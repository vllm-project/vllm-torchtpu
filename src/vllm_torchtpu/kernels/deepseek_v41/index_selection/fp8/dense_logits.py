# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged FP8 MQA logits for TPU v7."""

from dataclasses import replace
from functools import partial, reduce
from typing import cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .block_scores import TileBlockWriter, supports_candidate_publication
from .cache import PAGE_RECORDS, PAGE_ROWS, PackedFP8Layout
from .common import (
    QueryTile,
    _dma_rows,
    _head_order,
    _next_query_tile,
    _reduce_heads,
    _reduce_on_mxu,
    _weighted_selector,
)
from .tuned import (
    PACKED_FP8_MQA_CONFIG,
    paged_mqa_key_tile,
    paged_mqa_query_tile,
)


def _encode_page_table(page_ids, layout, first_bank_rows):
    """Records hold the bank-local row << 2, a valid bit and the bank bit."""
    valid = (page_ids >= 0) & (page_ids < layout.pages)
    # Spread masked reads: repeating page zero serializes a partially filled tile.
    safe = jnp.where(valid, page_ids, jnp.arange(page_ids.shape[-1]) % layout.pages)
    global_rows = layout.page_offset + safe * layout.page_stride
    bank = (global_rows >= first_bank_rows).astype(jnp.int32)
    local_rows = global_rows - bank * first_bank_rows
    return (local_rows << 2) | (valid.astype(jnp.int32) << 1) | bank


def _kernel(
    offsets,
    lengths,
    query_start,
    inputs,
    *refs,
    config,
    info,
    layout,
    reuse_keys,
    candidate_block_size=None,
):
    """Score query tiles against paged K; refs are the outputs, then scratch.

    The block-maxima output follows the scores when candidate_block_size is set.
    """
    *outputs, scratch = refs
    out, block_out = outputs[0], (outputs[1] if len(outputs) > 1 else None)
    mxu_size, sublanes = info.mxu_column_size, info.num_sublanes
    q, weights, pool, table = inputs
    buffers, semaphores = scratch
    queries, keys, scores, counts, pages, padding = buffers[:6]
    read_sem, write_sem, q_sem, page_sem, pad_sem = semaphores[:5]
    weights = pltpu.annotate(weights, no_store=True)
    keys = pltpu.annotate(keys, no_store=True)
    tokens, heads, dim = q.shape
    begin = query_start[()]
    end = begin + out.shape[0]
    q_tile, key_tile = config.query_tile, config.key_tile
    page_size, page_rows = layout.page_size, layout.page_rows
    page_count, tile_pages = table.shape[-1], key_tile // page_size
    tiles, fragments = out.shape[1] // key_tile, key_tile // mxu_size
    # Partial tiles whose rows all end in the first half score only that
    # half; four-fragment steps keep the -inf fills lane-aligned.
    half = pl.cdiv(pl.cdiv(fragments, 2), 4) * 4
    if candidate_block_size is not None:
        # One contiguous block row per query slot; tiles merge into it.
        block_scores, block_sem = buffers[6], semaphores[5]

        def block_rows(pair):
            return block_scores.at[pair, :, pl.ds(0, block_out.shape[1])]

    def page_record(i, slot):
        # Keep SMEM flat: 2D indexing adds scalar address work in masked tiles.
        return pages[slot * page_count + i]

    def copy_queries(query, slot, wait=False, zero=False):
        dst = queries.at[pl.ds(slot * q_tile, q_tile)]
        if zero:
            dst[...] = jnp.zeros(dst.shape, dst.dtype)
        _dma_rows(q, dst, q_sem.at[slot], query.count, wait, src_start=query.start)

    def release_outputs(pair):
        # counts: rows pending per score slot [0, 4), block rows [4, 6) and
        # -inf padding row copies [6, 8) per query slot.
        for slot in (2 * pair, 2 * pair + 1)[:tiles]:
            src = scores.at[slot]
            _dma_rows(src, src, write_sem.at[slot], counts[slot], wait=True)
        if candidate_block_size is not None:
            src = block_rows(pair)
            _dma_rows(src, src, block_sem.at[pair], counts[4 + pair], wait=True)
        row = pltpu.make_async_copy(
            padding.at[pl.ds(0, 1)],
            out.at[pl.ds(0, 1), pl.ds(0, key_tile)],
            pad_sem.at[pair],
        )
        lax.fori_loop(0, counts[6 + pair], lambda i, _: row.wait(), None)

    def load_page(i, page_slot, slot):
        record = page_record(i, page_slot)
        address = lax.shift_right_logical(record, jnp.int32(2))

        def load(bank):
            pltpu.make_async_copy(
                pool[bank].at[pl.ds(address, page_rows)],
                keys.at[slot, pl.ds(i * page_rows, page_rows)],
                read_sem.at[slot],
            ).start()

        if len(pool) == 1:
            load(0)
        else:
            lax.cond((record & 1) == 0, lambda: load(0), lambda: load(1))

    # Pages past a half tile's live part issue from a dynamic loop: unrolling
    # them all costs code size and scalar issue slots.
    half_pages = pl.cdiv(half * mxu_size, page_size) if half < fragments else tile_pages
    rest_pages = tile_pages - half_pages
    rest_chunk = next(c for c in (8, 6, 4, 2, 1) if rest_pages % c == 0)

    def load_rest(page_slot, slot):
        def body(chunk, _):
            for j in range(rest_chunk):
                load_page(half_pages + chunk * rest_chunk + j, page_slot, slot)

        if rest_pages:
            lax.fori_loop(0, rest_pages // rest_chunk, body, None)

    next_query = partial(_next_query_tile, offsets=offsets, tile_size=q_tile, end=end)

    def row_ids(query):
        return [jnp.minimum(query.start + i, tokens - 1) for i in range(q_tile)]

    def row_lengths(query):
        rows = enumerate(row_ids(query))
        return [jnp.where(i < query.count, lengths[r], 0) for i, r in rows]

    def live_tiles(query):
        limit = reduce(jnp.maximum, row_lengths(query), jnp.int32(0))
        return pl.cdiv(jnp.minimum(limit, out.shape[1]), key_tile)

    owners = _head_order(
        jnp.repeat(jnp.arange(q_tile), heads)[:, None], q_tile, sublanes
    )[:, 0]
    selector = (jnp.arange(q_tile)[:, None] == owners[None, :]).astype(jnp.bfloat16)

    def query_block(state):
        query, following, lookahead, tile_counts, step, tile_base, ready = state
        tile_count, next_tiles, ahead_tiles = tile_counts
        total_tiles = tile_count + next_tiles + ahead_tiles
        q_slot = step & 1
        farthest = next_query(lookahead)
        far_tiles = live_tiles(farthest)
        # Page tables run three tiles ahead of the current tile, so with one
        # tile per group they reach into the group after the lookahead.
        table_tiles = total_tiles + far_tiles

        def key_slot(tile):
            return 0 if reuse_keys else lax.rem(tile_base + tile, config.key_buffers)

        def page_slot(tile):
            return (tile_base + tile) & 3

        def output_slot(tile):
            return 2 * q_slot + (tile & 1)

        copy_queries(following, 1 - q_slot, zero=True)
        lengths_vector = jnp.stack(row_lengths(query))
        maximum_length = reduce(jnp.maximum, row_lengths(query))

        first_row = query.start - begin

        def publish(tile, wait=False):
            slot = output_slot(tile)
            src, sem = scores.at[slot], write_sem.at[slot]
            dst = out.at[:, pl.ds(tile * key_tile, key_tile)]
            _dma_rows(src, dst, sem, query.count, wait, dst_start=first_row)

        def owner(future):
            """Request row and local tile of a future tile."""
            ends = (tile_count, tile_count + next_tiles, total_tiles)

            def pick(*values):
                # values[g] belongs to group g; the last one is the farthest.
                picked = values[-1]
                for end_, value in zip(ends[::-1], values[-2::-1]):
                    picked = jnp.where(future < end_, value, picked)
                return picked

            row = pick(*(g.page_row for g in (query, following, lookahead, farthest)))
            return row, future - pick(0, *ends)

        def load_table(future, wait=False):
            row, tile = owner(future)
            slot = page_slot(future)
            dst = pages.at[pl.ds(slot * page_count, page_count)]
            dma = pltpu.make_async_copy(table.at[row, tile], dst, page_sem.at[slot])
            (dma.wait if wait else dma.start)()

        # The previous query already prefetched up to three tiles from these
        # three groups. Fill the newly visible part of the farthest group.
        for i in range(3):

            @pl.when(((step == 0) | (i >= total_tiles)) & (i < table_tiles))
            def prefill_table(i=i):
                load_table(i)

        release_outputs(q_slot)
        if candidate_block_size is not None:
            block_scores[q_slot] = jnp.full(
                block_scores.shape[1:], -jnp.inf, jnp.float32
            )
        if reuse_keys:
            pl.when(tile_count > 0)(lambda: load_table(0, wait=True))

            # A complete K tile stays resident across query groups of one request.
            # In this mode ready identifies the retained tile's request owner.
            @pl.when((tile_count > 0) & (ready != query.page_row))
            def fill_keys():
                for page in range(tile_pages):
                    load_page(page, page_slot(0), 0)
                src = keys.at[0]
                pltpu.make_async_copy(src, src, read_sem.at[0]).wait()
        else:
            for i in range(2):

                @pl.when((i >= ready) & (i < total_tiles))
                def prefill(i=i):
                    load_table(i, wait=True)
                    for page in range(half_pages):
                        load_page(page, page_slot(i), key_slot(i))
                    load_rest(page_slot(i), key_slot(i))

        copy_queries(query, q_slot, wait=True)
        weight = jnp.concatenate([weights[r][:, None] for r in row_ids(query)])
        head_weights = _head_order(weight, q_tile, sublanes)
        mxu_selector = _weighted_selector(
            selector, weights, row_ids(query), sublanes=sublanes
        )
        reduce_heads = partial(
            _reduce_heads,
            weights=head_weights,
            selector=mxu_selector,
            sublanes=sublanes,
        )
        query_slice = pl.ds(q_slot * q_tile, q_tile)
        lhs = _head_order(
            queries[query_slice].reshape(q_tile * heads, dim), q_tile, sublanes
        )

        def emit_tile(tile, *, prefetch, count=fragments):
            # Only the first `count` fragments are scored; the rest are -inf.
            live_pages = count * mxu_size // PAGE_RECORDS

            metadata, following_slot = page_slot(tile + 2), key_slot(tile + 2)

            def prefetching(fn):
                if prefetch:
                    fn()

            def load_pages(first, last):
                for page in range(first, last):
                    load_page(page, metadata, following_slot)

            prefetching(lambda: load_table(tile + 2, wait=True))

            def key_rows(start, count):
                return keys[key_slot(tile), pl.ds(start, count)]

            base = tile * key_tile + jnp.arange(PAGE_RECORDS)
            remaining = jnp.maximum(lengths_vector, 0)[:, None] - base[None, :]

            def qk(fragment):
                parts = [
                    pltpu.bitcast(key_rows(page * PAGE_ROWS, PAGE_ROWS - 1), q.dtype)
                    for page in range(
                        fragment * mxu_size // PAGE_RECORDS,
                        (fragment + 1) * mxu_size // PAGE_RECORDS,
                    )
                ]
                logits = lax.dot(
                    lhs,
                    jnp.concatenate(parts, axis=1),
                    preferred_element_type=jnp.float32,
                )
                return jnp.maximum(logits.astype(jnp.bfloat16), jnp.bfloat16(0))

            writer = (
                TileBlockWriter(
                    block_scores.at[q_slot],
                    candidate_block_size,
                    tile,
                    key_tile,
                    lengths_vector,
                    live_pages,
                )
                if candidate_block_size is not None
                else None
            )

            def store(fragment, values):
                for offset in range(0, mxu_size, PAGE_RECORDS):
                    local_page = (fragment * mxu_size + offset) // PAGE_RECORDS
                    valid = (page_record(local_page, page_slot(tile)) & 2) != 0
                    scale_row = (local_page + 1) * PAGE_ROWS - 1
                    scale = lax.bitcast_convert_type(
                        key_rows(scale_row, 1), jnp.float32
                    )
                    scaled = values[:, offset : offset + PAGE_RECORDS] * scale
                    column = local_page * PAGE_RECORDS
                    bound = jnp.where(valid, column, jnp.iinfo(jnp.int32).max)
                    scaled = jnp.where(remaining > bound, scaled, -jnp.inf)
                    scores[output_slot(tile), :, pl.ds(column, PAGE_RECORDS)] = scaled
                    if writer is not None:
                        writer.add(local_page, scaled)

            pending, depth = {}, config.issue_depth
            every = 3 if candidate_block_size else 4
            # Half tiles interleave only the first half of the next tile's
            # pages and drop the issue-only stages; the rest follow in a loop.
            split = count < fragments
            inline_pages = half_pages if split else tile_pages
            stages = count + depth if split else max(count + depth, fragments)
            for stage in range(stages):
                first = min(pl.cdiv(stage * mxu_size, page_size), inline_pages)
                last = min(pl.cdiv((stage + 1) * mxu_size, page_size), inline_pages)
                middle = (first + last + 1) // 2
                if first < middle:
                    prefetching(partial(load_pages, first, middle))
                if stage < count:
                    pending[stage] = qk(stage)
                if depth <= stage < count + depth:
                    fragment = stage - depth
                    mxu = _reduce_on_mxu(fragment, stage, count, every=every, drain=4)
                    store(fragment, reduce_heads(pending.pop(fragment), use_mxu=mxu))
                if middle < last:
                    prefetching(partial(load_pages, middle, last))
            if count < fragments:
                slot, start = output_slot(tile), count * mxu_size
                scores[slot, :, pl.ds(start, key_tile - start)] = jnp.full(
                    (q_tile, key_tile - start), -jnp.inf, jnp.float32
                )
            if split:
                prefetching(lambda: load_rest(metadata, following_slot))

        def wait_keys(slot):
            src = keys.at[slot]
            pltpu.make_async_copy(src, src, read_sem.at[slot]).wait()

        def step_tile(tile, _):
            # Four slots retain the current mask while prefetching metadata
            # one tile ahead of the two-tile K prefetch.
            pl.when(tile + 3 < table_tiles)(lambda: load_table(tile + 3))
            if not reuse_keys:
                wait_keys(key_slot(tile))
            pl.when(tile >= 2)(lambda: publish(tile - 2, wait=True))

            def scored(prefetch):
                emit = partial(emit_tile, tile, prefetch=prefetch)
                # Every tile takes the masked path: the extra selects are free
                # next to the MXU/VPOP bound, and one emit family halves code.
                if half >= fragments:
                    emit()
                else:
                    lax.cond(
                        maximum_length - tile * key_tile <= half * mxu_size,
                        lambda: emit(count=half),
                        emit,
                    )

            # One branch per tile: a per-page predicate costs scalar issue
            # slots in every fragment stage.
            if reuse_keys:
                scored(False)
            else:
                lax.cond(
                    tile + 2 < total_tiles,
                    lambda: scored(True),
                    lambda: scored(False),
                )
            publish(tile)

        if tiles == 1:
            pl.when(tile_count > 0)(lambda: step_tile(jnp.int32(0), None))
        else:
            lax.fori_loop(0, tile_count, step_tile, None)

        # The last two tiles' writes stay in flight until this slot pair
        # is reused; tile t publishes from score slot t & 1.
        for local in range(min(tiles, 2)):
            counts[2 * q_slot + local] = jnp.where(tile_count > local, query.count, 0)

        # Unscored tiles copy from a constant -inf buffer, so their writes
        # need not wait for the scored tiles' writes to drain.
        def pad_tile(tile, _):
            dst = out.at[:, pl.ds(tile * key_tile, key_tile)]
            _dma_rows(
                padding, dst, pad_sem.at[q_slot], query.count, dst_start=first_row
            )

        lax.fori_loop(tile_count, tiles, pad_tile, None)
        counts[6 + q_slot] = (tiles - tile_count) * query.count

        if candidate_block_size is not None:
            assert block_out is not None
            src, sem = block_rows(q_slot), block_sem.at[q_slot]
            _dma_rows(src, block_out, sem, query.count, dst_start=first_row)
            counts[4 + q_slot] = query.count

        return (
            following,
            lookahead,
            farthest,
            (next_tiles, ahead_tiles, live_tiles(farthest)),
            step + 1,
            tile_base + tile_count,
            jnp.where(tile_count > 0, query.page_row, ready)
            if reuse_keys
            else jnp.minimum(2, next_tiles + ahead_tiles),
        )

    zero = jnp.int32(0)
    initial = [next_query(QueryTile(begin, zero, zero))]
    for _ in range(2):
        initial.append(next_query(initial[-1]))
    for slot in range(8):
        counts[slot] = jnp.int32(0)
    padding[...] = jnp.full(padding.shape, -jnp.inf, jnp.float32)
    copy_queries(initial[0], 0, zero=True)
    lax.while_loop(
        lambda state: state[0].start < end,
        query_block,
        (
            *initial,
            tuple(live_tiles(q) for q in initial),
            zero,
            zero,
            jnp.int32(-1) if reuse_keys else zero,
        ),
    )
    release_outputs(0)
    release_outputs(1)


@jax.jit(
    static_argnames=(
        "cache_layout",
        "max_model_len",
        "query_size",
        "query_count",
        "reuse_keys",
        "candidate_block_size",
    )
)
def fp8_paged_mqa_logits(
    q: jax.Array,
    kv_cache: jax.Array | tuple[jax.Array, jax.Array],
    weights: jax.Array,
    context_lens: jax.Array,
    block_table: jax.Array,
    *,
    query_start_loc: jax.Array,
    max_model_len: int,
    q_scale: jax.Array,
    query_size: int,
    cache_layout: PackedFP8Layout | None = None,
    query_start: jax.Array | int = 0,
    query_count: int | None = None,
    reuse_keys: bool = False,
    candidate_block_size: int | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Return FP32 scores for the query window, with invalid positions set to -inf.

    Args:
        q: FP8 E4M3FN [T,32,128], in request order.
        kv_cache: One or two int32 [rows,128] banks in fp8.cache format.
        weights: FP32 [T,32]; weighted BF16 products must be finite.
        context_lens: Int32 [T] visible index-key counts, zero for padding.
        block_table: Int32 [G,P] page IDs covering the requested output width.
        query_start_loc: Int32 [G+1], monotonic from 0 to T; repeats mean empty.
        max_model_len: Nonnegative output width in index keys.
        q_scale: FP32 [T,32] query scales; K scales reside inside each page.
        query_size: Static query-tile bound, normally speculative tokens+1.
        cache_layout: Shared cache placement; None means tightly packed pages.
        query_start: First query to read from the full input arrays and offsets.
        query_count: Output rows; defaults to the full query batch.
        reuse_keys: Keep a single complete K tile resident across query groups
            belonging to the same request. Multi-tile outputs retain prefetching.
        candidate_block_size: Also return visible block maxima with the newest
            visible block forced to +inf; blocks must divide the packed record width.
    """
    tokens, heads, dim = q.shape
    tokens = tokens if query_count is None else query_count
    banks = kv_cache if isinstance(kv_cache, tuple) else (kv_cache,)
    layout = cache_layout or PackedFP8Layout(
        sum(bank.shape[0] for bank in banks) // PAGE_ROWS
    )
    layout.validate(tuple(bank.shape[0] for bank in banks))
    if layout.page_size != PAGE_RECORDS:
        raise ValueError(f"Dense FP8 scoring requires {PAGE_RECORDS}-record pages")
    if candidate_block_size is not None and not supports_candidate_publication(
        candidate_block_size
    ):
        raise ValueError(
            "Fused publication requires a positive block size "
            "dividing the packed record width"
        )
    tile = paged_mqa_query_tile(query_size, heads)
    key_tile = paged_mqa_key_tile(
        max_model_len, layout.page_size, PACKED_FP8_MQA_CONFIG.key_tile
    )
    config = replace(PACKED_FP8_MQA_CONFIG, query_tile=tile, key_tile=key_tile)
    width = pl.cdiv(max_model_len, key_tile) * key_tile
    widths = (max_model_len,)
    if candidate_block_size is not None:
        widths += (pl.cdiv(max_model_len, candidate_block_size),)
    if tokens == 0 or width == 0:
        empty = tuple(jnp.empty((tokens, w), jnp.float32) for w in widths)
        return empty if len(empty) > 1 else empty[0]
    info = cast(pltpu.TpuInfo, pltpu.get_tpu_info())
    page_count = width // layout.page_size
    tile_pages = config.key_tile // layout.page_size
    page_ids = block_table[:, :page_count]
    page_ids = jnp.pad(
        page_ids, ((0, 0), (0, page_count - page_ids.shape[1])), constant_values=-1
    )
    table = _encode_page_table(page_ids, layout, banks[0].shape[0])
    page_count = pl.cdiv(tile_pages, info.num_lanes) * info.num_lanes
    table = jnp.pad(
        table.reshape(page_ids.shape[0], -1, tile_pages),
        ((0, 0), (0, 0), (0, page_count - tile_pages)),
    )
    scratch = (
        (
            pltpu.VMEM((2 * tile, heads, dim), q.dtype),
            pltpu.VMEM(
                (config.key_buffers, tile_pages * layout.page_rows, dim), jnp.int32
            ),
            pltpu.VMEM((4, tile, config.key_tile), jnp.float32),
            pltpu.SMEM((8,), jnp.int32),
            pltpu.SMEM((4 * page_count,), jnp.int32),
            pltpu.VMEM((tile, config.key_tile), jnp.float32),
        ),
        tuple(
            pltpu.SemaphoreType.DMA(shape)
            for shape in ((config.key_buffers,), (4, tile), (2,), (4,), (2,))
        ),
    )
    hbm, vmem = (pl.BlockSpec(memory_space=s) for s in (pltpu.HBM, pltpu.VMEM))
    inputs = (q, weights * q_scale, banks, table)
    kernel = partial(
        _kernel,
        config=config,
        info=info,
        layout=layout,
        reuse_keys=reuse_keys and width == config.key_tile,
        candidate_block_size=candidate_block_size,
    )
    out_shape = jax.ShapeDtypeStruct((tokens, width), jnp.float32)
    out_specs = hbm
    if candidate_block_size is not None:
        blocks = width // candidate_block_size
        # One spare vector for the last unaligned tile write.
        block_columns = (pl.cdiv(blocks, info.num_lanes) + 1) * info.num_lanes
        scratch = (
            (*scratch[0], pltpu.VMEM((2, tile, block_columns), jnp.float32)),
            (*scratch[1], pltpu.SemaphoreType.DMA((2, tile))),
        )
        # Lane-aligned row DMAs; the -inf tail is sliced off below.
        lanes = pl.cdiv(blocks, info.num_lanes) * info.num_lanes
        out_shape = (out_shape, jax.ShapeDtypeStruct((tokens, lanes), jnp.float32))
        out_specs = (hbm, hbm)
    result = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(1,),
            in_specs=((hbm, vmem, tuple(hbm for _ in banks), hbm),),
            out_specs=out_specs,
            scratch_shapes=(scratch,),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=info.vmem_capacity_bytes,
            disable_bounds_checks=True,
        ),
        name="fp8_paged_mqa_logits_dense",
    )(query_start_loc, context_lens, jnp.asarray(query_start, jnp.int32), inputs)
    if candidate_block_size is None:
        return result[:, :max_model_len]
    return tuple(x[:, :w] for x, w in zip(result, widths))
