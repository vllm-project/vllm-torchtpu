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

from .cache import PAGE_RECORDS, PAGE_ROWS, PackedFP8Layout
from .common import (
    QueryTile,
    _dma_rows,
    _head_order,
    _next_query_tile,
    _reduce_heads,
)
from .tuned import (
    PACKED_FP8_MQA_CONFIG,
    paged_mqa_key_tile,
    paged_mqa_query_tile,
)


def supports_candidate_publication(block_size: int) -> bool:
    """Block reductions fit independently in each packed-record vector."""
    return block_size > 0 and PAGE_RECORDS % block_size == 0


def _encode_page_table(page_ids, layout, first_bank_rows, tile_pages):
    valid = (page_ids >= 0) & (page_ids < layout.pages)
    # Spread masked reads: repeating page zero serializes a partially filled tile.
    safe = jnp.where(valid, page_ids, jnp.arange(page_ids.shape[-1]) % layout.pages)
    global_rows = layout.page_offset + safe * layout.page_stride
    bank = (global_rows >= first_bank_rows).astype(jnp.int32)
    local_rows = global_rows - bank * first_bank_rows
    full = jnp.all(valid.reshape(-1, tile_pages), axis=1).astype(jnp.int32)
    full = jnp.repeat(full, tile_pages).reshape(page_ids.shape)
    return (local_rows << 3) | (full << 2) | (valid.astype(jnp.int32) << 1) | bank


def _kernel(
    offsets,
    lengths,
    query_start,
    inputs,
    out,
    scratch,
    *,
    config,
    info,
    layout,
    reuse_keys,
    block_out=None,
    candidate_block_size=None,
):
    mxu_size, sublanes = info.mxu_column_size, info.num_sublanes
    q, weights, pool, table = inputs
    buffers, semaphores = scratch
    queries, keys, scores, counts, pages = buffers[:5]
    read_sem, write_sem, q_sem, page_sem = semaphores[:4]
    if candidate_block_size is not None:
        block_scores, block_sem = buffers[5], semaphores[4]
        block_columns = block_scores.shape[-1]
    weights = pltpu.annotate(weights, no_store=True)
    keys = pltpu.annotate(keys, no_store=True)
    tokens, heads, dim = q.shape
    begin = query_start[()]
    end = begin + out.shape[0]
    q_tile, key_tile = config.query_tile, config.key_tile
    page_size, page_rows = layout.page_size, layout.page_rows
    page_count, tile_pages = table.shape[-1], key_tile // page_size
    tiles, fragments = out.shape[1] // key_tile, key_tile // mxu_size

    def page_record(i, slot):
        # Keep SMEM flat: 2D indexing adds scalar address work in masked tiles.
        return pages[slot * page_count + i]

    def copy_queries(query, slot, wait=False, zero=False):
        dst = queries.at[pl.ds(slot * q_tile, q_tile)]
        if zero:
            dst[...] = jnp.zeros(dst.shape, dst.dtype)
        _dma_rows(q, dst, q_sem.at[slot], query.count, wait, src_start=query.start)

    def release_outputs(pair):
        count = counts[pair]
        for slot in (2 * pair, 2 * pair + 1)[:tiles]:
            src = scores.at[slot]
            _dma_rows(src, src, write_sem.at[slot], count, wait=True)
            if candidate_block_size is not None:
                src = block_scores.at[slot]
                _dma_rows(src, src, block_sem.at[slot], count, wait=True)

    def load_page(i, page_slot, slot):
        record = page_record(i, page_slot)
        address = lax.shift_right_logical(record, jnp.int32(3))

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

        def key_slot(tile):
            return 0 if reuse_keys else (tile_base + tile) & (config.key_buffers - 1)

        def page_slot(tile):
            return (tile_base + tile) & 3

        def output_slot(tile):
            return 2 * q_slot + (tile & 1)

        copy_queries(following, 1 - q_slot, zero=True)
        lengths_vector = jnp.stack(row_lengths(query))
        minimum_length = reduce(jnp.minimum, row_lengths(query))

        def publish(tile, wait=False, slot=None):
            slot = output_slot(tile) if slot is None else slot
            src, sem = scores.at[slot], write_sem.at[slot]
            dst = out.at[:, pl.ds(tile * key_tile, key_tile)]
            _dma_rows(src, dst, sem, query.count, wait, dst_start=query.start - begin)
            if candidate_block_size is not None:
                assert block_out is not None
                _dma_rows(
                    block_scores.at[slot],
                    block_out.at[:, pl.ds(tile * block_columns, block_columns)],
                    block_sem.at[slot],
                    query.count,
                    wait,
                    dst_start=query.start - begin,
                )

        def load_table(future, wait=False):
            first, after = future < tile_count, future - tile_count
            third = after >= next_tiles
            row = jnp.where(
                first,
                query.page_row,
                jnp.where(third, lookahead.page_row, following.page_row),
            )
            tile = jnp.where(first, future, after - jnp.where(third, next_tiles, 0))
            slot = page_slot(future)
            dst = pages.at[pl.ds(slot * page_count, page_count)]
            dma = pltpu.make_async_copy(table.at[row, tile], dst, page_sem.at[slot])
            (dma.wait if wait else dma.start)()

        # The previous query already prefetched up to three tiles from these
        # first two groups. Fill the newly visible part of the lookahead.
        for i in range(3):

            @pl.when(((step == 0) | (i >= tile_count + next_tiles)) & (i < total_tiles))
            def prefill_table(i=i):
                load_table(i)

        release_outputs(q_slot)
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
                    for page in range(tile_pages):
                        load_page(page, page_slot(i), key_slot(i))

        copy_queries(query, q_slot, wait=True)

        def emit_tile(tile, *, prefetch, full):
            weight = jnp.concatenate([weights[r][:, None] for r in row_ids(query)])
            head_weights = _head_order(weight, q_tile, sublanes)
            query_slice = pl.ds(q_slot * q_tile, q_tile)
            lhs = _head_order(
                queries[query_slice].reshape(q_tile * heads, dim), q_tile, sublanes
            )
            if prefetch:
                load_table(tile + 2, wait=True)
            metadata = page_slot(tile + 2)
            following_slot = key_slot(tile + 2)

            def key_rows(start, count):
                return keys[key_slot(tile), pl.ds(start, count)]

            base = tile * key_tile + jnp.arange(PAGE_RECORDS)
            remaining = jnp.maximum(lengths_vector, 0)[:, None] - base[None, :]

            def qk(fragment):
                parts = []
                if page_size != PAGE_RECORDS:
                    for start in range(
                        fragment * mxu_size,
                        (fragment + 1) * mxu_size,
                        min(page_size, mxu_size),
                    ):
                        page, column = divmod(start, page_size)
                        raw = key_rows(
                            page * page_rows,
                            (PAGE_ROWS - 1) * page_size // PAGE_RECORDS,
                        )
                        if page_size < PAGE_RECORDS:
                            raw = jnp.repeat(raw, PAGE_RECORDS // page_size, axis=0)
                            columns = (
                                jnp.arange(PAGE_ROWS - 1)[:, None] * page_size
                                + jnp.arange(PAGE_RECORDS)[None, :] % page_size
                            ) % PAGE_RECORDS
                            words = jnp.take_along_axis(raw, columns, axis=1)[
                                :, :page_size
                            ]
                        else:
                            words = raw.reshape(PAGE_ROWS - 1, page_size)
                        parts.append(
                            words[:, column : column + min(page_size, mxu_size)]
                        )
                    rhs = pltpu.bitcast(jnp.concatenate(parts, axis=1), q.dtype)
                    logits = lax.dot(lhs, rhs, preferred_element_type=jnp.float32)
                    return jnp.maximum(logits.astype(jnp.bfloat16), jnp.bfloat16(0))
                for page in range(
                    fragment * mxu_size // PAGE_RECORDS,
                    (fragment + 1) * mxu_size // PAGE_RECORDS,
                ):
                    words = key_rows(page * PAGE_ROWS, PAGE_ROWS - 1)
                    parts.append(pltpu.bitcast(words, q.dtype))
                logits = lax.dot(
                    lhs,
                    jnp.concatenate(parts, axis=1),
                    preferred_element_type=jnp.float32,
                )
                return jnp.maximum(logits.astype(jnp.bfloat16), jnp.bfloat16(0))

            block_pending = []

            def publish_block_scores(column, scaled):
                assert candidate_block_size is not None
                local_page = column // PAGE_RECORDS
                reduced = scaled
                for bit in range(candidate_block_size.bit_length() - 1):
                    reduced = jnp.maximum(
                        reduced, pltpu.roll(reduced, 1 << bit, axis=1)
                    )
                blocks_per_page = PAGE_RECORDS // candidate_block_size
                indices = jnp.broadcast_to(
                    jnp.arange(blocks_per_page)[None, :] * candidate_block_size
                    + candidate_block_size
                    - 1,
                    (q_tile, blocks_per_page),
                )
                reduced = jnp.take_along_axis(reduced, indices, axis=1)
                block_pending.append(reduced)
                pages_per_write = info.num_lanes // blocks_per_page
                if (
                    local_page % pages_per_write == pages_per_write - 1
                    or local_page == key_tile // PAGE_RECORDS - 1
                ):
                    packed = jnp.concatenate(block_pending, axis=1)
                    start = (
                        column // candidate_block_size
                        + blocks_per_page
                        - packed.shape[1]
                    )
                    last = (lengths_vector - 1) // candidate_block_size
                    positions = tile * (key_tile // candidate_block_size)
                    positions += start + jnp.arange(packed.shape[1])
                    block_scores[
                        output_slot(tile),
                        :,
                        pl.ds(start, packed.shape[1]),
                    ] = jnp.where(positions[None, :] == last[:, None], jnp.inf, packed)
                    block_pending.clear()

            def store(fragment, values):
                if page_size != PAGE_RECORDS:
                    scales, masks = [], []
                    for start in range(
                        fragment * mxu_size,
                        (fragment + 1) * mxu_size,
                        min(page_size, mxu_size),
                    ):
                        page, column = divmod(start, page_size)
                        scale = key_rows(page * page_rows, page_rows).reshape(-1)[
                            (PAGE_ROWS - 1) * page_size : PAGE_ROWS * page_size
                        ]
                        scales.append(
                            lax.bitcast_convert_type(
                                scale[column : column + min(page_size, mxu_size)],
                                jnp.float32,
                            )
                        )
                        valid = full or (page_record(page, page_slot(tile)) & 2) != 0
                        masks.append(
                            jnp.full((min(page_size, mxu_size),), valid, jnp.int32)
                        )
                    scaled = values * jnp.concatenate(scales)[None, :]
                    if not full:
                        positions = (
                            tile * key_tile + fragment * mxu_size + jnp.arange(mxu_size)
                        )
                        scaled = jnp.where(
                            (jnp.concatenate(masks)[None, :] != 0)
                            & (lengths_vector[:, None] > positions[None, :]),
                            scaled,
                            -jnp.inf,
                        )
                    scores[
                        output_slot(tile), :, pl.ds(fragment * mxu_size, mxu_size)
                    ] = scaled
                    if candidate_block_size is not None:
                        for offset in range(0, mxu_size, PAGE_RECORDS):
                            publish_block_scores(
                                fragment * mxu_size + offset,
                                scaled[:, offset : offset + PAGE_RECORDS],
                            )
                    return
                for offset in range(0, mxu_size, PAGE_RECORDS):
                    local_page = (fragment * mxu_size + offset) // PAGE_RECORDS
                    valid = full or (page_record(local_page, page_slot(tile)) & 2) != 0
                    scale_row = (local_page + 1) * PAGE_ROWS - 1
                    scale = lax.bitcast_convert_type(
                        key_rows(scale_row, 1), jnp.float32
                    )
                    scaled = values[:, offset : offset + PAGE_RECORDS] * scale
                    column = local_page * PAGE_RECORDS
                    if not full:
                        bound = jnp.where(valid, column, jnp.iinfo(jnp.int32).max)
                        scaled = jnp.where(remaining > bound, scaled, -jnp.inf)
                    scores[output_slot(tile), :, pl.ds(column, PAGE_RECORDS)] = scaled
                    if candidate_block_size is not None:
                        publish_block_scores(column, scaled)

            pending = {}
            for stage in range(fragments + config.issue_depth):
                first = pl.cdiv(stage * mxu_size, page_size)
                last = pl.cdiv((stage + 1) * mxu_size, page_size)
                middle = (first + last + 1) // 2
                if stage < fragments:
                    if prefetch:
                        for page in range(first, middle):
                            load_page(page, metadata, following_slot)
                    pending[stage] = qk(stage)
                if stage >= config.issue_depth:
                    fragment = stage - config.issue_depth
                    values = _reduce_heads(
                        pending.pop(fragment) * head_weights,
                        selector,
                        sublanes=sublanes,
                        use_mxu=stage < fragments and bool(fragment % 2),
                    )
                    store(fragment, values)
                if stage < fragments and prefetch:
                    for page in range(middle, last):
                        load_page(page, metadata, following_slot)

        def step_tile(tile, _):
            # Four slots retain the current mask while prefetching metadata
            # one tile ahead of the two-tile K prefetch.
            pl.when(tile + 3 < total_tiles)(lambda: load_table(tile + 3))
            src = keys.at[key_slot(tile)]
            if not reuse_keys:
                pltpu.make_async_copy(src, src, read_sem.at[key_slot(tile)]).wait()
            pl.when(tile >= 2)(lambda: publish(tile - 2, wait=True))
            pending = tile + 2 < total_tiles
            encoded = page_record(0, page_slot(tile))
            full = ((tile + 1) * key_tile <= minimum_length) & ((encoded & 4) != 0)

            def emit(full):
                if reuse_keys:
                    emit_tile(tile, prefetch=False, full=full)
                else:
                    lax.cond(
                        pending,
                        lambda: emit_tile(tile, prefetch=True, full=full),
                        lambda: emit_tile(tile, prefetch=False, full=full),
                    )

            lax.cond(full, lambda: emit(True), lambda: emit(False))
            publish(tile)

        if tiles == 1:
            pl.when(tile_count > 0)(lambda: step_tile(jnp.int32(0), None))
        else:
            lax.fori_loop(0, tile_count, step_tile, None)

        counts[q_slot] = jnp.where(tile_count == tiles, query.count, 0)

        @pl.when(tile_count < tiles)
        def fill_padding():
            @pl.when(tile_count > 0)
            def drain():
                publish(tile_count - 1, wait=True)
                pl.when(tile_count > 1)(lambda: publish(tile_count - 2, wait=True))

            scores[output_slot(0)] = jnp.full(scores.shape[1:], -jnp.inf, jnp.float32)
            if candidate_block_size is not None:
                block_scores[output_slot(0)] = jnp.full(
                    block_scores.shape[1:], -jnp.inf, jnp.float32
                )
            for wait in (False, True):

                def body(tile, _, wait=wait):
                    publish(tile, wait=wait, slot=output_slot(0))

                lax.fori_loop(tile_count, tiles, body, None)

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
    first = next_query(QueryTile(begin, zero, zero))
    second = next_query(first)
    third = next_query(second)
    initial = (first, second, third)
    for pair in range(2):
        counts[pair] = jnp.int32(0)
    copy_queries(first, 0, zero=True)
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


def _kernel_with_block_scores(
    offsets,
    lengths,
    query_start,
    inputs,
    out,
    block_out,
    scratch,
    *,
    config,
    info,
    layout,
    reuse_keys,
    candidate_block_size,
):
    _kernel(
        offsets,
        lengths,
        query_start,
        inputs,
        out,
        scratch,
        config=config,
        info=info,
        layout=layout,
        reuse_keys=reuse_keys,
        block_out=block_out,
        candidate_block_size=candidate_block_size,
    )


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
    if candidate_block_size is not None and not supports_candidate_publication(
        candidate_block_size
    ):
        raise ValueError(
            "Fused publication requires a positive block size "
            "dividing the packed record width"
        )
    tile = paged_mqa_query_tile(query_size, heads)
    config = PACKED_FP8_MQA_CONFIG
    config = replace(
        config,
        query_tile=tile,
        key_tile=paged_mqa_key_tile(max_model_len, layout.page_size, config.key_tile),
    )
    width = pl.cdiv(max_model_len, config.key_tile) * config.key_tile
    if tokens == 0 or width == 0:
        empty = jnp.empty((tokens, max_model_len), jnp.float32)
        if candidate_block_size is not None:
            return empty, jnp.empty(
                (tokens, pl.cdiv(max_model_len, candidate_block_size)), jnp.float32
            )
        return empty
    info = cast(pltpu.TpuInfo, pltpu.get_tpu_info())
    page_count = width // layout.page_size
    tile_pages = config.key_tile // layout.page_size
    page_ids = block_table[:, :page_count]
    page_ids = jnp.pad(
        page_ids, ((0, 0), (0, page_count - page_ids.shape[1])), constant_values=-1
    )
    table = _encode_page_table(page_ids, layout, banks[0].shape[0], tile_pages)
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
            pltpu.SMEM((2,), jnp.int32),
            pltpu.SMEM((4 * page_count,), jnp.int32),
        ),
        tuple(
            pltpu.SemaphoreType.DMA(shape)
            for shape in ((config.key_buffers,), (4, tile), (2,), (4,))
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
    )
    out_shape = jax.ShapeDtypeStruct((tokens, width), jnp.float32)
    out_specs = hbm
    if candidate_block_size is not None:
        block_columns = (
            pl.cdiv(config.key_tile // candidate_block_size, info.num_lanes)
            * info.num_lanes
        )
        scratch = (
            (*scratch[0], pltpu.VMEM((4, tile, block_columns), jnp.float32)),
            (*scratch[1], pltpu.SemaphoreType.DMA((4, tile))),
        )
        kernel = partial(
            _kernel_with_block_scores,
            config=config,
            info=info,
            layout=layout,
            reuse_keys=reuse_keys and width == config.key_tile,
            candidate_block_size=candidate_block_size,
        )
        out_shape = (
            out_shape,
            jax.ShapeDtypeStruct(
                (tokens, width // config.key_tile * block_columns), jnp.float32
            ),
        )
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
    if candidate_block_size is not None:
        scores, blocks = result
        # Each tile's block output has its own HBM lane padding.
        blocks = blocks.reshape(tokens, width // config.key_tile, -1)
        blocks = blocks[:, :, : config.key_tile // candidate_block_size]
        blocks = blocks.reshape(tokens, width // candidate_block_size)
        return scores[:, :max_model_len], blocks[
            :, : pl.cdiv(max_model_len, candidate_block_size)
        ]
    return result[:, :max_model_len]
