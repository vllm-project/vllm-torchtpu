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

from vllm_torchtpu import envs

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
from .tuned import PACKED_FP8_MQA_CONFIG, paged_mqa_key_tile, paged_mqa_query_tile


def _encode_page_table(page_ids, layout, first_bank_rows):
    """Records hold the bank-local row << 2, a valid bit and the bank bit."""
    valid = (page_ids >= 0) & (page_ids < layout.pages)
    # Spread masked reads: repeating page zero serializes a partially filled tile.
    safe = jnp.where(valid, page_ids, jnp.arange(page_ids.shape[-1]) % layout.pages)
    global_rows = layout.page_offset + safe * layout.page_stride
    bank = (global_rows >= first_bank_rows).astype(jnp.int32)
    local_rows = global_rows - bank * first_bank_rows
    return (local_rows << 2) | (valid.astype(jnp.int32) << 1) | bank


def _pick(future, ends, values):
    """values[g] belongs to the tile group ending at ends[g]; the last is open."""
    picked = values[-1]
    for end, value in zip(ends[::-1], values[-2::-1]):
        picked = jnp.where(future < end, value, picked)
    return picked


class _DenseKernel:
    """Refs and static tiling of one call."""

    def __init__(self, smem, inputs, refs, config, info, layout, reuse_keys):
        self.out, scratch = refs
        self.offsets, self.lengths, query_start = smem
        self.config, self.layout, self.reuse_keys = config, layout, reuse_keys
        self.mxu_size, self.sublanes = info.mxu_column_size, info.num_sublanes
        self.q, weights, self.pool, self.table = inputs
        buffers, semaphores = scratch
        self.queries, keys, self.scores, self.counts, self.pages, self.padding = (
            buffers[:6]
        )
        self.read_sem, self.write_sem, self.q_sem, self.page_sem, self.pad_sem = (
            semaphores[:5]
        )
        self.weights = pltpu.annotate(weights, no_store=True)
        self.keys = pltpu.annotate(keys, no_store=True)
        self.tokens, self.heads, self.dim = self.q.shape
        self.begin = query_start[()]
        self.end = self.begin + self.out.shape[0]
        self.q_tile, self.key_tile = config.query_tile, config.key_tile
        self.page_count = self.table.shape[-1]
        self.tile_pages = self.key_tile // layout.page_size
        self.tiles = self.out.shape[1] // self.key_tile
        self.fragments = self.key_tile // self.mxu_size
        # Partial tiles whose rows all end in the first half score only that
        # half; four-fragment steps keep the -inf fills lane-aligned.
        self.half = pl.cdiv(pl.cdiv(self.fragments, 2), 4) * 4
        # Pages past a half tile's live part issue from a dynamic loop:
        # unrolling them all costs code size and scalar issue slots.
        half_pages = pl.cdiv(self.half * self.mxu_size, layout.page_size)
        self.half_pages = half_pages if self.half < self.fragments else self.tile_pages
        self.rest_pages = self.tile_pages - self.half_pages
        self.rest_chunk = next(c for c in (8, 6, 4, 2, 1) if self.rest_pages % c == 0)
        owners = jnp.repeat(jnp.arange(self.q_tile), self.heads)[:, None]
        owners = _head_order(owners, self.q_tile, self.sublanes)[:, 0]
        selector = jnp.arange(self.q_tile)[:, None] == owners[None, :]
        self.selector = selector.astype(jnp.bfloat16)

    def page_record(self, i, slot):
        # Keep SMEM flat: 2D indexing adds scalar address work in masked tiles.
        return self.pages[slot * self.page_count + i]

    def copy_queries(self, query, slot, wait=False, zero=False):
        dst = self.queries.at[pl.ds(slot * self.q_tile, self.q_tile)]
        if zero:
            dst[...] = jnp.zeros(dst.shape, dst.dtype)
        sem = self.q_sem.at[slot]
        _dma_rows(self.q, dst, sem, query.count, wait, src_start=query.start)

    def release_outputs(self, pair):
        # counts: rows pending per score slot [0, 4), [4, 6) unused and
        # -inf padding row copies [6, 8) per query slot.
        for slot in (2 * pair, 2 * pair + 1)[: self.tiles]:
            src = self.scores.at[slot]
            _dma_rows(src, src, self.write_sem.at[slot], self.counts[slot], wait=True)
        src = self.padding.at[pl.ds(0, 1)]
        dst = self.out.at[pl.ds(0, 1), pl.ds(0, self.key_tile)]
        row = pltpu.make_async_copy(src, dst, self.pad_sem.at[pair])
        lax.fori_loop(0, self.counts[6 + pair], lambda i, _: row.wait(), None)

    def load_page(self, i, page_slot, slot):
        record = self.page_record(i, page_slot)
        address = lax.shift_right_logical(record, jnp.int32(2))
        loads = [partial(self._start_load, b, address, i, slot) for b in (0, 1)]
        if len(self.pool) == 1:
            loads[0]()
        else:
            lax.cond((record & 1) == 0, *loads)

    def _start_load(self, bank, address, i, slot):
        rows = self.layout.page_rows
        pltpu.make_async_copy(
            self.pool[bank].at[pl.ds(address, rows)],
            self.keys.at[slot, pl.ds(i * rows, rows)],
            self.read_sem.at[slot],
        ).start()

    def load_pages(self, first, last, page_slot, slot):
        for page in range(first, last):
            self.load_page(page, page_slot, slot)

    def load_rest(self, page_slot, slot):
        if self.rest_pages:
            chunks = self.rest_pages // self.rest_chunk
            body = partial(self._load_chunk, page_slot, slot)
            lax.fori_loop(0, chunks, body, None)

    def _load_chunk(self, page_slot, slot, chunk, _):
        first = self.half_pages + chunk * self.rest_chunk
        for j in range(self.rest_chunk):
            self.load_page(first + j, page_slot, slot)

    def wait_keys(self, slot):
        src = self.keys.at[slot]
        pltpu.make_async_copy(src, src, self.read_sem.at[slot]).wait()

    def next_query(self, query):
        return _next_query_tile(query, self.offsets, self.q_tile, self.end)

    def row_ids(self, query):
        last = self.tokens - 1
        return [jnp.minimum(query.start + i, last) for i in range(self.q_tile)]

    def row_lengths(self, query):
        rows = enumerate(self.row_ids(query))
        return [jnp.where(i < query.count, self.lengths[r], 0) for i, r in rows]

    def live_tiles(self, query):
        limit = reduce(jnp.maximum, self.row_lengths(query), jnp.int32(0))
        return pl.cdiv(jnp.minimum(limit, self.out.shape[1]), self.key_tile)

    def query_block(self, state):
        return _QueryBlock(self, state).run()

    def run(self):
        zero = jnp.int32(0)
        initial = [self.next_query(QueryTile(self.begin, zero, zero))]
        for _ in range(2):
            initial.append(self.next_query(initial[-1]))
        for slot in range(8):
            self.counts[slot] = zero
        self.padding[...] = jnp.full(self.padding.shape, -jnp.inf, jnp.float32)
        self.copy_queries(initial[0], 0, zero=True)
        tiles = tuple(self.live_tiles(q) for q in initial)
        ready = jnp.int32(-1) if self.reuse_keys else zero
        state = (*initial, tiles, zero, zero, ready)
        lax.while_loop(lambda s: s[0].start < self.end, self.query_block, state)
        self.release_outputs(0)
        self.release_outputs(1)


class _QueryBlock:
    """One query tile's key tiles, with K prefetch for the following tiles."""

    def __init__(self, kernel, state):
        self.k = kernel
        self.query, self.following, self.lookahead, tile_counts, *rest = state
        self.step, self.tile_base, self.ready = rest
        self.tile_count, self.next_tiles, self.ahead_tiles = tile_counts
        self.total_tiles = self.tile_count + self.next_tiles + self.ahead_tiles
        self.q_slot = self.step & 1
        self.farthest = kernel.next_query(self.lookahead)
        self.far_tiles = kernel.live_tiles(self.farthest)
        # Page tables run three tiles ahead of the current tile, so with one
        # tile per group they reach into the group after the lookahead.
        self.table_tiles = self.total_tiles + self.far_tiles
        self.first_row = self.query.start - kernel.begin

    def key_slot(self, tile):
        if self.k.reuse_keys:
            return 0
        return lax.rem(self.tile_base + tile, self.k.config.key_buffers)

    def page_slot(self, tile):
        return (self.tile_base + tile) & 3

    def output_slot(self, tile):
        return 2 * self.q_slot + (tile & 1)

    def publish(self, tile, wait=False):
        k, slot = self.k, self.output_slot(tile)
        src, sem = k.scores.at[slot], k.write_sem.at[slot]
        dst = k.out.at[:, pl.ds(tile * k.key_tile, k.key_tile)]
        _dma_rows(src, dst, sem, self.query.count, wait, dst_start=self.first_row)

    def owner(self, future):
        """Request row and local tile of a future tile."""
        ends = (self.tile_count, self.tile_count + self.next_tiles, self.total_tiles)
        groups = (self.query, self.following, self.lookahead, self.farthest)
        row = _pick(future, ends, [g.page_row for g in groups])
        return row, future - _pick(future, ends, (0, *ends))

    def load_table(self, future, wait=False):
        k = self.k
        row, tile = self.owner(future)
        slot = self.page_slot(future)
        dst = k.pages.at[pl.ds(slot * k.page_count, k.page_count)]
        dma = pltpu.make_async_copy(k.table.at[row, tile], dst, k.page_sem.at[slot])
        (dma.wait if wait else dma.start)()

    def fill_keys(self):
        # A complete K tile stays resident across query groups of one request.
        self.k.load_pages(0, self.k.tile_pages, self.page_slot(0), 0)
        self.k.wait_keys(0)

    def prefill(self, i):
        page_slot, key_slot = self.page_slot(i), self.key_slot(i)
        self.load_table(i, wait=True)
        self.k.load_pages(0, self.k.half_pages, page_slot, key_slot)
        self.k.load_rest(page_slot, key_slot)

    def key_rows(self, tile, part, field, count):
        """`count` fields from `field` of 128-record part `part` of the tile.

        A page of P parts holds field f of part h at row f * P + h, so a
        part's fields are a sublane-strided load and need no relayout.
        """
        k = self.k
        parts = k.layout.page_size // PAGE_RECORDS
        page, part = divmod(part, parts)
        start = page * k.layout.page_rows + field * parts + part
        rows = pl.ds(start, count, None if parts == 1 else parts)
        return k.keys[self.key_slot(tile), rows]

    def qk(self, tile, fragment):
        k = self.k
        per_fragment = k.mxu_size // PAGE_RECORDS
        pages = range(fragment * per_fragment, (fragment + 1) * per_fragment)
        rows = [self.key_rows(tile, p, 0, PAGE_ROWS - 1) for p in pages]
        rhs = jnp.concatenate([pltpu.bitcast(r, k.q.dtype) for r in rows], axis=1)
        logits = lax.dot(self.lhs, rhs, preferred_element_type=jnp.float32)
        return jnp.maximum(logits.astype(jnp.bfloat16), jnp.bfloat16(0))

    def store(self, tile, fragment, values, remaining):
        k = self.k
        for offset in range(0, k.mxu_size, PAGE_RECORDS):
            local_page = (fragment * k.mxu_size + offset) // PAGE_RECORDS
            page = local_page // (k.layout.page_size // PAGE_RECORDS)
            valid = (k.page_record(page, self.page_slot(tile)) & 2) != 0
            scale = self.key_rows(tile, local_page, PAGE_ROWS - 1, 1)
            scale = lax.bitcast_convert_type(scale, jnp.float32)
            scaled = values[:, offset : offset + PAGE_RECORDS] * scale
            column = local_page * PAGE_RECORDS
            bound = jnp.where(valid, column, jnp.iinfo(jnp.int32).max)
            scaled = jnp.where(remaining > bound, scaled, -jnp.inf)
            k.scores[self.output_slot(tile), :, pl.ds(column, PAGE_RECORDS)] = scaled

    def emit_tile(self, tile, prefetch, count):
        """Score the first `count` fragments of `tile`; the rest are -inf."""
        k = self.k
        metadata, following = self.page_slot(tile + 2), self.key_slot(tile + 2)
        if prefetch:
            self.load_table(tile + 2, wait=True)
        base = tile * k.key_tile + jnp.arange(PAGE_RECORDS)
        remaining = jnp.maximum(self.lengths_vector, 0)[:, None] - base[None, :]
        pending, depth = {}, k.config.issue_depth
        # Half tiles interleave only the first half of the next tile's
        # pages and drop the issue-only stages; the rest follow in a loop.
        split = count < k.fragments
        inline_pages = k.half_pages if split else k.tile_pages
        stages = count + depth if split else max(count + depth, k.fragments)
        page_size = k.layout.page_size
        for stage in range(stages):
            first = min(pl.cdiv(stage * k.mxu_size, page_size), inline_pages)
            last = min(pl.cdiv((stage + 1) * k.mxu_size, page_size), inline_pages)
            middle = (first + last + 1) // 2
            if prefetch:
                k.load_pages(first, middle, metadata, following)
            if stage < count:
                pending[stage] = self.qk(tile, stage)
            if depth <= stage < count + depth:
                fragment = stage - depth
                # Keyed to the tuned depth, so another issue depth keeps each
                # fragment's head-sum arithmetic.
                tuned = fragment + PACKED_FP8_MQA_CONFIG.issue_depth
                mxu = _reduce_on_mxu(fragment, tuned, count, every=4, drain=4)
                products = pending.pop(fragment)
                values = _reduce_heads(
                    products, *self.reduction, sublanes=k.sublanes, use_mxu=mxu
                )
                self.store(tile, fragment, values, remaining)
            if prefetch:
                k.load_pages(middle, last, metadata, following)
        if split:
            start, width = count * k.mxu_size, k.key_tile - count * k.mxu_size
            fill = jnp.full((k.q_tile, width), -jnp.inf, jnp.float32)
            k.scores[self.output_slot(tile), :, pl.ds(start, width)] = fill
            if prefetch:
                k.load_rest(metadata, following)

    def score_tile(self, tile, prefetch):
        # Every tile takes the masked path: the extra selects are free next
        # to the MXU/VPOP bound, and one emit family halves code.
        k = self.k
        if k.half >= k.fragments:
            self.emit_tile(tile, prefetch, k.fragments)
        else:
            short = self.maximum_length - tile * k.key_tile <= k.half * k.mxu_size
            emit = partial(self.emit_tile, tile, prefetch)
            lax.cond(short, partial(emit, k.half), partial(emit, k.fragments))

    def step_tile(self, tile, _):
        # Four slots retain the current mask while prefetching metadata
        # one tile ahead of the two-tile K prefetch.
        pl.when(tile + 3 < self.table_tiles)(partial(self.load_table, tile + 3))
        if not self.k.reuse_keys:
            self.k.wait_keys(self.key_slot(tile))
        pl.when(tile >= 2)(partial(self.publish, tile - 2, wait=True))
        # One branch per tile: a per-page predicate costs scalar issue
        # slots in every fragment stage.
        if self.k.reuse_keys:
            self.score_tile(tile, False)
        else:
            score = partial(self.score_tile, tile)
            prefetch = tile + 2 < self.total_tiles
            lax.cond(prefetch, partial(score, True), partial(score, False))
        self.publish(tile)

    def pad_tile(self, tile, _):
        # Unscored tiles copy from a constant -inf buffer, so their writes
        # need not wait for the scored tiles' writes to drain.
        k = self.k
        dst = k.out.at[:, pl.ds(tile * k.key_tile, k.key_tile)]
        sem = k.pad_sem.at[self.q_slot]
        _dma_rows(k.padding, dst, sem, self.query.count, dst_start=self.first_row)

    def run(self):
        k, query, q_slot = self.k, self.query, self.q_slot
        k.copy_queries(self.following, 1 - q_slot, zero=True)
        self.lengths_vector = jnp.stack(k.row_lengths(query))
        self.maximum_length = reduce(jnp.maximum, k.row_lengths(query))
        # The previous query already prefetched up to three tiles from these
        # three groups. Fill the newly visible part of the farthest group.
        for i in range(3):
            newly = (self.step == 0) | (i >= self.total_tiles)
            pl.when(newly & (i < self.table_tiles))(partial(self.load_table, i))
        k.release_outputs(q_slot)
        scored = self.tile_count > 0
        if k.reuse_keys:
            pl.when(scored)(partial(self.load_table, 0, wait=True))
            # In this mode ready identifies the retained tile's request owner.
            pl.when(scored & (self.ready != query.page_row))(self.fill_keys)
        else:
            for i in range(2):
                needed = (i >= self.ready) & (i < self.total_tiles)
                pl.when(needed)(partial(self.prefill, i))
        k.copy_queries(query, q_slot, wait=True)
        rows = k.row_ids(query)
        weight = jnp.concatenate([k.weights[r][:, None] for r in rows])
        selector = _weighted_selector(k.selector, k.weights, rows, sublanes=k.sublanes)
        self.reduction = (_head_order(weight, k.q_tile, k.sublanes), selector)
        queries = k.queries[pl.ds(q_slot * k.q_tile, k.q_tile)]
        queries = queries.reshape(k.q_tile * k.heads, k.dim)
        self.lhs = _head_order(queries, k.q_tile, k.sublanes)
        if k.tiles == 1:
            pl.when(scored)(partial(self.step_tile, jnp.int32(0), None))
        else:
            lax.fori_loop(0, self.tile_count, self.step_tile, None)

        # The last two tiles' writes stay in flight until this slot pair
        # is reused; tile t publishes from score slot t & 1.
        for local in range(min(k.tiles, 2)):
            live = jnp.where(self.tile_count > local, query.count, 0)
            k.counts[2 * q_slot + local] = live
        lax.fori_loop(self.tile_count, k.tiles, self.pad_tile, None)
        k.counts[6 + q_slot] = (k.tiles - self.tile_count) * query.count

        ready = jnp.minimum(2, self.next_tiles + self.ahead_tiles)
        if k.reuse_keys:
            ready = jnp.where(scored, query.page_row, self.ready)
        tiles = (self.next_tiles, self.ahead_tiles, self.far_tiles)
        step, base = self.step + 1, self.tile_base + self.tile_count
        return self.following, self.lookahead, self.farthest, tiles, step, base, ready


def _kernel(static, offsets, lengths, query_start, inputs, *refs):
    """Score query tiles against paged K; refs are the outputs, then scratch."""
    _DenseKernel((offsets, lengths, query_start), inputs, refs, *static).run()


@jax.jit(
    static_argnames=(
        "cache_layout",
        "max_model_len",
        "query_size",
        "query_count",
        "reuse_keys",
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
) -> jax.Array:
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
    """
    tokens, heads, dim = q.shape
    tokens = tokens if query_count is None else query_count
    banks = kv_cache if isinstance(kv_cache, tuple) else (kv_cache,)
    bank_rows = tuple(bank.shape[0] for bank in banks)
    layout = cache_layout or PackedFP8Layout(sum(bank_rows) // PAGE_ROWS)
    layout.validate(bank_rows)
    if layout.page_size % PAGE_RECORDS:
        raise ValueError(
            f"Dense FP8 scoring requires pages of whole {PAGE_RECORDS}-record parts"
        )
    tile, config = paged_mqa_query_tile(query_size, heads), PACKED_FP8_MQA_CONFIG
    key_tile = paged_mqa_key_tile(max_model_len, layout.page_size, config.key_tile)
    depth = envs.TPU_DEEPSEEK_V41_INDEXER_ISSUE_DEPTH
    config = replace(
        config,
        query_tile=tile,
        key_tile=key_tile,
        issue_depth=depth or config.issue_depth,
    )
    width = pl.cdiv(max_model_len, key_tile) * key_tile
    if tokens == 0 or width == 0:
        return jnp.empty((tokens, max_model_len), jnp.float32)
    info = cast(pltpu.TpuInfo, pltpu.get_tpu_info())
    page_count = width // layout.page_size
    tile_pages = config.key_tile // layout.page_size
    page_ids = block_table[:, :page_count]
    pad = ((0, 0), (0, page_count - page_ids.shape[1]))
    page_ids = jnp.pad(page_ids, pad, constant_values=-1)
    table = _encode_page_table(page_ids, layout, banks[0].shape[0])
    page_count = pl.cdiv(tile_pages, info.num_lanes) * info.num_lanes
    table = table.reshape(page_ids.shape[0], -1, tile_pages)
    table = jnp.pad(table, ((0, 0), (0, 0), (0, page_count - tile_pages)))
    dma, key_rows = pltpu.SemaphoreType.DMA, tile_pages * layout.page_rows
    scratch = (
        (
            pltpu.VMEM((2 * tile, heads, dim), q.dtype),
            pltpu.VMEM((config.key_buffers, key_rows, dim), jnp.int32),
            pltpu.VMEM((4, tile, config.key_tile), jnp.float32),
            pltpu.SMEM((8,), jnp.int32),
            pltpu.SMEM((4 * page_count,), jnp.int32),
            pltpu.VMEM((tile, config.key_tile), jnp.float32),
        ),
        tuple(dma(s) for s in ((config.key_buffers,), (4, tile), (2,), (4,), (2,))),
    )
    hbm, vmem = (pl.BlockSpec(memory_space=s) for s in (pltpu.HBM, pltpu.VMEM))
    inputs = (q, weights * q_scale, banks, table)
    reuse_keys = reuse_keys and width == config.key_tile
    kernel = partial(_kernel, (config, info, layout, reuse_keys))
    out_shape = jax.ShapeDtypeStruct((tokens, width), jnp.float32)
    result = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(1,),
            in_specs=((hbm, vmem, tuple(hbm for _ in banks), hbm),),
            out_specs=hbm,
            scratch_shapes=(scratch,),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=info.vmem_capacity_bytes,
            disable_bounds_checks=True,
        ),
        name="fp8_paged_mqa_logits_dense",
    )(query_start_loc, context_lens, jnp.asarray(query_start, jnp.int32), inputs)
    return result[:, :max_model_len]
