# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Masked FP8 MQA logits over contiguous candidate unions after gathering."""

from dataclasses import replace
from functools import partial, reduce
from typing import cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .cache import PAGE_RECORDS, PAGE_ROWS
from .common import (
    QueryTile,
    _dma_rows,
    _head_order,
    _next_query_tile,
    _reduce_heads,
)
from .tuned import (
    PACKED_FP8_PREPARED_MQA_CONFIG,
    PACKED_FP8_PREPARED_PREFIX_TOKENS,
)


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
    membership_block_size,
):
    mxu_size, sublanes = info.mxu_column_size, info.num_sublanes
    q, weights, pool, block_owners = inputs
    buffers, semaphores = scratch
    queries, keys, scores, counts = buffers
    read_sem, write_sem, q_sem = semaphores
    weights = pltpu.annotate(weights, no_store=True)
    keys = pltpu.annotate(keys, no_store=True)
    tokens, heads, dim = q.shape
    begin = query_start[()]
    end = begin + out.shape[0]
    q_tile, key_tile = config.query_tile, config.key_tile
    tile_rows = key_tile // PAGE_RECORDS * PAGE_ROWS
    group_rows = out.shape[1] // PAGE_RECORDS * PAGE_ROWS
    tiles = out.shape[1] // key_tile

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

    def key_rows_to_read(tile, limit):
        valid = jnp.minimum(key_tile, jnp.maximum(limit - tile * key_tile, 0))
        return pl.cdiv(valid, PAGE_RECORDS) * PAGE_ROWS

    def load_keys(tile, group, slot, limit):
        count = key_rows_to_read(tile, limit)
        pltpu.make_async_copy(
            pool.at[pl.ds(group * group_rows + tile * tile_rows, count)],
            keys.at[slot, pl.ds(0, count)],
            read_sem.at[slot],
        ).start()

    next_query = partial(_next_query_tile, offsets=offsets, tile_size=q_tile, end=end)

    def row_ids(query):
        return [jnp.minimum(query.start + i, tokens - 1) for i in range(q_tile)]

    def row_lengths(query):
        rows = enumerate(row_ids(query))
        return [jnp.where(i < query.count, lengths[r], 0) for i, r in rows]

    def query_limit(query):
        limit = reduce(jnp.maximum, row_lengths(query), jnp.int32(0))
        return jnp.minimum(limit, out.shape[1])

    def live_tiles(query):
        return pl.cdiv(query_limit(query), key_tile)

    owners = _head_order(
        jnp.repeat(jnp.arange(q_tile), heads)[:, None], q_tile, sublanes
    )[:, 0]
    selector = (jnp.arange(q_tile)[:, None] == owners[None, :]).astype(jnp.bfloat16)

    def query_block(state):
        query, following, lookahead, tile_counts, step, key_base, ready = state
        tile_count, next_tiles, ahead_tiles = tile_counts
        q_slot = step & 1
        farthest = next_query(lookahead)

        def key_slot(tile):
            return (key_base + tile) % config.key_buffers

        def output_slot(tile):
            return 2 * q_slot + (tile & 1)

        copy_queries(following, 1 - q_slot, zero=True)
        lengths_vector = jnp.stack(row_lengths(query))
        query_bits = 1 << jnp.minimum(
            query.start - offsets[query.page_row] + jnp.arange(q_tile), 1
        )

        def publish(tile, wait=False, slot=None):
            slot = output_slot(tile) if slot is None else slot
            src, sem = scores.at[slot], write_sem.at[slot]
            dst = out.at[:, pl.ds(tile * key_tile, key_tile)]
            _dma_rows(src, dst, sem, query.count, wait, dst_start=query.start - begin)

        def tile_source(future):
            first, after = future < tile_count, future - tile_count
            third = after >= next_tiles
            return (
                jnp.where(first, future, after - jnp.where(third, next_tiles, 0)),
                jnp.where(
                    first,
                    query.page_row,
                    jnp.where(third, lookahead.page_row, following.page_row),
                ),
                jnp.where(
                    first,
                    query_limit(query),
                    jnp.where(third, query_limit(lookahead), query_limit(following)),
                ),
            )

        release_outputs(q_slot)
        for i in range(2):

            @pl.when((i >= ready) & (i < tile_count + next_tiles + ahead_tiles))
            def prefill(i=i):
                target, owner, limit = tile_source(i)
                load_keys(target, owner, key_slot(i), limit)

        copy_queries(query, q_slot, wait=True)

        def emit_tile(tile):
            weight = jnp.concatenate([weights[r][:, None] for r in row_ids(query)])
            head_weights = _head_order(weight, q_tile, sublanes)
            query_slice = pl.ds(q_slot * q_tile, q_tile)
            lhs = _head_order(
                queries[query_slice].reshape(q_tile * heads, dim), q_tile, sublanes
            )

            def key_rows(start, count):
                return keys[key_slot(tile), pl.ds(start, count)]

            base = tile * key_tile + jnp.arange(PAGE_RECORDS)
            remaining = jnp.maximum(lengths_vector, 0)[:, None] - base[None, :]

            def qk(fragment):
                fragment = jnp.asarray(fragment, jnp.uint32)
                parts = []
                for offset in range(0, mxu_size, PAGE_RECORDS):
                    page = (fragment * mxu_size + offset) // PAGE_RECORDS
                    words = key_rows(page * PAGE_ROWS, PAGE_ROWS - 1)
                    parts.append(pltpu.bitcast(words, q.dtype))
                logits = lax.dot(
                    lhs,
                    jnp.concatenate(parts, axis=1),
                    preferred_element_type=jnp.float32,
                )
                return jnp.maximum(logits.astype(jnp.bfloat16), jnp.bfloat16(0))

            def store(fragment, values):
                fragment = jnp.asarray(fragment, jnp.uint32)
                for offset in range(0, mxu_size, PAGE_RECORDS):
                    local_page = (fragment * mxu_size + offset) // PAGE_RECORDS
                    scale_row = (local_page + 1) * PAGE_ROWS - 1
                    scale = lax.bitcast_convert_type(
                        key_rows(scale_row, 1), jnp.float32
                    )
                    scaled = values[:, offset : offset + PAGE_RECORDS] * scale
                    column = local_page * PAGE_RECORDS
                    columns = (
                        tile * key_tile + column + jnp.arange(PAGE_RECORDS)
                    ) // membership_block_size
                    first_flag = (
                        (tile * key_tile + column)
                        // membership_block_size
                        // info.num_lanes
                        * info.num_lanes
                    )
                    local_flags = block_owners[
                        query.page_row,
                        0,
                        pl.ds(
                            pl.multiple_of(first_flag, info.num_lanes),
                            info.num_lanes,
                        ),
                    ]
                    flags = jnp.take_along_axis(
                        jnp.broadcast_to(local_flags, (q_tile, info.num_lanes)),
                        jnp.broadcast_to(
                            (columns - first_flag)[None, :], (q_tile, PAGE_RECORDS)
                        ),
                        axis=1,
                    )
                    allowed = (flags & query_bits[:, None]) != 0
                    scaled = jnp.where(allowed & (remaining > column), scaled, -jnp.inf)
                    scores[output_slot(tile), :, pl.ds(column, PAGE_RECORDS)] = scaled

            valid_keys = jnp.minimum(key_tile, query_limit(query) - tile * key_tile)
            fragments = pl.cdiv(valid_keys, mxu_size)
            scores[output_slot(tile)] = jnp.full(
                scores.shape[1:], -jnp.inf, jnp.float32
            )

            def emit_fragments(base, count):
                pending = {}
                for stage in range(count + config.issue_depth):
                    if stage < count:
                        pending[stage] = qk(base + stage)
                    if stage >= config.issue_depth:
                        fragment = stage - config.issue_depth
                        values = _reduce_heads(
                            pending.pop(fragment) * head_weights,
                            selector,
                            sublanes=sublanes,
                            use_mxu=stage < count and bool(fragment % 2),
                        )
                        store(base + fragment, values)

            # Static fragment offsets fold page/mask addressing at compile time.
            # Select one long prefix instead of restarting short pipelines.
            prefix = min(key_tile, PACKED_FP8_PREPARED_PREFIX_TOKENS) // mxu_size
            total = key_tile // mxu_size
            branches = tuple(
                partial(emit_fragments, 0, min(end, total))
                for end in range(prefix, total + prefix, prefix)
            )
            lax.switch(pl.cdiv(fragments, prefix) - 1, branches)

        def step_tile(tile, _):
            count = key_rows_to_read(tile, query_limit(query))
            src = keys.at[key_slot(tile), pl.ds(0, count)]
            pltpu.make_async_copy(src, src, read_sem.at[key_slot(tile)]).wait()
            pl.when(tile >= 2)(lambda: publish(tile - 2, wait=True))

            @pl.when(tile + 2 < tile_count + next_tiles + ahead_tiles)
            def prefetch():
                target, owner, limit = tile_source(tile + 2)
                load_keys(target, owner, key_slot(tile + 2), limit)

            emit_tile(tile)
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
            key_slot(tile_count),
            jnp.minimum(2, next_tiles + ahead_tiles),
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
            zero,
        ),
    )
    release_outputs(0)
    release_outputs(1)


@jax.jit(
    static_argnames=(
        "membership_block_size",
        "query_count",
        "key_tile",
    )
)
def fp8_gathered_mqa_logits(
    q: jax.Array,
    keys: jax.Array,
    weights: jax.Array,
    context_lens: jax.Array,
    block_owners: jax.Array,
    *,
    query_start_loc: jax.Array,
    membership_block_size: int,
    query_start: jax.Array | int = 0,
    query_count: int | None = None,
    key_tile: int | None = None,
) -> jax.Array:
    """Score gathered FP8 K with packed block ownership for query pairs.

    Args:
        q: FP8 queries [queries, heads, dim].
        keys: Packed int32 K [groups * group_rows, 128], with equal-sized
            regions per group; each 33x128 page includes FP8 K and FP32 scales.
        weights: Per-head weights [queries, heads], including query scales.
        context_lens: Visible token counts [queries] in sorted union order.
        block_owners: Int32 [groups, union_blocks]; bit 0/1 marks membership
            for the first/second query of each group. Padding has no bits set.
        query_start_loc: Query offsets [groups + 1]; group g owns queries
            [query_start_loc[g], query_start_loc[g + 1]) and shares gathered K.
            Each nonempty group contains one or two queries.
        membership_block_size: Tokens covered by each membership column.
        query_start: First query row to score.
        query_count: Number of query rows to score; None uses q.shape[0].
        key_tile: K tokens per tile; None uses the tuned configuration.

    Returns:
        Float32 logits [rows, union_blocks * membership_block_size] in gathered
        union order, where rows is query_count if provided, else q.shape[0].
        Invisible or nonmember tokens are set to -inf.
    """
    rows = q.shape[0] if query_count is None else query_count
    score_width = block_owners.shape[1] * membership_block_size
    if rows == 0 or score_width == 0:
        return jnp.empty((rows, score_width), jnp.float32)
    info = cast(pltpu.TpuInfo, pltpu.get_tpu_info())
    groups = query_start_loc.shape[0] - 1
    if block_owners.dtype != jnp.int32 or block_owners.shape[0] != groups:
        raise ValueError("Block ownership must be int32 [groups, union_blocks]")
    group_rows = keys.shape[0] // groups
    capacity = group_rows // PAGE_ROWS * PAGE_RECORDS
    if keys.shape != (groups * group_rows, PAGE_RECORDS) or group_rows % PAGE_ROWS:
        raise ValueError("Gathered K must contain equal packed regions per group")
    if score_width > capacity:
        raise ValueError("Score width exceeds gathered K capacity")
    config = PACKED_FP8_PREPARED_MQA_CONFIG
    tile = min(
        config.key_tile if key_tile is None else key_tile,
        pl.cdiv(score_width, info.mxu_column_size) * info.mxu_column_size,
    )
    if tile % info.mxu_column_size or tile % membership_block_size:
        raise ValueError("K tile must align to MXU columns and membership blocks")
    config = replace(config, key_tile=tile)
    width = pl.cdiv(score_width, tile) * tile
    padded_rows = width // PAGE_RECORDS * PAGE_ROWS
    keys = keys.reshape(groups, group_rows, PAGE_RECORDS)[:, :padded_rows]
    keys = jnp.pad(keys, ((0, 0), (0, max(0, padded_rows - group_rows)), (0, 0)))
    keys = keys.reshape(-1, PAGE_RECORDS)
    columns = pl.cdiv(width // membership_block_size, info.num_lanes) * info.num_lanes
    block_owners = jnp.pad(
        block_owners[:, :columns],
        ((0, 0), (0, max(0, columns - block_owners.shape[1]))),
    )
    # Keep the dynamic group index outside the tiled sublane/lane dimensions.
    block_owners = block_owners[:, None, :]
    q_tile, heads, dim = config.query_tile, q.shape[1], q.shape[2]
    scratch = (
        (
            pltpu.VMEM((2 * q_tile, heads, dim), q.dtype),
            pltpu.VMEM(
                (config.key_buffers, tile // PAGE_RECORDS * PAGE_ROWS, dim), jnp.int32
            ),
            pltpu.VMEM((4, q_tile, tile), jnp.float32),
            pltpu.SMEM((2,), jnp.int32),
        ),
        tuple(
            pltpu.SemaphoreType.DMA(shape)
            for shape in ((config.key_buffers,), (4, q_tile), (2,))
        ),
    )
    hbm, vmem = (pl.BlockSpec(memory_space=s) for s in (pltpu.HBM, pltpu.VMEM))
    return pl.pallas_call(
        partial(
            _kernel,
            config=config,
            info=info,
            membership_block_size=membership_block_size,
        ),
        out_shape=jax.ShapeDtypeStruct((rows, width), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(1,),
            in_specs=((hbm, vmem, hbm, vmem),),
            out_specs=hbm,
            scratch_shapes=(scratch,),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=info.vmem_capacity_bytes,
            disable_bounds_checks=True,
        ),
        name="fp8_gathered_mqa_logits",
    )(
        query_start_loc,
        context_lens,
        jnp.asarray(query_start, jnp.int32),
        (q, weights, keys, block_owners),
    )[:, :score_width]
