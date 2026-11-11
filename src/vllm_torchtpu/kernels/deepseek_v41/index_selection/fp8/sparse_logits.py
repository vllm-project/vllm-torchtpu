# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 pair-union scoring with contiguous per-query output windows."""

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
    _reduce_on_mxu,
    _weighted_selector,
)
from .tuned import (
    PACKED_FP8_PREPARED_MQA_CONFIG,
    PACKED_FP8_PREPARED_PAIRS,
    PACKED_FP8_PREPARED_PREFIX_TOKENS,
)

# Decode batches up to this size keep every query resident in VMEM, so no
# per-row q DMAs queue between the large K transfers.
_RESIDENT_QUERY_BYTES = 4 << 20


def candidate_windows(
    candidate_block_ids: jax.Array,
    block_spans: jax.Array,
    lengths: jax.Array,
    block: int,
) -> jax.Array:
    """Two causal prefixes in union coordinates: start, split, end1, end2."""
    start, split = block_spans[:, 0], block_spans[:, 1]
    visible = jnp.where(
        candidate_block_ids >= 0,
        jnp.clip(lengths[:, None] - candidate_block_ids * block, 0, block),
        0,
    )
    first = jnp.arange(candidate_block_ids.shape[1])[None, :] < split[:, None]
    return jnp.stack(
        (
            start * block,
            (start + split) * block,
            start * block + jnp.sum(jnp.where(first, visible, 0), axis=1),
            (start + split) * block + jnp.sum(jnp.where(first, 0, visible), axis=1),
        ),
        axis=1,
    )


def _kernel(
    offsets, windows, query_start, inputs, out, scratch, *, config, info, resident
):
    q, weights, pool = inputs
    queries, keys, scores, counts, read_sem, write_sem, q_sem = scratch
    keys = pltpu.annotate(keys, no_store=True)
    weights = pltpu.annotate(weights, no_store=True)
    tokens, heads, dim = weights.shape[0], weights.shape[1], q.shape[-1]
    q_tile, mxu, sublanes = config.query_tile, info.mxu_column_size, info.num_sublanes
    pairs = PACKED_FP8_PREPARED_PAIRS
    begin, width = query_start[()], out.shape[1]
    group_rows = pool.shape[0] // (offsets.shape[0] - 1)
    next_query = partial(
        _next_query_tile, offsets=offsets, tile_size=q_tile, end=begin + out.shape[0]
    )
    lanes = jnp.arange(PAGE_RECORDS)
    owners = _head_order(
        jnp.repeat(jnp.arange(q_tile), heads)[:, None], q_tile, sublanes
    )[:, 0]
    selector = (jnp.arange(q_tile)[:, None] == owners[None, :]).astype(jnp.bfloat16)
    # Keys load two iterations ahead when the buffers hold three groups.
    key_ahead = 2 if 3 * pairs <= config.key_buffers else 1

    def rows(query):
        return [
            jnp.minimum(offsets[query.page_row] + i, tokens - 1) for i in range(q_tile)
        ]

    def limit(query):
        return reduce(
            jnp.maximum,
            [
                jnp.where(
                    (query.count > 0)
                    & (i < offsets[query.page_row + 1] - offsets[query.page_row]),
                    jnp.maximum(windows[r * 4 + 2], windows[r * 4 + 3]),
                    0,
                )
                for i, r in enumerate(rows(query))
            ],
        )

    def slot_of(step, pair):
        # Query/score slots alternate by iteration: the previous iteration's
        # scores stay live until their deferred finish.
        return (step & 1) * pairs + pair

    def key_buffer(step, pair):
        return (step * pairs + pair) % config.key_buffers

    def load_query(query, slot, wait=False):
        if resident:
            return
        destination = queries.at[pl.ds(slot * q_tile, q_tile)]
        if not wait:
            destination[...] = jnp.zeros(destination.shape, destination.dtype)
        _dma_rows(
            q,
            destination,
            q_sem.at[slot],
            jnp.where(
                query.count > 0,
                offsets[query.page_row + 1] - offsets[query.page_row],
                0,
            ),
            wait,
            src_start=offsets[query.page_row],
        )

    def load_keys(query, buffer, wait=False):
        count = pl.cdiv(limit(query), PAGE_RECORDS) * PAGE_ROWS

        @pl.when(count > 0)
        def transfer():
            copy = pltpu.make_async_copy(
                pool.at[pl.ds(query.page_row * group_rows, count)],
                keys.at[buffer, pl.ds(0, count)],
                read_sem.at[buffer],
            )
            (copy.wait if wait else copy.start)()

    def output(slot):
        return scores.at[slot, :, pl.ds(0, width)]

    def release(slot):
        src = output(slot)
        _dma_rows(src, src, write_sem.at[slot], counts[slot], wait=True)

    def visible(row, offset):
        # Window mask in output columns; column zero is union token offset.
        start, split, end1, end2 = (windows[row * 4 + i] - offset for i in range(4))
        shape = (width // PAGE_RECORDS, PAGE_RECORDS)
        positions = lax.broadcasted_iota(jnp.int32, shape, 0) * PAGE_RECORDS
        positions += lax.broadcasted_iota(jnp.int32, shape, 1)
        return ((positions >= start) & (positions < end1)) | (
            (positions >= split) & (positions < end2)
        )

    def finish(query, slot):
        # Rotate the second row to its window start, mask both rows and publish.
        # Every output column is overwritten, so the buffer needs no -inf fill.
        query_rows = rows(query)
        start = windows[query_rows[1] * 4]
        aligned_start = start // PAGE_RECORDS * PAGE_RECORDS
        window = scores[
            slot,
            1:2,
            pl.ds(pl.multiple_of(aligned_start, PAGE_RECORDS), width + PAGE_RECORDS),
        ]
        shift = PAGE_RECORDS - start % PAGE_RECORDS
        rotated = pltpu.roll(window.reshape(-1, PAGE_RECORDS), shift, axis=1)
        rotated = jnp.where(lanes[None, :] < shift, rotated[:-1], rotated[1:])
        # Dense [pages, 128] masks cost far less than per-fragment [2, 128] ones.
        scores[slot, 1:2, pl.ds(0, width)] = jnp.where(
            visible(query_rows[1], start), rotated, -jnp.inf
        ).reshape(1, -1)
        first = scores[slot, 0:1, pl.ds(0, width)].reshape(-1, PAGE_RECORDS)
        scores[slot, 0:1, pl.ds(0, width)] = jnp.where(
            visible(query_rows[0], 0), first, -jnp.inf
        ).reshape(1, -1)
        _dma_rows(
            output(slot),
            out,
            write_sem.at[slot],
            query.count,
            src_start=query.start - offsets[query.page_row],
            dst_start=query.start - begin,
        )
        counts[slot] = query.count

    def key_parts(buffer, fragment):
        return [
            pltpu.bitcast(
                keys[
                    buffer,
                    pl.ds(
                        (fragment * mxu + offset) // PAGE_RECORDS * PAGE_ROWS,
                        PAGE_ROWS - 1,
                    ),
                ],
                q.dtype,
            )
            for offset in range(0, mxu, PAGE_RECORDS)
        ]

    def body(state):
        tiles, step, previous = state
        group = tiles[:pairs]
        for p, query in enumerate(group):
            load_query(query, slot_of(step, p), wait=True)
            load_keys(query, key_buffer(step, p), wait=True)
            release(slot_of(step, p))
        for p, query in enumerate(tiles[pairs : 2 * pairs]):
            load_query(query, slot_of(step + 1, p))
        for p, query in enumerate(tiles[key_ahead * pairs :]):
            load_keys(query, key_buffer(step + key_ahead, p))

        def finish_previous():
            for p, query in enumerate(previous):
                finish(query, slot_of(step - 1, p))

        def context(p, query):
            slot, buffer = slot_of(step, p), key_buffer(step, p)
            query_rows = rows(query)
            weight = jnp.concatenate([weights[r][:, None] for r in query_rows])
            head_weights = _head_order(weight, q_tile, sublanes)
            mxu_selector = _weighted_selector(
                selector, weights, query_rows, sublanes=sublanes
            )
            # A short group's resident second row is the next group's query;
            # its scores are never published and K limits ignore it.
            source = (
                jnp.concatenate([q[r][None] for r in query_rows])
                if resident
                else queries[pl.ds(slot * q_tile, q_tile)]
            )
            lhs = _head_order(source.reshape(q_tile * heads, dim), q_tile, sublanes)

            def qk(fragment):
                parts = key_parts(buffer, fragment)
                if fragment < 2:
                    # One matmul per K page: the first issues after one page loads.
                    logits = jnp.concatenate(
                        [
                            lax.dot(lhs, part, preferred_element_type=jnp.float32)
                            for part in parts
                        ],
                        axis=1,
                    )
                else:
                    logits = lax.dot(
                        lhs,
                        jnp.concatenate(parts, axis=1),
                        preferred_element_type=jnp.float32,
                    )
                return jnp.maximum(logits.astype(jnp.bfloat16), jnp.bfloat16(0))

            def store(products, fragment, stage, count):
                values = _reduce_heads(
                    products,
                    head_weights,
                    mxu_selector,
                    sublanes=sublanes,
                    # Stagger the MXU reductions of the group's pairs.
                    use_mxu=_reduce_on_mxu(
                        fragment * pairs + p, stage, count, every=5, drain=2
                    ),
                )
                for offset in range(0, mxu, PAGE_RECORDS):
                    page = (fragment * mxu + offset) // PAGE_RECORDS
                    scale = lax.bitcast_convert_type(
                        keys[buffer, pl.ds((page + 1) * PAGE_ROWS - 1, 1)], jnp.float32
                    )
                    # Unmasked; windows are applied once per output row.
                    scores[slot, :, pl.ds(page * PAGE_RECORDS, PAGE_RECORDS)] = (
                        values[:, offset : offset + PAGE_RECORDS] * scale
                    )

            return qk, store

        contexts = [context(p, query) for p, query in enumerate(group)]

        def emit(count):
            # The group's fragments share one issue window, so MXU fill and
            # drain latency are paid once per group instead of once per pair.
            depth, pending = config.issue_depth, {}
            for stage in range(count + depth):
                if stage < count:
                    for p, (qk, _) in enumerate(contexts):
                        pending[p, stage] = qk(stage)
                if stage == min(count, depth) - 1:
                    # The previous group's masks fill this group's MXU fill.
                    finish_previous()
                if stage >= depth:
                    for p, (_, store) in enumerate(contexts):
                        fragment = stage - depth
                        store(pending.pop((p, fragment)), fragment, stage, count)

        prefix = min(config.key_tile, PACKED_FP8_PREPARED_PREFIX_TOKENS) // mxu
        total = config.key_tile // mxu
        branches = tuple(
            partial(emit, min(end, total))
            for end in range(prefix, total + prefix, prefix)
        )
        # Shorter pairs score stale K past their limit; their masks drop it.
        span = reduce(jnp.maximum, [limit(query) for query in group])
        live = span > 0
        pl.when(live)(lambda: lax.switch(pl.cdiv(span, prefix * mxu) - 1, branches))
        pl.when(~live)(finish_previous)
        following = list(tiles[pairs:])
        for _ in range(pairs):
            following.append(next_query(following[-1]))
        return tuple(following), step + 1, tuple(group)

    zero = jnp.int32(0)
    tiles = [next_query(QueryTile(begin, zero, zero))]
    while len(tiles) < (key_ahead + 1) * pairs:
        tiles.append(next_query(tiles[-1]))
    for slot in range(2 * pairs):
        counts[slot] = zero
    for p in range(pairs):
        load_query(tiles[p], slot_of(zero, p))
    for g in range(key_ahead):
        for p in range(pairs):
            load_keys(tiles[g * pairs + p], key_buffer(jnp.int32(g), p))
    placeholder = tuple(tiles[0]._replace(count=zero) for _ in range(pairs))
    _, steps, last = lax.while_loop(
        lambda state: state[0][0].start < begin + out.shape[0],
        body,
        (tuple(tiles), zero, placeholder),
    )
    # The first iteration finished empty placeholders; finish the last group.
    for p, query in enumerate(last):
        finish(query, slot_of(steps - 1, p))
    for slot in range(2 * pairs):
        release(slot)


@jax.jit(static_argnames=("output_width", "query_count"))
def fp8_gathered_mqa_logits(
    q: jax.Array,
    keys: jax.Array,
    weights: jax.Array,
    windows: jax.Array,
    *,
    query_start_loc: jax.Array,
    output_width: int,
    query_start: jax.Array | int = 0,
    query_count: int | None = None,
) -> jax.Array:
    """Score pair unions and emit fixed-width contiguous per-query windows.

    ``windows[query]`` contains union-token coordinates [start, split, end1,
    end2]. Each query owns [start, end1) and [split, end2); all other scores
    are -inf. Output column zero corresponds to ``start``. K retains one
    equal-sized packed union region per group, independently of output width.
    """
    rows = q.shape[0] if query_count is None else query_count
    if rows == 0 or output_width == 0:
        return jnp.empty((rows, output_width), jnp.float32)
    info = cast(pltpu.TpuInfo, pltpu.get_tpu_info())
    groups = query_start_loc.shape[0] - 1
    group_rows = keys.shape[0] // groups
    if keys.shape != (groups * group_rows, PAGE_RECORDS) or group_rows % PAGE_ROWS:
        raise ValueError("Gathered K must contain equal packed regions per group")
    if windows.shape != (q.shape[0], 4) or windows.dtype != jnp.int32:
        raise ValueError("Query windows must be int32 [queries, 4]")
    capacity = group_rows // PAGE_ROWS * PAGE_RECORDS
    if output_width > capacity:
        raise ValueError("Output width exceeds gathered K capacity")
    tile = pl.cdiv(capacity, info.mxu_column_size) * info.mxu_column_size
    padded_rows = tile // PAGE_RECORDS * PAGE_ROWS
    keys = jnp.pad(
        keys.reshape(groups, group_rows, PAGE_RECORDS),
        ((0, 0), (0, padded_rows - group_rows), (0, 0)),
    ).reshape(-1, PAGE_RECORDS)
    config = replace(PACKED_FP8_PREPARED_MQA_CONFIG, key_tile=tile)
    width = pl.cdiv(output_width, PAGE_RECORDS) * PAGE_RECORDS
    q_tile, heads, dim = config.query_tile, q.shape[1], q.shape[2]
    if q_tile != 2:
        raise ValueError("Contiguous pair scoring requires a two-query tile")
    # Two iterations of query/score slots per pair: score one, finish the other.
    slots = 2 * PACKED_FP8_PREPARED_PAIRS
    if config.key_buffers < slots:
        raise ValueError("Pair groups need two K buffers per pair")
    resident = q.size * q.dtype.itemsize <= _RESIDENT_QUERY_BYTES
    scratch = (
        pltpu.VMEM((slots * q_tile, heads, dim), q.dtype),
        pltpu.VMEM((config.key_buffers, padded_rows, dim), jnp.int32),
        pltpu.VMEM((slots, q_tile, tile + PAGE_RECORDS), jnp.float32),
        pltpu.SMEM((slots,), jnp.int32),
        pltpu.SemaphoreType.DMA((config.key_buffers,)),
        pltpu.SemaphoreType.DMA((slots, q_tile)),
        pltpu.SemaphoreType.DMA((slots,)),
    )
    hbm, vmem = (pl.BlockSpec(memory_space=s) for s in (pltpu.HBM, pltpu.VMEM))
    return pl.pallas_call(
        partial(_kernel, config=config, info=info, resident=resident),
        out_shape=jax.ShapeDtypeStruct((rows, width), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(1,),
            in_specs=((vmem if resident else hbm, vmem, hbm),),
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
        # Flatten to avoid padding each four-int row to an SMEM lane boundary.
        windows.reshape(-1),
        jnp.asarray(query_start, jnp.int32),
        (q, weights, keys),
    )[:, :output_width]
