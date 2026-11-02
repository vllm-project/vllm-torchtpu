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
from .common import QueryTile, _dma_rows, _head_order, _next_query_tile, _reduce_heads
from .tuned import PACKED_FP8_PREPARED_MQA_CONFIG, PACKED_FP8_PREPARED_PREFIX_TOKENS


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


def _kernel(offsets, windows, query_start, inputs, out, scratch, *, config, info):
    q, weights, pool = inputs
    queries, keys, scores, counts, read_sem, write_sem, q_sem = scratch
    keys = pltpu.annotate(keys, no_store=True)
    weights = pltpu.annotate(weights, no_store=True)
    tokens, heads, dim = q.shape
    q_tile, mxu = config.query_tile, info.mxu_column_size
    begin, width = query_start[()], out.shape[1]
    group_rows = pool.shape[0] // (offsets.shape[0] - 1)
    next_query = partial(
        _next_query_tile, offsets=offsets, tile_size=q_tile, end=begin + out.shape[0]
    )
    lanes = jnp.arange(PAGE_RECORDS)
    owners = _head_order(
        jnp.repeat(jnp.arange(q_tile), heads)[:, None], q_tile, info.num_sublanes
    )[:, 0]
    selector = (jnp.arange(q_tile)[:, None] == owners[None, :]).astype(jnp.bfloat16)

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

    def load_query(query, step, wait=False):
        destination = queries.at[pl.ds((step & 1) * q_tile, q_tile)]
        if not wait:
            destination[...] = jnp.zeros(destination.shape, destination.dtype)
        _dma_rows(
            q,
            destination,
            q_sem.at[step & 1],
            jnp.where(
                query.count > 0,
                offsets[query.page_row + 1] - offsets[query.page_row],
                0,
            ),
            wait,
            src_start=offsets[query.page_row],
        )

    def load_keys(query, step, wait=False):
        count = pl.cdiv(limit(query), PAGE_RECORDS) * PAGE_ROWS

        @pl.when(count > 0)
        def transfer():
            copy = pltpu.make_async_copy(
                pool.at[pl.ds(query.page_row * group_rows, count)],
                keys.at[step % config.key_buffers, pl.ds(0, count)],
                read_sem.at[step % config.key_buffers],
            )
            (copy.wait if wait else copy.start)()

    def output(slot):
        return scores.at[slot, :, pl.ds(0, width)]

    def release(slot):
        src = output(slot)
        _dma_rows(src, src, write_sem.at[slot], counts[slot], wait=True)

    def body(state):
        query, following, lookahead, step = state
        slot = step & 1
        load_query(query, step, wait=True)
        load_keys(query, step, wait=True)
        release(slot)
        load_query(following, step + 1)
        load_keys(lookahead, step + 2)
        scores[slot] = jnp.full(scores.shape[1:], -jnp.inf, jnp.float32)
        query_rows = rows(query)
        weight = jnp.concatenate([weights[r][:, None] for r in query_rows])
        head_weights = _head_order(weight, q_tile, info.num_sublanes)
        lhs = _head_order(
            queries[pl.ds(slot * q_tile, q_tile)].reshape(q_tile * heads, dim),
            q_tile,
            info.num_sublanes,
        )

        def key_rows(start, count):
            return keys[step % config.key_buffers, pl.ds(start, count)]

        bounds = [
            jnp.stack([windows[row * 4 + i] for row in query_rows])[:, None]
            for i in range(4)
        ]

        def store_page(page, values):
            positions = page * PAGE_RECORDS + lanes[None, :]
            start, split, end1, end2 = bounds
            visible = ((positions >= start) & (positions < end1)) | (
                (positions >= split) & (positions < end2)
            )
            scores[slot, :, pl.ds(page * PAGE_RECORDS, PAGE_RECORDS)] = jnp.where(
                visible, values, -jnp.inf
            )

        def qk(fragment):
            parts = [
                pltpu.bitcast(
                    key_rows(
                        (fragment * mxu + offset) // PAGE_RECORDS * PAGE_ROWS,
                        PAGE_ROWS - 1,
                    ),
                    q.dtype,
                )
                for offset in range(0, mxu, PAGE_RECORDS)
            ]
            logits = lax.dot(
                lhs, jnp.concatenate(parts, axis=1), preferred_element_type=jnp.float32
            )
            return jnp.maximum(logits.astype(jnp.bfloat16), jnp.bfloat16(0))

        def emit(count):
            pending = {}
            for stage in range(count + config.issue_depth):
                if stage < count:
                    pending[stage] = qk(stage)
                if stage >= config.issue_depth:
                    fragment = stage - config.issue_depth
                    values = _reduce_heads(
                        pending.pop(fragment) * head_weights,
                        selector,
                        sublanes=info.num_sublanes,
                        use_mxu=stage < count and bool(fragment % 2),
                    )
                    for offset in range(0, mxu, PAGE_RECORDS):
                        page = (fragment * mxu + offset) // PAGE_RECORDS
                        scale = lax.bitcast_convert_type(
                            key_rows((page + 1) * PAGE_ROWS - 1, 1), jnp.float32
                        )
                        store_page(
                            page, values[:, offset : offset + PAGE_RECORDS] * scale
                        )

        prefix = min(config.key_tile, PACKED_FP8_PREPARED_PREFIX_TOKENS) // mxu
        total = config.key_tile // mxu
        branches = tuple(
            partial(emit, min(end, total))
            for end in range(prefix, total + prefix, prefix)
        )
        pl.when(limit(query) > 0)(
            lambda: lax.switch(pl.cdiv(limit(query), prefix * mxu) - 1, branches)
        )
        # One contiguous load/rotation after MXU issue avoids carrying a
        # partial output page through every static dot-product fragment.
        start = windows[query_rows[1] * 4]
        aligned_start = start // PAGE_RECORDS * PAGE_RECORDS
        window = scores[
            slot,
            1:2,
            pl.ds(pl.multiple_of(aligned_start, PAGE_RECORDS), width + PAGE_RECORDS),
        ]
        shift = PAGE_RECORDS - start % PAGE_RECORDS
        rotated = pltpu.roll(window.reshape(-1, PAGE_RECORDS), shift, axis=1)
        scores[slot, 1:2, pl.ds(0, width)] = jnp.where(
            lanes[None, :] < shift, rotated[:-1], rotated[1:]
        ).reshape(1, width)
        _dma_rows(
            output(slot),
            out,
            write_sem.at[slot],
            query.count,
            src_start=query.start - offsets[query.page_row],
            dst_start=query.start - begin,
        )
        counts[slot] = query.count
        return following, lookahead, next_query(lookahead), step + 1

    zero = jnp.int32(0)
    first = next_query(QueryTile(begin, zero, zero))
    for slot in range(2):
        counts[slot] = zero
    second = next_query(first)
    load_query(first, zero)
    load_keys(first, zero)
    load_keys(second, jnp.int32(1))
    lax.while_loop(
        lambda state: state[0].start < begin + out.shape[0],
        body,
        (first, second, next_query(second), zero),
    )
    release(0)
    release(1)


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
    config = PACKED_FP8_PREPARED_MQA_CONFIG
    config = replace(config, key_tile=tile)
    width = pl.cdiv(output_width, PAGE_RECORDS) * PAGE_RECORDS
    q_tile, heads, dim = config.query_tile, q.shape[1], q.shape[2]
    if q_tile != 2:
        raise ValueError("Contiguous pair scoring requires a two-query tile")
    scratch = (
        pltpu.VMEM((2 * q_tile, heads, dim), q.dtype),
        pltpu.VMEM((config.key_buffers, padded_rows, dim), jnp.int32),
        pltpu.VMEM((2, q_tile, tile + PAGE_RECORDS), jnp.float32),
        pltpu.SMEM((2,), jnp.int32),
        pltpu.SemaphoreType.DMA((config.key_buffers,)),
        pltpu.SemaphoreType.DMA((2, q_tile)),
        pltpu.SemaphoreType.DMA((2,)),
    )
    hbm, vmem = (pl.BlockSpec(memory_space=s) for s in (pltpu.HBM, pltpu.VMEM))
    return pl.pallas_call(
        partial(_kernel, config=config, info=info),
        out_shape=jax.ShapeDtypeStruct((rows, width), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(1,),
            in_specs=((hbm, vmem, hbm),),
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
