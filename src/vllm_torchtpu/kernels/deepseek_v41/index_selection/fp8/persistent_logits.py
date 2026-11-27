# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resume dense scoring across HBM windows without resetting query groups."""

import math
from collections.abc import Callable
from dataclasses import replace
from functools import reduce
from typing import cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs

from .common import (
    QueryTile,
    _dma_rows,
    _head_order,
    _reduce_heads,
    _reduce_on_mxu,
    _weighted_selector,
)
from .dense_logits import _DenseKernel, _encode_page_table, _QueryBlock
from .tuned import PACKED_FP8_GROUPED_PREFILL, PACKED_FP8_MQA_CONFIG, paged_mqa_key_tile


class _PersistentKernel(_DenseKernel):
    """Retain the query/K prefetch ring and the last query group's scores."""

    reference_rows: int | None = None

    def retained_key_id(self, query, tile_count):
        # Reload if the request changes or its visible prefix gains a K tile.
        return query.page_row * (self.tiles + 1) + tile_count

    def next_query(self, query):
        query = super().next_query(query)
        if self.reference_rows is not None:
            end = (query.start // self.reference_rows + 1) * self.reference_rows
            query = query._replace(count=jnp.minimum(query.count, end - query.start))
        if self.q_tile == 12:
            remaining = self.stop - query.start
            count = jnp.where(
                (remaining > 0) & (remaining < 12), pl.cdiv(remaining, 6) * 6, 12
            )
            query = query._replace(count=jnp.minimum(query.count, count))
        return query

    def query_block(self, state):
        block = _GroupedQueryBlock if self.q_tile == 12 else _QueryBlock
        result = block(self, state).run()
        query = state[0]
        self.loop[15], self.loop[16] = query.start, query.count
        # Preserve which of the last group's tiles were actually written.
        self.counts[4 + (state[4] & 1)] = state[3][0]
        return result

    def run_window(self, loop, stop, initialize):
        self.loop, self.stop = loop, stop
        zero = jnp.int32(0)
        if initialize:
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
        else:
            state = (
                *(QueryTile(*(loop[i + j] for j in range(3))) for i in (0, 3, 6)),
                tuple(loop[i] for i in range(9, 12)),
                *(loop[i] for i in range(12, 15)),
            )

            @pl.when(loop[15] >= self.begin)
            def carry():
                pair = (loop[12] - 1) & 1
                for tile in range(self.tiles):
                    slot = 2 * pair + tile
                    src = self.scores.at[slot]
                    dst = self.out.at[:, pl.ds(tile * self.key_tile, self.key_tile)]
                    count = jnp.where(tile < self.counts[4 + pair], loop[16], 0)
                    _dma_rows(
                        src,
                        dst,
                        self.write_semaphore(slot),
                        count,
                        dst_start=loop[15] - self.begin,
                    )
                    self.counts[slot] = count

        state = lax.while_loop(
            lambda s: cast(QueryTile, s[0]).start < stop, self.query_block, state
        )
        for i, value in enumerate(jax.tree.leaves(state)):
            loop[i] = value
        self.release_outputs(0)
        self.release_outputs(1)
        for i in (0, 1, 2, 3, 6, 7):
            self.counts[i] = zero


class _GroupedQueryBlock(_QueryBlock):
    """Pair independent six-row groups without changing their head reductions."""

    def prepare_queries(self):
        k = self.k
        rows = k.row_ids(self.query)

        def order(x):
            return jnp.concatenate(
                [_head_order(x[i * 32 : (i + 6) * 32], 6, k.sublanes) for i in (0, 6)]
            )

        weights = jnp.concatenate([k.weights[r][:, None] for r in rows])
        self.head_weights = order(weights)
        owners = _head_order(jnp.repeat(jnp.arange(6), 32)[:, None], 6, k.sublanes)[
            :, 0
        ]
        selector = (jnp.arange(6)[:, None] == owners[None, :]).astype(jnp.bfloat16)
        self.selectors = tuple(
            _weighted_selector(
                selector, k.weights, rows[i : i + 6], sublanes=k.sublanes
            )
            for i in (0, 6)
        )
        values = k.queries[pl.ds(self.q_slot * k.q_tile, k.q_tile)]
        self.lhs = order(values.reshape(k.q_tile * 32, 128))
        # Tokamax streamindex_topk, commit 16864d8: interleave independent
        # packed row blocks across MXUs, then restore the original head order.
        self.row_block = k.sublanes * (32 // jax.dtypes.itemsize_bits(k.q.dtype))
        self.blocks = self.lhs.shape[0] // self.row_block
        self.groups = math.gcd(pltpu.get_tpu_info().num_mxus, self.blocks)
        self.parts = [
            jnp.concatenate(
                [
                    self.lhs[pl.ds(b * self.row_block, self.row_block)]
                    for b in range(g, self.blocks, self.groups)
                ]
            )
            for g in range(self.groups)
        ]
        lengths = k.row_lengths(self.query)
        self.group_lengths = [reduce(jnp.maximum, lengths[i : i + 6]) for i in (0, 6)]

    def dot(self, rhs):
        products = [
            lax.dot(part, rhs, preferred_element_type=jnp.float32)
            for part in self.parts
        ]
        return jnp.concatenate(
            [
                products[b % self.groups][
                    pl.ds(b // self.groups * self.row_block, self.row_block)
                ]
                for b in range(self.blocks)
            ]
        )

    def reduce_fragment(self, products, fragment, tile, count):
        k = self.k
        stage = fragment + PACKED_FP8_MQA_CONFIG.issue_depth
        mxu = _reduce_on_mxu(fragment, stage, count, every=4, drain=4)
        half = _reduce_on_mxu(fragment, stage, min(count, k.half), every=4, drain=4)
        values = []
        for i, selector in enumerate(self.selectors):
            rows = pl.ds(i * 6 * 32, 6 * 32)

            def reduce_group(use_mxu, rows=rows, selector=selector):
                return _reduce_heads(
                    products[rows],
                    self.head_weights[rows],
                    selector,
                    sublanes=k.sublanes,
                    use_mxu=use_mxu,
                )

            value = reduce_group(mxu)
            if fragment < k.half and mxu != half:
                value = jnp.where(
                    self.group_lengths[i] <= tile * k.key_tile + k.half * k.mxu_size,
                    reduce_group(half),
                    value,
                )
            values.append(value)
        return jnp.concatenate(values)


def score_window(
    q,
    banks,
    weights,
    lengths,
    pages,
    offsets,
    *,
    layout,
    width,
    start,
    rows,
    state=None,
    reference_rows=None,
):
    """Produce a window plus a six-row halo; return its resident TC state.

    Windows are issued in order. Request-local six-query groups follow the
    standalone scorer across every window boundary, preserving BF16 head sums.
    Only one or two K tiles are supported so the final group fits the score ring.
    """
    info = pltpu.get_tpu_info()
    config = PACKED_FP8_MQA_CONFIG
    key_tile = paged_mqa_key_tile(width, layout.page_size, config.key_tile)
    config = replace(
        config,
        key_tile=key_tile,
        # Retained keys use one slot per live tile, never the four-slot DMA ring.
        key_buffers=2,
        issue_depth=envs.TPU_DEEPSEEK_V41_INDEXER_ISSUE_DEPTH or config.issue_depth,
    )
    grouped = PACKED_FP8_GROUPED_PREFILL.get(
        (info.generation, layout.page_size, key_tile)
    )
    if (
        grouped is not None
        and offsets.shape == (2,)
        and (
            state[0][0][0].shape[0] == 2 * grouped[0]
            if state is not None
            else rows + 2 * config.query_tile >= grouped[2]
        )
    ):
        config = replace(config, query_tile=grouped[0], issue_depth=grouped[1])
    width = pl.cdiv(width, key_tile) * key_tile
    if width > 2 * key_tile:
        raise ValueError("Persistent scoring supports at most two K tiles")
    tile, tile_pages = config.query_tile, key_tile // layout.page_size
    count = width // layout.page_size
    ids = pages[:, :count]
    ids = jnp.pad(ids, ((0, 0), (0, count - ids.shape[1])), constant_values=-1)
    table = _encode_page_table(ids, layout, banks[0].shape[0])
    page_count = pl.cdiv(tile_pages, info.num_lanes) * info.num_lanes
    table = table.reshape(ids.shape[0], -1, tile_pages)
    table = jnp.pad(table, ((0, 0), (0, 0), (0, page_count - tile_pages)))
    dma = pltpu.SemaphoreType.DMA
    scratch = (
        (
            pltpu.VMEM((2 * tile, 32, 128), q.dtype),
            pltpu.VMEM(
                (config.key_buffers, tile_pages * layout.page_rows, 128), jnp.int32
            ),
            pltpu.VMEM((4, tile, key_tile), jnp.float32),
            pltpu.SMEM((8,), jnp.int32),
            pltpu.SMEM((4 * page_count,), jnp.int32),
            pltpu.VMEM((tile, key_tile), jnp.float32),
        ),
        tuple(dma(s) for s in ((config.key_buffers,), (4 * tile,), (2,), (4,), (2,))),
    )
    # Flattened loop state: three QueryTiles (9 scalars), three tile counts,
    # step/base/ready, then the last group's start/count for the HBM halo.
    specs = (
        scratch,
        pltpu.SMEM((17,), jnp.int32),
        pltpu.SMEM(offsets.shape, jnp.int32),
        pltpu.SMEM(lengths.shape, jnp.int32),
        pltpu.VMEM(weights.shape, jnp.float32),
        pltpu.SMEM((2,), jnp.int32),
    )
    mesh = pltpu.TensorCoreMesh(axis_name="tc", num_cores=1)
    initialize = state is None
    out_type = jax.ShapeDtypeStruct((rows + 12, width), jnp.float32)

    def resume(inputs, retained, out):
        buffers, loop, query_offsets, row_lengths, weight, bounds = retained
        kernel = _PersistentKernel(
            (query_offsets, row_lengths, bounds.at[0]),
            (inputs[0], weight, inputs[1], inputs[2]),
            (out, buffers),
            config,
            info,
            layout,
            True,
        )
        kernel.end = q.shape[0]
        kernel.reference_rows = reference_rows
        stop = jnp.minimum(bounds[1], q.shape[0])
        kernel.run_window(loop, stop, initialize)
        bounds[0] = jnp.where(loop[15] + loop[16] > stop, loop[15], stop)
        bounds[1] = stop + rows

    params = pltpu.CompilerParams(disable_bounds_checks=True)
    inputs = (q, banks, table)
    if initialize:
        bounds = jnp.zeros((info.num_lanes,), jnp.int32).at[1].set(start + rows)

        def begin(offsets, lengths, weight, bounds, inputs, out, retained, staging):
            pltpu.sync_copy(offsets, retained[2])
            pltpu.sync_copy(lengths, retained[3])
            pltpu.sync_copy(weight, retained[4])
            pltpu.sync_copy(bounds, staging)
            retained[5][0], retained[5][1] = staging[0], staging[1]
            resume(inputs, retained, out)

        begin_call = cast(
            Callable[..., tuple[jax.Array, tuple]],
            pl.kernel(
                begin,
                mesh=mesh,
                out_type=(out_type, jax.tree.map(lambda s: s @ mesh, specs)),
                scratch_types=(pltpu.SMEM(bounds.shape, jnp.int32),),
                compiler_params=params,
                name="fp8_score_stream_start",
            ),
        )
        result, retained = begin_call(offsets, lengths, weights, bounds, inputs)
        state = jax.tree.map(
            lambda x, s: jax.new_ref(x, memory_space=s.memory_space @ mesh),
            retained,
            specs,
        )
    else:
        resume_call = cast(
            Callable[..., jax.Array],
            pl.kernel(
                resume,
                mesh=mesh,
                out_type=out_type,
                compiler_params=params,
                name="fp8_score_stream_resume",
            ),
        )
        result = resume_call(inputs, state)
    return result, state
