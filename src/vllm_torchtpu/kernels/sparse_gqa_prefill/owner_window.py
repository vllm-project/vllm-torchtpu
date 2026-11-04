# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bounded read-only SparseCore source over an unchanged native KV owner."""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as meta
from vllm_torchtpu.kernels.sparse_gqa_prefill import adaptive, grouped, slab_gather
from vllm_torchtpu.kernels.sparse_gqa_prefill.tuned_block_sizes import (
    SC_GATHER_MAX_CACHE_BYTES,
)


class PagePlan(NamedTuple):
    physical: jax.Array
    count: jax.Array
    table: jax.Array
    valid: jax.Array


def window_data_pages(cache, starts, table, workspace_bytes: int | None) -> int:
    """Bound allocation by table namespaces, caller memory policy and SC ABI."""
    page_bytes = math.prod(cache.shape[1:]) * jnp.dtype(cache.dtype).itemsize
    limit = SC_GATHER_MAX_CACHE_BYTES
    if workspace_bytes is not None:
        if workspace_bytes < 0:
            raise ValueError("Workspace bytes must be nonnegative")
        limit = min(limit, workspace_bytes)
    rows_per_page = cache.shape[1] * cache.shape[2]
    pages = min(limit // page_bytes, jnp.iinfo(jnp.int32).max // rows_per_page)
    return max(0, min(pages - 1, (starts.shape[0] - 1) * table.shape[1]))


def is_supported_by_owner_window(
    query, gate, blocks, lengths, table, cache, starts, *, kv_heads: int, ratio: int
) -> bool:
    """Check actual bounded-source geometry without admitting the native pool to SC."""
    if not adaptive._has_query_and_request_geometry(
        query, gate, kv_heads, ratio, starts
    ):
        return False
    if (
        cache.ndim != 5
        or cache.shape[0] <= 0
        or cache.shape[1] <= 0
        or table.ndim != 2
        or table.shape[1] <= 0
    ):
        return False
    capacity = window_data_pages(cache, starts, table, None)
    if capacity == 0:
        return False
    bounded = jax.ShapeDtypeStruct((capacity + 1, *cache.shape[1:]), cache.dtype)
    return adaptive.is_supported_by_adaptive_prefill(
        query,
        gate,
        blocks,
        lengths,
        table,
        bounded,
        starts,
        kv_heads=kv_heads,
        ratio=ratio,
        k_scale=1.0,
        v_scale=1.0,
    )


def plan_pages(
    blocks, lengths, table, starts, *, page_size, owner_pages, ratio, data_pages
) -> PagePlan:
    """Certify request-local live prefixes; table padding has no meaning."""
    batch, columns = table.shape
    requests = starts.shape[0] - 1
    rows = jnp.arange(batch, dtype=jnp.int32)
    # Counts of completed requests handle repeated empty starts without reads.
    request = jnp.sum(rows[:, None] >= starts[None, 1:], axis=1, dtype=jnp.int32)
    safe_request = jnp.clip(request, 0, requests - 1)
    maximum = jax.ops.segment_max(lengths, safe_request, num_segments=requests)
    maximum = jnp.maximum(maximum, 0)
    pages = (maximum + page_size - 1) // page_size
    offsets = jnp.cumsum(pages) - pages
    representatives = table[jnp.clip(starts[:-1], 0, batch - 1)]
    logical = jnp.arange(columns, dtype=jnp.int32)
    live = logical[None, :] < pages[:, None]
    count = jnp.sum(pages, dtype=jnp.int32)
    selected = jnp.nonzero(live.reshape(-1), size=data_pages, fill_value=0)[0]
    physical = representatives.reshape(-1)[selected]
    row_live = (lengths[:, None] > 0) & (logical < pages[safe_request, None])
    consistent = jnp.all(
        jnp.where(row_live, table == representatives[safe_request], True)
    )
    physical_valid = jnp.all(
        jnp.where(live, (representatives >= 0) & (representatives < owner_pages), True)
    )
    slots = jnp.arange(blocks.shape[1])[None, :]
    active = slots < jnp.minimum(lengths[:, None] // ratio, blocks.shape[1])
    selected_valid = jnp.all(
        jnp.where(active, (blocks >= 0) & (blocks < lengths[:, None] // ratio), True)
    )
    valid = (
        jnp.all((starts >= 0) & (starts <= batch))
        & jnp.all(
            jnp.where((rows < starts[0]) | (rows >= starts[-1]), lengths == 0, True)
        )
        & jnp.all(jnp.diff(starts) >= 0)
        & jnp.all((lengths >= 0) & (lengths <= columns * page_size))
        & (count <= data_pages)
        & consistent
        & physical_valid
        & selected_valid
    )
    remapped = jnp.where(row_live, offsets[safe_request, None] + logical + 1, 0)
    return PagePlan(physical, count, remapped, valid)


def stage_pages(cache, physical, count):
    """Write null page and active raw pages; the remaining tail is unwritten."""
    shape = (physical.shape[0] + 1, *cache.shape[1:])
    words = cache.shape[2] * cache.shape[-1]

    def move(ids, active, source, target, scratch, sem):
        page = pl.program_id(0)
        raw = scratch.bitcast(jnp.int32).reshape(1, cache.shape[1], words)
        dst = target.bitcast(jnp.int32).reshape(shape[0], cache.shape[1], words)

        def write():
            dma = pltpu.make_async_copy(raw, dst.at[pl.ds(page, 1)], sem)
            dma.start()
            dma.wait()

        @pl.when(page == 0)
        def null():
            raw[...] = jnp.zeros(raw.shape, jnp.int32)
            write()

        @pl.when((page > 0) & (page <= active[0]))
        def copy():
            src = source.bitcast(jnp.int32).reshape(
                cache.shape[0], cache.shape[1], words
            )
            dma = pltpu.make_async_copy(src.at[pl.ds(ids[page - 1], 1)], raw, sem)
            dma.start()
            dma.wait()
            write()

    return pl.pallas_call(
        move,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(shape[0],),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((1, *cache.shape[1:]), cache.dtype),
                pltpu.SemaphoreType.DMA,
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(shape, cache.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name="qsa_owned_window_raw_stage",
    )(physical, count.reshape(1), cache)


def _lookahead_prefill(
    q, gate, blocks, lengths, table, cache, *, chunk_size, kv_heads, ratio, padding_rows
):
    # Full microchunks use null-page padding for inactive rows.
    query = q
    count = jnp.int32(len(query) // chunk_size)
    assert len(query) // chunk_size > 0 and len(query) % chunk_size == 0
    info = pltpu.get_tpu_info().sparse_core
    rows_per_core = 4 * info.num_lanes
    cores = info.num_cores * info.num_subcores
    head_axes = cache.shape[2]
    capacity = pl.cdiv(blocks.shape[1] * ratio + ratio - 1, 256) * 256
    granularity = math.lcm(rows_per_core * cores, head_axes)
    rows = pl.cdiv(chunk_size * capacity * head_axes, granularity) * granularity
    shape = (rows, *cache.shape[3:])
    slab = slab_gather.pack_query_metadata(blocks, lengths, table)
    slotA = jax.ref.new_ref(jnp.zeros(shape, cache.dtype), memory_space=pltpu.HBM)
    slotB = jax.ref.new_ref(jnp.zeros(shape, cache.dtype), memory_space=pltpu.HBM)

    def write(slot, chunk):
        slab_gather._gather_from_slab_ref(
            slot,
            slab,
            cache,
            budget=blocks.shape[1],
            table_cols=table.shape[1],
            ratio=ratio,
            query_group=8,
            tile_k=256,
            padding_rows=padding_rows,
            query_base=chunk * chunk_size,
            chunk_queries=chunk_size,
        )

    def consume(slot, chunk, output):
        q = lax.dynamic_slice_in_dim(query, chunk * chunk_size, chunk_size)
        g = lax.dynamic_slice_in_dim(gate, chunk * chunk_size, chunk_size)
        n = lax.dynamic_slice_in_dim(lengths, chunk * chunk_size, chunk_size)
        y = grouped._attend_compact_ref(
            q,
            g,
            n,
            slot,
            kv_heads=kv_heads,
            ratio=ratio,
            query_group=8,
            source_group=8,
            tile_k=256,
            capacity=capacity,
            budget=blocks.shape[1],
        )
        return lax.dynamic_update_slice_in_dim(output, y, chunk * chunk_size, 0)

    write(slotA, jnp.int32(0))

    def pair(p, output):
        write(slotB, 2 * p + 1)
        output = consume(slotA, 2 * p, output)
        write(slotA, 2 * p + 2)
        return consume(slotB, 2 * p + 1, output)

    output = lax.fori_loop(0, (count - 1) // 2, pair, jnp.zeros_like(query))

    def even_tail(output):
        write(slotB, count - 1)
        output = consume(slotA, count - 2, output)
        return consume(slotB, count - 1, output)

    def odd_tail(output):
        return consume(slotA, count - 1, output)

    return lax.cond(count % 2 == 0, even_tail, odd_tail, output), cache


def owner_window_attention(
    query,
    gate,
    blocks,
    lengths,
    table,
    cache,
    starts,
    *,
    kv_heads: int,
    ratio: int,
    kv_tile: int = 256,
    workspace_bytes: int | None = None,
) -> jax.Array:
    """Read-only prefill; an uncertified window uses TC.

    workspace_bytes is a caller-owned staging allocation ceiling, including
    null page. None uses the existing SparseCore parameter byte limit. The
    owner is never donated, updated or returned by this entry.
    """

    def fallback(_):
        return adaptive._tc(
            query,
            gate,
            blocks,
            lengths,
            table,
            cache,
            starts,
            kv_heads=kv_heads,
            ratio=ratio,
            kv_tile=kv_tile,
        )

    if len(query) == 0:
        return jnp.zeros_like(query)
    data_pages = window_data_pages(cache, starts, table, workspace_bytes)
    window_shape = jax.ShapeDtypeStruct((data_pages + 1, *cache.shape[1:]), cache.dtype)
    if data_pages == 0 or not adaptive.is_supported_by_adaptive_prefill(
        query,
        gate,
        blocks,
        lengths,
        table,
        window_shape,
        starts,
        kv_heads=kv_heads,
        ratio=ratio,
        k_scale=1.0,
        v_scale=1.0,
    ):
        return fallback(None)
    padded = pl.cdiv(len(query), 64) * 64
    if not grouped._supports_whole_slab(
        jax.ShapeDtypeStruct((padded, blocks.shape[1]), blocks.dtype),
        jax.ShapeDtypeStruct((padded, table.shape[1]), table.dtype),
        window_shape,
        query_group=8,
        tile_k=256,
        padding_rows=cache.shape[1] * cache.shape[2],
    ):
        return fallback(None)
    plan = plan_pages(
        blocks,
        lengths,
        table,
        starts,
        page_size=cache.shape[1],
        owner_pages=cache.shape[0],
        ratio=ratio,
        data_pages=data_pages,
    )

    def sparse(_):
        window = stage_pages(cache, plan.physical, plan.count)
        # Full microchunks keep inactive rows on the initialized null page.
        padded = pl.cdiv(len(query), 64) * 64

        def pad(x, value=0):
            return jnp.pad(
                x,
                ((0, padded - len(query)), *((0, 0),) * (x.ndim - 1)),
                constant_values=value,
            )

        attention, _ = _lookahead_prefill(
            pad(query),
            jnp.full_like(pad(gate), jnp.inf),
            pad(blocks, -1),
            pad(lengths),
            pad(plan.table),
            window,
            chunk_size=64,
            kv_heads=kv_heads,
            ratio=ratio,
            padding_rows=cache.shape[1] * cache.shape[2],
        )
        return meta._gate_output(attention[: len(query)], gate, lengths)

    return lax.cond(plan.valid, sparse, fallback, None)
