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
"""Shared-query metadata slab and native packed SparseCore payload gather."""

import math
from typing import cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.sparse_gqa_prefill.gather import _gather_vectors


def field_layout(budget, table_cols):
    info = pltpu.get_tpu_info()
    sc = info.sparse_core
    assert sc is not None
    lanes = sc.num_lanes
    tile = (info.num_sublanes, info.num_lanes)
    block = pl.cdiv(1, lanes) * lanes
    table = pl.cdiv(block + budget, lanes) * lanes
    stride = pl.cdiv(table + table_cols, math.prod(tile)) * math.prod(tile)
    return dict(
        length_offset=0,
        block_offset=block,
        table_offset=table,
        stride_words=stride,
        slab_rows=stride // tile[1],
        TC_tile=tile,
        SC_lanes=lanes,
    )


def pack_query_metadata(blocks, lengths, table):
    layout = field_layout(blocks.shape[1], table.shape[1])
    batch, budget = blocks.shape
    block = layout["block_offset"]
    t = layout["table_offset"]
    row = jnp.concatenate(
        (
            lengths[:, None],
            jnp.zeros((batch, block - 1), jnp.int32),
            blocks,
            jnp.zeros((batch, t - block - budget), jnp.int32),
            table,
            jnp.zeros((batch, layout["stride_words"] - t - table.shape[1]), jnp.int32),
        ),
        axis=1,
    )
    return row.reshape(batch, layout["slab_rows"], layout["TC_tile"][1])


def lane_take(x, indices):
    return lax.gather(
        x,
        indices[..., None],
        dimension_numbers=lax.GatherDimensionNumbers(
            offset_dims=(),
            collapsed_slice_dims=(0,),
            start_index_map=(0,),
        ),
        slice_sizes=(1,),
        mode="promise_in_bounds",
    )


def buffer_bytes(budget: int, table_cols: int, dim: int) -> dict[str, int]:
    """Return named double-buffer allocations per SC vector subcore."""
    info = pltpu.get_tpu_info().sparse_core
    assert info is not None
    streams = 4
    word_bytes = jnp.dtype(jnp.int32).itemsize
    metadata = 2 * field_layout(budget, table_cols)["stride_words"] * word_bytes
    payload = 2 * streams * info.num_lanes * dim * word_bytes
    return {"metadata": metadata, "payload": payload, "total": metadata + payload}


def _gather_from_slab(
    slab,
    cache,
    *,
    budget,
    table_cols,
    ratio,
    query_group,
    tile_k,
    padding_rows,
    query_base=None,
    chunk_queries=None,
):
    compact = _gather_from_slab_impl(
        slab,
        cache,
        budget=budget,
        table_cols=table_cols,
        ratio=ratio,
        query_group=query_group,
        tile_k=tile_k,
        padding_rows=padding_rows,
        query_base=query_base,
        chunk_queries=chunk_queries,
    )
    assert compact is not None
    return compact


def _gather_from_slab_ref(
    destination,
    slab,
    cache,
    *,
    budget,
    table_cols,
    ratio,
    query_group,
    tile_k,
    padding_rows,
    query_base,
    chunk_queries,
    tail_rows=0,
):
    return _gather_from_slab_impl(
        slab,
        cache,
        budget=budget,
        table_cols=table_cols,
        ratio=ratio,
        query_group=query_group,
        tile_k=tile_k,
        padding_rows=padding_rows,
        query_base=query_base,
        chunk_queries=chunk_queries,
        destination=destination,
        tail_rows=tail_rows,
    )


def _gather_from_slab_impl(
    slab,
    cache,
    *,
    budget,
    table_cols,
    ratio,
    query_group,
    tile_k,
    padding_rows,
    query_base=None,
    chunk_queries=None,
    destination=None,
    tail_rows=0,
):
    cache_shape = cache.shape
    info = pltpu.get_tpu_info()
    sc = info.sparse_core
    assert sc is not None
    lanes, streams = (sc.num_lanes, 4)
    cores = sc.num_cores * sc.num_subcores
    rows_per_core = streams * lanes
    head_axes = cache_shape[2]
    tile_rows = tile_k * head_axes
    layout = field_layout(budget, table_cols)
    tc_tile = layout["TC_tile"]
    batch = slab.shape[0] if chunk_queries is None else chunk_queries
    capacity = pl.cdiv(budget * ratio + ratio - 1, tile_k) * tile_k
    full_tiles = capacity // tile_k - bool(tail_rows)
    # Store all full tiles first, then one aligned tail segment per query.
    full_rows = batch * full_tiles * tile_rows
    logical_rows = full_rows + batch * tail_rows * head_axes
    if tail_rows:
        assert full_tiles > 0 and tail_rows * head_axes % rows_per_core == 0
    granularity = math.lcm(rows_per_core * cores, head_axes)
    block_count = (
        pl.cdiv(logical_rows, granularity) * granularity // (rows_per_core * cores)
    )
    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )

    def kernel_body(cache_ref, slab_ref, output_ref, sem_ref, descriptor_ref=None):
        core = lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))
        src = cache_ref.bitcast(jnp.int32).reshape(-1, cache_shape[-1])
        dst = output_ref.bitcast(jnp.int32).reshape(-1, cache_shape[-1])

        def coordinates(block, stream):
            row = (block * cores + core) * rows_per_core + stream * lanes
            group_tile = row // (query_group * tile_rows)
            group = group_tile // full_tiles
            tile = group_tile % full_tiles
            query = group * query_group + row // tile_rows % query_group
            local = row % tile_rows
            if tail_rows:
                tail = row >= full_rows
                tail_row = row - full_rows
                tile = jnp.where(tail, full_tiles, tile)
                query = jnp.where(tail, tail_row // (tail_rows * head_axes), query)
                local = jnp.where(tail, tail_row % (tail_rows * head_axes), local)
            return row, tile, query, local

        def shared_query(block):
            return jnp.clip(coordinates(block, 0)[2], 0, batch - 1)

        def address_words(metadata, block, stream):
            row, tile, query, local = coordinates(block, stream)
            lane = local + jnp.arange(lanes, dtype=jnp.int32)
            token = tile * tile_k + lane // head_axes
            slot = token // ratio
            length_panel = metadata[pl.ds(layout["length_offset"], lanes)]
            length = lane_take(length_panel, jnp.zeros((lanes,), jnp.int32))
            nonnegative = jnp.maximum(length, 0)
            complete = nonnegative // ratio
            count = jnp.minimum(complete, budget)
            selected_len = count * ratio + nonnegative % ratio
            selected = jnp.zeros((lanes,), jnp.int32)
            for panel in range(pl.cdiv(budget, lanes)):
                delta = slot - panel * lanes
                source = metadata[pl.ds(layout["block_offset"] + panel * lanes, lanes)]
                value = lane_take(source, jnp.clip(delta, 0, lanes - 1))
                selected = jnp.where((delta >= 0) & (delta < lanes), value, selected)
            selected = jnp.where(slot < budget, selected, 0)
            selected = jnp.where(slot == count, complete, selected)
            logical = selected // (cache_shape[1] // ratio)
            physical = jnp.zeros((lanes,), jnp.int32)
            for panel in range(pl.cdiv(table_cols, lanes)):
                delta = logical - panel * lanes
                source = metadata[pl.ds(layout["table_offset"] + panel * lanes, lanes)]
                value = lane_take(source, jnp.clip(delta, 0, lanes - 1))
                physical = jnp.where((delta >= 0) & (delta < lanes), value, physical)
            physical = jnp.where((logical >= 0) & (logical < table_cols), physical, 0)
            physical_token = (
                physical * cache_shape[1]
                + selected % (cache_shape[1] // ratio) * ratio
                + token % ratio
            )
            index = physical_token * head_axes + lane % head_axes
            spread = (
                (query * capacity * head_axes + tile * tile_rows + lane)
                * 104729
                % padding_rows
            )
            index = jnp.where(token < selected_len, index, spread)
            global_rows = row + jnp.arange(lanes, dtype=jnp.int32)
            guard = (global_rows - logical_rows) * 15485863 % padding_rows
            return jnp.where(global_rows < logical_rows, index, guard)

        def body(local_slab):
            block = pl.program_id(0)
            metadata = local_slab.reshape(-1)
            indices = tuple(
                address_words(metadata, block, stream) for stream in range(streams)
            )
            output_row = (block * cores + core) * rows_per_core
            _gather_vectors(
                src,
                dst,
                sem_ref,
                indices,
                output_row,
                streams=streams,
                lanes=lanes,
                width=cache_shape[-1],
            )

        spec = pl.BlockSpec(
            (None, layout["slab_rows"], tc_tile[1]),
            lambda b: (shared_query(b), 0, 0),
        )
        if descriptor_ref is None:
            pltpu.emit_pipeline(body, grid=(block_count,), in_specs=spec)(slab_ref)
        else:

            def with_base(local_base):
                base = local_base[...][0]
                chunk_ref = slab_ref.at[pl.ds(base, batch)]
                pltpu.emit_pipeline(body, grid=(block_count,), in_specs=spec)(chunk_ref)

            pltpu.emit_pipeline(
                with_base,
                grid=(1,),
                in_specs=pl.BlockSpec((lanes,), lambda _: (0,)),
            )(descriptor_ref)

    if query_base is None:
        kernel = kernel_body
        inputs = (cache, slab)
        name = "sc_prefill_shared_slab_gather"
    else:
        assert chunk_queries is not None

        def kernel(cache_ref, slab_ref, descriptor_ref, output_ref, sem_ref):
            kernel_body(cache_ref, slab_ref, output_ref, sem_ref, descriptor_ref)

        inputs = (cache, slab, jnp.full((lanes,), query_base, jnp.int32))
        name = "sc_prefill_whole_slab_window_gather"
    if destination is not None:
        assert query_base is not None and chunk_queries is not None

        def writer(cache_ref, slab_ref, descriptor_ref, sem_ref):
            kernel_body(cache_ref, slab_ref, destination, sem_ref, descriptor_ref)

        pl.kernel(
            writer,
            out_type=(),
            scratch_types=(pltpu.SemaphoreType.DMA((streams,)),),
            mesh=mesh,
            compiler_params=pltpu.CompilerParams(
                use_tc_tiling_on_sc=True,
                needs_layout_passes=True,
                disable_bounds_checks=True,
            ),
            name="destination_ref_slab_writer",
        )(*inputs)
        return None

    compact = pl.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct(
            (block_count * cores * rows_per_core, *cache_shape[3:]), cache.dtype
        ),
        scratch_types=(pltpu.SemaphoreType.DMA((streams,)),),
        mesh=mesh,
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        name=name,
    )(*inputs)
    return cast(jax.Array, compact).reshape(-1, head_axes, *cache_shape[3:])
