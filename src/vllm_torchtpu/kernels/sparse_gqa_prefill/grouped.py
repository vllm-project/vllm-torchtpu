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
"""SC-interleaved selected KV and grouped-query prefill MXU tiles.

Ported from standalone qsa_prefill v0.4 grouped.py; imports only changed.
Transport uses the native packed SparseCore gather. Unlike block-union
prefill, the gathered rows are owned by individual queries. A diagonal mask
keeps their softmaxes independent while a small query group fills MXU rows.
QK, probabilities, PV and online state retain the original prefill FP32
arithmetic; only the original output/gate boundaries round to BF16.
"""

import functools
import math
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.experimental.batched_rpa import utils
from vllm_torchtpu.kernels.qwen4_exp.rounding import bf16
from vllm_torchtpu.kernels.sparse_gqa_prefill.gather import _gather_kernel, _take_lane


def _interleaved_indices(
    lengths_ref,
    blocks_ref,
    table_ref,
    out_ref,
    *,
    ratio,
    page_size,
    budget,
    query_group,
    tile_k,
    head_axes,
    capacity,
    source_rows,
):
    b = pl.program_id(0)
    tile = pl.program_id(1)
    lane = jnp.arange(tile_k * head_axes, dtype=jnp.int32)[None, :]
    token = tile * tile_k + lane // head_axes
    slot = token // ratio
    lengths = jnp.stack([lengths_ref[b * query_group + i] for i in range(query_group)])[
        :, None
    ]
    complete = lengths // ratio
    count = jnp.minimum(complete, budget)
    selected_len = count * ratio + lengths % ratio
    slots = jnp.broadcast_to(slot, (query_group, tile_k * head_axes))
    selected = jnp.zeros_like(slots)
    for c in range(blocks_ref.shape[1] // 128):
        local = slots - c * 128
        values = _take_lane(
            blocks_ref[:, c * 128 : (c + 1) * 128], jnp.clip(local, 0, 127)
        )
        selected = jnp.where((local >= 0) & (local < 128), values, selected)
    selected = jnp.where(slots == count, complete, selected)
    logical = selected // (page_size // ratio)
    physical = jnp.zeros_like(logical)
    for c in range(table_ref.shape[1] // 128):
        local = logical - c * 128
        values = _take_lane(
            table_ref[:, c * 128 : (c + 1) * 128], jnp.clip(local, 0, 127)
        )
        physical = jnp.where((local >= 0) & (local < 128), values, physical)
    physical_token = (
        physical * page_size + (selected % (page_size // ratio)) * ratio + token % ratio
    )
    indices = physical_token * head_axes + lane % head_axes
    spread = (
        (
            (b * query_group + jnp.arange(query_group)[:, None]) * capacity * head_axes
            + tile * tile_k * head_axes
            + lane
        )
        * 104729
        % source_rows
    )
    out_ref[...] = jnp.where(token < selected_len, indices, spread).reshape(1, -1)


def gather_interleaved(
    blocks: jax.Array,
    lengths: jax.Array,
    table: jax.Array,
    cache: jax.Array,
    *,
    query_group: int,
    tile_k: int,
    ratio: int,
) -> tuple[jax.Array, int]:
    batch, budget = blocks.shape
    head_axes, packing, dim = cache.shape[2:]
    capacity = pl.cdiv(budget * ratio + ratio - 1, tile_k) * tile_k
    if batch % query_group or cache.nbytes > 16 * 2**30:
        raise ValueError("Aligned query groups and cache <=16 GiB are required")
    padded_budget = pl.cdiv(budget, 128) * 128
    padded_table = pl.cdiv(table.shape[1], 128) * 128
    indices = pl.pallas_call(
        functools.partial(
            _interleaved_indices,
            ratio=ratio,
            page_size=cache.shape[1],
            budget=budget,
            query_group=query_group,
            tile_k=tile_k,
            head_axes=head_axes,
            capacity=capacity,
            source_rows=cache.shape[0] * cache.shape[1] * head_axes,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(batch // query_group, capacity // tile_k),
            in_specs=(
                pl.BlockSpec((query_group, padded_budget), lambda b, t, *_: (b, 0)),
                pl.BlockSpec((query_group, padded_table), lambda b, t, *_: (b, 0)),
            ),
            out_specs=pl.BlockSpec(
                (None, 1, query_group * tile_k * head_axes),
                lambda b, t, *_: (b * (capacity // tile_k) + t, 0, 0),
            ),
        ),
        out_shape=jax.ShapeDtypeStruct(
            (
                batch // query_group * (capacity // tile_k),
                1,
                query_group * tile_k * head_axes,
            ),
            jnp.int32,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
        name="sc_prefill_prepare_interleaved_indices",
    )(
        lengths,
        jnp.pad(blocks, ((0, 0), (0, padded_budget - budget))),
        jnp.pad(table, ((0, 0), (0, padded_table - table.shape[1]))),
    ).reshape(-1)
    info = pltpu.get_tpu_info().sparse_core
    if info is None:
        raise NotImplementedError("Requires SparseCore")
    streams = 4
    granularity = math.lcm(
        info.num_lanes * info.num_cores * info.num_subcores * streams, head_axes
    )
    padded = pl.cdiv(indices.size, granularity) * granularity
    source_rows = cache.shape[0] * cache.shape[1] * head_axes
    indices = jnp.concatenate(
        (
            indices,
            (jnp.arange(padded - indices.size, dtype=jnp.int32) * 15485863)
            % source_rows,
        )
    )
    mesh = plsc.VectorSubcoreMesh(
        num_cores=info.num_cores,
        num_subcores=info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    gather = cast(
        Callable[[jax.Array, jax.Array], jax.Array],
        pl.kernel(
            functools.partial(
                _gather_kernel,
                streams=streams,
                core_axis=mesh.core_axis_name,
                subcore_axis=mesh.subcore_axis_name,
            ),
            out_type=jax.ShapeDtypeStruct((padded, packing, dim), cache.dtype),
            scratch_types=(pltpu.SemaphoreType.DMA((streams,)),),
            compiler_params=pltpu.CompilerParams(
                use_tc_tiling_on_sc=True,
                needs_layout_passes=True,
                disable_bounds_checks=True,
            ),
            mesh=mesh,
            name="sc_prefill_interleaved_gather",
        ),
    )
    compact = gather(cache, indices)
    return compact.reshape(-1, head_axes, packing, dim), capacity


def _attention(
    lengths_ref,
    q_ref,
    gate_ref,
    compact_ref,
    out_ref,
    kv_ref,
    sem_ref,
    m_ref,
    l_ref,
    acc_ref,
    *,
    query_group,
    head_group,
    kv_heads,
    dim,
    tile_k,
    capacity,
    budget,
    ratio,
    k_scale,
    v_scale,
):
    block = pl.program_id(0)
    rows = query_group * head_group
    width = query_group * tile_k
    packing = 32 // jax.dtypes.itemsize_bits(compact_ref.dtype)
    head_axes = compact_ref.shape[1]
    selected = [
        jnp.minimum(lengths_ref[block * query_group + i] // ratio, budget) * ratio
        + lengths_ref[block * query_group + i] % ratio
        for i in range(query_group)
    ]
    query_owner = jnp.arange(rows)[:, None] % query_group
    row_lengths = jnp.zeros((rows, 128), jnp.int32)
    max_length = selected[0]
    for i in range(query_group):
        row_lengths = jnp.where(query_owner == i, selected[i], row_lengths)
        max_length = jnp.maximum(max_length, selected[i])
    row_lengths = row_lengths[:, :1]
    tile_count = jnp.maximum(1, pl.cdiv(max_length, tile_k))
    key_owner = jnp.arange(width)[None, :] // tile_k
    key_offset = jnp.arange(width)[None, :] % tile_k

    def start(tile, buf):
        offset = (block * (capacity // tile_k) + tile) * width
        pltpu.make_async_copy(
            compact_ref.at[pl.ds(offset, width)], kv_ref.at[buf], sem_ref.at[buf]
        ).start()

    m_ref[...] = jnp.full(m_ref.shape, -jnp.inf, jnp.float32)
    l_ref[...] = jnp.zeros(l_ref.shape, jnp.float32)
    acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)
    start(0, 0)

    def step(tile, _):
        buf = tile % 2
        dst = kv_ref.at[buf]
        pltpu.make_async_copy(dst, dst, sem_ref.at[buf]).wait()

        @pl.when(tile + 1 < tile_count)
        def prefetch():
            start(tile + 1, 1 - buf)

        valid = (query_owner == key_owner) & (tile * tile_k + key_offset < row_lengths)
        token = lax.broadcasted_iota(jnp.int32, (width, dim), 0)
        live = jnp.zeros((width, dim), jnp.bool_)
        for i in range(query_group):
            live |= (token // tile_k == i) & (
                tile * tile_k + token % tile_k < selected[i]
            )
        words = dst.bitcast(jnp.uint32).reshape(-1, dim)
        for head in range(kv_heads):
            packed = utils.strided_load(
                words, head // (packing // 2), width * head_axes, head_axes
            )
            key, value = utils.convert_to_target_bitwidth(
                packed, target_bitwidth=32 // packing, kv_dtype=compact_ref.dtype
            )[head % (packing // 2)]
            key = key.astype(jnp.float32)
            value = value.astype(jnp.float32)
            if compact_ref.dtype == jnp.float8_e4m3fn:
                key = jnp.where(live & jnp.isfinite(key), key, 0)
                value = jnp.where(live & jnp.isfinite(value), value, 0)
            else:
                key = jnp.where(live, key, 0)
                value = jnp.where(live, value, 0)
            score = jnp.dot(
                q_ref[head].astype(jnp.float32),
                key.T.astype(jnp.float32),
                preferred_element_type=jnp.float32,
            ) * (dim**-0.5 * k_scale)
            score = jnp.where(valid, score, -jnp.inf)
            old_m = m_ref[head, :, :1]
            old_l = l_ref[head, :, :1]
            next_m = jnp.maximum(old_m, jnp.max(score, axis=1, keepdims=True))
            safe_m = jnp.where(jnp.isfinite(next_m), next_m, 0)
            prob = jnp.where(valid, jnp.exp(score - safe_m), 0)
            alpha = jnp.exp(old_m - safe_m)
            weighted = jnp.dot(
                prob, value.astype(jnp.float32), preferred_element_type=jnp.float32
            )
            acc_ref[head] = acc_ref[head] * alpha + weighted * v_scale
            m_ref[head] = jnp.broadcast_to(next_m, (rows, 128))
            l_ref[head] = jnp.broadcast_to(
                old_l * alpha + jnp.sum(prob, axis=1, keepdims=True), (rows, 128)
            )
        return None

    lax.fori_loop(0, tile_count, step, None)
    for head in range(kv_heads):
        mass = l_ref[head, :, :1]
        result = acc_ref[head] / jnp.where(mass > 0, mass, 1)
        gated = bf16(
            bf16(result).astype(jnp.float32)
            * bf16(jax.nn.sigmoid(gate_ref[head].astype(jnp.float32))).astype(
                jnp.float32
            )
        )
        out_ref[head] = jnp.where(row_lengths > 0, gated, 0)


def sc_grouped_attention(
    q: jax.Array,
    gate: jax.Array,
    blocks: jax.Array,
    lengths: jax.Array,
    table: jax.Array,
    cache: jax.Array,
    *,
    kv_heads: int,
    ratio: int = 4,
    query_group: int = 8,
    tile_k: int = 128,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    if q.ndim != 3 or gate.shape != q.shape or cache.ndim != 5:
        raise ValueError("Expected Q/gate[T,H,D] and native 5D cache")
    batch, heads, dim = q.shape
    if (
        kv_heads <= 0
        or heads % kv_heads
        or query_group <= 0
        or tile_k <= 0
        or tile_k % 128
        or ratio <= 0
        or tile_k % ratio
        or cache.shape[1] % ratio
    ):
        raise ValueError("Invalid GQA, query-group, tile or page geometry")
    if (
        q.dtype != jnp.bfloat16
        or gate.dtype != jnp.bfloat16
        or cache.dtype not in (jnp.bfloat16, jnp.float8_e4m3fn)
    ):
        raise NotImplementedError("BF16 Q/gate and BF16 or FP8 E4M3FN KV are required")
    packing = 32 // jax.dtypes.itemsize_bits(cache.dtype)
    if cache.shape[2:] != (pl.cdiv(2 * kv_heads, packing), packing, dim):
        raise ValueError("Invalid native KV layout")
    if (
        blocks.ndim != 2
        or blocks.shape[0] != batch
        or blocks.shape[1] <= 0
        or lengths.shape != (batch,)
        or table.ndim != 2
        or table.shape[0] != batch
        or table.shape[1] <= 0
        or any(x.dtype != jnp.int32 for x in (blocks, lengths, table))
    ):
        raise ValueError("Invalid selection metadata")
    if batch % query_group or query_group % 8 or dim % 128:
        raise ValueError("Aligned query groups (multiple of 8) and D are required")
    real_heads = heads // kv_heads
    head_group = pl.cdiv(real_heads, 8) * 8
    compact, capacity = gather_interleaved(
        blocks,
        lengths,
        table,
        cache,
        query_group=query_group,
        tile_k=tile_k,
        ratio=ratio,
    )

    def pack(x):
        x = x.reshape(batch // query_group, query_group, kv_heads, real_heads, dim)
        x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, head_group - real_heads), (0, 0)))
        return x.transpose(0, 2, 3, 1, 4).reshape(
            batch // query_group, kv_heads, head_group * query_group, dim
        )

    spec = pl.BlockSpec(
        (None, kv_heads, head_group * query_group, dim), lambda b, *_: (b, 0, 0, 0)
    )
    state = (kv_heads, head_group * query_group, 128)
    y = pl.pallas_call(
        functools.partial(
            _attention,
            query_group=query_group,
            head_group=head_group,
            kv_heads=kv_heads,
            dim=dim,
            tile_k=tile_k,
            capacity=capacity,
            budget=blocks.shape[1],
            ratio=ratio,
            k_scale=k_scale,
            v_scale=v_scale,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(batch // query_group,),
            in_specs=(spec, spec, pl.BlockSpec(memory_space=pltpu.HBM)),
            out_specs=spec,
            scratch_shapes=(
                pltpu.VMEM((2, query_group * tile_k, *cache.shape[2:]), cache.dtype),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.VMEM(state, jnp.float32),
                pltpu.VMEM(state, jnp.float32),
                pltpu.VMEM((kv_heads, head_group * query_group, dim), jnp.float32),
            ),
        ),
        out_shape=jax.ShapeDtypeStruct(
            (batch // query_group, kv_heads, head_group * query_group, dim),
            jnp.bfloat16,
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name="sc_compact_grouped_prefill",
    )(lengths, pack(q), pack(gate), compact)
    y = y.reshape(
        batch // query_group, kv_heads, head_group, query_group, dim
    ).transpose(0, 3, 1, 2, 4)
    return y[:, :, :, :real_heads].reshape(q.shape), cache
