# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

#
# Portions of this file are modified copies of code Copyright 2026 Google LLC,
# licensed under the Apache License, Version 2.0, from
# vllm_torchtpu/kernels/sparse_core/{core_map_helper,ragged_gather_v2,
# ragged_gather_reduce_v3}.py, vllm_torchtpu/kernels/megablox/gmm_v2.py and
# vllm_torchtpu/layers/core/fused_moe_gmm.py.
"""Kimi-K3 prefill megakernel: the 93 layers over a packed window at TP32.

A window (512, 1024 or 2048..8192 tokens) packs up to ``segments``
sequences at 256-aligned offsets, described by an int32 ``segment_desc``
(offsets, lengths, KDA checkpoint positions). Activations are
sequence-parallel in 64-row stripes: rank r holds rows
``[2048 b + 64 r, 2048 b + 64 (r + 1))`` of every 2048-token block b, in
block order (``sp_token_index``), so each block is a whole ring all-gather
of its own; a 512- or 1024-token window is one block of 16- or 32-row
stripes (``window_stripe``). Weights are
family-major: each family holds every layer's copy, a runtime slot picks one.

The prefill program (``kimi_k3/prefill.py``) runs four ops per window:

* ``make_embedding_sp``, then ``make_prefill_stack_sp_indexed`` (depth 4):
  the embedding and layers 0..3 (a dense FFN at layer 0);
* ``make_regular_group8_sp_indexed``, eleven times: layers 4..91, as pairs
  of four-layer groups (three KDA layers, one MLA layer, each with its MoE);
* ``make_final_mla_sp_indexed`` and ``make_prefill_logits_sp``: layer 92
  and each sequence's first-token logits.

Per layer:

* KDA, ``make_norm_gather_kda_sp``: one Pallas call for the residual mixture
  and RMSNorm, the 32-rank all-gather of the window, the chunked gated delta
  rule (``kda_prefill_tiled_body``) and the output projection ring;
* MLA, ``make_norm_gather_sp`` then ``make_mla_attention_layer_sp_ring``:
  the projections, packed flash attention over the window, context-parallel
  attention over every rank's cached prefix pages (``CpPrefix``), the
  projection ring;
* MoE, ``make_indexed_moe_sp_ring``: sequence-parallel routing, then by
  ``moe_reduce_dtype``. FP32 (the default): ``prefill_moe.fused``, one
  EP16/TP2 Pallas schedule for every window (512..8192). BF16:
  ``prefill_moe.layer`` (packed latent/shared overlap, indexed gate/up,
  weighted down fused with the reduce-scatter) for 4K/8K windows; 512..2048
  use SparseCore row movement and the grouped matmuls and ring below.

The sections run bottom-up: the TPU kernel library (SparseCore gathers,
TP32 rings, the grouped matmul), the K3 building blocks, the layers, then
the groups and the stack. Model math shared with the decode megakernel is
in ``helpers``. The SparseCore, grouped-matmul and counting-sort code is
adapted from ``vllm_torchtpu.kernels`` and ``vllm_torchtpu.layers`` (see
the header); the TP32 rings from ``vllm_torchtpu.kernels.kimi_k3.collectives``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
from abc import ABC, abstractmethod
from collections import namedtuple
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src import api as _src_api
from jax._src import core as jax_core
from jax._src import dispatch as _src_dispatch
from jax._src import lax as _src_lax
from jax._src import tree_util as _src_tree_util
from jax._src.interpreters import mlir as _src_mlir
from jax._src.lib.mlir import ir as _src_ir
from jax._src.lib.mlir.dialects import hlo as _src_hlo
from jax._src.pallas import core as pl_core
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel.common import pool_alias
from vllm_torchtpu.megakernel.kimi_k3.config import active as _config
from vllm_torchtpu.megakernel.kimi_k3.kernels import mla_sparse_prefill
from vllm_torchtpu.megakernel.kimi_k3.kernels.helpers import (
    e8m0_scale,
    expert_activation,
    kda_decay,
    kda_normalize_query_key,
)

# ============================================================================
# SparseCore kernels through core_map
# ============================================================================
#
# ``pl.kernel`` dispatches to ``mpmd_map``; ``core_map_kernel`` calls
# ``core_map`` directly on the given single SparseCore mesh.


# A kernel's output buffer: ``lax.empty``'s AllocateBuffer, taking the kernel's
# operands, so that the buffers of two calls stay distinct after inlining: one
# AllocateBuffer shared by several kernels makes XLA copy it into each.
_output_buffer_p = jax_core.Primitive("kernel_output_buffer")
_output_buffer_p.def_impl(functools.partial(_src_dispatch.apply_primitive, _output_buffer_p))
_output_buffer_p.def_abstract_eval(
    lambda *operands, shape, dtype, out_sharding: jax_core.ShapedArray(shape, dtype, sharding=out_sharding))


def _output_buffer_lowering(ctx, *operands, shape, dtype, out_sharding):
    (aval,) = ctx.avals_out
    physical = jax_core.physical_aval(aval)
    buffer = _src_hlo.CustomCallOp(
        [_src_ir.RankedTensorType.get(list(physical.shape), _src_mlir.dtype_to_ir_type(physical.dtype))],
        list(operands),
        call_target_name=_src_ir.StringAttr.get("AllocateBuffer"),
        has_side_effect=_src_ir.BoolAttr.get(False),
    )
    return [_src_mlir.lower_with_sharding_in_types(ctx, buffer.results[0], physical)]


_src_mlir.register_lowering(_output_buffer_p, _output_buffer_lowering, "tpu")
_src_mlir.register_lowering(
    _output_buffer_p,
    lambda ctx, *operands, shape, dtype, out_sharding: _src_lax.lax._empty_lower(
        ctx, shape=shape, dtype=dtype, out_sharding=out_sharding))


def _empty_out_ref(out_type, operands=()):
    # Mirrors JAX's private ``pl.kernel`` output-ref setup; recheck on JAX upgrades.
    aval = pl_core._convert_out_shape_to_aval(out_type)
    memory_space = (
        None
        if isinstance(aval.memory_space, jax_core.MemorySpace)
        else aval.memory_space
    )
    value = _output_buffer_p.bind(*operands, shape=aval.shape, dtype=aval.dtype, out_sharding=aval.sharding)
    return jax_core.new_ref(value, memory_space=memory_space)


def core_map_kernel(
    body,
    *,
    out_type,
    mesh,
    scratch_types=(),
    compiler_params=None,
    interpret=False,
    cost_estimate=None,
    debug=False,
    name=None,
    metadata=None,
):
    """Near drop-in for ``pl.kernel`` that lowers via ``core_map``.

    Differs in that ``out_shape`` is renamed to ``out_type`` and
    ``scratch_shapes`` (list) is renamed to ``scratch_types`` (dict).
    """
    single_output = not isinstance(out_type, (tuple, list))
    out_types = (out_type,) if single_output else out_type

    @_src_api.jit
    def run(*operands):
        arg_refs = _src_tree_util.tree_map(jax_core.new_ref, operands)
        out_refs = _src_tree_util.tree_map(lambda out_type: _empty_out_ref(out_type, operands), out_types)

        @pl_core.core_map(
            mesh,
            scratch_shapes=scratch_types,
            compiler_params=compiler_params,
            interpret=interpret,
            cost_estimate=cost_estimate,
            debug=debug,
            name=name,
            metadata=metadata,
        )
        def _(*scratch_refs, **scratch_kwrefs):
            return body(*arg_refs, *out_refs, *scratch_refs, **scratch_kwrefs)

        outs = _src_tree_util.tree_map(lambda ref: ref[...], out_refs)
        return outs[0] if single_output else outs

    return run


# ============================================================================
# SparseCore ragged gather
# ============================================================================
#
# Rows gathered by index on the SparseCore (``ragged_gather_v2``): the routed
# MoE's move into expert-major order.


def calculate_col_size(hidden_size: int, packing: int) -> int:
    """Calculates the max column size bounded by VMEM limits and hidden_size divisibility."""
    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core
    assert sc_info is not None, "SparseCore info is missing."
    lanes = sc_info.num_lanes

    match tpu_info.generation:
        case 6:
            target_bytes = (256 * 1024) * 0.8
        case 7:
            target_bytes = (512 * 1024) * 0.8
        case _:
            target_bytes = (128 * 1024) * 0.8

    # Calculate max safe column size based on VMEM budget and align to 128.
    num_buffers = 2
    bytes_per_col = (lanes + lanes // packing) * 4 * num_buffers
    max_safe_col = int((target_bytes // bytes_per_col) // 128) * 128

    # Search for the largest divisor of hidden_size bounded by max_safe_col.
    # The first divisor found is the maximum.
    start_col = (min(hidden_size, max_safe_col) // 128) * 128
    for c in range(start_col, 127, -128):
        if hidden_size % c == 0:
            return c
    return max_safe_col


def main_kernel_v2(
    start_ref: jax.Ref,
    end_ref: jax.Ref,
    in_hbm_ref: jax.Ref,
    indices_hbm_ref: jax.Ref,
    out_hbm_ref: jax.Ref,
    start_vmem_ref: jax.Ref,
    end_vmem_ref: jax.Ref,
    sem_ref: jax.Ref,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_row_subchunks: int,
):
    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core
    assert sc_info is not None
    num_simd_lanes = sc_info.num_lanes
    hidden_size = in_hbm_ref.shape[-1]
    dtype_bits = jax.dtypes.itemsize_bits(out_hbm_ref.dtype)
    packing = 32 // dtype_bits
    col_size = calculate_col_size(hidden_size, packing)

    assert isinstance(hidden_size, int), (
        f"hidden_size must be int, got {type(hidden_size)}"
    )
    num_cores = jax.lax.axis_size((core_axis_name, subcore_axis_name))
    row_subchunk_size = num_simd_lanes
    row_chunk_size = row_subchunk_size * num_row_subchunks
    block_size = row_chunk_size * num_cores

    recv_sem = sem_ref.at[0]

    copy_start = pltpu.make_async_copy(start_ref, start_vmem_ref.at[:1], recv_sem)
    copy_end = pltpu.make_async_copy(end_ref, end_vmem_ref.at[:1], recv_sem)
    copy_start.start()
    copy_end.start()
    copy_start.wait()
    copy_end.wait()

    start = start_vmem_ref[...][0]
    end = end_vmem_ref[...][0]

    block_start = start // block_size
    block_end = pl.cdiv(end, block_size)
    num_blocks = block_end - block_start
    num_blocks = jnp.where(end <= start, 0, num_blocks)

    num_cols = pl.cdiv(hidden_size, col_size)

    dtype = out_hbm_ref.dtype
    dtype_bits = jax.dtypes.itemsize_bits(dtype)
    packing = 32 // dtype_bits

    core_index = lax.axis_index((core_axis_name, subcore_axis_name))

    # SparseCore `.bitcast()` leverages hardware Row-Packing for 16-bit -> 32-bit conversion.
    # The logical row count halves, while physical column dimensions remain unchanged.
    in_hbm_i32 = in_hbm_ref.bitcast(jnp.int32)
    out_hbm_i32 = out_hbm_ref.bitcast(jnp.int32)

    num_phys_cols = col_size

    # The outer pipeline runs on the Vector Core, hoisting the integer arithmetic required
    # to decode the row-packed indices. This prevents scalar-core instruction starvation
    # during the execution of the indirect `in_specs` block lambdas.

    # TODO(guoweij): The nested `emit_pipeline` design creates a pipeline bubble (DMA wait)
    # when the inner pipeline empties and restarts with new indices. This is a known
    # high-level API limitation, currently amortized by setting a large num_row_subchunks
    # and shouldn't be an issue for most use cases. We should still monitor
    # the bubble's impact on E2E performance and, if needed, manually reimplement
    # this using primitive operations to eliminate the bubble entirely.

    def col_loop(col_base, gather_ref, out_ref, idx_rem, unpack_col_chunk):
        col_slice = pl.ds(col_base, unpack_col_chunk)
        if packing == 1:
            gather_dt = gather_ref.bitcast(dtype)
            out_dt = out_ref.bitcast(dtype)
            out_dt[:, col_slice] = gather_dt[:, col_slice]
        else:
            # Manual bitwise extraction and packing for packing >= 2 (bfloat16, int8, int4)
            # bf16: 0xFFFF, int8: 0xFF, int4: 0xF
            mask = (1 << dtype_bits) - 1
            shift_multiplier = dtype_bits.bit_length() - 1
            for i in range(num_simd_lanes // packing):
                packed_row = jnp.zeros((1, unpack_col_chunk), dtype=jnp.int32)
                for j in range(packing):
                    k = i * packing + j
                    dynamic_shift = jnp.left_shift(idx_rem[k], shift_multiplier)
                    val = jnp.bitwise_right_shift(
                        gather_ref[pl.ds(k, 1), col_slice], dynamic_shift
                    )
                    val = jnp.bitwise_and(val, mask)
                    pack_shift = j * dtype_bits
                    packed_row = jnp.bitwise_or(
                        packed_row, jnp.left_shift(val, pack_shift)
                    )
                out_ref[pl.ds(i, 1), col_slice] = packed_row

    def inner_pipeline(gather_ref, out_ref, idx_ref, unpack_col_chunk):
        row_slice = pl.ds(pl.program_id(0) * row_subchunk_size, row_subchunk_size)
        subchunk_idxs = idx_ref[row_slice]
        if packing > 1:
            # Equivalent to `subchunk_idxs % packing`
            idx_rem = jnp.bitwise_and(subchunk_idxs, packing - 1)
        else:
            idx_rem = jnp.zeros_like(subchunk_idxs)

        col_loop_fn = functools.partial(
            col_loop,
            gather_ref=gather_ref,
            out_ref=out_ref,
            idx_rem=idx_rem,
            unpack_col_chunk=unpack_col_chunk,
        )
        plsc.parallel_loop(0, num_phys_cols, step=unpack_col_chunk)(col_loop_fn)

    def outer_pipeline(idx_ref):
        b = pl.program_id(0)
        b_global = b + block_start

        unpack_col_chunk = 128
        assert num_phys_cols % unpack_col_chunk == 0
        shift_amount = packing.bit_length() - 1
        pltpu.emit_pipeline(
            functools.partial(
                inner_pipeline, idx_ref=idx_ref, unpack_col_chunk=unpack_col_chunk
            ),
            grid=(num_row_subchunks, num_cols),
            in_specs=pl.BlockSpec(
                (pl.Indirect(row_subchunk_size), num_phys_cols),
                lambda r, col_id: (
                    jnp.bitwise_right_shift(
                        idx_ref[pl.ds(r * row_subchunk_size, row_subchunk_size)],
                        shift_amount,
                    ),
                    col_id,
                ),
            ),
            out_specs=pl.BlockSpec(
                (row_subchunk_size // packing, num_phys_cols),
                lambda r, col_id: (
                    (b_global * num_cores + core_index) * num_row_subchunks + r,
                    col_id,
                ),
            ),
        )(in_hbm_i32, out_hbm_i32)

    pltpu.emit_pipeline(
        outer_pipeline,
        grid=(num_blocks,),
        in_specs=pl.BlockSpec(
            (row_chunk_size,),
            lambda b: ((b + block_start) * num_cores + core_index,),
        ),
    )(indices_hbm_ref)


@jax.jit
def ragged_gather_v2(
    x: jax.Array, indices: jax.Array, start: jax.Array, end: jax.Array
) -> jax.Array:
    """Perform gather on indices within dynamic array start and end using BlockSpec."""

    assert x.ndim == 2, "Ragged gather only supports 2d inputs."
    assert indices.ndim == 1, "Ragged gather only supports 1d indices."

    if jnp.isscalar(start):
        start = start[None]
    if jnp.isscalar(end):
        end = end[None]

    dtype = x.dtype
    if dtype not in (jnp.bfloat16, jnp.float32, jnp.int8, jnp.float8_e4m3fn, jnp.int4):
        raise ValueError(f"dtype must be f32, bf16, int8, fp8 (e4m3fn), or int4, but got {dtype}")

    sc_info = pltpu.get_tpu_info().sparse_core
    if sc_info is None:
        return x[indices]

    hidden_size = x.shape[-1]
    out_size = indices.size

    dtype_bits = jax.dtypes.itemsize_bits(dtype)
    packing = 32 // dtype_bits
    col_size = calculate_col_size(hidden_size, packing)

    aligned_hidden_size = ((hidden_size + col_size - 1) // col_size) * col_size

    num_simd_lanes = sc_info.num_lanes
    num_cores = sc_info.num_cores * sc_info.num_subcores
    base_block_size = num_simd_lanes * num_cores

    # Calculate ideal num_row_subchunks to avoid too much padding overhead.
    num_row_subchunks = max(
        1, min(4, (out_size + base_block_size - 1) // base_block_size)
    )

    row_subchunk_size = num_simd_lanes
    row_chunk_size = row_subchunk_size * num_row_subchunks
    block_size = row_chunk_size * num_cores

    out_pad_size = ((out_size + block_size - 1) // block_size) * block_size - out_size
    indices = jnp.pad(indices, ((0, out_pad_size)))

    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=sc_info.num_cores,
        num_subcores=sc_info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    return core_map_kernel(
        functools.partial(
            main_kernel_v2,
            core_axis_name=vector_mesh.core_axis_name,
            subcore_axis_name=vector_mesh.subcore_axis_name,
            num_row_subchunks=num_row_subchunks,
        ),
        out_type=jax.ShapeDtypeStruct(
            (out_size + out_pad_size, aligned_hidden_size), dtype
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        scratch_types=[
            pltpu.VMEM((16,), jnp.int32),
            pltpu.VMEM((16,), jnp.int32),
            pltpu.SemaphoreType.DMA((1,)),
        ],
        mesh=vector_mesh,
        # Without an estimate the scheduler treats the call as free and puts
        # its done right after its start, leaving the TensorCore idle.
        cost_estimate=pl.CostEstimate(
            flops=0, transcendentals=0,
            bytes_accessed=2 * out_size * hidden_size * jnp.dtype(dtype).itemsize),
        name="sc_ragged_gather_v2",
    )(start, end, x, indices)[:out_size, :hidden_size]


# ============================================================================
# SparseCore ragged gather-reduce
# ============================================================================
#
# The routed MoE's return to token order (``ragged_gather_reduce``): each core
# scatter-adds its destination block's valid routes into FP32 VMEM.


_TOKEN_SUBCHUNK = 16
_NUM_TOKEN_SUBCHUNKS = 4
_TOKEN_BLOCK = _TOKEN_SUBCHUNK * _NUM_TOKEN_SUBCHUNKS
_ROUTE_INDEX_BITS = 20
_ROUTE_INDEX_MASK = (1 << _ROUTE_INDEX_BITS) - 1


def _align_to(value: int, alignment: int) -> int:
    return pl.cdiv(value, alignment) * alignment


def _fallback_implementation(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    out = x[indices] * topk_weights[:, None].astype(jnp.float32)
    out = jnp.where(valid_rows_mask[:, None], out, 0)
    out = out.reshape(-1, reduce_group_size, out.shape[-1])
    return jnp.sum(out, axis=1).astype(x.dtype)


def _calculate_num_column_partitions(
    hidden_size: int, num_cores: int, num_lanes: int
) -> int:
    preferred_num_stages = 4
    num_column_partitions = 1
    while (
        num_cores % (num_column_partitions * 2) == 0
        and hidden_size % (num_lanes * num_column_partitions * 2) == 0
        and hidden_size // (num_column_partitions * 2 * num_lanes)
        >= preferred_num_stages
    ):
        num_column_partitions *= 2
    # Destination-major accumulation keeps a dense FP32 token block in VMEM.
    # For columns up to 1024, using one fewer column partition amortizes the
    # per-block metadata, clear, and output-DMA overhead while still leaving
    # enough room for the double-buffered indirect gather.  Wider columns stay
    # with the conservative partitioning selected above.
    if num_column_partitions >= 2 and hidden_size <= 1024 * (
        num_column_partitions // 2
    ):
        num_column_partitions //= 2
    return num_column_partitions


def partition_counts(hidden_size: int, tpu_info) -> tuple[int, int]:
    """Column and row partition counts of the SparseCore subcores for a width."""
    sc_info = tpu_info.sparse_core
    num_cores = sc_info.num_cores * sc_info.num_subcores
    num_column_partitions = _calculate_num_column_partitions(
        hidden_size, num_cores, tpu_info.num_lanes
    )
    return num_column_partitions, num_cores // num_column_partitions


def _main_kernel(
    x_hbm_ref: jax.Ref,
    route_metadata_hbm_ref: jax.Ref,
    route_weights_hbm_ref: jax.Ref,
    route_counts_hbm_ref: jax.Ref,
    out_hbm_ref: jax.Ref,
    accum_vmem_ref: jax.Ref,
    out_tile_vmem_ref: jax.Ref,
    sem_ref: jax.Ref,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_row_partitions: int,
    num_column_partitions: int,
    blocks_per_row_partition: int,
    topk: int,
):
    sc_info = pltpu.get_tpu_info().sparse_core
    assert sc_info is not None
    num_simd_lanes = sc_info.num_lanes
    assert num_simd_lanes == _TOKEN_SUBCHUNK

    core_id = jax.lax.axis_index((core_axis_name, subcore_axis_name))
    row_partition_id = core_id // num_column_partitions
    col_partition_id = core_id % num_column_partitions
    col_size = accum_vmem_ref.shape[-1]
    col_start = col_partition_id * col_size
    metadata_block_size = topk * _TOKEN_BLOCK
    x_32b_hbm_ref = x_hbm_ref.bitcast(jnp.int32)
    out_32b_hbm_ref = out_hbm_ref.bitcast(jnp.uint32)

    def metadata_block_index(block_id):
        return (row_partition_id * blocks_per_row_partition + block_id,)

    @functools.partial(
        pltpu.emit_pipeline,
        grid=(blocks_per_row_partition,),
        in_specs=(
            pl.BlockSpec((metadata_block_size,), metadata_block_index),
            pl.BlockSpec((metadata_block_size,), metadata_block_index),
            pl.BlockSpec((_TOKEN_SUBCHUNK,), metadata_block_index),
        ),
        out_specs=(),
    )
    def token_block_pipeline(
        route_metadata_ref, route_weights_ref, route_counts_ref, output_sem_ref
    ):
        block_id = pl.program_id(0)
        global_block_id = row_partition_id * blocks_per_row_partition + block_id
        valid_route_count = route_counts_ref[...][0]
        fixed_two_routes = valid_route_count == 2 * _TOKEN_BLOCK
        num_route_chunks = pl.cdiv(valid_route_count, _TOKEN_SUBCHUNK)

        # Scatter-add below reads the previous accumulator value, so initialize
        # the complete dense destination block first.  This also supplies zeros
        # for tokens with no local route.
        def zero_column(col_offset):
            col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
            zero = jnp.zeros((_TOKEN_SUBCHUNK,), jnp.float32)
            for token_row in range(_TOKEN_BLOCK):
                accum_vmem_ref[token_row, col_slice] = zero

        plsc.parallel_loop(0, col_size, step=_TOKEN_SUBCHUNK)(zero_column)

        def route_indices_slice(route_chunk):
            start = route_chunk * _TOKEN_SUBCHUNK
            metadata = route_metadata_ref[pl.ds(start, _TOKEN_SUBCHUNK)]
            return jnp.bitwise_and(metadata, _ROUTE_INDEX_MASK)

        @functools.partial(
            pltpu.emit_pipeline,
            grid=(num_route_chunks,),
            in_specs=pl.BlockSpec(
                (pl.Indirect(_TOKEN_SUBCHUNK), col_size),
                lambda route_chunk: (
                    jnp.bitwise_right_shift(
                        route_indices_slice(route_chunk),
                        1,
                    ),
                    col_partition_id,
                ),
            ),
            out_specs=(),
        )
        def route_pipeline(gather_ref):
            route_chunk = pl.program_id(0)
            metadata_start = route_chunk * _TOKEN_SUBCHUNK
            metadata_slice = pl.ds(metadata_start, _TOKEN_SUBCHUNK)
            route_metadata = route_metadata_ref[metadata_slice]
            source_indices = jnp.bitwise_and(route_metadata, _ROUTE_INDEX_MASK)
            route_destinations = jnp.bitwise_right_shift(
                route_metadata, _ROUTE_INDEX_BITS
            )
            route_weights = route_weights_ref[metadata_slice]

            def fixed_two_route_column_loop(col_offset):
                col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
                feature_indices = col_offset + jnp.arange(
                    _TOKEN_SUBCHUNK, dtype=jnp.int32
                )
                route_batch_size = 16
                for route_batch_start in range(0, _TOKEN_SUBCHUNK, route_batch_size):
                    weighted_values = []
                    for route_lane in range(
                        route_batch_start, route_batch_start + route_batch_size
                    ):
                        value_i32 = gather_ref[route_lane, col_slice]
                        shift = jnp.where(
                            jnp.bitwise_and(source_indices[route_lane], 1) == 0,
                            16,
                            0,
                        )
                        shifted = jnp.bitwise_and(
                            jnp.left_shift(value_i32, shift),
                            jnp.int32(-65536),
                        )
                        value_f32 = plsc.bitcast(shifted, jnp.float32)
                        value_f32 *= route_weights[route_lane]
                        weighted_values.append(value_f32)

                    for pair_start in range(0, route_batch_size, 2):
                        route_lane = route_batch_start + pair_start
                        destination_indices = jnp.full_like(
                            feature_indices, route_destinations[route_lane]
                        )
                        plsc.addupdate_scatter(
                            accum_vmem_ref,
                            (destination_indices, feature_indices),
                            weighted_values[pair_start]
                            + weighted_values[pair_start + 1],
                        )

            def generic_column_loop(col_offset):
                col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
                feature_indices = col_offset + jnp.arange(
                    _TOKEN_SUBCHUNK, dtype=jnp.int32
                )
                route_batch_size = 4
                for route_batch_start in range(0, _TOKEN_SUBCHUNK, route_batch_size):
                    weighted_values = []
                    for route_lane in range(
                        route_batch_start, route_batch_start + route_batch_size
                    ):
                        value_i32 = gather_ref[route_lane, col_slice]
                        shift = jnp.where(
                            jnp.bitwise_and(source_indices[route_lane], 1) == 0,
                            16,
                            0,
                        )
                        shifted = jnp.bitwise_and(
                            jnp.left_shift(value_i32, shift),
                            jnp.int32(-65536),
                        )
                        value_f32 = plsc.bitcast(shifted, jnp.float32)
                        value_f32 *= route_weights[route_lane]
                        weighted_values.append(value_f32)

                    for route_batch_lane in range(route_batch_size):
                        route_lane = route_batch_start + route_batch_lane
                        destination_indices = jnp.full_like(
                            feature_indices, route_destinations[route_lane]
                        )
                        route_valid = metadata_start + route_lane < valid_route_count
                        scatter_mask = jnp.broadcast_to(route_valid, (_TOKEN_SUBCHUNK,))
                        plsc.addupdate_scatter(
                            accum_vmem_ref,
                            (destination_indices, feature_indices),
                            weighted_values[route_batch_lane],
                            mask=scatter_mask,
                        )

            @pl.when(fixed_two_routes)
            def run_fixed_two_route_path():
                plsc.parallel_loop(0, col_size, step=_TOKEN_SUBCHUNK)(
                    fixed_two_route_column_loop
                )

            @pl.when(jnp.logical_not(fixed_two_routes))
            def run_generic_path():
                plsc.parallel_loop(0, col_size, step=_TOKEN_SUBCHUNK)(
                    generic_column_loop
                )

        route_pipeline(x_32b_hbm_ref)

        # Convert pairs of FP32 destination rows to row-packed BF16 and write
        # four complete, tile-aligned 16-row output tiles.
        for token_subchunk_pair in range(_NUM_TOKEN_SUBCHUNKS // 2):
            copies = []
            for output_buffer in range(2):
                token_subchunk = token_subchunk_pair * 2 + output_buffer
                accum_row_start = token_subchunk * _TOKEN_SUBCHUNK

                def cast_column(col_offset):
                    col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
                    for token_lane in range(0, _TOKEN_SUBCHUNK, 2):
                        accum_row = accum_row_start + token_lane
                        packed_bf16 = plsc.pack(
                            accum_vmem_ref[accum_row, col_slice],
                            accum_vmem_ref[accum_row + 1, col_slice],
                            format=plsc.PackFormat.INTERLEAVED,
                        )
                        out_tile_vmem_ref[output_buffer, token_lane // 2, col_slice] = (
                            plsc.bitcast(packed_bf16, jnp.uint32)
                        )

                plsc.parallel_loop(0, col_size, step=_TOKEN_SUBCHUNK)(cast_column)

                output_row = (
                    global_block_id * _TOKEN_BLOCK + token_subchunk * _TOKEN_SUBCHUNK
                )
                output_row_packed = pl.multiple_of(
                    output_row // 2, _TOKEN_SUBCHUNK // 2
                )
                copy = pltpu.make_async_copy(
                    out_tile_vmem_ref.at[output_buffer],
                    out_32b_hbm_ref.at[
                        pl.ds(output_row_packed, _TOKEN_SUBCHUNK // 2),
                        pl.ds(col_start, col_size),
                    ],
                    output_sem_ref.at[output_buffer],
                )
                copy.start()
                copies.append(copy)
            for copy in copies:
                copy.wait()

    token_block_pipeline(
        route_metadata_hbm_ref,
        route_weights_hbm_ref,
        route_counts_hbm_ref,
        scratches=(sem_ref,),
    )


@functools.partial(jax.jit, static_argnames=("reduce_group_size",))
def ragged_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    """Destination-major SparseCore gather-reduce: the routed MoE's combine."""
    sc_info = pltpu.get_tpu_info().sparse_core
    if (
        sc_info is None
        or x.dtype != jnp.bfloat16
        or x.shape[-1] % pltpu.get_tpu_info().num_lanes != 0
    ):
        return _fallback_implementation(
            x,
            indices,
            topk_weights,
            valid_rows_mask,
            reduce_group_size,
        )
    dtype_bytes = jax.dtypes.itemsize_bits(x.dtype) // 8
    if jnp.size(x) * dtype_bytes * 2 < pltpu.get_tpu_info().vmem_capacity_bytes * 0.6:
        return _fallback_implementation(
            x,
            indices,
            topk_weights,
            valid_rows_mask,
            reduce_group_size,
        )

    input_size = indices.size
    if input_size % reduce_group_size != 0:
        raise ValueError(f"{input_size=} must be divisible by {reduce_group_size=}")
    # SparseCore row-packing views BF16 as pairs of rows.
    if x.shape[0] % 2:
        x = jnp.pad(x, ((0, 1), (0, 0)))
    num_tokens = input_size // reduce_group_size
    hidden_size = x.shape[-1]
    if x.shape[0] > 1 << _ROUTE_INDEX_BITS:
        raise ValueError(
            f"destination-major gather-reduce supports at most "
            f"{1 << _ROUTE_INDEX_BITS} source rows, got {x.shape[0]}"
        )
    num_column_partitions, num_row_partitions = partition_counts(
        hidden_size, pltpu.get_tpu_info()
    )

    token_alignment = num_row_partitions * _TOKEN_BLOCK
    padded_num_tokens = _align_to(num_tokens, token_alignment)
    token_padding = padded_num_tokens - num_tokens

    indices_2d = indices.reshape(num_tokens, reduce_group_size)
    weights_2d = topk_weights.reshape(num_tokens, reduce_group_size)
    valid_2d = valid_rows_mask.reshape(num_tokens, reduce_group_size)
    if token_padding:
        indices_2d = jnp.pad(indices_2d, ((0, token_padding), (0, 0)))
        weights_2d = jnp.pad(weights_2d, ((0, token_padding), (0, 0)))
        valid_2d = jnp.pad(valid_2d, ((0, token_padding), (0, 0)))

    num_blocks = padded_num_tokens // _TOKEN_BLOCK
    metadata_block_size = _TOKEN_BLOCK * reduce_group_size
    indices_blocks = indices_2d.reshape(num_blocks, metadata_block_size)
    valid_blocks = valid_2d.reshape(num_blocks, metadata_block_size)
    destination_template = jnp.repeat(
        jnp.arange(_TOKEN_BLOCK, dtype=jnp.int32), reduce_group_size
    )
    destination_blocks = jnp.broadcast_to(destination_template, indices_blocks.shape)

    # A stable boolean partition keeps routes for each destination adjacent
    # while placing the exact valid prefix first in every destination block.
    route_order = jnp.argsort(~valid_blocks, axis=1, stable=True)
    compact_valid = jnp.take_along_axis(valid_blocks, route_order, axis=1)
    route_indices = jnp.take_along_axis(indices_blocks, route_order, axis=1)
    route_destinations = jnp.take_along_axis(destination_blocks, route_order, axis=1)
    weights_blocks = weights_2d.reshape(num_blocks, metadata_block_size)
    route_weights = jnp.take_along_axis(weights_blocks, route_order, axis=1)
    route_indices = jnp.where(compact_valid, route_indices, 0).astype(jnp.int32)
    route_destinations = jnp.where(compact_valid, route_destinations, 0).astype(
        jnp.int32
    )
    route_metadata = jnp.bitwise_or(
        route_indices,
        jnp.left_shift(route_destinations, _ROUTE_INDEX_BITS),
    ).reshape(-1)
    route_weights = jnp.where(compact_valid, route_weights, 0).reshape(-1)
    route_weights = route_weights.astype(jnp.float32)
    route_counts = jnp.sum(valid_blocks, axis=1, dtype=jnp.int32)
    paired_destinations = route_destinations[:, : 2 * _TOKEN_BLOCK].reshape(
        num_blocks, _TOKEN_BLOCK, 2
    )
    fixed_two_route_blocks = jnp.logical_and(
        route_counts == 2 * _TOKEN_BLOCK,
        jnp.all(paired_destinations[:, :, 0] == paired_destinations[:, :, 1], axis=1),
    )
    # Count 128 selects the adjacent-pair fast path in the SC kernel. If a full
    # two-route block does not actually have adjacent equal destinations,
    # expose one already-zero compacted slot as a harmless sentinel so the block
    # takes the generic path instead.
    needs_zero_weight_sentinel = jnp.logical_and(
        route_counts == 2 * _TOKEN_BLOCK,
        jnp.logical_not(fixed_two_route_blocks),
    )
    route_counts += needs_zero_weight_sentinel.astype(jnp.int32)
    route_counts = jnp.broadcast_to(
        route_counts[:, None], (num_blocks, _TOKEN_SUBCHUNK)
    ).reshape(-1)

    col_size = hidden_size // num_column_partitions
    blocks_per_row_partition = num_blocks // num_row_partitions
    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=sc_info.num_cores,
        num_subcores=sc_info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    out = core_map_kernel(
        functools.partial(
            _main_kernel,
            core_axis_name=vector_mesh.core_axis_name,
            subcore_axis_name=vector_mesh.subcore_axis_name,
            num_row_partitions=num_row_partitions,
            num_column_partitions=num_column_partitions,
            blocks_per_row_partition=blocks_per_row_partition,
            topk=reduce_group_size,
        ),
        out_type=jax.ShapeDtypeStruct((padded_num_tokens, hidden_size), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            disable_bounds_checks=True,
            needs_layout_passes=False,
        ),
        scratch_types=(
            pltpu.VMEM((_TOKEN_BLOCK, col_size), jnp.float32),
            pltpu.VMEM((2, _TOKEN_SUBCHUNK // 2, col_size), jnp.uint32),
            pltpu.SemaphoreType.DMA((2,)),
        ),
        mesh=vector_mesh,
        name="sc_ragged_gather_reduce_v3",
    )(x, route_metadata, route_weights, route_counts)
    return out[:num_tokens]


# ============================================================================
# TP32 sequence-parallel rings
# ============================================================================
#
# Window all-gather, projection reduce-scatter and the MoE reduce-scatter.
#
# Physical ring position, mesh index, and Torch token owner are distinct.
# Tables explicitly translate all three; never permute the JAX mesh to match
# physical coordinates in the multi-process runtime.


CYCLES = (
    (0, 4, 12, 13, 14, 15, 7, 3, 11, 10, 2, 6, 5, 1, 9, 8),
    (0, 1, 9, 13, 14, 10, 2, 3, 11, 15, 7, 6, 5, 4, 12, 8),
    (0, 1, 2, 3, 7, 6, 14, 15, 11, 10, 9, 8, 12, 13, 5, 4),
)


def ring_tables(devices, token_ranks):
    """Build [family,position/peer/token-owner,mesh-index] lookup tables."""
    if len(devices) != 32 or sorted(token_ranks) != list(range(32)):
        raise ValueError("K3 rings require 32 devices and a token-owner permutation")
    if [d.id for d in devices] != sorted(d.id for d in devices):
        raise ValueError("Production collective mesh must use ascending device IDs")
    coords = [sorted({d.coords[a] for d in devices}) for a in range(3)]
    if tuple(map(len, coords)) != (2, 2, 4):
        raise ValueError("K3 rings require a physical 2x2x4 slice")
    physical_to_mesh = {}
    for i, device in enumerate(devices):
        x, y, z = [coords[a].index(device.coords[a]) for a in range(3)]
        if device.core_on_chip not in (0, 1):
            raise ValueError("Expected two cores per chip")
        physical = ((x * 2 + y) * 4 + z) * 2 + device.core_on_chip
        physical_to_mesh[physical] = i
    if len(physical_to_mesh) != 32:
        raise ValueError("Duplicate physical cores")
    result = []
    for cycle in CYCLES:
        peers = tuple(
            physical_to_mesh[chip * 2 + core] for chip in cycle for core in range(2)
        )
        positions = tuple(peers.index(i) for i in range(32))
        result.append((positions, peers, tuple(token_ranks)))
    return np.asarray(result, np.int32)


# One host's chips, neighbour to neighbour: a pipeline stage's ring visits
# them in this order, both cores of each in turn.
HOST_CHIP_CYCLE = ((0, 0), (1, 0), (1, 1), (0, 1))


def stage_ring_tables(devices, token_ranks=None):
    """``ring_tables`` for a pipeline stage over one host's 2x2 chips (8
    cores). ``devices``: records with chip ``coords`` and ``core_on_chip``
    in mesh order. The host has a single chip cycle, so the three families
    are that cycle: each carries a column third, as on the slice."""
    devices = list(devices)
    if token_ranks is None:
        token_ranks = tuple(range(len(devices)))
    if len(devices) != 8 or sorted(token_ranks) != list(range(8)):
        raise ValueError("K3 stage rings require one host's 8 cores and a token-owner permutation")
    xs = sorted({d.coords[0] for d in devices})
    ys = sorted({d.coords[1] for d in devices})
    where = {
        (xs.index(d.coords[0]), ys.index(d.coords[1]), d.core_on_chip): i
        for i, d in enumerate(devices)
    }
    cycle = [(x, y, c) for x, y in HOST_CHIP_CYCLE for c in (0, 1)]
    if set(where) != set(cycle):
        raise ValueError("K3 stage rings require one host's 2x2 chips with two cores each")
    peers = tuple(where[core] for core in cycle)
    positions = tuple(peers.index(i) for i in range(8))
    return np.asarray([(positions, peers, tuple(token_ranks))] * 3, np.int32)


def mesh_ring_tables(mesh, token_ranks=None):
    """The ring tables of a flat ``k3_sp`` mesh: the 2x2x4 slice's three
    cycles at 32 ranks, one host's chip cycle at 8 (a pipeline stage, whose
    mesh records carry no chip coordinates: the runtime's do)."""
    devices = list(mesh.devices.flat)
    if len(devices) == 32:
        return ring_tables(devices, tuple(range(32)) if token_ranks is None else token_ranks)
    from torch_tpu._internal.distributed import tpu_distributed

    attributes = {a.id: a for a in tpu_distributed.global_device_attributes()}
    return stage_ring_tables([attributes[d.id] for d in devices], token_ranks)


def check_mesh(mesh, what):
    """The ranks of a flat ``k3_sp`` mesh: the slice (32) or a stage (8)."""
    if mesh.size not in (8, 32) or tuple(mesh.axis_names) != ("k3_sp",):
        raise ValueError(f"{what} requires a flat k3_sp mesh of 32 ranks or a stage of 8")
    return mesh.size


def _regions(rows, width):
    """The ring streams' ``(row, rows, column, columns)`` regions: two groups
    x three cycles x two directions. From 64 rows a stream carries a quarter
    of the rows and its cycle's column third; below that a quarter would be
    under a BF16 tile (16 rows), so each stream carries every row and a
    twelfth of the columns."""
    if rows % 16 or width % 128 or width < 384:
        raise ValueError("K3 rings need 16-row ownership and 128-column alignment")
    blocks = width // 128
    if rows < 64:
        if blocks < 12:
            raise ValueError("K3 rings under 64 rows need at least 12 column blocks")
        widths = [128 * (blocks // 12 + (i < blocks % 12)) for i in range(12)]
        cols = np.cumsum([0, *widths[:-1]])
        return tuple((0, rows, int(col), w) for col, w in zip(cols, widths, strict=True))
    if rows % 64:
        raise ValueError("K3 rings need 64-row ownership from 64 rows")
    widths = [128 * (blocks // 3 + (i < blocks % 3)) for i in range(3)]
    cols = [0, widths[0], widths[0] + widths[1]]
    return tuple(
        ((2 * group + direction) * (rows // 4), rows // 4, cols[family], widths[family])
        for group in range(2)
        for family in range(3)
        for direction in range(2)
    )


# Sequence-parallel ownership over ``ranks`` ranks (32 on the slice, 8 in a
# pipeline stage): a rank holds ``stripe`` rows of every ``ranks * stripe``
# tokens. Local row i of rank r is window token
# (i // stripe) * ranks stripe + r * stripe + i % stripe. The stripe is
# STRIPE rows; a window under ranks STRIPE tokens (512, 1024 at 32 ranks) is
# one block of window / ranks-row stripes (``window_stripe``).
STRIPE = 64
SP_BLOCK = 32 * STRIPE
SMALL_WINDOWS = (512, 1024)
MAX_WINDOW = {32: 8192, 8: 16384}


def check_window(seq_len, what, ranks=32):
    """Reject a window the sequence-parallel kernels can't run: 512, 1024,
    or a multiple of 2048 up to 8192 on the slice; a multiple of 512 up to
    16384 in a stage."""
    block = ranks * STRIPE
    small = SMALL_WINDOWS if ranks == 32 else ()
    if seq_len not in small and (seq_len % block or not block <= seq_len <= MAX_WINDOW[ranks]):
        raise ValueError(
            f"{what} needs T in {small} or {block}..{MAX_WINDOW[ranks]} (a multiple of {block})"
        )


def window_stripe(local_rows):
    """The stripe of a window with ``local_rows`` rows per rank."""
    return min(STRIPE, local_rows)


def sp_token_index(rank, local, stripe, ranks=32):
    """Window token index of local row(s) ``local`` on rank ``rank``."""
    return (local // stripe) * ranks * stripe + rank * stripe + local % stripe


def sp_order(window, ranks=32):
    """The window token indices in rank-major ownership order: rank r's
    local rows are ``sp_order(window)[r * window // ranks : (r + 1) * window // ranks]``."""
    stripe = window_stripe(window // ranks)
    return np.arange(window).reshape(-1, ranks, stripe).transpose(1, 0, 2).reshape(-1)


def _stripe_pieces(begin, size, stripe):
    """Split the static local rows [begin, begin + size) at stripe ends:
    (offset in the range, local row, rows) pieces, each contiguous in the
    window."""
    pieces, row = [], begin
    while row < begin + size:
        rows = min(begin + size, (row // stripe + 1) * stripe) - row
        pieces.append((row - begin, row, rows))
        row += rows
    return pieces


def _reduce_scatter_kernel(
    tables_ref,
    x_ref,
    o_ref,
    local_sems,
    send_sems,
    recv_sems,
    credits,
    store_sem,
    local_buf,
    payload,
    received,
    *,
    streams,
    buffers,
):
    tile = pl.program_id(0)
    step = pl.program_id(1)
    rank = lax.axis_index("k3_sp")
    n = tables_ref.shape[-1]
    descriptors = []
    reads = []
    positions = [tables_ref[f, 0, rank] for f in range(3)]

    def ring_coords(pos, family):
        peer = tables_ref[family, 1, pos]
        return (peer,)

    @pl.when(step == 0)
    def enter():
        peers = [
            ring_coords((positions[f] + direction) % n, jnp.int32(f))
            for f in range(3)
            for direction in (-1, 1)
        ]
        sem = pltpu.get_barrier_semaphore()
        for peer in peers:
            pl.semaphore_signal(
                sem, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(sem, len(peers))

        @functools.partial(pl.run_scoped, ready=pltpu.SemaphoreType.REGULAR)
        def initialized(ready):
            for peer in peers:
                pl.semaphore_signal(
                    ready, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH
                )
            pl.semaphore_wait(ready, len(peers))

    for stream, region in enumerate(streams):
        begin, size = region[:2]
        col_begin, width = region[2:] if len(region) == 4 else (0, o_ref.shape[1])
        cols = pl.ds(col_begin, width)
        family = (stream // 2) % 3
        direction = -1 if stream % 2 == 0 else 1
        position = positions[family]
        target = ring_coords((position + direction) % n, jnp.int32(family))
        source = ring_coords((position - direction) % n, jnp.int32(family))
        owner_mesh = ring_coords(
            (position - direction * (step + 1)) % n, jnp.int32(family)
        )[0]
        owner = tables_ref[family, 2, owner_mesh]
        rows = pl.ds(begin, size)
        read = pltpu.make_async_copy(
            x_ref.at[
                pl.ds(owner * o_ref.shape[0] + tile * payload.shape[0] + begin, size),
                cols,
            ],
            local_buf.at[rows, cols],
            local_sems.at[stream],
        )
        read.start()
        reads.append(read)
        descriptors.append((target, source, rows, cols))

    for stream, (target, source, rows, cols) in enumerate(descriptors):

        def remote(index):
            return pltpu.make_async_remote_copy(
                payload.at[rows, cols],
                received.at[index % buffers, rows, cols],
                send_sems.at[stream],
                recv_sems.at[index % buffers, stream],
                device_id=target,
                device_id_type=pl.DeviceIdType.MESH,
            )

        @pl.when(step > 0)
        def wait():
            remote(step - 1).wait()

        reads[stream].wait()

        @pl.when(step == 0)
        def seed():
            payload.at[rows, cols][...] = local_buf.at[rows, cols][...].astype(
                payload.dtype
            )

        @pl.when(step > 0)
        def add():
            payload.at[rows, cols][...] = (
                local_buf.at[rows, cols][...].astype(payload.dtype)
                + received.at[(step - 1) % buffers, rows, cols][...]
            )
            pl.semaphore_signal(
                credits.at[stream],
                1,
                device_id=source,
                device_id_type=pl.DeviceIdType.MESH,
            )

        @pl.when(step < n - 1)
        def forward():
            @pl.when(step >= buffers)
            def acquire():
                pl.semaphore_wait(credits.at[stream], 1)

            remote(step).start()

        @pl.when(step == n - 1)
        def finish():
            pl.semaphore_wait(credits.at[stream], buffers)
            output = pltpu.make_async_copy(
                payload.at[rows, cols],
                o_ref.at[pl.ds(tile * payload.shape[0] + rows.start, rows.size), cols],
                store_sem,
            )
            output.start()
            output.wait()


def _ring_enter(tables, rank):
    """Meet the six ring neighbours (both directions of the three cycles)
    before the first remote copy: each has entered the kernel."""
    n = tables.shape[-1]
    positions = [tables[f, 0, rank] for f in range(3)]
    peers = [tables[f, 1, (positions[f] + d) % n] for f in range(3) for d in (-1, 1)]
    barrier = pltpu.get_barrier_semaphore()
    for p in peers:
        pl.semaphore_signal(barrier, 1, device_id=(p,), device_id_type=pl.DeviceIdType.MESH)
    pl.semaphore_wait(barrier, len(peers))

    @functools.partial(pl.run_scoped, ready=pltpu.SemaphoreType.REGULAR)
    def initialized(ready):
        for p in peers:
            pl.semaphore_signal(ready, 1, device_id=(p,), device_id_type=pl.DeviceIdType.MESH)
        pl.semaphore_wait(ready, len(peers))


def _all_gather_kernel(
    tables,
    x,
    out,
    load_sems,
    send_sems,
    recv_sems,
    credits,
    store_sems,
    payload,
    received,
    *,
    regions,
    buffers,
    striped=False,
):
    """Ring all-gather of every rank's rows. ``striped``: the rows are SP
    stripes (``sp_token_index``) and land in window order; otherwise owner
    o's rows land at ``[o rows, (o + 1) rows)``."""
    tile = pl.program_id(0)
    step = pl.program_id(1)
    rank = lax.axis_index("k3_sp")
    n = tables.shape[-1]
    positions = [tables[f, 0, rank] for f in range(3)]
    local_first = tile * payload.shape[0]
    stripe = window_stripe(x.shape[0])

    def owner(pos, family):
        return tables[family, 1, pos % n]

    def peer(pos, family):
        r = owner(pos, family)
        return (r,)

    pl.when(step == 0)(lambda: _ring_enter(tables, rank))

    @pl.when(step == 0)
    def load():
        for s, (begin, size, col, width) in enumerate(regions):
            rows, cols = pl.ds(begin, size), pl.ds(col, width)
            pltpu.make_async_copy(
                x.at[pl.ds(local_first + begin, size), cols],
                payload.at[rows, cols],
                load_sems.at[s],
            ).start()

    for s, (begin, size, col, width) in enumerate(regions):
        family = (s // 2) % 3
        direction = -1 if s % 2 == 0 else 1
        pos = positions[family]
        target = peer(pos + direction, jnp.int32(family))
        source = peer(pos - direction, jnp.int32(family))
        rows, cols = pl.ds(begin, size), pl.ds(col, width)

        def remote(index):
            return pltpu.make_async_remote_copy(
                payload.at[rows, cols],
                received.at[index % buffers, rows, cols],
                send_sems.at[s],
                recv_sems.at[index % buffers, s],
                device_id=target,
                device_id_type=pl.DeviceIdType.MESH,
            )

        def store(index, begin=begin, size=size, cols=cols, s=s):
            owner_mesh = owner(pos - direction * index, jnp.int32(family))
            dest = tables[family, 2, owner_mesh]
            if not striped:
                return [pltpu.make_async_copy(
                    payload.at[pl.ds(begin, size), cols],
                    out.at[
                        pl.ds(dest * x.shape[0] + tile * payload.shape[0] + begin, size),
                        cols,
                    ],
                    store_sems.at[s],
                )]
            return [
                pltpu.make_async_copy(
                    payload.at[pl.ds(row, count), cols],
                    out.at[pl.ds(sp_token_index(dest, local_first + row, stripe, n), count), cols],
                    store_sems.at[s],
                )
                for _, row, count in _stripe_pieces(begin, size, stripe)
            ]

        @pl.when(step == 0)
        def seed():
            pltpu.make_async_copy(
                x.at[pl.ds(local_first + begin, size), cols],
                payload.at[rows, cols],
                load_sems.at[s],
            ).wait()

        @pl.when(step > 0)
        def receive():
            remote(step - 1).wait()
            for copy in store(step - 1):
                copy.wait()
            payload.at[rows, cols][...] = received.at[(step - 1) % buffers, rows, cols][
                ...
            ]
            pl.semaphore_signal(
                credits.at[s], 1, device_id=source, device_id_type=pl.DeviceIdType.MESH
            )

        for copy in store(step):
            copy.start()

        @pl.when(step < n - 1)
        def forward():
            @pl.when(step >= buffers)
            def acquire():
                pl.semaphore_wait(credits.at[s], 1)

            remote(step).start()

        @pl.when(step == n - 1)
        def finish():
            for copy in store(step):
                copy.wait()
            pl.semaphore_wait(credits.at[s], buffers)


def _project_ring_kernel(
    tables_ref,
    x_ref,
    w_ref,
    o_ref,
    local_sems,
    send_sems,
    recv_sems,
    credits,
    store_sem,
    local_buf,
    payload,
    received,
    weights,
    inputs,
    weight_sem,
    *,
    streams,
    buffers,
):
    tile = pl.program_id(0)
    step = pl.program_id(1)

    descriptors = []
    reads = []
    rank = lax.axis_index("k3_sp")
    n = tables_ref.shape[-1]
    positions = [tables_ref[f, 0, rank] for f in range(3)]
    stripe = window_stripe(o_ref.shape[0])

    def ring_coords(pos, family):
        peer = tables_ref[family, 1, pos]
        return (peer,)

    @pl.when(step == 0)
    def enter():
        peers = [
            ring_coords((positions[f] + direction) % n, jnp.int32(f))
            for f in range(3)
            for direction in (-1, 1)
        ]
        sem = pltpu.get_barrier_semaphore()
        for peer in peers:
            pl.semaphore_signal(
                sem, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(sem, len(peers))

        @functools.partial(pl.run_scoped, ready=pltpu.SemaphoreType.REGULAR)
        def initialized(ready):
            for peer in peers:
                pl.semaphore_signal(
                    ready, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH
                )
            pl.semaphore_wait(ready, len(peers))

    @pl.when((tile == 0) & (step == 0))
    def load_weight():
        pltpu.async_copy(w_ref, weights, weight_sem).wait()

    for stream, region in enumerate(streams):
        begin, size = region[:2]
        col_begin, width = region[2:] if len(region) == 4 else (0, o_ref.shape[1])
        cols = pl.ds(col_begin, width)
        family = (stream // 2) % 3
        direction = -1 if stream % 2 == 0 else 1
        position = positions[family]
        target = ring_coords((position + direction) % n, jnp.int32(family))
        source = ring_coords((position - direction) % n, jnp.int32(family))
        peer = tables_ref[family, 1, (position - direction * (step + 1)) % n]
        owner = tables_ref[family, 2, peer]
        rows = pl.ds(begin, size)
        # The window rows of the owner's stripes (``sp_token_index``).
        read = [
            pltpu.make_async_copy(
                x_ref.at[
                    pl.ds(sp_token_index(owner, tile * payload.shape[0] + row, stripe, n), count), :
                ],
                inputs.at[stream, pl.ds(offset, count)],
                local_sems.at[stream],
            )
            for offset, row, count in _stripe_pieces(begin, size, stripe)
        ]
        for copy in read:
            copy.start()
        reads.append(read)
        descriptors.append((target, source, rows, cols))

    for stream, (target, source, rows, cols) in enumerate(descriptors):

        def remote(index):
            return pltpu.make_async_remote_copy(
                payload.at[rows, cols],
                received.at[index % buffers, rows, cols],
                send_sems.at[stream],
                recv_sems.at[index % buffers, stream],
                device_id=target,
                device_id_type=pl.DeviceIdType.MESH,
            )

        @pl.when(step > 0)
        def wait_before():
            remote(step - 1).wait()

        for copy in reads[stream]:
            copy.wait()
        local_buf.at[rows, cols][...] = jnp.dot(
            inputs[stream], weights[:, cols], preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16)

        @pl.when(step == 0)
        def seed():
            payload.at[rows, cols][...] = local_buf.at[rows, cols][...].astype(
                payload.dtype
            )

        @pl.when(step > 0)
        def add():
            payload.at[rows, cols][...] = (
                local_buf.at[rows, cols][...].astype(payload.dtype)
                + received.at[(step - 1) % buffers, rows, cols][...]
            )
            pl.semaphore_signal(
                credits.at[stream],
                1,
                device_id=source,
                device_id_type=pl.DeviceIdType.MESH,
            )

        @pl.when(step < n - 1)
        def forward():
            @pl.when(step >= buffers)
            def acquire():
                pl.semaphore_wait(credits.at[stream], 1)

            remote(step).start()

        @pl.when(step == n - 1)
        def finish():
            pl.semaphore_wait(credits.at[stream], buffers)
            local_buf.at[rows, cols][...] = payload.at[rows, cols][...].astype(
                local_buf.dtype
            )
            output = pltpu.make_async_copy(
                local_buf.at[rows, cols],
                o_ref.at[pl.ds(tile * payload.shape[0] + rows.start, rows.size), cols],
                store_sem,
            )
            output.start()
            output.wait()


def project_reduce_scatter(x, w, mesh, tables, *, collective_id, tile_rows=256):
    """Fuse a BF16 attention output projection (the rank's heads) with the
    sequence-parallel reduce-scatter, over row tiles of ``tile_rows``."""
    n = check_mesh(mesh, "K3 projection/RS")
    width = x.shape[2]
    if x.shape[0] != n or x.shape[1] % n or width % 128 or w.shape != (n, width, 7168):
        raise ValueError("K3 projection/RS requires rank-leading [ranks, T, heads * 128] inputs")
    if x.dtype != jnp.bfloat16 or w.dtype != jnp.bfloat16:
        raise ValueError("K3 projection/RS requires BF16 operands")
    tokens = x.shape[1]
    rows = tokens // n
    tile = min(rows, tile_rows)
    if rows % tile:
        raise ValueError("K3 projection/RS needs whole row tiles per rank")
    regions = _regions(tile, 7168)
    streams = len(regions)
    chunk = max(size for _, size, _, _ in regions)

    def local(x, w):
        x = x.reshape(tokens, width)
        w = w.reshape(width, 7168)
        hbm = pl.BlockSpec(memory_space=pltpu.HBM)
        scratch = (
            pltpu.SemaphoreType.DMA((streams,)),
            pltpu.SemaphoreType.DMA((streams,)),
            pltpu.SemaphoreType.DMA((2, streams)),
            pltpu.SemaphoreType.REGULAR((streams,)),
            pltpu.SemaphoreType.DMA,
            pltpu.VMEM((tile, 7168), jnp.bfloat16),
            pltpu.VMEM((tile, 7168), jnp.bfloat16),
            pltpu.VMEM((2, tile, 7168), jnp.bfloat16),
            pltpu.VMEM((width, 7168), jnp.bfloat16),
            pltpu.VMEM((streams, chunk, width), jnp.bfloat16),
            pltpu.SemaphoreType.DMA,
        )
        return pl.pallas_call(
            functools.partial(_project_ring_kernel, streams=regions, buffers=2),
            out_shape=jax.ShapeDtypeStruct((rows, 7168), jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=(hbm, hbm),
                out_specs=hbm,
                scratch_shapes=scratch,
                grid=(rows // tile, n),
            ),
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                vmem_limit_bytes=(40 if n == 32 else 64) * 1024**2,
            ),
            name="kimi_attention_project_ring_rs",
        )(jnp.asarray(tables, jnp.int32), x, w)

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P("k3_sp", None, None), P("k3_sp", None, None)),
        out_specs=P("k3_sp", None),
        check_vma=False,
    )(x, w)


# ============================================================================
# Fused grouped matmul (expert FFN)
# ============================================================================
#
# The grouped matmul adapted from ``kernels/megablox/gmm_v2.py``: the routed
# experts' gate/up (fused SiTU) and down projections, MXFP4 weights.


def situ_and_mul(
    gate: jax.Array, up: jax.Array, beta: float, linear_beta: float | None
) -> jax.Array:
    gate = beta * jnp.tanh(gate / beta) * jax.nn.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * jnp.tanh(up / linear_beta)
    return gate * up


def interleave_lane(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    """Interleaves two arrays along lane dim at zero-cost."""
    assert lhs.shape == rhs.shape
    chunk_size = pltpu.get_tpu_info().num_lanes
    num_chunks = lhs.shape[-1] // chunk_size
    lhs_chunks = jnp.split(lhs, num_chunks, axis=-1)
    rhs_chunks = jnp.split(rhs, num_chunks, axis=-1)
    interleaved = []
    for i in range(num_chunks):
        interleaved += [lhs_chunks[i], rhs_chunks[i]]
    return jnp.concat(interleaved, axis=-1)


def deinterleave_lane(val: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Deinterleaves an array along lane dim at zero-cost."""
    chunk_size = pltpu.get_tpu_info().num_lanes
    num_chunks = val.shape[-1] // chunk_size
    chunks = jnp.split(val, num_chunks, axis=-1)
    lhs = jnp.concat(chunks[0::2], axis=-1)
    rhs = jnp.concat(chunks[1::2], axis=-1)
    return lhs, rhs


def apply_act_fn(acc: jax.Array, fuse_act: str | None):
    """Applies a fused activation function to the accumulator.

    This function is used when an activation function is fused with the matrix
    multiplication. The input accumulator `acc` is expected to contain
    concatenated results for both the 'gate' and 'up' projections.

    Args:
        acc: The accumulator array, with the last dimension being 2 * tile_n.
        fuse_act: None, or ``"situ:<beta>:<linear_beta>"`` (``linear_beta``
            may be ``none``) for the SiTU gated activation.

    Returns:
        The result of applying the activation function.

    Raises:
        NotImplementedError: If an unsupported `fuse_act` is provided.
    """

    if fuse_act is None:
        return acc

    acc_gate, acc_up = deinterleave_lane(acc)
    match fuse_act:
        case str() if fuse_act.startswith("situ:"):
            _, beta, linear_beta = fuse_act.split(":")
            linear_beta = None if linear_beta == "none" else float(linear_beta)
            return situ_and_mul(acc_gate, acc_up, float(beta), linear_beta)
        case _:
            raise NotImplementedError(f"Unsupported activation function: {fuse_act}")


def align_to(x, a):
    return pl.cdiv(x, a) * a


def _get_lhs_sublane_size(dtype: jnp.dtype, size_m: int) -> int:
    """Sublane block size the kernel processes lhs rows in."""
    size_lhs_sublane = pltpu.get_tpu_info().get_sublane_tiling(dtype)
    size_lhs_sublane = min(size_lhs_sublane, size_m)
    return size_lhs_sublane


def get_packing_factor(storage_dtype: jnp.dtype, quant_dtype: jnp.dtype | None) -> int:
    if quant_dtype is None or quant_dtype == storage_dtype:
        return 1
    storage_bits = jax.dtypes.itemsize_bits(storage_dtype)
    quant_bits = jax.dtypes.itemsize_bits(quant_dtype)
    packing_factor, remainder = divmod(storage_bits, quant_bits)
    if remainder != 0:
        raise ValueError(
            f"Storage dtype {storage_dtype} is not divisible by "
            f"quant dtype {quant_dtype}"
        )
    return packing_factor


class RhsRef(ABC):
    """Abstract class that defines interfaces for rhs values."""

    @abstractmethod
    def get_weight(self) -> jax.Array: ...

    @abstractmethod
    def get_scale(self, replicate_size: int | None = None) -> jax.Array:
        """Returns scale array, optionally replicated across sublanes."""
        ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WeightsRef(RhsRef):
    """Dataclass for a single weights."""

    weight: Any
    scale: Any | None

    def get_weight(self) -> jax.Array:
        return self.weight[...]

    def get_scale(self, replicate_size: int | None = None) -> jax.Array:
        assert self.scale is not None
        if self.scale.dtype == jnp.uint8:  # E8M0 exponents [blocks, n]
            exponents = self.scale[...]
            powers = e8m0_scale(exponents)[:, None, :]
            return jnp.broadcast_to(powers, (exponents.shape[0], replicate_size or 1, exponents.shape[1]))
        if replicate_size is not None:
            # Perform zero-stride load for efficient broadcasting across sublanes.
            return self.scale[:, pl.ds(0, replicate_size, 0), :]
        return self.scale[...]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FusedWeightsRef(RhsRef):
    """Dataclass for gate and up weights used in fused activation."""

    gate: WeightsRef
    up: WeightsRef

    def get_weight(self) -> jax.Array:
        w_gate = self.gate.get_weight()
        w_up = self.up.get_weight()
        return interleave_lane(w_gate, w_up)

    def get_scale(self, replicate_size: int | None = None) -> jax.Array:
        s_gate = self.gate.get_scale(replicate_size)
        s_up = self.up.get_scale(replicate_size)
        return interleave_lane(s_gate, s_up)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
    gm_id_to_group_id: jax.Array
    gm_id_to_m_offset: jax.Array


@dataclasses.dataclass(frozen=True)
class TileSizes:
    tile_m: int
    tile_k: int
    tile_n: int
    bucket_base: int


@dataclasses.dataclass(frozen=True)
class Dimensions:
    size_m: int
    size_k: int
    size_n: int
    size_group: int
    size_lhs_group: int
    size_lhs_sublane: int


@dataclasses.dataclass(frozen=True)
class InputConfigs:
    quant_dtype: jnp.dtype | None
    quant_block_size: int | None
    dtype: jnp.dtype
    has_scale: bool = False
    # Scales as raw MXFP4 E8M0 exponent bytes ([group, blocks, n] uint8),
    # converted per tile in VMEM, instead of [group, blocks, 1, n] FP32.
    scale_e8m0: bool = False

    @property
    def should_unpack(self) -> bool:
        """True if rhs weights are sub-byte (e.g. INT4) packed in INT32/UINT32 carriers."""
        return get_packing_factor(self.dtype, self.quant_dtype) > 1

    @property
    def should_dequantize_before_matmul(self) -> bool:
        """Dequantize rhs before matmul if block size limits MXU utilization."""
        if not self.has_scale:
            return False
        assert self.quant_block_size is not None
        mxu_size = pltpu.get_tpu_info().mxu_column_size
        return self.quant_block_size < mxu_size

    @property
    def should_dequantize_after_matmul(self) -> bool:
        return self.has_scale and not self.should_dequantize_before_matmul


@dataclasses.dataclass(frozen=True)
class GmmConfigs:
    tiles: TileSizes
    dims: Dimensions
    lhs_cfgs: InputConfigs
    rhs_cfgs: InputConfigs
    out_dtype: jnp.dtype
    acc_dtype: jnp.dtype
    fuse_act: str | None

    @property
    def num_quant_blocks_per_tile_k(self) -> int:
        return pl.cdiv(self.tiles.tile_k, self.rhs_cfgs.quant_block_size)

    @property
    def out_size_n(self) -> int:
        if self.fuse_act is None:
            return self.dims.size_n
        else:
            return self.dims.size_n // 2


TileFn = Callable[[Dimensions, InputConfigs, InputConfigs, int, str | None], TileSizes]


class IndexMaps:
    """Index maps for GMM kernel."""

    def __init__(self, metadata_ref: MetadataRef, cfgs: GmmConfigs):
        self.metadata_ref = metadata_ref
        self.cfgs = cfgs

    def lhs_index_map(self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array):
        m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
        m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

        row_start = m_start // self.cfgs.dims.size_lhs_sublane
        row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
        row_size = row_end - row_start

        return (pl.ds(row_start, row_size), 0, k_id)

    def rhs_weight_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array):
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        return (group_id, k_id, n_id)

    def rhs_scale_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array):
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        # Simply multiplying k_id by num_quant_blocks_per_tile_k will not work
        # since a single quant block could be shared along multiple k tile.
        k_row = k_id * self.cfgs.tiles.tile_k
        b_row = k_row // self.cfgs.rhs_cfgs.quant_block_size
        b_tile_id = b_row // self.cfgs.num_quant_blocks_per_tile_k
        if self.cfgs.rhs_cfgs.scale_e8m0:
            return (group_id, b_tile_id, n_id)
        return (group_id, b_tile_id, 0, n_id)

    def out_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array):
        is_last_gm = gm_id == (pl.num_programs(1) - 1)
        m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
        m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

        row_start = m_start // self.cfgs.dims.size_lhs_sublane
        capped_row_end = m_end // self.cfgs.dims.size_lhs_sublane
        last_row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
        row_end = jnp.where(is_last_gm, last_row_end, capped_row_end)
        row_size = row_end - row_start

        return (pl.ds(row_start, row_size), 0, n_id)


def generate_block_specs(
    metadata_ref: MetadataRef, cfgs: GmmConfigs
) -> tuple[tuple[pl.BlockSpec, WeightsRef], pl.BlockSpec]:
    """Generates block specs for the given lhs, rhs, and out refs."""

    index_map = IndexMaps(metadata_ref, cfgs)
    bounded_slice_gm = pl.BoundedSlice(cfgs.tiles.tile_m // cfgs.dims.size_lhs_sublane)

    lhs_block_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_k),
        index_map.lhs_index_map,
    )

    rhs_weight_spec = pl.BlockSpec(
        (None, cfgs.tiles.tile_k, cfgs.tiles.tile_n),
        index_map.rhs_weight_index_map,
        pipeline_mode=pl.Buffered(buffer_count=3),
    )
    rhs_scale_block_spec = None
    if cfgs.rhs_cfgs.has_scale:
        rhs_scale_block_spec = pl.BlockSpec(
            (None, cfgs.num_quant_blocks_per_tile_k, cfgs.tiles.tile_n) if cfgs.rhs_cfgs.scale_e8m0
            else (None, cfgs.num_quant_blocks_per_tile_k, 1, cfgs.tiles.tile_n),
            index_map.rhs_scale_index_map,
        )

    rhs_block_spec = WeightsRef(
        weight=rhs_weight_spec,
        scale=rhs_scale_block_spec,
    )

    out_block_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_n),
        index_map.out_index_map,
    )

    return (lhs_block_spec, rhs_block_spec), out_block_spec


def inner_kernel(
    # In
    tiled_lhs_ref: jax.Array,
    # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_k]
    tiled_rhs_ref: RhsRef,  # [tile_k, tile_n]
    # Out
    tiled_out_ref: jax.Array,
    # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_n]
    # Scratch
    partial_out_ref: jax.Array,  # [size_lhs_sublane, tile_n]
    acc_ref: jax.Array,  # [tile_m, tile_n]
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
):
    """Inner kernel invoked by emit_pipeline to perform matmul.

    tiled_lhs_ref and tiled_out_ref point to rows [m_start:m_end] of lhs and out.
    Additionally, m_start and m_end do not have to align with tile boundaries
    [m_offset:m_offset+tile_m]. Therefore, rows [m_offset:m_start] and
    [m_end:m_offset+tile_m] of tiled_lhs_ref and tiled_out_ref will contain
    invalid data and need to be masked out.

    Args:
        tiled_lhs_ref: Contains value lhs[m_start:m_end, k_start:k_end]
        tiled_rhs_ref: Contains value rhs[g_id, k_start:k_end, n_start:n_end], where
            g_id is the group associated with lhs[m_start:m_end, :]
        tiled_out_ref: Contains value out[m_start:m_end, n_start:n_end]
        partial_out_ref: Contains last size_lhs_sublane rows of the previous output.
            Will be initialized to zero if this is first tile for grid[n_id, :, :].
        acc_ref: Reference to the accumulator.
        metadata_ref: Reference to the metadata.
        cfgs: GmmConfigs.
    """

    gm_id = pl.program_id(1)
    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
    m_offset = m_start - m_start % cfgs.dims.size_lhs_sublane

    m_start_local = m_start - m_offset
    m_end_local = m_end - m_offset

    def _matmul(is_first_k_step: bool, is_last_k_step: bool, bucket_m: int):
        tpu_info = pltpu.get_tpu_info()
        mxu_size = tpu_info.mxu_column_size

        # Step 1: Input pre-processing.
        tiled_lhs = tiled_lhs_ref.reshape(-1, cfgs.tiles.tile_k)[:bucket_m]
        tiled_rhs = tiled_rhs_ref.get_weight()

        # This should only be taken in the case where we don't requantize
        # the scales and thus we need to dequantize inside VMEM to avoid small
        # contracting dimensions.
        rhs_tile_n = tiled_rhs.shape[1]
        rhs_qbs = cfgs.rhs_cfgs.quant_block_size
        if cfgs.rhs_cfgs.should_dequantize_before_matmul:
            tiled_rhs_scale = tiled_rhs_ref.get_scale(replicate_size=rhs_qbs).astype(
                cfgs.lhs_cfgs.dtype
            )
            num_blocks = cfgs.num_quant_blocks_per_tile_k
            tiled_rhs_dequant = tiled_rhs.astype(cfgs.lhs_cfgs.dtype).reshape(
                num_blocks, rhs_qbs, rhs_tile_n
            )
            tiled_rhs_dequant = tiled_rhs_dequant * tiled_rhs_scale
            tiled_rhs = tiled_rhs_dequant.reshape(cfgs.tiles.tile_k, rhs_tile_n)
            rhs_qbs = cfgs.tiles.tile_k

        valid_k = cfgs.dims.size_k % cfgs.tiles.tile_k
        if is_last_k_step and valid_k != 0:
            mask_rhs = lax.broadcasted_iota(jnp.int32, tiled_rhs.shape, 0) < valid_k
            tiled_rhs = jnp.where(mask_rhs, tiled_rhs, 0)

        # Step 2: Matmul.
        acc_list = []
        for start_n in range(0, rhs_tile_n, mxu_size):
            end_n = min(rhs_tile_n, start_n + mxu_size)
            col_size = end_n - start_n

            acc_n = jnp.zeros((bucket_m, col_size), dtype=acc_ref.dtype)
            for start_k in range(0, cfgs.tiles.tile_k, rhs_qbs):
                end_k = min(cfgs.tiles.tile_k, start_k + rhs_qbs)

                block_acc = jnp.matmul(
                    tiled_lhs[:, start_k:end_k],
                    tiled_rhs[start_k:end_k, start_n:end_n],
                    preferred_element_type=jnp.float32,
                ).astype(acc_ref.dtype)

                if cfgs.rhs_cfgs.should_dequantize_after_matmul:
                    b_id = start_k // rhs_qbs
                    rhs_scale_replicated = tiled_rhs_ref.get_scale(
                        replicate_size=bucket_m
                    )[b_id, :, start_n : start_n + col_size]
                    block_acc *= rhs_scale_replicated.astype(acc_ref.dtype)

                acc_n += block_acc
            acc_list.append(acc_n)
        acc = jnp.concatenate(acc_list, axis=1)

        # Step 3: Output post-processing.
        if not is_first_k_step:
            acc = jnp.pad(acc, ((cfgs.tiles.tile_m - bucket_m, 0), (0, 0)))
            acc += acc_ref[...]
        acc_m = acc.shape[0]

        if is_last_k_step:
            acc = apply_act_fn(acc, cfgs.fuse_act)

            # Mask out rows that do not belong to the current group.
            iota = lax.broadcasted_iota(jnp.int32, acc.shape, 0)
            mask = jnp.logical_and(m_start_local <= iota, iota < m_end_local)
            acc_masked = jnp.where(mask, acc, 0)

            # Write the final output to the output ref.
            tiled_out_2d_ref = tiled_out_ref.reshape(-1, cfgs.tiles.tile_n)
            tiled_out_2d_ref[:acc_m] = acc_masked.astype(tiled_out_ref.dtype)

            # If this is the first tile for grid[n_id, :, :], we initialize the
            # partial out to zeros. Otherwise, partial out from last tile of
            # grid[n_id-1, :, :] can be used and cause numeric issues.
            partial_out_zeros = jnp.zeros_like(partial_out_ref)

            # Accumulate the partial output from the previous step.
            tiled_out_ref[0] += jnp.where(
                gm_id == 0, partial_out_zeros, partial_out_ref[...]
            )

            # Consider following case where size_lhs_sublane = 4, number denotes group
            # id and | denotes boundaries between sublanes:
            # | 0 0 1 2 | 2 2 2 2 | 3 3 4 4 |
            #
            # Assuming group id of current step is 1, current step will not completely
            # fill size_lhs_sublane rows and will be revisited at the next step. By
            # storing the partial rows into the partial_out_ref, the next step can
            # read them and accumulate to them.  Additionally, for group id of 2,
            # since it completely fills the size_lhs_sublane rows, we need to zero out
            # partial_out_ref to avoid numeric error for group 3.
            last_row = m_end_local // cfgs.dims.size_lhs_sublane
            partial_out_ref[...] = jnp.where(
                m_end_local % cfgs.dims.size_lhs_sublane == 0,
                partial_out_zeros,
                tiled_out_ref[last_row],
            )
        else:
            acc_ref[:acc_m] = acc

    def run_matmul_step(bucket_idx: int):
        bucket_m = cfgs.tiles.bucket_base * (bucket_idx + 1)

        @jax.named_scope(f"bm{bucket_m}_first_last")
        def matmul_first_last():
            _matmul(is_first_k_step=True, is_last_k_step=True, bucket_m=bucket_m)

        @jax.named_scope(f"bm{bucket_m}_first")
        def matmul_first():
            _matmul(is_first_k_step=True, is_last_k_step=False, bucket_m=bucket_m)

        @jax.named_scope(f"bm{bucket_m}_mid")
        def matmul_mid():
            _matmul(is_first_k_step=False, is_last_k_step=False, bucket_m=bucket_m)

        @jax.named_scope(f"bm{bucket_m}_last")
        def matmul_last():
            _matmul(is_first_k_step=False, is_last_k_step=True, bucket_m=bucket_m)

        num_k = pl.num_programs(2)
        k_id = pl.program_id(2)
        is_first_k_step = k_id == 0
        is_last_k_step = k_id == (num_k - 1)

        if bucket_m == cfgs.tiles.tile_m:
            lax.cond(
                is_first_k_step,
                lambda: lax.cond(is_last_k_step, matmul_first_last, matmul_first),
                lambda: lax.cond(is_last_k_step, matmul_last, matmul_mid),
            )
        else:
            # partial m is only invoked at last matmul.
            lax.cond(is_first_k_step, matmul_first_last, matmul_last)

    branches = []
    for bucket_idx in range(cfgs.tiles.tile_m // cfgs.tiles.bucket_base):
        branches.append(functools.partial(run_matmul_step, bucket_idx=bucket_idx))
    bucket_idx = m_end_local // cfgs.tiles.bucket_base
    lax.switch(bucket_idx, branches)


def fill_metadata(
    lhs_group_sizes_ref: jax.Array,  # int32[size_lhs_group]
    group_offset_ref: jax.Array,  # int32[1]
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
) -> jax.Array:
    """Fills the metadata for the given lhs group sizes and group offset.

    Iterates over the lhs group sizes and if the group id is valid, determines
    the number of gm tiles that are needed to process the current group. Then,
    it fills starting and ending offset (gm_id_to_m_offset), and the group id
    (gm_id_to_group_id) for each gm tile.

    Args:
        lhs_group_sizes_ref: The group sizes of lhs.
        group_offset_ref: Offset of the first group to process.
        metadata_ref: Metadata that is used to determine the group id and m offsets
            for each gmm tile.
        cfgs: GmmConfigs.

    Returns:
        The number of gm tiles to process lhs with given group offset.
    """

    group_offset = group_offset_ref[0]
    max_num_group = group_offset + cfgs.dims.size_group
    metadata_ref.gm_id_to_m_offset[0] = 0

    @jax.named_scope("inner_tm_loop")
    def inner_tm_loop(tm_id, curr_m_offset, *, end_m_offset, group_id):
        local_offset = curr_m_offset % cfgs.dims.size_lhs_sublane
        tm_size = jnp.minimum(
            cfgs.tiles.tile_m - local_offset, end_m_offset - curr_m_offset
        )

        metadata_ref.gm_id_to_group_id[tm_id] = group_id

        next_m_offset = curr_m_offset + tm_size
        metadata_ref.gm_id_to_m_offset[tm_id] = curr_m_offset
        metadata_ref.gm_id_to_m_offset[tm_id + 1] = next_m_offset

        return next_m_offset

    @jax.named_scope("outer_group_loop")
    def outer_group_loop(lhs_group_id, carry):
        num_gm, start_m_offset = carry

        group_id = lhs_group_id - group_offset
        group_size = lhs_group_sizes_ref[lhs_group_id]
        end_m_offset = start_m_offset + group_size

        # Assume following arguments:
        # - size_lhs_sublane & tile_m = 4
        # - group_size = 3
        # - start_m_offset = 7
        #
        # If we visualize it, it will look like this where:
        # - |: denotes boundaries between sublanes
        # - 0: denotes values for other groups
        # - 1: denotes values for the current group
        # | 0 0 0 0 | 0 0 0 1 | 1 1 0 0 |
        #
        # In this example, we see that we require processing 2 m tiles.
        # But, performing a naive cdiv(group_size, tile_m) will return 1.
        # Instead, adding local_offset will give us the correct value.
        local_offset = start_m_offset % cfgs.dims.size_lhs_sublane
        aligned_group_size = group_size + local_offset
        curr_num_gm = pl.cdiv(aligned_group_size, cfgs.tiles.tile_m)

        # We need to handle cases where we should not process the group.
        # 1. Even if group_size is 0, if local_offset is not 0, cdiv will return 1.
        # 2. If group comes before the group_offset, we should not process it.
        should_process = jnp.logical_and(group_size > 0, group_id >= 0)
        curr_num_gm = jnp.where(should_process, curr_num_gm, 0)
        next_num_gm = num_gm + curr_num_gm

        tm_loop_fn = functools.partial(
            inner_tm_loop,
            end_m_offset=end_m_offset,
            group_id=group_id,
        )
        lax.fori_loop(num_gm, next_num_gm, tm_loop_fn, start_m_offset)

        return next_num_gm, end_m_offset

    num_gm, _ = lax.fori_loop(0, max_num_group, outer_group_loop, (0, 0))
    return num_gm


def kernel_main(
    # Scalar prefetch
    lhs_group_sizes_ref: jax.Array,  # int32[size_lhs_group]
    group_offset_ref: jax.Array,  # int32[1]
    rhs_layer_ref: jax.Array,  # int32[1]
    # In
    lhs_ref: jax.Array,  # [size_m, size_k]
    rhs_ref: WeightsRef,  # [size_group, size_k, size_n]
    # Out
    out_ref: jax.Array,  # [size_m, size_n]
    # Scratch memory
    partial_out_ref: jax.Array,  # [size_lhs_sublane, tile_n]
    acc_ref: jax.Array,  # [tile_m, tile_n]
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
):
    """Entry point for GMM kernel.

    Computes metadata to determine which rows of lhs needs processing and how
    they will be tiled. And then, invoke inner kernel using metadata.

    Uses the following notation:
    - g: rhs group dimension
    - m: Batch dimension
    - gm: Batch tiling dimension. Aligned to size_lhs_sublane and has tile size
        of tile_m. Skips over empty groups and accounts for revisited tiles.
    - k: in dimension
    - n: out dimension

    Args:
        lhs_group_sizes_ref: Reference to the group sizes of lhs.
        group_offset_ref: Reference to the group offset.
        lhs_ref: Reference to the lhs.
        rhs_ref: Reference to the rhs.
        out_ref: Reference to the out.
        partial_out_ref: Reference to the partial output.
        acc_ref: Reference to the accumulator.
        metadata_ref: Reference to the metadata.
        cfgs: GmmConfigs.
    """

    num_k = pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k)
    num_n = pl.cdiv(cfgs.out_size_n, cfgs.tiles.tile_n)

    if rhs_ref.weight.ndim == 4:
        # Select the HBM view before the pipeline loads a tile. An array-level
        # dynamic slice would copy every expert in this layer into a new buffer.
        rhs_ref = dataclasses.replace(
            rhs_ref, weight=rhs_ref.weight.at[rhs_layer_ref[0]],
        )
        if rhs_ref.scale is not None and rhs_ref.scale.dtype == jnp.uint8 and rhs_ref.scale.ndim == 4:
            rhs_ref = dataclasses.replace(rhs_ref, scale=rhs_ref.scale.at[rhs_layer_ref[0]])

    if cfgs.rhs_cfgs.should_unpack:
        rhs_weight = rhs_ref.weight.bitcast(cfgs.rhs_cfgs.quant_dtype)
        rhs_ref = dataclasses.replace(rhs_ref, weight=rhs_weight)

    # Fill metadata buffer and return number of group & m iterations.
    num_gm = fill_metadata(
        lhs_group_sizes_ref,
        group_offset_ref,
        metadata_ref,
        cfgs=cfgs,
    )

    (lhs_spec, rhs_spec), out_spec = generate_block_specs(metadata_ref, cfgs)

    if cfgs.fuse_act is not None:
        rhs_up_ref = jax.tree.map(lambda x: x.at[..., cfgs.out_size_n :], rhs_ref)
        rhs_ref = FusedWeightsRef(gate=rhs_ref, up=rhs_up_ref)

        rhs_spec = FusedWeightsRef(
            gate=rhs_spec,
            up=rhs_spec,
        )

    # Execute the inner kernel.
    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(inner_kernel, cfgs=cfgs),
        grid=(num_n, num_gm, num_k),
        in_specs=(lhs_spec, rhs_spec),
        out_specs=out_spec,
    )

    # Bounded slice requires second last dim to be aligned to the sublane size.
    # rhs_ref uses static tiling thus reshape is not needed.
    lhs_in = lhs_ref.reshape(-1, cfgs.dims.size_lhs_sublane, lhs_ref.shape[-1])
    out_in = out_ref.reshape(-1, cfgs.dims.size_lhs_sublane, out_ref.shape[-1])
    scratches = [partial_out_ref, acc_ref, metadata_ref]
    pipeline_fn(lhs_in, rhs_ref, out_in, scratches=scratches)


def calculate_tiling(
    dims: Dimensions,
    lhs_cfgs: InputConfigs,
    rhs_cfgs: InputConfigs,
    vmem_limit_bytes: int,
    fuse_act: str | None = None,
) -> TileSizes:
    """Calculate optimal tile sizes for GMM kernel."""

    lhs_dtype = lhs_cfgs.dtype
    rhs_dtype = rhs_cfgs.quant_dtype or rhs_cfgs.dtype

    lhs_bits = jax.dtypes.itemsize_bits(lhs_dtype)
    rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)

    # When using bf16 for lhs and rhs, 128 is the largest tile_m value that is
    # safe to use for most scenarios. But if lower bitwidth is used, we need
    # to tweak tile_m to account for using faster hardware unit.
    # TODO(kyuyeunk): Account for different TPU hardware specs.
    bf16_bf16_tile_m = 128
    rhs_mod = min(pl.cdiv(16, rhs_bits), 2)
    tile_m = bf16_bf16_tile_m // rhs_mod
    tile_m = min(tile_m, dims.size_m)

    # To avoid stalling MXU, we add some buffer room where tile_n cannot go
    # smaller than 2x of mxu_column_size.
    tile_n_limit = pltpu.get_tpu_info().mxu_column_size * 2
    tile_n_limit = min(tile_n_limit, dims.size_n)

    size_n_per_rhs = dims.size_n
    fuse_act_factor = 1
    if fuse_act is not None:
        # When computing activation function, rhs is concatenated along dim n.
        fuse_act_factor = 2
        size_n_per_rhs //= fuse_act_factor
        tile_n_limit //= fuse_act_factor

    def _is_tile_k_quant_block_compatible(tk: int) -> bool:
        return (
            tk % rhs_cfgs.quant_block_size == 0 or rhs_cfgs.quant_block_size % tk == 0
        )

    # Initialize tile_k and tile_n to their maximum valid values.
    num_k_tiles = num_n_tiles = 1
    num_lanes = pltpu.get_tpu_info().num_lanes
    tile_k = align_to(dims.size_k, num_lanes)
    tile_n = align_to(size_n_per_rhs, num_lanes)

    def _gmm_vmem_estimate(tm: int, tn: int, tk: int) -> int:
        # 1. LHS tile (double-buffered HBM load)
        lhs_tile_bytes = lhs_bits // 8
        lhs_vmem = 2 * tm * tk * lhs_tile_bytes

        # 2. RHS tile (triple-buffered, includes scale if present)
        # If fuse_act is enabled, we have both gate and up weights,
        # so RHS memory is doubled.
        rhs_weight_vmem = tk * tn * rhs_bits // 8
        rhs_scale_vmem = 0
        if rhs_cfgs.has_scale and rhs_cfgs.quant_block_size is not None:
            num_quant_blocks_per_tile_k = pl.cdiv(tk, rhs_cfgs.quant_block_size)
            rhs_scale_vmem = num_quant_blocks_per_tile_k * tn * 4
        rhs_vmem = fuse_act_factor * (3 * rhs_weight_vmem + 2 * rhs_scale_vmem)

        # 3. Accumulator
        acc_cols = fuse_act_factor * tn
        acc_vmem = tm * acc_cols * 4  # FP32

        # 4. Output tile (double-buffered)
        out_dtype_bytes = jax.dtypes.itemsize_bits(lhs_cfgs.dtype) // 8
        out_vmem = 2 * tm * tn * out_dtype_bytes

        return lhs_vmem + rhs_vmem + acc_vmem + out_vmem

    # Multiple k tiles will introduce accumulation overhead. Thus, we first try
    # to fit the tensors into vmem by only adjusting tile_n.

    # Decrease tile_n until total memory fits in vmem limit.
    while (
        _gmm_vmem_estimate(tile_m, tile_n, tile_k) > vmem_limit_bytes
        and tile_n > tile_n_limit
    ):
        num_n_tiles += 1
        tile_n = align_to(size_n_per_rhs, num_n_tiles * num_lanes) // num_n_tiles

    # If decreasing tile_n is no longer possible, we decrease tile_k instead.
    if tile_n < tile_n_limit:
        num_n_tiles -= 1
        tile_n = align_to(size_n_per_rhs, num_n_tiles * num_lanes) // num_n_tiles

        # Decrease tile_k until total memory fits in vmem limit and tile_k is valid.
        while (
            _gmm_vmem_estimate(tile_m, tile_n, tile_k) > vmem_limit_bytes
            or not _is_tile_k_quant_block_compatible(tile_k)
        ) and tile_k > num_lanes:
            num_k_tiles += 1
            tile_k = align_to(dims.size_k, num_k_tiles * num_lanes) // num_k_tiles

    if tile_n == 0 or tile_k == 0:
        final_estimate = _gmm_vmem_estimate(tile_m, tile_n, tile_k)
        raise ValueError(
            f"Could not find valid tile sizes for {dims=} and"
            f" {final_estimate=} (limit: {vmem_limit_bytes})."
        )

    # TODO(alynie, kyuyeunk): max number of buckets was chosen empirically.
    # Revisit the value to be based on number of instruction memory size.
    max_num_buckets = 4
    bucket_base = tile_m
    for _ in range(1, max_num_buckets):
        new_tile_m = tile_m + bucket_base
        if new_tile_m > dims.size_m:
            break
        if _gmm_vmem_estimate(new_tile_m, tile_n, tile_k) > vmem_limit_bytes:
            break
        tile_m = new_tile_m

    return TileSizes(
        tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, bucket_base=bucket_base
    )


def validate_inputs(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    fuse_act: str | None = None,
    packing_factor: int = 1,
) -> Dimensions:
    """Validates the inputs for the GMM kernel."""

    size_m = lhs.shape[0]
    size_group, size_k, size_n = rhs.shape
    size_k *= packing_factor
    size_lhs_group = group_sizes.shape[0]

    assert size_group <= size_lhs_group
    assert lhs.shape == (size_m, size_k)
    assert rhs.shape == (size_group, size_k // packing_factor, size_n)
    if rhs_scale is not None:
        num_quant_blocks = rhs_scale.shape[1]
        if rhs_scale.dtype == jnp.uint8:  # E8M0 exponents
            assert rhs_scale.shape == (size_group, num_quant_blocks, size_n)
        else:
            assert rhs_scale.shape == (size_group, num_quant_blocks, 1, size_n)
        assert size_k % num_quant_blocks == 0

    assert group_offset.shape == (1,)

    size_lhs_sublane = _get_lhs_sublane_size(lhs.dtype, size_m)
    if fuse_act is not None:
        num_lanes = pltpu.get_tpu_info().num_lanes
        if size_n % (2 * num_lanes) != 0:
            raise ValueError(
                f"{size_n=} should be divisible by 2 * num_lanes when fuse_act is "
                "enabled since we need to split n dimension for gate and up."
            )

    return Dimensions(
        size_m=size_m,
        size_k=size_k,
        size_n=size_n,
        size_group=size_group,
        size_lhs_group=size_lhs_group,
        size_lhs_sublane=size_lhs_sublane,
    )


def get_cost_estimate(cfgs: GmmConfigs):
    """Returns the cost estimate for the GMM kernel."""

    dims = cfgs.dims
    lhs_dtype = cfgs.lhs_cfgs.dtype
    rhs_dtype = cfgs.rhs_cfgs.quant_dtype or cfgs.rhs_cfgs.dtype

    # We use bits for rhs since it could be a sub-byte dtype like int4.
    rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)
    fp32_bytes = jnp.dtype(jnp.float32).itemsize

    # TODO(kyuyeunk): Add compute flops for dequant.
    flops = 2 * dims.size_m * dims.size_k * dims.size_n

    lhs_bytes = dims.size_m * dims.size_k * lhs_dtype.itemsize

    rhs_size = dims.size_group * dims.size_k * dims.size_n
    rhs_bytes = rhs_size * rhs_bits // 8
    if cfgs.rhs_cfgs.has_scale:
        num_quant_blocks = pl.cdiv(dims.size_k, cfgs.rhs_cfgs.quant_block_size)
        rhs_bytes += dims.size_group * num_quant_blocks * dims.size_n * fp32_bytes

    out_bytes = dims.size_m * cfgs.out_size_n * cfgs.out_dtype.itemsize

    total_bytes = lhs_bytes + rhs_bytes + out_bytes

    return pl.CostEstimate(
        flops=flops,
        bytes_accessed=total_bytes,
        transcendentals=0,
    )


def get_scope_name(cfgs: GmmConfigs) -> str:
    dims = cfgs.dims
    tiles = cfgs.tiles
    return (
        f"gmm_v2-g_{dims.size_group}-m_{dims.size_m}-k_{dims.size_k}-act_{cfgs.fuse_act}"
        f"-n_{dims.size_n}-tm_{tiles.tile_m}-tk_{tiles.tile_k}-tn_{tiles.tile_n}"
    )


def make_gmm_configs(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    *,
    tile_info: TileSizes | TileFn,
    vmem_limit_bytes: int | None,
    out_dtype: jnp.dtype | None,
    acc_dtype: jnp.dtype | None,
    fuse_act: str | None = None,
    rhs_quant_dtype: jnp.dtype | None = None,
):
    """Fills the GMM config for the GMM kernel."""

    packing_factor = get_packing_factor(rhs.dtype, rhs_quant_dtype)

    dims = validate_inputs(
        lhs,
        rhs,
        rhs_scale,
        group_sizes,
        group_offset,
        fuse_act,
        packing_factor,
    )

    if rhs_scale is not None:
        has_scale = True
        rhs_quant_dtype = rhs_quant_dtype or rhs.dtype
        num_blocks = rhs_scale.shape[1]
        block_size = dims.size_k // num_blocks
    else:
        has_scale = False
        block_size = dims.size_k

    rhs_cfgs = InputConfigs(
        quant_dtype=rhs_quant_dtype,
        quant_block_size=block_size,
        dtype=rhs.dtype,
        has_scale=has_scale,
        scale_e8m0=rhs_scale is not None and rhs_scale.dtype == jnp.uint8,
    )

    # The lhs is never quantized: the matmul runs in its own dtype.
    lhs_cfgs = InputConfigs(quant_dtype=None, quant_block_size=None, dtype=lhs.dtype)

    if out_dtype is None:
        out_dtype = lhs.dtype

    if acc_dtype is None:
        acc_dtype = jnp.float32.dtype

    if isinstance(tile_info, TileSizes):
        tiles = tile_info
    else:
        tiles = tile_info(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act)

    return GmmConfigs(
        dims=dims,
        tiles=tiles,
        lhs_cfgs=lhs_cfgs,
        rhs_cfgs=rhs_cfgs,
        out_dtype=jnp.dtype(out_dtype),
        acc_dtype=jnp.dtype(acc_dtype),
        fuse_act=fuse_act,
    )


def get_metadata(cfgs: GmmConfigs) -> dict[str, str | int | float]:
    cfgs_dict = dataclasses.asdict(cfgs)
    ret = {}
    for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
        key = jax.tree_util.keystr(path, simple=True, separator=".")
        if not isinstance(val, str | int | float):
            val = str(val)
        ret[key] = val
    return ret


@jax.jit(
    static_argnames=[
        "tile_info",
        "vmem_limit_bytes",
        "precision",
        "preferred_element_type",
        "acc_dtype",
        "fuse_act",
        "rhs_quant_dtype",
    ]
)
def gmm_v2(
    lhs: jax.Array,  # [size_m, size_k]
    rhs: jax.Array,  # [size_group, size_k, size_n]
    group_sizes: jax.Array,  # int32[size_lhs_group]
    rhs_scale: jax.Array | None = None,  # [size_group, num_blocks, 1, out_size], or E8M0 uint8 [size_group, num_blocks, out_size]
    group_offset: jax.Array | None = None,  # int32[1]
    *,
    rhs_layer: jax.Array | None = None,
    tile_info: TileSizes | TileFn = calculate_tiling,
    vmem_limit_bytes: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: jnp.dtype | None = None,
    acc_dtype: jnp.dtype | None = None,
    fuse_act: str | None = None,
    rhs_quant_dtype: jnp.dtype | None = None,
) -> jax.Array:
    """GMM kernel implemented with emit_pipeline.

    Dynamically calculate offset lhs/out tiles to reduce redundant computations.
    Additionally, it adjusts dma size based on number of valid rows and utilize
    triple buffering on weights to better utilize memory.

    Args:
        lhs: lhs with shape [size_m, size_k].
        rhs: Native [size_group, size_k, size_n] or packed
            [size_group, size_k/8, size_n]. A leading layer axis is allowed
            when rhs_layer is supplied; the kernel selects it without copying.
        rhs_layer: Optional int32[1] index into a resident RHS weight family.
            Scales still describe the selected layer.
        group_sizes: The group sizes of lhs rows of shape [size_lhs_group,].
        rhs_scale: The rhs scale of shape [size_group, num_blocks, 1, out_size],
            or E8M0 uint8 [size_group, num_blocks, out_size].
        group_offset: Optional. The group offset of shape [1,].
        tile_info: The tile sizes or tile function to use.
        vmem_limit_bytes: Optional vmem limit in bytes.
        precision: Unused. Exists for compatibility reasons.
        preferred_element_type: Optional jnp.dtype for the output matrix.
        acc_dtype: Optional jnp.dtype for the accumulator.
        fuse_act: Activation function to fuse with GMM (``"situ:..."``), None
            if no fusion. Output rows outside every group are not written.
        rhs_quant_dtype: Quantized dtype encoded by a pre-packed RHS carrier, packed along k.

    Returns:
        Output of shape [size_m, size_n].
    """

    del precision

    # The kernel reshapes lhs by its sublane block, so pad the rows up to it.
    # The extra rows sit past every group, so they add no work, and they are
    # sliced back off the output below.
    size_m = lhs.shape[0]
    size_lhs_sublane = _get_lhs_sublane_size(lhs.dtype, size_m)
    padded_size_m = align_to(size_m, size_lhs_sublane)
    if padded_size_m != size_m:
        lhs = jnp.pad(lhs, ((0, padded_size_m - size_m), (0, 0)))

    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    else:
        if jnp.isscalar(group_offset):
            group_offset = group_offset[None]

    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)

    family = rhs.ndim == 4
    if family != (rhs_layer is not None):
        raise ValueError("A family RHS requires a runtime layer index")
    # E8M0 scales may also keep the family's layer axis ([L, E, blocks, N]).
    scale_family = family and rhs_scale is not None and rhs_scale.dtype == jnp.uint8 and rhs_scale.ndim == 4
    if rhs_layer is None:
        rhs_layer = jnp.zeros((1,), jnp.int32)
    elif rhs_layer.shape != (1,) or rhs_layer.dtype != jnp.int32:
        raise ValueError("rhs_layer must have shape (1,) and dtype int32")
    cfgs = make_gmm_configs(
        lhs,
        jax.ShapeDtypeStruct(rhs.shape[1:], rhs.dtype) if family else rhs,
        jax.ShapeDtypeStruct(rhs_scale.shape[1:], rhs_scale.dtype) if scale_family else rhs_scale,
        group_sizes,
        group_offset,
        tile_info=tile_info,
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=preferred_element_type,
        acc_dtype=acc_dtype,
        fuse_act=fuse_act,
        rhs_quant_dtype=rhs_quant_dtype,
    )
    dims = cfgs.dims
    tiles = cfgs.tiles

    rhs_scale_spec = None
    if rhs_scale is not None:
        if rhs_scale.dtype != jnp.uint8:  # E8M0 exponents stay bytes
            rhs_scale = rhs_scale.astype(jnp.float32)
        rhs_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    max_num_gm = dims.size_group + pl.cdiv(dims.size_m, tiles.tile_m) - 1
    acc_cols = 2 * tiles.tile_n if cfgs.fuse_act is not None else tiles.tile_n
    scratch_shapes = [
        # partial_out_ref
        pltpu.VMEM((dims.size_lhs_sublane, tiles.tile_n), cfgs.out_dtype),
        # acc_ref
        pltpu.VMEM((tiles.tile_m, acc_cols), cfgs.acc_dtype),
        # metadata_ref
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
        ),
    ]

    num_lanes = pltpu.get_tpu_info().num_lanes
    aligned_n = align_to(cfgs.out_size_n, num_lanes)
    out_init = jax.ShapeDtypeStruct((dims.size_m, aligned_n), cfgs.out_dtype)
    rhs_weights = WeightsRef(weight=rhs, scale=rhs_scale)

    return pl.pallas_call(
        functools.partial(kernel_main, cfgs=cfgs),
        out_shape=out_init,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                WeightsRef(
                    weight=pl.BlockSpec(memory_space=pltpu.HBM),
                    scale=rhs_scale_spec,
                ),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        name=get_scope_name(cfgs),
        cost_estimate=get_cost_estimate(cfgs),
        metadata=get_metadata(cfgs),
    )(group_sizes, group_offset, rhs_layer, lhs, rhs_weights)[:size_m, : cfgs.out_size_n]


# ============================================================================
# Prefill building blocks
# ============================================================================
#
# Chunked KDA math, the MLA projections and the logits' residual mixture.


# Structural constants of the K3 per-rank (TP32) layout.
_HEADS = 3  # attention heads per rank (96 total)
_CHANNELS = 128
_KDA_CHUNK = 32
_KDA_SUB_BLOCK = 16


def _dot(x, w):
    """[rows, K] @ [K, N] with an FP32 accumulator; rows must be 8-aligned."""
    if x.shape[0] % 8:
        raise ValueError("prefill dots use native MXU row tiles (multiples of 8)")
    return jax.lax.dot_general(
        x, w, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
    )


def _dot3(x, y, contracting=2):
    """Batched head-major dot: [H, M, K] @ [H, K, N] -> [H, M, N] in FP32."""
    return jax.lax.dot_general(
        x,
        y,
        (((contracting,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )


def _row_where(mask, x, fill=0):
    """``jnp.where(mask[..., None...], x, fill)`` with bool-safe broadcasting."""
    m = mask.astype(jnp.int32)  # numeric while reshaping/broadcasting
    while len(m.shape) < len(x.shape):
        m = m[..., None]
    m = jnp.broadcast_to(m, x.shape) != 0
    return jnp.where(m, x, fill)


def _mask_where(mask, x, fill):
    """jnp.where with a broadcasting bool mask, lowered through int32 so the
    Mosaic i1-reshape limitation is not hit."""
    mask = jnp.broadcast_to(mask.astype(jnp.int32), x.shape) != 0
    return jnp.where(mask, x, fill)


# ---------------------------------------------------------------------------
# Chunked KDA math (pure functions; also called on ref reads in the kernel).
# ---------------------------------------------------------------------------


def _inclusive_cumsum_rows(x):
    """Inclusive FP32 cumsum over axis -2 via exact log-shift doubling."""
    shift = 1
    rows = x.shape[-2]
    while shift < rows:
        x = x + jnp.pad(x, ((0, 0), (shift, 0), (0, 0)))[..., :rows, :]
        shift *= 2
    return x


# ---------------------------------------------------------------------------
# MLA projections
# ---------------------------------------------------------------------------


def mla_projections(hidden, weights, *, epsilon=1e-5, return_latent=False, query_scale=1.0):
    """MLA Q/K/V projections for a token window (per-rank layout).

    ``hidden``: [T, hidden] BF16. Returns (query [T,h,256] times
    ``query_scale`` before its BF16 rounding, new_key [T,h,256],
    new_value [T,h,128], output_gate [T,128 h]) for the rank's h heads,
    all BF16; with
    ``return_latent`` also the compressed KV that vLLM's MLA cache stores,
    ``[T, 576]`` BF16: the normalized 512-lane latent then the 64 rope lanes
    (K3's MLA is NoPE; the rope lanes are used unrotated). It is the same on
    every rank.
    """
    query_latent, latent = mla_latents(hidden, weights, epsilon=epsilon)
    query, new_key, new_value = mla_heads(query_latent, latent, weights, query_scale=query_scale)
    output_gate = _dot(hidden, weights.output_gate_projection).astype(jnp.bfloat16)
    if return_latent:
        return query, new_key, new_value, output_gate, latent
    return query, new_key, new_value, output_gate


def mla_latents(hidden, weights, *, epsilon=1e-5):
    """The row-local half of ``mla_projections``: the normalized query
    latent ``[T, 1536]`` and the compressed KV ``[T, 576]``, both BF16."""
    query_latent = _dot(hidden, weights.query_down_projection)
    query_latent = (
        query_latent
        * jax.lax.rsqrt(
            jnp.mean(query_latent * query_latent, axis=1, keepdims=True) + epsilon
        )
        * weights.query_normalization
    ).astype(jnp.bfloat16)
    compressed = _dot(hidden, weights.key_value_down_projection).astype(
        jnp.bfloat16
    )
    kv_latent = compressed[:, :512].astype(jnp.float32)
    kv_latent = (
        kv_latent
        * jax.lax.rsqrt(
            jnp.mean(kv_latent * kv_latent, axis=1, keepdims=True) + epsilon
        )
        * weights.key_value_normalization
    ).astype(jnp.bfloat16)
    return query_latent, jnp.concatenate((kv_latent, compressed[:, 512:576]), axis=1)


def mla_heads(query_latent, latent, weights, *, query_scale=1.0):
    """The per-head half of ``mla_projections``: ``(query, new_key,
    new_value)`` of the weights' heads from ``mla_latents``' outputs."""
    tokens = query_latent.shape[0]
    heads = weights.query_up_projection.shape[-1] // 256
    query = (_dot(query_latent, weights.query_up_projection) * query_scale).astype(jnp.bfloat16)
    query = query.reshape(tokens, heads, 256)
    expanded = _dot(latent[:, :512], weights.key_value_up_projection).astype(jnp.bfloat16)
    expanded = expanded.reshape(tokens, heads, 256)
    new_key = jnp.concatenate(
        (
            expanded[:, :, :128],
            jnp.broadcast_to(latent[:, None, 512:576], (tokens, heads, 64)),
            jnp.zeros((tokens, heads, 64), jnp.bfloat16),
        ),
        axis=2,
    )
    return query, new_key, expanded[:, :, 128:]


# ---------------------------------------------------------------------------
# Residual mixture (dense form, used by the logits; the kernel streams it)
# ---------------------------------------------------------------------------


def residual_mixture(blocks, prefix, folded, active_block_count, *, epsilon=1e-5):
    """Learned residual mixture over [8, T, H] blocks + [T, H] prefix.

    Returns [T, H] BF16.
    """
    values = jnp.concatenate(
        (blocks.astype(jnp.float32), prefix[None].astype(jnp.float32)), axis=0
    )
    scores = jnp.sum(values * folded.astype(jnp.float32), axis=-1)
    scores *= jax.lax.rsqrt(jnp.mean(values * values, axis=-1) + epsilon)
    index = jax.lax.broadcasted_iota(jnp.int32, (9, 1), 0)
    active = (index < active_block_count) | (index == 8)
    probabilities = jax.nn.softmax(jnp.where(active, scores, -jnp.inf), axis=0)
    mixed = jnp.sum(values * probabilities[..., None], axis=0)
    return mixed.astype(jnp.bfloat16)


# ============================================================================
# Residual mixture and RMSNorm (HBM-tiled)
# ============================================================================
#
# An online softmax: one read of each active block per tile, one block in VMEM.


def mixture_norm_tiled_body(
    active_ref,  # [1] int32 scalar prefetch; active block count in [0, 8]
    prefix_ref,  # [T, H] BF16 HBM
    blocks_ref,  # [8, T, H] BF16 HBM
    folded_ref,  # [1, H] FP32 HBM
    norm_ref,  # [1, H] BF16 HBM
    output_ref,  # [T, H] BF16 HBM
    prefix_bufs,  # [2, tile, H] BF16
    block_buf,  # [tile, H] BF16
    output_bufs,  # [2, tile, H] BF16
    numerator_buf,  # [tile, H] FP32
    row_max_buf,  # [tile, 1] FP32
    row_sum_buf,  # [tile, 1] FP32
    folded_buf,  # [1, H] FP32
    norm_buf,  # [1, H] BF16
    prefix_sems,
    block_sem,
    output_sems,
    weight_sems,
    *,
    tile,
    eps,
):
    """Run residual mixture plus RMSNorm with caller-owned refs and scratch."""
    tokens, hidden = prefix_ref.shape
    n_tiles = tokens // tile
    active_count = active_ref[0]

    weights = (
        pltpu.make_async_copy(folded_ref, folded_buf, weight_sems.at[0]),
        pltpu.make_async_copy(norm_ref, norm_buf, weight_sems.at[1]),
    )
    for copy in weights:
        copy.start()
    for copy in weights:
        copy.wait()
    folded = folded_buf[...]
    norm = norm_buf[...]

    def load_prefix(t, slot):
        return pltpu.make_async_copy(
            prefix_ref.at[pl.ds(t * tile, tile), :],
            prefix_bufs.at[slot],
            prefix_sems.at[slot],
        )

    def store_output(t, slot):
        return pltpu.make_async_copy(
            output_bufs.at[slot],
            output_ref.at[pl.ds(t * tile, tile), :],
            output_sems.at[slot],
        )

    load_prefix(0, 0).start()

    def tile_body(t):
        slot = t % 2
        load_prefix(t, slot).wait()

        @pl.when(t >= 2)
        def wait_previous_output():
            store_output(t - 2, slot).wait()

        @pl.when(t + 1 < n_tiles)
        def prefetch_next_prefix():
            load_prefix(t + 1, 1 - slot).start()

        prefix = prefix_bufs.at[slot][...].astype(jnp.float32)
        prefix_score = jnp.sum(prefix * folded, axis=1)
        prefix_score *= jax.lax.rsqrt(jnp.mean(prefix * prefix, axis=1) + eps)
        numerator_buf[...] = prefix
        row_max_buf[...] = prefix_score[:, None]
        row_sum_buf[...] = jnp.ones((tile, 1), jnp.float32)

        for block in range(8):
            @pl.when(block < active_count)
            def include_block():
                copy = pltpu.make_async_copy(
                    blocks_ref.at[block, pl.ds(t * tile, tile), :],
                    block_buf,
                    block_sem.at[0],
                )
                copy.start()
                copy.wait()
                values = block_buf[...].astype(jnp.float32)
                score = jnp.sum(values * folded, axis=1)
                score *= jax.lax.rsqrt(jnp.mean(values * values, axis=1) + eps)
                old_max = row_max_buf[...]
                new_max = jnp.maximum(old_max, score[:, None])
                old_scale = jnp.exp(old_max - new_max)
                new_scale = jnp.exp(score[:, None] - new_max)
                numerator_buf[...] = (
                    numerator_buf[...] * old_scale + values * new_scale
                )
                row_sum_buf[...] = row_sum_buf[...] * old_scale + new_scale
                row_max_buf[...] = new_max

        mixed = (numerator_buf[...] / row_sum_buf[...]).astype(jnp.bfloat16)
        mixed = mixed.astype(jnp.float32)
        normalized = (
            mixed
            * jax.lax.rsqrt(jnp.mean(mixed * mixed, axis=1, keepdims=True) + eps)
            * norm
        ).astype(jnp.bfloat16)
        output_bufs.at[slot][...] = normalized
        store_output(t, slot).start()

    pl.loop(0, n_tiles)(tile_body)
    store_output(n_tiles - 1, (n_tiles - 1) % 2).wait()
    if n_tiles > 1:
        store_output(n_tiles - 2, (n_tiles - 2) % 2).wait()


def mixture_norm_tiled_scratch_shapes(hidden_size, tile):
    """Scratch specs in the order expected by ``mixture_norm_tiled_body``."""
    return (
        pltpu.VMEM((2, tile, hidden_size), jnp.bfloat16),
        pltpu.VMEM((tile, hidden_size), jnp.bfloat16),
        pltpu.VMEM((2, tile, hidden_size), jnp.bfloat16),
        pltpu.VMEM((tile, hidden_size), jnp.float32),
        pltpu.VMEM((tile, 1), jnp.float32),
        pltpu.VMEM((tile, 1), jnp.float32),
        pltpu.VMEM((1, hidden_size), jnp.float32),
        pltpu.VMEM((1, hidden_size), jnp.bfloat16),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((1,)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
    )


def make_mixture_norm_tiled(*, seq_len, tile=256, eps=1e-5, interpret=False):
    """Build ``(prefix, blocks, folded, norm, active) -> normalized``.

    ``active`` is an int32 array of shape [1] with a value from 0 to 8.
    The sequence length and tile must be positive, and seq_len must be a
    multiple of tile. Inputs and output are HBM-backed in the Pallas body.
    """
    if seq_len <= 0 or tile <= 0 or seq_len % tile:
        raise ValueError("seq_len must be a positive multiple of tile")

    def run(prefix, blocks, folded, norm, active):
        if len(prefix.shape) != 2 or prefix.shape[0] != seq_len:
            raise ValueError("prefix must have shape [seq_len, hidden]")
        hidden = prefix.shape[1]
        if hidden % 128 or prefix.dtype != jnp.bfloat16:
            raise ValueError("prefix must be BF16 with a 128-aligned width")
        if blocks.shape != (8, seq_len, hidden) or blocks.dtype != jnp.bfloat16:
            raise ValueError("blocks must be BF16 with shape [8, seq_len, hidden]")
        if folded.shape != (1, hidden) or folded.dtype != jnp.float32:
            raise ValueError("folded must be FP32 with shape [1, hidden]")
        if norm.shape != (1, hidden) or norm.dtype != jnp.bfloat16:
            raise ValueError("norm must be BF16 with shape [1, hidden]")
        if active.shape != (1,) or active.dtype != jnp.int32:
            raise ValueError("active must be int32 with shape [1]")
        call = pl.pallas_call(
            functools.partial(mixture_norm_tiled_body, tile=tile, eps=eps),
            out_shape=jax.ShapeDtypeStruct(prefix.shape, jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=[pl.BlockSpec(memory_space=pltpu.HBM) for _ in range(4)],
                out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
                scratch_shapes=mixture_norm_tiled_scratch_shapes(hidden, tile),
            ),
            name="mixture_norm_tiled",
            interpret=interpret,
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 1024**2),
        )
        return call(active, prefix, blocks, folded, norm)

    return run


def make_mixture_norm_sp(mesh, *, seq_len, eps=1e-5):
    """Apply the tiled mixture/RMSNorm to each rank's T/n local rows."""
    n = check_mesh(mesh, "mixture SP")
    if seq_len % n:
        raise ValueError("mixture SP requires T divisible by the ranks")
    axis_name = mesh.axis_names[0]
    local_tokens = seq_len // n
    tile = min(256, local_tokens)
    if local_tokens % tile:
        raise ValueError("local token count must be divisible by the mixture tile")
    op = make_mixture_norm_tiled(seq_len=local_tokens, tile=tile, eps=eps)

    def local(prefix, blocks, folded, norm, active):
        return op(prefix[0], blocks[0], folded[0], norm[0], active)[None]

    return jax.jit(jax.shard_map(
        local, mesh=mesh,
        in_specs=(P(axis_name), P(axis_name), P(axis_name), P(axis_name), P()),
        out_specs=P(axis_name), check_vma=False,
    ))


# ============================================================================
# KDA (sequence-tiled)
# ============================================================================
#
# The window streams through VMEM in tiles; the KDA state carries across them.


def _invert_delta_rule_forward(low, block=8):
    """(I + L)^{-1} for ``low`` [H, C, C] FP32 strictly lower triangular.

    Splits L = L_diag + F into the block-diagonal part (sub-blocks of ``block``
    rows) and the off-block-diagonal remainder, so (I + L)^{-1} =
    (I + G)^{-1} (I + L_diag)^{-1} with G = (I + L_diag)^{-1} F. The diagonal
    blocks are inverted by forward substitution on the VPU: they sit side by
    side along lanes, so each elimination step is one FP32 multiply-subtract
    over all of them, instead of a chain of small MXU products each waiting
    on the previous result. (I + G)^{-1} uses exact nilpotent Neumann
    doubling (G^(C/block) = 0), whose largest power stays small enough to be
    well conditioned under reduced-precision MXU matmul. Mirrors the
    vllm-torchtpu chunk_kda solve.
    """
    heads, rows, columns = low.shape
    assert rows == columns and rows % block == 0
    shift = block.bit_length() - 1
    diagonal = jnp.concatenate([low[:, first:first + block, first:first + block] for first in range(0, rows, block)], axis=2)
    solved = jnp.broadcast_to(jnp.tile(jnp.eye(block, dtype=jnp.float32), (1, rows // block))[None], diagonal.shape)
    row = jnp.arange(block)[None, :, None]
    for pivot in range(block - 1):
        # Explicit lane indices: each lane reads its own block's pivot column.
        indices = jnp.broadcast_to(((jnp.arange(rows, dtype=jnp.int32) >> shift) * block + pivot)[None, None, :], diagonal.shape)
        factors = jnp.take_along_axis(diagonal, indices, axis=2, mode='promise_in_bounds')
        update = factors * solved[:, pivot:pivot + 1, :]
        solved = solved - jnp.where(row > pivot, update, 0)
    p = jnp.concatenate([jnp.pad(solved[:, :, first:first + block], ((0, 0), (0, 0), (first, columns - first - block))) for first in range(0, rows, block)], axis=1)
    if block == rows:
        return p
    idx = jnp.arange(rows)
    same_block = idx[:, None] >> shift == idx[None, :] >> shift
    f = low - low * same_block[None].astype(jnp.float32)
    g = _dot3(p, f)
    h = -g
    inv_ig = jnp.eye(rows, dtype=jnp.float32)[None] + h
    hk = h
    for _ in range((rows // block).bit_length() - 2):
        hk = _dot3(hk, hk)
        inv_ig = inv_ig + _dot3(inv_ig, hk)
    return _dot3(inv_ig, p)


def _kda_chunk_prepare(q, k, v, gate, beta, *, sub_block=_KDA_SUB_BLOCK):
    """The state-independent part of the Kimi delta rule in chunked form for
    a batch of chunks (heads and chunks folded into the leading axis): the
    triangular solve, its right-hand sides, the decay factors and the QK
    products, which need no incoming recurrent state. Returns ``(u, w, aqk,
    read_q, read_decay, kg, last_decay)`` for `_kda_chunk_apply`.

    ``q``, ``k``: [B, C, K] FP32, with ``q`` already L2-normalized and scaled
    by K**-0.5 and ``k`` L2-normalized. ``v``: [B, C, V] FP32. ``gate``:
    [B, C, K] FP32 log-decay per channel (<= 0, natural log). ``beta``:
    [B, C] FP32 in [0, 1].

    The intra-chunk decay factors use the sub-block anchor factorization
    exp(g_r - g_t) = exp(g_r - a_i) * exp(a_i - g_t), ``a_i`` the gate
    cumsum at the middle row of r's sub-block, so each sub-block's rows
    (its diagonal block included) are one matmul. For the bounded K3 gate
    (|gate| <= 5) the row factors stay within exp(+-40); a column factor
    underflows only where the true product is below exp(-47), and products
    above the diagonal (possibly inf) are masked out afterwards. The matmul
    runs at HIGHEST precision: the factors' spread would otherwise cost the
    diagonal blocks their FP32 accuracy.
    """
    head_count, chunk_size, channels = q.shape
    value_channels = v.shape[-1]
    if chunk_size % sub_block:
        raise ValueError("chunk size must be a multiple of the sub-block")
    nc = chunk_size // sub_block
    gc = _inclusive_cumsum_rows(gate)
    beta_k = beta[..., None].astype(jnp.float32)
    rows = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 0)
    cols = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 1)
    causal = (cols <= rows)[None]
    strictly_causal = (cols < rows)[None]
    aqk_parts = []
    akk_parts = []
    col_iota = jax.lax.broadcasted_iota(jnp.int32, (chunk_size,), 0)
    for i in range(nc):
        i_s = i * sub_block
        anchor = gc[:, i_s + sub_block // 2, :][:, None, :]
        row_scale = jnp.exp(gc[:, i_s:i_s + sub_block, :] - anchor)
        through = (col_iota < i_s + sub_block).astype(jnp.float32)[None, :, None]
        k_scaled = k * jnp.exp(jnp.minimum(anchor - gc, 80.0)) * through
        both = jax.lax.dot_general(
            jnp.concatenate((q[:, i_s:i_s + sub_block] * row_scale, k[:, i_s:i_s + sub_block] * row_scale), axis=1),
            k_scaled.transpose(0, 2, 1), (((2,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32, precision=jax.lax.Precision.HIGHEST)
        aqk_parts.append(both[:, :sub_block])
        akk_parts.append(both[:, sub_block:])
    aqk = jnp.concatenate(aqk_parts, axis=1)
    akk = jnp.concatenate(akk_parts, axis=1)
    aqk = _mask_where(causal, aqk, jnp.float32(0))
    # L is strictly causal, and beta scales rows: delta_r = beta_r * (v_r -
    # k_r @ state_{r-1}), so L[r, t] = beta_r * (k_r . k_t) * exp(gc_r-gc_t).
    low = _mask_where(strictly_causal, akk * beta[:, :, None], jnp.float32(0))
    solve = _invert_delta_rule_forward(low)
    rhs = jnp.concatenate((v * beta_k, k * jnp.exp(gc) * beta_k), axis=-1)
    solved = _dot3(solve, rhs)
    u, w = (solved[..., :value_channels], solved[..., value_channels:])
    anchor0 = gc[:, 0, :]
    read_q = q * jnp.exp(gc - anchor0[:, None, :])
    read_decay = jnp.exp(anchor0)
    g_last = gc[:, -1, :]
    kg = k * jnp.exp(g_last[:, None, :] - gc)
    last_decay = jnp.exp(g_last)
    return (u, w, aqk, read_q, read_decay, kg, last_decay)


def _kda_chunk_apply(prepared, state):
    """The state-dependent rest of the chunked delta rule on one chunk's
    `_kda_chunk_prepare` results and ``state`` [H, K, V] FP32: ``(out
    [H, C, V], new_state [H, K, V])``."""
    u, w, aqk, read_q, read_decay, kg, last_decay = prepared
    v_new = u - _dot3(w, state)
    out = _dot3(read_q, state * read_decay[:, :, None])
    out = out + _dot3(aqk, v_new)
    new_state = state * last_decay[:, :, None] + _dot3(kg, v_new, contracting=1)
    return (out, new_state)


def kda_prefill_tiled_body(
    hidden_ref,  # [T, hidden + 128] BF16 HBM: the rows, then their gate compression
    fb_w_ref,  # [128, 384] BF16
    proj_w_ref,  # [hidden, 1664] BF16
    conv_w_ref,  # [4, 3, 3, 128]
    alog_ref,  # [3, 1] FP32
    dt_ref,  # [3, 128] FP32
    norm_ref,  # [8, 128] BF16
    rec_state_in_ref,  # [3, 128, 128] FP32
    conv_hist_in_ref,  # [3, 3, 3, 128] BF16
    out_ref,  # [T, 384] BF16 HBM output (attended)
    rec_state_out_ref,  # [3, 128, 128] FP32
    conv_hist_out_ref,  # [3, 3, 3, 128] BF16
    # scratch
    hidden_bufs,  # (2, tile, hidden + 128) BF16 double buffer
    out_bufs,  # (2, tile, 384) BF16 double buffer
    fb_w_vmem,
    proj_w_vmem,
    conv_w_vmem,
    alog_vmem,
    dt_vmem,
    norm_vmem,
    rec_state_vmem,  # (3, 128, 128) FP32
    conv_hist_vmem,  # (3, 3, 3, 128) BF16
    hidden_sems,
    out_sems,
    weight_sems,
    *,
    tile,
    decay_lower_bound,
    epsilon,
    checkpoint,
    segments,
    tiles,
    block_tiles,
    projection,
):
    """Run one KDA layer over a packed window, reading and writing tiled
    activations in HBM.

    ``segments``: ``(ref, count)``; ``ref`` the SMEM int32 ``[3 * count]``
    offsets (ascending multiples of ``tile``), lengths (0: an unused slot,
    whose output is its input) and checkpoint positions (from the sequence
    start). State refs lead with the sequence, each loaded at its first tile
    and stored after its last; rows outside every sequence change nothing.
    ``checkpoint``: ``(rec_out_ref, conv_out_ref)``; also store each
    sequence's state after its checkpoint position (a multiple of ``tile``;
    0: the incoming state).
    ``tiles``: ``(first_tile, end_tile, first, last[, load])``; run only
    those tiles, loading the weights when ``first`` (or ``load``, if given)
    and finishing when ``last``.
    ``block_tiles``: hidden rows arrive in blocks of this many tiles; no load
    crosses a block and a block's stores complete at its last tile.
    ``projection``: ``(weight_ref, weight_vmem, partial_ref, partial_bufs,
    partial_sems)``; also write each tile's BF16 output projection to
    ``partial_ref`` ``[T, hidden]``, the reduce-scatter's partial sums
    (``None``: the attended rows only). Scratch refs follow
    ``kda_prefill_tiled_scratch_shapes``.
    """
    head_count, channels = _HEADS, _CHANNELS
    width = head_count * channels  # 384
    segment_ref, count = segments

    def field(index, s):  # descriptor field ``index`` of sequence ``s``
        return segment_ref[index * count + s]

    # ---- load the layer weights into VMEM once ----
    weight_copies = (
        pltpu.make_async_copy(fb_w_ref, fb_w_vmem, weight_sems.at[1]),
        pltpu.make_async_copy(proj_w_ref, proj_w_vmem, weight_sems.at[2]),
        pltpu.make_async_copy(conv_w_ref, conv_w_vmem, weight_sems.at[3]),
        pltpu.make_async_copy(alog_ref, alog_vmem, weight_sems.at[4]),
        pltpu.make_async_copy(dt_ref, dt_vmem, weight_sems.at[5]),
        pltpu.make_async_copy(norm_ref, norm_vmem, weight_sems.at[6]),
    )
    if projection is not None:
        weight_ref, weight_vmem, partial_ref, partial_bufs, partial_sems = projection
        weight_copies += (pltpu.make_async_copy(weight_ref, weight_vmem, weight_sems.at[13]),)
    first_tile, end_tile, first, last = tiles[:4]

    @pl.when(tiles[4] if len(tiles) > 4 else first)
    def load_weights():
        scope = jax.named_scope("kda_weights") if _config().phase_scopes else contextlib.nullcontext()
        with scope:
            for c in weight_copies:
                c.start()
            for c in weight_copies:
                c.wait()

    def copy_now(pairs, first_semaphore):
        copies = [pltpu.make_async_copy(source, destination, weight_sems.at[first_semaphore + i])
                  for i, (source, destination) in enumerate(pairs)]
        for copy in copies:
            copy.start()
        for copy in copies:
            copy.wait()

    rec_checkpoint_ref, hist_checkpoint_ref = checkpoint

    def write_checkpoint(segment):
        copy_now(((rec_state_vmem, rec_checkpoint_ref.at[segment]),
                  (conv_hist_vmem, hist_checkpoint_ref.at[segment])), 11)

    # An unused slot: its outputs are its inputs. The copies start with the
    # first tiles and are waited for after the last: one blocking round trip
    # per slot would hold up the tiles (~0.15 ms a layer at 16 slots).
    def unused_copies(s):
        pairs = [(rec_state_in_ref.at[s], rec_state_out_ref.at[s]),
                 (conv_hist_in_ref.at[s], conv_hist_out_ref.at[s]),
                 (rec_state_in_ref.at[s], rec_checkpoint_ref.at[s]),
                 (conv_hist_in_ref.at[s], hist_checkpoint_ref.at[s])]
        return [pltpu.make_async_copy(source, target, weight_sems.at[14 + i % 2])
                for i, (source, target) in enumerate(pairs)]

    def each_unused(action):
        @pl.loop(0, count)
        def visit(s):
            @pl.when(field(1, s) == 0)
            def unused():
                for copy in unused_copies(s):
                    action(copy)

    pl.when(first)(lambda: each_unused(lambda copy: copy.start()))

    def load_tile(t, buf):
        return pltpu.make_async_copy(
            hidden_ref.at[pl.ds(t * tile, tile), :],
            hidden_bufs.at[buf],
            hidden_sems.at[buf],
        )

    def store_tile(t, buf):
        copies = [pltpu.make_async_copy(
            out_bufs.at[buf], out_ref.at[pl.ds(t * tile, tile), :], out_sems.at[buf])]
        if projection is not None:
            copies.append(pltpu.make_async_copy(
                partial_bufs.at[buf], partial_ref.at[pl.ds(t * tile, tile), :],
                partial_sems.at[buf]))
        return _Copies(copies)

    def tile_body(t):
        buf = t % 2
        in_block = t % block_tiles

        @pl.when(in_block == 0)
        def load_first_tile():  # the block's later tiles are prefetched
            load_tile(t, buf).start()

        load_tile(t, buf).wait()
        # The tile's sequence (-1: padding only; offsets are page-aligned, so
        # no tile holds two) and where its rows end. The state copies index
        # that one sequence: unrolled per slot, they would not fit the fused
        # layer's instruction memory at 16 slots.
        first_row = t * tile
        segment, segment_end = -1, 0
        for s in range(count):
            start = field(0, s)
            end = start + field(1, s)
            inside = (first_row >= start) & (first_row < end)
            segment = jnp.where(inside, s, segment)
            segment_end = jnp.where(inside, end, segment_end)
        current = jnp.maximum(segment, 0)
        offset, position = field(0, current), field(2, current)

        @pl.when((segment >= 0) & (first_row == offset))
        def load_segment_state():
            copy_now(((rec_state_in_ref.at[current], rec_state_vmem),
                      (conv_hist_in_ref.at[current], conv_hist_vmem)), 7)
            pl.when(position == 0)(lambda: write_checkpoint(current))
        row_end = segment_end

        @pl.when(in_block >= 2)
        def wait_previous_store():
            store_tile(t - 2, buf).wait()

        @pl.when(in_block + 1 < block_tiles)
        def prefetch_next_tile():
            load_tile(t + 1, 1 - buf).start()

        # The weights are read here: a value live across the loop would be a
        # VMEM copy (22.75 MiB for the projection).
        conv_filter = conv_w_vmem[...].astype(jnp.float32)
        a_log = jnp.exp(alog_vmem[...])  # [3, 1] FP32
        dt_bias = dt_vmem[...]  # [3, 128] FP32
        norm = norm_vmem[0]
        # ---- projections ----
        # The gate compression arrives with the rows (see _kda_layer_body).
        hidden_width = hidden_bufs.shape[-1] - 128
        hidden_tile = hidden_bufs.at[buf][..., :hidden_width]  # [tile, hidden]
        gate_features = hidden_bufs.at[buf][..., hidden_width:]  # [tile, 128]
        raw_gate = _dot(gate_features, fb_w_vmem[...]).astype(jnp.bfloat16).reshape(
            tile, head_count, channels
        )  # [tile, 3, 128]
        projected = _dot(hidden_tile, proj_w_vmem[...]).astype(jnp.bfloat16)  # [tile, 1664]
        qkv = projected[:, : 3 * width].reshape(tile, 3, head_count, channels)
        output_gate = projected[:, 3 * width : 4 * width].reshape(
            tile, head_count, channels
        )
        update_rate = projected[:, 4 * width : 4 * width + head_count]  # [tile, 3]
        real_rows = jnp.arange(tile, dtype=jnp.int32) + t * tile < row_end
        raw_gate = _row_where(real_rows, raw_gate, -1e30)
        update_rate = _row_where(real_rows, update_rate, -1e30)

        # ---- 4-tap causal conv (carrying the 3-token history) ----
        window = jnp.concatenate((conv_hist_vmem[...], qkv), axis=0)  # [tile+3, 3,3,128]
        convolved = jnp.zeros((tile, 3, head_count, channels), jnp.float32)
        for tap in range(4):
            convolved = (
                convolved
                + window[tap : tap + tile].astype(jnp.float32) * conv_filter[tap]
            )
        filtered = jax.nn.silu(convolved).astype(jnp.bfloat16)
        filtered = _row_where(real_rows, filtered)
        # The history starts at the end of the real prefix. Mosaic does not
        # lower dynamic_slice on a value, so only the one partial tile uses a
        # masked reduction; full tiles use a static slice.
        real_count = jnp.clip(row_end - t * tile, 0, tile)

        @pl.when(real_count == tile)
        def carry_full_tile():
            conv_hist_vmem[...] = window[tile : tile + 3]

        @pl.when((real_count > 0) & (real_count < tile))
        def carry_partial_tile():
            rows = jnp.arange(tile + 3, dtype=jnp.int32)
            picked = []
            for i in range(3):
                picked.append(
                    jnp.sum(_row_where(rows == real_count + i, window), axis=0)
                )
            conv_hist_vmem[...] = jnp.stack(picked)

        query = filtered[:, 0].astype(jnp.float32)
        key = filtered[:, 1].astype(jnp.float32)
        value = filtered[:, 2].astype(jnp.float32)
        query, key = kda_normalize_query_key(query, key, channels)
        decay = kda_decay(decay_lower_bound, a_log[None], raw_gate, dt_bias[None])  # [tile, 3, 128] FP32
        beta = jax.nn.sigmoid(update_rate.astype(jnp.float32))  # [tile, 3] FP32

        # ---- chunked delta rule over this tile's 32-token chunks ----
        # The work that needs no incoming state runs for every chunk of the
        # tile as one batch, so it does not sit between the chunks' serial
        # state updates.
        chunk_count = tile // _KDA_CHUNK

        def batch(x):  # [tile, H, ...] -> [chunks * H, _KDA_CHUNK, ...]
            if x.ndim == 3:
                return x.reshape(chunk_count, _KDA_CHUNK, head_count, channels).transpose(0, 2, 1, 3).reshape(chunk_count * head_count, _KDA_CHUNK, channels)
            return x.reshape(chunk_count, _KDA_CHUNK, head_count).transpose(0, 2, 1).reshape(chunk_count * head_count, _KDA_CHUNK)

        prepared = _kda_chunk_prepare(batch(query), batch(key), batch(value), batch(decay), batch(beta))
        outputs = []
        for c in range(chunk_count):
            selected = tuple(x[c * head_count:(c + 1) * head_count] for x in prepared)
            chunk_out, new_state = _kda_chunk_apply(selected, rec_state_vmem[...])
            rec_state_vmem[...] = new_state
            outputs.append(chunk_out.transpose(1, 0, 2))

        attended = jnp.concatenate(outputs, axis=0)  # [tile, 3, 128] FP32
        attended = attended.astype(jnp.bfloat16).astype(jnp.float32)
        attended *= jax.lax.rsqrt(
            jnp.mean(attended * attended, axis=-1, keepdims=True) + epsilon
        )
        attended = (attended * norm).astype(jnp.bfloat16) * jax.nn.sigmoid(
            output_gate.astype(jnp.float32)
        ).astype(jnp.bfloat16)
        attended = attended.reshape(tile, width)
        out_bufs.at[buf][...] = attended
        if projection is not None:
            columns = weight_vmem.shape[1] // 4
            for c in range(4):  # in column quarters: bounded temporaries
                partial_bufs.at[buf, :, pl.ds(c * columns, columns)][...] = _dot(
                    attended, weight_vmem[:, pl.ds(c * columns, columns)]
                ).astype(jnp.bfloat16)

        pl.when((segment >= 0) & (position > 0)
                & ((t + 1) * tile == offset + position))(
            lambda: write_checkpoint(current))

        @pl.when((segment >= 0) & (t == (segment_end - 1) // tile))
        def store_segment_state():
            copy_now(((rec_state_vmem, rec_state_out_ref.at[current]),
                      (conv_hist_vmem, conv_hist_out_ref.at[current])), 9)

        # Write this tile back (overlaps with the next tile's compute).
        store_tile(t, buf).start()

        @pl.when(in_block == block_tiles - 1)
        def drain():  # the block's last two stores
            store_tile(t, buf).wait()
            if block_tiles > 1:
                store_tile(t - 1, 1 - buf).wait()

    pl.loop(first_tile, end_tile)(tile_body)
    # Each sequence's state was stored after its last tile; wait for the
    # unused slots' copies.
    pl.when(last)(lambda: each_unused(lambda copy: copy.wait()))


class _Copies:
    """Several async copies started and awaited together."""

    def __init__(self, copies):
        self.copies = copies

    def start(self):
        for copy in self.copies:
            copy.start()

    def wait(self):
        for copy in self.copies:
            copy.wait()


def kda_prefill_tiled_scratch_shapes(
    hidden_size, tile, fb_w, proj_w, conv_w, alog, dt, norm
):
    """Scratch specs, in the order expected by ``kda_prefill_tiled_body``."""
    width = _HEADS * _CHANNELS
    return (
        pltpu.VMEM((2, tile, hidden_size), jnp.bfloat16),
        pltpu.VMEM((2, tile, width), jnp.bfloat16),
        pltpu.VMEM(fb_w.shape, fb_w.dtype),
        pltpu.VMEM(proj_w.shape, proj_w.dtype),
        pltpu.VMEM(conv_w.shape, conv_w.dtype),
        pltpu.VMEM(alog.shape, alog.dtype),
        pltpu.VMEM(dt.shape, dt.dtype),
        pltpu.VMEM(norm.shape, norm.dtype),
        pltpu.VMEM((_HEADS, _CHANNELS, _CHANNELS), jnp.float32),
        pltpu.VMEM((3, 3, _HEADS, _CHANNELS), jnp.bfloat16),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((16,)),
    )


# ============================================================================
# MLA flash attention (long windows)
# ============================================================================
#
# Each rank's 3 heads over the packed window and every rank's cached prefixes.


TILE = 128
SCALE = 192.0**-0.5
# The online softmax runs in base 2 on queries pre-scaled by SCALE * log2(e)
# (``mla_projections(query_scale=SCALE_LOG2)``, before their BF16 rounding),
# so scores leave the MXU ready for exp2.
SCALE_LOG2 = SCALE * 1.4426950408889634


def _flash_kernel_packed(length_ref, segment_ref, *refs, tile, q_tile, count, prefixed=False, selected=False,
                         capacity=0):
    """Causal MLA flash attention over a packed window of ``count``
    sequences, per (head, ``q_tile``-row query tile), with ``tile``-row
    key/value blocks double-buffered from HBM.

    ``segment_ref``: the SMEM ``[3 * count]`` offsets, lengths and
    checkpoints, then, when ``prefixed``, ``count`` arena bases and ``count``
    cached lengths. Row r of sequence s attends arena rows
    ``[base_s, base_s + cached_s)`` of the layer in the third scalar ref
    (``prefixed``), then window keys ``[offset_s, r]``; rows outside every
    sequence return zero and their values are zeroed. The window is a
    one-layer arena (keys ``[1, 3, T, 256]``, values transposed
    ``[1, 3, 128, T]``). Offsets are multiples of 256.

    ``selected``: a scalar ref after the layer's, the dense flag, and an
    HBM ref after the keys and values, the selection words transposed
    (``[(capacity + T) / 32, T]`` int32; ``capacity`` the arena's rows):
    unless the flag is set, a key counts for a row only where its bit is
    set too, double-buffered per key block with the keys (``_Selection``)."""
    selection = None
    if selected:
        refs = list(refs)
        dense = refs.pop(1 if prefixed else 0)[0]
        words = refs.pop(6 if prefixed else 3)
        *refs, select_buf, select_sems = refs
        selection = (words, select_buf, select_sems, jnp.where(dense == 0, 0, -1).astype(jnp.int32))
    if prefixed:
        layer_ref, q_ref, k_hbm, v_hbm, prefix_k_hbm, prefix_v_hbm, out_ref, k_buf, v_buf, sems = refs
    else:
        q_ref, k_hbm, v_hbm, out_ref, k_buf, v_buf, sems = refs
    head, query_block = pl.program_id(0), pl.program_id(1)
    length = length_ref[0]
    offsets = [segment_ref[s] for s in range(count)]
    ends = [segment_ref[s] + segment_ref[count + s] for s in range(count)]
    first_row = query_block * q_tile
    rows = first_row + jnp.arange(q_tile)
    q = q_ref[0]
    row_start = rows + 1  # padding rows: no keys
    lowest = jnp.int32(1 << 30)
    # The first block after the tile's sequence start, when one sequence
    # holds every row of the tile.
    interior = jnp.int32(1 << 30)
    for s in range(count):
        inside = (rows >= offsets[s]) & (rows < ends[s])
        row_start = jnp.where(inside, offsets[s], row_start)
        meets = (offsets[s] < first_row + q_tile) & (ends[s] > first_row) & (ends[s] > offsets[s])
        lowest = jnp.where(meets, jnp.minimum(lowest, offsets[s]), lowest)
        holds = (offsets[s] <= first_row) & (first_row + q_tile <= ends[s])
        interior = jnp.where(holds, (offsets[s] + tile - 1) // tile, interior)
    first_block = jnp.where(lowest < (1 << 30), lowest // tile, 0)
    last_query_key = ((query_block + 1) * q_tile + tile - 1) // tile
    last_real_key = (length + tile - 1) // tile
    count_blocks = jnp.where((first_row < length) & (lowest < (1 << 30)),
                             jnp.minimum(last_query_key, last_real_key), 0)

    def real(indices):
        inside = indices < 0
        for s in range(count):
            inside = inside | ((indices >= offsets[s]) & (indices < ends[s]))
        return inside

    def window_mask(key_rows, sl):
        # Keys in [row_start, row] of a real row are its own sequence's; a
        # padding row's range is empty (row_start = row + 1).
        return (key_rows <= rows[None, sl]) & (key_rows >= row_start[None, sl])

    carry = (
        jnp.full((1, q_tile), -jnp.inf, jnp.float32),
        jnp.zeros((128 + _ONES, q_tile), jnp.float32),
    )
    select = None if selection is None else _Selection(*selection, first_row, capacity)
    if prefixed:
        carry = _prefix_phase(
            carry, q, segment_ref, layer_ref[0], head, first_row, rows, first_row < length,
            prefix_k_hbm, prefix_v_hbm, k_buf, v_buf, sems, tile=tile, q_tile=q_tile,
            count=count, select=None if select is None else select._replace(column=0))

    # Blocks [a, b) see every row and need no mask: past the sequence start,
    # wholly before the tile's first row.
    b = jnp.minimum(count_blocks, first_row // tile)
    a = jnp.maximum(first_block, interior)
    a, b = jnp.where(a < b, a, count_blocks), jnp.where(a < b, b, count_blocks)
    loop = functools.partial(_key_loop, q=q, layer=0, head=head, k_hbm=k_hbm, v_hbm=v_hbm, k_buf=k_buf,
                             v_buf=v_buf, sems=sems, tile=tile, select=select)
    carry = loop(carry, first_block, a, mask=window_mask, value_mask=real)
    carry = loop(carry, a, b, streamed=True)
    carry = loop(carry, b, count_blocks, mask=window_mask, value_mask=real)
    _, accumulator = carry
    denominator = accumulator[128:129]
    safe = jnp.where(denominator > 0, denominator, 1.0)
    out_ref[0] = (accumulator[:128] / safe).T.astype(jnp.bfloat16)


def _prefix_phase(carry, q, segment_ref, layer, head, first_row, rows, live, k_hbm, v_hbm, k_buf, v_buf, sems,
                  *, tile, q_tile, count, select=None):
    """The cached-prefix part of ``_flash_kernel_packed``: fold each
    sequence's arena rows into the online softmax ``carry`` of the query
    tile ``q`` (window ``rows``, from ``first_row``), once per distinct
    sequence of the tile: the key blocks wholly inside its rows stream
    unmasked over the whole tile and only its rows keep the result; the
    partial blocks at either end are masked."""
    offsets = [segment_ref[s] for s in range(count)]
    ends = [segment_ref[s] + segment_ref[count + s] for s in range(count)]
    row_sequence = jnp.full(rows.shape, -1, jnp.int32)
    for s in range(count):
        row_sequence = jnp.where((rows >= offsets[s]) & (rows < ends[s]), s, row_sequence)

    # A rolled sequence loop avoids emitting a full attention body for every
    # 256-row query slice, including slices that repeat the same sequence.
    def visit(sequence, carry):
        offset = segment_ref[sequence]
        end = offset + segment_ref[count + sequence]
        base = segment_ref[3 * count + sequence]
        size = jnp.where(live & (offset < first_row + q_tile) & (end > first_row),
                         segment_ref[4 * count + sequence], 0)
        lo_row, hi_row = base, base + size
        lo = base // tile
        hi = jnp.where(size > 0, (base + size + tile - 1) // tile, lo)

        def mask(key_rows, sl):
            return (row_sequence[None, sl] == sequence) & (key_rows >= lo_row) & (key_rows < hi_row)

        def value_mask(value_rows):
            return (value_rows >= lo_row) & (value_rows < hi_row)

        loop = functools.partial(_key_loop, q=q, layer=layer, head=head, k_hbm=k_hbm, v_hbm=v_hbm,
                                 k_buf=k_buf, v_buf=v_buf, sems=sems, tile=tile, select=select)
        # Whole blocks [a, b); without one, [lo, hi) is all edge.
        a, b = (lo_row + tile - 1) // tile, hi_row // tile
        a, b = jnp.where(a < b, a, hi), jnp.where(a < b, b, hi)

        # Both edges, [lo, a) and [b, hi), through one rolled masked body.
        def edge(i, carry):
            return loop(carry, jnp.where(i == 0, lo, b), jnp.where(i == 0, a, hi), mask=mask,
                        value_mask=value_mask)

        carry = jax.lax.fori_loop(0, 2, edge, carry)
        streamed = loop(carry, a, b, streamed=True)
        own = row_sequence[None] == sequence
        return tuple(_mask_where(own, new, old) for new, old in zip(streamed, carry))

    # Visit only the sequences that meet the tile (contiguous: offsets
    # ascend); any other visit is a no-op, but one per slot and query tile
    # costs ~0.6 ms a layer at 16 slots.
    first, end = jnp.int32(count), jnp.int32(0)
    for s in range(count):
        meets = live & (offsets[s] < first_row + q_tile) & (ends[s] > first_row)
        first = jnp.where(meets, jnp.minimum(first, s), first)
        end = jnp.where(meets, jnp.maximum(end, s + 1), end)
    return jax.lax.fori_loop(first, jnp.maximum(first, end), visit, carry)


class _Selection(NamedTuple):
    """A query tile's selection bits in ``_flash_kernel_packed``: ``words``
    (HBM ``[C / 32, T]`` int32: bit ``c % 32`` of word ``[c // 32, row]``
    for key column ``c``) streamed per key block into ``buf`` (``[2, tile
    / 32, q_tile]``, the keys' slots) on ``sems``; ``column`` is the loop's
    first key column (arena 0, window ``capacity``). ``fill`` -1 (dense):
    no copies, every bit set; 0: the copied bits."""

    words: Any
    buf: Any
    sems: Any
    fill: Any
    first_row: Any
    column: int

    def copy(self, key_block, slot):
        rows, queries = self.buf.shape[1:]
        return pltpu.make_async_copy(
            self.words.at[pl.ds(self.column // 32 + key_block * rows, rows), pl.ds(self.first_row, queries)],
            self.buf.at[slot], self.sems.at[slot])

    def start(self, key_block, slot):
        pl.when(self.fill == 0)(self.copy(key_block, slot).start)

    def wait(self, key_block, slot):
        pl.when(self.fill == 0)(self.copy(key_block, slot).wait)

    def keep(self, slot, sl):
        """``[tile, fragment]`` bool: the block's keys each query of ``sl`` keeps."""
        words = self.buf[slot, :, sl] | self.fill
        rows, queries = words.shape
        bits = jnp.broadcast_to(words[:, None], (rows, 32, queries)).reshape(32 * rows, queries)
        shift = jax.lax.broadcasted_iota(jnp.int32, bits.shape, 0) & 31
        return ((bits >> shift) & 1) != 0


# The online softmax runs transposed, keys by queries (S^T = K Q^T,
# out^T = [V^T; ones] P^T): per-query statistics reduce over sublanes, and PV
# streams only the value rows (plus _ONES rows of ones for the denominators)
# against the probabilities as MXU weights. QK and softmax for FRAGMENT-query
# fragment i overlap PV for i - 1 (separate physical/query-compute tiling and
# delayed PV follow the scheduling idea in heavyball-research/flywheel-tpu
# (Apache-2.0), f646686, pallas/fwd_pipeline.py).
FRAGMENT = 512
_ONES = 8
# Streamed blocks exponentiate against the maximum of the earlier key
# blocks; probabilities reach at most 2**STALE_LIMIT before the
# end-of-block rescale, far inside FP32/BF16 range even summed over a block.
STALE_LIMIT = 64.0


def _key_loop(carry, lo, hi, *, q, layer, head, k_hbm, v_hbm, k_buf, v_buf, sems, tile, mask=None,
              value_mask=None, streamed=False, select=None):
    """Fold key blocks ``[lo, hi)`` of arena layer ``layer`` (keys
    ``[L, 3, C, 256]``, values ``[L, 3, 128, C]``) into the transposed online
    softmax ``carry``: ``[1, Q]`` running maxima and ``[128 + _ONES, Q]``
    value and denominator accumulators of the base-2 queries ``q``.

    ``mask(key_rows, fragment slice)`` selects the scores that count and
    ``value_mask(key_rows)`` the real value rows (others, possibly NaN, are
    zeroed); both ``None`` when every row sees every key. ``streamed``
    (unmasked only): blocks after the first exponentiate against the running
    maximum (``_stale_step``), and the loop reruns exactly if a block's
    maximum exceeds it by ``STALE_LIMIT``. ``select`` (``_Selection``): each
    block's selection bits stream with its keys and mask its scores too,
    streamed blocks included."""

    def copies(key_block, slot):
        return (
            pltpu.make_async_copy(k_hbm.at[layer, head, pl.ds(key_block * tile, tile), :], k_buf.at[slot],
                                  sems.at[0, slot]),
            pltpu.make_async_copy(v_hbm.at[layer, head, :, pl.ds(key_block * tile, tile)], v_buf.at[slot],
                                  sems.at[1, slot]),
        )

    slots = k_buf.shape[0]

    def start(key_block, slot):
        for copy in copies(key_block, slot):
            copy.start()
        if select is not None:
            select.start(key_block, slot)

    def prime():
        start(lo, lo % slots)

    def block(key_block):
        """Wait for this block's key/value tile and prefetch the next."""
        slot = key_block % slots
        for copy in copies(key_block, slot):
            copy.wait()
        if select is not None:
            select.wait(key_block, slot)

        @pl.when(key_block + 1 < hi)
        def prefetch():
            start(key_block + 1, (key_block + 1) % slots)

        k, v = k_buf[slot], v_buf[slot]
        if value_mask is not None:
            # One row of key indices: the mask varies along the keys only.
            value_keys = key_block * tile + jax.lax.broadcasted_iota(jnp.int32, (1, tile), 1)
            v = jnp.where(value_mask(value_keys), v, jnp.zeros((), v.dtype))
        keep = None if select is None else functools.partial(select.keep, slot)
        return k, jnp.concatenate((v, jnp.ones((_ONES, tile), v.dtype)), axis=0), keep

    def step(key_block, carry):
        k, v, keep = block(key_block)
        key_rows = key_block * tile + jax.lax.broadcasted_iota(jnp.int32, (tile, 1), 0)
        scores_mask = None if mask is None else functools.partial(mask, key_rows)
        if keep is not None:
            scores_mask = keep if scores_mask is None else (
                lambda sl, masked=scores_mask: masked(sl) & keep(sl))
        return _exact_step(q, k, v, carry, scores_mask)

    def exact(c):
        pl.when(lo < hi)(prime)
        return jax.lax.fori_loop(lo, hi, step, c)

    if not streamed:
        return exact(carry)

    def stream(c):
        def body(key_block, state):
            c, worst = state
            k, v, keep = block(key_block)
            c, gap = _stale_step(q, k, v, c, keep)
            return c, jnp.maximum(worst, gap)

        prime()
        first = step(lo, c)
        worst = jnp.full(c[0].shape, -jnp.inf, jnp.float32)
        result, worst = jax.lax.fori_loop(lo + 1, hi, body, (first, worst))
        return jax.lax.cond(jnp.max(worst) > STALE_LIMIT, exact, lambda _: result, c)

    return jax.lax.cond(lo < hi, stream, lambda c: c, carry)


def _exact_step(q, k, v, carry, mask):
    """One key block of the transposed online softmax (``_key_loop``), each
    query's maximum updated before exponentiating (a query may start with
    no key: a -inf maximum). ``v`` is ``[V^T; ones]``."""
    maximum, accumulator = carry
    fragment = min(FRAGMENT, q.shape[0])
    maxima, accumulators = [], []
    previous = None

    def fold(sl, probabilities, correction):
        weighted = jax.lax.dot_general(v, probabilities, (((1,), (0,)), ((), ())),
                                       preferred_element_type=jnp.float32)
        accumulators.append(accumulator[:, sl] * correction + weighted)

    for qi in range(q.shape[0] // fragment):
        sl = slice(qi * fragment, (qi + 1) * fragment)
        scores = jax.lax.dot_general(k, q[sl], (((1,), (1,)), ((), ())),
                                     preferred_element_type=jnp.float32)  # [keys, queries]
        if mask is not None:
            scores = jnp.where(mask(sl), scores, -jnp.inf)
        new_maximum = jnp.maximum(maximum[:, sl], jnp.max(scores, axis=0, keepdims=True))
        # A query with no key yet keeps a -inf maximum; exponentiate against 0.
        safe_maximum = jnp.where(new_maximum == -jnp.inf, 0.0, new_maximum)
        probabilities = jnp.exp2(scores - safe_maximum).astype(jnp.bfloat16)
        maxima.append(new_maximum)
        if previous is not None:
            fold(*previous)
        previous = sl, probabilities, jnp.exp2(maximum[:, sl] - safe_maximum)
    assert previous is not None  # every query tile has a fragment
    fold(*previous)
    return jnp.concatenate(maxima, axis=1), jnp.concatenate(accumulators, axis=1)


def _stale_step(q, k, v, carry, keep=None):
    """One unmasked key block with every query's running maximum finite:
    exponentiate the block's scores against the carried maximum (not the
    block's own), so each score feeds the max and the exponent straight
    from the MXU instead of being stored and read back, and rescale once
    the block is folded in. The carry means the same as after an exact
    step. Returns (carry, each query's block maximum less the carried
    one, ``[1, Q]``). ``keep(fragment slice)``: the selected scores; a
    query the selection has left no key so far exponentiates against 0, its
    gap the block maximum's distance from 0."""
    maximum, accumulator = carry
    fragment = min(FRAGMENT, q.shape[0])
    maxima, accumulators, gaps = [], [], []
    previous = None

    def fold(sl, probabilities, reference, target):
        weighted = jax.lax.dot_general(v, probabilities, (((1,), (0,)), ((), ())),
                                       preferred_element_type=jnp.float32)
        accumulators.append((accumulator[:, sl] + weighted) * jnp.exp2(reference - target))

    for qi in range(q.shape[0] // fragment):
        sl = slice(qi * fragment, (qi + 1) * fragment)
        scores = jax.lax.dot_general(k, q[sl], (((1,), (1,)), ((), ())),
                                     preferred_element_type=jnp.float32)  # [keys, queries]
        carried = reference = maximum[:, sl]
        if keep is not None:
            scores = jnp.where(keep(sl), scores, -jnp.inf)
            reference = jnp.where(carried == -jnp.inf, 0.0, carried)
        block_maximum = jnp.max(scores, axis=0, keepdims=True)
        probabilities = jnp.exp2(scores - reference).astype(jnp.bfloat16)
        new_maximum = jnp.maximum(carried, block_maximum)
        maxima.append(new_maximum)
        gap = block_maximum - reference
        target = new_maximum
        if keep is not None:
            # A query with no key yet reruns exactly unless the block's maximum is within the limit of 0 either
            # way: its probabilities against 0 then neither overflow nor underflow wholesale.
            first = jnp.where(block_maximum == -jnp.inf, -jnp.inf, jnp.abs(block_maximum))
            gap = jnp.where(carried == -jnp.inf, first, gap)
            target = jnp.where(new_maximum == -jnp.inf, reference, new_maximum)  # still no key
        gaps.append(gap)
        if previous is not None:
            fold(*previous)
        previous = sl, probabilities, reference, target
    assert previous is not None  # every query tile has a fragment
    fold(*previous)
    carry = jnp.concatenate(maxima, axis=1), jnp.concatenate(accumulators, axis=1)
    return carry, jnp.concatenate(gaps, axis=1)


class CpPrefix(NamedTuple):
    """Context parallel prefixes (vLLM decode context parallelism): each
    rank's pool holds only its share of every sequence's pages (page p on
    rank p % ranks); each MLA layer all-gathers every rank's pages and
    attends its heads over all of them (see make_mla_attention_layer_sp_ring).
    ``pools``: the op's MLA layers' unified pools, in order (rank-sharded);
    ``ids``: ``[32, 1 + P]`` int32, rank-sharded -- each rank's own prefix
    page count then its pages, sequences end to end; ``local``: ``[32, 2 *
    count]`` int32, rank-sharded -- each sequence's base in that rank's
    arena and its token count there."""

    pools: tuple
    ids: Any
    local: Any

    def split(self):
        return tuple(CpPrefix((pool, ), self.ids, self.local) for pool in self.pools)


GATHER_STREAMS = 6  # the three physical ring cycles, both directions
GATHER_PARTS = 4  # chunks per page (32 of its 128 rows each)


def _prefix_page_gather_kernel(tables, counts_ref, ids_ref, pool_hbm, out_hbm, local_sem, send_sems, recv_sems,
                               done, *, slots, flat=0):
    """All-gather every rank's valid prefix pages over the six TP32 rings.

    Rank r's ``counts[r]`` pages (``ids[1:]``) land in ``out`` from slot
    ``r * slots + 1``. Chunk c of an owner's pages (``GATHER_PARTS`` per
    page) rides stream ``s = c % GATHER_STREAMS`` (family ``s // 2 % 3``,
    direction ``s % 2``); at step k a rank receives owner ``pos - dir k``'s
    chunks and forwards them. Each owner has its own slots, so no credits
    are needed; a neighbour may run steps ahead, so each step has its own
    receive semaphore. Entry barrier: the neighbours' ``out`` exists; exit
    handshake: no neighbour signals the next call's barrier early. ``flat``
    (a page's tokens): ``out`` is token rows ``[slots * flat + extra,
    lanes]``, slot ``s`` at rows ``s * flat ..``, each page retiled by its
    local copy."""
    ranks = tables.shape[-1]
    rows = out_hbm.shape[1] // GATHER_PARTS

    def target(slot, part=None):
        if flat:
            size = flat if part is None else flat // GATHER_PARTS
            return out_hbm.at[pl.ds(slot * flat + (0 if part is None else part * size), size)]
        return out_hbm.at[slot] if part is None else out_hbm.at[slot, pl.ds(part * rows, rows)]
    rank = lax.axis_index("k3_sp")
    positions = [tables[f, 0, rank] for f in range(3)]

    def ring(family, position):
        return tables[family, 1, position % ranks]

    neighbours = [ring(f, positions[f] + d) for f in range(3) for d in (-1, 1)]

    def chunks(owner, stream):
        return jnp.maximum(counts_ref[owner] * GATHER_PARTS - stream + GATHER_STREAMS - 1, 0) // GATHER_STREAMS

    def chunk(owner, stream, i):
        c = stream + GATHER_STREAMS * i
        return target(owner * slots + 1 + c // GATHER_PARTS, c % GATHER_PARTS)

    def remote(src, stream, step, peer):
        return pltpu.make_async_remote_copy(src, src, send_sems.at[stream], recv_sems.at[stream, step],
                                            device_id=(peer, ), device_id_type=pl.DeviceIdType.MESH)

    def local(q):
        page = pool_hbm.at[ids_ref[1 + q]]
        if flat:
            page = page.reshape(flat, out_hbm.shape[-1])
        return pltpu.make_async_copy(page, target(rank * slots + 1 + q), local_sem)

    count = counts_ref[rank]

    @pl.loop(0, count)
    def start_local(q):
        local(q).start()

    barrier = pltpu.get_barrier_semaphore()
    for peer in neighbours:
        pl.semaphore_signal(barrier, 1, device_id=(peer, ), device_id_type=pl.DeviceIdType.MESH)
    pl.semaphore_wait(barrier, len(neighbours))

    @pl.loop(0, count)
    def finish_local(q):
        local(q).wait()

    streams = []
    for stream in range(GATHER_STREAMS):
        family, direction = stream // 2 % 3, (-1 if stream % 2 == 0 else 1)
        streams.append((stream, family, direction, ring(family, positions[family] + direction)))

    for stream, _, _, downstream in streams:
        @pl.loop(0, chunks(rank, stream))
        def seed(i, stream=stream, downstream=downstream):
            remote(chunk(rank, stream, i), stream, 1, downstream).start()

    @pl.loop(1, ranks)
    def step(k):
        for stream, family, direction, downstream in streams:
            owner = ring(family, positions[family] - direction * k)
            size = chunks(owner, stream)

            @pl.loop(0, size)
            def receive(i, stream=stream, owner=owner):
                remote(chunk(owner, stream, i), stream, k, rank).wait_recv()

            @pl.when(k < ranks - 1)
            def forward(stream=stream, owner=owner, size=size, downstream=downstream):
                @pl.loop(0, size)
                def send(i):
                    remote(chunk(owner, stream, i), stream, k + 1, downstream).start()

    # Every chunk this rank sent: its own (step 0) and owners 1..30 steps upstream.
    @pl.loop(0, ranks - 1)
    def drain(k):
        for stream, family, direction, downstream in streams:
            owner = ring(family, positions[family] - direction * k)

            @pl.loop(0, chunks(owner, stream))
            def sent(i, stream=stream, owner=owner, downstream=downstream):
                remote(chunk(owner, stream, i), stream, k + 1, downstream).wait_send()

    for peer in neighbours:
        pl.semaphore_signal(done, 1, device_id=(peer, ), device_id_type=pl.DeviceIdType.MESH)
    pl.semaphore_wait(done, len(neighbours))


def gathered_prefix_pages(pool, ids, local, tables, axis_name, *, count, capacity, page_tokens=None,
                          flat_extra=None):
    """Every rank's cached prefix pages, gathered (inside ``shard_map``):
    ``(pages, order)`` for ``expand_prefix_pages``. ``pool``/``ids``/``local``
    are this rank's view (see ``CpPrefix``). ``pages`` holds every rank's
    ``1 + P`` page slots in mesh order (``[32 (1 + P), ...]``; only valid
    pages are written); ``order`` is ``prefix_page_order``'s, ``capacity``
    the most prefix tokens a window holds. ``page_tokens``: the tokens a
    page of ``pool`` holds (default: a latent pool's, ``[pages, rows,
    packing, lanes]``). ``flat_extra`` (a latent pool's): ``pages`` as token
    rows instead, ``[32 (1 + P) * page_tokens + flat_extra, lanes]`` (slot
    ``s``'s token ``t`` at row ``s * page_tokens + t``), the last
    ``flat_extra`` rows unwritten."""
    if page_tokens is None:
        page_tokens = pool.shape[-3] * pool.shape[-2]
    slots = ids.shape[0]  # 1 + P: slot 0 holds the count (never referenced)
    every = jax.lax.all_gather(local, axis_name)  # [32, 2 * count]: bases, token counts
    ranks = every.shape[0]
    counts = jnp.sum(every[:, count:], axis=1) // page_tokens  # pages per rank
    flat = 0 if flat_extra is None else page_tokens
    shape = ((ranks * slots, *pool.shape[1:]) if flat_extra is None
             else (ranks * slots * page_tokens + flat_extra, pool.shape[-1]))
    pages = pl.pallas_call(
        functools.partial(_prefix_page_gather_kernel, slots=slots, flat=flat),
        out_shape=jax.ShapeDtypeStruct(shape, pool.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA((GATHER_STREAMS, )),
                            pltpu.SemaphoreType.DMA((GATHER_STREAMS, ranks)), pltpu.SemaphoreType.REGULAR),
        ),
        compiler_params=pltpu.CompilerParams(collective_id=44),
        name="kimi_prefill_prefix_page_gather",
    )(jnp.asarray(tables, jnp.int32), counts.astype(jnp.int32), ids.astype(jnp.int32), pool)
    return pages, prefix_page_order(every, count=count, page_tokens=page_tokens, slots=slots,
                                    capacity=capacity)


def prefix_page_order(every, *, count, page_tokens, slots, capacity):
    """The arena order of ``gathered_prefix_pages``: ``[1 + capacity / page]``
    int32, the total page count, then the gathered slot of each arena page.
    ``every`` is every rank's ``local`` (``[ranks, 2 * count]``: each
    sequence's base and token count in that rank's pages); rank r's local
    page q sits at gathered slot ``r * slots + 1 + q``. Arena runs go
    sequence-major, then mesh rank, so each sequence is contiguous at its
    global base (descriptor ``[3 * count:5 * count]``). Within a page the
    arena keeps the page's token order (``expand_prefix_pages``), so arena
    row ``c`` is token ``c % page_tokens`` of gathered slot ``order[1 + c //
    page_tokens]``: Kimi-K3.1's selection names arena columns by it."""
    ranks = every.shape[0]
    starts = (every[:, :count] // page_tokens).T.reshape(-1)  # [count * ranks] local first pages
    sizes = (every[:, count:] // page_tokens).T.reshape(-1)
    run_rank = jnp.tile(jnp.arange(ranks, dtype=jnp.int32), count)
    ends = jnp.cumsum(sizes)
    if capacity % page_tokens:
        raise ValueError(f"the prefix capacity must be whole {page_tokens}-token pages")
    arena = jnp.arange(capacity // page_tokens, dtype=jnp.int32)
    run = jnp.minimum(jnp.sum(arena[:, None] >= ends[None, :], axis=1), sizes.shape[0] - 1)
    within = arena - (ends[run] - sizes[run])
    source = run_rank[run] * slots + 1 + starts[run] + within
    total = ends[-1]
    order = jnp.where(arena < total, source, 0).astype(jnp.int32)
    return jnp.concatenate((total[None].astype(jnp.int32), order))


EXPAND_PAGES = 4  # pages per expansion step


def _expand_pages_kernel(order_ref, pages_hbm, up_ref, keys_hbm, values_hbm, buf, kbuf, vbuf, sems, *, heads):
    """``expand_prefix_pages``: double-buffered page reads by ``order``,
    each step's rows up-projected per head into VMEM, then written to the
    head-major arena. Only ``order[0]`` pages are visited."""
    total = order_ref[0]
    _, pages, page_rows, packing, lanes = buf.shape
    page_tokens = page_rows * packing
    span = pages * page_tokens
    steps = (total + pages - 1) // pages
    words = buf.bitcast(jnp.uint32)  # a 32-bit word packs `packing` tokens

    def reads(c, slot):
        return [pltpu.make_async_copy(pages_hbm.at[order_ref[1 + c * pages + j]], buf.at[slot, j], sems.at[0, slot])
                for j in range(pages)]

    def writes(c, slot):
        rows = pl.ds(c * span, span)
        return [pltpu.make_async_copy(kbuf.at[slot], keys_hbm.at[:, rows], sems.at[1, slot]),
                pltpu.make_async_copy(vbuf.at[slot], values_hbm.at[:, :, rows], sems.at[2, slot])]

    @pl.when(steps > 0)
    def prime():
        for copy in reads(0, 0):
            copy.start()

    def step(c, carry):
        slot = c % 2
        for copy in reads(c, slot):
            copy.wait()

        @pl.when(c + 1 < steps)
        def prefetch():
            for copy in reads(c + 1, 1 - slot):
                copy.start()

        @pl.when(c >= 2)
        def reuse():  # this slot's writes from two steps back
            for copy in writes(c - 2, slot):
                copy.wait()

        for j in range(pages):
            # Page row r, pack i is token r * packing + i, and stays arena row j * page_tokens + it
            # (bitcast's packed-row order): the DSA selection names arena columns by it.
            x = pltpu.bitcast(words.at[slot, j].reshape(page_rows, lanes)[...], buf.dtype).astype(jnp.bfloat16)
            x = jnp.where(c * pages + j < total, x, jnp.zeros((), x.dtype))  # stale slots may hold NaN
            rows = pl.ds(j * page_tokens, page_tokens)
            rope = x[:, 512:576]
            for h in range(heads):
                expanded = jnp.dot(x[:, :512], up_ref[:, h * 256:(h + 1) * 256],
                                   preferred_element_type=jnp.float32).astype(jnp.bfloat16)
                kbuf[slot, h, rows, :] = jnp.concatenate(
                    (expanded[:, :128], rope, jnp.zeros((page_tokens, 64), jnp.bfloat16)), axis=-1)
                vbuf[slot, h, :, rows] = expanded[:, 128:].T
        for copy in writes(c, slot):
            copy.start()
        return carry

    jax.lax.fori_loop(0, steps, step, 0)

    @pl.when(steps >= 2)
    def drain_previous():
        for copy in writes(steps - 2, (steps - 2) % 2):
            copy.wait()

    @pl.when(steps >= 1)
    def drain_last():
        for copy in writes(steps - 1, (steps - 1) % 2):
            copy.wait()


def expand_prefix_pages(pages, order, key_value_up):
    """One MLA layer's cached prefixes as a one-layer arena for
    ``mla_flash_prefill_packed``: keys ``[1, H, C, 256]`` and values
    transposed, ``[1, H, 128, C]``, with H = ``key_value_up``'s heads (this
    rank's 3) and C = ``(len(order) - 1) * page_tokens``. Arena page i is
    gathered page ``order[1 + i]`` (unified-pool layout
    ``[page_rows, packing, 640]``, lanes 0-575 the latent; token ``r *
    packing + i`` at ``[r, i]``) in its token order, up-projected exactly as
    ``mla_projections`` makes a new token's key and value. Rows
    past the ``order[0]`` pages (rounded up to ``EXPAND_PAGES``) are never
    written: the flash kernel reads only each sequence's rows."""
    heads = key_value_up.shape[1] // 256
    page_rows, packing, lanes = pages.shape[1:]
    page_tokens = page_rows * packing
    span = EXPAND_PAGES * page_tokens
    capacity = (order.shape[0] - 1) * page_tokens
    if capacity % span:
        raise ValueError(f"the prefix capacity must be a multiple of {span} tokens")
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    keys, values = pl.pallas_call(
        functools.partial(_expand_pages_kernel, heads=heads),
        out_shape=(jax.ShapeDtypeStruct((heads, capacity, 256), jnp.bfloat16),
                   jax.ShapeDtypeStruct((heads, 128, capacity), jnp.bfloat16)),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[hbm, pl.BlockSpec(memory_space=pltpu.VMEM)],
            out_specs=(hbm, hbm),
            scratch_shapes=(
                pltpu.VMEM((2, EXPAND_PAGES, page_rows, packing, lanes), pages.dtype),
                pltpu.VMEM((2, heads, span, 256), jnp.bfloat16),
                pltpu.VMEM((2, heads, 128, span), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((3, 2)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 1024**2),
        name="kimi_prefill_expand_prefix_pages",
    )(order.astype(jnp.int32), pages, key_value_up)
    return keys[None], values[None]


def mla_flash_prefill_packed(query, key, value, length, segments, *, count, tile=TILE, arena=None, layer=0,
                             select=None, dense=None, select_transposed=False):
    """Causal MLA over a packed window of ``count`` sequences, per head.

    Args:
        query, key, value: The window's head-major ``[3, T, 256]``,
            ``[3, T, 256]`` and ``[3, T, 128]`` BF16 rows, queries
            pre-scaled by ``SCALE_LOG2``.
        length: ``[1]``, the end of the last sequence.
        segments: The int32 descriptor (``_flash_kernel_packed``),
            ``[3 * count]``, or ``[5 * count]`` with ``arena``.
        arena: Optional ``([L, 3, C, 256], [L, 3, 128, C])`` cached-prefix
            keys and transposed values, read in place from HBM.
        layer: The arena layer, a ``[1]`` or scalar int32.
        select: Optional uint32 ``[T, W]``, ``W`` at least ``(C + T) / 32``
            (C 0 without ``arena``): bit ``c % 32`` of word ``c // 32`` keeps the
            ``[arena | window]`` key column ``c`` for the row, beside the
            causal, segment and arena rule.
        dense: With ``select``, int32 ``[1]``: nonzero ignores ``select``.
        select_transposed: ``select`` is given as ``[W, T]``.

    Returns:
        ``[3, T, 128]`` BF16; padded rows are zero.
    """
    if query.ndim != 3 or query.shape[2] != 256:
        raise ValueError("query must be [H,T,256]")
    heads, tokens = query.shape[:2]
    if key.shape != (heads, tokens, 256) or value.shape != (heads, tokens, 128):
        raise ValueError("packed MLA keys/values are the window's rows")
    fields = 3 if arena is None else 5
    if segments.shape != (fields * count,):
        raise ValueError(f"segments must be [{fields} * count]")
    length = jnp.asarray(length, jnp.int32).reshape((1,))
    q_tile, k_tile = _tiles(tokens, tokens, tile)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    scalars = (length, segments.astype(jnp.int32))
    operands = (query, key[None], jnp.swapaxes(value, 1, 2)[None])  # the window as a one-layer arena
    in_specs = (pl.BlockSpec((1, q_tile, 256), lambda h, i, *_: (h, i, 0)), hbm, hbm)
    if arena is not None:
        arena_keys, arena_values = arena
        if (arena_keys.ndim != 4 or arena_keys.shape[1:] != (heads, arena_keys.shape[2], 256)
                or arena_values.shape != (*arena_keys.shape[:2], 128, arena_keys.shape[2])
                or arena_keys.shape[2] % k_tile):
            raise ValueError(f"arena must be [L,{heads},C,256] / [L,{heads},128,C] with C a multiple of {k_tile}")
        scalars += (jnp.asarray(layer, jnp.int32).reshape((1,)),)
        operands += (arena_keys, arena_values)
        in_specs += (hbm, hbm)
    scratch = (
        pltpu.VMEM((2, k_tile, 256), jnp.bfloat16),
        pltpu.VMEM((2, 128, k_tile), jnp.bfloat16),
        pltpu.SemaphoreType.DMA((2, 2)),
    )
    capacity = 0 if arena is None else arena[0].shape[2]
    if select is not None:
        rows, words = select.shape[::-1] if select_transposed else select.shape
        if rows != tokens or words < (capacity + tokens) // 32 or select.dtype != jnp.uint32:
            raise ValueError(f"select must be uint32 [{tokens}, at least {(capacity + tokens) // 32}]")
        dense = jnp.zeros((1,), jnp.int32) if dense is None else dense
        scalars += (jnp.asarray(dense, jnp.int32).reshape((1,)),)
        words = jax.lax.bitcast_convert_type(select, jnp.int32)
        operands += (words if select_transposed else jnp.swapaxes(words, 0, 1),)
        in_specs += (hbm,)
        scratch += (pltpu.VMEM((2, k_tile // 32, q_tile), jnp.int32), pltpu.SemaphoreType.DMA((2,)))
    elif dense is not None:
        raise ValueError("dense applies to a select")
    name = ("kimi_mla_flash_prefill_packed" + ("_prefixed" if arena is not None else "")
            + ("_selected" if select is not None else ""))
    return pl.pallas_call(
        functools.partial(_flash_kernel_packed, tile=k_tile, q_tile=q_tile, count=count,
                          prefixed=arena is not None, selected=select is not None, capacity=capacity),
        out_shape=jax.ShapeDtypeStruct((heads, tokens, 128), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalars),
            grid=(heads, tokens // q_tile),
            in_specs=in_specs,
            out_specs=pl.BlockSpec((1, q_tile, 128), lambda h, i, *_: (h, i, 0)),
            scratch_shapes=scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"), vmem_limit_bytes=64 * 1024**2,
        ),
        name=name,
    )(*scalars, *operands)


def _tiles(tokens, capacity, tile):
    """(query tile, key tile). Default 1024-row query and 2048-row key
    tiles; ``KernelConfig.mla_flash_qtile`` / ``mla_flash_ktile`` override."""
    q_tile = _config().mla_flash_qtile
    k_tile = _config().mla_flash_ktile
    q_tile = q_tile if tokens % q_tile == 0 else tile
    k_tile = k_tile if capacity % k_tile == 0 else (512 if capacity % 512 == 0 else tile)
    return q_tile, k_tile


# ============================================================================
# SP block half rings
# ============================================================================
#
# The KDA layer's window all-gather and reduce-scatter, one SP block at a time.


# Half rings for one SP block. Each of the three cycles carries a column
# third; on each cycle of n ranks a rank's stripe travels all its rows both
# ways, ``half_hops(n)[+1]`` = n / 2 hops forward and n / 2 - 1 back, so a
# ring takes n / 2 + 1 steps instead of the full ring's n at the same bytes
# per link (17 instead of 32 on the slice).
def half_hops(n):
    return {1: n // 2, -1: n // 2 - 1}


def half_steps(n):
    return n // 2 + 1


HALF_STEPS = half_steps(32)


def _half_ring_streams(width):
    blocks = width // 128
    widths = [128 * (blocks // 3 + (i < blocks % 3)) for i in range(3)]
    cols = [0, widths[0], widths[0] + widths[1]]
    return tuple(
        (family, direction, cols[family], widths[family])
        for family in range(3) for direction in (1, -1)
    )


def _half_gather_step(tables_ref, local_ref, window_ref, send_sems, recv_sems, own_sem,
                      *, step, block):
    """Step ``step`` (0..16) of SP block ``block``'s window all-gather on
    half rings: copies go HBM to the peer's HBM, each owner's rows straight
    to their window rows (forwarded from there), so no buffer is reused and
    no credits are needed. ``local_ref``: the rank's rows; ``window_ref``:
    the window."""
    rank = jax.lax.axis_index("k3_sp")
    n = tables_ref.shape[-1]
    positions = [tables_ref[f, 0, rank] for f in range(3)]
    stripe = window_stripe(local_ref.shape[0])
    local_rows = pl.ds(block * stripe, stripe)

    def window_rows(family, position):
        owner = tables_ref[family, 2, tables_ref[family, 1, position % n]]
        return pl.ds(block * n * stripe + owner * stripe, stripe)

    @pl.when(step == 0)
    def own_rows():
        copy = pltpu.make_async_copy(
            local_ref.at[local_rows], window_ref.at[window_rows(0, positions[0])], own_sem)
        copy.start()
        if local_ref.shape[0] not in (16, 32, 64, 128, 256, 512, 1024, 2048):
            copy.wait()

    for stream, (family, direction, col, width) in enumerate(_half_ring_streams(local_ref.shape[1])):
        cols, hops, position = pl.ds(col, width), half_hops(n)[direction], positions[family]
        target = (tables_ref[family, 1, (position + direction) % n],)

        def send(index, source, family=family, direction=direction, cols=cols,
                 position=position, target=target, stream=stream):
            # The rows of the owner ``index`` hops back (index 0: this rank).
            rows = window_rows(family, position - direction * index)
            return pltpu.make_async_remote_copy(
                source if source is not None else window_ref.at[rows, cols],
                window_ref.at[rows, cols], send_sems.at[stream], recv_sems.at[stream],
                device_id=target, device_id_type=pl.DeviceIdType.MESH,
            )

        own = local_ref.at[local_rows, cols]
        pl.when(step == 0)(lambda send=send, own=own: send(0, own).start())

        @pl.when((step >= 1) & (step <= hops))
        def arrived(send=send):
            copy = send(step - 1, None)
            copy.wait_recv()
            copy.wait_send()

        @pl.when((step >= 1) & (step < hops))
        def forward(send=send):
            send(step, None).start()

    # The local destination is disjoint from every remote stripe. Let the
    # copy run while the half rings progress, then join before KDA consumes
    # this block and before own_sem is reused for the next block.
    if local_ref.shape[0] in (16, 32, 64, 128, 256, 512, 1024, 2048):
        @pl.when(step == half_steps(n) - 1)
        def finish_local():
            pltpu.make_async_copy(
                local_ref.at[local_rows], window_ref.at[window_rows(0, positions[0])], own_sem
            ).wait()


def _half_reduce_step(
    tables_ref, length_ref, partial_ref, prefix_ref, output_ref,
    read_sems, send_sems, recv_sems, credits, store_sem,
    local_bufs, payload, received, *, step, block, replace,
):
    """Step ``step`` (0..16) of SP block ``block``'s reduce-scatter of the
    rank's output-projection partials (``partial_ref`` ``[T, hidden]``) on
    half rings, adding the residual at the last step. On each cycle the
    partial sum for the owner ``hops - step`` positions ahead moves one hop
    a step, gaining this rank's partial; the owner adds both directions'
    sums and its own. ``local_bufs`` / ``received`` ``[2, 2, stripe,
    hidden]`` (slot, direction) and ``payload`` ``[2, stripe, hidden]``.

    ``partial_ref`` ``[G, T, hidden]``: the partials of G head groups, read
    into ``local_bufs`` ``[2, 2, G, stripe, hidden]`` and summed in FP32."""
    planes = partial_ref.ndim == 3

    def own(slot, side, cols):  # this rank's partial: BF16, or the planes' FP32 sum
        if not planes:
            return local_bufs.at[slot, side, :, cols][...]
        total = local_bufs.at[slot, side, 0, :, cols][...].astype(jnp.float32)
        for g in range(1, partial_ref.shape[0]):
            total += local_bufs.at[slot, side, g, :, cols][...].astype(jnp.float32)
        return total

    def stage(slot, side):  # a [stripe, hidden] buffer
        return local_bufs.at[slot, side, 0] if planes else local_bufs.at[slot, side]

    rank = jax.lax.axis_index("k3_sp")
    n = tables_ref.shape[-1]
    token_owner = tables_ref[0, 2, rank]
    positions = [tables_ref[f, 0, rank] for f in range(3)]
    stripe = payload.shape[1]
    local_first, window_first = block * stripe, block * n * stripe
    streams = _half_ring_streams(partial_ref.shape[-1])

    for stream, (family, direction, col, width) in enumerate(streams):
        cols, hops, position = pl.ds(col, width), half_hops(n)[direction], positions[family]
        side = 0 if direction == 1 else 1
        target = (tables_ref[family, 1, (position + direction) % n],)
        source = (tables_ref[family, 1, (position - direction) % n],)

        def read(index, family=family, direction=direction, cols=cols, position=position,
                 hops=hops, side=side, stream=stream):
            # This rank's partial for the owner ``hops - index`` positions ahead.
            peer = tables_ref[family, 1, (position + direction * (hops - index)) % n]
            rows = pl.ds(window_first + tables_ref[family, 2, peer] * stripe, stripe)
            semaphore = read_sems.at[index % 2, stream]
            if planes:
                return _Copies([pltpu.make_async_copy(
                    partial_ref.at[g, rows, cols], local_bufs.at[index % 2, side, g, :, cols],
                    semaphore) for g in range(partial_ref.shape[0])])
            return pltpu.make_async_copy(
                partial_ref.at[rows, cols], local_bufs.at[index % 2, side, :, cols], semaphore)

        def remote(index, cols=cols, side=side, target=target, stream=stream):
            return pltpu.make_async_remote_copy(
                payload.at[side, :, cols], received.at[index % 2, side, :, cols],
                send_sems.at[stream], recv_sems.at[index % 2, stream],
                device_id=target, device_id_type=pl.DeviceIdType.MESH,
            )

        pl.when(step == 0)(lambda read=read: read(0).start())

        @pl.when(step <= hops)
        def accumulate(read=read, remote=remote, cols=cols, side=side, hops=hops,
                       stream=stream, source=source):
            read(step).wait()
            pl.when(step < hops)(lambda: read(step + 1).start())
            local = own(step % 2, side, cols)

            @pl.when(step == 0)
            def seed():
                payload.at[side, :, cols][...] = local.astype(jnp.bfloat16)

            @pl.when((step > 0) & (step < hops))
            def add():
                remote(step - 1).wait()  # this rank's last send, and the next sum
                arrived = received.at[(step - 1) % 2, side, :, cols][...]
                if planes:
                    arrived = arrived.astype(jnp.float32)
                payload.at[side, :, cols][...] = (local + arrived).astype(jnp.bfloat16)
                pl.semaphore_signal(credits.at[stream], 1, device_id=source,
                                    device_id_type=pl.DeviceIdType.MESH)

            @pl.when(step < hops)
            def forward():
                pl.when(step >= 2)(lambda: pl.semaphore_wait(credits.at[stream], 1))
                remote(step).start()

            @pl.when(step == hops)
            def last():  # this rank's own sum has arrived
                remote(step - 1).wait()

    # The backward ring (n / 2 - 1 hops) is done after that step, and its
    # local buffer [1, 1] is next a partial target at the next block's step
    # 1: read the whole residual stripe into it one step early.
    prefix_copy = pltpu.make_async_copy(prefix_ref.at[pl.ds(local_first, stripe)], stage(1, 1), store_sem)
    pl.when(step == half_steps(n) - 2)(prefix_copy.start)

    @pl.when(step == half_steps(n) - 1)
    def finish():
        prefix_copy.wait()
        for stream, (family, direction, col, width) in enumerate(streams):
            if direction != 1:
                continue
            cols = pl.ds(col, width)
            # Both directions' sums (the backward ring's arrived a step
            # earlier), this rank's own partial (the forward ring's last
            # read) and the residual, staged in the backward ring's free
            # local buffer.
            staging = stage(1, 1).at[:, cols]
            valid = sp_token_index(
                token_owner,
                local_first + jax.lax.broadcasted_iota(jnp.int32, (stripe, width), 0),
                stripe,
                n,
            ) < length_ref[0]
            projection = (
                own(0, 0, cols).astype(jnp.float32)
                + received.at[1, 0, :, cols][...].astype(jnp.float32)
                + received.at[0, 1, :, cols][...].astype(jnp.float32)
            )
            residual = staging[...].astype(jnp.float32)
            # First layer of a residual block: the attention output starts
            # the new partial sum (the finished one is already a block).
            residual = jnp.where(replace, jnp.zeros_like(residual), residual)
            staging[...] = (residual + jnp.where(valid, projection, 0)).astype(jnp.bfloat16)
        # Unchanged per-column arithmetic; one write of the whole stripe.
        output = pltpu.make_async_copy(stage(1, 1), output_ref.at[pl.ds(local_first, stripe)], store_sem)
        output.start()
        output.wait()
        for stream in range(len(streams)):  # the senders' last buffer is free
            pl.semaphore_signal(
                credits.at[stream], 1,
                device_id=(tables_ref[streams[stream][0], 1,
                                      (positions[streams[stream][0]] - streams[stream][1]) % n],),
                device_id_type=pl.DeviceIdType.MESH)
        for stream in range(len(streams)):
            pl.semaphore_wait(credits.at[stream], 2)


# ============================================================================
# MLA attention layer (sequence-parallel)
# ============================================================================
#
# Projections, flash attention, output gate, projection ring, residual add.


_PROJECTION_NAMES = (
    "query_down_projection", "query_up_projection",
    "key_value_down_projection", "key_value_up_projection",
    "output_gate_projection", "query_normalization",
    "key_value_normalization",
)
ProjectionWeights = namedtuple("ProjectionWeights", _PROJECTION_NAMES)  # pyrefly: ignore[bad-class-definition]


def make_mla_attention_layer_sp_ring(mesh, *, seq_len, segments, prefix_capacity, token_ranks=None,
                                     collective_id=31, flash_tile=128, dsa=None, rope_probe=False):
    """One MLA layer over a packed window of ``segments`` sequences.

    ``run(hidden, prefix, weights, segment_desc, length, arena)``:
    ``hidden`` is the gathered window ([T, 7168] on every rank), ``prefix``
    the rank's residual rows, ``segment_desc`` the ``[5 * segments]``
    descriptor (see ``_flash_kernel_packed``) and ``arena`` the layer's
    ``CpPrefix`` (at most ``prefix_capacity`` prefix tokens). Returns the
    new residual rows and the rank-leading ``[32, T, 576]`` compressed KV
    (identical on every rank). ``output_projection_repacked`` is [384,7168]
    per rank: the rank's attention features over the full hidden width, built
    from the chip pair's decode shards at load time
    (``rank_loader._sequence_parallel_output``).

    With ``dsa`` (Kimi-K3.1's ``DsaConfig``), ``weights`` also hold the
    indexer (``dsa.STAGE_DSA_FAMILIES``, every head) and ``run`` takes
    ``index_cache``: the decode-layout index-key cache (``[blocks, 24, P +
    32, 128]`` uint8, read only) and the layer's slot in it. Every rank keys
    the whole window (``dsa_prefill.index_keys``), scores and selects its
    own ``T / 32`` rows (``dsa_prefill_select.select_words``) and the words
    are exchanged (``gather_words``) for the flash kernel; two more outputs,
    the window's key bytes ``[32, T, 128]`` and exponents ``[32, T]`` uint8,
    zero from ``length`` on. With ``KernelConfig.mla_sparse_prefill``
    ``rows``, each rank instead attends its own rows at every head over their
    selected tokens (``mla_sparse_prefill.attend_rows``: no word exchange, no
    expansion of the arena) and an all-to-all returns its heads' rows.
    """
    if not segments:
        raise ValueError("the MLA layer runs packed windows")
    n = check_mesh(mesh, "ring MLA")
    check_window(seq_len, "ring MLA", n)
    tables = mesh_ring_tables(mesh, token_ranks)
    local_tokens = seq_len // n
    stripe = window_stripe(local_tokens)

    def selection(local_weights, full_hidden, query_latent, window_keys, segment_desc, length, ids, local,
                  order, page, index_cache, own_rows=False):
        """The flash kernel's ``select``/``dense`` for Kimi-K3.1 (see the docstring); with ``own_rows``, this
        rank's rows' words (``[T / 32, W]``, not exchanged) for the sparse attention."""
        from vllm_torchtpu.megakernel.kimi_k3.kernels import (
            dsa_cache,
            dsa_prefill,
            dsa_prefill_select,
        )

        topk = dsa.index_topk
        desc = segment_desc[:5 * segments]
        dense = jnp.all(desc[4 * segments:] + desc[segments:2 * segments] <= topk)
        cache, slot = index_cache
        rows_per_page = page + dsa_cache.SCALE_ROWS
        cache = jax.lax.dynamic_index_in_dim(cache, slot[0], axis=1, keepdims=False)
        index_pages, _ = gathered_prefix_pages(
            cache.reshape(-1, rows_per_page // 4, 4, 128), ids, local, tables, "k3_sp",
            count=segments, capacity=prefix_capacity, page_tokens=page)
        positions = sp_token_index(jax.lax.axis_index("k3_sp"), jnp.arange(local_tokens), stripe, n)
        q, weights = dsa_prefill.index_query(query_latent[positions], local_weights["dsa_wq"],
                                             full_hidden[positions], local_weights["dsa_wv"])

        def selected(_):
            words = dsa_prefill_select.select_words(
                q, weights, window_keys, index_pages.reshape(-1, rows_per_page, 128), order, desc, length,
                positions, count=segments, capacity=prefix_capacity, topk=topk)
            if own_rows:
                return words
            return dsa_prefill_select.gather_words(words, desc, count=segments, capacity=prefix_capacity,
                                                   tokens=seq_len, ranks=n, stripe=stripe)

        if _config().dsa_prefill_select == "xla":
            def reference(_):
                held = index_pages.reshape(-1, rows_per_page, 128)[order[1:]]
                arena_keys = dsa_cache.dequantize_index_keys(
                    held[:, :page].reshape(prefix_capacity, 128),
                    held[:, page:page + page // 128].reshape(prefix_capacity))
                arena_keys = jnp.where((jnp.arange(prefix_capacity) < order[0] * page)[:, None], arena_keys, 0)
                valid = dsa_prefill.candidate_valid(desc, length, count=segments, tokens=seq_len,
                                                    capacity=prefix_capacity, rows=positions)
                scores = dsa_prefill.candidate_scores(q, weights, jnp.concatenate((arena_keys, window_keys)))
                words = dsa_prefill.pack_bits(dsa_prefill.select_mask(scores, valid, topk))
                if own_rows:
                    return words
                words = jax.lax.all_gather(words, "k3_sp")
                return words.reshape(n, local_tokens // stripe, stripe, -1).swapaxes(0, 1).reshape(seq_len, -1)

            def dense_rows(_):
                return jnp.zeros((local_tokens if own_rows else seq_len, (prefix_capacity + seq_len) // 32),
                                 jnp.uint32)

            return dict(select=jax.lax.cond(dense, dense_rows, reference, None),
                        dense=dense.astype(jnp.int32).reshape(1), select_transposed=False)

        def skipped(_):
            shape = (dsa_prefill_select.word_columns(prefix_capacity, seq_len), seq_len)
            return jnp.zeros((local_tokens, shape[0]) if own_rows else shape, jnp.uint32)

        return dict(select=jax.lax.cond(dense, skipped, selected, None),
                    dense=dense.astype(jnp.int32).reshape(1), select_transposed=not own_rows)

    def exchanged(select, segment_desc, capacity=prefix_capacity):
        """``selection(own_rows=True)``'s words in the flash kernel's layout
        (every window row's, exchanged as ``selection`` does without
        ``own_rows``), for an arena of ``capacity`` tokens (at most the
        program's: the words of the arena below it, then the window's)."""
        from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_prefill_select

        words = select["select"]
        if capacity < prefix_capacity:
            words = jnp.concatenate((words[:, :capacity // 32], words[:, prefix_capacity // 32:]), axis=1)
        if _config().dsa_prefill_select == "xla":
            words = jax.lax.all_gather(words, "k3_sp")
            words = words.reshape(n, local_tokens // stripe, stripe, -1).swapaxes(0, 1).reshape(seq_len, -1)
            return dict(select, select=words, select_transposed=False)
        words = dsa_prefill_select.gather_words(words, segment_desc[:5 * segments], count=segments,
                                                capacity=capacity, tokens=seq_len, ranks=n, stripe=stripe)
        return dict(select, select=words, select_transposed=True)

    def local_packed_cp(full_hidden, weights, segment_desc, length, pool, ids, local, *index_cache):
        """One rank's view: this rank's heads over every rank's prefix
        pages and the window."""
        local_weights = jax.tree.map(lambda value: value[0], weights)
        projections = ProjectionWeights(
            *(local_weights[name] for name in _PROJECTION_NAMES)
        )
        # mla_sparse_prefill "rows": this rank's rows at every head (every head's up-projections are loaded).
        rows = dsa is not None and _config().mla_sparse_prefill == "rows"
        query_latent, latent = mla_latents(full_hidden, projections)
        output_gate = _dot(full_hidden, projections.output_gate_projection).astype(jnp.bfloat16)
        tokens = full_hidden.shape[0]
        real = (jnp.arange(tokens) < length.reshape(()))[None, :, None]
        pages, order = gathered_prefix_pages(pool, ids[0], local[0], tables, "k3_sp", count=segments,
                                             capacity=prefix_capacity,
                                             flat_extra=mla_sparse_prefill.table_extra(tokens) if rows else None)
        select, keys = {}, ()
        if dsa is not None:
            from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache, dsa_prefill

            key_bytes, exponents = dsa_prefill.index_keys(
                full_hidden, latent, local_weights["dsa_wk"], local_weights["dsa_knorm"],
                rope_probe=rope_probe, eps=dsa.norm_eps)
            select = selection(local_weights, full_hidden, query_latent,
                               dsa_cache.dequantize_index_keys(key_bytes, exponents), segment_desc, length,
                               ids[0], local[0], order, pool.shape[-3] * pool.shape[-2], index_cache,
                               own_rows=rows)
            filled = jnp.arange(tokens) < length.reshape(())
            keys = (jnp.where(filled[:, None], key_bytes, 0)[None], jnp.where(filled, exponents, 0)[None])
        def masked(pages, projections, select, arena_order=order):
            query, new_key, new_value = mla_heads(query_latent, latent, projections, query_scale=SCALE_LOG2)
            new_key = jnp.where(real, new_key.transpose(1, 0, 2), 0)
            new_value = jnp.where(real, new_value.transpose(1, 0, 2), 0)
            attended = mla_flash_prefill_packed(
                query.transpose(1, 0, 2), new_key, new_value, length, segment_desc[:5 * segments],
                count=segments, tile=flash_tile,
                arena=expand_prefix_pages(pages, arena_order, projections.key_value_up_projection), **select)
            return attended.transpose(1, 0, 2).reshape(tokens, -1)

        def sparse(_):
            positions = sp_token_index(jax.lax.axis_index("k3_sp"), jnp.arange(local_tokens), stripe, n)
            words = mla_sparse_prefill.selection_words(
                select["select"], select["dense"], segment_desc[:5 * segments], length, count=segments,
                capacity=prefix_capacity, tokens=tokens, transposed=False, rows=positions)
            own = (_dot(query_latent[positions], projections.query_up_projection) * SCALE_LOG2).astype(jnp.bfloat16)
            block = min(_config().mla_sparse_block, local_tokens)
            compact = "sparse_core" if jax.default_backend() == "tpu" else "xla"
            fp8 = _config().mla_sparse_fp8_gather
            if block > 1:
                mine = mla_sparse_prefill.attend_blocks(
                    own.reshape(local_tokens, -1, 256), jnp.where(real[0], latent, 0),
                    projections.key_value_up_projection, pages, order, words, capacity=prefix_capacity,
                    gather=lambda table, index, end: ragged_gather_v2(table, index, jnp.int32(0), end),
                    runs=segments * n, block=block, compact=compact, fp8_gather=fp8)
            else:
                mine = mla_sparse_prefill.attend_rows(
                    own.reshape(local_tokens, -1, 256), jnp.where(real[0], latent, 0),
                    projections.key_value_up_projection, pages, order, words, capacity=prefix_capacity,
                    gather=lambda table, index, end: ragged_gather_v2(table, index, jnp.int32(0), end),
                    runs=segments * n, compact=compact, fp8_gather=fp8,
                    groups=min(_config().mla_sparse_groups, local_tokens // 2),
                    split=_config().mla_sparse_query_split)
            # Back to this rank's heads at every window row (the reduce-scatter's layout).
            heads = mine.shape[1] // n
            mine = jax.lax.all_to_all(mine.reshape(local_tokens, n, heads * 128).transpose(1, 0, 2), "k3_sp", 0, 0)
            attended = mine.reshape(n, local_tokens // stripe, stripe, heads * 128).transpose(1, 0, 2, 3)
            return attended.reshape(tokens, heads * 128)

        dense_below = _config().mla_sparse_dense_below if seq_len <= 2048 else _config().mla_sparse_dense_below_wide
        if rows and dense_below:
            # Short contexts: the masked flash at this rank's heads over the whole context costs less than the
            # token gathers (both exact over the same selection).
            def dense(_):
                width = projections.key_value_up_projection.shape[-1] // n
                first = jax.lax.axis_index("k3_sp") * width
                mine = projections._replace(**{
                    name: jax.lax.dynamic_slice_in_dim(getattr(projections, name), first, width, axis=1)
                    for name in ("query_up_projection", "key_value_up_projection")})
                # The branch serves windows whose every sequence ends within dense_below: their prefixes, laid
                # end to end (a page of rounding each), fit `room`, so its pages, arena and exchanged words are
                # sized by it, not by the program's capacity (at 1M tokens those would be several GiB of
                # temporaries). The pages are gathered again as pages: viewing the flat token rows as pages copies
                # the whole table, and its layout then slows the sparse path's gathers.
                page = pool.shape[-3] * pool.shape[-2]
                room = min(prefix_capacity, -(-segments * (dense_below + page) // 4096) * 4096)
                share = min(ids.shape[-1] - 1, -(-(room // n + segments * page) // page))  # a rank's pages of it
                held, held_order = gathered_prefix_pages(pool, ids[0][:1 + share], local[0], tables, "k3_sp",
                                                         count=segments, capacity=room)
                return masked(held, mine, exchanged(select, segment_desc, capacity=room), arena_order=held_order)

            desc = segment_desc[:5 * segments]
            context = jnp.max(desc[4 * segments:] + desc[segments:2 * segments])
            attended = jax.lax.cond(context <= dense_below, dense, sparse, None)
        elif rows:
            attended = sparse(None)
        else:
            attended = masked(pages, projections, select)
        gate = jax.nn.sigmoid(output_gate.astype(jnp.float32)).astype(jnp.bfloat16)
        return (attended * gate).astype(jnp.bfloat16)[None], latent[None], *keys

    def run(hidden, prefix, weights, segment_desc, length, arena, index_cache=()):
        (pool, ) = arena.pools
        if (dsa is None) != (not index_cache):
            raise ValueError("the slice MLA layer takes an index-key cache exactly with dsa")
        gated, latent, *keys = jax.shard_map(
            local_packed_cp, mesh=mesh,
            in_specs=(P(), P("k3_sp"), P(), P(), P("k3_sp"), P("k3_sp"), P("k3_sp"))
            + ((P("k3_sp"), P()) if dsa is not None else ()),
            out_specs=(P("k3_sp"),) * (2 if dsa is None else 4),
            check_vma=False,
        )(hidden, weights, segment_desc, length, pool, arena.ids, arena.local, *index_cache)
        projected = project_reduce_scatter(
            gated, weights["output_projection_repacked"], mesh, tables,
            collective_id=collective_id,
        ).reshape(n, local_tokens, 7168)
        # Masked residual add inside a shard_map: an auto-sharded iota over the
        # rank axis needs the partitioner's partition-id, which runtimes that
        # only accept manual SPMD programs (torch_tpu's jax_op) reject.
        def residual(prefix, projected, length):
            rank = jax.lax.axis_index("k3_sp")
            real = sp_token_index(
                rank, jnp.arange(local_tokens), window_stripe(local_tokens), n) < length[0]
            return (prefix[0] + jnp.where(real[:, None], projected[0], 0))[None]

        output = jax.shard_map(
            residual, mesh=mesh, in_specs=(P("k3_sp"), P("k3_sp"), P()),
            out_specs=P("k3_sp"), check_vma=False,
        )(prefix, projected, length)
        return output.astype(jnp.bfloat16), latent, *keys

    return jax.jit(run)


# ============================================================================
# MoE reduce-scatter ring
# ============================================================================
#
# The ``kernels/kimi_k3/collectives.py`` ring (whose wrapper is BF16 only) with
# an FP32 or BF16 running sum; FP32 avoids rounding the partials at every hop.


def ring_reduce_scatter_fp32(partials, mesh, tables, *, collective_id=43, dtype=jnp.float32):
    """Reduce [n,T,D] partials (BF16 or FP32) to [n,T/n,D] token shards in
    row tiles of at most 256; the running sum is ``dtype`` (FP32, or BF16)
    at every hop."""
    n = check_mesh(mesh, "FP32 ring")
    if partials.ndim != 3 or partials.shape[0] != n or partials.dtype not in (jnp.float32, jnp.bfloat16):
        raise ValueError("FP32 ring needs [n,T,D] BF16 or FP32 partials")
    tokens, width = partials.shape[1:]
    if tokens % n:
        raise ValueError("FP32 ring needs whole local rows")
    local_rows = tokens // n
    rows = min(local_rows, 256)
    regions = _regions(rows, width)
    streams, buffers = len(regions), 2
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    def local(part):
        scratch = (
            pltpu.SemaphoreType.DMA((streams,)),
            pltpu.SemaphoreType.DMA((streams,)),
            pltpu.SemaphoreType.DMA((buffers, streams)),
            pltpu.SemaphoreType.REGULAR((streams,)),
            pltpu.SemaphoreType.DMA,
            pltpu.VMEM((rows, width), partials.dtype),  # local rows, as given
            pltpu.VMEM((rows, width), dtype),
            pltpu.VMEM((buffers, rows, width), dtype),
        )
        call = pl.pallas_call(
            functools.partial(
                _reduce_scatter_kernel, streams=regions,
                buffers=buffers,
            ),
            out_shape=jax.ShapeDtypeStruct((local_rows, width), dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(local_rows // rows, n), in_specs=(hbm,), out_specs=hbm,
                scratch_shapes=scratch,
            ),
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                vmem_limit_bytes=4 * rows * width * 4 + 8 * 1024**2,
            ),
            name="kimi_prefill_fp32_ring_rs",
        )
        return call(jnp.asarray(tables, jnp.int32), part[0])

    return jax.shard_map(
        local, mesh=mesh,
        in_specs=P("k3_sp", None, None),
        out_specs=P("k3_sp", None), check_vma=False,
    )(partials)


# ============================================================================
# Routed MoE (expert-major)
# ============================================================================
#
# Each EP shard runs its experts over the whole window; a ring sums the partials.


def _counting_sort_positions(keys, buckets):
    """Stable pair-to-expert-major positions using the MXU rank pass."""
    size = keys.shape[0]
    block = min(512, size)
    if buckets > block:
        raise ValueError("counting-sort bucket count exceeds its MXU block")
    pad = -size % block
    if pad:
        keys = jnp.pad(keys, (0, pad), constant_values=buckets - 1)
    onehot = jax.nn.one_hot(
        keys.reshape(-1, block), buckets, dtype=jnp.bfloat16
    )
    lower = jnp.concatenate((
        jnp.ones((1, block), jnp.bfloat16),
        jnp.tril(jnp.ones((block, block), jnp.bfloat16), -1),
    ), axis=0)
    both = jnp.einsum(
        "ij,bje->bie", lower, onehot, preferred_element_type=jnp.float32
    )
    hist = both[:, 0, :]
    rank = both[:, 1:, :]
    counts = hist.sum(axis=0)
    base = jnp.cumsum(hist, axis=0) - hist + jnp.cumsum(counts) - counts
    positions = ((base[:, None, :] + rank) * onehot).sum(axis=2)
    return positions.reshape(-1)[:size].astype(jnp.int32), counts.astype(jnp.int32)


def expert_family_slot(family_count, slot):
    """Map a global MoE slot (0..91) to its index in a resident expert bank.

    Expert weights are banked by execution group (``_EXPERT_BANK_SIZES`` in
    ``kimi_k3/prefill.py``): eight layers per regular group (slot ``3 + 8 i +
    j`` -> ``j``) and one for the final layer (-> 0). Any other count, the
    first bank (slots 0..2) or a complete 92-layer family, takes the global
    slot. Dense (non-expert) families are not banked.
    """
    if family_count == 8:
        return (slot - 3) % 8
    if family_count == 1:
        return jnp.zeros_like(slot)
    return slot


def routed_experts(
    latent,
    topk_ids,
    topk_probabilities,
    expert_gate_up,
    expert_gate_up_scales,
    expert_down,
    expert_down_scales,
    *,
    first_expert,
    expert_layer=None,
):
    """Compute one EP shard's routed contribution for all tokens.

    `latent` is [T, K] BF16. `topk_ids` and normalized probabilities are
    [T, topk]. Expert tensors are the K3 K-major packed MXFP4 layout. Every
    selected pair occupies one sorted row; nonlocal pairs sort to the end and
    the GMM skips them using dynamic group sizes. The result is [T, N] BF16.
    If expert_layer is supplied, packed weights keep a leading layer axis,
    and so may their E8M0 scales. No capacity limit or route drop is applied.
    The SparseCore moves the rows: ``ragged_gather_v2`` into expert-major
    order, ``ragged_gather_reduce`` back, weighting and summing each token's
    top-k contributions in FP32.
    """
    if expert_layer is not None:
        expert_layer = expert_family_slot(expert_gate_up.shape[0], expert_layer)
    tokens, input_width = latent.shape
    local_experts = expert_gate_up.shape[-3]
    if topk_ids.shape != topk_probabilities.shape or topk_ids.shape[0] != tokens:
        raise ValueError("topk ids and probabilities must have shape [T, topk]")
    if expert_gate_up.shape[-2] * 8 != input_width:
        raise ValueError("latent width must match the packed gate/up contraction")
    if expert_gate_up.shape[-1] != 2 * expert_down.shape[-2] * 8:
        raise ValueError("gate/up and down intermediate widths disagree")

    topk = topk_ids.shape[1]
    local_ids = topk_ids.astype(jnp.int32) - jnp.asarray(first_expert, jnp.int32)
    valid = (local_ids >= 0) & (local_ids < local_experts) & (topk_probabilities != 0)
    keys = jnp.where(valid, local_ids, local_experts).reshape(-1)
    token_id = jnp.broadcast_to(
        jnp.arange(tokens, dtype=jnp.int32)[:, None], (tokens, topk)
    ).reshape(-1)
    token_bits = max(1, (tokens - 1).bit_length())
    if (local_experts + 1).bit_length() + token_bits >= 31:
        raise ValueError("expert and token ids do not fit the packed sort key")
    # Sorting one packed integer gives both an expert-major order and its
    # source token without gathering a second large permutation vector.
    packed = (keys << token_bits) | token_id
    order = jnp.argsort(packed, stable=False)
    sorted_tokens = token_id[order]
    inverse, all_counts = _counting_sort_positions(keys, local_experts + 1)
    counts = all_counts[:local_experts]
    x = ragged_gather_v2(
        latent, sorted_tokens, jnp.array([0], jnp.int32),
        jnp.sum(counts, dtype=jnp.int32)[None],
    )

    # The E8M0 exponent bytes go to the GMM as they are (it converts each
    # tile in VMEM): an FP32 [E, blocks, 1, N] copy pads 8x in HBM.
    gu_scale = expert_gate_up_scales.astype(jnp.uint8)
    down_scale = expert_down_scales.astype(jnp.uint8)
    # gmm_v2 splits [gate, up] and interleaves lanes for the fused SiTU
    # activation (expert_activation).
    middle = gmm_v2(
        x, expert_gate_up, counts, rhs_scale=gu_scale, rhs_layer=expert_layer,
        rhs_quant_dtype=jnp.float4_e2m1fn,
        fuse_act="situ:4:25",
    )
    down = gmm_v2(
        middle, expert_down, counts, rhs_scale=down_scale, rhs_layer=expert_layer,
        rhs_quant_dtype=jnp.float4_e2m1fn,
    )

    # Restore the token-major pair order, mask the unwritten nonlocal GMM rows,
    # and reduce the top-k contributions per token in FP32.
    return ragged_gather_reduce(
        down, inverse, topk_probabilities.reshape(-1),
        valid.reshape(-1), reduce_group_size=topk,
    )


def _moe_latent_all_gather(latent, tables):
    """Gather BF16 SP latents in mesh-rank order over the ring (n ranks).

    The gather's owners are mesh ranks, independent of the routed-output
    ring's token ownership permutation.
    """
    rows, width = latent.shape
    n = tables.shape[-1]
    regions = _regions(rows, width)
    streams, buffers = len(regions), 2
    return pl.pallas_call(
        functools.partial(_all_gather_kernel, regions=regions, buffers=buffers),
        out_shape=jax.ShapeDtypeStruct((n * rows, width), latent.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1, n),
            in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(
                pltpu.SemaphoreType.DMA((streams,)),
                pltpu.SemaphoreType.DMA((streams,)),
                pltpu.SemaphoreType.DMA((buffers, streams)),
                pltpu.SemaphoreType.REGULAR((streams,)),
                pltpu.SemaphoreType.DMA((streams,)),
                pltpu.VMEM((rows, width), latent.dtype),
                pltpu.VMEM((buffers, rows, width), latent.dtype),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=41,
            vmem_limit_bytes=3 * rows * width * 2 + 8 * 1024**2,
        ),
        name="kimi_prefill_moe_latent_ag",
    )(jnp.asarray(tables, jnp.int32), latent)


def moe_local_routed(hidden_local, length, weights, *, axis_name, gather_tables):
    """This rank's routed-expert partial for every token of the window, and
    its FP32 half of the TP2 shared expert over its chip's rows (reduced
    across the chip pair in the tail).

    Routing is sequence-parallel (each rank routes its own ``T/n`` rows);
    the latent, ids and probabilities are then gathered so that each EP
    shard runs its experts over all tokens (``routed_experts``). The caller's
    reduce-scatter ring sums the partials. ``latent_down`` is [7168,4096] on
    every rank; EP x TP2 experts: 56 local experts at 32 ranks (EP16), 224 in
    a stage of 8 (EP4).
    """
    rank = jax.lax.axis_index(axis_name)
    n = gather_tables.shape[-1]
    local_tokens, hidden_width = hidden_local.shape
    if hidden_width != 7168:
        raise ValueError("K3 sequence-parallel MoE expects [T/n,7168]")
    if weights["latent_down"].shape[-2:] != (7168, 4096):
        raise ValueError("latent projection weights must use replicated SP layout")
    local_experts = weights["expert_gu"].shape[-3]
    if local_experts * n != 896 * 2:
        raise ValueError("K3 SP MoE requires EP x TP2 expert weights")
    expert_tp = n * local_experts // 896
    tokens = local_tokens * n

    logits = jax.lax.dot_general(
        hidden_local, weights["router"], (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    raw = jax.nn.sigmoid(logits)
    _, local_ids = jax.lax.top_k(raw + weights["router_bias"], 16)
    local_probs = jnp.take_along_axis(raw, local_ids, axis=1)
    local_probs /= jnp.maximum(local_probs.sum(axis=1, keepdims=True), 1e-30)
    valid_rows = sp_token_index(
        rank, jnp.arange(local_tokens), window_stripe(local_tokens), n) < length.reshape(())
    local_probs = jnp.where(valid_rows[:, None], local_probs, 0)

    if weights["latent_down"].ndim == 3:  # the resident family: select the layer in the kernel
        latent_local = gmm_v2(
            hidden_local, weights["latent_down"][:, None], jnp.full((1,), local_tokens, jnp.int32),
            rhs_layer=weights["expert_layer"])
    else:
        latent_local = hidden_local @ weights["latent_down"]
    latent_all = _moe_latent_all_gather(latent_local[:, :3584], gather_tables)
    # One all-gather for ids and probability bits.
    routes = jnp.concatenate(
        (local_ids.astype(jnp.int32), jax.lax.bitcast_convert_type(local_probs, jnp.int32)), axis=1)
    routes = jax.lax.all_gather(routes, axis_name).reshape(tokens, 32)
    ids_all = routes[:, :16]
    probs_all = jax.lax.bitcast_convert_type(routes[:, 16:], jnp.float32)
    routed = routed_experts(
        latent_all, ids_all, probs_all,
        weights["expert_gu"], weights["expert_gus"],
        weights["expert_down"], weights["expert_ds"],
        first_expert=(rank // expert_tp) * local_experts,
        expert_layer=weights.get("expert_layer"),
    )
    shared = _moe_sp_shared_partial(_moe_sp_shared_rows(hidden_local, axis_name, n), weights)
    return routed, shared


def shared_pairs(n):
    """The chip-local TP2 groups of an n-rank mesh."""
    return tuple((2 * i, 2 * i + 1) for i in range(n // 2))


SHARED_PAIRS = shared_pairs(32)


def _moe_sp_shared_rows(hidden_local, axis_name, n=32):
    """Both cores' token rows of this chip, for the TP2 shared expert."""
    local_tokens = hidden_local.shape[0]
    return jax.lax.all_gather(hidden_local, axis_name, axis_index_groups=shared_pairs(n)).reshape(
        2 * local_tokens, 7168)


def _moe_sp_shared_partial(chip_rows, weights):
    """This core's half of the shared expert over the chip's rows: FP32
    partials for _moe_sp_shared_reduce. Pure matmuls (no collectives), so
    it can overlap SparseCore work."""
    # One-group GMMs: the shared expert's M (the chip's rows) is small, and
    # the GMM keeps each weight tile on the MXU for all of it, with the
    # activation fused on the FP32 accumulator.
    rows = jnp.full((1,), chip_rows.shape[0], jnp.int32)
    # Resident layer families ([L, ...]): the kernel selects the layer (a
    # sliced operand would be copied).
    family = weights["shared_gu"].ndim == 3
    layer = dict(rhs_layer=weights["expert_layer"]) if family else {}

    def one_group(w):
        return w[:, None] if family else w[None]

    shared_mid = gmm_v2(chip_rows, one_group(weights["shared_gu"]), rows,
                        fuse_act="situ:4:25", **layer)
    return gmm_v2(shared_mid, one_group(weights["shared_down"]), rows,
                  preferred_element_type=jnp.float32, **layer)


def _moe_sp_shared_reduce(partial, axis_name, n=32):
    return jax.lax.psum_scatter(
        partial, axis_name, scatter_dimension=0, tiled=True, axis_index_groups=shared_pairs(n),
    ).astype(jnp.bfloat16)


def _moe_sp_tail(routed, shared, prefix_local, length, weights, *,
                 axis_name, eps, shared_reduced=False, n=32):
    """Finish the routed branch from local reduced MoE rows and add the
    shared expert's rows."""
    rank = jax.lax.axis_index(axis_name)
    local_tokens = prefix_local.shape[0]
    valid_rows = sp_token_index(
        rank, jnp.arange(local_tokens), window_stripe(local_tokens), n) < length.reshape(())
    routed = routed.astype(jnp.float32)
    routed *= jax.lax.rsqrt(jnp.mean(routed*routed, axis=1, keepdims=True)+eps)
    routed = (routed * weights["latent_norm"][:, :3584]).astype(jnp.bfloat16)
    up = routed @ weights["latent_up"]
    if not shared_reduced:
        shared = _moe_sp_shared_reduce(shared, axis_name, n)
    output = (up + shared).astype(jnp.bfloat16)
    output = jnp.where(valid_rows[:, None], output, 0)
    return (prefix_local + output).astype(jnp.bfloat16)


def make_moe_fused_sp_repacked(mesh, *, seq_len, eps=1e-5, collective_id=42,
                               token_ranks=None):
    """Build the TP32 MoE layer for repacked projection weights: the local
    routed partials, their reduce-scatter ring, then the latent-up and
    shared-expert tail on each rank's rows."""
    n = check_mesh(mesh, "MoE Pallas ring reduction")
    check_window(seq_len, "MoE Pallas ring reduction", n)
    axis_name = mesh.axis_names[0]
    tables = mesh_ring_tables(mesh, token_ranks)
    gather_tables = mesh_ring_tables(mesh)
    local_tokens = seq_len // n

    def local_partial(hidden, prefix, weights, length):
        local_weights = jax.tree.map(lambda value: value[0], weights)
        routed, shared = moe_local_routed(
            hidden[0], length, local_weights, axis_name=axis_name,
            gather_tables=gather_tables,
        )
        return routed[None], shared[None]

    partial_op = jax.jit(jax.shard_map(
        local_partial, mesh=mesh,
        in_specs=(P(axis_name), P(axis_name), P(axis_name), P()),
        out_specs=(P(axis_name), P(axis_name)), check_vma=False,
    ))

    def local_tail(reduced, shared, prefix, weights, length):
        local_weights = jax.tree.map(lambda value: value[0], weights)
        return _moe_sp_tail(
            reduced[0], shared[0], prefix[0], length, local_weights,
            axis_name=axis_name, eps=eps, n=n,
        )[None]

    tail_op = jax.jit(jax.shard_map(
        local_tail, mesh=mesh,
        in_specs=(P(axis_name), P(axis_name), P(axis_name),
                  P(axis_name), P()),
        out_specs=P(axis_name), check_vma=False,
    ))

    def run_ring(hidden, prefix, weights, length):
        partials, shared = partial_op(hidden, prefix, weights, length)
        reduced = ring_reduce_scatter_fp32(
            partials, mesh, tables,
            collective_id=collective_id, dtype=jnp.dtype(_config().moe_reduce_dtype),
        )
        reduced = reduced.reshape(n, local_tokens, 3584)
        return tail_op(reduced, shared, prefix, weights, length)

    return jax.jit(run_ring)


# ============================================================================
# MLA weight families
# ============================================================================
#
# Each family holds every MLA layer's copy; a runtime slot picks the layer.


MLA_FAMILIES = (
    "m_qa", "m_qb", "m_ka", "m_kb", "m_gate", "m_qnorm", "m_knorm",
    "m_o",
)
MLA_FIELD_NAMES = (
    "query_down_projection", "query_up_projection",
    "key_value_down_projection", "key_value_up_projection",
    "output_gate_projection", "query_normalization",
    "key_value_normalization", "output_projection_repacked",
)


def make_indexed_mla_layer_sp_ring(mesh, *, seq_len=8192, segments, prefix_capacity, flash_tile=256, dsa=None,
                                   rope_probe=False):
    """The MLA layer at MLA slot ``slot`` of the family-major weights:
    ``run(hidden, prefix, families, segment_desc, length, slot, arena,
    index_cache)`` (``arena``: the ``CpPrefix`` of the slot's pool; with
    ``dsa`` the families hold the indexer's too and ``index_cache`` is the
    index-key cache, read at the slot: see make_mla_attention_layer_sp_ring)."""
    from vllm_torchtpu.megakernel.kimi_k3.dsa import STAGE_DSA_FAMILIES

    base = make_mla_attention_layer_sp_ring(
        mesh, seq_len=seq_len, flash_tile=flash_tile, segments=segments, prefix_capacity=prefix_capacity,
        dsa=dsa, rope_probe=rope_probe,
    )
    fields = tuple(zip(MLA_FAMILIES, MLA_FIELD_NAMES)) + (
        () if dsa is None else tuple(zip(STAGE_DSA_FAMILIES, STAGE_DSA_FAMILIES)))

    def select(families, slot):
        return {
            field: jax.lax.dynamic_index_in_dim(
                families[name], slot[0], axis=1, keepdims=False,
            )
            for name, field in fields
        }

    def run(hidden, prefix, families, segment_desc, length, slot, arena, index_cache=()):
        index = () if not index_cache else (index_cache[0], slot)
        return base(hidden, prefix, select(families, slot), segment_desc, length, arena, index)

    return jax.jit(run)


# ============================================================================
# MoE weight families
# ============================================================================
#
# Each family holds every MoE layer's copy; a runtime slot picks the layer.


MOE_FAMILIES = (
    "router", "router_bias", "latent_down", "latent_up", "latent_norm",
    "shared_gu", "shared_down", "expert_gu", "expert_gus",
    "expert_down", "expert_ds",
)


def make_indexed_moe_sp_ring(mesh, *, seq_len=8192):
    """Select a MoE layer, keeping packed expert families resident for tile DMA."""
    from ..config import active

    if active().moe_reduce_dtype == "float32":
        if seq_len in (512, 1024, 2048, 4096, 8192):
            from .prefill_moe.fused import make_indexed_moe_sp_ring as make_fused_moe

            return make_fused_moe(mesh, seq_len=seq_len)
    elif seq_len in (4096, 8192):
        # The per-hop BF16 reduction.
        from .prefill_moe.layer import make_indexed_moe_sp_ring as make_packed_moe

        return make_packed_moe(mesh, seq_len=seq_len)
    return _make_indexed_moe_sp_ring_reference(mesh, seq_len=seq_len)


def _make_indexed_moe_sp_ring_reference(mesh, *, seq_len=8192):
    """SparseCore gather/combine path, for the windows the fused and packed
    MoEs do not cover (BF16 reduction below 4K)."""
    base = make_moe_fused_sp_repacked(mesh, seq_len=seq_len)

    def run(hidden, prefix, families, length, slot):
        selected = {
            name: (
                families[name] if name in ("expert_gu", "expert_down", "expert_gus", "expert_ds", "latent_down",
                                           "shared_gu", "shared_down")
                else jax.lax.dynamic_index_in_dim(
                    families[name], slot[0], axis=1, keepdims=False,
                )
            )
            for name in MOE_FAMILIES
        }
        selected["expert_layer"] = jnp.broadcast_to(slot[None], (mesh.size, 1))
        return base(hidden, prefix, selected, length)

    return jax.jit(run)


# ============================================================================
# Norm + all-gather + KDA
# ============================================================================
#
# Mixture, RMSNorm and window all-gather in one Pallas call, optionally with KDA.


def _norm_gather_body(
    active_ref, tables_ref, prefix_ref, blocks_ref, folded_ref, norm_ref,
    normalized_ref, gathered_ref, *scratch, regions, eps, mix_tile,
):
    mix_scratch = scratch[:12]
    ring_scratch = scratch[12:]
    @pl.when((pl.program_id(0) == 0) & (pl.program_id(1) == 0))
    def normalize():
        mixture_norm_tiled_body(
            active_ref, prefix_ref, blocks_ref, folded_ref, norm_ref,
            normalized_ref, *mix_scratch, tile=mix_tile, eps=eps,
        )

    _all_gather_kernel(
        tables_ref, normalized_ref, gathered_ref, *ring_scratch,
        regions=regions, buffers=2, striped=True,
    )


def make_norm_gather_sp(mesh, *, seq_len, token_ranks=None,
                        collective_id=37, eps=1e-5):
    """Return ``(normalized [n,T/n,H], gathered [T,H])``.

    The first result is token-sharded, and the second is replicated. The
    caller supplies rank-leading prefix, snapshots, and norm weights. Ranks
    gather their rows in tiles of at most 256.
    """
    n = check_mesh(mesh, "norm/gather")
    check_window(seq_len, "fused norm/gather", n)
    rows, width = seq_len // n, 7168
    tile = min(rows, 256)
    tables = jnp.asarray(mesh_ring_tables(mesh, token_ranks), jnp.int32)
    regions = _regions(tile, width)
    streams = len(regions)
    mix_scratch = mixture_norm_tiled_scratch_shapes(width, tile)
    ring_scratch = (
        pltpu.SemaphoreType.DMA((streams,)),
        pltpu.SemaphoreType.DMA((streams,)),
        pltpu.SemaphoreType.DMA((2, streams)),
        pltpu.SemaphoreType.REGULAR((streams,)),
        pltpu.SemaphoreType.DMA((streams,)),
        pltpu.VMEM((tile, width), jnp.bfloat16),
        pltpu.VMEM((2, tile, width), jnp.bfloat16),
    )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    call = pl.pallas_call(
        functools.partial(_norm_gather_body, regions=regions, eps=eps,
                          mix_tile=tile),
        out_shape=(
            jax.ShapeDtypeStruct((rows, width), jnp.bfloat16),
            jax.ShapeDtypeStruct((seq_len, width), jnp.bfloat16),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(rows // tile, n),
            in_specs=(hbm, hbm, hbm, hbm),
            out_specs=(hbm, hbm),
            scratch_shapes=(*mix_scratch, *ring_scratch),
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id, vmem_limit_bytes=64 * 1024**2,
        ),
        name="kimi_prefill_norm_gather_sp",
    )

    def local(prefix, blocks, folded, norm, active):
        normalized, gathered = call(
            active, tables, prefix[0], blocks[0], folded[0], norm[0],
        )
        return normalized[None], gathered

    return jax.jit(jax.shard_map(
        local, mesh=mesh,
        in_specs=(P("k3_sp"), P("k3_sp"), P("k3_sp"), P("k3_sp"), P()),
        out_specs=(P("k3_sp"), P()), check_vma=False,
    ))


def _kda_layer_body(
    slot_ref, segment_ref, active_ref, tables_ref, length_ref,
    prefix_ref, blocks_ref, folded_ref, norm_ref,
    gate_family_ref, fb_family_ref, proj_family_ref, conv_family_ref,
    alog_family_ref, dt_family_ref, kda_norm_family_ref,
    rec_in_ref, hist_in_ref, output_family_ref,
    normalized_ref, gathered_ref, attended_ref, rec_out_ref, hist_out_ref,
    final_ref, rec_checkpoint_ref, hist_checkpoint_ref, partial_ref,
    pool, *scratch, eps, count, blocks,
):
    """One KDA layer of the family-major weights (``slot_ref``) over a
    packed window of ``count`` sequences, pipelined over its ``blocks`` SP
    blocks on half rings: grid step ``17 i + j`` runs ring step j of block
    i's window all-gather, a share of block ``i - 1``'s KDA tiles (with
    their output-projection partials) and ring step j of block ``i - 2``'s
    reduce-scatter, so both rings run under the delta rule. Steps b < blocks
    first apply the mixture and RMSNorm to block b's local rows.
    ``segment_ref`` is the SMEM ``[3 * count]`` descriptor (see
    ``kda_prefill_tiled_body``). KDA slot ``9k`` is layer ``12k``, the first
    layer of a residual block, whose attention output replaces the partial
    sum instead of adding to it."""
    slot = slot_ref[0]
    step = pl.program_id(1)
    phase, ring_step = step // HALF_STEPS, step % HALF_STEPS
    stripe = window_stripe(prefix_ref.shape[0])
    mix_small, gather_sems = scratch[:8], scratch[8:11]
    reduce_sems, projection_scratch = scratch[11:16], scratch[16:19]
    gate_w_vmem, kda_scratch = scratch[19], scratch[20:]

    def views(shapes):  # the shared pool: the mixture at step 0, then the projection ring
        offset, result = 0, []
        for shape, dtype in shapes:
            size = pool_alias.storage_size(shape, jnp.dtype(dtype))
            result.append(pool_alias.view(
                pool.at[pl.ds(offset // 128, size // 128), :], shape, dtype))
            offset += size
        return result

    pl.when(step == 0)(lambda: _ring_enter(tables_ref, jax.lax.axis_index("k3_sp")))

    @pl.when(step < blocks)  # only block 0's mixture delays the first ring
    def normalize():
        rows = pl.ds(step * stripe, stripe)
        mix = views(_kda_mix_shapes(stripe))
        mixture_norm_tiled_body(
            active_ref, prefix_ref.at[rows], blocks_ref.at[:, rows], folded_ref, norm_ref,
            normalized_ref.at[rows, :7168], *mix, *mix_small,
            tile=stripe, eps=eps,
        )

        # Every TP rank holds the whole gate compression weight, so instead of
        # all 32 ranks projecting the gathered window, each projects its own
        # rows (still in mix[2]) and appends the 128 columns to its normalized
        # rows: they travel the half rings with the hidden state.
        @pl.when(step == 0)
        def load_gate():
            # Semaphore 0 of the KDA weight loads, which the tile body leaves unused.
            copy = pltpu.make_async_copy(gate_family_ref.at[slot], gate_w_vmem, kda_scratch[-1].at[0])
            copy.start()
            copy.wait()

        @functools.partial(pl.run_scoped, features=pltpu.VMEM((stripe, 128), jnp.bfloat16))
        def exchange_features(features):
            features[...] = _dot(mix[2][0], gate_w_vmem[...]).astype(jnp.bfloat16)
            # Before the first ring step reads these rows, and before the pool is reused.
            copy = pltpu.make_async_copy(features, normalized_ref.at[rows, 7168:], gather_sems[2])
            copy.start()
            copy.wait()

    # One call site for the KDA tile: inlined twice, the layer would no
    # longer fit the instruction memory.
    block_tiles = 32 * stripe // 128
    attending = (phase >= 1) & (phase <= blocks)
    first_tile = jnp.where(
        attending, (phase - 1) * block_tiles + ring_step * block_tiles // HALF_STEPS, 0)
    end_tile = jnp.where(
        attending, (phase - 1) * block_tiles + (ring_step + 1) * block_tiles // HALF_STEPS, 0)
    kda_prefill_tiled_body(
        gathered_ref, fb_family_ref.at[slot],
        proj_family_ref.at[slot], conv_family_ref.at[slot],
        alog_family_ref.at[slot], dt_family_ref.at[slot],
        kda_norm_family_ref.at[slot], rec_in_ref, hist_in_ref,
        attended_ref, rec_out_ref, hist_out_ref, *kda_scratch,
        tile=128, decay_lower_bound=-5.0, epsilon=eps,
        checkpoint=(rec_checkpoint_ref, hist_checkpoint_ref),
        segments=(segment_ref, count),
        tiles=(first_tile, end_tile, step == HALF_STEPS,
               step == HALF_STEPS * (blocks + 1) - 1),
        block_tiles=block_tiles,
        projection=(output_family_ref.at[slot], *projection_scratch[:1], partial_ref,
                    *projection_scratch[1:]),
    )

    # The three bodies of a step touch different SP blocks (gather: block
    # ``phase``, KDA: ``phase - 1``, reduce: ``phase - 2``) with no data
    # dependency between them; issuing the KDA tiles first keeps them from
    # queuing behind the rings' DMA issue and waits.
    @pl.when(phase < blocks)
    def gather():
        _half_gather_step(tables_ref, normalized_ref, gathered_ref, *gather_sems,
                          step=ring_step, block=phase)

    @pl.when(phase >= 2)
    def reduce():
        _half_reduce_step(
            tables_ref, length_ref, partial_ref, prefix_ref, final_ref,
            *reduce_sems, *views(_kda_reduce_shapes(stripe)),
            step=ring_step, block=phase - 2, replace=slot % 9 == 0,
        )


def _kda_mix_shapes(mix_tile):
    return (
        ((2, mix_tile, 7168), jnp.bfloat16),
        ((mix_tile, 7168), jnp.bfloat16),
        ((2, mix_tile, 7168), jnp.bfloat16),
        ((mix_tile, 7168), jnp.float32),
    )


def _kda_reduce_shapes(stripe, groups=()):
    return (
        ((2, 2, *groups, stripe, 7168), jnp.bfloat16),  # the local partials (slot, direction)
        ((2, stripe, 7168), jnp.bfloat16),  # payload (direction)
        ((2, 2, stripe, 7168), jnp.bfloat16),  # received (slot, direction)
    )


def make_norm_gather_kda_sp(mesh, *, seq_len, segments, token_ranks=None,
                            collective_id=38, eps=1e-5):
    """One KDA layer as one Pallas call (``_kda_layer_body``): the local
    residual mixture and RMSNorm, then per SP block the 32-rank all-gather
    of the window, KDA over the packed window (``segments`` sequences, their
    states leading with the sequence: ``[1, segments, ...]`` per rank) and
    the output projection ring with the residual add, pipelined so that the
    rings run under the delta rule.

    ``run(prefix, blocks, folded, norm, families, rec, hist, active, length,
    slot, segment_desc)`` returns ``(normalized, gathered, attended, rec,
    hist, output, rec_at, hist_at)``: the ``*_at`` states are each
    sequence's at its checkpoint position.
    """
    if mesh.size != 32 or tuple(mesh.axis_names) != ("k3_sp",):
        raise ValueError("norm/gather/KDA requires a flat k3_sp TP32 mesh")
    check_window(seq_len, "fused norm/gather/KDA")
    if not segments:
        raise ValueError("the KDA layer runs packed windows")
    state_lead = (segments, )
    rows, width = seq_len // 32, 7168
    stripe = window_stripe(rows)
    blocks = rows // stripe
    if token_ranks is None:
        token_ranks = tuple(range(32))
    tables = jnp.asarray(
        ring_tables(list(mesh.devices.flat), token_ranks), jnp.int32,
    )
    streams = len(_half_ring_streams(width))
    pool_bytes = max(
        sum(pool_alias.storage_size(shape, jnp.dtype(dtype))
            for shape, dtype in _kda_mix_shapes(stripe)),
        sum(pool_alias.storage_size(shape, jnp.dtype(dtype))
            for shape, dtype in _kda_reduce_shapes(stripe)),
    )
    shapes = (
        jax.ShapeDtypeStruct((128, 384), jnp.bfloat16),
        jax.ShapeDtypeStruct((width, 1664), jnp.bfloat16),
        jax.ShapeDtypeStruct((4, 3, 3, 128), jnp.float32),
        jax.ShapeDtypeStruct((3, 1), jnp.float32),
        jax.ShapeDtypeStruct((3, 128), jnp.float32),
        jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )
    scratch = (
        pltpu.VMEM((pool_bytes // 128, 128), jnp.uint8),
        *mixture_norm_tiled_scratch_shapes(width, stripe)[4:],
        pltpu.SemaphoreType.DMA((streams,)),  # the window all-gather
        pltpu.SemaphoreType.DMA((streams,)),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA((2, streams)),  # the reduce-scatter
        pltpu.SemaphoreType.DMA((streams,)),
        pltpu.SemaphoreType.DMA((2, streams)),
        pltpu.SemaphoreType.REGULAR((streams,)),
        pltpu.SemaphoreType.DMA,
        pltpu.VMEM((384, width), jnp.bfloat16),  # the output projection
        pltpu.VMEM((2, 128, width), jnp.bfloat16),  # its partials
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.VMEM((width, 128), jnp.bfloat16),  # the gate compression weight
        *kda_prefill_tiled_scratch_shapes(width + 128, 128, *shapes),
    )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    output_shapes = (
        # normalized and gathered rows, with the exchanged gate compression appended
        jax.ShapeDtypeStruct((rows, width + 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq_len, width + 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq_len, 384), jnp.bfloat16),
        jax.ShapeDtypeStruct((*state_lead, 3, 128, 128), jnp.float32),
        jax.ShapeDtypeStruct((*state_lead, 3, 3, 3, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, width), jnp.bfloat16),
        jax.ShapeDtypeStruct((*state_lead, 3, 128, 128), jnp.float32),
        jax.ShapeDtypeStruct((*state_lead, 3, 3, 3, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq_len, width), jnp.bfloat16),  # projection partials
    )
    call = pl.pallas_call(
        functools.partial(
            _kda_layer_body, count=segments, eps=eps, blocks=blocks,
        ),
        out_shape=output_shapes,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(1, HALF_STEPS * (blocks + 2)),
            in_specs=(hbm,) * 14,
            out_specs=(hbm,) * len(output_shapes),
            scratch_shapes=scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id, vmem_limit_bytes=64 * 1024**2,
        ),
        name="kimi_prefill_norm_gather_kda_sp",
    )

    def local(prefix, blocks, folded, norm, weights, rec, hist, active, length,
              slot, segment_desc):
        w = jax.tree.map(lambda value: value[0], weights)
        outputs = call(
            slot, segment_desc, active, tables, length,
            prefix[0], blocks[0], folded[0], norm[0],
            w["k_gate"], w["k_fb"], w["k_projection"], w["k_conv"],
            w["k_a_log"], w["k_dt"], w["k_norm"], rec[0], hist[0], w["k_o"],
        )
        normalized, gathered, attended, new_rec, new_hist, final, rec_at, hist_at, _ = outputs
        return (normalized[None, :, :width], gathered[:, :width], attended[None], new_rec[None],
                new_hist[None], final[None], rec_at[None], hist_at[None])

    return jax.jit(jax.shard_map(
        local, mesh=mesh,
        in_specs=(P("k3_sp"),) * 7 + (P(), P(), P(), P()),
        out_specs=(P("k3_sp"), P(), P("k3_sp"), P("k3_sp"), P("k3_sp"),
                   P("k3_sp"), P("k3_sp"), P("k3_sp")),
        check_vma=False,
    ))


# ============================================================================
# Dense FFN layer
# ============================================================================
#
# Layer 0's dense FFN, TP32 over the intermediate dimension.


def make_dense_ffn_sp(mesh, *, seq_len):
    """Layer-0 dense FFN over local rows with a TP output reduction."""
    n = check_mesh(mesh, "dense FFN SP")
    if seq_len % n:
        raise ValueError("dense FFN SP requires T divisible by the ranks")
    axis_name = mesh.axis_names[0]
    local_tokens = seq_len // n
    shard = 1152 * (32 // n)  # the TP32 shards (padded 1056-column slices), concatenated

    def local(hidden, prefix, gate_up, down, length):
        x = hidden[0]
        if (x.shape != (local_tokens, 7168)
                or gate_up[0].shape != (7168, 2 * shard)
                or down[0].shape != (shard, 7168)):
            raise ValueError("dense FFN weights must use K3 TP shapes")
        # Rows are sequence-parallel (``sp_token_index`` stripes) while the
        # weights are TP32 over the intermediate dimension, so every rank
        # applies its shard to every rank's rows (gathered in rank order) and
        # the partial outputs are reduce-scattered back to their owners.
        x = jax.lax.all_gather(x, axis_name, tiled=True)
        gu = jax.lax.dot_general(
            x, gate_up[0], (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        gate, value = jnp.split(gu, 2, axis=1)
        gate, value = gate.astype(jnp.float32), value.astype(jnp.float32)
        middle = expert_activation(gate, value).astype(jnp.bfloat16)
        partial = jax.lax.dot_general(
            middle, down[0], (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        reduced = jax.lax.psum_scatter(
            partial, axis_name, scatter_dimension=0, tiled=True,
        ).astype(jnp.bfloat16)
        rank = jax.lax.axis_index(axis_name)
        valid = sp_token_index(rank, jnp.arange(local_tokens), window_stripe(local_tokens), n) < length[0]
        return (prefix[0] + jnp.where(valid[:, None], reduced, 0))[None]

    return jax.jit(jax.shard_map(
        local, mesh=mesh,
        in_specs=(P(axis_name), P(axis_name), P(axis_name), P(axis_name), P()),
        out_specs=P(axis_name), check_vma=False,
    ))


# ============================================================================
# Four-layer groups
# ============================================================================
#
# One straight-line group program reused through a runtime group index:
# SparseCore row movement does not lower inside a dynamic JAX loop.


def make_regular_group4_sp_indexed(mesh, *, seq_len=8192, segments, prefix_capacity, mla_flash_tile=256, dsa=None,
                                   rope_probe=False):
    """One runtime-indexed group at layers 4g..4g+3 (g = 1..22) over a
    packed window: three KDA layers and one MLA layer, each followed by its
    MoE. ``run(prefix, blocks, common, kda_families, mla_families,
    moe_families, kda_state, mla_state, length, group, segment_desc,
    arena, index_cache)``: see make_prefill_stack_sp_indexed; ``arena`` is the
    ``CpPrefix`` of the group's MLA pool.
    """
    if mesh.size != 32 or tuple(mesh.axis_names) != ("k3_sp",):
        raise ValueError("indexed four-layer group requires flat k3_sp TP32")
    check_window(seq_len, "indexed four-layer group")
    sharding = NamedSharding(mesh, P("k3_sp"))
    mixture = make_mixture_norm_sp(mesh, seq_len=seq_len)
    norm_gather = make_norm_gather_sp(mesh, seq_len=seq_len)
    kda = make_norm_gather_kda_sp(mesh, seq_len=seq_len, segments=segments)
    kda_outputs = (3, 4, 6, 7)  # rec, hist, rec_at, hist_at in the KDA op's result
    mla = make_indexed_mla_layer_sp_ring(
        mesh, seq_len=seq_len, flash_tile=mla_flash_tile, segments=segments,
        prefix_capacity=prefix_capacity, dsa=dsa, rope_probe=rope_probe,
    )
    mla_states = 1 if dsa is None else 3  # latents (, key bytes, exponents)
    moe = make_indexed_moe_sp_ring(mesh, seq_len=seq_len)

    def choose(family, slot):
        return jax.lax.dynamic_index_in_dim(
            family, slot, axis=1, keepdims=False,
        )

    def put_state(family, state, slot):
        return jax.lax.dynamic_update_slice(
            family, state[:, None],
            (0, slot, *([0] * (family.ndim - 2))),
        )

    def scalar(value):
        return jnp.asarray([value], jnp.int32)

    def run(prefix, blocks, common, kda_families, mla_families,
            moe_families, kda_state, mla_state, length, group, segment_desc, arena, index_cache=()):
        base = group[0] * 4
        kda_states = tuple(kda_state)
        if len(kda_states) != len(kda_outputs):
            raise ValueError("kda_state is (rec, hist, rec_at, hist_at)")
        mla_cache = tuple(mla_state)
        if len(mla_cache) != mla_states:
            raise ValueError(f"mla_state holds {mla_states} families")
        blocks = jax.lax.cond(
            base % 12 == 0,
            lambda x: x.at[:, base // 12].set(prefix),
            lambda x: x,
            blocks,
        )
        tap = prefix
        for offset in range(4):
            index = base + offset
            if offset == 2:
                tap = prefix  # entering layer base + 2 (DSpark aux layer 90 in group 22)
            attn_res = choose(common["attn_res"], index)
            attn_norm = choose(common["attn_norm"], index)
            attn_active = scalar((index - 1) // 12 + 1)
            if offset < 3:
                slot = base - group[0] + offset
                result = kda(
                    prefix, blocks, attn_res, attn_norm, kda_families,
                    choose(kda_states[0], slot), choose(kda_states[1], slot),
                    attn_active, length, scalar(slot), segment_desc,
                )
                after_attention = result[5]
                kda_states = tuple(put_state(family, result[i], slot)
                                   for family, i in zip(kda_states, kda_outputs, strict=True))
            else:
                slot = group[0]
                _, gathered = norm_gather(
                    prefix, blocks, attn_res, attn_norm, attn_active,
                )
                after_attention, *updates = mla(
                    gathered, prefix, mla_families, segment_desc, length, scalar(slot), arena,
                    index_cache,
                )
                mla_cache = tuple(
                    put_state(family, update, slot)
                    for family, update in zip(mla_cache, updates, strict=True))
            ffn_input = mixture(
                after_attention, blocks, choose(common["mlp_res"], index),
                choose(common["ffn_norm"], index),
                scalar(index // 12 + 1),
            )
            prefix = moe(
                ffn_input, after_attention, moe_families,
                length, scalar(index - 1),
            )
        return prefix, blocks, kda_states, mla_cache, tap

    return jax.jit(run, out_shardings=(
        sharding, sharding, (sharding,) * len(kda_outputs), (sharding,) * mla_states, sharding,
    ))


def make_regular_group8_sp_indexed(mesh, *, seq_len=8192, segments, prefix_capacity, mla_flash_tile=256, dsa=None,
                                   rope_probe=False):
    """Two consecutive regular four-layer groups as one executable
    (``pair`` p runs groups 2p + 1 and 2p + 2; ``arena`` carries both
    groups' MLA pools, in order). Also returns the prefix between the two
    groups: the residual stream entering layer 8p + 12 (pair 10's is the
    DSpark draft's aux layer 88)."""
    group4 = make_regular_group4_sp_indexed(
        mesh, seq_len=seq_len, mla_flash_tile=mla_flash_tile, segments=segments,
        prefix_capacity=prefix_capacity, dsa=dsa, rope_probe=rope_probe,
    )
    mla_states = 1 if dsa is None else 3
    sharding = NamedSharding(mesh, P("k3_sp"))

    def run(prefix, blocks, common, kda_families, mla_families,
            moe_families, kda_state, mla_state, length, pair, segment_desc, arena, index_cache=()):
        first = jnp.asarray([2 * pair[0] + 1], jnp.int32)
        second = first + 1
        arenas = arena.split()
        middle, blocks, kda_state, mla_state, _ = group4(
            prefix, blocks, common, kda_families, mla_families,
            moe_families, kda_state, mla_state, length, first,
            segment_desc, arenas[0], index_cache,
        )
        prefix, blocks, kda_state, mla_state, tap = group4(
            middle, blocks, common, kda_families, mla_families,
            moe_families, kda_state, mla_state, length, second,
            segment_desc, arenas[1], index_cache,
        )
        return prefix, blocks, kda_state, mla_state, middle, tap

    return jax.jit(run, out_shardings=(
        sharding, sharding, (sharding,) * 4, (sharding,) * mla_states, sharding, sharding,
    ))


def make_final_mla_sp_indexed(mesh, *, seq_len=8192, segments, prefix_capacity, mla_flash_tile=256, dsa=None,
                              rope_probe=False):
    """K3's last layer (92: MLA, then MoE) over a packed window:
    ``run(prefix, blocks, common, mla_families, moe_families, mla_state,
    length, segment_desc, arena)``; see make_prefill_stack_sp_indexed."""
    if mesh.size != 32 or tuple(mesh.axis_names) != ("k3_sp",):
        raise ValueError("final MLA layer requires flat k3_sp TP32")
    check_window(seq_len, "final MLA layer")
    sharding = NamedSharding(mesh, P("k3_sp"))
    mixture = make_mixture_norm_sp(mesh, seq_len=seq_len)
    norm_gather = make_norm_gather_sp(mesh, seq_len=seq_len)
    mla = make_indexed_mla_layer_sp_ring(
        mesh, seq_len=seq_len, flash_tile=mla_flash_tile, segments=segments,
        prefix_capacity=prefix_capacity, dsa=dsa, rope_probe=rope_probe,
    )
    mla_states = 1 if dsa is None else 3
    moe = make_indexed_moe_sp_ring(mesh, seq_len=seq_len)

    def run(prefix, blocks, common, mla_families, moe_families,
            mla_state, length, segment_desc, arena, index_cache=()):
        if len(mla_state) != mla_states:
            raise ValueError(f"mla_state holds {mla_states} families")
        attn_res = common["attn_res"][:, 92]
        attn_norm = common["attn_norm"][:, 92]
        active = jnp.asarray([8], jnp.int32)
        _, gathered = norm_gather(
            prefix, blocks, attn_res, attn_norm, active,
        )
        after_attention, *updates = mla(
            gathered, prefix, mla_families, segment_desc, length,
            jnp.asarray([23], jnp.int32), arena, index_cache,
        )
        mla_cache = tuple(
            jax.lax.dynamic_update_slice(
                family, update[:, None], (0, 23, *([0] * (family.ndim - 2))))
            for family, update in zip(mla_state, updates, strict=True))
        ffn_input = mixture(
            after_attention, blocks, common["mlp_res"][:, 92],
            common["ffn_norm"][:, 92], active,
        )
        prefix = moe(
            ffn_input, after_attention, moe_families,
            length, jnp.asarray([91], jnp.int32),
        )
        return prefix, blocks, mla_cache

    return jax.jit(run, out_shardings=(
        sharding, sharding, (sharding,) * mla_states,
    ))


# ============================================================================
# Layer stack
# ============================================================================
#
# Layer 0's KDA and dense FFN, then a ``fori_loop`` picking KDA or MLA per layer.


def make_prefill_stack_sp_indexed(mesh, *, seq_len=8192, depth=93, segments, prefix_capacity,
                                  mla_flash_tile=256, unroll=False, tap_layer=None, dsa=None, rope_probe=False):
    """Layers ``0 .. depth - 1`` over a packed window of ``segments``
    sequences, on family-major K3 weights (the prefill runs the first four
    here, then ``make_regular_group8_sp_indexed`` and
    ``make_final_mla_sp_indexed``).

    ``run(hidden, common, kda_families, mla_families, moe_families,
    kda_state, mla_state, length, segment_desc, arena)``:
    ``segment_desc`` is the ``[3 * segments]`` descriptor (offsets, lengths,
    checkpoint positions; see kda_prefill_tiled_body), ``length`` the end of
    the last sequence and ``arena`` the ``CpPrefix`` of the MLA pools. The
    KDA state is ``(rec, hist, rec_at, hist_at)``, leading with the sequence
    (``[32, 69, segments, ...]``); ``rec_at``/``hist_at`` receive each
    layer's state at each sequence's checkpoint. The MLA state is
    ``(latents,)``, ``[32, 24, T, 576]``: each MLA layer's compressed KV.
    ``tap_layer`` (``1 .. depth - 1``): also return the residual stream
    entering that layer (the MLA DSpark draft's aux layer 3).
    """
    if mesh.size != 32 or tuple(mesh.axis_names) != ("k3_sp",):
        raise ValueError("indexed prefill stack requires flat k3_sp TP32 mesh")
    if tap_layer is not None and not 1 <= tap_layer < depth:
        raise ValueError("tap_layer must be one of the stack's later layers")
    check_window(seq_len, "indexed prefill stack")
    if not 2 <= depth <= 93:
        raise ValueError("indexed prefill stack needs depth 2..93")
    sharding = NamedSharding(mesh, P("k3_sp"))
    mixture = make_mixture_norm_sp(mesh, seq_len=seq_len)
    norm_gather = make_norm_gather_sp(mesh, seq_len=seq_len)
    kda = make_norm_gather_kda_sp(mesh, seq_len=seq_len, segments=segments)
    mla = make_indexed_mla_layer_sp_ring(
        mesh, seq_len=seq_len, flash_tile=mla_flash_tile, segments=segments,
        prefix_capacity=prefix_capacity, dsa=dsa, rope_probe=rope_probe,
    )
    mla_states = 1 if dsa is None else 3
    moe = make_indexed_moe_sp_ring(mesh, seq_len=seq_len)
    dense = make_dense_ffn_sp(mesh, seq_len=seq_len)

    def choose(family, slot):
        return jax.lax.dynamic_index_in_dim(
            family, slot, axis=1, keepdims=False,
        )

    def put_state(family, state, slot):
        return jax.lax.dynamic_update_slice(
            family, state[:, None],
            (0, slot, *([0] * (family.ndim - 2))),
        )

    def active(value):
        return jnp.asarray([value], jnp.int32)

    kda_outputs = (3, 4, 6, 7)  # rec, hist, rec_at, hist_at in the KDA op's result

    def run(hidden, common, kda_families, mla_families, moe_families,
            kda_state, mla_state, length, segment_desc, arena, index_cache=()):
        kda_state = tuple(kda_state)
        if len(kda_state) != len(kda_outputs):
            raise ValueError("kda_state is (rec, hist, rec_at, hist_at)")
        if len(mla_state) != mla_states:
            raise ValueError(f"mla_state holds {mla_states} families")

        def run_kda(prefix, blocks, attn_res, attn_norm, states, attn_active, slot):
            result = kda(
                prefix, blocks, attn_res, attn_norm, kda_families,
                choose(states[0], slot), choose(states[1], slot),
                attn_active, length, active(slot), segment_desc,
            )
            return result[5], tuple(put_state(family, result[i], slot)
                                    for family, i in zip(states, kda_outputs, strict=True))

        prefix = hidden
        blocks = jnp.zeros((32, 8, seq_len // 32, 7168), jnp.bfloat16)
        blocks = blocks.at[:, 0].set(prefix)
        after_attention, kda_states = run_kda(
            prefix, blocks, choose(common["attn_res"], 0),
            choose(common["attn_norm"], 0), kda_state, active(0), 0,
        )
        ffn_input = mixture(
            after_attention, blocks, choose(common["mlp_res"], 0),
            choose(common["ffn_norm"], 0), active(1),
        )
        prefix = dense(
            ffn_input, after_attention, common["dense_gu"],
            common["dense_down"], length,
        )

        def body(index, carry):
            prefix, blocks, kda_states, mla_cache, tap = carry
            if tap_layer is not None:
                tap = jnp.where(index == tap_layer, prefix, tap)
            blocks = jax.lax.cond(
                index % 12 == 0,
                lambda x: x.at[:, index // 12].set(prefix),
                lambda x: x,
                blocks,
            )
            attn_res = choose(common["attn_res"], index)
            attn_norm = choose(common["attn_norm"], index)
            attn_active = active((index - 1) // 12 + 1)

            def kda_branch(state):
                kda_states, mla_cache = state
                slot = index - (index + 1) // 4
                after, kda_states = run_kda(prefix, blocks, attn_res, attn_norm, kda_states,
                                            attn_active, slot)
                return after, kda_states, mla_cache

            def mla_branch(state):
                kda_states, mla_cache = state
                slot = index // 4
                _, gathered = norm_gather(
                    prefix, blocks, attn_res, attn_norm, attn_active,
                )
                after, *updates = mla(
                    gathered, prefix, mla_families, segment_desc, length, active(slot), arena,
                    index_cache,
                )
                return (
                    after, kda_states,
                    tuple(put_state(family, update, slot)
                          for family, update in zip(mla_cache, updates, strict=True)),
                )

            after_attention, kda_states, mla_cache = jax.lax.cond(
                (index % 4 == 3) | (index == 92), mla_branch, kda_branch,
                (kda_states, mla_cache),
            )
            ffn_input = mixture(
                after_attention, blocks, choose(common["mlp_res"], index),
                choose(common["ffn_norm"], index),
                active(index // 12 + 1),
            )
            prefix = moe(
                ffn_input, after_attention, moe_families,
                length, active(index - 1),
            )
            return prefix, blocks, kda_states, mla_cache, tap

        result = jax.lax.fori_loop(
            1, depth, body,
            (prefix, blocks, kda_states, tuple(mla_state), prefix if tap_layer is not None else None),
            unroll=unroll,
        )
        if tap_layer is not None:
            return (result[0], result[1], result[2], result[3], result[4])
        return (result[0], result[1], result[2], result[3])

    return jax.jit(run, out_shardings=(
        sharding, sharding, (sharding,) * len(kda_outputs), (sharding,) * mla_states,
        *((sharding,) if tap_layer is not None else ()),
    ))


# ============================================================================
# Embedding and logits
# ============================================================================
#
# The vocab-sharded embedding and each packed sequence's first-token logits.


def make_embedding_sp(mesh, *, seq_len):
    """Embed rank-local token IDs using a vocab-sharded table."""
    n = check_mesh(mesh, "embedding SP")
    if seq_len % n:
        raise ValueError("embedding SP requires T divisible by the ranks")
    axis_name = mesh.axis_names[0]
    local_tokens = seq_len // n
    vocab_per_rank = 163840 // n

    def local(tokens, embedding, length):
        rank = jax.lax.axis_index(axis_name)
        token_ids = jax.lax.all_gather(tokens[0], axis_name).reshape(seq_len)
        local_ids = token_ids - rank * vocab_per_rank
        matched = (local_ids >= 0) & (local_ids < vocab_per_rank)
        table = embedding[0]
        if table.shape != (vocab_per_rank, 7168):
            raise ValueError("embedding table has wrong per-rank shape")
        partial = jnp.where(
            matched[:, None], table[jnp.clip(local_ids, 0, vocab_per_rank - 1)], 0
        ).astype(jnp.float32)
        hidden = jax.lax.psum_scatter(
            partial, axis_name, scatter_dimension=0, tiled=True
        ).astype(jnp.bfloat16)
        valid = sp_token_index(rank, jnp.arange(local_tokens), window_stripe(local_tokens), n) < length[0]
        return jnp.where(valid[:, None], hidden, 0)[None]

    return jax.jit(jax.shard_map(
        local, mesh=mesh,
        in_specs=(P(axis_name), P(axis_name), P()),
        out_specs=P(axis_name), check_vma=False,
    ))


def make_prefill_logits_sp(mesh, *, seq_len, layers, segments):
    """Each packed sequence's last-token residual mixture, RMSNorm, and
    vocab-sharded logits: the op takes the ``[3 * segments]`` descriptor
    (offsets, lengths, ...; see kda_prefill_tiled_body) and returns
    ``[segments, V/n]`` per rank (zero for unused slots).
    """
    n = check_mesh(mesh, "logits SP")
    if seq_len % n:
        raise ValueError("logits SP requires T divisible by the ranks")
    if not 2 <= layers <= 93:
        raise ValueError("K3 layer count must be 2..93")
    axis_name = mesh.axis_names[0]
    active = (layers - 1) // 12 + 1
    stripe = window_stripe(seq_len // n)

    def local_packed(prefix, blocks, output_res, final_norm, lm_head, segment_desc):
        rank = jax.lax.axis_index(axis_name)
        rows = []
        for s in range(segments):
            used = segment_desc[segments + s] > 0
            last = jnp.maximum(segment_desc[s] + segment_desc[segments + s] - 1, 0)
            block, within = last // (n * stripe), last % (n * stripe)
            owner, row = within // stripe, block * stripe + within % stripe
            local_prefix = jax.lax.dynamic_slice(prefix[0], (row, 0), (1, 7168))
            local_blocks = jax.lax.dynamic_slice(blocks[0], (0, row, 0), (8, 1, 7168))
            mixed = residual_mixture(local_blocks, local_prefix, output_res[0], active).astype(jnp.float32)
            normalized = (
                mixed * jax.lax.rsqrt(jnp.mean(mixed * mixed, axis=1, keepdims=True) + 1e-5)
                * final_norm[0]
            ).astype(jnp.bfloat16)
            rows.append(jnp.where((rank == owner) & used, normalized, 0))
        hidden = jax.lax.psum(jnp.concatenate(rows, axis=0), axis_name)
        logits = jax.lax.dot_general(
            hidden, lm_head[0], (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        return logits[None]

    return jax.jit(jax.shard_map(
        local_packed, mesh=mesh,
        in_specs=(P(axis_name), P(axis_name), P(axis_name), P(axis_name),
                  P(axis_name), P()),
        out_specs=P(axis_name), check_vma=False,
    ))
