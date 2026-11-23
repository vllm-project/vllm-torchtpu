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
# Source provenance: inferact/k3-megakernel@30c9ba966b
#   src/vllm_torchtpu/megakernel/kimi_k3/kernels/prefill_megakernel.py
# Source provenance: vllm_torchtpu/kernels/megablox/gmm_v2.py@644426d851 (modified copy)
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

The prefill program (``models/kimi_k3/megakernel/prefill.py``) exposes five
kernel builders (running 13 ops per window: first, 11 group ops, and final):

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
  EP16 x TP2 Pallas schedule for every window (512..8192). BF16: reference
  subcore row movement with grouped matmuls and reduce-scatter ring
  (4K/8K windows emit a one-time retirement warning).

The sections run bottom-up: the TPU kernel library (SparseCore gathers,
TP32 rings, the grouped matmul), the K3 building blocks, the layers, then
the groups and the stack. Value-level model math helpers live in
``helpers``. The SparseCore, grouped-matmul and counting-sort code is
adapted from ``vllm_torchtpu.kernels`` and ``vllm_torchtpu.layers`` (see
the header).
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.core.collectives.spmd import mesh_axis_index
from vllm_torchtpu.models.kimi_k3.megakernel import config as _config_mod
from vllm_torchtpu.models.kimi_k3.megakernel.common import pool_alias
from vllm_torchtpu.models.kimi_k3.megakernel.common.constants import (
    AXIS_SP,
    DP2_GROUPS,
    HEAD_DIM,
    HIDDEN_SIZE,
    KDA_HEADS,
    MIB_BYTES,
    NUM_LAYERS,
    OVERFLOW_WINDOW_TOKENS,
    PAGE_TOKENS,
    PREFILL_WINDOWS,
    ROUTED_EXPERTS,
    STORE_TILE,
    SUBLANE_ROWS,
    TP_PEERS,
    TP_SIZE,
    VREG_LANE_COUNT,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels.helpers import (
    e8m0_scale,
    expert_activation,
    kda_decay,
    kda_normalize_query_key,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Protocol

    from jax._src.pallas import core as pl_core

    from vllm_torchtpu.models.kimi_k3.megakernel.common.pool_alias import (
        TransformedRef,
    )
    from vllm_torchtpu.models.kimi_k3.megakernel.common.protocols import (
        AsyncCopyProtocol,
        PallasRefProtocol,
        ReshapableProtocol,
    )

    class _RingDeviceProtocol(Protocol):
        """Device descriptor protocol for TP32 ring topology lookup."""

        id: int
        coords: tuple[int, int, int]
        core_on_chip: int


logger = init_logger(__name__)


def _config() -> _config_mod.KernelConfig:
    """Return the active Kimi-K3 kernel configuration."""
    return _config_mod.active()


# ============================================================================
# SparseCore kernels through core_map
# ============================================================================
#
# ``pl.kernel`` dispatches to ``mpmd_map``; ``core_map_kernel`` calls
# ``core_map`` directly on the given single SparseCore mesh.

_GEN_TRILLIUM = 6
_GEN_IRONWOOD = 7
_PACKED_WORD_BITS = 32
_RAGGED_NDIM_2D = 2
_COL_PARTITION_DOUBLE = 2
_COL_PARTITION_WIDTH_CAP = 1024
_UPPER_BF16_MASK = -65536
_V6_TARGET_BYTES = (256 * 1024) * 0.8
_V7_TARGET_BYTES = (512 * 1024) * 0.8
_DEFAULT_TARGET_BYTES = (128 * 1024) * 0.8


# ============================================================================
# SparseCore ragged gather
# ============================================================================
#
# Rows gathered by index on the SparseCore (``ragged_gather_v2``): the routed
# MoE's move into expert-major order.


def calculate_col_size(hidden_size: int, packing: int) -> int:
    """Compute the largest 128-aligned column divisor fitting local scratch."""
    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core
    if sc_info is None:
        msg = "Subcore info is missing."
        raise ValueError(msg)
    lanes = sc_info.num_lanes

    match tpu_info.generation:
        case int(gen) if gen == _GEN_TRILLIUM:
            target_bytes = _V6_TARGET_BYTES
        case int(gen) if gen == _GEN_IRONWOOD:
            target_bytes = _V7_TARGET_BYTES
        case _:
            target_bytes = _DEFAULT_TARGET_BYTES

    num_buffers = 2
    bytes_per_col = (lanes + lanes // packing) * 4 * num_buffers
    max_safe_col = (int(target_bytes // bytes_per_col) // VREG_LANE_COUNT) * (
        VREG_LANE_COUNT
    )
    start_col = (min(hidden_size, max_safe_col) // VREG_LANE_COUNT) * VREG_LANE_COUNT
    for c in range(start_col, VREG_LANE_COUNT - 1, -VREG_LANE_COUNT):
        if hidden_size % c == 0:
            return c
    return max_safe_col


@dataclasses.dataclass(frozen=True, slots=True)
class _RaggedGatherSpec:
    """Static packing and tile parameters for ``main_kernel_v2``."""

    dtype: jnp.dtype
    dtype_bits: int
    packing: int
    num_simd_lanes: int
    row_subchunk_size: int
    num_phys_cols: int
    unpack_col_chunk: int


def _ragged_gather_col_loop(
    col_base: jax.Array,
    *,
    spec: _RaggedGatherSpec,
    gather_ref: PallasRefProtocol,
    out_ref: PallasRefProtocol,
    idx_rem: jax.Array,
) -> None:
    """Unpack and repack one column chunk of gathered 32-bit words."""
    col_slice = pl.ds(col_base, spec.unpack_col_chunk)
    if spec.packing == 1:
        gather_dt = gather_ref.bitcast(spec.dtype)
        out_dt = out_ref.bitcast(spec.dtype)
        out_dt[:, col_slice] = gather_dt[:, col_slice]
        return
    mask = (1 << spec.dtype_bits) - 1
    shift_multiplier = spec.dtype_bits.bit_length() - 1
    for i in range(spec.num_simd_lanes // spec.packing):
        packed_row = jnp.zeros((1, spec.unpack_col_chunk), dtype=jnp.int32)
        for j in range(spec.packing):
            lane = i * spec.packing + j
            dynamic_shift = jnp.left_shift(idx_rem[lane], shift_multiplier)
            val = jnp.bitwise_right_shift(
                gather_ref[pl.ds(lane, 1), col_slice], dynamic_shift
            )
            val = jnp.bitwise_and(val, mask)
            pack_shift = j * spec.dtype_bits
            packed_row = jnp.bitwise_or(packed_row, jnp.left_shift(val, pack_shift))
        out_ref[pl.ds(i, 1), col_slice] = packed_row


def _ragged_gather_inner_pipeline(
    gather_ref: PallasRefProtocol,
    out_ref: PallasRefProtocol,
    *,
    spec: _RaggedGatherSpec,
    idx_ref: PallasRefProtocol,
) -> None:
    """Gather and repack one row subchunk across all physical column chunks."""
    row_slice = pl.ds(pl.program_id(0) * spec.row_subchunk_size, spec.row_subchunk_size)
    subchunk_idxs = idx_ref[row_slice]
    if spec.packing > 1:
        idx_rem = jnp.bitwise_and(subchunk_idxs, spec.packing - 1)
    else:
        idx_rem = jnp.zeros_like(subchunk_idxs)
    col_loop_fn = functools.partial(
        _ragged_gather_col_loop,
        spec=spec,
        gather_ref=gather_ref,
        out_ref=out_ref,
        idx_rem=idx_rem,
    )
    plsc.parallel_loop(0, spec.num_phys_cols, step=spec.unpack_col_chunk)(col_loop_fn)


def main_kernel_v2(
    start_ref: PallasRefProtocol,
    end_ref: PallasRefProtocol,
    in_hbm_ref: PallasRefProtocol,
    indices_hbm_ref: PallasRefProtocol,
    out_hbm_ref: PallasRefProtocol,
    start_vmem_ref: PallasRefProtocol,
    end_vmem_ref: PallasRefProtocol,
    sem_ref: PallasRefProtocol,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_row_subchunks: int,
) -> None:
    """Gather dynamic index ranges from ``in_hbm_ref`` into ``out_hbm_ref``."""
    sc_info = pltpu.get_tpu_info().sparse_core
    if sc_info is None:
        msg = "Subcore info is missing."
        raise ValueError(msg)
    num_simd_lanes = sc_info.num_lanes
    hidden_size = in_hbm_ref.shape[-1]
    if not isinstance(hidden_size, int):
        msg = f"hidden_size must be int, got {type(hidden_size)}"
        raise TypeError(msg)
    dtype = jnp.dtype(out_hbm_ref.dtype)
    dtype_bits = jax.dtypes.itemsize_bits(dtype)
    packing = _PACKED_WORD_BITS // dtype_bits
    col_size = calculate_col_size(hidden_size, packing)
    if col_size % VREG_LANE_COUNT != 0:
        msg = f"col_size {col_size} must be divisible by {VREG_LANE_COUNT}"
        raise ValueError(msg)

    num_cores = jax.lax.axis_size((core_axis_name, subcore_axis_name))
    row_chunk_size = num_simd_lanes * num_row_subchunks
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
    num_blocks = jnp.where(end <= start, 0, pl.cdiv(end, block_size) - block_start)
    num_cols = pl.cdiv(hidden_size, col_size)
    core_index = lax.axis_index((core_axis_name, subcore_axis_name))
    # SparseCore `.bitcast()` leverages hardware Row-Packing for 16-bit -> 32-bit
    # conversion. The logical row count halves, while physical column dimensions
    # remain unchanged.
    in_hbm_i32 = in_hbm_ref.bitcast(jnp.int32)
    out_hbm_i32 = out_hbm_ref.bitcast(jnp.int32)
    spec = _RaggedGatherSpec(
        dtype=dtype,
        dtype_bits=dtype_bits,
        packing=packing,
        num_simd_lanes=num_simd_lanes,
        row_subchunk_size=num_simd_lanes,
        num_phys_cols=col_size,
        unpack_col_chunk=VREG_LANE_COUNT,
    )

    # The outer pipeline runs on the vector core, moving the integer arithmetic
    # required to decode row-packed indices. This prevents scalar-core instruction
    # starvation during execution of the indirect `in_specs` block lambdas.

    # Upstream TODO(guoweij): The nested `emit_pipeline` design creates a pipeline
    # bubble (DMA wait) when the inner pipeline empties and restarts with new
    # indices. This is a known high-level API limitation, amortized by setting a
    # large num_row_subchunks and shouldn't be an issue for most use cases. We
    # should still monitor the bubble's impact on E2E performance and, if needed,
    # manually reimplement this with primitives to remove the bubble entirely.

    def outer_pipeline(idx_ref: PallasRefProtocol) -> None:
        """Emit inner indirect gather pipeline for one index block."""
        b_global = pl.program_id(0) + block_start
        shift_amount = packing.bit_length() - 1

        def in_index_map(
            r: jax.Array, col_id: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            """Map subchunk and column indices to packed input rows."""
            return (
                jnp.bitwise_right_shift(
                    idx_ref[pl.ds(r * num_simd_lanes, num_simd_lanes)],
                    shift_amount,
                ),
                col_id,
            )

        def out_index_map(
            r: jax.Array, col_id: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            """Map subchunk and column indices to packed output rows."""
            return ((b_global * num_cores + core_index) * num_row_subchunks + r, col_id)

        pltpu.emit_pipeline(
            functools.partial(
                _ragged_gather_inner_pipeline, spec=spec, idx_ref=idx_ref
            ),
            grid=(num_row_subchunks, num_cols),
            in_specs=pl.BlockSpec(
                (pl.Indirect(num_simd_lanes), col_size), in_index_map
            ),
            out_specs=pl.BlockSpec(
                (num_simd_lanes // packing, col_size), out_index_map
            ),
        )(in_hbm_i32, out_hbm_i32)

    def idx_block_map(b: jax.Array) -> tuple[jax.Array]:
        """Select global index chunk for outer block ``b``."""
        return ((b + block_start) * num_cores + core_index,)

    pltpu.emit_pipeline(
        outer_pipeline,
        grid=(num_blocks,),
        in_specs=pl.BlockSpec((row_chunk_size,), idx_block_map),
    )(indices_hbm_ref)


@jax.jit
def ragged_gather_v2(
    x: jax.Array, indices: jax.Array, start: jax.Array, end: jax.Array
) -> jax.Array:
    """Gather rows of ``x`` indexed by ``indices[start:end]``."""
    if x.ndim != _RAGGED_NDIM_2D:
        msg = "Ragged gather only supports 2d inputs."
        raise ValueError(msg)
    if indices.ndim != 1:
        msg = "Ragged gather only supports 1d indices."
        raise ValueError(msg)

    if jnp.isscalar(start):
        start = start[None]
    if jnp.isscalar(end):
        end = end[None]

    dtype = x.dtype
    if dtype not in (jnp.bfloat16, jnp.float32, jnp.int8, jnp.int4):
        msg = f"dtype must be f32, bf16, int8, or int4, but got {dtype}"
        raise ValueError(msg)

    sc_info = pltpu.get_tpu_info().sparse_core
    if sc_info is None:
        return x[indices]

    hidden_size = x.shape[-1]
    out_size = indices.size
    dtype_bits = jax.dtypes.itemsize_bits(dtype)
    packing = _PACKED_WORD_BITS // dtype_bits
    col_size = calculate_col_size(hidden_size, packing)
    aligned_hidden_size = ((hidden_size + col_size - 1) // col_size) * col_size

    num_simd_lanes = sc_info.num_lanes
    num_cores = sc_info.num_cores * sc_info.num_subcores
    base_block_size = num_simd_lanes * num_cores
    num_row_subchunks = max(
        1, min(4, (out_size + base_block_size - 1) // base_block_size)
    )
    row_chunk_size = num_simd_lanes * num_row_subchunks
    block_size = row_chunk_size * num_cores

    out_pad_size = ((out_size + block_size - 1) // block_size) * block_size - out_size
    indices = jnp.pad(indices, ((0, out_pad_size)))

    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=sc_info.num_cores,
        num_subcores=sc_info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    gather_kernel = cast(
        "Callable[..., jax.Array]",
        jax.jit(
            pl.kernel(
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
                    flops=0,
                    transcendentals=0,
                    bytes_accessed=2
                    * out_size
                    * hidden_size
                    * jnp.dtype(dtype).itemsize,
                ),
                name="sc_ragged_gather_v2",
            )
        ),
    )
    return gather_kernel(start, end, x, indices)[:out_size, :hidden_size]


# ============================================================================
# SparseCore ragged gather-reduce
# ============================================================================
#
# The routed MoE's return to token order (``ragged_gather_reduce``): each core
# scatter-adds its destination block's valid routes into FP32 buffers.

_TOKEN_SUBCHUNK = 16
_NUM_TOKEN_SUBCHUNKS = 4
_TOKEN_BLOCK = _TOKEN_SUBCHUNK * _NUM_TOKEN_SUBCHUNKS
_ROUTE_INDEX_BITS = 20
_ROUTE_INDEX_MASK = (1 << _ROUTE_INDEX_BITS) - 1


def _align_to(value: int, alignment: int) -> int:
    """Round ``value`` up to the next multiple of ``alignment``."""
    return pl.cdiv(value, alignment) * alignment


def _fallback_implementation(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    """Gather, weight, mask, and sum routes using dense array operations."""
    out = x[indices] * topk_weights[:, None].astype(jnp.float32)
    out = jnp.where(valid_rows_mask[:, None], out, 0)
    out = out.reshape(-1, reduce_group_size, out.shape[-1])
    return jnp.sum(out, axis=1).astype(x.dtype)


def _calculate_num_column_partitions(
    hidden_size: int, num_cores: int, num_lanes: int
) -> int:
    """Select power-of-two column partition count across available subcores."""
    preferred_num_stages = 4
    num_column_partitions = 1
    while (
        num_cores % (num_column_partitions * _COL_PARTITION_DOUBLE) == 0
        and hidden_size % (num_lanes * num_column_partitions * _COL_PARTITION_DOUBLE)
        == 0
        and hidden_size // (num_column_partitions * _COL_PARTITION_DOUBLE * num_lanes)
        >= preferred_num_stages
    ):
        num_column_partitions *= _COL_PARTITION_DOUBLE
    # Destination-major accumulation keeps a dense FP32 token block in VMEM.
    # For columns up to 1024, using one fewer column partition amortizes the
    # per-block metadata, clear, and output-DMA overhead while still leaving
    # enough room for the double-buffered indirect gather. Wider columns stay
    # with the conservative partitioning selected above.
    if num_column_partitions >= _COL_PARTITION_DOUBLE and hidden_size <= (
        _COL_PARTITION_WIDTH_CAP * (num_column_partitions // _COL_PARTITION_DOUBLE)
    ):
        num_column_partitions //= _COL_PARTITION_DOUBLE
    return num_column_partitions


def partition_counts(hidden_size: int, tpu_info: pltpu.TpuInfo) -> tuple[int, int]:
    """Return ``(column_partitions, row_partitions)`` for ``hidden_size``."""
    sc_info = tpu_info.sparse_core
    if sc_info is None:
        msg = "Subcore info is missing."
        raise ValueError(msg)
    num_cores = sc_info.num_cores * sc_info.num_subcores
    num_column_partitions = _calculate_num_column_partitions(
        hidden_size, num_cores, tpu_info.num_lanes
    )
    return num_column_partitions, num_cores // num_column_partitions


@dataclasses.dataclass(frozen=True, slots=True)
class _RouteScatterCtx:
    """Per-chunk tensors and destination buffer for gather-reduce scatter-add."""

    accum_vmem_ref: PallasRefProtocol
    gather_ref: PallasRefProtocol
    source_indices: jax.Array
    route_destinations: jax.Array
    route_weights: jax.Array
    metadata_start: jax.Array
    valid_route_count: jax.Array


@dataclasses.dataclass(frozen=True, slots=True)
class _GatherReduceBlockCtx:
    """Per-core references and geometry for ``_main_kernel`` token blocks."""

    x_32b_hbm_ref: PallasRefProtocol
    out_32b_hbm_ref: PallasRefProtocol
    accum_vmem_ref: PallasRefProtocol
    out_tile_vmem_ref: PallasRefProtocol
    row_partition_id: jax.Array
    col_partition_id: jax.Array
    blocks_per_row_partition: int
    col_size: int
    col_start: jax.Array


def _extract_weighted_route_batch(
    route_context: _RouteScatterCtx,
    col_slice: pl.Slice,
    route_batch_start: int,
    route_batch_size: int,
) -> list[jax.Array]:
    """Unpack row-packed BF16 lanes to FP32 and apply routing weights."""
    weighted_values: list[jax.Array] = []
    for route_lane in range(route_batch_start, route_batch_start + route_batch_size):
        value_i32 = route_context.gather_ref[route_lane, col_slice]
        shift = jnp.where(
            jnp.bitwise_and(route_context.source_indices[route_lane], 1) == 0, 16, 0
        )
        shifted = jnp.bitwise_and(
            jnp.left_shift(value_i32, shift),
            jnp.int32(_UPPER_BF16_MASK),
        )
        value_f32 = (
            plsc.bitcast(shifted, jnp.float32) * route_context.route_weights[route_lane]
        )
        weighted_values.append(value_f32)
    return weighted_values


def _fixed_two_route_column_step(
    col_offset: jax.Array, *, route_context: _RouteScatterCtx
) -> None:
    """Scatter-add paired adjacent two-route contributions for one column tile."""
    col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
    feature_indices = col_offset + jnp.arange(_TOKEN_SUBCHUNK, dtype=jnp.int32)
    weighted_values = _extract_weighted_route_batch(
        route_context, col_slice, 0, _TOKEN_SUBCHUNK
    )
    accum_target = cast("TransformedRef", route_context.accum_vmem_ref)
    for pair_start in range(0, _TOKEN_SUBCHUNK, 2):
        destination_indices = jnp.full_like(
            feature_indices, route_context.route_destinations[pair_start]
        )
        plsc.addupdate_scatter(
            accum_target,
            (destination_indices, feature_indices),
            weighted_values[pair_start] + weighted_values[pair_start + 1],
        )


def _generic_route_column_step(
    col_offset: jax.Array, *, route_context: _RouteScatterCtx
) -> None:
    """Scatter-add masked variable-count route contributions for one column tile."""
    col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
    feature_indices = col_offset + jnp.arange(_TOKEN_SUBCHUNK, dtype=jnp.int32)
    route_batch_size = 4
    accum_target = cast("TransformedRef", route_context.accum_vmem_ref)
    for route_batch_start in range(0, _TOKEN_SUBCHUNK, route_batch_size):
        weighted_values = _extract_weighted_route_batch(
            route_context, col_slice, route_batch_start, route_batch_size
        )
        for route_batch_lane in range(route_batch_size):
            route_lane = route_batch_start + route_batch_lane
            destination_indices = jnp.full_like(
                feature_indices, route_context.route_destinations[route_lane]
            )
            route_valid = (
                route_context.metadata_start + route_lane
                < route_context.valid_route_count
            )
            scatter_mask = jnp.broadcast_to(route_valid, (_TOKEN_SUBCHUNK,))
            plsc.addupdate_scatter(
                accum_target,
                (destination_indices, feature_indices),
                weighted_values[route_batch_lane],
                mask=scatter_mask,
            )


def _store_gather_reduce_subchunk_pair(
    block_context: _GatherReduceBlockCtx,
    global_block_id: jax.Array,
    output_sem_ref: PallasRefProtocol,
    token_subchunk_pair: int,
) -> None:
    """Pack pairs of FP32 destination rows into BF16 and copy them to HBM."""
    copies: list[AsyncCopyProtocol] = []
    for output_buffer in range(2):
        token_subchunk = token_subchunk_pair * 2 + output_buffer
        accum_row_start = token_subchunk * _TOKEN_SUBCHUNK

        def cast_column(
            col_offset: jax.Array,
            *,
            row_base: int = accum_row_start,
            bank: int = output_buffer,
        ) -> None:
            """Pack 16 FP32 accumulator rows into 8 interleaved BF16 words."""
            col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
            for token_lane in range(0, _TOKEN_SUBCHUNK, 2):
                accum_row = row_base + token_lane
                packed_bf16 = plsc.pack(
                    block_context.accum_vmem_ref[accum_row, col_slice],
                    block_context.accum_vmem_ref[accum_row + 1, col_slice],
                    format=plsc.PackFormat.INTERLEAVED,
                )
                block_context.out_tile_vmem_ref[bank, token_lane // 2, col_slice] = (
                    plsc.bitcast(packed_bf16, jnp.uint32)
                )

        plsc.parallel_loop(0, block_context.col_size, step=_TOKEN_SUBCHUNK)(cast_column)
        output_row = global_block_id * _TOKEN_BLOCK + token_subchunk * _TOKEN_SUBCHUNK
        output_row_packed = pl.multiple_of(output_row // 2, _TOKEN_SUBCHUNK // 2)
        copy = cast(
            "AsyncCopyProtocol",
            pltpu.make_async_copy(
                block_context.out_tile_vmem_ref.at[output_buffer],
                block_context.out_32b_hbm_ref.at[
                    pl.ds(output_row_packed, _TOKEN_SUBCHUNK // 2),
                    pl.ds(block_context.col_start, block_context.col_size),
                ],
                output_sem_ref.at[output_buffer],
            ),
        )
        copy.start()
        copies.append(copy)
    for copy in copies:
        copy.wait()


def _run_gather_reduce_token_block(
    route_metadata_ref: PallasRefProtocol,
    route_weights_ref: PallasRefProtocol,
    route_counts_ref: PallasRefProtocol,
    output_sem_ref: PallasRefProtocol,
    *,
    block_context: _GatherReduceBlockCtx,
) -> None:
    """Accumulate all routes for one 64-token destination block and store BF16."""
    global_block_id = (
        block_context.row_partition_id * block_context.blocks_per_row_partition
        + pl.program_id(0)
    )
    valid_route_count = route_counts_ref[...][0]
    fixed_two_routes = valid_route_count == 2 * _TOKEN_BLOCK
    num_route_chunks = pl.cdiv(valid_route_count, _TOKEN_SUBCHUNK)

    def zero_column(col_offset: jax.Array) -> None:
        """Zero one 16-column strip across all 64 token accumulator rows."""
        col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
        zero = jnp.zeros((_TOKEN_SUBCHUNK,), jnp.float32)
        for token_row in range(_TOKEN_BLOCK):
            block_context.accum_vmem_ref[token_row, col_slice] = zero

    plsc.parallel_loop(0, block_context.col_size, step=_TOKEN_SUBCHUNK)(zero_column)

    def gather_index_map(route_chunk: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Decode packed source row indices for ``route_chunk``."""
        start = route_chunk * _TOKEN_SUBCHUNK
        metadata = route_metadata_ref[pl.ds(start, _TOKEN_SUBCHUNK)]
        source = jnp.bitwise_and(metadata, _ROUTE_INDEX_MASK)
        return (jnp.bitwise_right_shift(source, 1), block_context.col_partition_id)

    @functools.partial(
        pltpu.emit_pipeline,
        grid=(num_route_chunks,),
        in_specs=pl.BlockSpec(
            (pl.Indirect(_TOKEN_SUBCHUNK), block_context.col_size), gather_index_map
        ),
        out_specs=(),
    )
    def route_pipeline(gather_ref: PallasRefProtocol) -> None:
        """Scatter-add one 16-route chunk into the FP32 destination block."""
        metadata_start = pl.program_id(0) * _TOKEN_SUBCHUNK
        metadata_slice = pl.ds(metadata_start, _TOKEN_SUBCHUNK)
        route_metadata = route_metadata_ref[metadata_slice]
        route_context = _RouteScatterCtx(
            accum_vmem_ref=block_context.accum_vmem_ref,
            gather_ref=gather_ref,
            source_indices=jnp.bitwise_and(route_metadata, _ROUTE_INDEX_MASK),
            route_destinations=jnp.bitwise_right_shift(
                route_metadata, _ROUTE_INDEX_BITS
            ),
            route_weights=route_weights_ref[metadata_slice],
            metadata_start=metadata_start,
            valid_route_count=valid_route_count,
        )

        @pl.when(fixed_two_routes)
        def run_fixed_two_route_path() -> None:
            """Run fast paired scatter-add when every token has two routes."""
            plsc.parallel_loop(0, block_context.col_size, step=_TOKEN_SUBCHUNK)(
                functools.partial(
                    _fixed_two_route_column_step, route_context=route_context
                )
            )

        @pl.when(jnp.logical_not(fixed_two_routes))
        def run_generic_path() -> None:
            """Run masked scatter-add for arbitrary route counts."""
            plsc.parallel_loop(0, block_context.col_size, step=_TOKEN_SUBCHUNK)(
                functools.partial(
                    _generic_route_column_step, route_context=route_context
                )
            )

    route_pipeline(block_context.x_32b_hbm_ref)
    for token_subchunk_pair in range(_NUM_TOKEN_SUBCHUNKS // 2):
        _store_gather_reduce_subchunk_pair(
            block_context, global_block_id, output_sem_ref, token_subchunk_pair
        )


def _main_kernel(
    x_hbm_ref: PallasRefProtocol,
    route_metadata_hbm_ref: PallasRefProtocol,
    route_weights_hbm_ref: PallasRefProtocol,
    route_counts_hbm_ref: PallasRefProtocol,
    out_hbm_ref: PallasRefProtocol,
    accum_vmem_ref: PallasRefProtocol,
    out_tile_vmem_ref: PallasRefProtocol,
    sem_ref: PallasRefProtocol,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_column_partitions: int,
    blocks_per_row_partition: int,
    topk: int,
) -> None:
    """Execute destination-major gather-reduce across token blocks."""
    sc_info = pltpu.get_tpu_info().sparse_core
    if sc_info is None or sc_info.num_lanes != _TOKEN_SUBCHUNK:
        msg = f"Expected {_TOKEN_SUBCHUNK} subcore lanes"
        raise ValueError(msg)

    core_id = jax.lax.axis_index((core_axis_name, subcore_axis_name))
    row_partition_id = core_id // num_column_partitions
    col_partition_id = core_id % num_column_partitions
    col_size = accum_vmem_ref.shape[-1]
    metadata_block_size = topk * _TOKEN_BLOCK
    block_context = _GatherReduceBlockCtx(
        x_32b_hbm_ref=x_hbm_ref.bitcast(jnp.int32),
        out_32b_hbm_ref=out_hbm_ref.bitcast(jnp.uint32),
        accum_vmem_ref=accum_vmem_ref,
        out_tile_vmem_ref=out_tile_vmem_ref,
        row_partition_id=row_partition_id,
        col_partition_id=col_partition_id,
        blocks_per_row_partition=blocks_per_row_partition,
        col_size=col_size,
        col_start=col_partition_id * col_size,
    )

    def metadata_block_index(block_id: jax.Array) -> tuple[jax.Array]:
        """Select global token block index for ``block_id``."""
        return (row_partition_id * blocks_per_row_partition + block_id,)

    pltpu.emit_pipeline(
        functools.partial(_run_gather_reduce_token_block, block_context=block_context),
        grid=(blocks_per_row_partition,),
        in_specs=(
            pl.BlockSpec((metadata_block_size,), metadata_block_index),
            pl.BlockSpec((metadata_block_size,), metadata_block_index),
            pl.BlockSpec((_TOKEN_SUBCHUNK,), metadata_block_index),
        ),
        out_specs=(),
    )(
        route_metadata_hbm_ref,
        route_weights_hbm_ref,
        route_counts_hbm_ref,
        scratches=(sem_ref,),
    )


def _pack_gather_reduce_metadata(
    indices_2d: jax.Array,
    weights_2d: jax.Array,
    valid_2d: jax.Array,
    reduce_group_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compact valid routes per 64-token block into packed metadata arrays."""
    num_blocks = indices_2d.shape[0] // _TOKEN_BLOCK
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
    return route_metadata, route_weights, route_counts


@functools.partial(jax.jit, static_argnames=("reduce_group_size",))
def ragged_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    """Gather, weight, and sum routed expert outputs back into token order."""
    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core
    dtype_bytes = jax.dtypes.itemsize_bits(x.dtype) // 8
    if (
        sc_info is None
        or x.dtype != jnp.bfloat16
        or x.shape[-1] % tpu_info.num_lanes != 0
        or jnp.size(x) * dtype_bytes * 2 < tpu_info.vmem_capacity_bytes * 0.6
    ):
        return _fallback_implementation(
            x, indices, topk_weights, valid_rows_mask, reduce_group_size
        )

    input_size = indices.size
    if input_size % reduce_group_size != 0:
        msg = f"{input_size=} must be divisible by {reduce_group_size=}"
        raise ValueError(msg)
    if x.shape[0] % 2:
        x = jnp.pad(x, ((0, 1), (0, 0)))
    num_tokens = input_size // reduce_group_size
    hidden_size = x.shape[-1]
    if x.shape[0] > 1 << _ROUTE_INDEX_BITS:
        msg = (
            f"destination-major gather-reduce supports at most "
            f"{1 << _ROUTE_INDEX_BITS} source rows, got {x.shape[0]}"
        )
        raise ValueError(msg)

    num_column_partitions, num_row_partitions = partition_counts(hidden_size, tpu_info)
    padded_num_tokens = _align_to(num_tokens, num_row_partitions * _TOKEN_BLOCK)
    token_padding = padded_num_tokens - num_tokens
    indices_2d = indices.reshape(num_tokens, reduce_group_size)
    weights_2d = topk_weights.reshape(num_tokens, reduce_group_size)
    valid_2d = valid_rows_mask.reshape(num_tokens, reduce_group_size)
    if token_padding:
        indices_2d = jnp.pad(indices_2d, ((0, token_padding), (0, 0)))
        weights_2d = jnp.pad(weights_2d, ((0, token_padding), (0, 0)))
        valid_2d = jnp.pad(valid_2d, ((0, token_padding), (0, 0)))

    route_metadata, route_weights, route_counts = _pack_gather_reduce_metadata(
        indices_2d, weights_2d, valid_2d, reduce_group_size
    )
    col_size = hidden_size // num_column_partitions
    blocks_per_row_partition = (padded_num_tokens // _TOKEN_BLOCK) // num_row_partitions
    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=sc_info.num_cores,
        num_subcores=sc_info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    reduce_kernel = cast(
        "Callable[..., jax.Array]",
        jax.jit(
            pl.kernel(
                functools.partial(
                    _main_kernel,
                    core_axis_name=vector_mesh.core_axis_name,
                    subcore_axis_name=vector_mesh.subcore_axis_name,
                    num_column_partitions=num_column_partitions,
                    blocks_per_row_partition=blocks_per_row_partition,
                    topk=reduce_group_size,
                ),
                out_type=jax.ShapeDtypeStruct(
                    (padded_num_tokens, hidden_size), jnp.bfloat16
                ),
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
            )
        ),
    )
    out = reduce_kernel(x, route_metadata, route_weights, route_counts)
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

_MIN_RING_ROWS = 16
_FULL_RING_ROWS = 64
_RING_STREAM_COUNT = 12
_REGION_FOUR_TUPLE = 4
_PROJECTION_VMEM_LIMIT = 40 * MIB_BYTES
_PROJECTION_MAX_ROWS_PER_RANK = 256
_PROJECTION_HEAD_WIDTH = KDA_HEADS * HEAD_DIM

# Sequence-parallel ownership: a rank holds ``stripe`` rows of every
# ``32 * stripe`` tokens. Local row i of rank r is window token
# (i // stripe) * 32 stripe + r * stripe + i % stripe. The stripe is STRIPE
# rows; a window under 32 STRIPE tokens (512, 1024) is one block of
# window / 32-row stripes (``window_stripe``).
STRIPE = 64
SP_BLOCK = TP_SIZE * STRIPE
SMALL_WINDOWS = PREFILL_WINDOWS[:2]


def ring_tables(
    devices: Sequence[_RingDeviceProtocol], token_ranks: Sequence[int]
) -> np.ndarray:
    """Build ``[family, position/peer/token-owner, mesh-index]`` lookup tables."""
    if len(devices) != TP_SIZE or sorted(token_ranks) != list(range(TP_SIZE)):
        msg = "K3 rings require 32 devices and a token-owner permutation"
        raise ValueError(msg)
    if [d.id for d in devices] != sorted(d.id for d in devices):
        msg = "Production collective mesh must use ascending device IDs"
        raise ValueError(msg)
    coords = [sorted({d.coords[a] for d in devices}) for a in range(3)]
    if tuple(map(len, coords)) != (2, 2, 4):
        msg = "K3 rings require a physical 2x2x4 slice"
        raise ValueError(msg)
    physical_to_mesh: dict[int, int] = {}
    for i, device in enumerate(devices):
        x, y, z = [coords[a].index(device.coords[a]) for a in range(3)]
        if device.core_on_chip not in (0, 1):
            msg = "Expected two cores per chip"
            raise ValueError(msg)
        physical = ((x * 2 + y) * 4 + z) * 2 + device.core_on_chip
        physical_to_mesh[physical] = i
    if len(physical_to_mesh) != TP_SIZE:
        msg = "Duplicate physical cores"
        raise ValueError(msg)
    result = []
    for cycle in CYCLES:
        peers = tuple(
            physical_to_mesh[chip * 2 + core] for chip in cycle for core in range(2)
        )
        positions = tuple(peers.index(i) for i in range(TP_SIZE))
        result.append((positions, peers, tuple(token_ranks)))
    return np.asarray(result, np.int32)


def regions(rows: int, width: int) -> tuple[tuple[int, int, int, int], ...]:
    """Partition ``(rows, width)`` into 12 bidirectional 3-cycle ring stream slices."""
    if rows % _MIN_RING_ROWS or width % VREG_LANE_COUNT or width < 3 * VREG_LANE_COUNT:
        msg = "K3 rings need 16-row ownership and 128-column alignment"
        raise ValueError(msg)
    blocks = width // VREG_LANE_COUNT
    if rows < _FULL_RING_ROWS:
        if blocks < _RING_STREAM_COUNT:
            msg = "K3 rings under 64 rows need at least 12 column blocks"
            raise ValueError(msg)
        widths = [
            VREG_LANE_COUNT
            * (blocks // _RING_STREAM_COUNT + (i < blocks % _RING_STREAM_COUNT))
            for i in range(_RING_STREAM_COUNT)
        ]
        cols = np.cumsum([0, *widths[:-1]])
        return tuple(
            (0, rows, int(col), w) for col, w in zip(cols, widths, strict=True)
        )
    if rows % _FULL_RING_ROWS:
        msg = "K3 rings need 64-row ownership from 64 rows"
        raise ValueError(msg)
    widths = [VREG_LANE_COUNT * (blocks // 3 + (i < blocks % 3)) for i in range(3)]
    cols_3 = [0, widths[0], widths[0] + widths[1]]
    return tuple(
        (
            (2 * group + direction) * (rows // 4),
            rows // 4,
            cols_3[family],
            widths[family],
        )
        for group in range(2)
        for family in range(3)
        for direction in range(2)
    )


def check_window(seq_len: int, what: str) -> None:
    """Validate that ``seq_len`` is 512, 1024, or a multiple of 2048 up to 8192."""
    if seq_len not in SMALL_WINDOWS and (
        seq_len % SP_BLOCK or not SP_BLOCK <= seq_len <= OVERFLOW_WINDOW_TOKENS
    ):
        msg = (
            f"{what} needs T in {SMALL_WINDOWS} or "
            f"2048..{OVERFLOW_WINDOW_TOKENS} (a multiple of 2048)"
        )
        raise ValueError(msg)


def window_stripe(local_rows: int) -> int:
    """Return sequence-parallel stripe height for ``local_rows`` rows per rank."""
    return min(STRIPE, local_rows)


def sp_token_index(
    rank: jax.Array | int, local: jax.Array | int, stripe: int
) -> jax.Array:
    """Map local row index ``local`` on ``rank`` to global window token index."""
    return jnp.asarray(
        (local // stripe) * TP_SIZE * stripe + rank * stripe + local % stripe
    )


def sp_order(window: int) -> np.ndarray:
    """Return global window token indices arranged in rank-major ownership order."""
    stripe = window_stripe(window // TP_SIZE)
    return np.arange(window).reshape(-1, TP_SIZE, stripe).transpose(1, 0, 2).reshape(-1)


def _stripe_pieces(begin: int, size: int, stripe: int) -> list[tuple[int, int, int]]:
    """Split local row interval ``[begin, begin + size)`` at stripe boundaries."""
    pieces: list[tuple[int, int, int]] = []
    row = begin
    while row < begin + size:
        rows = min(begin + size, (row // stripe + 1) * stripe) - row
        pieces.append((row - begin, row, rows))
        row += rows
    return pieces


def ring_enter(tables: PallasRefProtocol, rank: jax.Array) -> None:
    """Synchronize with all six ring neighbors before issuing remote transfers."""
    positions = [tables[f, 0, rank] for f in range(3)]
    peers = [
        tables[f, 1, (positions[f] + d) % TP_SIZE] for f in range(3) for d in (-1, 1)
    ]
    barrier = pltpu.get_barrier_semaphore()
    for p in peers:
        pl.semaphore_signal(
            barrier, 1, device_id=(p,), device_id_type=pl.DeviceIdType.MESH
        )
    pl.semaphore_wait(barrier, len(peers))

    @functools.partial(pl.run_scoped, ready=pltpu.SemaphoreType.REGULAR)
    def initialized(ready: PallasRefProtocol) -> None:
        """Complete secondary local semaphore handshake across ring neighbors."""
        for p in peers:
            pl.semaphore_signal(
                ready, 1, device_id=(p,), device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(ready, len(peers))


@dataclasses.dataclass(frozen=True, slots=True)
class _RingStreamDescriptor:
    """Target/source peer coordinates and slice bounds for one ring stream."""

    target: tuple[jax.Array]
    source: tuple[jax.Array]
    rows: pl.Slice
    cols: pl.Slice


@dataclasses.dataclass(frozen=True, slots=True)
class _RsKernelCtx:
    """Shared buffers, semaphores, and grid coordinates for reduce-scatter."""

    tables_ref: PallasRefProtocol
    x_ref: PallasRefProtocol
    o_ref: PallasRefProtocol
    local_sems: PallasRefProtocol
    send_sems: PallasRefProtocol
    recv_sems: PallasRefProtocol
    credit_sems: PallasRefProtocol
    store_sem: PallasRefProtocol
    local_buf: PallasRefProtocol
    payload: PallasRefProtocol
    received: PallasRefProtocol
    streams: Sequence[tuple[int, ...]]
    buffers: int
    tile: jax.Array
    step: jax.Array
    positions: list[jax.Array]


def _start_rs_reads(
    ctx: _RsKernelCtx,
) -> tuple[list[AsyncCopyProtocol], list[_RingStreamDescriptor]]:
    """Start local HBM-to-scratch reads for all reduce-scatter streams."""
    descriptors: list[_RingStreamDescriptor] = []
    reads: list[AsyncCopyProtocol] = []
    for stream, region in enumerate(ctx.streams):
        begin, size = region[:2]
        col_begin, width = (
            (region[2], region[3])
            if len(region) == _REGION_FOUR_TUPLE
            else (0, ctx.o_ref.shape[1])
        )
        cols = pl.ds(col_begin, width)
        family = (stream // 2) % 3
        direction = -1 if stream % 2 == 0 else 1
        position = ctx.positions[family]
        target = (ctx.tables_ref[family, 1, (position + direction) % TP_SIZE],)
        source = (ctx.tables_ref[family, 1, (position - direction) % TP_SIZE],)
        owner_mesh = ctx.tables_ref[
            family, 1, (position - direction * (ctx.step + 1)) % TP_SIZE
        ]
        owner = ctx.tables_ref[family, 2, owner_mesh]
        rows = pl.ds(begin, size)
        read = cast(
            "AsyncCopyProtocol",
            pltpu.make_async_copy(
                ctx.x_ref.at[
                    pl.ds(
                        owner * ctx.o_ref.shape[0]
                        + ctx.tile * ctx.payload.shape[0]
                        + begin,
                        size,
                    ),
                    cols,
                ],
                ctx.local_buf.at[rows, cols],
                ctx.local_sems.at[stream],
            ),
        )
        read.start()
        reads.append(read)
        descriptors.append(
            _RingStreamDescriptor(target=target, source=source, rows=rows, cols=cols)
        )
    return reads, descriptors


def _step_rs_stream(
    ctx: _RsKernelCtx,
    stream: int,
    read_copy: AsyncCopyProtocol,
    desc: _RingStreamDescriptor,
) -> None:
    """Advance one stream of ``_reduce_scatter_kernel`` at ``ctx.step``."""
    step = ctx.step
    rows, cols = desc.rows, desc.cols

    def remote(index: jax.Array) -> AsyncCopyProtocol:
        """Create remote copy forwarding accumulated payload at ``index``."""
        return cast(
            "AsyncCopyProtocol",
            pltpu.make_async_remote_copy(
                ctx.payload.at[rows, cols],
                ctx.received.at[index % ctx.buffers, rows, cols],
                ctx.send_sems.at[stream],
                ctx.recv_sems.at[index % ctx.buffers, stream],
                device_id=desc.target,
                device_id_type=pl.DeviceIdType.MESH,
            ),
        )

    @pl.when(step > 0)
    def wait_remote() -> None:
        """Wait for prior hop remote transfer completion."""
        remote(step - 1).wait()

    read_copy.wait()

    @pl.when(step == 0)
    def seed() -> None:
        """Initialize payload from step-0 local slice."""
        ctx.payload.at[rows, cols][...] = ctx.local_buf.at[rows, cols][...].astype(
            ctx.payload.dtype
        )

    @pl.when(step > 0)
    def add() -> None:
        """Accumulate local slice into received payload and return credit."""
        ctx.payload.at[rows, cols][...] = (
            ctx.local_buf.at[rows, cols][...].astype(ctx.payload.dtype)
            + ctx.received.at[(step - 1) % ctx.buffers, rows, cols][...]
        )
        pl.semaphore_signal(
            ctx.credit_sems.at[stream],
            1,
            device_id=desc.source,
            device_id_type=pl.DeviceIdType.MESH,
        )

    @pl.when(step < TP_PEERS)
    def forward() -> None:
        """Acquire downstream credit if needed and start remote hop."""

        @pl.when(step >= ctx.buffers)
        def acquire() -> None:
            """Wait for downstream receive-slot credit."""
            pl.semaphore_wait(ctx.credit_sems.at[stream], 1)

        remote(step).start()

    @pl.when(step == TP_PEERS)
    def finish() -> None:
        """Drain credits and store reduced slice on final hop."""
        pl.semaphore_wait(ctx.credit_sems.at[stream], ctx.buffers)
        output = pltpu.make_async_copy(
            ctx.payload.at[rows, cols],
            ctx.o_ref.at[
                pl.ds(ctx.tile * ctx.payload.shape[0] + rows.start, rows.size), cols
            ],
            ctx.store_sem,
        )
        output.start()
        output.wait()


def _reduce_scatter_kernel(
    tables_ref: PallasRefProtocol,
    x_ref: PallasRefProtocol,
    o_ref: PallasRefProtocol,
    local_sems: PallasRefProtocol,
    send_sems: PallasRefProtocol,
    recv_sems: PallasRefProtocol,
    credit_sems: PallasRefProtocol,
    store_sem: PallasRefProtocol,
    local_buf: PallasRefProtocol,
    payload: PallasRefProtocol,
    received: PallasRefProtocol,
    *,
    streams: Sequence[tuple[int, ...]],
    buffers: int,
) -> None:
    """Execute 32-step bidirectional 3-cycle ring reduce-scatter."""
    tile = pl.program_id(0)
    step = pl.program_id(1)
    rank = lax.axis_index(AXIS_SP)
    positions = [tables_ref[f, 0, rank] for f in range(3)]

    @pl.when(step == 0)
    def enter() -> None:
        """Synchronize all six ring neighbors at step 0."""
        ring_enter(tables_ref, rank)

    ctx = _RsKernelCtx(
        tables_ref=tables_ref,
        x_ref=x_ref,
        o_ref=o_ref,
        local_sems=local_sems,
        send_sems=send_sems,
        recv_sems=recv_sems,
        credit_sems=credit_sems,
        store_sem=store_sem,
        local_buf=local_buf,
        payload=payload,
        received=received,
        streams=streams,
        buffers=buffers,
        tile=tile,
        step=step,
        positions=positions,
    )
    reads, descriptors = _start_rs_reads(ctx)
    for stream, desc in enumerate(descriptors):
        _step_rs_stream(ctx, stream, reads[stream], desc)


@dataclasses.dataclass(frozen=True, slots=True)
class _AgKernelCtx:
    """Shared buffers, semaphores, and grid state for ``_all_gather_kernel``."""

    tables: PallasRefProtocol
    x: PallasRefProtocol
    out: PallasRefProtocol
    load_sems: PallasRefProtocol
    send_sems: PallasRefProtocol
    recv_sems: PallasRefProtocol
    credit_sems: PallasRefProtocol
    store_sems: PallasRefProtocol
    payload: PallasRefProtocol
    received: PallasRefProtocol
    buffers: int
    striped: bool
    tile: jax.Array
    step: jax.Array
    local_first: jax.Array
    stripe: int
    positions: list[jax.Array]


def _ag_store_copies(
    ctx: _AgKernelCtx,
    s: int,
    region: tuple[int, int, int, int],
    index: jax.Array,
) -> list[AsyncCopyProtocol]:
    """Build output copies for the rank arriving at ring hop ``index``."""
    begin, size, col, width = region
    family = (s // 2) % 3
    direction = -1 if s % 2 == 0 else 1
    pos = ctx.positions[family]
    cols = pl.ds(col, width)
    owner_mesh = ctx.tables[family, 1, (pos - direction * index) % TP_SIZE]
    dest = ctx.tables[family, 2, owner_mesh]
    if not ctx.striped:
        return [
            cast(
                "AsyncCopyProtocol",
                pltpu.make_async_copy(
                    ctx.payload.at[pl.ds(begin, size), cols],
                    ctx.out.at[
                        pl.ds(
                            dest * ctx.x.shape[0]
                            + ctx.tile * ctx.payload.shape[0]
                            + begin,
                            size,
                        ),
                        cols,
                    ],
                    ctx.store_sems.at[s],
                ),
            )
        ]
    return [
        cast(
            "AsyncCopyProtocol",
            pltpu.make_async_copy(
                ctx.payload.at[pl.ds(row, count), cols],
                ctx.out.at[pl.ds(sp_token_index(dest, row, ctx.stripe), count), cols],
                ctx.store_sems.at[s],
            ),
        )
        for _, row, count in _stripe_pieces(begin, size, ctx.stripe)
    ]


def _step_ag_stream(
    ctx: _AgKernelCtx,
    s: int,
    region: tuple[int, int, int, int],
) -> None:
    """Advance stream ``s`` of ``_all_gather_kernel`` at ``ctx.step``."""
    begin, size, col, width = region
    step = ctx.step
    family = (s // 2) % 3
    direction = -1 if s % 2 == 0 else 1
    pos = ctx.positions[family]
    target = (ctx.tables[family, 1, (pos + direction) % TP_SIZE],)
    source = (ctx.tables[family, 1, (pos - direction) % TP_SIZE],)
    rows, cols = pl.ds(begin, size), pl.ds(col, width)

    def remote(index: jax.Array) -> AsyncCopyProtocol:
        """Create remote copy forwarding payload at ``index``."""
        return cast(
            "AsyncCopyProtocol",
            pltpu.make_async_remote_copy(
                ctx.payload.at[rows, cols],
                ctx.received.at[index % ctx.buffers, rows, cols],
                ctx.send_sems.at[s],
                ctx.recv_sems.at[index % ctx.buffers, s],
                device_id=target,
                device_id_type=pl.DeviceIdType.MESH,
            ),
        )

    @pl.when(step == 0)
    def seed() -> None:
        """Wait for initial local load into ``payload``."""
        pltpu.make_async_copy(
            ctx.x.at[pl.ds(ctx.local_first + begin, size), cols],
            ctx.payload.at[rows, cols],
            ctx.load_sems.at[s],
        ).wait()

    @pl.when(step > 0)
    def receive() -> None:
        """Wait for prior remote transfer and store, then advance payload."""
        remote(step - 1).wait()
        for copy in _ag_store_copies(ctx, s, region, step - 1):
            copy.wait()
        ctx.payload.at[rows, cols][...] = ctx.received.at[
            (step - 1) % ctx.buffers, rows, cols
        ][...]
        pl.semaphore_signal(
            ctx.credit_sems.at[s],
            1,
            device_id=source,
            device_id_type=pl.DeviceIdType.MESH,
        )

    for copy in _ag_store_copies(ctx, s, region, step):
        copy.start()

    @pl.when(step < TP_PEERS)
    def forward() -> None:
        """Acquire downstream credit if needed and forward payload."""

        @pl.when(step >= ctx.buffers)
        def acquire() -> None:
            """Wait for downstream receive-slot credit."""
            pl.semaphore_wait(ctx.credit_sems.at[s], 1)

        remote(step).start()

    @pl.when(step == TP_PEERS)
    def finish() -> None:
        """Wait for final output stores and remaining credits."""
        for copy in _ag_store_copies(ctx, s, region, step):
            copy.wait()
        pl.semaphore_wait(ctx.credit_sems.at[s], ctx.buffers)


def _all_gather_kernel(
    tables: PallasRefProtocol,
    x: PallasRefProtocol,
    out: PallasRefProtocol,
    load_sems: PallasRefProtocol,
    send_sems: PallasRefProtocol,
    recv_sems: PallasRefProtocol,
    credit_sems: PallasRefProtocol,
    store_sems: PallasRefProtocol,
    payload: PallasRefProtocol,
    received: PallasRefProtocol,
    *,
    regions: Sequence[tuple[int, int, int, int]],
    buffers: int,
    striped: bool = False,
) -> None:
    """Gather all ranks' rows over 12 bidirectional ring streams."""
    tile = pl.program_id(0)
    step = pl.program_id(1)
    rank = lax.axis_index(AXIS_SP)
    positions = [tables[f, 0, rank] for f in range(3)]
    local_first = tile * payload.shape[0]
    stripe = window_stripe(x.shape[0])

    @pl.when(step == 0)
    def enter_and_load() -> None:
        """Synchronize ring neighbors and start step-0 local loads."""
        ring_enter(tables, rank)
        for s, (begin, size, col, width) in enumerate(regions):
            rows, cols = pl.ds(begin, size), pl.ds(col, width)
            pltpu.make_async_copy(
                x.at[pl.ds(local_first + begin, size), cols],
                payload.at[rows, cols],
                load_sems.at[s],
            ).start()

    ctx = _AgKernelCtx(
        tables=tables,
        x=x,
        out=out,
        load_sems=load_sems,
        send_sems=send_sems,
        recv_sems=recv_sems,
        credit_sems=credit_sems,
        store_sems=store_sems,
        payload=payload,
        received=received,
        buffers=buffers,
        striped=striped,
        tile=tile,
        step=step,
        local_first=local_first,
        stripe=stripe,
        positions=positions,
    )
    for s, region in enumerate(regions):
        _step_ag_stream(ctx, s, region)


@dataclasses.dataclass(frozen=True, slots=True)
class _ProjRingCtx:
    """Shared buffers, weights, and grid coordinates for ``_project_ring_kernel``."""

    tables_ref: PallasRefProtocol
    x_ref: PallasRefProtocol
    o_ref: PallasRefProtocol
    local_sems: PallasRefProtocol
    send_sems: PallasRefProtocol
    recv_sems: PallasRefProtocol
    credit_sems: PallasRefProtocol
    store_sem: PallasRefProtocol
    local_buf: PallasRefProtocol
    payload: PallasRefProtocol
    received: PallasRefProtocol
    weights: PallasRefProtocol
    inputs: PallasRefProtocol
    streams: Sequence[tuple[int, ...]]
    buffers: int
    tile: jax.Array
    step: jax.Array
    stripe: int
    positions: list[jax.Array]


def _start_proj_ring_reads(
    ctx: _ProjRingCtx,
) -> tuple[list[list[AsyncCopyProtocol]], list[_RingStreamDescriptor]]:
    """Start striped token input gathers for all projection-ring streams."""
    descriptors: list[_RingStreamDescriptor] = []
    reads: list[list[AsyncCopyProtocol]] = []
    for stream, region in enumerate(ctx.streams):
        begin, size = region[:2]
        col_begin, width = (
            (region[2], region[3])
            if len(region) == _REGION_FOUR_TUPLE
            else (0, ctx.o_ref.shape[1])
        )
        cols = pl.ds(col_begin, width)
        family = (stream // 2) % 3
        direction = -1 if stream % 2 == 0 else 1
        position = ctx.positions[family]
        target = (ctx.tables_ref[family, 1, (position + direction) % TP_SIZE],)
        source = (ctx.tables_ref[family, 1, (position - direction) % TP_SIZE],)
        peer = ctx.tables_ref[
            family, 1, (position - direction * (ctx.step + 1)) % TP_SIZE
        ]
        owner = ctx.tables_ref[family, 2, peer]
        rows = pl.ds(begin, size)
        read_copies = [
            cast(
                "AsyncCopyProtocol",
                pltpu.make_async_copy(
                    ctx.x_ref.at[
                        pl.ds(sp_token_index(owner, row, ctx.stripe), count), :
                    ],
                    ctx.inputs.at[stream, pl.ds(offset, count)],
                    ctx.local_sems.at[stream],
                ),
            )
            for offset, row, count in _stripe_pieces(begin, size, ctx.stripe)
        ]
        for copy in read_copies:
            copy.start()
        reads.append(read_copies)
        descriptors.append(
            _RingStreamDescriptor(target=target, source=source, rows=rows, cols=cols)
        )
    return reads, descriptors


def _step_proj_ring_stream(
    ctx: _ProjRingCtx,
    stream: int,
    read_copies: list[AsyncCopyProtocol],
    desc: _RingStreamDescriptor,
) -> None:
    """Compute projection slice and advance one stream of ``_project_ring_kernel``."""
    step = ctx.step
    rows, cols = desc.rows, desc.cols

    def remote(index: jax.Array) -> AsyncCopyProtocol:
        """Create remote copy forwarding projected partial at ``index``."""
        return cast(
            "AsyncCopyProtocol",
            pltpu.make_async_remote_copy(
                ctx.payload.at[rows, cols],
                ctx.received.at[index % ctx.buffers, rows, cols],
                ctx.send_sems.at[stream],
                ctx.recv_sems.at[index % ctx.buffers, stream],
                device_id=desc.target,
                device_id_type=pl.DeviceIdType.MESH,
            ),
        )

    @pl.when(step > 0)
    def wait_before() -> None:
        """Wait for prior hop remote transfer completion."""
        remote(step - 1).wait()

    for copy in read_copies:
        copy.wait()
    ctx.local_buf.at[rows, cols][...] = jnp.dot(
        ctx.inputs[stream], ctx.weights[:, cols], preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)

    @pl.when(step == 0)
    def seed() -> None:
        """Initialize payload from step-0 projected slice."""
        ctx.payload.at[rows, cols][...] = ctx.local_buf.at[rows, cols][...].astype(
            ctx.payload.dtype
        )

    @pl.when(step > 0)
    def add() -> None:
        """Add projected slice into received sum and signal upstream credit."""
        ctx.payload.at[rows, cols][...] = (
            ctx.local_buf.at[rows, cols][...].astype(ctx.payload.dtype)
            + ctx.received.at[(step - 1) % ctx.buffers, rows, cols][...]
        )
        pl.semaphore_signal(
            ctx.credit_sems.at[stream],
            1,
            device_id=desc.source,
            device_id_type=pl.DeviceIdType.MESH,
        )

    @pl.when(step < TP_PEERS)
    def forward() -> None:
        """Acquire downstream credit if needed and forward payload."""

        @pl.when(step >= ctx.buffers)
        def acquire() -> None:
            """Wait for downstream receive-slot credit."""
            pl.semaphore_wait(ctx.credit_sems.at[stream], 1)

        remote(step).start()

    @pl.when(step == TP_PEERS)
    def finish() -> None:
        """Drain credits and store final BF16 reduced slice."""
        pl.semaphore_wait(ctx.credit_sems.at[stream], ctx.buffers)
        ctx.local_buf.at[rows, cols][...] = ctx.payload.at[rows, cols][...].astype(
            ctx.local_buf.dtype
        )
        output = pltpu.make_async_copy(
            ctx.local_buf.at[rows, cols],
            ctx.o_ref.at[
                pl.ds(ctx.tile * ctx.payload.shape[0] + rows.start, rows.size), cols
            ],
            ctx.store_sem,
        )
        output.start()
        output.wait()


def _project_ring_kernel(
    tables_ref: PallasRefProtocol,
    x_ref: PallasRefProtocol,
    w_ref: PallasRefProtocol,
    o_ref: PallasRefProtocol,
    local_sems: PallasRefProtocol,
    send_sems: PallasRefProtocol,
    recv_sems: PallasRefProtocol,
    credit_sems: PallasRefProtocol,
    store_sem: PallasRefProtocol,
    local_buf: PallasRefProtocol,
    payload: PallasRefProtocol,
    received: PallasRefProtocol,
    weights: PallasRefProtocol,
    inputs: PallasRefProtocol,
    weight_sem: PallasRefProtocol,
    *,
    streams: Sequence[tuple[int, ...]],
    buffers: int,
) -> None:
    """Fuse per-rank output projection with 32-step ring reduce-scatter."""
    tile = pl.program_id(0)
    step = pl.program_id(1)
    rank = lax.axis_index(AXIS_SP)
    positions = [tables_ref[f, 0, rank] for f in range(3)]
    stripe = window_stripe(o_ref.shape[0])

    @pl.when(step == 0)
    def enter_and_load_weight() -> None:
        """Synchronize ring neighbors and load projection weights at step 0."""
        ring_enter(tables_ref, rank)
        pltpu.async_copy(w_ref, weights, weight_sem).wait()

    ctx = _ProjRingCtx(
        tables_ref=tables_ref,
        x_ref=x_ref,
        o_ref=o_ref,
        local_sems=local_sems,
        send_sems=send_sems,
        recv_sems=recv_sems,
        credit_sems=credit_sems,
        store_sem=store_sem,
        local_buf=local_buf,
        payload=payload,
        received=received,
        weights=weights,
        inputs=inputs,
        streams=streams,
        buffers=buffers,
        tile=tile,
        step=step,
        stripe=stripe,
        positions=positions,
    )
    reads, descriptors = _start_proj_ring_reads(ctx)
    for stream, desc in enumerate(descriptors):
        _step_proj_ring_stream(ctx, stream, reads[stream], desc)


def project_reduce_scatter(
    x: jax.Array,
    w: jax.Array,
    mesh: jax.sharding.Mesh,
    tables: np.ndarray,
    *,
    collective_id: int,
) -> jax.Array:
    """Fuse three-head BF16 attention output projection with SP32 reduce-scatter."""
    if (
        x.shape[0] != TP_SIZE
        or x.shape[1] % TP_SIZE
        or x.shape[2] != _PROJECTION_HEAD_WIDTH
        or w.shape != (TP_SIZE, _PROJECTION_HEAD_WIDTH, HIDDEN_SIZE)
    ):
        msg = "K3 projection/RS requires SP32 and 3 local heads"
        raise ValueError(msg)
    if x.dtype != jnp.bfloat16 or w.dtype != jnp.bfloat16:
        msg = "K3 projection/RS requires BF16 operands"
        raise ValueError(msg)
    tokens = x.shape[1]
    rows = tokens // TP_SIZE
    if rows > _PROJECTION_MAX_ROWS_PER_RANK:
        msg = (
            "K3 projection/RS needs at most "
            f"{_PROJECTION_MAX_ROWS_PER_RANK} rows per rank, got {rows}"
        )
        raise ValueError(msg)
    stream_regions = regions(rows, HIDDEN_SIZE)
    streams = len(stream_regions)
    chunk = max(size for _, size, _, _ in stream_regions)

    def local(x_shard: jax.Array, w_shard: jax.Array) -> jax.Array:
        """Run fused projection and reduce-scatter on one rank."""
        x_2d = x_shard.reshape(tokens, _PROJECTION_HEAD_WIDTH)
        w_2d = w_shard.reshape(_PROJECTION_HEAD_WIDTH, HIDDEN_SIZE)
        hbm = pl.BlockSpec(memory_space=pltpu.HBM)
        scratch = (
            pltpu.SemaphoreType.DMA((streams,)),
            pltpu.SemaphoreType.DMA((streams,)),
            pltpu.SemaphoreType.DMA((2, streams)),
            pltpu.SemaphoreType.REGULAR((streams,)),
            pltpu.SemaphoreType.DMA,
            pltpu.VMEM((rows, HIDDEN_SIZE), jnp.bfloat16),
            pltpu.VMEM((rows, HIDDEN_SIZE), jnp.bfloat16),
            pltpu.VMEM((2, rows, HIDDEN_SIZE), jnp.bfloat16),
            pltpu.VMEM((_PROJECTION_HEAD_WIDTH, HIDDEN_SIZE), jnp.bfloat16),
            pltpu.VMEM((streams, chunk, _PROJECTION_HEAD_WIDTH), jnp.bfloat16),
            pltpu.SemaphoreType.DMA,
        )
        return cast(
            "jax.Array",
            pl.pallas_call(
                functools.partial(
                    _project_ring_kernel, streams=stream_regions, buffers=2
                ),
                out_shape=jax.ShapeDtypeStruct((rows, HIDDEN_SIZE), jnp.bfloat16),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=1,
                    in_specs=(hbm, hbm),
                    out_specs=hbm,
                    scratch_shapes=scratch,
                    grid=(1, TP_SIZE),
                ),
                compiler_params=pltpu.CompilerParams(
                    collective_id=collective_id,
                    vmem_limit_bytes=_PROJECTION_VMEM_LIMIT,
                ),
                name="kimi_attention_project_ring_rs",
            )(jnp.asarray(tables, jnp.int32), x_2d, w_2d),
        )

    return cast(
        "jax.Array",
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(AXIS_SP, None, None), P(AXIS_SP, None, None)),
            out_specs=P(AXIS_SP, None),
            check_vma=False,
        )(x, w),
    )


# ============================================================================
# Fused grouped matmul (expert FFN)
# ============================================================================
#
# The grouped matmul adapted from ``kernels/megablox/gmm_v2.py``: the routed
# experts' gate/up (fused SiTU) and down projections, MXFP4 weights.

_FAMILY_WEIGHT_NDIM = 4
_PYTREE_LOCK = threading.Lock()
_PYTREES_REGISTERED: list[bool] = [False]


def ensure_prefill_pytrees_registered() -> None:
    """Register grouped-matmul reference dataclasses as JAX PyTrees once."""
    if _PYTREES_REGISTERED[0]:
        return
    with _PYTREE_LOCK:
        if _PYTREES_REGISTERED[0]:
            return
        jax.tree_util.register_dataclass(WeightsRef)
        jax.tree_util.register_dataclass(FusedWeightsRef)
        jax.tree_util.register_dataclass(MetadataRef)
        _PYTREES_REGISTERED[0] = True


def situ_and_mul(
    gate: jax.Array, up: jax.Array, beta: float, linear_beta: float | None
) -> jax.Array:
    """Apply SiTU activation to ``gate`` and multiply with optional tanh ``up``."""
    gate = beta * jnp.tanh(gate / beta) * jax.nn.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * jnp.tanh(up / linear_beta)
    return gate * up


def interleave_lane(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    """Interleave ``lhs`` and ``rhs`` chunks along the trailing lane axis."""
    if lhs.shape != rhs.shape:
        msg = f"interleave_lane shape mismatch: {lhs.shape} vs {rhs.shape}"
        raise ValueError(msg)
    chunk_size = pltpu.get_tpu_info().num_lanes
    num_chunks = lhs.shape[-1] // chunk_size
    lhs_chunks = jnp.split(lhs, num_chunks, axis=-1)
    rhs_chunks = jnp.split(rhs, num_chunks, axis=-1)
    interleaved: list[jax.Array] = []
    for i in range(num_chunks):
        interleaved += [lhs_chunks[i], rhs_chunks[i]]
    return jnp.concat(interleaved, axis=-1)


def deinterleave_lane(val: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Split lane-interleaved ``val`` back into ``(lhs, rhs)`` halves."""
    chunk_size = pltpu.get_tpu_info().num_lanes
    num_chunks = val.shape[-1] // chunk_size
    chunks = jnp.split(val, num_chunks, axis=-1)
    lhs = jnp.concat(chunks[0::2], axis=-1)
    rhs = jnp.concat(chunks[1::2], axis=-1)
    return lhs, rhs


def apply_act_fn(acc: jax.Array, fuse_act: str | None) -> jax.Array:
    """Apply optional ``situ:<beta>:<linear_beta>`` fused activation to ``acc``."""
    if fuse_act is None:
        return acc
    acc_gate, acc_up = deinterleave_lane(acc)
    match fuse_act:
        case str() if fuse_act.startswith("situ:"):
            _, beta, linear_beta_str = fuse_act.split(":")
            linear_beta = None if linear_beta_str == "none" else float(linear_beta_str)
            return situ_and_mul(acc_gate, acc_up, float(beta), linear_beta)
        case _:
            msg = f"Unsupported activation function: {fuse_act}"
            raise NotImplementedError(msg)


def align_to(x: int, a: int) -> int:
    """Round integer ``x`` up to a multiple of ``a``."""
    return pl.cdiv(x, a) * a


def _get_lhs_sublane_size(dtype: jnp.dtype, size_m: int) -> int:
    """Return sublane row block size bounded by ``size_m``."""
    return min(pltpu.get_tpu_info().get_sublane_tiling(dtype), size_m)


def get_packing_factor(storage_dtype: jnp.dtype, quant_dtype: jnp.dtype | None) -> int:
    """Return element packing factor of ``quant_dtype`` inside ``storage_dtype``."""
    if quant_dtype is None or quant_dtype == storage_dtype:
        return 1
    storage_bits = jax.dtypes.itemsize_bits(storage_dtype)
    quant_bits = jax.dtypes.itemsize_bits(quant_dtype)
    packing_factor, remainder = divmod(storage_bits, quant_bits)
    if remainder != 0:
        msg = (
            f"Storage dtype {storage_dtype} is not divisible by "
            f"quant dtype {quant_dtype}"
        )
        raise ValueError(msg)
    return packing_factor


class RhsRef(ABC):
    """Abstract interface for grouped-matmul right-hand-side weight references."""

    @abstractmethod
    def get_weight(self) -> jax.Array:
        """Load the current weight tile."""

    @abstractmethod
    def get_scale(self, replicate_size: int | None = None) -> jax.Array:
        """Load the current scale tile, optionally replicated across sublanes."""


@dataclasses.dataclass(frozen=True)
class WeightsRef(RhsRef):
    """Single-matrix weight and optional block-scale container."""

    weight: jax.Array | PallasRefProtocol | pl.BlockSpec
    scale: jax.Array | PallasRefProtocol | pl.BlockSpec | None

    def __post_init__(self) -> None:
        """Ensure PyTree registration before JAX traverses ``WeightsRef``."""
        ensure_prefill_pytrees_registered()

    def get_weight(self) -> jax.Array:
        """Dereference the current weight tile."""
        return cast("PallasRefProtocol", self.weight)[...]

    def get_scale(self, replicate_size: int | None = None) -> jax.Array:
        """Dereference and expand E8M0 or FP32 block scales."""
        if self.scale is None:
            msg = "WeightsRef.scale is None"
            raise ValueError(msg)
        scale_ref = cast("PallasRefProtocol", self.scale)
        if scale_ref.dtype == jnp.uint8:
            exponents = scale_ref[...]
            powers = e8m0_scale(exponents)[:, None, :]
            return jnp.broadcast_to(
                powers,
                (exponents.shape[0], replicate_size or 1, exponents.shape[1]),
            )
        if replicate_size is not None:
            # Perform zero-stride load for efficient broadcasting across sublanes.
            return scale_ref[:, pl.ds(0, replicate_size, 0), :]
        return scale_ref[...]


@dataclasses.dataclass(frozen=True)
class FusedWeightsRef(RhsRef):
    """Paired gate and up weight references interleaved along lanes."""

    gate: WeightsRef
    up: WeightsRef

    def __post_init__(self) -> None:
        """Ensure PyTree registration before JAX traverses ``FusedWeightsRef``."""
        ensure_prefill_pytrees_registered()

    def get_weight(self) -> jax.Array:
        """Load and lane-interleave gate and up weight tiles."""
        return interleave_lane(self.gate.get_weight(), self.up.get_weight())

    def get_scale(self, replicate_size: int | None = None) -> jax.Array:
        """Load and lane-interleave gate and up scale tiles."""
        return interleave_lane(
            self.gate.get_scale(replicate_size), self.up.get_scale(replicate_size)
        )


@dataclasses.dataclass(frozen=True)
class MetadataRef:
    """Group ID and row-offset scratch references for ragged grouped matmul."""

    gm_id_to_group_id: jax.Array | PallasRefProtocol | pl_core.MemoryRef
    gm_id_to_m_offset: jax.Array | PallasRefProtocol | pl_core.MemoryRef

    def __post_init__(self) -> None:
        """Ensure PyTree registration before JAX traverses ``MetadataRef``."""
        ensure_prefill_pytrees_registered()


@dataclasses.dataclass(frozen=True)
class TileSizes:
    """Tile dimensions and bucket granularity for grouped matmul."""

    tile_m: int
    tile_k: int
    tile_n: int
    bucket_base: int


@dataclasses.dataclass(frozen=True)
class Dimensions:
    """Logical problem dimensions for grouped matmul."""

    size_m: int
    size_k: int
    size_n: int
    size_group: int
    size_lhs_group: int
    size_lhs_sublane: int


@dataclasses.dataclass(frozen=True)
class InputConfigs:
    """Quantization and element-type configuration for a matmul operand."""

    quant_dtype: jnp.dtype | None
    quant_block_size: int | None
    dtype: jnp.dtype
    has_scale: bool = False
    scale_e8m0: bool = False

    @property
    def should_unpack(self) -> bool:
        """Return True when sub-byte weights are packed into wider carriers."""
        return get_packing_factor(self.dtype, self.quant_dtype) > 1

    @property
    def should_dequantize_before_matmul(self) -> bool:
        """Return True when quant block size is narrower than matrix panel width."""
        if not self.has_scale:
            return False
        if self.quant_block_size is None:
            msg = "quant_block_size must be set when has_scale is True"
            raise ValueError(msg)
        return self.quant_block_size < pltpu.get_tpu_info().mxu_column_size

    @property
    def should_dequantize_after_matmul(self) -> bool:
        """Return True when block scales apply after panel matmul."""
        return self.has_scale and not self.should_dequantize_before_matmul


@dataclasses.dataclass(frozen=True)
class GmmConfigs:
    """Complete tiling, shape, and dtype configuration for ``gmm_v2``."""

    tiles: TileSizes
    dims: Dimensions
    lhs_cfgs: InputConfigs
    rhs_cfgs: InputConfigs
    out_dtype: jnp.dtype
    acc_dtype: jnp.dtype
    fuse_act: str | None

    @property
    def num_quant_blocks_per_tile_k(self) -> int:
        """Return number of quantization blocks spanning one ``tile_k``."""
        qbs = self.rhs_cfgs.quant_block_size
        if qbs is None:
            msg = "rhs_cfgs.quant_block_size must not be None"
            raise ValueError(msg)
        return pl.cdiv(self.tiles.tile_k, qbs)

    @property
    def out_size_n(self) -> int:
        """Return output column width after optional gate/up fusion."""
        if self.fuse_act is None:
            return self.dims.size_n
        return self.dims.size_n // 2


TileFn = Callable[[Dimensions, InputConfigs, InputConfigs, int, str | None], TileSizes]


class IndexMaps:
    """Pipeline block index maps for ``gmm_v2``."""

    def __init__(self, metadata_ref: MetadataRef, cfgs: GmmConfigs) -> None:
        """Bind ``metadata_ref`` and ``cfgs`` for pipeline index mapping."""
        self.metadata_ref = metadata_ref
        self.cfgs = cfgs

    def lhs_index_map(
        self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array
    ) -> tuple[pl.Slice, int, jax.Array]:
        """Map ``(gm_id, k_id)`` to bounded sublane-aligned LHS slice."""
        offsets = cast("PallasRefProtocol", self.metadata_ref.gm_id_to_m_offset)
        m_start = offsets[gm_id]
        m_end = offsets[gm_id + 1]
        row_start = m_start // self.cfgs.dims.size_lhs_sublane
        row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
        return (pl.ds(row_start, row_end - row_start), 0, k_id)

    def rhs_weight_index_map(
        self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Map ``(n_id, gm_id, k_id)`` to expert weight tile coordinates."""
        groups = cast("PallasRefProtocol", self.metadata_ref.gm_id_to_group_id)
        return (groups[gm_id], k_id, n_id)

    def rhs_scale_index_map(
        self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array
    ) -> tuple[jax.Array | int, ...]:
        """Map ``(n_id, gm_id, k_id)`` to expert scale tile coordinates."""
        groups = cast("PallasRefProtocol", self.metadata_ref.gm_id_to_group_id)
        group_id = groups[gm_id]
        qbs = self.cfgs.rhs_cfgs.quant_block_size
        if qbs is None:
            msg = "quant_block_size must not be None when indexing rhs scale"
            raise ValueError(msg)
        # Simply multiplying k_id by num_quant_blocks_per_tile_k will not work
        # since a single quant block could be shared along multiple k tile.
        k_row = k_id * self.cfgs.tiles.tile_k
        b_row = k_row // qbs
        b_tile_id = b_row // self.cfgs.num_quant_blocks_per_tile_k
        if self.cfgs.rhs_cfgs.scale_e8m0:
            return (group_id, b_tile_id, n_id)
        return (group_id, b_tile_id, 0, n_id)

    def out_index_map(
        self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array
    ) -> tuple[pl.Slice, int, jax.Array]:
        """Map ``(n_id, gm_id)`` to bounded sublane-aligned output slice."""
        offsets = cast("PallasRefProtocol", self.metadata_ref.gm_id_to_m_offset)
        is_last_gm = gm_id == (pl.num_programs(1) - 1)
        m_start = offsets[gm_id]
        m_end = offsets[gm_id + 1]
        row_start = m_start // self.cfgs.dims.size_lhs_sublane
        capped_row_end = m_end // self.cfgs.dims.size_lhs_sublane
        last_row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
        row_end = jnp.where(is_last_gm, last_row_end, capped_row_end)
        return (pl.ds(row_start, row_end - row_start), 0, n_id)


def generate_block_specs(
    metadata_ref: MetadataRef, cfgs: GmmConfigs
) -> tuple[tuple[pl.BlockSpec, WeightsRef], pl.BlockSpec]:
    """Build LHS, RHS, and output ``BlockSpec`` objects for ``cfgs``."""
    ensure_prefill_pytrees_registered()
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
        scale_shape = (
            (None, cfgs.num_quant_blocks_per_tile_k, cfgs.tiles.tile_n)
            if cfgs.rhs_cfgs.scale_e8m0
            else (None, cfgs.num_quant_blocks_per_tile_k, 1, cfgs.tiles.tile_n)
        )
        rhs_scale_block_spec = pl.BlockSpec(scale_shape, index_map.rhs_scale_index_map)

    rhs_block_spec = WeightsRef(weight=rhs_weight_spec, scale=rhs_scale_block_spec)
    out_block_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_n),
        index_map.out_index_map,
    )
    return (lhs_block_spec, rhs_block_spec), out_block_spec


@dataclasses.dataclass(frozen=True, slots=True)
class _GmmTileCtx:
    """Per-tile references and local row bounds for ``inner_kernel``."""

    tiled_lhs_ref: jax.Array | PallasRefProtocol | ReshapableProtocol
    tiled_rhs_ref: RhsRef
    tiled_out_ref: PallasRefProtocol
    partial_out_ref: PallasRefProtocol
    acc_ref: PallasRefProtocol
    cfgs: GmmConfigs
    gm_id: jax.Array
    m_start_local: jax.Array
    m_end_local: jax.Array


def _preprocess_gmm_inputs(
    tile_context: _GmmTileCtx, bucket_m: int, *, is_last_k_step: bool
) -> tuple[jax.Array, jax.Array, int]:
    """Slice ``bucket_m`` LHS rows and optionally pre-dequantize or mask RHS."""
    cfgs = tile_context.cfgs
    lhs_2d = cast("ReshapableProtocol", tile_context.tiled_lhs_ref).reshape(
        -1, cfgs.tiles.tile_k
    )
    tiled_lhs = cast("PallasRefProtocol", lhs_2d)[:bucket_m]
    tiled_rhs = tile_context.tiled_rhs_ref.get_weight()
    rhs_tile_n = tiled_rhs.shape[1]
    rhs_qbs = cfgs.rhs_cfgs.quant_block_size
    if rhs_qbs is None:
        rhs_qbs = cfgs.tiles.tile_k
    if cfgs.rhs_cfgs.should_dequantize_before_matmul:
        tiled_rhs_scale = tile_context.tiled_rhs_ref.get_scale(
            replicate_size=rhs_qbs
        ).astype(cfgs.lhs_cfgs.dtype)
        num_blocks = cfgs.num_quant_blocks_per_tile_k
        tiled_rhs_dequant = tiled_rhs.astype(cfgs.lhs_cfgs.dtype).reshape(
            num_blocks, rhs_qbs, rhs_tile_n
        )
        tiled_rhs = (tiled_rhs_dequant * tiled_rhs_scale).reshape(
            cfgs.tiles.tile_k, rhs_tile_n
        )
        rhs_qbs = cfgs.tiles.tile_k

    valid_k = cfgs.dims.size_k % cfgs.tiles.tile_k
    if is_last_k_step and valid_k != 0:
        mask_rhs = lax.broadcasted_iota(jnp.int32, tiled_rhs.shape, 0) < valid_k
        tiled_rhs = jnp.where(mask_rhs, tiled_rhs, 0)
    return tiled_lhs, tiled_rhs, rhs_qbs


def _compute_gmm_panels(
    tile_context: _GmmTileCtx,
    tiled_lhs: jax.Array,
    tiled_rhs: jax.Array,
    rhs_qbs: int,
    bucket_m: int,
) -> jax.Array:
    """Accumulate panel matrix products and optional post-matmul scales."""
    cfgs = tile_context.cfgs
    panel_cols = pltpu.get_tpu_info().mxu_column_size
    rhs_tile_n = tiled_rhs.shape[1]
    acc_dtype = tile_context.acc_ref.dtype
    acc_list: list[jax.Array] = []
    for start_n in range(0, rhs_tile_n, panel_cols):
        end_n = min(rhs_tile_n, start_n + panel_cols)
        col_size = end_n - start_n
        acc_n = jnp.zeros((bucket_m, col_size), dtype=acc_dtype)
        for start_k in range(0, cfgs.tiles.tile_k, rhs_qbs):
            end_k = min(cfgs.tiles.tile_k, start_k + rhs_qbs)
            block_acc = jnp.matmul(
                tiled_lhs[:, start_k:end_k],
                tiled_rhs[start_k:end_k, start_n:end_n],
                preferred_element_type=jnp.float32,
            ).astype(acc_dtype)
            if cfgs.rhs_cfgs.should_dequantize_after_matmul:
                b_id = start_k // rhs_qbs
                rhs_scale_rep = tile_context.tiled_rhs_ref.get_scale(
                    replicate_size=bucket_m
                )[b_id, :, start_n : start_n + col_size]
                block_acc *= rhs_scale_rep.astype(acc_dtype)
            acc_n += block_acc
        acc_list.append(acc_n)
    return jnp.concatenate(acc_list, axis=1)


def _run_gmm_matmul(
    tile_context: _GmmTileCtx,
    bucket_m: int,
    *,
    is_first_k_step: bool,
    is_last_k_step: bool,
) -> None:
    """Execute one K-step of grouped matmul for ``bucket_m`` active rows."""
    cfgs = tile_context.cfgs
    tiled_lhs, tiled_rhs, rhs_qbs = _preprocess_gmm_inputs(
        tile_context, bucket_m, is_last_k_step=is_last_k_step
    )
    acc = _compute_gmm_panels(tile_context, tiled_lhs, tiled_rhs, rhs_qbs, bucket_m)
    if not is_first_k_step:
        acc = jnp.pad(acc, ((cfgs.tiles.tile_m - bucket_m, 0), (0, 0)))
        acc += tile_context.acc_ref[...]
    acc_m = acc.shape[0]
    if not is_last_k_step:
        tile_context.acc_ref[:acc_m] = acc
        return

    acc = apply_act_fn(acc, cfgs.fuse_act)
    iota = lax.broadcasted_iota(jnp.int32, acc.shape, 0)
    mask = jnp.logical_and(
        tile_context.m_start_local <= iota, iota < tile_context.m_end_local
    )
    acc_masked = jnp.where(mask, acc, 0)
    tiled_out_2d_ref = cast(
        "PallasRefProtocol", tile_context.tiled_out_ref.reshape(-1, cfgs.tiles.tile_n)
    )
    tiled_out_2d_ref[:acc_m] = acc_masked.astype(tile_context.tiled_out_ref.dtype)

    # If this is the first tile for grid[n_id, :, :], we initialize the
    # partial out to zeros. Otherwise, partial out from last tile of
    # grid[n_id-1, :, :] can be used and cause numeric issues.
    partial_out_zeros = jnp.zeros_like(tile_context.partial_out_ref)
    tile_context.tiled_out_ref[0] += jnp.where(
        tile_context.gm_id == 0, partial_out_zeros, tile_context.partial_out_ref[...]
    )
    last_row = tile_context.m_end_local // cfgs.dims.size_lhs_sublane
    tile_context.partial_out_ref[...] = jnp.where(
        tile_context.m_end_local % cfgs.dims.size_lhs_sublane == 0,
        partial_out_zeros,
        tile_context.tiled_out_ref[last_row],
    )


def _dispatch_gmm_bucket(tile_context: _GmmTileCtx, *, bucket_idx: int) -> None:
    """Dispatch first/middle/last K-step branch for ``bucket_idx``."""
    bucket_m = tile_context.cfgs.tiles.bucket_base * (bucket_idx + 1)

    @jax.named_scope(f"bm{bucket_m}_first_last")
    def matmul_first_last() -> None:
        """Run single-step K accumulation and output store."""
        _run_gmm_matmul(
            tile_context, bucket_m, is_first_k_step=True, is_last_k_step=True
        )

    @jax.named_scope(f"bm{bucket_m}_first")
    def matmul_first() -> None:
        """Run initial K-step into accumulator scratch."""
        _run_gmm_matmul(
            tile_context, bucket_m, is_first_k_step=True, is_last_k_step=False
        )

    @jax.named_scope(f"bm{bucket_m}_mid")
    def matmul_mid() -> None:
        """Run intermediate K-step accumulating into scratch."""
        _run_gmm_matmul(
            tile_context, bucket_m, is_first_k_step=False, is_last_k_step=False
        )

    @jax.named_scope(f"bm{bucket_m}_last")
    def matmul_last() -> None:
        """Run final K-step and write masked output."""
        _run_gmm_matmul(
            tile_context, bucket_m, is_first_k_step=False, is_last_k_step=True
        )

    num_k = pl.num_programs(2)
    k_id = pl.program_id(2)
    is_first_k = k_id == 0
    is_last_k = k_id == (num_k - 1)
    if bucket_m == tile_context.cfgs.tiles.tile_m:

        def first_branch() -> None:
            """Dispatch first K-step whether single-step or multi-step."""
            lax.cond(is_last_k, matmul_first_last, matmul_first)

        def later_branch() -> None:
            """Dispatch non-first K-step whether middle or final."""
            lax.cond(is_last_k, matmul_last, matmul_mid)

        lax.cond(is_first_k, first_branch, later_branch)
    else:
        lax.cond(is_first_k, matmul_first_last, matmul_last)


def inner_kernel(
    tiled_lhs_ref: jax.Array | PallasRefProtocol | ReshapableProtocol,
    tiled_rhs_ref: RhsRef,
    tiled_out_ref: PallasRefProtocol,
    partial_out_ref: PallasRefProtocol,
    acc_ref: PallasRefProtocol,
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
) -> None:
    """Execute one pipelined grouped-matmul tile across M-size buckets."""
    gm_id = pl.program_id(1)
    offsets = cast("PallasRefProtocol", metadata_ref.gm_id_to_m_offset)
    m_start = offsets[gm_id]
    m_end = offsets[gm_id + 1]
    m_offset = m_start - m_start % cfgs.dims.size_lhs_sublane
    tile_context = _GmmTileCtx(
        tiled_lhs_ref=tiled_lhs_ref,
        tiled_rhs_ref=tiled_rhs_ref,
        tiled_out_ref=tiled_out_ref,
        partial_out_ref=partial_out_ref,
        acc_ref=acc_ref,
        cfgs=cfgs,
        gm_id=gm_id,
        m_start_local=m_start - m_offset,
        m_end_local=m_end - m_offset,
    )
    branches = [
        functools.partial(_dispatch_gmm_bucket, tile_context, bucket_idx=bucket_idx)
        for bucket_idx in range(cfgs.tiles.tile_m // cfgs.tiles.bucket_base)
    ]
    lax.switch(tile_context.m_end_local // cfgs.tiles.bucket_base, branches)


def fill_metadata(
    lhs_group_sizes_ref: jax.Array | PallasRefProtocol,
    group_offset_ref: jax.Array | PallasRefProtocol,
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
) -> jax.Array:
    """Populate ``metadata_ref`` group IDs and row offsets and return tile count."""
    group_offset = group_offset_ref[0]
    max_num_group = group_offset + cfgs.dims.size_group
    group_ids = cast("PallasRefProtocol", metadata_ref.gm_id_to_group_id)
    m_offsets = cast("PallasRefProtocol", metadata_ref.gm_id_to_m_offset)
    m_offsets[0] = 0

    @jax.named_scope("inner_tm_loop")
    def inner_tm_loop(
        tm_id: jax.Array,
        curr_m_offset: jax.Array,
        *,
        end_m_offset: jax.Array,
        group_id: jax.Array,
    ) -> jax.Array:
        """Record group ID and M-offset bounds for tile ``tm_id``."""
        local_offset = curr_m_offset % cfgs.dims.size_lhs_sublane
        tm_size = jnp.minimum(
            cfgs.tiles.tile_m - local_offset, end_m_offset - curr_m_offset
        )
        group_ids[tm_id] = group_id
        next_m_offset = curr_m_offset + tm_size
        m_offsets[tm_id] = curr_m_offset
        m_offsets[tm_id + 1] = next_m_offset
        return next_m_offset

    @jax.named_scope("outer_group_loop")
    def outer_group_loop(
        lhs_group_id: jax.Array, carry: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        """Emit all M-tiles required by ``lhs_group_id``."""
        num_gm, start_m_offset = carry
        group_id = lhs_group_id - group_offset
        group_size = lhs_group_sizes_ref[lhs_group_id]
        end_m_offset = start_m_offset + group_size
        local_offset = start_m_offset % cfgs.dims.size_lhs_sublane
        aligned_group_size = group_size + local_offset
        curr_num_gm = pl.cdiv(aligned_group_size, cfgs.tiles.tile_m)
        should_process = jnp.logical_and(group_size > 0, group_id >= 0)
        curr_num_gm = jnp.where(should_process, curr_num_gm, 0)
        next_num_gm = num_gm + curr_num_gm
        tm_loop_fn = functools.partial(
            inner_tm_loop, end_m_offset=end_m_offset, group_id=group_id
        )
        lax.fori_loop(num_gm, next_num_gm, tm_loop_fn, start_m_offset)
        return next_num_gm, end_m_offset

    num_gm, _ = lax.fori_loop(
        0, max_num_group, outer_group_loop, (jnp.int32(0), jnp.int32(0))
    )
    return num_gm


def kernel_main(
    lhs_group_sizes_ref: PallasRefProtocol,
    group_offset_ref: PallasRefProtocol,
    rhs_layer_ref: PallasRefProtocol,
    lhs_ref: PallasRefProtocol,
    rhs_ref: WeightsRef,
    out_ref: PallasRefProtocol,
    partial_out_ref: PallasRefProtocol,
    acc_ref: PallasRefProtocol,
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
) -> None:
    """Build tile metadata and run the pipelined grouped-matmul kernel."""
    num_k = pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k)
    num_n = pl.cdiv(cfgs.out_size_n, cfgs.tiles.tile_n)
    weight_ref = cast("PallasRefProtocol", rhs_ref.weight)
    if weight_ref.ndim == _FAMILY_WEIGHT_NDIM:
        # Select the HBM view before the pipeline loads a tile. An array-level
        # dynamic slice would copy every expert in this layer into a new buffer.
        rhs_ref = dataclasses.replace(rhs_ref, weight=weight_ref.at[rhs_layer_ref[0]])
        if rhs_ref.scale is not None:
            scale_ref = cast("PallasRefProtocol", rhs_ref.scale)
            if scale_ref.dtype == jnp.uint8 and scale_ref.ndim == _FAMILY_WEIGHT_NDIM:
                rhs_ref = dataclasses.replace(
                    rhs_ref, scale=scale_ref.at[rhs_layer_ref[0]]
                )

    if cfgs.rhs_cfgs.should_unpack:
        if cfgs.rhs_cfgs.quant_dtype is None:
            msg = "quant_dtype must not be None when should_unpack is True"
            raise ValueError(msg)
        rhs_weight = cast("PallasRefProtocol", rhs_ref.weight).bitcast(
            cfgs.rhs_cfgs.quant_dtype
        )
        rhs_ref = dataclasses.replace(rhs_ref, weight=rhs_weight)

    num_gm = fill_metadata(
        lhs_group_sizes_ref, group_offset_ref, metadata_ref, cfgs=cfgs
    )
    (lhs_spec, rhs_spec), out_spec = generate_block_specs(metadata_ref, cfgs)

    rhs_in: RhsRef = rhs_ref
    rhs_in_spec: WeightsRef | FusedWeightsRef = rhs_spec
    if cfgs.fuse_act is not None:

        def slice_up_half(x: PallasRefProtocol) -> object:
            """Slice the second half of columns holding up-projection weights."""
            return x.at[..., cfgs.out_size_n :]

        rhs_up_ref = jax.tree.map(slice_up_half, rhs_ref)
        rhs_in = FusedWeightsRef(gate=rhs_ref, up=rhs_up_ref)
        rhs_in_spec = FusedWeightsRef(gate=rhs_spec, up=rhs_spec)

    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(inner_kernel, cfgs=cfgs),
        grid=(num_n, num_gm, num_k),
        in_specs=(lhs_spec, rhs_in_spec),
        out_specs=out_spec,
    )
    lhs_in = lhs_ref.reshape(-1, cfgs.dims.size_lhs_sublane, lhs_ref.shape[-1])
    out_in = out_ref.reshape(-1, cfgs.dims.size_lhs_sublane, out_ref.shape[-1])
    pipeline_fn(
        lhs_in, rhs_in, out_in, scratches=[partial_out_ref, acc_ref, metadata_ref]
    )


def _estimate_gmm_scratch_bytes(
    lhs_cfgs: InputConfigs,
    rhs_cfgs: InputConfigs,
    fuse_act_factor: int,
    tiles_mnk: tuple[int, int, int],
) -> int:
    """Estimate total buffered scratch footprint in bytes for ``tiles_mnk``."""
    tm, tn, tk = tiles_mnk
    lhs_bits = jax.dtypes.itemsize_bits(lhs_cfgs.dtype)
    rhs_bits = jax.dtypes.itemsize_bits(rhs_cfgs.quant_dtype or rhs_cfgs.dtype)
    lhs_bytes = 2 * tm * tk * (lhs_bits // 8)
    rhs_weight_bytes = tk * tn * rhs_bits // 8
    rhs_scale_bytes = 0
    if rhs_cfgs.has_scale and rhs_cfgs.quant_block_size is not None:
        rhs_scale_bytes = pl.cdiv(tk, rhs_cfgs.quant_block_size) * tn * 4
    rhs_bytes = fuse_act_factor * (3 * rhs_weight_bytes + 2 * rhs_scale_bytes)
    acc_bytes = tm * fuse_act_factor * tn * 4
    out_bytes = 2 * tm * tn * (lhs_bits // 8)
    return lhs_bytes + rhs_bytes + acc_bytes + out_bytes


def calculate_tiling(
    dims: Dimensions,
    lhs_cfgs: InputConfigs,
    rhs_cfgs: InputConfigs,
    vmem_limit_bytes: int,
    fuse_act: str | None = None,
) -> TileSizes:
    """Calculate optimal M, K, and N tile sizes within ``vmem_limit_bytes``."""
    rhs_dtype = rhs_cfgs.quant_dtype or rhs_cfgs.dtype
    rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)

    # When using bf16 for lhs and rhs, 128 is the largest tile_m value that is
    # safe to use for most scenarios. But if lower bitwidth is used, we need
    # to tweak tile_m to account for using faster hardware unit.
    # TODO(kyuyeunk): Account for different TPU hardware specs.
    bf16_bf16_tile_m = 128
    rhs_mod = min(pl.cdiv(16, rhs_bits), 2)
    tile_m = min(bf16_bf16_tile_m // rhs_mod, dims.size_m)
    tile_n_limit = min(pltpu.get_tpu_info().mxu_column_size * 2, dims.size_n)

    size_n_per_rhs = dims.size_n
    fuse_act_factor = 1
    if fuse_act is not None:
        fuse_act_factor = 2
        size_n_per_rhs //= fuse_act_factor
        tile_n_limit //= fuse_act_factor

    def is_k_compatible(tk: int) -> bool:
        """Check compatibility between ``tk`` and ``rhs_cfgs.quant_block_size``."""
        qbs = rhs_cfgs.quant_block_size
        if qbs is None:
            return True
        return tk % qbs == 0 or qbs % tk == 0

    def estimate(tm: int, tn: int, tk: int) -> int:
        """Return scratch byte estimate for ``(tm, tn, tk)``."""
        return _estimate_gmm_scratch_bytes(
            lhs_cfgs, rhs_cfgs, fuse_act_factor, (tm, tn, tk)
        )

    num_k_tiles = num_n_tiles = 1
    num_lanes = pltpu.get_tpu_info().num_lanes
    tile_k = align_to(dims.size_k, num_lanes)
    tile_n = align_to(size_n_per_rhs, num_lanes)

    while estimate(tile_m, tile_n, tile_k) > vmem_limit_bytes and tile_n > tile_n_limit:
        num_n_tiles += 1
        tile_n = align_to(size_n_per_rhs, num_n_tiles * num_lanes) // num_n_tiles

    if tile_n < tile_n_limit:
        num_n_tiles -= 1
        tile_n = align_to(size_n_per_rhs, num_n_tiles * num_lanes) // num_n_tiles
        while (
            estimate(tile_m, tile_n, tile_k) > vmem_limit_bytes
            or not is_k_compatible(tile_k)
        ) and tile_k > num_lanes:
            num_k_tiles += 1
            tile_k = align_to(dims.size_k, num_k_tiles * num_lanes) // num_k_tiles

    if tile_n == 0 or tile_k == 0:
        final_estimate = estimate(tile_m, tile_n, tile_k)
        msg = (
            f"Could not find valid tile sizes for {dims=} and"
            f" {final_estimate=} (limit: {vmem_limit_bytes})."
        )
        raise ValueError(msg)

    # TODO(alynie, kyuyeunk): max number of buckets was chosen empirically.
    # Revisit the value to be based on number of instruction memory size.
    max_num_buckets = 4
    bucket_base = tile_m
    for _ in range(1, max_num_buckets):
        new_tile_m = tile_m + bucket_base
        if (
            new_tile_m > dims.size_m
            or estimate(new_tile_m, tile_n, tile_k) > vmem_limit_bytes
        ):
            break
        tile_m = new_tile_m

    return TileSizes(
        tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, bucket_base=bucket_base
    )


def validate_inputs(
    lhs: jax.Array | jax.ShapeDtypeStruct,
    rhs: jax.Array | jax.ShapeDtypeStruct,
    rhs_scale: jax.Array | jax.ShapeDtypeStruct | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    fuse_act: str | None = None,
    packing_factor: int = 1,
) -> Dimensions:
    """Validate operand shapes for ``gmm_v2`` and return ``Dimensions``."""
    size_m = lhs.shape[0]
    size_group, size_k, size_n = rhs.shape
    size_k *= packing_factor
    size_lhs_group = group_sizes.shape[0]
    if (
        size_group > size_lhs_group
        or lhs.shape != (size_m, size_k)
        or rhs.shape != (size_group, size_k // packing_factor, size_n)
        or group_offset.shape != (1,)
    ):
        msg = "Invalid lhs, rhs, or group_offset shapes for grouped matmul"
        raise ValueError(msg)

    if rhs_scale is not None:
        num_quant_blocks = rhs_scale.shape[1]
        expected_scale_shape = (
            (size_group, num_quant_blocks, size_n)
            if rhs_scale.dtype == jnp.uint8
            else (size_group, num_quant_blocks, 1, size_n)
        )
        if rhs_scale.shape != expected_scale_shape or size_k % num_quant_blocks != 0:
            msg = f"Invalid rhs_scale shape {rhs_scale.shape} for size_k={size_k}"
            raise ValueError(msg)

    size_lhs_sublane = _get_lhs_sublane_size(jnp.dtype(lhs.dtype), size_m)
    if fuse_act is not None:
        num_lanes = pltpu.get_tpu_info().num_lanes
        if size_n % (2 * num_lanes) != 0:
            msg = (
                f"{size_n=} should be divisible by 2 * num_lanes when fuse_act is "
                "enabled since we need to split n dimension for gate and up."
            )
            raise ValueError(msg)

    return Dimensions(
        size_m=size_m,
        size_k=size_k,
        size_n=size_n,
        size_group=size_group,
        size_lhs_group=size_lhs_group,
        size_lhs_sublane=size_lhs_sublane,
    )


def get_cost_estimate(cfgs: GmmConfigs) -> pl.CostEstimate:
    """Return FLOP and HBM byte cost estimate for ``cfgs``."""
    dims = cfgs.dims
    lhs_dtype = cfgs.lhs_cfgs.dtype
    rhs_dtype = cfgs.rhs_cfgs.quant_dtype or cfgs.rhs_cfgs.dtype
    rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)
    fp32_bytes = jnp.dtype(jnp.float32).itemsize

    # TODO(kyuyeunk): Add compute flops for dequant.
    flops = 2 * dims.size_m * dims.size_k * dims.size_n
    lhs_bytes = dims.size_m * dims.size_k * lhs_dtype.itemsize
    rhs_size = dims.size_group * dims.size_k * dims.size_n
    rhs_bytes = rhs_size * rhs_bits // 8
    if cfgs.rhs_cfgs.has_scale and cfgs.rhs_cfgs.quant_block_size is not None:
        num_quant_blocks = pl.cdiv(dims.size_k, cfgs.rhs_cfgs.quant_block_size)
        rhs_bytes += dims.size_group * num_quant_blocks * dims.size_n * fp32_bytes
    out_bytes = dims.size_m * cfgs.out_size_n * cfgs.out_dtype.itemsize
    return pl.CostEstimate(
        flops=flops,
        bytes_accessed=lhs_bytes + rhs_bytes + out_bytes,
        transcendentals=0,
    )


def get_scope_name(cfgs: GmmConfigs) -> str:
    """Format descriptive kernel scope name for ``cfgs``."""
    dims = cfgs.dims
    tiles = cfgs.tiles
    return (
        f"gmm_v2-g_{dims.size_group}-m_{dims.size_m}-k_{dims.size_k}-act_{cfgs.fuse_act}"
        f"-n_{dims.size_n}-tm_{tiles.tile_m}-tk_{tiles.tile_k}-tn_{tiles.tile_n}"
    )


def make_gmm_configs(
    lhs: jax.Array | jax.ShapeDtypeStruct,
    rhs: jax.Array | jax.ShapeDtypeStruct,
    rhs_scale: jax.Array | jax.ShapeDtypeStruct | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    *,
    tile_info: TileSizes | TileFn,
    vmem_limit_bytes: int | None,
    out_dtype: jax.typing.DTypeLike | None,
    acc_dtype: jax.typing.DTypeLike | None,
    fuse_act: str | None = None,
    rhs_quant_dtype: jax.typing.DTypeLike | None = None,
) -> GmmConfigs:
    """Construct validated ``GmmConfigs`` for ``gmm_v2`` and fused MoE callers."""
    ensure_prefill_pytrees_registered()
    rhs_dt = jnp.dtype(rhs.dtype)
    lhs_dt = jnp.dtype(lhs.dtype)
    quant_dt = jnp.dtype(rhs_quant_dtype) if rhs_quant_dtype is not None else None
    packing_factor = get_packing_factor(rhs_dt, quant_dt)
    dims = validate_inputs(
        lhs, rhs, rhs_scale, group_sizes, group_offset, fuse_act, packing_factor
    )
    if rhs_scale is not None:
        has_scale = True
        quant_dt = quant_dt or rhs_dt
        block_size = dims.size_k // rhs_scale.shape[1]
    else:
        has_scale = False
        block_size = dims.size_k

    rhs_cfgs = InputConfigs(
        quant_dtype=quant_dt,
        quant_block_size=block_size,
        dtype=rhs_dt,
        has_scale=has_scale,
        scale_e8m0=rhs_scale is not None and rhs_scale.dtype == jnp.uint8,
    )
    lhs_cfgs = InputConfigs(quant_dtype=None, quant_block_size=None, dtype=lhs_dt)
    resolved_out = jnp.dtype(lhs_dt if out_dtype is None else out_dtype)
    resolved_acc = jnp.dtype(jnp.float32 if acc_dtype is None else acc_dtype)
    if isinstance(tile_info, TileSizes):
        tiles = tile_info
    else:
        limit = (
            int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
            if vmem_limit_bytes is None
            else vmem_limit_bytes
        )
        tiles = tile_info(dims, lhs_cfgs, rhs_cfgs, limit, fuse_act)

    return GmmConfigs(
        dims=dims,
        tiles=tiles,
        lhs_cfgs=lhs_cfgs,
        rhs_cfgs=rhs_cfgs,
        out_dtype=resolved_out,
        acc_dtype=resolved_acc,
        fuse_act=fuse_act,
    )


def get_metadata(cfgs: GmmConfigs) -> dict[str, str]:
    """Serialize ``cfgs`` leaf fields into string metadata for ``pallas_call``."""
    cfgs_dict = dataclasses.asdict(cfgs)
    ret: dict[str, str] = {}
    for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
        key = jax.tree_util.keystr(path, simple=True, separator=".")
        ret[key] = str(val)
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
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    rhs_scale: jax.Array | None = None,
    group_offset: jax.Array | None = None,
    *,
    rhs_layer: jax.Array | None = None,
    tile_info: TileSizes | TileFn = calculate_tiling,
    vmem_limit_bytes: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    acc_dtype: jax.typing.DTypeLike | None = None,
    fuse_act: str | None = None,
    rhs_quant_dtype: jax.typing.DTypeLike | None = None,
) -> jax.Array:
    """Compute ragged grouped matrix multiplication via ``emit_pipeline``."""
    del precision
    ensure_prefill_pytrees_registered()
    size_m = lhs.shape[0]
    size_lhs_sublane = _get_lhs_sublane_size(lhs.dtype, size_m)
    padded_size_m = align_to(size_m, size_lhs_sublane)
    if padded_size_m != size_m:
        lhs = jnp.pad(lhs, ((0, padded_size_m - size_m), (0, 0)))

    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    elif jnp.isscalar(group_offset):
        group_offset = group_offset[None]

    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)

    family = rhs.ndim == _FAMILY_WEIGHT_NDIM
    if family != (rhs_layer is not None):
        msg = "A family RHS requires a runtime layer index"
        raise ValueError(msg)
    arg_rhs: jax.Array | jax.ShapeDtypeStruct = (
        jax.ShapeDtypeStruct(rhs.shape[1:], rhs.dtype) if family else rhs
    )
    arg_scale: jax.Array | jax.ShapeDtypeStruct | None = rhs_scale
    if (
        family
        and rhs_scale is not None
        and rhs_scale.dtype == jnp.uint8
        and rhs_scale.ndim == _FAMILY_WEIGHT_NDIM
    ):
        arg_scale = jax.ShapeDtypeStruct(rhs_scale.shape[1:], rhs_scale.dtype)
    if rhs_layer is None:
        rhs_layer = jnp.zeros((1,), jnp.int32)
    elif rhs_layer.shape != (1,) or rhs_layer.dtype != jnp.int32:
        msg = "rhs_layer must have shape (1,) and dtype int32"
        raise ValueError(msg)

    cfgs = make_gmm_configs(
        lhs,
        arg_rhs,
        arg_scale,
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
        if rhs_scale.dtype != jnp.uint8:
            rhs_scale = rhs_scale.astype(jnp.float32)
        rhs_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    max_num_gm = dims.size_group + pl.cdiv(dims.size_m, tiles.tile_m) - 1
    acc_cols = 2 * tiles.tile_n if cfgs.fuse_act is not None else tiles.tile_n
    scratch_shapes = cast(
        "pl_core.ScratchShapeTree",
        [
            pltpu.VMEM((dims.size_lhs_sublane, tiles.tile_n), cfgs.out_dtype),
            pltpu.VMEM((tiles.tile_m, acc_cols), cfgs.acc_dtype),
            MetadataRef(
                gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
                gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
            ),
        ],
    )
    aligned_n = align_to(cfgs.out_size_n, pltpu.get_tpu_info().num_lanes)
    out_init = jax.ShapeDtypeStruct((dims.size_m, aligned_n), cfgs.out_dtype)
    rhs_weights = WeightsRef(weight=rhs, scale=rhs_scale)

    full_out = cast(
        "jax.Array",
        pl.pallas_call(
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
        )(group_sizes, group_offset, rhs_layer, lhs, rhs_weights),
    )
    return full_out[:size_m, : cfgs.out_size_n]


# ============================================================================
# Prefill building blocks
# ============================================================================
#
# Chunked KDA math, the MLA projections and the logits' residual mixture.

_HEADS = KDA_HEADS
_CHANNELS = HEAD_DIM
_KDA_CHUNK = 64
_KDA_SUB_BLOCK = 16
_RESIDUAL_BLOCKS = 8
_DOUBLE_BUFFER_SLOTS = 2
_MIXTURE_VMEM_LIMIT = 64 * MIB_BYTES
_CONV_HISTORY_LEN = 3
_CONV_TAPS = 4


def _dot(x: jax.Array, w: jax.Array) -> jax.Array:
    """Compute ``[rows, K] @ [K, N]`` in FP32 for 8-row-aligned ``x``."""
    if x.shape[0] % SUBLANE_ROWS:
        msg = "prefill dots use 8-row-aligned tiles"
        raise ValueError(msg)
    return jax.lax.dot_general(
        x, w, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
    )


def _dot3(x: jax.Array, y: jax.Array, contracting: int = 2) -> jax.Array:
    """Compute batched head-major ``[H, M, K] @ [H, K, N]`` in FP32."""
    return jax.lax.dot_general(
        x,
        y,
        (((contracting,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )


def _row_where(mask: jax.Array, x: jax.Array, fill: float = 0) -> jax.Array:
    """Broadcast boolean ``mask`` along trailing dimensions of ``x`` via int32."""
    m = mask.astype(jnp.int32)
    while len(m.shape) < len(x.shape):
        m = m[..., None]
    m = jnp.broadcast_to(m, x.shape) != 0
    return jnp.where(m, x, fill)


def _mask_where(mask: jax.Array, x: jax.Array, fill: jax.Array | float) -> jax.Array:
    """Select between ``x`` and ``fill`` using a broadcasting bool mask.

    Lowered through int32 so the Mosaic i1-reshape limitation is not hit.
    """
    cond = jnp.broadcast_to(mask.astype(jnp.int32), x.shape) != 0
    return jnp.where(cond, x, fill)


# ---------------------------------------------------------------------------
# Chunked KDA math (pure functions; also called on ref reads in the kernel).
# ---------------------------------------------------------------------------


def _inclusive_cumsum_rows(x: jax.Array) -> jax.Array:
    """Compute inclusive FP32 cumulative sum along axis ``-2`` via doubling."""
    shift = 1
    rows = x.shape[-2]
    while shift < rows:
        x = x + jnp.pad(x, ((0, 0), (shift, 0), (0, 0)))[..., :rows, :]
        shift *= 2
    return x


def _invert_delta_rule_two_level(low: jax.Array, block: int = 8) -> jax.Array:
    """Invert ``I + low`` via two-level nilpotent Neumann doubling."""
    chunk_size = low.shape[-1]
    nc_inv = chunk_size // block
    eye = jnp.eye(chunk_size, dtype=jnp.float32)[None]
    idx = jnp.arange(chunk_size, dtype=jnp.int32)
    same_block = (idx[:, None] // block == idx[None, :] // block).astype(jnp.float32)
    l_diag = low * same_block[None]
    f = low - l_diag
    neg_ld = -l_diag
    p = eye + neg_ld
    mk = neg_ld
    for _ in range(block.bit_length() - 2):
        mk = _dot3(mk, mk)
        p = p + _dot3(p, mk)
    g = _dot3(p, f)
    h = -g
    inv_ig = eye + h
    hk = h
    for _ in range(nc_inv.bit_length() - 2):
        hk = _dot3(hk, hk)
        inv_ig = inv_ig + _dot3(inv_ig, hk)
    return _dot3(inv_ig, p)


def _kda_chunk_core(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    gate: jax.Array,
    beta: jax.Array,
    state: jax.Array,
    *,
    sub_block: int = _KDA_SUB_BLOCK,
    invert: Callable[[jax.Array], jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Evaluate one chunk of the parallel Kimi gated delta rule."""
    if invert is None:
        invert = _invert_delta_rule_two_level
    _head_count, chunk_size, _channels = q.shape
    value_channels = v.shape[-1]
    if chunk_size % sub_block:
        msg = "chunk size must be a multiple of the sub-block"
        raise ValueError(msg)
    nc = chunk_size // sub_block
    gc = _inclusive_cumsum_rows(gate)
    beta_k = beta[..., None].astype(jnp.float32)

    rows = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 0)
    cols = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 1)
    causal = (cols <= rows)[None]
    strictly_causal = (cols < rows)[None]

    aqk_parts: list[jax.Array] = []
    akk_parts: list[jax.Array] = []
    col_iota = jax.lax.broadcasted_iota(jnp.int32, (chunk_size,), 0)
    for i in range(nc):
        i_s = i * sub_block
        anchor = gc[:, i_s + sub_block // 2, :][:, None, :]
        row_scale = jnp.exp(gc[:, i_s : i_s + sub_block, :] - anchor)
        through = (col_iota < i_s + sub_block).astype(jnp.float32)[None, :, None]
        k_scaled = k * jnp.exp(jnp.minimum(anchor - gc, 80.0)) * through
        both = jax.lax.dot_general(
            jnp.concatenate(
                (
                    q[:, i_s : i_s + sub_block] * row_scale,
                    k[:, i_s : i_s + sub_block] * row_scale,
                ),
                axis=1,
            ),
            k_scaled.transpose(0, 2, 1),
            (((2,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        )
        aqk_parts.append(both[:, :sub_block])
        akk_parts.append(both[:, sub_block:])
    aqk = _mask_where(causal, jnp.concatenate(aqk_parts, axis=1), jnp.float32(0))
    akk = jnp.concatenate(akk_parts, axis=1)
    low = _mask_where(strictly_causal, akk * beta[:, :, None], jnp.float32(0))
    solve = invert(low)

    rhs = jnp.concatenate((v * beta_k, k * jnp.exp(gc) * beta_k), axis=-1)
    solved = _dot3(solve, rhs)
    u, w = solved[..., :value_channels], solved[..., value_channels:]
    v_new = u - _dot3(w, state)

    anchor0 = gc[:, 0, :]
    out = _dot3(
        q * jnp.exp(gc - anchor0[:, None, :]),
        state * jnp.exp(anchor0)[:, :, None],
    ) + _dot3(aqk, v_new)

    g_last = gc[:, -1, :]
    kg = k * jnp.exp(g_last[:, None, :] - gc)
    new_state = state * jnp.exp(g_last)[:, :, None] + _dot3(kg, v_new, contracting=1)
    aux = {"gc": gc, "aqk": aqk, "low": low, "solve": solve, "v_new": v_new}
    return out, new_state, aux


def kda_chunk(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    gate: jax.Array,
    beta: jax.Array,
    state: jax.Array,
    *,
    sub_block: int = _KDA_SUB_BLOCK,
    invert: Callable[[jax.Array], jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate one chunk of the Kimi delta rule and return ``(out, new_state)``."""
    out, new_state, _ = _kda_chunk_core(
        q, k, v, gate, beta, state, sub_block=sub_block, invert=invert
    )
    return out, new_state


# ---------------------------------------------------------------------------
# MLA projections
# ---------------------------------------------------------------------------


def _mla_projections_with_latent(
    hidden: jax.Array,
    weights: ProjectionWeights,
    *,
    epsilon: float = 1e-5,
    query_scale: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute per-rank MLA Q/K/V, output gate, and compressed KV latent."""
    tokens = hidden.shape[0]
    query_latent = _dot(hidden, weights.query_down_projection)
    query_latent = (
        query_latent
        * jax.lax.rsqrt(
            jnp.mean(query_latent * query_latent, axis=1, keepdims=True) + epsilon
        )
        * weights.query_normalization
    ).astype(jnp.bfloat16)
    query = (_dot(query_latent, weights.query_up_projection) * query_scale).astype(
        jnp.bfloat16
    )
    query = query.reshape(tokens, _HEADS, 256)

    compressed = _dot(hidden, weights.key_value_down_projection).astype(jnp.bfloat16)
    kv_latent = compressed[:, :512].astype(jnp.float32)
    kv_latent = (
        kv_latent
        * jax.lax.rsqrt(
            jnp.mean(kv_latent * kv_latent, axis=1, keepdims=True) + epsilon
        )
        * weights.key_value_normalization
    ).astype(jnp.bfloat16)
    expanded = _dot(kv_latent, weights.key_value_up_projection).astype(jnp.bfloat16)
    expanded = expanded.reshape(tokens, _HEADS, 256)
    new_key = jnp.concatenate(
        (
            expanded[:, :, :128],
            jnp.broadcast_to(compressed[:, None, 512:576], (tokens, _HEADS, 64)),
            jnp.zeros((tokens, _HEADS, 64), jnp.bfloat16),
        ),
        axis=2,
    )
    new_value = expanded[:, :, 128:]
    output_gate = _dot(hidden, weights.output_gate_projection).astype(jnp.bfloat16)
    latent = jnp.concatenate((kv_latent, compressed[:, 512:576]), axis=1)
    return query, new_key, new_value, output_gate, latent


# ---------------------------------------------------------------------------
# Residual mixture (dense form, used by the logits; the kernel streams it)
# ---------------------------------------------------------------------------


def residual_mixture(
    blocks: jax.Array,
    prefix: jax.Array,
    folded: jax.Array,
    active_block_count: jax.Array | int,
    *,
    epsilon: float = 1e-5,
) -> jax.Array:
    """Compute learned softmax residual mixture over ``blocks`` and ``prefix``."""
    values = jnp.concatenate(
        (blocks.astype(jnp.float32), prefix[None].astype(jnp.float32)), axis=0
    )
    scores = jnp.sum(values * folded.astype(jnp.float32), axis=-1)
    scores *= jax.lax.rsqrt(jnp.mean(values * values, axis=-1) + epsilon)
    index = jax.lax.broadcasted_iota(jnp.int32, (_RESIDUAL_BLOCKS + 1, 1), 0)
    active = (index < active_block_count) | (index == _RESIDUAL_BLOCKS)
    probabilities = jax.nn.softmax(jnp.where(active, scores, -jnp.inf), axis=0)
    mixed = jnp.sum(values * probabilities[..., None], axis=0)
    return mixed.astype(jnp.bfloat16)


# ============================================================================
# Residual mixture and RMSNorm (HBM-tiled)
# ============================================================================


@dataclasses.dataclass(frozen=True, slots=True)
class _MixNormCtx:
    """Per-call references and constants for ``mixture_norm_tiled_body``."""

    prefix_ref: PallasRefProtocol
    blocks_ref: PallasRefProtocol
    output_ref: PallasRefProtocol
    prefix_bufs: PallasRefProtocol
    block_buf: PallasRefProtocol
    output_bufs: PallasRefProtocol
    numerator_buf: PallasRefProtocol
    row_max_buf: PallasRefProtocol
    row_sum_buf: PallasRefProtocol
    prefix_sems: PallasRefProtocol
    block_sem: PallasRefProtocol
    output_sems: PallasRefProtocol
    folded: jax.Array
    norm: jax.Array
    active_count: jax.Array
    n_tiles: int
    tile: int
    eps: float


def _include_mix_block(mix_context: _MixNormCtx, t: jax.Array, block: int) -> None:
    """Fold active residual block ``block`` into online softmax accumulators."""

    @pl.when(block < mix_context.active_count)
    def include_block() -> None:
        """Load block tile and update online softmax numerator and sum."""
        copy = pltpu.make_async_copy(
            mix_context.blocks_ref.at[
                block, pl.ds(t * mix_context.tile, mix_context.tile), :
            ],
            mix_context.block_buf,
            mix_context.block_sem.at[0],
        )
        copy.start()
        copy.wait()
        values = mix_context.block_buf[...].astype(jnp.float32)
        score = jnp.sum(values * mix_context.folded, axis=1)
        score *= jax.lax.rsqrt(jnp.mean(values * values, axis=1) + mix_context.eps)
        old_max = mix_context.row_max_buf[...]
        new_max = jnp.maximum(old_max, score[:, None])
        old_scale = jnp.exp(old_max - new_max)
        new_scale = jnp.exp(score[:, None] - new_max)
        mix_context.numerator_buf[...] = (
            mix_context.numerator_buf[...] * old_scale + values * new_scale
        )
        mix_context.row_sum_buf[...] = (
            mix_context.row_sum_buf[...] * old_scale + new_scale
        )
        mix_context.row_max_buf[...] = new_max


def _run_mix_norm_tile(t: jax.Array, *, mix_context: _MixNormCtx) -> None:
    """Process one row tile of online residual mixture and RMSNorm."""
    slot = t % _DOUBLE_BUFFER_SLOTS
    pltpu.make_async_copy(
        mix_context.prefix_ref.at[pl.ds(t * mix_context.tile, mix_context.tile), :],
        mix_context.prefix_bufs.at[slot],
        mix_context.prefix_sems.at[slot],
    ).wait()

    @pl.when(t >= _DOUBLE_BUFFER_SLOTS)
    def wait_previous_output() -> None:
        """Wait for output DMA from two tiles back before reusing ``slot``."""
        pltpu.make_async_copy(
            mix_context.output_bufs.at[slot],
            mix_context.output_ref.at[
                pl.ds((t - 2) * mix_context.tile, mix_context.tile), :
            ],
            mix_context.output_sems.at[slot],
        ).wait()

    @pl.when(t + 1 < mix_context.n_tiles)
    def prefetch_next_prefix() -> None:
        """Prefetch next tile's prefix rows into alternate slot."""
        pltpu.make_async_copy(
            mix_context.prefix_ref.at[
                pl.ds((t + 1) * mix_context.tile, mix_context.tile), :
            ],
            mix_context.prefix_bufs.at[1 - slot],
            mix_context.prefix_sems.at[1 - slot],
        ).start()

    prefix = mix_context.prefix_bufs.at[slot][...].astype(jnp.float32)
    prefix_score = jnp.sum(prefix * mix_context.folded, axis=1)
    prefix_score *= jax.lax.rsqrt(jnp.mean(prefix * prefix, axis=1) + mix_context.eps)
    mix_context.numerator_buf[...] = prefix
    mix_context.row_max_buf[...] = prefix_score[:, None]
    mix_context.row_sum_buf[...] = jnp.ones((mix_context.tile, 1), jnp.float32)

    for block in range(_RESIDUAL_BLOCKS):
        _include_mix_block(mix_context, t, block)

    mixed = (
        (mix_context.numerator_buf[...] / mix_context.row_sum_buf[...])
        .astype(jnp.bfloat16)
        .astype(jnp.float32)
    )
    normalized = (
        mixed
        * jax.lax.rsqrt(
            jnp.mean(mixed * mixed, axis=1, keepdims=True) + mix_context.eps
        )
        * mix_context.norm
    ).astype(jnp.bfloat16)
    mix_context.output_bufs.at[slot][...] = normalized
    pltpu.make_async_copy(
        mix_context.output_bufs.at[slot],
        mix_context.output_ref.at[pl.ds(t * mix_context.tile, mix_context.tile), :],
        mix_context.output_sems.at[slot],
    ).start()


def mixture_norm_tiled_body(
    active_ref: PallasRefProtocol,
    prefix_ref: PallasRefProtocol,
    blocks_ref: PallasRefProtocol,
    folded_ref: PallasRefProtocol,
    norm_ref: PallasRefProtocol,
    output_ref: PallasRefProtocol,
    prefix_bufs: PallasRefProtocol,
    block_buf: PallasRefProtocol,
    output_bufs: PallasRefProtocol,
    numerator_buf: PallasRefProtocol,
    row_max_buf: PallasRefProtocol,
    row_sum_buf: PallasRefProtocol,
    folded_buf: PallasRefProtocol,
    norm_buf: PallasRefProtocol,
    prefix_sems: PallasRefProtocol,
    block_sem: PallasRefProtocol,
    output_sems: PallasRefProtocol,
    weight_sems: PallasRefProtocol,
    *,
    tile: int,
    eps: float,
) -> None:
    """Run tiled online residual mixture and RMSNorm into ``output_ref``."""
    tokens, _hidden = prefix_ref.shape
    n_tiles = tokens // tile
    weights = (
        pltpu.make_async_copy(folded_ref, folded_buf, weight_sems.at[0]),
        pltpu.make_async_copy(norm_ref, norm_buf, weight_sems.at[1]),
    )
    for copy in weights:
        copy.start()
    for copy in weights:
        copy.wait()

    pltpu.make_async_copy(
        prefix_ref.at[pl.ds(0, tile), :], prefix_bufs.at[0], prefix_sems.at[0]
    ).start()
    mix_context = _MixNormCtx(
        prefix_ref=prefix_ref,
        blocks_ref=blocks_ref,
        output_ref=output_ref,
        prefix_bufs=prefix_bufs,
        block_buf=block_buf,
        output_bufs=output_bufs,
        numerator_buf=numerator_buf,
        row_max_buf=row_max_buf,
        row_sum_buf=row_sum_buf,
        prefix_sems=prefix_sems,
        block_sem=block_sem,
        output_sems=output_sems,
        folded=folded_buf[...],
        norm=norm_buf[...],
        active_count=active_ref[0],
        n_tiles=n_tiles,
        tile=tile,
        eps=eps,
    )
    pl.loop(0, n_tiles)(functools.partial(_run_mix_norm_tile, mix_context=mix_context))
    pltpu.make_async_copy(
        output_bufs.at[(n_tiles - 1) % 2],
        output_ref.at[pl.ds((n_tiles - 1) * tile, tile), :],
        output_sems.at[(n_tiles - 1) % 2],
    ).wait()
    if n_tiles > 1:
        pltpu.make_async_copy(
            output_bufs.at[(n_tiles - 2) % 2],
            output_ref.at[pl.ds((n_tiles - 2) * tile, tile), :],
            output_sems.at[(n_tiles - 2) % 2],
        ).wait()


def mixture_norm_tiled_scratch_shapes(
    hidden_size: int, tile: int
) -> tuple[object, ...]:
    """Return scratch specs expected by ``mixture_norm_tiled_body``."""
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


def make_mixture_norm_tiled(
    *,
    seq_len: int,
    tile: int = PAGE_TOKENS,
    eps: float = 1e-5,
    interpret: bool = False,
) -> Callable[..., jax.Array]:
    """Build single-device tiled residual mixture and RMSNorm callable."""
    if seq_len <= 0 or tile <= 0 or seq_len % tile:
        msg = "seq_len must be a positive multiple of tile"
        raise ValueError(msg)

    def run(
        prefix: jax.Array,
        blocks: jax.Array,
        folded: jax.Array,
        norm: jax.Array,
        active: jax.Array,
    ) -> jax.Array:
        """Validate operand shapes and invoke ``mixture_norm_tiled_body``."""
        if len(prefix.shape) != _RAGGED_NDIM_2D or prefix.shape[0] != seq_len:
            msg = "prefix must have shape [seq_len, hidden]"
            raise ValueError(msg)
        hidden = prefix.shape[1]
        if hidden % VREG_LANE_COUNT or prefix.dtype != jnp.bfloat16:
            msg = "prefix must be BF16 with a 128-aligned width"
            raise ValueError(msg)
        if (
            blocks.shape != (_RESIDUAL_BLOCKS, seq_len, hidden)
            or blocks.dtype != jnp.bfloat16
        ):
            msg = "blocks must be BF16 with shape [8, seq_len, hidden]"
            raise ValueError(msg)
        if folded.shape != (1, hidden) or folded.dtype != jnp.float32:
            msg = "folded must be FP32 with shape [1, hidden]"
            raise ValueError(msg)
        if norm.shape != (1, hidden) or norm.dtype != jnp.bfloat16:
            msg = "norm must be BF16 with shape [1, hidden]"
            raise ValueError(msg)
        if active.shape != (1,) or active.dtype != jnp.int32:
            msg = "active must be int32 with shape [1]"
            raise ValueError(msg)
        scratch = cast(
            "pl_core.ScratchShapeTree",
            mixture_norm_tiled_scratch_shapes(hidden, tile),
        )
        call = pl.pallas_call(
            functools.partial(mixture_norm_tiled_body, tile=tile, eps=eps),
            out_shape=jax.ShapeDtypeStruct(prefix.shape, jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=[pl.BlockSpec(memory_space=pltpu.HBM) for _ in range(4)],
                out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
                scratch_shapes=scratch,
            ),
            name="mixture_norm_tiled",
            interpret=interpret,
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=_MIXTURE_VMEM_LIMIT),
        )
        return cast("jax.Array", call(active, prefix, blocks, folded, norm))

    return run


def make_mixture_norm_sp(
    mesh: jax.sharding.Mesh, *, seq_len: int, eps: float = 1e-5
) -> Callable[..., jax.Array]:
    """Build TP32 sequence-parallel tiled residual mixture and RMSNorm callable."""
    if mesh.size != TP_SIZE or len(mesh.axis_names) != 1 or seq_len % TP_SIZE:
        msg = "mixture SP requires a flat TP32 mesh and T divisible by 32"
        raise ValueError(msg)
    axis_name = mesh.axis_names[0]
    local_tokens = seq_len // TP_SIZE
    tile = min(PAGE_TOKENS, local_tokens)
    if local_tokens % tile:
        msg = "local token count must be divisible by the mixture tile"
        raise ValueError(msg)
    op = make_mixture_norm_tiled(seq_len=local_tokens, tile=tile, eps=eps)

    def local(
        prefix: jax.Array,
        blocks: jax.Array,
        folded: jax.Array,
        norm: jax.Array,
        active: jax.Array,
    ) -> jax.Array:
        """Apply tiled mixture/RMSNorm to one rank's local token slice."""
        return op(prefix[0], blocks[0], folded[0], norm[0], active)[None]

    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(axis_name), P(axis_name), P(axis_name), P(axis_name), P()),
            out_specs=P(axis_name),
            check_vma=False,
        )
    )


# ============================================================================
# KDA (sequence-tiled)
# ============================================================================


class _Copies:
    """Group of async copies started and awaited together."""

    def __init__(self, copies: Sequence[AsyncCopyProtocol]) -> None:
        """Store ``copies`` sequence."""
        self.copies = copies

    def start(self) -> None:
        """Start every async copy in ``self.copies``."""
        for copy in self.copies:
            copy.start()

    def wait(self) -> None:
        """Wait on every async copy in ``self.copies``."""
        for copy in self.copies:
            copy.wait()


@dataclasses.dataclass(frozen=True, slots=True)
class _KdaTiledCtx:
    """Shared references, scratch buffers, and configuration for tiled KDA."""

    hidden_ref: PallasRefProtocol
    rec_state_in_ref: PallasRefProtocol
    conv_hist_in_ref: PallasRefProtocol
    out_ref: PallasRefProtocol
    rec_state_out_ref: PallasRefProtocol
    conv_hist_out_ref: PallasRefProtocol
    rec_checkpoint_ref: PallasRefProtocol
    hist_checkpoint_ref: PallasRefProtocol
    partial_ref: PallasRefProtocol
    segment_ref: PallasRefProtocol
    hidden_bufs: PallasRefProtocol
    out_bufs: PallasRefProtocol
    partial_bufs: PallasRefProtocol
    gate_w_vmem: PallasRefProtocol
    fb_w_vmem: PallasRefProtocol
    proj_w_vmem: PallasRefProtocol
    conv_w_vmem: PallasRefProtocol
    alog_vmem: PallasRefProtocol
    dt_vmem: PallasRefProtocol
    norm_vmem: PallasRefProtocol
    weight_vmem: PallasRefProtocol
    rec_state_vmem: PallasRefProtocol
    conv_hist_vmem: PallasRefProtocol
    hidden_sems: PallasRefProtocol
    out_sems: PallasRefProtocol
    partial_sems: PallasRefProtocol
    weight_sems: PallasRefProtocol
    count: int
    tile: int
    block_tiles: int
    decay_lower_bound: float
    epsilon: float


def _copy_kda_pairs_now(
    kda_context: _KdaTiledCtx,
    pairs: Sequence[tuple[object, object]],
    first_semaphore: int,
) -> None:
    """Synchronously copy ``pairs`` using consecutive ``weight_sems``."""
    copies = [
        cast(
            "AsyncCopyProtocol",
            pltpu.make_async_copy(
                source, destination, kda_context.weight_sems.at[first_semaphore + i]
            ),
        )
        for i, (source, destination) in enumerate(pairs)
    ]
    for copy in copies:
        copy.start()
    for copy in copies:
        copy.wait()


def _write_kda_checkpoint(kda_context: _KdaTiledCtx, segment: jax.Array) -> None:
    """Store recurrent and conv history state to checkpoint buffers."""
    _copy_kda_pairs_now(
        kda_context,
        (
            (kda_context.rec_state_vmem, kda_context.rec_checkpoint_ref.at[segment]),
            (kda_context.conv_hist_vmem, kda_context.hist_checkpoint_ref.at[segment]),
        ),
        11,
    )


def _each_unused_kda_slot(kda_context: _KdaTiledCtx, *, wait: bool) -> None:
    """Start or wait on pass-through state copies for zero-length segments."""
    # An unused slot: its outputs are its inputs. The copies start with the
    # first tiles and are waited for after the last: one blocking round trip
    # per slot would hold up the tiles (~0.15 ms a layer at 16 slots).

    @pl.loop(0, kda_context.count)
    def visit(s: jax.Array) -> None:
        """Process unused segment slot ``s`` when its length is zero."""

        @pl.when(kda_context.segment_ref[kda_context.count + s] == 0)
        def unused() -> None:
            """Copy input states directly to output and checkpoint slots."""
            pairs = (
                (
                    kda_context.rec_state_in_ref.at[s],
                    kda_context.rec_state_out_ref.at[s],
                ),
                (
                    kda_context.conv_hist_in_ref.at[s],
                    kda_context.conv_hist_out_ref.at[s],
                ),
                (
                    kda_context.rec_state_in_ref.at[s],
                    kda_context.rec_checkpoint_ref.at[s],
                ),
                (
                    kda_context.conv_hist_in_ref.at[s],
                    kda_context.hist_checkpoint_ref.at[s],
                ),
            )
            for i, (source, target) in enumerate(pairs):
                copy = cast(
                    "AsyncCopyProtocol",
                    pltpu.make_async_copy(
                        source, target, kda_context.weight_sems.at[14 + i % 2]
                    ),
                )
                if wait:
                    copy.wait()
                else:
                    copy.start()


def _kda_load_tile(
    kda_context: _KdaTiledCtx, t: jax.Array, buf: jax.Array
) -> AsyncCopyProtocol:
    """Create async copy loading hidden tile ``t`` into ``buf``."""
    return cast(
        "AsyncCopyProtocol",
        pltpu.make_async_copy(
            kda_context.hidden_ref.at[pl.ds(t * kda_context.tile, kda_context.tile), :],
            kda_context.hidden_bufs.at[buf],
            kda_context.hidden_sems.at[buf],
        ),
    )


def _kda_store_tile(kda_context: _KdaTiledCtx, t: jax.Array, buf: jax.Array) -> _Copies:
    """Create grouped async copies storing attended and projected tile ``t``."""
    return _Copies(
        [
            cast(
                "AsyncCopyProtocol",
                pltpu.make_async_copy(
                    kda_context.out_bufs.at[buf],
                    kda_context.out_ref.at[
                        pl.ds(t * kda_context.tile, kda_context.tile), :
                    ],
                    kda_context.out_sems.at[buf],
                ),
            ),
            cast(
                "AsyncCopyProtocol",
                pltpu.make_async_copy(
                    kda_context.partial_bufs.at[buf],
                    kda_context.partial_ref.at[
                        pl.ds(t * kda_context.tile, kda_context.tile), :
                    ],
                    kda_context.partial_sems.at[buf],
                ),
            ),
        ]
    )


def _prepare_kda_tile_segment(
    kda_context: _KdaTiledCtx, t: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Identify owning sequence of tile ``t`` and load its state at segment start."""
    # The tile's sequence (-1: padding only; offsets are page-aligned, so
    # no tile holds two) and where its rows end. The state copies index
    # that one sequence: unrolled per slot, they would not fit the fused
    # layer's instruction memory at 16 slots.
    first_row = t * kda_context.tile
    segment = jnp.int32(-1)
    segment_end = jnp.int32(0)
    for s in range(kda_context.count):
        start = kda_context.segment_ref[s]
        end = start + kda_context.segment_ref[kda_context.count + s]
        inside = (first_row >= start) & (first_row < end)
        segment = jnp.where(inside, s, segment)
        segment_end = jnp.where(inside, end, segment_end)
    current = jnp.maximum(segment, 0)
    offset = kda_context.segment_ref[current]
    position = kda_context.segment_ref[2 * kda_context.count + current]

    @pl.when((segment >= 0) & (first_row == offset))
    def load_segment_state() -> None:
        """Load initial recurrent and conv states and checkpoint if position 0."""
        _copy_kda_pairs_now(
            kda_context,
            (
                (kda_context.rec_state_in_ref.at[current], kda_context.rec_state_vmem),
                (kda_context.conv_hist_in_ref.at[current], kda_context.conv_hist_vmem),
            ),
            7,
        )

        @pl.when(position == 0)
        def checkpoint_zero() -> None:
            """Write incoming state when checkpoint position is 0."""
            _write_kda_checkpoint(kda_context, current)

    return segment, segment_end, current, offset, position


def _compute_kda_tile_math(
    kda_context: _KdaTiledCtx, t: jax.Array, buf: jax.Array, row_end: jax.Array
) -> None:
    """Project, convolve, run chunked delta rule, and project output for tile ``t``."""
    width = _HEADS * _CHANNELS
    # The weights are read here: a value live across the loop would be a
    # VMEM copy (22.75 MiB for the projection).
    conv_filter = kda_context.conv_w_vmem[...].astype(jnp.float32)
    a_log = jnp.exp(kda_context.alog_vmem[...])
    dt_bias = kda_context.dt_vmem[...]
    norm = kda_context.norm_vmem[0]
    hidden_tile = kda_context.hidden_bufs.at[buf][...]

    gate_features = _dot(hidden_tile, kda_context.gate_w_vmem[...]).astype(jnp.bfloat16)
    raw_gate = (
        _dot(gate_features, kda_context.fb_w_vmem[...])
        .astype(jnp.bfloat16)
        .reshape(kda_context.tile, _HEADS, _CHANNELS)
    )
    projected = _dot(hidden_tile, kda_context.proj_w_vmem[...]).astype(jnp.bfloat16)
    qkv = projected[:, : 3 * width].reshape(kda_context.tile, 3, _HEADS, _CHANNELS)
    output_gate = projected[:, 3 * width : 4 * width].reshape(
        kda_context.tile, _HEADS, _CHANNELS
    )
    update_rate = projected[:, 4 * width : 4 * width + _HEADS]
    real_rows = (
        jnp.arange(kda_context.tile, dtype=jnp.int32) + t * kda_context.tile < row_end
    )
    raw_gate = _row_where(real_rows, raw_gate, -1e30)
    update_rate = _row_where(real_rows, update_rate, -1e30)

    window = jnp.concatenate((kda_context.conv_hist_vmem[...], qkv), axis=0)
    convolved = jnp.zeros((kda_context.tile, 3, _HEADS, _CHANNELS), jnp.float32)
    for tap in range(_CONV_TAPS):
        convolved = (
            convolved
            + window[tap : tap + kda_context.tile].astype(jnp.float32)
            * conv_filter[tap]
        )
    filtered = _row_where(real_rows, jax.nn.silu(convolved).astype(jnp.bfloat16))
    real_count = jnp.clip(row_end - t * kda_context.tile, 0, kda_context.tile)

    @pl.when(real_count == kda_context.tile)
    def carry_full_tile() -> None:
        """Carry trailing 3 tokens of full tile into conv history."""
        kda_context.conv_hist_vmem[...] = window[
            kda_context.tile : kda_context.tile + _CONV_HISTORY_LEN
        ]

    @pl.when((real_count > 0) & (real_count < kda_context.tile))
    def carry_partial_tile() -> None:
        """Carry trailing 3 valid tokens of partial tile via masked sum."""
        rows = jnp.arange(kda_context.tile + _CONV_HISTORY_LEN, dtype=jnp.int32)
        picked = [
            jnp.sum(_row_where(rows == real_count + i, window), axis=0)
            for i in range(_CONV_HISTORY_LEN)
        ]
        kda_context.conv_hist_vmem[...] = jnp.stack(picked)

    query, key = kda_normalize_query_key(
        filtered[:, 0].astype(jnp.float32),
        filtered[:, 1].astype(jnp.float32),
        _CHANNELS,
    )
    value = filtered[:, 2].astype(jnp.float32)
    decay = kda_decay(
        kda_context.decay_lower_bound, a_log[None], raw_gate, dt_bias[None]
    )
    beta = jax.nn.sigmoid(update_rate.astype(jnp.float32))

    outputs: list[jax.Array] = []
    for c in range(kda_context.tile // _KDA_CHUNK):
        s0, s1 = c * _KDA_CHUNK, (c + 1) * _KDA_CHUNK
        chunk_out, new_state = kda_chunk(
            query[s0:s1].transpose(1, 0, 2),
            key[s0:s1].transpose(1, 0, 2),
            value[s0:s1].transpose(1, 0, 2),
            decay[s0:s1].transpose(1, 0, 2),
            beta[s0:s1].transpose(1, 0),
            kda_context.rec_state_vmem[...],
        )
        kda_context.rec_state_vmem[...] = new_state
        outputs.append(chunk_out.transpose(1, 0, 2))

    attended = jnp.concatenate(outputs, axis=0).astype(jnp.bfloat16).astype(jnp.float32)
    attended *= jax.lax.rsqrt(
        jnp.mean(attended * attended, axis=-1, keepdims=True) + kda_context.epsilon
    )
    attended = (attended * norm).astype(jnp.bfloat16) * jax.nn.sigmoid(
        output_gate.astype(jnp.float32)
    ).astype(jnp.bfloat16)
    attended_2d = attended.reshape(kda_context.tile, width)
    kda_context.out_bufs.at[buf][...] = attended_2d
    columns = kda_context.weight_vmem.shape[1] // 4
    for c in range(4):
        kda_context.partial_bufs.at[buf, :, pl.ds(c * columns, columns)][...] = _dot(
            attended_2d, kda_context.weight_vmem[:, pl.ds(c * columns, columns)]
        ).astype(jnp.bfloat16)


def _run_kda_tile(t: jax.Array, *, kda_context: _KdaTiledCtx) -> None:
    """Execute load, state management, KDA math, and store for tile ``t``."""
    buf = t % _DOUBLE_BUFFER_SLOTS
    in_block = t % kda_context.block_tiles

    @pl.when(in_block == 0)
    def load_first_tile() -> None:
        """Start first tile load of the current SP block."""
        _kda_load_tile(kda_context, t, buf).start()

    _kda_load_tile(kda_context, t, buf).wait()
    segment, segment_end, current, offset, position = _prepare_kda_tile_segment(
        kda_context, t
    )

    @pl.when(in_block >= _DOUBLE_BUFFER_SLOTS)
    def wait_previous_store() -> None:
        """Wait for output stores from two tiles back in this block."""
        _kda_store_tile(kda_context, t - 2, buf).wait()

    @pl.when(in_block + 1 < kda_context.block_tiles)
    def prefetch_next_tile() -> None:
        """Prefetch next hidden tile within the current block."""
        _kda_load_tile(kda_context, t + 1, 1 - buf).start()

    _compute_kda_tile_math(kda_context, t, buf, segment_end)

    @pl.when(
        (segment >= 0)
        & (position > 0)
        & ((t + 1) * kda_context.tile == offset + position)
    )
    def checkpoint_after_tile() -> None:
        """Write sequence state when tile end matches checkpoint position."""
        _write_kda_checkpoint(kda_context, current)

    @pl.when((segment >= 0) & (t == (segment_end - 1) // kda_context.tile))
    def store_segment_state() -> None:
        """Store final recurrent and conv states at sequence end."""
        _copy_kda_pairs_now(
            kda_context,
            (
                (kda_context.rec_state_vmem, kda_context.rec_state_out_ref.at[current]),
                (kda_context.conv_hist_vmem, kda_context.conv_hist_out_ref.at[current]),
            ),
            9,
        )

    _kda_store_tile(kda_context, t, buf).start()

    @pl.when(in_block == kda_context.block_tiles - 1)
    def drain_block_stores() -> None:
        """Wait for the block's trailing tile stores."""
        _kda_store_tile(kda_context, t, buf).wait()
        if kda_context.block_tiles > 1:
            _kda_store_tile(kda_context, t - 1, 1 - buf).wait()


def kda_prefill_tiled_body(
    hidden_ref: PallasRefProtocol,
    gate_w_ref: PallasRefProtocol,
    fb_w_ref: PallasRefProtocol,
    proj_w_ref: PallasRefProtocol,
    conv_w_ref: PallasRefProtocol,
    alog_ref: PallasRefProtocol,
    dt_ref: PallasRefProtocol,
    norm_ref: PallasRefProtocol,
    rec_state_in_ref: PallasRefProtocol,
    conv_hist_in_ref: PallasRefProtocol,
    out_ref: PallasRefProtocol,
    rec_state_out_ref: PallasRefProtocol,
    conv_hist_out_ref: PallasRefProtocol,
    hidden_bufs: PallasRefProtocol,
    out_bufs: PallasRefProtocol,
    gate_w_vmem: PallasRefProtocol,
    fb_w_vmem: PallasRefProtocol,
    proj_w_vmem: PallasRefProtocol,
    conv_w_vmem: PallasRefProtocol,
    alog_vmem: PallasRefProtocol,
    dt_vmem: PallasRefProtocol,
    norm_vmem: PallasRefProtocol,
    rec_state_vmem: PallasRefProtocol,
    conv_hist_vmem: PallasRefProtocol,
    hidden_sems: PallasRefProtocol,
    out_sems: PallasRefProtocol,
    weight_sems: PallasRefProtocol,
    *,
    tile: int,
    decay_lower_bound: float,
    epsilon: float,
    checkpoint: tuple[PallasRefProtocol, PallasRefProtocol],
    segments: tuple[PallasRefProtocol, int],
    tiles: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    block_tiles: int,
    projection: tuple[
        PallasRefProtocol,
        PallasRefProtocol,
        PallasRefProtocol,
        PallasRefProtocol,
        PallasRefProtocol,
    ],
) -> None:
    """Run tiled KDA attention and output projection across ``tiles``."""
    segment_ref, count = segments
    rec_checkpoint_ref, hist_checkpoint_ref = checkpoint
    weight_ref, weight_vmem, partial_ref, partial_bufs, partial_sems = projection
    first_tile, end_tile, first, last = tiles
    weight_copies = (
        pltpu.make_async_copy(gate_w_ref, gate_w_vmem, weight_sems.at[0]),
        pltpu.make_async_copy(fb_w_ref, fb_w_vmem, weight_sems.at[1]),
        pltpu.make_async_copy(proj_w_ref, proj_w_vmem, weight_sems.at[2]),
        pltpu.make_async_copy(conv_w_ref, conv_w_vmem, weight_sems.at[3]),
        pltpu.make_async_copy(alog_ref, alog_vmem, weight_sems.at[4]),
        pltpu.make_async_copy(dt_ref, dt_vmem, weight_sems.at[5]),
        pltpu.make_async_copy(norm_ref, norm_vmem, weight_sems.at[6]),
        pltpu.make_async_copy(weight_ref, weight_vmem, weight_sems.at[13]),
    )
    kda_context = _KdaTiledCtx(
        hidden_ref=hidden_ref,
        rec_state_in_ref=rec_state_in_ref,
        conv_hist_in_ref=conv_hist_in_ref,
        out_ref=out_ref,
        rec_state_out_ref=rec_state_out_ref,
        conv_hist_out_ref=conv_hist_out_ref,
        rec_checkpoint_ref=rec_checkpoint_ref,
        hist_checkpoint_ref=hist_checkpoint_ref,
        partial_ref=partial_ref,
        segment_ref=segment_ref,
        hidden_bufs=hidden_bufs,
        out_bufs=out_bufs,
        partial_bufs=partial_bufs,
        gate_w_vmem=gate_w_vmem,
        fb_w_vmem=fb_w_vmem,
        proj_w_vmem=proj_w_vmem,
        conv_w_vmem=conv_w_vmem,
        alog_vmem=alog_vmem,
        dt_vmem=dt_vmem,
        norm_vmem=norm_vmem,
        weight_vmem=weight_vmem,
        rec_state_vmem=rec_state_vmem,
        conv_hist_vmem=conv_hist_vmem,
        hidden_sems=hidden_sems,
        out_sems=out_sems,
        partial_sems=partial_sems,
        weight_sems=weight_sems,
        count=count,
        tile=tile,
        block_tiles=block_tiles,
        decay_lower_bound=decay_lower_bound,
        epsilon=epsilon,
    )

    @pl.when(first)
    def load_weights_and_unused() -> None:
        """Load KDA weights and start unused-slot pass-through copies."""
        for c in weight_copies:
            c.start()
        for c in weight_copies:
            c.wait()
        _each_unused_kda_slot(kda_context, wait=False)

    pl.loop(first_tile, end_tile)(
        functools.partial(_run_kda_tile, kda_context=kda_context)
    )

    @pl.when(last)
    def finish_unused() -> None:
        """Wait for unused-slot pass-through state copies after last tile."""
        _each_unused_kda_slot(kda_context, wait=True)


def kda_prefill_tiled_scratch_shapes(
    hidden_size: int,
    tile: int,
    gate_w: jax.ShapeDtypeStruct,
    fb_w: jax.ShapeDtypeStruct,
    proj_w: jax.ShapeDtypeStruct,
    conv_w: jax.ShapeDtypeStruct,
    alog: jax.ShapeDtypeStruct,
    dt: jax.ShapeDtypeStruct,
    norm: jax.ShapeDtypeStruct,
) -> tuple[object, ...]:
    """Return scratch specs expected by ``kda_prefill_tiled_body``."""
    width = _HEADS * _CHANNELS
    return (
        pltpu.VMEM((2, tile, hidden_size), jnp.bfloat16),
        pltpu.VMEM((2, tile, width), jnp.bfloat16),
        pltpu.VMEM(gate_w.shape, gate_w.dtype),
        pltpu.VMEM(fb_w.shape, fb_w.dtype),
        pltpu.VMEM(proj_w.shape, proj_w.dtype),
        pltpu.VMEM(conv_w.shape, conv_w.dtype),
        pltpu.VMEM(alog.shape, alog.dtype),
        pltpu.VMEM(dt.shape, dt.dtype),
        pltpu.VMEM(norm.shape, norm.dtype),
        pltpu.VMEM((_HEADS, _CHANNELS, _CHANNELS), jnp.float32),
        pltpu.VMEM((_CONV_HISTORY_LEN, 3, _HEADS, _CHANNELS), jnp.bfloat16),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((16,)),
    )


# ============================================================================
# MLA flash attention (long windows)
# ============================================================================
#
# Each rank's 3 heads over the packed window and every rank's cached prefixes.

TILE = STORE_TILE
SCALE = 192.0**-0.5
# The online softmax runs in base 2 on queries pre-scaled by SCALE * log2(e)
# (before their BF16 rounding), so scores leave the MXU ready for exp2.
SCALE_LOG2 = SCALE * 1.4426950408889634
# The online softmax runs transposed, keys by queries (S^T = K Q^T,
# out^T = [V^T; ones] P^T): per-query statistics reduce over sublanes, and PV
# streams only the value rows (plus _ONES rows of ones for the denominators)
# against the probabilities as MXU weights. QK and softmax for FRAGMENT-query
# fragment i overlap PV for i - 1 (separate physical/query-compute tiling and
# delayed PV follow the scheduling idea in heavyball-research/flywheel-tpu
# (Apache-2.0), f646686, pallas/fwd_pipeline.py).
FRAGMENT = 512
_ONES = SUBLANE_ROWS
# Streamed blocks exponentiate against the maximum of the earlier key
# blocks; probabilities reach at most 2**STALE_LIMIT before the
# end-of-block rescale, far inside FP32/BF16 range even summed over a block.
STALE_LIMIT = 64.0
_SENTINEL_OFFSET = 1 << 30
_MLA_NDIM_3D = 3
_MLA_NDIM_4D = 4
_MLA_QK_WIDTH = 2 * HEAD_DIM
_MLA_V_WIDTH = HEAD_DIM
_FLASH_VMEM_LIMIT = 64 * MIB_BYTES


def _flash_kernel_packed(
    length_ref: PallasRefProtocol,
    segment_ref: PallasRefProtocol,
    *refs: PallasRefProtocol,
    tile: int,
    q_tile: int,
    count: int,
    prefixed: bool = False,
) -> None:
    """Compute causal packed MLA flash attention for one head and query tile."""
    if prefixed:
        (
            layer_ref,
            q_ref,
            k_hbm,
            v_hbm,
            prefix_k_hbm,
            prefix_v_hbm,
            out_ref,
            k_buf,
            v_buf,
            sems,
        ) = refs
    else:
        layer_ref = prefix_k_hbm = prefix_v_hbm = None
        q_ref, k_hbm, v_hbm, out_ref, k_buf, v_buf, sems = refs
    head, query_block = pl.program_id(0), pl.program_id(1)
    length = length_ref[0]
    offsets = [segment_ref[s] for s in range(count)]
    ends = [segment_ref[s] + segment_ref[count + s] for s in range(count)]
    first_row = query_block * q_tile
    rows = first_row + jnp.arange(q_tile)
    q = q_ref[0]
    row_start = rows + 1
    lowest = jnp.int32(_SENTINEL_OFFSET)
    interior = jnp.int32(_SENTINEL_OFFSET)
    for s in range(count):
        inside = (rows >= offsets[s]) & (rows < ends[s])
        row_start = jnp.where(inside, offsets[s], row_start)
        meets = (
            (offsets[s] < first_row + q_tile)
            & (ends[s] > first_row)
            & (ends[s] > offsets[s])
        )
        lowest = jnp.where(meets, jnp.minimum(lowest, offsets[s]), lowest)
        holds = (offsets[s] <= first_row) & (first_row + q_tile <= ends[s])
        interior = jnp.where(holds, (offsets[s] + tile - 1) // tile, interior)
    first_block = jnp.where(lowest < _SENTINEL_OFFSET, lowest // tile, 0)
    last_query_key = ((query_block + 1) * q_tile + tile - 1) // tile
    last_real_key = (length + tile - 1) // tile
    count_blocks = jnp.where(
        (first_row < length) & (lowest < _SENTINEL_OFFSET),
        jnp.minimum(last_query_key, last_real_key),
        0,
    )

    def real(indices: jax.Array) -> jax.Array:
        """Return mask of indices falling within any valid packed sequence."""
        inside = indices < 0
        for s in range(count):
            inside = inside | ((indices >= offsets[s]) & (indices < ends[s]))
        return inside

    def window_mask(key_rows: jax.Array, sl: slice) -> jax.Array:
        """Return causal intra-sequence mask for query fragment ``sl``."""
        return (key_rows <= rows[None, sl]) & (key_rows >= row_start[None, sl])

    carry = (
        jnp.full((1, q_tile), -jnp.inf, jnp.float32),
        jnp.zeros((_MLA_V_WIDTH + _ONES, q_tile), jnp.float32),
    )
    if (
        prefixed
        and layer_ref is not None
        and prefix_k_hbm is not None
        and prefix_v_hbm is not None
    ):
        carry = _prefix_phase(
            carry,
            q,
            segment_ref,
            layer_ref[0],
            head,
            first_row,
            rows,
            first_row < length,
            prefix_k_hbm,
            prefix_v_hbm,
            k_buf,
            v_buf,
            sems,
            tile=tile,
            q_tile=q_tile,
            count=count,
        )

    b = jnp.minimum(count_blocks, first_row // tile)
    a = jnp.maximum(first_block, interior)
    has_streamed = a < b
    a = jnp.where(has_streamed, a, count_blocks)
    b = jnp.where(has_streamed, b, count_blocks)
    loop = functools.partial(
        _key_loop,
        q=q,
        layer=0,
        head=head,
        k_hbm=k_hbm,
        v_hbm=v_hbm,
        k_buf=k_buf,
        v_buf=v_buf,
        sems=sems,
        tile=tile,
    )
    carry = loop(carry, first_block, a, mask=window_mask, value_mask=real)
    carry = loop(carry, a, b, streamed=True)
    carry = loop(carry, b, count_blocks, mask=window_mask, value_mask=real)
    _, accumulator = carry
    denominator = accumulator[_MLA_V_WIDTH : _MLA_V_WIDTH + 1]
    safe = jnp.where(denominator > 0, denominator, 1.0)
    out_ref[0] = (accumulator[:_MLA_V_WIDTH] / safe).T.astype(jnp.bfloat16)


def _prefix_phase(
    carry: tuple[jax.Array, jax.Array],
    q: jax.Array,
    segment_ref: PallasRefProtocol,
    layer: jax.Array | int,
    head: jax.Array,
    first_row: jax.Array,
    rows: jax.Array,
    live: jax.Array,
    k_hbm: PallasRefProtocol,
    v_hbm: PallasRefProtocol,
    k_buf: PallasRefProtocol,
    v_buf: PallasRefProtocol,
    sems: PallasRefProtocol,
    *,
    tile: int,
    q_tile: int,
    count: int,
) -> tuple[jax.Array, jax.Array]:
    """Fold cached prefix arena keys/values into online softmax ``carry``."""
    offsets = [segment_ref[s] for s in range(count)]
    ends = [segment_ref[s] + segment_ref[count + s] for s in range(count)]
    row_sequence = jnp.full(rows.shape, -1, jnp.int32)
    for s in range(count):
        row_sequence = jnp.where(
            (rows >= offsets[s]) & (rows < ends[s]), s, row_sequence
        )

    # A rolled sequence loop avoids emitting a full attention body for every
    # 256-row query slice, including slices that repeat the same sequence.
    def visit(
        sequence: jax.Array, seq_carry: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        """Attend cached prefix blocks belonging to ``sequence``."""
        offset = segment_ref[sequence]
        end = offset + segment_ref[count + sequence]
        base = segment_ref[3 * count + sequence]
        size = jnp.where(
            live & (offset < first_row + q_tile) & (end > first_row),
            segment_ref[4 * count + sequence],
            0,
        )
        lo_row, hi_row = base, base + size
        lo = base // tile
        hi = jnp.where(size > 0, (base + size + tile - 1) // tile, lo)

        def mask(key_rows: jax.Array, sl: slice) -> jax.Array:
            """Restrict prefix keys to ``[lo_row, hi_row)`` on ``sequence`` rows."""
            return (
                (row_sequence[None, sl] == sequence)
                & (key_rows >= lo_row)
                & (key_rows < hi_row)
            )

        def value_mask(value_rows: jax.Array) -> jax.Array:
            """Select valid cached value rows within ``[lo_row, hi_row)``."""
            return (value_rows >= lo_row) & (value_rows < hi_row)

        loop = functools.partial(
            _key_loop,
            q=q,
            layer=layer,
            head=head,
            k_hbm=k_hbm,
            v_hbm=v_hbm,
            k_buf=k_buf,
            v_buf=v_buf,
            sems=sems,
            tile=tile,
        )
        a, b = (lo_row + tile - 1) // tile, hi_row // tile
        a, b = jnp.where(a < b, a, hi), jnp.where(a < b, b, hi)

        def edge(
            i: jax.Array, edge_carry: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            """Process leading or trailing partial prefix key block."""
            return loop(
                edge_carry,
                jnp.where(i == 0, lo, b),
                jnp.where(i == 0, a, hi),
                mask=mask,
                value_mask=value_mask,
            )

        seq_carry = jax.lax.fori_loop(0, 2, edge, seq_carry)
        streamed = loop(seq_carry, a, b, streamed=True)
        own = row_sequence[None] == sequence
        return (
            _mask_where(own, streamed[0], seq_carry[0]),
            _mask_where(own, streamed[1], seq_carry[1]),
        )

    # Visit only the sequences that meet the tile (contiguous: offsets
    # ascend); any other visit is a no-op, but one per slot and query tile
    # costs ~0.6 ms a layer at 16 slots.
    first, end = jnp.int32(count), jnp.int32(0)
    for s in range(count):
        meets = live & (offsets[s] < first_row + q_tile) & (ends[s] > first_row)
        first = jnp.where(meets, jnp.minimum(first, s), first)
        end = jnp.where(meets, jnp.maximum(end, s + 1), end)
    return jax.lax.fori_loop(first, jnp.maximum(first, end), visit, carry)


@dataclasses.dataclass(frozen=True, slots=True)
class _KeyLoopCtx:
    """Shared buffers and masks for ``_key_loop`` online softmax blocks."""

    q: jax.Array
    lo: jax.Array | int
    hi: jax.Array | int
    layer: jax.Array | int
    head: jax.Array
    k_hbm: PallasRefProtocol
    v_hbm: PallasRefProtocol
    k_buf: PallasRefProtocol
    v_buf: PallasRefProtocol
    sems: PallasRefProtocol
    tile: int
    mask: Callable[[jax.Array, slice], jax.Array] | None
    value_mask: Callable[[jax.Array], jax.Array] | None


def _key_loop_copies(
    loop_context: _KeyLoopCtx, key_block: jax.Array | int, slot: jax.Array | int
) -> tuple[AsyncCopyProtocol, AsyncCopyProtocol]:
    """Create K and V tile async copies for ``key_block`` into ``slot``."""
    return (
        cast(
            "AsyncCopyProtocol",
            pltpu.make_async_copy(
                loop_context.k_hbm.at[
                    loop_context.layer,
                    loop_context.head,
                    pl.ds(key_block * loop_context.tile, loop_context.tile),
                    :,
                ],
                loop_context.k_buf.at[slot],
                loop_context.sems.at[0, slot],
            ),
        ),
        cast(
            "AsyncCopyProtocol",
            pltpu.make_async_copy(
                loop_context.v_hbm.at[
                    loop_context.layer,
                    loop_context.head,
                    :,
                    pl.ds(key_block * loop_context.tile, loop_context.tile),
                ],
                loop_context.v_buf.at[slot],
                loop_context.sems.at[1, slot],
            ),
        ),
    )


def _load_key_value_block(
    loop_context: _KeyLoopCtx, key_block: jax.Array | int
) -> tuple[jax.Array, jax.Array]:
    """Wait for ``key_block`` K/V copies, prefetch next block, and pad ones."""
    slots = loop_context.k_buf.shape[0]
    slot = key_block % slots
    for copy in _key_loop_copies(loop_context, key_block, slot):
        copy.wait()

    @pl.when(key_block + 1 < loop_context.hi)
    def prefetch() -> None:
        """Start K/V copies for ``key_block + 1``."""
        for copy in _key_loop_copies(
            loop_context, key_block + 1, (key_block + 1) % slots
        ):
            copy.start()

    k_val, v_val = loop_context.k_buf[slot], loop_context.v_buf[slot]
    if loop_context.value_mask is not None:
        value_keys = key_block * loop_context.tile + jax.lax.broadcasted_iota(
            jnp.int32, (1, loop_context.tile), 1
        )
        v_val = jnp.where(
            loop_context.value_mask(value_keys), v_val, jnp.zeros((), v_val.dtype)
        )
    return k_val, jnp.concatenate(
        (v_val, jnp.ones((_ONES, loop_context.tile), v_val.dtype)), axis=0
    )


def _exact_key_loop_range(
    loop_context: _KeyLoopCtx, carry: tuple[jax.Array, jax.Array]
) -> tuple[jax.Array, jax.Array]:
    """Fold ``[loop_context.lo, loop_context.hi)`` using per-block running maxima."""
    slots = loop_context.k_buf.shape[0]

    @pl.when(loop_context.lo < loop_context.hi)
    def prime() -> None:
        """Prefetch initial key/value block ``loop_context.lo``."""
        for copy in _key_loop_copies(
            loop_context, loop_context.lo, loop_context.lo % slots
        ):
            copy.start()

    def step(
        key_block: jax.Array, c: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        """Fold ``key_block`` with exact maximum update."""
        k_val, v_val = _load_key_value_block(loop_context, key_block)
        key_rows = key_block * loop_context.tile + jax.lax.broadcasted_iota(
            jnp.int32, (loop_context.tile, 1), 0
        )
        block_mask = (
            None
            if loop_context.mask is None
            else functools.partial(loop_context.mask, key_rows)
        )
        return _exact_step(loop_context.q, k_val, v_val, c, block_mask)

    return jax.lax.fori_loop(loop_context.lo, loop_context.hi, step, carry)


def _key_loop(
    carry: tuple[jax.Array, jax.Array],
    lo: jax.Array | int,
    hi: jax.Array | int,
    *,
    q: jax.Array,
    layer: jax.Array | int,
    head: jax.Array,
    k_hbm: PallasRefProtocol,
    v_hbm: PallasRefProtocol,
    k_buf: PallasRefProtocol,
    v_buf: PallasRefProtocol,
    sems: PallasRefProtocol,
    tile: int,
    mask: Callable[[jax.Array, slice], jax.Array] | None = None,
    value_mask: Callable[[jax.Array], jax.Array] | None = None,
    streamed: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Fold key/value blocks ``[lo, hi)`` into transposed online softmax ``carry``."""
    loop_context = _KeyLoopCtx(
        q=q,
        lo=lo,
        hi=hi,
        layer=layer,
        head=head,
        k_hbm=k_hbm,
        v_hbm=v_hbm,
        k_buf=k_buf,
        v_buf=v_buf,
        sems=sems,
        tile=tile,
        mask=mask,
        value_mask=value_mask,
    )
    if not streamed:
        return _exact_key_loop_range(loop_context, carry)

    def stream(c: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        """Stream blocks past ``lo`` using carried maxima with exact fallback."""
        slots = k_buf.shape[0]
        for copy in _key_loop_copies(loop_context, lo, lo % slots):
            copy.start()
        k0, v0 = _load_key_value_block(loop_context, lo)
        first = _exact_step(q, k0, v0, c, None)

        def body(
            key_block: jax.Array,
            state: tuple[tuple[jax.Array, jax.Array], jax.Array],
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            """Fold ``key_block`` against stale running maximum and track gap."""
            cur_c, worst = state
            k_val, v_val = _load_key_value_block(loop_context, key_block)
            next_c, gap = _stale_step(q, k_val, v_val, cur_c)
            return next_c, jnp.maximum(worst, gap)

        worst_init = jnp.full(c[0].shape, -jnp.inf, jnp.float32)
        result, worst = jax.lax.fori_loop(lo + 1, hi, body, (first, worst_init))

        def keep_streamed(
            _: tuple[jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array]:
            """Return streamed carry when maximum drift stayed within bound."""
            return result

        return jax.lax.cond(
            jnp.max(worst) > STALE_LIMIT,
            functools.partial(_exact_key_loop_range, loop_context),
            keep_streamed,
            c,
        )

    def unchanged(c: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        """Return ``c`` unchanged when ``lo >= hi``."""
        return c

    return jax.lax.cond(lo < hi, stream, unchanged, carry)


def _exact_step(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    carry: tuple[jax.Array, jax.Array],
    mask: Callable[[slice], jax.Array] | None,
) -> tuple[jax.Array, jax.Array]:
    """Fold one key block into ``carry`` updating per-query maxima before exp2."""
    maximum, accumulator = carry
    fragment = min(FRAGMENT, q.shape[0])
    maxima: list[jax.Array] = []
    accumulators: list[jax.Array] = []
    previous: tuple[slice, jax.Array, jax.Array] | None = None

    def fold(sl: slice, probabilities: jax.Array, correction: jax.Array) -> None:
        """Contract ``v`` with ``probabilities`` and rescale ``accumulator``."""
        weighted = jax.lax.dot_general(
            v,
            probabilities,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        accumulators.append(accumulator[:, sl] * correction + weighted)

    for qi in range(q.shape[0] // fragment):
        sl = slice(qi * fragment, (qi + 1) * fragment)
        scores = jax.lax.dot_general(
            k, q[sl], (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        )
        if mask is not None:
            scores = jnp.where(mask(sl), scores, -jnp.inf)
        new_maximum = jnp.maximum(
            maximum[:, sl], jnp.max(scores, axis=0, keepdims=True)
        )
        # A query with no key yet keeps a -inf maximum; exponentiate against 0.
        safe_maximum = jnp.where(new_maximum == -jnp.inf, 0.0, new_maximum)
        probabilities = jnp.exp2(scores - safe_maximum).astype(jnp.bfloat16)
        maxima.append(new_maximum)
        if previous is not None:
            fold(*previous)
        previous = (sl, probabilities, jnp.exp2(maximum[:, sl] - safe_maximum))
    if previous is None:
        msg = "Query tile must contain at least one fragment"
        raise ValueError(msg)
    fold(*previous)
    return jnp.concatenate(maxima, axis=1), jnp.concatenate(accumulators, axis=1)


def _stale_step(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    carry: tuple[jax.Array, jax.Array],
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """Fold one unmasked key block exponentiating against carried maxima."""
    maximum, accumulator = carry
    fragment = min(FRAGMENT, q.shape[0])
    maxima: list[jax.Array] = []
    accumulators: list[jax.Array] = []
    gaps: list[jax.Array] = []
    previous: tuple[slice, jax.Array, jax.Array, jax.Array] | None = None

    def fold(
        sl: slice,
        probabilities: jax.Array,
        reference: jax.Array,
        new_maximum: jax.Array,
    ) -> None:
        """Contract ``v`` with ``probabilities`` and apply post-block rescale."""
        weighted = jax.lax.dot_general(
            v,
            probabilities,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        accumulators.append(
            (accumulator[:, sl] + weighted) * jnp.exp2(reference - new_maximum)
        )

    for qi in range(q.shape[0] // fragment):
        sl = slice(qi * fragment, (qi + 1) * fragment)
        scores = jax.lax.dot_general(
            k, q[sl], (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        )
        reference = maximum[:, sl]
        block_maximum = jnp.max(scores, axis=0, keepdims=True)
        probabilities = jnp.exp2(scores - reference).astype(jnp.bfloat16)
        new_maximum = jnp.maximum(reference, block_maximum)
        maxima.append(new_maximum)
        gaps.append(block_maximum - reference)
        if previous is not None:
            fold(*previous)
        previous = (sl, probabilities, reference, new_maximum)
    if previous is None:
        msg = "Query tile must contain at least one fragment"
        raise ValueError(msg)
    fold(*previous)
    next_carry = (
        jnp.concatenate(maxima, axis=1),
        jnp.concatenate(accumulators, axis=1),
    )
    return next_carry, jnp.concatenate(gaps, axis=1)


class CpPrefix(NamedTuple):
    """Context-parallel cached prefix pools, page IDs, and local sequence spans."""

    pools: tuple[jax.Array | jax.ShapeDtypeStruct, ...]
    ids: jax.Array | jax.ShapeDtypeStruct
    local: jax.Array | jax.ShapeDtypeStruct

    def split(self) -> tuple[CpPrefix, ...]:
        """Split ``pools`` into one single-layer ``CpPrefix`` per MLA layer."""
        return tuple(CpPrefix((pool,), self.ids, self.local) for pool in self.pools)


GATHER_STREAMS = 6
GATHER_PARTS = 4


@dataclasses.dataclass(frozen=True, slots=True)
class _PageGatherCtx:
    """Shared buffers, topology tables, and rank metadata for page all-gather."""

    tables: PallasRefProtocol
    counts_ref: PallasRefProtocol
    out_hbm: PallasRefProtocol
    send_sems: PallasRefProtocol
    recv_sems: PallasRefProtocol
    rank: jax.Array
    rows: int
    slots: int
    positions: list[jax.Array]
    streams: list[tuple[int, int, int, jax.Array]]


def _page_gather_chunks(
    gather_context: _PageGatherCtx, owner: jax.Array, stream: int
) -> jax.Array:
    """Return chunk count owned by ``owner`` on ``stream``."""
    return (
        jnp.maximum(
            gather_context.counts_ref[owner] * GATHER_PARTS
            - stream
            + GATHER_STREAMS
            - 1,
            0,
        )
        // GATHER_STREAMS
    )


def _page_gather_remote(
    gather_context: _PageGatherCtx,
    owner: jax.Array,
    stream: int,
    idx: jax.Array,
    hop_and_peer: tuple[jax.Array | int, jax.Array],
) -> AsyncCopyProtocol:
    """Build remote chunk transfer for ``(owner, stream, idx)``."""
    hop, peer = hop_and_peer
    c = stream + GATHER_STREAMS * idx
    src = gather_context.out_hbm.at[
        owner * gather_context.slots + 1 + c // GATHER_PARTS,
        pl.ds((c % GATHER_PARTS) * gather_context.rows, gather_context.rows),
    ]
    return cast(
        "AsyncCopyProtocol",
        pltpu.make_async_remote_copy(
            src,
            src,
            gather_context.send_sems.at[stream],
            gather_context.recv_sems.at[stream, hop],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        ),
    )


def _step_page_gather_hop(hop: jax.Array, *, gather_context: _PageGatherCtx) -> None:
    """Receive and forward all six streams' page chunks at ring ``hop``."""
    for stream, family, direction, downstream in gather_context.streams:
        owner = gather_context.tables[
            family, 1, (gather_context.positions[family] - direction * hop) % TP_SIZE
        ]
        size = _page_gather_chunks(gather_context, owner, stream)

        @pl.loop(0, size)
        def receive(i: jax.Array, *, st: int = stream, own: jax.Array = owner) -> None:
            """Wait for incoming remote chunk ``i``."""
            _page_gather_remote(
                gather_context, own, st, i, (hop, gather_context.rank)
            ).wait_recv()

        @pl.when(hop < TP_PEERS)
        def forward(
            *,
            st: int = stream,
            own: jax.Array = owner,
            sz: jax.Array = size,
            dst: jax.Array = downstream,
        ) -> None:
            """Forward received chunks one hop downstream."""

            @pl.loop(0, sz)
            def send(i: jax.Array) -> None:
                """Start downstream remote copy for chunk ``i``."""
                _page_gather_remote(gather_context, own, st, i, (hop + 1, dst)).start()


def _drain_page_gather_hop(hop: jax.Array, *, gather_context: _PageGatherCtx) -> None:
    """Wait on send completions for chunks transmitted at ``hop + 1``."""
    for stream, family, direction, downstream in gather_context.streams:
        owner = gather_context.tables[
            family, 1, (gather_context.positions[family] - direction * hop) % TP_SIZE
        ]

        @pl.loop(0, _page_gather_chunks(gather_context, owner, stream))
        def sent(
            i: jax.Array,
            *,
            st: int = stream,
            own: jax.Array = owner,
            dst: jax.Array = downstream,
        ) -> None:
            """Wait for send completion on chunk ``i``."""
            _page_gather_remote(gather_context, own, st, i, (hop + 1, dst)).wait_send()


def _prefix_page_gather_kernel(
    tables: PallasRefProtocol,
    counts_ref: PallasRefProtocol,
    ids_ref: PallasRefProtocol,
    pool_hbm: PallasRefProtocol,
    out_hbm: PallasRefProtocol,
    local_sem: PallasRefProtocol,
    send_sems: PallasRefProtocol,
    recv_sems: PallasRefProtocol,
    done: PallasRefProtocol,
    *,
    slots: int,
) -> None:
    """All-gather every rank's valid cached prefix pages over six TP32 rings."""
    rows = out_hbm.shape[1] // GATHER_PARTS
    rank = lax.axis_index(AXIS_SP)
    positions = [tables[f, 0, rank] for f in range(3)]
    neighbours = [
        tables[f, 1, (positions[f] + d) % TP_SIZE] for f in range(3) for d in (-1, 1)
    ]
    count = counts_ref[rank]

    def local_copy(q: jax.Array) -> AsyncCopyProtocol:
        """Create local HBM copy placing owned page ``q`` into ``out_hbm``."""
        return cast(
            "AsyncCopyProtocol",
            pltpu.make_async_copy(
                pool_hbm.at[ids_ref[1 + q]],
                out_hbm.at[rank * slots + 1 + q],
                local_sem,
            ),
        )

    @pl.loop(0, count)
    def start_local(q: jax.Array) -> None:
        """Start local copy for page ``q``."""
        local_copy(q).start()

    barrier = pltpu.get_barrier_semaphore()
    for peer in neighbours:
        pl.semaphore_signal(
            barrier, 1, device_id=(peer,), device_id_type=pl.DeviceIdType.MESH
        )
    pl.semaphore_wait(barrier, len(neighbours))

    @pl.loop(0, count)
    def finish_local(q: jax.Array) -> None:
        """Wait for local copy of page ``q``."""
        local_copy(q).wait()

    streams = [
        (
            st,
            st // 2 % 3,
            -1 if st % 2 == 0 else 1,
            tables[
                st // 2 % 3,
                1,
                (positions[st // 2 % 3] + (-1 if st % 2 == 0 else 1)) % TP_SIZE,
            ],
        )
        for st in range(GATHER_STREAMS)
    ]
    gather_context = _PageGatherCtx(
        tables=tables,
        counts_ref=counts_ref,
        out_hbm=out_hbm,
        send_sems=send_sems,
        recv_sems=recv_sems,
        rank=rank,
        rows=rows,
        slots=slots,
        positions=positions,
        streams=streams,
    )
    for stream, _, _, downstream in streams:

        @pl.loop(0, _page_gather_chunks(gather_context, rank, stream))
        def seed(
            i: jax.Array, *, st: int = stream, dst: jax.Array = downstream
        ) -> None:
            """Start hop-1 send of owned chunk ``i``."""
            _page_gather_remote(gather_context, rank, st, i, (1, dst)).start()

    pl.loop(1, TP_SIZE)(
        functools.partial(_step_page_gather_hop, gather_context=gather_context)
    )
    pl.loop(0, TP_PEERS)(
        functools.partial(_drain_page_gather_hop, gather_context=gather_context)
    )
    for peer in neighbours:
        pl.semaphore_signal(
            done, 1, device_id=(peer,), device_id_type=pl.DeviceIdType.MESH
        )
    pl.semaphore_wait(done, len(neighbours))


def gathered_prefix_pages(
    pool: jax.Array,
    ids: jax.Array,
    local: jax.Array,
    tables: np.ndarray,
    axis_name: str,
    *,
    count: int,
    capacity: int,
) -> tuple[jax.Array, jax.Array]:
    """All-gather cached prefix pages across ``axis_name`` and compute arena order."""
    page_tokens = pool.shape[-3] * pool.shape[-2]
    slots = ids.shape[0]
    every = jax.lax.all_gather(local, axis_name)
    ranks = every.shape[0]
    counts = jnp.sum(every[:, count:], axis=1) // page_tokens
    pages = cast(
        "jax.Array",
        pl.pallas_call(
            functools.partial(_prefix_page_gather_kernel, slots=slots),
            out_shape=jax.ShapeDtypeStruct(
                (ranks * slots, *pool.shape[1:]), pool.dtype
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=3,
                in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
                out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
                scratch_shapes=(
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA((GATHER_STREAMS,)),
                    pltpu.SemaphoreType.DMA((GATHER_STREAMS, ranks)),
                    pltpu.SemaphoreType.REGULAR,
                ),
            ),
            compiler_params=pltpu.CompilerParams(collective_id=44),
            name="kimi_prefill_prefix_page_gather",
        )(
            jnp.asarray(tables, jnp.int32),
            counts.astype(jnp.int32),
            ids.astype(jnp.int32),
            pool,
        ),
    )
    return pages, prefix_page_order(
        every, count=count, page_tokens=page_tokens, slots=slots, capacity=capacity
    )


def prefix_page_order(
    every: jax.Array,
    *,
    count: int,
    page_tokens: int,
    slots: int,
    capacity: int,
) -> jax.Array:
    """Build ``[1 + capacity / page_tokens]`` gathered-slot order for prefix arena."""
    ranks = every.shape[0]
    starts = (every[:, :count] // page_tokens).T.reshape(-1)
    sizes = (every[:, count:] // page_tokens).T.reshape(-1)
    run_rank = jnp.tile(jnp.arange(ranks, dtype=jnp.int32), count)
    ends = jnp.cumsum(sizes)
    if capacity % page_tokens:
        msg = f"the prefix capacity must be whole {page_tokens}-token pages"
        raise ValueError(msg)
    arena = jnp.arange(capacity // page_tokens, dtype=jnp.int32)
    run = jnp.minimum(
        jnp.sum(arena[:, None] >= ends[None, :], axis=1), sizes.shape[0] - 1
    )
    within = arena - (ends[run] - sizes[run])
    source = run_rank[run] * slots + 1 + starts[run] + within
    total = ends[-1]
    order = jnp.where(arena < total, source, 0).astype(jnp.int32)
    return jnp.concatenate((total[None].astype(jnp.int32), order))


EXPAND_PAGES = 4


@dataclasses.dataclass(frozen=True, slots=True)
class _ExpandPagesCtx:
    """References and shapes for ``_expand_pages_kernel``."""

    order_ref: PallasRefProtocol
    pages_hbm: PallasRefProtocol
    up_ref: PallasRefProtocol
    keys_hbm: PallasRefProtocol
    values_hbm: PallasRefProtocol
    buf: PallasRefProtocol
    words: PallasRefProtocol
    kbuf: PallasRefProtocol
    vbuf: PallasRefProtocol
    sems: PallasRefProtocol
    total: jax.Array
    steps: jax.Array
    pages: int
    page_rows: int
    page_tokens: int
    lanes: int
    span: int
    heads: int


def _expand_page_reads(
    expand_context: _ExpandPagesCtx, c: jax.Array | int, slot: jax.Array | int
) -> list[AsyncCopyProtocol]:
    """Create HBM page read copies for batch ``c`` into ``slot``."""
    return [
        cast(
            "AsyncCopyProtocol",
            pltpu.make_async_copy(
                expand_context.pages_hbm.at[
                    expand_context.order_ref[1 + c * expand_context.pages + j]
                ],
                expand_context.buf.at[slot, j],
                expand_context.sems.at[0, slot],
            ),
        )
        for j in range(expand_context.pages)
    ]


def _expand_page_writes(
    expand_context: _ExpandPagesCtx, c: jax.Array | int, slot: jax.Array | int
) -> list[AsyncCopyProtocol]:
    """Create HBM arena key/value write copies for batch ``c`` from ``slot``."""
    rows = pl.ds(c * expand_context.span, expand_context.span)
    return [
        cast(
            "AsyncCopyProtocol",
            pltpu.make_async_copy(
                expand_context.kbuf.at[slot],
                expand_context.keys_hbm.at[:, rows],
                expand_context.sems.at[1, slot],
            ),
        ),
        cast(
            "AsyncCopyProtocol",
            pltpu.make_async_copy(
                expand_context.vbuf.at[slot],
                expand_context.values_hbm.at[:, :, rows],
                expand_context.sems.at[2, slot],
            ),
        ),
    ]


def _step_expand_pages(
    c: jax.Array, carry: jax.Array, *, expand_context: _ExpandPagesCtx
) -> jax.Array:
    """Up-project ``expand_context.pages`` cached pages at step ``c`` into K/V arena."""
    slot = c % _DOUBLE_BUFFER_SLOTS
    for copy in _expand_page_reads(expand_context, c, slot):
        copy.wait()

    @pl.when(c + 1 < expand_context.steps)
    def prefetch() -> None:
        """Prefetch next batch of gathered pages."""
        for copy in _expand_page_reads(expand_context, c + 1, 1 - slot):
            copy.start()

    @pl.when(c >= _DOUBLE_BUFFER_SLOTS)
    def reuse() -> None:
        """Wait for arena writes from two batches back."""
        for copy in _expand_page_writes(expand_context, c - 2, slot):
            copy.wait()

    for j in range(expand_context.pages):
        # Token order within a page is free (a page never spans sequences).
        x = pltpu.bitcast(
            expand_context.words.at[slot, j].reshape(
                expand_context.page_rows, expand_context.lanes
            )[...],
            expand_context.buf.dtype,
        ).astype(jnp.bfloat16)
        # Mask out trailing pages; stale slots may hold NaN.
        x = jnp.where(
            c * expand_context.pages + j < expand_context.total,
            x,
            jnp.zeros((), x.dtype),
        )
        rows = pl.ds(j * expand_context.page_tokens, expand_context.page_tokens)
        rope = x[:, 512:576]
        for h in range(expand_context.heads):
            expanded = jnp.dot(
                x[:, :512],
                expand_context.up_ref[:, h * _MLA_QK_WIDTH : (h + 1) * _MLA_QK_WIDTH],
                preferred_element_type=jnp.float32,
            ).astype(jnp.bfloat16)
            expand_context.kbuf[slot, h, rows, :] = jnp.concatenate(
                (
                    expanded[:, :_MLA_V_WIDTH],
                    rope,
                    jnp.zeros((expand_context.page_tokens, 64), jnp.bfloat16),
                ),
                axis=-1,
            )
            expand_context.vbuf[slot, h, :, rows] = expanded[:, _MLA_V_WIDTH:].T
    for copy in _expand_page_writes(expand_context, c, slot):
        copy.start()
    return carry


def _expand_pages_kernel(
    order_ref: PallasRefProtocol,
    pages_hbm: PallasRefProtocol,
    up_ref: PallasRefProtocol,
    keys_hbm: PallasRefProtocol,
    values_hbm: PallasRefProtocol,
    buf: PallasRefProtocol,
    kbuf: PallasRefProtocol,
    vbuf: PallasRefProtocol,
    sems: PallasRefProtocol,
    *,
    heads: int,
) -> None:
    """Expand gathered compressed KV pages into head-major K and V arenas."""
    total = order_ref[0]
    _, pages, page_rows, packing, lanes = buf.shape
    page_tokens = page_rows * packing
    span = pages * page_tokens
    steps = (total + pages - 1) // pages
    expand_context = _ExpandPagesCtx(
        order_ref=order_ref,
        pages_hbm=pages_hbm,
        up_ref=up_ref,
        keys_hbm=keys_hbm,
        values_hbm=values_hbm,
        buf=buf,
        words=buf.bitcast(jnp.uint32),
        kbuf=kbuf,
        vbuf=vbuf,
        sems=sems,
        total=total,
        steps=steps,
        pages=pages,
        page_rows=page_rows,
        page_tokens=page_tokens,
        lanes=lanes,
        span=span,
        heads=heads,
    )

    @pl.when(steps > 0)
    def prime() -> None:
        """Start first batch of page reads."""
        for copy in _expand_page_reads(expand_context, 0, 0):
            copy.start()

    jax.lax.fori_loop(
        0,
        steps,
        functools.partial(_step_expand_pages, expand_context=expand_context),
        jnp.int32(0),
    )

    @pl.when(steps >= _DOUBLE_BUFFER_SLOTS)
    def drain_previous() -> None:
        """Wait for penultimate batch writes."""
        for copy in _expand_page_writes(expand_context, steps - 2, (steps - 2) % 2):
            copy.wait()

    @pl.when(steps >= 1)
    def drain_last() -> None:
        """Wait for final batch writes."""
        for copy in _expand_page_writes(expand_context, steps - 1, (steps - 1) % 2):
            copy.wait()


def expand_prefix_pages(
    pages: jax.Array, order: jax.Array, key_value_up: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Expand gathered prefix pages into a single-layer ``(keys, values)`` arena."""
    heads = key_value_up.shape[1] // _MLA_QK_WIDTH
    page_rows, packing, lanes = pages.shape[1:]
    page_tokens = page_rows * packing
    span = EXPAND_PAGES * page_tokens
    capacity = (order.shape[0] - 1) * page_tokens
    if capacity % span:
        msg = f"the prefix capacity must be a multiple of {span} tokens"
        raise ValueError(msg)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    keys, values = cast(
        "tuple[jax.Array, jax.Array]",
        pl.pallas_call(
            functools.partial(_expand_pages_kernel, heads=heads),
            out_shape=(
                jax.ShapeDtypeStruct((heads, capacity, _MLA_QK_WIDTH), jnp.bfloat16),
                jax.ShapeDtypeStruct((heads, _MLA_V_WIDTH, capacity), jnp.bfloat16),
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=[hbm, pl.BlockSpec(memory_space=pltpu.VMEM)],
                out_specs=(hbm, hbm),
                scratch_shapes=(
                    pltpu.VMEM(
                        (2, EXPAND_PAGES, page_rows, packing, lanes), pages.dtype
                    ),
                    pltpu.VMEM((2, heads, span, _MLA_QK_WIDTH), jnp.bfloat16),
                    pltpu.VMEM((2, heads, _MLA_V_WIDTH, span), jnp.bfloat16),
                    pltpu.SemaphoreType.DMA((3, 2)),
                ),
            ),
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=_FLASH_VMEM_LIMIT),
            name="kimi_prefill_expand_prefix_pages",
        )(order.astype(jnp.int32), pages, key_value_up),
    )
    return keys[None], values[None]


def mla_flash_prefill_packed(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    length: jax.Array | int,
    segments: jax.Array,
    *,
    count: int,
    tile: int = TILE,
    arena: tuple[jax.Array, jax.Array] | None = None,
    layer: jax.Array | int = 0,
) -> jax.Array:
    """Compute causal packed MLA attention over ``count`` sequences per head."""
    if query.ndim != _MLA_NDIM_3D or query.shape[2] != _MLA_QK_WIDTH:
        msg = "query must be [H,T,256]"
        raise ValueError(msg)
    heads, tokens = query.shape[:2]
    if heads != _HEADS:
        msg = "the window phase takes this rank's 3 heads"
        raise ValueError(msg)
    if key.shape != (_HEADS, tokens, _MLA_QK_WIDTH) or value.shape != (
        _HEADS,
        tokens,
        _MLA_V_WIDTH,
    ):
        msg = "packed MLA keys/values are the window's rows"
        raise ValueError(msg)
    fields = 3 if arena is None else 5
    if segments.shape != (fields * count,):
        msg = f"segments must be [{fields} * count]"
        raise ValueError(msg)
    length_arr = jnp.asarray(length, jnp.int32).reshape((1,))
    q_tile, k_tile = _tiles(tokens, tokens, tile)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    def q_index_map(
        h: jax.Array, i: jax.Array, *_args: object
    ) -> tuple[jax.Array, jax.Array, int]:
        """Select head ``h`` and query block ``i``."""
        return (h, i, 0)

    scalars: tuple[jax.Array, ...] = (length_arr, segments.astype(jnp.int32))
    operands: tuple[jax.Array, ...] = (
        query,
        key[None],
        jnp.swapaxes(value, 1, 2)[None],
    )
    in_specs: tuple[pl.BlockSpec, ...] = (
        pl.BlockSpec((1, q_tile, _MLA_QK_WIDTH), q_index_map),
        hbm,
        hbm,
    )
    if arena is not None:
        arena_keys, arena_values = arena
        if (
            arena_keys.ndim != _MLA_NDIM_4D
            or arena_keys.shape[1:] != (heads, arena_keys.shape[2], _MLA_QK_WIDTH)
            or arena_values.shape
            != (*arena_keys.shape[:2], _MLA_V_WIDTH, arena_keys.shape[2])
            or arena_keys.shape[2] % k_tile
        ):
            msg = (
                f"arena must be [L,{heads},C,256] / [L,{heads},128,C] "
                f"with C a multiple of {k_tile}"
            )
            raise ValueError(msg)
        scalars += (jnp.asarray(layer, jnp.int32).reshape((1,)),)
        operands += (arena_keys, arena_values)
        in_specs += (hbm, hbm)
    name = "kimi_mla_flash_prefill_packed" + ("_prefixed" if arena is not None else "")
    return cast(
        "jax.Array",
        pl.pallas_call(
            functools.partial(
                _flash_kernel_packed,
                tile=k_tile,
                q_tile=q_tile,
                count=count,
                prefixed=arena is not None,
            ),
            out_shape=jax.ShapeDtypeStruct((heads, tokens, _MLA_V_WIDTH), jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalars),
                grid=(heads, tokens // q_tile),
                in_specs=in_specs,
                out_specs=pl.BlockSpec((1, q_tile, _MLA_V_WIDTH), q_index_map),
                scratch_shapes=(
                    pltpu.VMEM((2, k_tile, _MLA_QK_WIDTH), jnp.bfloat16),
                    pltpu.VMEM((2, _MLA_V_WIDTH, k_tile), jnp.bfloat16),
                    pltpu.SemaphoreType.DMA((2, 2)),
                ),
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel"),
                vmem_limit_bytes=_FLASH_VMEM_LIMIT,
            ),
            name=name,
        )(*scalars, *operands),
    )


def _tiles(tokens: int, capacity: int, tile: int) -> tuple[int, int]:
    """Select divisible ``(query_tile, key_tile)`` sizes for packed flash MLA."""
    q_tile = _config().mla_flash_qtile
    k_tile = _config().mla_flash_ktile
    q_tile = q_tile if tokens % q_tile == 0 else tile
    k_tile = (
        k_tile
        if capacity % k_tile == 0
        else (FRAGMENT if capacity % FRAGMENT == 0 else tile)
    )
    return q_tile, k_tile


# ============================================================================
# SP block half rings
# ============================================================================

# Half rings for one SP block. Each of the three cycles carries a column
# third; on each cycle a rank's stripe travels all its rows both
# ways, HALF_HOPS[+1] = 16 hops forward and 15 back, so a ring takes 17
# steps instead of the full ring's 32 at the same bytes per link.
HALF_HOPS = {1: 16, -1: 15}
HALF_STEPS = 17
_ASYNC_OWN_SHAPES = tuple(window // TP_SIZE for window in PREFILL_WINDOWS)


def _half_ring_streams(width: int) -> tuple[tuple[int, int, int, int], ...]:
    """Partition ``width`` across the three cycles and two half-ring directions."""
    blocks = width // VREG_LANE_COUNT
    widths = [VREG_LANE_COUNT * (blocks // 3 + (i < blocks % 3)) for i in range(3)]
    cols = [0, widths[0], widths[0] + widths[1]]
    return tuple(
        (family, direction, cols[family], widths[family])
        for family in range(3)
        for direction in (1, -1)
    )


@dataclasses.dataclass(frozen=True, slots=True)
class _HalfGatherCtx:
    """Shared references and step metadata for ``_half_gather_step``."""

    tables_ref: PallasRefProtocol
    local_ref: PallasRefProtocol
    window_ref: PallasRefProtocol
    send_sems: PallasRefProtocol
    recv_sems: PallasRefProtocol
    positions: list[jax.Array]
    stripe: int
    local_rows: pl.Slice
    block: jax.Array
    step: jax.Array


def _half_gather_window_rows(
    gather_context: _HalfGatherCtx, family: int, position: jax.Array
) -> pl.Slice:
    """Return global window row slice owned by ``(family, position)``."""
    owner = gather_context.tables_ref[
        family, 2, gather_context.tables_ref[family, 1, position % TP_SIZE]
    ]
    return pl.ds(
        gather_context.block * TP_SIZE * gather_context.stripe
        + owner * gather_context.stripe,
        gather_context.stripe,
    )


def _step_half_gather_stream(
    gather_context: _HalfGatherCtx,
    stream: int,
    spec: tuple[int, int, int, int],
) -> None:
    """Advance stream ``stream`` of ``_half_gather_step`` at ``gather_context.step``."""
    family, direction, col, width = spec
    cols = pl.ds(col, width)
    hops = HALF_HOPS[direction]
    position = gather_context.positions[family]
    target = (gather_context.tables_ref[family, 1, (position + direction) % TP_SIZE],)
    step = gather_context.step

    def send(index: jax.Array | int, source: object) -> AsyncCopyProtocol:
        """Create half-ring remote copy for owner ``index`` hops upstream."""
        rows = _half_gather_window_rows(
            gather_context, family, position - direction * index
        )
        src = gather_context.window_ref.at[rows, cols] if source is None else source
        return cast(
            "AsyncCopyProtocol",
            pltpu.make_async_remote_copy(
                src,
                gather_context.window_ref.at[rows, cols],
                gather_context.send_sems.at[stream],
                gather_context.recv_sems.at[stream],
                device_id=target,
                device_id_type=pl.DeviceIdType.MESH,
            ),
        )

    own = gather_context.local_ref.at[gather_context.local_rows, cols]

    @pl.when(step == 0)
    def seed_own() -> None:
        """Send this rank's stripe on step 0."""
        send(0, own).start()

    @pl.when((step >= 1) & (step <= hops))
    def arrived() -> None:
        """Wait on incoming and outgoing copies from ``step - 1``."""
        copy = send(step - 1, None)
        copy.wait_recv()
        copy.wait_send()

    @pl.when((step >= 1) & (step < hops))
    def forward() -> None:
        """Forward newly arrived stripe at ``step``."""
        send(step, None).start()


def _half_gather_step(
    tables_ref: PallasRefProtocol,
    local_ref: PallasRefProtocol,
    window_ref: PallasRefProtocol,
    send_sems: PallasRefProtocol,
    recv_sems: PallasRefProtocol,
    own_sem: PallasRefProtocol,
    *,
    step: jax.Array,
    block: jax.Array,
) -> None:
    """Advance step ``step`` of SP block ``block``'s half-ring all-gather."""
    rank = jax.lax.axis_index(AXIS_SP)
    positions = [tables_ref[f, 0, rank] for f in range(3)]
    stripe = window_stripe(local_ref.shape[0])
    local_rows = pl.ds(block * stripe, stripe)
    gather_context = _HalfGatherCtx(
        tables_ref=tables_ref,
        local_ref=local_ref,
        window_ref=window_ref,
        send_sems=send_sems,
        recv_sems=recv_sems,
        positions=positions,
        stripe=stripe,
        local_rows=local_rows,
        block=block,
        step=step,
    )

    @pl.when(step == 0)
    def own_rows() -> None:
        """Start local copy of this rank's stripe into ``window_ref``."""
        copy = pltpu.make_async_copy(
            local_ref.at[local_rows],
            window_ref.at[_half_gather_window_rows(gather_context, 0, positions[0])],
            own_sem,
        )
        copy.start()
        if local_ref.shape[0] not in _ASYNC_OWN_SHAPES:
            copy.wait()

    for stream, spec in enumerate(_half_ring_streams(local_ref.shape[1])):
        _step_half_gather_stream(gather_context, stream, spec)

    # The local destination is disjoint from every remote stripe. Let the
    # copy run while the half rings progress, then join before KDA consumes
    # this block and before own_sem is reused for the next block.
    if local_ref.shape[0] in _ASYNC_OWN_SHAPES:

        @pl.when(step == HALF_STEPS - 1)
        def finish_local() -> None:
            """Join local stripe copy before KDA consumes ``block``."""
            pltpu.make_async_copy(
                local_ref.at[local_rows],
                window_ref.at[
                    _half_gather_window_rows(gather_context, 0, positions[0])
                ],
                own_sem,
            ).wait()


@dataclasses.dataclass(frozen=True, slots=True)
class _HalfReduceCtx:
    """Shared buffers, semaphores, and step parameters for ``_half_reduce_step``."""

    tables_ref: PallasRefProtocol
    length_ref: PallasRefProtocol
    partial_ref: PallasRefProtocol
    prefix_ref: PallasRefProtocol
    output_ref: PallasRefProtocol
    read_sems: PallasRefProtocol
    send_sems: PallasRefProtocol
    recv_sems: PallasRefProtocol
    credit_sems: PallasRefProtocol
    store_sem: PallasRefProtocol
    local_bufs: PallasRefProtocol
    payload: PallasRefProtocol
    received: PallasRefProtocol
    step: jax.Array
    local_first: jax.Array
    window_first: jax.Array
    stripe: int
    replace: jax.Array
    token_owner: jax.Array
    positions: list[jax.Array]
    streams: tuple[tuple[int, int, int, int], ...]


def _half_reduce_read_copy(
    reduce_context: _HalfReduceCtx,
    stream: int,
    spec: tuple[int, int, int, int],
    index: jax.Array | int,
) -> AsyncCopyProtocol:
    """Create local partial read for owner ``hops - index`` hops ahead."""
    family, direction, col, width = spec
    cols = pl.ds(col, width)
    hops = HALF_HOPS[direction]
    position = reduce_context.positions[family]
    side = 0 if direction == 1 else 1
    peer = reduce_context.tables_ref[
        family, 1, (position + direction * (hops - index)) % TP_SIZE
    ]
    owner = reduce_context.tables_ref[family, 2, peer]
    return cast(
        "AsyncCopyProtocol",
        pltpu.make_async_copy(
            reduce_context.partial_ref.at[
                pl.ds(
                    reduce_context.window_first + owner * reduce_context.stripe,
                    reduce_context.stripe,
                ),
                cols,
            ],
            reduce_context.local_bufs.at[index % 2, side, :, cols],
            reduce_context.read_sems.at[index % 2, stream],
        ),
    )


def _half_reduce_remote_copy(
    reduce_context: _HalfReduceCtx,
    stream: int,
    spec: tuple[int, int, int, int],
    index: jax.Array | int,
) -> AsyncCopyProtocol:
    """Create remote half-ring transfer for ``stream`` at ``index``."""
    family, direction, col, width = spec
    cols = pl.ds(col, width)
    position = reduce_context.positions[family]
    side = 0 if direction == 1 else 1
    target = (reduce_context.tables_ref[family, 1, (position + direction) % TP_SIZE],)
    return cast(
        "AsyncCopyProtocol",
        pltpu.make_async_remote_copy(
            reduce_context.payload.at[side, :, cols],
            reduce_context.received.at[index % 2, side, :, cols],
            reduce_context.send_sems.at[stream],
            reduce_context.recv_sems.at[index % 2, stream],
            device_id=target,
            device_id_type=pl.DeviceIdType.MESH,
        ),
    )


def _step_half_reduce_stream(
    reduce_context: _HalfReduceCtx, stream: int, spec: tuple[int, int, int, int]
) -> None:
    """Advance stream ``stream`` of ``_half_reduce_step`` at ``reduce_context.step``."""
    family, direction, col, width = spec
    step = reduce_context.step
    cols = pl.ds(col, width)
    hops = HALF_HOPS[direction]
    position = reduce_context.positions[family]
    side = 0 if direction == 1 else 1
    source = (reduce_context.tables_ref[family, 1, (position - direction) % TP_SIZE],)

    @pl.when(step == 0)
    def start_initial_read() -> None:
        """Start step-0 local partial read."""
        _half_reduce_read_copy(reduce_context, stream, spec, 0).start()

    @pl.when(step <= hops)
    def accumulate() -> None:
        """Accumulate local partial and advance remote half-ring hop."""
        _half_reduce_read_copy(reduce_context, stream, spec, step).wait()

        @pl.when(step < hops)
        def prefetch_next_read() -> None:
            """Prefetch next hop's local partial."""
            _half_reduce_read_copy(reduce_context, stream, spec, step + 1).start()

        local = reduce_context.local_bufs.at[step % 2, side, :, cols][...]

        @pl.when(step == 0)
        def seed() -> None:
            """Seed payload from step-0 local partial."""
            reduce_context.payload.at[side, :, cols][...] = local

        @pl.when((step > 0) & (step < hops))
        def add() -> None:
            """Add local partial into received sum and return credit."""
            _half_reduce_remote_copy(reduce_context, stream, spec, step - 1).wait()
            reduce_context.payload.at[side, :, cols][...] = (
                local + reduce_context.received.at[(step - 1) % 2, side, :, cols][...]
            )
            pl.semaphore_signal(
                reduce_context.credit_sems.at[stream],
                1,
                device_id=source,
                device_id_type=pl.DeviceIdType.MESH,
            )

        @pl.when(step < hops)
        def forward() -> None:
            """Acquire credit when ``step >= 2`` and forward sum."""

            @pl.when(step >= _DOUBLE_BUFFER_SLOTS)
            def wait_credit() -> None:
                """Wait for downstream receive credit."""
                pl.semaphore_wait(reduce_context.credit_sems.at[stream], 1)

            _half_reduce_remote_copy(reduce_context, stream, spec, step).start()

        @pl.when(step == hops)
        def last() -> None:
            """Wait for final incoming partial sum on ``step == hops``."""
            _half_reduce_remote_copy(reduce_context, stream, spec, step - 1).wait()


def _finish_half_reduce_block(reduce_context: _HalfReduceCtx) -> None:
    """Combine forward/backward half-ring sums with residual and release credits."""
    for _, direction, col, width in reduce_context.streams:
        if direction != 1:
            continue
        cols = pl.ds(col, width)
        staging = reduce_context.local_bufs.at[1, 1, :, cols]
        prefix_copy = pltpu.make_async_copy(
            reduce_context.prefix_ref.at[
                pl.ds(reduce_context.local_first, reduce_context.stripe), cols
            ],
            staging,
            reduce_context.store_sem,
        )
        prefix_copy.start()
        prefix_copy.wait()
        valid = (
            sp_token_index(
                reduce_context.token_owner,
                reduce_context.local_first
                + jax.lax.broadcasted_iota(
                    jnp.int32, (reduce_context.stripe, width), 0
                ),
                reduce_context.stripe,
            )
            < reduce_context.length_ref[0]
        )
        projection = (
            reduce_context.local_bufs.at[0, 0, :, cols][...].astype(jnp.float32)
            + reduce_context.received.at[1, 0, :, cols][...].astype(jnp.float32)
            + reduce_context.received.at[0, 1, :, cols][...].astype(jnp.float32)
        )
        residual = staging[...].astype(jnp.float32)
        residual = jnp.where(reduce_context.replace, jnp.zeros_like(residual), residual)
        staging[...] = (residual + jnp.where(valid, projection, 0)).astype(jnp.bfloat16)
        output = pltpu.make_async_copy(
            staging,
            reduce_context.output_ref.at[
                pl.ds(reduce_context.local_first, reduce_context.stripe), cols
            ],
            reduce_context.store_sem,
        )
        output.start()
        output.wait()
    for stream, (fam, dir_, _, _) in enumerate(reduce_context.streams):
        peer = reduce_context.tables_ref[
            fam, 1, (reduce_context.positions[fam] - dir_) % TP_SIZE
        ]
        pl.semaphore_signal(
            reduce_context.credit_sems.at[stream],
            1,
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
    for stream in range(len(reduce_context.streams)):
        pl.semaphore_wait(reduce_context.credit_sems.at[stream], _DOUBLE_BUFFER_SLOTS)


def _half_reduce_step(
    tables_ref: PallasRefProtocol,
    length_ref: PallasRefProtocol,
    partial_ref: PallasRefProtocol,
    prefix_ref: PallasRefProtocol,
    output_ref: PallasRefProtocol,
    read_sems: PallasRefProtocol,
    send_sems: PallasRefProtocol,
    recv_sems: PallasRefProtocol,
    credit_sems: PallasRefProtocol,
    store_sem: PallasRefProtocol,
    local_bufs: PallasRefProtocol,
    payload: PallasRefProtocol,
    received: PallasRefProtocol,
    *,
    step: jax.Array,
    block: jax.Array,
    replace: jax.Array,
) -> None:
    """Advance step ``step`` of SP block ``block``'s half-ring reduce-scatter."""
    rank = jax.lax.axis_index(AXIS_SP)
    stripe = payload.shape[1]
    streams = _half_ring_streams(partial_ref.shape[1])
    reduce_context = _HalfReduceCtx(
        tables_ref=tables_ref,
        length_ref=length_ref,
        partial_ref=partial_ref,
        prefix_ref=prefix_ref,
        output_ref=output_ref,
        read_sems=read_sems,
        send_sems=send_sems,
        recv_sems=recv_sems,
        credit_sems=credit_sems,
        store_sem=store_sem,
        local_bufs=local_bufs,
        payload=payload,
        received=received,
        step=step,
        local_first=block * stripe,
        window_first=block * TP_SIZE * stripe,
        stripe=stripe,
        replace=replace,
        token_owner=tables_ref[0, 2, rank],
        positions=[tables_ref[f, 0, rank] for f in range(3)],
        streams=streams,
    )
    for stream, spec in enumerate(streams):
        _step_half_reduce_stream(reduce_context, stream, spec)

    @pl.when(step == HALF_STEPS - 1)
    def finish() -> None:
        """Finalize block reduction and residual add at step 16."""
        _finish_half_reduce_block(reduce_context)


# ============================================================================
# MLA attention layer (sequence-parallel)
# ============================================================================


class ProjectionWeights(NamedTuple):
    """Per-rank MLA projection and RMSNorm weight tensors."""

    query_down_projection: jax.Array
    query_up_projection: jax.Array
    key_value_down_projection: jax.Array
    key_value_up_projection: jax.Array
    output_gate_projection: jax.Array
    query_normalization: jax.Array
    key_value_normalization: jax.Array


def make_mla_attention_layer_sp_ring(
    mesh: jax.sharding.Mesh,
    *,
    seq_len: int,
    segments: int,
    prefix_capacity: int,
    token_ranks: Sequence[int] | None = None,
    collective_id: int = 31,
    flash_tile: int = STORE_TILE,
) -> Callable[..., tuple[jax.Array, jax.Array]]:
    """Build sequence-parallel MLA layer callable over a packed window."""
    if not segments:
        msg = "the MLA layer runs packed windows"
        raise ValueError(msg)
    if mesh.size != TP_SIZE or tuple(mesh.axis_names) != (AXIS_SP,):
        msg = "ring MLA requires a flat TP32 sequence-parallel mesh"
        raise ValueError(msg)
    check_window(seq_len, "ring MLA")
    if token_ranks is None:
        token_ranks = tuple(range(TP_SIZE))
    tables = ring_tables(list(mesh.devices.flat), token_ranks)
    local_tokens = seq_len // TP_SIZE

    def unshard_first(value: jax.Array) -> jax.Array:
        """Select leading local rank slice."""
        return value[0]

    def local_packed_cp(
        full_hidden: jax.Array,
        weights: Mapping[str, jax.Array],
        segment_desc: jax.Array,
        length: jax.Array,
        pool: jax.Array,
        ids: jax.Array,
        local: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Run one rank's 3 MLA heads over gathered prefixes and ``full_hidden``."""
        local_weights = jax.tree.map(unshard_first, weights)
        projections = ProjectionWeights(
            query_down_projection=local_weights["query_down_projection"],
            query_up_projection=local_weights["query_up_projection"],
            key_value_down_projection=local_weights["key_value_down_projection"],
            key_value_up_projection=local_weights["key_value_up_projection"],
            output_gate_projection=local_weights["output_gate_projection"],
            query_normalization=local_weights["query_normalization"],
            key_value_normalization=local_weights["key_value_normalization"],
        )
        query, new_key, new_value, output_gate, latent = _mla_projections_with_latent(
            full_hidden, projections, query_scale=SCALE_LOG2
        )
        tokens = query.shape[0]
        real = (jnp.arange(tokens) < length.reshape(()))[None, :, None]
        new_key = jnp.where(real, new_key.transpose(1, 0, 2), 0)
        new_value = jnp.where(real, new_value.transpose(1, 0, 2), 0)
        query_hm = query.transpose(1, 0, 2)
        pages, order = gathered_prefix_pages(
            pool,
            ids[0],
            local[0],
            tables,
            AXIS_SP,
            count=segments,
            capacity=prefix_capacity,
        )
        arena_kv = expand_prefix_pages(
            pages, order, projections.key_value_up_projection
        )
        attended = mla_flash_prefill_packed(
            query_hm,
            new_key,
            new_value,
            length,
            segment_desc[: 5 * segments],
            count=segments,
            tile=flash_tile,
            arena=arena_kv,
        )
        attended_2d = attended.transpose(1, 0, 2).reshape(
            tokens, _PROJECTION_HEAD_WIDTH
        )
        gate = jax.nn.sigmoid(output_gate.astype(jnp.float32)).astype(jnp.bfloat16)
        return (attended_2d * gate).astype(jnp.bfloat16)[None], latent[None]

    def run(
        hidden: jax.Array,
        prefix: jax.Array,
        weights: Mapping[str, jax.Array],
        segment_desc: jax.Array,
        length: jax.Array,
        arena: CpPrefix,
    ) -> tuple[jax.Array, jax.Array]:
        """Execute sequence-parallel MLA projections, attention, and ring output."""
        (pool,) = arena.pools
        gated, latent = jax.shard_map(
            local_packed_cp,
            mesh=mesh,
            in_specs=(
                P(),
                P(AXIS_SP),
                P(),
                P(),
                P(AXIS_SP),
                P(AXIS_SP),
                P(AXIS_SP),
            ),
            out_specs=(P(AXIS_SP), P(AXIS_SP)),
            check_vma=False,
        )(
            hidden,
            weights,
            segment_desc,
            length,
            cast("jax.Array", pool),
            cast("jax.Array", arena.ids),
            cast("jax.Array", arena.local),
        )
        projected = project_reduce_scatter(
            gated,
            weights["output_projection_repacked"],
            mesh,
            tables,
            collective_id=collective_id,
        ).reshape(TP_SIZE, local_tokens, HIDDEN_SIZE)

        # Masked residual add inside a shard_map: an auto-sharded iota over the
        # rank axis needs the partitioner's partition-id, which runtimes that
        # only accept manual SPMD programs (torch_tpu's jax_op) reject.
        def residual(
            prefix_shard: jax.Array, projected_shard: jax.Array, len_arr: jax.Array
        ) -> jax.Array:
            """Add masked projected attention output onto ``prefix_shard``."""
            rank = mesh_axis_index(AXIS_SP)
            real = (
                sp_token_index(
                    rank, jnp.arange(local_tokens), window_stripe(local_tokens)
                )
                < len_arr[0]
            )
            return (prefix_shard[0] + jnp.where(real[:, None], projected_shard[0], 0))[
                None
            ]

        output = jax.shard_map(
            residual,
            mesh=mesh,
            in_specs=(P(AXIS_SP), P(AXIS_SP), P()),
            out_specs=P(AXIS_SP),
            check_vma=False,
        )(prefix, projected, length)
        return output.astype(jnp.bfloat16), latent

    return jax.jit(run)


# ============================================================================
# MoE reduce-scatter ring
# ============================================================================
#
# Running sum uses FP32 or BF16; FP32 avoids rounding the partials at every hop.

_RING_RS_HEADROOM = 8 * MIB_BYTES
_COUNTING_SORT_BLOCK = 512
_EXPERT_BANK_GROUP_SIZE = 8
_PACKED_SORT_KEY_BITS = 31
_EP_SIZE = TP_SIZE // DP2_GROUPS
_LOCAL_EXPERTS_EP16 = ROUTED_EXPERTS // _EP_SIZE
_LATENT_DOWN_COLS = 4096
_LATENT_ROUTED_COLS = 3584
_FAMILY_SHARED_NDIM = 3
_MOE_PARTIAL_NDIM = 3


def ring_reduce_scatter_fp32(
    partials: jax.Array,
    mesh: jax.sharding.Mesh,
    tables: np.ndarray,
    *,
    collective_id: int = 43,
    dtype: jax.typing.DTypeLike = jnp.float32,
) -> jax.Array:
    """Reduce ``[32, T, D]`` partials to ``[32, T/32, D]`` shards in ``dtype``."""
    if mesh.size != TP_SIZE or tuple(mesh.axis_names) != (AXIS_SP,):
        msg = "FP32 ring requires a flat TP32 sequence-parallel mesh"
        raise ValueError(msg)
    if (
        partials.ndim != _MOE_PARTIAL_NDIM
        or partials.shape[0] != TP_SIZE
        or partials.dtype not in (jnp.float32, jnp.bfloat16)
    ):
        msg = "FP32 ring needs [32,T,D] BF16 or FP32 partials"
        raise ValueError(msg)
    tokens, width = partials.shape[1:]
    if tokens % TP_SIZE or tokens // TP_SIZE > PAGE_TOKENS:
        msg = "FP32 ring needs at most 256 local rows"
        raise ValueError(msg)
    rows = tokens // TP_SIZE
    stream_regions = regions(rows, width)
    streams, buffers = len(stream_regions), 2
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    ring_dt = jnp.dtype(dtype)

    def local(part: jax.Array) -> jax.Array:
        """Run single-rank reduce-scatter call."""
        scratch = (
            pltpu.SemaphoreType.DMA((streams,)),
            pltpu.SemaphoreType.DMA((streams,)),
            pltpu.SemaphoreType.DMA((buffers, streams)),
            pltpu.SemaphoreType.REGULAR((streams,)),
            pltpu.SemaphoreType.DMA,
            pltpu.VMEM((rows, width), partials.dtype),
            pltpu.VMEM((rows, width), ring_dt),
            pltpu.VMEM((buffers, rows, width), ring_dt),
        )
        call = pl.pallas_call(
            functools.partial(
                _reduce_scatter_kernel, streams=stream_regions, buffers=buffers
            ),
            out_shape=jax.ShapeDtypeStruct((rows, width), ring_dt),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(1, TP_SIZE),
                in_specs=(hbm,),
                out_specs=hbm,
                scratch_shapes=scratch,
            ),
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                vmem_limit_bytes=4 * rows * width * 4 + _RING_RS_HEADROOM,
            ),
            name="kimi_prefill_fp32_ring_rs",
        )
        return cast("jax.Array", call(jnp.asarray(tables, jnp.int32), part[0]))

    return cast(
        "jax.Array",
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=P(AXIS_SP, None, None),
            out_specs=P(AXIS_SP, None),
            check_vma=False,
        )(partials),
    )


# ============================================================================
# Routed MoE (expert-major)
# ============================================================================


def _counting_sort_positions(
    keys: jax.Array, buckets: int
) -> tuple[jax.Array, jax.Array]:
    """Compute stable expert-major destination positions and bucket counts."""
    size = keys.shape[0]
    block = min(_COUNTING_SORT_BLOCK, size)
    if buckets > block:
        msg = "counting-sort bucket count exceeds its block width"
        raise ValueError(msg)
    pad = -size % block
    if pad:
        keys = jnp.pad(keys, (0, pad), constant_values=buckets - 1)
    onehot = jax.nn.one_hot(keys.reshape(-1, block), buckets, dtype=jnp.bfloat16)
    lower = jnp.concatenate(
        (
            jnp.ones((1, block), jnp.bfloat16),
            jnp.tril(jnp.ones((block, block), jnp.bfloat16), -1),
        ),
        axis=0,
    )
    both = jnp.einsum("ij,bje->bie", lower, onehot, preferred_element_type=jnp.float32)
    hist = both[:, 0, :]
    rank = both[:, 1:, :]
    counts = hist.sum(axis=0)
    base = jnp.cumsum(hist, axis=0) - hist + jnp.cumsum(counts) - counts
    positions = ((base[:, None, :] + rank) * onehot).sum(axis=2)
    return positions.reshape(-1)[:size].astype(jnp.int32), counts.astype(jnp.int32)


def expert_family_slot(family_count: int, slot: jax.Array) -> jax.Array:
    """Map global MoE layer ``slot`` to its resident expert-bank index."""
    if family_count == _EXPERT_BANK_GROUP_SIZE:
        return (slot - 3) % _EXPERT_BANK_GROUP_SIZE
    if family_count == 1:
        return jnp.zeros_like(slot)
    return slot


def routed_experts(
    latent: jax.Array,
    topk_ids: jax.Array,
    topk_probabilities: jax.Array,
    expert_gate_up: jax.Array,
    expert_gate_up_scales: jax.Array,
    expert_down: jax.Array,
    expert_down_scales: jax.Array,
    *,
    first_expert: jax.Array | int,
    expert_layer: jax.Array | None = None,
) -> jax.Array:
    """Compute one EP shard's routed expert contribution across all tokens."""
    if expert_layer is not None:
        expert_layer = expert_family_slot(expert_gate_up.shape[0], expert_layer)
    tokens, input_width = latent.shape
    local_experts = expert_gate_up.shape[-3]
    if topk_ids.shape != topk_probabilities.shape or topk_ids.shape[0] != tokens:
        msg = "topk ids and probabilities must have shape [T, topk]"
        raise ValueError(msg)
    if expert_gate_up.shape[-2] * 8 != input_width:
        msg = "latent width must match the packed gate/up contraction"
        raise ValueError(msg)
    if expert_gate_up.shape[-1] != 2 * expert_down.shape[-2] * 8:
        msg = "gate/up and down intermediate widths disagree"
        raise ValueError(msg)

    topk = topk_ids.shape[1]
    local_ids = topk_ids.astype(jnp.int32) - jnp.asarray(first_expert, jnp.int32)
    valid = (local_ids >= 0) & (local_ids < local_experts) & (topk_probabilities != 0)
    keys = jnp.where(valid, local_ids, local_experts).reshape(-1)
    token_id = jnp.broadcast_to(
        jnp.arange(tokens, dtype=jnp.int32)[:, None], (tokens, topk)
    ).reshape(-1)
    token_bits = max(1, (tokens - 1).bit_length())
    if (local_experts + 1).bit_length() + token_bits >= _PACKED_SORT_KEY_BITS:
        msg = "expert and token ids do not fit the packed sort key"
        raise ValueError(msg)
    packed = (keys << token_bits) | token_id
    order = jnp.argsort(packed, stable=False)
    sorted_tokens = token_id[order]
    inverse, all_counts = _counting_sort_positions(keys, local_experts + 1)
    counts = all_counts[:local_experts]
    x = ragged_gather_v2(
        latent,
        sorted_tokens,
        jnp.array([0], jnp.int32),
        jnp.sum(counts, dtype=jnp.int32)[None],
    )
    # The E8M0 exponent bytes go to the GMM as they are (it converts each
    # tile in VMEM): an FP32 [E, blocks, 1, N] copy pads 8x in HBM.
    gu_scale = expert_gate_up_scales.astype(jnp.uint8)
    down_scale = expert_down_scales.astype(jnp.uint8)
    middle = gmm_v2(
        x,
        expert_gate_up,
        counts,
        rhs_scale=gu_scale,
        rhs_layer=expert_layer,
        rhs_quant_dtype=jnp.float4_e2m1fn,
        fuse_act="situ:4:25",
    )
    down = gmm_v2(
        middle,
        expert_down,
        counts,
        rhs_scale=down_scale,
        rhs_layer=expert_layer,
        rhs_quant_dtype=jnp.float4_e2m1fn,
    )
    # Restore the token-major pair order, mask the unwritten nonlocal GMM rows,
    # and reduce the top-k contributions per token in FP32.
    return ragged_gather_reduce(
        down,
        inverse,
        topk_probabilities.reshape(-1),
        valid.reshape(-1),
        reduce_group_size=topk,
    )


def _moe_latent_all_gather(latent: jax.Array, tables: np.ndarray) -> jax.Array:
    """All-gather BF16 sequence-parallel latents in mesh-rank order."""
    rows, width = latent.shape
    stream_regions = regions(rows, width)
    streams, buffers = len(stream_regions), 2
    return cast(
        "jax.Array",
        pl.pallas_call(
            functools.partial(
                _all_gather_kernel, regions=stream_regions, buffers=buffers
            ),
            out_shape=jax.ShapeDtypeStruct((TP_SIZE * rows, width), latent.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(1, TP_SIZE),
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
                vmem_limit_bytes=3 * rows * width * 2 + _RING_RS_HEADROOM,
            ),
            name="kimi_prefill_moe_latent_ag",
        )(jnp.asarray(tables, jnp.int32), latent),
    )


def moe_local_routed(
    hidden_local: jax.Array,
    length: jax.Array,
    weights: Mapping[str, jax.Array],
    *,
    axis_name: str,
    gather_tables: np.ndarray,
) -> tuple[jax.Array, jax.Array]:
    """Compute one rank's routed MoE partial and chip-local TP2 shared partial."""
    rank = mesh_axis_index(axis_name)
    local_tokens, hidden_width = hidden_local.shape
    if hidden_width != HIDDEN_SIZE:
        msg = "K3 sequence-parallel MoE expects [T/32,7168]"
        raise ValueError(msg)
    if weights["latent_down"].shape[-2:] != (HIDDEN_SIZE, _LATENT_DOWN_COLS):
        msg = "latent projection weights must use replicated SP layout"
        raise ValueError(msg)
    local_experts = weights["expert_gu"].shape[-3]
    if local_experts != _LOCAL_EXPERTS_EP16:
        msg = "K3 SP MoE requires EP16 x TP2 expert weights"
        raise ValueError(msg)
    expert_tp = TP_SIZE * local_experts // ROUTED_EXPERTS
    tokens = local_tokens * TP_SIZE

    logits = jax.lax.dot_general(
        hidden_local,
        weights["router"],
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    raw = jax.nn.sigmoid(logits)
    _, local_ids = jax.lax.top_k(raw + weights["router_bias"], 16)
    local_probs = jnp.take_along_axis(raw, local_ids, axis=1)
    local_probs /= jnp.maximum(local_probs.sum(axis=1, keepdims=True), 1e-30)
    valid_rows = sp_token_index(
        rank, jnp.arange(local_tokens), window_stripe(local_tokens)
    ) < length.reshape(())
    local_probs = jnp.where(valid_rows[:, None], local_probs, 0)

    if weights["latent_down"].ndim == _FAMILY_SHARED_NDIM:
        latent_local = gmm_v2(
            hidden_local,
            weights["latent_down"][:, None],
            jnp.full((1,), local_tokens, jnp.int32),
            rhs_layer=weights["expert_layer"],
        )
    else:
        latent_local = hidden_local @ weights["latent_down"]
    latent_all = _moe_latent_all_gather(
        latent_local[:, :_LATENT_ROUTED_COLS], gather_tables
    )
    routes = jnp.concatenate(
        (
            local_ids.astype(jnp.int32),
            jax.lax.bitcast_convert_type(local_probs, jnp.int32),
        ),
        axis=1,
    )
    routes = jax.lax.all_gather(routes, axis_name).reshape(tokens, TP_SIZE)
    ids_all = routes[:, :16]
    probs_all = jax.lax.bitcast_convert_type(routes[:, 16:], jnp.float32)
    routed = routed_experts(
        latent_all,
        ids_all,
        probs_all,
        weights["expert_gu"],
        weights["expert_gus"],
        weights["expert_down"],
        weights["expert_ds"],
        first_expert=(rank // expert_tp) * local_experts,
        expert_layer=weights.get("expert_layer"),
    )
    shared = _moe_sp_shared_partial(
        _moe_sp_shared_rows(hidden_local, axis_name), weights
    )
    return routed, shared


SHARED_PAIRS = tuple((2 * i, 2 * i + 1) for i in range(16))


def _moe_sp_shared_rows(hidden_local: jax.Array, axis_name: str) -> jax.Array:
    """All-gather both chip-local cores' token rows for TP2 shared expert."""
    local_tokens = hidden_local.shape[0]
    return jax.lax.all_gather(
        hidden_local, axis_name, axis_index_groups=SHARED_PAIRS
    ).reshape(2 * local_tokens, HIDDEN_SIZE)


def _moe_sp_shared_partial(
    chip_rows: jax.Array, weights: Mapping[str, jax.Array]
) -> jax.Array:
    """Compute one core's FP32 half of the TP2 shared expert over ``chip_rows``."""
    rows = jnp.full((1,), chip_rows.shape[0], jnp.int32)
    # Resident layer families ([L, ...]): the kernel selects the layer (a
    # sliced operand would be copied).
    family = weights["shared_gu"].ndim == _FAMILY_SHARED_NDIM
    layer_arg = weights["expert_layer"] if family else None

    def one_group(w: jax.Array) -> jax.Array:
        """Insert singleton expert-group axis."""
        return w[:, None] if family else w[None]

    shared_mid = gmm_v2(
        chip_rows,
        one_group(weights["shared_gu"]),
        rows,
        rhs_layer=layer_arg,
        fuse_act="situ:4:25",
    )
    return gmm_v2(
        shared_mid,
        one_group(weights["shared_down"]),
        rows,
        rhs_layer=layer_arg,
        preferred_element_type=jnp.float32,
    )


def _moe_sp_shared_reduce(partial: jax.Array, axis_name: str) -> jax.Array:
    """Reduce-scatter TP2 shared expert partials across ``SHARED_PAIRS``."""
    return jax.lax.psum_scatter(
        partial,
        axis_name,
        scatter_dimension=0,
        tiled=True,
        axis_index_groups=SHARED_PAIRS,
    ).astype(jnp.bfloat16)


def _moe_sp_tail(
    routed: jax.Array,
    shared: jax.Array,
    prefix_local: jax.Array,
    length: jax.Array,
    weights: Mapping[str, jax.Array],
    *,
    axis_name: str,
    eps: float,
    shared_reduced: bool = False,
) -> jax.Array:
    """Normalize reduced routed output, project up, and add shared + residual."""
    rank = mesh_axis_index(axis_name)
    local_tokens = prefix_local.shape[0]
    valid_rows = sp_token_index(
        rank, jnp.arange(local_tokens), window_stripe(local_tokens)
    ) < length.reshape(())
    routed_f32 = routed.astype(jnp.float32)
    routed_f32 *= jax.lax.rsqrt(
        jnp.mean(routed_f32 * routed_f32, axis=1, keepdims=True) + eps
    )
    routed_bf16 = (routed_f32 * weights["latent_norm"][:, :_LATENT_ROUTED_COLS]).astype(
        jnp.bfloat16
    )
    up = routed_bf16 @ weights["latent_up"]
    if not shared_reduced:
        shared = _moe_sp_shared_reduce(shared, axis_name)
    output = (up + shared).astype(jnp.bfloat16)
    output = jnp.where(valid_rows[:, None], output, 0)
    return (prefix_local + output).astype(jnp.bfloat16)


moe_sp_tail = _moe_sp_tail


def make_moe_fused_sp_repacked(
    mesh: jax.sharding.Mesh,
    *,
    seq_len: int,
    eps: float = 1e-5,
    collective_id: int = 42,
    token_ranks: Sequence[int] | None = None,
) -> Callable[..., jax.Array]:
    """Build reference TP32 MoE layer callable over repacked projection weights."""
    if tuple(mesh.axis_names) != (AXIS_SP,):
        msg = "MoE Pallas ring reduction needs TP32 sequence-parallel axis"
        raise ValueError(msg)
    check_window(seq_len, "MoE Pallas ring reduction")
    axis_name = mesh.axis_names[0]
    if token_ranks is None:
        token_ranks = tuple(range(TP_SIZE))
    tables = ring_tables(list(mesh.devices.flat), token_ranks)
    gather_tables = ring_tables(list(mesh.devices.flat), tuple(range(TP_SIZE)))
    local_tokens = seq_len // TP_SIZE

    def unshard_first(value: jax.Array) -> jax.Array:
        """Select leading rank slice."""
        return value[0]

    def local_partial(
        hidden: jax.Array,
        weights: Mapping[str, jax.Array],
        length: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute per-rank routed and shared MoE partials."""
        local_weights = jax.tree.map(unshard_first, weights)
        routed, shared = moe_local_routed(
            hidden[0],
            length,
            local_weights,
            axis_name=axis_name,
            gather_tables=gather_tables,
        )
        return routed[None], shared[None]

    partial_op = jax.jit(
        jax.shard_map(
            local_partial,
            mesh=mesh,
            in_specs=(P(axis_name), P(axis_name), P()),
            out_specs=(P(axis_name), P(axis_name)),
            check_vma=False,
        )
    )

    def local_tail(
        reduced: jax.Array,
        shared: jax.Array,
        prefix: jax.Array,
        weights: Mapping[str, jax.Array],
        length: jax.Array,
    ) -> jax.Array:
        """Apply RMSNorm, latent-up projection, shared reduction, and residual."""
        local_weights = jax.tree.map(unshard_first, weights)
        return _moe_sp_tail(
            reduced[0],
            shared[0],
            prefix[0],
            length,
            local_weights,
            axis_name=axis_name,
            eps=eps,
        )[None]

    tail_op = jax.jit(
        jax.shard_map(
            local_tail,
            mesh=mesh,
            in_specs=(
                P(axis_name),
                P(axis_name),
                P(axis_name),
                P(axis_name),
                P(),
            ),
            out_specs=P(axis_name),
            check_vma=False,
        )
    )

    def run_ring(
        hidden: jax.Array,
        prefix: jax.Array,
        weights: Mapping[str, jax.Array],
        length: jax.Array,
    ) -> jax.Array:
        """Run routed partials, ring reduce-scatter, and MoE tail."""
        partials, shared = partial_op(hidden, weights, length)
        reduced = ring_reduce_scatter_fp32(
            partials,
            mesh,
            tables,
            collective_id=collective_id,
            dtype=jnp.dtype(_config().moe_reduce_dtype),
        ).reshape(TP_SIZE, local_tokens, _LATENT_ROUTED_COLS)
        return tail_op(reduced, shared, prefix, weights, length)

    return jax.jit(run_ring)


# ============================================================================
# MLA weight families
# ============================================================================

MLA_FAMILIES = (
    "m_qa",
    "m_qb",
    "m_ka",
    "m_kb",
    "m_gate",
    "m_qnorm",
    "m_knorm",
    "m_o",
)
MLA_FIELD_NAMES = (
    "query_down_projection",
    "query_up_projection",
    "key_value_down_projection",
    "key_value_up_projection",
    "output_gate_projection",
    "query_normalization",
    "key_value_normalization",
    "output_projection_repacked",
)


def make_indexed_mla_layer_sp_ring(
    mesh: jax.sharding.Mesh,
    *,
    seq_len: int = OVERFLOW_WINDOW_TOKENS,
    segments: int,
    prefix_capacity: int,
    flash_tile: int = PAGE_TOKENS,
) -> Callable[..., tuple[jax.Array, jax.Array]]:
    """Build runtime-slot-indexed MLA layer callable over family-major weights."""
    base = make_mla_attention_layer_sp_ring(
        mesh,
        seq_len=seq_len,
        flash_tile=flash_tile,
        segments=segments,
        prefix_capacity=prefix_capacity,
    )

    def select(
        families: Mapping[str, jax.Array], slot: jax.Array
    ) -> dict[str, jax.Array]:
        """Slice layer ``slot`` from each MLA weight family."""
        return {
            field: jax.lax.dynamic_index_in_dim(
                families[name], slot[0], axis=1, keepdims=False
            )
            for name, field in zip(MLA_FAMILIES, MLA_FIELD_NAMES, strict=True)
        }

    def run(
        hidden: jax.Array,
        prefix: jax.Array,
        families: Mapping[str, jax.Array],
        segment_desc: jax.Array,
        length: jax.Array,
        slot: jax.Array,
        arena: CpPrefix,
    ) -> tuple[jax.Array, jax.Array]:
        """Execute indexed MLA layer at ``slot``."""
        return base(hidden, prefix, select(families, slot), segment_desc, length, arena)

    return jax.jit(run)


# ============================================================================
# MoE weight families
# ============================================================================

MOE_FAMILIES = (
    "router",
    "router_bias",
    "latent_down",
    "latent_up",
    "latent_norm",
    "shared_gu",
    "shared_down",
    "expert_gu",
    "expert_gus",
    "expert_down",
    "expert_ds",
)

_RESIDENT_MOE_FAMILIES = (
    "expert_gu",
    "expert_down",
    "expert_gus",
    "expert_ds",
    "latent_down",
    "shared_gu",
    "shared_down",
)
_RETIRED_PACKED_BF16_WINDOWS = PREFILL_WINDOWS[-2:]


def make_indexed_moe_sp_ring(
    mesh: jax.sharding.Mesh, *, seq_len: int = OVERFLOW_WINDOW_TOKENS
) -> Callable[..., jax.Array]:
    """Select fused FP32 or reference MoE layer callable for ``seq_len``."""
    reduce_dtype = _config().moe_reduce_dtype
    if reduce_dtype == "float32" and seq_len in PREFILL_WINDOWS:
        # Deferred import breaks the prefill_megakernel <-> prefill_moe.fused cycle.
        fused_mod = importlib.import_module(".prefill_moe.fused", __package__)
        return cast(
            "Callable[..., jax.Array]",
            fused_mod.make_indexed_moe_sp_ring(mesh, seq_len=seq_len),
        )
    if reduce_dtype == "bfloat16" and seq_len in _RETIRED_PACKED_BF16_WINDOWS:
        logger.warning_once(
            "Packed BF16 prefill MoE is retired at seq_len=%d; using reference MoE.",
            seq_len,
        )
    return _make_indexed_moe_sp_ring_reference(mesh, seq_len=seq_len)


def _make_indexed_moe_sp_ring_reference(
    mesh: jax.sharding.Mesh, *, seq_len: int = OVERFLOW_WINDOW_TOKENS
) -> Callable[..., jax.Array]:
    """Build reference subcore gather/combine MoE callable for ``seq_len``."""
    base = make_moe_fused_sp_repacked(mesh, seq_len=seq_len)

    def run(
        hidden: jax.Array,
        prefix: jax.Array,
        families: Mapping[str, jax.Array],
        length: jax.Array,
        slot: jax.Array,
    ) -> jax.Array:
        """Select non-resident layer slices and invoke reference MoE."""
        selected = {
            name: (
                families[name]
                if name in _RESIDENT_MOE_FAMILIES
                else jax.lax.dynamic_index_in_dim(
                    families[name], slot[0], axis=1, keepdims=False
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

_KDA_LAYER_VMEM_LIMIT = 64 * MIB_BYTES
_DENSE_GU_COLS = 2304
_DENSE_DOWN_ROWS = 1152


def _norm_gather_body(
    active_ref: PallasRefProtocol,
    tables_ref: PallasRefProtocol,
    prefix_ref: PallasRefProtocol,
    blocks_ref: PallasRefProtocol,
    folded_ref: PallasRefProtocol,
    norm_ref: PallasRefProtocol,
    normalized_ref: PallasRefProtocol,
    gathered_ref: PallasRefProtocol,
    *scratch: PallasRefProtocol,
    regions: Sequence[tuple[int, int, int, int]],
    eps: float,
    mix_tile: int,
) -> None:
    """Run local residual mixture + RMSNorm at step 0 followed by window all-gather."""
    mix_scratch = scratch[:12]
    ring_scratch = scratch[12:]

    @pl.when(pl.program_id(1) == 0)
    def normalize() -> None:
        """Normalize local token rows before step-0 ring loads."""
        mixture_norm_tiled_body(
            active_ref,
            prefix_ref,
            blocks_ref,
            folded_ref,
            norm_ref,
            normalized_ref,
            *mix_scratch,
            tile=mix_tile,
            eps=eps,
        )

    _all_gather_kernel(
        tables_ref,
        normalized_ref,
        gathered_ref,
        *ring_scratch,
        regions=regions,
        buffers=2,
        striped=True,
    )


def make_norm_gather_sp(
    mesh: jax.sharding.Mesh,
    *,
    seq_len: int,
    token_ranks: Sequence[int] | None = None,
    collective_id: int = 37,
    eps: float = 1e-5,
) -> Callable[..., tuple[jax.Array, jax.Array]]:
    """Build fused residual mixture, RMSNorm, and TP32 window all-gather callable."""
    if mesh.size != TP_SIZE or tuple(mesh.axis_names) != (AXIS_SP,):
        msg = "norm/gather requires a flat TP32 sequence-parallel mesh"
        raise ValueError(msg)
    check_window(seq_len, "fused norm/gather")
    rows, width = seq_len // TP_SIZE, HIDDEN_SIZE
    if token_ranks is None:
        token_ranks = tuple(range(TP_SIZE))
    tables = jnp.asarray(ring_tables(list(mesh.devices.flat), token_ranks), jnp.int32)
    stream_regions = regions(rows, width)
    streams = len(stream_regions)
    mix_scratch = mixture_norm_tiled_scratch_shapes(width, rows)
    ring_scratch = (
        pltpu.SemaphoreType.DMA((streams,)),
        pltpu.SemaphoreType.DMA((streams,)),
        pltpu.SemaphoreType.DMA((2, streams)),
        pltpu.SemaphoreType.REGULAR((streams,)),
        pltpu.SemaphoreType.DMA((streams,)),
        pltpu.VMEM((rows, width), jnp.bfloat16),
        pltpu.VMEM((2, rows, width), jnp.bfloat16),
    )
    all_scratch = cast(
        "pl_core.ScratchShapeTree",
        (*mix_scratch, *ring_scratch),
    )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    call = pl.pallas_call(
        functools.partial(
            _norm_gather_body, regions=stream_regions, eps=eps, mix_tile=rows
        ),
        out_shape=(
            jax.ShapeDtypeStruct((rows, width), jnp.bfloat16),
            jax.ShapeDtypeStruct((seq_len, width), jnp.bfloat16),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1, TP_SIZE),
            in_specs=(hbm, hbm, hbm, hbm),
            out_specs=(hbm, hbm),
            scratch_shapes=all_scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id, vmem_limit_bytes=_KDA_LAYER_VMEM_LIMIT
        ),
        name="kimi_prefill_norm_gather_sp",
    )

    def local(
        prefix: jax.Array,
        blocks: jax.Array,
        folded: jax.Array,
        norm: jax.Array,
        active: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Invoke fused norm/gather on one rank."""
        normalized, gathered = cast(
            "tuple[jax.Array, jax.Array]",
            call(active, tables, prefix[0], blocks[0], folded[0], norm[0]),
        )
        return normalized[None], gathered

    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(AXIS_SP), P(AXIS_SP), P(AXIS_SP), P(AXIS_SP), P()),
            out_specs=(P(AXIS_SP), P()),
            check_vma=False,
        )
    )


_PIPELINE_PHASE_OFFSET = 2


def _kda_layer_body(
    slot_ref: PallasRefProtocol,
    segment_ref: PallasRefProtocol,
    active_ref: PallasRefProtocol,
    tables_ref: PallasRefProtocol,
    length_ref: PallasRefProtocol,
    prefix_ref: PallasRefProtocol,
    blocks_ref: PallasRefProtocol,
    folded_ref: PallasRefProtocol,
    norm_ref: PallasRefProtocol,
    gate_family_ref: PallasRefProtocol,
    fb_family_ref: PallasRefProtocol,
    proj_family_ref: PallasRefProtocol,
    conv_family_ref: PallasRefProtocol,
    alog_family_ref: PallasRefProtocol,
    dt_family_ref: PallasRefProtocol,
    kda_norm_family_ref: PallasRefProtocol,
    rec_in_ref: PallasRefProtocol,
    hist_in_ref: PallasRefProtocol,
    output_family_ref: PallasRefProtocol,
    normalized_ref: PallasRefProtocol,
    gathered_ref: PallasRefProtocol,
    attended_ref: PallasRefProtocol,
    rec_out_ref: PallasRefProtocol,
    hist_out_ref: PallasRefProtocol,
    final_ref: PallasRefProtocol,
    rec_checkpoint_ref: PallasRefProtocol,
    hist_checkpoint_ref: PallasRefProtocol,
    partial_ref: PallasRefProtocol,
    pool: PallasRefProtocol,
    *scratch: PallasRefProtocol,
    eps: float,
    count: int,
    blocks: int,
) -> None:
    """Execute one pipelined KDA layer overlapping half-rings with tiled KDA."""
    slot = slot_ref[0]
    step = pl.program_id(1)
    phase, ring_step = step // HALF_STEPS, step % HALF_STEPS
    stripe = window_stripe(prefix_ref.shape[0])
    mix_small, gather_sems = scratch[:8], scratch[8:11]
    reduce_sems, projection_scratch = scratch[11:16], scratch[16:19]
    kda_scratch = scratch[19:]

    def views(
        shapes: Sequence[tuple[tuple[int, ...], jax.typing.DTypeLike]],
    ) -> list[PallasRefProtocol]:
        """Carve typed aliased views out of ``pool``."""
        offset = 0
        result: list[PallasRefProtocol] = []
        for shape, dtype in shapes:
            size = pool_alias.storage_size(shape, jnp.dtype(dtype))
            result.append(
                cast(
                    "PallasRefProtocol",
                    pool_alias.view(
                        pool.at[
                            pl.ds(offset // VREG_LANE_COUNT, size // VREG_LANE_COUNT), :
                        ],
                        shape,
                        dtype,
                    ),
                )
            )
            offset += size
        return result

    @pl.when(step == 0)
    def enter() -> None:
        """Synchronize ring neighbors before step-0 half-ring copies."""
        ring_enter(tables_ref, jax.lax.axis_index(AXIS_SP))

    @pl.when(step < blocks)
    def normalize() -> None:
        """Normalize SP block ``step``'s local rows."""
        rows = pl.ds(step * stripe, stripe)
        mixture_norm_tiled_body(
            active_ref,
            prefix_ref.at[rows],
            blocks_ref.at[:, rows],
            folded_ref,
            norm_ref,
            normalized_ref.at[rows],
            *views(_kda_mix_shapes(stripe)),
            *mix_small,
            tile=stripe,
            eps=eps,
        )

    @pl.when(phase < blocks)
    def gather() -> None:
        """Advance half-ring all-gather for SP block ``phase``."""
        _half_gather_step(
            tables_ref,
            normalized_ref,
            gathered_ref,
            *gather_sems,
            step=ring_step,
            block=phase,
        )

    @pl.when(phase >= _PIPELINE_PHASE_OFFSET)
    def reduce() -> None:
        """Advance half-ring reduce-scatter for SP block ``phase - 2``."""
        _half_reduce_step(
            tables_ref,
            length_ref,
            partial_ref,
            prefix_ref,
            final_ref,
            *reduce_sems,
            *views(_kda_reduce_shapes(stripe)),
            step=ring_step,
            block=phase - _PIPELINE_PHASE_OFFSET,
            replace=slot % 9 == 0,
        )

    # One call site for the KDA tile: inlined twice, the layer would no
    # longer fit the instruction memory.
    block_tiles = TP_SIZE * stripe // STORE_TILE
    attending = (phase >= 1) & (phase <= blocks)
    first_tile = jnp.where(
        attending, (phase - 1) * block_tiles + ring_step * block_tiles // HALF_STEPS, 0
    )
    end_tile = jnp.where(
        attending,
        (phase - 1) * block_tiles + (ring_step + 1) * block_tiles // HALF_STEPS,
        0,
    )
    kda_prefill_tiled_body(
        gathered_ref,
        gate_family_ref.at[slot],
        fb_family_ref.at[slot],
        proj_family_ref.at[slot],
        conv_family_ref.at[slot],
        alog_family_ref.at[slot],
        dt_family_ref.at[slot],
        kda_norm_family_ref.at[slot],
        rec_in_ref,
        hist_in_ref,
        attended_ref,
        rec_out_ref,
        hist_out_ref,
        *kda_scratch,
        tile=STORE_TILE,
        decay_lower_bound=-5.0,
        epsilon=eps,
        checkpoint=(rec_checkpoint_ref, hist_checkpoint_ref),
        segments=(segment_ref, count),
        tiles=(
            first_tile,
            end_tile,
            step == HALF_STEPS,
            step == HALF_STEPS * (blocks + 1) - 1,
        ),
        block_tiles=block_tiles,
        projection=(
            output_family_ref.at[slot],
            projection_scratch[0],
            partial_ref,
            projection_scratch[1],
            projection_scratch[2],
        ),
    )


def _kda_mix_shapes(
    mix_tile: int,
) -> tuple[tuple[tuple[int, ...], jax.typing.DTypeLike], ...]:
    """Return pooled scratch shape specs for KDA residual mixture."""
    return (
        ((2, mix_tile, HIDDEN_SIZE), jnp.bfloat16),
        ((mix_tile, HIDDEN_SIZE), jnp.bfloat16),
        ((2, mix_tile, HIDDEN_SIZE), jnp.bfloat16),
        ((mix_tile, HIDDEN_SIZE), jnp.float32),
    )


def _kda_reduce_shapes(
    stripe: int,
) -> tuple[tuple[tuple[int, ...], jax.typing.DTypeLike], ...]:
    """Return pooled scratch shape specs for KDA half-ring reduce-scatter."""
    return (
        ((2, 2, stripe, HIDDEN_SIZE), jnp.bfloat16),
        ((2, stripe, HIDDEN_SIZE), jnp.bfloat16),
        ((2, 2, stripe, HIDDEN_SIZE), jnp.bfloat16),
    )


def make_norm_gather_kda_sp(
    mesh: jax.sharding.Mesh,
    *,
    seq_len: int,
    segments: int,
    token_ranks: Sequence[int] | None = None,
    collective_id: int = 38,
    eps: float = 1e-5,
) -> Callable[..., tuple[jax.Array, ...]]:
    """Build single-call pipelined mixture, all-gather, KDA, and reduce-scatter."""
    if mesh.size != TP_SIZE or tuple(mesh.axis_names) != (AXIS_SP,):
        msg = "norm/gather/KDA requires a flat TP32 sequence-parallel mesh"
        raise ValueError(msg)
    check_window(seq_len, "fused norm/gather/KDA")
    if not segments:
        msg = "the KDA layer runs packed windows"
        raise ValueError(msg)
    state_lead = (segments,)
    rows, width = seq_len // TP_SIZE, HIDDEN_SIZE
    stripe = window_stripe(rows)
    blocks = rows // stripe
    if token_ranks is None:
        token_ranks = tuple(range(TP_SIZE))
    tables = jnp.asarray(ring_tables(list(mesh.devices.flat), token_ranks), jnp.int32)
    streams = len(_half_ring_streams(width))
    pool_bytes = max(
        sum(
            pool_alias.storage_size(shape, jnp.dtype(dtype))
            for shape, dtype in _kda_mix_shapes(stripe)
        ),
        sum(
            pool_alias.storage_size(shape, jnp.dtype(dtype))
            for shape, dtype in _kda_reduce_shapes(stripe)
        ),
    )
    shapes = (
        jax.ShapeDtypeStruct((width, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((HEAD_DIM, _PROJECTION_HEAD_WIDTH), jnp.bfloat16),
        jax.ShapeDtypeStruct((width, 1664), jnp.bfloat16),
        jax.ShapeDtypeStruct((4, 3, 3, HEAD_DIM), jnp.float32),
        jax.ShapeDtypeStruct((3, 1), jnp.float32),
        jax.ShapeDtypeStruct((3, HEAD_DIM), jnp.float32),
        jax.ShapeDtypeStruct((8, HEAD_DIM), jnp.float32),
    )
    scratch = cast(
        "pl_core.ScratchShapeTree",
        (
            pltpu.VMEM((pool_bytes // VREG_LANE_COUNT, VREG_LANE_COUNT), jnp.uint8),
            *mixture_norm_tiled_scratch_shapes(width, stripe)[4:],
            pltpu.SemaphoreType.DMA((streams,)),
            pltpu.SemaphoreType.DMA((streams,)),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((2, streams)),
            pltpu.SemaphoreType.DMA((streams,)),
            pltpu.SemaphoreType.DMA((2, streams)),
            pltpu.SemaphoreType.REGULAR((streams,)),
            pltpu.SemaphoreType.DMA,
            pltpu.VMEM((_PROJECTION_HEAD_WIDTH, width), jnp.bfloat16),
            pltpu.VMEM((2, STORE_TILE, width), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((2,)),
            *kda_prefill_tiled_scratch_shapes(width, STORE_TILE, *shapes),
        ),
    )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    output_shapes = (
        jax.ShapeDtypeStruct((rows, width), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq_len, width), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq_len, _PROJECTION_HEAD_WIDTH), jnp.bfloat16),
        jax.ShapeDtypeStruct((*state_lead, 3, HEAD_DIM, HEAD_DIM), jnp.float32),
        jax.ShapeDtypeStruct((*state_lead, 3, 3, 3, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, width), jnp.bfloat16),
        jax.ShapeDtypeStruct((*state_lead, 3, HEAD_DIM, HEAD_DIM), jnp.float32),
        jax.ShapeDtypeStruct((*state_lead, 3, 3, 3, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq_len, width), jnp.bfloat16),
    )
    call = pl.pallas_call(
        functools.partial(_kda_layer_body, count=segments, eps=eps, blocks=blocks),
        out_shape=output_shapes,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(1, HALF_STEPS * (blocks + 2)),
            in_specs=(hbm,) * 14,
            out_specs=(hbm,) * len(output_shapes),
            scratch_shapes=scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id, vmem_limit_bytes=_KDA_LAYER_VMEM_LIMIT
        ),
        name="kimi_prefill_norm_gather_kda_sp",
    )

    def unshard_first(value: jax.Array) -> jax.Array:
        """Select leading rank slice."""
        return value[0]

    def local(
        prefix: jax.Array,
        blocks_in: jax.Array,
        folded: jax.Array,
        norm: jax.Array,
        weights: Mapping[str, jax.Array],
        rec: jax.Array,
        hist: jax.Array,
        active: jax.Array,
        length: jax.Array,
        slot: jax.Array,
        segment_desc: jax.Array,
    ) -> tuple[jax.Array, ...]:
        """Invoke fused KDA layer on one rank."""
        w = jax.tree.map(unshard_first, weights)
        outputs = cast(
            "tuple[jax.Array, ...]",
            call(
                slot,
                segment_desc,
                active,
                tables,
                length,
                prefix[0],
                blocks_in[0],
                folded[0],
                norm[0],
                w["k_gate"],
                w["k_fb"],
                w["k_projection"],
                w["k_conv"],
                w["k_a_log"],
                w["k_dt"],
                w["k_norm"],
                rec[0],
                hist[0],
                w["k_o"],
            ),
        )
        normalized, gathered, attended, new_rec, new_hist, final, rec_at, hist_at, _ = (
            outputs
        )
        return (
            normalized[None],
            gathered,
            attended[None],
            new_rec[None],
            new_hist[None],
            final[None],
            rec_at[None],
            hist_at[None],
        )

    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(AXIS_SP),) * 7 + (P(), P(), P(), P()),
            out_specs=(
                P(AXIS_SP),
                P(),
                P(AXIS_SP),
                P(AXIS_SP),
                P(AXIS_SP),
                P(AXIS_SP),
                P(AXIS_SP),
                P(AXIS_SP),
            ),
            check_vma=False,
        )
    )


# ============================================================================
# Dense FFN layer
# ============================================================================


def make_dense_ffn_sp(
    mesh: jax.sharding.Mesh, *, seq_len: int
) -> Callable[..., jax.Array]:
    """Build layer-0 dense FFN callable with TP32 output reduce-scatter."""
    if mesh.size != TP_SIZE or len(mesh.axis_names) != 1 or seq_len % TP_SIZE:
        msg = "dense FFN SP requires flat TP32 and T divisible by 32"
        raise ValueError(msg)
    axis_name = mesh.axis_names[0]
    local_tokens = seq_len // TP_SIZE

    def local(
        hidden: jax.Array,
        prefix: jax.Array,
        gate_up: jax.Array,
        down: jax.Array,
        length: jax.Array,
    ) -> jax.Array:
        """Compute one rank's dense FFN shard and reduce-scatter back."""
        x = hidden[0]
        if (
            x.shape != (local_tokens, HIDDEN_SIZE)
            or gate_up[0].shape != (HIDDEN_SIZE, _DENSE_GU_COLS)
            or down[0].shape != (_DENSE_DOWN_ROWS, HIDDEN_SIZE)
        ):
            msg = "dense FFN weights must use K3 TP32 shapes"
            raise ValueError(msg)
        x_all = jax.lax.all_gather(x, axis_name, tiled=True)
        gu = jax.lax.dot_general(
            x_all,
            gate_up[0],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        gate, value = jnp.split(gu, 2, axis=1)
        middle = expert_activation(
            gate.astype(jnp.float32), value.astype(jnp.float32)
        ).astype(jnp.bfloat16)
        partial = jax.lax.dot_general(
            middle,
            down[0],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        reduced = jax.lax.psum_scatter(
            partial, axis_name, scatter_dimension=0, tiled=True
        ).astype(jnp.bfloat16)
        rank = mesh_axis_index(axis_name)
        valid = (
            sp_token_index(rank, jnp.arange(local_tokens), window_stripe(local_tokens))
            < length[0]
        )
        return (prefix[0] + jnp.where(valid[:, None], reduced, 0))[None]

    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(
                P(axis_name),
                P(axis_name),
                P(axis_name),
                P(axis_name),
                P(),
            ),
            out_specs=P(axis_name),
            check_vma=False,
        )
    )


# ============================================================================
# Four-layer groups
# ============================================================================

_GROUP4_TAP_OFFSET = 2
_GROUP4_KDA_COUNT = 3
_KDA_OUTPUT_INDICES = (3, 4, 6, 7)


def _choose_family_slot(family: jax.Array, slot: jax.Array | int) -> jax.Array:
    """Slice ``slot`` along axis 1 of ``family``."""
    return jax.lax.dynamic_index_in_dim(family, slot, axis=1, keepdims=False)


@dataclasses.dataclass(frozen=True, slots=True)
class _Group4Ops:
    """Compiled sublayer callables shared across a 4-layer regular group."""

    mixture: Callable[..., jax.Array]
    norm_gather: Callable[..., tuple[jax.Array, jax.Array]]
    kda: Callable[..., tuple[jax.Array, ...]]
    mla: Callable[..., tuple[jax.Array, jax.Array]]
    moe: Callable[..., jax.Array]


@dataclasses.dataclass(frozen=True, slots=True)
class _Group4StepInputs:
    """Per-call family dictionaries and descriptors for ``_run_group4_layer``."""

    common: Mapping[str, jax.Array]
    kda_families: Mapping[str, jax.Array]
    mla_families: Mapping[str, jax.Array]
    moe_families: Mapping[str, jax.Array]
    length: jax.Array
    group: jax.Array
    segment_desc: jax.Array
    arena: CpPrefix


def _run_group4_layer(
    ops: _Group4Ops,
    step_inputs: _Group4StepInputs,
    carry: tuple[jax.Array, jax.Array, tuple[jax.Array, ...], tuple[jax.Array, ...]],
    offset: int,
) -> tuple[jax.Array, tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    """Execute layer ``offset`` (0..2 KDA, 3 MLA) plus MoE inside a 4-layer group."""
    prefix, blocks, kda_states, mla_cache = carry
    base = step_inputs.group[0] * 4
    index = base + offset
    attn_res = _choose_family_slot(step_inputs.common["attn_res"], index)
    attn_norm = _choose_family_slot(step_inputs.common["attn_norm"], index)
    attn_active = jnp.asarray([(index - 1) // 12 + 1], jnp.int32)
    if offset < _GROUP4_KDA_COUNT:
        slot = base - step_inputs.group[0] + offset
        result = ops.kda(
            prefix,
            blocks,
            attn_res,
            attn_norm,
            step_inputs.kda_families,
            _choose_family_slot(kda_states[0], slot),
            _choose_family_slot(kda_states[1], slot),
            attn_active,
            step_inputs.length,
            jnp.asarray([slot], jnp.int32),
            step_inputs.segment_desc,
        )
        after_attention = result[5]
        kda_updates = tuple(result[i] for i in _KDA_OUTPUT_INDICES)
        kda_states = _update_state_families(kda_states, kda_updates, slot)
    else:
        slot = step_inputs.group[0]
        _, gathered = ops.norm_gather(prefix, blocks, attn_res, attn_norm, attn_active)
        after_attention, *updates = ops.mla(
            gathered,
            prefix,
            step_inputs.mla_families,
            step_inputs.segment_desc,
            step_inputs.length,
            jnp.asarray([slot], jnp.int32),
            step_inputs.arena,
        )
        mla_cache = _update_state_families(mla_cache, updates, slot)
    ffn_input = ops.mixture(
        after_attention,
        blocks,
        _choose_family_slot(step_inputs.common["mlp_res"], index),
        _choose_family_slot(step_inputs.common["ffn_norm"], index),
        jnp.asarray([index // 12 + 1], jnp.int32),
    )
    next_prefix = ops.moe(
        ffn_input,
        after_attention,
        step_inputs.moe_families,
        step_inputs.length,
        jnp.asarray([index - 1], jnp.int32),
    )
    return next_prefix, kda_states, mla_cache


# One straight-line group program reused through a runtime group index:
# SparseCore row movement does not lower inside a dynamic JAX loop.
def make_regular_group4_sp_indexed(
    mesh: jax.sharding.Mesh,
    *,
    seq_len: int = OVERFLOW_WINDOW_TOKENS,
    segments: int,
    prefix_capacity: int,
    mla_flash_tile: int = PAGE_TOKENS,
) -> Callable[
    ...,
    tuple[
        jax.Array,
        jax.Array,
        tuple[jax.Array, ...],
        tuple[jax.Array, ...],
        jax.Array,
    ],
]:
    """Build runtime-indexed 4-layer group callable (3 KDA + 1 MLA, each + MoE)."""
    if mesh.size != TP_SIZE or tuple(mesh.axis_names) != (AXIS_SP,):
        msg = "indexed four-layer group requires flat TP32 sequence-parallel mesh"
        raise ValueError(msg)
    check_window(seq_len, "indexed four-layer group")
    sharding = NamedSharding(mesh, P(AXIS_SP))
    ops = _Group4Ops(
        mixture=make_mixture_norm_sp(mesh, seq_len=seq_len),
        norm_gather=make_norm_gather_sp(mesh, seq_len=seq_len),
        kda=make_norm_gather_kda_sp(mesh, seq_len=seq_len, segments=segments),
        mla=make_indexed_mla_layer_sp_ring(
            mesh,
            seq_len=seq_len,
            flash_tile=mla_flash_tile,
            segments=segments,
            prefix_capacity=prefix_capacity,
        ),
        moe=make_indexed_moe_sp_ring(mesh, seq_len=seq_len),
    )

    def run(
        prefix: jax.Array,
        blocks: jax.Array,
        common: Mapping[str, jax.Array],
        kda_families: Mapping[str, jax.Array],
        mla_families: Mapping[str, jax.Array],
        moe_families: Mapping[str, jax.Array],
        kda_state: Sequence[jax.Array],
        mla_state: Sequence[jax.Array],
        length: jax.Array,
        group: jax.Array,
        segment_desc: jax.Array,
        arena: CpPrefix,
    ) -> tuple[
        jax.Array,
        jax.Array,
        tuple[jax.Array, ...],
        tuple[jax.Array, ...],
        jax.Array,
    ]:
        """Execute layers ``4*group .. 4*group + 3`` over a packed window."""
        base = group[0] * 4
        kda_states = tuple(kda_state)
        mla_cache = tuple(mla_state)
        _validate_stack_states(kda_states, mla_cache)

        def set_residual_block(x: jax.Array) -> jax.Array:
            """Record residual block snapshot at 12-layer boundary."""
            return x.at[:, base // 12].set(prefix)

        def keep_residual_block(x: jax.Array) -> jax.Array:
            """Leave residual blocks unchanged inside a 12-layer span."""
            return x

        blocks = jax.lax.cond(
            base % 12 == 0, set_residual_block, keep_residual_block, blocks
        )
        step_inputs = _Group4StepInputs(
            common=common,
            kda_families=kda_families,
            mla_families=mla_families,
            moe_families=moe_families,
            length=length,
            group=group,
            segment_desc=segment_desc,
            arena=arena,
        )
        tap = prefix
        for offset in range(4):
            if offset == _GROUP4_TAP_OFFSET:
                tap = prefix
            prefix, kda_states, mla_cache = _run_group4_layer(
                ops, step_inputs, (prefix, blocks, kda_states, mla_cache), offset
            )
        return prefix, blocks, kda_states, mla_cache, tap

    return jax.jit(
        run,
        out_shardings=(
            sharding,
            sharding,
            (sharding,) * len(_KDA_OUTPUT_INDICES),
            (sharding,),
            sharding,
        ),
    )


def make_regular_group8_sp_indexed(
    mesh: jax.sharding.Mesh,
    *,
    seq_len: int = OVERFLOW_WINDOW_TOKENS,
    segments: int,
    prefix_capacity: int,
    mla_flash_tile: int = PAGE_TOKENS,
) -> Callable[
    ...,
    tuple[
        jax.Array,
        jax.Array,
        tuple[jax.Array, ...],
        tuple[jax.Array, ...],
        jax.Array,
        jax.Array,
    ],
]:
    """Build a compiled callable running two consecutive four-layer groups."""
    group4 = make_regular_group4_sp_indexed(
        mesh,
        seq_len=seq_len,
        mla_flash_tile=mla_flash_tile,
        segments=segments,
        prefix_capacity=prefix_capacity,
    )
    sharding = NamedSharding(mesh, P(AXIS_SP))

    def run(
        prefix: jax.Array,
        blocks: jax.Array,
        common: Mapping[str, jax.Array],
        kda_families: Mapping[str, jax.Array],
        mla_families: Mapping[str, jax.Array],
        moe_families: Mapping[str, jax.Array],
        kda_state: Sequence[jax.Array],
        mla_state: Sequence[jax.Array],
        length: jax.Array,
        pair: jax.Array,
        segment_desc: jax.Array,
        arena: CpPrefix,
    ) -> tuple[
        jax.Array,
        jax.Array,
        tuple[jax.Array, ...],
        tuple[jax.Array, ...],
        jax.Array,
        jax.Array,
    ]:
        """Run consecutive four-layer groups ``2 * pair + 1`` and ``2 * pair + 2``."""
        first = jnp.asarray([2 * pair[0] + 1], jnp.int32)
        second = first + 1
        arenas = arena.split()
        middle, blocks, next_kda, next_mla, _ = group4(
            prefix,
            blocks,
            common,
            kda_families,
            mla_families,
            moe_families,
            kda_state,
            mla_state,
            length,
            first,
            segment_desc,
            arenas[0],
        )
        out_prefix, blocks, next_kda, next_mla, tap = group4(
            middle,
            blocks,
            common,
            kda_families,
            mla_families,
            moe_families,
            next_kda,
            next_mla,
            length,
            second,
            segment_desc,
            arenas[1],
        )
        return out_prefix, blocks, next_kda, next_mla, middle, tap

    return jax.jit(
        run,
        out_shardings=(
            sharding,
            sharding,
            (sharding,) * 4,
            (sharding,),
            sharding,
            sharding,
        ),
    )


def make_final_mla_sp_indexed(
    mesh: jax.sharding.Mesh,
    *,
    seq_len: int = OVERFLOW_WINDOW_TOKENS,
    segments: int,
    prefix_capacity: int,
    mla_flash_tile: int = PAGE_TOKENS,
) -> Callable[..., tuple[jax.Array, jax.Array, tuple[jax.Array, ...]]]:
    """Build a compiled callable running final decoder layer 92 over a window."""
    if mesh.size != TP_SIZE or tuple(mesh.axis_names) != (AXIS_SP,):
        msg = "final MLA layer requires flat TP32 sequence-parallel mesh"
        raise ValueError(msg)
    check_window(seq_len, "final MLA layer")
    sharding = NamedSharding(mesh, P(AXIS_SP))
    mixture = make_mixture_norm_sp(mesh, seq_len=seq_len)
    norm_gather = make_norm_gather_sp(mesh, seq_len=seq_len)
    mla = make_indexed_mla_layer_sp_ring(
        mesh,
        seq_len=seq_len,
        flash_tile=mla_flash_tile,
        segments=segments,
        prefix_capacity=prefix_capacity,
    )
    moe = make_indexed_moe_sp_ring(mesh, seq_len=seq_len)

    def run(
        prefix: jax.Array,
        blocks: jax.Array,
        common: Mapping[str, jax.Array],
        mla_families: Mapping[str, jax.Array],
        moe_families: Mapping[str, jax.Array],
        mla_state: Sequence[jax.Array],
        length: jax.Array,
        segment_desc: jax.Array,
        arena: CpPrefix,
    ) -> tuple[jax.Array, jax.Array, tuple[jax.Array, ...]]:
        """Execute final MLA + MoE layer 92 and update MLA state slot 23."""
        if len(mla_state) != 1:
            msg = "mla_state is (latents,)"
            raise ValueError(msg)
        attn_res = common["attn_res"][:, 92]
        attn_norm = common["attn_norm"][:, 92]
        active = jnp.asarray([_RESIDUAL_BLOCKS], jnp.int32)
        _, gathered = norm_gather(prefix, blocks, attn_res, attn_norm, active)
        after_attention, *updates = mla(
            gathered,
            prefix,
            mla_families,
            segment_desc,
            length,
            jnp.asarray([23], jnp.int32),
            arena,
        )
        mla_cache = _update_state_families(mla_state, updates, 23)
        ffn_input = mixture(
            after_attention,
            blocks,
            common["mlp_res"][:, 92],
            common["ffn_norm"][:, 92],
            active,
        )
        out_prefix = moe(
            ffn_input,
            after_attention,
            moe_families,
            length,
            jnp.asarray([91], jnp.int32),
        )
        return out_prefix, blocks, mla_cache

    return jax.jit(run, out_shardings=(sharding, sharding, (sharding,)))


# ============================================================================
# Layer stack
# ============================================================================

_MIN_STACK_DEPTH = 2
_KDA_STATE_ARITY = 4
_FINAL_LAYER_INDEX = NUM_LAYERS - 1
_MLA_GROUP_REM = 3


def _validate_prefill_stack(
    mesh: jax.sharding.Mesh,
    *,
    seq_len: int,
    depth: int,
    tap_layer: int | None,
) -> None:
    """Validate mesh sharding, window size, depth, and optional tap layer."""
    if mesh.size != TP_SIZE or tuple(mesh.axis_names) != (AXIS_SP,):
        msg = "indexed prefill stack requires flat TP32 sequence-parallel mesh"
        raise ValueError(msg)
    if tap_layer is not None and not 1 <= tap_layer < depth:
        msg = "tap_layer must be one of the stack's later layers"
        raise ValueError(msg)
    check_window(seq_len, "indexed prefill stack")
    if not _MIN_STACK_DEPTH <= depth <= NUM_LAYERS:
        msg = "indexed prefill stack needs depth 2..93"
        raise ValueError(msg)


def _validate_stack_states(
    kda_state: tuple[jax.Array, ...],
    mla_state: Sequence[jax.Array],
) -> None:
    """Validate tuple arities of KDA and MLA state containers."""
    if len(kda_state) != _KDA_STATE_ARITY:
        msg = "kda_state is (rec, hist, rec_at, hist_at)"
        raise ValueError(msg)
    if len(mla_state) != 1:
        msg = "mla_state is (latents,)"
        raise ValueError(msg)


def _put_family_state(
    family: jax.Array,
    state: jax.Array,
    slot: int | jax.Array,
) -> jax.Array:
    """Write a single layer state slice into a family-major state tensor."""
    return jax.lax.dynamic_update_slice(
        family,
        state[:, None],
        (0, slot, *([0] * (family.ndim - 2))),
    )


def _update_state_families(
    families: Sequence[jax.Array],
    updates: Sequence[jax.Array],
    slot: int | jax.Array,
) -> tuple[jax.Array, ...]:
    """Update each family-major state tensor at the target layer slot."""
    return tuple(
        _put_family_state(family, update, slot)
        for family, update in zip(families, updates, strict=True)
    )


@dataclasses.dataclass(frozen=True, slots=True)
class _StackOps:
    """Compiled sub-layer callables shared across the indexed prefill stack."""

    mixture: Callable[..., jax.Array]
    norm_gather: Callable[..., tuple[jax.Array, jax.Array]]
    kda: Callable[..., tuple[jax.Array, ...]]
    mla: Callable[..., tuple[jax.Array, ...]]
    moe: Callable[..., jax.Array]
    dense: Callable[..., jax.Array]


@dataclasses.dataclass(frozen=True, slots=True)
class _StackRunInputs:
    """Per-call weights, control inputs, and tap layer for the prefill stack."""

    common: Mapping[str, jax.Array]
    kda_families: Mapping[str, jax.Array]
    mla_families: Mapping[str, jax.Array]
    moe_families: Mapping[str, jax.Array]
    length: jax.Array
    segment_desc: jax.Array
    arena: CpPrefix
    tap_layer: int | None


def _stack_active(value: jax.Array | int) -> jax.Array:
    """Pack integer ``value`` into a single-element int32 array."""
    return jnp.asarray([value], jnp.int32)


def _run_stack_kda(
    ops: _StackOps,
    stack_inputs: _StackRunInputs,
    cur_state: tuple[jax.Array, jax.Array, tuple[jax.Array, ...]],
    attn_inputs: tuple[jax.Array, jax.Array, jax.Array],
    slot: jax.Array | int,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    """Run one KDA attention layer at ``slot`` and update KDA state families."""
    cur_prefix, cur_blocks, states = cur_state
    attn_res, attn_norm, attn_active = attn_inputs
    result = ops.kda(
        cur_prefix,
        cur_blocks,
        attn_res,
        attn_norm,
        stack_inputs.kda_families,
        _choose_family_slot(states[0], slot),
        _choose_family_slot(states[1], slot),
        attn_active,
        stack_inputs.length,
        _stack_active(slot),
        stack_inputs.segment_desc,
    )
    kda_updates = (result[3], result[4], result[6], result[7])
    return result[5], _update_state_families(states, kda_updates, slot)


def _stack_layer_step(
    ops: _StackOps,
    stack_inputs: _StackRunInputs,
    index: jax.Array,
    carry: tuple[
        jax.Array,
        jax.Array,
        tuple[jax.Array, ...],
        tuple[jax.Array, ...],
        jax.Array | None,
    ],
) -> tuple[
    jax.Array,
    jax.Array,
    tuple[jax.Array, ...],
    tuple[jax.Array, ...],
    jax.Array | None,
]:
    """Execute one decoder layer ``index`` inside the prefill stack loop."""
    cur_prefix, cur_blocks, cur_kda, cur_mla, tap = carry
    if tap is not None and stack_inputs.tap_layer is not None:
        tap = jnp.where(index == stack_inputs.tap_layer, cur_prefix, tap)

    def set_block(x: jax.Array) -> jax.Array:
        """Store residual block at 12-layer boundary."""
        return x.at[:, index // 12].set(cur_prefix)

    def keep_block(x: jax.Array) -> jax.Array:
        """Preserve residual blocks between boundaries."""
        return x

    cur_blocks = jax.lax.cond(index % 12 == 0, set_block, keep_block, cur_blocks)
    attn_res = _choose_family_slot(stack_inputs.common["attn_res"], index)
    attn_norm = _choose_family_slot(stack_inputs.common["attn_norm"], index)
    attn_active = _stack_active((index - 1) // 12 + 1)

    def kda_branch(
        state: tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]],
    ) -> tuple[jax.Array, tuple[jax.Array, ...], tuple[jax.Array, ...]]:
        """Run KDA attention branch for layer ``index``."""
        st_kda, st_mla = state
        slot = index - (index + 1) // 4
        after, next_kda = _run_stack_kda(
            ops,
            stack_inputs,
            (cur_prefix, cur_blocks, st_kda),
            (attn_res, attn_norm, attn_active),
            slot,
        )
        return after, next_kda, st_mla

    def mla_branch(
        state: tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]],
    ) -> tuple[jax.Array, tuple[jax.Array, ...], tuple[jax.Array, ...]]:
        """Run MLA attention branch for layer ``index``."""
        st_kda, st_mla = state
        slot = index // 4
        _, gathered = ops.norm_gather(
            cur_prefix, cur_blocks, attn_res, attn_norm, attn_active
        )
        after, *updates = ops.mla(
            gathered,
            cur_prefix,
            stack_inputs.mla_families,
            stack_inputs.segment_desc,
            stack_inputs.length,
            _stack_active(slot),
            stack_inputs.arena,
        )
        return after, st_kda, _update_state_families(st_mla, updates, slot)

    after_attn, cur_kda, cur_mla = jax.lax.cond(
        (index % 4 == _MLA_GROUP_REM) | (index == _FINAL_LAYER_INDEX),
        mla_branch,
        kda_branch,
        (cur_kda, cur_mla),
    )
    next_ffn = ops.mixture(
        after_attn,
        cur_blocks,
        _choose_family_slot(stack_inputs.common["mlp_res"], index),
        _choose_family_slot(stack_inputs.common["ffn_norm"], index),
        _stack_active(index // 12 + 1),
    )
    next_prefix = ops.moe(
        next_ffn,
        after_attn,
        stack_inputs.moe_families,
        stack_inputs.length,
        _stack_active(index - 1),
    )
    return next_prefix, cur_blocks, cur_kda, cur_mla, tap


def make_prefill_stack_sp_indexed(
    mesh: jax.sharding.Mesh,
    *,
    seq_len: int = OVERFLOW_WINDOW_TOKENS,
    depth: int = NUM_LAYERS,
    segments: int,
    prefix_capacity: int,
    mla_flash_tile: int = PAGE_TOKENS,
    unroll: bool = False,
    tap_layer: int | None = None,
) -> Callable[
    ...,
    tuple[
        jax.Array,
        jax.Array,
        tuple[jax.Array, ...],
        tuple[jax.Array, ...],
        *tuple[jax.Array, ...],
    ],
]:
    """Build a compiled callable running layers 0..depth-1 over a packed window."""
    _validate_prefill_stack(mesh, seq_len=seq_len, depth=depth, tap_layer=tap_layer)
    sharding = NamedSharding(mesh, P(AXIS_SP))
    ops = _StackOps(
        mixture=make_mixture_norm_sp(mesh, seq_len=seq_len),
        norm_gather=make_norm_gather_sp(mesh, seq_len=seq_len),
        kda=make_norm_gather_kda_sp(mesh, seq_len=seq_len, segments=segments),
        mla=make_indexed_mla_layer_sp_ring(
            mesh,
            seq_len=seq_len,
            flash_tile=mla_flash_tile,
            segments=segments,
            prefix_capacity=prefix_capacity,
        ),
        moe=make_indexed_moe_sp_ring(mesh, seq_len=seq_len),
        dense=make_dense_ffn_sp(mesh, seq_len=seq_len),
    )

    def run(
        hidden: jax.Array,
        common: Mapping[str, jax.Array],
        kda_families: Mapping[str, jax.Array],
        mla_families: Mapping[str, jax.Array],
        moe_families: Mapping[str, jax.Array],
        kda_state: Sequence[jax.Array],
        mla_state: Sequence[jax.Array],
        length: jax.Array,
        segment_desc: jax.Array,
        arena: CpPrefix,
    ) -> tuple[
        jax.Array,
        jax.Array,
        tuple[jax.Array, ...],
        tuple[jax.Array, ...],
        *tuple[jax.Array, ...],
    ]:
        """Run layer 0 (dense FFN) then ``fori_loop`` over layers 1..depth-1."""
        kda_tuple = tuple(kda_state)
        _validate_stack_states(kda_tuple, mla_state)
        stack_inputs = _StackRunInputs(
            common=common,
            kda_families=kda_families,
            mla_families=mla_families,
            moe_families=moe_families,
            length=length,
            segment_desc=segment_desc,
            arena=arena,
            tap_layer=tap_layer,
        )
        prefix = hidden
        blocks = (
            jnp.zeros(
                (TP_SIZE, _RESIDUAL_BLOCKS, seq_len // TP_SIZE, HIDDEN_SIZE),
                jnp.bfloat16,
            )
            .at[:, 0]
            .set(prefix)
        )
        after_attention, kda_states = _run_stack_kda(
            ops,
            stack_inputs,
            (prefix, blocks, kda_tuple),
            (
                _choose_family_slot(common["attn_res"], 0),
                _choose_family_slot(common["attn_norm"], 0),
                _stack_active(0),
            ),
            0,
        )
        ffn_input = ops.mixture(
            after_attention,
            blocks,
            _choose_family_slot(common["mlp_res"], 0),
            _choose_family_slot(common["ffn_norm"], 0),
            _stack_active(1),
        )
        prefix = ops.dense(
            ffn_input,
            after_attention,
            common["dense_gu"],
            common["dense_down"],
            length,
        )
        init_tap = prefix if tap_layer is not None else None
        result = jax.lax.fori_loop(
            1,
            depth,
            functools.partial(_stack_layer_step, ops, stack_inputs),
            (prefix, blocks, kda_states, tuple(mla_state), init_tap),
            unroll=unroll,
        )
        if tap_layer is not None and result[4] is not None:
            return (result[0], result[1], result[2], result[3], result[4])
        return (result[0], result[1], result[2], result[3])

    return jax.jit(
        run,
        out_shardings=(
            sharding,
            sharding,
            (sharding,) * len(_KDA_OUTPUT_INDICES),
            (sharding,),
            *((sharding,) if tap_layer is not None else ()),
        ),
    )


# ============================================================================
# Embedding and logits
# ============================================================================


def make_embedding_sp(
    mesh: jax.sharding.Mesh,
    *,
    seq_len: int,
    vocab_per_rank: int = 5120,
) -> Callable[..., jax.Array]:
    """Build a compiled callable embedding token IDs via a vocab-sharded table."""
    if mesh.size != TP_SIZE or len(mesh.axis_names) != 1 or seq_len % TP_SIZE:
        msg = "embedding SP requires flat TP32 and T divisible by 32"
        raise ValueError(msg)
    axis_name = mesh.axis_names[0]
    local_tokens = seq_len // TP_SIZE

    def local(tokens: jax.Array, embedding: jax.Array, length: jax.Array) -> jax.Array:
        """Look up vocab-sharded embeddings and reduce-scatter to SP rows."""
        rank = mesh_axis_index(axis_name)
        token_ids = jax.lax.all_gather(tokens[0], axis_name).reshape(seq_len)
        local_ids = token_ids - rank * vocab_per_rank
        matched = (local_ids >= 0) & (local_ids < vocab_per_rank)
        table = embedding[0]
        if table.shape != (vocab_per_rank, HIDDEN_SIZE):
            msg = "embedding table has wrong per-rank shape"
            raise ValueError(msg)
        partial = jnp.where(
            matched[:, None], table[jnp.clip(local_ids, 0, vocab_per_rank - 1)], 0
        ).astype(jnp.float32)
        hidden = jax.lax.psum_scatter(
            partial, axis_name, scatter_dimension=0, tiled=True
        ).astype(jnp.bfloat16)
        valid = (
            sp_token_index(rank, jnp.arange(local_tokens), window_stripe(local_tokens))
            < length[0]
        )
        return jnp.where(valid[:, None], hidden, 0)[None]

    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(axis_name), P(axis_name), P()),
            out_specs=P(axis_name),
            check_vma=False,
        )
    )


def make_prefill_logits_sp(
    mesh: jax.sharding.Mesh,
    *,
    seq_len: int,
    layers: int,
    segments: int,
) -> Callable[..., jax.Array]:
    """Build a compiled callable computing last-token vocab-sharded logits."""
    if mesh.size != TP_SIZE or len(mesh.axis_names) != 1 or seq_len % TP_SIZE:
        msg = "logits SP requires flat TP32 and T divisible by 32"
        raise ValueError(msg)
    if not _MIN_STACK_DEPTH <= layers <= NUM_LAYERS:
        msg = "K3 layer count must be 2..93"
        raise ValueError(msg)
    axis_name = mesh.axis_names[0]
    active = (layers - 1) // 12 + 1
    stripe = window_stripe(seq_len // TP_SIZE)

    def local_packed(
        prefix: jax.Array,
        blocks: jax.Array,
        output_res: jax.Array,
        final_norm: jax.Array,
        lm_head: jax.Array,
        segment_desc: jax.Array,
    ) -> jax.Array:
        """Gather each segment's last token, RMS-normalize, and project logits."""
        rank = mesh_axis_index(axis_name)
        rows: list[jax.Array] = []
        for s in range(segments):
            used = segment_desc[segments + s] > 0
            last = jnp.maximum(segment_desc[s] + segment_desc[segments + s] - 1, 0)
            block = last // (TP_SIZE * stripe)
            within = last % (TP_SIZE * stripe)
            owner = within // stripe
            row = block * stripe + within % stripe
            local_prefix = jax.lax.dynamic_slice(prefix[0], (row, 0), (1, HIDDEN_SIZE))
            local_blocks = jax.lax.dynamic_slice(
                blocks[0], (0, row, 0), (_RESIDUAL_BLOCKS, 1, HIDDEN_SIZE)
            )
            mixed = residual_mixture(
                local_blocks, local_prefix, output_res[0], active
            ).astype(jnp.float32)
            normalized = (
                mixed
                * jax.lax.rsqrt(jnp.mean(mixed * mixed, axis=1, keepdims=True) + 1e-5)
                * final_norm[0]
            ).astype(jnp.bfloat16)
            rows.append(jnp.where((rank == owner) & used, normalized, 0))
        hidden = jax.lax.psum(jnp.concatenate(rows, axis=0), axis_name)
        logits = jax.lax.dot_general(
            hidden,
            lm_head[0],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        return logits[None]

    return jax.jit(
        jax.shard_map(
            local_packed,
            mesh=mesh,
            in_specs=(
                P(axis_name),
                P(axis_name),
                P(axis_name),
                P(axis_name),
                P(axis_name),
                P(),
            ),
            out_specs=P(axis_name),
            check_vma=False,
        )
    )
