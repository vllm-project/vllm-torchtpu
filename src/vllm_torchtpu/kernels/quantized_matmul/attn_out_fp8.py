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
"""Pallas kernel fusing the attention-output relayout with o_proj FP8 quant.

The batched ragged paged attention kernel emits its output head-major,
``[num_kv_heads, M, q_per_kv // q_packing, q_packing, head_dim]``, and
``finalize_outputs`` hands the model ``o.swapaxes(1, 0).reshape(M, K)``. On
the dynamic FP8 path XLA then fuses ``abs_max`` into that transpose but
materialises the scaled activation as an ``[M, K]`` float32 temporary plus a
float32 relayout copy before the ``o_proj`` ``dot_general``.

This kernel reads the attention output in its native layout via
``unprepare_outputs_to_head_major`` (which folds away against
``finalize_outputs``), computes the per-row ``abs_max`` scale and writes
``[M, K]`` in ``quant_dtype`` plus a lane-broadcast ``[M, num_lanes]``
float32 scale in one VMEM pass. ``quantize_attn_out`` returns ``(x_q,
scale[:, :1])`` so it plugs straight into ``xla_quantized_matmul``.

Numerics match ``xla_quantized_matmul(..., quantize_activation=True)``
bit-for-bit: ``scale = abs_max / dtype_max`` where
``(dtype_max, dtype_min) = get_max_min(quant_dtype)`` (``1.0`` when the row is
all zero), ``x_q = clip(f32(x) / scale, dtype_min, dtype_max).astype(quant_dtype)``,
and ``out = dot(x_q, w_q) * scale * w_scale``.
"""

from __future__ import annotations

import functools

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from vllm_torchtpu.kernels.experimental.batched_rpa.utils import get_dtype_packing
from vllm_torchtpu.kernels.experimental.batched_rpa.wrapper import (
    unprepare_outputs_to_head_major,
)
from vllm_torchtpu.kernels.quantized_matmul.util import get_max_min

_MIB = 1024 * 1024
# Keep scoped Pallas VMEM within half of chip capacity so surrounding XLA ops
# retain their half-VMEM budget and cross-program weight prefetches, and leave
# an extra 1/8 margin inside that cap for Mosaic compiler spills.
_MAX_VMEM_FRACTION_DENOM = 2
_TILE_BUDGET_NUMERATOR = 7
_TILE_BUDGET_DENOMINATOR = 8
# Multi-tile grids need a multiple of 32 rows for 1-byte FP8 sublane packing.
_ROW_TILES = (512, 256, 128, 64, 32)
MIN_ROW_TILE = _ROW_TILES[-1]


def _max_scoped_vmem_bytes(vmem_capacity_bytes: int) -> int:
    """Upper bound on scoped VMEM reserved by ``attn_out_fp8_quant``."""
    return vmem_capacity_bytes // _MAX_VMEM_FRACTION_DENOM


def _tile_budget_bytes(vmem_capacity_bytes: int) -> int:
    """VMEM budget for double-buffered blocks plus in-kernel temporaries."""
    return (
        _max_scoped_vmem_bytes(vmem_capacity_bytes) * _TILE_BUDGET_NUMERATOR
    ) // _TILE_BUDGET_DENOMINATOR


def _row_vmem_bytes(
    num_kv_heads: int,
    q_per_kv: int,
    head_dim: int,
    act_dtype: jnp.dtype,
    quant_dtype: jnp.dtype,
    num_lanes: int,
) -> int:
    """Estimated VMEM bytes per tile row, including temporaries."""
    act_dt = jnp.dtype(act_dtype)
    quant_dt = jnp.dtype(quant_dtype)
    q_packing = get_dtype_packing(act_dt)
    k = num_kv_heads * q_per_kv * head_dim
    num_groups_total = num_kv_heads * (q_per_kv // q_packing)
    # Double-buffered BlockSpec I/O: x_ref + q_ref + f32 s_ref.
    io_bytes = 2 * (act_dt.itemsize * k + quant_dt.itemsize * k + num_lanes * 4)
    # In-kernel temporaries: f32 head-group loads across groups (4*k) plus
    # active group scaling scratch (4 * q_packing * head_dim), quantized
    # `q_blocks` plus the concatenated `[rows, K]` output (`2*k` when
    # concatenating, `k` for a single group), and lane-padded f32
    # `abs_max` / `scale`.
    quant_temp_bytes = (2 if num_groups_total > 1 else 1) * quant_dt.itemsize * k
    temp_bytes = 4 * k + 4 * q_packing * head_dim + quant_temp_bytes + 2 * num_lanes * 4
    return io_bytes + temp_bytes


def vmem_limit_bytes(
    row_tile: int,
    num_kv_heads: int,
    q_per_kv: int,
    head_dim: int,
    act_dtype: jnp.dtype = jnp.bfloat16,
    quant_dtype: jnp.dtype = jnp.float8_e4m3fn,
) -> int:
    """Per-tile scoped VMEM byte limit passed to ``pltpu.CompilerParams``."""
    tpu_info = pltpu.get_tpu_info()
    vmem_capacity_bytes = int(tpu_info.vmem_capacity_bytes)
    num_lanes = int(tpu_info.num_lanes)
    working_set = row_tile * _row_vmem_bytes(
        num_kv_heads, q_per_kv, head_dim, act_dtype, quant_dtype, num_lanes
    )
    padded = working_set + (working_set + 7) // 8
    aligned = ((padded + _MIB - 1) // _MIB) * _MIB
    return min(_max_scoped_vmem_bytes(vmem_capacity_bytes), aligned)


def select_row_tile(
    m: int,
    num_kv_heads: int,
    q_per_kv: int,
    head_dim: int,
    act_dtype: jnp.dtype = jnp.bfloat16,
    quant_dtype: jnp.dtype = jnp.float8_e4m3fn,
) -> int | None:
    """Row tile for an ``[m, K]`` attention output, or None if unsupported.

    Derives packing from ``act_dtype`` and lane count / VMEM capacity from
    ``pltpu.get_tpu_info()``. Called inside JAX lowering, where ``m`` is a
    static per-bucket int, so no symbolic token-count guards reach TorchDynamo.
    """
    act_dt = jnp.dtype(act_dtype)
    quant_dt = jnp.dtype(quant_dtype)
    q_packing = get_dtype_packing(act_dt)
    if (
        m <= 0
        or num_kv_heads <= 0
        or q_per_kv <= 0
        or head_dim <= 0
        or q_packing <= 0
        or q_per_kv % q_packing != 0
        or quant_dt.itemsize != 1
    ):
        return None
    tpu_info = pltpu.get_tpu_info()
    vmem_capacity_bytes = int(tpu_info.vmem_capacity_bytes)
    num_lanes = int(tpu_info.num_lanes)
    if vmem_capacity_bytes <= 0 or num_lanes <= 0 or head_dim % num_lanes != 0:
        return None
    budget = _tile_budget_bytes(vmem_capacity_bytes)
    bytes_per_row = _row_vmem_bytes(
        num_kv_heads, q_per_kv, head_dim, act_dt, quant_dt, num_lanes
    )
    for row_tile in _ROW_TILES:
        if m >= row_tile and m % row_tile == 0 and row_tile * bytes_per_row <= budget:
            return row_tile
    return None


def _attn_out_fp8_kernel(
    x_ref,
    q_ref,
    s_ref,
    *,
    num_kv_heads: int,
    num_groups: int,
    q_packing: int,
    head_dim: int,
    num_lanes: int,
    quant_dtype: jnp.dtype,
) -> None:
    """``x_ref``: ``[kv, rows, q_per_kv // q_packing, q_packing, d]``,
    ``q_ref``: ``[rows, K]``.
    """
    group_width = q_packing * head_dim
    groups = [
        (kv_head, group)
        for kv_head in range(num_kv_heads)
        for group in range(num_groups)
    ]

    def load_group(kv_head: int, group: int) -> jax.Array:
        # Load each packed head group as one lane-dense `[rows, q_packing * d]`
        # block; its C-order flatten already matches `q_ref`'s slot order.
        if q_packing == 1:
            return x_ref[kv_head, :, group, 0, :].astype(jnp.float32)
        packed = x_ref[kv_head, :, group, :, :].astype(jnp.float32)
        return packed.reshape(packed.shape[0], group_width)

    abs_max = None
    for kv_head, group in groups:
        group_max = jnp.max(jnp.abs(load_group(kv_head, group)), axis=1, keepdims=True)
        abs_max = group_max if abs_max is None else jnp.maximum(abs_max, group_max)
    assert abs_max is not None
    dtype_max, dtype_min = get_max_min(quant_dtype)
    scale = abs_max / dtype_max
    scale = jnp.where(scale == 0.0, 1.0, scale)
    q_blocks = [
        jnp.clip(load_group(kv_head, group) / scale, dtype_min, dtype_max).astype(
            quant_dtype
        )
        for kv_head, group in groups
    ]
    q_ref[...] = (
        q_blocks[0] if len(q_blocks) == 1 else jnp.concatenate(q_blocks, axis=1)
    )
    s_ref[...] = jnp.broadcast_to(scale, (scale.shape[0], num_lanes))


def quantize_attn_out(
    x: jax.Array,
    quant_dtype: jnp.dtype = jnp.float8_e4m3fn,
    *,
    num_kv_heads: int,
    q_per_kv: int,
    head_dim: int,
) -> tuple[jax.Array, jax.Array]:
    """Quantize the ``[M, K]`` attention output per row to ``quant_dtype``.

    Returns ``(x_q [M, K], scale [M, 1] float32)``, matching the
    ``quantize_tensor`` contract expected by ``xla_quantized_matmul``.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D attention output [M, K], got {x.shape=}")
    m, k = x.shape
    if k != num_kv_heads * q_per_kv * head_dim:
        raise ValueError(f"{k=} must equal {num_kv_heads=} * {q_per_kv=} * {head_dim=}")
    row_tile = select_row_tile(
        m,
        num_kv_heads,
        q_per_kv,
        head_dim,
        act_dtype=x.dtype,
        quant_dtype=quant_dtype,
    )
    if row_tile is None:
        raise ValueError(
            f"No attention-output FP8 row tile for {x.shape=}, {x.dtype=}, "
            f"{q_per_kv=}, {head_dim=}"
        )
    q_packing = get_dtype_packing(x.dtype)
    num_lanes = int(pltpu.get_tpu_info().num_lanes)
    num_groups = q_per_kv // q_packing
    x_head_major = unprepare_outputs_to_head_major(x, num_kv_heads, q_per_kv, head_dim)
    x_q, scale_wide = pl.pallas_call(
        functools.partial(
            _attn_out_fp8_kernel,
            num_kv_heads=num_kv_heads,
            num_groups=num_groups,
            q_packing=q_packing,
            head_dim=head_dim,
            num_lanes=num_lanes,
            quant_dtype=quant_dtype,
        ),
        out_shape=[
            jax.ShapeDtypeStruct((m, k), quant_dtype),
            jax.ShapeDtypeStruct((m, num_lanes), jnp.float32),
        ],
        grid=(m // row_tile,),
        in_specs=[
            pl.BlockSpec(
                (num_kv_heads, row_tile, num_groups, q_packing, head_dim),
                lambda i: (0, i, 0, 0, 0),
            )
        ],
        out_specs=[
            pl.BlockSpec((row_tile, k), lambda i: (i, 0)),
            pl.BlockSpec((row_tile, num_lanes), lambda i: (i, 0)),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
            vmem_limit_bytes=vmem_limit_bytes(
                row_tile,
                num_kv_heads,
                q_per_kv,
                head_dim,
                act_dtype=x.dtype,
                quant_dtype=quant_dtype,
            ),
        ),
        name="attn_out_fp8_quant",
    )(x_head_major)
    return x_q, scale_wide[:, :1]
