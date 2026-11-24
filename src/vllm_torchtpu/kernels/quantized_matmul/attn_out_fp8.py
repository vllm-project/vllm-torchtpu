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
``[num_kv_heads, M, q_per_kv // 2, 2, head_dim]``, and the wrapper hands the
model ``o.swapaxes(1, 0).reshape(M, K)``. On the dynamic FP8 path XLA then
fuses ``abs_max`` into that transpose but materialises the scaled activation
as an ``[M, K]`` float32 temporary plus a float32 relayout copy before the
``o_proj`` ``dot_general``.

This kernel reads the attention output in its native layout (the
``reshape``/``swapaxes`` pair in ``quantize_attn_out`` folds away against the
wrapper's), computes the per-row ``abs_max`` scale and writes ``[M, K]`` in
``quant_dtype`` plus a lane-broadcast ``[M, 128]`` float32 scale in one VMEM
pass.

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

from vllm_torchtpu.kernels.quantized_matmul.util import get_max_min

_NUM_LANES = 128
# The RPA kernels pack bf16 query heads two per 32-bit word, so their output
# holds the query heads of one KV head as `q_per_kv // 2` pairs.
_Q_PACKING = 2
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


def vmem_limit_bytes(
    row_tile: int,
    num_kv_heads: int,
    q_per_kv: int,
    head_dim: int,
    vmem_capacity_bytes: int,
) -> int:
    """Per-tile scoped VMEM byte limit passed to ``pltpu.CompilerParams``."""
    working_set = row_tile * _row_vmem_bytes(num_kv_heads, q_per_kv, head_dim)
    padded = working_set + (working_set + 7) // 8
    aligned = ((padded + _MIB - 1) // _MIB) * _MIB
    return min(_max_scoped_vmem_bytes(vmem_capacity_bytes), aligned)


def _row_vmem_bytes(num_kv_heads: int, q_per_kv: int, head_dim: int) -> int:
    """Estimated VMEM bytes per tile row, including temporaries."""
    k = num_kv_heads * q_per_kv * head_dim
    num_pairs_total = num_kv_heads * (q_per_kv // _Q_PACKING)
    # Double-buffered BlockSpec I/O: bf16 x_ref (2*k) + 1B q_ref (k) + f32 s_ref.
    io_bytes = 2 * (2 * k + k + _NUM_LANES * 4)
    # In-kernel temporaries: f32 head-pair loads across pairs (4*k) plus active
    # pair scaling scratch (4 * 2 * head_dim), quantized `q_blocks` plus the
    # concatenated `[rows, K]` FP8 value (`2*k` when concatenating, `k` for a
    # single pair), and lane-padded f32 `abs_max` / `scale`.
    fp8_temp_bytes = 2 * k if num_pairs_total > 1 else k
    temp_bytes = 4 * k + 4 * _Q_PACKING * head_dim + fp8_temp_bytes + 2 * _NUM_LANES * 4
    return io_bytes + temp_bytes


def select_row_tile(
    m: int,
    num_kv_heads: int,
    q_per_kv: int,
    head_dim: int,
    vmem_capacity_bytes: int | None = None,
) -> int | None:
    """Row tile for an ``[m, K]`` attention output, or None if unsupported.

    The largest tile that divides ``m`` and whose double-buffered blocks plus
    in-kernel temporaries fit the chip's VMEM budget. Called inside JAX
    lowering, where ``m`` is a static per-bucket int, so no symbolic
    token-count guards reach TorchDynamo.
    """
    if (
        m <= 0
        or num_kv_heads <= 0
        or q_per_kv <= 0
        or head_dim <= 0
        or q_per_kv % _Q_PACKING != 0
        or head_dim % _NUM_LANES != 0
    ):
        return None
    if vmem_capacity_bytes is None:
        vmem_capacity_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes)
    if vmem_capacity_bytes <= 0:
        return None
    budget = _tile_budget_bytes(vmem_capacity_bytes)
    bytes_per_row = _row_vmem_bytes(num_kv_heads, q_per_kv, head_dim)
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
    num_pairs: int,
    head_dim: int,
    quant_dtype: jnp.dtype,
) -> None:
    """``x_ref``: ``[kv, rows, q_per_kv // 2, 2, d]``; ``q_ref``: ``[rows, K]``."""
    pair_width = _Q_PACKING * head_dim
    pairs = [
        (kv_head, pair) for kv_head in range(num_kv_heads) for pair in range(num_pairs)
    ]

    def load_pair(kv_head: int, pair: int) -> jax.Array:
        # Load each packed head pair as one lane-dense `[rows, 2 * d]` block;
        # its C-order flatten already matches `q_ref`'s `[slot0 | slot1]` order.
        packed = x_ref[kv_head, :, pair, :, :].astype(jnp.float32)
        return packed.reshape(packed.shape[0], pair_width)

    abs_max = None
    for kv_head, pair in pairs:
        pair_max = jnp.max(jnp.abs(load_pair(kv_head, pair)), axis=1, keepdims=True)
        abs_max = pair_max if abs_max is None else jnp.maximum(abs_max, pair_max)
    assert abs_max is not None
    dtype_max, dtype_min = get_max_min(quant_dtype)
    scale = abs_max / dtype_max
    scale = jnp.where(scale == 0.0, 1.0, scale)
    q_blocks = [
        jnp.clip(load_pair(kv_head, pair) / scale, dtype_min, dtype_max).astype(
            quant_dtype
        )
        for kv_head, pair in pairs
    ]
    q_ref[...] = (
        q_blocks[0] if len(q_blocks) == 1 else jnp.concatenate(q_blocks, axis=1)
    )
    s_ref[...] = jnp.broadcast_to(scale, (scale.shape[0], _NUM_LANES))


def quantize_attn_out(
    x: jax.Array,
    num_kv_heads: int,
    q_per_kv: int,
    head_dim: int,
    quant_dtype: jnp.dtype = jnp.float8_e4m3fn,
    *,
    vmem_capacity_bytes: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Quantize the ``[M, K]`` attention output per row to ``quant_dtype``.

    Returns ``(x_q [M, K], scale [M, 128] float32)``; column 0 of ``scale``
    is the per-row scale.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D attention output [M, K], got {x.shape=}")
    m, k = x.shape
    if k != num_kv_heads * q_per_kv * head_dim:
        raise ValueError(f"{k=} must equal {num_kv_heads=} * {q_per_kv=} * {head_dim=}")
    if vmem_capacity_bytes is None:
        vmem_capacity_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes)
    row_tile = select_row_tile(
        m,
        num_kv_heads,
        q_per_kv,
        head_dim,
        vmem_capacity_bytes=vmem_capacity_bytes,
    )
    if row_tile is None:
        raise ValueError(
            f"No attention-output FP8 row tile for {x.shape=}, "
            f"{q_per_kv=}, {head_dim=}, {vmem_capacity_bytes=}"
        )
    num_pairs = q_per_kv // _Q_PACKING
    # Undo the wrapper's `o.reshape(kv, m, q_per_kv, d).swapaxes(1, 0).reshape(m, k)`
    # in reverse order so XLA cancels the swapaxes and reshapes even when both
    # `num_kv_heads > 1` and `num_pairs > 1`, reading the attention output
    # buffer directly without a relayout copy.
    x_head_major = (
        x.reshape(m, num_kv_heads, q_per_kv, head_dim)
        .swapaxes(1, 0)
        .reshape(num_kv_heads, m, num_pairs, _Q_PACKING, head_dim)
    )
    return pl.pallas_call(
        functools.partial(
            _attn_out_fp8_kernel,
            num_kv_heads=num_kv_heads,
            num_pairs=num_pairs,
            head_dim=head_dim,
            quant_dtype=quant_dtype,
        ),
        out_shape=[
            jax.ShapeDtypeStruct((m, k), quant_dtype),
            jax.ShapeDtypeStruct((m, _NUM_LANES), jnp.float32),
        ],
        grid=(m // row_tile,),
        in_specs=[
            pl.BlockSpec(
                (num_kv_heads, row_tile, num_pairs, _Q_PACKING, head_dim),
                lambda i: (0, i, 0, 0, 0),
            )
        ],
        out_specs=[
            pl.BlockSpec((row_tile, k), lambda i: (i, 0)),
            pl.BlockSpec((row_tile, _NUM_LANES), lambda i: (i, 0)),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
            vmem_limit_bytes=vmem_limit_bytes(
                row_tile, num_kv_heads, q_per_kv, head_dim, vmem_capacity_bytes
            ),
        ),
        name="attn_out_fp8_quant",
    )(x_head_major)


def attn_out_fp8_matmul(
    x: jax.Array,
    w_q: jax.Array,
    w_s: jax.Array,
    num_kv_heads: int,
    q_per_kv: int,
    head_dim: int,
) -> jax.Array:
    """``o_proj`` W8A8 matmul with the fused attention-output quantization.

    ``w_q`` is ``[K, N]`` FP8 and ``w_s`` the ``[N]`` per-channel scale.
    """
    x_q, x_scale = quantize_attn_out(x, num_kv_heads, q_per_kv, head_dim, w_q.dtype)
    out = jax.lax.dot_general(
        x_q,
        w_q,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    out *= x_scale[:, :1]
    out *= jnp.expand_dims(w_s, 0)
    return out.astype(x.dtype)
