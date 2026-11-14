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
"""Pallas kernel fusing gated-MLP activation with dynamic per-row FP8 quantization.

Data contract:
- Input ``gate_up`` has shape ``[M, 2 * D]`` in ``bfloat16`` where the first ``D``
  columns hold the gate projection and the last ``D`` columns hold the up
  projection, with ``D >= 128`` and ``D % 128 == 0`` on the fused Pallas path
  (smaller or unaligned ``D`` falls back cleanly to ``xla_quantized_matmul``).
- Dual ``BlockSpec((bm, D), (i, 0))`` and ``BlockSpec((bm, D), (i, 1))`` DMAs stream
  the two halves directly from ``[M, 2 * D]`` into VMEM without materializing a
  strided HBM slice.
- Inside VMEM, the kernel evaluates the gated activation in ``float32``, rounds to
  ``bfloat16`` before multiplying by ``up`` (matching ``vllm``'s ``GeluAndMul`` /
  ``SiluAndMul`` rounding boundary), quantizes each row via ``quantize_block``, and
  writes ``[M, D]`` 1-byte FP8 activations plus ``[M, 128]`` lane-broadcast
  ``float32`` scales for ``dot_general``.

Note: Do not add ``from __future__ import annotations`` in this module;
``torch_tpu``'s ``pallas.jax_op`` inspects live parameter annotations to infer
``static_argnums`` for ``act_kind``, ``w8a16_max_rows``, and ``vmem_capacity_bytes``.
"""

import enum
import functools

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_block,
    xla_quantized_matmul,
)


class GatedActivationKind(enum.IntEnum):
    """Supported gated-MLP elementwise activations on Pallas TPU."""

    GELU_TANH = 0
    SILU = 1


FP8_MIN_TOKENS = 128
_LANE_ALIGN = 128
_SUBLANE_ALIGN = 8
_FP8_MULTI_TILE_M_ALIGN = 32  # 8 sublanes * 4-way 1-byte FP8 sublane packing
_SCALE_LANES = 128
# Cap scoped Pallas VMEM at 7/8 of chip capacity (56 MiB on 64 MiB TPU_7X).
_VMEM_LIMIT_NUMERATOR = 7
_VMEM_LIMIT_DENOMINATOR = 8
# Multi-tile footprint per (bm, d) element: 10 bytes for double-buffered I/O
# (2 * (2B gate + 2B up + 1B fp8 out)) + 12 bytes for f32 temporaries
# (gate_f32, act_f32) and VREG spill headroom = 22 bytes/elem.
# Single-tile (grid == 1) uses 5 bytes I/O + 12 bytes f32 temporaries = 17 bytes/elem.
_MULTI_TILE_BYTES_PER_ELEM = 22
_SINGLE_TILE_BYTES_PER_ELEM = 17
_VALID_ACT_KINDS = frozenset(
    {int(GatedActivationKind.GELU_TANH), int(GatedActivationKind.SILU)}
)


def vmem_limit_bytes(vmem_capacity_bytes: int) -> int:
    """Scoped VMEM byte budget passed to ``pltpu.CompilerParams``."""
    return (vmem_capacity_bytes * _VMEM_LIMIT_NUMERATOR) // _VMEM_LIMIT_DENOMINATOR


def _validate_fp8_dtype(dtype: jnp.dtype, caller: str) -> jnp.dtype:
    resolved = jnp.dtype(dtype)
    if not (jnp.issubdtype(resolved, jnp.floating) and resolved.itemsize == 1):
        raise ValueError(
            f"{caller} requires a 1-byte floating FP8 dtype, got {resolved}"
        )
    return resolved


def _validate_gate_up_bf16(gate_up: jax.Array, caller: str) -> tuple[int, int]:
    if gate_up.ndim != 2 or gate_up.dtype != jnp.bfloat16:
        raise ValueError(
            f"{caller} requires 2D bfloat16 gate_up [M, 2*D], "
            f"got shape={gate_up.shape}, dtype={gate_up.dtype}"
        )
    m, two_d = gate_up.shape
    if two_d <= 0 or two_d % 2 != 0:
        raise ValueError(
            f"{caller} requires positive even last dim on gate_up, got {gate_up.shape}"
        )
    return m, two_d // 2


def _validate_act_kind(act_kind: int) -> int:
    if isinstance(act_kind, bool):
        raise ValueError(f"Unsupported gated activation kind: {act_kind}")
    resolved = int(act_kind)
    if resolved not in _VALID_ACT_KINDS:
        raise ValueError(f"Unsupported gated activation kind: {act_kind}")
    return resolved


def _apply_gate_act(gate_f32: jax.Array, act_kind: int) -> jax.Array:
    if act_kind == GatedActivationKind.GELU_TANH:
        return jax.nn.gelu(gate_f32, approximate=True)
    if act_kind == GatedActivationKind.SILU:
        return jax.nn.silu(gate_f32)
    raise ValueError(f"Unsupported gated activation kind: {act_kind}")


def _gated_act_fp8_kernel(
    gate_ref,
    up_ref,
    q_ref,
    s_ref,
    *,
    act_kind: int,
    quant_dtype: jnp.dtype,
):
    gate_f32 = gate_ref[...].astype(jnp.float32)
    # Match upstream GeluAndMul / SiluAndMul bf16 rounding before dynamic FP8 quant.
    act_bf16 = _apply_gate_act(gate_f32, act_kind).astype(gate_ref.dtype) * up_ref[...]
    act_f32 = act_bf16.astype(jnp.float32)
    q, scale = quantize_block(act_f32, axis=1, target_dtype=quant_dtype)
    q_ref[...] = q
    # Write [bm, 128] lane-broadcast f32 to avoid lane-sparse VMEM stores;
    # fused_gated_act_fp8_quant slices back to [:, :1] outside the kernel.
    s_ref[...] = jnp.broadcast_to(scale, (gate_ref.shape[0], _SCALE_LANES))


def select_block_m(m: int, d: int, vmem_capacity_bytes: int) -> int | None:
    """Pick the largest pipelined ``block_m`` fitting the scoped VMEM budget.

    Requires ``d >= 128`` and ``d % 128 == 0`` for full-lane ``(bm, d)`` TPU
    DMAs. Multi-tile grids (``m > bm``) require ``bm % 32 == 0`` for 1-byte FP8
    sublane packing (``8 * 4 = 32``)—satisfied by construction for all candidates
    ``(256, 128, 64, 32)``—and enforce ``m // bm >= 2`` (or ``>= 4`` when
    ``m <= 256`` and ``d >= 8192``) so DMA and vector compute overlap across
    pipeline stages. Standard power-of-two token buckets hit the multi-tile path;
    custom unaligned buckets (``m <= 256``, ``m % 8 == 0``) fall back to
    single-tile ``bm = m`` when no multi-tile candidate divides ``m``.
    """
    if m <= 0 or d < _LANE_ALIGN or d % _LANE_ALIGN != 0 or vmem_capacity_bytes <= 0:
        return None
    limit = vmem_limit_bytes(vmem_capacity_bytes)
    min_grid = 4 if (m <= 256 and d >= 8192) else 2
    for candidate in (256, 128, 64, 32):
        if (
            m >= candidate * min_grid
            and m % candidate == 0
            and _MULTI_TILE_BYTES_PER_ELEM * candidate * d <= limit
        ):
            return candidate
    if (
        m <= 256
        and m % _SUBLANE_ALIGN == 0
        and _SINGLE_TILE_BYTES_PER_ELEM * m * d <= limit
    ):
        return m
    return None


def fused_gated_act_fp8_quant(
    gate_up: jax.Array,
    quant_dtype: jnp.dtype,
    act_kind: int = int(GatedActivationKind.GELU_TANH),
    *,
    bm: int,
    vmem_capacity_bytes: int,
) -> tuple[jax.Array, jax.Array]:
    """Run fused gated activation + dynamic per-row FP8 quant -> ``(x_q, x_s)``.

    Also serves as a direct entry point for intermediate-quantization tests, so
    dtype, shape, lane/sublane alignment, and scoped VMEM bounds are validated
    here independently of ``select_block_m``.
    """
    quant_dtype = _validate_fp8_dtype(quant_dtype, "fused_gated_act_fp8_quant")
    act_kind = _validate_act_kind(act_kind)
    m, d = _validate_gate_up_bf16(gate_up, "fused_gated_act_fp8_quant")
    if d < _LANE_ALIGN or d % _LANE_ALIGN != 0:
        raise ValueError(
            f"fused_gated_act_fp8_quant requires d >= {_LANE_ALIGN} and "
            f"d % {_LANE_ALIGN} == 0, got d={d}"
        )
    if vmem_capacity_bytes <= 0:
        raise ValueError(
            f"vmem_capacity_bytes must be positive, got {vmem_capacity_bytes}"
        )
    limit = vmem_limit_bytes(vmem_capacity_bytes)
    bytes_per_elem = (
        _MULTI_TILE_BYTES_PER_ELEM if m > bm else _SINGLE_TILE_BYTES_PER_ELEM
    )
    valid_bm = (
        bm > 0
        and m > 0
        and m % bm == 0
        and bm % _SUBLANE_ALIGN == 0
        and (m == bm or bm % _FP8_MULTI_TILE_M_ALIGN == 0)
        and bytes_per_elem * bm * d <= limit
    )
    if not valid_bm:
        raise ValueError(
            f"Invalid tile config (m={m}, d={d}, bm={bm}, vmem={vmem_capacity_bytes})"
        )

    x_q, s_wide = pl.pallas_call(
        functools.partial(
            _gated_act_fp8_kernel,
            act_kind=act_kind,
            quant_dtype=quant_dtype,
        ),
        out_shape=[
            jax.ShapeDtypeStruct((m, d), quant_dtype),
            jax.ShapeDtypeStruct((m, _SCALE_LANES), jnp.float32),
        ],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(m // bm,),
            in_specs=[
                pl.BlockSpec((bm, d), lambda i: (i, 0)),
                pl.BlockSpec((bm, d), lambda i: (i, 1)),
            ],
            out_specs=[
                pl.BlockSpec((bm, d), lambda i: (i, 0)),
                pl.BlockSpec((bm, _SCALE_LANES), lambda i: (i, 0)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
            vmem_limit_bytes=limit,
        ),
        name="fused_gated_act_fp8",
    )(gate_up, gate_up)
    return x_q, s_wide[:, :1]


def fused_gated_act_fp8_matmul(
    gate_up: jax.Array,
    w_q: jax.Array,
    w_s: jax.Array,
    act_kind: int = int(GatedActivationKind.GELU_TANH),
    w8a16_max_rows: int = 0,
    vmem_capacity_bytes: int = 0,
) -> jax.Array:
    """Fuse gated activation + dynamic per-row FP8 quant ahead of ``dot_general``.

    Token-count and ``D``-alignment branching (``m < FP8_MIN_TOKENS``,
    ``m <= w8a16_max_rows``, unaligned ``m``, or ``d < 128`` / ``d % 128 != 0``)
    happens here inside JAX lowering—where ``m`` and ``d`` are static per
    compiled bucket—so the outer PyTorch Dynamo FX graph stays independent of
    ``num_tokens`` (matching ``linear_common.quantized_matmul``). Requires 2D
    ``bfloat16`` ``gate_up [M, 2*D]``, 2D 1-byte FP8 ``w_q [D, N]``, and 1D
    per-channel ``[N]`` or per-tensor ``[1]`` ``w_s``.
    """
    act_kind = _validate_act_kind(act_kind)
    m, d = _validate_gate_up_bf16(gate_up, "fused_gated_act_fp8_matmul")
    if w_q.ndim != 2 or w_q.shape[0] != d or w_q.shape[1] <= 0:
        raise ValueError(f"Expected 2D w_q with shape ({d}, N>0), got {w_q.shape}")
    _validate_fp8_dtype(w_q.dtype, "fused_gated_act_fp8_matmul")
    n = w_q.shape[1]
    if (
        w_s.ndim != 1
        or w_s.shape[0] not in (1, n)
        or not jnp.issubdtype(w_s.dtype, jnp.floating)
    ):
        raise ValueError(
            f"Expected 1D floating per-channel ({n},) or per-tensor (1,) w_s, "
            f"got w_s.shape={w_s.shape}, dtype={w_s.dtype} for w_q.shape={w_q.shape}"
        )

    if vmem_capacity_bytes <= 0:
        vmem_capacity_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes)

    use_w8a16 = 0 < m <= w8a16_max_rows
    bm = (
        select_block_m(m, d, vmem_capacity_bytes)
        if (m >= FP8_MIN_TOKENS and not use_w8a16)
        else None
    )
    if bm is None:
        gate, up = gate_up[:, :d], gate_up[:, d:]
        act = (
            _apply_gate_act(gate.astype(jnp.float32), act_kind).astype(gate_up.dtype)
            * up
        )
        return xla_quantized_matmul(act, w_q, w_s, quantize_activation=not use_w8a16)

    x_q, x_s = fused_gated_act_fp8_quant(
        gate_up,
        w_q.dtype,
        act_kind,
        bm=bm,
        vmem_capacity_bytes=vmem_capacity_bytes,
    )
    out = jax.lax.dot_general(
        x_q,
        w_q,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    out = out * x_s * jnp.expand_dims(w_s, 0)
    return out.astype(gate_up.dtype)
