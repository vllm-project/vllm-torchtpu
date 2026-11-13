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
"""Pallas kernel fusing pre-MLP RMSNorm with dynamic per-row FP8 quantization.

Data contract:
- Input ``x`` has shape ``[M, K]`` in ``bfloat16`` with ``K >= 128`` and
  ``K % 128 == 0`` on the fused Pallas path (smaller/unaligned ``K`` or buckets
  rejected by ``should_use_pallas_rmsnorm_fp8`` fall back to
  ``_xla_rmsnorm_fp8_matmul``); ``gamma`` has shape ``[K]`` in ``bfloat16`` or
  ``float32``.
- Inside VMEM, the kernel evaluates RMSNorm in ``float32``, rounds to
  ``bfloat16`` after multiplying by ``gamma + gamma_offset`` (reproducing
  ``ir_ops.rms_norm`` CUDA rounding boundaries), quantizes each row via
  ``quantize_block``, and writes ``[M, K]`` 1-byte FP8 activations plus
  ``[M, 128]`` lane-broadcast ``float32`` scales for ``jax.lax.dot_general``.

Note: Do not add ``from __future__ import annotations`` in this module;
``torch_tpu``'s ``pallas.jax_op`` inspects live parameter annotations to infer
``static_argnums`` for ``eps``, ``gamma_offset``, ``w8a16_max_rows``, and
``vmem_capacity_bytes``.
"""

import functools

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_block,
    xla_quantized_matmul,
)

FP8_RMSNORM_MIN_TOKENS = 128
_LANE_ALIGN = 128
_SUBLANE_ALIGN = 8
_FP8_MULTI_TILE_M_ALIGN = 32  # 8 sublanes * 4-way 1-byte FP8 sublane packing
_SCALE_LANES = 128
# Cap scoped Pallas VMEM at 7/8 of chip capacity (56 MiB on 64 MiB TPU_7X).
_VMEM_LIMIT_NUMERATOR = 7
_VMEM_LIMIT_DENOMINATOR = 8
# Multi-tile footprint per (bm, k) element: 6 bytes for double-buffered I/O
# (2 * (2B bf16 in + 1B fp8 out)) + 8 bytes for two float32 (bm, k) temporaries
# + 2 bytes spill headroom = 16 bytes/elem.
# Single-tile (grid == 1) uses 3 bytes I/O + 8 bytes float32 temporaries +
# 2 bytes spill headroom = 13 bytes/elem.
_MULTI_TILE_BYTES_PER_ELEM = 16
_SINGLE_TILE_BYTES_PER_ELEM = 13
# On narrow per-rank output shards (high-TP gate_up_proj with N < 6144) at
# large prefill buckets (M > 8192), XLA's native M-loop overlaps activation
# quantization into dot_general faster than a standalone Pallas custom call.
_NARROW_N_PREFILL_MAX_N = 6144
_NARROW_N_PREFILL_MAX_M = 8192


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


def _validate_x_and_gamma(
    x: jax.Array, gamma: jax.Array, caller: str
) -> tuple[int, int]:
    if x.ndim != 2 or x.dtype != jnp.bfloat16:
        raise ValueError(
            f"{caller} requires 2D bfloat16 input x [M, K], "
            f"got shape={x.shape}, dtype={x.dtype}"
        )
    m, k = x.shape
    if m <= 0 or k <= 0:
        raise ValueError(
            f"{caller} requires positive dimensions on x, got shape={x.shape}"
        )
    if gamma.shape != (k,) or gamma.dtype not in (jnp.bfloat16, jnp.float32):
        raise ValueError(
            f"{caller} requires 1D bfloat16/float32 gamma of shape ({k},), "
            f"got shape={gamma.shape}, dtype={gamma.dtype}"
        )
    return m, k


def _rmsnorm_fp8_kernel(
    x_ref,
    gamma_ref,
    q_ref,
    s_ref,
    *,
    eps: float,
    gamma_offset: float,
    quant_dtype: jnp.dtype,
) -> None:
    orig_dtype = x_ref.dtype
    x_f32 = x_ref[...].astype(jnp.float32)
    var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    rsqrt_var = jax.lax.rsqrt(var + eps)
    gamma_f32 = gamma_ref[...].astype(jnp.float32)
    if gamma_offset != 0.0:
        gamma_f32 = gamma_f32 + gamma_offset
    # Match ir_ops.rms_norm: multiply gamma in FP32 and round to orig_dtype once.
    x_norm_bf16 = (x_f32 * rsqrt_var * gamma_f32).astype(orig_dtype)
    # Upcast to float32 for Pallas cross-lane max reduction (exact for bf16).
    q, scale = quantize_block(
        x_norm_bf16.astype(jnp.float32), axis=1, target_dtype=quant_dtype
    )
    q_ref[...] = q
    # Write [bm, 128] lane-broadcast f32 to avoid lane-sparse VMEM stores;
    # fused_rmsnorm_fp8_quant slices back to [:, :1] outside the kernel.
    s_ref[...] = jnp.broadcast_to(scale, (x_norm_bf16.shape[0], _SCALE_LANES))


def _tile_vmem_bytes(m: int, bm: int, k: int) -> int:
    bytes_per_elem = (
        _MULTI_TILE_BYTES_PER_ELEM if m > bm else _SINGLE_TILE_BYTES_PER_ELEM
    )
    scale_buffers = 2 if m > bm else 1
    return bm * (bytes_per_elem * k + scale_buffers * _SCALE_LANES * 4) + 4 * k


def select_block_m(
    m: int,
    k: int,
    vmem_capacity_bytes: int,
) -> int | None:
    """Pick the largest valid ``block_m`` fitting the scoped VMEM budget.

    Requires ``k >= 128`` and ``k % 128 == 0`` for full-lane ``(bm, k)`` TPU
    DMAs. Multi-tile grids (``m > bm``) require ``bm % 32 == 0`` for 1-byte FP8
    sublane packing (``8 * 4 = 32``); single-tile grids (``bm == m <= 512``)
    require ``m % 8 == 0``.
    """
    if m <= 0 or k < _LANE_ALIGN or k % _LANE_ALIGN != 0 or vmem_capacity_bytes <= 0:
        return None
    limit = vmem_limit_bytes(vmem_capacity_bytes)
    for candidate in (512, 256, 128, 64, 32):
        if (
            m >= candidate
            and m % candidate == 0
            and _tile_vmem_bytes(m, candidate, k) <= limit
        ):
            return candidate
    if m <= 512 and m % _SUBLANE_ALIGN == 0 and _tile_vmem_bytes(m, m, k) <= limit:
        return m
    return None


def should_use_pallas_rmsnorm_fp8(
    m: int,
    k: int,
    n: int,
    vmem_capacity_bytes: int,
    *,
    w8a16_max_rows: int = 0,
) -> bool:
    """True when padded ``compile_sizes`` bucket ``(m, k, n)`` lowers via Pallas.

    Keeping token-count gating inside JAX lowering (where ``m`` compares against
    the concrete padded ``compile_sizes`` bucket) avoids installing ``SymInt``
    guards in TorchDynamo that would split buckets across multiple FX traces.
    """
    if m < FP8_RMSNORM_MIN_TOKENS or (0 < m <= w8a16_max_rows) or n <= 0:
        return False
    if n < _NARROW_N_PREFILL_MAX_N and m > _NARROW_N_PREFILL_MAX_M:
        return False
    return select_block_m(m, k, vmem_capacity_bytes) is not None


def fused_rmsnorm_fp8_quant(
    x: jax.Array,
    gamma: jax.Array,
    quant_dtype: jnp.dtype,
    eps: float = 1e-6,
    gamma_offset: float = 0.0,
    *,
    bm: int,
    vmem_capacity_bytes: int,
) -> tuple[jax.Array, jax.Array]:
    """Run fused RMSNorm + dynamic per-row FP8 quantization -> ``(x_q, x_s)``."""
    quant_dtype = _validate_fp8_dtype(quant_dtype, "fused_rmsnorm_fp8_quant")
    m, k = _validate_x_and_gamma(x, gamma, "fused_rmsnorm_fp8_quant")
    if k < _LANE_ALIGN or k % _LANE_ALIGN != 0:
        raise ValueError(
            f"fused_rmsnorm_fp8_quant requires k >= {_LANE_ALIGN} and "
            f"k % {_LANE_ALIGN} == 0, got k={k}"
        )
    if vmem_capacity_bytes <= 0:
        raise ValueError(
            f"vmem_capacity_bytes must be positive, got {vmem_capacity_bytes}"
        )
    limit = vmem_limit_bytes(vmem_capacity_bytes)
    valid_bm = (
        bm > 0
        and m % bm == 0
        and bm % _SUBLANE_ALIGN == 0
        and (m == bm or bm % _FP8_MULTI_TILE_M_ALIGN == 0)
        and _tile_vmem_bytes(m, bm, k) <= limit
    )
    if not valid_bm:
        raise ValueError(
            f"Invalid tile config (m={m}, k={k}, bm={bm}, vmem={vmem_capacity_bytes})"
        )

    x_q, s_wide = pl.pallas_call(
        functools.partial(
            _rmsnorm_fp8_kernel,
            eps=eps,
            gamma_offset=gamma_offset,
            quant_dtype=quant_dtype,
        ),
        out_shape=[
            jax.ShapeDtypeStruct((m, k), quant_dtype),
            jax.ShapeDtypeStruct((m, _SCALE_LANES), jnp.float32),
        ],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(m // bm,),
            in_specs=[
                pl.BlockSpec((bm, k), lambda i: (i, 0)),
                pl.BlockSpec((k,), lambda i: (0,)),
            ],
            out_specs=[
                pl.BlockSpec((bm, k), lambda i: (i, 0)),
                pl.BlockSpec((bm, _SCALE_LANES), lambda i: (i, 0)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
            vmem_limit_bytes=limit,
        ),
        name="fused_rmsnorm_fp8",
    )(x, gamma)
    return x_q, s_wide[:, :1]


def _xla_rmsnorm_fp8_matmul(
    x: jax.Array,
    gamma: jax.Array,
    w_q: jax.Array,
    w_s: jax.Array,
    eps: float,
    gamma_offset: float,
    *,
    quantize_activation: bool = True,
) -> jax.Array:
    """Native XLA fallback matching ``ir_ops.rms_norm`` + ``xla_quantized_matmul``."""
    xf = x.astype(jnp.float32)
    inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + eps)
    gf = gamma.astype(jnp.float32)
    if gamma_offset != 0.0:
        gf = gf + gamma_offset
    normed = (xf * inv_rms * gf).astype(x.dtype)
    return xla_quantized_matmul(
        normed, w_q, w_s, quantize_activation=quantize_activation
    )


def fused_rmsnorm_fp8_matmul(
    x: jax.Array,
    gamma: jax.Array,
    w_q: jax.Array,
    w_s: jax.Array,
    eps: float = 1e-6,
    gamma_offset: float = 0.0,
    w8a16_max_rows: int = 0,
    vmem_capacity_bytes: int = 0,
) -> jax.Array:
    """Fuse RMSNorm + dynamic per-row FP8 quantization ahead of ``dot_general``."""
    m, k = _validate_x_and_gamma(x, gamma, "fused_rmsnorm_fp8_matmul")
    if w_q.ndim != 2 or w_q.shape[0] != k or w_q.shape[1] <= 0:
        raise ValueError(f"Expected 2D w_q with shape ({k}, N>0), got {w_q.shape}")
    _validate_fp8_dtype(w_q.dtype, "fused_rmsnorm_fp8_matmul")
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
    if not should_use_pallas_rmsnorm_fp8(
        m, k, n, vmem_capacity_bytes, w8a16_max_rows=w8a16_max_rows
    ):
        return _xla_rmsnorm_fp8_matmul(
            x, gamma, w_q, w_s, eps, gamma_offset, quantize_activation=not use_w8a16
        )

    bm = select_block_m(m, k, vmem_capacity_bytes)
    assert bm is not None
    x_q, x_s = fused_rmsnorm_fp8_quant(
        x,
        gamma,
        w_q.dtype,
        eps=eps,
        gamma_offset=gamma_offset,
        bm=bm,
        vmem_capacity_bytes=vmem_capacity_bytes,
    )
    out = jax.lax.dot_general(
        x_q,
        w_q,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(jnp.float32)
    out = out * x_s * jnp.expand_dims(w_s, 0)
    return out.astype(x.dtype)
