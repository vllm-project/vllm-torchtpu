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
"""Gated RMS norm and per-token fp8 quantization in one pass.

Per head ``h`` of a token: ``y = x_h * rsqrt(mean(x_h^2) + eps) * w * silu(z_h)``,
rounded to bf16 as ``RMSNormGated`` returns it; then ``scale = max|y| / 448``
over the token (1 for an all-zero row) and ``q = clip(y / scale)`` in
float8_e4m3fn -- the rows and scales the fp8 linear would compute for itself.

Fused Pallas kernel for gated RMSNorm (RMSNormGated) and per-token dynamic FP8 quantization.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

FP8 = jnp.float8_e4m3fn
FP8_MAX = float(jnp.finfo(FP8).max)
VREG_F32_LANES = 8 * 128  # 1024 fp32 lanes per TPU VREG (4 KiB)
TARGET_VREG_BUDGET = 64   # Hardware VREG count per TPU core (256 KiB)


def _kernel(o_ref, z_ref, w_ref, q_ref, s_ref, *, eps: float):
    o = o_ref[...].astype(jnp.float32)  # [tile, heads, dim]
    z = z_ref[...].astype(jnp.float32)
    w = w_ref[...].astype(jnp.float32)
    var = jnp.mean(o * o, axis=-1, keepdims=True)
    y = o * jax.lax.rsqrt(var + eps) * w[None, None, :]
    y = y * (z * jax.nn.sigmoid(z))
    y = y.astype(jnp.bfloat16).astype(jnp.float32)
    amax = jnp.max(jnp.abs(y), axis=(1, 2), keepdims=True)
    scale = amax * jnp.float32(1.0 / FP8_MAX)
    scale = jnp.where(scale == 0.0, 1.0, scale)
    scale_inv = jnp.float32(1.0) / scale
    q_ref[...] = jnp.clip(y * scale_inv, -FP8_MAX, FP8_MAX).astype(FP8)
    s_ref[...] = scale.reshape(scale.shape[0], 1)


def legal_tile(rows: int, cap: int) -> int:
    """The largest multiple of 8 up to ``cap`` dividing ``rows``, else ``rows``.

    A block equal to the full dimension is always legal, which covers row
    counts such as 36 or 44 that have no multiple-of-8 divisor.
    """
    for t in range(min(cap, rows), 7, -1):
        if rows % t == 0 and t % 8 == 0:
            return t
    return rows


def default_tile_cap(tokens: int, cols: int) -> int:
    """Select an optimal multiple-of-8 tile cap for ``[tokens, cols]`` quant.

    1. Keeps live fp32 vectors within the 64-VREG hardware file (e.g. ``tile=64``
       instead of ``512`` for ``cols=1024`` at ``N=2048``, preventing 6 MiB spills).
    2. For decode/verify sizes (``tokens <= 256``, e.g., ``N=96, 128``), caps
       ``tile`` so ``grid >= 6`` (``tile=16`` -> ``grid=(6,)`` at ``N=96`` and
       ``grid=(8,)`` at ``N=128``) so VMEM DMAs double-buffer with compute.
    """
    vregs_per_row = max(1, (cols + VREG_F32_LANES - 1) // VREG_F32_LANES)
    vreg_cap = max(16, (TARGET_VREG_BUDGET // vregs_per_row // 8) * 8)
    if tokens <= 256:
        dma_cap = max(8, (tokens // 6 // 8) * 8)
        return min(vreg_cap, dma_cap)
    return vreg_cap


def gated_norm_quant(
    o: jax.Array,
    z: jax.Array,
    w: jax.Array,
    eps: float,
    *,
    tile: int | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """o, z: [tokens, heads, dim] bf16; w: [dim].

    Returns the fp8 rows [tokens, heads, dim] and their f32 scales [tokens, 1].
    """
    tokens, heads, dim = o.shape
    assert z.shape == o.shape and w.shape == (dim,), (o.shape, z.shape, w.shape)
    cap = default_tile_cap(tokens, heads * dim) if tile is None else tile
    tile = legal_tile(tokens, cap)
    spec = pl.BlockSpec((tile, heads, dim), lambda i: (i, 0, 0))
    return pl.pallas_call(
        functools.partial(_kernel, eps=float(eps)),
        grid=(tokens // tile,),
        in_specs=[spec, spec, pl.BlockSpec((dim,), lambda i: (0,))],
        out_specs=[spec, pl.BlockSpec((tile, 1), lambda i: (i, 0))],
        out_shape=[
            jax.ShapeDtypeStruct((tokens, heads, dim), FP8),
            jax.ShapeDtypeStruct((tokens, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(o, z, w)


def reference(
    o: jax.Array, z: jax.Array, w: jax.Array, eps: float
) -> tuple[jax.Array, jax.Array]:
    """RMSNormGated followed by the fp8 linear's per-token quantization."""
    x = o.astype(jnp.float32)
    zz = z.astype(jnp.float32)
    var = jnp.mean(x * x, axis=-1, keepdims=True)
    y = x * jax.lax.rsqrt(var + eps) * w.astype(jnp.float32)[None, None, :]
    y = (y * (zz * jax.nn.sigmoid(zz))).astype(jnp.bfloat16)
    flat = y.reshape(y.shape[0], -1)
    amax = jnp.max(jnp.abs(flat), axis=-1, keepdims=True).astype(jnp.float32)
    scale = amax * jnp.float32(1.0 / FP8_MAX)
    scale = jnp.where(scale == 0.0, 1.0, scale)
    scale_inv = jnp.float32(1.0) / scale
    q = jnp.clip(flat.astype(jnp.float32) * scale_inv, -FP8_MAX, FP8_MAX).astype(FP8)
    return q.reshape(o.shape), scale
