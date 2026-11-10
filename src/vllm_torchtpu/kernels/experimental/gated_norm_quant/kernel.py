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

Fused Pallas kernel for gated RMSNorm (RMSNormGated) and per-token dynamic
FP8 quantization.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

FP8 = jnp.float8_e4m3fn
FP8_MAX = float(jnp.finfo(FP8).max)
VREG_F32_LANES = 8 * 128  # 1024 fp32 lanes per TPU VREG (4 KiB)
TARGET_VREG_BUDGET = 64  # Hardware VREG count per TPU core (256 KiB)


def _kernel(o_ref, z_ref, w_ref, q_ref, s_ref, *, step_m: int, eps: float):
    bm = o_ref.shape[0]
    step = min(bm, step_m)
    w = w_ref[...].astype(jnp.float32)[None, None, :]
    for r in range(bm // step):
        sl = slice(r * step, (r + 1) * step)
        o = o_ref[sl, :, :].astype(jnp.float32)  # [step, heads, dim]
        z = z_ref[sl, :, :].astype(jnp.float32)
        var = jnp.mean(o * o, axis=-1, keepdims=True)
        y = o * jax.lax.rsqrt(var + eps) * w
        y = y * (z * jax.nn.sigmoid(z))
        y = y.astype(jnp.bfloat16).astype(jnp.float32)
        amax = jnp.max(jnp.abs(y), axis=(1, 2), keepdims=True)
        scale = amax * jnp.float32(1.0 / FP8_MAX)
        scale = jnp.where(scale == 0.0, 1.0, scale)
        scale_inv = jnp.float32(1.0) / scale
        q_ref[sl, :, :] = jnp.clip(y * scale_inv, -FP8_MAX, FP8_MAX).astype(FP8)
        s_ref[sl, :] = scale.reshape(step, 1)


def _kernel_2dz(
    o_ref, z_ref, w_ref, q_ref, s_ref, scratch_ref, *, step_m: int, eps: float
):
    bm, heads, dim = o_ref.shape
    step = min(bm, step_m)
    w = w_ref[...].astype(jnp.float32)[None, None, :]
    flat_s = scratch_ref.reshape(step * heads, dim)
    for r in range(bm // step):
        sl = slice(r * step, (r + 1) * step)
        o = o_ref[sl, :, :].astype(jnp.float32)
        var = jnp.mean(o * o, axis=-1, keepdims=True)
        scratch_ref[...] = o * jax.lax.rsqrt(var + eps) * w
        o_norm_2d = jnp.concatenate(
            [flat_s[h : step * heads : heads] for h in range(heads)],
            axis=1,
        )
        z = z_ref[sl, :].astype(jnp.float32)
        y = (
            (o_norm_2d * (z * jax.nn.sigmoid(z)))
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        amax = jnp.max(jnp.abs(y), axis=1, keepdims=True)
        scale = amax * jnp.float32(1.0 / FP8_MAX)
        scale = jnp.where(scale == 0.0, 1.0, scale)
        scale_inv = jnp.float32(1.0) / scale
        q_ref[sl, :] = jnp.clip(y * scale_inv, -FP8_MAX, FP8_MAX).astype(FP8)
        s_ref[sl, :] = scale


def legal_tile(rows: int, cap: int) -> int:
    """The largest multiple of 8 up to ``cap`` dividing ``rows``, else ``rows``.

    A block equal to the full dimension is always legal, which covers row
    counts such as 36 or 44 that have no multiple-of-8 divisor.
    """
    for t in range(min(cap, rows), 7, -1):
        if rows % t == 0 and t % 8 == 0:
            return t
    return rows


def default_vreg_step(cols: int) -> int:
    """Select an inner VREG sub-tile row step keeping live fp32 arrays in 64 VREGs."""
    vregs_per_row = max(1, (cols + VREG_F32_LANES - 1) // VREG_F32_LANES)
    return max(16, (TARGET_VREG_BUDGET // vregs_per_row // 8) * 8)


def default_tile_cap(tokens: int, cols: int) -> int:
    """Select an optimal multiple-of-8 VMEM BlockSpec tile cap for ``[tokens, cols]``.

    1. For decode/verify sizes (``tokens <= 256``, e.g., ``N=96, 128``), caps
       ``tile`` so ``grid >= 6`` (``tile=16`` -> ``grid=(6,)`` at ``N=96`` and
       ``grid=(8,)`` at ``N=128``) so VMEM DMAs double-buffer with compute.
    2. For large prefill sizes (``tokens > 256``), uses a larger VMEM block (up
       to 512 rows) while the kernel iterates in VREG-sized sub-tiles internally,
       preventing both VREG spills and large-grid overhead at ``T=8192``.
    """
    vreg_cap = default_vreg_step(cols)
    if tokens <= 256:
        dma_cap = max(8, (tokens // 6 // 8) * 8)
        return min(vreg_cap, dma_cap)
    return max(vreg_cap, 512)


def gated_norm_quant(
    o: jax.Array,
    z: jax.Array,
    w: jax.Array,
    eps: float,
    *,
    tile: int | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """o: [tokens, heads, dim] bf16; z: 3D or 2D [tokens, heads * dim] bf16; w: [dim].

    Returns the fp8 rows (matching ``z.shape``) and their f32 scales [tokens, 1].
    """
    tokens, heads, dim = o.shape
    cols = heads * dim
    assert w.shape == (dim,), (o.shape, z.shape, w.shape)
    cap = default_tile_cap(tokens, cols) if tile is None else tile
    bm = legal_tile(tokens, cap)
    step_m = legal_tile(bm, default_vreg_step(cols))

    if z.ndim == 2:
        assert z.shape == (tokens, cols), (o.shape, z.shape, w.shape)
        if (
            (bm % 32 == 0 or (bm == tokens and bm % 8 == 0))
            and (step_m % 32 == 0 or step_m == bm)
            and dim % 128 == 0
        ):
            return pl.pallas_call(
                functools.partial(_kernel_2dz, step_m=step_m, eps=float(eps)),
                grid=(tokens // bm,),
                in_specs=[
                    pl.BlockSpec((bm, heads, dim), lambda i: (i, 0, 0)),
                    pl.BlockSpec((bm, cols), lambda i: (i, 0)),
                    pl.BlockSpec((dim,), lambda i: (0,)),
                ],
                out_specs=[
                    pl.BlockSpec((bm, cols), lambda i: (i, 0)),
                    pl.BlockSpec((bm, 1), lambda i: (i, 0)),
                ],
                out_shape=[
                    jax.ShapeDtypeStruct((tokens, cols), FP8),
                    jax.ShapeDtypeStruct((tokens, 1), jnp.float32),
                ],
                scratch_shapes=[pltpu.VMEM((step_m, heads, dim), jnp.float32)],
                compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
                interpret=interpret,
            )(o, z, w)
        q_3d, s = gated_norm_quant(
            o, z.reshape(tokens, heads, dim), w, eps, tile=tile, interpret=interpret
        )
        return q_3d.reshape(tokens, cols), s

    assert z.shape == o.shape, (o.shape, z.shape, w.shape)
    spec = pl.BlockSpec((bm, heads, dim), lambda i: (i, 0, 0))
    return pl.pallas_call(
        functools.partial(_kernel, step_m=step_m, eps=float(eps)),
        grid=(tokens // bm,),
        in_specs=[spec, spec, pl.BlockSpec((dim,), lambda i: (0,))],
        out_specs=[spec, pl.BlockSpec((bm, 1), lambda i: (i, 0))],
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
    zz = z.reshape(o.shape).astype(jnp.float32)
    var = jnp.mean(x * x, axis=-1, keepdims=True)
    y = x * jax.lax.rsqrt(var + eps) * w.astype(jnp.float32)[None, None, :]
    y = (y * (zz * jax.nn.sigmoid(zz))).astype(jnp.bfloat16)
    flat = y.reshape(y.shape[0], -1)
    amax = jnp.max(jnp.abs(flat), axis=-1, keepdims=True).astype(jnp.float32)
    scale = amax * jnp.float32(1.0 / FP8_MAX)
    scale = jnp.where(scale == 0.0, 1.0, scale)
    scale_inv = jnp.float32(1.0) / scale
    q = jnp.clip(flat.astype(jnp.float32) * scale_inv, -FP8_MAX, FP8_MAX).astype(FP8)
    return q.reshape(z.shape), scale
