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
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

FP8 = jnp.float8_e4m3fn
FP8_MAX = float(jnp.finfo(FP8).max)
VREG_F32_LANES = 8 * 128  # 1024 fp32 lanes per TPU VREG (4 KiB)
TARGET_VREG_BUDGET = 64  # Hardware VREG count per TPU core (256 KiB)
# VMEM the double-buffered blocks of one grid step may take. The kernel runs
# under the 32 MiB scoped default (it sets no vmem_limit_bytes); the budget
# holds the unsharded GDN width (64 heads x 128 = 8192 columns, TP1 under
# pipeline parallelism) to 200 rows (128 once legal_tile divides a step)
# and leaves widths of 1638 columns or
# less at the 512-row cap.
TILE_VMEM_BUDGET_BYTES = 16 << 20
# Bytes per element of a block across its buffers: two bf16 inputs and one
# fp8 output, double buffered.
TILE_BYTES_PER_ELEMENT = 2 * (2 + 2 + 1)
# Token-major output tile override (default unset: the rule below is unchanged).
TOKEN_MAJOR_TILE_ENV = "TPU_GDN_NORM_QUANT_2D_TILE"
# Test-only numerics mutant of the token-major path (bench validity): the 2D-z
# kernel gathers head h + 1 where it should gather head h. Must break identity.
TEST_MUTANT_ENV = "TPU_GDN_NORM_QUANT_TEST_MUTANT"


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
    o_ref,
    z_ref,
    w_ref,
    q_ref,
    s_ref,
    scratch_ref,
    *,
    step_m: int,
    eps: float,
    head_shift: int = 0,
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
            [
                flat_s[(h + head_shift) % heads : step * heads : heads]
                for h in range(heads)
            ],
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


# Lane r1p GDN, DN2: fp8 rows written token-major ([tokens, heads * dim]) so the out_proj dot
# reads them at the FA o_proj rate. Default unset (the rules below are unchanged).
#   "swap": z arrives token-major 2D; the normalized o is moved to token-major by a value
#           transpose of each 8-token sub-block (no VMEM scratch round trip).
#   "late": z arrives head-major (as the PCP GDN kernel writes it); the whole computation runs in
#           the 3D kernel's layout and only the final clipped values are transposed.
#   "late16": as "late", but the bf16-rounded y is transposed (16-token sub-blocks, half the vregs)
#           and the per-token scale is applied after the transpose (same per-element ops).
TOKEN_MAJOR_MODE_ENV = "TPU_GDN_NORM_QUANT_TM"


def token_major_mode() -> str:
    mode = os.environ.get(TOKEN_MAJOR_MODE_ENV, "").strip().lower()
    if mode not in ("", "swap", "late", "late16"):
        raise ValueError(f"{TOKEN_MAJOR_MODE_ENV}={mode!r}: expected '', swap, late or late16")
    return mode


_TM_SUB = 8  # tokens per transposed sub-block: one f32 vreg row group per head


def _kernel_tm_swap(
    o_ref, z_ref, w_ref, q_ref, s_ref, *, step_m: int, eps: float, head_shift: int = 0
):
    bm, heads, dim = o_ref.shape
    step = min(bm, step_m)
    w = w_ref[...].astype(jnp.float32)[None, None, :]
    for r in range(bm // step):
        rows = []
        for u in range(step // _TM_SUB):
            o = o_ref[pl.ds(r * step + u * _TM_SUB, _TM_SUB), :, :].astype(jnp.float32)
            var = jnp.mean(o * o, axis=-1, keepdims=True)
            o_norm = jnp.swapaxes(o * jax.lax.rsqrt(var + eps) * w, 0, 1)  # [heads, sub, dim]
            rows.append(
                jnp.concatenate(
                    [o_norm[(h + head_shift) % heads] for h in range(heads)], axis=1
                )
            )
        o_norm_2d = jnp.concatenate(rows, axis=0)  # [step, heads * dim]
        sl = slice(r * step, (r + 1) * step)
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


def _kernel_tm_late(
    o_ref, z_ref, w_ref, q_ref, s_ref, *, step_m: int, eps: float, head_shift: int = 0
):
    bm, heads, dim = o_ref.shape
    step = min(bm, step_m)
    w = w_ref[...].astype(jnp.float32)[None, None, :]
    for r in range(bm // step):
        q_rows, s_rows = [], []
        for u in range(step // _TM_SUB):
            sub = pl.ds(r * step + u * _TM_SUB, _TM_SUB)
            o = o_ref[sub, :, :].astype(jnp.float32)  # [sub, heads, dim]
            z = z_ref[sub, :, :].astype(jnp.float32)
            var = jnp.mean(o * o, axis=-1, keepdims=True)
            y = o * jax.lax.rsqrt(var + eps) * w
            y = y * (z * jax.nn.sigmoid(z))
            y = y.astype(jnp.bfloat16).astype(jnp.float32)
            amax = jnp.max(jnp.abs(y), axis=(1, 2), keepdims=True)
            scale = amax * jnp.float32(1.0 / FP8_MAX)
            scale = jnp.where(scale == 0.0, 1.0, scale)
            scale_inv = jnp.float32(1.0) / scale
            q_hm = jnp.swapaxes(jnp.clip(y * scale_inv, -FP8_MAX, FP8_MAX), 0, 1)
            q_rows.append(
                jnp.concatenate([q_hm[(h + head_shift) % heads] for h in range(heads)], axis=1)
            )
            s_rows.append(scale.reshape(_TM_SUB, 1))
        sl = slice(r * step, (r + 1) * step)
        q_ref[sl, :] = jnp.concatenate(q_rows, axis=0).astype(FP8)
        s_ref[sl, :] = jnp.concatenate(s_rows, axis=0)


def _kernel_tm_late16(
    o_ref, z_ref, w_ref, q_ref, s_ref, *, step_m: int, eps: float, head_shift: int = 0
):
    bm, heads, dim = o_ref.shape
    step = min(bm, step_m)
    sub_n = 16
    w = w_ref[...].astype(jnp.float32)[None, None, :]
    for r in range(bm // step):
        q_rows, s_rows = [], []
        for u in range(step // sub_n):
            sub = pl.ds(r * step + u * sub_n, sub_n)
            o = o_ref[sub, :, :].astype(jnp.float32)
            z = z_ref[sub, :, :].astype(jnp.float32)
            var = jnp.mean(o * o, axis=-1, keepdims=True)
            y = o * jax.lax.rsqrt(var + eps) * w
            y = (y * (z * jax.nn.sigmoid(z))).astype(jnp.bfloat16)
            amax = jnp.max(jnp.abs(y.astype(jnp.float32)), axis=(1, 2), keepdims=True)
            scale = amax * jnp.float32(1.0 / FP8_MAX)
            scale = jnp.where(scale == 0.0, 1.0, scale)
            scale_inv = (jnp.float32(1.0) / scale).reshape(sub_n, 1)
            y_hm = jnp.swapaxes(y, 0, 1)  # [heads, sub, dim] bf16
            y_tm = jnp.concatenate(
                [y_hm[(h + head_shift) % heads] for h in range(heads)], axis=1
            ).astype(jnp.float32)
            q_rows.append(jnp.clip(y_tm * scale_inv, -FP8_MAX, FP8_MAX))
            s_rows.append(scale.reshape(sub_n, 1))
        sl = slice(r * step, (r + 1) * step)
        q_ref[sl, :] = jnp.concatenate(q_rows, axis=0).astype(FP8)
        s_ref[sl, :] = jnp.concatenate(s_rows, axis=0)


def _token_major_call(o, z, w, eps, mode, bm, step_m, interpret):
    tokens, heads, dim = o.shape
    cols = heads * dim
    mutant = {}
    if os.environ.get(TEST_MUTANT_ENV, "") == "head_shift":
        mutant = {"head_shift": 1}
    if mode == "swap":
        kernel, z_in = _kernel_tm_swap, z.reshape(tokens, cols)
        z_spec = pl.BlockSpec((bm, cols), lambda i: (i, 0))
    else:
        kernel = _kernel_tm_late16 if mode == "late16" else _kernel_tm_late
        z_in = z.reshape(tokens, heads, dim)
        z_spec = pl.BlockSpec((bm, heads, dim), lambda i: (i, 0, 0))
    return pl.pallas_call(
        functools.partial(kernel, step_m=step_m, eps=float(eps), **mutant),
        grid=(tokens // bm,),
        in_specs=[
            pl.BlockSpec((bm, heads, dim), lambda i: (i, 0, 0)),
            z_spec,
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
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name=f"gated_norm_quant_tm_{mode}",
        interpret=interpret,
    )(o, z_in, w)


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
    3. Either way the block's double-buffered bytes stay under
       ``TILE_VMEM_BUDGET_BYTES``.
    """
    vreg_cap = default_vreg_step(cols)
    budget_cap = max(
        8, TILE_VMEM_BUDGET_BYTES // (TILE_BYTES_PER_ELEMENT * cols) // 8 * 8
    )
    if tokens <= 256:
        dma_cap = max(8, (tokens // 6 // 8) * 8)
        return min(vreg_cap, dma_cap, budget_cap)
    return min(max(vreg_cap, 512), budget_cap)


def token_major_tile(tokens: int, dim: int, default: str = "") -> tuple[int, int] | None:
    """``TPU_GDN_NORM_QUANT_2D_TILE=BM[/STEP]``: a 2D-z block (rows, VREG step).

    The 2D-z kernel writes token-major fp8 rows ``[tokens, heads * dim]``,
    which the next fp8 matmul reads in place. The default rule takes it only
    when the block and its step are multiples of 32 rows; at 4,352 or 6,528
    GDN rows (64 heads) it picks 136 / 8 and 192 / 16 and falls back to the 3D
    kernel, whose ``[tokens, heads, dim]`` fp8 output reaches the matmul as
    head-major tiles. With the knob set: the largest multiple of 32 rows up to
    ``BM`` that divides ``tokens`` and ``STEP`` (default 32, a multiple of 32).
    Unset, or no such block: None (the default rule).
    """
    raw = os.environ.get(TOKEN_MAJOR_TILE_ENV, "").strip() or default
    if not raw:
        return None
    parts = [int(x) for x in raw.split("/")]
    cap = parts[0]
    step = parts[1] if len(parts) > 1 else 32
    if step <= 0 or step % 32 or dim % 128:
        return None
    for bm in range(min(cap, tokens) // 32 * 32, 0, -32):
        if tokens % bm == 0 and bm % step == 0:
            return bm, step
    return None


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
    tm_mode = token_major_mode() if tile is None else ""
    if tm_mode and dim % 128 == 0 and z.size == o.size:
        tm_tile = token_major_tile(tokens, dim, default="128/32")
        if tm_tile is not None and tm_tile[1] % (16 if tm_mode == "late16" else _TM_SUB) == 0:
            q2d, s = _token_major_call(o, z, w, eps, tm_mode, *tm_tile, interpret)
            return q2d.reshape(z.shape), s
    cap = default_tile_cap(tokens, cols) if tile is None else tile
    bm = legal_tile(tokens, cap)
    step_m = legal_tile(bm, default_vreg_step(cols))

    if z.ndim == 2:
        assert z.shape == (tokens, cols), (o.shape, z.shape, w.shape)
        token_major = token_major_tile(tokens, dim) if tile is None else None
        mutant = {}
        if token_major is not None:
            bm, step_m = token_major
            if os.environ.get(TEST_MUTANT_ENV, "") == "head_shift":
                mutant = {"head_shift": 1}
        if (
            (bm % 32 == 0 or (bm == tokens and bm % 8 == 0))
            and (step_m % 32 == 0 or step_m == bm)
            and dim % 128 == 0
        ):
            return pl.pallas_call(
                functools.partial(
                    _kernel_2dz, step_m=step_m, eps=float(eps), **mutant
                ),
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
