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
"""``TPU_FUSED_NORM_QKVA``: the MLA decode prologue as one Pallas call.

At decode each MLA layer runs three XLA ops in front of its attention:

    1. the add + RMSNorm of the residual stream (``input_layernorm``),
    2. ``fused_qkv_a_proj``: per-row fp8 quant of the norm output and an
       fp8 x fp8 dot against the replicated ``[6144, 2624]`` fp8 weight
       (16 MB, read whole by every core every layer),
    3. the latent norms: ``q_a_layernorm`` on the 2048 q columns and
       ``kv_a_layernorm`` on the next 512; the 64 rope columns pass through.

This kernel does all three in one call. It starts the weight stream first
(K-major row slabs through a VMEM ring, as ``fp8_skinny``'s ``k`` split) and
computes the add + RMSNorm and the activation quant while the first slabs are
in flight; the latent norms run on the f32 accumulators after the last slab.
The norm output and the new residual go back to HBM by DMA under the stream.

Numerics, step for step today's ops (``vllm/ir/ops/layernorm.py`` and
``quantized_matmul.util``):

    s        = f32(x) + f32(residual)          residual' = bf16(s)
    y        = bf16(bf16(s * rsqrt(mean(s^2) + eps)) * w_ln)
    scale    = f32(max|y|) / fp8_max;  0 -> 1  x_q = fp8(clip(f32(y) / scale))
    qkv      = bf16((dot(x_q, W) * scale) * w_scale)
    q_c      = rms_norm(qkv[:, :2048], w_qa),  kv_c = rms_norm(qkv[:, 2048:2560], w_kva)
    k_pe     = qkv[:, 2560:2624]

The one deliberate difference is the dot: x_q is held as bf16 (every e4m3
value is exact in bf16) and fed to a bf16 x fp8 MXU dot with f32
accumulation, one 1024-row slab at a time, as ``fp8_skinny`` does. XLA's fp8 x
fp8 dot accumulates below f32 precision on TPU, so ``qkv`` (and the norms of
it) can differ from today's in the last bf16 ulp, closer to the exact product.
Free elsewhere: the f32 summation order of the three mean squares.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Rows of the decode programs this serves (4, 8, 16, 32, 48, 64).
MAX_ROWS = 64
# K-slab rows and ring depth: fp8_skinny's measured config for this weight
# ((6144, 2688) -> ("k", 1024, 3)).
TILE_K = 1024
NBUF = 3
# Rows per slice of the add + RMSNorm prologue: one bf16 vreg's sublanes.
NORM_ROWS = 16


def supports(
    rows: int, hidden: int, n_out: int, q_rank: int, kv_rank: int, rope: int
) -> bool:
    return (
        0 < rows <= MAX_ROWS
        and hidden % TILE_K == 0
        and n_out == q_rank + kv_rank + rope
        and q_rank % 128 == 0
        and kv_rank % 128 == 0
        and 0 < rope <= 128
    )


def _rms(v, w, eps, dtype):
    """``ir.ops.rms_norm`` on an f32 array of bf16 values: the f32 mean of
    squares, the normalized value rounded to ``dtype``, times the weight,
    rounded again (a bf16 x bf16 product is exact in f32)."""
    var = jnp.mean(v * v, axis=-1, keepdims=True)
    normed = (v * jax.lax.rsqrt(var + eps)).astype(dtype)
    return (normed.astype(jnp.float32) * w.astype(jnp.float32)).astype(dtype)


def _quant_rows(v, q_max, q_dtype):
    """``quantize_tensor``'s per-row quant of the bf16 values ``v`` (f32):
    scale = max|v| / fp8_max, 0 -> 1; v_q = fp8(clip(v / scale))."""
    scale = jnp.max(jnp.abs(v), axis=-1, keepdims=True) / jnp.float32(q_max)
    scale = jnp.where(scale == 0, jnp.float32(1.0), scale)
    return jnp.clip(v / scale, -q_max, q_max).astype(q_dtype), scale


def _kernel(
    x_hbm,
    r_hbm,
    lnw_ref,
    ws_ref,
    qaw_ref,
    kvaw_ref,
    w_hbm,
    y_hbm,
    ro_hbm,
    qc_ref,
    kvc_ref,
    kpe_ref,
    *rest,
    eps,
    q_eps,
    kv_eps,
    q_rank,
    kv_rank,
    rope,
    t,
    nbuf,
    q_max,
    emit_q8,
):
    q8_ref, qs_ref = rest[:2] if emit_q8 else (None, None)
    (x_v, r_v, xq_v, xs_v, wbuf, acc_q, acc_kv, acc_pe, io_sem, w_sem) = (
        rest[2:] if emit_q8 else rest
    )
    rows, hidden = x_v.shape
    n_tiles = hidden // t
    dtype = x_v.dtype

    # The activations first (small), then the weight ring: both in flight
    # before any compute.
    x_in = pltpu.make_async_copy(x_hbm, x_v, io_sem.at[0])
    r_in = pltpu.make_async_copy(r_hbm, r_v, io_sem.at[1])
    x_in.start()
    r_in.start()

    # The weight as XLA stores the canonical [hidden, n] parameter: N-major
    # ([n, hidden] row-major), so a K-slab is a column block of it.
    def w_copy(j):
        return pltpu.make_async_copy(
            w_hbm.at[:, pl.ds(j * t, t)], wbuf.at[j % nbuf], w_sem.at[j % nbuf]
        )

    for j in range(min(nbuf, n_tiles)):
        w_copy(j).start()

    x_in.wait()
    r_in.wait()

    # Add + RMSNorm + per-row fp8 quant, one row slice at a time, under the
    # first weight slabs. y and the new residual overwrite x and r in place
    # (each slice reads its rows before it writes them).
    slice_rows = min(rows, NORM_ROWS)
    for start in range(0, rows, slice_rows):
        n = min(slice_rows, rows - start)
        sl = pl.ds(start, n)
        s = x_v[sl, :].astype(jnp.float32) + r_v[sl, :].astype(jnp.float32)
        r_v[sl, :] = s.astype(dtype)
        y = _rms(s, lnw_ref[...], eps, dtype)
        x_v[sl, :] = y
        x_q, scale = _quant_rows(y.astype(jnp.float32), q_max, wbuf.dtype)
        xs_v[sl, :] = scale
        xq_v[sl, :] = x_q.astype(xq_v.dtype)

    y_out = pltpu.make_async_copy(x_v, y_hbm, io_sem.at[0])
    r_out = pltpu.make_async_copy(r_v, ro_hbm, io_sem.at[1])
    y_out.start()
    r_out.start()

    # The weight stream: each slab's three column blocks (q, kv, rope) into
    # their own f32 accumulators, so every later slice is lane-aligned.
    dn = (((1,), (1,)), ((), ()))
    kv_end = q_rank + kv_rank
    for j in range(n_tiles):
        w_copy(j).wait()
        slot = j % nbuf
        xt = xq_v[:, pl.ds(j * t, t)]
        parts = (
            (acc_q, wbuf[slot, pl.ds(0, q_rank), :]),
            (acc_kv, wbuf[slot, pl.ds(q_rank, kv_rank), :]),
            (acc_pe, wbuf[slot, pl.ds(kv_end, rope), :]),
        )
        for acc, w in parts:
            p = jax.lax.dot_general(xt, w, dn, preferred_element_type=jnp.float32)
            if j == 0:
                acc[...] = p
            else:
                acc[...] += p
        if j + nbuf < n_tiles:
            w_copy(j + nbuf).start()

    # Dequant exactly as xla_quantized_matmul: (acc * x_scale) * w_scale.
    xs = xs_v[...]
    q = (acc_q[...] * xs * ws_ref[:, pl.ds(0, q_rank)]).astype(dtype)
    kv = (acc_kv[...] * xs * ws_ref[:, pl.ds(q_rank, kv_rank)]).astype(dtype)
    pe = (acc_pe[...] * xs * ws_ref[:, pl.ds(kv_end, rope)]).astype(dtype)
    q_c = _rms(q.astype(jnp.float32), qaw_ref[...], q_eps, dtype)
    qc_ref[...] = q_c
    if q8_ref is not None:
        # q_b_proj's own activation quant, for its XLA matmul above 16 rows.
        q8, q_scale = _quant_rows(q_c.astype(jnp.float32), q_max, q8_ref.dtype)
        q8_ref[...] = q8
        qs_ref[...] = q_scale
    kvc_ref[...] = _rms(kv.astype(jnp.float32), kvaw_ref[...], kv_eps, dtype)
    kpe_ref[...] = pe

    y_out.wait()
    r_out.wait()


@functools.cache
def _build(
    rows,
    hidden,
    n_out,
    dtype,
    w_dtype,
    eps,
    q_eps,
    kv_eps,
    q_rank,
    kv_rank,
    rope,
    t,
    nbuf,
    emit_q8,
    interpret=False,
):
    if hidden % t or t % 128:
        raise ValueError(f"slab rows {t} must divide hidden {hidden} in lane multiples")
    nbuf = max(1, min(nbuf, hidden // t))
    q_max = float(jnp.finfo(w_dtype).max)
    any_spec = pl.BlockSpec(memory_space=pl.ANY)

    def row_block(width):
        return pl.BlockSpec((1, width), lambda i: (0, 0))

    def out_block(width):
        return pl.BlockSpec((rows, width), lambda i: (0, 0))

    act_bytes = rows * hidden * jnp.dtype(dtype).itemsize
    ring_bytes = nbuf * t * (-(-n_out // 128) * 128)
    vmem = 3 * act_bytes + ring_bytes + rows * (n_out + 384) * 4 * 2
    vmem_limit = min(64 * 1024 * 1024, max(32 * 1024 * 1024, 2 * vmem))
    kern = functools.partial(
        _kernel,
        eps=eps,
        q_eps=q_eps,
        kv_eps=kv_eps,
        q_rank=q_rank,
        kv_rank=kv_rank,
        rope=rope,
        t=t,
        nbuf=nbuf,
        q_max=q_max,
        emit_q8=emit_q8,
    )
    q8_specs = [out_block(q_rank), out_block(1)] if emit_q8 else []
    q8_shapes = (
        [
            jax.ShapeDtypeStruct((rows, q_rank), w_dtype),
            jax.ShapeDtypeStruct((rows, 1), jnp.float32),
        ]
        if emit_q8
        else []
    )
    return pl.pallas_call(
        kern,
        grid=(1,),
        in_specs=[
            any_spec,  # x
            any_spec,  # residual
            row_block(hidden),  # input_layernorm weight
            row_block(n_out),  # per-channel weight scale
            row_block(q_rank),  # q_a_layernorm weight
            row_block(kv_rank),  # kv_a_layernorm weight
            any_spec,  # fp8 weight [hidden, n_out], streamed
        ],
        out_specs=[
            any_spec,  # y
            any_spec,  # new residual
            out_block(q_rank),
            out_block(kv_rank),
            out_block(rope),
            *q8_specs,
        ],
        out_shape=[
            jax.ShapeDtypeStruct((rows, hidden), dtype),
            jax.ShapeDtypeStruct((rows, hidden), dtype),
            jax.ShapeDtypeStruct((rows, q_rank), dtype),
            jax.ShapeDtypeStruct((rows, kv_rank), dtype),
            jax.ShapeDtypeStruct((rows, rope), dtype),
            *q8_shapes,
        ],
        scratch_shapes=[
            pltpu.VMEM((rows, hidden), dtype),  # x, then y
            pltpu.VMEM((rows, hidden), dtype),  # residual, then the new one
            pltpu.VMEM((rows, hidden), jnp.bfloat16),  # x_q held as bf16
            pltpu.VMEM((rows, 1), jnp.float32),  # x scale
            pltpu.VMEM((nbuf, n_out, t), w_dtype),  # weight ring, N-major slabs
            pltpu.VMEM((rows, q_rank), jnp.float32),
            pltpu.VMEM((rows, kv_rank), jnp.float32),
            pltpu.VMEM((rows, rope), jnp.float32),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((nbuf,)),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=int(vmem_limit),
        ),
        name=f"fused_norm_qkva_m{rows}_k{hidden}_n{n_out}_t{t}_b{nbuf}"
        f"{'_q8' if emit_q8 else ''}",
        interpret=pltpu.InterpretParams() if interpret else False,
    )


@functools.partial(jax.jit, static_argnums=(0,))
def _call_hbm_pinned(call, x, residual, ln_w, w_s, qa_w, kva_w, w_q):
    # As fp8_skinny `hbm_pin`: the in_spec's ANY only types the ref; XLA's
    # memory-space assignment reads the operand's aval, and without this it may
    # stage the 16 MB weight into VMEM with a copy ahead of the call.
    w_q = pltpu.with_memory_space_constraint(w_q, pltpu.HBM)
    return call(x, residual, ln_w, w_s, qa_w, kva_w, w_q)


def _n_major(w_q: jax.Array) -> jax.Array:
    """[hidden, n] -> [n, hidden]: the canonical parameter is stored N-major
    (layout {0,1}), so this transpose is a bitcast, where handing the kernel
    the [hidden, n] row-major operand made XLA transpose the 16 MB weight
    before every call (37.4 us at decode-b1, j-709ea6e5-6a38-48b9-abb7)."""
    return w_q.T


def fused_norm_qkva(
    x: jax.Array,
    residual: jax.Array,
    ln_w: jax.Array,
    w_q: jax.Array,
    w_s: jax.Array,
    qa_w: jax.Array,
    kva_w: jax.Array,
    *,
    eps: float,
    q_eps: float,
    kv_eps: float,
    q_rank: int,
    kv_rank: int,
    emit_q8: bool = False,
    hbm_pin: bool = True,
    interpret: bool = False,
):
    """Returns ``(y, new_residual, q_c, kv_c_normed, k_pe)``, then with
    ``emit_q8`` ``(q_c_fp8 [rows, q_rank], q_c_scale f32 [rows, 1])``, bit for
    bit ``quantize_tensor(q_c, fp8)`` (q_b_proj's prequantized input).

    ``x``/``residual``: ``[rows <= 64, hidden]`` bf16; ``ln_w`` ``[hidden]``;
    ``w_q``: fp8 ``[hidden, q_rank + kv_rank + rope]`` (K-major, the canonical
    flipped layout); ``w_s``: f32 ``[n_out]``; ``qa_w`` ``[q_rank]``,
    ``kva_w`` ``[kv_rank]``. ``q_c`` is ``q_a_layernorm`` of the q columns,
    ``kv_c_normed`` ``kv_a_layernorm`` of the next ``kv_rank``, ``k_pe`` the
    raw last ``rope`` columns, all bf16.
    """
    rows, hidden = x.shape
    n_out = w_q.shape[1]
    rope = n_out - q_rank - kv_rank
    if not supports(rows, hidden, n_out, q_rank, kv_rank, rope):
        raise ValueError(f"{x.shape=} {w_q.shape=} {q_rank=} {kv_rank=}")
    if w_q.shape[0] != hidden or residual.shape != x.shape:
        raise ValueError(f"{x.shape=} {residual.shape=} {w_q.shape=}")
    call = _build(
        rows,
        hidden,
        n_out,
        x.dtype,
        w_q.dtype,
        float(eps),
        float(q_eps),
        float(kv_eps),
        q_rank,
        kv_rank,
        rope,
        TILE_K,
        NBUF,
        bool(emit_q8),
        bool(interpret),
    )
    args = (
        x,
        residual,
        ln_w.reshape(1, hidden),
        w_s.reshape(1, n_out).astype(jnp.float32),
        qa_w.reshape(1, q_rank),
        kva_w.reshape(1, kv_rank),
        _n_major(w_q),
    )
    if hbm_pin:
        return tuple(_call_hbm_pinned(call, *args))
    return tuple(call(*args))


def reference(
    x, residual, ln_w, w_q, w_s, qa_w, kva_w, *, eps, q_eps, kv_eps, q_rank, kv_rank
):
    """Today's three ops in JAX (XLA): ``fused_add_rmsnorm.reference``,
    ``xla_quantized_matmul``, the split and ``ir.ops.rms_norm`` twice; then
    q_b_proj's activation quant of q_c (``quantize_tensor``)."""
    from vllm_torchtpu.kernels.fused_add_rmsnorm import reference as add_rms_norm
    from vllm_torchtpu.kernels.quantized_matmul.util import (
        quantize_tensor,
        xla_quantized_matmul,
    )

    def rms_norm(v, w, e):
        f = v.astype(jnp.float32)
        var = jnp.mean(f * f, axis=-1, keepdims=True)
        return ((f * jax.lax.rsqrt(var + e)).astype(w.dtype) * w).astype(v.dtype)

    y, new_res = add_rms_norm(x, residual, ln_w, eps)
    qkv = xla_quantized_matmul(y, w_q, w_s)
    q_c = rms_norm(qkv[:, :q_rank], qa_w, q_eps)
    kv_c = rms_norm(qkv[:, q_rank : q_rank + kv_rank], kva_w, kv_eps)
    k_pe = qkv[:, q_rank + kv_rank :]
    q8, q_scale = quantize_tensor(q_c, w_q.dtype)
    return y, new_res, q_c, kv_c, k_pe, q8, q_scale
