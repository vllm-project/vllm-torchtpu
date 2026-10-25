# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused FP32 Block Attention Residual for token-local K3 activations."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _mix_kernel_2d(
    p_ref,
    h_ref,
    w_ref,
    out_ref,
    m_scratch,
    d_scratch,
    acc_scratch,
    *,
    eps: float,
    blocks: int,
    tile_t: int,
    hidden: int,
    step_h: int,
):
    k = pl.program_id(1)
    n_steps = hidden // step_h
    zero_vec = jnp.zeros((tile_t, step_h), jnp.float32)

    def body_k(i, carry):
        sq_v, dot_v = carry
        c = i * step_h
        wc = w_ref[:, pl.ds(c, step_h)]
        hk = h_ref[:, pl.ds(c, step_h)].astype(jnp.float32)
        return sq_v + hk * hk, dot_v + hk * wc

    sq_vec, dot_vec = jax.lax.fori_loop(0, n_steps, body_k, (zero_vec, zero_vec))
    sq_k = jnp.broadcast_to(jnp.sum(sq_vec, axis=-1, keepdims=True), (tile_t, 128))
    dot_k = jnp.broadcast_to(jnp.sum(dot_vec, axis=-1, keepdims=True), (tile_t, 128))
    inv_k = jax.lax.rsqrt(sq_k * (1.0 / hidden) + eps)
    sk = dot_k * inv_k

    @pl.when(k == 0)
    def init():
        def body_p(i, carry):
            sq_v, dot_v = carry
            c = i * step_h
            wc = w_ref[:, pl.ds(c, step_h)]
            pc = p_ref[:, pl.ds(c, step_h)].astype(jnp.float32)
            acc_scratch[:, pl.ds(c, step_h)] = pc
            return sq_v + pc * pc, dot_v + pc * wc

        sq_p_vec, dot_p_vec = jax.lax.fori_loop(
            0, n_steps, body_p, (zero_vec, zero_vec)
        )
        sq_p = jnp.broadcast_to(
            jnp.sum(sq_p_vec, axis=-1, keepdims=True), (tile_t, 128)
        )
        dot_p = jnp.broadcast_to(
            jnp.sum(dot_p_vec, axis=-1, keepdims=True), (tile_t, 128)
        )
        inv_p = jax.lax.rsqrt(sq_p * (1.0 / hidden) + eps)
        m_scratch[...] = dot_p * inv_p
        d_scratch[...] = jnp.ones((tile_t, 128), jnp.float32)

    m_old = m_scratch[...]
    d_old = d_scratch[...]
    m_new = jnp.maximum(m_old, sk)
    scale_old = jnp.exp(m_old - m_new)
    scale_k = jnp.exp(sk - m_new)
    d_new = d_old * scale_old + scale_k
    m_scratch[...] = m_new
    d_scratch[...] = d_new
    so = scale_old[:, :1]
    sk1 = scale_k[:, :1]

    def body_acc(i, _):
        c = i * step_h
        hk = h_ref[:, pl.ds(c, step_h)].astype(jnp.float32)
        ac = acc_scratch[:, pl.ds(c, step_h)]
        acc_scratch[:, pl.ds(c, step_h)] = ac * so + hk * sk1
        return None

    jax.lax.fori_loop(0, n_steps, body_acc, None)

    @pl.when(k == blocks - 1)
    def finalize():
        inv_d = 1.0 / d_scratch[:, :1]

        def body_fin(i, _):
            c = i * step_h
            ac = acc_scratch[:, pl.ds(c, step_h)]
            out_ref[:, pl.ds(c, step_h)] = (ac * inv_d).astype(out_ref.dtype)
            return None

        jax.lax.fori_loop(0, n_steps, body_fin, None)


def _mix_kernel_3d(
    p_ref,
    h_ref,
    w_ref,
    out_ref,
    *,
    eps: float,
    blocks: int,
    tile_t: int,
    hidden: int,
    step_h: int,
):
    n_steps = hidden // step_h
    zero_p = jnp.zeros((tile_t, step_h), jnp.float32)
    zero_h = jnp.zeros((tile_t, blocks, step_h), jnp.float32)

    def body_stat(i, carry):
        sq_p, dot_p, sq_h, dot_h = carry
        c = i * step_h
        wc = w_ref[pl.ds(c, step_h)].astype(jnp.float32)
        pc = p_ref[:, pl.ds(c, step_h)].astype(jnp.float32)
        hc = h_ref[:, :, pl.ds(c, step_h)].astype(jnp.float32)
        return (
            sq_p + pc * pc,
            dot_p + pc * wc,
            sq_h + hc * hc,
            dot_h + hc * wc,
        )

    sq_p_v, dot_p_v, sq_h_v, dot_h_v = jax.lax.fori_loop(
        0, n_steps, body_stat, (zero_p, zero_p, zero_h, zero_h)
    )
    sq_p = jnp.sum(sq_p_v, axis=-1, keepdims=True)
    dot_p = jnp.sum(dot_p_v, axis=-1, keepdims=True)
    sq_h = jnp.sum(sq_h_v, axis=-1)
    dot_h = jnp.sum(dot_h_v, axis=-1)
    inv_p = jax.lax.rsqrt(sq_p * (1.0 / hidden) + eps)
    inv_h = jax.lax.rsqrt(sq_h * (1.0 / hidden) + eps)
    sp = dot_p * inv_p
    sh = dot_h * inv_h
    m = jnp.maximum(jnp.max(sh, axis=-1, keepdims=True), sp)
    eh = jnp.exp(sh - m)
    ep = jnp.exp(sp - m)
    inv_d = 1.0 / (jnp.sum(eh, axis=-1, keepdims=True) + ep)
    prob_h = (eh * inv_d)[:, :, None]
    prob_p = ep * inv_d

    def body_mix(i, _):
        c = i * step_h
        pc = p_ref[:, pl.ds(c, step_h)].astype(jnp.float32)
        hc = h_ref[:, :, pl.ds(c, step_h)].astype(jnp.float32)
        acc = jnp.sum(hc * prob_h, axis=1) + pc * prob_p
        out_ref[:, pl.ds(c, step_h)] = acc.astype(out_ref.dtype)
        return None

    jax.lax.fori_loop(0, n_steps, body_mix, None)


def attention_residual(prefix, history, folded_weight, *, eps=1e-5, interpret=False):
    """Mix [tokens,hidden] prefix and saved residual blocks.

    Accepts either [blocks * tokens, hidden] 2D block-major history (zero-copy
    prefix slices from KimiDecoderLayer) or [tokens, blocks, hidden] 3D history.
    """
    tokens, hidden = prefix.shape
    if folded_weight.shape != (hidden,):
        raise ValueError(
            "Attention Residual needs one folded weight per hidden feature"
        )
    if history.ndim == 2:
        if history.shape[1] != hidden or (
            tokens > 0 and history.shape[0] % tokens != 0
        ):
            raise ValueError("Attention Residual prefix/history shapes disagree")
        blocks = 0 if tokens == 0 else history.shape[0] // tokens
        if blocks == 0 or tokens == 0:
            return prefix
        if tokens % 8:
            raise ValueError("Fused Attention Residual requires 8-token tiles")
        if tokens % 64 == 0:
            tile_t, target_step_h = 64, 128
        elif tokens % 32 == 0:
            tile_t, target_step_h = 32, 256
        elif tokens % 16 == 0:
            tile_t, target_step_h = 16, 512
        else:
            tile_t, target_step_h = 8, 512
        tiles_t = tokens // tile_t
        step_h = next(
            s for s in (target_step_h, 512, 256, 128) if hidden % s == 0 and s <= hidden
        )
        w_2d = jax.lax.broadcast_in_dim(
            folded_weight.astype(jnp.float32), (tile_t, hidden), (1,)
        )
        return pl.pallas_call(
            functools.partial(
                _mix_kernel_2d,
                eps=eps,
                blocks=blocks,
                tile_t=tile_t,
                hidden=hidden,
                step_h=step_h,
            ),
            out_shape=jax.ShapeDtypeStruct(prefix.shape, prefix.dtype),
            grid=(tiles_t, blocks),
            in_specs=(
                pl.BlockSpec((tile_t, hidden), lambda i, k: (i, 0)),
                pl.BlockSpec((tile_t, hidden), lambda i, k: (k * tiles_t + i, 0)),
                pl.BlockSpec((tile_t, hidden), lambda i, k: (0, 0)),
            ),
            out_specs=pl.BlockSpec((tile_t, hidden), lambda i, k: (i, 0)),
            scratch_shapes=(
                pltpu.VMEM((tile_t, 128), jnp.float32),
                pltpu.VMEM((tile_t, 128), jnp.float32),
                pltpu.VMEM((tile_t, hidden), jnp.float32),
            ),
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=63 * 1024**2),
            interpret=interpret,
            name="kimi_attnres_compact_m16",
        )(prefix, history, w_2d)

    if history.shape[0] != tokens or history.shape[-1] != hidden:
        raise ValueError("Attention Residual prefix/history shapes disagree")
    blocks = history.shape[1]
    if blocks == 0 or tokens == 0:
        return prefix
    if tokens % 8:
        raise ValueError("Fused Attention Residual requires 8-token tiles")
    tile_t = 8
    step_h = next(s for s in (256, 128) if hidden % s == 0)
    return pl.pallas_call(
        functools.partial(
            _mix_kernel_3d,
            eps=eps,
            blocks=blocks,
            tile_t=tile_t,
            hidden=hidden,
            step_h=step_h,
        ),
        out_shape=jax.ShapeDtypeStruct(prefix.shape, prefix.dtype),
        grid=(tokens // tile_t,),
        in_specs=(
            pl.BlockSpec((tile_t, hidden), lambda i: (i, 0)),
            pl.BlockSpec((tile_t, blocks, hidden), lambda i: (i, 0, 0)),
            pl.BlockSpec((hidden,), lambda i: (0,)),
        ),
        out_specs=pl.BlockSpec((tile_t, hidden), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=63 * 1024**2),
        interpret=interpret,
        name="kimi_attnres_compact_m16",
    )(prefix, history, folded_weight)
