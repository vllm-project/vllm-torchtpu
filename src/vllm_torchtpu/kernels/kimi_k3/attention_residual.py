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
    unroll: int,
):
    k = pl.program_id(1)
    n_outer = hidden // (step_h * unroll)
    zero_vec = jnp.zeros((tile_t, step_h), jnp.float32)

    def body_k(i, carry):
        sq_v, dot_v = carry
        base = i * (step_h * unroll)
        for u in range(unroll):
            c = base + u * step_h
            wc = w_ref[pl.ds(c, step_h)].astype(jnp.float32)
            hk = h_ref[:, pl.ds(c, step_h)].astype(jnp.float32)
            sq_v = sq_v + hk * hk
            dot_v = dot_v + hk * wc
        return sq_v, dot_v

    sq_vec, dot_vec = jax.lax.fori_loop(0, n_outer, body_k, (zero_vec, zero_vec))
    sq_k = jnp.broadcast_to(jnp.sum(sq_vec, axis=-1, keepdims=True), (tile_t, 128))
    dot_k = jnp.broadcast_to(jnp.sum(dot_vec, axis=-1, keepdims=True), (tile_t, 128))
    inv_k = jax.lax.rsqrt(sq_k * (1.0 / hidden) + eps)
    sk = dot_k * inv_k

    @pl.when(k == 0)
    def init():
        step_p = min(256, step_h)
        n_steps_p = hidden // step_p
        zero_p = jnp.zeros((tile_t, step_p), jnp.float32)

        def body_p(i, carry):
            sq_v, dot_v = carry
            c = i * step_p
            wc = w_ref[pl.ds(c, step_p)].astype(jnp.float32)
            pc = p_ref[:, pl.ds(c, step_p)].astype(jnp.float32)
            acc_scratch[:, pl.ds(c, step_p)] = pc
            return sq_v + pc * pc, dot_v + pc * wc

        sq_p_vec, dot_p_vec = jax.lax.fori_loop(
            0, n_steps_p, body_p, (zero_p, zero_p)
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
        base = i * (step_h * unroll)
        for u in range(unroll):
            c = base + u * step_h
            hk = h_ref[:, pl.ds(c, step_h)].astype(jnp.float32)
            ac = acc_scratch[:, pl.ds(c, step_h)]
            acc_scratch[:, pl.ds(c, step_h)] = ac * so + hk * sk1
        return None

    jax.lax.fori_loop(0, n_outer, body_acc, None)

    @pl.when(k == blocks - 1)
    def finalize():
        inv_d = 1.0 / d_scratch[:, :1]

        def body_fin(i, _):
            base = i * (step_h * unroll)
            for u in range(unroll):
                c = base + u * step_h
                ac = acc_scratch[:, pl.ds(c, step_h)]
                out_ref[:, pl.ds(c, step_h)] = (ac * inv_d).astype(out_ref.dtype)
            return None

        jax.lax.fori_loop(0, n_outer, body_fin, None)


def _mix_kernel_2d_inv(
    p_ref,
    h_ref,
    w_ref,
    inv_ref,
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
    unroll: int,
    packed_inv: bool,
):
    k = pl.program_id(1)
    n_outer = hidden // (step_h * unroll)
    zero_vec = jnp.zeros((tile_t, step_h), jnp.float32)

    def body_k(i, dot_v):
        base = i * (step_h * unroll)
        for u in range(unroll):
            c = base + u * step_h
            wc = w_ref[pl.ds(c, step_h)].astype(jnp.float32)
            hk = h_ref[:, pl.ds(c, step_h)].astype(jnp.float32)
            dot_v = dot_v + hk * wc
        return dot_v

    dot_vec = jax.lax.fori_loop(0, n_outer, body_k, zero_vec)
    dot_k = jnp.broadcast_to(jnp.sum(dot_vec, axis=-1, keepdims=True), (tile_t, 128))
    if packed_inv:
        sub = 128 // tile_t
        iota_b = jax.lax.broadcasted_iota(jnp.int32, (blocks, 128), 0)
        inv_row = jnp.sum(
            jnp.where(iota_b == k, inv_ref[...], 0.0), axis=0, keepdims=True
        )
        row_idx = jax.lax.broadcasted_iota(jnp.int32, (tile_t, 128), 0)
        col_idx = jax.lax.broadcasted_iota(jnp.int32, (tile_t, 128), 1)
        target_col = (pl.program_id(0) % sub) * tile_t + row_idx
        inv_k_1 = jnp.sum(
            jnp.where(
                col_idx == target_col,
                jnp.broadcast_to(inv_row, (tile_t, 128)),
                0.0,
            ),
            axis=-1,
            keepdims=True,
        )
        inv_k = jnp.broadcast_to(inv_k_1, (tile_t, 128))
    else:
        inv_k = inv_ref[...]
    sk = dot_k * inv_k

    @pl.when(k == 0)
    def init():
        step_p = min(256, step_h)
        n_steps_p = hidden // step_p
        zero_p = jnp.zeros((tile_t, step_p), jnp.float32)

        def body_p(i, carry):
            sq_v, dot_v = carry
            c = i * step_p
            wc = w_ref[pl.ds(c, step_p)].astype(jnp.float32)
            pc = p_ref[:, pl.ds(c, step_p)].astype(jnp.float32)
            acc_scratch[:, pl.ds(c, step_p)] = pc
            return sq_v + pc * pc, dot_v + pc * wc

        sq_p_vec, dot_p_vec = jax.lax.fori_loop(
            0, n_steps_p, body_p, (zero_p, zero_p)
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
        base = i * (step_h * unroll)
        for u in range(unroll):
            c = base + u * step_h
            hk = h_ref[:, pl.ds(c, step_h)].astype(jnp.float32)
            ac = acc_scratch[:, pl.ds(c, step_h)]
            acc_scratch[:, pl.ds(c, step_h)] = ac * so + hk * sk1
        return None

    jax.lax.fori_loop(0, n_outer, body_acc, None)

    @pl.when(k == blocks - 1)
    def finalize():
        inv_d = 1.0 / d_scratch[:, :1]

        def body_fin(i, _):
            base = i * (step_h * unroll)
            for u in range(unroll):
                c = base + u * step_h
                ac = acc_scratch[:, pl.ds(c, step_h)]
                out_ref[:, pl.ds(c, step_h)] = (ac * inv_d).astype(out_ref.dtype)
            return None

        jax.lax.fori_loop(0, n_outer, body_fin, None)


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
    unroll: int,
):
    n_outer = hidden // (step_h * unroll)
    zero_p = jnp.zeros((tile_t, step_h), jnp.float32)
    zero_h = jnp.zeros((tile_t, blocks, step_h), jnp.float32)

    def body_stat(i, carry):
        sq_p, dot_p, sq_h, dot_h = carry
        base = i * (step_h * unroll)
        for u in range(unroll):
            c = base + u * step_h
            wc = w_ref[pl.ds(c, step_h)].astype(jnp.float32)
            pc = p_ref[:, pl.ds(c, step_h)].astype(jnp.float32)
            hc = h_ref[:, :, pl.ds(c, step_h)].astype(jnp.float32)
            sq_p = sq_p + pc * pc
            dot_p = dot_p + pc * wc
            sq_h = sq_h + hc * hc
            dot_h = dot_h + hc * wc
        return sq_p, dot_p, sq_h, dot_h

    sq_p_v, dot_p_v, sq_h_v, dot_h_v = jax.lax.fori_loop(
        0, n_outer, body_stat, (zero_p, zero_p, zero_h, zero_h)
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
        base = i * (step_h * unroll)
        for u in range(unroll):
            c = base + u * step_h
            pc = p_ref[:, pl.ds(c, step_h)].astype(jnp.float32)
            hc = h_ref[:, :, pl.ds(c, step_h)].astype(jnp.float32)
            acc = jnp.sum(hc * prob_h, axis=1) + pc * prob_p
            out_ref[:, pl.ds(c, step_h)] = acc.astype(out_ref.dtype)
        return None

    jax.lax.fori_loop(0, n_outer, body_mix, None)


def _mix_kernel_3d_inv(
    p_ref,
    h_ref,
    w_ref,
    inv_ref,
    out_ref,
    *,
    eps: float,
    blocks: int,
    tile_t: int,
    hidden: int,
    step_h: int,
    unroll: int,
):
    n_outer = hidden // (step_h * unroll)
    zero_p = jnp.zeros((tile_t, step_h), jnp.float32)
    zero_h = jnp.zeros((tile_t, blocks, step_h), jnp.float32)

    def body_stat(i, carry):
        sq_p, dot_p, dot_h = carry
        base = i * (step_h * unroll)
        for u in range(unroll):
            c = base + u * step_h
            wc = w_ref[pl.ds(c, step_h)].astype(jnp.float32)
            pc = p_ref[:, pl.ds(c, step_h)].astype(jnp.float32)
            hc = h_ref[:, :, pl.ds(c, step_h)].astype(jnp.float32)
            sq_p = sq_p + pc * pc
            dot_p = dot_p + pc * wc
            dot_h = dot_h + hc * wc
        return sq_p, dot_p, dot_h

    sq_p_v, dot_p_v, dot_h_v = jax.lax.fori_loop(
        0, n_outer, body_stat, (zero_p, zero_p, zero_h)
    )
    sq_p = jnp.sum(sq_p_v, axis=-1, keepdims=True)
    dot_p = jnp.sum(dot_p_v, axis=-1, keepdims=True)
    dot_h = jnp.sum(dot_h_v, axis=-1)
    inv_p = jax.lax.rsqrt(sq_p * (1.0 / hidden) + eps)
    inv_h = inv_ref[:, :, 0]
    sp = dot_p * inv_p
    sh = dot_h * inv_h
    m = jnp.maximum(jnp.max(sh, axis=-1, keepdims=True), sp)
    eh = jnp.exp(sh - m)
    ep = jnp.exp(sp - m)
    inv_d = 1.0 / (jnp.sum(eh, axis=-1, keepdims=True) + ep)
    prob_h = (eh * inv_d)[:, :, None]
    prob_p = ep * inv_d

    def body_mix(i, _):
        base = i * (step_h * unroll)
        for u in range(unroll):
            c = base + u * step_h
            pc = p_ref[:, pl.ds(c, step_h)].astype(jnp.float32)
            hc = h_ref[:, :, pl.ds(c, step_h)].astype(jnp.float32)
            acc = jnp.sum(hc * prob_h, axis=1) + pc * prob_p
            out_ref[:, pl.ds(c, step_h)] = acc.astype(out_ref.dtype)
        return None

    jax.lax.fori_loop(0, n_outer, body_mix, None)


def attention_residual(
    prefix,
    history,
    folded_weight,
    block_inv_rms=None,
    *,
    eps=1e-5,
    interpret=False,
):
    """Mix [tokens,hidden] prefix and saved residual blocks.

    Accepts either [blocks * tokens, hidden] 2D block-major history (zero-copy
    prefix slices from KimiDecoderLayer) or [tokens, blocks, hidden] 3D history.
    When ``block_inv_rms`` ([blocks, tokens] or [tokens, blocks] float32) is
    provided, skips recomputing historical block RMS norms inside the kernel.
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
        tile_t = 8 if tokens % 16 else 16
        tiles_t = tokens // tile_t
        step_h = next(s for s in (512, 256, 128) if hidden % s == 0 and s <= hidden)
        n_steps = hidden // step_h
        if block_inv_rms is not None:
            inv = block_inv_rms.astype(jnp.float32)
            if inv.ndim == 2 and inv.shape == (tokens, blocks) and tokens != blocks:
                inv = jnp.transpose(inv, (1, 0))
            else:
                inv = inv.reshape(blocks, tokens)
            unroll = next(u for u in (7, 2, 1) if n_steps % u == 0)
            packed_inv = tokens % 128 == 0
            if packed_inv:
                sub = 128 // tile_t
                inv_arg = inv
                inv_spec = pl.BlockSpec((blocks, 128), lambda i, k: (0, i // sub))
            else:
                inv_arg = jnp.broadcast_to(
                    inv.reshape(blocks * tokens, 1), (blocks * tokens, 128)
                )
                inv_spec = pl.BlockSpec(
                    (tile_t, 128), lambda i, k: (k * tiles_t + i, 0)
                )
            return pl.pallas_call(
                functools.partial(
                    _mix_kernel_2d_inv,
                    eps=eps,
                    blocks=blocks,
                    tile_t=tile_t,
                    hidden=hidden,
                    step_h=step_h,
                    unroll=unroll,
                    packed_inv=packed_inv,
                ),
                out_shape=jax.ShapeDtypeStruct(prefix.shape, prefix.dtype),
                grid=(tiles_t, blocks),
                in_specs=(
                    pl.BlockSpec((tile_t, hidden), lambda i, k: (i, 0)),
                    pl.BlockSpec((tile_t, hidden), lambda i, k: (k * tiles_t + i, 0)),
                    pl.BlockSpec((hidden,), lambda i, k: (0,)),
                    inv_spec,
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
            )(prefix, history, folded_weight, inv_arg)

        unroll = next(u for u in (2, 1) if n_steps % u == 0)
        return pl.pallas_call(
            functools.partial(
                _mix_kernel_2d,
                eps=eps,
                blocks=blocks,
                tile_t=tile_t,
                hidden=hidden,
                step_h=step_h,
                unroll=unroll,
            ),
            out_shape=jax.ShapeDtypeStruct(prefix.shape, prefix.dtype),
            grid=(tiles_t, blocks),
            in_specs=(
                pl.BlockSpec((tile_t, hidden), lambda i, k: (i, 0)),
                pl.BlockSpec((tile_t, hidden), lambda i, k: (k * tiles_t + i, 0)),
                pl.BlockSpec((hidden,), lambda i, k: (0,)),
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
        )(prefix, history, folded_weight)

    if history.shape[0] != tokens or history.shape[-1] != hidden:
        raise ValueError("Attention Residual prefix/history shapes disagree")
    blocks = history.shape[1]
    if blocks == 0 or tokens == 0:
        return prefix
    if tokens % 8:
        raise ValueError("Fused Attention Residual requires 8-token tiles")
    tile_t = 8
    step_h = next(s for s in (256, 128) if hidden % s == 0)
    n_steps = hidden // step_h
    unroll = next(u for u in (2, 1) if n_steps % u == 0)
    if block_inv_rms is not None:
        inv = block_inv_rms.astype(jnp.float32)
        if inv.ndim == 2 and inv.shape == (blocks, tokens) and blocks != tokens:
            inv = jnp.transpose(inv, (1, 0))
        else:
            inv = inv.reshape(tokens, blocks)
        inv_3d = jnp.broadcast_to(inv[:, :, None], (tokens, blocks, 128))
        return pl.pallas_call(
            functools.partial(
                _mix_kernel_3d_inv,
                eps=eps,
                blocks=blocks,
                tile_t=tile_t,
                hidden=hidden,
                step_h=step_h,
                unroll=unroll,
            ),
            out_shape=jax.ShapeDtypeStruct(prefix.shape, prefix.dtype),
            grid=(tokens // tile_t,),
            in_specs=(
                pl.BlockSpec((tile_t, hidden), lambda i: (i, 0)),
                pl.BlockSpec((tile_t, blocks, hidden), lambda i: (i, 0, 0)),
                pl.BlockSpec((hidden,), lambda i: (0,)),
                pl.BlockSpec((tile_t, blocks, 128), lambda i: (i, 0, 0)),
            ),
            out_specs=pl.BlockSpec((tile_t, hidden), lambda i: (i, 0)),
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=63 * 1024**2),
            interpret=interpret,
            name="kimi_attnres_compact_m16",
        )(prefix, history, folded_weight, inv_3d)

    return pl.pallas_call(
        functools.partial(
            _mix_kernel_3d,
            eps=eps,
            blocks=blocks,
            tile_t=tile_t,
            hidden=hidden,
            step_h=step_h,
            unroll=unroll,
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
