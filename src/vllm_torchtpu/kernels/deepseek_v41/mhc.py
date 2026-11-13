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
"""DeepSeek-V4.1 Single-Pass Delayed mHC Pallas kernels."""

import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.mhc import utils
from vllm_torchtpu.kernels.deepseek_v4.rope import _lane_gather


def _packed_mixes(x: jax.Array, fn_72_ref, mix_hc: int) -> jax.Array:
    """Single-pass MXU projection on the [3 * mix_hc, hc_dim] bf16 split of hc_fn."""
    parts = jax.lax.dot_general(
        x,
        fn_72_ref[...],
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    return (
        parts[:, :mix_hc]
        + parts[:, mix_hc : 2 * mix_hc]
        + parts[:, 2 * mix_hc : 3 * mix_hc]
    )


def _rms_norm_to_ref(collapsed_bf16, nw_ref, normed_ref, rms_eps):
    cf32 = collapsed_bf16.astype(jnp.float32)
    rstd = jax.lax.rsqrt(jnp.mean(cf32 * cf32, axis=-1, keepdims=True) + rms_eps)
    normed_ref[...] = (cf32 * rstd).astype(jnp.bfloat16) * nw_ref[...]


def _v41_in_kernel_gates(
    acc: jax.Array,
    sqr: jax.Array,
    scale_ref,
    hb_ref,
    post_out_ref,
    comb_out_ref,
    pre_out_ref,
    *,
    hc_mult: int,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> None:
    """In-kernel mhc_pre_gates (sigmoid + softmax + Sinkhorn-Knopp) for hc_mult=4."""
    mixes = acc * jax.lax.rsqrt(sqr / (hc_mult * hidden_size) + rms_eps)
    logits = mixes * scale_ref[...] + hb_ref[...]

    pre_logits = logits[:, :4]
    pre_out_ref[...] = jax.nn.sigmoid(pre_logits) + hc_pre_eps

    idx4 = jnp.arange(4, dtype=jnp.int32)
    post_logits = _lane_gather(logits, idx4 + 4)
    post_out_ref[...] = jax.nn.sigmoid(post_logits) * hc_post_mult_value

    idx16 = jnp.arange(16, dtype=jnp.int32)
    if acc.shape[0] <= 16:
        comb = _lane_gather(logits, idx16 + 8)

        row_swap1 = idx16 ^ 1
        row_swap2 = idx16 ^ 2
        col_swap1 = idx16 ^ 4
        col_swap2 = idx16 ^ 8

        m1 = jnp.maximum(comb, _lane_gather(comb, row_swap1))
        row_max = jnp.maximum(m1, _lane_gather(m1, row_swap2))
        exp_x = jnp.exp(comb - row_max)
        s1 = exp_x + _lane_gather(exp_x, row_swap1)
        row_sum = s1 + _lane_gather(s1, row_swap2)
        comb_mix = exp_x / row_sum + hc_sinkhorn_eps

        def _row_sum(v: jax.Array) -> jax.Array:
            r1 = v + _lane_gather(v, row_swap1)
            return r1 + _lane_gather(r1, row_swap2)

        def _col_sum(v: jax.Array) -> jax.Array:
            c1 = v + _lane_gather(v, col_swap1)
            return c1 + _lane_gather(c1, col_swap2)

        comb_mix = comb_mix * jax.lax.reciprocal(_col_sum(comb_mix) + hc_sinkhorn_eps)
        for _ in range(sinkhorn_repeat - 1):
            comb_mix = comb_mix * jax.lax.reciprocal(
                _row_sum(comb_mix) + hc_sinkhorn_eps
            )
            comb_mix = comb_mix * jax.lax.reciprocal(
                _col_sum(comb_mix) + hc_sinkhorn_eps
            )

        comb_out_ref[...] = comb_mix
        return

    idx32 = jnp.arange(32, dtype=jnp.int32)
    perm_in = 8 + (idx32 & 3) * 4 + (idx32 >> 3)
    t = jnp.transpose(_lane_gather(logits, perm_in))
    mask4 = (jnp.arange(8, dtype=jnp.int32)[:, None] < 4).astype(jnp.float32)

    c0 = t[0:8, :]
    c1 = t[8:16, :]
    c2 = t[16:24, :]
    c3 = t[24:32, :]

    m = jnp.maximum(jnp.maximum(c0, c1), jnp.maximum(c2, c3))
    e0 = jnp.exp(c0 - m) * mask4
    e1 = jnp.exp(c1 - m) * mask4
    e2 = jnp.exp(c2 - m) * mask4
    e3 = jnp.exp(c3 - m) * mask4
    rs = (e0 + e1) + (e2 + e3)
    inv_rs = jnp.where(mask4 > 0, jax.lax.reciprocal(rs), 0.0)
    eps_v = hc_sinkhorn_eps * mask4
    c0 = e0 * inv_rs + eps_v
    c1 = e1 * inv_rs + eps_v
    c2 = e2 * inv_rs + eps_v
    c3 = e3 * inv_rs + eps_v

    c0 = c0 * jax.lax.reciprocal(jnp.sum(c0, axis=0, keepdims=True) + hc_sinkhorn_eps)
    c1 = c1 * jax.lax.reciprocal(jnp.sum(c1, axis=0, keepdims=True) + hc_sinkhorn_eps)
    c2 = c2 * jax.lax.reciprocal(jnp.sum(c2, axis=0, keepdims=True) + hc_sinkhorn_eps)
    c3 = c3 * jax.lax.reciprocal(jnp.sum(c3, axis=0, keepdims=True) + hc_sinkhorn_eps)

    for _ in range(sinkhorn_repeat - 1):
        inv_rs = jax.lax.reciprocal(((c0 + c1) + (c2 + c3)) + hc_sinkhorn_eps)
        c0 = c0 * inv_rs
        c1 = c1 * inv_rs
        c2 = c2 * inv_rs
        c3 = c3 * inv_rs

        c0 = c0 * jax.lax.reciprocal(
            jnp.sum(c0, axis=0, keepdims=True) + hc_sinkhorn_eps
        )
        c1 = c1 * jax.lax.reciprocal(
            jnp.sum(c1, axis=0, keepdims=True) + hc_sinkhorn_eps
        )
        c2 = c2 * jax.lax.reciprocal(
            jnp.sum(c2, axis=0, keepdims=True) + hc_sinkhorn_eps
        )
        c3 = c3 * jax.lax.reciprocal(
            jnp.sum(c3, axis=0, keepdims=True) + hc_sinkhorn_eps
        )

    out32 = jnp.transpose(jnp.concatenate([c0, c1, c2, c3], axis=0))
    perm_out = (idx16 & 3) * 8 + (idx16 >> 2)
    comb_out_ref[...] = _lane_gather(out32, perm_out)


def _v41_pre_kernel(
    res_ref,
    pre_in_ref,
    nw_ref,
    fn_72_ref,
    scale_ref,
    hb_ref,
    post_out_ref,
    comb_out_ref,
    normed_ref,
    pre_out_ref,
    *,
    hc_mult: int,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    use_stream0: bool,
):
    """mhc_pre: 1-pass packed mix GEMM + strip-tiled sqrsum + in-kernel Sinkhorn gates + RMSNorm."""
    mix_hc = hc_mult * (hc_mult + 2)
    x = res_ref[...]
    acc = _packed_mixes(x, fn_72_ref, mix_hc)
    pre_in = None if use_stream0 else pre_in_ref[...]
    dh = 1280 if hidden_size % 1280 == 0 else hidden_size
    num_strips = hidden_size // dh
    sqr = None
    for s in range(num_strips):
        s_lo = s * dh
        s_hi = s_lo + dh
        col_strip = None
        for j in range(hc_mult):
            lo = j * hidden_size + s_lo
            xj_f32 = x[:, lo : lo + dh].astype(jnp.float32)
            sj = jnp.sum(xj_f32 * xj_f32, axis=-1, keepdims=True)
            sqr = sj if sqr is None else sqr + sj
            if not use_stream0:
                assert pre_in is not None
                term = pre_in[:, j : j + 1] * xj_f32
                col_strip = term if col_strip is None else col_strip + term
        if use_stream0:
            normed_ref[:, s_lo:s_hi] = x[:, s_lo:s_hi]
        else:
            assert col_strip is not None
            normed_ref[:, s_lo:s_hi] = col_strip.astype(jnp.bfloat16)

    assert sqr is not None
    _v41_in_kernel_gates(
        acc,
        sqr,
        scale_ref,
        hb_ref,
        post_out_ref,
        comb_out_ref,
        pre_out_ref,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )
    _rms_norm_to_ref(normed_ref[...], nw_ref, normed_ref, rms_eps)


def _v41_pre_mixes_kernel(
    res_ref,
    pre_in_ref,
    nw_ref,
    fn_72_ref,
    mixes_ref,
    sqr_ref,
    normed_ref,
    *,
    hc_mult: int,
    hidden_size: int,
    rms_eps: float,
    use_stream0: bool,
):
    """mhc_pre (multi-block): 1-pass packed mix GEMM + strip-tiled sqrsum + RMSNorm."""
    mix_hc = hc_mult * (hc_mult + 2)
    x = res_ref[...]
    mixes_ref[...] = _packed_mixes(x, fn_72_ref, mix_hc)
    pre_in = None if use_stream0 else pre_in_ref[...]
    dh = 1280 if hidden_size % 1280 == 0 else hidden_size
    num_strips = hidden_size // dh
    sqr = None
    for s in range(num_strips):
        s_lo = s * dh
        s_hi = s_lo + dh
        col_strip = None
        for j in range(hc_mult):
            lo = j * hidden_size + s_lo
            xj_f32 = x[:, lo : lo + dh].astype(jnp.float32)
            sj = jnp.sum(xj_f32 * xj_f32, axis=-1, keepdims=True)
            sqr = sj if sqr is None else sqr + sj
            if not use_stream0:
                assert pre_in is not None
                term = pre_in[:, j : j + 1] * xj_f32
                col_strip = term if col_strip is None else col_strip + term
        if use_stream0:
            normed_ref[:, s_lo:s_hi] = x[:, s_lo:s_hi]
        else:
            assert col_strip is not None
            normed_ref[:, s_lo:s_hi] = col_strip.astype(jnp.bfloat16)

    assert sqr is not None
    sqr_ref[...] = sqr
    _rms_norm_to_ref(normed_ref[...], nw_ref, normed_ref, rms_eps)


def _v41_fused_post_pre_1strip_body(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    pre_in_ref,
    nw_ref,
    fn_72_ref,
    scale_ref,
    hb_ref,
    newres_ref,
    post_out_ref,
    comb_out_ref,
    normed_ref,
    pre_out_ref,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    write_bf16: bool = True,
):
    """Seam kernel (single-block): 1-pass packed GEMM + in-kernel Sinkhorn."""
    post = post_ref[...]
    comb = comb_ref[...]
    pre_in = pre_in_ref[...]
    hidden_size = x_ref.shape[1]
    hc_mult = post.shape[1]
    mix_hc = hc_mult * (hc_mult + 2)
    x_bf = x_ref[...]
    old_bf = [
        res_ref[:, i * hidden_size : (i + 1) * hidden_size] for i in range(hc_mult)
    ]

    streams_bf = []
    collapsed_f32 = None
    sqr = None
    strip_exact = x_ref.shape[0] > 16 and hidden_size % 1280 == 0
    dh = 1280 if strip_exact else hidden_size
    num_strips = hidden_size // dh
    nfs = []
    for j in range(hc_mult):
        acc = post[:, j : j + 1] * x_bf.astype(jnp.float32)
        for i in range(hc_mult):
            k = i * hc_mult + j
            acc = acc + comb[:, k : k + 1] * old_bf[i].astype(jnp.float32)
        new_bf = acc.astype(jnp.bfloat16)
        newres_ref[:, j * hidden_size : (j + 1) * hidden_size] = new_bf
        streams_bf.append(new_bf)
        nf = new_bf.astype(jnp.float32)
        nfs.append(nf)
        if not strip_exact:
            sj = jnp.sum(nf * nf, axis=-1, keepdims=True)
            sqr = sj if sqr is None else sqr + sj
        term = pre_in[:, j : j + 1] * nf
        collapsed_f32 = term if collapsed_f32 is None else collapsed_f32 + term

    if strip_exact:
        for s in range(num_strips):
            s_lo, s_hi = s * dh, (s + 1) * dh
            for j in range(hc_mult):
                sj = jnp.sum(
                    nfs[j][:, s_lo:s_hi] * nfs[j][:, s_lo:s_hi],
                    axis=-1,
                    keepdims=True,
                )
                sqr = sj if sqr is None else sqr + sj

    g_bf = jnp.concatenate(streams_bf, axis=1)
    acc = _packed_mixes(
        newres_ref[...] if x_ref.shape[0] > 32 else g_bf, fn_72_ref, mix_hc
    )
    assert sqr is not None
    _v41_in_kernel_gates(
        acc,
        sqr,
        scale_ref,
        hb_ref,
        post_out_ref,
        comb_out_ref,
        pre_out_ref,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )
    assert collapsed_f32 is not None
    cf32 = collapsed_f32.astype(jnp.bfloat16).astype(jnp.float32)
    if strip_exact:
        ns = None
        for s in range(num_strips):
            p = jnp.sum(
                cf32[:, s * dh : (s + 1) * dh] ** 2, axis=-1, keepdims=True
            )
            ns = p if ns is None else ns + p
        rstd = jax.lax.rsqrt(ns / hidden_size + rms_eps)
    else:
        rstd = jax.lax.rsqrt(jnp.mean(cf32 * cf32, axis=-1, keepdims=True) + rms_eps)
    val_bf16 = (cf32 * rstd).astype(jnp.bfloat16) * nw_ref[...]
    if write_bf16:
        normed_ref[...] = val_bf16
        return None
    return val_bf16


def _v41_fused_post_pre_1strip_kernel(*args, **kwargs):
    _v41_fused_post_pre_1strip_body(*args, **kwargs)


def _v41_fused_post_pre_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    pre_in_ref,
    nw_ref,
    fn_72_ref,
    scale_ref,
    hb_ref,
    newres_ref,
    post_out_ref,
    comb_out_ref,
    normed_ref,
    pre_out_ref,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
):
    """Seam kernel: strip-tiled recombine + overlapped packed GEMM + in-kernel Sinkhorn."""
    post = post_ref[...]
    comb = comb_ref[...]
    pre_in = pre_in_ref[...]
    hidden_size = x_ref.shape[1]
    hc_mult = post.shape[1]
    mix_hc = hc_mult * (hc_mult + 2)
    dh = 1280 if hidden_size % 1280 == 0 else hidden_size
    num_strips = hidden_size // dh

    sqr_acc = None
    norm_sqr_acc = None
    for s in range(num_strips):
        s_lo = s * dh
        s_hi = s_lo + dh
        x_f32 = x_ref[:, s_lo:s_hi].astype(jnp.float32)
        accs = [post[:, j : j + 1] * x_f32 for j in range(hc_mult)]
        for i in range(hc_mult):
            old_i = res_ref[
                :, i * hidden_size + s_lo : i * hidden_size + s_hi
            ].astype(jnp.float32)
            for j in range(hc_mult):
                k = i * hc_mult + j
                accs[j] = accs[j] + comb[:, k : k + 1] * old_i
        col_strip = None
        for j in range(hc_mult):
            lo = j * hidden_size + s_lo
            hi = lo + dh
            new_bf = accs[j].astype(jnp.bfloat16)
            newres_ref[:, lo:hi] = new_bf
            nf = new_bf.astype(jnp.float32)
            sj = jnp.sum(nf * nf, axis=-1, keepdims=True)
            sqr_acc = sj if sqr_acc is None else sqr_acc + sj
            term = pre_in[:, j : j + 1] * nf
            col_strip = term if col_strip is None else col_strip + term
        assert col_strip is not None
        col_bf = col_strip.astype(jnp.bfloat16)
        normed_ref[:, s_lo:s_hi] = col_bf
        cf32 = col_bf.astype(jnp.float32)
        ns = jnp.sum(cf32 * cf32, axis=-1, keepdims=True)
        norm_sqr_acc = ns if norm_sqr_acc is None else norm_sqr_acc + ns

    assert sqr_acc is not None and norm_sqr_acc is not None
    acc = _packed_mixes(newres_ref[...], fn_72_ref, mix_hc)
    _v41_in_kernel_gates(
        acc,
        sqr_acc,
        scale_ref,
        hb_ref,
        post_out_ref,
        comb_out_ref,
        pre_out_ref,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )
    rstd = jax.lax.rsqrt(norm_sqr_acc / hidden_size + rms_eps)
    for s in range(num_strips):
        s_lo = s * dh
        s_hi = s_lo + dh
        cf32_s = normed_ref[:, s_lo:s_hi].astype(jnp.float32)
        normed_ref[:, s_lo:s_hi] = (
            cf32_s * rstd
        ).astype(jnp.bfloat16) * nw_ref[:, s_lo:s_hi]


def _v41_fused_post_pre_mixes_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    pre_in_ref,
    nw_ref,
    fn_72_ref,
    newres_ref,
    mixes_ref,
    sqr_ref,
    normed_ref,
    *,
    rms_eps: float,
):
    """Seam kernel (multi-block): strip-tiled recombine + overlapped packed mix GEMM + RMSNorm."""
    post = post_ref[...]
    comb = comb_ref[...]
    pre_in = pre_in_ref[...]
    hidden_size = x_ref.shape[1]
    hc_mult = post.shape[1]
    mix_hc = hc_mult * (hc_mult + 2)
    dh = 1280 if hidden_size % 1280 == 0 else hidden_size
    num_strips = hidden_size // dh

    sqr_acc = None
    norm_sqr_acc = None
    for s in range(num_strips):
        s_lo = s * dh
        s_hi = s_lo + dh
        x_f32 = x_ref[:, s_lo:s_hi].astype(jnp.float32)
        accs = [post[:, j : j + 1] * x_f32 for j in range(hc_mult)]
        for i in range(hc_mult):
            old_i = res_ref[
                :, i * hidden_size + s_lo : i * hidden_size + s_hi
            ].astype(jnp.float32)
            for j in range(hc_mult):
                k = i * hc_mult + j
                accs[j] = accs[j] + comb[:, k : k + 1] * old_i
        col_strip = None
        for j in range(hc_mult):
            lo = j * hidden_size + s_lo
            hi = lo + dh
            new_bf = accs[j].astype(jnp.bfloat16)
            newres_ref[:, lo:hi] = new_bf
            nf = new_bf.astype(jnp.float32)
            sj = jnp.sum(nf * nf, axis=-1, keepdims=True)
            sqr_acc = sj if sqr_acc is None else sqr_acc + sj
            term = pre_in[:, j : j + 1] * nf
            col_strip = term if col_strip is None else col_strip + term
        assert col_strip is not None
        col_bf = col_strip.astype(jnp.bfloat16)
        normed_ref[:, s_lo:s_hi] = col_bf
        cf32 = col_bf.astype(jnp.float32)
        ns = jnp.sum(cf32 * cf32, axis=-1, keepdims=True)
        norm_sqr_acc = ns if norm_sqr_acc is None else norm_sqr_acc + ns

    assert sqr_acc is not None and norm_sqr_acc is not None
    mixes_ref[...] = _packed_mixes(newres_ref[...], fn_72_ref, mix_hc)
    sqr_ref[...] = sqr_acc
    rstd = jax.lax.rsqrt(norm_sqr_acc / hidden_size + rms_eps)
    for s in range(num_strips):
        s_lo = s * dh
        s_hi = s_lo + dh
        cf32_s = normed_ref[:, s_lo:s_hi].astype(jnp.float32)
        normed_ref[:, s_lo:s_hi] = (
            cf32_s * rstd
        ).astype(jnp.bfloat16) * nw_ref[:, s_lo:s_hi]


def _v41_mhc_pallas_call(
    kernel,
    token_inputs: tuple[jax.Array, ...],
    norm_weight: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array | None,
    hc_base: jax.Array | None,
    hc_mult: int,
    out_widths_dtypes: tuple[tuple[int, Any], ...],
    *,
    max_token_block: int = 128,
) -> tuple[jax.Array, ...]:
    """Run a V4.1 mHC kernel over token blocks with resident weights in VMEM."""
    num_tokens = token_inputs[0].shape[0]
    has_fp8_out = any(dt == jnp.float8_e4m3fn for _, dt in out_widths_dtypes)
    sublane = 32 if has_fp8_out else utils.SUBLANE
    if (not has_fp8_out and num_tokens <= 64) or (16 <= num_tokens <= 64):
        tb, padded_tokens = utils.select_token_block(num_tokens, 64)
    else:
        padded_tokens = ((num_tokens + sublane - 1) // sublane) * sublane
        tb = min(max_token_block, padded_tokens)
        while tb > sublane and padded_tokens % tb != 0:
            tb -= sublane
    token_inputs = utils.pad_to(padded_tokens, *token_inputs)
    nw_2d = norm_weight.reshape(1, -1).astype(jnp.bfloat16)

    mix_hc = hc_mult * (hc_mult + 2)
    if hc_fn.ndim == 3 and hc_fn.dtype == jnp.bfloat16:
        hc_dim = hc_fn.shape[2]
        fn_72 = hc_fn.reshape(3 * mix_hc, hc_dim)
    elif (
        hc_fn.ndim == 2 and hc_fn.dtype == jnp.bfloat16 and hc_fn.shape[0] == 3 * mix_hc
    ):
        hc_dim = hc_fn.shape[1]
        fn_72 = hc_fn
    else:
        hc_dim = hc_fn.shape[1]
        fn_72 = jnp.concatenate(utils.split_fn3(hc_fn), axis=0)
    fn_spec = pl.BlockSpec((3 * mix_hc, hc_dim), lambda i: (0, 0))

    extra_inputs: tuple[jax.Array, ...] = ()
    extra_specs: tuple[pl.BlockSpec, ...] = ()
    if hc_scale is not None and hc_base is not None:
        if hc_scale.size == mix_hc:
            scale_vec = hc_scale.reshape(1, mix_hc).astype(jnp.float32)
        else:
            scale_vec = (
                jnp.repeat(
                    hc_scale.reshape(3),
                    jnp.array([hc_mult, hc_mult, hc_mult * hc_mult]),
                    total_repeat_length=mix_hc,
                )
                .reshape(1, mix_hc)
                .astype(jnp.float32)
            )
        hb_vec = hc_base.reshape(1, mix_hc).astype(jnp.float32)
        extra_inputs = (scale_vec, hb_vec)
        extra_specs = (
            pl.BlockSpec((1, mix_hc), lambda i: (0, 0)),
            pl.BlockSpec((1, mix_hc), lambda i: (0, 0)),
        )

    buf_mode = pl.Buffered(buffer_count=1 if (padded_tokens == tb and padded_tokens == num_tokens) else 2)
    outs = pl.pallas_call(
        kernel,
        grid=(padded_tokens // tb,),
        in_specs=[
            *(
                pl.BlockSpec((tb, a.shape[1]), lambda i: (i, 0), pipeline_mode=buf_mode)
                for a in token_inputs
            ),
            pl.BlockSpec(nw_2d.shape, lambda i: (0, 0)),
            fn_spec,
            *extra_specs,
        ],
        out_specs=[
            pl.BlockSpec((tb, w), lambda i: (i, 0), pipeline_mode=buf_mode)
            for w, _ in out_widths_dtypes
        ],
        out_shape=[
            jax.ShapeDtypeStruct((padded_tokens, w), dt) for w, dt in out_widths_dtypes
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=utils.DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
    )(*token_inputs, nw_2d, fn_72, *extra_inputs)
    return utils.trim_to(num_tokens, *outs)


def mhc_pre_v41(
    residual: jax.Array,
    pre_mix_in: jax.Array,
    norm_weight: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    use_stream0: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    *outer_shape, hc_mult, hidden_size = residual.shape
    res2d = residual.reshape(-1, hc_mult * hidden_size)
    num_tokens = res2d.shape[0]
    if num_tokens <= 256:
        post_mix, comb_mix, normed, next_pre_mix = _v41_mhc_pallas_call(
            functools.partial(
                _v41_pre_kernel,
                hc_mult=hc_mult,
                hidden_size=hidden_size,
                rms_eps=rms_eps,
                hc_pre_eps=hc_pre_eps,
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
                use_stream0=use_stream0,
            ),
            (res2d, pre_mix_in.reshape(num_tokens, hc_mult)),
            norm_weight,
            hc_fn,
            hc_scale,
            hc_base,
            hc_mult,
            (
                (hc_mult, jnp.float32),
                (hc_mult * hc_mult, jnp.float32),
                (hidden_size, jnp.bfloat16),
                (hc_mult, jnp.float32),
            ),
            max_token_block=128,
        )
    else:
        mix_hc = hc_mult * (hc_mult + 2)
        mixes, sqr, normed = _v41_mhc_pallas_call(
            functools.partial(
                _v41_pre_mixes_kernel,
                hc_mult=hc_mult,
                hidden_size=hidden_size,
                rms_eps=rms_eps,
                use_stream0=use_stream0,
            ),
            (res2d, pre_mix_in.reshape(num_tokens, hc_mult)),
            norm_weight,
            hc_fn,
            None,
            None,
            hc_mult,
            (
                (mix_hc, jnp.float32),
                (1, jnp.float32),
                (hidden_size, jnp.bfloat16),
            ),
            max_token_block=128,
        )
        next_pre_mix, post_mix, comb_mix = utils.mhc_pre_gates(
            mixes,
            sqr,
            hc_mult,
            hidden_size,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
    return (
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
        normed.reshape(*outer_shape, hidden_size),
        next_pre_mix.reshape(*outer_shape, hc_mult),
    )


def mhc_fused_post_pre_v41(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    pre_mix_in: jax.Array,
    norm_weight: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    *outer_shape, hc_mult, hidden_size = residual.shape
    num_tokens = x.size // hidden_size
    if num_tokens <= 256:
        seam_fn = (
            _v41_fused_post_pre_1strip_kernel
            if num_tokens <= 192
            else _v41_fused_post_pre_kernel
        )
        new_res2d, post_mix, comb_mix, normed, next_pre_mix = _v41_mhc_pallas_call(
            functools.partial(
                seam_fn,
                rms_eps=rms_eps,
                hc_pre_eps=hc_pre_eps,
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
            ),
            (
                x.reshape(num_tokens, hidden_size).astype(jnp.bfloat16),
                residual.reshape(num_tokens, hc_mult * hidden_size),
                post_layer_mix.reshape(num_tokens, hc_mult),
                comb_res_mix.reshape(num_tokens, hc_mult * hc_mult),
                pre_mix_in.reshape(num_tokens, hc_mult),
            ),
            norm_weight,
            hc_fn,
            hc_scale,
            hc_base,
            hc_mult,
            (
                (hc_mult * hidden_size, jnp.bfloat16),
                (hc_mult, jnp.float32),
                (hc_mult * hc_mult, jnp.float32),
                (hidden_size, jnp.bfloat16),
                (hc_mult, jnp.float32),
            ),
            max_token_block=192 if num_tokens <= 192 else 128,
        )
    else:
        mix_hc = hc_mult * (hc_mult + 2)
        new_res2d, mixes, sqr, normed = _v41_mhc_pallas_call(
            functools.partial(
                _v41_fused_post_pre_mixes_kernel,
                rms_eps=rms_eps,
            ),
            (
                x.reshape(num_tokens, hidden_size).astype(jnp.bfloat16),
                residual.reshape(num_tokens, hc_mult * hidden_size),
                post_layer_mix.reshape(num_tokens, hc_mult),
                comb_res_mix.reshape(num_tokens, hc_mult * hc_mult),
                pre_mix_in.reshape(num_tokens, hc_mult),
            ),
            norm_weight,
            hc_fn,
            None,
            None,
            hc_mult,
            (
                (hc_mult * hidden_size, jnp.bfloat16),
                (mix_hc, jnp.float32),
                (1, jnp.float32),
                (hidden_size, jnp.bfloat16),
            ),
            max_token_block=128,
        )
        next_pre_mix, post_mix, comb_mix = utils.mhc_pre_gates(
            mixes,
            sqr,
            hc_mult,
            hidden_size,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
    return (
        new_res2d.reshape(residual.shape),
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
        normed.reshape(*outer_shape, hidden_size),
        next_pre_mix.reshape(*outer_shape, hc_mult),
    )

_FP8_E4M3_MAX = float(jnp.finfo(jnp.float8_e4m3fn).max)


def _rms_norm_quant_to_refs(
    collapsed_bf16, nw_ref, normed_ref, normed_fp8_ref, normed_scale_ref, rms_eps
):
    cf32 = collapsed_bf16.astype(jnp.float32)
    rstd = jax.lax.rsqrt(jnp.mean(cf32 * cf32, axis=-1, keepdims=True) + rms_eps)
    val_bf16 = (cf32 * rstd).astype(jnp.bfloat16) * nw_ref[...]
    normed_ref[...] = val_bf16
    val_f32 = val_bf16.astype(jnp.float32)
    abs_max = jnp.max(jnp.abs(val_f32), axis=-1, keepdims=True)
    scale = abs_max / _FP8_E4M3_MAX
    scale = jnp.where(scale == 0, 1.0, scale)
    normed_scale_ref[...] = scale
    inv_scale = 1.0 / scale
    normed_fp8_ref[...] = jnp.clip(
        val_f32 * inv_scale, -_FP8_E4M3_MAX, _FP8_E4M3_MAX
    ).astype(jnp.float8_e4m3fn)


def _v41_pre_quant_kernel(
    res_ref,
    pre_in_ref,
    nw_ref,
    fn_72_ref,
    scale_ref,
    hb_ref,
    post_out_ref,
    comb_out_ref,
    normed_ref,
    pre_out_ref,
    normed_fp8_ref,
    normed_scale_ref,
    *,
    hc_mult: int,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    use_stream0: bool,
):
    """mhc_pre + strip-tiled sqrsum + in-VMEM dynamic FP8 activation quantization."""
    mix_hc = hc_mult * (hc_mult + 2)
    x = res_ref[...]
    acc = _packed_mixes(x, fn_72_ref, mix_hc)
    pre_in = None if use_stream0 else pre_in_ref[...]
    dh = 1280 if hidden_size % 1280 == 0 else hidden_size
    num_strips = hidden_size // dh
    sqr = None
    for s in range(num_strips):
        s_lo = s * dh
        s_hi = s_lo + dh
        col_strip = None
        for j in range(hc_mult):
            lo = j * hidden_size + s_lo
            xj_f32 = x[:, lo : lo + dh].astype(jnp.float32)
            sj = jnp.sum(xj_f32 * xj_f32, axis=-1, keepdims=True)
            sqr = sj if sqr is None else sqr + sj
            if not use_stream0:
                assert pre_in is not None
                term = pre_in[:, j : j + 1] * xj_f32
                col_strip = term if col_strip is None else col_strip + term
        if use_stream0:
            normed_ref[:, s_lo:s_hi] = x[:, s_lo:s_hi]
        else:
            assert col_strip is not None
            normed_ref[:, s_lo:s_hi] = col_strip.astype(jnp.bfloat16)

    assert sqr is not None
    _v41_in_kernel_gates(
        acc,
        sqr,
        scale_ref,
        hb_ref,
        post_out_ref,
        comb_out_ref,
        pre_out_ref,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )
    _rms_norm_quant_to_refs(
        normed_ref[...], nw_ref, normed_ref, normed_fp8_ref, normed_scale_ref, rms_eps
    )


def _v41_pre_quant_mixes_kernel(
    res_ref,
    pre_in_ref,
    nw_ref,
    fn_72_ref,
    mixes_ref,
    sqr_ref,
    normed_ref,
    normed_fp8_ref,
    normed_scale_ref,
    *,
    hc_mult: int,
    hidden_size: int,
    rms_eps: float,
    use_stream0: bool,
):
    """mhc_pre (multi-block) + strip-tiled sqrsum + in-VMEM dynamic FP8 activation quantization."""
    mix_hc = hc_mult * (hc_mult + 2)
    x = res_ref[...]
    mixes_ref[...] = _packed_mixes(x, fn_72_ref, mix_hc)
    pre_in = None if use_stream0 else pre_in_ref[...]
    dh = 1280 if hidden_size % 1280 == 0 else hidden_size
    num_strips = hidden_size // dh
    sqr = None
    for s in range(num_strips):
        s_lo = s * dh
        s_hi = s_lo + dh
        col_strip = None
        for j in range(hc_mult):
            lo = j * hidden_size + s_lo
            xj_f32 = x[:, lo : lo + dh].astype(jnp.float32)
            sj = jnp.sum(xj_f32 * xj_f32, axis=-1, keepdims=True)
            sqr = sj if sqr is None else sqr + sj
            if not use_stream0:
                assert pre_in is not None
                term = pre_in[:, j : j + 1] * xj_f32
                col_strip = term if col_strip is None else col_strip + term
        if use_stream0:
            normed_ref[:, s_lo:s_hi] = x[:, s_lo:s_hi]
        else:
            assert col_strip is not None
            normed_ref[:, s_lo:s_hi] = col_strip.astype(jnp.bfloat16)

    assert sqr is not None
    sqr_ref[...] = sqr
    _rms_norm_quant_to_refs(
        normed_ref[...], nw_ref, normed_ref, normed_fp8_ref, normed_scale_ref, rms_eps
    )


def _v41_fused_post_pre_quant_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    pre_in_ref,
    nw_ref,
    fn_72_ref,
    scale_ref,
    hb_ref,
    newres_ref,
    post_out_ref,
    comb_out_ref,
    normed_ref,
    pre_out_ref,
    normed_fp8_ref,
    normed_scale_ref,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    write_bf16: bool = True,
):
    """Seam kernel + in-VMEM dynamic FP8 activation quantization."""
    post = post_ref[...]
    comb = comb_ref[...]
    pre_in = pre_in_ref[...]
    hidden_size = x_ref.shape[1]
    hc_mult = post.shape[1]
    mix_hc = hc_mult * (hc_mult + 2)
    dh = 1280 if hidden_size % 1280 == 0 else hidden_size
    num_strips = hidden_size // dh

    sqr_acc = None
    norm_sqr_acc = None
    stage_vmem = write_bf16 and x_ref.shape[0] > 16
    cf32_strips = []
    for s in range(num_strips):
        s_lo = s * dh
        s_hi = s_lo + dh
        x_f32 = x_ref[:, s_lo:s_hi].astype(jnp.float32)
        accs = [post[:, j : j + 1] * x_f32 for j in range(hc_mult)]
        for i in range(hc_mult):
            old_i = res_ref[
                :, i * hidden_size + s_lo : i * hidden_size + s_hi
            ].astype(jnp.float32)
            for j in range(hc_mult):
                k = i * hc_mult + j
                accs[j] = accs[j] + comb[:, k : k + 1] * old_i
        col_strip = None
        for j in range(hc_mult):
            lo = j * hidden_size + s_lo
            hi = lo + dh
            new_bf = accs[j].astype(jnp.bfloat16)
            newres_ref[:, lo:hi] = new_bf
            nf = new_bf.astype(jnp.float32)
            sj = jnp.sum(nf * nf, axis=-1, keepdims=True)
            sqr_acc = sj if sqr_acc is None else sqr_acc + sj
            term = pre_in[:, j : j + 1] * nf
            col_strip = term if col_strip is None else col_strip + term
        assert col_strip is not None
        col_bf = col_strip.astype(jnp.bfloat16)
        if stage_vmem:
            normed_ref[:, s_lo:s_hi] = col_bf
        cf32 = col_bf.astype(jnp.float32)
        if not stage_vmem:
            cf32_strips.append(cf32)
        ns = jnp.sum(cf32 * cf32, axis=-1, keepdims=True)
        norm_sqr_acc = ns if norm_sqr_acc is None else norm_sqr_acc + ns

    assert sqr_acc is not None and norm_sqr_acc is not None
    acc = _packed_mixes(newres_ref[...], fn_72_ref, mix_hc)
    _v41_in_kernel_gates(
        acc,
        sqr_acc,
        scale_ref,
        hb_ref,
        post_out_ref,
        comb_out_ref,
        pre_out_ref,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )
    rstd = jax.lax.rsqrt(norm_sqr_acc / hidden_size + rms_eps)
    val_strips = []
    abs_max_acc = None
    for s in range(num_strips):
        s_lo = s * dh
        s_hi = s_lo + dh
        cf32_s = (
            normed_ref[:, s_lo:s_hi].astype(jnp.float32)
            if stage_vmem
            else cf32_strips[s]
        )
        val_bf16 = (
            cf32_s * rstd
        ).astype(jnp.bfloat16) * nw_ref[:, s_lo:s_hi]
        if write_bf16:
            normed_ref[:, s_lo:s_hi] = val_bf16
        val_f32 = val_bf16.astype(jnp.float32)
        val_strips.append(val_f32)
        s_abs = jnp.max(jnp.abs(val_f32), axis=-1, keepdims=True)
        abs_max_acc = (
            s_abs if abs_max_acc is None else jnp.maximum(abs_max_acc, s_abs)
        )
    assert abs_max_acc is not None
    scale = abs_max_acc / _FP8_E4M3_MAX
    scale = jnp.where(scale == 0, 1.0, scale)
    normed_scale_ref[...] = scale
    inv_scale = 1.0 / scale
    for s in range(num_strips):
        s_lo = s * dh
        s_hi = s_lo + dh
        normed_fp8_ref[:, s_lo:s_hi] = jnp.clip(
            val_strips[s] * inv_scale, -_FP8_E4M3_MAX, _FP8_E4M3_MAX
        ).astype(jnp.float8_e4m3fn)


def _v41_fused_post_pre_quant_mixes_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    pre_in_ref,
    nw_ref,
    fn_72_ref,
    newres_ref,
    mixes_ref,
    sqr_ref,
    normed_ref,
    normed_fp8_ref,
    normed_scale_ref,
    *,
    rms_eps: float,
):
    """Seam kernel (multi-block) + in-VMEM dynamic FP8 activation quantization."""
    post = post_ref[...]
    comb = comb_ref[...]
    pre_in = pre_in_ref[...]
    hidden_size = x_ref.shape[1]
    hc_mult = post.shape[1]
    mix_hc = hc_mult * (hc_mult + 2)
    dh = 1280 if hidden_size % 1280 == 0 else hidden_size
    num_strips = hidden_size // dh
    write_bf16 = normed_ref.shape[1] == hidden_size

    sqr_acc = None
    norm_sqr_acc = None
    cf32_strips = []
    for s in range(num_strips):
        s_lo = s * dh
        s_hi = s_lo + dh
        x_f32 = x_ref[:, s_lo:s_hi].astype(jnp.float32)
        accs = [post[:, j : j + 1] * x_f32 for j in range(hc_mult)]
        for i in range(hc_mult):
            old_i = res_ref[
                :, i * hidden_size + s_lo : i * hidden_size + s_hi
            ].astype(jnp.float32)
            for j in range(hc_mult):
                k = i * hc_mult + j
                accs[j] = accs[j] + comb[:, k : k + 1] * old_i
        col_strip = None
        for j in range(hc_mult):
            lo = j * hidden_size + s_lo
            hi = lo + dh
            new_bf = accs[j].astype(jnp.bfloat16)
            newres_ref[:, lo:hi] = new_bf
            nf = new_bf.astype(jnp.float32)
            sj = jnp.sum(nf * nf, axis=-1, keepdims=True)
            sqr_acc = sj if sqr_acc is None else sqr_acc + sj
            term = pre_in[:, j : j + 1] * nf
            col_strip = term if col_strip is None else col_strip + term
        assert col_strip is not None
        col_bf = col_strip.astype(jnp.bfloat16)
        if write_bf16:
            normed_ref[:, s_lo:s_hi] = col_bf
        cf32 = col_bf.astype(jnp.float32)
        if not write_bf16:
            cf32_strips.append(cf32)
        ns = jnp.sum(cf32 * cf32, axis=-1, keepdims=True)
        norm_sqr_acc = ns if norm_sqr_acc is None else norm_sqr_acc + ns

    assert sqr_acc is not None and norm_sqr_acc is not None
    mixes_ref[...] = _packed_mixes(newres_ref[...], fn_72_ref, mix_hc)
    sqr_ref[...] = sqr_acc
    rstd = jax.lax.rsqrt(norm_sqr_acc / hidden_size + rms_eps)
    val_strips = []
    abs_max_acc = None
    for s in range(num_strips):
        s_lo = s * dh
        s_hi = s_lo + dh
        cf32_s = (
            normed_ref[:, s_lo:s_hi].astype(jnp.float32)
            if write_bf16
            else cf32_strips[s]
        )
        val_bf16 = (
            cf32_s * rstd
        ).astype(jnp.bfloat16) * nw_ref[:, s_lo:s_hi]
        if write_bf16:
            normed_ref[:, s_lo:s_hi] = val_bf16
        val_f32 = val_bf16.astype(jnp.float32)
        val_strips.append(val_f32)
        s_abs = jnp.max(jnp.abs(val_f32), axis=-1, keepdims=True)
        abs_max_acc = (
            s_abs if abs_max_acc is None else jnp.maximum(abs_max_acc, s_abs)
        )
    assert abs_max_acc is not None
    scale = abs_max_acc / _FP8_E4M3_MAX
    scale = jnp.where(scale == 0, 1.0, scale)
    normed_scale_ref[...] = scale
    inv_scale = 1.0 / scale
    for s in range(num_strips):
        s_lo = s * dh
        s_hi = s_lo + dh
        normed_fp8_ref[:, s_lo:s_hi] = jnp.clip(
            val_strips[s] * inv_scale, -_FP8_E4M3_MAX, _FP8_E4M3_MAX
        ).astype(jnp.float8_e4m3fn)


def mhc_pre_quant_v41(
    residual: jax.Array,
    pre_mix_in: jax.Array,
    norm_weight: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    use_stream0: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    *outer_shape, hc_mult, hidden_size = residual.shape
    res2d = residual.reshape(-1, hc_mult * hidden_size)
    num_tokens = res2d.shape[0]
    if num_tokens <= 256:
        (
            post_mix,
            comb_mix,
            normed,
            next_pre_mix,
            normed_fp8,
            normed_scale,
        ) = _v41_mhc_pallas_call(
            functools.partial(
                _v41_pre_quant_kernel,
                hc_mult=hc_mult,
                hidden_size=hidden_size,
                rms_eps=rms_eps,
                hc_pre_eps=hc_pre_eps,
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
                use_stream0=use_stream0,
            ),
            (res2d, pre_mix_in.reshape(num_tokens, hc_mult)),
            norm_weight,
            hc_fn,
            hc_scale,
            hc_base,
            hc_mult,
            (
                (hc_mult, jnp.float32),
                (hc_mult * hc_mult, jnp.float32),
                (hidden_size, jnp.bfloat16),
                (hc_mult, jnp.float32),
                (hidden_size, jnp.float8_e4m3fn),
                (1, jnp.float32),
            ),
            max_token_block=128,
        )
    else:
        mix_hc = hc_mult * (hc_mult + 2)
        mixes, sqr, normed, normed_fp8, normed_scale = _v41_mhc_pallas_call(
            functools.partial(
                _v41_pre_quant_mixes_kernel,
                hc_mult=hc_mult,
                hidden_size=hidden_size,
                rms_eps=rms_eps,
                use_stream0=use_stream0,
            ),
            (res2d, pre_mix_in.reshape(num_tokens, hc_mult)),
            norm_weight,
            hc_fn,
            None,
            None,
            hc_mult,
            (
                (mix_hc, jnp.float32),
                (1, jnp.float32),
                (hidden_size, jnp.bfloat16),
                (hidden_size, jnp.float8_e4m3fn),
                (1, jnp.float32),
            ),
            max_token_block=128,
        )
        next_pre_mix, post_mix, comb_mix = utils.mhc_pre_gates(
            mixes,
            sqr,
            hc_mult,
            hidden_size,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
    return (
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
        normed.reshape(*outer_shape, hidden_size),
        next_pre_mix.reshape(*outer_shape, hc_mult),
        normed_fp8.reshape(*outer_shape, hidden_size),
        normed_scale.reshape(*outer_shape, 1),
    )


def _v41_fused_post_pre_quant_1strip_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    pre_in_ref,
    nw_ref,
    fn_72_ref,
    scale_ref,
    hb_ref,
    newres_ref,
    post_out_ref,
    comb_out_ref,
    normed_ref,
    pre_out_ref,
    normed_fp8_ref,
    normed_scale_ref,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    write_bf16: bool = True,
):
    """Seam kernel (num_tokens <= 128): 1-pass packed GEMM + Sinkhorn + contiguous FP8 quant."""
    val_bf16 = _v41_fused_post_pre_1strip_body(
        x_ref,
        res_ref,
        post_ref,
        comb_ref,
        pre_in_ref,
        nw_ref,
        fn_72_ref,
        scale_ref,
        hb_ref,
        newres_ref,
        post_out_ref,
        comb_out_ref,
        normed_ref,
        pre_out_ref,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        write_bf16=write_bf16,
    )
    val_f32 = (normed_ref[...] if write_bf16 else val_bf16).astype(jnp.float32)
    abs_max = jnp.max(jnp.abs(val_f32), axis=-1, keepdims=True)
    scale = abs_max / _FP8_E4M3_MAX
    scale = jnp.where(scale == 0, 1.0, scale)
    normed_scale_ref[...] = scale
    normed_fp8_ref[...] = jnp.clip(
        val_f32 * (1.0 / scale),
        -_FP8_E4M3_MAX,
        _FP8_E4M3_MAX,
    ).astype(jnp.float8_e4m3fn)


def mhc_fused_post_pre_quant_v41(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    pre_mix_in: jax.Array,
    norm_weight: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    write_bf16: bool = True,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    *outer_shape, hc_mult, hidden_size = residual.shape
    num_tokens = x.size // hidden_size
    normed_w = hidden_size
    if num_tokens <= 256:
        seam_fn = (
            _v41_fused_post_pre_quant_1strip_kernel
            if num_tokens <= 128
            else _v41_fused_post_pre_quant_kernel
        )
        (
            new_res2d,
            post_mix,
            comb_mix,
            normed,
            next_pre_mix,
            normed_fp8,
            normed_scale,
        ) = _v41_mhc_pallas_call(
            functools.partial(
                seam_fn,
                rms_eps=rms_eps,
                hc_pre_eps=hc_pre_eps,
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
                write_bf16=write_bf16,
            ),
            (
                x.reshape(num_tokens, hidden_size).astype(jnp.bfloat16),
                residual.reshape(num_tokens, hc_mult * hidden_size),
                post_layer_mix.reshape(num_tokens, hc_mult),
                comb_res_mix.reshape(num_tokens, hc_mult * hc_mult),
                pre_mix_in.reshape(num_tokens, hc_mult),
            ),
            norm_weight,
            hc_fn,
            hc_scale,
            hc_base,
            hc_mult,
            (
                (hc_mult * hidden_size, jnp.bfloat16),
                (hc_mult, jnp.float32),
                (hc_mult * hc_mult, jnp.float32),
                (normed_w, jnp.bfloat16),
                (hc_mult, jnp.float32),
                (hidden_size, jnp.float8_e4m3fn),
                (1, jnp.float32),
            ),
            max_token_block=64 if num_tokens <= 64 else 128,
        )
    else:
        mix_hc = hc_mult * (hc_mult + 2)
        (
            new_res2d,
            mixes,
            sqr,
            normed,
            normed_fp8,
            normed_scale,
        ) = _v41_mhc_pallas_call(
            functools.partial(
                _v41_fused_post_pre_quant_mixes_kernel,
                rms_eps=rms_eps,
            ),
            (
                x.reshape(num_tokens, hidden_size).astype(jnp.bfloat16),
                residual.reshape(num_tokens, hc_mult * hidden_size),
                post_layer_mix.reshape(num_tokens, hc_mult),
                comb_res_mix.reshape(num_tokens, hc_mult * hc_mult),
                pre_mix_in.reshape(num_tokens, hc_mult),
            ),
            norm_weight,
            hc_fn,
            None,
            None,
            hc_mult,
            (
                (hc_mult * hidden_size, jnp.bfloat16),
                (mix_hc, jnp.float32),
                (1, jnp.float32),
                (normed_w, jnp.bfloat16),
                (hidden_size, jnp.float8_e4m3fn),
                (1, jnp.float32),
            ),
            max_token_block=128,
        )
        next_pre_mix, post_mix, comb_mix = utils.mhc_pre_gates(
            mixes,
            sqr,
            hc_mult,
            hidden_size,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
    return (
        new_res2d.reshape(residual.shape),
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
        normed.reshape(*outer_shape, normed_w),
        next_pre_mix.reshape(*outer_shape, hc_mult),
        normed_fp8.reshape(*outer_shape, hidden_size),
        normed_scale.reshape(*outer_shape, 1),
    )


def _engram_gate_in_vmem(
    streams_bf: list[jax.Array],
    kv_ref,
    qk_ref,
    *,
    hidden_size: int,
    eps: float,
) -> list[jax.Array]:
    """In-VMEM Engram gated residual update across hc_mult streams on 2D T(8,128)."""
    hc_mult = len(streams_bf)
    inv_sqrt_d = hidden_size**-0.5
    v_f32 = kv_ref[:, hc_mult * hidden_size : (hc_mult + 1) * hidden_size].astype(
        jnp.float32
    )
    out_streams = []
    for j in range(hc_mult):
        lo = j * hidden_size
        hi = lo + hidden_size
        h_f32 = streams_bf[j].astype(jnp.float32)
        k_f32 = kv_ref[:, lo:hi].astype(jnp.float32)
        w_f32 = qk_ref[:, lo:hi]
        h_msq = jnp.mean(h_f32 * h_f32, axis=-1, keepdims=True)
        k_msq = jnp.mean(k_f32 * k_f32, axis=-1, keepdims=True)
        rstd = jax.lax.rsqrt(h_msq + eps) * jax.lax.rsqrt(k_msq + eps)
        dot = (
            jnp.sum(h_f32 * w_f32 * k_f32, axis=-1, keepdims=True)
            * rstd
            * inv_sqrt_d
        )
        abs_sqrt = jnp.sqrt(jnp.maximum(jnp.abs(dot), 1e-6))
        signed = jnp.where(dot < 0.0, -abs_sqrt, abs_sqrt)
        gate = jax.nn.sigmoid(signed)
        out_streams.append((h_f32 + gate * v_f32).astype(jnp.bfloat16))
    return out_streams


def _engram_gate_kernel(
    res_ref,
    kv_ref,
    qk_ref,
    out_ref,
    *,
    hc_mult: int,
    hidden_size: int,
    eps: float,
):
    streams_bf = [
        res_ref[:, j * hidden_size : (j + 1) * hidden_size] for j in range(hc_mult)
    ]
    gated = _engram_gate_in_vmem(
        streams_bf, kv_ref, qk_ref, hidden_size=hidden_size, eps=eps
    )
    for j in range(hc_mult):
        out_ref[:, j * hidden_size : (j + 1) * hidden_size] = gated[j]


def engram_gate_pallas(
    residual: jax.Array,
    kv: jax.Array,
    qk_weight: jax.Array,
    *,
    eps: float = 1e-6,
) -> jax.Array:
    """2D-tiled Pallas Engram gating without 3D T(4,128) relayouts or f32 HBM spills."""
    *outer_shape, hc_mult, hidden_size = residual.shape
    num_tokens = residual.size // (hc_mult * hidden_size)
    max_tb = (
        128
        if num_tokens % utils.SUBLANE == 0
        else 128
    )
    tb, padded_tokens = utils.select_token_block(num_tokens, max_tb)
    res2d = residual.reshape(num_tokens, hc_mult * hidden_size).astype(jnp.bfloat16)
    kv2d = kv.reshape(num_tokens, (hc_mult + 1) * hidden_size).astype(jnp.bfloat16)
    res_pad, kv_pad = utils.pad_to(padded_tokens, res2d, kv2d)
    qk_2d = qk_weight.reshape(1, hc_mult * hidden_size).astype(jnp.float32)
    buf_mode = pl.Buffered(buffer_count=2)
    out_pad = pl.pallas_call(
        functools.partial(
            _engram_gate_kernel,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            eps=eps,
        ),
        grid=(padded_tokens // tb,),
        in_specs=[
            pl.BlockSpec((tb, res_pad.shape[1]), lambda i: (i, 0), pipeline_mode=buf_mode),
            pl.BlockSpec((tb, kv_pad.shape[1]), lambda i: (i, 0), pipeline_mode=buf_mode),
            pl.BlockSpec(qk_2d.shape, lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec(
            (tb, res_pad.shape[1]), lambda i: (i, 0), pipeline_mode=buf_mode
        ),
        out_shape=jax.ShapeDtypeStruct(res_pad.shape, jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=utils.DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
    )(res_pad, kv_pad, qk_2d)
    (out_2d,) = utils.trim_to(num_tokens, out_pad)
    return out_2d.reshape(*outer_shape, hc_mult, hidden_size)


def _v41_fused_post_engram_pre_1strip_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    pre_in_ref,
    kv_ref,
    nw_ref,
    fn_72_ref,
    qk_ref,
    scale_ref,
    hb_ref,
    newres_ref,
    post_out_ref,
    comb_out_ref,
    normed_ref,
    pre_out_ref,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
):
    """Fused mhc_post + engram_gate + mhc_pre (num_tokens <= 128) in VMEM."""
    post = post_ref[...]
    comb = comb_ref[...]
    pre_in = pre_in_ref[...]
    hidden_size = x_ref.shape[1]
    hc_mult = post.shape[1]
    mix_hc = hc_mult * (hc_mult + 2)
    dh = 1280 if hidden_size % 1280 == 0 else hidden_size
    num_strips = hidden_size // dh

    for s in range(num_strips):
        s_lo = s * dh
        s_hi = s_lo + dh
        x_f32 = x_ref[:, s_lo:s_hi].astype(jnp.float32)
        old_s = [
            res_ref[:, i * hidden_size + s_lo : i * hidden_size + s_hi].astype(
                jnp.float32
            )
            for i in range(hc_mult)
        ]
        for j in range(hc_mult):
            acc = post[:, j : j + 1] * x_f32
            for i in range(hc_mult):
                k = i * hc_mult + j
                acc = acc + comb[:, k : k + 1] * old_s[i]
            newres_ref[:, j * hidden_size + s_lo : j * hidden_size + s_hi] = (
                acc.astype(jnp.bfloat16)
            )

    post_streams_bf = [
        newres_ref[:, j * hidden_size : (j + 1) * hidden_size] for j in range(hc_mult)
    ]
    gated_streams_bf = _engram_gate_in_vmem(
        post_streams_bf, kv_ref, qk_ref, hidden_size=hidden_size, eps=rms_eps
    )
    for j in range(hc_mult):
        newres_ref[:, j * hidden_size : (j + 1) * hidden_size] = gated_streams_bf[j]

    sqr = None
    for s in range(num_strips):
        s_lo = s * dh
        s_hi = s_lo + dh
        col_strip = None
        for j in range(hc_mult):
            lo = j * hidden_size + s_lo
            nf = newres_ref[:, lo : lo + dh].astype(jnp.float32)
            sj = jnp.sum(nf * nf, axis=-1, keepdims=True)
            sqr = sj if sqr is None else sqr + sj
            term = pre_in[:, j : j + 1] * nf
            col_strip = term if col_strip is None else col_strip + term
        assert col_strip is not None
        normed_ref[:, s_lo:s_hi] = col_strip.astype(jnp.bfloat16)

    acc = _packed_mixes(newres_ref[...], fn_72_ref, mix_hc)
    assert sqr is not None
    _v41_in_kernel_gates(
        acc,
        sqr,
        scale_ref,
        hb_ref,
        post_out_ref,
        comb_out_ref,
        pre_out_ref,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )
    _rms_norm_to_ref(normed_ref[...], nw_ref, normed_ref, rms_eps)


def _v41_fused_post_engram_pre_mixes_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    pre_in_ref,
    kv_ref,
    nw_ref,
    fn_72_ref,
    qk_ref,
    newres_ref,
    mixes_ref,
    sqr_ref,
    normed_ref,
    *,
    rms_eps: float,
):
    """Fused mhc_post + engram_gate + mhc_pre (multi-block) in VMEM."""
    post = post_ref[...]
    comb = comb_ref[...]
    pre_in = pre_in_ref[...]
    hidden_size = x_ref.shape[1]
    hc_mult = post.shape[1]
    mix_hc = hc_mult * (hc_mult + 2)
    dh = 1280 if hidden_size % 1280 == 0 else hidden_size
    num_strips = hidden_size // dh

    for s in range(num_strips):
        s_lo = s * dh
        s_hi = s_lo + dh
        x_f32 = x_ref[:, s_lo:s_hi].astype(jnp.float32)
        old_s = [
            res_ref[:, i * hidden_size + s_lo : i * hidden_size + s_hi].astype(
                jnp.float32
            )
            for i in range(hc_mult)
        ]
        for j in range(hc_mult):
            acc = post[:, j : j + 1] * x_f32
            for i in range(hc_mult):
                k = i * hc_mult + j
                acc = acc + comb[:, k : k + 1] * old_s[i]
            newres_ref[:, j * hidden_size + s_lo : j * hidden_size + s_hi] = (
                acc.astype(jnp.bfloat16)
            )

    post_streams_bf = [
        newres_ref[:, j * hidden_size : (j + 1) * hidden_size] for j in range(hc_mult)
    ]
    gated_streams_bf = _engram_gate_in_vmem(
        post_streams_bf, kv_ref, qk_ref, hidden_size=hidden_size, eps=rms_eps
    )
    for j in range(hc_mult):
        newres_ref[:, j * hidden_size : (j + 1) * hidden_size] = gated_streams_bf[j]

    sqr = None
    for s in range(num_strips):
        s_lo = s * dh
        s_hi = s_lo + dh
        col_strip = None
        for j in range(hc_mult):
            lo = j * hidden_size + s_lo
            nf = newres_ref[:, lo : lo + dh].astype(jnp.float32)
            sj = jnp.sum(nf * nf, axis=-1, keepdims=True)
            sqr = sj if sqr is None else sqr + sj
            term = pre_in[:, j : j + 1] * nf
            col_strip = term if col_strip is None else col_strip + term
        assert col_strip is not None
        normed_ref[:, s_lo:s_hi] = col_strip.astype(jnp.bfloat16)

    mixes_ref[...] = _packed_mixes(newres_ref[...], fn_72_ref, mix_hc)
    assert sqr is not None
    sqr_ref[...] = sqr
    _rms_norm_to_ref(normed_ref[...], nw_ref, normed_ref, rms_eps)


def mhc_fused_post_engram_pre_v41(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    pre_mix_in: jax.Array,
    engram_kv: jax.Array,
    engram_qk_weight: jax.Array,
    norm_weight: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Single-pass fused mhc_post + Engram gate + mhc_pre_v41 kernel."""
    *outer_shape, hc_mult, hidden_size = residual.shape
    num_tokens = x.size // hidden_size
    max_tb = (
        64
        if num_tokens <= 64
        else (
            max(utils.SUBLANE, math.gcd(num_tokens, 128))
            if num_tokens % utils.SUBLANE == 0
            else 128
        )
    )
    tb, padded_tokens = utils.select_token_block(num_tokens, max_tb)
    token_inputs = utils.pad_to(
        padded_tokens,
        x.reshape(num_tokens, hidden_size).astype(jnp.bfloat16),
        residual.reshape(num_tokens, hc_mult * hidden_size),
        post_layer_mix.reshape(num_tokens, hc_mult),
        comb_res_mix.reshape(num_tokens, hc_mult * hc_mult),
        pre_mix_in.reshape(num_tokens, hc_mult),
        engram_kv.reshape(num_tokens, (hc_mult + 1) * hidden_size).astype(jnp.bfloat16),
    )
    nw_2d = norm_weight.reshape(1, -1).astype(jnp.bfloat16)
    qk_2d = engram_qk_weight.reshape(1, hc_mult * hidden_size).astype(jnp.float32)

    mix_hc = hc_mult * (hc_mult + 2)
    if hc_fn.ndim == 3 and hc_fn.dtype == jnp.bfloat16:
        hc_dim = hc_fn.shape[2]
        fn_72 = hc_fn.reshape(3 * mix_hc, hc_dim)
    elif (
        hc_fn.ndim == 2 and hc_fn.dtype == jnp.bfloat16 and hc_fn.shape[0] == 3 * mix_hc
    ):
        hc_dim = hc_fn.shape[1]
        fn_72 = hc_fn
    else:
        hc_dim = hc_fn.shape[1]
        fn_72 = jnp.concatenate(utils.split_fn3(hc_fn), axis=0)

    buf_mode = pl.Buffered(buffer_count=2)
    if num_tokens <= 128:
        if hc_scale.size == mix_hc:
            scale_vec = hc_scale.reshape(1, mix_hc).astype(jnp.float32)
        else:
            scale_vec = (
                jnp.repeat(
                    hc_scale.reshape(3),
                    jnp.array([hc_mult, hc_mult, hc_mult * hc_mult]),
                    total_repeat_length=mix_hc,
                )
                .reshape(1, mix_hc)
                .astype(jnp.float32)
            )
        hb_vec = hc_base.reshape(1, mix_hc).astype(jnp.float32)
        out_widths_dtypes: tuple[tuple[int, Any], ...] = (
            (hc_mult * hidden_size, jnp.bfloat16),
            (hc_mult, jnp.float32),
            (hc_mult * hc_mult, jnp.float32),
            (hidden_size, jnp.bfloat16),
            (hc_mult, jnp.float32),
        )
        outs = pl.pallas_call(
            functools.partial(
                _v41_fused_post_engram_pre_1strip_kernel,
                rms_eps=rms_eps,
                hc_pre_eps=hc_pre_eps,
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
            ),
            grid=(padded_tokens // tb,),
            in_specs=[
                *(
                    pl.BlockSpec(
                        (tb, a.shape[1]), lambda i: (i, 0), pipeline_mode=buf_mode
                    )
                    for a in token_inputs
                ),
                pl.BlockSpec(nw_2d.shape, lambda i: (0, 0)),
                pl.BlockSpec((3 * mix_hc, hc_dim), lambda i: (0, 0)),
                pl.BlockSpec(qk_2d.shape, lambda i: (0, 0)),
                pl.BlockSpec((1, mix_hc), lambda i: (0, 0)),
                pl.BlockSpec((1, mix_hc), lambda i: (0, 0)),
            ],
            out_specs=[
                pl.BlockSpec((tb, w), lambda i: (i, 0), pipeline_mode=buf_mode)
                for w, _ in out_widths_dtypes
            ],
            out_shape=[
                jax.ShapeDtypeStruct((padded_tokens, w), dt)
                for w, dt in out_widths_dtypes
            ],
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel",),
                vmem_limit_bytes=utils.DEFAULT_VMEM_LIMIT_BYTES,
                disable_bounds_checks=True,
            ),
        )(*token_inputs, nw_2d, fn_72, qk_2d, scale_vec, hb_vec)
        new_res2d, post_mix, comb_mix, normed, next_pre_mix = utils.trim_to(
            num_tokens, *outs
        )
    else:
        out_widths_dtypes = (
            (hc_mult * hidden_size, jnp.bfloat16),
            (mix_hc, jnp.float32),
            (1, jnp.float32),
            (hidden_size, jnp.bfloat16),
        )
        outs = pl.pallas_call(
            functools.partial(
                _v41_fused_post_engram_pre_mixes_kernel,
                rms_eps=rms_eps,
            ),
            grid=(padded_tokens // tb,),
            in_specs=[
                *(
                    pl.BlockSpec(
                        (tb, a.shape[1]), lambda i: (i, 0), pipeline_mode=buf_mode
                    )
                    for a in token_inputs
                ),
                pl.BlockSpec(nw_2d.shape, lambda i: (0, 0)),
                pl.BlockSpec((3 * mix_hc, hc_dim), lambda i: (0, 0)),
                pl.BlockSpec(qk_2d.shape, lambda i: (0, 0)),
            ],
            out_specs=[
                pl.BlockSpec((tb, w), lambda i: (i, 0), pipeline_mode=buf_mode)
                for w, _ in out_widths_dtypes
            ],
            out_shape=[
                jax.ShapeDtypeStruct((padded_tokens, w), dt)
                for w, dt in out_widths_dtypes
            ],
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel",),
                vmem_limit_bytes=utils.DEFAULT_VMEM_LIMIT_BYTES,
                disable_bounds_checks=True,
            ),
        )(*token_inputs, nw_2d, fn_72, qk_2d)
        new_res2d, mixes, sqr, normed = utils.trim_to(num_tokens, *outs)
        next_pre_mix, post_mix, comb_mix = utils.mhc_pre_gates(
            mixes,
            sqr,
            hc_mult,
            hidden_size,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
    return (
        new_res2d.reshape(residual.shape),
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
        normed.reshape(*outer_shape, hidden_size),
        next_pre_mix.reshape(*outer_shape, hc_mult),
    )


def _v41_fused_post_engram_pre_quant_1strip_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    pre_in_ref,
    kv_ref,
    nw_ref,
    fn_72_ref,
    qk_ref,
    scale_ref,
    hb_ref,
    newres_ref,
    post_out_ref,
    comb_out_ref,
    normed_ref,
    pre_out_ref,
    normed_fp8_ref,
    normed_scale_ref,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
):
    """Fused mhc_post + engram_gate + mhc_pre_quant (num_tokens <= 128) in VMEM."""
    post = post_ref[...]
    comb = comb_ref[...]
    pre_in = pre_in_ref[...]
    hidden_size = x_ref.shape[1]
    hc_mult = post.shape[1]
    mix_hc = hc_mult * (hc_mult + 2)
    dh = 1280 if hidden_size % 1280 == 0 else hidden_size
    num_strips = hidden_size // dh

    for s in range(num_strips):
        s_lo = s * dh
        s_hi = s_lo + dh
        x_f32 = x_ref[:, s_lo:s_hi].astype(jnp.float32)
        old_s = [
            res_ref[:, i * hidden_size + s_lo : i * hidden_size + s_hi].astype(
                jnp.float32
            )
            for i in range(hc_mult)
        ]
        for j in range(hc_mult):
            acc = post[:, j : j + 1] * x_f32
            for i in range(hc_mult):
                k = i * hc_mult + j
                acc = acc + comb[:, k : k + 1] * old_s[i]
            newres_ref[:, j * hidden_size + s_lo : j * hidden_size + s_hi] = (
                acc.astype(jnp.bfloat16)
            )

    post_streams_bf = [
        newres_ref[:, j * hidden_size : (j + 1) * hidden_size] for j in range(hc_mult)
    ]
    gated_streams_bf = _engram_gate_in_vmem(
        post_streams_bf, kv_ref, qk_ref, hidden_size=hidden_size, eps=rms_eps
    )
    for j in range(hc_mult):
        newres_ref[:, j * hidden_size : (j + 1) * hidden_size] = gated_streams_bf[j]

    sqr = None
    for s in range(num_strips):
        s_lo = s * dh
        s_hi = s_lo + dh
        col_strip = None
        for j in range(hc_mult):
            lo = j * hidden_size + s_lo
            nf = newres_ref[:, lo : lo + dh].astype(jnp.float32)
            sj = jnp.sum(nf * nf, axis=-1, keepdims=True)
            sqr = sj if sqr is None else sqr + sj
            term = pre_in[:, j : j + 1] * nf
            col_strip = term if col_strip is None else col_strip + term
        assert col_strip is not None
        normed_ref[:, s_lo:s_hi] = col_strip.astype(jnp.bfloat16)

    acc = _packed_mixes(newres_ref[...], fn_72_ref, mix_hc)
    assert sqr is not None
    _v41_in_kernel_gates(
        acc,
        sqr,
        scale_ref,
        hb_ref,
        post_out_ref,
        comb_out_ref,
        pre_out_ref,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )
    _rms_norm_quant_to_refs(
        normed_ref[...],
        nw_ref,
        normed_ref,
        normed_fp8_ref,
        normed_scale_ref,
        rms_eps,
    )


def _v41_fused_post_engram_pre_quant_mixes_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    pre_in_ref,
    kv_ref,
    nw_ref,
    fn_72_ref,
    qk_ref,
    newres_ref,
    mixes_ref,
    sqr_ref,
    normed_ref,
    normed_fp8_ref,
    normed_scale_ref,
    *,
    rms_eps: float,
):
    """Fused mhc_post + engram_gate + mhc_pre_quant (multi-block) in VMEM."""
    post = post_ref[...]
    comb = comb_ref[...]
    pre_in = pre_in_ref[...]
    hidden_size = x_ref.shape[1]
    hc_mult = post.shape[1]
    mix_hc = hc_mult * (hc_mult + 2)
    dh = 1280 if hidden_size % 1280 == 0 else hidden_size
    num_strips = hidden_size // dh

    for s in range(num_strips):
        s_lo = s * dh
        s_hi = s_lo + dh
        x_f32 = x_ref[:, s_lo:s_hi].astype(jnp.float32)
        old_s = [
            res_ref[:, i * hidden_size + s_lo : i * hidden_size + s_hi].astype(
                jnp.float32
            )
            for i in range(hc_mult)
        ]
        for j in range(hc_mult):
            acc = post[:, j : j + 1] * x_f32
            for i in range(hc_mult):
                k = i * hc_mult + j
                acc = acc + comb[:, k : k + 1] * old_s[i]
            newres_ref[:, j * hidden_size + s_lo : j * hidden_size + s_hi] = (
                acc.astype(jnp.bfloat16)
            )

    post_streams_bf = [
        newres_ref[:, j * hidden_size : (j + 1) * hidden_size] for j in range(hc_mult)
    ]
    gated_streams_bf = _engram_gate_in_vmem(
        post_streams_bf, kv_ref, qk_ref, hidden_size=hidden_size, eps=rms_eps
    )
    for j in range(hc_mult):
        newres_ref[:, j * hidden_size : (j + 1) * hidden_size] = gated_streams_bf[j]

    sqr = None
    for s in range(num_strips):
        s_lo = s * dh
        s_hi = s_lo + dh
        col_strip = None
        for j in range(hc_mult):
            lo = j * hidden_size + s_lo
            nf = newres_ref[:, lo : lo + dh].astype(jnp.float32)
            sj = jnp.sum(nf * nf, axis=-1, keepdims=True)
            sqr = sj if sqr is None else sqr + sj
            term = pre_in[:, j : j + 1] * nf
            col_strip = term if col_strip is None else col_strip + term
        assert col_strip is not None
        normed_ref[:, s_lo:s_hi] = col_strip.astype(jnp.bfloat16)

    mixes_ref[...] = _packed_mixes(newres_ref[...], fn_72_ref, mix_hc)
    assert sqr is not None
    sqr_ref[...] = sqr
    _rms_norm_quant_to_refs(
        normed_ref[...],
        nw_ref,
        normed_ref,
        normed_fp8_ref,
        normed_scale_ref,
        rms_eps,
    )


def mhc_fused_post_engram_pre_quant_v41(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    pre_mix_in: jax.Array,
    engram_kv: jax.Array,
    engram_qk_weight: jax.Array,
    norm_weight: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Single-pass fused mhc_post + Engram gate + mhc_pre_quant_v41 kernel."""
    *outer_shape, hc_mult, hidden_size = residual.shape
    num_tokens = x.size // hidden_size
    if num_tokens <= 64:
        tb, padded_tokens = utils.select_token_block(num_tokens, 64)
    else:
        padded_tokens = ((num_tokens + 31) // 32) * 32
        tb = min(128, padded_tokens)
        while tb > 32 and padded_tokens % tb != 0:
            tb -= 32
    token_inputs = utils.pad_to(
        padded_tokens,
        x.reshape(num_tokens, hidden_size).astype(jnp.bfloat16),
        residual.reshape(num_tokens, hc_mult * hidden_size),
        post_layer_mix.reshape(num_tokens, hc_mult),
        comb_res_mix.reshape(num_tokens, hc_mult * hc_mult),
        pre_mix_in.reshape(num_tokens, hc_mult),
        engram_kv.reshape(num_tokens, (hc_mult + 1) * hidden_size).astype(jnp.bfloat16),
    )
    nw_2d = norm_weight.reshape(1, -1).astype(jnp.bfloat16)
    qk_2d = engram_qk_weight.reshape(1, hc_mult * hidden_size).astype(jnp.float32)

    mix_hc = hc_mult * (hc_mult + 2)
    if hc_fn.ndim == 3 and hc_fn.dtype == jnp.bfloat16:
        hc_dim = hc_fn.shape[2]
        fn_72 = hc_fn.reshape(3 * mix_hc, hc_dim)
    elif (
        hc_fn.ndim == 2 and hc_fn.dtype == jnp.bfloat16 and hc_fn.shape[0] == 3 * mix_hc
    ):
        hc_dim = hc_fn.shape[1]
        fn_72 = hc_fn
    else:
        hc_dim = hc_fn.shape[1]
        fn_72 = jnp.concatenate(utils.split_fn3(hc_fn), axis=0)

    buf_mode = pl.Buffered(buffer_count=2)
    if num_tokens <= 128:
        if hc_scale.size == mix_hc:
            scale_vec = hc_scale.reshape(1, mix_hc).astype(jnp.float32)
        else:
            scale_vec = (
                jnp.repeat(
                    hc_scale.reshape(3),
                    jnp.array([hc_mult, hc_mult, hc_mult * hc_mult]),
                    total_repeat_length=mix_hc,
                )
                .reshape(1, mix_hc)
                .astype(jnp.float32)
            )
        hb_vec = hc_base.reshape(1, mix_hc).astype(jnp.float32)
        out_widths_dtypes: tuple[tuple[int, Any], ...] = (
            (hc_mult * hidden_size, jnp.bfloat16),
            (hc_mult, jnp.float32),
            (hc_mult * hc_mult, jnp.float32),
            (hidden_size, jnp.bfloat16),
            (hc_mult, jnp.float32),
            (hidden_size, jnp.float8_e4m3fn),
            (1, jnp.float32),
        )
        outs = pl.pallas_call(
            functools.partial(
                _v41_fused_post_engram_pre_quant_1strip_kernel,
                rms_eps=rms_eps,
                hc_pre_eps=hc_pre_eps,
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
            ),
            grid=(padded_tokens // tb,),
            in_specs=[
                *(
                    pl.BlockSpec(
                        (tb, a.shape[1]), lambda i: (i, 0), pipeline_mode=buf_mode
                    )
                    for a in token_inputs
                ),
                pl.BlockSpec(nw_2d.shape, lambda i: (0, 0)),
                pl.BlockSpec((3 * mix_hc, hc_dim), lambda i: (0, 0)),
                pl.BlockSpec(qk_2d.shape, lambda i: (0, 0)),
                pl.BlockSpec((1, mix_hc), lambda i: (0, 0)),
                pl.BlockSpec((1, mix_hc), lambda i: (0, 0)),
            ],
            out_specs=[
                pl.BlockSpec((tb, w), lambda i: (i, 0), pipeline_mode=buf_mode)
                for w, _ in out_widths_dtypes
            ],
            out_shape=[
                jax.ShapeDtypeStruct((padded_tokens, w), dt)
                for w, dt in out_widths_dtypes
            ],
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel",),
                vmem_limit_bytes=utils.DEFAULT_VMEM_LIMIT_BYTES,
                disable_bounds_checks=True,
            ),
        )(*token_inputs, nw_2d, fn_72, qk_2d, scale_vec, hb_vec)
        (
            new_res2d,
            post_mix,
            comb_mix,
            normed,
            next_pre_mix,
            normed_fp8,
            normed_scale,
        ) = utils.trim_to(num_tokens, *outs)
    else:
        out_widths_dtypes = (
            (hc_mult * hidden_size, jnp.bfloat16),
            (mix_hc, jnp.float32),
            (1, jnp.float32),
            (hidden_size, jnp.bfloat16),
            (hidden_size, jnp.float8_e4m3fn),
            (1, jnp.float32),
        )
        outs = pl.pallas_call(
            functools.partial(
                _v41_fused_post_engram_pre_quant_mixes_kernel,
                rms_eps=rms_eps,
            ),
            grid=(padded_tokens // tb,),
            in_specs=[
                *(
                    pl.BlockSpec(
                        (tb, a.shape[1]), lambda i: (i, 0), pipeline_mode=buf_mode
                    )
                    for a in token_inputs
                ),
                pl.BlockSpec(nw_2d.shape, lambda i: (0, 0)),
                pl.BlockSpec((3 * mix_hc, hc_dim), lambda i: (0, 0)),
                pl.BlockSpec(qk_2d.shape, lambda i: (0, 0)),
            ],
            out_specs=[
                pl.BlockSpec((tb, w), lambda i: (i, 0), pipeline_mode=buf_mode)
                for w, _ in out_widths_dtypes
            ],
            out_shape=[
                jax.ShapeDtypeStruct((padded_tokens, w), dt)
                for w, dt in out_widths_dtypes
            ],
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel",),
                vmem_limit_bytes=utils.DEFAULT_VMEM_LIMIT_BYTES,
                disable_bounds_checks=True,
            ),
        )(*token_inputs, nw_2d, fn_72, qk_2d)
        new_res2d, mixes, sqr, normed, normed_fp8, normed_scale = utils.trim_to(
            num_tokens, *outs
        )
        next_pre_mix, post_mix, comb_mix = utils.mhc_pre_gates(
            mixes,
            sqr,
            hc_mult,
            hidden_size,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
    return (
        new_res2d.reshape(residual.shape),
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
        normed.reshape(*outer_shape, hidden_size),
        next_pre_mix.reshape(*outer_shape, hc_mult),
        normed_fp8.reshape(*outer_shape, hidden_size),
        normed_scale.reshape(*outer_shape, 1),
    )


def _v41_post_collapse_norm_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    pre_in_ref,
    nw_ref,
    newres_ref,
    normed_ref,
    *,
    rms_eps: float,
):
    """Stage-boundary kernel: strip-tiled mhc_post + mhc_collapse + RMSNorm."""
    post = post_ref[...]
    comb = comb_ref[...]
    pre_in = pre_in_ref[...]
    hidden_size = x_ref.shape[1]
    hc_mult = post.shape[1]
    dh = 1280 if hidden_size % 1280 == 0 else hidden_size
    num_strips = hidden_size // dh
    for s in range(num_strips):
        s_lo = s * dh
        s_hi = s_lo + dh
        x_f32 = x_ref[:, s_lo:s_hi].astype(jnp.float32)
        old_s = [
            res_ref[:, i * hidden_size + s_lo : i * hidden_size + s_hi].astype(
                jnp.float32
            )
            for i in range(hc_mult)
        ]
        col_strip = None
        for j in range(hc_mult):
            acc = post[:, j : j + 1] * x_f32
            for i in range(hc_mult):
                k = i * hc_mult + j
                acc = acc + comb[:, k : k + 1] * old_s[i]
            new_bf = acc.astype(jnp.bfloat16)
            newres_ref[:, j * hidden_size + s_lo : j * hidden_size + s_hi] = new_bf
            term = pre_in[:, j : j + 1] * new_bf.astype(jnp.float32)
            col_strip = term if col_strip is None else col_strip + term
        assert col_strip is not None
        normed_ref[:, s_lo:s_hi] = col_strip.astype(jnp.bfloat16)
    _rms_norm_to_ref(normed_ref[...], nw_ref, normed_ref, rms_eps)


def _v41_final_collapse_norm_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    pre_in_ref,
    nw_ref,
    normed_ref,
    *,
    rms_eps: float,
):
    """Final stage-exit kernel: strip-tiled mhc_post + mhc_collapse + RMSNorm."""
    post = post_ref[...]
    comb = comb_ref[...]
    pre_in = pre_in_ref[...]
    hidden_size = x_ref.shape[1]
    hc_mult = post.shape[1]
    dh = 1280 if hidden_size % 1280 == 0 else hidden_size
    num_strips = hidden_size // dh
    for s in range(num_strips):
        s_lo = s * dh
        s_hi = s_lo + dh
        x_f32 = x_ref[:, s_lo:s_hi].astype(jnp.float32)
        old_s = [
            res_ref[:, i * hidden_size + s_lo : i * hidden_size + s_hi].astype(
                jnp.float32
            )
            for i in range(hc_mult)
        ]
        col_strip = None
        for j in range(hc_mult):
            acc = post[:, j : j + 1] * x_f32
            for i in range(hc_mult):
                k = i * hc_mult + j
                acc = acc + comb[:, k : k + 1] * old_s[i]
            new_bf = acc.astype(jnp.bfloat16)
            term = pre_in[:, j : j + 1] * new_bf.astype(jnp.float32)
            col_strip = term if col_strip is None else col_strip + term
        assert col_strip is not None
        normed_ref[:, s_lo:s_hi] = col_strip.astype(jnp.bfloat16)
    _rms_norm_to_ref(normed_ref[...], nw_ref, normed_ref, rms_eps)


def mhc_post_collapse_norm_v41(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    pre_mix_in: jax.Array,
    norm_weight: jax.Array,
    *,
    rms_eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array]:
    """Fused mhc_post + mhc_collapse + RMSNorm returning (new_residual, normed)."""
    *outer_shape, hc_mult, hidden_size = residual.shape
    num_tokens = x.size // hidden_size
    max_tb = (
        max(utils.SUBLANE, math.gcd(num_tokens, 128))
        if num_tokens % utils.SUBLANE == 0
        else 128
    )
    tb, padded_tokens = utils.select_token_block(num_tokens, max_tb)
    token_inputs = utils.pad_to(
        padded_tokens,
        x.reshape(num_tokens, hidden_size).astype(jnp.bfloat16),
        residual.reshape(num_tokens, hc_mult * hidden_size),
        post_layer_mix.reshape(num_tokens, hc_mult),
        comb_res_mix.reshape(num_tokens, hc_mult * hc_mult),
        pre_mix_in.reshape(num_tokens, hc_mult),
    )
    nw_2d = norm_weight.reshape(1, -1).astype(jnp.bfloat16)
    buf_mode = pl.Buffered(buffer_count=2)
    out_widths_dtypes: tuple[tuple[int, Any], ...] = (
        (hc_mult * hidden_size, jnp.bfloat16),
        (hidden_size, jnp.bfloat16),
    )
    outs = pl.pallas_call(
        functools.partial(_v41_post_collapse_norm_kernel, rms_eps=rms_eps),
        grid=(padded_tokens // tb,),
        in_specs=[
            *(
                pl.BlockSpec(
                    (tb, a.shape[1]), lambda i: (i, 0), pipeline_mode=buf_mode
                )
                for a in token_inputs
            ),
            pl.BlockSpec(nw_2d.shape, lambda i: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((tb, w), lambda i: (i, 0), pipeline_mode=buf_mode)
            for w, _ in out_widths_dtypes
        ],
        out_shape=[
            jax.ShapeDtypeStruct((padded_tokens, w), dt)
            for w, dt in out_widths_dtypes
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=utils.DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
    )(*token_inputs, nw_2d)
    new_res2d, normed = utils.trim_to(num_tokens, *outs)
    return (
        new_res2d.reshape(residual.shape),
        normed.reshape(*outer_shape, hidden_size),
    )


def mhc_final_collapse_norm_v41(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    pre_mix_in: jax.Array,
    norm_weight: jax.Array,
    *,
    rms_eps: float = 1e-6,
) -> jax.Array:
    """Fused mhc_post + mhc_collapse + RMSNorm returning normed only (no dead residual write)."""
    *outer_shape, hc_mult, hidden_size = residual.shape
    num_tokens = x.size // hidden_size
    max_tb = (
        max(utils.SUBLANE, math.gcd(num_tokens, 128))
        if num_tokens % utils.SUBLANE == 0
        else 128
    )
    tb, padded_tokens = utils.select_token_block(num_tokens, max_tb)
    token_inputs = utils.pad_to(
        padded_tokens,
        x.reshape(num_tokens, hidden_size).astype(jnp.bfloat16),
        residual.reshape(num_tokens, hc_mult * hidden_size),
        post_layer_mix.reshape(num_tokens, hc_mult),
        comb_res_mix.reshape(num_tokens, hc_mult * hc_mult),
        pre_mix_in.reshape(num_tokens, hc_mult),
    )
    nw_2d = norm_weight.reshape(1, -1).astype(jnp.bfloat16)
    buf_mode = pl.Buffered(buffer_count=2)
    normed_pad = pl.pallas_call(
        functools.partial(_v41_final_collapse_norm_kernel, rms_eps=rms_eps),
        grid=(padded_tokens // tb,),
        in_specs=[
            *(
                pl.BlockSpec(
                    (tb, a.shape[1]), lambda i: (i, 0), pipeline_mode=buf_mode
                )
                for a in token_inputs
            ),
            pl.BlockSpec(nw_2d.shape, lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec(
            (tb, hidden_size), lambda i: (i, 0), pipeline_mode=buf_mode
        ),
        out_shape=jax.ShapeDtypeStruct((padded_tokens, hidden_size), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=utils.DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
    )(*token_inputs, nw_2d)
    (normed,) = utils.trim_to(num_tokens, normed_pad)
    return normed.reshape(*outer_shape, hidden_size)

