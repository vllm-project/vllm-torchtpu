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

import functools
from typing import cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.mhc import utils
from vllm_torchtpu.kernels.deepseek_v4.mhc.sinkhorn_kernel import single_token_post_comb
from vllm_torchtpu.kernels.deepseek_v4.mhc.utils import DEFAULT_VMEM_LIMIT_BYTES


def _fused_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    fn_refs,
    sc_ref,
    hb_ref,
    newres_ref,
    mixes_ref,
    sqrsum_ref,
    layer_ref,
    *,
    hc_mult,
    hidden_size,
    gemm_precision,
    rms_eps,
    hc_pre_eps,
    fused_gates=False,
    hc_sinkhorn_eps=1e-6,
    hc_post_mult_value=2.0,
    sinkhorn_repeat=20,
    finish_layer=None,
):
    """Recombine and collapse; decode uses an FP32 vector projection."""
    native_streams = fused_gates and res_ref.shape == (hc_mult, hidden_size)
    post = post_ref[...]  # (tb, hc_mult) f32
    comb = comb_ref[...]  # (tb, hc_mult * hc_mult) f32, row-major (i, j)

    # Keep the streams (and x) in bf16 and upcast per use: transient f32
    # values instead of 4 held f32 copies halves the resident working set,
    # leaving VMEM headroom for the pipeline to prefetch the next block.
    # bf16 -> f32 casts are exact, so results are unchanged.
    x_bf = x_ref[...]  # (tb, hidden) bf16
    if native_streams:
        combined = post * x_bf.astype(jnp.float32)
        for i in range(hc_mult):
            combined = combined + (
                comb[i : i + 1, :].T * res_ref[i : i + 1, :].astype(jnp.float32)
            )
        combined = combined.astype(jnp.bfloat16)
        newres_ref[...] = combined
        streams_bf = [combined[i : i + 1, :] for i in range(hc_mult)]
    else:
        old_bf = [
            res_ref[:, i * hidden_size : (i + 1) * hidden_size] for i in range(hc_mult)
        ]

        streams_bf = []
        for j in range(hc_mult):
            acc = post[:, j : j + 1] * x_bf.astype(jnp.float32)
            for i in range(hc_mult):
                k = i * hc_mult + j
                acc = acc + comb[:, k : k + 1] * old_bf[i].astype(jnp.float32)
            # Round to bf16 BEFORE the GEMM: the unfused path's pre reads the
            # bf16 residual that post stored, and parity requires matching it.
            new_bf = acc.astype(jnp.bfloat16)
            newres_ref[:, j * hidden_size : (j + 1) * hidden_size] = new_bf
            streams_bf.append(new_bf)

    g_bf = jnp.concatenate(streams_bf, axis=1)  # (tb, hc_mult * hidden_size)
    # One token is a matrix-vector product; VPU avoids BF16 splitting and MXU padding.
    if fused_gates and gemm_precision == "highest":
        acc = jnp.sum(
            fn_refs[0][...] * g_bf.astype(jnp.float32), axis=1, keepdims=True
        ).T
    else:
        fn_hi, fn_mid, fn_lo = (
            utils.split_fn3(fn_refs[0][...])
            if len(fn_refs) == 1
            else tuple(ref[...] for ref in fn_refs)
        )
        dn = (((1,), (1,)), ((), ()))
        acc = jax.lax.dot_general(g_bf, fn_hi, dn, preferred_element_type=jnp.float32)
        if gemm_precision == "highest":
            acc = acc + jax.lax.dot_general(
                g_bf, fn_mid, dn, preferred_element_type=jnp.float32
            )
            acc = acc + jax.lax.dot_general(
                g_bf, fn_lo, dn, preferred_element_type=jnp.float32
            )
    if not fused_gates:
        mixes_ref[...] = acc
    gf = g_bf.astype(jnp.float32)  # f32 for the squared sum + collapse
    sqr = jnp.sum(gf * gf, axis=-1, keepdims=True)

    m, h = hc_mult, hidden_size
    scaled = acc[:, :m] * jax.lax.rsqrt(sqr / (m * h) + rms_eps)
    pre_mix = jax.nn.sigmoid(scaled * sc_ref[0, 0] + hb_ref[:, :m]) + hc_pre_eps
    lay = pre_mix[:, 0:1] * gf[:, :h]
    for i in range(1, m):
        lay = lay + pre_mix[:, i : i + 1] * gf[:, i * h : (i + 1) * h]
    layer = lay.astype(jnp.bfloat16)
    if finish_layer is None:
        layer_ref[...] = layer
    else:
        finish_layer(layer)

    # Release the collapsed streams before the iterative gates epilogue.
    if fused_gates:
        scaled_all = acc * jax.lax.rsqrt(sqr / (hc_mult * hidden_size) + rms_eps)
        post, comb = single_token_post_comb(
            scaled_all,
            sc_ref[...],
            hb_ref[...],
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
        mixes_ref[...] = post.T if native_streams else post
        sqrsum_ref[...] = comb
    else:
        sqrsum_ref[...] = sqr


@functools.partial(
    jax.jit,
    static_argnames=(
        "fused_gates",
        "hc_sinkhorn_eps",
        "hc_post_mult_value",
        "sinkhorn_repeat",
        "rms_eps",
        "hc_pre_eps",
        "token_block_size",
        "gemm_precision",
        "vmem_limit_bytes",
    ),
)
def fused_post_pre_mixes(
    x2d: jax.Array,
    res2d: jax.Array,
    post2d: jax.Array,
    comb2d: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    *,
    fused_gates: bool = False,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 2.0,
    sinkhorn_repeat: int = 20,
    token_block_size: int = 32,
    gemm_precision: str = "highest",
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The Pallas seam kernel: post recombine fused with pre's mix GEMM
    and in-block collapse.

    Args:
        x2d: (num_tokens, hidden_size), bfloat16 — sublayer output.
        res2d: (num_tokens, hc_mult * hidden_size), bfloat16 — old streams.
        post2d: (num_tokens, hc_mult), float32.
        comb2d: (num_tokens, hc_mult * hc_mult), float32, row-major (i, j).
        fn: (hc_mult3, hc_mult * hidden_size), float32 — next sublayer's
            gate projection.
        hc_scale: (3,), float32 — pre/post/comb logit scales.
        hc_base: (hc_mult3,), float32 — logit biases, pre entries first.
        rms_eps: static; RMS-norm epsilon for the in-block pre gates.
        hc_pre_eps: static; additive floor on the sigmoid pre gates.
        token_block_size: tokens per grid step. Default 32 keeps the
            worst-case VMEM footprint (f32 old streams + bf16 blocks +
            the three resident bf16 fn chunks) near 12 MB at DeepSeek-V4
            shapes; 64 also fits.
        gemm_precision: "highest" (3-pass exact decomposition, the
            default) or "default" (hi-chunk pass only; pre_mix and the
            collapse then also see the hi-only mixes).

    Returns:
        new_res2d: (num_tokens, hc_mult * hidden_size), bfloat16.
        mixes: (num_tokens, hc_mult3), float32 — raw, unscaled, for the
            XLA gates.
        sqrsum: (num_tokens, 1), float32.
        layer2d: (num_tokens, hidden_size), bfloat16.
        With fused_gates, mixes and sqrsum instead hold post (1, 4)
        and comb (4, 4); this requires one token and four streams.
    """
    assert x2d.dtype == jnp.bfloat16, x2d.dtype
    assert res2d.dtype == jnp.bfloat16, res2d.dtype
    assert post2d.dtype == jnp.float32, post2d.dtype
    assert comb2d.dtype == jnp.float32, comb2d.dtype
    assert fn.dtype == jnp.float32, fn.dtype

    num_tokens, hidden_size = x2d.shape
    hc_hidden = res2d.shape[1]
    hc_mult = hc_hidden // hidden_size
    hc_mult3 = fn.shape[0]
    assert res2d.shape == (num_tokens, hc_mult * hidden_size)
    assert fn.shape == (hc_mult3, hc_hidden), (fn.shape, res2d.shape)
    assert gemm_precision in ("highest", "default"), gemm_precision

    # This kernel keeps a fixed block: its default (32) already fits the
    # budget at DeepSeek-V4 shapes, so there is no vmem_need to shrink by.
    tb, padded_tokens = utils.select_token_block(num_tokens, token_block_size)
    if num_tokens == 1 and hc_mult == 4:
        tb, padded_tokens = 1, 1
    x2d, res2d, post2d, comb2d = utils.pad_to(padded_tokens, x2d, res2d, post2d, comb2d)

    # Decode consumes weights once; split in VMEM to avoid HBM intermediates.
    fn_parts = (fn,) if num_tokens == 1 and fn.shape[0] == 24 else utils.split_fn3(fn)
    sc2d = hc_scale.reshape(1, 3).astype(jnp.float32)
    hb2d = hc_base.reshape(1, hc_mult3).astype(jnp.float32)

    compiler_params = pltpu.CompilerParams(
        dimension_semantics=("parallel",),
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
    )

    mix_shape = (1, 4) if fused_gates else (padded_tokens, hc_mult3)
    sum_shape = (4, 4) if fused_gates else (padded_tokens, 1)
    mix_block = (1, 4) if fused_gates else (tb, hc_mult3)
    sum_block = (4, 4) if fused_gates else (tb, 1)
    resident = pl.BlockSpec((hc_mult3, hc_hidden), lambda i: (0, 0))
    new_res, mixes, sqrsum, layer2d = pl.pallas_call(
        functools.partial(
            _fused_kernel,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            gemm_precision=gemm_precision,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            fused_gates=fused_gates,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
        ),
        grid=(padded_tokens // tb,),
        in_specs=[
            pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_hidden), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult * hc_mult), lambda i: (i, 0)),
            tuple(resident for _ in fn_parts),
            pl.BlockSpec((1, 3), lambda i: (0, 0)),
            pl.BlockSpec((1, hc_mult3), lambda i: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((tb, hc_hidden), lambda i: (i, 0)),
            pl.BlockSpec(mix_block, lambda i: (i, 0)),
            pl.BlockSpec(sum_block, lambda i: (i, 0)),
            pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((padded_tokens, hc_hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct(mix_shape, jnp.float32),
            jax.ShapeDtypeStruct(sum_shape, jnp.float32),
            jax.ShapeDtypeStruct((padded_tokens, hidden_size), jnp.bfloat16),
        ],
        compiler_params=compiler_params,
    )(x2d, res2d, post2d, comb2d, fn_parts, sc2d, hb2d)

    return cast(
        tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        utils.trim_to(num_tokens, new_res, mixes, sqrsum, layer2d),
    )


def mhc_fused_post_pre(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    token_block_size: int = 32,
    gemm_precision: str = "highest",
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fused seam, including gates for single-token four-stream decode.
    Returns (residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur).
    """
    assert residual.dtype == jnp.bfloat16

    outer_shape = residual.shape[:-2]
    hc_mult, hidden_size = residual.shape[-2:]

    x2d = x.reshape(-1, hidden_size).astype(jnp.bfloat16)
    res2d = residual.reshape(-1, hc_mult * hidden_size)
    post2d = post_layer_mix.reshape(-1, hc_mult)
    comb2d = comb_res_mix.reshape(-1, hc_mult * hc_mult)

    fused_gates = (
        x2d.shape[0] == 1
        and hc_mult == 4
        and fn.dtype == jnp.float32
        and sinkhorn_repeat >= 1
        and jax.default_backend() == "tpu"
    )
    new_res2d, mixes, sqrsum, layer_input_cur = fused_post_pre_mixes(
        x2d,
        res2d,
        post2d,
        comb2d,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        fused_gates=fused_gates,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        token_block_size=token_block_size,
        gemm_precision=gemm_precision,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    if fused_gates:
        post_mix_cur, comb_mix_cur = mixes, sqrsum
    else:
        _, post_mix_cur, comb_mix_cur = utils.mhc_pre_gates(
            mixes,
            sqrsum,
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
        new_res2d.reshape(*outer_shape, hc_mult, hidden_size),
        post_mix_cur.reshape(*outer_shape, hc_mult, 1),
        comb_mix_cur.reshape(*outer_shape, hc_mult, hc_mult),
        layer_input_cur.reshape(*outer_shape, hidden_size),
    )
