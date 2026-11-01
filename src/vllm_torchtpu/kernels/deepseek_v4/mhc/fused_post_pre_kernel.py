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
from vllm_torchtpu.kernels.deepseek_v4.mhc.utils import DEFAULT_VMEM_LIMIT_BYTES


def _fused_kernel(x_ref, *refs, **params):
    """One token block: recombine, store once, 3-pass GEMM, collapse."""
    _fused_block(x_ref[...], *refs, **params)


def _fused_add_kernel(x_ref, addend_ref, *refs, addend_scale, **params):
    """`_fused_kernel` of the sublayer output `x + addend * addend_scale`."""
    _fused_block(
        utils.add_scaled_bf16(x_ref[...], addend_ref[...], addend_scale),
        *refs,
        **params,
    )


def _fused_block(
    x_bf,
    res_ref,
    post_ref,
    comb_ref,
    fn_hi_ref,
    fn_mid_ref,
    fn_lo_ref,
    sc_ref,
    hb_ref,
    newres_ref,
    post_out_ref,
    comb_out_ref,
    layer_ref,
    logits_ref,
    logits_t_ref,
    gates_t_ref,
    *,
    hc_mult,
    hidden_size,
    gemm_precision,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
):
    """`_fused_kernel`'s body, given the block's (tb, hidden) bf16 sublayer
    output `x_bf`."""
    post = post_ref[...]  # (tb, hc_mult) f32
    comb = comb_ref[...]  # (tb, hc_mult * hc_mult) f32, row-major (i, j)

    # Keep the streams (and x) in bf16 and upcast per use: transient f32
    # values instead of 4 held f32 copies halves the resident working set,
    # leaving VMEM headroom for the pipeline to prefetch the next block.
    # bf16 -> f32 casts are exact, so results are unchanged.
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
    dn = (((1,), (1,)), ((), ()))
    acc = jax.lax.dot_general(
        g_bf, fn_hi_ref[...], dn, preferred_element_type=jnp.float32
    )
    if gemm_precision == "highest":
        acc = acc + jax.lax.dot_general(
            g_bf, fn_mid_ref[...], dn, preferred_element_type=jnp.float32
        )
        acc = acc + jax.lax.dot_general(
            g_bf, fn_lo_ref[...], dn, preferred_element_type=jnp.float32
        )
    gf = g_bf.astype(jnp.float32)  # f32 for the squared sum + collapse
    sqr = jnp.sum(gf * gf, axis=-1, keepdims=True)

    m, h = hc_mult, hidden_size
    scaled = acc * jax.lax.rsqrt(sqr / (m * h) + rms_eps)
    pre_mix = jax.nn.sigmoid(scaled[:, :m] * sc_ref[0, 0] + hb_ref[:, :m]) + hc_pre_eps
    lay = pre_mix[:, 0:1] * gf[:, :h]
    for i in range(1, m):
        lay = lay + pre_mix[:, i : i + 1] * gf[:, i * h : (i + 1) * h]
    layer_ref[...] = lay.astype(jnp.bfloat16)

    utils.post_comb_gates_epilogue(
        scaled,
        sc_ref,
        hb_ref,
        post_out_ref,
        comb_out_ref,
        logits_ref,
        logits_t_ref,
        gates_t_ref,
        hc_mult=m,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "rms_eps",
        "hc_pre_eps",
        "hc_sinkhorn_eps",
        "hc_post_mult_value",
        "sinkhorn_repeat",
        "addend_scale",
        "token_block_size",
        "gemm_precision",
        "vmem_limit_bytes",
    ),
)
def fused_post_pre_gates(
    x2d: jax.Array,
    res2d: jax.Array,
    post2d: jax.Array,
    comb2d: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    addend: jax.Array | None = None,
    addend_scale: float = 1.0,
    token_block_size: int = 32,
    gemm_precision: str = "highest",
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The Pallas seam kernel: post recombine fused with pre's mix GEMM,
    in-block collapse and the post/comb gates.

    Args:
        x2d: (num_tokens, hidden_size), bfloat16 — sublayer output.
        res2d: (num_tokens, hc_mult * hidden_size), bfloat16 — old streams.
        post2d: (num_tokens, hc_mult), float32.
        comb2d: (num_tokens, hc_mult * hc_mult), float32, row-major (i, j).
        fn: (hc_mult3, hc_mult * hidden_size), float32 — next sublayer's
            gate projection.
        hc_scale: (3,), float32 — pre/post/comb logit scales.
        hc_base: (hc_mult3,), float32 — logit biases, pre entries first.
        rms_eps: static; RMS-norm epsilon of the gates.
        hc_pre_eps: static; additive floor on the sigmoid pre gates.
        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat: static;
            the post/comb gate constants of ``utils.mhc_pre_gates``.
        addend: optional (num_tokens, hidden_size), bfloat16. When given,
            the sublayer output is `x2d + addend * addend_scale`, rounded
            as that bf16 expression is.
        addend_scale: static; the addend's scale.
        token_block_size: tokens per grid step. Default 32 keeps the
            worst-case VMEM footprint (f32 old streams + bf16 blocks +
            the three resident bf16 fn chunks) near 12 MB at DeepSeek-V4
            shapes; 64 also fits.
        gemm_precision: "highest" (3-pass exact decomposition, the
            default) or "default" (hi-chunk pass only; pre_mix and the
            collapse then also see the hi-only mixes).

    Returns:
        new_res2d: (num_tokens, hc_mult * hidden_size), bfloat16.
        post2d: (num_tokens, hc_mult), float32 — post_mix.
        comb2d: (num_tokens, hc_mult * hc_mult), float32 — comb_mix,
            row-major (i, j).
        layer2d: (num_tokens, hidden_size), bfloat16.
    """
    assert x2d.dtype == jnp.bfloat16, x2d.dtype
    assert addend is None or addend.dtype == jnp.bfloat16, addend.dtype
    assert res2d.dtype == jnp.bfloat16, res2d.dtype
    assert post2d.dtype == jnp.float32, post2d.dtype
    assert comb2d.dtype == jnp.float32, comb2d.dtype
    assert fn.dtype == jnp.float32, fn.dtype

    num_tokens, hidden_size = x2d.shape
    hc_hidden = res2d.shape[1]
    hc_mult = hc_hidden // hidden_size
    hc_mult3 = fn.shape[0]
    assert res2d.shape == (num_tokens, hc_mult * hidden_size)
    assert addend is None or addend.shape == x2d.shape, (addend.shape, x2d.shape)
    assert fn.shape == (hc_mult3, hc_hidden), (fn.shape, res2d.shape)
    assert gemm_precision in ("highest", "default"), gemm_precision

    # This kernel keeps a fixed block: its default (32) already fits the
    # budget at DeepSeek-V4 shapes, so there is no vmem_need to shrink by.
    tb, padded_tokens = utils.select_token_block(num_tokens, token_block_size)
    x2d, res2d, post2d, comb2d = utils.pad_to(padded_tokens, x2d, res2d, post2d, comb2d)
    if addend is None:
        kernel, x_inputs = _fused_kernel, (x2d,)
    else:
        kernel = functools.partial(_fused_add_kernel, addend_scale=addend_scale)
        x_inputs = (x2d, *utils.pad_to(padded_tokens, addend))

    fn_hi, fn_mid, fn_lo = utils.split_fn3(fn)
    sc2d = hc_scale.reshape(1, 3).astype(jnp.float32)
    hb2d = hc_base.reshape(1, hc_mult3).astype(jnp.float32)

    # The last block's gates read every block's logits.
    compiler_params = pltpu.CompilerParams(
        dimension_semantics=("arbitrary",),
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
    )

    resident = pl.BlockSpec((hc_mult3, hc_hidden), lambda i: (0, 0))
    x_spec = pl.BlockSpec((tb, hidden_size), lambda i: (i, 0))
    gate_specs, gate_shapes, gate_scratch = utils.post_comb_gates_specs(
        padded_tokens, hc_mult
    )
    new_res, post, comb, layer2d = pl.pallas_call(
        functools.partial(
            kernel,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            gemm_precision=gemm_precision,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
        ),
        grid=(padded_tokens // tb,),
        in_specs=[
            *[x_spec] * len(x_inputs),
            pl.BlockSpec((tb, hc_hidden), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult * hc_mult), lambda i: (i, 0)),
            resident,
            resident,
            resident,
            pl.BlockSpec((1, 3), lambda i: (0, 0)),
            pl.BlockSpec((1, hc_mult3), lambda i: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((tb, hc_hidden), lambda i: (i, 0)),
            *gate_specs,
            pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((padded_tokens, hc_hidden), jnp.bfloat16),
            *gate_shapes,
            jax.ShapeDtypeStruct((padded_tokens, hidden_size), jnp.bfloat16),
        ],
        scratch_shapes=gate_scratch,
        compiler_params=compiler_params,
    )(*x_inputs, res2d, post2d, comb2d, fn_hi, fn_mid, fn_lo, sc2d, hb2d)

    return cast(
        tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        utils.trim_to(num_tokens, new_res, post, comb, layer2d),
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
    addend: jax.Array | None = None,
    addend_scale: float = 1.0,
    token_block_size: int = 32,
    gemm_precision: str = "highest",
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fused seam op: one Pallas post+GEMM+collapse+gates kernel.
    Returns (residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur).

    With `addend`, the sublayer output is `x + addend * addend_scale`; see
    `fused_post_pre_gates`.
    """
    assert residual.dtype == jnp.bfloat16

    outer_shape = residual.shape[:-2]
    hc_mult, hidden_size = residual.shape[-2:]

    x2d = x.reshape(-1, hidden_size).astype(jnp.bfloat16)
    res2d = residual.reshape(-1, hc_mult * hidden_size)
    post2d = post_layer_mix.reshape(-1, hc_mult)
    comb2d = comb_res_mix.reshape(-1, hc_mult * hc_mult)
    addend2d = None if addend is None else addend.reshape(-1, hidden_size)

    new_res2d, post_mix_cur, comb_mix_cur, layer_input_cur = fused_post_pre_gates(
        x2d,
        res2d,
        post2d,
        comb2d,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        addend=addend2d,
        addend_scale=addend_scale,
        token_block_size=token_block_size,
        gemm_precision=gemm_precision,
        vmem_limit_bytes=vmem_limit_bytes,
    )

    return (
        new_res2d.reshape(*outer_shape, hc_mult, hidden_size),
        post_mix_cur.reshape(*outer_shape, hc_mult, 1),
        comb_mix_cur.reshape(*outer_shape, hc_mult, hc_mult),
        layer_input_cur.reshape(*outer_shape, hidden_size),
    )
