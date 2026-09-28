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

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.mhc import tuning, utils

# Percent by which the analytic ``_vmem_need`` window arithmetic must be
# scaled to match XLA's reported program footprint. Measured on v7x: the
# model predicts 66.25 MiB at tb=256 where XLA reported 77.69 MiB (1.17x).
# Rounded up to 120 so the guard errs on the side of a smaller block.
_VMEM_MODEL_CALIBRATION = 120


def _fused_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    fn3_ref,
    sc_ref,
    hb_ref,
    newres_ref,
    post_mix_ref,
    comb_mix_ref,
    layer_ref,
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
    """One token block: recombine, store once, 3-pass GEMM, collapse, gates."""
    post = post_ref[...]  # (tb, hc_mult) f32
    comb = comb_ref[...]  # (tb, hc_mult * hc_mult) f32, row-major (i, j)

    # Keep the streams (and x) in bf16 and upcast per use: transient f32
    # values instead of 4 held f32 copies halves the resident working set,
    # leaving VMEM headroom for the pipeline to prefetch the next block.
    # bf16 -> f32 casts are exact, so results are unchanged.
    x_bf = x_ref[...]  # (tb, hidden) bf16
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
    # The hi/mid/lo chunks are stacked along N, so one GEMM replaces three.
    # N stays inside a single 256-wide MXU tile either way (72 <= 256), so
    # this cuts tile passes 3x -- 64 instead of 192 at hc_hidden=16384 --
    # rather than trading them for a wider tile.
    #
    # NOT bit-exact, despite summing the row groups in the original
    # hi + mid + lo order. Measured on v6e against the three-dot version:
    # `mixes` differs by <=8.9e-08 absolute (~1 ULP of f32), and that
    # propagates to <=2.5e-06 relative in the gates and a single bf16
    # quantum on 26 of 524288 `layer` elements. The cause is that XLA is
    # free to reassociate a sum of dots sharing an LHS into one wider
    # accumulation, so the unstacked form was not evaluating in source
    # order either; the two forms simply tile K differently.
    n3 = hc_mult * (hc_mult + 2)
    if gemm_precision == "highest":
        acc3 = jax.lax.dot_general(
            g_bf, fn3_ref[...], dn, preferred_element_type=jnp.float32
        )
        acc = acc3[:, :n3] + acc3[:, n3 : 2 * n3]
        acc = acc + acc3[:, 2 * n3 :]
    else:
        acc = jax.lax.dot_general(
            g_bf, fn3_ref[...], dn, preferred_element_type=jnp.float32
        )
    gf = g_bf.astype(jnp.float32)  # f32 for the squared sum + collapse
    sqr = jnp.sum(gf * gf, axis=-1, keepdims=True)

    m, h = hc_mult, hidden_size
    # All three gate groups in-block. `mixes`/`sqrsum` used to be written out
    # so XLA could redo this; keeping it here removes that round-trip and,
    # far more importantly, the Sinkhorn loop that followed it.
    pre_mix, post_mix, comb_flat = utils.mhc_gates_inblock(
        acc,
        sqr,
        m,
        h,
        sc_ref[0, 0],
        sc_ref[0, 1],
        sc_ref[0, 2],
        hb_ref,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    post_mix_ref[...] = post_mix
    comb_mix_ref[...] = comb_flat

    lay = pre_mix[:, 0:1] * gf[:, :h]
    for i in range(1, m):
        lay = lay + pre_mix[:, i : i + 1] * gf[:, i * h : (i + 1) * h]
    layer_ref[...] = lay.astype(jnp.bfloat16)


@functools.partial(
    jax.jit,
    static_argnames=(
        "rms_eps",
        "hc_pre_eps",
        "hc_sinkhorn_eps",
        "hc_post_mult_value",
        "sinkhorn_repeat",
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
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    token_block_size: int = 32,
    gemm_precision: str = "highest",
    vmem_limit_bytes: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The Pallas seam kernel: post recombine fused with pre's mix GEMM,
    the in-block collapse, and all three gate groups.

    Args:
        x2d: (num_tokens, hidden_size), bfloat16 — sublayer output.
        res2d: (num_tokens, hc_mult * hidden_size), bfloat16 — old streams.
        post2d: (num_tokens, hc_mult), float32.
        comb2d: (num_tokens, hc_mult * hc_mult), float32, row-major (i, j).
        fn: (hc_mult3, hc_mult * hidden_size), float32 — next sublayer's
            gate projection.
        hc_scale: (3,), float32 — pre/post/comb logit scales.
        hc_base: (hc_mult3,), float32 — logit biases, pre entries first.
        rms_eps: static; RMS-norm epsilon for the in-block gates.
        hc_pre_eps: static; additive floor on the sigmoid pre gates.
        hc_sinkhorn_eps: static; additive floor inside the Sinkhorn loop.
        hc_post_mult_value: static; scale on the sigmoid post gates.
        sinkhorn_repeat: static; Sinkhorn iteration count.
        token_block_size: tokens per grid step. Default 32 keeps the
            worst-case VMEM footprint (f32 old streams + bf16 blocks +
            the three resident bf16 fn chunks) near 12 MB at DeepSeek-V4
            shapes; 64 also fits.
        gemm_precision: "highest" (3-pass exact decomposition, the
            default) or "default" (hi-chunk pass only; pre_mix and the
            collapse then also see the hi-only mixes).

    Returns:
        new_res2d: (num_tokens, hc_mult * hidden_size), bfloat16.
        post_mix: (num_tokens, hc_mult), float32.
        comb_mix: (num_tokens, hc_mult * hc_mult), float32, row-major.
        layer2d: (num_tokens, hidden_size), bfloat16.
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

    def _vmem_need(tb: int) -> int:
        # Double-buffered windows (x2d, res2d, new_res, layer2d:
        # 2 levels * 2 * tb * (hc_hidden + hidden_size) * 2B) + the f32/bf16
        # stream scratch (g_bf + gf: tb * hc_hidden * (2 + 4)B) + the
        # resident stacked fn (3 * hc_mult3 rows * hc_hidden * 2B; the same
        # byte count whether it is three chunks or one stack, so this term
        # and its calibration below are unchanged by the stacking).
        modelled = (
            tb * (4 * (hc_hidden + hidden_size) * 2 + hc_hidden * 6)
            + 3 * hc_mult3 * hc_hidden * 2
        )
        # Calibration against hardware: on v7x at hidden_size=4096,
        # hc_hidden=16384, hc_mult3=24, tb=256, this model gives 66.25 MiB
        # while XLA reported an actual 77.69 MiB program footprint. The
        # shortfall is VREG spills and Mosaic bookkeeping, which the window
        # arithmetic above cannot see, so scale the estimate to match.
        return (modelled * _VMEM_MODEL_CALIBRATION) // 100

    vmem_limit_bytes = tuning.resolve_vmem_limit_bytes(vmem_limit_bytes)
    tb, padded_tokens = utils.select_token_block(
        num_tokens,
        token_block_size,
        vmem_need=_vmem_need,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    x2d, res2d, post2d, comb2d = utils.pad_to(padded_tokens, x2d, res2d, post2d, comb2d)

    if gemm_precision == "highest":
        fn3 = utils.split_fn3_stacked(fn)
    else:
        fn3 = utils.split_fn3(fn)[0]
    sc2d = hc_scale.reshape(1, 3).astype(jnp.float32)
    hb2d = hc_base.reshape(1, hc_mult3).astype(jnp.float32)

    compiler_params = pltpu.CompilerParams(
        dimension_semantics=("parallel",),
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
    )

    resident = pl.BlockSpec(fn3.shape, lambda i: (0, 0))
    new_res, post_mix, comb_mix, layer2d = pl.pallas_call(
        functools.partial(
            _fused_kernel,
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
            pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_hidden), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult * hc_mult), lambda i: (i, 0)),
            resident,
            pl.BlockSpec((1, 3), lambda i: (0, 0)),
            pl.BlockSpec((1, hc_mult3), lambda i: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((tb, hc_hidden), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult * hc_mult), lambda i: (i, 0)),
            pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((padded_tokens, hc_hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((padded_tokens, hc_mult), jnp.float32),
            jax.ShapeDtypeStruct((padded_tokens, hc_mult * hc_mult), jnp.float32),
            jax.ShapeDtypeStruct((padded_tokens, hidden_size), jnp.bfloat16),
        ],
        compiler_params=compiler_params,
    )(x2d, res2d, post2d, comb2d, fn3, sc2d, hb2d)

    return utils.trim_to(num_tokens, new_res, post_mix, comb_mix, layer2d)


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
    vmem_limit_bytes: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fused seam op: Pallas post+GEMM+collapse+gates kernel.

    Returns (residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur).
    """
    assert residual.dtype == jnp.bfloat16

    outer_shape = residual.shape[:-2]
    hc_mult, hidden_size = residual.shape[-2:]

    x2d = x.reshape(-1, hidden_size).astype(jnp.bfloat16)
    res2d = residual.reshape(-1, hc_mult * hidden_size)
    post2d = post_layer_mix.reshape(-1, hc_mult)
    comb2d = comb_res_mix.reshape(-1, hc_mult * hc_mult)

    new_res2d, post_mix_cur, comb_mix_cur, layer_input_cur = fused_post_pre_mixes(
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
