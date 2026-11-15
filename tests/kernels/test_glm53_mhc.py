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
"""Tests for the fused GLM 5.3 mHC Pallas kernels (`kernels/glm53_mhc`).

On CPU the kernels run in Pallas interpret mode against an unfused JAX
reference built from the shared DeepSeek-V4 mHC gate math.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.mhc import utils as dsv4_utils
from vllm_torchtpu.kernels.glm53_mhc import (
    SUBLANE,
    mhc_fused_post_pre_norm,
    mhc_pre_norm_fused,
    pad_to,
    select_fused_token_block,
    select_token_block,
    split_fn3,
    trim_to,
)

HIDDEN, HC_MULT, ITERS, HC_EPS, RMS_EPS = 256, 4, 20, 1e-6, 1e-5
POST_MULT = 2.0
MIX = (2 + HC_MULT) * HC_MULT


def _rms_norm_weighted(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    xf = x.astype(jnp.float32)
    var = jnp.mean(xf * xf, axis=-1, keepdims=True)
    xf = xf * jax.lax.rsqrt(var + eps)
    return (xf.astype(weight.dtype) * weight).astype(x.dtype)


@pytest.fixture(scope="module")
def weights():
    rng = np.random.default_rng(42)
    fn = jnp.asarray(rng.standard_normal((MIX, HC_MULT * HIDDEN)) * 0.05, jnp.float32)
    return {
        "fn": fn,
        "split": split_fn3(fn),
        "scale": jnp.asarray([0.8, 1.1, 0.9], jnp.float32),
        "base": jnp.asarray(rng.standard_normal(MIX) * 0.1, jnp.float32),
        "norm_weight": jnp.asarray(
            rng.standard_normal(HIDDEN) * 0.2 + 1.0, jnp.float32
        ),
    }


def _unfused_pre(res, w):
    fn_hi, fn_mid, fn_lo = w["split"]
    num_t = res.shape[0]
    dn = (((1,), (1,)), ((), ()))
    acc = (
        jax.lax.dot_general(res, fn_hi, dn, preferred_element_type=jnp.float32)
        + jax.lax.dot_general(res, fn_mid, dn, preferred_element_type=jnp.float32)
        + jax.lax.dot_general(res, fn_lo, dn, preferred_element_type=jnp.float32)
    )
    xf = res.astype(jnp.float32)
    sqr = jnp.sum(xf * xf, axis=-1, keepdims=True)
    pre_mix, post_mix, comb_mix = dsv4_utils.mhc_pre_gates(
        acc,
        sqr,
        HC_MULT,
        HIDDEN,
        w["scale"],
        w["base"],
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        POST_MULT,
        ITERS,
    )
    lay = pre_mix[:, 0:1] * xf[:, :HIDDEN]
    for i in range(1, HC_MULT):
        lay = lay + pre_mix[:, i : i + 1] * xf[:, i * HIDDEN : (i + 1) * HIDDEN]
    normed = _rms_norm_weighted(lay.astype(jnp.bfloat16), w["norm_weight"], RMS_EPS)
    return (
        post_mix.reshape(num_t, HC_MULT, 1),
        comb_mix.reshape(num_t, HC_MULT, HC_MULT),
        normed,
    )


def _unfused_post_pre(x, res, post, comb, w):
    num_t = res.shape[0]
    post2d = post.reshape(num_t, HC_MULT)
    comb2d = comb.reshape(num_t, HC_MULT * HC_MULT)
    old = [res[:, i * HIDDEN : (i + 1) * HIDDEN] for i in range(HC_MULT)]
    streams = []
    for j in range(HC_MULT):
        acc = post2d[:, j : j + 1] * x.astype(jnp.float32)
        for i in range(HC_MULT):
            k = i * HC_MULT + j
            acc = acc + comb2d[:, k : k + 1] * old[i].astype(jnp.float32)
        streams.append(acc.astype(jnp.bfloat16))
    new_res = jnp.concatenate(streams, axis=1)
    return (new_res, *_unfused_pre(new_res, w))


def test_split_fn3_matches_deepseek_v4_split_bitwise():
    rng = np.random.default_rng(0)
    for scale in (1e-4, 2e-2, 1.0, 1e2):
        fn = jnp.asarray(rng.standard_normal((24, 1024)) * scale, jnp.float32)
        for got, want in zip(split_fn3(fn), dsv4_utils.split_fn3(fn), strict=True):
            assert got.dtype == jnp.bfloat16
            np.testing.assert_array_equal(
                np.asarray(got).view(np.uint16), np.asarray(want).view(np.uint16)
            )


def test_token_block_selection_and_padding():
    assert select_token_block(1, 64) == (SUBLANE, SUBLANE)
    assert select_token_block(16, 64) == (16, 16)
    assert select_token_block(513, 64) == (64, 576)
    # A VMEM budget that only fits 16 rows halves the block down to it.
    assert (
        select_token_block(512, 64, vmem_need=lambda tb: tb, vmem_limit_bytes=16)[0]
        == 16
    )
    assert select_fused_token_block(256) == 128
    assert select_fused_token_block(200) == 32
    a = jnp.ones((5, 3))
    (padded,) = pad_to(8, a)
    assert padded.shape == (8, 3)
    assert float(padded[5:].sum()) == 0.0
    (trimmed,) = trim_to(5, padded)
    np.testing.assert_array_equal(np.asarray(trimmed), np.asarray(a))


def _check_fused_matches_unfused(weights, tokens):
    rng = np.random.default_rng(tokens)
    res = jnp.asarray(
        rng.standard_normal((tokens, HC_MULT * HIDDEN)) * 0.5, jnp.bfloat16
    )
    x = jnp.asarray(rng.standard_normal((tokens, HIDDEN)) * 0.5, jnp.bfloat16)
    interpret = jax.default_backend() != "tpu"
    fn_hi, fn_mid, fn_lo = weights["split"]
    consts = (RMS_EPS, HC_EPS, HC_EPS, POST_MULT, ITERS)
    w_args = (weights["scale"], weights["base"], weights["norm_weight"])

    post_f, comb_f, norm_f = mhc_pre_norm_fused(
        res, fn_hi, fn_mid, fn_lo, *w_args, *consts, interpret=interpret
    )
    post_u, comb_u, norm_u = jax.jit(_unfused_pre)(res, weights)
    np.testing.assert_allclose(post_f, post_u, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(comb_f, comb_u, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(norm_f, np.float32), np.asarray(norm_u, np.float32), 2e-2, 2e-2
    )

    got = mhc_fused_post_pre_norm(
        x, res, post_f, comb_f, fn_hi, fn_mid, fn_lo, *w_args, *consts,
        interpret=interpret,
    )  # fmt: skip
    want = jax.jit(_unfused_post_pre)(x, res, post_f, comb_f, weights)
    tols = ((1e-2, 1e-2), (1e-3, 1e-5), (1e-3, 1e-5), (2e-2, 2e-2))
    for g, w, (rtol, atol) in zip(got, want, tols, strict=True):
        assert g.shape == w.shape
        np.testing.assert_allclose(
            np.asarray(g, np.float32), np.asarray(w, np.float32), rtol, atol
        )


@pytest.mark.parametrize("tokens", [1, 8, 16, 32, 64, 128, 513])
def test_fused_kernels_match_unfused_reference(weights, tokens):
    _check_fused_matches_unfused(weights, tokens)


def test_fused_kernels_keep_gate_math_inside_the_kernel():
    """Lowered for TPU, no Sinkhorn/RMSNorm/split op is left outside the kernel."""
    consts = (RMS_EPS, HC_EPS, HC_EPS, POST_MULT, ITERS)
    pre_jit = jax.jit(
        lambda r, a, b, c, s, h, n: mhc_pre_norm_fused(r, a, b, c, s, h, n, *consts)
    )
    fused_jit = jax.jit(lambda *args: mhc_fused_post_pre_norm(*args, *consts))
    for b in (8, 32, 64, 128):
        res = jax.ShapeDtypeStruct((b, 16384), jnp.bfloat16)
        x = jax.ShapeDtypeStruct((b, 4096), jnp.bfloat16)
        post = jax.ShapeDtypeStruct((b, 4, 1), jnp.float32)
        comb = jax.ShapeDtypeStruct((b, 4, 4), jnp.float32)
        fn = jax.ShapeDtypeStruct((24, 16384), jnp.bfloat16)
        sc = jax.ShapeDtypeStruct((3,), jnp.float32)
        hb = jax.ShapeDtypeStruct((24,), jnp.float32)
        nw = jax.ShapeDtypeStruct((4096,), jnp.float32)
        for lowered in (
            pre_jit.trace(res, fn, fn, fn, sc, hb, nw),
            fused_jit.trace(x, res, post, comb, fn, fn, fn, sc, hb, nw),
        ):
            hlo = lowered.lower(lowering_platforms=("tpu",)).as_text()
            outside = "\n".join(
                line for line in hlo.splitlines() if "custom_call" not in line
            )
            for forbidden in (
                "stablehlo.reduce_precision",
                "stablehlo.divide",
                "stablehlo.reduce",
                "stablehlo.rsqrt",
                "stablehlo.exponential",
                "stablehlo.logistic",
                "stablehlo.pad",
                "stablehlo.slice",
            ):
                assert forbidden not in outside, (b, forbidden)


@pytest.mark.tpu_test
@pytest.mark.parametrize("tokens", [8, 64, 2048])
def test_fused_kernels_match_unfused_reference_on_tpu(weights, tokens):
    if jax.default_backend() != "tpu":
        pytest.skip("needs a TPU backend")
    _check_fused_matches_unfused(weights, tokens)
