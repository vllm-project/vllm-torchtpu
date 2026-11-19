# SPDX-License-Identifier: Apache-2.0
"""Folded-weight prefill HC mix (hc_prefill_mix(fold=True), P-DENSE D4) in
interpret mode: judged against a float64 truth by the 2x rule (its error at
most twice the served fused mix's), the prologue's combined input bitwise."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.hc_prefill_fused import fold_down, extend_down, hc_prefill_mix

BRANCHES, HIDDEN, LOWRANK, EPS, TILE = 4, 128, 64, 1e-6, 128
WIDTH = BRANCHES * HIDDEN


def _weights(seed):
    rng = np.random.default_rng(seed)
    n = lambda shape, s: jnp.asarray(rng.normal(size=shape) * s, jnp.bfloat16)  # noqa: E731
    return dict(w=n((WIDTH,), 0.3), down=n((LOWRANK, WIDTH), 0.05), up=n((WIDTH, LOWRANK), 0.1),
                inject=n((BRANCHES, WIDTH), 0.05)), rng


def _truth(x, p):
    """The HC mix in float64 without intermediate rounding: (mixed, injection)."""
    f = lambda a: np.asarray(jnp.asarray(a).astype(jnp.float32), np.float64)  # noqa: E731
    v = f(x).reshape(len(x), BRANCHES, HIDDEN)
    norm = v / np.sqrt(np.mean(v * v, -1, keepdims=True) + EPS) * (1.0 + f(p["w"]).reshape(BRANCHES, HIDDEN))
    norm = norm.reshape(len(x), WIDTH)
    g = norm @ f(p["down"]).T / BRANCHES
    g = g / (1.0 + np.exp(-g))
    lg = g @ f(p["up"]).T
    mixed = ((1.0 / (1.0 + np.exp(-lg))) * norm).reshape(len(x), BRANCHES, HIDDEN).sum(1) / BRANCHES
    inj = 2.0 / (1.0 + np.exp(-(norm @ f(p["inject"]).T) / BRANCHES))
    return mixed, inj


def _err(a, truth):
    d = np.asarray(jnp.asarray(a).astype(jnp.float32), np.float64) - truth
    return float(np.sqrt(np.mean(d * d))), float(np.max(np.abs(d)))


def _mix(x, p, **kw):
    return hc_prefill_mix(x, p["w"], p["down"], p["up"], p["inject"], branches=BRANCHES, hidden=HIDDEN,
                          eps=EPS, token_tile=TILE, interpret=True, **kw)


@pytest.mark.parametrize("tokens,partial", [(256, False), (384, False), (320, True)])
@pytest.mark.parametrize("norm_impl", ["vpu", "bf16ep64"])
def test_fold_meets_float64_rule_against_the_served_mix(tokens, partial, norm_impl):
    p, rng = _weights(7)
    x = jnp.asarray(rng.normal(size=(tokens, WIDTH)), jnp.bfloat16)
    truth_mixed, truth_inj = _truth(x, p)
    served = _mix(x, p, partial=partial, norm_impl=norm_impl)
    folded = _mix(x, p, partial=partial, norm_impl=norm_impl, fold=True)
    for got, ref, truth in ((folded[0], served[0], truth_mixed), (folded[1], served[1], truth_inj)):
        g, r = _err(got, truth), _err(ref, truth)
        assert g[0] <= 2 * r[0] and g[1] <= 2 * r[1], (g, r)
    assert np.all(np.isfinite(np.asarray(folded[0].astype(jnp.float32))))
    # Not the same program: the fold must change the result somewhere (engagement in interpret mode).
    assert not np.array_equal(np.asarray(folded[0]).view(np.uint16), np.asarray(served[0]).view(np.uint16))


@pytest.mark.parametrize("tokens", [256, 384])
def test_fold_prologue_combines_bitwise_and_meets_the_rule(tokens):
    p, rng = _weights(9)
    prev = jnp.asarray(rng.normal(size=(tokens, WIDTH)), jnp.bfloat16)
    block = jnp.asarray(rng.normal(size=(tokens, HIDDEN)), jnp.bfloat16)
    prev_inj = jnp.asarray(rng.uniform(0.0, 2.0, size=(tokens, BRANCHES)), jnp.float32)
    served = _mix(prev, p, block=block, prev_injection=prev_inj, norm_impl="bf16ep64")
    folded = _mix(prev, p, block=block, prev_injection=prev_inj, norm_impl="bf16ep64", fold=True)
    # h = bf16(f32(residual) + f32(block) * injection_prev[:, b]): the same formula in both kernels.
    np.testing.assert_array_equal(np.asarray(folded[2]).view(np.uint16), np.asarray(served[2]).view(np.uint16))
    truth_mixed, truth_inj = _truth(served[2], p)
    for got, ref, truth in ((folded[0], served[0], truth_mixed), (folded[1], served[1], truth_inj)):
        g, r = _err(got, truth), _err(ref, truth)
        assert g[0] <= 2 * r[0] and g[1] <= 2 * r[1], (g, r)


def test_fold_down_scales_rows_by_one_plus_weight():
    p, _ = _weights(3)
    ext = extend_down(p["down"], p["inject"])
    folded = fold_down(ext, p["w"])
    want = (np.asarray(ext.astype(jnp.float32)) * (1.0 + np.asarray(p["w"].astype(jnp.float32)))[None, :])
    np.testing.assert_array_equal(np.asarray(folded), np.asarray(jnp.asarray(want).astype(jnp.bfloat16)))
    assert folded.shape == ext.shape and folded.dtype == jnp.bfloat16


def test_fold_kernel_is_named_and_the_flag_reaches_it():
    p, rng = _weights(5)
    x = jnp.asarray(rng.normal(size=(256, WIDTH)), jnp.bfloat16)
    # The pallas_call (and its name) is in the jaxpr; interpret-mode lowering inlines the body.
    txt = str(jax.make_jaxpr(lambda a: _mix(a, p, fold=True))(x))
    assert "qwen4exp_hc_prefill_inj_m128_fold" in txt
    txt = str(jax.make_jaxpr(lambda a: _mix(a, p))(x))
    assert "qwen4exp_hc_prefill_inj_m128" in txt and "_fold" not in txt
