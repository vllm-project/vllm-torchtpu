# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""TPU parity, routing edge cases and dispatch for the fused TP MoE kernel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental import moe_fused_tp as kernel
from vllm_torchtpu.layers.core import fused_moe_gmm as wrapper

M_TOKENS, H, I, E, TOPK = 32, 512, 128, 64, 10


@pytest.fixture(scope="module")
def tensors():
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")
    rng = np.random.default_rng(31)
    w1 = jnp.asarray(rng.integers(-8, 9, size=(E, H, 2 * I)), jnp.float8_e4m3fn)
    w2 = jnp.asarray(rng.integers(-8, 9, size=(E, I, H)), jnp.float8_e4m3fn)
    s1 = jnp.asarray(rng.uniform(0.006, 0.012, size=(E, 1, 1, 2 * I)), jnp.float32)
    s2 = jnp.asarray(rng.uniform(0.006, 0.012, size=(E, 1, 1, H)), jnp.float32)
    return rng, w1, w2, s1, s2


def _oracle(x, w1, w2, s1, s2, weights, ids):
    """float64 reference with dequantized weights."""
    x = np.asarray(x.astype(jnp.float32), np.float64)
    w1 = np.asarray(w1.astype(jnp.float32), np.float64) * np.asarray(s1)[:, 0]
    w2 = np.asarray(w2.astype(jnp.float32), np.float64) * np.asarray(s2)[:, 0]
    weights = np.asarray(weights.astype(jnp.float32), np.float64)
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        for k in range(ids.shape[1]):
            if weights[t, k] == 0:
                continue
            hh = x[t] @ w1[ids[t, k]]
            g, u = hh[:I], hh[I:]
            out[t] += weights[t, k] * ((g / (1 + np.exp(-g)) * u) @ w2[ids[t, k]])
    return out


def _routing(rng, m, case):
    ids = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(m)]).astype(np.int32)
    weights = rng.dirichlet(np.ones(TOPK), size=m).astype(np.float32)
    if case == "padded":
        weights[m - 5 :] = 0          # bucket padding rows
    elif case == "single_expert":
        ids[:, :] = np.arange(TOPK)
        ids[:, 0] = 7
        weights[:, 1:] = 0
    elif case == "zero":
        weights[:] = 0
    return weights, ids


@pytest.mark.parametrize("m", [8, 16, 32, 48])
@pytest.mark.parametrize("case", ["random", "padded", "single_expert", "zero"])
@pytest.mark.parametrize("num_buffers", [2, 3, 5])
def test_matches_oracle_and_gmm(tensors, m, case, num_buffers):
    rng, w1, w2, s1, s2 = tensors
    x = jnp.asarray(rng.normal(size=(m, H)), jnp.bfloat16)
    weights, ids = _routing(rng, m, case)
    w_bf16 = jnp.asarray(weights, jnp.bfloat16)
    actual = np.asarray(
        kernel.fused_moe_tp(x, w1, w2, s1, s2, w_bf16, jnp.asarray(ids), num_buffers=num_buffers)
    ).astype(np.float64)
    assert actual.shape == (m, H)
    assert np.isfinite(actual).all()
    if case == "zero":
        assert not actual.any()
        return
    ref = _oracle(x, w1, w2, s1, s2, w_bf16, ids)
    if case == "padded":
        assert not actual[m - 5 :].any()
    valid = slice(None) if case != "padded" else slice(0, m - 5)
    gmm = np.asarray(
        wrapper.fused_moe_func(
            x, w1, w2, s1, s2, None, None, w_bf16, jnp.asarray(ids),
            jnp.zeros((), jnp.int32), topk=TOPK, use_ep=False, skip_padded_tokens=True,
        )
    ).astype(np.float64)

    def row_err(y):
        return np.max(
            np.linalg.norm(y[valid] - ref[valid], axis=1)
            / np.maximum(np.linalg.norm(ref[valid], axis=1), 1e-30)
        )

    # FP8 activations on both paths: no worse than 1.5x the GMM path's error.
    assert row_err(actual) <= max(1.5 * row_err(gmm), 2e-3), (row_err(actual), row_err(gmm))


def test_fault_is_caught(tensors):
    """Positive control: dropping one expert's contribution must fail the bound above."""
    rng, w1, w2, s1, s2 = tensors
    m = 32
    x = jnp.asarray(rng.normal(size=(m, H)), jnp.bfloat16)
    weights, ids = _routing(rng, m, "random")
    w_bf16 = jnp.asarray(weights, jnp.bfloat16)
    broken = np.array(weights)
    broken[ids == ids[0, 0]] = 0      # remove one expert from routing
    y = np.asarray(
        kernel.fused_moe_tp(x, w1, w2, s1, s2, jnp.asarray(broken, jnp.bfloat16), jnp.asarray(ids))
    ).astype(np.float64)
    ref = _oracle(x, w1, w2, s1, s2, w_bf16, ids)
    gmm = np.asarray(
        wrapper.fused_moe_func(
            x, w1, w2, s1, s2, None, None, w_bf16, jnp.asarray(ids),
            jnp.zeros((), jnp.int32), topk=TOPK, use_ep=False, skip_padded_tokens=True,
        )
    ).astype(np.float64)
    err = lambda a: np.max(  # noqa: E731
        np.linalg.norm(a - ref, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), 1e-30)
    )
    assert err(y) > max(1.5 * err(gmm), 2e-3)


def test_dispatch_selects_kernel_up_to_cap(tensors, monkeypatch):
    rng, w1, w2, s1, s2 = tensors
    weights, ids = _routing(rng, 32, "random")
    args = (
        jnp.asarray(rng.normal(size=(32, H)), jnp.bfloat16), w1, w2, s1, s2, None, None,
        jnp.asarray(weights, jnp.bfloat16), jnp.asarray(ids), jnp.zeros((), jnp.int32),
    )
    kwargs = dict(topk=TOPK, use_ep=False, skip_padded_tokens=True)
    for cap, expect in ((0, False), (16, False), (32, True)):
        monkeypatch.setattr(wrapper, "_FUSED_TP_MAX_TOKENS", cap)
        wrapper.fused_moe_func.clear_cache()
        hlo = wrapper.fused_moe_func.lower(*args, **kwargs).as_text()
        assert ("fused_moe_tp-e_64" in hlo) == expect
        assert ("gmm_v2" in hlo) == (not expect)


def test_admission_rejects_ep_shapes():
    x = jax.ShapeDtypeStruct((32, 512), jnp.bfloat16)
    w1 = jax.ShapeDtypeStruct((4, 512, 250), jnp.float8_e4m3fn)   # i not a multiple of 128
    w2 = jax.ShapeDtypeStruct((4, 125, 512), jnp.float8_e4m3fn)
    s1 = jax.ShapeDtypeStruct((4, 1, 1, 250), jnp.float32)
    s2 = jax.ShapeDtypeStruct((4, 1, 1, 512), jnp.float32)
    assert not kernel.can_use_fused_tp(
        x, w1, w2, s1, s2, None, None, activation="silu", rhs_quant_dtype=None
    )


def test_padded_rows_with_nonfinite_hidden_stay_zero(tensors):
    """Bucket padding rows carry real (possibly non-finite) hidden states and zero weight."""
    rng, w1, w2, s1, s2 = tensors
    m = 32
    x = np.asarray(rng.normal(size=(m, H)), np.float32)
    weights, ids = _routing(rng, m, "padded")
    clean = np.asarray(
        kernel.fused_moe_tp(jnp.asarray(x, jnp.bfloat16), w1, w2, s1, s2,
                            jnp.asarray(weights, jnp.bfloat16), jnp.asarray(ids))
    )
    x[m - 5 :] = rng.normal(size=(5, H)) * 1e3
    x[m - 2, :7] = np.nan
    x[m - 1, 3] = np.inf
    dirty = np.asarray(
        kernel.fused_moe_tp(jnp.asarray(x, jnp.bfloat16), w1, w2, s1, s2,
                            jnp.asarray(weights, jnp.bfloat16), jnp.asarray(ids))
    )
    assert np.array_equal(dirty[: m - 5], clean[: m - 5])
    assert not dirty[m - 5 :].astype(np.float32).any()
