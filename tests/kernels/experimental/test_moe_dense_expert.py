# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""TPU parity and routing edge cases for the opt-in dense-expert path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental import moe_dense_expert as kernel
from vllm_torchtpu.layers.core import fused_moe_gmm as wrapper


@pytest.fixture(scope="module")
def tensors():
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")
    m, h, i, e = 16, 512, 512, 4
    rng = np.random.default_rng(27)
    x = jnp.asarray(rng.normal(size=(m, h)), jnp.bfloat16)
    w1 = jnp.asarray(rng.integers(-8, 9, size=(e, h, 2 * i)), jnp.float8_e4m3fn)
    w2 = jnp.asarray(rng.integers(-8, 9, size=(e, i, h)), jnp.float8_e4m3fn)
    s1 = jnp.asarray(rng.uniform(0.006, 0.012, size=(e, 1, 1, 2 * i)), jnp.float32)
    s2 = jnp.asarray(rng.uniform(0.006, 0.012, size=(e, 1, 1, h)), jnp.float32)
    return x, w1, w2, s1, s2


@pytest.mark.parametrize(
    "case", ["mixed", "single", "three", "empty", "zero", "duplicate"]
)
@pytest.mark.parametrize("num_buffers", [1, 2, 3])
def test_same_entrypoint_and_ep_mask(tensors, monkeypatch, case, num_buffers):
    # Exercise each DMA schedule, including wraparound and short pipelines,
    # without requiring a different physical TPU for every capacity.
    monkeypatch.setattr(kernel, "_select_expert_buffers", lambda *args: num_buffers)
    kernel.dense_expert_moe.clear_cache()
    x, w1, w2, s1, s2 = tensors
    ids = np.tile(np.array([[64, 65, 63, 68]], np.int32), (16, 1))
    ids[8:, :2] = [66, 67]
    weights = np.tile(np.array([[0.125, 0.25, 0.375, 0.25]], np.float32), (16, 1))
    if case == "single":
        ids[:, :] = [67, 0, 1, 2]
    elif case == "three":
        ids[:, :] = [64, 65, 66, 0]
    elif case == "empty":
        ids[:, :] = [0, 1, 2, 3]
    elif case == "zero":
        weights[:] = 0
    elif case == "duplicate":
        ids[:, :] = [64, 64, 65, 65]
    args = (
        x,
        w1,
        w2,
        s1,
        s2,
        None,
        None,
        jnp.asarray(weights, jnp.bfloat16),
        jnp.asarray(ids),
    )
    kwargs = dict(
        experts_start=jnp.array(64, jnp.int32),
        topk=4,
        use_ep=True,
        onehot_moe_permute_threshold=32768,
    )
    monkeypatch.setattr(wrapper, "_DENSE_EXPERT_THRESHOLD", 0)
    wrapper.fused_moe_func.clear_cache()
    expected = np.asarray(wrapper.fused_moe_func(*args, **kwargs)).astype("float32")
    monkeypatch.setattr(wrapper, "_DENSE_EXPERT_THRESHOLD", 1024)
    wrapper.fused_moe_func.clear_cache()
    try:
        actual = np.asarray(wrapper.fused_moe_func(*args, **kwargs)).astype("float32")
        hlo = wrapper.fused_moe_func.lower(*args, **kwargs).as_text()
        assert "dense_expert_moe-e_4" in hlo
        assert f"buffers_{num_buffers}" in hlo
        assert "blockwise_onehot_unpermute" not in hlo
        assert np.isfinite(actual).all()
        if case in ("empty", "zero"):
            np.testing.assert_array_equal(actual, 0)
        else:
            rel = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
            assert rel < 0.008, rel
    finally:
        wrapper.fused_moe_func.clear_cache()
        kernel.dense_expert_moe.clear_cache()


@pytest.mark.parametrize(
    "unsupported", ["unaligned_tokens", "bias", "activation", "scale", "bf16_weights"]
)
def test_unsupported_inputs_keep_standard_path(tensors, monkeypatch, unsupported):
    x, w1, w2, s1, s2 = tensors
    b1, activation, padded = None, "silu", False
    if unsupported == "unaligned_tokens":
        x = x[:15]
    elif unsupported == "bias":
        b1 = jnp.zeros((4, 1, 1024), jnp.float32)
    elif unsupported == "activation":
        activation = "gelu"
    elif unsupported == "scale":
        s1 = None
    else:
        w1 = w1.astype(jnp.bfloat16)
    assert not kernel.can_use_dense_expert(
        x,
        w1,
        w2,
        s1,
        s2,
        b1,
        None,
        activation=activation,
        rhs_quant_dtype=None,
        skip_padded_tokens=padded,
    )


def test_unaligned_entrypoint_falls_back(tensors, monkeypatch):
    x, w1, w2, s1, s2 = tensors
    x = x[:15]  # An unsupported token alignment still uses GMM.
    args = (
        x,
        w1,
        w2,
        s1,
        s2,
        None,
        None,
        jnp.ones((15, 1), jnp.bfloat16),
        jnp.zeros((15, 1), jnp.int32),
    )
    monkeypatch.setattr(wrapper, "_DENSE_EXPERT_THRESHOLD", 0)
    wrapper.fused_moe_func.clear_cache()
    expected = wrapper.fused_moe_func(*args)
    monkeypatch.setattr(wrapper, "_DENSE_EXPERT_THRESHOLD", 1024)
    wrapper.fused_moe_func.clear_cache()
    try:
        actual = wrapper.fused_moe_func(*args)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        assert (
            "dense_expert_moe-e_" not in wrapper.fused_moe_func.lower(*args).as_text()
        )
    finally:
        wrapper.fused_moe_func.clear_cache()


@pytest.mark.parametrize("m", [1024, 1040])
def test_qwen_threshold_boundary_matches_gmm(tensors, monkeypatch, m):
    # Qwen's real H/I also checks that the largest enabled bucket compiles.
    rng = np.random.default_rng(20260922)
    h, i, e = 4096, 1024, 4
    x = jnp.asarray(rng.normal(size=(m, h)), jnp.bfloat16)
    w1 = jnp.asarray(rng.integers(-4, 5, size=(e, h, 2 * i)), jnp.float8_e4m3fn)
    w2 = jnp.asarray(rng.integers(-4, 5, size=(e, i, h)), jnp.float8_e4m3fn)
    s1 = jnp.full((e, 1, 1, 2 * i), 0.01, jnp.float32)
    s2 = jnp.full((e, 1, 1, h), 0.01, jnp.float32)
    ids = jnp.asarray(np.tile([0, 3], (m, 1)), jnp.int32)
    weights = jnp.full((m, 2), 0.5, jnp.bfloat16)
    args = (x, w1, w2, s1, s2, None, None, weights, ids)
    kwargs = dict(topk=2, use_sparse_core=False)
    monkeypatch.setattr(wrapper, "_DENSE_EXPERT_THRESHOLD", 0)
    wrapper.fused_moe_func.clear_cache()
    expected = np.asarray(wrapper.fused_moe_func(*args, **kwargs), dtype=np.float32)
    monkeypatch.setattr(wrapper, "_DENSE_EXPERT_THRESHOLD", 1024)
    wrapper.fused_moe_func.clear_cache()
    try:
        actual = np.asarray(wrapper.fused_moe_func(*args, **kwargs), dtype=np.float32)
        hlo = wrapper.fused_moe_func.lower(*args, **kwargs).as_text()
        assert ("dense_expert_moe-e_" in hlo) == (m <= 1024)
        assert np.isfinite(actual).all()
        if m > 1024:
            np.testing.assert_array_equal(actual, expected)
        else:
            relative_l2 = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
            assert relative_l2 < 0.008, relative_l2
    finally:
        wrapper.fused_moe_func.clear_cache()
        kernel.dense_expert_moe.clear_cache()
