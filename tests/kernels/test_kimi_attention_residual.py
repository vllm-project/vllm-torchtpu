# SPDX-License-Identifier: Apache-2.0
"""Compare fused Attention Residual with independent float64 arithmetic."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.kimi_k3.attention_residual import attention_residual


@pytest.mark.parametrize("blocks", [0, 1, 2, 4, 8])
def test_fused_mix_matches_float64(blocks):
    rng = np.random.default_rng(96)
    prefix = jnp.asarray(rng.normal(size=(16, 128)), jnp.bfloat16)
    history = jnp.asarray(rng.normal(size=(16, blocks, 128)), jnp.bfloat16)
    weight = jnp.asarray(rng.normal(0, 0.02, size=128), jnp.float32)
    actual = jax.jit(attention_residual, static_argnames=("interpret",))(
        prefix, history, weight, interpret=True
    )
    values = np.concatenate(
        (np.asarray(history, np.float64), np.asarray(prefix, np.float64)[:, None]),
        axis=1,
    )
    scores = (values @ np.asarray(weight, np.float64)) / np.sqrt(
        np.mean(values * values, axis=-1) + 1e-5
    )
    probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    expected = (values * probabilities[..., None]).sum(axis=1)
    np.testing.assert_allclose(
        np.asarray(actual, np.float32), expected, rtol=0.005, atol=0.005
    )


@pytest.mark.parametrize("tokens", [16, 128])
@pytest.mark.parametrize("blocks", [1, 4, 8])
def test_fused_mix_2d_with_precomputed_inv_rms_matches_float64(tokens, blocks):
    rng = np.random.default_rng(104)
    hidden = 512
    prefix = jnp.asarray(rng.normal(size=(tokens, hidden)), jnp.bfloat16)
    history_3d = jnp.asarray(rng.normal(size=(blocks, tokens, hidden)), jnp.bfloat16)
    history_2d = history_3d.reshape(blocks * tokens, hidden)
    weight = jnp.asarray(rng.normal(0, 0.02, size=hidden), jnp.float32)
    inv_rms = jax.lax.rsqrt(
        jnp.mean(history_3d.astype(jnp.float32) ** 2, axis=-1) + 1e-5
    )
    actual_no_inv = jax.jit(attention_residual, static_argnames=("interpret",))(
        prefix, history_2d, weight, interpret=True
    )
    actual_with_inv = jax.jit(attention_residual, static_argnames=("interpret",))(
        prefix, history_2d, weight, inv_rms, interpret=True
    )
    np.testing.assert_allclose(
        np.asarray(actual_with_inv, np.float32),
        np.asarray(actual_no_inv, np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    values = np.concatenate(
        (
            np.transpose(np.asarray(history_3d, np.float64), (1, 0, 2)),
            np.asarray(prefix, np.float64)[:, None],
        ),
        axis=1,
    )
    scores = (values @ np.asarray(weight, np.float64)) / np.sqrt(
        np.mean(values * values, axis=-1) + 1e-5
    )
    probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    expected = (values * probabilities[..., None]).sum(axis=1)
    np.testing.assert_allclose(
        np.asarray(actual_with_inv, np.float32), expected, rtol=0.005, atol=0.005
    )
