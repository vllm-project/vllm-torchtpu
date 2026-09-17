# SPDX-License-Identifier: Apache-2.0
"""Compare fused Attention Residual with independent float64 arithmetic."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.kimi_k3.attention_residual import attention_residual


@pytest.mark.parametrize('blocks', [0, 1, 2, 4, 8])
def test_fused_mix_matches_float64(blocks):
    rng = np.random.default_rng(96)
    prefix = jnp.asarray(rng.normal(size=(16, 128)), jnp.bfloat16)
    history = jnp.asarray(rng.normal(size=(16, blocks, 128)), jnp.bfloat16)
    weight = jnp.asarray(rng.normal(0, 0.02, size=128), jnp.float32)
    actual = jax.jit(attention_residual,
                     static_argnames=('interpret', ))(prefix,
                                                      history,
                                                      weight,
                                                      interpret=True)
    values = np.concatenate((np.asarray(
        history, np.float64), np.asarray(prefix, np.float64)[:, None]),
                            axis=1)
    scores = (values @ np.asarray(weight, np.float64)
              ) / np.sqrt(np.mean(values * values, axis=-1) + 1e-5)
    probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    expected = (values * probabilities[..., None]).sum(axis=1)
    np.testing.assert_allclose(np.asarray(actual, np.float32),
                               expected,
                               rtol=0.005,
                               atol=0.005)
