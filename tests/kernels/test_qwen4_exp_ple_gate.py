# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_PLE_FUSED_GATE kernel matches the Torch-order reference (interpret)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.ple_gate import normalized_gate_dot


@pytest.mark.parametrize("tokens,streams,hidden", [(13, 4, 1024), (8, 2, 640)])
def test_fused_gate_dot_matches_reference(tokens, streams, hidden):
    rng = np.random.default_rng(tokens)
    shape = (tokens, streams * hidden)
    key = jnp.asarray(rng.normal(size=shape), jnp.bfloat16)
    query = jnp.asarray(rng.normal(size=shape), jnp.bfloat16)
    kw = jnp.asarray(rng.normal(size=shape[1:]) * 0.1, jnp.bfloat16)
    qw = jnp.asarray(rng.normal(size=shape[1:]) * 0.1, jnp.bfloat16)

    def inv(x):
        g = x.astype(jnp.float32).reshape(tokens, streams, hidden)
        return jax.lax.rsqrt((g * g).mean(-1) + 1e-6)

    def norm(x, w):
        g = x.astype(jnp.float32).reshape(tokens, streams, hidden) * inv(x)[..., None]
        w = (1 + w.astype(jnp.float32)).reshape(streams, hidden)
        return (g * w).astype(jnp.bfloat16).astype(jnp.float32)

    out = normalized_gate_dot(
        key, query, inv(key), inv(query), kw, qw, streams=streams, interpret=True
    )
    ref = (norm(key, kw) * norm(query, qw)).astype(jnp.bfloat16)
    ref = ref.astype(jnp.float32).sum(-1).astype(jnp.bfloat16)
    assert out.shape == (tokens, streams) and out.dtype == jnp.bfloat16
    np.testing.assert_allclose(
        np.asarray(out, np.float32), np.asarray(ref, np.float32), rtol=1e-2, atol=1e-2
    )
