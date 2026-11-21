# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41 import engram_gate as eg

M, D = 4, 5120
EPS, CLAMP = 1e-20, 1e-6


def _inputs(tokens: int, seed: int = 0):
    ks = jax.random.split(jax.random.PRNGKey(seed), 5)
    hidden = (jax.random.normal(ks[0], (tokens, M * D)) * 2).astype(jnp.bfloat16)
    kv = (jax.random.normal(ks[1], (tokens, (M + 1) * D)) * 0.3).astype(jnp.bfloat16)
    q = (1 + 0.1 * jax.random.normal(ks[2], (M, D))).astype(jnp.bfloat16)
    k = (1 + 0.1 * jax.random.normal(ks[3], (M, D))).astype(jnp.bfloat16)
    mask = jax.random.uniform(ks[4], (tokens,)) > 0.2
    return hidden, kv, q, k, mask


@pytest.mark.parametrize("tokens", [6, 16, 144, 200, 1024])
@pytest.mark.parametrize("masked", [False, True])
def test_matches_xla_gate_within_one_rounding(tokens, masked):
    hidden, kv, q, k, mask = _inputs(tokens)
    mask = mask if masked else None
    interpret = jax.default_backend() != "tpu"
    got = eg.engram_gate(
        hidden, kv, q, k, mask, eps=EPS, clamp=CLAMP, interpret=interpret
    )
    ref = eg.engram_gate_reference(hidden, kv, q, k, mask, eps=EPS, clamp=CLAMP)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    got32, ref32 = np.asarray(got, np.float32), np.asarray(ref, np.float32)
    # Only the sums over D are reassociated: at most one BF16 step apart, rarely.
    step = np.abs(ref32) * 2.0**-7 + 1e-30
    assert np.all(np.abs(got32 - ref32) <= step)
    assert np.mean(got32 != ref32) < 1e-3
    if masked:
        closed = ~np.asarray(mask)
        np.testing.assert_array_equal(
            got32[closed], np.asarray(hidden, np.float32)[closed]
        )
