# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical checks for fused sliding-window accumulation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla.window_attention import attend_windows


@pytest.mark.parametrize(
    "heads,keys,queries,query_block",
    [
        (8, 192, 8, 8),
        (64, 256, 8, 8),
        (64, 192, 64, 64),
    ],
)
@pytest.mark.parametrize("interpret", [True, False])
def test_window_attention_matches_dense(heads, keys, queries, query_block, interpret):
    if not interpret and jax.default_backend() != "tpu":
        pytest.skip("Compiled Pallas kernel requires TPU")
    rng = np.random.default_rng(4)
    groups, dim = 2, 512
    q = jnp.asarray(rng.normal(size=(groups * queries, heads, dim)), jnp.bfloat16)
    kv = jnp.asarray(rng.normal(size=(groups, keys, dim)), jnp.bfloat16)
    mask = jnp.asarray(rng.uniform(size=(groups, queries, keys)) > 0.3)
    mask = mask.at[-1, -1].set(False)
    mask = mask.at[:, :, -8:].set(False)
    kv = kv.at[:, -8:].set(jnp.nan)
    pieces = jnp.stack(
        [
            kv[:, :, start + parity : start + 256 : 2]
            for start in (0, 256)
            for parity in (0, 1)
        ]
    )
    scale = dim**-0.5
    got = attend_windows(
        q, pieces, mask, scale, query_block=query_block, interpret=interpret
    )
    kv = jnp.where(mask.any(axis=1)[..., None], kv, 0)
    scores = (
        jnp.einsum(
            "gqhd,gkd->gqhk",
            q.reshape(groups, queries, heads, dim).astype(jnp.float32),
            kv.astype(jnp.float32),
        )
        * scale
    )
    scores = jnp.where(mask[:, :, None], scores, -jnp.inf)
    maximum = jnp.maximum(-1e30, scores.max(-1))
    prob = jnp.exp(scores - maximum[..., None])
    denominator = prob.sum(-1)
    numerator = jnp.einsum(
        "gqhk,gkd->gqhd",
        prob.astype(jnp.bfloat16),
        kv,
        preferred_element_type=jnp.float32,
    )
    np.testing.assert_allclose(
        got[0], maximum.reshape(got[0].shape), atol=2e-4, rtol=2e-4
    )
    np.testing.assert_allclose(
        got[1], denominator.reshape(got[1].shape), atol=2e-4, rtol=2e-4
    )
    safe_den = jnp.maximum(denominator, 1.0).reshape(*got[0].shape, 1)
    # Probability and output are rounded to BF16; compare normalized values.
    np.testing.assert_allclose(
        got[2].astype(jnp.float32) / safe_den,
        numerator.reshape(q.shape) / safe_den,
        atol=2e-3,
        rtol=8e-3,
    )
    assert np.isfinite(np.asarray(got[2], np.float32)).all()
    np.testing.assert_array_equal(got[2][-1], 0)


@pytest.mark.parametrize(
    "groups,heads,queries,keys",
    [
        (8, 8, 1, 128),
        (2, 8, 64, 256),
        (2, 64, 64, 256),
        (3, 16, 1, 256),
        (2, 32, 64, 256),
    ],
)
@pytest.mark.parametrize("interpret", [True, False])
def test_swa_native_layout_with_sink(groups, heads, queries, keys, interpret):
    """Native Q/O must match dense attention, including invalid NaN KV rows."""
    from vllm_torchtpu.kernels.mla.swa_attention import attend_swa

    if not interpret and jax.default_backend() != "tpu":
        pytest.skip("Compiled Pallas kernel requires TPU")
    rng = np.random.default_rng(83)
    dim = 512
    q = jnp.asarray(rng.normal(size=(groups * queries, heads, dim)), jnp.bfloat16)
    kv = jnp.asarray(rng.normal(size=(groups, keys, dim)), jnp.bfloat16)
    mask = jnp.asarray(rng.uniform(size=(groups, queries, keys)) > 0.3)
    mask = mask.at[-1, -1].set(False).at[:, :, -8:].set(False)
    kv = kv.at[:, -8:].set(jnp.nan)
    sink = jnp.asarray(rng.normal(size=heads), jnp.float32)
    # A large sink also exercises stable normalization without exp overflow.
    sink = sink.at[0].set(100.0)
    pieces = jnp.stack(
        [
            kv[:, :, start + parity : start + 256 : 2]
            for start in (0, 256)
            for parity in (0, 1)
        ]
    )
    got = attend_swa(q, pieces, mask, sink, dim**-0.5, interpret=interpret)
    kv = jnp.where(mask.any(1)[..., None], kv, 0)
    scores = (
        jnp.einsum(
            "gqhd,gkd->gqhk",
            q.reshape(groups, queries, heads, dim).astype(jnp.float32),
            kv.astype(jnp.float32),
        )
        * dim**-0.5
    )
    scores = jnp.where(mask[:, :, None, :], scores, -jnp.inf)
    maximum = jnp.maximum(scores.max(-1), sink)
    prob = jnp.exp(scores - maximum[..., None])
    denominator = prob.sum(-1) + jnp.exp(sink - maximum)
    expected = (
        jnp.einsum(
            "gqhk,gkd->gqhd",
            prob.astype(jnp.bfloat16),
            kv,
            preferred_element_type=jnp.float32,
        )
        / denominator[..., None]
    )
    np.testing.assert_allclose(
        np.asarray(got, np.float32),
        np.asarray(expected.reshape(q.shape), np.float32),
        atol=3e-3,
        rtol=1e-2,
    )
    np.testing.assert_array_equal(got[-1], 0)
    assert got.shape == q.shape and got.dtype == q.dtype
