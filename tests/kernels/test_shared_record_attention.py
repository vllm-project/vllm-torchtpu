# SPDX-License-Identifier: Apache-2.0
"""Validate pair output packing, partial windows and the attention sink."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.shared_record_attention import shared_record_attention


def _pieces(x):
    return jnp.stack(
        [
            x[..., start + parity : start + 256 : 2]
            for start in (0, 256)
            for parity in (0, 1)
        ]
    )


@pytest.mark.parametrize("valid_keys", [0, 1, 133, 192])
@pytest.mark.parametrize("enable_x64", [False, True])
def test_pair_matches_dense_with_sink(valid_keys, enable_x64):
    if jax.default_backend() != "tpu":
        pytest.skip("Compiled MXU kernel requires TPU")
    with jax.enable_x64(enable_x64):
        _check_pair_with_sink(valid_keys)


def _check_pair_with_sink(valid_keys):
    rng = np.random.default_rng(23)
    q = jnp.asarray(rng.normal(size=(2, 320, 512)), jnp.bfloat16)
    kv = jnp.asarray(rng.normal(size=(2, 192, 512)), jnp.bfloat16)
    valid = jnp.arange(192)[None, :] < jnp.asarray([valid_keys, 1])[:, None]
    kv = jnp.where(valid[..., None], kv, 0)
    sink = jnp.linspace(-1, 1, 320, dtype=jnp.float32)[:, None]
    mask = jnp.pad(valid.astype(jnp.int32), ((0, 0), (0, 64)))[:, None, :]
    actual = jax.jit(shared_record_attention)(mask, _pieces(q), _pieces(kv), sink)
    scores = (
        jnp.einsum("rqd,rkd->rqk", q.astype(jnp.float32), kv.astype(jnp.float32))
        * 512**-0.5
    )
    scores = jnp.where(valid[:, None, :], scores, -jnp.inf)
    maximum = jnp.maximum(scores.max(-1, keepdims=True), -1e30)
    probability = jnp.exp(scores - maximum)
    denominator = probability.sum(-1, keepdims=True) + jnp.exp(sink[None] - maximum)
    expected = (
        jnp.einsum(
            "rqk,rkd->rqd",
            probability.astype(jnp.bfloat16),
            kv,
            preferred_element_type=jnp.float32,
        )
        / denominator
    ).astype(jnp.bfloat16)
    np.testing.assert_allclose(
        actual.astype(jnp.float32), expected.astype(jnp.float32), atol=0.01, rtol=0.01
    )
    if valid_keys == 0:
        np.testing.assert_array_equal(actual[0], jnp.zeros_like(actual[0]))
