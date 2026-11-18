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
"""TPU_MOE_ONEHOT_COMBINE_SORTED: the combine built from the sorted token ids
equals the one-hot over [tokens, topk, routes] summed over topk."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.layers.core.fused_moe_gmm import (
    onehot_combine_from_sorted,
    sorted_route_weights,
)

LOCAL_EXPERTS, TOPK, HIDDEN = 16, 8, 128


def _routing(seed, num_tokens, num_padded):
    """Rank-local routing sorted the way prepare_routed_gmm_inputs sorts it:
    stable by local expert, invalid routes (key LOCAL_EXPERTS) last."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(-LOCAL_EXPERTS, LOCAL_EXPERTS, (num_tokens, TOPK))
    weights = (rng.random((num_tokens, TOPK)) + 0.01).astype(np.float32)
    if num_padded:
        weights[-num_padded:] = 0.0
    valid = (ids >= 0) & (weights != 0)
    keys = np.where(valid, ids, LOCAL_EXPERTS).reshape(-1)
    order = np.argsort(keys, kind="stable")
    revert = np.empty_like(order)
    revert[order] = np.arange(order.size)
    token_sorted = (np.arange(order.size) // TOPK)[order]
    w_flat = weights.reshape(-1)
    w_sorted = np.zeros_like(w_flat)
    w_sorted[revert] = w_flat
    gmm = rng.standard_normal((order.size, HIDDEN)).astype(np.float32)
    gmm[int(valid.sum()) :] = 0.0  # rows past the computed prefix are zeroed
    return weights, valid, revert, token_sorted, w_sorted, gmm


def _reference(gmm, revert, weights, valid, num_tokens):
    onehot = jax.nn.one_hot(revert.reshape(num_tokens, TOPK), revert.size)
    combine = (onehot * weights[..., None] * valid[..., None]).sum(axis=1)
    return np.asarray(combine @ gmm, dtype=np.float64)


@pytest.mark.parametrize(
    ("num_tokens", "num_padded"), [(4, 0), (16, 3), (48, 0), (96, 7), (384, 0)]
)
def test_combine_from_sorted_matches_onehot_sum(num_tokens, num_padded):
    weights, valid, revert, token_sorted, w_sorted, gmm = _routing(
        num_tokens, num_tokens, num_padded
    )
    want = _reference(gmm, revert, weights, valid, num_tokens)
    got = onehot_combine_from_sorted(
        jnp.asarray(gmm),
        jnp.asarray(token_sorted, jnp.int32),
        jnp.asarray(w_sorted),
        jnp.asarray([valid.sum()], jnp.int32),
        num_tokens=num_tokens,
    )
    assert got.shape == (num_tokens, HIDDEN)
    np.testing.assert_allclose(np.asarray(got, np.float64), want, rtol=1e-5, atol=1e-5)


def test_positive_control_swapped_tokens_is_caught():
    num_tokens = 16
    weights, valid, revert, token_sorted, w_sorted, gmm = _routing(5, num_tokens, 0)
    want = _reference(gmm, revert, weights, valid, num_tokens)
    bad = token_sorted.copy()
    a = 0
    b = next(i for i in range(1, int(valid.sum())) if bad[i] != bad[a])
    bad[a], bad[b] = bad[b], bad[a]
    got = onehot_combine_from_sorted(
        jnp.asarray(gmm),
        jnp.asarray(bad, jnp.int32),
        jnp.asarray(w_sorted),
        jnp.asarray([valid.sum()], jnp.int32),
        num_tokens=num_tokens,
    )
    assert not np.allclose(np.asarray(got, np.float64), want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("num_tokens", [4, 48, 384])
def test_sorted_route_weights_match_scatter(num_tokens):
    weights, _, revert, _, w_sorted, _ = _routing(num_tokens + 1, num_tokens, 0)
    got = sorted_route_weights(
        jnp.asarray(revert, jnp.int32), jnp.asarray(weights.reshape(-1))
    )
    np.testing.assert_array_equal(np.asarray(got), w_sorted)
