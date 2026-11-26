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
"""TPU numerics of the bounded EP one-hot permute/combine against the dense forms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox import moe_onehot_bounded

pytestmark = pytest.mark.tpu_test

HIDDEN, EXPERTS, LOCAL, TOPK = 6144, 128, 16, 4


def _routing(num_tokens, real_tokens, seed):
    """Expert-sorted EP layout as fused_moe_gmm builds it (non-local last)."""
    rng = np.random.default_rng(seed)
    experts = np.stack(
        [rng.choice(EXPERTS, TOPK, replace=False) for _ in range(num_tokens)]
    )
    weights = rng.uniform(0.1, 1.0, (num_tokens, TOPK)).astype(np.float32)
    weights[real_tokens:] = 0
    valid = (experts < LOCAL) & (weights != 0)
    keys = np.where(valid, experts, LOCAL).ravel()
    order = np.argsort(keys, kind="stable")
    revert = np.empty_like(order)
    revert[order] = np.arange(order.size)
    return (
        jnp.asarray(revert, jnp.int32),
        jnp.asarray(np.where(valid, weights, 0), jnp.bfloat16),
        jnp.asarray([valid.sum()], jnp.int32),
    )


def _dense_permute(hidden, revert):
    onehot = jax.nn.one_hot(
        revert.reshape(hidden.shape[0], TOPK), revert.size, dtype=hidden.dtype
    )
    return onehot.sum(axis=1).T @ hidden


def _dense_combine(routed, revert, weights, valid):
    routed = jnp.where((jnp.arange(routed.shape[0]) < valid[0])[:, None], routed, 0)
    onehot = jax.nn.one_hot(
        revert.reshape(weights.shape[0], TOPK), revert.size, dtype=routed.dtype
    )
    return (onehot * weights[..., None]).sum(axis=1) @ routed


@pytest.mark.parametrize(
    "num_tokens,real_tokens",
    # c48/c64 decode buckets off 128 tokens (192, 224, 320, 416, 448) too.
    [
        (256, 192),
        (384, 296),
        (512, 448),
        (192, 184),
        (224, 216),
        (320, 288),
        (416, 400),
        (448, 424),
    ],
)
def test_matches_dense_one_hot(num_tokens, real_tokens):
    revert, weights, valid = _routing(num_tokens, real_tokens, num_tokens)
    hidden = jax.random.normal(jax.random.PRNGKey(0), (num_tokens, HIDDEN)) * 0.1
    hidden = hidden.astype(jnp.bfloat16)
    routed = jax.random.normal(jax.random.PRNGKey(1), (revert.size, HIDDEN)) * 0.1
    # The GMM leaves rows past the valid prefix unwritten.
    routed = jnp.where(
        (jnp.arange(revert.size) < valid[0])[:, None], routed, jnp.nan
    ).astype(jnp.bfloat16)
    n = int(valid[0])

    permuted = moe_onehot_bounded.bounded_onehot_permute(hidden, revert, valid)
    np.testing.assert_array_equal(
        np.asarray(permuted[:n]), np.asarray(_dense_permute(hidden, revert)[:n])
    )
    combined = moe_onehot_bounded.bounded_onehot_combine(routed, revert, weights, valid)
    # Same products, f32 accumulation over fewer (all-zero) slots: equal up to
    # the order of the f32 sum, i.e. at most one bf16 rounding step.
    np.testing.assert_allclose(
        np.asarray(combined, np.float32),
        np.asarray(_dense_combine(routed, revert, weights, valid), np.float32),
        rtol=8e-3,
        atol=1e-3,
    )


def test_nan_token_stays_local_and_empty_prefix_is_zero():
    num_tokens = 256
    revert, weights, valid = _routing(num_tokens, num_tokens, 7)
    hidden = jnp.ones((num_tokens, HIDDEN), jnp.bfloat16).at[3].set(jnp.nan)
    permuted = moe_onehot_bounded.bounded_onehot_permute(hidden, revert, valid)
    assert np.isfinite(np.asarray(permuted[: int(valid[0])], np.float32)).all()

    routed = jnp.full((revert.size, HIDDEN), jnp.nan, jnp.bfloat16)
    empty = jnp.asarray([0], jnp.int32)
    combined = moe_onehot_bounded.bounded_onehot_combine(
        routed, revert, jnp.zeros_like(weights), empty
    )
    np.testing.assert_array_equal(np.asarray(combined, np.float32), 0.0)


@pytest.mark.parametrize("num_tokens,real_tokens", [(256, 216), (384, 368)])
def test_shared_down_matches_separate_add(num_tokens, real_tokens):
    revert, weights, valid = _routing(num_tokens, real_tokens, num_tokens + 1)
    routed = jax.random.normal(jax.random.PRNGKey(2), (revert.size, HIDDEN)) * 0.1
    routed = jnp.where(
        (jnp.arange(revert.size) < valid[0])[:, None], routed, jnp.nan
    ).astype(jnp.bfloat16)
    act = (jax.random.normal(jax.random.PRNGKey(3), (num_tokens, 384)) * 0.5).astype(
        jnp.bfloat16
    )
    w2 = (jax.random.normal(jax.random.PRNGKey(4), (384, HIDDEN)) * 2).astype(
        jnp.float8_e4m3fn
    )
    s2 = jax.random.uniform(jax.random.PRNGKey(5), (1, HIDDEN), jnp.float32) * 0.01

    fused = moe_onehot_bounded.bounded_onehot_combine(
        routed, revert, weights, valid, (act, w2, s2)
    )
    plain = moe_onehot_bounded.bounded_onehot_combine(routed, revert, weights, valid)
    # One rounding of the f32 sum, as XLA compiles the separate add
    # (fused_moe_gmm._add_shared_down) in the decode graph.
    shared = jnp.matmul(
        act, w2.astype(jnp.bfloat16), preferred_element_type=jnp.float32
    )
    expected = (plain.astype(jnp.float32) + shared * s2).astype(jnp.bfloat16)
    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(fused, jnp.int16)),
        np.asarray(jax.lax.bitcast_convert_type(expected, jnp.int16)),
    )
