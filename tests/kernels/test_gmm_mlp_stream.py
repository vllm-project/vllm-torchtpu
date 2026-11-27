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
"""Tests for the streamed decode expert MLP (gmm_mlp_stream)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox.gmm_mlp_stream import gmm_mlp_stream, supported
from vllm_torchtpu.kernels.megablox.gmm_v2 import gmm_v2_mlp

_HIDDEN, _INTER, _GROUPS, _BLOCK = 1024, 512, 16, 128
_FP4 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6, 0]


def _require_tpu() -> None:
    if jax.default_backend() != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {jax.default_backend()}.")


def _operands(batch_size: int):
    keys = jax.random.split(jax.random.key(0), 5)
    values = jnp.asarray(_FP4, jnp.float32)

    def fp4(key, k, n):
        idx = jax.random.randint(key, (_GROUPS, k, n), 0, 16)
        return values[idx].astype(jnp.float4_e2m1fn)

    w1 = fp4(keys[0], _HIDDEN, 2 * _INTER)
    w2 = fp4(keys[1], _INTER, _HIDDEN)
    s1 = jax.random.uniform(
        keys[2], (_GROUPS, _HIDDEN // _BLOCK, 1, 2 * _INTER), jnp.float32, 0.5, 2
    )
    s2 = jax.random.uniform(
        keys[3], (_GROUPS, _INTER // _BLOCK, 1, _HIDDEN), jnp.float32, 0.5, 2
    )
    lhs = jax.random.normal(keys[4], (batch_size, _HIDDEN), jnp.bfloat16)
    return lhs, w1, s1, w2, s2


def _reference(lhs, w1, s1, w2, s2, group_sizes):
    """float64 expert MLP on dequantized weights; the activation rounded to
    bf16 as both kernels store it."""
    x = np.asarray(lhs.astype(jnp.float32), np.float64)
    w1d = np.asarray(w1.astype(jnp.float32), np.float64) * np.repeat(
        np.asarray(s1)[:, :, 0], _BLOCK, axis=1
    )
    w2d = np.asarray(w2.astype(jnp.float32), np.float64) * np.repeat(
        np.asarray(s2)[:, :, 0], _BLOCK, axis=1
    )
    out, off = [], 0
    for e, n in enumerate(group_sizes):
        gu = x[off : off + n] @ w1d[e]
        off += n
        gate = np.minimum(gu[:, :_INTER], 7.0)
        up = np.clip(gu[:, _INTER:], -7.0, 7.0)
        act = (up + 1.0) * gate / (1.0 + np.exp(-1.702 * gate))
        act = np.asarray(jnp.asarray(act, jnp.float32).astype(jnp.bfloat16), np.float64)
        out.append(act @ w2d[e])
    return np.concatenate(out)


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "batch_size, group_sizes",
    [
        (32, [0, 2, 0, 1, 0, 0, 3, 0, 0, 0, 20, 0, 0, 1, 0, 0]),
        (128, [3, 0, 14, 0, 1, 30, 0, 0, 33, 0, 0, 0, 2, 0, 0, 1]),
        (384, [9, 12, 0, 17, 30, 0, 31, 2, 0, 0, 20, 0, 13, 0, 0, 5]),
    ],
)
def test_matches_gmm_v2_mlp_on_fp8_buckets(batch_size, group_sizes):
    """Where every group's rows fit gmm_v2's fp8-fed decode buckets (<= 48
    rows from the group's sublane tile), the stream equals gmm_v2_mlp bit for
    bit."""
    _require_tpu()
    lhs, w1, s1, w2, s2 = _operands(batch_size)
    assert supported(lhs, w1, s1, w2, s2, "swigluoai")
    group_sizes = jnp.asarray(group_sizes, jnp.int32)
    rows = int(group_sizes.sum())
    expected = gmm_v2_mlp(
        lhs,
        w1,
        w2,
        group_sizes,
        rhs1_scale=s1,
        rhs2_scale=s2,
        fuse_act="swigluoai",
        preferred_element_type=jnp.bfloat16,
        maybe_quantize_lhs=False,
    )
    actual = gmm_mlp_stream(lhs, w1, w2, group_sizes, s1, s2, fuse_act="swigluoai")
    np.testing.assert_array_equal(
        np.asarray(actual[:rows].astype(jnp.float32)),
        np.asarray(expected[:rows].astype(jnp.float32)),
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "batch_size, group_sizes",
    [
        # Groups spanning several 48-row windows and several passes.
        (384, [9, 60, 0, 17, 130, 0, 31, 2, 0, 0, 20, 0, 47, 0, 0, 5]),
        (512, [0] * 15 + [301]),
        # No local rows.
        (32, [0] * 16),
    ],
)
def test_matches_reference(batch_size, group_sizes):
    _require_tpu()
    lhs, w1, s1, w2, s2 = _operands(batch_size)
    rows = sum(group_sizes)
    actual = gmm_mlp_stream(
        lhs, w1, w2, jnp.asarray(group_sizes, jnp.int32), s1, s2, fuse_act="swigluoai"
    )
    if rows == 0:
        return
    expected = _reference(lhs, w1, s1, w2, s2, group_sizes)
    got = np.asarray(actual[:rows].astype(jnp.float32), np.float64)
    err = np.abs(got - expected).max() / np.abs(expected).max()
    assert err < 1e-2, err


@pytest.mark.tpu_test
@pytest.mark.parametrize("num_tokens, topk", [(256, 4), (384, 4), (64, 8)])
def test_gather_matches_permute_first(num_tokens, topk):
    """With ``revert`` the stream gathers each routed row from the tokens
    itself: bit-identical to permuting the rows first."""
    _require_tpu()
    tokens, w1, s1, w2, s2 = _operands(num_tokens)
    # Non-finite values are zeroed as the one-hot permute zeroes them.
    tokens = tokens.at[3, 5].set(jnp.nan).at[4, 7].set(-jnp.inf).at[5, 9].set(jnp.inf)
    rng = np.random.default_rng(num_tokens)
    experts = np.stack(
        [rng.choice(4 * _GROUPS, topk, replace=False) for _ in range(num_tokens)]
    )
    keys = np.where(experts < _GROUPS, experts, _GROUPS).ravel()
    order = np.argsort(keys, kind="stable")
    revert = np.empty_like(order)
    revert[order] = np.arange(order.size)
    group_sizes = jnp.asarray(np.bincount(keys, minlength=_GROUPS + 1)[:_GROUPS])
    rows = int(group_sizes.sum())
    permuted = jnp.where(jnp.isfinite(tokens), tokens, 0)[order // topk]
    expected = gmm_mlp_stream(
        permuted, w1, w2, group_sizes, s1, s2, fuse_act="swigluoai"
    )
    actual = gmm_mlp_stream(
        tokens,
        w1,
        w2,
        group_sizes,
        s1,
        s2,
        fuse_act="swigluoai",
        revert=jnp.asarray(revert, jnp.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(actual[:rows].astype(jnp.float32)),
        np.asarray(expected[:rows].astype(jnp.float32)),
    )
