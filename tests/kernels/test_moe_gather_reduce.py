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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox.moe_gather_reduce import gather_reduce
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v3 import (
    ragged_gather_reduce as ragged_gather_reduce_v3,
)

pytestmark = pytest.mark.tpu_test


def _reference(x, indices, weights, valid, topk):
    out = x[indices].astype(jnp.float32) * weights.astype(jnp.float32)[:, None]
    out = jnp.where(valid[:, None], out, 0)
    return out.reshape(-1, topk, x.shape[-1]).sum(axis=1).astype(x.dtype)


@pytest.mark.parametrize(
    "num_tokens,topk,hidden,real_tokens",
    [
        (8192, 4, 6144, 8192),
        (8192, 4, 6144, 7484),
        (300, 4, 6144, 300),
        (129, 2, 1024, 100),
        (512, 3, 2048, 512),
    ],
)
def test_gather_reduce_matches_reference(num_tokens, topk, hidden, real_tokens):
    args = _routed_rows(num_tokens, topk, hidden, real_tokens)
    actual = jax.jit(gather_reduce, static_argnames="reduce_group_size")(
        *args, reduce_group_size=topk
    )
    expected = jax.jit(_reference, static_argnames="topk")(*args, topk=topk)

    assert actual.shape == (num_tokens, hidden)
    assert np.isfinite(np.asarray(actual.astype(jnp.float32))).all()
    np.testing.assert_allclose(
        np.asarray(actual.astype(jnp.float32)),
        np.asarray(expected.astype(jnp.float32)),
        rtol=2**-7,
        atol=1e-6,
    )


@pytest.mark.parametrize("real_tokens", [8192, 7484])
def test_gather_reduce_equals_the_sparse_core_combine(real_tokens):
    """The same f32 sum in the same route order as ragged_gather_reduce_v3."""
    args = _routed_rows(8192, 4, 6144, real_tokens, num_experts=128)
    actual = jax.jit(gather_reduce, static_argnames="reduce_group_size")(
        *args, reduce_group_size=4
    )
    expected = jax.jit(ragged_gather_reduce_v3, static_argnames="reduce_group_size")(
        *args, reduce_group_size=4
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def _routed_rows(num_tokens, topk, hidden, real_tokens, num_experts=16):
    rng = np.random.default_rng(0)
    ids = np.stack(
        [rng.choice(num_experts, topk, replace=False) for _ in range(num_tokens)]
    )
    valid = np.zeros((num_tokens, topk), bool)
    valid[:real_tokens] = True
    # Expert-sorted rows, the invalid routes last, as the GMM leaves them.
    order = np.argsort(np.where(valid, ids, num_experts).reshape(-1), kind="stable")
    indices = np.empty_like(order)
    indices[order] = np.arange(order.size)
    num_valid = int(valid.sum())
    x = rng.standard_normal((num_tokens * topk, hidden), dtype=np.float32)
    x = np.array(jnp.asarray(x, jnp.bfloat16))
    # Rows the GMM never wrote must not reach the sum.
    x[num_valid:] = np.nan
    weights = jnp.asarray(rng.uniform(0.05, 0.6, num_tokens * topk), jnp.bfloat16)
    return (
        jnp.asarray(x),
        jnp.asarray(indices, jnp.int32),
        weights,
        jnp.asarray(valid.reshape(-1)),
    )
