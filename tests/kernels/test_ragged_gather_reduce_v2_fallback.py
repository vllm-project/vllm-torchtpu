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
"""TensorCore fallback of ragged_gather_reduce_v2 (the EP MoE combine at decode)."""

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2.wrapper import (
    _fallback_implementation,
)

TOPK = 10


def _decode_combine_inputs(num_tokens=96, hidden=512, seed=0):
    """An EP decode combine: each token keeps 0-4 of its TOPK routes on this
    rank, the GMM rows past the routed prefix are stale (NaN here)."""
    rng = np.random.default_rng(seed)
    routes = num_tokens * TOPK
    valid = np.zeros((num_tokens, TOPK), bool)
    for t in range(num_tokens):
        valid[t, rng.choice(TOPK, size=rng.integers(0, 5), replace=False)] = True
    valid = valid.reshape(-1)
    # Valid routes read the computed prefix, the others the stale tail.
    indices = np.empty(routes, np.int32)
    num_valid = int(valid.sum())
    indices[valid] = rng.permutation(num_valid)
    indices[~valid] = num_valid + rng.permutation(routes - num_valid)
    # 8-bit values times 4-bit weights: most products need more bits than
    # bf16 keeps, while a sum of TOPK of them still fits f32 exactly, so the
    # one rounding is the final one and the result is exact on any backend.
    x = rng.integers(128, 256, size=(routes, hidden)) * rng.choice(
        [-1, 1], (routes, hidden)
    )
    x = (x * np.exp2(rng.integers(-3, 1, size=(routes, 1)))).astype(np.float32)
    x = jnp.asarray(x, jnp.bfloat16).at[num_valid:].set(jnp.nan)
    weights = jnp.asarray(rng.integers(1, 16, size=routes) / 16.0, jnp.bfloat16)
    return x, jnp.asarray(indices), weights, jnp.asarray(valid)


def _round_to_bf16(v):
    """float64 -> nearest bfloat16 (ties to even), as float64."""
    m, e = np.frexp(v)
    return np.ldexp(np.rint(np.ldexp(m, 8)), e - 8)


def test_fallback_weighted_sum_is_one_matmul():
    # Written as an f32 multiply and a sum over the routes, XLA may keep the
    # products in bf16 when the sum is rounded to bf16 right after; the
    # served decode graph does, which rounds every expert term. A matmul
    # accumulates inside the MXU and rounds once.
    x, indices, weights, valid = _decode_combine_inputs()
    jaxpr = jax.make_jaxpr(_fallback_implementation, static_argnums=4)(
        x, indices, weights, valid, TOPK
    )
    prims = [eqn.primitive.name for eqn in jaxpr.jaxpr.eqns]
    assert "dot_general" in prims
    assert "reduce_sum" not in prims


def test_fallback_matches_correctly_rounded_sum():
    x, indices, weights, valid = _decode_combine_inputs()
    out = np.asarray(
        jax.jit(_fallback_implementation, static_argnums=4)(
            x, indices, weights, valid, TOPK
        ).astype(jnp.float32)
    )
    xs = np.asarray(x.astype(jnp.float32)).astype(np.float64)
    w = np.asarray(weights.astype(jnp.float32)).astype(np.float64)
    terms = np.where(valid.reshape(-1, 1), xs[np.asarray(indices)] * w[:, None], 0.0)
    expected = _round_to_bf16(terms.reshape(-1, TOPK, xs.shape[1]).sum(axis=1))
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out, expected)
