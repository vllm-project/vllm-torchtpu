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
"""CPU Pallas interpret and TPU tests for scaled router top-k selection."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels import router_topk


def test_select_scaled_lowest_index_tie_break_and_nan() -> None:
    num_tokens, num_experts, topk = 8, 128, 4
    scores = jnp.ones((num_tokens, num_experts), dtype=jnp.float32) * 0.5
    scores = scores.at[0, 3].set(jnp.nan)
    scale = jnp.arange(1, num_experts + 1, dtype=jnp.bfloat16)

    weights, indices, sel_scales = router_topk.select_scaled(
        scores, scale, topk=topk, interpret=True
    )
    np.testing.assert_array_equal(
        np.asarray(indices[0]), np.array([0, 1, 2, 4], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        np.asarray(indices[1]), np.array([0, 1, 2, 3], dtype=np.int32)
    )
    np.testing.assert_allclose(
        np.asarray(sel_scales[0]), np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float32)
    )
    assert np.all(np.isfinite(np.asarray(weights)))

    dead_scores = scores.at[2, :].set(jnp.nan)
    dead_weights, _, _ = router_topk.select_scaled(
        dead_scores, scale, topk=topk, interpret=True
    )
    assert np.all(np.isnan(np.asarray(dead_weights[2])))
    assert np.all(np.isfinite(np.asarray(dead_weights[:2])))


@pytest.mark.parametrize("topk", [2, 8])
def test_select_scaled_softmax_renorm_and_per_expert_scale(topk: int) -> None:
    num_tokens, num_experts = 16, 128
    key_l, key_s = jax.random.split(jax.random.key(77 + topk))
    probs = jax.nn.softmax(
        jax.random.normal(key_l, (num_tokens, num_experts), dtype=jnp.float32), axis=-1
    )
    per_expert = jax.nn.softplus(
        jax.random.normal(key_s, (num_experts,), dtype=jnp.float32)
    ).astype(jnp.bfloat16)

    weights, indices, sel_scales = router_topk.select_scaled(
        probs, per_expert, topk=topk, interpret=True
    )

    ref_vals, ref_idx = jax.lax.top_k(probs, topk)
    ref_s = jnp.take_along_axis(
        per_expert.astype(jnp.float32)[None, :], ref_idx, axis=-1
    )

    assert weights.dtype == jnp.float32
    assert indices.dtype == jnp.int32
    assert sel_scales.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(indices), np.asarray(ref_idx))
    np.testing.assert_array_equal(np.asarray(weights), np.asarray(ref_vals))
    np.testing.assert_array_equal(np.asarray(sel_scales), np.asarray(ref_s))


@pytest.mark.tpu_test
def test_router_topk_scaled_on_tpu() -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("Requires TPU backend")
    k1, k2 = jax.random.split(jax.random.key(321), 2)
    scores = jax.nn.sigmoid(jax.random.normal(k1, (16, 128), dtype=jnp.float32))
    per_expert = jax.nn.softplus(
        jax.random.normal(k2, (128,), dtype=jnp.float32)
    ).astype(jnp.bfloat16)

    w0, i0 = router_topk.select(scores, topk=8)
    ref_w0, ref_i0 = jax.lax.top_k(scores, 8)
    np.testing.assert_array_equal(np.asarray(i0), np.asarray(ref_i0))
    np.testing.assert_allclose(np.asarray(w0), np.asarray(ref_w0), rtol=1e-6)

    ws, isc, ss = router_topk.select_scaled(scores, per_expert, topk=8)
    np.testing.assert_array_equal(np.asarray(isc), np.asarray(ref_i0))
    np.testing.assert_allclose(np.asarray(ws), np.asarray(ref_w0), rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(ss),
        np.asarray(
            jnp.take_along_axis(
                per_expert.astype(jnp.float32)[None, :], ref_i0, axis=-1
            )
        ),
    )
