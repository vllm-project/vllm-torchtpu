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
"""weighted_sum: a route of weight zero contributes nothing, whatever its
gathered row holds."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest

from vllm_torchtpu.kernels.fused_moe.v3 import combine

pytestmark = pytest.mark.multichip


class CombineTest(absltest.TestCase):
    def setUp(self):
        if jax.devices()[0].platform != "tpu":
            self.skipTest("the combine runs on TPU only")

    def test_a_zero_weight_route_contributes_nothing(self):
        tokens, top_k, lane_blocks = 128, 4, 8
        lanes = combine.LANES
        hidden = lane_blocks * lanes
        rows = jax.random.normal(
            jax.random.PRNGKey(0), (tokens, top_k, lane_blocks, lanes), jnp.float32
        )
        weights = jax.random.uniform(
            jax.random.PRNGKey(1), (tokens, top_k), jnp.float32, 0.1, 1.0
        )
        # Slot 1 of every token: weight zero, and a row of NaN and Inf as a
        # row nobody wrote might hold.
        weights = weights.at[:, 1].set(0.0)
        poison = jnp.where(jnp.arange(lanes) % 2 == 0, jnp.nan, jnp.inf)
        rows = rows.at[:, 1].set(poison).astype(jnp.bfloat16)
        out = combine.weighted_sum(
            rows.reshape(tokens * top_k, lane_blocks, lanes),
            None,
            weights,
            tokens_per_shard=tokens,
            hidden=hidden,
            out_dtype=jnp.float32,
        )
        clean = rows.astype(jnp.float32).at[:, 1].set(0.0)
        expected = jnp.einsum("tkbl,tk->tbl", clean, weights).reshape(tokens, hidden)
        self.assertTrue(bool(jnp.isfinite(out).all()))
        np.testing.assert_allclose(
            np.asarray(out), np.asarray(expected), rtol=1e-2, atol=1e-2
        )

    def test_a_zero_weight_route_with_an_unwritten_scale_contributes_nothing(self):
        tokens, top_k, lane_blocks = 128, 4, 8
        lanes = combine.LANES
        hidden = lane_blocks * lanes
        rows = jax.random.normal(
            jax.random.PRNGKey(0), (tokens, top_k, lane_blocks, lanes), jnp.float32
        ).astype(jnp.float8_e4m3fn)
        scales = jax.random.uniform(
            jax.random.PRNGKey(2), (tokens, top_k), jnp.float32, 0.5, 2.0
        )
        weights = jax.random.uniform(
            jax.random.PRNGKey(1), (tokens, top_k), jnp.float32, 0.1, 1.0
        )
        # Slot 1 of every token: weight zero, and a NaN or Inf scale as an
        # fp8 result row's scale nobody wrote might hold.
        weights = weights.at[:, 1].set(0.0)
        poison = jnp.where(jnp.arange(tokens) % 2 == 0, jnp.nan, jnp.inf)
        scales = scales.at[:, 1].set(poison)
        out = combine.weighted_sum(
            rows.reshape(tokens * top_k, lane_blocks, lanes),
            scales.reshape(tokens * top_k),
            weights,
            tokens_per_shard=tokens,
            hidden=hidden,
            out_dtype=jnp.float32,
        )
        coef = jnp.where(weights != 0, weights * scales, 0.0)
        expected = jnp.einsum("tkbl,tk->tbl", rows.astype(jnp.float32), coef)
        self.assertTrue(bool(jnp.isfinite(out).all()))
        np.testing.assert_allclose(
            np.asarray(out),
            np.asarray(expected.reshape(tokens, hidden)),
            rtol=3e-2,
            atol=3e-2,
        )


if __name__ == "__main__":
    absltest.main()
