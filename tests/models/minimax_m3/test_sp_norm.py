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
"""`sp_norm.add_norm_quant` against model.py's unfused SP chain in jnp."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import sp_norm

EPS = 1e-6


def _reference(shard, update, weight):
    r = (shard.astype(jnp.float32) + update.astype(jnp.float32)).astype(jnp.bfloat16)
    x = r.astype(jnp.float32)
    var = jnp.mean(x * x, axis=-1, keepdims=True)
    normed = (x * jax.lax.rsqrt(var + EPS) * weight).astype(jnp.bfloat16)
    scale = jnp.abs(normed).max(-1, keepdims=True).astype(jnp.float32)
    scale = jnp.maximum(scale, 1e-12) / sp_norm.FP8_MAX
    q = (normed.astype(jnp.float32) / scale).astype(sp_norm.FP8)
    return r, q, scale


@pytest.mark.parametrize("rows,hidden,block", [(256, 512, 128), (64, 384, 128)])
def test_matches_reference(rows, hidden, block):
    rng = np.random.default_rng(rows + hidden)
    shard = jnp.asarray(rng.normal(0, 3, (rows, hidden)), jnp.bfloat16)
    update = jnp.asarray(rng.normal(0, 1, (rows, hidden)), jnp.bfloat16)
    weight = jnp.asarray(1 + rng.normal(0, 0.3, (hidden,)), jnp.float32)
    got = sp_norm.add_norm_quant(
        shard, update, weight, eps=EPS, block_rows=block, interpret=True
    )
    want = _reference(shard, update, weight)
    np.testing.assert_array_equal(
        np.asarray(got[0], np.float32), np.asarray(want[0], np.float32)
    )
    np.testing.assert_allclose(np.asarray(got[2]), np.asarray(want[2]), rtol=1e-6)
    gq, wq = np.asarray(got[1], np.float32), np.asarray(want[1], np.float32)
    # x * (1 / s) may round one e4m3 step away from x / s on a rare element.
    assert np.mean(gq != wq) < 1e-3
    np.testing.assert_allclose(gq, wq, rtol=0.13)
