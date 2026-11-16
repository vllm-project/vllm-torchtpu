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
"""quant._e4m3_rows_matmul against the bf16 path it replaces (dequantized rows
times the upcast weight, then the channel scale)."""

import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.models.minimax_m3.quant import _e4m3_rows_matmul


def test_matches_dequantized_bf16_matmul():
    rng = np.random.default_rng(0)
    rows, k, n = 64, 256, 384
    x = rng.normal(0, 1, (rows, k)) * rng.uniform(0.1, 10, (rows, 1))
    row_scale = np.abs(x).max(-1, keepdims=True) / 448.0
    q = jnp.asarray(x / row_scale, jnp.float8_e4m3fn)
    w = jnp.asarray(rng.normal(0, 30, (k, n)), jnp.float8_e4m3fn)
    w_scale = jnp.asarray(2.0 ** rng.integers(-12, -4, (n,)), jnp.bfloat16)
    row_scale = jnp.asarray(row_scale, jnp.float32)

    got = np.asarray(_e4m3_rows_matmul(q, row_scale, w, w_scale), np.float32)
    rows_bf16 = (q.astype(jnp.float32) * row_scale).astype(jnp.bfloat16)
    want = jnp.matmul(rows_bf16, w.astype(jnp.bfloat16)) * w_scale
    want = np.asarray(want, np.float32)
    assert got.shape == (rows, n)
    np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2 * np.abs(want).max())
