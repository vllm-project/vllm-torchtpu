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
"""e4m3 LM head kernel vs the f32 product of the dequantized weight.

``interpret=True`` runs the kernel body on the host; it does not gate Mosaic
lowering.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels import fp8_lm_head as kernel


def _quantize(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.maximum(np.abs(w).max(axis=1, keepdims=True) / kernel.E4M3_MAX, 1e-30)
    return (w / scale).astype(jnp.float8_e4m3fn), scale.reshape(1, -1)


# Rows below / above a row tile, vocab not a multiple of the vocab tile.
@pytest.mark.parametrize("rows,vocab", [(7, 1000), (8, 3600), (600, 1900)])
def test_matches_dequantized_matmul(rows: int, vocab: int) -> None:
    rng = np.random.default_rng(rows + vocab)
    w = (rng.standard_normal((vocab, 256)) * 0.02).astype(np.float32)
    x = jnp.asarray(rng.standard_normal((rows, 256)), jnp.bfloat16)
    q, scale = _quantize(w)
    want = np.asarray(x, np.float32) @ (q.astype(np.float32) * scale.T).T
    got = kernel.fp8_lm_head(x, jnp.asarray(q), jnp.asarray(scale), interpret=True)
    assert got.shape == (rows, vocab) and got.dtype == jnp.bfloat16
    # bf16 output rounding only.
    np.testing.assert_allclose(np.asarray(got, np.float32), want, rtol=1e-2, atol=1e-2)
