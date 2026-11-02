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
"""The fused-quantize fp8 matmul must match `xla_quantized_matmul` exactly."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.quantized_matmul import fused_quantize_matmul, util

pytestmark = pytest.mark.tpu_test


# DeepSeek-V4-Flash's per-channel fp8 projections at the prefill buckets.
@pytest.mark.parametrize("m", [512, 1024])
@pytest.mark.parametrize(
    "k,n",
    [
        (4096, 1536),
        (1024, 32768),
        (1024, 8192),
        (8192, 4096),
        (4096, 4096),
        (2048, 4096),
    ],
)
def test_matches_xla_quantized_matmul(m: int, k: int, n: int) -> None:
    kr, kx, kw, ks = jax.random.split(jax.random.key(0), 4)
    row_scale = jnp.exp(jax.random.uniform(kr, (m, 1), minval=-8, maxval=4))
    x = (jax.random.normal(kx, (m, k)) * row_scale).astype(jnp.bfloat16)
    x = x.at[3].set(0)  # a zero row takes scale 1
    w_q = (jax.random.normal(kw, (k, n)) * 0.5).astype(jnp.float8_e4m3fn)
    w_scale = jax.random.uniform(ks, (n,), minval=1e-3, maxval=1e-2)
    assert fused_quantize_matmul.applies(x, w_q, w_scale)

    got = fused_quantize_matmul.fused_quantize_matmul(x, w_q, w_scale)
    expected = util.xla_quantized_matmul(x, w_q, w_scale)
    np.testing.assert_array_equal(
        np.asarray(got.view(jnp.uint16)), np.asarray(expected.view(jnp.uint16))
    )


@pytest.mark.parametrize("k,n", [(1024, 32768), (4096, 1536)])
def test_returns_quantized_activation(k: int, n: int) -> None:
    """The returned activation stands in for `quantize_tensor`'s, so a second
    matmul of it matches `xla_quantized_matmul` too."""
    m = 1024
    kx, kw, ks, kw2, ks2 = jax.random.split(jax.random.key(1), 5)
    x = jax.random.normal(kx, (m, k), jnp.bfloat16).at[3].set(0)
    w_q = (jax.random.normal(kw, (k, n)) * 0.5).astype(jnp.float8_e4m3fn)
    w_scale = jax.random.uniform(ks, (n,), minval=1e-3, maxval=1e-2)
    w_q2 = (jax.random.normal(kw2, (k, 8192)) * 0.5).astype(jnp.float8_e4m3fn)
    w_scale2 = jax.random.uniform(ks2, (8192,), minval=1e-3, maxval=1e-2)

    out, x_q, x_scale = fused_quantize_matmul.fused_quantize_matmul(
        x, w_q, w_scale, return_quantized=True
    )
    expected_x_q, expected_x_scale = util.quantize_tensor(x, w_q.dtype)
    second = util.xla_prequantized_matmul(x_q, x_scale, w_q2, w_scale2, x.dtype)
    pairs = [
        (x_q, expected_x_q),
        (x_scale, expected_x_scale),
        (out, util.xla_quantized_matmul(x, w_q, w_scale)),
        (second, util.xla_quantized_matmul(x, w_q2, w_scale2)),
    ]
    for got, expected in pairs:
        np.testing.assert_array_equal(
            np.asarray(got).view(np.uint8), np.asarray(expected).view(np.uint8)
        )


@pytest.mark.parametrize(
    "m,w_dtype",
    [
        (256, jnp.float8_e4m3fn),  # decode bucket
        (520, jnp.float8_e4m3fn),  # not a whole number of row chunks
        (1024, jnp.bfloat16),  # unquantized weight
    ],
)
def test_keeps_xla_path(m: int, w_dtype: jnp.dtype) -> None:
    x = jnp.zeros((m, 4096), jnp.bfloat16)
    w_q = jnp.zeros((4096, 4096), w_dtype)
    assert not fused_quantize_matmul.applies(x, w_q, jnp.ones((4096,)))
