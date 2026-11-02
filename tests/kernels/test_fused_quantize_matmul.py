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
