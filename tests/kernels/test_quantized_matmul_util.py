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
"""The per-channel W8A8 matmul split into its activation quant and its matmul."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_tensor,
    xla_prequantized_matmul,
    xla_quantized_matmul,
)


@pytest.mark.parametrize("w_dtype", [jnp.float8_e4m3fn, jnp.int8])
def test_split_matmul_is_bitexact(w_dtype) -> None:
    """Quantizing rows apart from the matmul (e.g. before a gather) changes no bit."""
    x = jax.random.normal(jax.random.key(0), (48, 256), jnp.float32)
    x = (x * jnp.linspace(0.01, 30.0, 48)[:, None]).astype(jnp.bfloat16)
    w = jax.random.normal(jax.random.key(1), (256, 96), jnp.float32) * 40.0
    w_q = jnp.clip(w, -127, 127).astype(w_dtype)
    w_s = jax.random.uniform(jax.random.key(2), (96,), jnp.float32) * 0.01

    fused = jax.jit(xla_quantized_matmul)(x, w_q, w_s)

    def split(x, w_q, w_s):
        x_q, x_scale = quantize_tensor(x, w_q.dtype)
        # Row blocks quantized separately and concatenated, as a gather does.
        parts = [quantize_tensor(p, w_q.dtype) for p in jnp.split(x, 4)]
        x_q_cat = jnp.concatenate([p[0] for p in parts])
        x_scale_cat = jnp.concatenate([p[1] for p in parts])
        assert x_q_cat.dtype == x_q.dtype and x_scale_cat.dtype == jnp.float32
        return xla_prequantized_matmul(x_q_cat, x_scale_cat, w_q, w_s, x.dtype)

    out = jax.jit(split)(x, w_q, w_s)
    assert out.dtype == fused.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(out), np.asarray(fused))
