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
"""Tests for optimization_barrier in quantize_block."""

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.quantized_matmul.util import get_max_min, quantize_block


def _ref_quantize_block_without_barrier(data, axis, target_dtype):
    abs_max = jnp.max(jnp.abs(data), axis=axis, keepdims=True)
    dtype_max, dtype_min = get_max_min(target_dtype)
    scale = abs_max / dtype_max
    scale = jnp.where(scale == 0, 1.0, scale)
    if jnp.issubdtype(target_dtype, jnp.floating):
        data_q = (data / scale).clip(dtype_min, dtype_max).astype(target_dtype)
    else:
        data_q = jnp.round(data / scale).astype(target_dtype)
    return data_q, scale


def test_quantize_block_emits_optimization_barrier_and_preserves_bits():
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        rng = np.random.default_rng(0)
        raw = rng.standard_normal((16, 128), dtype=np.float32) * 12.0
        raw[0, :] = 0.0
        raw[1, 0] = 1000.0
        raw[1, 1] = -1000.0
        x = jnp.asarray(raw, dtype=jnp.bfloat16)

        jaxpr = jax.make_jaxpr(lambda inp: quantize_block(inp, -1, jnp.float8_e4m3fn))(
            x
        )
        assert "optimization_barrier" in str(jaxpr)

        q_actual, s_actual = quantize_block(x, -1, jnp.float8_e4m3fn)
        q_ref, s_ref = _ref_quantize_block_without_barrier(x, -1, jnp.float8_e4m3fn)
        np.testing.assert_array_equal(
            np.asarray(q_actual.astype(jnp.float32)),
            np.asarray(q_ref.astype(jnp.float32)),
        )
        np.testing.assert_array_equal(np.asarray(s_actual), np.asarray(s_ref))

        qi_actual, si_actual = quantize_block(x, -1, jnp.int8)
        qi_ref, si_ref = _ref_quantize_block_without_barrier(x, -1, jnp.int8)
        np.testing.assert_array_equal(np.asarray(qi_actual), np.asarray(qi_ref))
        np.testing.assert_array_equal(np.asarray(si_actual), np.asarray(si_ref))
