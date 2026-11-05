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

import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized

from tests.kernels.deepseek_v4.rope_test import assert_bits_equal, make_cos_sin_cache
from vllm_torchtpu.kernels.deepseek_v4.indexer_q_proj import q_proj_rope_quant
from vllm_torchtpu.kernels.deepseek_v4.rope import rope_quant
from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_tensor,
    xla_prequantized_matmul,
)

pytestmark = pytest.mark.tpu_test

# DeepSeek-V4-Flash's indexer: q_lora_rank 1024, 64 heads of 128, rope dim 64.
Q_LORA_RANK = 1024
NUM_HEADS = 64
HEAD_DIM = 128
ROTARY_DIM = 64


class QProjRopeQuantTest(parameterized.TestCase):
    @parameterized.parameters(512, 1024, 1536)
    def test_matches_projection_then_rope_quant(self, num_tokens):
        """Bit for bit the XLA projection followed by ``rope_quant``."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=(num_tokens, Q_LORA_RANK)).astype(np.float32)
        x[3] = 0.0  # an all-zero (padding) row
        x_q, x_scale = quantize_tensor(jnp.asarray(x, jnp.bfloat16), jnp.float8_e4m3fn)
        n = NUM_HEADS * HEAD_DIM
        w_q = jnp.asarray(rng.normal(size=(Q_LORA_RANK, n)) * 0.05, jnp.float8_e4m3fn)
        w_scale = jnp.asarray(rng.uniform(1e-3, 1e-2, size=(n,)), jnp.float32)
        max_position = 4096
        positions = jnp.asarray(
            rng.integers(0, max_position, size=(num_tokens,)), jnp.int32
        )
        cos_sin_cache = jnp.asarray(make_cos_sin_cache(max_position, ROTARY_DIM))

        q, scales = q_proj_rope_quant(
            x_q, x_scale, w_q, w_scale, positions, cos_sin_cache
        )
        projected = xla_prequantized_matmul(
            x_q, x_scale, w_q, w_scale, jnp.bfloat16
        ).reshape(num_tokens, NUM_HEADS, HEAD_DIM)
        q_expected, scales_expected = rope_quant(
            projected, positions, cos_sin_cache, quant_dtype=jnp.float8_e4m3fn
        )
        assert_bits_equal(q, q_expected)
        assert_bits_equal(scales, scales_expected)


if __name__ == "__main__":
    absltest.main()
