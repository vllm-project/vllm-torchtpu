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

from tests.kernels.deepseek_v4.rope_test import (
    HEAD_DIM,
    NUM_HEADS,
    ROTARY_DIM,
    assert_bits_equal,
    make_cos_sin_cache,
)
from vllm_torchtpu.kernels.deepseek_v4.o_projection import cos_sin_table, wo_a_cos_sin
from vllm_torchtpu.kernels.deepseek_v4.q_proj_norm_rope import q_proj_norm_rope
from vllm_torchtpu.kernels.deepseek_v4.rope import qnorm_rope
from vllm_torchtpu.kernels.quantized_matmul.fused_quantize_matmul import (
    fused_quantize_matmul,
)

pytestmark = pytest.mark.tpu_test

# DeepSeek-V4-Flash's wq_b: q_lora_rank 1024 to 64 heads of 512.
Q_LORA_RANK = 1024
EPS = 1e-6
# One bf16 ulp of the normed query, whose rows are O(1).
QUERY_TOL = 2**-7


class QProjNormRopeTest(parameterized.TestCase):
    @parameterized.product(num_tokens=(512, 1024, 2048), return_quantized=(False, True))
    def test_matches_projection_then_qnorm_rope(self, num_tokens, return_quantized):
        """``fused_quantize_matmul`` followed by ``qnorm_rope``: the quantized
        activation bit for bit, the query to within the 1-ulp rms difference
        the two norms' accumulation may round to."""
        rng = np.random.default_rng(0)
        row_scale = np.exp(rng.uniform(-8, 4, size=(num_tokens, 1)))
        x = rng.normal(size=(num_tokens, Q_LORA_RANK)) * row_scale
        x[3] = 0.0  # an all-zero (padding) row
        x = jnp.asarray(x, jnp.bfloat16)
        n = NUM_HEADS * HEAD_DIM
        w_q = jnp.asarray(rng.normal(size=(Q_LORA_RANK, n)) * 0.5, jnp.float8_e4m3fn)
        w_scale = jnp.asarray(rng.uniform(1e-3, 1e-2, size=(n,)), jnp.float32)
        max_position = 4096
        positions = rng.integers(0, max_position, size=(num_tokens,))
        positions[-7:] = 0  # padding rows
        positions = jnp.asarray(positions, jnp.int32)
        cos_sin_cache = jnp.asarray(make_cos_sin_cache(max_position, ROTARY_DIM))
        table = cos_sin_table(cos_sin_cache, inverse=True, num_positions=max_position)

        got = q_proj_norm_rope(
            x,
            w_q,
            w_scale,
            wo_a_cos_sin(positions, table),
            head_dim=HEAD_DIM,
            rotary_dim=ROTARY_DIM,
            eps=EPS,
            return_quantized=return_quantized,
        )
        projected = fused_quantize_matmul(
            x, w_q, w_scale, return_quantized=return_quantized
        )
        if not return_quantized:
            got, projected = (got,), (projected,)
        expected = qnorm_rope(
            projected[0].reshape(num_tokens, NUM_HEADS, HEAD_DIM),
            positions,
            cos_sin_cache,
            eps=EPS,
        ).reshape(num_tokens, n)
        np.testing.assert_allclose(
            np.asarray(got[0], np.float32),
            np.asarray(expected, np.float32),
            rtol=QUERY_TOL,
            atol=QUERY_TOL,
        )
        for g, e in zip(got[1:], projected[1:], strict=True):
            assert_bits_equal(g, e)


if __name__ == "__main__":
    absltest.main()
