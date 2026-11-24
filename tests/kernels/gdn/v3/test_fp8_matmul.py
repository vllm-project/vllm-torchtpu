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
"""CPU unit tests for GDN FP8 matmul and power-of-two scaling helpers."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.gdn.v3 import fp8_matmul


def _rel_l2(ref: jax.Array, got: jax.Array) -> float:
    r = np.asarray(ref, dtype=np.float32)
    y = np.asarray(got, dtype=np.float32)
    return float(np.linalg.norm(r - y) / (np.linalg.norm(r) + 1e-12))


class Fp8MatmulTest(parameterized.TestCase):
    @parameterized.parameters(16.0, 256.0, 300.0)
    def test_col_pow2_scale_exact_inverse_and_bound(self, max_abs: float):
        rng = np.random.default_rng(7)
        x = rng.standard_normal((64, 128)).astype(np.float32)
        x[:, 0] *= 1000.0
        x[:, 1] *= 1e-4
        x[:, 2] = 0.0
        x[:, 3] = max_abs
        x_bf16 = jnp.asarray(x, dtype=jnp.bfloat16)

        scale, inv_scale = jax.jit(
            fp8_matmul.compute_column_pow2_scales, static_argnames=("max_abs",)
        )(x_bf16, max_abs=max_abs)
        np.testing.assert_array_equal(
            np.asarray(scale * inv_scale, dtype=np.float32),
            np.ones((1, 128), dtype=np.float32),
        )
        scaled = x_bf16 * inv_scale
        self.assertLess(float(jnp.max(jnp.abs(scaled))), max_abs)
        np.testing.assert_array_equal(
            np.asarray(scaled * scale, dtype=np.float32),
            np.asarray(x_bf16, dtype=np.float32),
        )

    def test_pow2_scale_validation_errors(self):
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            fp8_matmul.compute_column_pow2_scales(
                jnp.ones((4, 4), dtype=jnp.float16), max_abs=16.0
            )
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            fp8_matmul.pow2_scales_from_amax(
                jnp.ones((1, 4), dtype=jnp.float32), max_abs=16.0
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            fp8_matmul.compute_column_pow2_scales(
                jnp.ones((4, 4), dtype=jnp.bfloat16), max_abs=0.0
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            fp8_matmul.pow2_scales_from_amax(
                jnp.ones((1, 4), dtype=jnp.bfloat16), max_abs=-1.0
            )

    @parameterized.parameters(
        (None, 5e-3),
        (jnp.float8_e4m3fn, 6e-3),
        (jnp.float8_e5m2, 1.5e-2),
    )
    def test_split_high_low_and_dot_match_float32(self, fp8_dtype, max_rel):
        rng = np.random.default_rng(11)
        lhs = jnp.asarray(rng.standard_normal((64, 128)), dtype=jnp.float32)
        rhs = jnp.asarray(rng.standard_normal((128, 64)), dtype=jnp.float32)
        dt = None if fp8_dtype is None else jnp.dtype(fp8_dtype)
        got = fp8_matmul.dot_general(
            fp8_matmul.split_high_low(lhs, dt),
            fp8_matmul.split_high_low(rhs, dt, truncate_e4m3=True),
            (((1,), (0,)), ((), ())),
        )
        ref = jnp.matmul(lhs, rhs, precision=jax.lax.Precision.HIGHEST)
        self.assertLessEqual(_rel_l2(ref, got), max_rel)

    def test_dot_general_operand_combinations_and_validation(self):
        rng = np.random.default_rng(13)
        lhs = jnp.asarray(rng.standard_normal((32, 64)), dtype=jnp.float32)
        rhs = jnp.asarray(rng.standard_normal((64, 32)), dtype=jnp.float32)
        dims = (((1,), (0,)), ((), ()))
        ref = jnp.matmul(lhs, rhs, precision=jax.lax.Precision.HIGHEST)
        dt = jnp.dtype(jnp.float8_e4m3fn)
        lhs_fp8 = fp8_matmul.split_high_low(lhs, dt)
        rhs_fp8 = fp8_matmul.split_high_low(rhs, dt)

        got_bare = fp8_matmul.dot_general(
            lhs.astype(jnp.bfloat16), rhs.astype(jnp.bfloat16), dims
        )
        self.assertLessEqual(_rel_l2(ref, got_bare), 5e-3)

        got_12 = fp8_matmul.dot_general(lhs.astype(jnp.bfloat16), rhs_fp8, dims)
        self.assertLessEqual(_rel_l2(ref, got_12), 6e-3)

        got_21 = fp8_matmul.dot_general(lhs_fp8, rhs.astype(jnp.bfloat16), dims)
        self.assertLessEqual(_rel_l2(ref, got_21), 6e-3)

        with self.assertRaises(ValueError):
            fp8_matmul.dot_general([], rhs_fp8, dims)
        with self.assertRaises(ValueError):
            fp8_matmul.dot_general(lhs_fp8 + [lhs_fp8[0]], rhs_fp8, dims)


if __name__ == "__main__":
    absltest.main()
