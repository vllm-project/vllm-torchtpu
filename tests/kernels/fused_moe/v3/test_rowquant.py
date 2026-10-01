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
"""quantize_rows: exact scales, no fp8 overflow, the documented flushes,
and the intermediate requantization every weight format's body goes
through."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.fused_moe.v3 import ffn
from vllm_torchtpu.kernels.fused_moe.v3.formats import WeightFormat
from vllm_torchtpu.kernels.fused_moe.v3.reference import (
    FLOAT_GRIDS,
    float_grid_rows,
    fp4_block_rows,
    fused_ep_moe_reference,
)
from vllm_torchtpu.kernels.fused_moe.v3.rowquant import FP8_MAX, quantize_rows

pytestmark = pytest.mark.cpu_test

# Tolerance for a row-scaled intermediate value against the value it came
# from, as a share of its row's largest. e4m3 carries three mantissa bits.
MID_QUANT_ROW_BOUND = 0.07


def reference_scale(x_bf16):
    x32 = x_bf16.astype(jnp.float32)
    return jnp.max(jnp.abs(x32), axis=-1, keepdims=True) / FP8_MAX


class QuantizeRowsTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("narrow", 8, 128), ("served_row", 16, 4096), ("wide_range", 32, 512)
    )
    def test_scale_is_exact_against_an_f32_reduce(self, rows, cols):
        rng = np.random.default_rng(12)
        magnitudes = np.exp(rng.uniform(-12, 12, size=(rows, 1)))
        x = jnp.asarray(
            (rng.normal(size=(rows, cols)) * magnitudes).astype(np.float32)
        ).astype(jnp.bfloat16)
        _, scale = quantize_rows(x)
        np.testing.assert_array_equal(
            np.asarray(scale.view(jnp.int32)),
            np.asarray(reference_scale(x).view(jnp.int32)),
        )

    def test_a_block_is_quantized_as_a_row_of_its_own(self):
        """With a block, each block of a row carries the scale the whole-row
        rule would give that block on its own, value for value."""
        x = jax.random.normal(jax.random.key(3), (16, 512), jnp.float32).astype(
            jnp.bfloat16
        )
        q, s = quantize_rows(x, block=128)
        self.assertEqual(s.shape, (16, 4))
        for b in range(4):
            q_b, s_b = quantize_rows(x[:, b * 128 : (b + 1) * 128])
            np.testing.assert_array_equal(
                np.asarray(q[:, b * 128 : (b + 1) * 128]), np.asarray(q_b)
            )
            np.testing.assert_array_equal(np.asarray(s[:, b : b + 1]), np.asarray(s_b))
        with self.assertRaisesRegex(ValueError, "whole number"):
            quantize_rows(x, block=384)

    def test_a_zero_row_has_a_zero_scale_and_stays_zero(self):
        quantized, scale = quantize_rows(jnp.zeros((4, 128), jnp.bfloat16))
        np.testing.assert_array_equal(np.asarray(scale), np.zeros((4, 1), np.float32))
        np.testing.assert_array_equal(
            np.asarray(quantized.astype(jnp.float32)), np.zeros((4, 128), np.float32)
        )

    def test_no_fp8_overflow_through_the_bf16_inverse_scale(self):
        rng = np.random.default_rng(13)
        for _ in range(64):
            magnitude = np.exp(rng.uniform(-20, 20, size=(1, 1)))
            x = jnp.asarray(
                (rng.normal(size=(1, 256)) * magnitude).astype(np.float32)
            ).astype(jnp.bfloat16)
            quantized, _ = quantize_rows(x)
            upcast = np.asarray(quantized.astype(jnp.float32))
            self.assertTrue(np.isfinite(upcast).all())
            self.assertLessEqual(float(np.abs(upcast).max()), FP8_MAX)

    def test_no_fp8_overflow_on_rows_whose_maximum_sits_at_the_boundary(self):
        peaks = np.unique(
            np.asarray(
                jnp.asarray(np.linspace(440.0, 456.0, 2048, dtype=np.float32))
                .astype(jnp.bfloat16)
                .astype(jnp.float32)
            )
        )
        rng = np.random.default_rng(14)
        for peak in peaks:
            row = (rng.random((1, 128)) * peak).astype(np.float32)
            row[0, 0] = peak
            quantized, scale = quantize_rows(jnp.asarray(row).astype(jnp.bfloat16))
            upcast = np.asarray(quantized.astype(jnp.float32))
            self.assertTrue(np.isfinite(upcast).all(), f"peak {peak}")
            self.assertLessEqual(float(np.abs(upcast).max()), FP8_MAX)
            self.assertGreater(float(scale[0, 0]), 0.0)

    def test_a_non_finite_row_is_zeroed_whole_and_stays_local(self):
        for seed, bad in ((15, np.inf), (16, -np.inf), (17, np.nan)):
            rng = np.random.default_rng(seed)
            x = rng.normal(size=(4, 128)).astype(np.float32)
            clean_q, clean_s = quantize_rows(jnp.asarray(x).astype(jnp.bfloat16))
            poisoned = x.copy()
            poisoned[2, 5] = bad
            quantized, scale = quantize_rows(jnp.asarray(poisoned).astype(jnp.bfloat16))
            upcast = np.asarray(quantized.astype(jnp.float32))
            self.assertEqual(float(scale[2, 0]), 0.0)
            np.testing.assert_array_equal(upcast[2], np.zeros(128, np.float32))
            keep = [0, 1, 3]
            np.testing.assert_array_equal(
                upcast[keep], np.asarray(clean_q.astype(jnp.float32))[keep]
            )
            np.testing.assert_array_equal(
                np.asarray(scale)[keep], np.asarray(clean_s)[keep]
            )

    def test_an_all_non_finite_row_is_zeroed_the_same_way(self):
        for bad in (np.inf, np.nan):
            row = jnp.asarray(np.full((1, 128), bad, np.float32)).astype(jnp.bfloat16)
            quantized, scale = quantize_rows(row)
            self.assertEqual(float(scale[0, 0]), 0.0)
            np.testing.assert_array_equal(
                np.asarray(quantized.astype(jnp.float32)),
                np.zeros((1, 128), np.float32),
            )


class IntermediateRowsTest(parameterized.TestCase):
    ROWS = 8

    def halves(self, seed, inter):
        """A gate|up accumulator whose first column slice holds the larger
        values."""
        rng = np.random.default_rng(seed)
        acc = rng.normal(size=(self.ROWS, 2 * inter)).astype(np.float32)
        chunk = ffn.ACTIVATION_COLUMNS_PER_PASS_DEFAULT
        acc[:, chunk:inter] *= 0.01
        acc[:, inter + chunk :] *= 0.01
        return jnp.asarray(acc)

    def intermediate(self, acc1, inter):
        return ffn.intermediate_rows(
            acc1,
            jnp.ones((self.ROWS, 1), jnp.float32),
            inter,
            w1_scales=None,
            activation="silu",
            w1_bias=None,
            no_gate=False,
        )

    def test_a_zero_accumulator_leaves_a_zero_row_and_a_zero_scale(self):
        inter = 256
        mid, scale = self.intermediate(
            jnp.zeros((self.ROWS, 2 * inter), jnp.float32), inter
        )
        np.testing.assert_array_equal(
            np.asarray(mid.astype(jnp.float32)),
            np.zeros((self.ROWS, inter), np.float32),
        )
        np.testing.assert_array_equal(
            np.asarray(scale), np.zeros((self.ROWS, 1), np.float32)
        )

    def test_the_row_scale_spans_every_column_slice(self):
        inter = 2 * ffn.ACTIVATION_COLUMNS_PER_PASS_DEFAULT
        acc1 = self.halves(31, inter)
        _, scale = self.intermediate(acc1, inter)
        mid = (jax.nn.silu(acc1[:, :inter]) * acc1[:, inter:]).astype(jnp.bfloat16)
        first = jnp.max(
            jnp.abs(mid[:, : ffn.ACTIVATION_COLUMNS_PER_PASS_DEFAULT]), axis=-1
        )
        last = jnp.max(
            jnp.abs(mid[:, ffn.ACTIVATION_COLUMNS_PER_PASS_DEFAULT :]), axis=-1
        )
        self.assertTrue(bool(jnp.all(first > last)))  # the premise
        amax = jnp.max(jnp.abs(mid), axis=-1, keepdims=True)
        np.testing.assert_array_equal(
            np.asarray(scale), np.asarray(amax.astype(jnp.float32) / FP8_MAX)
        )

    def test_the_quantized_intermediate_dequantizes_back_to_its_rows(self):
        inter = 2 * ffn.ACTIVATION_COLUMNS_PER_PASS_DEFAULT
        acc1 = self.halves(32, inter)
        mid, scale = self.intermediate(acc1, inter)
        want = (jax.nn.silu(acc1[:, :inter]) * acc1[:, inter:]).astype(jnp.float32)
        got = mid.astype(jnp.float32) * scale
        worst = jnp.max(jnp.abs(got - want), axis=-1, keepdims=True)
        amax = jnp.max(jnp.abs(want), axis=-1, keepdims=True)
        self.assertTrue(bool(jnp.all(worst <= MID_QUANT_ROW_BOUND * amax)))


class ResultRowFormsTest(parameterized.TestCase):
    """The reference's result-row roundings: the small-float grid
    (float_grid_rows) with e2m1's parameters is the native
    float4_e2m1fn cast, on its own and through the reference."""

    @parameterized.named_parameters(("b16", 16), ("b32", 32), ("b128", 128))
    def test_the_e2m1_grid_is_the_native_fp4_cast(self, block):
        rng = np.random.default_rng(5)
        magnitudes = np.exp(rng.uniform(-10, 10, size=(24, 1)))
        rows = (rng.normal(size=(24, 512)) * magnitudes).astype(np.float32)
        rows[3] = 0.0  # a zero block scale
        rows[4, :block] = 0.0
        x = jnp.asarray(rows).astype(jnp.bfloat16)
        exp_bits, man_bits, bias = FLOAT_GRIDS["fp4_grid"]
        grid = float_grid_rows(
            x, block, exp_bits=exp_bits, man_bits=man_bits, bias=bias
        )
        np.testing.assert_array_equal(
            np.asarray(grid), np.asarray(fp4_block_rows(x, block))
        )

    def test_the_grid_forms_reach_the_reference_unchanged(self):
        """The grid form named fp4_grid gives the reference's fp4 output
        exactly (a shadowed expert index once made every grid form garbage)."""
        rng = np.random.default_rng(0)
        shape = dict(tokens=64, top_k=2, experts=8, hidden=256, inter=128)
        x = jnp.asarray(rng.normal(size=(64, 256)).astype(np.float32), jnp.bfloat16)
        w1 = jnp.asarray(
            rng.normal(size=(8, 256, 2 * 128)) * 0.05, jnp.bfloat16
        ).astype(jnp.float8_e4m3fn)
        w2 = jnp.asarray(rng.normal(size=(8, 128, 256)) * 0.05, jnp.bfloat16).astype(
            jnp.float8_e4m3fn
        )
        ones = jnp.ones((8, 2 * 128), jnp.float32), jnp.ones((8, 256), jnp.float32)
        logits = jnp.asarray(rng.normal(size=(64, 8)).astype(np.float32))

        def reference(**kw):
            return np.asarray(
                fused_ep_moe_reference(
                    x,
                    w1,
                    w2,
                    ones[0],
                    ones[1],
                    logits,
                    None,
                    None,
                    top_k=shape["top_k"],
                    renormalize=True,
                    weight_format=WeightFormat.FP8,
                    **kw,
                ),
                np.float32,
            )

        fp4 = reference(result_rows="fp4", result_block=16)
        np.testing.assert_array_equal(
            reference(result_rows="fp4_grid", result_block=16), fp4
        )


if __name__ == "__main__":
    absltest.main()
