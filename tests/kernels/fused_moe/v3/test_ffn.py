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
"""The expert FFN body as plain jnp: where each bias lands, the epilogue's
arithmetic."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.fused_moe.v3 import ffn
from vllm_torchtpu.kernels.fused_moe.v3.formats import FP4
from vllm_torchtpu.kernels.fused_moe.v3.rowquant import FP8, FP8_MAX, quantize_rows

pytestmark = pytest.mark.cpu_test

# Tolerance for the biased body against a reference that biases the
# dequantized halves. The gap is the intermediate requantization.
FFN_BIAS_RELATIVE_BOUND = 0.05
MID_QUANT_ROW_BOUND = 0.07


class FfnTest(parameterized.TestCase):
    ROWS, HIDDEN, INTER, BLOCK = 16, 256, 128, 128

    def operands(self, seed=0, fp4=False):
        key = jax.random.key(seed)
        kx, k1, k2, kb1, kb2 = jax.random.split(key, 5)
        x = jax.random.normal(kx, (self.ROWS, self.HIDDEN), jnp.float32) / 4
        w1 = jax.random.normal(k1, (self.HIDDEN, 2 * self.INTER), jnp.float32) / 16
        w2 = jax.random.normal(k2, (self.INTER, self.HIDDEN), jnp.float32) / 16
        b1 = jax.random.normal(kb1, (1, 2 * self.INTER), jnp.float32) / 2
        b2 = jax.random.normal(kb2, (1, self.HIDDEN), jnp.float32) * 2
        rows, scales = quantize_rows(x.astype(jnp.bfloat16))
        quant = self.block_quant if fp4 else self.channel_quant
        q1, s1 = quant(w1)
        q2, s2 = quant(w2)
        return dict(rows=rows, scales=scales, q1=q1, s1=s1, q2=q2, s2=s2, b1=b1, b2=b2)

    def channel_quant(self, w):
        amax = jnp.max(jnp.abs(w), axis=0, keepdims=True)
        scale = jnp.where(amax == 0, 1.0, amax / FP8_MAX).astype(jnp.float32)
        return (w / scale).astype(FP8), scale

    def block_quant(self, w):
        k, n = w.shape
        blocks = w.reshape(k // self.BLOCK, self.BLOCK, n)
        amax = jnp.max(jnp.abs(blocks), axis=1)
        scale = jnp.where(amax == 0, 1.0, amax / 6.0).astype(jnp.float32)
        return (blocks / scale[:, None, :]).astype(FP4).reshape(k, n), scale

    def body(self, op, *, fp4, activation, w1_bias, activation_block=0):
        """The dequantized down projection the body produces, under jit as
        the kernel runs it (the eager fp8 dot on the TPU is lossy)."""
        return jax.jit(
            lambda op: self._body(
                op,
                fp4=fp4,
                activation=activation,
                w1_bias=w1_bias,
                activation_block=activation_block,
            )
        )(op)

    def _body(self, op, *, fp4, activation, w1_bias, activation_block=0):
        if fp4:

            def block_of(q):
                return lambda b: q[b * self.BLOCK : (b + 1) * self.BLOCK].astype(FP8)

            acc2, scales = ffn.expert_ffn_fp4(
                op["rows"],
                op["scales"],
                block_of(op["q1"]),
                block_of(op["q2"]),
                op["s1"],
                op["s2"],
                block=self.BLOCK,
                activation=activation,
                w1_bias=w1_bias,
                no_gate=False,
                activation_block=activation_block,
            )
            return acc2 if scales is None else acc2 * scales
        acc2, scales = ffn.expert_ffn_fp8(
            op["rows"],
            op["scales"],
            op["q1"],
            op["q2"],
            op["s1"],
            activation=activation,
            w1_bias=w1_bias,
            no_gate=False,
            block=activation_block,
        )
        return (acc2 if scales is None else acc2 * scales) * op["s2"]

    def exact(self, op, *, activation, w1_bias):
        """The same layer on the dequantized operands in f32."""
        n = op["scales"].shape[1]
        width = self.HIDDEN // n
        rows = jnp.concatenate(
            [
                op["rows"][:, b * width : (b + 1) * width].astype(jnp.float32)
                * op["scales"][:, b : b + 1]
                for b in range(n)
            ],
            axis=1,
        )
        if op["s1"].shape[0] == 1:  # per-channel fp8
            w1 = op["q1"].astype(jnp.float32) * op["s1"]
            w2 = op["q2"].astype(jnp.float32) * op["s2"]
        else:  # block-scaled fp4
            k = self.HIDDEN // op["s1"].shape[0]
            w1 = (
                op["q1"].astype(jnp.float32).reshape(-1, k, 2 * self.INTER)
                * op["s1"][:, None, :]
            ).reshape(self.HIDDEN, 2 * self.INTER)
            k2 = self.INTER // op["s2"].shape[0]
            w2 = (
                op["q2"].astype(jnp.float32).reshape(-1, k2, self.HIDDEN)
                * op["s2"][:, None, :]
            ).reshape(self.INTER, self.HIDDEN)
        acc1 = rows @ w1
        if w1_bias is not None:
            acc1 = acc1 + w1_bias
        mid = ffn.apply_activation(
            acc1[:, : self.INTER], acc1[:, self.INTER :], activation
        )
        return mid @ w2

    def rel(self, got, want):
        got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
        return float(np.linalg.norm(got - want) / np.linalg.norm(want))

    @parameterized.parameters(False, True)
    def test_a_blocked_body_stays_as_close_to_exact_as_the_whole_row(self, fp4):
        """With an activation block, both contractions carry per-block row
        scales; the result stays within the fp8 rounding band of the exact
        f32 layer, like the whole-row body does."""
        op = self.operands(fp4=fp4)
        op["rows"], op["scales"] = quantize_rows(
            jax.random.normal(
                jax.random.key(0), (self.ROWS, self.HIDDEN), jnp.float32
            ).astype(jnp.bfloat16)
            / 4,
            block=self.BLOCK,
        )
        got = self.body(
            op, fp4=fp4, activation="silu", w1_bias=None, activation_block=self.BLOCK
        )
        want = self.exact(op, activation="silu", w1_bias=None)
        self.assertLess(self.rel(got, want), 0.15)
        with self.assertRaisesRegex(ValueError, "activation block"):
            self.body(
                op,
                fp4=True,
                activation="silu",
                w1_bias=None,
                activation_block=2 * self.BLOCK,
            )

    def test_one_block_is_the_whole_row(self):
        """blocked_rows_dot over one block is the dot scaled by the row.

        Two exact checks, because XLA's fp8 matmul path on the TPU is not
        exact (4.2e-3 relative L2 against float64 on a 256-long
        contraction, in every form XLA turns into a native fp8 dot;
        studies/fp8_dot_probe2.py): the bf16 form of the helper, run
        eagerly, against float64; and the fp8 form under jit against the
        plain fp8 dot under the same jit, which XLA treats alike."""
        op = self.operands()
        rows, q1, scales = op["rows"], op["q1"], op["scales"]
        exact = (
            np.asarray(rows, np.float64) @ np.asarray(q1, np.float64)
        ) * np.asarray(scales, np.float64)
        bf16_form = ffn.blocked_rows_dot(
            rows, q1, scales, self.HIDDEN, upcast=jnp.bfloat16
        )
        np.testing.assert_allclose(
            np.asarray(bf16_form, np.float64), exact, rtol=1e-5, atol=1e-3
        )
        fp8_form, plain = jax.jit(
            lambda r, q, s: (
                ffn.blocked_rows_dot(r, q, s, self.HIDDEN),
                jax.lax.dot_general(
                    r, q, ffn.DOT_ROWS_BY_COLUMNS, preferred_element_type=jnp.float32
                )
                * s,
            )
        )(rows, q1, scales)
        np.testing.assert_array_equal(np.asarray(fp8_form), np.asarray(plain))

    @parameterized.product(fp4=(False, True), activation=("silu", "swigluoai"))
    def test_a_zero_bias_leaves_the_body_bitwise_unchanged(self, fp4, activation):
        op = self.operands(fp4=fp4)
        plain = self.body(op, fp4=fp4, activation=activation, w1_bias=None)
        zeroed = self.body(
            op, fp4=fp4, activation=activation, w1_bias=jnp.zeros_like(op["b1"])
        )
        np.testing.assert_array_equal(np.asarray(zeroed), np.asarray(plain))

    @parameterized.product(fp4=(False, True), activation=("silu", "swigluoai"))
    def test_the_gate_and_up_bias_land_before_the_activation(self, fp4, activation):
        op = self.operands(fp4=fp4)
        got = self.body(op, fp4=fp4, activation=activation, w1_bias=op["b1"])
        x = op["rows"].astype(jnp.float32) * op["scales"]
        if fp4:
            w1 = (
                op["q1"].astype(jnp.float32).reshape(-1, self.BLOCK, 2 * self.INTER)
                * op["s1"][:, None, :]
            ).reshape(self.HIDDEN, 2 * self.INTER)
            w2 = (
                op["q2"].astype(jnp.float32).reshape(-1, self.BLOCK, self.HIDDEN)
                * op["s2"][:, None, :]
            ).reshape(self.INTER, self.HIDDEN)
        else:
            w1 = op["q1"].astype(jnp.float32) * op["s1"]
            w2 = op["q2"].astype(jnp.float32) * op["s2"]
        acc1 = x @ w1 + op["b1"]
        mid = ffn.apply_activation(
            acc1[:, : self.INTER], acc1[:, self.INTER :], activation
        )
        want = mid.astype(jnp.bfloat16).astype(jnp.float32) @ w2
        rel = float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))
        self.assertLess(rel, FFN_BIAS_RELATIVE_BOUND)

    @parameterized.parameters(False, True)
    def test_a_zero_down_projection_leaves_the_row_zero(self, fp4):
        op = self.operands(fp4=fp4)
        op = dict(op, q2=jnp.zeros_like(op["q2"]))
        row = self.body(op, fp4=fp4, activation="swigluoai", w1_bias=op["b1"])
        np.testing.assert_array_equal(
            np.asarray(row), np.zeros((self.ROWS, self.HIDDEN), np.float32)
        )

    def test_the_result_row_is_the_epilogue_applied_exactly_once(self):
        op = self.operands()
        acc2 = jax.random.normal(
            jax.random.key(7), (self.ROWS, self.HIDDEN), jnp.float32
        )
        mid_scales = jnp.full((self.ROWS, 1), 0.25, jnp.float32)
        rows, scales = ffn.result_rows(
            acc2, mid_scales, w2_scales=op["s2"], w2_bias=op["b2"]
        )
        want = ((acc2 * mid_scales) * op["s2"] + op["b2"]).astype(jnp.bfloat16)
        self.assertEqual(jnp.dtype(rows.dtype), jnp.dtype(FP8))
        got = rows.astype(jnp.float32) * scales
        amax = jnp.max(jnp.abs(want.astype(jnp.float32)), axis=-1, keepdims=True)
        worst = jnp.max(jnp.abs(got - want.astype(jnp.float32)), axis=-1, keepdims=True)
        self.assertTrue(bool(jnp.all(worst <= MID_QUANT_ROW_BOUND * amax)))

    def test_doubling_the_down_bias_moves_the_result_row(self):
        """On the dequantized row: an fp8 row takes its scale from the
        row's maximum, so doubling every value moves only the scale."""
        op = self.operands()
        acc2 = jnp.zeros((self.ROWS, self.HIDDEN), jnp.float32)
        ones = jnp.ones((self.ROWS, 1), jnp.float32)

        def row_of(bias):
            rows, scales = ffn.result_rows(acc2, ones, w2_scales=None, w2_bias=bias)
            return rows.astype(jnp.float32) * scales

        self.assertFalse(bool(jnp.allclose(row_of(op["b2"]), row_of(2.0 * op["b2"]))))


if __name__ == "__main__":
    absltest.main()
