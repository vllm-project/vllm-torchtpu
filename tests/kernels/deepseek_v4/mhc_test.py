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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import vllm.model_executor.kernels.mhc as mhc_kernels
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.deepseek_v4.mhc import utils

HC_MULT = 4
# Values from the official DeepSeek-V4 config.json: rms_norm_eps=1e-6,
# hc_eps=1e-6 (feeds BOTH pre_eps and sinkhorn_eps), hc_sinkhorn_iters=20;
# hc_post_alpha=2.0 is hardcoded in the model (deepseek_v4.py).
RMS_EPS = 1e-6
HC_PRE_EPS = 1e-6
HC_SINKHORN_EPS = 1e-6
HC_POST_MULT_VALUE = 2.0

# f32 outputs: absorb summation-order differences of the (D=hc_mult*H)-long
# f32 accumulations between torch and XLA.
F32_RTOL, F32_ATOL = 1e-4, 1e-5
# bf16 outputs: ~1-2 ulp at magnitude O(1).
BF16_RTOL, BF16_ATOL = 2e-2, 2e-2
# Fused-path f32 gate outputs: the seam kernel's recombine and the
# reference's einsum can round near-boundary stream values to different
# adjacent bf16 values before the mix GEMM, so gates carry ~1 bf16-ulp of
# extra input noise (worst at small H, where each element weighs more).
FUSED_F32_RTOL, FUSED_F32_ATOL = 1e-3, 1e-5
# At the tiny test H=256 each flipped element weighs 28x more than at the
# real H=7168, and the flip pattern is backend-dependent (measured up to
# 2.4e-3 on the TPU backend); the small-shape fused tests use this
# looser bound, the real-shape tests keep FUSED_F32_RTOL.
FUSED_SMALL_H_RTOL = 5e-3
# In-kernel vs XLA gates on identical mixes: transcendental/division
# lowerings differ by a few ulp, and 20 Sinkhorn sweeps carry that along.
GATES_RTOL, GATES_ATOL = 1e-5, 1e-7


def _ref_mhc_pre_mixes(x2d: jax.Array, fn: jax.Array) -> tuple[jax.Array, jax.Array]:
    x = x2d.astype(jnp.float32)
    mixes = jax.lax.dot_general(
        x,
        fn,
        dimension_numbers=(((1,), (1,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
    )
    sqrsum = jnp.sum(x * x, axis=-1, keepdims=True)
    return mixes, sqrsum


def _ref_mhc_pre_collapse(
    pre_mix: jax.Array,
    x2d: jax.Array,
    hc_mult: int,
    hidden_size: int,
) -> jax.Array:
    out = pre_mix[:, 0:1] * x2d[:, :hidden_size].astype(jnp.float32)
    for i in range(1, hc_mult):
        out = out + (
            pre_mix[:, i : i + 1]
            * x2d[:, i * hidden_size : (i + 1) * hidden_size].astype(jnp.float32)
        )
    return out.astype(jnp.bfloat16)


def _ref_mhc_pre(
    residual: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    assert residual.dtype == jnp.bfloat16
    assert fn.dtype == jnp.float32

    outer_shape = residual.shape[:-2]
    hc_mult, hidden_size = residual.shape[-2:]
    x2d = residual.reshape(-1, hc_mult * hidden_size)

    mixes, sqrsum = _ref_mhc_pre_mixes(x2d, fn)
    pre_mix, post_mix, comb_mix = utils.mhc_pre_gates(
        mixes,
        sqrsum,
        hc_mult,
        hidden_size,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    layer_input = _ref_mhc_pre_collapse(pre_mix, x2d, hc_mult, hidden_size)

    return (
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
        layer_input.reshape(*outer_shape, hidden_size),
    )


def _ref_mhc_post(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    *,
    precision: jax.lax.Precision | None = None,
) -> jax.Array:
    mixed_residual = jnp.einsum(
        "...ij,...ih->...jh",
        comb_res_mix.astype(jnp.float32),
        residual.astype(jnp.float32),
        precision=precision,
    )
    post_term = post_layer_mix.astype(jnp.float32) * x[..., None, :].astype(jnp.float32)
    return (mixed_residual + post_term).astype(residual.dtype)


def _ref_mhc_fused_post_pre(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    residual_cur = _ref_mhc_post(
        x, residual, post_layer_mix, comb_res_mix, precision=jax.lax.Precision.HIGHEST
    )
    post_mix_cur, comb_mix_cur, layer_input_cur = _ref_mhc_pre(
        residual_cur,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur


def _to_np(x) -> np.ndarray:
    """jax array (any dtype incl. bf16) -> float32 numpy."""
    return np.asarray(jnp.asarray(x).astype(jnp.float32))


def _make_pre_inputs(rng, num_tokens, hidden_size):
    hc_mult3 = 2 * HC_MULT + HC_MULT * HC_MULT
    residual = rng.standard_normal((num_tokens, HC_MULT, hidden_size), dtype=np.float32)
    fn = rng.standard_normal((hc_mult3, HC_MULT * hidden_size), dtype=np.float32) * 0.02
    hc_scale = (1.0 + 0.1 * rng.standard_normal(3)).astype(np.float32)
    hc_base = (0.5 * rng.standard_normal(hc_mult3)).astype(np.float32)
    return residual, fn, hc_scale, hc_base


class MhcPreParityTest(parameterized.TestCase):
    @parameterized.product(
        num_tokens=[1, 8, 17, 128],
        hidden_size=[256, 7168],
        sinkhorn_repeat=[1, 2, 20],
    )
    def test_matches_torch_reference(self, num_tokens, hidden_size, sinkhorn_repeat):
        rng = np.random.default_rng(0)
        residual, fn, hc_scale, hc_base = _make_pre_inputs(rng, num_tokens, hidden_size)

        want_post, want_comb, want_layer = mhc_kernels.mhc_pre_torch(
            torch.from_numpy(residual).to(torch.bfloat16),
            torch.from_numpy(fn),
            torch.from_numpy(hc_scale),
            torch.from_numpy(hc_base),
            RMS_EPS,
            HC_PRE_EPS,
            HC_SINKHORN_EPS,
            HC_POST_MULT_VALUE,
            sinkhorn_repeat,
        )

        got_post, got_comb, got_layer = _ref_mhc_pre(
            jnp.asarray(residual).astype(jnp.bfloat16),
            jnp.asarray(fn),
            jnp.asarray(hc_scale),
            jnp.asarray(hc_base),
            RMS_EPS,
            HC_PRE_EPS,
            HC_SINKHORN_EPS,
            HC_POST_MULT_VALUE,
            sinkhorn_repeat,
        )

        self.assertEqual(got_post.shape, (num_tokens, HC_MULT, 1))
        self.assertEqual(got_comb.shape, (num_tokens, HC_MULT, HC_MULT))
        self.assertEqual(got_layer.shape, (num_tokens, hidden_size))
        self.assertEqual(got_layer.dtype, jnp.bfloat16)

        np.testing.assert_allclose(
            _to_np(got_post), want_post.float().numpy(), rtol=F32_RTOL, atol=F32_ATOL
        )
        np.testing.assert_allclose(
            _to_np(got_comb), want_comb.float().numpy(), rtol=F32_RTOL, atol=F32_ATOL
        )
        np.testing.assert_allclose(
            _to_np(got_layer),
            want_layer.float().numpy(),
            rtol=BF16_RTOL,
            atol=BF16_ATOL,
        )

    def test_padded_zero_rows_are_finite(self):
        """vLLM pads token buckets with all-zero rows; they must not NaN."""

        rng = np.random.default_rng(1)
        residual, fn, hc_scale, hc_base = _make_pre_inputs(rng, 8, 256)
        residual[4:] = 0.0  # padded tail

        got_post, got_comb, got_layer = _ref_mhc_pre(
            jnp.asarray(residual).astype(jnp.bfloat16),
            jnp.asarray(fn),
            jnp.asarray(hc_scale),
            jnp.asarray(hc_base),
            RMS_EPS,
            HC_PRE_EPS,
            HC_SINKHORN_EPS,
            HC_POST_MULT_VALUE,
            2,
        )
        for out in (got_post, got_comb, got_layer):
            self.assertTrue(bool(jnp.isfinite(out.astype(jnp.float32)).all()))


class MhcPostParityTest(parameterized.TestCase):
    @parameterized.product(num_tokens=[1, 17, 128], hidden_size=[256, 7168])
    def test_matches_torch_reference(self, num_tokens, hidden_size):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((num_tokens, hidden_size), dtype=np.float32)
        residual = rng.standard_normal(
            (num_tokens, HC_MULT, hidden_size), dtype=np.float32
        )
        # Mimic mhc_pre outputs: positive doubly-normalized-ish comb mix and
        # sigmoid-scaled post mix, both f32.
        post_mix = (
            2.0 / (1.0 + np.exp(-rng.standard_normal((num_tokens, HC_MULT, 1))))
        ).astype(np.float32)
        comb_raw = (
            np.abs(rng.standard_normal((num_tokens, HC_MULT, HC_MULT))).astype(
                np.float32
            )
            + 0.01
        )
        comb_mix = comb_raw / comb_raw.sum(axis=-1, keepdims=True)

        want = mhc_kernels.mhc_post_torch(
            torch.from_numpy(x).to(torch.bfloat16),
            torch.from_numpy(residual).to(torch.bfloat16),
            torch.from_numpy(post_mix),
            torch.from_numpy(comb_mix),
        )

        got = _ref_mhc_post(
            jnp.asarray(x).astype(jnp.bfloat16),
            jnp.asarray(residual).astype(jnp.bfloat16),
            jnp.asarray(post_mix),
            jnp.asarray(comb_mix),
        )

        self.assertEqual(got.shape, (num_tokens, HC_MULT, hidden_size))
        self.assertEqual(got.dtype, jnp.bfloat16)
        np.testing.assert_allclose(
            _to_np(got), want.float().numpy(), rtol=BF16_RTOL, atol=BF16_ATOL
        )


def _make_post_inputs(rng, num_tokens, hidden_size):
    x = rng.standard_normal((num_tokens, hidden_size), dtype=np.float32)
    residual = rng.standard_normal((num_tokens, HC_MULT, hidden_size), dtype=np.float32)
    post_mix = (
        2.0 / (1.0 + np.exp(-rng.standard_normal((num_tokens, HC_MULT, 1))))
    ).astype(np.float32)
    comb_raw = (
        np.abs(rng.standard_normal((num_tokens, HC_MULT, HC_MULT))).astype(np.float32)
        + 0.01
    )
    comb_mix = comb_raw / comb_raw.sum(axis=-1, keepdims=True)
    return (
        jnp.asarray(x).astype(jnp.bfloat16),
        jnp.asarray(residual).astype(jnp.bfloat16),
        jnp.asarray(post_mix),
        jnp.asarray(comb_mix),
    )


@pytest.mark.tpu_test
class MhcPostPallasSmallShapeTest(parameterized.TestCase):
    """Post kernel vs the jnp oracle at small shapes and ragged token
    counts (block indexing and padding); needs TPU hardware."""

    def setUp(self):
        super().setUp()
        if jax.default_backend() != "tpu":
            self.skipTest("Pallas TPU kernel needs TPU hardware")

    @parameterized.product(num_tokens=[1, 17, 64, 200], hidden_size=[256])
    def test_matches_reference(self, num_tokens, hidden_size):
        from vllm_torchtpu.kernels.deepseek_v4.mhc import post_kernel

        rng = np.random.default_rng(7)
        args = _make_post_inputs(rng, num_tokens, hidden_size)

        want = _ref_mhc_post(*args)
        got = post_kernel.mhc_post(*args, token_block_size=64)

        self.assertEqual(got.shape, want.shape)
        self.assertEqual(got.dtype, jnp.bfloat16)
        np.testing.assert_allclose(
            _to_np(got), _to_np(want), rtol=BF16_RTOL, atol=BF16_ATOL
        )


@pytest.mark.tpu_test
class MhcPostPallasTpuTest(parameterized.TestCase):
    """Compiled (Mosaic) post kernel vs the jnp reference. Needs TPU."""

    def setUp(self):
        super().setUp()
        if jax.default_backend() != "tpu":
            self.skipTest("Pallas TPU kernel needs TPU hardware")

    @parameterized.product(
        num_tokens=[16, 17, 128, 2048],
        token_block_size=[32, 64],
    )
    def test_matches_reference(self, num_tokens, token_block_size):
        from vllm_torchtpu.kernels.deepseek_v4.mhc import post_kernel

        rng = np.random.default_rng(8)
        args = _make_post_inputs(rng, num_tokens, 7168)

        want = _ref_mhc_post(*args)
        got = post_kernel.mhc_post(*args, token_block_size=token_block_size)

        self.assertEqual(got.shape, want.shape)
        np.testing.assert_allclose(
            _to_np(got), _to_np(want), rtol=BF16_RTOL, atol=BF16_ATOL
        )


def _make_fused_inputs(rng, num_tokens, hidden_size):
    """x/residual/gates (post-style) plus fn/scale/base (pre-style)."""
    x, residual, post_mix, comb_mix = _make_post_inputs(rng, num_tokens, hidden_size)
    _, fn, hc_scale, hc_base = _make_pre_inputs(rng, num_tokens, hidden_size)
    return (
        x,
        residual,
        post_mix,
        comb_mix,
        jnp.asarray(fn),
        jnp.asarray(hc_scale),
        jnp.asarray(hc_base),
    )


def _fused_args(rng, num_tokens, hidden_size, sinkhorn_repeat=20):
    x, res, plm, crm, fn, sc, hb = _make_fused_inputs(rng, num_tokens, hidden_size)
    return (
        x,
        res,
        plm,
        crm,
        fn,
        sc,
        hb,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT_VALUE,
        sinkhorn_repeat,
    )


@pytest.mark.tpu_test
class MhcFusedPostPrePallasSmallShapeTest(parameterized.TestCase):
    """Fused seam kernel vs the sequential oracle at small shapes and
    ragged token counts; needs TPU hardware."""

    def setUp(self):
        super().setUp()
        if jax.default_backend() != "tpu":
            self.skipTest("Pallas TPU kernel needs TPU hardware")

    @parameterized.product(num_tokens=[1, 17, 64, 200], hidden_size=[256])
    def test_matches_sequential_reference(self, num_tokens, hidden_size):
        from vllm_torchtpu.kernels.deepseek_v4.mhc import fused_post_pre_kernel

        args = _fused_args(np.random.default_rng(9), num_tokens, hidden_size)
        fused_fn = fused_post_pre_kernel.mhc_fused_post_pre

        want_res, want_post, want_comb, want_layer = _ref_mhc_fused_post_pre(*args)
        got_res, got_post, got_comb, got_layer = fused_fn(*args, token_block_size=64)

        self.assertEqual(got_res.shape, want_res.shape)
        self.assertEqual(got_res.dtype, jnp.bfloat16)
        self.assertEqual(got_layer.dtype, jnp.bfloat16)
        np.testing.assert_allclose(
            _to_np(got_res), _to_np(want_res), rtol=BF16_RTOL, atol=BF16_ATOL
        )
        np.testing.assert_allclose(
            _to_np(got_post),
            _to_np(want_post),
            rtol=FUSED_SMALL_H_RTOL,
            atol=FUSED_F32_ATOL,
        )
        np.testing.assert_allclose(
            _to_np(got_comb),
            _to_np(want_comb),
            rtol=FUSED_SMALL_H_RTOL,
            atol=FUSED_F32_ATOL,
        )
        np.testing.assert_allclose(
            _to_np(got_layer), _to_np(want_layer), rtol=BF16_RTOL, atol=BF16_ATOL
        )


@pytest.mark.tpu_test
class MhcFusedPostPrePallasTpuTest(parameterized.TestCase):
    """Compiled (Mosaic) fused seam kernel vs the sequential reference.
    Needs TPU hardware."""

    def setUp(self):
        super().setUp()
        if jax.default_backend() != "tpu":
            self.skipTest("Pallas TPU kernel needs TPU hardware")

    @parameterized.product(
        num_tokens=[16, 17, 128, 2048],
        token_block_size=[32, 64],
    )
    def test_matches_sequential_reference(self, num_tokens, token_block_size):
        from vllm_torchtpu.kernels.deepseek_v4.mhc import fused_post_pre_kernel

        args = _fused_args(np.random.default_rng(10), num_tokens, 7168)
        fused_fn = fused_post_pre_kernel.mhc_fused_post_pre

        want = _ref_mhc_fused_post_pre(*args)
        got = fused_fn(*args, token_block_size=token_block_size)

        for g, w, tol in zip(got, want, ("bf16", "f32", "f32", "bf16")):
            rtol, atol = (
                (BF16_RTOL, BF16_ATOL)
                if tol == "bf16"
                else (FUSED_F32_RTOL, FUSED_F32_ATOL)
            )
            np.testing.assert_allclose(_to_np(g), _to_np(w), rtol=rtol, atol=atol)


@pytest.mark.tpu_test
class MhcPreMixesPallasSmallShapeTest(parameterized.TestCase):
    """Mix-GEMM kernel vs the jnp oracle at small shapes and ragged
    token counts, validating block indexing and padding; needs TPU
    hardware."""

    def setUp(self):
        super().setUp()
        if jax.default_backend() != "tpu":
            self.skipTest("Pallas TPU kernel needs TPU hardware")

    @parameterized.product(num_tokens=[1, 17, 64, 200], hidden_size=[256])
    def test_matches_reference(self, num_tokens, hidden_size):
        from vllm_torchtpu.kernels.deepseek_v4.mhc import pre_kernel

        rng = np.random.default_rng(4)
        residual, fn, _, _ = _make_pre_inputs(rng, num_tokens, hidden_size)
        x2d = (
            jnp.asarray(residual)
            .astype(jnp.bfloat16)
            .reshape(num_tokens, HC_MULT * hidden_size)
        )
        fn_j = jnp.asarray(fn)

        want_mixes, want_sqrsum = _ref_mhc_pre_mixes(x2d, fn_j)
        got_mixes, got_sqrsum = pre_kernel.mhc_pre_mixes(
            x2d,
            fn_j,
            token_block_size=64,
        )

        self.assertEqual(got_mixes.shape, want_mixes.shape)
        self.assertEqual(got_sqrsum.shape, want_sqrsum.shape)
        np.testing.assert_allclose(
            _to_np(got_mixes), _to_np(want_mixes), rtol=F32_RTOL, atol=F32_ATOL
        )
        np.testing.assert_allclose(
            _to_np(got_sqrsum), _to_np(want_sqrsum), rtol=F32_RTOL, atol=F32_ATOL
        )


@pytest.mark.tpu_test
class MhcPrePallasTpuTest(parameterized.TestCase):
    """Compiled (Mosaic) kernel vs the jnp reference. Needs TPU hardware."""

    def setUp(self):
        super().setUp()
        if jax.default_backend() != "tpu":
            self.skipTest("Pallas TPU kernel needs TPU hardware")

    @parameterized.product(
        num_tokens=[16, 17, 128, 2048],
        token_block_size=[64, 128],
    )
    def test_mixes_matches_reference(self, num_tokens, token_block_size):
        from vllm_torchtpu.kernels.deepseek_v4.mhc import pre_kernel

        hidden_size = 7168  # DeepSeek-V4 shape; D = 28672
        rng = np.random.default_rng(5)
        residual, fn, _, _ = _make_pre_inputs(rng, num_tokens, hidden_size)
        x2d = (
            jnp.asarray(residual)
            .astype(jnp.bfloat16)
            .reshape(num_tokens, HC_MULT * hidden_size)
        )
        fn_j = jnp.asarray(fn)

        want_mixes, want_sqrsum = _ref_mhc_pre_mixes(x2d, fn_j)
        got_mixes, got_sqrsum = pre_kernel.mhc_pre_mixes(
            x2d, fn_j, token_block_size=token_block_size
        )

        np.testing.assert_allclose(
            _to_np(got_mixes), _to_np(want_mixes), rtol=F32_RTOL, atol=F32_ATOL
        )
        np.testing.assert_allclose(
            _to_np(got_sqrsum), _to_np(want_sqrsum), rtol=F32_RTOL, atol=F32_ATOL
        )

    @parameterized.product(num_tokens=[16, 128], sinkhorn_repeat=[1, 20])
    def test_full_op_matches_reference(self, num_tokens, sinkhorn_repeat):
        from vllm_torchtpu.kernels.deepseek_v4.mhc import pre_kernel

        hidden_size = 7168
        rng = np.random.default_rng(6)
        residual, fn, hc_scale, hc_base = _make_pre_inputs(rng, num_tokens, hidden_size)
        args = (
            jnp.asarray(residual).astype(jnp.bfloat16),
            jnp.asarray(fn),
            jnp.asarray(hc_scale),
            jnp.asarray(hc_base),
            RMS_EPS,
            HC_PRE_EPS,
            HC_SINKHORN_EPS,
            HC_POST_MULT_VALUE,
            sinkhorn_repeat,
        )

        want_post, want_comb, want_layer = _ref_mhc_pre(*args)
        got_post, got_comb, got_layer = pre_kernel.mhc_pre(*args)

        self.assertEqual(got_layer.dtype, jnp.bfloat16)
        np.testing.assert_allclose(
            _to_np(got_post), _to_np(want_post), rtol=F32_RTOL, atol=F32_ATOL
        )
        np.testing.assert_allclose(
            _to_np(got_comb), _to_np(want_comb), rtol=F32_RTOL, atol=F32_ATOL
        )
        np.testing.assert_allclose(
            _to_np(got_layer), _to_np(want_layer), rtol=BF16_RTOL, atol=BF16_ATOL
        )


@pytest.mark.tpu_test
class MhcGatesInKernelTpuTest(parameterized.TestCase):
    """``gates_in_kernel`` vs the XLA-gates path of the same kernels.

    The streams and layer input never see the post/comb gates, so they must
    be bit-identical; the gates differ only by transcendental/division
    rounding (Mosaic vs XLA lowering). Needs TPU hardware.
    """

    def setUp(self):
        super().setUp()
        if jax.default_backend() != "tpu":
            self.skipTest("Pallas TPU kernel needs TPU hardware")

    def _check_gates(self, got_post, got_comb, want_post, want_comb):
        np.testing.assert_allclose(
            _to_np(got_post), _to_np(want_post), rtol=GATES_RTOL, atol=GATES_ATOL
        )
        np.testing.assert_allclose(
            _to_np(got_comb), _to_np(want_comb), rtol=GATES_RTOL, atol=GATES_ATOL
        )

    @parameterized.product(num_tokens=[1, 8, 16, 17, 32, 128], sinkhorn_repeat=[1, 20])
    def test_fused(self, num_tokens, sinkhorn_repeat):
        from vllm_torchtpu.kernels.deepseek_v4.mhc import fused_post_pre_kernel

        args = _fused_args(np.random.default_rng(11), num_tokens, 7168, sinkhorn_repeat)
        fused_fn = fused_post_pre_kernel.mhc_fused_post_pre
        want_res, want_post, want_comb, want_layer = fused_fn(*args)
        got_res, got_post, got_comb, got_layer = fused_fn(*args, gates_in_kernel=True)

        self.assertEqual(got_post.shape, want_post.shape)
        self.assertEqual(got_comb.shape, want_comb.shape)
        np.testing.assert_array_equal(_to_np(got_res), _to_np(want_res))
        np.testing.assert_array_equal(_to_np(got_layer), _to_np(want_layer))
        self._check_gates(got_post, got_comb, want_post, want_comb)

    @parameterized.product(
        num_tokens=[1, 3, 4, 8], sinkhorn_repeat=[1, 20], norm=[False, True],
        stacked=[False, True],
    )
    def test_fused_small_block(self, num_tokens, sinkhorn_repeat, norm, stacked):
        """small_block=8 (one block of exactly the T<=8 rows) vs the 16-row
        block: streams bit-identical; the squared-sum reduction runs over a
        different tile layout, so the gates move by f32 ulps (gate
        tolerance) and a rare layer-input element by one bf16 ulp."""
        from vllm_torchtpu.kernels.deepseek_v4.mhc import fused_post_pre_kernel, utils

        rng = np.random.default_rng(15)
        args = _fused_args(rng, num_tokens, 7168, sinkhorn_repeat)
        if stacked:
            args = args[:4] + (utils.cat_fn3(jnp.asarray(args[4])),) + args[5:]
        kw = dict(gates_in_kernel=True, gates_max_tokens=0)
        if norm:
            kw.update(
                norm_weight=jnp.asarray(rng.uniform(0.5, 1.5, (7168,)), jnp.bfloat16),
                norm_eps=RMS_EPS,
            )
        fused_fn = fused_post_pre_kernel.mhc_fused_post_pre
        want = fused_fn(*args, small_block=16, **kw)
        got = fused_fn(*args, small_block=8, **kw)
        for g, w in zip(got, want):
            self.assertEqual(g.shape, w.shape)
            self.assertTrue(np.isfinite(_to_np(g)).all())
        np.testing.assert_array_equal(_to_np(got[0]), _to_np(want[0]))
        got_l, want_l = _to_np(got[3]), _to_np(want[3])
        np.testing.assert_allclose(got_l, want_l, rtol=8e-3, atol=1e-6)
        self.assertLessEqual(np.mean(got_l != want_l), 1e-3)
        self._check_gates(got[1], got[2], want[1], want[2])

    @parameterized.product(num_tokens=[1, 8, 16, 32, 128, 300])
    def test_fused_two_fn_chunks(self, num_tokens):
        """fn_chunks=2 (hi + mid of the stacked fn) vs a float64 oracle of
        the mix logits: within 1e-5 of the logit scale (3-chunk ~2e-7);
        new_res unchanged."""
        from vllm_torchtpu.kernels.deepseek_v4.mhc import fused_post_pre_kernel, utils

        rng = np.random.default_rng(16)
        x, res, plm, crm, fn, sc, hb = _make_fused_inputs(rng, num_tokens, 7168)
        cat = utils.cat_fn3(jnp.asarray(fn))
        ops = (
            jnp.asarray(x).astype(jnp.bfloat16).reshape(num_tokens, -1),
            jnp.asarray(res).astype(jnp.bfloat16).reshape(num_tokens, -1),
            jnp.asarray(plm, jnp.float32).reshape(num_tokens, -1),
            jnp.asarray(crm, jnp.float32).reshape(num_tokens, -1),
            cat,
            jnp.asarray(sc),
            jnp.asarray(hb),
            RMS_EPS,
            HC_PRE_EPS,
        )
        res3, mix3, _, _ = fused_post_pre_kernel.fused_post_pre_mixes(*ops, fn_chunks=3)
        res2, mix2, _, _ = fused_post_pre_kernel.fused_post_pre_mixes(*ops, fn_chunks=2)
        np.testing.assert_array_equal(_to_np(res2), _to_np(res3))
        g = _to_np(res3).astype(np.float64)
        ref = g @ np.asarray(fn, np.float64).T
        scale = np.abs(ref).max()
        self.assertLess(np.abs(_to_np(mix3) - ref).max() / scale, 1e-6)
        self.assertLess(np.abs(_to_np(mix2) - ref).max() / scale, 1e-5)

    @parameterized.product(num_tokens=[8, 16, 17], sinkhorn_repeat=[1, 20])
    def test_pre(self, num_tokens, sinkhorn_repeat):
        from vllm_torchtpu.kernels.deepseek_v4.mhc import pre_kernel

        rng = np.random.default_rng(12)
        residual, fn, hc_scale, hc_base = _make_pre_inputs(rng, num_tokens, 7168)
        args = (
            jnp.asarray(residual).astype(jnp.bfloat16),
            jnp.asarray(fn),
            jnp.asarray(hc_scale),
            jnp.asarray(hc_base),
            RMS_EPS,
            HC_PRE_EPS,
            HC_SINKHORN_EPS,
            HC_POST_MULT_VALUE,
            sinkhorn_repeat,
        )
        want_post, want_comb, want_layer = pre_kernel.mhc_pre(*args)
        got_post, got_comb, got_layer = pre_kernel.mhc_pre(*args, gates_in_kernel=True)

        np.testing.assert_array_equal(_to_np(got_layer), _to_np(want_layer))
        self._check_gates(got_post, got_comb, want_post, want_comb)

    @parameterized.product(
        num_tokens=[8, 16, 17], weight_dtype=[jnp.bfloat16, jnp.float32]
    )
    def test_fused_norm(self, num_tokens, weight_dtype):
        """Fused attn/ffn RMSNorm vs vLLM's rms_norm on the unnormalized
        layer input: at most a rare 1-ulp bf16 flip (f32 mean-reduction
        order)."""
        from vllm_torchtpu.kernels.deepseek_v4.mhc import (
            fused_post_pre_kernel,
            pre_kernel,
        )

        rng = np.random.default_rng(14)
        weight = jnp.asarray(rng.uniform(0.5, 1.5, (7168,)), weight_dtype)
        norm = dict(norm_weight=weight, norm_eps=RMS_EPS)

        def ref_norm(x):
            xf = x.astype(jnp.float32)
            var = jnp.mean(xf * xf, axis=-1, keepdims=True)
            y = (xf * jax.lax.rsqrt(var + RMS_EPS)).astype(weight.dtype) * weight
            return y.astype(jnp.bfloat16)

        args = _fused_args(rng, num_tokens, 7168)
        fused_fn = fused_post_pre_kernel.mhc_fused_post_pre
        want = fused_fn(*args, gates_in_kernel=True)
        got = fused_fn(*args, gates_in_kernel=True, **norm)
        for g, w in zip(got[:3], want[:3]):
            np.testing.assert_array_equal(_to_np(g), _to_np(w))
        np.testing.assert_allclose(
            _to_np(got[3]), _to_np(ref_norm(want[3])), rtol=8e-3, atol=1e-6
        )

        residual, fn, hc_scale, hc_base = _make_pre_inputs(rng, num_tokens, 7168)
        pre_args = (
            jnp.asarray(residual).astype(jnp.bfloat16),
            jnp.asarray(fn),
            jnp.asarray(hc_scale),
            jnp.asarray(hc_base),
            RMS_EPS,
            HC_PRE_EPS,
            HC_SINKHORN_EPS,
            HC_POST_MULT_VALUE,
            20,
        )
        want = pre_kernel.mhc_pre(*pre_args, gates_in_kernel=True)
        got = pre_kernel.mhc_pre(*pre_args, gates_in_kernel=True, **norm)
        np.testing.assert_allclose(
            _to_np(got[2]), _to_np(ref_norm(want[2])), rtol=8e-3, atol=1e-6
        )

    def test_padded_zero_rows_are_finite(self):
        from vllm_torchtpu.kernels.deepseek_v4.mhc import pre_kernel

        rng = np.random.default_rng(13)
        residual, fn, hc_scale, hc_base = _make_pre_inputs(rng, 16, 7168)
        residual[5:] = 0.0
        outs = pre_kernel.mhc_pre(
            jnp.asarray(residual).astype(jnp.bfloat16),
            jnp.asarray(fn),
            jnp.asarray(hc_scale),
            jnp.asarray(hc_base),
            RMS_EPS,
            HC_PRE_EPS,
            HC_SINKHORN_EPS,
            HC_POST_MULT_VALUE,
            20,
            gates_in_kernel=True,
        )
        for out in outs:
            self.assertTrue(bool(jnp.isfinite(out.astype(jnp.float32)).all()))


@pytest.mark.tpu_test
class MhcStackedFnTpuTest(parameterized.TestCase):
    """Stacked-chunk single-pass GEMM (utils.CatFn), the 128-token block and
    the uncapped in-kernel gates vs the default three-chunk path. Streams are bit-identical (they never see fn); the
    gates differ only by GEMM partial-sum rounding (tolerance tier)."""

    def setUp(self):
        super().setUp()
        if jax.default_backend() != "tpu":
            self.skipTest("Pallas TPU kernel needs TPU hardware")

    @parameterized.product(
        num_tokens=[8, 16, 17, 128, 600],
        variant=["cat", "cat_tb128_gkall"],
    )
    def test_fused(self, num_tokens, variant):
        from vllm_torchtpu.kernels.deepseek_v4.mhc import fused_post_pre_kernel

        args = list(_fused_args(np.random.default_rng(21), num_tokens, 7168))
        fused_fn = fused_post_pre_kernel.mhc_fused_post_pre
        want = fused_fn(*args, gates_in_kernel=True, gates_max_tokens=0)
        args[4] = utils.cat_fn3(args[4])
        kw = dict(gates_in_kernel=True, gates_max_tokens=0)
        if variant == "cat_tb128_gkall":
            kw["token_block_size"] = 128
        got = fused_fn(*args, **kw)
        np.testing.assert_array_equal(_to_np(got[0]), _to_np(want[0]))
        for g, w in zip(got[1:3], want[1:3]):
            np.testing.assert_allclose(
                _to_np(g), _to_np(w), rtol=GATES_RTOL * 10, atol=GATES_ATOL
            )
        # layer input: pre_mix moves by a few f32 ulp -> rare 1-ulp bf16 flips
        np.testing.assert_allclose(
            _to_np(got[3]), _to_np(want[3]), rtol=8e-3, atol=1e-6
        )

    def test_pre(self):
        from vllm_torchtpu.kernels.deepseek_v4.mhc import pre_kernel

        rng = np.random.default_rng(22)
        residual, fn, hc_scale, hc_base = _make_pre_inputs(rng, 16, 7168)
        args = [
            jnp.asarray(residual).astype(jnp.bfloat16),
            jnp.asarray(fn),
            jnp.asarray(hc_scale),
            jnp.asarray(hc_base),
            RMS_EPS,
            HC_PRE_EPS,
            HC_SINKHORN_EPS,
            HC_POST_MULT_VALUE,
            20,
        ]
        want = pre_kernel.mhc_pre(*args, gates_in_kernel=True)
        args[1] = utils.cat_fn3(args[1])
        got = pre_kernel.mhc_pre(*args, gates_in_kernel=True)
        for g, w in zip(got[:2], want[:2]):
            np.testing.assert_allclose(
                _to_np(g), _to_np(w), rtol=GATES_RTOL * 10, atol=GATES_ATOL
            )
        np.testing.assert_allclose(_to_np(got[2]), _to_np(want[2]), rtol=8e-3, atol=1e-6)


@pytest.mark.tpu_test
class MhcPostHeadTpuTest(parameterized.TestCase):
    """Fused last post + hc head collapse + stream mean (post_head_kernel)."""

    def setUp(self):
        super().setUp()
        if jax.default_backend() != "tpu":
            self.skipTest("Pallas TPU kernel needs TPU hardware")

    @parameterized.parameters(8, 16, 130)
    def test_matches_post_then_head(self, num_tokens):
        from vllm_torchtpu.kernels.deepseek_v4.mhc import post_head_kernel, post_kernel

        rng = np.random.default_rng(31)
        x, res, plm, crm = _make_post_inputs(rng, num_tokens, 7168)
        fn = jnp.asarray(rng.standard_normal((HC_MULT, HC_MULT * 7168), np.float32) * 0.02)
        sc = jnp.asarray((1 + 0.1 * rng.standard_normal(1)).astype(np.float32))
        hb = jnp.asarray((0.5 * rng.standard_normal(HC_MULT)).astype(np.float32))
        head, mean = post_head_kernel.mhc_post_head(
            x, res, plm, crm, fn, sc, hb, RMS_EPS, HC_PRE_EPS, want_mean=True
        )
        rec = post_kernel.mhc_post(x, res, plm, crm)
        # The stream mean: bit-identical to post followed by a 4-way mean.
        f = _to_np(rec)
        want_mean = (((f[:, 0] + f[:, 2]) + (f[:, 1] + f[:, 3])) * 0.25).astype(np.float32)
        np.testing.assert_array_equal(
            _to_np(mean), _to_np(jnp.asarray(want_mean).astype(jnp.bfloat16))
        )
        # The head: f32 projection, vs a float64 evaluation of mhc_collapse_head.
        r64 = f.astype(np.float64).reshape(num_tokens, -1)
        nrm = r64 / np.sqrt((r64 * r64).mean(-1, keepdims=True) + RMS_EPS)
        z = nrm @ np.asarray(fn, np.float64).T
        pm = 1 / (1 + np.exp(-(z * float(sc[0]) + np.asarray(hb, np.float64)))) + HC_PRE_EPS
        y = (pm[:, :, None] * f.astype(np.float64)).sum(1)
        np.testing.assert_allclose(_to_np(head), y, rtol=8e-3, atol=2e-2)


if __name__ == "__main__":
    absltest.main()
