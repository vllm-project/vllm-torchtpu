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
class MhcGenericGatesTest(parameterized.TestCase):
    """Generic gates retain FP32 normalization for one or multiple tokens."""

    def setUp(self):
        super().setUp()
        if jax.default_backend() != "tpu":
            self.skipTest("Pallas TPU kernel needs TPU hardware")

    @parameterized.product(
        num_tokens=[1, 8],
        repeats=[1, 2, 20],
        amplitude=[0.0, 1.0, 100.0],
    )
    def test_matches_independent_normalization(self, num_tokens, repeats, amplitude):
        rng = np.random.default_rng(713)
        mixes = rng.normal(size=(num_tokens, 24)).astype(np.float32) * amplitude
        squares = rng.uniform(8192, 32768, size=(num_tokens, 1)).astype(np.float32)
        scale = np.array([0.3, 0.5, 0.7], dtype=np.float32)
        base = rng.normal(size=24).astype(np.float32) * amplitude
        normalized = mixes.astype(np.float64) / np.sqrt(squares / 16384 + RMS_EPS)
        pre = 1 / (1 + np.exp(-(normalized[:, :4] * scale[0] + base[:4])))
        post = 1 / (1 + np.exp(-(normalized[:, 4:8] * scale[1] + base[4:8])))
        logits = (normalized[:, 8:] * scale[2] + base[8:]).reshape(-1, 4, 4)
        comb = np.exp(logits - logits.max(axis=-1, keepdims=True))
        comb = comb / comb.sum(axis=-1, keepdims=True) + HC_SINKHORN_EPS
        comb = comb / (comb.sum(axis=-2, keepdims=True) + HC_SINKHORN_EPS)
        for _ in range(repeats - 1):
            comb = comb / (comb.sum(axis=-1, keepdims=True) + HC_SINKHORN_EPS)
            comb = comb / (comb.sum(axis=-2, keepdims=True) + HC_SINKHORN_EPS)

        def call(m, q, c, b):
            return utils.mhc_pre_gates(
                m,
                q,
                4,
                4096,
                c,
                b,
                RMS_EPS,
                HC_PRE_EPS,
                HC_SINKHORN_EPS,
                HC_POST_MULT_VALUE,
                repeats,
            )

        compiled = (
            jax.jit(call)
            .lower(
                jnp.asarray(mixes),
                jnp.asarray(squares),
                jnp.asarray(scale),
                jnp.asarray(base),
            )
            .compile()
        )
        got = compiled(mixes, squares, scale, base)
        expected = (pre + HC_PRE_EPS, post * HC_POST_MULT_VALUE, comb)
        for actual, wanted in zip(got, expected):
            self.assertEqual(actual.dtype, jnp.float32)
            np.testing.assert_allclose(actual, wanted, rtol=1e-5, atol=1e-6)
        self.assertNotIn("mhc_sinkhorn_fp32_single_token", compiled.as_text())


@pytest.mark.tpu_test
class MhcSingleTokenNoPaddingTest(parameterized.TestCase):
    @parameterized.product(repeats=[1, 2, 20], hidden_size=[256, 4096])
    def test_seam_avoids_hbm_token_padding(self, repeats, hidden_size):
        from vllm_torchtpu.kernels.deepseek_v4.mhc import fused_post_pre_kernel

        args = list(_fused_args(np.random.default_rng(711), 1, hidden_size))
        args[-1] = repeats
        constants = args[-5:]
        compiled = (
            jax.jit(lambda *a: fused_post_pre_kernel.mhc_fused_post_pre(*a, *constants))
            .lower(*args[:-5])
            .compile()
        )
        actual = compiled(*args[:-5])
        expected = _ref_mhc_fused_post_pre(*args)
        for got, want in zip(actual, expected):
            rtol, atol = (
                (BF16_RTOL, BF16_ATOL)
                if got.dtype == jnp.bfloat16
                else (FUSED_SMALL_H_RTOL, FUSED_F32_ATOL)
            )
            np.testing.assert_allclose(_to_np(got), _to_np(want), rtol=rtol, atol=atol)
        self.assertNotIn("bf16[16,", compiled.as_text())


@pytest.mark.tpu_test
class MhcWeightSplitPlacementTest(parameterized.TestCase):
    @parameterized.product(
        num_tokens=[1, 8], hidden_size=[256, 4096], path=["pre", "fused"]
    )
    def test_single_token_consumes_fp32_weights_directly(
        self, num_tokens, hidden_size, path
    ):
        from vllm_torchtpu.kernels.deepseek_v4.mhc import (
            fused_post_pre_kernel,
            pre_kernel,
        )

        args = _fused_args(np.random.default_rng(823), num_tokens, hidden_size)
        if path == "pre":
            inputs = (args[1], args[4], args[5], args[6])
            constants = args[-5:]
            operation = pre_kernel.mhc_pre
            expected = _ref_mhc_pre(*inputs, *constants)
        else:
            inputs = args[:-5]
            constants = args[-5:]
            operation = fused_post_pre_kernel.mhc_fused_post_pre
            expected = _ref_mhc_fused_post_pre(*args)
        compiled = (
            jax.jit(lambda *a: operation(*a, *constants)).lower(*inputs).compile()
        )
        actual = compiled(*inputs)
        for got, want in zip(actual, expected):
            rtol, atol = (
                (BF16_RTOL, BF16_ATOL)
                if got.dtype == jnp.bfloat16
                else (FUSED_SMALL_H_RTOL, FUSED_F32_ATOL)
            )
            np.testing.assert_allclose(_to_np(got), _to_np(want), rtol=rtol, atol=atol)
        # Decode passes original FP32 weights to Pallas; batched execution
        # reuses the external split across token blocks.
        fp32_weight = f"f32[24,{4 * hidden_size}]"
        direct = any(
            "custom-call(" in line and fp32_weight in line
            for line in compiled.as_text().splitlines()
        )
        self.assertEqual(direct, num_tokens == 1)
        # Decode now completes gates in the projection kernel.
        has_separate_sinkhorn = "mhc_sinkhorn_fp32_single_token" in compiled.as_text()
        if num_tokens == 1:
            self.assertFalse(has_separate_sinkhorn)


@pytest.mark.tpu_test
class MHCNormQuantTest(parameterized.TestCase):
    """Check the fused boundary against stored BF16 mHC output and Torch norm."""

    @parameterized.product(
        hidden_size=[256, 4096, 7168, 16384],
        fused=[False, True],
        norm_dtype=[jnp.bfloat16, jnp.float32],
    )
    def test_norm_quant_preserves_residual_and_gates(
        self, hidden_size, fused, norm_dtype
    ):
        from vllm_torchtpu.kernels.deepseek_v4.mhc import (
            fused_post_pre_kernel,
            norm_quant_kernel,
            pre_kernel,
        )
        from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor

        rng = np.random.default_rng(913)
        residual, fn, scale, base = _make_pre_inputs(rng, 1, hidden_size)
        residual = jnp.asarray(residual, jnp.bfloat16)
        fn, scale, base = (jnp.asarray(v, jnp.float32) for v in (fn, scale, base))
        weight = jnp.asarray(rng.normal(1.0, 0.1, size=hidden_size), dtype=norm_dtype)
        constants = (RMS_EPS, HC_PRE_EPS, HC_SINKHORN_EPS, HC_POST_MULT_VALUE, 20)
        if fused:
            x = jnp.asarray(rng.normal(size=(1, hidden_size)), jnp.bfloat16)
            post = jnp.ones((1, 4, 1), jnp.float32)
            comb = jnp.eye(4, dtype=jnp.float32)[None]
            args = (x, residual, post, comb, fn, scale, base)
            baseline = fused_post_pre_kernel.mhc_fused_post_pre(*args, *constants)
            got = norm_quant_kernel.mhc_fused_post_pre_norm_quant(
                *args,
                weight,
                constants=constants,
                norm_eps=RMS_EPS,
                quant_dtype=jnp.float8_e4m3fn,
            )
        else:
            args = (residual, fn, scale, base)
            baseline = pre_kernel.mhc_pre(*args, *constants)
            got = norm_quant_kernel.mhc_pre_norm_quant(
                *args,
                weight,
                constants=constants,
                norm_eps=RMS_EPS,
                quant_dtype=jnp.float8_e4m3fn,
            )
        for ref, value in zip(baseline[:-1], got[:-2]):
            np.testing.assert_array_equal(_to_np(ref), _to_np(value))
        raw = torch.from_numpy(_to_np(baseline[-1])).to(torch.bfloat16)
        wt = torch.from_numpy(_to_np(weight)).to(
            torch.bfloat16 if norm_dtype == jnp.bfloat16 else torch.float32
        )
        xf = raw.float()
        normalized = (
            xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + RMS_EPS)
        ).to(wt.dtype)
        expected = (normalized * wt).to(raw.dtype).float().numpy()
        np.testing.assert_allclose(
            _to_np(got[-2]), expected, rtol=BF16_RTOL, atol=BF16_ATOL
        )
        q, qs = quantize_tensor(got[-2], jnp.float8_e4m3fn)
        np.testing.assert_array_equal(_to_np(got[-1][0]), _to_np(q))
        np.testing.assert_array_equal(_to_np(got[-1][1]), _to_np(qs))

        if fused:
            norm_only = norm_quant_kernel.mhc_fused_post_pre_norm_quant(
                *args,
                weight,
                constants=constants,
                norm_eps=RMS_EPS,
                quant_dtype=None,
            )
            self.assertEmpty(norm_only[-1])
            for ref, value in zip(got[:-1], norm_only[:-1]):
                np.testing.assert_array_equal(_to_np(ref), _to_np(value))

        from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul
        from vllm_torchtpu.layers.adapter.linear_common import (
            _prequantized_bf16_matmul_jax,
        )

        w = jnp.asarray(rng.normal(size=(hidden_size, 128)), jnp.float8_e4m3fn)
        ws = jnp.ones((128,), jnp.float32)
        projected = _prequantized_bf16_matmul_jax(got[-1][0], got[-1][1], w, ws)
        reference = xla_quantized_matmul(got[-2], w, ws)
        np.testing.assert_array_equal(_to_np(projected), _to_np(reference))

    @parameterized.parameters(
        (1, 4, 4096, True), (8, 4, 4096, False), (1, 2, 4096, False), (1, 4, 257, False)
    )
    def test_supported_shapes(self, tokens, streams, hidden, expected):
        from vllm_torchtpu.kernels.deepseek_v4.mhc.norm_quant_kernel import (
            supports_mhc_norm_quant,
        )

        self.assertEqual(supports_mhc_norm_quant(tokens, streams, hidden), expected)

    def test_zero_input_has_unit_quant_scale(self):
        from vllm_torchtpu.kernels.deepseek_v4.mhc import norm_quant_kernel

        h = 256
        post, comb, y, (q, scale) = norm_quant_kernel.mhc_pre_norm_quant(
            jnp.zeros((1, 4, h), jnp.bfloat16),
            jnp.zeros((24, 4 * h), jnp.float32),
            jnp.ones((3,), jnp.float32),
            jnp.zeros((24,), jnp.float32),
            jnp.ones((h,), jnp.bfloat16),
            constants=(RMS_EPS, HC_PRE_EPS, HC_SINKHORN_EPS, HC_POST_MULT_VALUE, 20),
            norm_eps=RMS_EPS,
            quant_dtype=jnp.float8_e4m3fn,
        )
        np.testing.assert_array_equal(_to_np(y), 0)
        np.testing.assert_array_equal(_to_np(q), 0)
        np.testing.assert_array_equal(_to_np(scale), 1)
        self.assertTrue(np.isfinite(_to_np(post)).all())
        self.assertTrue(np.isfinite(_to_np(comb)).all())


@pytest.mark.tpu_test
@pytest.mark.parametrize("hidden_size", [128, 4096])
@pytest.mark.parametrize("flat", [False, True])
def test_single_token_post_matches_batched_recombination(hidden_size, flat):
    """Stream layout changes must preserve nontrivial post/comb coefficients."""
    from vllm_torchtpu.kernels.deepseek_v4.mhc import post_kernel

    rng = np.random.default_rng(1210)
    h = hidden_size
    x = jnp.asarray(rng.normal(size=(2, h)), jnp.bfloat16)
    residual = jnp.asarray(rng.normal(size=(2, 4, h)), jnp.bfloat16)
    post = jnp.asarray(rng.uniform(0, 2, size=(2, 4, 1)), jnp.float32)
    comb = jax.nn.softmax(jnp.asarray(rng.normal(size=(2, 4, 4)), jnp.float32), axis=-1)
    expected = post_kernel.mhc_post(x, residual, post, comb)[:1]
    if flat:
        actual = post_kernel.mhc_post_2d(
            x[:1], residual[:1].reshape(1, 4 * h), post[:1], comb[:1]
        ).reshape(1, 4, h)
    else:
        actual = post_kernel.mhc_post(x[:1], residual[:1], post[:1], comb[:1])
    np.testing.assert_array_equal(_to_np(actual), _to_np(expected))


@pytest.mark.tpu_test
@pytest.mark.parametrize("hidden_size", [256, 4096])
@pytest.mark.parametrize("repeats", [1, 20])
@pytest.mark.parametrize("norm_dtype", [jnp.bfloat16, jnp.float32])
def test_broadcast_pre_matches_repeated_streams(hidden_size, repeats, norm_dtype):
    from vllm_torchtpu.kernels.deepseek_v4.mhc import norm_quant_kernel
    from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor

    rng = np.random.default_rng(1012)
    h = hidden_size
    x = jnp.asarray(rng.normal(size=(1, h)), jnp.bfloat16)
    fn = jnp.asarray(rng.normal(0, 0.01, size=(24, 4 * h)), jnp.float32)
    scale = jnp.array([0.3, 0.5, 0.7], jnp.float32)
    base = jnp.asarray(rng.normal(size=24), jnp.float32)
    weight = jnp.asarray(rng.normal(1, 0.1, size=h), norm_dtype)
    constants = (RMS_EPS, HC_PRE_EPS, HC_SINKHORN_EPS, HC_POST_MULT_VALUE, repeats)
    residual = jnp.broadcast_to(x[:, None, :], (1, 4, h))
    expected = norm_quant_kernel.mhc_pre_norm_quant(
        residual, fn, scale, base, weight, constants=constants, norm_eps=RMS_EPS
    )
    actual_residual, *actual = norm_quant_kernel.mhc_broadcast_pre_norm_quant(
        x,
        fn.reshape(24, 4, h).sum(axis=1),
        scale,
        base,
        weight,
        constants=constants,
        norm_eps=RMS_EPS,
    )
    np.testing.assert_array_equal(_to_np(actual_residual), _to_np(residual))
    for got, want in zip(actual[:2], expected[:2]):
        np.testing.assert_allclose(_to_np(got), _to_np(want), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        _to_np(actual[2]), _to_np(expected[2]), rtol=BF16_RTOL, atol=BF16_ATOL
    )
    q, qs = quantize_tensor(actual[2], jnp.float8_e4m3fn)
    np.testing.assert_array_equal(_to_np(actual[3][0]), _to_np(q))
    np.testing.assert_array_equal(_to_np(actual[3][1]), _to_np(qs))


@pytest.mark.tpu_test
@pytest.mark.parametrize("repeats", [1, 2, 7, 20])
@pytest.mark.parametrize("logit_scale", [0.0, 0.01, 1.0, 20.0])
@pytest.mark.parametrize("eps", [1e-6, 1e-3])
def test_fused_sinkhorn_preserves_fp32_reduction_tree(repeats, logit_scale, eps):
    """Register layout changes must preserve both four-value reduction trees."""
    from jax.experimental import pallas as pl

    from vllm_torchtpu.kernels.deepseek_v4.mhc.sinkhorn_kernel import (
        single_token_post_comb,
    )

    def kernel(base_ref, logits_ref, output_ref, reference_ref):
        _, comb = single_token_post_comb(
            jnp.zeros((1, 24), jnp.float32),
            jnp.ones((1, 3), jnp.float32),
            base_ref[...],
            eps,
            HC_POST_MULT_VALUE,
            repeats,
        )
        output_ref[...] = comb
        # Matrix-axis reference, independent of the fused lane-replicated layout.
        value = jax.nn.softmax(logits_ref[...], axis=-1) + eps
        value = value / (jnp.sum(value, axis=0, keepdims=True) + eps)

        def normalize(_, matrix):
            matrix = matrix / (jnp.sum(matrix, axis=1, keepdims=True) + eps)
            return matrix / (jnp.sum(matrix, axis=0, keepdims=True) + eps)

        reference_ref[...] = jax.lax.fori_loop(0, repeats - 1, normalize, value)

    shape = jax.ShapeDtypeStruct((4, 4), jnp.float32)
    fused = pl.pallas_call(kernel, out_shape=(shape, shape))
    for seed in (913, 1210):
        rng = np.random.default_rng(seed)
        logits = jnp.asarray(rng.normal(size=(1, 4, 4)) * logit_scale, jnp.float32)
        base = jnp.concatenate(
            [jnp.zeros((1, 8), jnp.float32), logits.reshape(1, 16)], axis=1
        )
        actual, expected = fused(base, logits[0])
        np.testing.assert_array_equal(_to_np(actual), _to_np(expected))


if __name__ == "__main__":
    absltest.main()
