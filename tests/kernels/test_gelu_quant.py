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
"""Pallas interpret and TPU tests for fused GeGLU + per-token FP8 quantization."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels import gelu_quant


def _ref_gelu_tanh_and_mul_quantize(
    x: jax.Array, quant_dtype: jnp.dtype = jnp.float8_e4m3fn
) -> tuple[jax.Array, jax.Array]:
    d = x.shape[1] // 2
    gate = x[:, :d].astype(jnp.float32)
    up = x[:, d:].astype(jnp.float32)
    act = (jax.nn.gelu(gate, approximate=True) * up).astype(x.dtype)
    f32 = act.astype(jnp.float32)
    finfo = jnp.finfo(quant_dtype)
    qmax = float(finfo.max)
    scale = jnp.max(jnp.abs(f32), axis=1, keepdims=True) / qmax
    inv = jnp.where(scale == 0, 0.0, 1.0 / scale)
    q = jnp.clip(f32 * inv, float(finfo.min), qmax).astype(quant_dtype)
    return q, scale


@pytest.mark.parametrize("shape", [(16, 512), (64, 2048)])
def test_gelu_tanh_and_mul_quantize_matches_fp32_reference(
    shape: tuple[int, int],
) -> None:
    n, cols = shape
    x = jax.random.normal(
        jax.random.key(n * 13 + cols), (n, cols), dtype=jnp.float32
    ).astype(jnp.bfloat16)

    q, scale = gelu_quant.gelu_tanh_and_mul_quantize(
        x, quant_dtype=jnp.float8_e4m3fn, max_block_rows=32, interpret=True
    )
    q_ref, s_ref = _ref_gelu_tanh_and_mul_quantize(x)

    assert q.shape == (n, cols // 2)
    assert q.dtype == jnp.float8_e4m3fn
    assert scale.shape == (n, 1)
    assert scale.dtype == jnp.float32

    deq = np.asarray(q.astype(jnp.float32) * scale)
    deq_ref = np.asarray(q_ref.astype(jnp.float32) * s_ref)
    if jax.default_backend() == "cpu":
        np.testing.assert_allclose(np.asarray(scale), np.asarray(s_ref), rtol=1e-5)
        np.testing.assert_allclose(deq, deq_ref, rtol=2e-2, atol=2e-2)
    else:
        np.testing.assert_allclose(
            np.asarray(scale), np.asarray(s_ref), rtol=5e-3, atol=1e-6
        )
        rel_l2 = float(np.linalg.norm(deq - deq_ref) / np.linalg.norm(deq_ref))
        assert rel_l2 < 2e-2
        np.testing.assert_allclose(deq, deq_ref, rtol=0.16, atol=3e-2)


def test_gelu_tanh_and_mul_quantize_tiny_and_zero_rows_stay_finite() -> None:
    d = 256
    tiny = jnp.full((8, 2 * d), 1e-4, dtype=jnp.bfloat16)
    zeros = jnp.zeros((8, 2 * d), dtype=jnp.bfloat16)
    x = jnp.concatenate([tiny, zeros], axis=0)

    q, scale = gelu_quant.gelu_tanh_and_mul_quantize(
        x, jnp.float8_e4m3fn, interpret=True
    )
    q_f32 = np.asarray(q.astype(jnp.float32))
    s_f32 = np.asarray(scale)

    assert np.all(np.isfinite(q_f32))
    assert np.all(np.isfinite(s_f32))
    np.testing.assert_array_equal(q_f32[8:], 0.0)
    np.testing.assert_array_equal(s_f32[8:], 1.0)
    np.testing.assert_array_equal(q_f32[8:] * s_f32[8:], 0.0)
    deq = q_f32[:8] * s_f32[:8]
    ref_q, ref_s = _ref_gelu_tanh_and_mul_quantize(x[:8])
    ref_deq = np.asarray(ref_q.astype(jnp.float32) * ref_s)
    np.testing.assert_allclose(deq, ref_deq, rtol=5e-2, atol=1e-10)


@pytest.mark.tpu_test
def test_gelu_tanh_and_mul_quantize_on_tpu() -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("Requires TPU backend")
    x = jax.random.normal(jax.random.key(55), (32, 1024), dtype=jnp.bfloat16)
    q, scale = gelu_quant.gelu_tanh_and_mul_quantize(x, jnp.float8_e4m3fn)
    q_ref, s_ref = _ref_gelu_tanh_and_mul_quantize(x)
    np.testing.assert_allclose(
        np.asarray(scale), np.asarray(s_ref), rtol=5e-3, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(q.astype(jnp.float32) * scale),
        np.asarray(q_ref.astype(jnp.float32) * s_ref),
        rtol=0.16,
        atol=3e-2,
    )
