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
"""Pallas interpret, TPU, and contract tests for decode GEMV kernels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu import tpu_info
from vllm_torchtpu.kernels.quantized_matmul import decode_gemv, util


def test_is_tpu_v7x_returns_false_on_cpu_backend(monkeypatch) -> None:
    if jax.default_backend() != "cpu":
        pytest.skip("CPU-backend fallback check")
    if not hasattr(tpu_info.envs, "CPU_ONLY"):
        monkeypatch.setattr(tpu_info.envs, "CPU_ONLY", False, raising=False)
    assert tpu_info.is_tpu_v7x() is False


def test_decode_gemv_fp8_and_prequantized_and_nt_match_reference() -> None:
    assert decode_gemv.supports(16, 256, 512, jnp.float8_e4m3fn, 1)
    assert decode_gemv.supports(8, 256, 512, jnp.float8_e4m3fn, 1, row_multiple=8)
    assert not decode_gemv.supports(8, 256, 512, jnp.float8_e4m3fn, 2)
    assert decode_gemv.supports_nt(16, 256, 512, itemsize=2)

    m, k, n = 16, 256, 512
    k1, k2, k3 = jax.random.split(jax.random.key(301), 3)
    x = jax.random.normal(k1, (m, k), dtype=jnp.bfloat16)
    w_f32 = jax.random.normal(k2, (n, k), dtype=jnp.float32) * 0.25
    w_n_k, w_scale = util.quantize_tensor(w_f32, jnp.float8_e4m3fn, dim=1)
    w_k_n = w_n_k.T
    s_1d = w_scale.squeeze(-1)

    y_gemv = decode_gemv.fp8_decode_matmul(
        x, w_k_n, s_1d, tile_bytes=65536, interpret=True
    )
    y_ref = util.xla_quantized_matmul(x, w_k_n, s_1d)
    np.testing.assert_allclose(
        np.asarray(y_gemv, dtype=np.float32),
        np.asarray(y_ref, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )

    x_q, x_scale = util.quantize_tensor(x, jnp.float8_e4m3fn, dim=1)
    y_pre_pallas = decode_gemv.fp8_decode_prequantized_matmul(
        x_q,
        x_scale,
        w_k_n,
        s_1d,
        out_dtype=jnp.bfloat16,
        tile_bytes=65536,
        interpret=True,
    )
    y_pre_xla = util.xla_prequantized_matmul(
        x_q, x_scale, w_k_n, s_1d, out_dtype=jnp.bfloat16
    )
    np.testing.assert_allclose(
        np.asarray(y_pre_pallas, dtype=np.float32),
        np.asarray(y_pre_xla, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )

    w_bf16 = jax.random.normal(k3, (n, k), dtype=jnp.bfloat16) * 0.2
    y_nt = decode_gemv.decode_matmul_nt(x, w_bf16, tile_bytes=65536, interpret=True)
    y_nt_ref = jnp.dot(x, w_bf16.T, preferred_element_type=jnp.float32).astype(
        jnp.bfloat16
    )
    np.testing.assert_allclose(
        np.asarray(y_nt, dtype=np.float32),
        np.asarray(y_nt_ref, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.tpu_test
def test_decode_gemv_on_tpu() -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("Requires TPU backend")
    k1, k2, k3 = jax.random.split(jax.random.key(909), 3)
    x = jax.random.normal(k1, (16, 6144), dtype=jnp.bfloat16)
    w_n_k, s_n = util.quantize_tensor(
        jax.random.normal(k2, (2688, 6144), dtype=jnp.float32) * 0.1,
        jnp.float8_e4m3fn,
        dim=1,
    )
    w_k_n, s_1d = w_n_k.T, s_n.squeeze(-1).astype(jnp.float32)
    y_ref = np.asarray(util.xla_quantized_matmul(x, w_k_n, s_1d), np.float32)
    y = np.asarray(decode_gemv.fp8_decode_matmul(x, w_k_n, s_1d), np.float32)
    np.testing.assert_allclose(y, y_ref, rtol=2e-2, atol=2e-1)

    x_q, x_scale = util.quantize_tensor(x, jnp.float8_e4m3fn, dim=1)
    y_pre = np.asarray(
        decode_gemv.fp8_decode_prequantized_matmul(
            x_q, x_scale, w_k_n, s_1d, out_dtype=jnp.bfloat16
        ),
        np.float32,
    )
    y_pre_ref = np.asarray(
        util.xla_prequantized_matmul(x_q, x_scale, w_k_n, s_1d), np.float32
    )
    np.testing.assert_allclose(y_pre, y_pre_ref, rtol=2e-2, atol=2e-1)

    w_bf16 = jax.random.normal(k3, (2048, 6144), dtype=jnp.bfloat16) * 0.1
    y_nt = np.asarray(decode_gemv.decode_matmul_nt(x, w_bf16), np.float32)
    y_nt_ref = np.asarray(
        jnp.dot(x, w_bf16.T, preferred_element_type=jnp.float32).astype(jnp.bfloat16),
        np.float32,
    )
    np.testing.assert_allclose(y_nt, y_nt_ref, rtol=2e-2, atol=2e-1)
