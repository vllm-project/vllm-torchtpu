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
"""Tests for W8A16 on small FP8 dense-linear steps (TPU_FP8_LINEAR_W8A16_MAX_ROWS)."""

from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_torchtpu.layers.adapter import linear_common
from vllm_torchtpu.layers.adapter.linear_common import _quantized_matmul_jax
from vllm_torchtpu.layers.adapter.quantization import fp8
from vllm_torchtpu.layers.adapter.quantization.configs import VllmQuantLinearConfig
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU

_K, _N = 256, 128


def _weights(seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((_K, _N)).astype(np.float32)
    scale = np.abs(w).max(axis=0) / 448.0
    w_q = jnp.asarray(w / scale, dtype=jnp.float8_e4m3fn)
    return w_q, jnp.asarray(scale, dtype=jnp.float32)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "rows,max_rows,expect_quantized",
    [
        (16, 0, True),  # default: W8A8 everywhere
        (16, 16, False),
        (17, 16, True),
        (64, 64, False),
        (256, 64, True),
    ],
)
def test_activation_quantization_follows_threshold(
    monkeypatch, rows, max_rows, expect_quantized
):
    seen = {}
    real = linear_common.xla_quantized_matmul

    def spy(x, w_q, w_s, quantize_activation=True):
        seen["quantize_activation"] = quantize_activation
        return real(x, w_q, w_s, quantize_activation=quantize_activation)

    monkeypatch.setattr(linear_common, "xla_quantized_matmul", spy)
    w_q, w_s = _weights()
    out = _quantized_matmul_jax(
        jnp.ones((rows, _K), jnp.bfloat16), w_q, w_s, w8a16_max_rows=max_rows
    )
    assert out.shape == (rows, _N)
    assert out.dtype == jnp.bfloat16
    assert seen["quantize_activation"] is expect_quantized


@pytest.mark.cpu_test
def test_int8_weights_keep_w8a8_below_threshold(monkeypatch):
    seen = {}
    real = linear_common.xla_quantized_matmul

    def spy(x, w_q, w_s, quantize_activation=True):
        seen["quantize_activation"] = quantize_activation
        return real(x, w_q, w_s, quantize_activation=quantize_activation)

    monkeypatch.setattr(linear_common, "xla_quantized_matmul", spy)
    w_q = jnp.ones((_K, _N), jnp.int8)
    w_s = jnp.ones((_N,), jnp.float32)
    _quantized_matmul_jax(jnp.ones((16, _K), jnp.bfloat16), w_q, w_s, 64)
    assert seen["quantize_activation"] is True


@pytest.mark.cpu_test
def test_blockwise_path_ignores_threshold(monkeypatch):
    seen = {}

    def fake_gmm_v2(lhs, rhs, maybe_quantize_lhs, **kwargs):
        seen["maybe_quantize_lhs"] = maybe_quantize_lhs
        return jnp.zeros((lhs.shape[0], rhs.shape[-1]), lhs.dtype)

    monkeypatch.setattr(linear_common, "gmm_v2", fake_gmm_v2)
    w_q, _ = _weights()
    num_blocks = 2
    w_s = jnp.ones((1, num_blocks, 1, _N), jnp.float32)
    _quantized_matmul_jax(jnp.ones((16, _K), jnp.bfloat16), w_q, w_s, 64)
    assert seen["maybe_quantize_lhs"] is True


@pytest.mark.cpu_test
def test_direct_jax_bridge_defaults_to_w8a8(monkeypatch):
    seen = {}
    real = linear_common.xla_quantized_matmul

    def spy(x, w_q, w_s, quantize_activation=True):
        seen["quantize_activation"] = quantize_activation
        return real(x, w_q, w_s, quantize_activation=quantize_activation)

    monkeypatch.setattr(linear_common, "xla_quantized_matmul", spy)
    w_q, w_s = _weights()
    _quantized_matmul_jax(jnp.ones((16, _K), jnp.bfloat16), w_q, w_s)
    assert seen["quantize_activation"] is True


@pytest.mark.cpu_test
@pytest.mark.parametrize("env_val,expected", [(None, 0), ("0", 0), ("16", 16)])
def test_fp8_linear_method_forwards_w8a16_max_rows(monkeypatch, env_val, expected):
    if env_val is None:
        monkeypatch.delenv("TPU_FP8_LINEAR_W8A16_MAX_ROWS", raising=False)
    else:
        monkeypatch.setenv("TPU_FP8_LINEAR_W8A16_MAX_ROWS", env_val)

    method = VllmFp8LinearMethodTPU(Fp8Config(is_checkpoint_fp8_serialized=True))
    assert method.w8a16_max_rows == expected

    seen = {}

    def fake_quantized_matmul(x, w_q, w_s, w8a16_max_rows=0):
        seen["w8a16_max_rows"] = w8a16_max_rows
        return torch.zeros((x.shape[0], w_q.shape[-1]), dtype=x.dtype)

    monkeypatch.setattr(fp8, "quantized_matmul", fake_quantized_matmul)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros((_K, _N), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.ones((_N,), dtype=torch.float32), requires_grad=False
    )
    method.apply(layer, torch.ones((16, _K), dtype=torch.bfloat16))
    assert seen["w8a16_max_rows"] == expected


@pytest.mark.cpu_test
def test_fp8_linear_method_takes_w8a16_max_rows_from_linear_config(monkeypatch):
    # Production path: VllmQuantLinearConfig reads the env when the layer is
    # built, and the linear method uses that value, not a later env read.
    layer = MagicMock(spec=LinearBase)
    layer.output_size = _N
    monkeypatch.setenv("TPU_FP8_LINEAR_W8A16_MAX_ROWS", "32")
    linear_config = VllmQuantLinearConfig(None, layer)
    assert linear_config.w8a16_max_rows == 32
    monkeypatch.setenv("TPU_FP8_LINEAR_W8A16_MAX_ROWS", "0")

    method = VllmFp8LinearMethodTPU(
        Fp8Config(is_checkpoint_fp8_serialized=True), linear_config=linear_config
    )
    assert method.w8a16_max_rows == 32

    seen = {}

    def fake_quantized_matmul(x, w_q, w_s, w8a16_max_rows=0):
        seen["w8a16_max_rows"] = w8a16_max_rows
        return torch.zeros((x.shape[0], w_q.shape[-1]), dtype=x.dtype)

    monkeypatch.setattr(fp8, "quantized_matmul", fake_quantized_matmul)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros((_K, _N), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.ones((_N,), dtype=torch.float32), requires_grad=False
    )
    method.apply(layer, torch.ones((16, _K), dtype=torch.bfloat16))
    assert seen["w8a16_max_rows"] == 32


@pytest.mark.cpu_test
def test_quantized_matmul_op_accepts_w8a16_max_rows():
    op = linear_common._get_quantized_matmul_op()
    with torch.device("meta"):
        x = torch.empty((16, _K), dtype=torch.bfloat16)
        w_q = torch.empty((_K, _N), dtype=torch.float8_e4m3fn)
        w_s = torch.empty((_N,), dtype=torch.float32)
        out = op(x, w_q, w_s, 64)
    assert out.shape == (16, _N)
    assert out.dtype == torch.bfloat16


@pytest.mark.cpu_test
def test_w8a16_skips_activation_rounding():
    """Kernel-level check only: W8A16 matches the bf16-activation reference at
    least as closely as W8A8, because it skips rounding activations to FP8.
    The reference is the W8A16 computation in f32, so this says nothing about
    end-to-end model accuracy."""
    w_q, w_s = _weights()
    x = jnp.asarray(
        np.random.default_rng(1).standard_normal((16, _K)), dtype=jnp.bfloat16
    )
    # Reference: bf16 activations against the dequantized FP8 weights, in f32.
    ref = np.asarray(x, np.float32) @ (
        np.asarray(w_q.astype(jnp.float32)) * np.asarray(w_s)[None, :]
    )

    w8a8 = np.asarray(_quantized_matmul_jax(x, w_q, w_s, w8a16_max_rows=0), np.float32)
    w8a16 = np.asarray(
        _quantized_matmul_jax(x, w_q, w_s, w8a16_max_rows=16), np.float32
    )

    err_w8a8 = np.abs(w8a8 - ref).max()
    err_w8a16 = np.abs(w8a16 - ref).max()
    assert err_w8a16 <= err_w8a8
