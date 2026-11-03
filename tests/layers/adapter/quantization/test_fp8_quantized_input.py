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
"""Tests for the FP8 linear's split activation quant (quantize_input + matmul)."""

import functools

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_torchtpu.layers.adapter import linear_common
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    _prequantized_matmul_bf16_jax,
    _quantize_activation_jax,
    _quantized_matmul_jax,
)
from vllm_torchtpu.layers.adapter.quantization import fp8
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU

_K, _N = 256, 128


@pytest.mark.cpu_test
@pytest.mark.parametrize("w_dtype", [jnp.float8_e4m3fn, jnp.float8_e5m2, jnp.int8])
def test_split_ops_match_quantized_matmul(w_dtype):
    """Rows quantized in shards, gathered, then multiplied == the fused path."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, _K)) * np.linspace(0.01, 30.0, 64)[:, None]
    x = jnp.asarray(x, jnp.bfloat16)
    w_q = jnp.asarray(np.clip(rng.standard_normal((_K, _N)) * 40, -127, 127), w_dtype)
    w_s = jnp.asarray(rng.uniform(size=_N) * 0.01, jnp.float32)
    quantize = functools.partial(_quantize_activation_jax, quant_dtype=w_q.dtype)
    shards = [quantize(part) for part in jnp.split(x, 8)]
    x_q = jnp.concatenate([q for q, _ in shards])
    x_scale = jnp.concatenate([s for _, s in shards])
    assert x_q.dtype == w_q.dtype and x_scale.shape == (64, 1)
    np.testing.assert_array_equal(
        np.asarray(_prequantized_matmul_bf16_jax(x_q, x_scale, w_q, w_s)),
        np.asarray(_quantized_matmul_jax(x, w_q, w_s)),
    )


@pytest.mark.cpu_test
def test_every_requantized_weight_dtype_has_an_activation_quant():
    assert set(fp8._MOE_REQUANT_WEIGHT_DTYPES.values()) <= set(
        linear_common._quantize_activation_ops
    )


def _layer(weight_scale: torch.Tensor, params_dtype=torch.bfloat16):
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros((_K, _N), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    layer.params_dtype = params_dtype
    return layer


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("scale_shape", "params_dtype", "keep_layout", "rows", "max_rows", "expected"),
    [
        ((_N,), torch.bfloat16, False, 16384, 0, True),
        ((_N,), torch.bfloat16, False, 64, 64, False),  # W8A16 bucket
        ((_N,), torch.bfloat16, False, 65, 64, True),
        ((1, 2, 1, _N), torch.bfloat16, False, 16384, 0, False),  # blockwise
        ((_N,), torch.float16, False, 16384, 0, False),
        ((_N,), torch.bfloat16, True, 16384, 0, False),  # raw-layout consumer
    ],
)
def test_supports_quantized_input(
    monkeypatch, scale_shape, params_dtype, keep_layout, rows, max_rows, expected
):
    monkeypatch.setenv("TPU_FP8_LINEAR_W8A16_MAX_ROWS", str(max_rows))
    method = VllmFp8LinearMethodTPU(Fp8Config(is_checkpoint_fp8_serialized=True))
    layer = _layer(torch.ones(scale_shape), params_dtype)
    if keep_layout:
        setattr(layer, KEEP_VLLM_LAYOUT_ATTR, True)
    assert method.supports_quantized_input(layer, rows) is expected


@pytest.mark.cpu_test
def test_quantized_input_round_trip_uses_layer_weights(monkeypatch):
    monkeypatch.setenv("TPU_FP8_DECODE_GEMV_MAX_ROWS", "16")
    calls = []

    def fake_quantize(x, w_q):
        calls.append(("quantize", w_q))
        return x.to(torch.float8_e4m3fn), torch.ones((x.shape[0], 1))

    def fake_matmul(x_q, x_scale, w_q, w_s, decode_gemv_max_rows=0):
        calls.append(("matmul", w_q, w_s, decode_gemv_max_rows))
        return torch.zeros((x_q.shape[0], w_q.shape[-1]), dtype=torch.bfloat16)

    monkeypatch.setattr(fp8, "quantize_activation", fake_quantize)
    monkeypatch.setattr(fp8, "prequantized_matmul", fake_matmul)
    method = VllmFp8LinearMethodTPU(Fp8Config(is_checkpoint_fp8_serialized=True))
    layer = _layer(torch.ones(_N))
    x_q, x_scale = method.quantize_input(layer, torch.ones((8, _K)))
    out = method.apply_quantized_input(layer, x_q, x_scale)
    assert out.shape == (8, _N)
    assert calls[0] == ("quantize", layer.weight)
    assert calls[1][1] is layer.weight and calls[1][2] is layer.weight_scale
    assert method.decode_gemv_max_rows == 16 and calls[1][3] == 16
