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
"""Tests for load-time FP4 W4A8 linear requantization, ModelOpt mixed online
FP8, and NVFP4 MoE W4A8 / FP8-all-gather gates."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.model_executor.layers.linear import LinearBase

from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8Config,
    VllmFp8LinearMethodTPU,
    _get_linear_quant_config,
)
from vllm_torchtpu.layers.adapter.quantization.modelopt_mixed import (
    VllmModelOptMixedConfig,
)
from vllm_torchtpu.layers.adapter.quantization.nvfp4 import VllmNvfp4MoEMethod
from vllm_torchtpu.layers.adapter.quantization.online_fp8 import attach_online_fp8


def _make_linear(n_in: int = 256, n_out: int = 64) -> LinearBase:
    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    layer.input_size = n_in
    layer.output_size = n_out
    layer.logical_widths = [n_out]
    return layer


def _fake_quantize_fp4_kmajor(
    w_kmajor_3d: torch.Tensor, block: int, pack: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    assert not pack
    groups, k_dim, n_out = w_kmajor_3d.shape
    assert k_dim % block == 0 and n_out % 2 == 0
    w_q = torch.zeros(
        (groups, k_dim, n_out // 2),
        dtype=torch.uint8,
        device=w_kmajor_3d.device,
    ).view(torch.float4_e2m1fn_x2)
    w_s = torch.ones(
        (groups, k_dim // block, 1, n_out),
        dtype=torch.float32,
        device=w_kmajor_3d.device,
    )
    return w_q, w_s


def test_linear_quant_config_fp4_requires_block_kernel(monkeypatch):
    monkeypatch.setenv("REQUANTIZE_WEIGHT_DTYPE", "fp4")
    monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "0")
    monkeypatch.delenv("REQUANTIZE_BLOCK_SIZE", raising=False)
    with pytest.raises(ValueError, match="ENABLE_QUANTIZED_MATMUL_KERNEL=1"):
        _get_linear_quant_config(None)

    monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "1")
    monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", "128")
    name, dtype, block, use_kernel = _get_linear_quant_config(None)
    assert name == "fp4"
    assert dtype == torch.float4_e2m1fn_x2
    assert block == 128
    assert use_kernel is True


def test_fp8_serialized_linear_requants_to_kmajor_fp4(monkeypatch):
    monkeypatch.setenv("REQUANTIZE_WEIGHT_DTYPE", "fp4")
    monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "1")
    monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", "128")
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.quantize_native_fp4_kmajor",
        _fake_quantize_fp4_kmajor,
    )

    layer = _make_linear(n_in=256, n_out=64)
    layer.weight = torch.nn.Parameter(
        torch.ones((64, 256), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.tensor([0.5], dtype=torch.float32), requires_grad=False
    )

    cfg = VllmFp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="static")
    method = VllmFp8LinearMethodTPU(cfg)
    method.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.float4_e2m1fn_x2
    assert tuple(layer.weight.shape) == (256, 32)
    assert tuple(layer.weight_scale.shape) == (1, 2, 1, 64)
    assert getattr(layer, WEIGHT_FLIPPED_ATTR, False) is True
    assert layer.weight_block_size == 128

    called = {}

    def _fake_matmul_fp4(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor):
        called["shapes"] = (tuple(x.shape), tuple(w.shape), tuple(s.shape))
        return torch.zeros((x.shape[0], s.shape[-1]), dtype=x.dtype)

    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.quantized_matmul_fp4",
        _fake_matmul_fp4,
    )
    x = torch.randn((5, 256), dtype=torch.bfloat16)
    bias = torch.ones((64,), dtype=torch.bfloat16)
    out = method.apply(layer, x, bias=bias)
    assert called["shapes"] == ((5, 256), (256, 32), (1, 2, 1, 64))
    assert tuple(out.shape) == (5, 64)
    assert torch.all(out == 1)


def test_onload_bf16_linear_requants_to_kmajor_fp4_and_fallbacks(monkeypatch):
    monkeypatch.setenv("REQUANTIZE_WEIGHT_DTYPE", "fp4")
    monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "1")
    monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", "128")
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.quantize_native_fp4_kmajor",
        _fake_quantize_fp4_kmajor,
    )

    shared = _make_linear(n_in=256, n_out=64)
    shared.weight = torch.nn.Parameter(
        torch.randn((64, 256), dtype=torch.bfloat16), requires_grad=False
    )
    cfg = VllmFp8Config(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")
    method = VllmFp8LinearMethodTPU(cfg)
    method.process_weights_after_loading(shared)
    assert shared.weight.dtype == torch.float4_e2m1fn_x2
    assert tuple(shared.weight.shape) == (256, 32)
    assert tuple(shared.weight_scale.shape) == (1, 2, 1, 64)

    kept = _make_linear(n_in=256, n_out=64)
    kept.weight = torch.nn.Parameter(
        torch.randn((64, 256), dtype=torch.bfloat16), requires_grad=False
    )
    setattr(kept, KEEP_VLLM_LAYOUT_ATTR, True)
    method_kept = VllmFp8LinearMethodTPU(cfg)
    method_kept.process_weights_after_loading(kept)
    assert kept.weight.dtype == torch.float8_e4m3fn
    assert tuple(kept.weight.shape) == (64, 256)
    assert not getattr(kept, WEIGHT_FLIPPED_ATTR, False)


def test_modelopt_mixed_supports_online_fp8():
    raw_cfg = {
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
            "model.layers.0.mlp.experts": {
                "quant_algo": "NVFP4",
                "group_size": 16,
            }
        },
        "ignore": ["mtp*"],
    }
    cfg = VllmModelOptMixedConfig.from_config(raw_cfg)
    vllm_cfg = MagicMock()
    vllm_cfg.additional_config = {
        "tpu_online_fp8": {
            "enabled": True,
            "weight_scheme": "per_channel",
            "targets": ["model.layers.0.mlp.shared_expert.*"],
        }
    }
    attach_online_fp8(cfg, vllm_cfg)
    layer = _make_linear(n_in=256, n_out=64)
    method = cfg.get_quant_method(
        layer, "model.layers.0.mlp.shared_expert.gate_up_proj"
    )
    assert isinstance(method, VllmFp8LinearMethodTPU)
    assert method.online_fp8 is True


def test_nvfp4_moe_fp8_all_gather_gate(monkeypatch):
    monkeypatch.setenv("TPU_MOE_FP8_ALL_GATHER", "1")
    monkeypatch.setenv("TPU_MOE_COLLECTION_CHUNK_SIZE", "0")
    cfg = MagicMock(group_size=16)
    moe_cfg = MagicMock()
    moe_cfg.experts_per_token = 2
    method = VllmNvfp4MoEMethod(cfg, moe_cfg)
    assert method.supports_internal_mk is True

    with (
        patch(
            "vllm_torchtpu.layers.adapter.quantization.nvfp4.moe_routing.route",
            return_value=(torch.ones((2, 2)), torch.zeros((2, 2), dtype=torch.int32)),
        ),
        patch(
            "vllm_torchtpu.layers.adapter.quantization.nvfp4.pipelined_fused_moe_gmm",
            return_value="pipelined_fp8_ag",
        ) as mock_pipe,
    ):
        layer = MagicMock()
        layer.activation = MagicMock(value="silu")
        layer.moe_config = moe_cfg
        layer._experts_start = 0
        out = method.apply_monolithic(layer, torch.zeros((2, 16)), torch.zeros((2, 4)))
        assert out == "pipelined_fp8_ag"
        mock_pipe.assert_called_once()
