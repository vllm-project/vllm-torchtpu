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
"""Tests for load-time FP4 (W4A8) LinearBase weight requantization."""

from __future__ import annotations

from unittest.mock import patch

import jax
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


def _make_linear(n_in: int = 256, n_out: int = 64, prefix: str = "") -> LinearBase:
    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    layer.input_size = n_in
    layer.output_size = n_out
    layer.logical_widths = [n_out]
    layer.prefix = prefix
    return layer


def _fake_quantize_fp4_kmajor(
    w_kmajor_2d: torch.Tensor, block: int, pack: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    assert not pack
    k_dim, n_out = w_kmajor_2d.shape
    assert k_dim % block == 0 and n_out % 2 == 0
    w_q = torch.zeros(
        (k_dim, n_out // 2),
        dtype=torch.uint8,
        device=w_kmajor_2d.device,
    ).view(torch.float4_e2m1fn_x2)
    w_s = torch.ones(
        (k_dim // block, 1, n_out),
        dtype=torch.float32,
        device=w_kmajor_2d.device,
    )
    return w_q, w_s


def test_linear_quant_config_fp4_requires_kernel_and_defaults_block_size(monkeypatch):
    monkeypatch.setenv("REQUANTIZE_WEIGHT_DTYPE", "fp4")
    monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "0")
    monkeypatch.delenv("REQUANTIZE_BLOCK_SIZE", raising=False)
    with pytest.raises(ValueError, match="ENABLE_QUANTIZED_MATMUL_KERNEL=1"):
        _get_linear_quant_config(None)

    monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "1")
    name, dtype, block, use_kernel = _get_linear_quant_config(None)
    assert name == "fp4"
    assert dtype == torch.float4_e2m1fn_x2
    assert block == 512
    assert use_kernel is True

    monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", "256")
    _, _, explicit_block, _ = _get_linear_quant_config(None)
    assert explicit_block == 256


def test_fp8_serialized_linear_requants_to_kmajor_fp4_and_dispatches_apply(
    monkeypatch,
):
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

    x = torch.ones((4, 256), dtype=torch.bfloat16)
    bias = torch.full((64,), 0.25, dtype=torch.bfloat16)
    with patch(
        "vllm_torchtpu.layers.adapter.quantization.fp8.quantized_matmul_fp4",
        return_value=torch.zeros((4, 64), dtype=torch.bfloat16),
    ) as mock_fp4_mm:
        out = method.apply(layer, x, bias=bias)
        mock_fp4_mm.assert_called_once()
    assert torch.allclose(out, torch.full((4, 64), 0.25, dtype=torch.bfloat16))


def test_bf16_onload_linear_requants_to_fp4_and_respects_keep_vllm_layout(monkeypatch):
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


def test_requantize_fp4_prefixes_honors_method_or_layer_prefix(monkeypatch):
    monkeypatch.setenv("REQUANTIZE_WEIGHT_DTYPE", "fp4")
    monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "1")
    monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", "256")
    monkeypatch.setenv("REQUANTIZE_FP4_PREFIXES", "shared_expert.,lm_head")
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.quantize_native_fp4_kmajor",
        _fake_quantize_fp4_kmajor,
    )

    fp8_cfg = VllmFp8Config(
        is_checkpoint_fp8_serialized=True, activation_scheme="static"
    )
    # Excluded via layer.prefix even when VllmFp8LinearMethodTPU(fp8_cfg) omits prefix.
    gate_ba = _make_linear(
        n_in=512, n_out=128, prefix="model.layers.0.linear_attn.in_proj_ba"
    )
    gate_ba.weight = torch.nn.Parameter(
        torch.ones((128, 512), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    gate_ba.weight_scale = torch.nn.Parameter(
        torch.tensor([0.25], dtype=torch.float32), requires_grad=False
    )
    method_ba = VllmFp8LinearMethodTPU(fp8_cfg)
    method_ba.process_weights_after_loading(gate_ba)
    assert gate_ba.weight.dtype == torch.float8_e4m3fn
    assert tuple(gate_ba.weight.shape) == (512, 128)

    onload_cfg = VllmFp8Config(
        is_checkpoint_fp8_serialized=False, activation_scheme="dynamic"
    )
    shared = _make_linear(
        n_in=512,
        n_out=128,
        prefix="model.layers.0.mlp.shared_expert.gate_up_proj",
    )
    shared.weight = torch.nn.Parameter(
        torch.randn((128, 512), dtype=torch.bfloat16), requires_grad=False
    )
    method_shared = VllmFp8LinearMethodTPU(onload_cfg)
    method_shared.process_weights_after_loading(shared)
    assert shared.weight.dtype == torch.float4_e2m1fn_x2
    assert tuple(shared.weight.shape) == (512, 64)


@pytest.mark.tpu_test
def test_tpu_device_fp4_linear_w4a8_b512(monkeypatch):
    if jax.default_backend() != "tpu":
        pytest.skip("Requires TPU backend")
    monkeypatch.setenv("REQUANTIZE_WEIGHT_DTYPE", "fp4")
    monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "1")
    monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", "512")
    monkeypatch.delenv("REQUANTIZE_FP4_PREFIXES", raising=False)

    torch.manual_seed(0)
    w_ref = torch.randn((256, 1024), dtype=torch.bfloat16, device="tpu") * 0.05
    x = torch.randn((16, 1024), dtype=torch.bfloat16, device="tpu")
    y_ref = (x.float() @ w_ref.float().T).to(torch.bfloat16)

    layer = _make_linear(
        n_in=1024,
        n_out=256,
        prefix="model.layers.0.mlp.shared_expert.down_proj",
    )
    layer.weight = torch.nn.Parameter(w_ref.clone(), requires_grad=False)
    cfg = VllmFp8Config(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")
    method = VllmFp8LinearMethodTPU(cfg)
    method.process_weights_after_loading(layer)
    assert layer.weight.device.type == "tpu"
    assert layer.weight.dtype == torch.float4_e2m1fn_x2
    assert tuple(layer.weight.shape) == (1024, 128)
    assert tuple(layer.weight_scale.shape) == (1, 2, 1, 256)

    y_fp4 = method.apply(layer, x)
    cos = torch.nn.functional.cosine_similarity(
        y_ref.float().flatten(), y_fp4.float().flatten(), dim=0
    ).item()
    assert cos > 0.985


@pytest.mark.tpu_test
def test_tpu_device_fp4_linear_qkvz_large_shape_vmem(monkeypatch):
    """Regression test for m=2048, k=8192, n={8192, 18432} scoped-VMEM budget."""
    if jax.default_backend() != "tpu":
        pytest.skip("Requires TPU backend")
    monkeypatch.setenv("REQUANTIZE_WEIGHT_DTYPE", "fp4")
    monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "1")
    monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", "512")
    monkeypatch.delenv("REQUANTIZE_FP4_PREFIXES", raising=False)

    x = torch.randn((2048, 8192), dtype=torch.bfloat16, device="tpu") * 0.1
    cfg = VllmFp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="static")
    for n_out in (8192, 18432):
        layer = _make_linear(
            n_in=8192,
            n_out=n_out,
            prefix="model.layers.0.linear_attn.in_proj_qkvz",
        )
        layer.weight = torch.nn.Parameter(
            torch.ones((n_out, 8192), dtype=torch.float8_e4m3fn, device="tpu"),
            requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.tensor([0.02], dtype=torch.float32, device="tpu"),
            requires_grad=False,
        )
        method = VllmFp8LinearMethodTPU(cfg)
        method.process_weights_after_loading(layer)
        out = method.apply(layer, x)
        assert tuple(out.shape) == (2048, n_out)
