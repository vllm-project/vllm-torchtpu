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
from jax.experimental.pallas import tpu as pltpu
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import VllmMLAAttention
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8Config,
    VllmFp8LinearMethodTPU,
    _get_linear_quant_config,
)
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform


def _make_linear(n_in: int = 512, n_out: int = 64, prefix: str = "") -> LinearBase:
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


def test_linear_quant_config_fp4_validation_and_block_multiples(monkeypatch):
    assert "REQUANTIZE_FP4_PREFIXES" in TpuPlatform.additional_env_vars

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

    monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", "1024")
    _, _, explicit_block, _ = _get_linear_quant_config(None)
    assert explicit_block == 1024

    # Non-512-multiple block sizes must raise NotImplementedError rather than
    # silently mis-indexing gmm_v2's W4A8 LHS block loop.
    monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", "256")
    with pytest.raises(NotImplementedError, match="multiple of 512"):
        _get_linear_quant_config(None)

    # Invalid REQUANTIZE_WEIGHT_DTYPE typos must fail eagerly on onload configs too.
    monkeypatch.setenv("REQUANTIZE_WEIGHT_DTYPE", "e2m1")
    onload_cfg = VllmFp8Config(
        is_checkpoint_fp8_serialized=False, activation_scheme="dynamic"
    )
    with pytest.raises(ValueError, match="Unsupported REQUANTIZE_WEIGHT_DTYPE"):
        VllmFp8LinearMethodTPU(onload_cfg)


def test_fp8_serialized_linear_requants_to_kmajor_fp4_and_dispatches_apply(
    monkeypatch,
):
    monkeypatch.setenv("REQUANTIZE_WEIGHT_DTYPE", "fp4")
    monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "1")
    monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", "512")
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8._ensure_native_fp4_supported",
        lambda _dtype: None,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.quantize_native_fp4_kmajor",
        _fake_quantize_fp4_kmajor,
    )

    layer = _make_linear(n_in=1024, n_out=64)
    layer.weight = torch.nn.Parameter(
        torch.ones((64, 1024), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.tensor([0.5], dtype=torch.float32), requires_grad=False
    )

    cfg = VllmFp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="static")
    method = VllmFp8LinearMethodTPU(cfg, prefix="model.layers.0.self_attn.o_proj")
    method.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.float4_e2m1fn_x2
    assert tuple(layer.weight.shape) == (1024, 32)
    assert tuple(layer.weight_scale.shape) == (1, 2, 1, 64)
    assert getattr(layer, WEIGHT_FLIPPED_ATTR, False) is True
    assert layer.weight_block_size == 512

    x = torch.ones((4, 1024), dtype=torch.bfloat16)
    bias = torch.full((64,), 0.25, dtype=torch.bfloat16)
    with patch(
        "vllm_torchtpu.layers.adapter.quantization.fp8.quantized_matmul_fp4",
        return_value=torch.zeros((4, 64), dtype=torch.bfloat16),
    ) as mock_fp4_mm:
        out = method.apply(layer, x, bias=bias)
        mock_fp4_mm.assert_called_once()
    assert torch.allclose(out, torch.full((4, 64), 0.25, dtype=torch.bfloat16))


def test_mla_kv_b_proj_and_parallel_lm_head_stay_fp8(monkeypatch):
    """MLA kv_b_proj (KEEP_VLLM_LAYOUT_ATTR) and ParallelLMHead never turn into FP4."""
    monkeypatch.setenv("REQUANTIZE_WEIGHT_DTYPE", "fp4")
    monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "1")
    monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", "512")
    monkeypatch.delenv("REQUANTIZE_FP4_PREFIXES", raising=False)
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8._ensure_native_fp4_supported",
        lambda _dtype: None,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.quantize_native_fp4_kmajor",
        _fake_quantize_fp4_kmajor,
    )

    # 1. VllmMLAAttention marks its FP8 kv_b_proj KEEP_VLLM_LAYOUT_ATTR.
    kv_b = _make_linear(
        n_in=512, n_out=256, prefix="model.layers.0.self_attn.kv_b_proj"
    )

    def _stub_mla_init(self, *args, **kwargs):
        self.kv_cache_dtype = "auto"
        self.layer_name = "model.layers.0.self_attn.attn"

    with patch(
        "vllm.model_executor.layers.attention.mla_attention.MLAAttention.__init__",
        _stub_mla_init,
    ):
        VllmMLAAttention(
            num_heads=2,
            scale=1.0,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            q_lora_rank=None,
            kv_lora_rank=512,
            kv_b_proj=kv_b,
        )
    assert getattr(kv_b, KEEP_VLLM_LAYOUT_ATTR, False) is True
    kv_b.weight = torch.nn.Parameter(
        torch.ones((256, 512), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    kv_b.weight_scale = torch.nn.Parameter(
        torch.tensor([0.25], dtype=torch.float32), requires_grad=False
    )
    fp8_cfg = VllmFp8Config(
        is_checkpoint_fp8_serialized=True, activation_scheme="static"
    )
    VllmFp8LinearMethodTPU(fp8_cfg, prefix=kv_b.prefix).process_weights_after_loading(
        kv_b
    )
    assert kv_b.weight.dtype == torch.float8_e4m3fn
    assert tuple(kv_b.weight.shape) == (256, 512)
    assert not getattr(kv_b, WEIGHT_FLIPPED_ATTR, False)

    # 2. ParallelLMHead (VocabParallelEmbedding, not LinearBase) stays on W8A16 FP8.
    lm_head = ParallelLMHead.__new__(ParallelLMHead)
    torch.nn.Module.__init__(lm_head)
    lm_head.weight = torch.nn.Parameter(
        torch.randn((128, 512), dtype=torch.bfloat16), requires_grad=False
    )
    onload_cfg = VllmFp8Config(
        is_checkpoint_fp8_serialized=False, activation_scheme="dynamic"
    )
    VllmFp8LinearMethodTPU(onload_cfg, prefix="lm_head").process_weights_after_loading(
        lm_head
    )
    assert lm_head.weight.dtype == torch.float8_e4m3fn
    assert tuple(lm_head.weight.shape) == (512, 128)


@pytest.mark.parametrize("serialized", [True, False])
@pytest.mark.parametrize(
    ("n_in", "n_out"),
    [
        (640, 64),  # n_in not a multiple of the block
        (256, 64),  # n_in smaller than the block
        (1024, 63),  # odd n_out cannot pair into fp4x2
    ],
)
def test_fp4_ineligible_shape_falls_back_to_per_channel_fp8(
    monkeypatch, serialized: bool, n_in: int, n_out: int
):
    monkeypatch.setenv("REQUANTIZE_WEIGHT_DTYPE", "fp4")
    monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "1")
    monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", "512")
    monkeypatch.delenv("REQUANTIZE_FP4_PREFIXES", raising=False)
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8._ensure_native_fp4_supported",
        lambda _dtype: None,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.quantize_native_fp4_kmajor",
        _fake_quantize_fp4_kmajor,
    )

    layer = _make_linear(n_in=n_in, n_out=n_out, prefix="model.layers.0.mlp.down_proj")
    if serialized:
        layer.weight = torch.nn.Parameter(
            torch.ones((n_out, n_in), dtype=torch.float8_e4m3fn), requires_grad=False
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.tensor([0.5], dtype=torch.float32), requires_grad=False
        )
    else:
        layer.weight = torch.nn.Parameter(
            torch.randn((n_out, n_in), dtype=torch.bfloat16), requires_grad=False
        )
    cfg = VllmFp8Config(
        is_checkpoint_fp8_serialized=serialized,
        activation_scheme="static" if serialized else "dynamic",
    )
    method = VllmFp8LinearMethodTPU(cfg, prefix=layer.prefix)
    method.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.float8_e4m3fn
    assert tuple(layer.weight.shape) == (n_in, n_out)
    assert tuple(layer.weight_scale.shape) == (n_out,)

    x = torch.ones((4, n_in), dtype=torch.bfloat16)
    fp8_module = "vllm_torchtpu.layers.adapter.quantization.fp8"
    with (
        patch(
            f"{fp8_module}.quantized_matmul",
            return_value=torch.zeros((4, n_out), dtype=torch.bfloat16),
        ) as mock_fp8_mm,
        patch(f"{fp8_module}.quantized_matmul_fp4") as mock_fp4_mm,
    ):
        method.apply(layer, x)
    mock_fp8_mm.assert_called_once()
    mock_fp4_mm.assert_not_called()


def test_requantize_fp4_prefixes_blocks_empty_and_non_matching_prefixes(monkeypatch):
    monkeypatch.setenv("REQUANTIZE_WEIGHT_DTYPE", "fp4")
    monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "1")
    monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", "512")
    monkeypatch.setenv("REQUANTIZE_FP4_PREFIXES", "shared_expert.")
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8._ensure_native_fp4_supported",
        lambda _dtype: None,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.quantize_native_fp4_kmajor",
        _fake_quantize_fp4_kmajor,
    )

    fp8_cfg = VllmFp8Config(
        is_checkpoint_fp8_serialized=True, activation_scheme="static"
    )
    # 1. Empty prefix must NOT bypass a non-empty REQUANTIZE_FP4_PREFIXES allowlist.
    unnamed = _make_linear(n_in=512, n_out=128, prefix="")
    unnamed.weight = torch.nn.Parameter(
        torch.ones((128, 512), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    unnamed.weight_scale = torch.nn.Parameter(
        torch.tensor([0.25], dtype=torch.float32), requires_grad=False
    )
    VllmFp8LinearMethodTPU(fp8_cfg).process_weights_after_loading(unnamed)
    assert unnamed.weight.dtype == torch.float8_e4m3fn

    # 2. Non-matching prefix stays FP8.
    gate_ba = _make_linear(
        n_in=512, n_out=128, prefix="model.layers.0.linear_attn.in_proj_ba"
    )
    gate_ba.weight = torch.nn.Parameter(
        torch.ones((128, 512), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    gate_ba.weight_scale = torch.nn.Parameter(
        torch.tensor([0.25], dtype=torch.float32), requires_grad=False
    )
    VllmFp8LinearMethodTPU(fp8_cfg).process_weights_after_loading(gate_ba)
    assert gate_ba.weight.dtype == torch.float8_e4m3fn

    # 3. Matching prefix requantizes to FP4.
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
    VllmFp8LinearMethodTPU(onload_cfg).process_weights_after_loading(shared)
    assert shared.weight.dtype == torch.float4_e2m1fn_x2
    assert tuple(shared.weight.shape) == (512, 64)


def _require_tpu_v7x() -> None:
    if jax.default_backend() != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {jax.default_backend()!r}")
    if pltpu.get_tpu_info().generation < 7:
        pytest.skip("Native float4_e2m1fn W4A8 requires TPU v7x+")


@pytest.mark.tpu_test
@pytest.mark.parametrize("n_out", [192, 256])
def test_tpu_device_fp4_linear_w4a8_b512(monkeypatch, n_out: int):
    """Validates 128-aligned (256) and non-aligned (192) output widths + 3D x."""
    _require_tpu_v7x()
    monkeypatch.setenv("REQUANTIZE_WEIGHT_DTYPE", "fp4")
    monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "1")
    monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", "512")
    monkeypatch.delenv("REQUANTIZE_FP4_PREFIXES", raising=False)

    torch.manual_seed(0)
    w_ref = torch.randn((n_out, 1024), dtype=torch.bfloat16, device="tpu") * 0.05
    x_3d = torch.randn((2, 8, 1024), dtype=torch.bfloat16, device="tpu")
    y_ref = (x_3d.float() @ w_ref.float().T).to(torch.bfloat16)

    layer = _make_linear(
        n_in=1024,
        n_out=n_out,
        prefix="model.layers.0.mlp.shared_expert.down_proj",
    )
    layer.weight = torch.nn.Parameter(w_ref.clone(), requires_grad=False)
    cfg = VllmFp8Config(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")
    method = VllmFp8LinearMethodTPU(cfg, prefix=layer.prefix)
    method.process_weights_after_loading(layer)
    assert layer.weight.device.type == "tpu"
    assert layer.weight.dtype == torch.float4_e2m1fn_x2
    assert tuple(layer.weight.shape) == (1024, n_out // 2)
    assert tuple(layer.weight_scale.shape) == (1, 2, 1, n_out)

    y_fp4 = method.apply(layer, x_3d)
    assert tuple(y_fp4.shape) == (2, 8, n_out)
    cos = torch.nn.functional.cosine_similarity(
        y_ref.float().flatten(), y_fp4.float().flatten(), dim=0
    ).item()
    assert cos > 0.985


@pytest.mark.tpu_test
def test_tpu_device_fp4_linear_qkvz_large_shape_vmem(monkeypatch):
    """Regression test for m=2048, k=8192, n={8192, 18432} scoped-VMEM budget."""
    _require_tpu_v7x()
    monkeypatch.setenv("REQUANTIZE_WEIGHT_DTYPE", "fp4")
    monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "1")
    monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", "512")
    monkeypatch.delenv("REQUANTIZE_FP4_PREFIXES", raising=False)

    x = torch.randn((2048, 8192), dtype=torch.bfloat16, device="tpu") * 0.1
    # Every dequantized weight is 0.02, so each output column is 0.02 * sum(x).
    y_ref = x.float().sum(dim=-1, keepdim=True) * 0.02
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
        method = VllmFp8LinearMethodTPU(cfg, prefix=layer.prefix)
        method.process_weights_after_loading(layer)
        out = method.apply(layer, x)
        assert tuple(out.shape) == (2048, n_out)
        cos = torch.nn.functional.cosine_similarity(
            y_ref.expand(-1, n_out).flatten(), out.float().flatten(), dim=0
        ).item()
        assert cos > 0.99
