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
"""MiniMax M3 MXFP8 linears kept e4m3 (additional_config minimax_fp8_linears /
minimax_fp8_draft_linears), on CPU."""

from types import SimpleNamespace

import pytest
import torch
import vllm.config
from vllm.model_executor.layers.linear import LinearBase

from vllm_torchtpu.models.minimax_m3 import quant
from vllm_torchtpu.models.minimax_m3.quant import (
    MiniMaxM3Mxfp8LinearMethod,
    dequantize_mxfp8,
    requantize_mxfp8_per_channel,
)


def _mxfp8(rows, cols, seed, spread=2):
    g = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, cols, generator=g).to(torch.float8_e4m3fn)
    scale = torch.randint(127 - spread, 127 + spread, (rows, cols // 32), generator=g)
    return weight, scale.to(torch.uint8)


def test_requantized_weight_dequantizes_to_the_checkpoint():
    """Within e4m3's normal range the block shift is exact: e4m3 x channel
    scale equals the bf16 dequantized weight bit for bit, except for the
    `changed` elements."""
    weight, scale = _mxfp8(64, 256, 0)
    w8, channel_scale, changed = requantize_mxfp8_per_channel(weight, scale)
    assert w8.dtype == torch.float8_e4m3fn and w8.shape == (256, 64)
    got = (w8.float().T * channel_scale[:, None]).to(torch.bfloat16)
    want = dequantize_mxfp8(weight, scale)
    differ = int((got != want).sum())
    assert differ <= changed < weight.numel() // 20


def test_requantize_is_exact_with_equal_block_scales():
    weight, _ = _mxfp8(16, 128, 1)
    scale = torch.full((16, 4), 125, dtype=torch.uint8)
    w8, channel_scale, changed = requantize_mxfp8_per_channel(weight, scale)
    assert changed == 0
    assert torch.equal(
        (w8.float().T * channel_scale[:, None]).to(torch.bfloat16),
        dequantize_mxfp8(weight, scale),
    )


class _Linear(torch.nn.Module):
    def __init__(self, weight, scale):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
        self.bias = None


@pytest.mark.parametrize("equal_scales", [True, False])
@pytest.mark.parametrize("rows", [3, 64])
def test_e4m3_linear_matches_the_bf16_linear(rows, equal_scales):
    """With no element moved below e4m3's normal range (equal block scales)
    the matmul is the bf16 one bit for bit (a power-of-two scale after one
    rounding); otherwise those elements lose bits: within 2% of the output's
    scale."""
    weight, scale = _mxfp8(64, 256, 2)
    if equal_scales:
        scale = torch.full_like(scale, 126)
    bf16 = dequantize_mxfp8(weight, scale)
    layer = _Linear(weight.clone(), scale.clone())
    method = MiniMaxM3Mxfp8LinearMethod(keep_fp8=True)
    method.process_weights_after_loading(layer)
    assert layer.weight.dtype == torch.float8_e4m3fn
    x = torch.randn(rows, 256, dtype=torch.bfloat16)
    got = method.apply(layer, x)
    want = method_bf16_apply(bf16, x)
    if equal_scales:
        assert torch.equal(got, want)
    else:
        err = (got.float() - want.float()).abs().max()
        assert err <= 0.02 * want.float().abs().max()


def method_bf16_apply(bf16, x):
    """The key-off linear: x @ W with the bf16 dequantized weight."""
    return torch.matmul(x, bf16.T.contiguous())


def test_key_off_keeps_the_bf16_linear():
    weight, scale = _mxfp8(64, 256, 3)
    layer = _Linear(weight.clone(), scale.clone())
    method = MiniMaxM3Mxfp8LinearMethod()
    assert method.keep_fp8 is False
    method.process_weights_after_loading(layer)
    assert layer.weight.dtype == torch.bfloat16
    assert not hasattr(layer, "weight_scale")


def _config(is_draft):
    cfg = object.__new__(quant.MiniMaxM3QuantConfig)
    cfg.is_draft = is_draft
    cfg.is_layer_excluded = lambda prefix: False
    return cfg


@pytest.mark.parametrize(
    "additional,draft,prefix,want",
    [
        ({}, False, "model.layers.0.self_attn.o_proj", False),
        ({"minimax_fp8_linears": True}, False, "model.layers.0.self_attn.o_proj", True),
        ({"minimax_fp8_linears": True}, True, "dspark.layers.0.mlp.down_proj", False),
        ({"minimax_fp8_draft_linears": True}, True, "dspark.layers.0.mlp.down_proj", True),
        ({"minimax_fp8_draft_linears": True}, True, "dspark.layers.0.self_attn.o_proj", False),
        ({"minimax_fp8_draft_linears": True}, False, "model.layers.0.mlp.down_proj", False),
    ],
)
def test_keys_pick_the_linears(monkeypatch, additional, draft, prefix, want):
    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config",
        lambda: SimpleNamespace(additional_config=additional),
    )
    layer = object.__new__(LinearBase)
    method = _config(draft)._get_checkpoint_quant_method(layer, prefix)
    assert isinstance(method, MiniMaxM3Mxfp8LinearMethod)
    assert method.keep_fp8 is want
