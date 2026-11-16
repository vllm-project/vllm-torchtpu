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
"""The model-owned FP8 MoE method, on CPU."""

from types import SimpleNamespace as NS

import pytest
import torch

from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU
from vllm_torchtpu.models.qwen38_flash.experimental import moe_method
from vllm_torchtpu.models.qwen38_flash.experimental.kernels.adaptive_fused_moe import (
    vllm_adapter as adaptive_moe,
)


def _shared_method():
    return VllmFp8MoEMethodTPU.__new__(VllmFp8MoEMethodTPU)


@pytest.mark.parametrize("sp", [False, True])
def test_own_moe_layer_swaps_only_the_shared_fp8_method(sp):
    layer = torch.nn.Module()
    layer.quant_method = method = _shared_method()
    assert moe_method.own_moe_layer(layer, sequence_parallel_variant=sp)
    assert layer.quant_method is method
    assert type(method) is moe_method.Qwen4ExpFp8MoEMethod
    assert getattr(layer, adaptive_moe.AG_LEAD_AXIS_ATTR) is True
    assert getattr(layer, adaptive_moe.SEQUENCE_PARALLEL_VARIANT_ATTR) is sp

    other = torch.nn.Module()
    other.quant_method = NS()
    assert not moe_method.own_moe_layer(other, sequence_parallel_variant=True)
    assert not hasattr(other, adaptive_moe.AG_LEAD_AXIS_ATTR)


def _owned(adaptive):
    method = _shared_method()
    method.__class__ = moe_method.Qwen4ExpFp8MoEMethod
    method._use_adaptive_fused_moe = adaptive
    return method


def test_apply_runs_the_fused_decode_first(monkeypatch):
    from vllm_torchtpu.models.qwen38_flash.experimental import moe_decode

    monkeypatch.setattr(
        moe_decode, "fused_moe_decode_forward", lambda *a: "decoded", raising=False
    )
    assert _owned(True).apply_monolithic(NS(), None, None) == "decoded"


def test_apply_runs_the_model_adaptive_op_with_its_scales(monkeypatch):
    from vllm_torchtpu.models.qwen38_flash.experimental import moe_decode

    monkeypatch.setattr(
        moe_decode, "fused_moe_decode_forward", lambda *a: None, raising=False
    )
    seen = []
    monkeypatch.setattr(
        adaptive_moe,
        "run_adaptive_fused_moe",
        lambda owner, x, w1, w2, s1, s2, logits: seen.append((s1, s2)) or "out",
    )
    layer = torch.nn.Module()
    layer.w13_weight = layer.w2_weight = None
    layer.w13_weight_scale_inv = torch.ones(4, 1, 1, 8)
    layer.w2_weight_scale_inv = torch.ones(4, 1, 1, 6)
    assert _owned(True).apply_monolithic(layer, "x", "logits") == "out"
    ((s1, s2),) = seen
    assert s1.shape == (4, 1, 1, 8)
    assert s2.shape == (4, 1, 1, 6)


def test_apply_without_adaptive_keeps_the_shared_path(monkeypatch):
    from vllm_torchtpu.models.qwen38_flash.experimental import moe_decode

    monkeypatch.setattr(
        moe_decode, "fused_moe_decode_forward", lambda *a: None, raising=False
    )
    monkeypatch.setattr(
        VllmFp8MoEMethodTPU, "apply_monolithic", lambda self, *a: "shared"
    )
    assert _owned(False).apply_monolithic(NS(), "x", "logits") == "shared"
