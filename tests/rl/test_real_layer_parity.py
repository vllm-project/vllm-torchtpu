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
"""CPU parity tests against real TorchTPU PWAL implementations."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_torchtpu.rl.manifest_generator import (
    DP_VOCAB_ATTR,
    WEIGHT_FLIPPED_ATTR,
    generate_manifest,
)
from vllm_torchtpu.rl.weight_conversion import from_hf


def test_mirrored_attr_constants_match_layer_adapters() -> None:
    pytest.importorskip("torch_tpu")
    from vllm_torchtpu.layers.adapter.dp_vocab_parallel import (
        DP_VOCAB_ATTR as adapter_dp_vocab,
    )
    from vllm_torchtpu.layers.adapter.linear_common import (
        WEIGHT_FLIPPED_ATTR as adapter_flipped,
    )

    assert adapter_flipped == WEIGHT_FLIPPED_ATTR
    assert adapter_dp_vocab == DP_VOCAB_ATTR


def test_real_unquantized_linear_pwal_matches_manifest() -> None:
    pytest.importorskip("torch_tpu")
    from vllm_torchtpu.layers.adapter.quantization.unquantized import (
        VllmUnquantizedLinearMethod,
    )

    class ReplicatedLinear(nn.Module):
        def __init__(self, weight: torch.Tensor) -> None:
            super().__init__()
            self.weight = nn.Parameter(weight.clone(), requires_grad=False)
            self.tp_size = 1

    hf_w = torch.randn(16, 32, dtype=torch.bfloat16)
    model = nn.Module()
    model.proj = ReplicatedLinear(hf_w)
    VllmUnquantizedLinearMethod().process_weights_after_loading(model.proj)
    assert getattr(model.proj, WEIGHT_FLIPPED_ATTR, False)

    converted = from_hf(generate_manifest(model, tp_size=1), {"proj.weight": hf_w})
    assert torch.equal(converted["proj.weight"], model.proj.weight.data)


def test_real_moe_pwal_parity_and_reentry_idempotency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch_tpu")
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.fused_moe.config import MoEActivation

    from vllm_torchtpu.layers.adapter.quantization import unquantized as unq_mod

    monkeypatch.setattr(unq_mod, "prebuild_fused_moe_kernel", lambda **kw: None)
    num_e, hidden, inter = 2, 64, 96  # inter=96 triggers 128-tile right padding
    gate = torch.randn(num_e, inter, hidden, dtype=torch.bfloat16)
    up = torch.randn(num_e, inter, hidden, dtype=torch.bfloat16)
    down = torch.randn(num_e, hidden, inter, dtype=torch.bfloat16)

    layer = RoutedExperts.__new__(RoutedExperts)
    nn.Module.__init__(layer)
    layer.activation = MoEActivation.SILU
    layer.w13_weight = nn.Parameter(torch.cat([gate, up], dim=1), requires_grad=False)
    layer.w2_weight = nn.Parameter(down.clone(), requires_grad=False)
    layer.moe_config = SimpleNamespace(
        num_experts=num_e,
        num_logical_experts=num_e,
        hidden_dim=hidden,
        intermediate_size_per_partition=inter,
        ep_size=1,
        tp_size=1,
        has_bias=False,
        moe_parallel_config=SimpleNamespace(use_ep=False),
    )

    method = unq_mod.VllmUnquantizedFusedMoEMethod(layer.moe_config)
    layer.quant_method = method
    method.process_weights_after_loading(layer)
    assert getattr(layer, WEIGHT_FLIPPED_ATTR, False)
    w13_first, w2_first = layer.w13_weight, layer.w2_weight

    # Second PWAL call must be a no-op (same Parameter objects, no double-transpose).
    method.process_weights_after_loading(layer)
    assert layer.w13_weight is w13_first
    assert layer.w2_weight is w2_first

    model = nn.Module()
    model.experts = layer
    converted = from_hf(
        generate_manifest(model, tp_size=1, ep_size=1),
        {
            "experts.gate_proj.weight": gate,
            "experts.up_proj.weight": up,
            "experts.down_proj.weight": down,
        },
    )
    assert torch.equal(converted["experts.w13_weight"], layer.w13_weight.data)
    assert torch.equal(converted["experts.w2_weight"], layer.w2_weight.data)
