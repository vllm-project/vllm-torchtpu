# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""The shared latent MoE, in a configuration Kimi K3 does not use.

Kimi K3's tests (tests/models/vllm/test_kimi_k3.py) cover sigmoid routing
with a score-correction bias and a latent norm; these cover softmax routing
without either, as OLMo 3.5 routes.
"""

from __future__ import annotations

import zlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_torchtpu.layers.adapter import latent_moe
from vllm_torchtpu.layers.adapter.latent_moe import LatentMoE, RoutedOutputTransform

pytestmark = pytest.mark.usefixtures("cpu_model_tests")


def test_routed_output_transform_leaves_reduction_to_vllm_runner() -> None:
    class SquareNorm(nn.Module):
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states.square()

    class IdentityLinear(nn.Module):
        def forward(self, hidden_states: torch.Tensor):
            return hidden_states, None

    transform = RoutedOutputTransform(SquareNorm(), IdentityLinear())
    hidden_states = torch.tensor([[2.0, 3.0]])

    # vLLM v0.27 reduces the routed latent output before invoking this
    # nonlinear transform. Reducing again here underweights it at TP > 1.
    torch.testing.assert_close(transform(hidden_states), hidden_states.square())


class _Recorder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.args = args
        self.kwargs = kwargs


def test_softmax_latent_moe_without_bias_or_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a score-correction bias the router has none and the runner
    gets none; without a norm epsilon the latent output goes straight to the
    up projection. The experts run at the latent width, next to the caller's
    shared experts."""
    fused_moe_args = {}

    def fake_fused_moe(**kwargs):
        fused_moe_args.update(kwargs)
        return SimpleNamespace(routed_experts=None)

    monkeypatch.setattr(latent_moe, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(latent_moe, "GateLinear", _Recorder)
    monkeypatch.setattr(latent_moe, "make_latent_projection", _Recorder)
    monkeypatch.setattr(latent_moe, "FusedMoEFactory", fake_fused_moe)
    shared_experts = nn.Identity()

    layer = LatentMoE(
        hidden_size=16,
        num_experts=8,
        top_k=2,
        intermediate_size=128,
        hidden_act="silu",
        quant_config=None,
        prefix="model.layers.1.moe",
        shared_experts=shared_experts,
        scoring_func="softmax",
        renormalize=True,
        routed_scaling_factor=1.0,
        use_score_correction_bias=False,
        latent_size=4,
    )

    assert not hasattr(layer.gate, "e_score_correction_bias")
    assert layer.routed_expert_norm is None
    assert layer.routed_expert_down_proj.args[:2] == (16, 4)
    assert layer.routed_expert_up_proj.args[:2] == (4, 16)
    assert fused_moe_args["hidden_size"] == 4
    assert fused_moe_args["e_score_correction_bias"] is None
    assert fused_moe_args["scoring_func"] == "softmax"
    assert fused_moe_args["shared_experts"] is shared_experts
    assert fused_moe_args["routed_input_transform"] is layer.routed_expert_down_proj
    output_transform = fused_moe_args["routed_output_transform"]
    assert output_transform.norm is None
    assert output_transform.up_proj is layer.routed_expert_up_proj


def test_dummy_router_without_bias_draws_only_fan_in_weights(monkeypatch) -> None:
    """With no correction bias, the dummy router's weight is the first draw of
    the layer's generator, scaled by the fan-in."""
    config = SimpleNamespace(load_config=SimpleNamespace(load_format="dummy"))
    monkeypatch.setattr(latent_moe, "get_current_vllm_config_or_none", lambda: config)
    module = LatentMoE.__new__(LatentMoE)
    nn.Module.__init__(module)
    module.prefix = "model.layers.1.moe"
    module.use_score_correction_bias = False
    module.gate = nn.Module()
    module.gate.input_size = 128
    module.gate.weight = nn.Parameter(torch.zeros(64, 128))

    module.initialize_dummy_router()

    generator = torch.Generator(device="cpu").manual_seed(
        1234 + zlib.crc32(module.prefix.encode())
    )
    expected = torch.randn((64, 128), generator=generator) * 128**-0.5
    torch.testing.assert_close(module.gate.weight.detach(), expected, rtol=0, atol=0)
