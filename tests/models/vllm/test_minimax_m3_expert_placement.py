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
"""MiniMax-M3's expert placement: weights and router rows move together."""

import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.models.minimax_m3 import expert_placement
from vllm_torchtpu.models.minimax_m3.expert_placement import ExpertPlacement

_SHIPPED = "minimax_m3.1_flash_agentx_ep4x2.json"
_EXPERT = "model.layers.{}.block_sparse_moe.experts.{}.w1.weight"
_GATE = "model.layers.{}.block_sparse_moe.gate.weight"


def test_identity_without_a_placement(monkeypatch):
    monkeypatch.setattr(envs, "TPU_MOE_EXPERT_PLACEMENT", "")
    placement = ExpertPlacement(128)
    assert not placement
    rows = torch.arange(128.0)
    assert placement.slot(_EXPERT.format(10, 37), 37) == 37
    assert placement.to_slots(_GATE.format(10), rows) is rows


@pytest.mark.parametrize("layer", [3, 31, 59])
def test_router_picks_the_slot_that_holds_the_expert(monkeypatch, layer):
    """A token routed to checkpoint expert e must be routed to the slot whose
    weights are expert e: gate row of slot s == checkpoint gate row of the
    expert loaded into s."""
    monkeypatch.setattr(envs, "TPU_MOE_EXPERT_PLACEMENT", _SHIPPED)
    expert_placement._load.cache_clear()
    placement = ExpertPlacement(128)
    assert placement
    gate = torch.randn(128, 16)
    slotted = placement.to_slots(_GATE.format(layer), gate)
    for expert in range(128):
        slot = placement.slot(_EXPERT.format(layer, expert), expert)
        assert torch.equal(slotted[slot], gate[expert])
    # The same hidden state scores slot s exactly as the expert placed there.
    x = torch.randn(4, 16)
    for s, e in enumerate(expert_placement._load(_SHIPPED, 128)[layer]):
        assert torch.equal(x @ slotted[s], x @ gate[e])


def test_checkpoint_and_model_names_find_the_same_layer(monkeypatch):
    """The EP weight filter sees checkpoint names, the model's load_weights the
    same names without the model prefix: both must place an expert alike."""
    monkeypatch.setattr(envs, "TPU_MOE_EXPERT_PLACEMENT", _SHIPPED)
    expert_placement._load.cache_clear()
    placement = ExpertPlacement(128)
    ckpt = "language_model.model.layers.10.block_sparse_moe.experts.37.w1.weight_packed"
    model = "layers.10.block_sparse_moe.experts.w13_weight"
    slot = expert_placement._load(_SHIPPED, 128)[10].index(37)
    assert placement.slot(ckpt, 37) == placement.slot(model, 37) == slot


def test_rejects_a_placement_that_is_not_a_permutation(tmp_path, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text('{"num_experts": 4, "layers": {"3": [0, 1, 1, 3]}}')
    monkeypatch.setattr(envs, "TPU_MOE_EXPERT_PLACEMENT", str(bad))
    with pytest.raises(ValueError, match="not a permutation"):
        ExpertPlacement(4)
