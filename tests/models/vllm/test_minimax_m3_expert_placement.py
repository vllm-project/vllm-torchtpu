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
_REPLICAS = "minimax_m3.1_flash_agentx_ep4x2_r4.json"
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
    for s, e in enumerate(expert_placement._load(_SHIPPED, 128).slots[layer]):
        assert torch.equal(x @ slotted[s], x @ gate[e])


def test_checkpoint_and_model_names_find_the_same_layer(monkeypatch):
    """The EP weight filter sees checkpoint names, the model's load_weights the
    same names without the model prefix: both must place an expert alike."""
    monkeypatch.setattr(envs, "TPU_MOE_EXPERT_PLACEMENT", _SHIPPED)
    expert_placement._load.cache_clear()
    placement = ExpertPlacement(128)
    ckpt = "language_model.model.layers.10.block_sparse_moe.experts.37.w1.weight_packed"
    model = "layers.10.block_sparse_moe.experts.w13_weight"
    slot = expert_placement._load(_SHIPPED, 128).slots[10].index(37)
    assert placement.slot(ckpt, 37) == placement.slot(model, 37) == slot


def test_rejects_a_placement_that_is_not_a_permutation(tmp_path, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text('{"num_experts": 4, "layers": {"3": [0, 1, 1, 3]}}')
    monkeypatch.setattr(envs, "TPU_MOE_EXPERT_PLACEMENT", str(bad))
    with pytest.raises(ValueError, match="not a permutation"):
        ExpertPlacement(4)


@pytest.mark.parametrize("layer", [3, 31, 59])
def test_replica_numbering_agrees_between_loader_and_kernel(monkeypatch, layer):
    """Block q owns physical experts [q (E/G + R), (q + 1) (E/G + R)): a slot's
    home copy is where fused_moe_func's home mapping sends it, each replica
    is where the kernel's table says, the loader writes the checkpoint expert
    to all of them, and the EP weight filter keeps it on every block that
    holds one."""
    monkeypatch.setattr(envs, "TPU_MOE_EXPERT_PLACEMENT", _REPLICAS)
    expert_placement._load.cache_clear()
    placement = ExpertPlacement(128)
    groups, per_group = placement.groups, placement.replicas_per_group
    assert (groups, per_group) == (4, 4)
    assert placement.num_physical_experts == 128 + groups * per_group
    block, local = 128 // groups, 128 // groups + per_group
    table = placement.replica_table(f"model.layers.{layer}.block_sparse_moe")
    rows = [table[i : i + 4] for i in range(0, len(table), 4)]
    assert len(rows) == groups * per_group
    copies = {}
    for slot, home, copy, copy_local in rows:
        assert home == slot // block != copy
        assert block <= copy_local < local
        copies.setdefault(slot, []).append(copy * local + copy_local)
    # Each block's replica slots are used once.
    assert len({c for cs in copies.values() for c in cs}) == len(rows)
    name = _EXPERT.format(layer, 0)
    for expert in range(128):
        slot = placement.slot(name, expert)
        home_physical = slot + slot // block * per_group  # fused_moe_func
        physical = placement.physical_experts(name, expert)
        assert physical == [home_physical] + copies.get(slot, [])
        owners = placement.owner_slots(name, expert)
        assert sorted(o // block for o in owners) == sorted(
            p // local for p in physical
        )


def test_replicas_need_a_block_other_than_home(tmp_path, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text(
        '{"num_experts": 4, "layers": {}, "groups": 2, "replicas_per_group": 1,'
        ' "replicas": {"3": [[0, 0]]}}'
    )
    monkeypatch.setattr(envs, "TPU_MOE_EXPERT_PLACEMENT", str(bad))
    with pytest.raises(ValueError, match="home"):
        ExpertPlacement(4)
