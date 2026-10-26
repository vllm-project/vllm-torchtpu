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
"""Host-side checks for the static expert layout (vllm_torchtpu/expert_placement.py)."""

import json
import os
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import expert_placement

pytestmark = pytest.mark.cpu_test


def _table(experts=8, shards=2, layers=None, draft=None):
    layers = layers or {"3": [[7, 0, 5, 2], [1, 6, 3, 4]]}
    table = {"experts": experts, "shards": shards, "layers": layers}
    if draft is not None:
        table["draft"] = draft
    return table


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(expert_placement, "_APPLIED_LAYERS", set())
    expert_placement._read_once.cache_clear()
    yield
    expert_placement._read_once.cache_clear()


@pytest.fixture
def placement_file(tmp_path, monkeypatch):
    path = tmp_path / "placement.json"
    path.write_text(json.dumps(_table()))
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", str(path))
    yield path


def test_slot_table_inverts_the_rank_lists():
    p = expert_placement.parse(_table())
    assert p.slots(3) == 4 and p.target.physical == 8
    assert p.slot_table(3) == [[1], [4], [3], [6], [7], [2], [5], [0]]
    assert p.target.expert_map(3, 1) == [-1, 0, -1, 2, 3, -1, 1, -1]
    assert p.ranks(4) is None


def test_spare_slots_hold_replicas():
    table = _table(layers={"3": [[7, 0, 5, 2, 3], [1, 6, 3, 4, 7]]})
    p = expert_placement.parse(table)
    assert p.slots(3) == 5 and p.target.physical == 10
    rows = p.slot_table(3)
    assert rows[3] == [4, 7] and rows[7] == [0, 9]
    assert rows[0] == [1, -1]


def test_explicit_slots_field_must_match_the_lists():
    table = _table(layers={"3": [[7, 0, 5, 2, 3], [1, 6, 3, 4, 7]]})
    table["slots"] = 5
    assert expert_placement.parse(table).target.slots == 5
    table["slots"] = 4
    with pytest.raises(ValueError, match=r"4 slots each.*widths \[5\]"):
        expert_placement.parse(table)


@pytest.mark.parametrize(
    "layers",
    [
        {"3": [[7, 0, 5, 2], [1, 6, 3, 3]]},  # 4 missing, 3 twice
        {"3": [[7, 0, 5], [1, 6, 3, 4, 2]]},  # uneven ranks
        {"3": [[7, 0, 5, 2]]},  # one rank short
        {"3": [[7, 7, 5, 2], [1, 6, 3, 4]]},  # 0 missing, 7 twice on rank 0
        {"3": [[7, 0, 5, 2], [1, 6, 3, 4]], "4": [[0, 1, 2], [3, 4, 5]]},  # too few
    ],
)
def test_a_table_that_is_not_a_layout_is_refused(layers):
    with pytest.raises(ValueError):
        expert_placement.parse(_table(layers=layers))


def test_draft_group_has_its_own_experts_and_slots():
    draft = {"experts": 4, "layers": {"0": [[0, 1, 2], [2, 3, 1]]}}
    p = expert_placement.parse(_table(draft=draft))
    assert p.draft is not None and p.draft.experts == 4
    assert p.draft.shards == 2 and p.draft.slots == 3
    assert p.draft.slot_table(0) == [[0, -1], [1, 5], [2, 3], [4, -1]]
    assert p.group("draft") is p.draft and p.group("target") is p.target


def test_layer_index_reads_module_prefixes_and_checkpoint_names():
    assert expert_placement.layer_index("model.layers.12.ffn.experts") == 12
    assert expert_placement.layer_index("layers.3.ffn.experts.17.w1.weight") == 3
    assert expert_placement.layer_index("mtp.0.ffn.experts.1.w1.weight") is None
    assert expert_placement.layer_index("model.sublayers.3.x") is None


@pytest.fixture(autouse=True)
def _no_vllm_config(monkeypatch):
    """Outside a served model the target group applies and the env var selects."""
    import vllm.config

    def _raise():
        raise AssertionError("no current vllm config in this test")

    monkeypatch.setattr(vllm.config, "get_current_vllm_config", _raise)


def _fake_layer(name, rank, experts=8, shards=2):
    manager = SimpleNamespace(
        global_num_experts=experts,
        ep_size=shards,
        ep_rank=rank,
        num_fused_shared_experts=0,
        expert_mask=None,
        _expert_map=torch.arange(experts, dtype=torch.int32),
    )
    manager.expert_map = manager._expert_map

    class Layer(torch.nn.Module):
        def update_expert_map_info(self):
            self.local_num_experts = self.expert_map_manager._local_num_experts
            self.register_buffer("_expert_map", self.expert_map_manager._expert_map)

    layer = Layer()
    layer.layer_name = name
    layer.expert_map_manager = manager
    return layer


def test_apply_maps_slots_and_gates_the_fetch_filter(placement_file):
    assert expert_placement.fetch_filter_ids("layers.3.ffn.experts.0.w1", 1) is None
    layer = _fake_layer("model.layers.3.ffn.experts", rank=1)
    assert expert_placement.apply_to_layer(layer)
    assert layer._expert_map.tolist() == [-1, 0, -1, 2, 3, -1, 1, -1]
    assert layer.local_num_experts == 4
    slots = expert_placement.expert_slot_table(layer)
    assert slots.dtype == torch.int32
    assert slots.tolist() == [[1], [4], [3], [6], [7], [2], [5], [0]]
    assert expert_placement.physical_experts(layer) == 8
    assert expert_placement.fetch_filter_ids(
        "layers.3.ffn.experts.0.w1.weight", 1
    ) == frozenset({1, 6, 3, 4})
    # Unlisted layers and drafter tensors keep the default filter.
    assert expert_placement.fetch_filter_ids("layers.4.ffn.experts.0.w1", 1) is None
    assert expert_placement.fetch_filter_ids("mtp.3.ffn.experts.0.w1", 1) is None


def test_replicas_widen_the_fetch_set_and_the_slot_count(tmp_path, monkeypatch):
    path = tmp_path / "replicas.json"
    path.write_text(
        json.dumps(_table(layers={"3": [[7, 0, 5, 2, 3], [1, 6, 3, 4, 7]]}))
    )
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", str(path))
    layer = _fake_layer("model.layers.3.ffn.experts", rank=0)
    assert expert_placement.apply_to_layer(layer)
    assert layer.local_num_experts == 5
    assert layer._expert_map.tolist() == [1, -1, 3, 4, -1, 2, -1, 0]
    assert expert_placement.physical_experts(layer) == 10
    assert expert_placement.fetch_filter_ids(
        "layers.3.ffn.experts.0.w1.weight", 0
    ) == frozenset({7, 0, 5, 2, 3})


def test_unlisted_layer_is_untouched_and_any_path_may_serve_it(placement_file):
    layer = _fake_layer("model.layers.4.ffn.experts", rank=0)
    assert not expert_placement.apply_to_layer(layer)
    assert expert_placement.expert_slot_table(layer) is None
    assert expert_placement.physical_experts(layer) is None
    expert_placement.check_layer_served(layer, fused_armed=False)


def test_a_placed_layer_must_be_served_by_the_fused_kernel(placement_file):
    layer = _fake_layer("model.layers.3.ffn.experts", rank=0)
    expert_placement.apply_to_layer(layer)
    expert_placement.check_layer_served(layer, fused_armed=True)
    with pytest.raises(RuntimeError, match="not armed"):
        expert_placement.check_layer_served(layer, fused_armed=False)


def test_a_table_for_another_mesh_is_refused(placement_file):
    layer = _fake_layer("model.layers.3.ffn.experts", rank=0, experts=16, shards=4)
    with pytest.raises(ValueError, match="is for 2 ranks x 8 experts"):
        expert_placement.apply_to_layer(layer)


def test_drafter_layers_take_the_draft_group_by_runner_type(tmp_path, monkeypatch):
    import vllm.config

    path = tmp_path / "drafter.json"
    draft = {"experts": 4, "layers": {"1": [[0, 1, 2], [2, 3, 1]]}}
    path.write_text(json.dumps(_table(draft=draft)))
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", str(path))
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            runner_type="draft", hf_config=SimpleNamespace(num_hidden_layers=6)
        )
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: runner)
    # A 6-layer target: the drafter's module `model.layers.7` is draft layer 1;
    # its `model.layers.6` is draft layer 0, which this layout does not list.
    assert not expert_placement.apply_to_layer(
        _fake_layer("model.layers.6.ffn.experts", 1, experts=4)
    )
    layer = _fake_layer("model.layers.7.ffn.experts", 1, experts=4)
    assert expert_placement.apply_to_layer(layer)
    assert layer.local_num_experts == 3
    assert expert_placement.physical_experts(layer) == 6
    assert expert_placement.fetch_filter_ids("mtp.1.ffn.experts.0.w1.weight", 1) == (
        frozenset({2, 3, 1})
    )
    assert expert_placement.fetch_filter_ids("mtp.0.ffn.experts.0.w1", 1) is None
    # The target's layer 1 is a different layer and was not applied.
    assert expert_placement.fetch_filter_ids("layers.1.ffn.experts.0.w1", 1) is None


def test_the_policy_field_selects_the_layout_over_the_env(tmp_path, monkeypatch):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(_table()))
    monkeypatch.delenv("TPU_EXPERT_PLACEMENT", raising=False)
    config = SimpleNamespace(
        additional_config={"deepseek_v41": {"expert_layout": str(path)}}
    )
    assert expert_placement.configured_value(config) == str(path)
    assert expert_placement.current(config).target.slots == 4
    assert expert_placement.content_hash(config) is not None
    assert (
        expert_placement.configured_value(SimpleNamespace(additional_config={})) is None
    )


def test_shipped_layouts_resolve_by_name_and_parse():
    shipped = sorted(
        f for f in os.listdir(expert_placement.PLACEMENT_DIR) if f.endswith(".json")
    )
    assert shipped
    for f in shipped:
        path = expert_placement.resolve(f[:-5])
        assert path == os.path.join(expert_placement.PLACEMENT_DIR, f)
        with open(path) as fh:
            expert_placement.parse(json.load(fh), path)
    with pytest.raises(FileNotFoundError, match="shipped layout"):
        expert_placement.resolve("no_such_layout")
    assert expert_placement.resolve("/tmp/x.json") == "/tmp/x.json"


def test_content_hash_tracks_table_bytes(tmp_path, monkeypatch):
    monkeypatch.delenv("TPU_EXPERT_PLACEMENT", raising=False)
    assert expert_placement.content_hash() is None
    path = tmp_path / "placement.json"
    path.write_text(json.dumps(_table()))
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", str(path))
    first = expert_placement.content_hash()
    assert len(first) == 64
    path.write_text(json.dumps(_table(layers={"3": [[0, 1, 2, 3], [4, 5, 6, 7]]})))
    expert_placement._read_once.cache_clear()
    assert expert_placement.content_hash() != first
