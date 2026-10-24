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
"""Host-side checks for TPU_EXPERT_PLACEMENT (vllm_torchtpu/expert_placement.py)."""

import json
import os
import subprocess
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import expert_placement


def _table(experts=8, shards=2, layers=None):
    layers = layers or {"3": [[7, 0, 5, 2], [1, 6, 3, 4]]}
    return {"experts": experts, "shards": shards, "layers": layers}


@pytest.fixture(autouse=True)
def _fresh_tables():
    expert_placement.clear_cache()
    yield
    expert_placement.clear_cache()


@pytest.fixture
def placement_file(tmp_path, monkeypatch):
    path = tmp_path / "placement.json"
    path.write_text(json.dumps(_table()))
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", str(path))
    monkeypatch.setenv("USE_MOE_FUSED_EP_KERNEL", "1")
    monkeypatch.setattr(expert_placement, "_APPLIED_LAYERS", {})
    yield path


def test_slot_table_inverts_the_rank_lists():
    p = expert_placement.parse(_table())
    assert p.slot_table(3) == [1, 4, 3, 6, 7, 2, 5, 0]
    assert p.ranks(4) is None


@pytest.mark.parametrize(
    "layers",
    [
        {"3": [[7, 0, 5, 2], [1, 6, 3, 3]]},  # 4 missing, 3 twice
        {"3": [[7, 0, 5], [1, 6, 3, 4, 2]]},  # uneven ranks
        {"3": [[7, 0, 5, 2]]},  # one rank short
        {"3": [[7, 7, 5, 2], [1, 6, 3, 4]]},  # 0 missing, 7 twice on rank 0
        {"3": [[7, 0, 5, 2, 3], [1, 6, 3, 4, 7]]},  # replicas: not a permutation
    ],
)
def test_a_table_that_is_not_a_permutation_is_refused(layers):
    with pytest.raises(ValueError):
        expert_placement.parse(_table(layers=layers))


def test_layer_index_reads_module_prefixes_and_checkpoint_names():
    assert expert_placement.layer_index("model.layers.12.ffn.experts") == 12
    assert expert_placement.layer_index("layers.3.ffn.experts.17.w1.weight") == 3
    assert expert_placement.layer_index("mtp.0.ffn.experts.1.w1.weight") is None
    assert expert_placement.layer_index("model.sublayers.3.x") is None


@pytest.fixture(autouse=True)
def _target_has_six_layers(monkeypatch):
    import vllm.config

    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(num_hidden_layers=6))
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: config)


def _fake_layer(name, rank, experts=8, shards=2):
    manager = SimpleNamespace(
        global_num_experts=experts,
        ep_size=shards,
        ep_rank=rank,
        local_num_experts=experts // shards,
        num_fused_shared_experts=0,
        expert_mask=None,
        _expert_map=torch.arange(experts, dtype=torch.int32),
    )
    manager.expert_map = manager._expert_map

    class Layer(torch.nn.Module):
        def update_expert_map_info(self):
            self.register_buffer("_expert_map", self.expert_map_manager._expert_map)

    layer = Layer()
    layer.layer_name = name
    layer.expert_map_manager = manager
    return layer


def test_apply_maps_slots_and_gates_the_fetch_filter(placement_file):
    assert expert_placement.fetch_filter_ids("layers.3.ffn.experts.0.w1") is None
    layer = _fake_layer("model.layers.3.ffn.experts", rank=1)
    assert expert_placement.apply_to_layer(layer)
    assert layer._expert_map.tolist() == [-1, 0, -1, 2, 3, -1, 1, -1]
    slots = expert_placement.expert_slot_table(layer)
    assert slots.dtype == torch.int32
    assert slots.tolist() == [1, 4, 3, 6, 7, 2, 5, 0]
    # The filter keeps rank 1's experts because the expert map was built for
    # rank 1; it is never handed a rank of its own to disagree with.
    assert expert_placement.fetch_filter_ids(
        "layers.3.ffn.experts.0.w1.weight"
    ) == frozenset({1, 6, 3, 4})
    # Unlisted layers and drafter layers keep the default filter.
    assert expert_placement.fetch_filter_ids("layers.4.ffn.experts.0.w1") is None
    assert expert_placement.fetch_filter_ids("mtp.0.ffn.experts.0.w1") is None


def test_the_fetch_filter_follows_the_rank_the_map_was_built_for(placement_file):
    expert_placement.apply_to_layer(_fake_layer("model.layers.3.ffn.experts", 0))
    assert expert_placement.fetch_filter_ids(
        "layers.3.ffn.experts.5.w2.weight"
    ) == frozenset({7, 0, 5, 2})


def test_unlisted_layer_is_untouched_and_any_path_may_serve_it(placement_file):
    layer = _fake_layer("model.layers.4.ffn.experts", rank=0)
    assert not expert_placement.apply_to_layer(layer)
    assert expert_placement.expert_slot_table(layer) is None
    expert_placement.check_layer_served(layer, fused_armed=False)


def test_a_placed_layer_must_be_served_by_the_fused_kernel(placement_file):
    layer = _fake_layer("model.layers.3.ffn.experts", rank=0)
    expert_placement.apply_to_layer(layer)
    expert_placement.check_layer_served(layer, fused_armed=True)
    with pytest.raises(RuntimeError, match="not armed"):
        expert_placement.check_layer_served(layer, fused_armed=False)


def test_a_placement_without_the_fused_kernel_fails_before_the_load(
    placement_file, monkeypatch
):
    monkeypatch.setenv("USE_MOE_FUSED_EP_KERNEL", "0")
    with pytest.raises(RuntimeError, match="USE_MOE_FUSED_EP_KERNEL"):
        expert_placement.apply_to_layer(_fake_layer("model.layers.3.ffn.experts", 0))
    # An unlisted layer is no reason to refuse.
    assert not expert_placement.apply_to_layer(
        _fake_layer("model.layers.4.ffn.experts", 0)
    )


def test_a_table_for_another_mesh_is_refused(placement_file):
    layer = _fake_layer("model.layers.3.ffn.experts", rank=0, experts=16, shards=4)
    with pytest.raises(ValueError, match="is for 2 ranks x 8 experts"):
        expert_placement.apply_to_layer(layer)


def test_drafter_layers_take_their_mtp_tensors(tmp_path, monkeypatch):
    monkeypatch.setattr(expert_placement, "_APPLIED_LAYERS", {})
    monkeypatch.setenv("USE_MOE_FUSED_EP_KERNEL", "1")
    expert_placement._target_layers.cache_clear()
    path = tmp_path / "drafter.json"
    path.write_text(json.dumps(_table(layers={"7": [[7, 0, 5, 2], [1, 6, 3, 4]]})))
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", str(path))
    assert expert_placement.apply_to_layer(_fake_layer("model.layers.7.ffn.experts", 1))
    # A 6-layer target: mtp.1 is module layers.7.
    name = "mtp.1.ffn.experts.0.w1.weight"
    assert expert_placement.fetch_filter_ids(name) == frozenset({1, 6, 3, 4})
    assert expert_placement.fetch_filter_ids("mtp.0.ffn.experts.0.w1") is None
    expert_placement._target_layers.cache_clear()


def test_shipped_placements_resolve_by_name_and_parse():
    shipped = sorted(os.listdir(expert_placement.PLACEMENT_DIR))
    assert shipped
    for f in shipped:
        path = expert_placement.resolve(f[:-5])
        assert path == os.path.join(expert_placement.PLACEMENT_DIR, f)
        with open(path) as fh:
            expert_placement.parse(json.load(fh), path)
    with pytest.raises(FileNotFoundError, match="shipped placement"):
        expert_placement.resolve("no_such_placement")
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
    expert_placement.clear_cache()
    assert expert_placement.content_hash() != first


def test_the_cache_key_and_the_served_table_come_from_one_read(placement_file):
    """A table rewritten mid-startup must not key one placement and serve
    another: both answers stay those of the first read."""
    digest, placement = expert_placement.content_hash(), expert_placement.current()
    placement_file.write_text(
        json.dumps(_table(layers={"3": [[0, 1, 2, 3], [4, 5, 6, 7]]}))
    )
    assert expert_placement.content_hash() == digest
    assert expert_placement.current() == placement


def test_a_gs_table_without_gcloud_says_so(monkeypatch):
    def missing(*_args, **_kwargs):
        raise FileNotFoundError("gcloud")

    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="gcloud"):
        expert_placement._read("gs://bucket/placement.json")
