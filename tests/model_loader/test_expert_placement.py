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
"""Static expert placement (expert_placement.py) on synthetic tables: the
table checks, the maps, the router rows, the EP weight filter, and a target
layer and an MTP draft layer built through the installed vLLM hooks."""

import json
from types import SimpleNamespace

import pytest
import torch
import vllm.config
from vllm.model_executor.layers.fused_moe import expert_map_manager
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.model_loader import default_loader, runai_streamer_loader

import vllm_torchtpu
from vllm_torchtpu import envs
from vllm_torchtpu import expert_placement as ep
from vllm_torchtpu.model_loader_patches import _should_skip

pytestmark = pytest.mark.cpu_test

TARGET = "language_model.model.layers.0.mlp.experts"
DRAFT = "mtp.layers.0.mlp.experts"


def _plain_table():
    # Rank 0 holds experts 3, 0; rank 1 holds 2, 1.
    cores = [[3, 0], [2, 1]]
    return {
        "shards": 2,
        "experts": 4,
        "slots_per_core": 2,
        "owned_per_core": 2,
        "layers": {"0": {"cores": cores}, "1": {"cores": cores}},
    }


def _copies_table():
    return {
        "shards": 2,
        "experts": 4,
        "owned_per_core": 2,
        "slots_per_core": 3,
        "copies": True,
        "layers": {"0": {"cores": [[0, 1, 2], [2, 3, 0]], "copies": {"2": 0, "0": 1}}},
    }


@pytest.fixture
def use_table(tmp_path, monkeypatch):
    def use(table):
        path = tmp_path / "placement.json"
        path.write_text(json.dumps(table))
        monkeypatch.setenv("TPU_EXPERT_PLACEMENT", str(path))
        ep._load.cache_clear()
        return str(path)

    monkeypatch.setattr(ep, "_LOCAL_IDS", {})
    yield use
    ep._load.cache_clear()


def test_maps_and_router_rows(use_table):
    use_table(_plain_table())
    table = ep.placement()
    assert (table.ranks, table.slots, table.owned, table.experts) == (2, 2, 2, 4)
    assert ep.virtual_order(table, 0) == [3, 0, 2, 1]
    assert ep.expert_map(table, 0, 0) == [1, -1, -1, 0]
    assert ep.expert_map(table, 0, 1) == [-1, 1, 0, -1]
    assert ep.num_primary_experts(TARGET) is None
    w = torch.arange(4.0).reshape(4, 1)
    out = ep.permute_router_rows("model.layers.0.mlp.gate.weight", w)
    assert out[:, 0].tolist() == [3.0, 0.0, 2.0, 1.0]
    assert (
        ep.permute_router_rows("model.layers.0.mlp.shared_expert_gate.weight", w) is w
    )
    assert ep.permute_router_rows("model.layers.7.mlp.gate.weight", w) is w
    with pytest.raises(ValueError, match="5 rows"):
        ep.permute_router_rows("model.layers.1.mlp.gate.weight", torch.zeros(5, 1))


def test_unset_is_the_identity(monkeypatch):
    monkeypatch.delenv("TPU_EXPERT_PLACEMENT", raising=False)
    assert ep.placement() is None
    w = torch.zeros(4, 1)
    assert ep.permute_router_rows("model.layers.0.mlp.gate.weight", w) is w
    assert ep.local_expert_ids("model.layers.0.mlp.experts.3.w1.weight") is None


def test_mtp_draft_layers_keep_linear_placement(use_table):
    use_table(_copies_table())
    w = torch.arange(4.0).reshape(4, 1)
    assert ep.layer_index(DRAFT) is None
    assert ep.layer_index(TARGET) == 0
    assert ep.permute_router_rows("mtp.layers.0.mlp.gate.weight", w) is w
    assert ep.num_primary_experts(DRAFT) is None
    assert ep.copy_layout(DRAFT, 0) is None


def test_copy_layout_in_router_row_ids(use_table):
    use_table(_copies_table())
    assert ep.num_primary_experts(TARGET) == 2
    assert ep.placement().hot[0] == (2, 0)
    assert ep.copy_layout(TARGET, 0) == ([0, 1, 2], [1, -1, 0, -1], [2, 0])
    assert ep.copy_layout(TARGET, 1) == ([2, 3, 0], [1, -1, 0, -1], [2, 0])


@pytest.mark.parametrize(
    "case",
    ["duplicate_primary", "invalid_copy", "duplicate_copy", "own_copy", "order"],
)
def test_invalid_tables_are_refused(use_table, case):
    table = _copies_table()
    layer = table["layers"]["0"]
    if case == "duplicate_primary":
        layer["cores"] = [[0, 0, 2], [2, 3, 1]]
    elif case == "invalid_copy":
        layer["cores"][0][-1] = 4
    elif case == "duplicate_copy":
        table["slots_per_core"] = 4
        layer["cores"] = [[0, 1, 2, 2], [2, 3, 0, 1]]
    elif case == "own_copy":
        layer["cores"][0][-1] = 1
    else:
        layer["copies"] = {"2": 1, "0": 0}
    use_table(table)
    with pytest.raises(ValueError, match="expert placement"):
        ep.placement()


def test_weight_filter_follows_the_placed_layer(use_table, monkeypatch):
    use_table(_plain_table())
    monkeypatch.setattr(ep, "_LOCAL_IDS", {0: frozenset({2, 1})})
    linear = {0, 1}
    name = "model.language_model.layers.{}.mlp.experts.{}.gate_proj.weight"
    assert _should_skip(name.format(0, 0), linear)
    assert not _should_skip(name.format(0, 2), linear)
    assert not _should_skip(name.format(3, 0), linear)
    assert _should_skip(name.format(3, 2), linear)
    assert not _should_skip("mtp.layers.0.mlp.experts.0.gate_proj.weight", linear)


class _Experts(torch.nn.Module):
    """The RoutedExperts state update_expert_map_info reads."""

    def __init__(self, layer_name, rank, quant_config=None):
        super().__init__()
        parallel = SimpleNamespace(
            use_ep=True, ep_size=2, ep_rank=rank, use_all2all_kernels=False
        )
        self.layer_name = layer_name
        self.quant_config = quant_config
        self.expert_map_manager = expert_map_manager.ExpertMapManager(
            max_num_batched_tokens=8,
            top_k=2,
            global_num_experts=4,
            num_redundant_experts=0,
            num_expert_group=None,
            moe_parallel_config=parallel,
            placement_strategy="linear",
            enable_eplb=False,
        )
        self.moe_config = SimpleNamespace(
            moe_parallel_config=parallel,
            num_local_experts=self.expert_map_manager.local_num_experts,
        )


@pytest.fixture
def installed(monkeypatch):
    """Main's host-built map; returns the placement install. Both undone."""
    manager_cls = expert_map_manager.ExpertMapManager
    for name in (
        "_calculate_expert_maps",
        "map_global_to_local",
        "get_compressed_map_string",
        "_tpu_expert_map_source",
        "_tpu_expert_map_version",
        "_tpu_expert_map_host",
        "_tpu_host_expert_map_patch",
    ):
        original = getattr(manager_cls, name, None)
        monkeypatch.setattr(manager_cls, name, original, raising=False)
    monkeypatch.setattr(
        expert_map_manager,
        "determine_expert_map",
        expert_map_manager.determine_expert_map,
    )
    monkeypatch.setattr(
        RoutedExperts, "update_expert_map_info", RoutedExperts.update_expert_map_info
    )
    monkeypatch.setattr(
        RoutedExperts, "_tpu_expert_placement_patch", False, raising=False
    )
    loader = default_loader.DefaultModelLoader
    streamer = runai_streamer_loader.RunaiModelStreamerLoader
    monkeypatch.setattr(loader, "get_all_weights", loader.get_all_weights)
    monkeypatch.setattr(
        streamer, "_get_weights_iterator", streamer._get_weights_iterator
    )
    vllm_torchtpu._patch_expert_map_host_lookup()
    _load_format(monkeypatch, "auto")
    return ep.install_expert_placement


def _load_format(monkeypatch, load_format):
    config = SimpleNamespace(load_config=SimpleNamespace(load_format=load_format))
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: config)


def _fp8_config():
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config

    return VllmFp8Config.__new__(VllmFp8Config)


@pytest.mark.parametrize("copies", [False, True])
def test_target_is_placed_and_the_mtp_draft_stays_linear(use_table, installed, copies):
    use_table(_copies_table() if copies else _plain_table())
    installed()
    table = ep.placement()
    target = _Experts(TARGET, rank=1, quant_config=_fp8_config())
    draft = _Experts(DRAFT, rank=1, quant_config=_fp8_config())
    for layer in (target, draft):
        RoutedExperts.update_expert_map_info(layer)

    manager = target.expert_map_manager
    assert target._expert_map.tolist() == ep.expert_map(table, 0, 1)
    assert target.local_num_experts == target.moe_config.num_local_experts
    assert target.local_num_experts == table.slots
    for expert, slot in enumerate(ep.expert_map(table, 0, 1)):
        assert manager.map_global_to_local(expert) == slot
    assert draft._expert_map.tolist() == [-1, -1, 0, 1]
    assert draft.local_num_experts == draft.moe_config.num_local_experts == 2
    assert {0: frozenset(table.cores[0][1])} == ep._LOCAL_IDS

    rows = torch.arange(4.0)
    target_rows = ep.permute_router_rows(
        "model.language_model.layers.0.mlp.gate.weight", rows
    )
    assert target_rows.tolist() == [float(e) for e in ep.virtual_order(table, 0)]
    assert ep.permute_router_rows("mtp.layers.0.mlp.gate.weight", rows) is rows


def test_weight_streams_reorder_router_rows(use_table, installed, monkeypatch):
    use_table(_plain_table())
    weights = [
        ("model.layers.0.mlp.gate.weight", torch.arange(4.0)),
        ("model.layers.0.mlp.experts.1.down_proj.weight", torch.ones(2)),
    ]
    loader = default_loader.DefaultModelLoader
    streamer = runai_streamer_loader.RunaiModelStreamerLoader
    monkeypatch.setattr(loader, "get_all_weights", lambda *a: iter(weights))
    monkeypatch.setattr(streamer, "_get_weights_iterator", lambda *a: iter(weights))
    installed()
    for stream in (
        loader.get_all_weights(None, None, None),
        streamer._get_weights_iterator(None, None, None),
    ):
        got = dict(stream)
        assert got["model.layers.0.mlp.gate.weight"].tolist() == [3.0, 0.0, 2.0, 1.0]
        assert got["model.layers.0.mlp.experts.1.down_proj.weight"] is weights[1][1]


@pytest.mark.parametrize("refusal", ["fused_ep", "not_fp8", "load_format"])
def test_unsupported_configs_are_refused_at_construction(
    use_table, installed, monkeypatch, refusal
):
    use_table(_plain_table() if refusal == "load_format" else _copies_table())
    installed()
    quant_config = None if refusal == "not_fp8" else _fp8_config()
    monkeypatch.setattr(
        envs, "USE_MOE_FUSED_EP_KERNEL", refusal == "fused_ep", raising=False
    )
    layer = _Experts(TARGET, rank=0, quant_config=quant_config)
    _load_format(monkeypatch, "dummy" if refusal == "load_format" else "auto")
    with pytest.raises(NotImplementedError):
        RoutedExperts.update_expert_map_info(layer)
