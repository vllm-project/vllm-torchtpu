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
from vllm_torchtpu.layers.adapter import moe_routing

# Rank 0 holds experts 7, 0, 5, 2 in slots 0-3; rank 1 holds 1, 6, 3, 4 in
# slots 4-7.
ORDER = (7, 0, 5, 2, 1, 6, 3, 4)
SLOT = [1, 4, 3, 6, 7, 2, 5, 0]


def _table(experts=8, shards=2, layers=None):
    layers = layers or {"3": [list(ORDER[:4]), list(ORDER[4:])]}
    return {"experts": experts, "shards": shards, "layers": layers}


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    expert_placement.clear_cache()
    monkeypatch.setattr(expert_placement, "_APPLIED_LAYERS", {})
    monkeypatch.setattr(expert_placement, "_PERMUTED_LAYERS", set())
    yield
    expert_placement.clear_cache()


@pytest.fixture(autouse=True)
def _target_has_six_layers(monkeypatch):
    import vllm.config

    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(num_hidden_layers=6))
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: config)
    expert_placement._target_layers.cache_clear()
    yield
    expert_placement._target_layers.cache_clear()


@pytest.fixture
def placement_file(tmp_path, monkeypatch):
    path = tmp_path / "placement.json"
    path.write_text(json.dumps(_table()))
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", str(path))
    return path


class _Experts(torch.nn.Module):
    """The RoutedExperts attributes placement and routing read."""

    def __init__(self, name, rank, experts, shards, score_bias, hash_table):
        super().__init__()
        self.layer_name = name
        self.global_num_experts = experts
        self.expert_map_manager = SimpleNamespace(
            global_num_experts=experts,
            ep_size=shards,
            ep_rank=rank,
            local_num_experts=experts // shards,
            num_fused_shared_experts=0,
            expert_mask=None,
            placement_strategy="linear",
            _expert_map=torch.arange(experts, dtype=torch.int32),
        )
        self.expert_map_manager.expert_map = self.expert_map_manager._expert_map
        self.moe_config = SimpleNamespace(
            experts_per_token=2,
            moe_parallel_config=SimpleNamespace(enable_eplb=False),
        )
        self.renormalize = True
        self.scoring_func = "sqrtsoftplus"
        self.use_grouped_topk = False
        self.num_expert_group = None
        self.topk_group = None
        self.routed_scaling_factor = 1.0
        self.custom_routing_function = None
        self.e_score_correction_bias = score_bias
        self.hash_indices_table = hash_table

    def update_expert_map_info(self):
        self.register_buffer("_expert_map", self.expert_map_manager._expert_map)


class _Runner(torch.nn.Module):
    """A MoE layer as the TPU MoE runner holds it: gate plus routed experts."""

    def __init__(
        self, name, rank=0, experts=8, shards=2, hidden=3, hash_layer=False, bias=False
    ):
        super().__init__()
        gen = torch.Generator().manual_seed(rank)
        self.gate = torch.nn.Linear(hidden, experts, bias=bias)
        self.gate.weight.data.copy_(torch.randn(experts, hidden, generator=gen))
        score_bias = None
        hash_table = None
        if hash_layer:
            hash_table = torch.nn.Parameter(
                torch.randint(0, experts, (5, 2), dtype=torch.int32, generator=gen),
                requires_grad=False,
            )
            self.gate.tid2eid = hash_table
        else:
            score_bias = torch.nn.Parameter(
                torch.randn(experts, generator=gen), requires_grad=False
            )
            self.gate.e_score_correction_bias = score_bias
        self.routed_experts = _Experts(
            name, rank, experts, shards, score_bias, hash_table
        )


def _model(*runners):
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(runners)
    return model


def test_slot_order_and_table_invert_each_other():
    p = expert_placement.parse(_table())
    assert p.slot_order(3) == ORDER
    assert p.slot_table(3) == SLOT
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


def test_apply_maps_slots_and_gates_the_fetch_filter(placement_file):
    assert expert_placement.fetch_filter_ids("layers.3.ffn.experts.0.w1") is None
    runner = _Runner("model.layers.3.ffn.experts", rank=1)
    assert expert_placement.apply(runner)
    layer = runner.routed_experts
    assert layer._expert_map.tolist() == [-1, 0, -1, 2, 3, -1, 1, -1]
    assert expert_placement.expert_order(layer) == ORDER
    # The filter keeps rank 1's experts because the expert map was built for
    # rank 1; it is never handed a rank of its own to disagree with.
    assert expert_placement.fetch_filter_ids(
        "layers.3.ffn.experts.0.w1.weight"
    ) == frozenset({1, 6, 3, 4})
    # Unlisted layers and drafter layers keep the default filter.
    assert expert_placement.fetch_filter_ids("layers.4.ffn.experts.0.w1") is None
    assert expert_placement.fetch_filter_ids("mtp.0.ffn.experts.0.w1") is None


def test_the_fetch_filter_follows_the_rank_the_map_was_built_for(placement_file):
    expert_placement.apply(_Runner("model.layers.3.ffn.experts", rank=0))
    assert expert_placement.fetch_filter_ids(
        "layers.3.ffn.experts.5.w2.weight"
    ) == frozenset({7, 0, 5, 2})


def test_unlisted_and_unplaced_layers_are_untouched(placement_file, monkeypatch):
    runner = _Runner("model.layers.4.ffn.experts")
    assert not expert_placement.apply(runner)
    assert expert_placement.expert_order(runner.routed_experts) is None
    assert runner.routed_experts.expert_map_manager._expert_map.tolist() == list(
        range(8)
    )
    monkeypatch.delenv("TPU_EXPERT_PLACEMENT")
    # Without a placement apply reads nothing off the layer.
    assert not expert_placement.apply(SimpleNamespace())
    assert not expert_placement.has_applied_layers()


def test_a_table_for_another_mesh_is_refused(placement_file):
    runner = _Runner("model.layers.3.ffn.experts", experts=16, shards=4)
    with pytest.raises(ValueError, match="is for 2 ranks x 8 experts"):
        expert_placement.apply(runner)


@pytest.mark.parametrize(
    "change, reason",
    [
        (lambda r: setattr(r, "gate", None), "outside the MoE layer"),
        (
            lambda r: setattr(r.routed_experts, "use_grouped_topk", True)
            or setattr(r.routed_experts, "num_expert_group", 4),
            "groups of contiguous",
        ),
        (
            lambda r: setattr(r.routed_experts, "custom_routing_function", len),
            "custom routing function",
        ),
        (
            lambda r: setattr(
                r.routed_experts.moe_config.moe_parallel_config, "enable_eplb", True
            ),
            "already places",
        ),
    ],
)
def test_routing_a_renumbering_would_change_is_refused(placement_file, change, reason):
    runner = _Runner("model.layers.3.ffn.experts")
    change(runner)
    with pytest.raises(ValueError, match=reason):
        expert_placement.apply(runner)


def test_drafter_layers_take_their_mtp_tensors(tmp_path, monkeypatch):
    path = tmp_path / "drafter.json"
    layers = {"7": [list(ORDER[:4]), list(ORDER[4:])]}
    path.write_text(json.dumps(_table(layers=layers)))
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", str(path))
    assert expert_placement.apply(_Runner("model.layers.7.ffn.experts", rank=1))
    # A 6-layer target: mtp.1 is module layers.7.
    name = "mtp.1.ffn.experts.0.w1.weight"
    assert expert_placement.fetch_filter_ids(name) == frozenset({1, 6, 3, 4})
    assert expert_placement.fetch_filter_ids("mtp.0.ffn.experts.0.w1") is None


@pytest.mark.parametrize("bias", [False, True])
def test_permute_routers_renumbers_gate_rows_and_score_bias(placement_file, bias):
    runner = _Runner("model.layers.3.ffn.experts", bias=bias)
    weight = runner.gate.weight.detach().clone()
    gate_bias = None if runner.gate.bias is None else runner.gate.bias.detach().clone()
    score_bias = runner.routed_experts.e_score_correction_bias.detach().clone()
    model = _model(runner)
    assert expert_placement.apply(runner)
    expert_placement.permute_routers(model)
    rows = list(ORDER)
    torch.testing.assert_close(runner.gate.weight.detach(), weight[rows])
    torch.testing.assert_close(
        runner.routed_experts.e_score_correction_bias.detach(), score_bias[rows]
    )
    if bias:
        torch.testing.assert_close(runner.gate.bias.detach(), gate_bias[rows])
    expert_placement.check_routers_permuted()


def test_permute_routers_maps_hash_table_ids_to_slots(placement_file):
    runner = _Runner("model.layers.3.ffn.experts", hash_layer=True)
    table = runner.routed_experts.hash_indices_table.detach().clone()
    model = _model(runner)
    expert_placement.apply(runner)
    expert_placement.permute_routers(model)
    permuted = runner.routed_experts.hash_indices_table
    assert permuted.dtype == torch.int32
    assert permuted.tolist() == [[SLOT[e] for e in row] for row in table.tolist()]


@pytest.mark.parametrize(
    "scoring, one_group",
    [
        ("sqrtsoftplus", False),  # DeepSeek-V4
        # DeepSeek-V2-style grouped top-k over a single group, as in GLM-5.
        ("sigmoid", True),
    ],
)
def test_routing_in_slot_order_selects_the_same_experts(
    placement_file, scoring, one_group
):
    """The renumbered router picks the same experts, with the same weights,
    as the checkpoint's: only their ids change, to the slots holding them."""
    runner = _Runner("model.layers.3.ffn.experts")
    layer = runner.routed_experts
    if one_group:
        layer.use_grouped_topk = True
        layer.num_expert_group = layer.topk_group = 1
    x = torch.randn(16, 3, generator=torch.Generator().manual_seed(7))

    def route():
        logits = runner.gate(x).detach()
        return moe_routing.select_experts(
            hidden_states=x,
            router_logits=logits,
            topk=2,
            renormalize=True,
            scoring_fn=scoring,
            layer=layer,
        )

    weights, ids = route()
    model = _model(runner)
    expert_placement.apply(runner)
    expert_placement.permute_routers(model)
    placed_weights, slots = route()
    assert slots.tolist() == [[SLOT[e] for e in row] for row in ids.tolist()]
    torch.testing.assert_close(placed_weights, weights)


def test_each_router_is_renumbered_once(placement_file):
    runner = _Runner("model.layers.3.ffn.experts")
    weight = runner.gate.weight.detach().clone()
    model = _model(runner)
    expert_placement.apply(runner)
    expert_placement.permute_routers(model)
    expert_placement.permute_routers(model)
    torch.testing.assert_close(runner.gate.weight.detach(), weight[list(ORDER)])


def test_a_placed_layer_whose_router_was_not_renumbered_is_refused(placement_file):
    runner = _Runner("model.layers.3.ffn.experts")
    expert_placement.apply(runner)
    with pytest.raises(RuntimeError, match=r"MoE layers \[3\].*permute_routers"):
        expert_placement.check_routers_permuted()
    expert_placement.permute_routers(_model(runner))
    expert_placement.check_routers_permuted()


def test_listed_layers_the_process_never_built_are_reported(tmp_path, monkeypatch):
    path = tmp_path / "two_layers.json"
    held = [list(ORDER[:4]), list(ORDER[4:])]
    path.write_text(json.dumps(_table(layers={"3": held, "4": held})))
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", str(path))
    warnings = []
    monkeypatch.setattr(
        expert_placement.logger, "warning", lambda *args: warnings.append(args)
    )
    runner = _Runner("model.layers.3.ffn.experts")
    expert_placement.apply(runner)
    expert_placement.permute_routers(_model(runner))
    expert_placement.check_routers_permuted()
    assert len(warnings) == 1 and warnings[0][2] == [4]


def test_the_loader_renumbers_routers_before_post_processing(
    placement_file, monkeypatch
):
    """No model code: the model-loader patch renumbers every placed router
    between the checkpoint write and process_weights_after_loading."""
    from vllm.model_executor.model_loader import base_loader

    from vllm_torchtpu import model_loader_patches

    seen = []

    def post_process(model, model_config, target_device):
        seen.append(model.layers[0].gate.weight.detach().clone())

    monkeypatch.setattr(base_loader, "process_weights_after_loading", post_process)
    model_loader_patches.patch_expert_placement_routers()
    model_loader_patches.patch_expert_placement_routers()
    assert base_loader.process_weights_after_loading.__wrapped__ is post_process
    runner = _Runner("model.layers.3.ffn.experts")
    weight = runner.gate.weight.detach().clone()
    expert_placement.apply(runner)
    base_loader.process_weights_after_loading(_model(runner), None, None)
    torch.testing.assert_close(seen[0], weight[list(ORDER)])
    expert_placement.check_routers_permuted()


def test_a_gate_with_tensors_the_renumbering_misses_is_refused(placement_file):
    runner = _Runner("model.layers.3.ffn.experts")
    runner.gate.weight_scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    model = _model(runner)
    expert_placement.apply(runner)
    with pytest.raises(ValueError, match="weight_scale"):
        expert_placement.permute_routers(model)


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


def test_a_table_is_read_once_per_process(placement_file):
    """A table rewritten mid-startup must not place some layers one way and
    the rest another: every reader keeps the first read."""
    placement = expert_placement.current()
    placement_file.write_text(
        json.dumps(_table(layers={"3": [[0, 1, 2, 3], [4, 5, 6, 7]]}))
    )
    assert expert_placement.current() == placement
    expert_placement.clear_cache()
    assert expert_placement.current() != placement


def test_a_gs_table_without_gcloud_says_so(monkeypatch):
    def missing(*_args, **_kwargs):
        raise FileNotFoundError("gcloud")

    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="gcloud"):
        expert_placement._read("gs://bucket/placement.json")
