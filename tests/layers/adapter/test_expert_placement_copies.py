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
"""Copy slots in TPU_EXPERT_PLACEMENT tables (layers/adapter/expert_placement.py).

A table without copies must behave exactly as main's permutation tables did,
so the parser is checked against a verbatim copy of main's parser from before
copies existed; a table with copies is checked for its semantics, its strict
validation, the hook that sizes its layers before vLLM allocates their
weights, and `apply`.
"""

import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.layers.adapter import expert_placement as ep

pytestmark = pytest.mark.cpu_test

VALIDATION = (
    Path(ep.__file__).resolve().parents[2] / "models" / "qwen3_5" / "validation"
)
DP8_TABLE = VALIDATION / "expert_placement_dp8_ep8.json"
TP8_TABLE = VALIDATION / "expert_placement_tp8_ep8_agentx_corr_c32.json"

# Rank 0: primaries 7, 0, 5, 2 then a copy of 1; rank 1: primaries 1, 6, 3, 4
# then a copy of 7.
ORDER = (7, 0, 5, 2, 1, 6, 3, 4)
SLOT = [1, 4, 3, 6, 7, 2, 5, 0]
LAYER = "model.layers.3.mlp.experts"


def _permutation():
    return {"experts": 8, "shards": 2, "layers": {"3": [[7, 0, 5, 2], [1, 6, 3, 4]]}}


def _copies(copy_order=True):
    table = {
        "experts": 8,
        "shards": 2,
        "copies_per_rank": 1,
        "layers": {"3": [[7, 0, 5, 2, 1], [1, 6, 3, 4, 7]]},
    }
    if copy_order:
        table["copy_order"] = {"3": [7, 1]}
    return table


# Main's parser before copy slots existed, verbatim apart from its name and
# its return value, so that permutation tables can be compared field by field.
def _base_parse(table: dict, source: str = "<dict>"):
    """Validate a placement table: each listed layer permutes its experts over
    the ranks, experts / shards to a rank."""
    experts = int(table["experts"])
    shards = int(table["shards"])
    if experts <= 0 or shards <= 0:
        raise ValueError(
            f"expert placement {source}: experts and shards must be positive, "
            f"not {experts} and {shards}"
        )
    if experts % shards:
        raise ValueError(
            f"expert placement {source}: {experts} experts do not divide over "
            f"{shards} ranks"
        )
    per_rank = experts // shards
    layers = {}
    for key, ranks in table["layers"].items():
        try:
            layer = int(key)
        except ValueError:
            raise ValueError(
                f"expert placement {source}: layer key {key!r} is not a layer index"
            ) from None
        if layer < 0:
            raise ValueError(f"expert placement {source}: layer {key} is negative")
        if layer in layers:
            raise ValueError(
                f"expert placement {source}: layer {layer} is listed twice"
            )
        if len(ranks) != shards or any(len(held) != per_rank for held in ranks):
            raise ValueError(
                f"expert placement {source}: layer {key} must list {shards} "
                f"ranks of {per_rank} experts each"
            )
        ranks = tuple(tuple(int(e) for e in held) for held in ranks)
        if sorted(e for held in ranks for e in held) != list(range(experts)):
            raise ValueError(
                f"expert placement {source}: layer {key} must hold each of the "
                f"{experts} experts exactly once"
            )
        layers[layer] = ranks
    return experts, shards, layers


def _canonical_digest(table: dict) -> str:
    """The data that defines a placement: shape, per-layer rank lists and copy
    order. The constants below are this digest of candidate4's two tables in
    their original format, computed from those files."""
    per_rank = table["experts"] // table["shards"]
    order = table.get("copy_order", {})
    canonical = {
        "experts": table["experts"],
        "shards": table["shards"],
        "slots_per_rank": per_rank + table.get("copies_per_rank", 0),
        "primaries_per_rank": per_rank,
        "layers": {
            str(int(n)): {
                "cores": ranks,
                "copy_order": [int(e) for e in order.get(n, [])],
            }
            for n, ranks in table["layers"].items()
        },
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


CANDIDATE4_DIGESTS = {
    DP8_TABLE: "da314f13b961c219b6a4d6d95dbf32cf0a88dbf66a7b84ec2f1491085e510232",
    TP8_TABLE: "d75857cc53b361c11c5e85edb8bbaaec3e726b9fe8ac156b95494fd69d16b320",
}


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    ep.clear_cache()
    monkeypatch.setattr(ep, "_APPLIED_LAYERS", {})
    monkeypatch.setattr(ep, "_PERMUTED_LAYERS", set())
    yield
    ep.clear_cache()


@pytest.fixture(autouse=True)
def _target_has_six_layers(monkeypatch):
    import vllm.config

    text_config = SimpleNamespace(num_hidden_layers=6)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(text_config=text_config),
            hf_text_config=text_config,
        )
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: config)
    ep._target_layers.cache_clear()
    yield
    ep._target_layers.cache_clear()


def _use(tmp_path, monkeypatch, table):
    path = tmp_path / "placement.json"
    path.write_text(json.dumps(table))
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", str(path))
    return path


# ---- tables without copies are main's tables ----


@pytest.mark.parametrize(
    "table",
    [_permutation(), json.loads(DP8_TABLE.read_text())],
    ids=["small", "qwen3_5_dp8_ep8"],
)
def test_permutation_tables_parse_as_before(table):
    experts, shards, layers = _base_parse(table)
    placement = ep.parse(table)
    assert (placement.experts, placement.shards, placement.layers) == (
        experts,
        shards,
        layers,
    )
    assert placement.copies_per_rank == 0 and placement.copy_order == {}
    for layer, ranks in layers.items():
        order = tuple(e for held in ranks for e in held)
        assert placement.slot_order(layer) == order
        slot = [0] * experts
        for s, e in enumerate(order):
            slot[e] = s
        assert placement.slot_table(layer) == slot
    assert not placement.copies_per_rank


def test_a_primary_listed_twice_is_still_refused():
    table = _permutation()
    table["layers"]["3"][1][0] = 7  # expert 7 twice, expert 1 never
    with pytest.raises(ValueError, match="exactly once"):
        ep.parse(table)
    with pytest.raises(ValueError, match="exactly once"):
        ep.parse({**_copies(), "layers": {"3": [[7, 0, 5, 2, 1], [7, 6, 3, 4, 2]]}})


# ---- copy slots ----


def test_copy_slots_extend_the_primaries():
    placement = ep.parse(_copies())
    assert (placement.per_rank, placement.slots_per_rank) == (4, 5)
    # The router and every kernel but the few-token one see the primaries only.
    assert placement.slot_order(3) == ORDER
    assert placement.slot_table(3) == SLOT
    gid, prim, copy = placement.holder_tables(3, 0)
    assert gid == [0, 1, 2, 3, 4]  # rank 0's slots, the copy of expert 1 last
    assert prim == [0, 0, 0, 0, 1, 1, 1, 1]
    # Expert 7 (id 0) has its copy on rank 1, expert 1 (id 4) on rank 0.
    assert copy == [1, -1, -1, -1, 0, -1, -1, -1]
    assert placement.holder_tables(3, 1)[0] == [4, 5, 6, 7, 0]


def test_hot_order_follows_copy_order_else_rank_order():
    assert ep.parse(_copies()).hot_order(3) == [0, 4]
    assert ep.parse(_copies(copy_order=False)).hot_order(3) == [4, 0]


@pytest.mark.parametrize(
    "change,match",
    [
        ({"copies_per_rank": True}, "copies_per_rank"),
        ({"copies_per_rank": -1}, "copies_per_rank"),
        ({"copies_per_rank": "1"}, "copies_per_rank"),
        ({"layers": {"3": [[7, 0, 5, 2], [1, 6, 3, 4]]}}, "ranks of 5"),
        ({"layers": {"3": [[7, 0, 5, 2, 0], [1, 6, 3, 4, 7]]}}, "own primary"),
        ({"layers": {"3": [[7, 0, 5, 2, 8], [1, 6, 3, 4, 7]]}}, "outside"),
        ({"layers": {"3": [[7, 0, 5, 2, 1], [1, 6, 3, 4, 0]]}}, "copy_order"),
        ({"layers": {"3": [[7, 0, 5, 2, 6], [1, 6, 3, 4, 6]]}}, "own primary"),
        ({"copy_order": {"3": [7]}}, "exactly the layer"),
        ({"copy_order": {"4": [7, 1]}}, "does not place"),
        ({"copy_order": [7, 1]}, "must be an object"),
    ],
)
def test_malformed_copy_tables_are_refused(change, match):
    with pytest.raises(ValueError, match=match):
        ep.parse({**_copies(), **change})


def test_a_copied_expert_has_one_copy():
    """Three ranks of two primaries and one copy: expert 4 (rank 2's) may be
    copied once, not on both other ranks."""
    table = {
        "experts": 6,
        "shards": 3,
        "copies_per_rank": 1,
        "layers": {"3": [[0, 1, 4], [2, 3, 5], [4, 5, 0]]},
    }
    ep.parse(table)
    table["layers"]["3"][1][2] = 4
    with pytest.raises(ValueError, match="copies an expert twice"):
        ep.parse(table)


# ---- the Qwen3.5 tables ----


@pytest.mark.parametrize("path", [DP8_TABLE, TP8_TABLE], ids=["dp8_ep8", "tp8"])
def test_tables_hold_candidate4s_placements(path):
    table = json.loads(path.read_text())
    assert _canonical_digest(table) == CANDIDATE4_DIGESTS[path]
    ep.parse(table, str(path))


def test_the_tp8_table_copies_32_experts_per_layer():
    placement = ep.parse(json.loads(TP8_TABLE.read_text()))
    assert (placement.experts, placement.shards) == (512, 8)
    assert (placement.per_rank, placement.slots_per_rank) == (64, 68)
    assert sorted(placement.layers) == list(range(60))
    for layer in placement.layers:
        _, prim, copy = placement.holder_tables(layer, 0)
        copied = [v for v in range(512) if copy[v] >= 0]
        assert len(copied) == 32 == len(placement.hot_order(layer))
        assert all(copy[v] != prim[v] for v in copied)


# ---- sizing a copies table's layers before their weights exist ----


def _manager(rank, local=4):
    return SimpleNamespace(
        use_ep=True,
        ep_size=2,
        ep_rank=rank,
        global_num_experts=8,
        num_fused_shared_experts=0,
        _local_num_experts=local,
        _expert_map=torch.arange(8, dtype=torch.int32),
    )


@pytest.fixture
def vllm_moe(monkeypatch):
    """vLLM's factory and map manager, restored after the test."""
    from vllm.model_executor.layers import fused_moe as fused_pkg
    from vllm.model_executor.layers.fused_moe import expert_map_manager as emm
    from vllm.model_executor.layers.fused_moe import layer as fused_layer

    monkeypatch.setattr(fused_layer, "FusedMoEFactory", fused_layer.FusedMoEFactory)
    monkeypatch.setattr(fused_pkg, "FusedMoEFactory", fused_pkg.FusedMoEFactory)
    monkeypatch.setattr(
        emm.ExpertMapManager,
        "_calculate_expert_maps",
        emm.ExpertMapManager._calculate_expert_maps,
    )
    monkeypatch.setattr(fused_layer, "_tpu_copy_slot_hook", False, raising=False)
    model = types.ModuleType("_b3_fake_model")
    model.FusedMoEFactory = fused_layer.FusedMoEFactory
    monkeypatch.setitem(sys.modules, model.__name__, model)
    return SimpleNamespace(
        layer=fused_layer,
        pkg=fused_pkg,
        emm=emm,
        model=model,
        factory=fused_layer.FusedMoEFactory,
        calc=emm.ExpertMapManager._calculate_expert_maps,
    )


@pytest.mark.parametrize("table", [None, _permutation()], ids=["none", "permutation"])
def test_without_copies_the_hook_is_never_installed(
    tmp_path, monkeypatch, vllm_moe, table
):
    if table is None:
        monkeypatch.setenv("TPU_EXPERT_PLACEMENT", "")
    else:
        _use(tmp_path, monkeypatch, table)
    assert ep.install_copy_slot_hook() is False
    assert vllm_moe.layer.FusedMoEFactory is vllm_moe.factory
    assert vllm_moe.pkg.FusedMoEFactory is vllm_moe.factory
    assert vllm_moe.model.FusedMoEFactory is vllm_moe.factory
    assert vllm_moe.emm.ExpertMapManager._calculate_expert_maps is vllm_moe.calc


def test_a_copies_table_installs_the_hook_by_every_name(
    tmp_path, monkeypatch, vllm_moe
):
    _use(tmp_path, monkeypatch, _copies())
    assert ep.install_copy_slot_hook() is True
    wrapped = vllm_moe.layer.FusedMoEFactory
    assert wrapped is not vllm_moe.factory
    assert vllm_moe.pkg.FusedMoEFactory is wrapped
    assert vllm_moe.model.FusedMoEFactory is wrapped
    assert vllm_moe.emm.ExpertMapManager._calculate_expert_maps is not vllm_moe.calc
    assert ep.install_copy_slot_hook() is True  # once only
    assert vllm_moe.layer.FusedMoEFactory is wrapped


def test_a_placed_layer_is_sized_to_its_primaries_and_copies(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, _copies())
    manager = _manager(rank=1)
    ep._size_copy_slots(manager, LAYER)
    assert manager._local_num_experts == 5
    expected = [-1] * 8
    for slot, expert in enumerate([1, 6, 3, 4, 7]):
        expected[expert] = slot
    assert manager._expert_map.tolist() == expected


@pytest.mark.parametrize("rank", [0, 3, 7])
def test_a_tp8_layer_is_sized_to_68_slots_with_its_copies(monkeypatch, rank):
    """The TP8 table: 64 primaries and 4 copies per rank, so a placed layer gets
    68 local slots and its expert map places the rank's copies in slots
    64-67."""
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", str(TP8_TABLE))
    placement = ep.current()
    manager = SimpleNamespace(
        use_ep=True,
        ep_size=8,
        ep_rank=rank,
        global_num_experts=512,
        num_fused_shared_experts=0,
        _local_num_experts=64,
        _expert_map=torch.arange(512, dtype=torch.int32),
    )
    ep._size_copy_slots(manager, "language_model.model.layers.17.mlp.experts")
    assert manager._local_num_experts == 68
    held = placement.ranks(17)[rank]
    expert_map = manager._expert_map.tolist()
    assert [expert_map[e] for e in held] == list(range(68))
    assert [expert_map[e] for e in held[64:]] == [64, 65, 66, 67]
    assert sum(slot >= 0 for slot in expert_map) == 68


def test_other_layers_keep_vllms_map(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, _copies())
    for prefix in ("model.layers.2.mlp.experts", ""):
        manager = _manager(rank=1)
        ep._size_copy_slots(manager, prefix)
        assert manager._local_num_experts == 4
        assert manager._expert_map.tolist() == list(range(8))


class _Experts(torch.nn.Module):
    def __init__(self, rank, local):
        super().__init__()
        self.layer_name = LAYER
        self.global_num_experts = 8
        self.expert_map_manager = SimpleNamespace(
            global_num_experts=8,
            ep_size=2,
            ep_rank=rank,
            local_num_experts=local,
            num_fused_shared_experts=0,
            expert_mask=None,
            placement_strategy="linear",
            expert_map=None,
            _expert_map=None,
        )
        self.moe_config = SimpleNamespace(
            moe_parallel_config=SimpleNamespace(enable_eplb=False)
        )
        self.use_grouped_topk = False
        self.num_expert_group = None
        self.custom_routing_function = None

    def update_expert_map_info(self):
        pass


def _runner(rank, local):
    return SimpleNamespace(
        gate=torch.nn.Linear(3, 8), routed_experts=_Experts(rank, local)
    )


def test_apply_takes_a_sized_layers_primaries_and_copies(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, _copies())
    runner = _runner(rank=0, local=5)
    assert ep.apply(runner)
    layer = runner.routed_experts
    assert getattr(layer, ep.PRIMARY_EXPERTS_ATTR) == 4
    assert ep.expert_order(layer) == ORDER
    assert ep.fetch_filter_ids("layers.3.mlp.experts.0.w1") == frozenset(
        {7, 0, 5, 2, 1}
    )


def test_apply_refuses_a_copies_layer_the_hook_did_not_size(tmp_path, monkeypatch):
    """Without the hook the layer keeps the checkpoint's 4 slots per rank, and
    its weights have no room for the copy."""
    _use(tmp_path, monkeypatch, _copies())
    with pytest.raises(ValueError, match="holds 4 experts on rank 0, the placement 5"):
        ep.apply(_runner(rank=0, local=4))


def test_a_permutation_layer_gets_no_primary_count(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, _permutation())
    runner = _runner(rank=0, local=4)
    assert ep.apply(runner)
    assert not hasattr(runner.routed_experts, ep.PRIMARY_EXPERTS_ATTR)
