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
table checks, the maps, the router rows, the EP weight filter, a target layer
and an MTP draft layer built through the installed vLLM hooks, and GLM-5.2's
layer names and routing."""

import hashlib
import json
import logging
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import vllm.config
from vllm.model_executor.layers.fused_moe import expert_map_manager
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.model_loader import default_loader, runai_streamer_loader

import vllm_torchtpu
from vllm_torchtpu import expert_placement as ep
from vllm_torchtpu import model_loader_patches
from vllm_torchtpu.model_loader_patches import _should_skip

TARGET = "language_model.model.layers.0.mlp.experts"
DRAFT = "mtp.layers.0.mlp.experts"


def _plain_table():
    # Rank 0 holds experts 3, 0; rank 1 holds 2, 1.
    cores = [[3, 0], [2, 1]]
    return {
        "shards": 2,
        "experts": 4,
        "slots_per_core": 2,
        "layers": {"0": {"cores": cores}, "1": {"cores": cores}},
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
    assert (table.ranks, table.slots, table.experts) == (2, 2, 4)
    assert ep.virtual_order(table, 0) == [3, 0, 2, 1]
    assert ep.expert_map(table, 0, 0) == [1, -1, -1, 0]
    assert ep.expert_map(table, 0, 1) == [-1, 1, 0, -1]
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


def test_gs_uri_is_read_with_gcloud(monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout=json.dumps(_plain_table()).encode())

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", "gs://bucket/placement.json")
    ep._load.cache_clear()
    try:
        assert ep.placement().cores[0] == ((3, 0), (2, 1))
    finally:
        ep._load.cache_clear()
    assert calls == [["gcloud", "storage", "cat", "gs://bucket/placement.json"]]


def test_mtp_draft_layers_keep_linear_placement(use_table):
    use_table(_plain_table())
    w = torch.arange(4.0).reshape(4, 1)
    assert ep.layer_index(DRAFT) is None
    assert ep.layer_index(TARGET) == 0
    assert ep.permute_router_rows("mtp.layers.0.mlp.gate.weight", w) is w


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("duplicate", "exactly once"),
        ("out_of_range", "exactly once"),
        ("ranks", "must list 2 ranks"),
        ("slots", "must list 2 ranks"),
        ("experts", "must equal 6 experts"),
        ("copies", "copies are not supported"),
    ],
)
def test_invalid_tables_are_refused(use_table, case, match):
    table = _plain_table()
    layer = table["layers"]["0"]
    if case == "duplicate":
        layer["cores"] = [[3, 3], [2, 1]]
    elif case == "out_of_range":
        layer["cores"] = [[4, 0], [2, 1]]
    elif case == "ranks":
        layer["cores"] = [[3, 0, 2, 1]]
    elif case == "slots":
        layer["cores"] = [[3, 0, 2], [1]]
    elif case == "experts":
        table["experts"] = 6
    else:
        # Rank 0 also holds a copy of expert 2, rank 1 one of expert 3.
        table.update(slots_per_core=3, owned_per_core=2, copies=True)
        table["layers"] = {"0": {"cores": [[3, 0, 2], [2, 1, 3]]}}
    use_table(table)
    with pytest.raises(ValueError, match=match):
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

    def __init__(self, layer_name, rank, ep_size=2):
        super().__init__()
        parallel = SimpleNamespace(
            use_ep=True, ep_size=ep_size, ep_rank=rank, use_all2all_kernels=False
        )
        self.layer_name = layer_name
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


def test_target_is_placed_and_the_mtp_draft_stays_linear(use_table, installed):
    use_table(_plain_table())
    installed()
    table = ep.placement()
    target = _Experts(TARGET, rank=1)
    draft = _Experts(DRAFT, rank=1)
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


def test_unsupported_load_format_is_refused_at_construction(
    use_table, installed, monkeypatch
):
    use_table(_plain_table())
    installed()
    layer = _Experts(TARGET, rank=0)
    _load_format(monkeypatch, "dummy")
    with pytest.raises(NotImplementedError, match="load-format dummy"):
        RoutedExperts.update_expert_map_info(layer)


def test_table_for_another_ep_size_is_refused_at_construction(use_table, installed):
    # Built for 2 ranks; the layer runs on 4.
    use_table(_plain_table())
    installed()
    layer = _Experts(TARGET, rank=0, ep_size=4)
    with pytest.raises(ValueError, match="the table is for 2 ranks"):
        RoutedExperts.update_expert_map_info(layer)


# GLM-5.2 checkpoint names: of its 78 decoder layers, 0-2 are dense and 3-77
# route over 256 experts (sigmoid scores, top 8, a single expert group). The
# router is `.mlp.gate`, with its selection bias `.mlp.gate.e_score_correction_bias`.
# The MTP layer is stored as layer 78 under the same names, and vLLM builds its
# experts as `model.layers.78.mlp.experts`: no `mtp` module in either name.
GLM_GATE = "model.layers.{}.mlp.gate.weight"
GLM_BIAS = "model.layers.{}.mlp.gate.e_score_correction_bias"
GLM_MTP_EXPERTS = "model.layers.78.mlp.experts"
GLM_EXPERTS, GLM_RANKS = 256, 16
GLM_OWNED = GLM_EXPERTS // GLM_RANKS


def _glm_table(layers, seed=0):
    """A GLM-5.2-shaped table, 16 ranks x 16 experts: a random permutation per
    listed layer."""
    generator = torch.Generator().manual_seed(seed)
    entries = {}
    for layer in layers:
        perm = torch.randperm(GLM_EXPERTS, generator=generator).tolist()
        cores = [perm[r * GLM_OWNED : (r + 1) * GLM_OWNED] for r in range(GLM_RANKS)]
        entries[str(layer)] = {"cores": cores}
    return {
        "shards": GLM_RANKS,
        "experts": GLM_EXPERTS,
        "slots_per_core": GLM_OWNED,
        "owned_per_core": GLM_OWNED,
        "layers": entries,
    }


def test_glm52_router_rows_follow_the_table(use_table):
    use_table(_glm_table(range(3, 78)))
    table = ep.placement()
    rows = torch.arange(float(GLM_EXPERTS))
    order = ep.virtual_order(table, 40)
    for name in (GLM_GATE, GLM_BIAS):
        placed = ep.permute_router_rows(name.format(40), rows)
        assert placed.tolist() == [float(e) for e in order]
        assert ep.permute_router_rows(name.format(78), rows) is rows
    # Router row v is held by rank v // 16, in local slot v % 16.
    for row, expert in enumerate(order):
        rank, slot = divmod(row, GLM_OWNED)
        assert ep.expert_map(table, 40, rank)[expert] == slot
    for name in (
        "model.layers.1.mlp.gate_proj.weight",
        "model.layers.40.mlp.shared_experts.gate_proj.weight",
        "model.layers.40.mlp.experts.3.gate_proj.weight",
        "model.layers.40.mlp.experts.3.gate_proj.weight_scale_inv",
    ):
        assert ep.permute_router_rows(name, rows) is rows
    assert ep.layer_index("model.layers.40.mlp.experts") == 40
    assert ep.layer_index("model.layers.1.mlp") is None


@pytest.mark.parametrize("listed", [False, True])
def test_glm52_mtp_layer_is_placed_only_when_listed(use_table, listed):
    """By name, the MTP layer's router rows and its experts resolve to the same
    layer; the deployment check refuses a table that lists it
    (test_glm52_table_listing_mtp_or_dense_layers_is_refused)."""
    use_table(_glm_table([40, 78] if listed else [40]))
    rows = torch.arange(float(GLM_EXPERTS))
    assert ep.layer_index(GLM_MTP_EXPERTS) == (78 if listed else None)
    for name in (GLM_GATE, GLM_BIAS):
        assert ep.layer_index(name.format(78)) == (78 if listed else None)
        moved = ep.permute_router_rows(name.format(78), rows) is not rows
        assert moved == listed


def test_glm52_weight_streams_place_the_router(use_table, installed, monkeypatch):
    use_table(_glm_table([40]))
    order = [float(e) for e in ep.virtual_order(ep.placement(), 40)]
    rows = torch.arange(float(GLM_EXPERTS))
    scale_name = "model.layers.40.mlp.experts.7.down_proj.weight_scale_inv"
    scale = torch.ones(2)
    weights = [
        (GLM_GATE.format(40), rows),
        (GLM_BIAS.format(40), rows),
        (GLM_GATE.format(78), rows),
        (scale_name, scale),
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
        assert got[GLM_GATE.format(40)].tolist() == order
        assert got[GLM_BIAS.format(40)].tolist() == order
        assert got[GLM_GATE.format(78)] is rows
        assert got[scale_name] is scale


def _glm_moe(x, gate, bias, expert_scale, top_k=8, scaling=2.5):
    """GLM-5.2 routing with scalar experts: the bias picks the experts, the
    sigmoid scores weigh them, renormalized and scaled."""
    scores = (x @ gate.T).sigmoid()
    ids = (scores + bias).topk(top_k, dim=-1).indices
    weights = scores.gather(-1, ids)
    weights = weights / weights.sum(-1, keepdim=True) * scaling
    return (weights * expert_scale[ids]).sum(-1, keepdim=True) * x


def _glm_placed_experts(table, layer, per_expert):
    """The ranks' local slots, rank-major, filled through the expert map as the
    EP weight loader fills them; the grouped path reads row v on rank v // 16."""
    placed = torch.full_like(per_expert, float("nan"))
    for rank in range(table.ranks):
        for expert, slot in enumerate(ep.expert_map(table, layer, rank)):
            if slot >= 0:
                placed[rank * table.slots + slot] = per_expert[expert]
    return placed


def test_glm52_placed_layer_matches_the_unplaced_layer(use_table):
    """Placing the router weight, its bias and the experts keeps a GLM-5.2
    layer's output; leaving any one of them in checkpoint order does not."""
    use_table(_glm_table([40]))
    table = ep.placement()
    torch.manual_seed(0)
    x = torch.randn(32, 8, dtype=torch.float64)
    gate = torch.randn(GLM_EXPERTS, 8, dtype=torch.float64)
    # Large next to the (0, 1) scores, so the bias changes which experts win.
    bias = torch.randn(GLM_EXPERTS, dtype=torch.float64)
    scale = torch.arange(1.0, GLM_EXPERTS + 1, dtype=torch.float64)
    ref = _glm_moe(x, gate, bias, scale)

    placed = {
        "gate": ep.permute_router_rows(GLM_GATE.format(40), gate),
        "bias": ep.permute_router_rows(GLM_BIAS.format(40), bias),
        "expert_scale": _glm_placed_experts(table, 40, scale),
    }
    torch.testing.assert_close(_glm_moe(x, **placed), ref)
    # Negative controls: one piece left in checkpoint order.
    for piece, unplaced in (("gate", gate), ("bias", bias), ("expert_scale", scale)):
        assert not torch.allclose(_glm_moe(x, **{**placed, piece: unplaced}), ref)


# GLM-5.2 deployment checks at EP16: the EP weight filter over 16 ranks, the
# once-per-worker "[eplb] placement" lines, and the refusals the first MoE
# layer raises (dense or MTP layers listed, EP x TP splitting, a placement
# strategy other than linear).
GLM_PLACED = (3, 40, 77)
GLM_UNPLACED = 10  # a routed layer the table does not list
GLM_MTP = 78
GLM_CKPT = "model.layers.{}.mlp.experts.{}.{}.{}"


class _GlmExperts(torch.nn.Module):
    """The RoutedExperts state update_expert_map_info reads, GLM-5.2 at EP16."""

    def __init__(self, layer_name, rank):
        super().__init__()
        parallel = SimpleNamespace(
            use_ep=True, ep_size=GLM_RANKS, ep_rank=rank, use_all2all_kernels=False
        )
        self.layer_name = layer_name
        self.expert_map_manager = expert_map_manager.ExpertMapManager(
            max_num_batched_tokens=8,
            top_k=8,
            global_num_experts=GLM_EXPERTS,
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


def _glm_config(monkeypatch, strategy="linear", **hf_overrides):
    hf = SimpleNamespace(
        **{
            "num_hidden_layers": 78,
            "first_k_dense_replace": 3,
            "n_group": 1,
            "topk_group": 1,
            **hf_overrides,
        }
    )
    config = SimpleNamespace(
        load_config=SimpleNamespace(load_format="auto"),
        parallel_config=SimpleNamespace(expert_placement_strategy=strategy),
        model_config=SimpleNamespace(hf_config=hf, hf_text_config=hf),
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: config)


@pytest.fixture
def worker(monkeypatch):
    """Returns a reset to a new worker process's placement state."""
    monkeypatch.setattr(ep, "_STATE", {})
    monkeypatch.setenv("TPU_MOE_EP_TP_SIZE", "1")

    def reset():
        ep._STATE.clear()
        ep._LOCAL_IDS.clear()

    return reset


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture
def eplb_lines():
    handler = _Lines()
    log = logging.getLogger("vllm.vllm_torchtpu.expert_placement")
    level = log.level
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    yield handler.lines
    log.removeHandler(handler)
    log.setLevel(level)


def _build_glm_worker(rank, layers=(*GLM_PLACED, GLM_UNPLACED)):
    """One worker's model construction: the target's MoE layers, then the MTP
    draft's layer 78."""
    built = {}
    for layer in (*layers, GLM_MTP):
        experts = _GlmExperts(f"model.layers.{layer}.mlp.experts", rank)
        RoutedExperts.update_expert_map_info(experts)
        built[layer] = experts
    return built


def _safetensors(path, names):
    header, offset = {}, 0
    for name in names:
        header[name] = {
            "dtype": "F8_E4M3",
            "shape": [4],
            "data_offsets": [offset, offset + 4],
        }
        offset += 4
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(offset))
    return 8 + len(blob)


def test_glm52_weight_filter_loads_the_table_experts_on_16_ranks(
    use_table, installed, worker, monkeypatch, tmp_path
):
    use_table(_glm_table(GLM_PLACED, seed=7))
    _glm_config(monkeypatch)
    installed()
    table = ep.placement()
    layers = (*GLM_PLACED, GLM_UNPLACED, GLM_MTP)
    names = [
        GLM_CKPT.format(layer, e, proj, suffix)
        for layer in layers
        for e in range(GLM_EXPERTS)
        for proj in ("gate_proj", "up_proj", "down_proj")
        for suffix in ("weight", "weight_scale_inv")
    ]
    path = tmp_path / "model-00001.safetensors"
    header_bytes = _safetensors(path, names)
    owners = {layer: [[] for _ in range(GLM_EXPERTS)] for layer in layers}
    for rank in range(GLM_RANKS):
        worker()
        built = _build_glm_worker(rank)
        linear = set(range(GLM_OWNED * rank, GLM_OWNED * (rank + 1)))
        kept = {layer: set() for layer in layers}
        scales_kept = {layer: set() for layer in layers}
        for name in names:
            # The default loader's predicate is the prefetch's and the Run:AI
            # sharded fetch plan's.
            skip = model_loader_patches._should_skip_weight_tpu(name, linear)
            assert skip == _should_skip(name, linear)
            if skip:
                continue
            layer, expert = int(name.split(".")[2]), int(name.split(".")[5])
            (kept if name.endswith(".weight") else scales_kept)[layer].add(expert)
        for layer in GLM_PLACED:
            assert kept[layer] == set(table.cores[layer][rank]), (rank, layer)
            local = built[layer]._expert_map.tolist()
            assert [local[e] for e in table.cores[layer][rank]] == list(
                range(GLM_OWNED)
            )
            assert sum(v >= 0 for v in local) == GLM_OWNED
        for layer in (GLM_UNPLACED, GLM_MTP):
            assert kept[layer] == linear, (rank, layer)
            local = built[layer]._expert_map.tolist()
            assert [e for e in range(GLM_EXPERTS) if local[e] >= 0] == sorted(linear)
        for layer in layers:
            for expert in kept[layer]:
                owners[layer][expert].append(rank)
            # Block scales are not EP-filtered: every rank reads them all and
            # the expert map drops the other ranks' (placed or not).
            assert scales_kept[layer] == set(range(GLM_EXPERTS))
        # The prefetch byte ranges cover exactly the kept tensors.
        ranges = model_loader_patches._local_tensor_ranges(str(path), linear)
        kept_bytes = 4 * sum(1 for n in names if not _should_skip(n, linear))
        assert sum(hi - lo for lo, hi in ranges) == kept_bytes
        assert ranges[0][0] >= header_bytes
    # Over the 16 ranks every expert of every layer is loaded exactly once.
    for layer, ranks in owners.items():
        assert all(len(r) == 1 for r in ranks), layer


def _glm_router_stream(monkeypatch):
    weights = [
        (GLM_GATE.format(3), torch.arange(float(GLM_EXPERTS))),
        (GLM_BIAS.format(3), torch.arange(float(GLM_EXPERTS))),
    ]
    loader = default_loader.DefaultModelLoader
    streamer = runai_streamer_loader.RunaiModelStreamerLoader
    monkeypatch.setattr(loader, "get_all_weights", lambda *a: iter(weights))
    monkeypatch.setattr(streamer, "_get_weights_iterator", lambda *a: iter(weights))


def test_glm52_installed_line_once_per_worker_on_16_ranks(
    use_table, installed, worker, monkeypatch, eplb_lines
):
    path = use_table(_glm_table(GLM_PLACED, seed=3))
    with open(path, "rb") as f:
        sha16 = hashlib.sha256(f.read()).hexdigest()[:16]
    _glm_config(monkeypatch)
    _glm_router_stream(monkeypatch)
    installed()
    loader = default_loader.DefaultModelLoader
    for rank in range(GLM_RANKS):
        worker()
        _build_glm_worker(rank)
        dict(loader.get_all_weights(None, None, None))  # the target's load
        dict(loader.get_all_weights(None, None, None))  # the MTP draft's load
    assert [m for m in eplb_lines if m.startswith("[eplb]")] == [
        f"[eplb] placement installed ep_rank={r} layers={len(GLM_PLACED)} "
        f"table_sha256={sha16}"
        for r in range(GLM_RANKS)
    ]


def test_glm52_none_line_once_per_worker_without_a_table(
    installed, worker, monkeypatch, eplb_lines
):
    monkeypatch.delenv("TPU_EXPERT_PLACEMENT", raising=False)
    ep._load.cache_clear()
    monkeypatch.setattr(
        RoutedExperts, "_tpu_expert_placement_none_marker", False, raising=False
    )
    _glm_config(monkeypatch)
    installed()
    for rank in range(GLM_RANKS):
        worker()
        for experts in _build_glm_worker(rank).values():
            local = experts._expert_map.tolist()
            assert [e for e in range(GLM_EXPERTS) if local[e] >= 0] == list(
                range(GLM_OWNED * rank, GLM_OWNED * (rank + 1))
            )
    assert [m for m in eplb_lines if m.startswith("[eplb]")] == [
        f"[eplb] placement none ep_rank={r}" for r in range(GLM_RANKS)
    ]
    assert not ep._LOCAL_IDS


@pytest.mark.parametrize(
    ("layers", "match"),
    [
        ((3, 40, 78), r"lists layers \[78\]"),
        ((2, 3), r"lists layers \[2\]"),
        ((0, 77), r"lists layers \[0\]"),
    ],
)
def test_glm52_table_listing_mtp_or_dense_layers_is_refused(
    use_table, installed, worker, monkeypatch, layers, match
):
    use_table(_glm_table(layers))
    _glm_config(monkeypatch)
    installed()
    with pytest.raises(ValueError, match=match):
        RoutedExperts.update_expert_map_info(
            _GlmExperts("model.layers.3.mlp.experts", 0)
        )


@pytest.mark.parametrize(
    "env", [("TPU_MOE_EP_TP_SIZE", "4"), ("TPU_MOE_HIERARCHICAL_EP", "1")]
)
def test_glm52_ep_tp_split_is_refused(use_table, installed, worker, monkeypatch, env):
    use_table(_glm_table(GLM_PLACED))
    _glm_config(monkeypatch)
    installed()
    monkeypatch.setenv(*env)
    with pytest.raises(ValueError, match="TPU_MOE_EP_TP_SIZE=1"):
        RoutedExperts.update_expert_map_info(
            _GlmExperts("model.layers.3.mlp.experts", 0)
        )


def test_glm52_round_robin_placement_strategy_is_refused(
    use_table, installed, worker, monkeypatch
):
    use_table(_glm_table(GLM_PLACED))
    _glm_config(monkeypatch, strategy="round_robin")
    installed()
    with pytest.raises(ValueError, match="'round_robin'"):
        RoutedExperts.update_expert_map_info(
            _GlmExperts("model.layers.3.mlp.experts", 0)
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        ("duplicate", "exactly once"),
        ("out_of_range", "exactly once"),
        ("15_ranks", "must list 16 ranks"),
        ("uneven_slots", "must list 16 ranks"),
        ("255_experts", "must equal"),
    ],
)
def test_glm52_bad_tables_are_refused_by_the_loader_and_the_validator(
    use_table, mutate, match
):
    table = _glm_table(GLM_PLACED)
    cores = table["layers"]["40"]["cores"]
    if mutate == "duplicate":
        cores[1][0] = cores[0][0]
    elif mutate == "out_of_range":
        cores[1][0] = GLM_EXPERTS
    elif mutate == "15_ranks":
        table["layers"]["40"]["cores"] = cores[:15]
    elif mutate == "uneven_slots":
        cores[0].append(cores[1].pop())
    else:
        table["experts"] = GLM_EXPERTS - 1
    use_table(table)
    with pytest.raises(ValueError, match=match):
        ep.placement()
    with pytest.raises(ValueError, match=match):
        ep.validate(
            json.dumps(table).encode(),
            "t",
            experts=GLM_EXPERTS,
            ranks=GLM_RANKS,
            slots=GLM_OWNED,
            first_moe_layer=3,
            num_layers=78,
        )


def test_glm52_validator_refuses_other_shapes_and_layers():
    deployment = {
        "experts": GLM_EXPERTS,
        "ranks": GLM_RANKS,
        "slots": GLM_OWNED,
        "first_moe_layer": 3,
        "num_layers": 78,
    }
    eight = {
        "shards": 8,
        "experts": GLM_EXPERTS,
        "slots_per_core": 32,
        "layers": {
            "3": {"cores": [list(range(32 * r, 32 * (r + 1))) for r in range(8)]}
        },
    }
    with pytest.raises(ValueError, match="needs 16 x 16 over 256"):
        ep.validate(json.dumps(eight).encode(), "t", **deployment)
    for bad in (78, 2):
        with pytest.raises(ValueError, match=rf"lists layers \[{bad}\]"):
            ep.validate(json.dumps(_glm_table((3, bad))).encode(), "t", **deployment)
    full = ep.validate(json.dumps(_glm_table(range(3, 78))).encode(), "t", **deployment)
    assert sorted(full.cores) == list(range(3, 78))
    assert len(full.sha256) == 64


def test_glm52_dense_and_mtp_layers_are_untouched_by_a_full_table(
    use_table, installed, worker, monkeypatch
):
    use_table(_glm_table(range(3, 78), seed=11))
    _glm_config(monkeypatch)
    installed()
    table = ep.placement()
    rows = torch.arange(float(GLM_EXPERTS))
    h = torch.zeros(4)
    for layer in (0, 1, 2):
        for proj in (
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "self_attn.q_a_proj",
        ):
            name = f"model.layers.{layer}.{proj}.weight"
            assert ep.permute_router_rows(name, h) is h
            assert ep.local_expert_ids(name) is None
            assert not _should_skip(name, set(range(GLM_OWNED)))
    for name in (GLM_GATE.format(78), GLM_BIAS.format(78)):
        assert ep.permute_router_rows(name, rows) is rows
    assert ep.layer_index("model.layers.78.mtp_block.mlp.experts") is None
    for proj in ("gate_proj", "down_proj"):
        name = f"model.layers.40.mlp.shared_experts.{proj}.weight"
        assert ep.permute_router_rows(name, h) is h
    for layer in range(3, 78):
        order = [float(e) for e in ep.virtual_order(table, layer)]
        for name in (GLM_GATE.format(layer), GLM_BIAS.format(layer)):
            assert ep.permute_router_rows(name, rows).tolist() == order
    # The MTP draft's experts keep the linear map on a worker.
    worker()
    local = _build_glm_worker(9, layers=(3,))[GLM_MTP]._expert_map.tolist()
    assert [e for e in range(GLM_EXPERTS) if local[e] >= 0] == list(range(144, 160))


@pytest.mark.parametrize(
    ("hf", "match"),
    [
        ({"n_group": 8, "topk_group": 4}, "grouped top-k"),
        ({"num_hash_layers": 4}, r"hash-routed layers \[3\]"),
    ],
)
def test_id_dependent_routing_is_refused(
    use_table, installed, worker, monkeypatch, hf, match
):
    use_table(_glm_table(GLM_PLACED))
    _glm_config(monkeypatch, **hf)
    installed()
    with pytest.raises(ValueError, match=match):
        RoutedExperts.update_expert_map_info(
            _GlmExperts("model.layers.3.mlp.experts", 0)
        )


@pytest.mark.parametrize("name", ["glm52_ep16_c32_4k.json", "glm52_ep16_c32_8k.json"])
def test_glm52_example_tables_validate(name):
    path = Path(__file__).parents[2] / "examples" / "expert_placement" / name
    raw = path.read_bytes()
    table = ep.validate(
        raw,
        name,
        experts=GLM_EXPERTS,
        ranks=GLM_RANKS,
        slots=GLM_OWNED,
        first_moe_layer=3,
        num_layers=78,
    )
    assert sorted(table.cores) == list(range(3, 78))
    assert table.sha256 == hashlib.sha256(raw).hexdigest()
