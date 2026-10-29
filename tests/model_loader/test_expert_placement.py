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

import functools
import hashlib
import json
import logging
import subprocess
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

TARGET = "language_model.model.layers.0.mlp.experts"
DRAFT = "mtp.layers.0.mlp.experts"


def _plain_table():
    cores = [[3, 0], [2, 1]]
    return {"shards": 2, "experts": 4, "layers": {"0": cores, "1": cores}}


@pytest.fixture
def use_table(tmp_path, monkeypatch):
    def use(table):
        path = tmp_path / "placement.json"
        path.write_text(json.dumps(table))
        monkeypatch.setenv("TPU_EXPERT_PLACEMENT", str(path))
        ep._load.cache_clear()
        return str(path)

    monkeypatch.setattr(ep, "_LOCAL_IDS", {})
    monkeypatch.setattr(ep, "_ROUTED", set())
    yield use
    ep._load.cache_clear()


def _mark_placed(rank=0):
    """Record every table layer as placed on `rank`, as building its experts
    does, so the router rows of those layers are reordered."""
    table = ep.placement()
    for layer, cores in table.cores.items():
        ep._LOCAL_IDS[layer] = frozenset(cores[rank])
    return table


def test_maps_and_router_rows(use_table):
    use_table(_plain_table())
    table = ep.placement()
    assert (table.ranks, table.slots, table.experts) == (2, 2, 4)
    assert ep.virtual_order(table, 0) == [3, 0, 2, 1]
    assert ep.expert_map(table, 0, 0) == [1, -1, -1, 0]
    assert ep.expert_map(table, 0, 1) == [-1, 1, 0, -1]
    w = torch.arange(4.0).reshape(4, 1)
    # Experts not placed by this process: the router keeps its order too.
    assert ep.permute_router_rows("model.layers.0.mlp.gate.weight", w) is w
    _mark_placed()
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


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("listed_twice", "layer 0 is listed twice"),
    ],
)
def test_invalid_tables_are_refused(use_table, case, match):
    table = _plain_table()
    table["layers"]["00"] = table["layers"]["0"]
    use_table(table)
    with pytest.raises(ValueError, match=match):
        ep.placement()


def _copies_table():
    return {
        "shards": 2,
        "experts": 4,
        "layers": {"0": [[3, 0, 2], [2, 1, 0]]},
        "copy_share": {"0": {"2": 0.5, "0": 0.5}},
    }


def test_copies_table_maps_every_slot(use_table, monkeypatch):
    use_table(_copies_table())
    table = ep.placement()
    assert (table.ranks, table.owned, table.slots, table.experts) == (2, 2, 3, 4)
    # Router rows follow the primaries only; each rank maps its copy too.
    assert ep.virtual_order(table, 0) == [3, 0, 2, 1]
    assert ep.expert_map(table, 0, 0) == [1, -1, 2, 0]
    assert ep.expert_map(table, 0, 1) == [2, 1, 0, -1]
    # Only a layer this worker placed reports its primaries.
    assert ep.num_primary_experts(TARGET) is None
    monkeypatch.setattr(ep, "_LOCAL_IDS", {0: frozenset(table.cores[0][0])})
    monkeypatch.setattr(ep, "_ROUTED", set())
    out = ep.permute_router_rows("model.layers.0.mlp.gate.weight", torch.arange(4.0))
    assert out.tolist() == [3.0, 0.0, 2.0, 1.0]
    assert ep.num_primary_experts(TARGET) == 2
    assert ep.num_primary_experts(DRAFT) is None
    assert ep.num_primary_experts("model.layers.1.mlp.experts") is None
    use_table(_plain_table())
    assert ep.num_primary_experts(TARGET) is None


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("duplicate_primary", "exactly once among the primaries"),
        ("invalid_copy", "invalid copy id"),
        ("duplicate_copy", "more than one copy slot"),
        ("own_copy", "on the rank that owns the expert"),
        ("uneven", "must list 2 ranks of 3 experts each"),
        ("no_share", "layer 0 has copy slots but no copy_share"),
        ("share_layer", r"copy_share lists layers \[5\]"),
        ("share_experts", "copy_share must list exactly its copied experts"),
        ("share_range", r"is 1.5, outside \[0, 1\]"),
        ("min_tokens", "copy_min_tokens must be a non-negative integer"),
    ],
)
def test_invalid_copies_tables_are_refused(use_table, case, match):
    table = _copies_table()
    layer = table["layers"]["0"]
    if case == "duplicate_primary":
        layer[0] = [3, 3, 2]
    elif case == "invalid_copy":
        layer[0][-1] = 4
    elif case == "duplicate_copy":
        layer[1][-1] = 2
    elif case == "own_copy":
        layer[0][-1] = 3
    elif case == "uneven":
        layer[1].append(3)
    elif case == "no_share":
        del table["copy_share"]
    elif case == "share_layer":
        table["copy_share"]["5"] = {"2": 0.5}
    elif case == "share_experts":
        table["copy_share"]["0"] = {"2": 0.5}
    elif case == "share_range":
        table["copy_share"]["0"] = {"2": 0.5, "0": 1.5}
    else:
        table["copy_min_tokens"] = True
    use_table(table)
    with pytest.raises(ValueError, match=match):
        ep.placement()


def test_copies_table_warns_when_a_kernel_would_decline_it(
    use_table, monkeypatch, eplb_lines
):
    monkeypatch.setattr(envs, "USE_MOE_TP_DECODE_KERNEL", True, raising=False)
    use_table(_copies_table())
    ep.placement()
    assert "layers with copies run on the grouped-matmul path" in eplb_lines[-1]


class _Experts(torch.nn.Module):
    """The RoutedExperts state update_expert_map_info reads."""

    def __init__(self, layer_name, rank, ep_size=2, experts=4):
        super().__init__()
        parallel = SimpleNamespace(
            use_ep=True, ep_size=ep_size, ep_rank=rank, use_all2all_kernels=False
        )
        self.layer_name = layer_name
        self.expert_map_manager = expert_map_manager.ExpertMapManager(
            max_num_batched_tokens=8,
            top_k=2,
            global_num_experts=experts,
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
        RoutedExperts, "_get_quant_method", RoutedExperts._get_quant_method
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


def test_fp8_weight_loader_fills_the_copy_slot(use_table, installed, monkeypatch):
    """vLLM's RoutedExperts.weight_loader, with the FP8 MoE method's
    block-scaled parameters, loads a copied expert into its copy slot on the
    copy rank: the weights and both block scales of slot `owned`."""
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        MoEActivation,
        RoutingMethodType,
    )

    from vllm_torchtpu.layers.adapter.quantization.fp8 import (
        VllmFp8Config,
        VllmFp8MoEMethodTPU,
    )

    use_table(_copies_table())
    installed()
    config = SimpleNamespace(
        load_config=SimpleNamespace(load_format="auto"),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True, expert_placement_strategy="linear"
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(expert_dtype="fp8")),
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    hidden, inter, rank = 256, 256, 1
    parallel = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=2,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=rank,
        sp_size=1,
        use_ep=True,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    manager = expert_map_manager.ExpertMapManager(
        max_num_batched_tokens=16,
        top_k=2,
        global_num_experts=4,
        num_redundant_experts=0,
        num_expert_group=None,
        moe_parallel_config=parallel,
        placement_strategy="linear",
        enable_eplb=False,
    )
    moe_config = FusedMoEConfig(
        num_experts=4,
        experts_per_token=2,
        hidden_dim=hidden,
        intermediate_size=inter,
        num_local_experts=manager.local_num_experts,
        num_logical_experts=4,
        activation=MoEActivation.SILU,
        device=torch.device("cpu"),
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
    )
    quant = VllmFp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )
    quant.vllm_config = config
    with set_current_vllm_config(config):
        layer = RoutedExperts(
            layer_name=TARGET,
            params_dtype=torch.bfloat16,
            moe_config=moe_config,
            quant_config=quant,
            expert_map_manager=manager,
        )
    assert isinstance(layer.quant_method, VllmFp8MoEMethodTPU)
    slots = 3
    assert layer.w13_weight.shape == (slots, 2 * inter, hidden)
    assert layer.w13_weight_scale_inv.shape == (slots, 2 * inter // 128, 2)
    params = {
        "w13": (layer.w13_weight, layer.w13_weight_scale_inv),
        "w2": (layer.w2_weight, layer.w2_weight_scale_inv),
    }
    for weight, scale in params.values():
        weight.data.fill_(7.0)
        scale.data.fill_(-1.0)
    gen = torch.Generator().manual_seed(0)
    data = {
        "w1": torch.full((inter, hidden), 2.0).to(torch.float8_e4m3fn),
        "w3": torch.full((inter, hidden), 3.0).to(torch.float8_e4m3fn),
        "w2": torch.full((hidden, inter), 4.0).to(torch.float8_e4m3fn),
    }
    scales = {sid: torch.rand(2, 2, generator=gen) + 0.5 for sid in data}
    expert = 0  # copied to rank 1's slot 2; owned by rank 0
    for sid in ("w1", "w3", "w2"):
        weight, scale = params["w2" if sid == "w2" else "w13"]
        layer.weight_loader(weight, data[sid], f"{sid}.weight", sid, expert)
        layer.weight_loader(scale, scales[sid], f"{sid}.weight_scale_inv", sid, expert)

    w13, s13 = (t.data.float() for t in params["w13"])
    w2, s2 = (t.data.float() for t in params["w2"])
    primaries = slice(0, 2)
    # The primaries' slots are untouched.
    assert torch.all(w13[primaries] == 7.0) and torch.all(w2[primaries] == 7.0)
    assert torch.all(s13[primaries] == -1.0) and torch.all(s2[primaries] == -1.0)
    owned = 2
    torch.testing.assert_close(w13[owned, :inter], data["w1"].float())
    torch.testing.assert_close(w13[owned, inter:], data["w3"].float())
    torch.testing.assert_close(w2[owned], data["w2"].float())
    torch.testing.assert_close(s13[owned, :2], scales["w1"])
    torch.testing.assert_close(s13[owned, 2:], scales["w3"])
    torch.testing.assert_close(s2[owned], scales["w2"])


def test_copies_need_the_fp8_moe_method(use_table, installed, monkeypatch):
    """RoutedExperts picks its quant method right after the expert map; a
    placed layer with copy slots must get the FP8 MoE method."""
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU

    fp8 = VllmFp8MoEMethodTPU.__new__(VllmFp8MoEMethodTPU)
    other = SimpleNamespace()
    picked = []
    monkeypatch.setattr(RoutedExperts, "_get_quant_method", lambda self, *a: picked[-1])
    use_table(_copies_table())
    installed()
    layer = _Experts(TARGET, rank=0)
    RoutedExperts.update_expert_map_info(layer)
    picked.append(fp8)
    assert RoutedExperts._get_quant_method(layer, TARGET, None, None) is fp8
    picked.append(other)
    with pytest.raises(NotImplementedError, match="only the FP8 MoE method"):
        RoutedExperts._get_quant_method(layer, TARGET, None, None)
    # The MTP draft is not placed, so any method serves it.
    draft = _Experts(DRAFT, rank=0)
    RoutedExperts.update_expert_map_info(draft)
    assert RoutedExperts._get_quant_method(draft, DRAFT, None, None) is other


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
        entries[str(layer)] = cores
    return {"shards": GLM_RANKS, "experts": GLM_EXPERTS, "layers": entries}


def test_glm52_weight_streams_place_the_router(use_table, installed, monkeypatch):
    use_table(_glm_table([40]))
    _mark_placed()
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


# GLM-5.2 deployment checks at EP16: the EP weight filter over 16 ranks, the
# once-per-worker "[eplb] placement" lines, and the refusals the first MoE
# layer raises (dense or MTP layers listed, EP x TP splitting, a placement
# strategy other than linear).
GLM_PLACED = (3, 40, 77)
GLM_UNPLACED = 10  # a routed layer the table does not list
GLM_MTP = 78
GLM_CKPT = "model.layers.{}.mlp.experts.{}.{}.{}"


_GlmExperts = functools.partial(_Experts, ep_size=GLM_RANKS, experts=GLM_EXPERTS)


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
    # Flat EP, so the EP x TP refusal fires only in the test that sets it.
    monkeypatch.setenv("TPU_MOE_EP_TP_SIZE", "1")

    def reset():
        ep._STATE.clear()
        ep._LOCAL_IDS.clear()
        ep._ROUTED.clear()

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


def test_glm52_weight_filter_loads_the_table_experts_on_16_ranks(
    use_table, installed, worker, monkeypatch
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
    owners = {layer: [[] for _ in range(GLM_EXPERTS)] for layer in layers}
    for rank in range(GLM_RANKS):
        worker()
        built = _build_glm_worker(rank)
        linear = set(range(GLM_OWNED * rank, GLM_OWNED * (rank + 1)))
        kept = {layer: set() for layer in layers}
        scales_kept = {layer: set() for layer in layers}
        for name in names:
            if _should_skip(name, linear):
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
    for layer, ranks in owners.items():
        assert all(len(r) == 1 for r in ranks), layer


def _glm_router_stream(monkeypatch, layers=GLM_PLACED):
    weights = [
        (name.format(layer), torch.arange(float(GLM_EXPERTS)))
        for layer in layers
        for name in (GLM_GATE, GLM_BIAS)
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


@pytest.mark.parametrize(
    ("layers", "match"),
    [
        ((3, 40, 78), r"lists layers \[78\]"),
        ((2, 3), r"lists layers \[2\]"),
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
        ("255_experts", "do not divide over 16 ranks"),
    ],
)
def test_glm52_bad_tables_are_refused(use_table, mutate, match):
    table = _glm_table(GLM_PLACED)
    cores = table["layers"]["40"]
    if mutate == "duplicate":
        cores[1][0] = cores[0][0]
    elif mutate == "out_of_range":
        cores[1][0] = GLM_EXPERTS
    elif mutate == "15_ranks":
        table["layers"]["40"] = cores[:15]
    elif mutate == "uneven_slots":
        cores[0].append(cores[1].pop())
    else:
        table["experts"] = GLM_EXPERTS - 1
    use_table(table)
    with pytest.raises(ValueError, match=match):
        ep.placement()


def test_glm52_dense_and_mtp_layers_are_untouched_by_a_full_table(
    use_table, installed, worker, monkeypatch
):
    use_table(_glm_table(range(3, 78), seed=11))
    _glm_config(monkeypatch)
    installed()
    table = _mark_placed()
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
    # Router row v is held by rank v // 16, in local slot v % 16.
    for row, expert in enumerate(ep.virtual_order(table, 40)):
        rank, slot = divmod(row, GLM_OWNED)
        assert ep.expert_map(table, 40, rank)[expert] == slot


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


# sha256 of "<layer>:<ids>;..." over the sorted layers (each layer's ids rank by
# rank) of the shipped tables, taken from their earlier {"cores": ...} files.
SHIPPED_SLOT_ORDERS = {
    "glm52_ep16_c32_4k": (
        "3e7c020ba503b7a77dd9cbd6b101b353cc77c2b8b2a264e83e9966d9ee3e5d82"
    ),
    "glm52_ep16_c32_8k": (
        "320cdc18136f6c80ec6bb7cdc8fae040ef4f604e00024e63e64ea96d29d341e1"
    ),
}


@pytest.mark.parametrize("name", sorted(SHIPPED_SLOT_ORDERS))
def test_glm52_shipped_tables_load_by_name(monkeypatch, name):
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", name)
    ep._load.cache_clear()
    try:
        table = ep.placement()
    finally:
        ep._load.cache_clear()
    assert (table.experts, table.ranks) == (GLM_EXPERTS, GLM_RANKS)
    ep.check_layers(table, 3, 78, name)
    assert sorted(table.cores) == list(range(3, 78))
    orders = ";".join(
        f"{n}:" + ",".join(map(str, ep.virtual_order(table, n)))
        for n in sorted(table.cores)
    )
    assert hashlib.sha256(orders.encode()).hexdigest() == SHIPPED_SLOT_ORDERS[name]


def test_unknown_table_name_is_refused(monkeypatch):
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", "glm52_no_such_table")
    with pytest.raises(FileNotFoundError, match="glm52_ep16_c32_4k"):
        ep.placement()


def test_glm52_ep_off_with_a_table_is_refused(
    use_table, installed, worker, monkeypatch
):
    use_table(_glm_table(GLM_PLACED))
    _glm_config(monkeypatch)
    installed()
    experts = _GlmExperts("model.layers.3.mlp.experts", 0)
    experts.moe_config.moe_parallel_config.use_ep = False
    with pytest.raises(ValueError, match="without expert parallelism"):
        RoutedExperts.update_expert_map_info(experts)


def test_glm52_stream_missing_a_placed_router_is_refused(
    use_table, installed, worker, monkeypatch
):
    use_table(_glm_table(GLM_PLACED))
    _glm_config(monkeypatch)
    _glm_router_stream(monkeypatch, layers=GLM_PLACED[:1])
    installed()
    _build_glm_worker(0)
    loader = default_loader.DefaultModelLoader
    with pytest.raises(ValueError, match=r"placed layers \[40, 77\] got no router"):
        dict(loader.get_all_weights(None, None, None))


def test_glm52_table_placing_no_layer_is_refused(
    use_table, installed, worker, monkeypatch
):
    use_table(_glm_table(GLM_PLACED))
    _glm_config(monkeypatch)
    _glm_router_stream(monkeypatch)
    installed()
    _build_glm_worker(0, layers=(GLM_UNPLACED,))
    loader = default_loader.DefaultModelLoader
    with pytest.raises(ValueError, match="placement EMPTY"):
        dict(loader.get_all_weights(None, None, None))


def _glm_copies_table(layers, seed=0):
    """16 ranks x (16 primaries + 1 copy): a random permutation per listed
    layer, and rank r copies the first primary of rank r + 1."""
    table = _glm_table(layers, seed)
    table["copy_share"] = {}
    for key, cores in table["layers"].items():
        copied = [cores[(r + 1) % GLM_RANKS][0] for r in range(GLM_RANKS)]
        table["layers"][key] = [c + [copied[r]] for r, c in enumerate(cores)]
        table["copy_share"][key] = {str(e): 0.5 for e in copied}
    return table


def test_glm52_copies_load_every_slot_on_16_ranks(
    use_table, installed, worker, monkeypatch, eplb_lines
):
    """Each rank builds 17 slots for a placed layer and keeps its 16
    primaries and its copy; over the ranks every expert is loaded once as a
    primary and the copied ones once more. The installed line keeps its
    prefix and names the copy slots."""
    path = use_table(_glm_copies_table(GLM_PLACED, seed=5))
    with open(path, "rb") as f:
        sha16 = hashlib.sha256(f.read()).hexdigest()[:16]
    _glm_config(monkeypatch)
    _glm_router_stream(monkeypatch)
    installed()
    table = ep.placement()
    loader = default_loader.DefaultModelLoader
    held = {layer: [0] * GLM_EXPERTS for layer in GLM_PLACED}
    for rank in range(GLM_RANKS):
        worker()
        built = _build_glm_worker(rank)
        linear = set(range(GLM_OWNED * rank, GLM_OWNED * (rank + 1)))
        for layer in GLM_PLACED:
            experts = built[layer]
            assert experts.local_num_experts == GLM_OWNED + 1
            assert experts.moe_config.num_local_experts == GLM_OWNED + 1
            assert ep.num_primary_experts(experts.layer_name) == GLM_OWNED
            local = experts._expert_map.tolist()
            assert [local[e] for e in table.cores[layer][rank]] == list(
                range(GLM_OWNED + 1)
            )
            for e in range(GLM_EXPERTS):
                name = GLM_CKPT.format(layer, e, "gate_proj", "weight")
                held[layer][e] += not _should_skip(name, linear)
        for layer in (GLM_UNPLACED, GLM_MTP):
            assert built[layer].local_num_experts == GLM_OWNED
            assert ep.num_primary_experts(built[layer].layer_name) is None
        dict(loader.get_all_weights(None, None, None))
    for layer in GLM_PLACED:
        copied = {c[GLM_OWNED] for c in table.cores[layer]}
        assert held[layer] == [2 if e in copied else 1 for e in range(GLM_EXPERTS)]
    assert [m for m in eplb_lines if m.startswith("[eplb]")] == [
        f"[eplb] placement installed ep_rank={r} layers={len(GLM_PLACED)} "
        f"table_sha256={sha16} copies=1"
        for r in range(GLM_RANKS)
    ]
