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
"""The static expert placement (expert_placement.py) on synthetic tables:
the table checks, the slot maps, the router-row permutation, the load-time
hooks on vLLM's real ExpertMapManager and loaders, and MoE output equality
with and without a placement. CPU only."""

import json
import random
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import expert_placement as ep
from vllm_torchtpu import model_loader_patches


def _experts_start(layer) -> int:
    """``moe_routing.get_experts_start`` (the contiguous-block rule every MoE
    path here dispatches by), or its formula where the layers package cannot
    import (no JAX on a CPU-only venv)."""
    try:
        from vllm_torchtpu.layers.adapter import moe_routing
    except ImportError:
        cfg = layer.moe_config
        base, rem = divmod(layer.global_num_experts, cfg.ep_size)
        return cfg.ep_rank * base + min(cfg.ep_rank, rem)
    return moe_routing.get_experts_start(layer)


def _write(tmp_path, table, name="placement.json"):
    path = tmp_path / name
    path.write_text(json.dumps(table))
    return str(path)


def _table(tmp_path, dup=False):
    order = [3, 3, 2, 1] if dup else [3, 0, 2, 1]  # rank 0: slots 0, 1; rank 1
    table = {
        "shards": 2,
        "experts": 4,
        "slots_per_core": 2,
        "layers": {str(n): {"cores": [order[:2], order[2:]]} for n in (0, 1)},
    }
    return _write(tmp_path, table), order


def _random_table(tmp_path, experts, ranks, layers, seed=0):
    rng = random.Random(seed)
    per = experts // ranks
    out = {}
    for n in range(layers):
        perm = list(range(experts))
        rng.shuffle(perm)
        out[str(n)] = {"cores": [perm[r * per : (r + 1) * per] for r in range(ranks)]}
    table = {"shards": ranks, "experts": experts, "slots_per_core": per, "layers": out}
    return _write(tmp_path, table)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(ep, "_EP_RANK", None)
    monkeypatch.setattr(ep, "_CURRENT_PREFIX", "")


def test_maps_and_router_rows(tmp_path, monkeypatch):
    path, order = _table(tmp_path)
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", path)
    assert ep.enabled()
    assert (ep.shards(), ep.slots(), ep.global_count()) == (2, 2, 4)
    assert ep.virtual_order(0) == order
    for rank in range(2):
        m = ep.expert_map(0, rank)
        for j, e in enumerate(ep.cores(0)[rank]):
            assert m[e] == j
            assert ep.virtual_order(0)[rank * 2 + j] == e
        assert ep.local_ids(0, rank) == frozenset(ep.cores(0)[rank])
        assert m.count(-1) == 2
    w = torch.arange(4.0).reshape(4, 1)
    out = ep.permute_router_rows("model.layers.0.mlp.gate.weight", w)
    assert out[:, 0].tolist() == [3.0, 0.0, 2.0, 1.0]
    assert ep.permute_router_rows("model.layers.0.mlp.up_proj.weight", w) is w
    # Qwen3.5's shared-expert gate is one row, not per expert.
    assert (
        ep.permute_router_rows("model.layers.0.mlp.shared_expert_gate.weight", w) is w
    )
    # The MTP draft layer keeps the default placement.
    assert ep.permute_router_rows("mtp.layers.0.mlp.gate.weight", w) is w
    with pytest.raises(ValueError):
        ep.permute_router_rows("model.layers.1.mlp.gate.weight", torch.zeros(5, 1))
    with pytest.raises(NotImplementedError):
        ep.permute_router_rows("model.layers.1.mlp.gate.weight_scale_inv", w)


def test_a_duplicated_expert_is_refused(tmp_path, monkeypatch):
    path, _ = _table(tmp_path, dup=True)
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", path)
    with pytest.raises(ValueError):
        ep.data()


def _copies_table(tmp_path, **changes):
    table = {
        "shards": 2,
        "experts": 4,
        "owned_per_core": 2,
        "slots_per_core": 3,
        "copies": True,
        # Primaries: rank 0 holds 1, 0; rank 1 holds 3, 2. Copies: rank 0
        # holds 2, rank 1 holds 1, declared hottest first.
        "layers": {"0": {"cores": [[1, 0, 2], [3, 2, 1]], "copies": {"1": 1, "2": 0}}},
    }
    table.update(changes)
    return _write(tmp_path, table)


def test_copy_tables(tmp_path, monkeypatch):
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", _copies_table(tmp_path))
    assert ep.has_copies() and ep.owned() == 2 and ep.slots() == 3
    # The router order and the loader see primaries; the map adds the copy.
    assert ep.virtual_order(0) == [1, 0, 3, 2]
    assert ep.expert_map(0, 0) == [1, 0, 2, -1]
    assert ep.local_ids(0, 1) == {3, 2, 1}
    # Virtual ids: 1 -> 0, 0 -> 1, 3 -> 2, 2 -> 3.
    gid, primary, copy = ep.holder_tables(0, 0)
    assert gid == [0, 1, 3]
    assert primary == [0, 0, 1, 1]
    assert copy == [1, -1, -1, 0]
    assert ep.holder_tables(0, 1)[0] == [2, 3, 0]
    assert ep.hot_order(0) == [0, 3]


@pytest.mark.parametrize(
    "cores",
    [
        [[1, 0, 0], [3, 2, 1]],  # a copy on its own primary rank
        [[1, 0, 2], [3, 2, 2]],  # an expert copied twice
        [[1, 0, 9], [3, 2, 1]],  # not an expert
    ],
)
def test_bad_copy_tables_are_refused(tmp_path, monkeypatch, cores):
    path = _copies_table(tmp_path, layers={"0": {"cores": cores}})
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", path)
    with pytest.raises(ValueError):
        ep.data()


def test_extra_slots_need_the_copies_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", _copies_table(tmp_path, copies=False))
    with pytest.raises(ValueError, match="must equal"):
        ep.data()


def test_unset_is_the_identity(monkeypatch):
    monkeypatch.delenv("TPU_EXPERT_PLACEMENT", raising=False)
    assert not ep.enabled()
    w = torch.zeros(4, 1)
    assert ep.permute_router_rows("model.layers.0.mlp.gate.weight", w) is w
    assert (
        ep.skip_expert_weight("model.layers.0.mlp.experts.1.up_proj.weight", 1) is None
    )
    # The loader filter keeps its linear behavior.
    name = "model.layers.0.mlp.experts.3.up_proj.weight"
    assert model_loader_patches._should_skip(name, {0, 1})
    assert not model_loader_patches._should_skip(name, {2, 3})


def test_loader_filter_follows_the_placement(tmp_path, monkeypatch):
    path, _ = _table(tmp_path)
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", path)
    name = "model.language_model.layers.0.mlp.experts.{}.down_proj.weight"
    with pytest.raises(RuntimeError, match="EP rank is unknown"):
        model_loader_patches._should_skip(name.format(3), {0, 1})
    monkeypatch.setattr(ep, "_EP_RANK", 0)
    linear_rank0 = {0, 1}
    # Rank 0 holds experts 3 and 0 under the placement, whatever the
    # linear set says.
    kept = {
        e
        for e in range(4)
        if not model_loader_patches._should_skip(name.format(e), linear_rank0)
    }
    assert kept == {0, 3}
    # MTP layers and non-expert tensors keep the linear filter.
    assert model_loader_patches._should_skip(
        "mtp.layers.0.mlp.experts.3.down_proj.weight", linear_rank0
    )
    assert not model_loader_patches._should_skip(
        "model.layers.0.mlp.gate.weight", linear_rank0
    )


def _parallel_config(ep_size, ep_rank, use_ep=True):
    return SimpleNamespace(
        use_ep=use_ep,
        ep_size=ep_size,
        ep_rank=ep_rank,
        tp_size=1,
        tp_rank=0,
        use_all2all_kernels=False,
        needs_round_robin_routing_tables=False,
        all2all_backend="naive",
        enable_eplb=False,
    )


def _manager(ep_size, ep_rank, experts, prefix):
    from vllm.model_executor.layers.fused_moe.expert_map_manager import (
        ExpertMapManager,
    )

    # One process simulates every rank here; a worker holds one.
    ep._EP_RANK = None
    ep._CURRENT_PREFIX = prefix
    try:
        return ExpertMapManager(
            max_num_batched_tokens=16,
            top_k=2,
            global_num_experts=experts,
            num_redundant_experts=0,
            num_expert_group=None,
            moe_parallel_config=_parallel_config(ep_size, ep_rank),
            placement_strategy="linear",
            enable_eplb=False,
        )
    finally:
        ep._CURRENT_PREFIX = ""


@pytest.fixture
def installed(monkeypatch):
    """install_model_patches on top of this package's host expert-map patch,
    with every patched attribute restored after the test."""
    from vllm.model_executor.layers import fused_moe as fused_pkg
    from vllm.model_executor.layers.fused_moe import expert_map_manager as emm
    from vllm.model_executor.layers.fused_moe import layer as fused_layer
    from vllm.model_executor.model_loader import default_loader as dl
    from vllm.model_executor.model_loader import runai_streamer_loader as rsl

    import vllm_torchtpu

    cls = emm.ExpertMapManager
    for attr in (
        "_calculate_expert_maps",
        "map_global_to_local",
        "get_compressed_map_string",
        "_tpu_expert_map_source",
        "_tpu_expert_map_version",
        "_tpu_expert_map_host",
        "_tpu_host_expert_map_patch",
    ):
        monkeypatch.setattr(cls, attr, getattr(cls, attr, None), raising=False)
    monkeypatch.setattr(fused_layer, "_tpu_placement_patch", False, raising=False)
    monkeypatch.setattr(fused_layer, "FusedMoEFactory", fused_layer.FusedMoEFactory)
    monkeypatch.setattr(fused_pkg, "FusedMoEFactory", fused_pkg.FusedMoEFactory)
    monkeypatch.setattr(
        dl.DefaultModelLoader, "get_all_weights", dl.DefaultModelLoader.get_all_weights
    )
    monkeypatch.setattr(
        rsl.RunaiModelStreamerLoader,
        "_get_weights_iterator",
        rsl.RunaiModelStreamerLoader._get_weights_iterator,
    )
    vllm_torchtpu._patch_expert_map_host_lookup()
    ep.install_model_patches()
    yield


def test_hooks_install_and_apply(tmp_path, monkeypatch, request):
    path = _random_table(tmp_path, experts=8, ranks=2, layers=2)
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", path)
    from vllm.model_executor.model_loader import default_loader as dl

    gate = torch.randn(8, 3)
    up = torch.randn(4, 3)

    def stream(self, *args, **kwargs):
        yield "model.layers.0.mlp.gate.weight", gate
        yield "model.layers.0.mlp.experts.0.up_proj.weight", up

    # A fake default-loader stream for install_model_patches to wrap.
    monkeypatch.setattr(dl.DefaultModelLoader, "get_all_weights", stream)
    request.getfixturevalue("installed")

    for rank in range(2):
        m = _manager(2, rank, 8, "model.layers.1.mlp.experts")
        assert m.expert_map.tolist() == ep.expert_map(1, rank)
        assert m.local_num_experts == 4
        for j, e in enumerate(ep.cores(1)[rank]):
            # The host-side lookup the MoE weight loader uses.
            assert m.map_global_to_local(e) == j
    assert ep._EP_RANK == 1
    # An MTP draft layer keeps the linear map.
    mtp = _manager(2, 0, 8, "mtp.layers.0.mlp.experts")
    assert mtp.expert_map.tolist() == [0, 1, 2, 3, -1, -1, -1, -1]
    # Outside the factory there is no prefix to place by.
    with pytest.raises(RuntimeError, match="outside the layer factory"):
        _manager(2, 0, 8, "")
    # A table for another EP size is refused.
    with pytest.raises(ValueError, match="ranks x"):
        _manager(4, 0, 8, "model.layers.0.mlp.experts")

    out = dict(dl.DefaultModelLoader.get_all_weights(None, None, None))
    assert torch.equal(out["model.layers.0.mlp.gate.weight"], gate[ep.virtual_order(0)])
    assert out["model.layers.0.mlp.experts.0.up_proj.weight"] is up


def test_placement_without_expert_parallel_is_refused(tmp_path, monkeypatch, request):
    path = _random_table(tmp_path, experts=8, ranks=2, layers=1)
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", path)
    request.getfixturevalue("installed")
    from vllm.model_executor.layers.fused_moe.expert_map_manager import (
        ExpertMapManager,
    )

    ep._CURRENT_PREFIX = "model.layers.0.mlp.experts"
    with pytest.raises(ValueError, match="enable-expert-parallel"):
        ExpertMapManager(
            max_num_batched_tokens=16,
            top_k=2,
            global_num_experts=8,
            num_redundant_experts=0,
            num_expert_group=None,
            moe_parallel_config=_parallel_config(1, 0, use_ep=False),
            placement_strategy="linear",
            enable_eplb=False,
        )


def _ffn(x, w13, w2):
    gate, up = w13.chunk(2, dim=0)
    return (torch.nn.functional.silu(x @ gate.T) * (x @ up.T)) @ w2.T


def _route(x, router, topk):
    probs = torch.softmax(x @ router.T, dim=-1)
    weights, ids = torch.topk(probs, topk, dim=-1)
    return weights / weights.sum(-1, keepdim=True), ids


@pytest.mark.parametrize("ranks", [2, 4])
def test_moe_output_matches_without_placement(tmp_path, monkeypatch, request, ranks):
    """A random MoE layer computed twice: the checkpoint's own placement
    (reference) and the placement realized purely at load (expert weights
    loaded through the placed expert map, router rows permuted), then
    dispatched by the contiguous-block rule every MoE path here uses
    (local slot = id - experts_start) and combined across ranks."""
    experts, hidden, inter, topk, tokens, layer = 16, 32, 16, 4, 24, 3
    path = _random_table(tmp_path, experts, ranks, layers=4, seed=ranks)
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", path)
    request.getfixturevalue("installed")
    torch.manual_seed(0)
    router = torch.randn(experts, hidden, dtype=torch.float64)
    w13 = torch.randn(experts, 2 * inter, hidden, dtype=torch.float64) / hidden**0.5
    w2 = torch.randn(experts, hidden, inter, dtype=torch.float64) / inter**0.5
    x = torch.randn(tokens, hidden, dtype=torch.float64)

    ref_w, ref_ids = _route(x, router, topk)
    ref = torch.zeros_like(x)
    for t in range(tokens):
        for k in range(topk):
            e = int(ref_ids[t, k])
            ref[t] += ref_w[t, k] * _ffn(x[t], w13[e], w2[e])

    prefix = f"model.language_model.layers.{layer}.mlp"
    placed_router = ep.permute_router_rows(f"{prefix}.gate.weight", router)
    out = torch.zeros_like(x)
    w, ids = _route(x, placed_router, topk)
    # The same experts with the same weights, named by virtual id.
    order = torch.tensor(ep.virtual_order(layer))
    assert torch.equal(order[ids], ref_ids)
    torch.testing.assert_close(w, ref_w)
    for rank in range(ranks):
        manager = _manager(ranks, rank, experts, f"{prefix}.experts")
        assert rank == ep._EP_RANK
        local = experts // ranks
        local_w13 = torch.full(
            (local, 2 * inter, hidden), float("nan"), dtype=w13.dtype
        )
        local_w2 = torch.full((local, hidden, inter), float("nan"), dtype=w2.dtype)
        for e in range(experts):
            name = f"{prefix}.experts.{e}.gate_proj.weight"
            linear = set(range(rank * local, (rank + 1) * local))
            skipped = model_loader_patches._should_skip(name, linear)
            slot = manager.map_global_to_local(e)
            assert skipped == (slot < 0)
            if slot >= 0:  # what FusedMoE.weight_loader does per expert
                local_w13[slot] = w13[e]
                local_w2[slot] = w2[e]
        layer_view = SimpleNamespace(
            global_num_experts=experts,
            moe_config=SimpleNamespace(
                moe_parallel_config=SimpleNamespace(use_ep=True),
                ep_size=ranks,
                ep_rank=rank,
            ),
        )
        start = _experts_start(layer_view)
        for t in range(tokens):
            for k in range(topk):
                slot = int(ids[t, k]) - start
                if 0 <= slot < local:
                    out[t] += w[t, k] * _ffn(x[t], local_w13[slot], local_w2[slot])
    torch.testing.assert_close(out, ref, rtol=1e-12, atol=1e-12)


def test_shipped_qwen35_table_is_valid(monkeypatch):
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts/vllm/benchmark/inferencex/qwen3.5/expert_placement_dp8_ep8.json"
    )
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", str(path))
    assert (ep.shards(), ep.global_count(), ep.slots()) == (8, 512, 64)
    assert len(ep.data()["layers"]) == 60


def test_shipped_qwen35_copies_table_is_valid(monkeypatch):
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts/vllm/benchmark/inferencex/qwen3.5"
        / "expert_placement_dp8_ep8_copies128.json"
    )
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", str(path))
    assert ep.has_copies()
    assert (ep.shards(), ep.global_count(), ep.owned(), ep.slots()) == (8, 512, 64, 80)
    assert len(ep.data()["layers"]) == 60
    gid, primary, copy = ep.holder_tables(0, 3)
    assert len(gid) == 80 and gid[:64] == list(range(3 * 64, 4 * 64))
    assert sum(c >= 0 for c in copy) == len(ep.hot_order(0)) == 128
