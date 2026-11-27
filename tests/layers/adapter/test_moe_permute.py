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
"""Static MoE expert permutation (VLLM_TPU_MOE_EXPERT_PERMUTATION), host-only.

Models the EP grouped-matmul contract on CPU like test_moe_redundant: each
rank keeps the routed pairs whose ``id - experts_start`` is in ``[0, L)``,
evaluates them with its local weights, and the partial outputs are summed.
"""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.layers.adapter import moe_permute as mp
from vllm_torchtpu.layers.adapter import moe_redundant as rx


class _FakeExperts(torch.nn.Module):
    def __init__(self, num_logical, ep, rank, name):
        super().__init__()
        self.layer_name = name
        self.global_num_experts = num_logical
        self.expert_placement_strategy = "linear"
        self.moe_config = SimpleNamespace(
            moe_parallel_config=SimpleNamespace(use_ep=True), ep_size=ep, ep_rank=rank
        )
        self._local = num_logical // ep
        self._rank = rank

    def _map_global_expert_id_to_local_expert_id(self, gid):
        lo = self._rank * self._local
        return gid - lo if lo <= gid < lo + self._local else -1

    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id,
                      return_success=False):
        from vllm_torchtpu import model_loader_patches as mlp

        e = self._map_global_expert_id_to_local_expert_id(expert_id)
        if e == -1:
            return False if return_success else None
        param.data[e].copy_(loaded_weight)
        # What the staging hook in model_loader_patches records.
        mlp._note_permuted_write(self, e, None, expert_id, weight_name, shard_id)
        return True if return_success else None


def _perm(num_logical, seed):
    p = list(range(num_logical))
    random.Random(seed).shuffle(p)
    return p


def _setup(monkeypatch, table):
    monkeypatch.setattr(mp, "ENABLED", True)
    monkeypatch.setattr(mp, "SPEC", "test")
    monkeypatch.setattr(mp, "_source_cache", table)
    monkeypatch.setattr(mp, "_ORDINAL", [0])
    monkeypatch.setattr(mp, "_patch_expert_id_filter", lambda: None)
    monkeypatch.setattr(mp, "_LOCAL_BY_LAYER", {})
    monkeypatch.setattr(mp, "_LOCAL_UNION", set())
    monkeypatch.setattr(mp, "_LOADING", [False])
    monkeypatch.setattr(mp, "_STATS", {"layers": 0, "moved": 0, "prepared": 0,
                                       "checked_experts": 0})
    monkeypatch.setattr(mp, "write_tracking", lambda: True)
    monkeypatch.setattr(rx, "ENABLED", False)


def _ep_moe(x, ids, w, w1_local, w2_local, starts):
    """Reference EP MoE: per rank keep local ids, silu-gated FFN, sum ranks."""
    out = torch.zeros_like(x)
    local = w1_local[0].shape[0]
    for r, start in enumerate(starts):
        part = torch.zeros_like(x)
        for t in range(x.shape[0]):
            for k in range(ids.shape[1]):
                e = int(ids[t, k]) - start
                if not (0 <= e < local) or w[t, k] == 0:
                    continue
                h = x[t] @ w1_local[r][e]
                g, u = h.chunk(2)
                part[t] += w[t, k] * ((torch.nn.functional.silu(g) * u) @ w2_local[r][e])
        out += part
    return out


@pytest.mark.parametrize("seed", [0, 1])
def test_loader_and_routing_give_the_linear_result(monkeypatch, seed):
    torch.manual_seed(seed)
    num_logical, ep, topk, hidden, inter, tokens = 32, 4, 4, 16, 8, 10
    local = num_logical // ep
    perm = _perm(num_logical, seed)
    x = torch.randn(tokens, hidden, dtype=torch.float64)
    w1 = torch.randn(num_logical, hidden, 2 * inter, dtype=torch.float64) * 0.2
    w2 = torch.randn(num_logical, inter, hidden, dtype=torch.float64) * 0.2
    ids = torch.stack([torch.randperm(num_logical)[:topk] for _ in range(tokens)])
    ids[-1] = -1  # rows outside every rank's range stay outside
    w = torch.rand(tokens, topk, dtype=torch.float64)
    starts = [r * local for r in range(ep)]
    base = _ep_moe(x, ids, w, [w1[s:s + local] for s in starts],
                   [w2[s:s + local] for s in starts], starts)

    _setup(monkeypatch, {"moe0": perm})
    w1_local, w2_local, layers = [], [], []
    for rank in range(ep):
        mp._ORDINAL[0] = 0
        layer = _FakeExperts(num_logical, ep, rank, "model.layers.3.mlp.experts")
        assert mp.prepare(layer)
        layer.w13_weight = torch.nn.Parameter(
            torch.full((local, hidden, 2 * inter), float("nan"), dtype=torch.float64),
            requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(
            torch.full((local, inter, hidden), float("nan"), dtype=torch.float64),
            requires_grad=False)
        for gid in range(num_logical):
            layer.weight_loader(layer.w13_weight, w1[gid], "w13", "w1", gid)
            layer.weight_loader(layer.w2_weight, w2[gid], "w2", "w2", gid)
        assert not torch.isnan(layer.w13_weight).any()  # every slot loaded once
        mp.install(layer)
        w1_local.append(layer.w13_weight.data)
        w2_local.append(layer.w2_weight.data)
        layers.append(layer)
    phys = mp.to_physical(layers[0], ids)
    assert phys.dtype == ids.dtype
    assert (phys[-1] == -1).all()
    for layer in layers[1:]:  # rank-independent table
        assert torch.equal(mp.to_physical(layer, ids), phys)
    got = _ep_moe(x, phys, w, w1_local, w2_local, starts)
    torch.testing.assert_close(got, base, rtol=1e-12, atol=1e-12)


def test_off_or_unsupported(monkeypatch):
    layer = _FakeExperts(16, 4, 0, "model.layers.0.mlp.experts")
    monkeypatch.setattr(mp, "ENABLED", False)
    assert not mp.prepare(layer)
    _setup(monkeypatch, {"moe0": _perm(16, 0)})
    monkeypatch.setattr(rx, "ENABLED", True)
    assert not mp.prepare(layer)
    monkeypatch.setattr(rx, "ENABLED", False)
    layer.moe_config.moe_parallel_config.use_ep = False
    assert not mp.prepare(layer)
    assert "_map_global_expert_id_to_local_expert_id" not in layer.__dict__
    ids = torch.tensor([[1, 2]])
    assert mp.to_physical(layer, ids) is ids


def test_layers_without_table_and_draft_graph(monkeypatch):
    _setup(monkeypatch, {"moe1": _perm(16, 3)})
    a = _FakeExperts(16, 4, 0, "model.layers.3.mlp.experts")
    b = _FakeExperts(16, 4, 0, "model.layers.4.mlp.experts")
    assert not mp.prepare(a)  # moe0 has no table: linear
    assert mp.prepare(b)
    monkeypatch.setattr(mp, "_model_tag", lambda: "eagle_head")
    c = _FakeExperts(16, 4, 0, "mtp.layers.0.mlp.experts")
    assert not mp.prepare(c)


def test_sources_and_validation(tmp_path):
    with pytest.raises(ValueError, match="permutation"):
        mp.validate([0, 0, 1, 2], 4)
    assert mp.validate([3, 2, 1, 0], 4) == [3, 2, 1, 0]
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"_src": "x", "moe0": [1, 0]}))
    assert mp.load_source(str(path)) == {"moe0": [1, 0]}
    assert mp.load_source('{"moe2": [0, 1]}') == {"moe2": [0, 1]}
    assert mp.load_source("") == {}
    with pytest.raises(ValueError, match="unknown builtin"):
        mp.load_source("builtin:does_not_exist")


@pytest.mark.parametrize(
    "name", ["minimax_m31_agentx_perm", "minimax_m31_agentx_prefill_k13"]
)
def test_builtin_table_is_valid(name):
    data = mp.load_source(f"builtin:{name}")
    assert len(data) == 57
    for key, perm in data.items():
        assert key.startswith("moe")
        mp.validate(perm, 128)


def _load_rank(num_logical, ep, rank, name, perm_table, gids):
    mp._ORDINAL[0] = 0
    layer = _FakeExperts(num_logical, ep, rank, name)
    assert mp.prepare(layer)
    local = num_logical // ep
    layer.w13_weight = torch.nn.Parameter(torch.zeros(local, 2), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.zeros(local, 2), requires_grad=False)
    for gid in gids:
        layer.weight_loader(layer.w13_weight, torch.ones(2), f"{name}.w13_weight", "w1", gid)
        layer.weight_loader(layer.w13_weight, torch.ones(2), f"{name}.w13_weight", "w3", gid)
        layer.weight_loader(layer.w2_weight, torch.ones(2), f"{name}.w2_weight", "w2", gid)
    return layer


def test_install_raises_when_permuted_experts_were_filtered(monkeypatch):
    """The broken load: a linear EP filter streams only linear-local experts,
    so experts permuted in from other ranks are never written."""
    num_logical, ep, rank = 32, 4, 1
    perm = _perm(num_logical, 5)
    _setup(monkeypatch, {"moe0": perm})
    local = num_logical // ep
    linear = range(rank * local, (rank + 1) * local)
    layer = _load_rank(num_logical, ep, rank, "model.layers.3.mlp.experts", perm, linear)
    with pytest.raises(RuntimeError, match="permuted-local experts not loaded"):
        mp.install(layer)

    # A partial write (w3 never arrives for one expert) also fails.
    _setup(monkeypatch, {"moe0": perm})
    layer = _load_rank(num_logical, ep, rank, "model.layers.3.mlp.experts", perm, [])
    lo = rank * local
    mine = [e for e, p in enumerate(perm) if lo <= p < lo + local]
    for gid in mine:
        layer.weight_loader(layer.w13_weight, torch.ones(2), "x.w13_weight", "w1", gid)
        if gid != mine[0]:
            layer.weight_loader(layer.w13_weight, torch.ones(2), "x.w13_weight", "w3", gid)
        layer.weight_loader(layer.w2_weight, torch.ones(2), "x.w2_weight", "w2", gid)
    with pytest.raises(RuntimeError, match=f"partial=\\[{mine[0]}\\]"):
        mp.install(layer)

    # The full stream passes and logs the complete count.
    _setup(monkeypatch, {"moe0": perm})
    layer = _load_rank(num_logical, ep, rank, "model.layers.3.mlp.experts", perm,
                       range(num_logical))
    mp.install(layer)
    assert mp.summary()["checked_experts"] == local


def test_per_layer_filter_streams_exactly_the_permuted_local_experts(monkeypatch):
    from vllm_torchtpu import model_loader_patches as mlp

    num_logical, ep, rank = 32, 4, 2
    local = num_logical // ep
    lo = rank * local
    p0, p1 = _perm(num_logical, 7), _perm(num_logical, 8)
    _setup(monkeypatch, {"moe0": p0, "moe2": p1})
    a = _FakeExperts(num_logical, ep, rank, "model.layers.3.block_sparse_moe.experts")
    b = _FakeExperts(num_logical, ep, rank, "model.layers.4.block_sparse_moe.experts")
    c = _FakeExperts(num_logical, ep, rank, "model.layers.5.block_sparse_moe.experts")
    assert mp.prepare(a) and not mp.prepare(b) and mp.prepare(c)
    model = torch.nn.Module()
    model.a, model.b, model.c = a, b, c
    mine0 = {e for e, p in enumerate(p0) if lo <= p < lo + local}
    mine2 = {e for e, p in enumerate(p1) if lo <= p < lo + local}
    linear = set(range(lo, lo + local))
    union = mine0 | mine2 | linear
    assert mp.widen_local_expert_ids(set(linear)) == union
    assert mp.widen_local_expert_ids(None) is None

    def kept(layer_idx):
        return {
            e for e in range(num_logical)
            if not mlp._should_skip(
                f"language_model.model.layers.{layer_idx}.block_sparse_moe.experts.{e}.w1.weight",
                union)
        }

    # Outside a permuted model's load the union filter applies.
    assert kept(3) == union
    with mp.loading(model):
        assert kept(3) == mine0
        assert kept(4) == linear  # layer without a table stays linear
        assert kept(5) == mine2
        assert kept(9) == union  # unknown layer: the caller's filter
        assert not mlp._should_skip("model.layers.3.self_attn.q_proj.weight", union)
    assert kept(5) == union
    # A model without permuted layers (the draft) never gets the per-layer filter.
    with mp.loading(torch.nn.Module()):
        assert kept(3) == union


def test_launcher_filter_is_widened(monkeypatch):
    import builtins

    num_logical, ep, rank = 16, 4, 0
    perm = _perm(num_logical, 2)
    _setup(monkeypatch, {"moe0": perm})
    layer = _FakeExperts(num_logical, ep, rank, "model.layers.3.mlp.experts")
    assert mp.prepare(layer)
    model = torch.nn.Module()
    model.layer = layer
    linear = {0, 1, 2, 3}
    monkeypatch.setattr(builtins, "_EXOS_MODEL_LOAD_FILTER",
                        (False, (), "model.layers.60.", set(linear)), raising=False)
    mp.widen_external_filter()  # not loading: untouched
    assert builtins._EXOS_MODEL_LOAD_FILTER[3] == linear
    with mp.loading(model):
        mp.widen_external_filter()
    got = builtins._EXOS_MODEL_LOAD_FILTER
    assert got[:3] == (False, (), "model.layers.60.")
    assert got[3] == linear | {e for e, p in enumerate(perm) if p < 4}
    # The MTP load keeps its own filter.
    monkeypatch.setattr(builtins, "_EXOS_MODEL_LOAD_FILTER", (True, (), None, set(linear)))
    with mp.loading(model):
        mp.widen_external_filter()
    assert builtins._EXOS_MODEL_LOAD_FILTER[3] == linear


@pytest.mark.parametrize("seed", [0, 1])
def test_permuted_router_emits_physical_ids(monkeypatch, seed):
    torch.manual_seed(seed)
    num_logical, ep, topk, hidden, inter, tokens = 32, 4, 4, 16, 8, 10
    local = num_logical // ep
    perm = _perm(num_logical, seed)
    x = torch.randn(tokens, hidden, dtype=torch.float64)
    gw = torch.randn(num_logical, hidden, dtype=torch.float64)
    gb = torch.randn(num_logical, dtype=torch.float64)
    w1 = torch.randn(num_logical, hidden, 2 * inter, dtype=torch.float64) * 0.2
    w2 = torch.randn(num_logical, inter, hidden, dtype=torch.float64) * 0.2

    def route(weight, bias):
        scores = torch.sigmoid(x @ weight.T)
        _, ids = torch.topk(scores + bias, topk, dim=-1)
        w = scores.gather(1, ids)
        return ids, w / w.sum(-1, keepdim=True)

    ids, w = route(gw, gb)
    starts = [r * local for r in range(ep)]
    base = _ep_moe(x, ids, w, [w1[s:s + local] for s in starts],
                   [w2[s:s + local] for s in starts], starts)

    _setup(monkeypatch, {"moe0": perm})
    w1_local, w2_local, routers = [], [], []
    for rank in range(ep):
        mp._ORDINAL[0] = 0
        layer = _FakeExperts(num_logical, ep, rank, "model.layers.3.mlp.experts")
        assert mp.prepare(layer)
        gate = torch.nn.Parameter(torch.full_like(gw, float("nan")), requires_grad=False)
        bias = torch.nn.Parameter(torch.full_like(gb, float("nan")), requires_grad=False)
        assert mp.permute_router(layer, gate_weight=gate, e_score_correction_bias=bias)
        with pytest.raises(RuntimeError, match="router parameters"):
            mp.install(layer)  # nothing loaded through the permuting loader yet
        gate.weight_loader(gate, gw)
        bias.weight_loader(bias, gb)
        layer.w13_weight = torch.nn.Parameter(
            torch.full((local, hidden, 2 * inter), float("nan"), dtype=torch.float64),
            requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(
            torch.full((local, inter, hidden), float("nan"), dtype=torch.float64),
            requires_grad=False)
        for gid in range(num_logical):
            layer.weight_loader(layer.w13_weight, w1[gid], "w13", "w1", gid)
            layer.weight_loader(layer.w2_weight, w2[gid], "w2", "w2", gid)
        mp.install(layer)
        w1_local.append(layer.w13_weight.data)
        w2_local.append(layer.w2_weight.data)
        routers.append((gate.data, bias.data))
        # Physical row p holds logical expert inv[p].
        assert torch.equal(gate.data[torch.tensor(perm)], gw)
        assert torch.equal(bias.data[torch.tensor(perm)], gb)
    phys, wp = route(*routers[0])
    assert torch.equal(phys, torch.tensor(perm)[ids])
    assert torch.equal(mp.to_physical(layer, phys), phys)  # no second mapping
    torch.testing.assert_close(wp, w, rtol=0, atol=0)
    got = _ep_moe(x, phys, wp, w1_local, w2_local, starts)
    torch.testing.assert_close(got, base, rtol=1e-12, atol=1e-12)


def test_per_pair_outputs_are_bitwise_and_only_rank_grouping_changes(monkeypatch):
    """A permutation moves whole (token, expert) pairs between ranks: every
    pair's weighted expert output is bitwise the linear one; only which rank's
    partial sum holds it changes (float32, the rank-grouped sum agrees to
    rounding)."""
    torch.manual_seed(0)
    num_logical, ep, topk, hidden, inter, tokens = 32, 4, 4, 16, 8, 64
    local = num_logical // ep
    perm = _perm(num_logical, 3)
    x = torch.randn(tokens, hidden)
    w1 = torch.randn(num_logical, hidden, 2 * inter) * 0.2
    w2 = torch.randn(num_logical, inter, hidden) * 0.2
    ids = torch.stack([torch.randperm(num_logical)[:topk] for _ in range(tokens)])
    w = torch.rand(tokens, topk)

    def pairs(slot_of, weights_of):
        out, parts = {}, torch.zeros(ep, tokens, hidden)
        for t in range(tokens):
            for k in range(topk):
                e = int(ids[t, k])
                slot = slot_of[e]
                h = x[t] @ weights_of[0][slot]
                g, u = h.chunk(2)
                y = w[t, k] * ((torch.nn.functional.silu(g) * u) @ weights_of[1][slot])
                out[(t, e)] = y
                parts[slot // local, t] += y
        return out, parts.sum(0)

    lin_pairs, lin_sum = pairs(list(range(num_logical)), (w1, w2))
    inv = [0] * num_logical
    for e, p in enumerate(perm):
        inv[p] = e
    phys_w = (w1[inv], w2[inv])  # physical slot p holds logical inv[p]
    perm_pairs, perm_sum = pairs(perm, phys_w)
    assert lin_pairs.keys() == perm_pairs.keys()
    for key, y in lin_pairs.items():
        assert torch.equal(perm_pairs[key], y), key
    torch.testing.assert_close(perm_sum, lin_sum, rtol=1e-5, atol=1e-5)
