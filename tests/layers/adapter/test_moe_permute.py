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
        e = self._map_global_expert_id_to_local_expert_id(expert_id)
        if e == -1:
            return False if return_success else None
        param.data[e].copy_(loaded_weight)
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
    monkeypatch.setattr(mp, "_widen_ep_weight_filter", lambda: None)
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


def test_builtin_table_is_valid():
    data = mp.load_source("builtin:minimax_m31_agentx_perm")
    assert len(data) == 57
    for key, perm in data.items():
        assert key.startswith("moe")
        mp.validate(perm, 128)
