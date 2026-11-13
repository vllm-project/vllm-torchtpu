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
"""Static redundant MoE experts (VLLM_TPU_MOE_REDUNDANT_EXPERTS), host-only.

The kernel side needs no change (``fused_moe_func`` only sees physical ids, a
physical ``experts_start`` and ``S`` local experts), so these tests model the
EP grouped-matmul contract on CPU: each rank keeps the routed pairs whose
``id - experts_start`` falls in ``[0, S)``, evaluates them with its local
weights, and the ranks' partial outputs are summed (the EP all-reduce).
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter import moe_redundant as rx


def _remap(plan, ids):
    return rx.remap_ids(ids, rx.copies_table(plan), ep=plan.ep, slots=plan.slots_total)


# --------------------------------------------------------------------------
# Placement
# --------------------------------------------------------------------------


@pytest.mark.parametrize("num_logical,ep,slots", [(512, 8, 1), (512, 8, 2), (8, 2, 2)])
def test_placement_invariants(num_logical, ep, slots):
    hot = [463, 19, 15, 370, 253, 0, 1, 2, 3, 4, 5, 6, 7]
    hot = [h % num_logical for h in hot]
    plan = rx.compute_placement(num_logical, ep, slots, hot)
    assert len(plan.rep_logical) == ep * slots
    assert len(set(plan.rep_logical)) == ep * slots
    for i, logical in enumerate(plan.rep_logical):
        assert plan.alt_rank(i) != plan.home_rank(logical)
    p2l = plan.physical_to_logical()
    assert -1 not in p2l
    assert sorted(set(p2l)) == list(range(num_logical))
    # Deterministic: every rank computes the same placement.
    assert rx.compute_placement(num_logical, ep, slots, hot) == plan


def test_placement_takes_hot_first_and_respects_home_cap():
    plan = rx.compute_placement(512, 8, 1, [463, 19, 15])
    assert {463, 19, 15} <= set(plan.rep_logical)
    # All hot experts on rank 0 (experts 0..63): at most 7 can be replicated.
    plan = rx.compute_placement(512, 8, 1, list(range(20)))
    assert sum(1 for e in plan.rep_logical if e < 64) == 7
    # Tiny case that needs the swap: E=8, ep=2, 2 slots each, rank-0 hot.
    plan = rx.compute_placement(8, 2, 2, [0, 1, 2, 3])
    for i, logical in enumerate(plan.rep_logical):
        assert plan.alt_rank(i) != plan.home_rank(logical)


def test_placement_weighted_shares_prefer_light_ranks():
    # Rank 0 carries the load; its hot expert's copy must go elsewhere and the
    # light ranks receive copies first.
    shares = {5: 9.0, 70: 1.0, 140: 1.0}
    plan = rx.compute_placement(512, 8, 1, shares)
    i = plan.rep_logical.index(5)
    assert plan.alt_rank(i) != 0


def test_layer_key_and_sources(tmp_path):
    assert rx.layer_key("language_model.model.layers.37.mlp.experts") == "L37"
    assert rx.layer_key("mtp.layers.0.mlp.experts") == "mtp0"
    assert rx._load_source('{"L0": [1, 2]}') == {"L0": [1, 2]}
    f = tmp_path / "p.json"
    f.write_text('{"mtp0": [15]}')
    assert rx._load_source(str(f)) == {"mtp0": [15]}
    assert rx.hot_for_layer({"L3": [9]}, "model.layers.3.mlp.experts") == [9]
    assert rx.hot_for_layer({"moe0": [7]}, "model.layers.3.mlp.experts", 0) == [7]
    with pytest.raises(ValueError):
        rx._load_source("builtin:nope")


# --------------------------------------------------------------------------
# Router remap
# --------------------------------------------------------------------------


def _random_routing(num_tokens, num_logical, topk, seed, hot=()):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(num_tokens, num_logical, generator=g)
    for h in hot:
        logits[:, h] += 4.0
    w, ids = torch.topk(logits.softmax(-1), topk, dim=-1)
    return ids.to(torch.int32), w


@pytest.mark.parametrize("num_tokens", [8, 64, 600])
def test_remap_preserves_logical_experts(num_tokens):
    plan = rx.compute_placement(512, 8, 2, [463, 19, 15, 370])
    ids, _ = _random_routing(num_tokens, 512, 10, seed=num_tokens, hot=(463, 19))
    phys = _remap(plan, ids)
    p2l = torch.tensor(plan.physical_to_logical())
    assert torch.equal(p2l[phys.long()], ids.long())
    assert phys.dtype == torch.int32
    assert int(phys.max()) < plan.num_physical


def _max_rank_hits(pids, slots, ep):
    ranks = {int(p) // slots for p in set(pids.flatten().tolist())}
    hits = [0] * ep
    for p in set(pids.flatten().tolist()):
        hits[int(p) // slots] += 1
    return max(hits), sum(hits), ranks


def test_remap_moves_whole_experts_and_balances():
    num_logical, ep = 64, 8  # 8 experts per rank
    hot = [0, 1, 2, 3]  # all on rank 0
    plan = rx.compute_placement(num_logical, ep, 1, hot)
    g = torch.Generator().manual_seed(0)
    ids = torch.stack(
        [
            torch.cat(
                [torch.tensor([0, 1, 2, 3]), torch.randperm(56, generator=g)[:2] + 8]
            )
            for _ in range(2)
        ]
    ).to(torch.int32)
    phys = _remap(plan, ids)
    for logical in hot:
        targets = set(phys[ids == logical].tolist())
        assert len(targets) == 1  # whole expert on one copy per step
    before, total_b, _ = _max_rank_hits(ids, num_logical // ep, ep)
    after, total_a, _ = _max_rank_hits(phys, plan.slots_total, ep)
    assert after < before
    assert total_a == total_b  # same distinct experts, no weight read twice


def test_remap_large_batches_use_native_copies():
    plan = rx.compute_placement(64, 8, 1, [5])
    tokens = rx.MAX_ROUTES // 2 + 1
    ids = torch.full((tokens, 2), 5, dtype=torch.int32)
    phys = _remap(plan, ids)
    assert set(phys.flatten().tolist()) == {plan.native_physical(5)}


def _greedy(ids, table, ep, slots):
    """Independent restatement of the assignment rule."""
    hit = list(dict.fromkeys(e for e in ids.reshape(-1).tolist() if e >= 0))
    load = [0] * ep
    pick = {}
    for e in hit:
        if table[e][1] < 0:
            pick[e] = table[e][0]
            load[pick[e] // slots] += 1
    for e in hit:
        if e in pick:
            continue
        opts = [p for p in table[e] if p >= 0]
        best = opts[0]
        for p in opts[1:]:
            if load[p // slots] < load[best // slots]:
                best = p
        pick[e] = best
        load[best // slots] += 1
    pick[-1] = -1
    return np.vectorize(pick.get)(ids)


@pytest.mark.parametrize(
    "tokens,slots,padded", [(1, 1, 0), (8, 2, 0), (24, 4, 8), (96, 8, 16)]
)
def test_kernel_matches_rule(tokens, slots, padded):
    jnp = pytest.importorskip("jax.numpy")
    from vllm_torchtpu.kernels.moe_replica_assign import _reference, replica_assign

    plan = rx.compute_placement(128, 8, slots, list(range(0, 128, 3)))
    table = rx.copies_table(plan).numpy()
    rng = np.random.default_rng(tokens)
    ids = np.stack([rng.choice(128, 4, replace=False) for _ in range(tokens)])
    if padded:
        ids[-padded:] = -1  # skipped rows
    want = _greedy(ids, table.tolist(), 8, plan.slots_total)
    ref = np.asarray(
        _reference(jnp.asarray(ids.reshape(-1)), jnp.asarray(table), ep=8,
                   slots=plan.slots_total)
    ).reshape(ids.shape)
    np.testing.assert_array_equal(ref, want)
    torch_ref = rx.remap_ids_reference(
        torch.from_numpy(ids), torch.from_numpy(table), ep=8, slots=plan.slots_total
    )
    np.testing.assert_array_equal(torch_ref.numpy(), want)
    try:
        got = replica_assign(
            jnp.asarray(ids), jnp.asarray(table), ep=8, slots=plan.slots_total,
            interpret=True,
        )
    except Exception as exc:  # pragma: no cover - no Pallas interpreter here
        pytest.skip(f"pallas interpret unavailable: {exc}")
    np.testing.assert_array_equal(np.asarray(got), want)


# --------------------------------------------------------------------------
# End-to-end numerics on the EP contract
# --------------------------------------------------------------------------


def _expert(x, w1, w2):
    gate, up = (x @ w1).chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate) * up) @ w2


def _ep_moe(x, ids, weights, w1_local, w2_local, starts):
    """Sum over ranks of each rank's local contribution (fused_moe_func EP)."""
    out = torch.zeros_like(x)
    for w1, w2, start in zip(w1_local, w2_local, starts):
        local = ids.long() - start
        valid = (local >= 0) & (local < w1.shape[0])
        for t, k in itertools.product(range(ids.shape[0]), range(ids.shape[1])):
            if valid[t, k] and weights[t, k] != 0:
                e = int(local[t, k])
                out[t] += weights[t, k] * _expert(x[t], w1[e], w2[e])
    return out


@pytest.mark.parametrize("slots", [1, 2, 4])
def test_moe_output_identical_with_and_without_replicas(slots):
    torch.manual_seed(0)
    num_logical, ep, topk, hidden, inter, tokens = 32, 4, 4, 16, 8, 12
    local = num_logical // ep
    x = torch.randn(tokens, hidden, dtype=torch.float64)
    w1 = torch.randn(num_logical, hidden, 2 * inter, dtype=torch.float64) * 0.2
    w2 = torch.randn(num_logical, inter, hidden, dtype=torch.float64) * 0.2
    ids, w = _random_routing(tokens, num_logical, topk, seed=3, hot=(0, 1, 9))
    w = w.to(torch.float64)
    w[-1] = 0  # a padded token (TPU_MOE_SKIP_PADDED_TOKENS)

    base = _ep_moe(
        x,
        ids,
        w,
        [w1[r * local : (r + 1) * local] for r in range(ep)],
        [w2[r * local : (r + 1) * local] for r in range(ep)],
        [r * local for r in range(ep)],
    )

    plan = rx.compute_placement(num_logical, ep, slots, [0, 1, 9, 2])
    p2l = plan.physical_to_logical()
    s = plan.slots_total
    # What the weight loader builds: every physical slot holds its logical
    # expert's checkpoint weights.
    w1_local = [w1[p2l[r * s : (r + 1) * s]] for r in range(ep)]
    w2_local = [w2[p2l[r * s : (r + 1) * s]] for r in range(ep)]
    phys = _remap(plan, ids)
    assert not torch.equal(phys, ids)  # some work actually moved
    got = _ep_moe(x, phys, w, w1_local, w2_local, [r * s for r in range(ep)])
    # Same experts, same weights: only the cross-rank summation grouping
    # differs, so float64 agrees to rounding.
    torch.testing.assert_close(got, base, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------
# Weight loading into replica slots
# --------------------------------------------------------------------------


class _FakeExperts(torch.nn.Module):
    """vLLM RoutedExperts' loader contract, reduced to what the wrapper uses."""

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

    def weight_loader(
        self,
        param,
        loaded_weight,
        weight_name,
        shard_id,
        expert_id,
        return_success=False,
    ):
        e = self._map_global_expert_id_to_local_expert_id(expert_id)
        if e == -1:
            return False if return_success else None
        param.data[e].copy_(loaded_weight)
        return True if return_success else None


def test_weight_loader_fills_replica_slots(monkeypatch):
    monkeypatch.setattr(rx, "ENABLED", True)
    monkeypatch.setattr(rx, "SLOTS_PER_RANK", 2)
    monkeypatch.setattr(rx, "_placement_cache", {"L3": [1, 2, 9]})
    num_logical, ep = 16, 4
    ckpt = torch.arange(num_logical, dtype=torch.float32).view(-1, 1, 1) * torch.ones(
        num_logical, 2, 3
    )
    for rank in range(ep):
        layer = _FakeExperts(num_logical, ep, rank, "model.layers.3.mlp.experts")
        extra = rx.prepare(layer)
        assert extra == 2
        s = num_logical // ep + extra
        layer.w13_weight = torch.nn.Parameter(
            torch.full((s, 2, 3), -1.0), requires_grad=False
        )
        layer.w13_weight.weight_loader = layer.weight_loader
        rx.wrap_weight_loaders(layer)
        for gid in range(num_logical):
            layer.w13_weight.weight_loader(
                layer.w13_weight,
                ckpt[gid],
                "w13_weight",
                "w1",
                gid,
                return_success=True,
            )
        # The override is gone after loading.
        assert "_map_global_expert_id_to_local_expert_id" not in layer.__dict__
        plan = layer._rx_plan
        p2l = plan.physical_to_logical()[rank * s : (rank + 1) * s]
        got = layer.w13_weight.data[:, 0, 0].tolist()
        assert got == [float(v) for v in p2l]
        assert {1, 2, 9} <= set(plan.rep_logical)


def test_prepare_off_or_unsupported(monkeypatch):
    layer = _FakeExperts(16, 4, 0, "model.layers.0.mlp.experts")
    monkeypatch.setattr(rx, "ENABLED", False)
    assert rx.prepare(layer) == 0
    monkeypatch.setattr(rx, "ENABLED", True)
    monkeypatch.setattr(rx, "SLOTS_PER_RANK", 1)
    monkeypatch.setattr(rx, "_placement_cache", {})
    layer.moe_config.moe_parallel_config.use_ep = False
    assert rx.prepare(layer) == 0
    assert not hasattr(layer, "_rx_plan")


def test_install_registers_physical_buffers(monkeypatch):
    monkeypatch.setattr(rx, "ENABLED", True)
    monkeypatch.setattr(rx, "SLOTS_PER_RANK", 1)
    monkeypatch.setattr(rx, "_placement_cache", {})
    layer = _FakeExperts(16, 4, 2, "model.layers.0.mlp.experts")
    assert rx.prepare(layer) == 1
    layer.w13_weight = torch.nn.Parameter(torch.zeros(5, 2, 2), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.zeros(5, 2, 2), requires_grad=False)
    for p in (layer.w13_weight, layer.w2_weight):
        p.weight_loader = layer.weight_loader
    rx.wrap_weight_loaders(layer)
    with pytest.raises(RuntimeError, match="never loaded"):
        rx.install(layer)
    for gid in range(16):
        for p in (layer.w13_weight, layer.w2_weight):
            p.weight_loader(p, torch.ones(2, 2), "w", "w1", gid)
    layer.register_buffer("_experts_start", torch.tensor(8, dtype=torch.int32))
    rx.install(layer)
    assert int(layer._experts_start) == 2 * 5
    assert layer._rx_copies.shape == (16, 2)
    assert rx.num_physical(layer) == 20
    ids = torch.tensor([[0, 15]], dtype=torch.int32)
    phys = rx.to_physical(layer, ids, torch.ones(1, 2))
    p2l = layer._rx_plan.physical_to_logical()
    assert [p2l[i] for i in phys[0].tolist()] == [0, 15]


def test_prepare_skips_graphs_traced_once_for_every_bucket(monkeypatch):
    """The MTP draft ("eagle_head") is not re-traced per bucket group; keep it
    on plain experts."""
    monkeypatch.setattr(rx, "ENABLED", True)
    monkeypatch.setattr(rx, "SLOTS_PER_RANK", 2)
    monkeypatch.setattr(rx, "_placement_cache", {})
    layer = _FakeExperts(16, 4, 0, "mtp.layers.0.mlp.experts")
    monkeypatch.setattr(rx, "_model_tag", lambda: "eagle_head")
    assert rx.prepare(layer) == 0
    assert not hasattr(layer, "_rx_plan")
    monkeypatch.setattr(rx, "_model_tag", lambda: "backbone")
    assert rx.prepare(layer) == 2


def test_assign_op_signature_resolves():
    """jax_op reads the body's type hints (module-level names only)."""
    import typing

    hints = typing.get_type_hints(rx._assign_jax)
    assert hints["ep"] is int and hints["slots"] is int
