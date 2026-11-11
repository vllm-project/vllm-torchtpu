# SPDX-License-Identifier: Apache-2.0
"""Tests for the layout solver: the vendored policy's structure and
determinism (DeepInfer's own checks), the rank-uniqueness constraint, and the
layout writer's output parsing under vllm_torchtpu.expert_placement."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_layout  # noqa: E402
import solve_layout  # noqa: E402
import tile_model as tm  # noqa: E402
from eplb_policy import (  # noqa: E402
    _batch_greedy_indices,
    rank_lists,
    rebalance_experts_global,
)

pytestmark = pytest.mark.cpu_test

CASES = [
    (1, 1, 16, 24, 3),
    (3, 2, 64, 72, 9),
    (2, 4, 128, 160, 20),
    (1, 1, 8, 8, 1),
    (4, 2, 64, 128, 16),
    (3, 3, 16, 64, 8),
    (5, 2, 384, 416, 13),
]


def _check_structure(phy2log, log2phy, log_count, num_physical):
    num_layers = phy2log.size(0)
    assert log_count.equal((log2phy >= 0).sum(-1))
    for layer in range(num_layers):
        for physical_index, logical_index in enumerate(phy2log[layer].tolist()):
            assert physical_index in log2phy[layer][logical_index].tolist()
    assert log_count.sum(-1).equal(
        torch.full((num_layers,), num_physical, dtype=torch.int)
    )


@pytest.mark.parametrize(
    "num_steps,num_layers,num_logical,num_physical,num_local", CASES
)
def test_rebalance_structure_and_rank_uniqueness(
    num_steps, num_layers, num_logical, num_physical, num_local
):
    torch.manual_seed(num_steps * 100 + num_layers * 10 + num_physical)
    tokens = torch.randint(0, 5000, (num_steps, num_layers, num_logical))
    phy2log, log2phy, log_count = rebalance_experts_global(
        tokens, num_physical, num_local
    )
    _check_structure(phy2log, log2phy, log_count, num_physical)
    for layer in rank_lists(phy2log, num_local):
        for held in layer:
            assert len(held) == num_local and len(set(held)) == num_local
        assert sorted(e for held in layer for e in held if True) == sorted(
            phy2log.new_tensor([e for held in layer for e in held]).tolist()
        )


@pytest.mark.parametrize(
    "num_steps,num_layers,num_logical,num_physical,num_local", CASES
)
def test_rebalance_is_deterministic(
    num_steps, num_layers, num_logical, num_physical, num_local
):
    torch.manual_seed(num_steps * 100 + num_layers * 10 + num_physical)
    tokens = torch.randint(0, 5000, (num_steps, num_layers, num_logical))
    first = rebalance_experts_global(tokens.clone(), num_physical, num_local)
    second = rebalance_experts_global(tokens.clone(), num_physical, num_local)
    for x, y in zip(first, second, strict=True):
        assert torch.equal(x, y)


def test_a_hot_expert_is_replicated_onto_different_ranks():
    # One expert carries half the rows; with 2 spare slots over 4 ranks it
    # gets copies, and no rank holds two of them.
    tokens = torch.ones(4, 1, 16)
    tokens[:, :, 5] = 400.0
    phy2log, log2phy, log_count = rebalance_experts_global(tokens, 20, 5)
    assert log_count[0, 5] >= 2
    ranks = rank_lists(phy2log, 5)[0]
    holders = [r for r, held in enumerate(ranks) if 5 in held]
    assert len(holders) == int(log_count[0, 5])
    for held in ranks:
        assert held.count(5) <= 1


def test_the_constraint_can_be_switched_off_for_comparison():
    tokens = torch.ones(2, 1, 4)
    tokens[:, :, 0] = 1000.0
    phy2log_map = torch.tensor([[0, 1, 2, 3, 0, 0, 0, 0]], dtype=torch.int32)
    counts = torch.tensor([[4, 1, 1, 1]], dtype=torch.int32)
    free = _batch_greedy_indices(tokens, phy2log_map, counts, 4, unique_per_rank=False)
    assert free.shape == (1, 8)
    with pytest.raises(ValueError, match="holding the expert twice"):
        # Four copies of expert 0 over two ranks cannot be unique per rank.
        _batch_greedy_indices(tokens, phy2log_map, counts, 4, unique_per_rank=True)


def test_batch_greedy_objective_and_ties_match_the_reference():
    history, physical, expected = (
        [[10, 0, 6, 0, 4, 0], [0, 10, 0, 6, 0, 4]],
        [0, 1, 2, 3, 4, 5],
        [0, 1, 5, 2, 3, 4],
    )
    tokens = torch.tensor(history).unsqueeze(1)
    counts = torch.bincount(torch.tensor(physical), minlength=6).unsqueeze(0)
    physical_map = torch.tensor([physical], dtype=torch.int32)
    indices = _batch_greedy_indices(tokens, physical_map, counts, 3)
    assert indices.tolist() == [expected]


def test_tile_model_replicas_halve_a_hot_expert_and_split_rows():
    rows = np.zeros(8)
    rows[0] = 512
    rows[1:] = 32
    contiguous = tm.contiguous_ranks(8, 2)
    slow_c, mean_c, passes_c, _ = tm.predict(rows, contiguous)
    ranks = [[0, 1, 2, 3, 0 + 4], [4, 5, 6, 7, 0]]
    ranks = [[0, 1, 2, 3, 4], [5, 6, 7, 0, 4]]
    reps = tm.replica_counts(ranks, 8)
    assert reps.tolist() == [2, 1, 1, 1, 2, 1, 1, 1]
    slow_l, mean_l, passes_l, _ = tm.predict(rows, ranks, reps)
    assert slow_l < slow_c and passes_l < passes_c
    assert abs(mean_l - mean_c) < 15  # the same rows, a few more tails


def _fake_tap(path, steps=6, layers=2, src=8, experts=32, draft_experts=8, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(3, size=(steps, layers, src, experts)).astype(np.int32)
    counts[:, :, :, 7] += 40  # one hot expert
    draft = rng.poisson(3, size=(steps, 1, src, draft_experts)).astype(np.int32)
    np.savez(
        path,
        counts=counts,
        layers=np.arange(layers),
        draft_counts=draft,
        draft_layers=np.array([layers]),
    )


def test_solve_layout_writes_a_parseable_layout_with_a_draft_group(tmp_path):
    tap = tmp_path / "routing_round1.npz"
    _fake_tap(tap)
    out = tmp_path / "layout.json"
    assert (
        solve_layout.main(
            [
                "--routing",
                str(tap),
                "--shards",
                "4",
                "--slots",
                "9",
                "--pool",
                "2",
                "--draft-slots",
                "3",
                "--out",
                str(out),
                "--name",
                "t",
            ]
        )
        == 0
    )
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from vllm_torchtpu import expert_placement

    doc = json.loads(out.read_text())
    p = expert_placement.parse(doc, str(out))
    assert p.target.shards == 4 and p.target.slots == 9 and p.target.physical == 36
    assert sorted(int(k) for k in doc["layers"]) == [0, 1]
    assert p.draft.experts == 8 and p.draft.slots == 3 and sorted(p.draft.layers) == [0]
    # The hot expert got a copy on every layer and the prediction improved.
    for lid in ("0", "1"):
        assert 7 in doc["meta"]["predicted"][lid]["replicated"]
        pred = doc["meta"]["predicted"][lid]
        assert pred["layout_slowest_us"] < pred["contiguous_slowest_us"]
    assert doc["meta"]["steps"] == 3  # 6 steps pooled by 2
    # The checker accepts it and the predicted ratio is reported.
    assert check_layout.main([str(out), "--routing", str(tap), "--pool", "2"]) == 0


def test_min_gain_gate_leaves_flat_layers_contiguous(tmp_path):
    tap = tmp_path / "routing_round1.npz"
    counts = np.full((4, 2, 8, 32), 3, dtype=np.int32)  # perfectly flat
    np.savez(tap, counts=counts, layers=np.arange(2))
    out = tmp_path / "flat.json"
    solve_layout.main(
        [
            "--routing",
            str(tap),
            "--shards",
            "4",
            "--slots",
            "9",
            "--min-gain-us",
            "1000",
            "--out",
            str(out),
        ]
    )
    doc = json.loads(out.read_text())
    assert doc["layers"] == {}
    assert all(not v["kept"] for v in doc["meta"]["predicted"].values())


def test_solver_is_deterministic(tmp_path):
    tap = tmp_path / "routing_round1.npz"
    _fake_tap(tap, seed=5)
    outs = []
    for i in range(2):
        out = tmp_path / f"l{i}.json"
        solve_layout.main(
            ["--routing", str(tap), "--shards", "4", "--slots", "9", "--out", str(out)]
        )
        outs.append(json.loads(out.read_text())["layers"])
    assert outs[0] == outs[1]
