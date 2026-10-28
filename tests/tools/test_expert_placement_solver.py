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
"""Expert-placement tools: cost model, solver, counts file and validator.

The tools live in tools/expert_placement (not a package); these tests put
that directory on sys.path. Routing is synthetic and seeded.
"""

import copy
import json
import pathlib
import sys

import numpy as np
import pytest

TOOLS = pathlib.Path(__file__).resolve().parents[2] / "tools" / "expert_placement"
sys.path.insert(0, str(TOOLS))

import aggregate_counts  # noqa: E402
import placement_model as pm  # noqa: E402
import solve_placement  # noqa: E402
import validate_placement  # noqa: E402

SMALL = pm.Shape(experts=64, ranks=8, topk=4)
GLM = pm.Shape(experts=256, ranks=16, topk=8)
MIX = {
    "decode_tokens": {64: 1.0, 128: 3.0, 96: 2.0},
    "decode_weight": 11.2,
    "prefill_steps": [[4096, 0.79], [440, 0.41]],
}


def _shares(shape, seed, alpha=0.5):
    return np.random.default_rng(seed).dirichlet(np.full(shape.experts, alpha))


def _cost(**over):
    return pm.CostModel.from_json(dict(MIX, **over))


@pytest.fixture(scope="module")
def solved_small():
    out = {}
    for seed in (1, 2, 3):
        p_dec, p_pre = _shares(SMALL, seed), _shares(SMALL, seed + 100)
        out[seed] = pm.solve_layer(p_dec, p_pre, SMALL, _cost())
    return out


def test_tables_are_permutations(solved_small):
    for seed, r in solved_small.items():
        for name, cores in r["tables"].items():
            assert pm.is_permutation(SMALL, cores), (seed, name)


def test_joint_never_worse_than_linear_qlpt_lpt(solved_small):
    for seed, r in solved_small.items():
        t = r["terms"]
        for start in ("linear", "qlpt", "lpt"):
            assert t["joint"]["J"] <= t[start]["J"] + 1e-9, (seed, start)
        # q-LPT optimizes the decode term: no worse than linear there.
        assert (
            t["qlpt"]["decode_emax_active"] <= t["linear"]["decode_emax_active"] + 1e-9
        )


@pytest.mark.parametrize(
    "over",
    [
        {"prefill_steps": []},
        {"decode_weight": 0.0},
    ],
    ids=["decode-only", "prefill-only"],
)
def test_variants_never_worse_than_linear(over):
    r = pm.solve_layer(
        _shares(SMALL, 7), _shares(SMALL, 8), SMALL, _cost(**over), max_iter=60
    )
    assert pm.is_permutation(SMALL, r["tables"]["joint"])
    assert r["terms"]["joint"]["J"] <= r["terms"]["linear"]["J"] + 1e-9


def test_weight_scale_does_not_change_the_search():
    """Weights as raw step counts (the counts' scale) or as rates give the
    same tables: the search tolerance is relative."""
    p_dec, p_pre = _shares(SMALL, 11), _shares(SMALL, 12)
    rate = _cost()
    raw = _cost(
        decode_weight=MIX["decode_weight"] * 3600,
        prefill_steps=[[t, w * 3600] for t, w in MIX["prefill_steps"]],
    )
    a = pm.solve_layer(p_dec, p_pre, SMALL, rate)
    b = pm.solve_layer(p_dec, p_pre, SMALL, raw)
    assert a["tables"]["joint"] == b["tables"]["joint"]
    assert a["joint_swaps"] == b["joint_swaps"] > 0


@pytest.mark.parametrize(
    "shape,over",
    [
        (pm.Shape(8, 1, 2), {}),
        (SMALL, {"decode_min_tokens": 10**6, "prefill_steps": []}),
    ],
    ids=["one-rank", "no-placement-dependent-term"],
)
def test_degenerate_layers_do_not_crash(shape, over):
    r = pm.solve_layer(_shares(shape, 1), _shares(shape, 2), shape, _cost(**over))
    assert pm.is_permutation(shape, r["tables"]["joint"])


def test_uniform_routing_stays_valid_and_level():
    u = np.ones(SMALL.experts)
    r = pm.solve_layer(u, u, SMALL, _cost(), max_iter=5)
    assert pm.is_permutation(SMALL, r["tables"]["joint"])
    assert abs(r["terms"]["joint"]["J"] - r["terms"]["linear"]["J"]) < 1e-6


def test_hot_block_is_spread():
    """Positive control: the hottest experts sit in rank 0's linear block;
    the table spreads them and cuts the busiest-rank cost."""
    p = np.full(GLM.experts, 1.0)
    p[: GLM.slots] = 40.0
    r = pm.solve_layer(p, p, GLM, _cost(), max_iter=40)
    joint = r["tables"]["joint"]
    assert pm.is_permutation(GLM, joint)
    hot_per_rank = [sum(e < GLM.slots for e in c) for c in joint]
    assert max(hot_per_rank) <= 2, hot_per_rank
    assert r["terms"]["joint"]["J"] < 0.8 * r["terms"]["linear"]["J"]


def test_analytic_matches_monte_carlo(solved_small):
    cost = _cost()
    for seed, r in solved_small.items():
        p_dec, p_pre = _shares(SMALL, seed), _shares(SMALL, seed + 100)
        lm = pm.LayerModel(p_dec, p_pre, SMALL, cost)
        mc = pm.monte_carlo(
            r["tables"],
            p_dec,
            p_pre,
            SMALL,
            cost,
            np.random.default_rng(seed),
            6000,
            1500,
        )
        for name, cores in r["tables"].items():
            an = lm.terms(np.array(cores))
            # Decode bins compress P(T): a small common bias. Prefill: the
            # normal approximation of the pairs per rank.
            assert abs(an["decode_emax_active"] - mc[name]["decode_emax_active"]) < 0.06
            for a, m in zip(an["prefill_emax_pairs"], mc[name]["prefill_emax_pairs"]):
                assert abs(a / m - 1) < 0.01, (seed, name, a, m)


def test_decode_bins_cover_distribution():
    hist = {4 * s: float(s % 7 + 1) for s in range(1, 33)}
    cost = pm.CostModel(decode_tokens=hist, decode_bins=8)
    t, w = cost.decode_bins_t()
    assert len(t) <= 8 and abs(w.sum() - 1) < 1e-12
    tv, p = cost.decode_probs()
    assert abs((t * w).sum() - (tv * p).sum()) < 1.0  # mean tokens kept
    cut = pm.CostModel(decode_tokens=hist, decode_min_tokens=65)
    t2, w2 = cut.decode_bins_t()
    share_le_64 = sum(v for k, v in hist.items() if k <= 64) / sum(hist.values())
    assert t2.min() > 64 and abs(w2.sum() - (1 - share_le_64)) < 1e-9


def test_cost_model_json_and_workload_defaults(tmp_path):
    path = tmp_path / "cost.json"
    path.write_text(
        json.dumps({"prefill_ns_per_pair": 31, "decode_tokens": {"128": 1}})
    )
    cost = pm.CostModel.load(str(path))
    assert cost.decode_tokens == {128: 1.0} and cost.prefill_ns_per_pair == 31
    assert pm.CostModel.from_json(cost.to_json()) == cost
    with pytest.raises(ValueError, match="unknown cost model keys"):
        pm.CostModel.from_json({"prefil_us": 1.0})
    wl = pm.workload_from_counts(np.full(8, 100.0), np.full(8, 4000.0), 10, 2, topk=4)
    assert wl["decode_tokens_per_step"] == 20 and wl["prefill_tokens_per_step"] == 4000
    res = pm.CostModel().resolved(wl)
    assert res.decode_tokens == {20: 1.0} and res.decode_weight == 10
    assert res.prefill_steps == [[4000, 2.0]]
    # Given fields win over the counts.
    assert pm.CostModel(decode_tokens={64: 1.0}).resolved(wl).decode_tokens == {64: 1.0}


# --------------------------------------------------------------------------
# Command line: counts file -> solve -> validate with the loader's checks.
# --------------------------------------------------------------------------

LAYERS = (3, 4, 5)
NUM_LAYERS = 6  # layers 0-2 dense, 3-5 routed


def _counts_file(tmp_path, shape, seed=0, prefill_steps=3):
    rng = np.random.default_rng(seed)
    e, k = shape.experts, shape.topk
    # Counter classes 0-2: decode, prefill, prefill_decode_rows.
    by_class = np.zeros((NUM_LAYERS, 3, e), dtype=np.int64)
    steps = np.zeros((NUM_LAYERS, 4), dtype=np.int64)
    for n in LAYERS:
        by_class[n, 0] = rng.multinomial(k * 50 * 128, _shares(shape, seed + n))
        p_pre = _shares(shape, seed + n + 50)
        by_class[n, 1] = rng.multinomial(k * prefill_steps * 4096, p_pre)
        steps[n] = (50, prefill_steps, 0, 0)
    agg = {
        "decode": by_class[:, 0],
        "prefill": by_class[:, 1] + by_class[:, 2],
        "steps": steps,
        "step_names": [
            "decode_steps",
            "prefill_steps",
            "unknown_steps",
            "pad_mismatch_rows",
        ],
        "layers": np.array(LAYERS),
        "num_experts": e,
        "topk": k,
        "workers": 1,
        "checks": {"ranks_identical": False},
        "sources": ["synthetic"],
    }
    path = tmp_path / "counts.npz"
    aggregate_counts.save(str(path), agg)
    return path


def _validate(table, shape, *extra):
    return validate_placement.main(
        [
            str(table),
            "--ranks",
            str(shape.ranks),
            "--experts",
            str(shape.experts),
            "--first-moe-layer",
            str(LAYERS[0]),
            "--num-layers",
            str(NUM_LAYERS),
            *extra,
        ]
    )


def test_cli_solve_then_validate(tmp_path, capsys):
    counts = _counts_file(tmp_path, SMALL)
    table = tmp_path / "table.json"
    assert (
        solve_placement.main(
            [
                "--counts",
                str(counts),
                "--ranks",
                str(SMALL.ranks),
                "--out",
                str(table),
                "--max-iter",
                "40",
                "--model",
                "synthetic",
            ]
        )
        == 0
    )
    t = json.loads(table.read_text())
    assert (t["experts"], t["shards"], len(t["layers"]["3"][0])) == (64, 8, 8)
    assert sorted(map(int, t["layers"])) == list(LAYERS)
    assert t["meta"]["workload_from_counts"]["decode_tokens_per_step"] == 128
    assert t["meta"]["workload_from_counts"]["prefill_tokens_per_step"] == 4096
    report = json.loads((tmp_path / "table.json.report.json").read_text())
    assert report["analytic_cost_saved_vs_linear_pct"]["joint"] >= 0
    sha = (tmp_path / "table.json.sha256").read_text().split()[0]
    capsys.readouterr()
    assert _validate(table, SMALL) == 0
    out = capsys.readouterr().out
    assert f"table_sha256={sha[:16]}" in out and "OK" in out


def test_cli_decode_only_counts(tmp_path):
    """A decode-only profile (for example a disaggregated decode node) still
    solves every layer, on the decode term alone."""
    counts = _counts_file(tmp_path, SMALL, prefill_steps=0)
    table = tmp_path / "table.json"
    assert (
        solve_placement.main(
            [
                "--counts",
                str(counts),
                "--ranks",
                "8",
                "--out",
                str(table),
                "--max-iter",
                "20",
            ]
        )
        == 0
    )
    t = json.loads(table.read_text())
    assert sorted(map(int, t["layers"])) == list(LAYERS)
    assert t["meta"]["cost_model"]["prefill_steps"] == []
    assert _validate(table, SMALL) == 0


def test_cost_weights_share_one_scale():
    wl = {
        "decode_steps": 10,
        "prefill_steps": 2,
        "decode_tokens_per_step": 64,
        "prefill_tokens_per_step": 4096,
    }
    with pytest.raises(ValueError, match="together"):
        pm.CostModel(decode_weight=11.2).resolved(wl)
    with pytest.raises(ValueError, match="together"):
        pm.CostModel(prefill_steps=[[4096, 0.8]]).resolved(wl)


def test_cli_validate_with_hf_config(tmp_path):
    table = tmp_path / "t.json"
    table.write_text(json.dumps(_table(SMALL)))
    cfg = {
        "n_routed_experts": 64,
        "first_k_dense_replace": 3,
        "num_hidden_layers": 6,
        "n_group": 1,
        "topk_group": 1,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    args = [str(table), "--ranks", "8", "--hf-config", str(tmp_path / "config.json")]
    assert validate_placement.main(args) == 0
    (tmp_path / "config.json").write_text(
        json.dumps(dict(cfg, n_group=8, topk_group=4))
    )
    assert validate_placement.main(args) == 1  # grouped top-k picks by contiguous id


def _table(shape, seed=0):
    rng = np.random.default_rng(seed)
    layers = {}
    for n in LAYERS:
        perm = rng.permutation(shape.experts).tolist()
        s = shape.slots
        layers[str(n)] = [perm[r * s : (r + 1) * s] for r in range(shape.ranks)]
    return {"experts": shape.experts, "shards": shape.ranks, "layers": layers}


NEGATIVE = {
    "mtp_layer_listed": lambda t: t["layers"].__setitem__(
        str(NUM_LAYERS), copy.deepcopy(t["layers"]["3"])
    ),
    "duplicate_expert": lambda t: t["layers"]["4"][0].__setitem__(
        0, t["layers"]["4"][1][0]
    ),
}


@pytest.mark.parametrize("name", sorted(NEGATIVE))
def test_validator_refuses(tmp_path, name):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_table(SMALL)))
    assert _validate(good, SMALL) == 0
    t = _table(SMALL)
    NEGATIVE[name](t)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(t))
    assert _validate(bad, SMALL) == 1


DEPLOYMENT = {
    "experts": SMALL.experts,
    "ranks": SMALL.ranks,
    "slots": SMALL.slots,
    "first_moe_layer": LAYERS[0],
    "num_layers": NUM_LAYERS,
}


def test_validate_refuses_another_deployment_shape():
    four = _table(pm.Shape(experts=64, ranks=4, topk=4))
    with pytest.raises(ValueError, match="needs 8 x 8 over 64"):
        validate_placement.validate(json.dumps(four).encode(), "t", **DEPLOYMENT)


def test_validate_takes_a_copies_table():
    t = _table(SMALL)
    t.update(copy_min_tokens=64, copy_share={})
    for key, cores in t["layers"].items():
        copied = [cores[(r + 1) % SMALL.ranks][0] for r in range(SMALL.ranks)]
        t["layers"][key] = [c + [copied[r]] for r, c in enumerate(cores)]
        t["copy_share"][key] = {str(e): 0.25 for e in copied}
    raw = json.dumps(t).encode()
    table = validate_placement.validate(raw, "t", **DEPLOYMENT)
    assert (table.owned, table.slots, table.copy_min_tokens) == (8, 9, 64)
    assert table.shares[3] == {int(e): 0.25 for e in t["copy_share"]["3"]}
    with pytest.raises(ValueError, match="8 ranks x 8 primaries over 64"):
        validate_placement.validate(raw, "t", **{**DEPLOYMENT, "slots": 9})


def test_validator_argument_errors(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_table(SMALL)))
    for argv in (
        [
            str(good),
            "--ranks",
            "0",
            "--experts",
            "64",
            "--first-moe-layer",
            "3",
            "--num-layers",
            "6",
        ],
        [str(good), "--ranks", "8"],
    ):
        with pytest.raises(SystemExit) as e:
            validate_placement.main(argv)
        assert e.value.code == 2, argv
