# SPDX-License-Identifier: Apache-2.0
"""Solve a static expert layout (vllm_torchtpu/expert_placement.py) from
tapped routing counts.

    python solve_layout.py --routing routing_round1.npz [--routing ...]
        --shards 32 --slots 13 --pool 2 [--draft-slots 5 --draft-pool 2]
        [--min-gain-us 0] [--name dsv41_flash_dp32_c768] --out layout.json

Input: the routing tap's `counts [steps, layers, source ranks, experts]` (and
`draft_counts [steps, draft layers, source ranks, draft experts]`), from
scripts/dsv41_bench/routing_tap/convert.py. A wider EP group than the tap's is
PROXIED by pooling `pool` consecutive steps' rows: `pool` steps of 8 ranks at
288 rows are 8 * pool ranks at 288 rows, i.e. 32 ranks at 288 * 8 * pool / 32
rows; pool 2 stands for DP32 at 24 requests per rank (C768), pool 4 for DP32
at 48 (C1536). The pooled rows share prompts across the pooled "ranks", so
they may overstate correlation.

Policy: eplb_policy.rebalance_experts_global (DeepInfer's) with the rank-
uniqueness constraint; replication and packing minimize the window's summed
per-step makespan in rows. `--min-gain-us` leaves a layer at the checkpoint's
contiguous blocks when the tile model predicts less than that gain on the
slowest device. The file records, under "meta", the predicted slowest and
mean device per layer for the contiguous and the solved layout.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tile_model as tm  # noqa: E402
from eplb_policy import rank_lists, rebalance_experts_global  # noqa: E402


def pooled_rows(counts, pool):
    """[steps // pool, layers, experts] ROWBLK-aligned rows per logical expert,
    pooling `pool` consecutive steps' source ranks into one wider step."""
    counts = np.asarray(counts)
    steps = counts.shape[0] // pool * pool
    counts = counts[:steps]
    s, layers, src, experts = counts.shape
    pooled = counts.reshape(s // pool, pool, layers, src, experts)
    pooled = pooled.transpose(0, 2, 1, 3, 4).reshape(
        s // pool, layers, pool * src, experts
    )
    aligned = (pooled + tm.ROWBLK - 1) // tm.ROWBLK * tm.ROWBLK
    return aligned.sum(axis=2)


def solve_group(rows, shards, slots, *, min_gain_us=0.0, layer_ids=None):
    """rows [steps, layers, experts] -> ({layer: ranks}, report) for a group."""
    steps, layers, experts = rows.shape
    if experts % shards:
        raise ValueError(f"{experts} experts do not divide over {shards} ranks")
    if slots < experts // shards:
        raise ValueError(
            f"{slots} slots per rank hold fewer than {experts // shards} experts"
        )
    physical = shards * slots
    phy2log, _, _ = rebalance_experts_global(
        torch.as_tensor(rows, dtype=torch.float32), physical, slots
    )
    lists = rank_lists(phy2log, slots)
    layer_ids = list(range(layers)) if layer_ids is None else list(layer_ids)
    contiguous = tm.contiguous_ranks(experts, shards)
    out, report = {}, {}
    mean_rows = rows.mean(axis=0)
    for i, lid in enumerate(layer_ids):
        ranks = lists[i]
        reps = tm.replica_counts(ranks, experts)
        before = np.array([tm.predict(r, contiguous)[:2] for r in rows[:, i]]).mean(0)
        after = np.array([tm.predict(r, ranks, reps)[:2] for r in rows[:, i]]).mean(0)
        passes_before = np.array(
            [tm.predict(r, contiguous)[2:] for r in rows[:, i]]
        ).mean(0)
        passes_after = np.array(
            [tm.predict(r, ranks, reps)[2:] for r in rows[:, i]]
        ).mean(0)
        gain = float(before[0] - after[0])
        # The gate is opt-in: at 0 every solved layer is written, even one the
        # tile model prices a hair worse (the solver balances rows, not tiles).
        kept = min_gain_us <= 0 or gain >= min_gain_us
        report[str(lid)] = {
            "contiguous_slowest_us": round(float(before[0]), 1),
            "contiguous_mean_us": round(float(before[1]), 1),
            "layout_slowest_us": round(float(after[0]), 1),
            "layout_mean_us": round(float(after[1]), 1),
            "contiguous_slowest_passes": round(float(passes_before[0]), 1),
            "layout_slowest_passes": round(float(passes_after[0]), 1),
            "layout_mean_passes": round(float(passes_after[1]), 1),
            "replicated": sorted(int(e) for e in np.flatnonzero(reps > 1)),
            "hottest_mean_rows": int(mean_rows[i].max()),
            "kept": kept,
        }
        if kept:
            out[str(lid)] = [[int(e) for e in held] for held in ranks]
    return out, report


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--routing", type=Path, action="append", required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--slots", type=int, required=True)
    ap.add_argument("--pool", type=int, default=1)
    ap.add_argument("--draft-slots", type=int, default=None)
    ap.add_argument("--draft-pool", type=int, default=None)
    ap.add_argument("--min-gain-us", type=float, default=0.0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    taps = [np.load(p) for p in a.routing]
    layer_ids = taps[0]["layers"].tolist()
    for t in taps[1:]:
        if t["layers"].tolist() != layer_ids:
            raise ValueError("routing tables cover different layers")
    rows = np.concatenate([pooled_rows(t["counts"], a.pool) for t in taps])
    target, report = solve_group(
        rows, a.shards, a.slots, min_gain_us=a.min_gain_us, layer_ids=layer_ids
    )
    experts = rows.shape[-1]
    doc = {
        "experts": int(experts),
        "shards": a.shards,
        "slots": a.slots,
        "layers": target,
        "meta": {
            "name": a.name,
            "source": [str(p) for p in a.routing],
            "pool": a.pool,
            "steps": int(rows.shape[0]),
            "cost_model": "scripts/dsv41_bench/expert_layout/tile_model.py PARAMS",
            "min_gain_us": a.min_gain_us,
            "predicted": report,
        },
    }
    if a.draft_slots is not None:
        draft_pool = a.draft_pool or a.pool
        draft_rows = np.concatenate(
            [pooled_rows(t["draft_counts"], draft_pool) for t in taps]
        )
        draft_ids = taps[0]["draft_layers"].tolist()
        # The tap names the drafter's layers after the target's; the draft
        # model's own modules are layers 0..n-1.
        draft_ids = list(range(len(draft_ids)))
        draft, draft_report = solve_group(
            draft_rows,
            a.shards,
            a.draft_slots,
            min_gain_us=a.min_gain_us,
            layer_ids=draft_ids,
        )
        doc["draft"] = {
            "experts": int(draft_rows.shape[-1]),
            "slots": a.draft_slots,
            "layers": draft,
        }
        doc["meta"]["draft_pool"] = draft_pool
        doc["meta"]["draft_predicted"] = draft_report
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    slow_b = np.mean([r["contiguous_slowest_us"] for r in report.values()])
    slow_a = np.mean([r["layout_slowest_us"] for r in report.values()])
    mean_a = np.mean([r["layout_mean_us"] for r in report.values()])
    print(
        f"{a.out}: {len(target)}/{len(report)} layers placed, {a.shards} ranks "
        f"x {a.slots} slots over {experts} experts, {rows.shape[0]} pooled steps "
        f"(pool {a.pool}); tile-model "
        f"slowest device {slow_b:.1f} -> {slow_a:.1f} us (mean device {mean_a:.1f} us, "
        f"ratio {slow_a / mean_a:.3f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
