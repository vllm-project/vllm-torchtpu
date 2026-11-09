# SPDX-License-Identifier: Apache-2.0
"""Check a static expert layout file and predict its balance on tapped routing.

    python check_layout.py layout.json [--routing routing_round0.npz] [--pool 2]

Exit 0 when the file parses under vllm_torchtpu.expert_placement (every
expert held, no rank holds one twice, equal slots). With `--routing`, replay
the counts (pooled as solve_layout.py does) and print per layer and overall
the tile model's slowest and mean device for the checkpoint's contiguous
blocks and for the layout, the predicted `moe_lb` (slowest rank's rows over
the mean rank's), and the hottest expert's share.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tile_model as tm  # noqa: E402
from solve_layout import pooled_rows  # noqa: E402


def _rows_per_rank(rows_e, ranks, reps):
    per_rank = np.zeros(len(ranks))
    for d, held in enumerate(ranks):
        for e in held:
            per_rank[d] += rows_e[e] / reps[e]
    return per_rank


def evaluate(group, rows, layer_ids):
    """Per layer: predicted metrics for the contiguous and the placed layout."""
    experts, shards = group.experts, group.shards
    contiguous = tm.contiguous_ranks(experts, shards)
    out = {}
    for i, lid in enumerate(layer_ids):
        ranks = group.ranks(lid)
        if ranks is None:
            ranks = contiguous
        reps = tm.replica_counts(ranks, experts)
        before = np.array([tm.predict(r, contiguous) for r in rows[:, i]]).mean(0)
        after = np.array([tm.predict(r, ranks, reps) for r in rows[:, i]]).mean(0)
        lb_before = np.mean(
            [
                (
                    _rows_per_rank(r, contiguous, np.ones(experts)).max()
                    / max(r.sum() / shards, 1)
                )
                for r in rows[:, i]
            ]
        )
        lb_after = np.mean(
            [
                (_rows_per_rank(r, ranks, reps).max() / max(r.sum() / shards, 1))
                for r in rows[:, i]
            ]
        )
        out[lid] = {
            "placed": group.ranks(lid) is not None,
            "contiguous": before,
            "layout": after,
            "moe_lb_contiguous": lb_before,
            "moe_lb_layout": lb_after,
            "hottest_share": float(rows[:, i].mean(0).max() / rows[:, i].mean(0).sum()),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("layout", type=Path)
    ap.add_argument("--routing", type=Path, action="append", default=[])
    ap.add_argument("--pool", type=int, default=1)
    ap.add_argument("--draft-pool", type=int, default=None)
    a = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from vllm_torchtpu import expert_placement

    text = a.layout.read_text()
    placement = expert_placement.parse(json.loads(text), str(a.layout))
    t = placement.target
    print(
        f"{a.layout}: ok; target {t.experts} experts x {t.shards} ranks "
        f"x {t.slots} slots "
        f"({t.physical} physical), {len(t.layers)} layers placed"
        + (
            f"; draft {placement.draft.experts} x {placement.draft.shards} x "
            f"{placement.draft.slots}, {len(placement.draft.layers)} layers"
            if placement.draft
            else ""
        )
    )
    if not a.routing:
        return 0
    taps = [np.load(p) for p in a.routing]
    layer_ids = taps[0]["layers"].tolist()
    rows = np.concatenate([pooled_rows(x["counts"], a.pool) for x in taps])
    res = evaluate(t, rows, layer_ids)
    print(
        f"{'layer':>5} {'placed':>6} {'slow_us':>8} {'->':>2} {'slow_us':>8} "
        f"{'mean_us':>8} {'passes':>7} {'->':>2} {'passes':>7} {'lb':>6} {'->':>2} "
        f"{'lb':>6} {'hot%':>5}"
    )
    for lid, r in res.items():
        b, af = r["contiguous"], r["layout"]
        print(
            f"{lid:>5} {str(r['placed']):>6} {b[0]:8.1f} {'':>2} {af[0]:8.1f} "
            f"{af[1]:8.1f} "
            f"{b[2]:7.1f} {'':>2} {af[2]:7.1f} {r['moe_lb_contiguous']:6.2f} {'':>2} "
            f"{r['moe_lb_layout']:6.2f} {100 * r['hottest_share']:5.1f}"
        )
    slow_b = np.mean([r["contiguous"][0] for r in res.values()])
    slow_a = np.mean([r["layout"][0] for r in res.values()])
    mean_a = np.mean([r["layout"][1] for r in res.values()])
    lb_a = np.mean([r["moe_lb_layout"] for r in res.values()])
    lb_b = np.mean([r["moe_lb_contiguous"] for r in res.values()])
    print(
        f"overall ({rows.shape[0]} pooled steps, pool {a.pool}): slowest device "
        f"{slow_b:.1f} -> {slow_a:.1f} us, mean device {mean_a:.1f} us "
        f"(slowest/mean {slow_a / mean_a:.3f}); moe_lb {lb_b:.2f} -> {lb_a:.2f}"
    )
    if placement.draft is not None and "draft_counts" in taps[0]:
        pool = a.draft_pool or a.pool
        drows = np.concatenate([pooled_rows(x["draft_counts"], pool) for x in taps])
        dres = evaluate(placement.draft, drows, list(range(drows.shape[1])))
        for lid, r in dres.items():
            print(
                f"draft {lid}: slowest {r['contiguous'][0]:.1f} -> "
                f"{r['layout'][0]:.1f} us, mean {r['layout'][1]:.1f}; "
                f"moe_lb {r['moe_lb_contiguous']:.2f} -> {r['moe_lb_layout']:.2f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
