#!/usr/bin/env python3
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
"""Build a TPU_EXPERT_PLACEMENT table from MOE_EXPERT_COUNT_TAP_DIR snapshots.

    moe_expert_placement.py --shards 16 --layers 3:60 \\
        --out placement.json TAP_DIR [TAP_DIR ...]

Each rank's load is its last snapshot minus its `--skip`-th (1-based; 0 keeps
everything), which drops warmup and compile traffic; ranks are then summed.
Only the latest run in a directory counts: a restarted server numbers its
snapshots after the old ones and starts its `step` over, so the snapshots from
the last drop in `step` on are one run's.

A layer is as slow as its costliest rank, and a rank's cost is the sum of its
slots' costs (VisitModel: weight stream floor, full and short tiles). Each
layer is placed to minimise the EXPECTED straggler over steps sampled from the
measured per-expert means and variances: costliest-first onto the cheapest
rank, then pairwise swaps off the rank that is most often the straggler. A
layer whose expected straggler does not drop by --min-gain-us stays
contiguous. A placement only permutes a layer's experts over the ranks,
experts / shards to a rank; it never replicates one. The cost defaults are
fitted to the fused EP kernel on DeepSeek-V4-Pro; for another kernel or model,
pass its own --floor-us, --visit-us, --tile-us, --short-tile-us, --row-us,
--rung and --tile-rows.
"""

import argparse
import collections
import glob
import json
import os
import re

import numpy as np

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def load_counts(dirs, skip):
    """{layer name: (mean [E] per step, variance [E] per step)} summed over ranks."""
    by_rank = collections.defaultdict(list)
    for d in dirs:
        pattern = os.path.join(d, "**", "counts_ep*_*.npz")
        for path in glob.glob(pattern, recursive=True):
            m = re.search(r"counts_ep(\d+)_(\d+)\.npz$", path)
            if m:
                by_rank[(d, int(m.group(1)))].append((int(m.group(2)), path))
    if not by_rank:
        raise SystemExit(f"no counts_ep*_*.npz under {dirs}")
    mean, var, names = None, None, None
    for key, snaps in sorted(by_rank.items()):
        snaps.sort()
        # The latest run: everything from the last restart of `step` on.
        steps = [int(np.load(path)["step"]) for _, path in snaps]
        restarts = [i for i in range(1, len(steps)) if steps[i] <= steps[i - 1]]
        snaps = snaps[restarts[-1] :] if restarts else snaps
        if len(snaps) <= skip:
            raise SystemExit(f"{key}: {len(snaps)} snapshots, need more than {skip}")
        last = np.load(snaps[-1][1])
        c = last["counts"].astype(np.int64)
        if skip:
            c = c - np.load(snaps[skip - 1][1])["counts"].astype(np.int64)
        steps = np.maximum(c[:, 2, :1], 1)
        mu = c[:, 0] / steps
        v = np.maximum(c[:, 1] / steps - mu * mu, 0)
        layer_names = [str(n) for n in last["layer_names"]]
        if names is None:
            names, mean, var = layer_names, mu, v
        else:
            assert names == layer_names, key
            mean, var = mean + mu, var + v
    return {n: (mean[i], var[i]) for i, n in enumerate(names)}


class VisitModel:
    """Device time of one expert slot holding `rows` rows in one step (us).

    A visited slot costs max(floor, its tiles) plus a fixed visit cost: full
    tiles of tile_rows rows, a tail tile at the short rung's cost when it fits
    in it, and a small per-row term. Not visited (no rows): nothing. Defaults
    are a least-squares fit to 36 measured EP8 cells of the fused EP kernel
    (synthetic Zipf and replayed production routing, three placements each;
    RMSE 9 us of 440-820 us layers).
    """

    def __init__(self, floor, visit, full_tile, short_tile, per_row, rung, tile_rows):
        self.floor, self.visit = floor, visit
        self.full, self.short, self.per_row = full_tile, short_tile, per_row
        self.rung, self.tile_rows = rung, tile_rows

    def cost(self, rows):
        rows = np.asarray(rows, np.float64)
        full = np.floor(rows / self.tile_rows)
        tail = rows - full * self.tile_rows
        tail_cost = np.where(tail <= self.rung, self.short, self.full)
        comp = full * self.full + (tail > 0) * tail_cost + self.per_row * rows
        return np.where(rows > 0, np.maximum(self.floor, comp) + self.visit, 0.0)


def sample_rows(mu, var, steps, rng):
    """[experts, steps] per-step rows: normal at the measured mean and variance
    (at least Poisson), rounded and clipped at zero."""
    sd = np.sqrt(np.maximum(var, mu))
    noise = rng.standard_normal((mu.size, steps))
    return np.maximum(0, np.rint(mu[:, None] + sd[:, None] * noise))


def place(mu, var, shards, model, steps=64, sweeps=200, seed=0):
    """(per rank the experts it holds, expected straggler us).

    The objective is the mean over sampled steps of the costliest rank, so a
    placement is judged on the steps it will actually see rather than on the
    expected rows alone. Greedy costliest-first, then pairwise swaps off the
    rank that is most often the straggler while the objective improves.
    """
    rng = np.random.default_rng(seed)
    experts = mu.size
    width = experts // shards
    rows = sample_rows(mu, var, steps, rng)
    cost = {e: model.cost(rows[e]) for e in range(experts)}
    order = sorted(range(experts), key=lambda e: -cost[e].mean())
    held = [[] for _ in range(shards)]
    total = np.zeros((shards, steps))
    for e in order:
        free = [r for r in range(shards) if len(held[r]) < width]
        r = min(free, key=lambda r: total[r].mean())
        held[r].append(e)
        total[r] += cost[e]
    objective = total.max(0).mean()
    for _ in range(sweeps):
        hi = int(np.argmax(np.bincount(total.argmax(0), minlength=shards)))
        best = None
        for i, a in enumerate(held[hi]):
            for lo in range(shards):
                if lo == hi:
                    continue
                cand = list(range(len(held[lo])))
                cb = np.stack([cost[b] for b in held[lo]])
                new_hi = total[hi] - cost[a] + cb
                new_lo = total[lo] + cost[a] - cb
                rest = np.delete(total, [hi, lo], axis=0).max(0, initial=0.0)
                o = np.maximum(np.maximum(new_hi, new_lo), rest).mean(1)
                k = int(np.argmin(o))
                if o[k] < objective - 1e-9 and (best is None or o[k] < best[0]):
                    best = (o[k], i, lo, cand[k])
        if best is None:
            break
        objective, i, lo, j = best
        a, b = held[hi][i], held[lo][j]
        held[hi][i], held[lo][j] = b, a
        total[hi] += cost[b] - cost[a]
        total[lo] += cost[a] - cost[b]
    return held, objective


def expected_straggler(held, mu, var, model, steps=256, seed=1):
    """Mean over sampled steps of the costliest rank of a given placement."""
    rows = sample_rows(mu, var, steps, np.random.default_rng(seed))
    total = np.stack([sum(model.cost(rows[e]) for e in h) for h in held])
    return total.max(0).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", type=int, default=1, help="snapshots to drop")
    ap.add_argument(
        "--layers",
        required=True,
        help="first:last decoder layers to place (a drafter's MoE layers are "
        "numbered after the target's)",
    )
    ap.add_argument("--min-gain-us", type=float, default=5.0)
    ap.add_argument("--floor-us", type=float, default=4.86)
    ap.add_argument("--visit-us", type=float, default=1.725)
    ap.add_argument("--tile-us", type=float, default=11.1)
    ap.add_argument("--short-tile-us", type=float, default=7.16)
    ap.add_argument("--row-us", type=float, default=0.01)
    ap.add_argument(
        "--rung",
        type=int,
        default=48,
        help="the kernel's short tile height at the served shape "
        "(kernel.tile_height_rungs; 48 at DSv4-Pro EP16 T128)",
    )
    ap.add_argument("--tile-rows", type=int, default=128)
    a = ap.parse_args()
    model = VisitModel(
        a.floor_us,
        a.visit_us,
        a.tile_us,
        a.short_tile_us,
        a.row_us,
        a.rung,
        a.tile_rows,
    )
    counts = load_counts(a.dirs, a.skip)
    lo, hi = (int(x) for x in a.layers.split(":"))
    layers = {}
    for name, (mu, var) in counts.items():
        m = _LAYER_RE.search(name)
        if m and lo <= int(m.group(1)) <= hi:
            layers[int(m.group(1))] = (mu, var)
    built = {n: place(mu, var, a.shards, model) for n, (mu, var) in layers.items()}
    table, before, after = {}, [], []
    for n, (mu, var) in sorted(layers.items()):
        slots = mu.size // a.shards
        contiguous = [list(range(r * slots, (r + 1) * slots)) for r in range(a.shards)]
        before.append(expected_straggler(contiguous, mu, var, model))
        after.append(expected_straggler(built[n][0], mu, var, model))
        keep = after[-1] < before[-1] - a.min_gain_us
        print(
            f"layer {n:3d} rows/step {mu.sum():8.1f} hottest expert "
            f"{mu.max() / mu.mean():5.1f}x mean | expected straggler us: "
            f"contiguous {before[-1]:6.1f} placed {after[-1]:6.1f}"
            f"{'' if keep else ' -> not placed'}"
        )
        if keep:
            table[str(n)] = built[n][0]
        else:
            after[-1] = before[-1]
    experts = next(iter(layers.values()))[0].size
    out = {"experts": experts, "shards": a.shards, "layers": table}
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(
        f"wrote {a.out}: {len(table)} layers, expected straggler summed over "
        f"layers "
        f"{sum(before) / 1e3:.2f} -> {sum(after) / 1e3:.2f} ms/step"
    )


if __name__ == "__main__":
    main()
