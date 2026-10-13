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
"""Correlation-aware expert placement from sampled decode steps (profiler v2).

The decode MoE kernel (moe_lowtok) streams the weights of every distinct expert a
step hits on a core, and the post-MoE all-reduce waits for the slowest core. So
the per-layer decode cost tracks E[max over cores of distinct hit experts]. The
v2 profiler keeps a ring of real per-step hit sets (`dec_ring`), which carry the
co-activation that marginal histograms lose. This tool:

  * replays a placement (primaries + optional copies, kernel copy rule) on them;
  * solves primaries by swap local search on that replayed objective;
  * picks copies greedily by replayed gain;
  * writes the PR #1383 JSON.

The output is the TPU_EXPERT_PLACEMENT JSON read by
vllm_torchtpu/expert_placement.py. Usage:

  python3 solve_placement.py <profile_dir> --copies 32 --out table.json \
      [--baseline other_table.json ...] [--recording "<what was profiled>"]

<profile_dir> is searched recursively for the profiler's
moe_profile_rank_*_pid_*_latest.npz dumps (see profiler/).
"""

import argparse
import glob
import json
import os

import numpy as np

R, E = 8, 512
OWN = E // R


def load_samples(profile_dir):
    """Per layer [S, E] bool hit sets and [S] valid rows, from one worker's ring."""
    files = sorted(
        glob.glob(
            os.path.join(profile_dir, "**", "moe_profile_rank_*_pid_*_latest.npz"),
            recursive=True,
        )
    )
    if not files:
        raise SystemExit(f"no profiler dumps under {profile_dir}")
    # Every EP rank sees the same router input, so one worker is enough; check two.
    z = np.load(files[0])
    ring, rows, steps = z["dec_ring"], z["dec_ring_rows"], z["dec_steps"][:, 0]
    if len(files) > 1:
        z2 = np.load(files[1])
        same = np.array_equal(z2["dec_step_hit"], z["dec_step_hit"])
        print(f"workers agree on decode step hits: {same}")
    out = {}
    for layer in range(ring.shape[0]):
        keep = rows[layer] > 0
        if keep.sum() == 0:
            continue
        out[layer] = (ring[layer][keep].astype(bool), rows[layer][keep])
    per_layer = min(len(v[0]) for v in out.values())
    print(
        f"{len(files)} dumps; layers with samples: {len(out)};"
        f" samples/layer: {per_layer}; decode steps: {int(steps.max())}"
    )
    return out, z


def replay(owner, copies, hits):
    """Mean over steps of max-core distinct experts under the kernel copy rule.

    owner: [E] primary core; copies: list of (expert, core) in hot order.
    hits: [S, E] bool. Returns (mean max, per-step max).
    """
    copied = np.zeros(E, bool)
    for e, _ in copies:
        copied[e] = True
    onehot = np.zeros((E, R), np.int32)
    onehot[np.arange(E), owner] = 1
    ld = (hits & ~copied).astype(np.int32) @ onehot  # [S, R]
    for e, c in copies:  # hot order, fewer hit experts wins, ties to primary
        h = hits[:, e]
        a = owner[e]
        pick_c = h & (ld[:, c] < ld[:, a])
        ld[:, c] += pick_c
        ld[:, a] += h & ~pick_c
    mx = ld.max(1)
    return mx.mean(), mx


def solve_primaries(hits, init_owner, iters=40000, seed=0):
    """Swap local search on the mean max-core load (copies ignored)."""
    rng = np.random.default_rng(seed)
    owner = init_owner.copy()
    H = hits.astype(np.int32)
    onehot = np.zeros((E, R), np.int32)
    onehot[np.arange(E), owner] = 1
    ld = H @ onehot
    cur = ld.max(1).mean()
    freq = H.mean(0)
    # Bias proposals toward experts that are hit at all.
    prob = (freq + 1e-3) / (freq + 1e-3).sum()
    for _ in range(iters):
        a = rng.choice(E, p=prob)
        b = rng.integers(E)
        ra, rb = owner[a], owner[b]
        if ra == rb:
            continue
        d = H[:, b] - H[:, a]
        la, lb = ld[:, ra] + d, ld[:, rb] - d
        others = ld.copy()
        others[:, ra] = la
        others[:, rb] = lb
        new = others.max(1).mean()
        if new < cur - 1e-9:
            ld, cur = others, new
            owner[a], owner[b] = rb, ra
    return owner, cur


def solve_copies(hits, owner, n_copies, per_core):
    """The n most often hit experts get one copy each, hottest first, each on the
    free core (not its owner) with the best replayed mean max-core load."""
    copies, free = [], np.full(R, per_core)
    freq = hits.mean(0)
    cur = None
    for e in np.argsort(-freq)[:n_copies]:
        e = int(e)
        best = None
        for c in range(R):
            if c == owner[e] or free[c] == 0:
                continue
            v, _ = replay(owner, copies + [(e, c)], hits)
            if best is None or v < best[0]:
                best = (v, c)
        if best is None:  # only the owner has a free slot: move an earlier copy there
            j = next(
                i
                for i, (x, c) in enumerate(copies)
                if owner[x] != owner[e] and c != owner[e]
            )
            x, c = copies[j]
            copies[j] = (x, int(owner[e]))
            free[owner[e]] -= 1
            free[c] += 1
            best = (None, c)
        copies.append((e, best[1]))
        free[best[1]] -= 1
    cur, _ = replay(owner, copies, hits)
    return copies, cur


def owner_from_table(table, layer):
    own = table["owned_per_core"]
    owner = np.empty(E, int)
    for r, c in enumerate(table["layers"][str(layer)]["cores"]):
        owner[c[:own]] = r
    copies = [
        (int(e), int(r))
        for e, r in table["layers"][str(layer)].get("copies", {}).items()
    ]
    return owner, copies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile_dir")
    ap.add_argument(
        "--copies", type=int, default=0, help="copies per layer (multiple of 8)"
    )
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--baseline",
        action="append",
        default=[],
        help="tables to evaluate on the same held-out steps",
    )
    ap.add_argument("--layers", default="", help="comma list to restrict (debug)")
    ap.add_argument(
        "--recording",
        default="",
        help="free text stored in the table: the profiled job and workload",
    )
    args = ap.parse_args()

    samples, _ = load_samples(args.profile_dir)
    layers = (
        sorted(samples) if not args.layers else [int(x) for x in args.layers.split(",")]
    )
    per_core = args.copies // R
    baselines = {}
    for path in args.baseline:
        with open(path) as f:
            baselines[path] = json.load(f)
    result = {
        "shards": R,
        "experts": E,
        "owned_per_core": OWN,
        "slots_per_core": OWN + per_core,
        "copies": per_core > 0,
        "extra_slots": per_core,
        "algorithm": (
            "swap local search of the primaries on replayed mean max-core"
            " distinct experts over sampled decode steps (train half), then"
            " greedy copies by replayed gain"
        ),
        "recording": args.recording,
        "layers": {},
    }
    report = {"contig": [], "solved": []} | {os.path.basename(p): [] for p in baselines}
    for layer in layers:
        hits, rows = samples[layer]
        idx = np.random.default_rng(layer).permutation(len(hits))
        train, test = hits[idx[::2]], hits[idx[1::2]]
        contig = np.arange(E) // OWN
        owner, _ = solve_primaries(train, contig, iters=args.iters, seed=layer)
        copies, _ = (
            solve_copies(train, owner, args.copies, per_core)
            if per_core
            else ([], None)
        )
        cores = [list(np.flatnonzero(owner == r)) for r in range(R)]
        for e, c in copies:
            cores[c].append(e)
        result["layers"][str(layer)] = {"cores": [[int(x) for x in c] for c in cores]}
        if copies:
            result["layers"][str(layer)]["copies"] = {str(e): int(c) for e, c in copies}
        report["contig"].append(replay(contig, [], test)[0])
        report["solved"].append(replay(owner, copies, test)[0])
        for p, t in baselines.items():
            if str(layer) in t["layers"]:
                o, cp = owner_from_table(t, layer)
                report[os.path.basename(p)].append(replay(o, cp, test)[0])
        ideal = test.sum(1).mean() / R
        print(
            f"L{layer:02d} rows~{rows.mean():.1f} ideal {ideal:.2f} "
            + " ".join(f"{k}={v[-1]:.2f}" for k, v in report.items() if v),
            flush=True,
        )
    print(
        "mean over layers (held-out steps): "
        + " ".join(f"{k}={np.mean(v):.3f}" for k, v in report.items() if v)
    )
    with open(args.out, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
