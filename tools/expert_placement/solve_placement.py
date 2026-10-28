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
"""Solve a static expert placement table from per-layer routing counts.

Reads ``counts.npz`` (aggregate_counts.py) and writes a ``TPU_EXPERT_PLACEMENT``
table: per layer, a permutation of the experts over ``--ranks`` ranks that
lowers the expected MoE time of the busiest rank (placement_model.py). Per
layer the search starts from q-LPT (decode term only) and from LPT on the
prefill pairs, runs pairwise swaps on the full objective, and keeps the best
(never worse than vLLM's linear placement on the objective).

Outputs: the table (compact JSON, sorted keys), ``<table>.sha256``, and a
report ``<table>.report.json`` with each layer's
objective terms for linear, q-LPT, LPT and the table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import placement_model as pm  # noqa: E402


def parse_layers(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out += range(int(lo), int(hi) + 1)
        elif part:
            out.append(int(part))
    return sorted(set(out))


def load_counts(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    steps = {str(n): d["steps"][:, i] for i, n in enumerate(d["step_names"])}
    return {
        "decode": d["decode"].astype(np.float64),
        "prefill": d["prefill"].astype(np.float64),
        "decode_steps": steps["decode_steps"].astype(np.float64),
        "prefill_steps": steps["prefill_steps"].astype(np.float64),
        "layers": [int(x) for x in d["layers"]],
        "num_experts": int(d["num_experts"]),
        "topk": int(d["topk"]),
    }


def workload(counts: dict, layers: list[int], topk: int) -> dict:
    """Mean tokens per step over ``layers`` (every layer sees every step)."""
    n = max(len(layers), 1)
    return pm.workload_from_counts(
        counts["decode"][layers].sum(axis=0) / n,
        counts["prefill"][layers].sum(axis=0) / n,
        float(counts["decode_steps"][layers].mean()) if layers else 0.0,
        float(counts["prefill_steps"][layers].mean()) if layers else 0.0,
        topk,
    )


def build_table(shape: pm.Shape, cores: dict[int, list[list[int]]], meta: dict) -> dict:
    return {
        "experts": shape.experts,
        "shards": shape.ranks,
        "layers": {
            str(n): [[int(e) for e in c] for c in cores[n]] for n in sorted(cores)
        },
        "meta": meta,
    }


def write_table(path: str, table: dict) -> str:
    """Writes compact JSON with sorted keys and ``<path>.sha256``; returns
    the sha256 of the bytes written (what the loader logs)."""
    raw = (json.dumps(table, sort_keys=True, separators=(",", ":")) + "\n").encode()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(raw)
    sha = hashlib.sha256(raw).hexdigest()
    with open(path + ".sha256", "w") as f:
        f.write(f"{sha}  {os.path.basename(path)}\n")
    return sha


def _solve_one(args):
    layer, p_dec, p_pre, shape, cost_json, max_iter = args
    cost = pm.CostModel.from_json(cost_json)
    t0 = time.time()
    r = pm.solve_layer(p_dec, p_pre, shape, cost, max_iter=max_iter)
    r["secs"] = time.time() - t0
    return layer, r


def solve(
    counts: dict,
    shape: pm.Shape,
    cost: pm.CostModel,
    layers: list[int],
    procs: int = 1,
    max_iter: int = 200,
) -> dict:
    """Per-layer results of ``pm.solve_layer`` keyed by layer."""
    jobs = []
    for n in layers:
        p_dec = counts["decode"][n]
        p_pre = counts["prefill"][n]
        # A layer without samples of one step type uses the other's shares
        # (that term's weight is then 0 unless the cost model gives one).
        if p_dec.sum() <= 0:
            p_dec = p_pre
        if p_pre.sum() <= 0:
            p_pre = p_dec
        jobs.append((n, p_dec, p_pre, shape, cost.to_json(), max_iter))
    if procs > 1:
        with mp.get_context("fork").Pool(procs) as pool:
            return dict(pool.map(_solve_one, jobs, chunksize=1))
    return dict(map(_solve_one, jobs))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--counts", required=True, help="aggregate_counts.py output")
    ap.add_argument("--ranks", type=int, required=True, help="EP ranks")
    ap.add_argument("--out", required=True, help="table .json to write")
    ap.add_argument("--cost", default="", help="cost model JSON (CostModel fields)")
    ap.add_argument("--layers", default="", help="e.g. 3-77 (default: all with data)")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--model", default="", help="model name recorded in meta")
    args = ap.parse_args(argv)

    counts = load_counts(args.counts)
    topk = counts["topk"]
    shape = pm.Shape(counts["num_experts"], args.ranks, topk)
    with_data = {
        n
        for n in counts["layers"]
        if counts["decode"][n].sum() + counts["prefill"][n].sum() > 0
    }
    layers = parse_layers(args.layers) if args.layers else sorted(with_data)
    empty = [n for n in layers if n not in with_data]
    layers = [n for n in layers if n not in empty]
    if not layers:
        print("no layer with routing counts to solve", file=sys.stderr)
        return 1
    wl = workload(counts, layers, topk)
    cost = pm.CostModel.load(args.cost).resolved(wl)
    t0 = time.time()
    res = solve(counts, shape, cost, layers, args.procs, args.max_iter)
    cores = {n: res[n]["tables"]["joint"] for n in layers}
    bad = [n for n, c in cores.items() if not pm.is_permutation(shape, c)]
    if bad:
        raise SystemExit(f"internal error: layers {bad} are not permutations")
    totals = {
        name: float(sum(res[n]["terms"][name]["J"] for n in layers))
        for name in ("linear", "qlpt", "lpt", "joint")
    }
    saved = {k: pm.pct_saved(totals["linear"], v) for k, v in totals.items()}
    meta = {
        "model": args.model,
        "tool": "tools/expert_placement/solve_placement.py",
        "algorithm": "per-layer pairwise swap search on the busiest-rank MoE "
        "cost, started from q-LPT and LPT",
        "cost_model": cost.to_json(),
        "workload_from_counts": wl,
        "topk": topk,
        "layers_without_data": empty,
        "analytic_cost_saved_vs_linear_pct": round(saved["joint"], 4),
        "solved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    table = build_table(shape, cores, meta)
    sha = write_table(args.out, table)
    report = {
        "table": os.path.abspath(args.out),
        "sha256": sha,
        "analytic_cost_total": totals,
        "analytic_cost_saved_vs_linear_pct": saved,
        "per_layer": {
            str(n): {
                k: res[n][k]
                for k in ("terms", "joint_start", "joint_iters", "joint_swaps", "secs")
            }
            for n in layers
        },
    }
    rpath = args.out + ".report.json"
    with open(rpath, "w") as f:
        json.dump(report, f, indent=1)
    print(
        f"wrote {args.out}: {len(layers)} layers, {shape.ranks} ranks x "
        f"{shape.slots} slots over {shape.experts} experts, sha256 {sha} "
        f"(loader marker table_sha256={sha[:16]})"
    )
    print(
        "analytic busiest-rank MoE cost saved vs linear: "
        + "  ".join(f"{k} {v:+.2f}%" for k, v in saved.items())
        + f"  ({time.time() - t0:.0f} s; report {rpath})"
    )
    if empty:
        print(f"layers without data, left linear (not listed): {empty}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
