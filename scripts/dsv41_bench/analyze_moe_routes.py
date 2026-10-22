# SPDX-License-Identifier: Apache-2.0
"""Analyze same-forward EP load from exact native-router replays.

The 64/128-row counts describe the current FP4 height ladder. They are not a
latency model; communication, weight fetch and overlap also determine latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_replay(path: Path) -> list[dict]:
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta["purpose"] != "native_tpu_router_replay_of_actual_capture":
        raise ValueError("Expected native TPU router replay")
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("Replay checksum mismatch")
    records = []
    with np.load(path, allow_pickle=False) as data:
        for layer in meta["routers"]:
            ids = data[layer["key"] + "_ids"]
            weights = data[layer["key"] + "_weights"]
            experts = layer["global_num_experts"]
            topk = layer["topk"]
            if (
                ids.shape != (meta["row"]["rows"], topk)
                or not np.issubdtype(ids.dtype, np.integer)
                or weights.shape != ids.shape
                or not np.isfinite(weights).all()
                or np.any(weights < 0)
                or np.any(ids < 0)
                or np.any(ids >= experts)
                or np.any(np.diff(np.sort(ids, axis=1), axis=1) <= 0)
            ):
                raise ValueError("Invalid routed indices/weights")
            records.append(
                {
                    **layer,
                    "rank": meta["rank"],
                    "sample": meta["sample"],
                    "forward_id": meta["row"]["forward_id"],
                    "counts": np.bincount(ids.ravel(), minlength=experts),
                }
            )
    return records


def aggregate(records: list[dict], source_row_alignment: int = 2) -> list[dict]:
    # Match host.build_routing_tables_sharded: each source/expert run is
    # rounded to ROWBLK before the runs are concatenated into an expert slab.
    if source_row_alignment < 1 or source_row_alignment & (source_row_alignment - 1):
        raise ValueError("Source row alignment must be a positive power of two")
    groups = defaultdict(list)
    for r in records:
        groups[(r["forward_id"], r["module"])].append(r)
    result = []
    for (forward, module), group in sorted(groups.items()):
        ref = group[0]
        ep = ref["ep_size"]
        experts = ref["global_num_experts"]
        if (
            len(group) != ep
            or {r["ep_rank"] for r in group} != set(range(ep))
            or len({r["rank"] for r in group}) != ep
        ):
            raise ValueError(
                f"Incomplete or duplicate same-forward EP group: {forward}, {module}"
            )
        fields = (
            "global_num_experts",
            "ep_size",
            "local_experts",
            "topk",
            "scoring_fn",
            "renormalize",
            "routed_scaling_factor",
            "bias_sha256",
        )
        if any(any(r[k] != ref[k] for k in fields) for r in group):
            raise ValueError(
                f"Inconsistent router/ownership metadata: {forward}, {module}"
            )
        if ep * ref["local_experts"] != experts:
            raise ValueError("Only equal contiguous expert ownership is supported")
        source_counts = np.stack([r["counts"] for r in group])
        aligned = (
            (source_counts + source_row_alignment - 1)
            // source_row_alignment
            * source_row_alignment
        ).sum(axis=0)
        result.append(
            {
                "forward_id": forward,
                "module": module,
                "ep_size": ep,
                "counts": source_counts.sum(axis=0),
                "aligned_counts": aligned,
                "source_row_alignment": source_row_alignment,
                "logits_sha256_by_rank": {r["rank"]: r["logits_sha256"] for r in group},
            }
        )
    return result


def placement(cost: np.ndarray, shards: int) -> np.ndarray:
    """Equal expert slots per shard, greedy descending cost; deterministic ties."""
    cost = np.asarray(cost, dtype=np.float64)
    if (
        cost.ndim != 1
        or cost.size % shards
        or not np.isfinite(cost).all()
        or np.any(cost < 0)
    ):
        raise ValueError("Expected finite nonnegative costs and equal shard slots")
    limit = cost.size // shards
    owner = np.full(cost.size, -1, dtype=np.int32)
    loads = np.zeros(shards)
    slots = np.zeros(shards, dtype=np.int32)
    for expert in np.lexsort((np.arange(cost.size), -cost)):
        available = np.flatnonzero(slots < limit)
        shard = min(available, key=lambda i: (loads[i], slots[i], i))
        owner[expert] = shard
        loads[shard] += cost[expert]
        slots[shard] += 1
    return owner


def load_summary(
    counts: np.ndarray, aligned_counts: np.ndarray, owners: np.ndarray, shards: int
) -> dict:
    if counts.shape != owners.shape or set(owners.tolist()) != set(range(shards)):
        raise ValueError("Placement must assign each expert to a valid shard")
    fields = {
        "routed_rows": counts,
        "aligned_rows": aligned_counts,
        "active_experts": counts > 0,
        "tiles_128": (aligned_counts + 127) // 128,
        "computed_rows_64": ((aligned_counts + 63) // 64) * 64,
    }
    result = {}
    for name, values in fields.items():
        loads = np.bincount(owners, weights=values, minlength=shards).astype(np.int64)
        result[name] = {
            "per_shard": loads.tolist(),
            "max": int(loads.max()),
            "mean": float(loads.mean()),
            "max_over_mean": float(loads.max() / max(1, loads.mean())),
        }
    return result


def summarize(steps: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in steps:
        groups[r["module"]].append(r)
    result = []
    for module, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["forward_id"])
        train = rows[0]
        ep = train["ep_size"]
        n = train["counts"].size
        if len(rows) < 2:
            raise ValueError("Need a later held-out forward per layer")
        contiguous = np.repeat(np.arange(ep), n // ep)
        costs = {
            "routed_rows": train["counts"],
            "tiles_128": (train["aligned_counts"] + 127) // 128,
            "computed_rows_64": ((train["aligned_counts"] + 63) // 64) * 64,
        }
        plans = {k: placement(v, ep) for k, v in costs.items()}
        round_robin = np.arange(n) % ep
        entries = []
        for r in rows:
            if r["ep_size"] != ep or r["counts"].size != n:
                raise ValueError("Expert topology changed")
            counts = r["counts"]
            aligned = r["aligned_counts"]
            entries.append(
                {
                    "forward_id": r["forward_id"],
                    "split": "training" if r is train else "held_out",
                    "total_routes": int(counts.sum()),
                    "source_row_alignment": r["source_row_alignment"],
                    "zero_experts": int(np.count_nonzero(counts == 0)),
                    "expert_rows_percentiles": dict(
                        zip(
                            ("min", "p50", "p90", "p95", "max"),
                            np.percentile(counts, [0, 50, 90, 95, 100]).tolist(),
                            strict=True,
                        )
                    ),
                    "contiguous": load_summary(counts, aligned, contiguous, ep),
                    "round_robin": load_summary(counts, aligned, round_robin, ep),
                    "fixed_training_placements": {
                        k: load_summary(counts, aligned, v, ep)
                        for k, v in plans.items()
                    },
                    "counts": counts.tolist(),
                    "aligned_counts": aligned.tolist(),
                }
            )
        result.append(
            {
                "module": module,
                "training_forward": train["forward_id"],
                "placements": {k: v.tolist() for k, v in plans.items()},
                "steps": entries,
            }
        )
    return {
        "scope": (
            "Same-forward native-router replay. Work counts only, not measured "
            "speedup. Placements are offline proposals, not runtime changes."
        ),
        "layers": result,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("replays", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--source-row-alignment",
        type=int,
        default=2,
        help="Current native routing-plan ROWBLK; align each source/expert run",
    )
    a = p.parse_args()
    paths = sorted(a.replays.glob("rank-*-sample-*.npz"))
    if not paths:
        p.error("No replay captures")
    records = [r for path in paths for r in read_replay(path)]
    report = summarize(aggregate(records, a.source_row_alignment))
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "captures": len(paths),
                "layers": len(report["layers"]),
                "scope": report["scope"],
            }
        )
    )


if __name__ == "__main__":
    main()
