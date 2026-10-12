# SPDX-License-Identifier: Apache-2.0
"""Validate per-rank cache hits and preemptions for one-request-per-rank waves."""

import argparse
import json
import re
import statistics
from pathlib import Path


def per_rank(path: Path, metric: str) -> dict[int, float]:
    values = {}
    for line in path.read_text().splitlines():
        if not line.startswith(metric + "{"):
            continue
        labels, value = line.rsplit("}", 1)
        engine = re.search(r'\bengine="(\d+)"', labels)
        if engine is None:
            raise ValueError(f"Missing engine label for {metric}")
        rank = int(engine[1])
        if rank in values:
            raise ValueError(f"Multiple series for {metric}, rank {rank}")
        values[rank] = float(value.strip())
    return values


def validate(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    ranks = manifest["ranks"]
    if len(set(ranks)) != len(ranks):
        raise ValueError("Per-request attribution requires one request per rank")
    results = json.loads((root / "results.json").read_text())
    measured = [item for item in results if not item["warmup"]]
    if len(measured) < 5:
        raise ValueError("Need at least five unprofiled measured waves")
    checked = []
    for item in results:
        wave = item["extend"]
        label = wave["label"]
        counters = {}
        for metric, expected in (
            ("vllm:prefix_cache_hits_total", manifest["prefix"]),
            ("vllm:request_prefill_time_seconds_count", 1),
            ("vllm:num_preemptions_total", 0),
        ):
            before, after = (
                per_rank(root / f"{label}-{part}.prom", metric)
                for part in ("before", "after")
            )
            if before.keys() != after.keys() or not set(ranks) <= after.keys():
                raise ValueError(f"{label}: missing/inconsistent rank counters")
            delta = {rank: value - before[rank] for rank, value in after.items()}
            for rank, value in delta.items():
                target = expected if rank in ranks else 0
                if value != target:
                    raise ValueError(
                        f"{label}: {metric} rank {rank}: {value} != {target}"
                    )
            counters[metric] = delta
        if sorted(row["rank"] for row in wave["requests"]) != sorted(ranks):
            raise ValueError(f"{label}: response ranks mismatch")
        for row in wave["requests"]:
            usage = row["usage"]
            if usage["prompt_tokens"] != manifest["prefix"] + manifest["extend"]:
                raise ValueError(f"{label}: prompt length mismatch")
            if usage["completion_tokens"] != 1:
                raise ValueError(f"{label}: incomplete output")
        checked.append({"label": label, "warmup": item["warmup"], "counters": counters})
    waves = [item["extend"] for item in measured]
    return {
        "input_sha256": manifest["input_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "measured_waves": len(waves),
        "per_rank_checks": checked,
        "extend_s": [wave["server_prefill_mean_s"] for wave in waves],
        "new_tokens_per_s": [wave["new_tokens_per_s"] for wave in waves],
        "ttft_mean_s": [
            statistics.mean(row["ttft_s"] for row in wave["requests"]) for wave in waves
        ],
        "scope": "Cache affinity, completion and timing validation; not model accuracy",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = validate(args.root)
    (args.root / "validated.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
