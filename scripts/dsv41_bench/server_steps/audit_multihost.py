# SPDX-License-Identifier: Apache-2.0
"""Audit both hosts after the DP16 CI launcher has collected their artifacts."""

import json
import sys
from pathlib import Path

from bench_cached_extend import assert_no_compilation
from bench_decode import rank_results, summarize


def audit(root: Path) -> dict:
    paths = [root / "benchmark", root / "worker/benchmark"]
    manifests = [json.loads((p / "manifest.json").read_text()) for p in paths]
    timings = [json.loads((p / "startup.json").read_text()) for p in paths]
    if not all(t["completed"] for t in timings):
        raise ValueError("Both hosts must complete")
    if timings[0]["formal_status"] != "passed" or not timings[1]["shutdown_requested"]:
        raise ValueError("Expected a completed head and orderly worker shutdown")
    for index, manifest in enumerate(manifests):
        if manifest["host_rank"] != index or manifest["scenario"] != "decode_dp16":
            raise ValueError("Unexpected host rank or scenario")
        for key, expected in {
            "compile_sizes": [288, 2048],
            "cache_mode": "disabled",
            "engram_hbm": "all",
            "engram_dp_shared_memory": False,
            "moe_kernel": "push",
            "moe_push_sc_prepush": 2,
            "moe_push_tc_exchange": True,
            "moe_preprocess": "topology",
            "allocator": "glibc",
            "reclaim_compile_memory": True,
        }.items():
            if manifest[key] != expected:
                raise ValueError(f"Host {index} differs from CI recipe: {key}")
    for key in ("torchtpu", "vllm", "model_weights", "model_config_sha256", "packages"):
        if manifests[0]["identity"][key] != manifests[1]["identity"][key]:
            raise ValueError(f"Hosts disagree on {key}")
    hostnames = [m["hostname"] for m in manifests]
    if len(set(hostnames)) != 2:
        raise ValueError("Expected two distinct serving hosts")
    fixed = paths[0] / "fixed"
    rounds = json.loads((fixed / "manifest.json").read_text())["rounds"]
    if rounds < 1:
        raise ValueError("No formal rounds")
    logs = [(p / "server.log").read_text() for p in paths]
    if not all(logs):
        raise ValueError("Missing server logs")
    for index in range(1, rounds + 1):
        records = json.loads((fixed / f"round-{index}-steps.json").read_text())
        # Recheck rank coverage and strict shape invariants from raw records.
        report = summarize(records["results"], "decode_dp16")
        saved = json.loads((fixed / f"round-{index}-summary.json").read_text())
        if saved["warmup"] or report["decode_throughput"] != saved["decode_throughput"]:
            raise ValueError("Formal summary does not match its raw records")
        ranks = rank_results(records["results"])
        for host_rank, hostname in enumerate(hostnames):
            local = sorted(
                (r for r in ranks if r["hostname"] == hostname), key=lambda r: r["rank"]
            )
            if [r["rank"] for r in local] != list(
                range(host_rank * 8, (host_rank + 1) * 8)
            ):
                raise ValueError("Unexpected host/rank placement")
            # Each log is audited against that host's own wall clock.
            intervals = [r["audit_wall_interval_ns"] for r in local]
            if any(low >= high for low, high in intervals):
                raise ValueError("Invalid host-local audit interval")
            assert_no_compilation(logs[host_rank], intervals)
    return {
        "completed": True,
        "hosts": hostnames,
        "ranks": list(range(16)),
        "rounds": rounds,
        "commit": manifests[0]["identity"]["torchtpu"]["commit"],
    }


if __name__ == "__main__":
    root = Path(sys.argv[1])
    result = audit(root)
    (root / "benchmark/multihost-audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
