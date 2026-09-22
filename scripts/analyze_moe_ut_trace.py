#!/usr/bin/env python3
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Summarize one-layer MoE XPlane captures without host dispatch overhead.

Requires xprof. Selects the latest capture per case; measures only TensorCore
XLA Ops and the enclosing module. Async lanes are not added a second time.
"""

import argparse
import csv
import gzip
import json
import statistics
from pathlib import Path


def duration(event):
    return (
        float(event.get("args", {}).get("device_duration_ps", event["dur"] * 1e6)) / 1e6
    )


def analyze(directory, expected_steps):
    from xprof.convert.raw_to_tool_data import xspace_to_tool_data

    captures = sorted((directory / "trace").rglob("*.xplane.pb"))
    assert captures, f"No capture in {directory}"
    converted = directory / "trace.json.gz"
    if converted.exists() and converted.stat().st_mtime > captures[-1].stat().st_mtime:
        with gzip.open(converted, "rt") as f:
            raw = f.read()
    else:
        raw, _ = xspace_to_tool_data([str(captures[-1])], "trace_viewer", {})
        with gzip.open(converted, "wt") as f:
            f.write(raw)
    events = json.loads(raw)["traceEvents"]
    devices = {
        e["pid"]: e["args"]["name"]
        for e in events
        if e.get("name") == "process_name"
        and "/device:TPU:" in e["args"]["name"]
        and "SparseCore" not in e["args"]["name"]
    }
    lanes = {
        (e["pid"], e["args"]["name"]): e["tid"]
        for e in events
        if e.get("name") == "thread_name"
    }
    assert devices, f"No device events in {captures[-1]}"
    rows, per_device = [], {}
    for pid, device in sorted(devices.items()):
        modules = sorted(
            [
                e
                for e in events
                if e.get("ph") == "X"
                and e.get("pid") == pid
                and e.get("tid") == lanes[pid, "XLA Modules"]
                and "run_local" in e["name"]
            ],
            key=lambda e: e["ts"],
        )
        assert len(modules) == expected_steps, (device, len(modules), expected_steps)
        ops = [
            e
            for e in events
            if e.get("ph") == "X"
            and e.get("pid") == pid
            and e.get("tid") == lanes[pid, "XLA Ops"]
        ]
        device_rows = []
        for step, module in enumerate(modules):
            selected = [
                e for e in ops if module["ts"] <= e["ts"] < module["ts"] + module["dur"]
            ]
            gmm1 = [
                e for e in selected if "gmm_v2" in e["name"] and "act_silu" in e["name"]
            ]
            gmm2 = [
                e for e in selected if "gmm_v2" in e["name"] and "act_None" in e["name"]
            ]
            dense = [e for e in selected if e["name"].startswith("dense_expert_moe-e_")]
            total = duration(module)
            row = {"device": device, "step": step, "total_us": total}
            if dense:
                assert len(dense) == 1 and not gmm1 and not gmm2
                routed = duration(dense[0])
                row["dense_expert_us"] = routed
            else:
                assert len(gmm1) == len(gmm2) == 1, (device, step, len(gmm1), len(gmm2))
                first, second = duration(gmm1[0]), duration(gmm2[0])
                routed = first + second
                row.update(gmm1_us=first, gmm2_us=second, gmm_us=routed)
            row.update(
                routed_kernel_us=routed,
                other_us=total - routed,
                op_sum_us=sum(duration(e) for e in selected),
                op_count=len(selected),
            )
            row["op_other_us"] = row["op_sum_us"] - routed
            row["module_gap_us"] = row["total_us"] - row["op_sum_us"]
            row["sort_count"] = sum(
                e.get("args", {}).get("hlo_category") == "sort" for e in selected
            )
            row["permute_kernel_count"] = sum(
                "onehot_unpermute" in e["name"] for e in selected
            )
            if dense:
                assert row["sort_count"] == row["permute_kernel_count"] == 0
            assert row["other_us"] >= 0
            device_rows.append(row)
            rows.append(row)
        keys = [k for k in device_rows[0] if k.endswith("_us")]
        assert len({r["sort_count"] for r in device_rows}) == 1
        per_device[device] = {
            k: statistics.median(r[k] for r in device_rows) for k in keys
        }
        per_device[device]["steps"] = len(device_rows)
        per_device[device]["sort_count_per_step"] = device_rows[0]["sort_count"]
        per_device[device]["mean"] = {
            k: statistics.mean(r[k] for r in device_rows) for k in keys
        }
    with (directory / "device_steps.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "source": str(captures[-1]),
        "scope": "one compiled MoE module, device duration",
        "statistics": "median across capture steps independently for each metric",
        "routed_kernel_scope": (
            "fused projection, activation, weighted combine"
            if "dense_expert_us" in rows[0]
            else "two GMMs only; permutation/combine are in other"
        ),
        "per_device": per_device,
        "median_max_rank_total_us": statistics.median(
            max(r["total_us"] for r in rows if r["step"] == step)
            for step in range(expected_steps)
        ),
    }
    (directory / "device_timing.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--steps", type=int, help="Override profile_steps recorded by the runner"
    )
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    steps = args.steps if args.steps is not None else summary["profile_steps"]
    if steps < 1:
        parser.error(
            "No profile steps recorded; run the benchmark with --profile-steps > 0"
        )
    for case, data in summary["cases"].items():
        data["device_timing"] = analyze(args.run / case, steps)
        print(case, json.dumps(data["device_timing"]["per_device"]))
    (args.run / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
