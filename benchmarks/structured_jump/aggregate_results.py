#!/usr/bin/env python3
# Copyright 2026 Google LLC
"""Consolidates benchmark sweep results across Arm 0, Arm 1, Arm 2, Arm 3."""

import json
from pathlib import Path


def load_results(jsonl_path: Path):
    if not jsonl_path.exists():
        return {}
    records = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            bs = data.get("parallel")
            records[bs] = data
    return records


def main():
    b0 = load_results(Path("/tmp/results_sweep_B0_gamma8.jsonl"))
    b1 = load_results(Path("/tmp/results_sweep_B1_gamma8.jsonl"))
    b2 = load_results(Path("/tmp/results_sweep_B2_gamma8.jsonl"))
    b3 = load_results(Path("/tmp/results_sweep_B3_gamma8.jsonl"))

    all_bs = sorted(
        list(
            set(
                list(b0.keys()) + list(b1.keys()) + list(b2.keys()) +
                list(b3.keys()))))

    print("=" * 120)
    print(
        "FOUR-ARM COMPREHENSIVE BENCHMARK MATRIX: Meta-Llama-3.1-8B-Instruct (Character JSON Schema)"
    )
    print("=" * 120)
    print(
        f"{'BS':<5} | {'B0 (Vanilla)':<15} | {'B1 (Raw Spec)':<15} | {'B2 (Composite)':<15} | {'B3 (Pure GaaD)':<15} | {'Speedup B3/B0':<14} | {'Speedup B3/B2':<14}"
    )
    print("-" * 120)

    for bs in all_bs:
        r0 = b0.get(bs, {})
        r1 = b1.get(bs, {})
        r2 = b2.get(bs, {})
        r3 = b3.get(bs, {})

        l0 = f"{r0['latency']:.2f}s" if "latency" in r0 else "Pending"
        l1 = f"{r1['latency']:.2f}s" if "latency" in r1 else "Pending"
        l2 = f"{r2['latency']:.2f}s" if "latency" in r2 else "Pending"
        l3 = f"{r3['latency']:.2f}s" if "latency" in r3 else "Pending"

        sp_b3_b0 = f"{r0['latency'] / r3['latency']:.2f}x" if (
            "latency" in r0 and "latency" in r3) else "-"
        sp_b3_b2 = f"{r2['latency'] / r3['latency']:.2f}x" if (
            "latency" in r2 and "latency" in r3) else "-"

        print(
            f"{bs:<5} | {l0:<15} | {l1:<15} | {l2:<15} | {l3:<15} | {sp_b3_b0:<14} | {sp_b3_b2:<14}"
        )


if __name__ == "__main__":
    main()
