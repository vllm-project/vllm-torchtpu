# Copyright 2026 Google LLC
"""Automated batch size sweep runner inside CDK container."""

import argparse
import subprocess
import sys
import time


def run_cmd(cmd: str):
    print(f"\n>>> Executing: {cmd}", flush=True)
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"FAILED with returncode {ret.returncode}", flush=True)
        sys.exit(ret.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Run Batch Size Sweep inside CDK Job")
    parser.add_argument(
        "--arm",
        type=str,
        required=True,
        help="Arm to evaluate (e.g. B0, B1, B2, B3, eagle3_base, eagle3_gaad)")
    parser.add_argument("--gamma",
                        type=int,
                        default=8,
                        help="Verification window gamma (spec_k = gamma - 1)")
    parser.add_argument("--batch-sizes",
                        type=int,
                        nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64, 128, 1024])
    parser.add_argument("--num-requests", type=int, default=50)
    parser.add_argument("--workload", type=str, default="character")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output-jsonl", type=str, default=None)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(
            f"Ignoring extra unrecognized arguments passed by template: {unknown}"
        )
    import os
    os.makedirs("/tmp/artifacts", exist_ok=True)
    if args.output_jsonl is None:
        args.output_jsonl = f"/tmp/artifacts/results_sweep_{args.arm}_gamma{args.gamma}.jsonl"

    print("=" * 80, flush=True)
    print(
        f"🚀 Starting Batch Sweep for Arm={args.arm} | Gamma={args.gamma} | Batches={args.batch_sizes}",
        flush=True)
    print("=" * 80, flush=True)

    for bs in args.batch_sizes:
        num_reqs = max(args.num_requests, bs)
        print("\n" + "-" * 70, flush=True)
        print(
            f" Running Arm {args.arm} | Gamma={args.gamma} | Batch Size = {bs} | Requests = {num_reqs}",
            flush=True)
        print("-" * 70, flush=True)

        model_arg = f"--model {args.model} " if args.model else ""
        cmd = (
            f"rm -f /tmp/libtpu_lockfile && "
            f"python3 /workspace/vllm-torchtpu/benchmarks/structured_jump/run_bench_jump.py "
            f"--arm {args.arm} "
            f"--gamma {args.gamma} "
            f"--workload {args.workload} "
            f"--num-requests {num_reqs} "
            f"--parallel {bs} "
            f"{model_arg}"
            f"--output-jsonl {args.output_jsonl}")
        run_cmd(cmd)
        time.sleep(1)

    print("\n" + "=" * 80, flush=True)
    print(
        f"✅ Completed Batch Sweep for Arm={args.arm}! Output written to {args.output_jsonl}",
        flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
