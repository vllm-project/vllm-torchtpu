#!/usr/bin/env python3
# Copyright 2026 Google LLC
"""Launch 10 parallel validation jobs with the optimal XLA flag configuration."""

import json
import os
import subprocess
import time

from scripts.xla_autotune.cdk_orchestrator import CDKOrchestrator
from scripts.xla_autotune.config import CDKConfig, ModelConfig
from scripts.xla_autotune.utils import parse_cdk_job_id

BEST_EXTRA_FLAGS = (
    "--xla_tpu_vmem_scavenging_mode=NONE "
    "--xla_tpu_enable_async_collective_merger=false "
    "--xla_tpu_enable_brkga_latency_hiding_scheduler=true "
    "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0 "
    "--xla_tpu_all_to_all_with_different_output_addresses=true"
)

RECIPE = "torchtpu-vllm-dsv4-pro-v7x-16-benchmark-10layers-baseline"
# NOTE: Adjust for your workspace
ARTIFACT_DIR = (
    "/mnt/workspace/xla_tuning/vllm-torchtpu/artifacts/dsv4_validation_10runs"
)
NUM_RUNS = 10


def launch_validation():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    orch = CDKOrchestrator(
        CDKConfig(recipe=RECIPE), ModelConfig(name="dsv4", model_path="/models")
    )
    ack_code = orch._ack_code
    print(f"Resolved CDK agent code: {ack_code}")
    print(f"Optimal Flags: {BEST_EXTRA_FLAGS}\n")

    jobs = []
    for i in range(1, NUM_RUNS + 1):
        trial_id = f"val_{i:02d}"
        trial_dir = os.path.join(ARTIFACT_DIR, "trials", trial_id)
        os.makedirs(trial_dir, exist_ok=True)

        cmd = ["cdk"]
        if ack_code:
            cmd.append(f"--agent-code={ack_code}")
        cmd.extend(
            [
                "job",
                "create",
                RECIPE,
                f"EXTRA_LIBTPU_INIT_ARGS={BEST_EXTRA_FLAGS}",
                "-t",
                f"agent-jobs,xla-autotune,dsv4-validation,trial-{trial_id}",
            ]
        )

        with open(os.path.join(trial_dir, "cdk_submit.cmd"), "w") as f:
            f.write(" ".join(cmd) + "\n")

        print(f"Submitting validation run {i}/{NUM_RUNS} ({trial_id})...")
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=orch._get_env(),
        )

        with open(os.path.join(trial_dir, "cdk_submit.log"), "w") as f:
            f.write(res.stdout)
            if res.stderr:
                f.write("\n--- STDERR ---\n" + res.stderr)

        job_id = parse_cdk_job_id(res.stdout)

        if job_id:
            print(f" -> [{trial_id}] Created job: {job_id}")
            jobs.append(
                {"trial_id": trial_id, "job_id": job_id, "flags": BEST_EXTRA_FLAGS}
            )
        else:
            print(f" -> [{trial_id}] Error: {res.stderr or res.stdout}")

        time.sleep(1)

    manifest_path = os.path.join(ARTIFACT_DIR, "validation_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(jobs, f, indent=2)

    print(f"\nSuccessfully launched {len(jobs)}/{NUM_RUNS} validation jobs.")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    launch_validation()
