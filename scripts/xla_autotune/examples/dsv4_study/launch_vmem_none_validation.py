#!/usr/bin/env python3
# Copyright 2026 Google LLC
"""Launch 5 parallel validation jobs for --xla_tpu_vmem_scavenging_mode=NONE."""

import json
import os
import subprocess
import time

from scripts.xla_autotune.cdk_orchestrator import CDKOrchestrator
from scripts.xla_autotune.config import CDKConfig, ModelConfig
from scripts.xla_autotune.utils import parse_cdk_job_id

TARGET_FLAG = "--xla_tpu_vmem_scavenging_mode=NONE"
RECIPE = "torchtpu-vllm-dsv4-pro-v7x-16-benchmark-10layers-baseline"
# NOTE: Adjust for your workspace
ARTIFACT_DIR = (
    "/mnt/workspace/xla_tuning/vllm-torchtpu/artifacts/dsv4_val_vmem_none_5runs"
)
NUM_RUNS = 5


def launch():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    orch = CDKOrchestrator(
        CDKConfig(recipe=RECIPE), ModelConfig(name="dsv4", model_path="/models")
    )
    ack_code = orch._ack_code
    print(f"Resolved CDK agent code: {ack_code}")
    print(f"Validation Target: {TARGET_FLAG} (5 runs)\n")

    jobs = []
    for i in range(1, NUM_RUNS + 1):
        trial_id = f"vmem_none_{i:02d}"
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
                f"EXTRA_LIBTPU_INIT_ARGS={TARGET_FLAG}",
                "-t",
                f"agent-jobs,xla-autotune,dsv4-vmem-none,trial-{trial_id}",
            ]
        )

        with open(os.path.join(trial_dir, "cdk_submit.cmd"), "w") as f:
            f.write(" ".join(cmd) + "\n")

        print(f"Submitting {trial_id} ({i}/{NUM_RUNS})...")
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
            jobs.append({"trial_id": trial_id, "job_id": job_id, "flag": TARGET_FLAG})
        else:
            print(f" -> [{trial_id}] Error: {res.stderr or res.stdout}")

        time.sleep(1)

    manifest_path = os.path.join(ARTIFACT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(jobs, f, indent=2)

    print(f"\nSuccessfully launched {len(jobs)}/{NUM_RUNS} jobs.")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    launch()
