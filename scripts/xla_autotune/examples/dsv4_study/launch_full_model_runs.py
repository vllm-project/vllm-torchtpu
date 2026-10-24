#!/usr/bin/env python3
# Copyright 2026 Google LLC
"""Launch 2 baseline runs and 2 runs with --xla_tpu_vmem_scavenging_mode=NONE for full DeepSeek-V4 Pro."""

import json
import os
import subprocess
import time

from scripts.xla_autotune.cdk_orchestrator import CDKOrchestrator
from scripts.xla_autotune.config import CDKConfig, ModelConfig
from scripts.xla_autotune.utils import parse_cdk_job_id

RECIPE = "torchtpu-vllm-dsv4-pro-v7x-16-benchmark"
# NOTE: Adjust for your workspace
ARTIFACT_DIR = "/mnt/workspace/xla_tuning/vllm-torchtpu/artifacts/dsv4_full_model_eval"

RUNS = [
    {
        "trial_id": "full_baseline_01",
        "kind": "baseline",
        "flag": None,
        "tag": "dsv4-full-baseline",
    },
    {
        "trial_id": "full_baseline_02",
        "kind": "baseline",
        "flag": None,
        "tag": "dsv4-full-baseline",
    },
    {
        "trial_id": "full_vmem_none_01",
        "kind": "tuned",
        "flag": "--xla_tpu_vmem_scavenging_mode=NONE",
        "tag": "dsv4-full-vmem-none",
    },
    {
        "trial_id": "full_vmem_none_02",
        "kind": "tuned",
        "flag": "--xla_tpu_vmem_scavenging_mode=NONE",
        "tag": "dsv4-full-vmem-none",
    },
]


def launch_full():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    orch = CDKOrchestrator(
        CDKConfig(recipe=RECIPE), ModelConfig(name="dsv4", model_path="/models")
    )
    ack_code = orch._ack_code
    print(f"Resolved CDK agent code: {ack_code}")
    print(f"Launching 4 full-model runs using recipe: {RECIPE}\n")

    jobs = []
    for r in RUNS:
        trial_id = r["trial_id"]
        trial_dir = os.path.join(ARTIFACT_DIR, "trials", trial_id)
        os.makedirs(trial_dir, exist_ok=True)

        cmd = ["cdk"]
        if ack_code:
            cmd.append(f"--agent-code={ack_code}")
        cmd.extend(["job", "create", RECIPE])

        if r["flag"]:
            cmd.append(f"EXTRA_LIBTPU_INIT_ARGS={r['flag']}")

        cmd.extend(["-t", f"agent-jobs,xla-autotune,{r['tag']},trial-{trial_id}"])

        with open(os.path.join(trial_dir, "cdk_submit.cmd"), "w") as f:
            f.write(" ".join(cmd) + "\n")

        print(f"Submitting {trial_id} (kind={r['kind']}, flag={r['flag']})...")
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
                {
                    "trial_id": trial_id,
                    "kind": r["kind"],
                    "flag": r["flag"],
                    "job_id": job_id,
                }
            )
        else:
            print(f" -> [{trial_id}] Error: {res.stderr or res.stdout}")

        time.sleep(1)

    manifest_path = os.path.join(ARTIFACT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(jobs, f, indent=2)

    print(f"\nSuccessfully launched {len(jobs)}/4 full model jobs.")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    launch_full()
