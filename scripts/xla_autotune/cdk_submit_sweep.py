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
"""Model-agnostic batch submission tool for XLA autotune candidates on CDK."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any

from scripts.xla_autotune.cdk_orchestrator import CDKOrchestrator
from scripts.xla_autotune.config import CDKConfig, ModelConfig, load_tuning_config
from scripts.xla_autotune.utils import load_flags, parse_cdk_job_id

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("xla_submit_all")


def submit_sweep(
    recipe: str,
    flags_file: str,
    artifact_dir: str,
    baseline_runs: int = 2,
    tags: list[str] | None = None,
    extra_recipe_params: dict[str, str] | None = None,
    agent_code: str | None = None,
    dry_run: bool = False,
) -> str:
    """Submit baseline runs followed by all candidate flags to CDK.

    Returns:
        Path to the generated submission_manifest.json file.
    """
    os.makedirs(artifact_dir, exist_ok=True)
    tags = tags or ["agent-jobs", "xla-autotune"]
    extra_params = extra_recipe_params or {}

    cdk_cfg = CDKConfig(
        recipe=recipe,
        agent_code=agent_code,
        extra_recipe_params=extra_params,
        tags=tags,
    )
    orch = CDKOrchestrator(cdk_cfg, ModelConfig(name="autotune_model"), dry_run=dry_run)
    ack_code = orch._ack_code
    logger.info("Using CDK recipe: %s", recipe)
    logger.info("Resolved CDK agent code: %s", ack_code)

    # 1. Prepare candidates from flags file
    candidates: list[str] = []
    if os.path.exists(flags_file):
        candidates = load_flags(flags_file)
    else:
        logger.warning(
            "Flags file '%s' not found or empty. Only baseline runs will be submitted.",
            flags_file,
        )

    # 2. Plan jobs: baselines first, then candidates
    planned_jobs = []
    for i in range(1, baseline_runs + 1):
        planned_jobs.append(
            {
                "trial_id": f"baseline_{i:02d}",
                "kind": "baseline",
                "flag": None,
            }
        )
    for i, flag in enumerate(candidates):
        planned_jobs.append(
            {
                "trial_id": f"cand_{i + 1:03d}",
                "kind": "candidate",
                "flag": flag,
            }
        )

    logger.info(
        "Planned %d total jobs: %d baseline(s), %d candidate flag(s)",
        len(planned_jobs),
        baseline_runs,
        len(candidates),
    )

    manifest_path = os.path.join(artifact_dir, "submission_manifest.json")
    submitted_jobs: list[dict[str, Any]] = []

    # If manifest exists from a previous invocation, reload to allow resuming
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as f:
                submitted_jobs = json.load(f)
            logger.info(
                "Loaded %d previously submitted jobs from existing manifest.",
                len(submitted_jobs),
            )
        except Exception as e:
            logger.warning("Could not read existing manifest: %s", e)

    already_submitted = {j["trial_id"]: j for j in submitted_jobs}

    for item in planned_jobs:
        trial_id = item["trial_id"]
        kind = item["kind"]
        flag = item["flag"]
        trial_dir = os.path.join(artifact_dir, "trials", trial_id)
        os.makedirs(trial_dir, exist_ok=True)

        if trial_id in already_submitted:
            logger.info(
                "[%s] Already recorded in manifest (Job: %s). Skipping.",
                trial_id,
                already_submitted[trial_id].get("job_id"),
            )
            continue

        cmd = ["cdk"]
        if ack_code:
            cmd.append(f"--agent-code={ack_code}")
        cmd.extend(["job", "create", recipe])

        # Extra template params
        for k, v in extra_params.items():
            cmd.append(f"{k}={v}")

        if flag:
            cmd.append(f"EXTRA_LIBTPU_INIT_ARGS={flag}")

        trial_tags = list(tags) + [f"trial-{trial_id}"]
        cmd.extend(["-t", ",".join(trial_tags)])

        with open(
            os.path.join(trial_dir, "cdk_submit.cmd"), "w", encoding="utf-8"
        ) as f:
            f.write(" ".join(cmd) + "\n")

        if dry_run:
            logger.info("[DRY-RUN] Would submit [%s]: %s", trial_id, " ".join(cmd))
            submitted_jobs.append(
                {
                    "trial_id": trial_id,
                    "kind": kind,
                    "flag": flag,
                    "job_id": f"dry-run-{trial_id}",
                }
            )
            continue

        logger.info(
            "Submitting [%s] (kind=%s, flag=%s)...", trial_id, kind, flag or "None"
        )
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=orch._get_env(),
        )

        with open(
            os.path.join(trial_dir, "cdk_submit.log"), "w", encoding="utf-8"
        ) as f:
            f.write(res.stdout)
            if res.stderr:
                f.write("\n--- STDERR ---\n" + res.stderr)

        job_id = parse_cdk_job_id(res.stdout)

        if job_id:
            logger.info(" -> [%s] Created CDK job: %s", trial_id, job_id)
            submitted_jobs.append(
                {
                    "trial_id": trial_id,
                    "kind": kind,
                    "flag": flag,
                    "job_id": job_id,
                }
            )
        else:
            logger.error(
                " -> [%s] Submission failed: %s", trial_id, res.stderr or res.stdout
            )

        # Write manifest incrementally so progress is preserved
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(submitted_jobs, f, indent=2)

        time.sleep(1)

    logger.info(
        "Submission complete. %d jobs written to %s", len(submitted_jobs), manifest_path
    )
    return manifest_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch submit XLA autotune candidates and baselines to CDK in a model-agnostic manner."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to autotuning YAML config file (e.g. configs/example_config.yaml).",
    )
    parser.add_argument(
        "--recipe", type=str, help="CDK recipe name (overrides config)."
    )
    parser.add_argument(
        "--flags-file",
        type=str,
        help="Path to text file containing candidate XLA flags.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        help="Directory to save sweep trial manifests and logs.",
    )
    parser.add_argument(
        "--baseline-runs",
        type=int,
        default=2,
        help="Number of baseline trials (default: 2).",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default="agent-jobs,xla-autotune",
        help="Comma-separated CDK tags.",
    )
    parser.add_argument(
        "--agent-code", type=str, default=None, help="CDK agent ACK code override."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without submitting.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    recipe = args.recipe
    flags_file = args.flags_file or "flags.txt"
    artifact_dir = args.artifact_dir or "artifacts/xla_sweep"
    baseline_runs = args.baseline_runs
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    extra_params = {}

    if args.config:
        cfg = load_tuning_config(args.config)
        if cfg.cdk:
            recipe = recipe or cfg.cdk.recipe
            tags = cfg.cdk.tags or tags
            extra_params = cfg.cdk.extra_recipe_params
        flags_file = args.flags_file or cfg.flag_list_file
        artifact_dir = args.artifact_dir or cfg.artifact_dir
        baseline_runs = cfg.baseline_runs

    if not recipe:
        logger.error(
            "No CDK recipe specified. Provide --recipe or a --config with a 'cdk' block."
        )
        sys.exit(1)

    submit_sweep(
        recipe=recipe,
        flags_file=flags_file,
        artifact_dir=artifact_dir,
        baseline_runs=baseline_runs,
        tags=tags,
        extra_recipe_params=extra_params,
        agent_code=args.agent_code,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
