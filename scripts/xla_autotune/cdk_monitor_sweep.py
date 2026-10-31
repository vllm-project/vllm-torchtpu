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
"""Model-agnostic parallel sweep monitor and leaderboard generator for CDK XLA autotuning."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time

from scripts.xla_autotune.config import TrialResult, load_tuning_config
from scripts.xla_autotune.utils import (
    generate_ranking_report,
    get_cdk_env,
    sync_and_parse_cdk_job,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("xla_sweep_monitor")


def monitor_sweep(
    artifact_dir: str,
    target_metric: str = "total_token_throughput",
    poll_interval_s: int = 60,
    run_once: bool = False,
):
    """Monitor all trials in submission_manifest.json until completion."""
    manifest_path = os.path.join(artifact_dir, "submission_manifest.json")
    if not os.path.exists(manifest_path):
        logger.error("Manifest not found: %s", manifest_path)
        return

    logger.info("Starting autotune sweep monitor for: %s", artifact_dir)

    while True:
        try:
            with open(manifest_path, encoding="utf-8") as f:
                all_items = json.load(f)
        except Exception as e:
            logger.warning("Could not read manifest: %s", e)
            all_items = []

        if not all_items:
            logger.warning("No items found in %s", manifest_path)
            if run_once:
                break
            time.sleep(poll_interval_s)
            continue

        # Fetch current CDK job states
        try:
            res = subprocess.run(
                ["cdk", "job", "list", "-a", "-n", "100", "-o", "json"],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
                env=get_cdk_env(),
            )
            jobs_by_id = {}
            if res.returncode == 0 and res.stdout.strip():
                jobs_list = json.loads(res.stdout)
                for j in jobs_list:
                    jobs_by_id[j.get("id")] = j
        except Exception as e:
            logger.warning("Failed to poll CDK job list: %s", e)
            jobs_by_id = {}

        results: list[TrialResult] = []
        pending_count = 0
        running_count = 0
        succeeded_count = 0
        failed_count = 0

        for item in all_items:
            trial_id = item["trial_id"]
            job_id = item["job_id"]
            flag = item.get("flag")
            kind = item.get("kind", "candidate" if flag else "baseline")
            trial_dir = os.path.join(artifact_dir, "trials", trial_id)
            os.makedirs(trial_dir, exist_ok=True)
            trial_json_path = os.path.join(trial_dir, f"{trial_id}.json")

            # Check if trial result already parsed and saved
            if os.path.exists(trial_json_path):
                try:
                    with open(trial_json_path, encoding="utf-8") as f:
                        data = json.load(f)
                    res_obj = TrialResult(
                        trial_id=data["trial_id"],
                        kind=data.get("kind", kind),
                        flag=data.get("flag"),
                        model=data.get("model", "unknown"),
                        target_metric=data.get("target_metric", target_metric),
                        target_value=data.get("target_value"),
                        success=data.get("success", False),
                        metrics=data.get("metrics", {}),
                        duration_sec=data.get("duration_sec", 0.0),
                        started_utc=data.get("started_utc", ""),
                        finished_utc=data.get("finished_utc", ""),
                        error=data.get("error", ""),
                        cdk_job_id=data.get("cdk_job_id", job_id),
                    )
                    results.append(res_obj)
                    if res_obj.success:
                        succeeded_count += 1
                    else:
                        failed_count += 1
                    continue
                except Exception as e:
                    logger.warning("Corrupt result file %s: %s", trial_json_path, e)

            # Query current state from CDK
            cdk_job = jobs_by_id.get(job_id)
            if not cdk_job:
                pending_count += 1
                continue

            status = cdk_job.get("job_status", "Pending")
            if status in ("Pending", "InQueue", "Starting"):
                pending_count += 1
            elif status == "Running":
                running_count += 1
            elif status in ("Succeeded", "Failed"):
                logger.info(
                    "Trial %s (Job %s) finished with status: %s",
                    trial_id,
                    job_id,
                    status,
                )
                metrics, err = sync_and_parse_cdk_job(job_id, trial_dir)
                val = metrics.get(target_metric)
                success = (status == "Succeeded") and (val is not None)

                res_obj = TrialResult(
                    trial_id=trial_id,
                    kind=kind,
                    flag=flag,
                    model="unknown",
                    target_metric=target_metric,
                    target_value=val,
                    success=success,
                    metrics=metrics,
                    started_utc=cdk_job.get("created_at", ""),
                    finished_utc=cdk_job.get("finished_at", ""),
                    error=err if not success else "",
                    cdk_job_id=job_id,
                )

                with open(trial_json_path, "w", encoding="utf-8") as f:
                    json.dump(res_obj.to_dict(), f, indent=2)

                results.append(res_obj)
                if success:
                    succeeded_count += 1
                else:
                    failed_count += 1

        # Calculate baseline average from successful baseline trials
        baseline_vals = [
            r.target_value
            for r in results
            if r.kind == "baseline" and r.success and r.target_value is not None
        ]
        baseline_avg = (
            (sum(baseline_vals) / len(baseline_vals)) if baseline_vals else None
        )

        # Generate summary and ranking leaderboard
        summary_data = {
            "evaluated_trials": len(results),
            "total_planned": len(all_items),
            "succeeded": succeeded_count,
            "failed": failed_count,
            "running": running_count,
            "pending": pending_count,
            "baseline_avg": baseline_avg,
            "results": [r.to_dict() for r in results],
        }
        with open(
            os.path.join(artifact_dir, "summary.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(summary_data, f, indent=2)

        ranking_path = os.path.join(artifact_dir, "ranking.md")
        generate_ranking_report(results, target_metric, ranking_path)

        logger.info(
            "Progress: %d/%d completed (%d succeeded, %d failed, %d running, %d pending) | Baseline avg: %s",
            len(results),
            len(all_items),
            succeeded_count,
            failed_count,
            running_count,
            pending_count,
            f"{baseline_avg:,.2f}" if baseline_avg else "Pending",
        )

        if len(results) == len(all_items) or run_once:
            logger.info("Sweep monitoring finished.")
            break

        time.sleep(poll_interval_s)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Model-agnostic sweep monitor for CDK XLA autotuning."
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="Artifact directory holding submission_manifest.json (overrides config).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to autotuning YAML config.",
    )
    parser.add_argument(
        "--target-metric",
        type=str,
        default="total_token_throughput",
        help="Benchmark target metric.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Poll interval in seconds (default: 60).",
    )
    parser.add_argument(
        "--once", action="store_true", help="Run a single status check and exit."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    artifact_dir = args.artifact_dir
    target_metric = args.target_metric
    poll_interval = args.poll_interval

    if args.config:
        cfg = load_tuning_config(args.config)
        target_metric = cfg.benchmark.target_metric or target_metric
        artifact_dir = artifact_dir or cfg.artifact_dir

    if not artifact_dir:
        logger.error(
            "No artifact directory specified. Provide --artifact-dir or a --config with 'artifact_dir'."
        )
        sys.exit(1)

    monitor_sweep(
        artifact_dir=artifact_dir,
        target_metric=target_metric,
        poll_interval_s=poll_interval,
        run_once=args.once,
    )


if __name__ == "__main__":
    main()
