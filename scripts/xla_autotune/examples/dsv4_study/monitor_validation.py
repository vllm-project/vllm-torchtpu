#!/usr/bin/env python3
# Copyright 2026 Google LLC
"""Monitor and aggregate metrics for the 10 validation runs."""

import json
import logging
import os
import subprocess
import time

from scripts.xla_autotune.utils import get_cdk_env, sync_and_parse_cdk_job

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("validation_monitor")

# NOTE: Adjust for your workspace
ARTIFACT_DIR = (
    "/mnt/workspace/xla_tuning/vllm-torchtpu/artifacts/dsv4_validation_10runs"
)
BASELINE_AVG = 193086.61  # Measured baseline throughput (tok/s)


def monitor_validation():
    manifest_path = os.path.join(ARTIFACT_DIR, "validation_manifest.json")
    if not os.path.exists(manifest_path):
        logger.error("Manifest not found: %s", manifest_path)
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    logger.info("Monitoring %d validation jobs...", len(manifest))

    while True:
        try:
            res = subprocess.run(
                ["cdk", "job", "list", "-a", "-n", "100", "-o", "json"],
                capture_output=True,
                text=True,
                check=False,
                env=get_cdk_env(),
            )
            jobs_by_id = (
                {j["id"]: j for j in json.loads(res.stdout)}
                if res.returncode == 0
                else {}
            )
        except Exception as e:
            logger.warning("Error fetching job list: %s", e)
            jobs_by_id = {}

        completed = []
        running = 0
        pending = 0

        for item in manifest:
            trial_id = item["trial_id"]
            job_id = item["job_id"]
            trial_dir = os.path.join(ARTIFACT_DIR, "trials", trial_id)
            json_path = os.path.join(trial_dir, f"{trial_id}.json")

            if os.path.exists(json_path):
                with open(json_path) as f:
                    d = json.load(f)
                completed.append(d)
                continue

            cdk_info = jobs_by_id.get(job_id, {})
            status = cdk_info.get("job_status", "Pending")

            if status == "Running":
                running += 1
            elif status in ("Pending", "InQueue"):
                pending += 1
            elif status in ("Succeeded", "Failed"):
                logger.info(
                    "Trial %s (%s) finished: %s. Fetching results...",
                    trial_id,
                    job_id,
                    status,
                )
                metrics, err = sync_and_parse_cdk_job(job_id, trial_dir, timeout=60)
                val = metrics.get("total_token_throughput") if metrics else None
                success = (status == "Succeeded") and (val is not None)

                data = {
                    "trial_id": trial_id,
                    "job_id": job_id,
                    "status": status,
                    "success": success,
                    "target_value": val,
                    "metrics": metrics,
                    "error": err if not success else "",
                }
                with open(json_path, "w") as f:
                    json.dump(data, f, indent=2)
                completed.append(data)

        # Generate validation report
        valid_vals = [
            c["target_value"]
            for c in completed
            if c.get("success") and c.get("target_value")
        ]
        report_lines = [
            "# XLA Autotune Validation Results (10 Runs)",
            "",
            f"- **Baseline Average Throughput**: `{BASELINE_AVG:,.2f} tok/s`",
            f"- **Completed Runs**: `{len(completed)}/{len(manifest)}` ({len(valid_vals)} succeeded)",
            "",
        ]

        if valid_vals:
            val_avg = sum(valid_vals) / len(valid_vals)
            val_min = min(valid_vals)
            val_max = max(valid_vals)
            diff_pct = ((val_avg - BASELINE_AVG) / BASELINE_AVG) * 100
            diff_str = f"+{diff_pct:.2f}%" if diff_pct >= 0 else f"{diff_pct:.2f}%"

            report_lines.extend(
                [
                    "### 📈 Validation Summary",
                    f"- **Validation Mean Throughput**: **`{val_avg:,.2f} tok/s`** ({diff_str} vs Baseline)",
                    f"- **Spread / Range**: `{val_min:,.2f}` – `{val_max:,.2f} tok/s`",
                    "",
                    "| Run | Job ID | Status | Throughput (tok/s) | vs Baseline |",
                    "|:---:|:---:|:---:|:---:|:---:|",
                ]
            )
            for c in completed:
                val = c.get("target_value")
                v_str = f"{val:,.2f}" if val else "N/A"
                if val:
                    pct = ((val - BASELINE_AVG) / BASELINE_AVG) * 100
                    p_str = f"+{pct:.2f}%" if pct >= 0 else f"{pct:.2f}%"
                else:
                    p_str = "N/A"
                report_lines.append(
                    f"| `{c['trial_id']}` | `{c['job_id']}` | `{c.get('status', 'Unknown')}` | {v_str} | {p_str} |"
                )
        else:
            report_lines.append("Validation runs in progress...")

        report_path = os.path.join(ARTIFACT_DIR, "validation_report.md")
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines) + "\n")

        logger.info(
            "Validation Progress: %d/%d completed (%d running, %d queued/pending)",
            len(completed),
            len(manifest),
            running,
            pending,
        )

        if len(completed) == len(manifest):
            logger.info("All 10 validation runs finished!")
            break

        time.sleep(60)


if __name__ == "__main__":
    monitor_validation()
