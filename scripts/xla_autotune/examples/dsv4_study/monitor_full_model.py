#!/usr/bin/env python3
# Copyright 2026 Google LLC
"""Monitor and report metrics for the full DeepSeek-V4 Pro model runs (2 baseline, 2 tuned)."""

import json
import logging
import os
import subprocess
import time

from scripts.xla_autotune.utils import get_cdk_env, sync_and_parse_cdk_job

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("full_model_monitor")

# NOTE: Adjust for your workspace
ARTIFACT_DIR = "/mnt/workspace/xla_tuning/vllm-torchtpu/artifacts/dsv4_full_model_eval"


def monitor():
    manifest_path = os.path.join(ARTIFACT_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        logger.error("Manifest not found: %s", manifest_path)
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    logger.info("Monitoring %d full model jobs...", len(manifest))

    while True:
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception as e:
            logger.warning("Error reading manifest: %s", e)
            manifest = []

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
            kind = item.get("kind", "unknown")
            flag = item.get("flag")
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
                    "Full model run %s (%s) finished: %s. Fetching results...",
                    trial_id,
                    job_id,
                    status,
                )
                metrics, err = sync_and_parse_cdk_job(job_id, trial_dir, timeout=120)
                val = metrics.get("total_token_throughput") if metrics else None
                success = (status == "Succeeded") and (val is not None)

                data = {
                    "trial_id": trial_id,
                    "job_id": job_id,
                    "kind": kind,
                    "flag": flag,
                    "status": status,
                    "success": success,
                    "target_value": val,
                    "metrics": metrics,
                    "error": err if not success else "",
                }
                with open(json_path, "w") as f:
                    json.dump(data, f, indent=2)
                completed.append(data)

        # Baseline vs Tuned stats
        baselines = [
            c
            for c in completed
            if c.get("kind") == "baseline"
            and c.get("success")
            and c.get("target_value")
        ]
        tuneds = [
            c
            for c in completed
            if c.get("kind") == "tuned" and c.get("success") and c.get("target_value")
        ]

        report_lines = [
            "# Full DeepSeek-V4 Pro Model Evaluation (v7x-16)",
            "",
            "- **Recipe**: `torchtpu-vllm-dsv4-pro-v7x-16-benchmark`",
            "- **Tuned Flag**: `--xla_tpu_vmem_scavenging_mode=NONE`",
            f"- **Completed Runs**: `{len(completed)}/{len(manifest)}`",
            "",
        ]

        if baselines or tuneds:
            b_avg = (
                (sum(b["target_value"] for b in baselines) / len(baselines))
                if baselines
                else None
            )
            t_avg = (
                (sum(t["target_value"] for t in tuneds) / len(tuneds))
                if tuneds
                else None
            )

            report_lines.append("### 📊 Performance Comparison")
            if b_avg is not None:
                report_lines.append(
                    f"- **Baseline Average Throughput**: **`{b_avg:,.2f} tok/s`** ({len(baselines)} run(s))"
                )
            if t_avg is not None:
                report_lines.append(
                    f"- **Tuned Average Throughput**: **`{t_avg:,.2f} tok/s`** ({len(tuneds)} run(s))"
                )

            if b_avg is not None and t_avg is not None:
                diff_pct = ((t_avg - b_avg) / b_avg) * 100
                diff_str = f"+{diff_pct:.2f}%" if diff_pct >= 0 else f"{diff_pct:.2f}%"
                report_lines.append(
                    f"- **Net Speedup on Full Model**: **`{diff_str}`**"
                )

            report_lines.extend(
                [
                    "",
                    "| Run | Kind | Job ID | Status | Throughput (tok/s) | Request (req/s) | TTFT (ms) | Notes |",
                    "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
                ]
            )
            for c in completed:
                val = c.get("target_value")
                m = c.get("metrics", {})
                req_tp = m.get("request_throughput", 0)
                ttft = m.get("mean_ttft_ms", 0)
                v_str = f"{val:,.2f}" if val else "N/A"
                report_lines.append(
                    f"| `{c['trial_id']}` | {c.get('kind')} | `{c['job_id']}` | `{c.get('status')}` | {v_str} | {req_tp:.2f} | {ttft:,.2f} | {c.get('error', '')} |"
                )
        else:
            report_lines.append("Full model runs in progress...")

        report_path = os.path.join(ARTIFACT_DIR, "full_model_report.md")
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines) + "\n")

        logger.info(
            "Full Model Progress: %d/%d completed (%d running, %d queued/pending)",
            len(completed),
            len(manifest),
            running,
            pending,
        )

        if len(completed) == len(manifest):
            logger.info("All 4 full model runs finished!")
            break

        time.sleep(60)


if __name__ == "__main__":
    monitor()
