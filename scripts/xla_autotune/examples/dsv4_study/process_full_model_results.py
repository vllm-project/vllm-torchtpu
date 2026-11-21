#!/usr/bin/env python3
# Copyright 2026 Google LLC
"""Sync outputs and aggregate results for the 4 full model jobs."""

import json
import logging
import os

from scripts.xla_autotune.utils import sync_and_parse_cdk_job

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("process_full_model")

# NOTE: Adjust for your workspace
ARTIFACT_DIR = "/mnt/workspace/xla_tuning/vllm-torchtpu/artifacts/dsv4_full_model_eval"


def main():
    manifest_path = os.path.join(ARTIFACT_DIR, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    logger.info("Processing %d full model jobs from %s", len(manifest), manifest_path)

    completed = []
    for item in manifest:
        trial_id = item["trial_id"]
        job_id = item["job_id"]
        kind = item.get("kind", "unknown")
        flag = item.get("flag")
        trial_dir = os.path.join(ARTIFACT_DIR, "trials", trial_id)
        os.makedirs(trial_dir, exist_ok=True)
        json_path = os.path.join(trial_dir, f"{trial_id}.json")

        logger.info("Processing trial: %s (job: %s, kind: %s)", trial_id, job_id, kind)
        metrics, err = sync_and_parse_cdk_job(job_id, trial_dir, timeout=180)
        val = metrics.get("total_token_throughput") if metrics else None
        success = val is not None

        data = {
            "trial_id": trial_id,
            "job_id": job_id,
            "kind": kind,
            "flag": flag,
            "status": "Succeeded",
            "success": success,
            "target_value": val,
            "metrics": metrics,
            "error": err if not success else "",
        }
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        completed.append(data)
        logger.info(
            "Trial %s result: throughput=%s, req/s=%s, ttft=%s",
            trial_id,
            f"{val:,.2f}" if val else "N/A",
            f"{metrics.get('request_throughput', 0):.2f}" if metrics else "N/A",
            f"{metrics.get('mean_ttft_ms', 0):,.2f} ms" if metrics else "N/A",
        )

    # Baseline vs Tuned stats
    baselines = [
        c
        for c in completed
        if c.get("kind") == "baseline" and c.get("success") and c.get("target_value")
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
        "- **Model**: Full 61 Layers (`DeepSeek-V4-Pro-0813`)",
        f"- **Completed Runs**: `{len(completed)}/{len(manifest)}`",
        "",
    ]

    b_avg = (
        (sum(b["target_value"] for b in baselines) / len(baselines))
        if baselines
        else None
    )
    t_avg = (sum(t["target_value"] for t in tuneds) / len(tuneds)) if tuneds else None

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
        report_lines.append(f"- **Net Speedup on Full Model**: **`{diff_str}`**")

    report_lines.extend(
        [
            "",
            "| Run | Kind | Job ID | Status | Throughput (tok/s) | Request (req/s) | TTFT (ms) | vs Baseline Mean | Notes |",
            "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
    )
    for c in completed:
        val = c.get("target_value")
        m = c.get("metrics", {})
        req_tp = m.get("request_throughput", 0)
        ttft = m.get("mean_ttft_ms", 0)
        v_str = f"{val:,.2f}" if val else "N/A"
        if val and b_avg:
            diff = ((val - b_avg) / b_avg) * 100
            diff_s = f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%"
        else:
            diff_s = "N/A"
        report_lines.append(
            f"| `{c['trial_id']}` | {c.get('kind')} | `{c['job_id']}` | `{c.get('status')}` | {v_str} | {req_tp:.2f} | {ttft:,.2f} | {diff_s} | {c.get('error', '')} |"
        )

    report_path = os.path.join(ARTIFACT_DIR, "full_model_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")

    logger.info("Full model report successfully written to %s", report_path)
    print("\n" + "\n".join(report_lines))


if __name__ == "__main__":
    main()
