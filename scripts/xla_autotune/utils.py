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
"""Shared CDK utility functions for XLA autotuning scripts."""

from __future__ import annotations

import gzip
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from .config import TrialResult
from .runner import parse_vllm_bench_serve_stdout

logger = logging.getLogger(__name__)


def get_cdk_env(cdk_root_dir: str | None = None) -> dict[str, str]:
    """Build environment dict for CDK CLI calls, ensuring CDK_ROOT_DIR is set.

    Resolution order:
    1. Explicit ``cdk_root_dir`` argument.
    2. Existing ``CDK_ROOT_DIR`` in the environment.
    3. Auto-detected ``cloud-devkit-recipes`` directories in common locations.
    """
    env = dict(os.environ)
    if cdk_root_dir:
        env["CDK_ROOT_DIR"] = cdk_root_dir
    elif "CDK_ROOT_DIR" not in env or not env["CDK_ROOT_DIR"]:
        for candidate in [
            Path.cwd() / "cloud-devkit-recipes",
            Path(__file__).resolve().parents[3] / "cloud-devkit-recipes",
            Path("/mnt/workspace/xla_tuning/cloud-devkit-recipes"),
        ]:
            if candidate.is_dir():
                env["CDK_ROOT_DIR"] = str(candidate)
                break
    return env


def parse_cdk_job_id(text: str) -> str | None:
    """Extract a CDK job ID from CLI output text.

    Tries the full UUID format first (``j-xxxxxxxx-xxxx-xxxx-xxxx``), then
    falls back to any ``j-<hex-and-dashes>`` pattern.

    Returns the matched job ID string, or ``None`` if no match is found.
    """
    m = re.search(r"\b(j-[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4})\b", text)
    if not m:
        m = re.search(r"\b(j-[a-f0-9\-]+)\b", text)
    return m.group(1) if m else None


def sync_and_parse_cdk_job(
    job_id: str,
    trial_dir: str,
    env: dict[str, str] | None = None,
    timeout: int = 90,
) -> tuple[dict[str, Any], str]:
    """Sync CDK job outputs and parse benchmark metrics from logs.

    1. Calls ``cdk job sync-outputs <job_id>`` to pull logs locally.
    2. Reads all log files (including ``.gz``) from the synced output directory.
    3. Writes a merged log to ``<trial_dir>/<job_id>.log``.
    4. Parses benchmark metrics via ``parse_vllm_bench_serve_stdout``.

    Args:
        job_id: CDK job identifier (e.g. ``j-7fb420a0-198f-42df-9d73``).
        trial_dir: Directory to write the merged log file into.
        env: Environment dict for the ``cdk`` subprocess. Defaults to
            ``get_cdk_env()`` if not provided.
        timeout: Timeout in seconds for the ``sync-outputs`` call.

    Returns:
        A tuple of ``(metrics_dict, error_message)``. On success the error
        message is an empty string; on failure the metrics dict is empty.
    """
    if env is None:
        env = get_cdk_env()

    os.makedirs(trial_dir, exist_ok=True)

    logger.info("Syncing outputs for job %s...", job_id)
    try:
        subprocess.run(
            ["cdk", "job", "sync-outputs", job_id],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        logger.warning("sync-outputs timed out after %ds for %s", timeout, job_id)

    home_dir = os.path.expanduser("~")
    output_dir = Path(home_dir) / "cloud-devkit-data" / "job-outputs" / job_id
    all_log_text = ""
    if output_dir.is_dir():
        bench_logs = sorted(output_dir.glob("**/sidecar-bench*.log*"))
        other_logs = sorted(output_dir.glob("**/*.log*"))
        log_files = bench_logs + [f for f in other_logs if f not in bench_logs]
        for lf in log_files:
            try:
                if lf.suffix == ".gz":
                    with gzip.open(lf, "rt", encoding="utf-8", errors="replace") as f:
                        all_log_text += f"\n=== {lf.name} ===\n" + f.read()
                else:
                    with open(lf, encoding="utf-8", errors="replace") as f:
                        all_log_text += f"\n=== {lf.name} ===\n" + f.read()
            except Exception as e:
                logger.warning("Error reading %s: %s", lf, e)

    log_path = os.path.join(trial_dir, f"{job_id}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(all_log_text)

    metrics = parse_vllm_bench_serve_stdout(all_log_text)
    if not metrics:
        return {}, "Could not parse benchmark metrics from output logs"
    return metrics, ""


def generate_ranking_report(
    results: list[TrialResult],
    target_metric: str,
    output_path: str,
) -> None:
    """Generate a clean markdown ranking comparing candidate flags to the baseline."""
    baselines = [
        r.target_value
        for r in results
        if r.kind == "baseline" and r.success and r.target_value is not None
    ]
    candidates = [
        r
        for r in results
        if r.kind == "candidate" and r.success and r.target_value is not None
    ]

    lines = [
        "# XLA Autotune Sweep Results\n",
        f"- Target Metric: `{target_metric}`",
        f"- Evaluated: {len(results)} trials ({len(baselines)} baselines, {len(candidates)} successful candidates)\n",
    ]

    baseline_mean = sum(baselines) / len(baselines) if baselines else None
    if baseline_mean is not None and baseline_mean > 0:
        lines.append(
            f"### Baseline Performance: `{baseline_mean:.2f}` (Spread: {min(baselines):.2f} - {max(baselines):.2f})\n"
        )

        # Sort candidates descending by target_value
        candidates.sort(key=lambda c: c.target_value or 0.0, reverse=True)

        lines.append(
            "| Rank | Trial | Flag | Metric Value | vs Baseline | CDK Job ID |"
        )
        lines.append("|:---:|:---:|:---|:---:|:---:|:---:|")

        for rank, c in enumerate(candidates, start=1):
            assert c.target_value is not None
            pct_diff = ((c.target_value - baseline_mean) / baseline_mean) * 100.0
            sign = "+" if pct_diff >= 0 else ""
            diff_str = (
                f"**{sign}{pct_diff:.2f}%**"
                if pct_diff > 0
                else f"{sign}{pct_diff:.2f}%"
            )
            job_str = c.cdk_job_id or "local"
            lines.append(
                f"| {rank} | `{c.trial_id}` | `{c.flag}` | {c.target_value:.2f} | {diff_str} | `{job_str}` |"
            )
    else:
        lines.append(
            "No successful baselines available to compute relative improvements.\n"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Ranking report written to: %s", output_path)


def load_flags(path: str) -> list[str]:
    """Load non-empty, non-comment lines from a flags file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Flags file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return [
            ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")
        ]
