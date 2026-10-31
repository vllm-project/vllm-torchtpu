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
"""Local One-Factor-At-A-Time (OFAT) XLA Autotuner for vllm-torchtpu."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import traceback
from datetime import UTC, datetime
from typing import Any

from scripts.xla_autotune.config import (
    BenchmarkConfig,
    ModelConfig,
    Trial,
    TrialResult,
    TuningConfig,
    load_tuning_config,
)
from scripts.xla_autotune.runner import get_benchmark_runner
from scripts.xla_autotune.server_manager import ServerManager
from scripts.xla_autotune.utils import generate_ranking_report, load_flags

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("xla_autotune")


def slice_flags(flags: list[str], idx: int, count: int) -> list[str]:
    """Return contiguous shard `idx` of `count` (1-based)."""
    if not (1 <= idx <= count):
        raise ValueError(f"slice_index {idx} must be between 1 and {count}")
    chunk = math.ceil(len(flags) / count)
    start = (idx - 1) * chunk
    end = len(flags) if idx == count else start + chunk
    return flags[start:end]


def plan_trials(
    candidates: list[str],
    baseline_runs: int,
    skip_candidates: int = 0,
) -> list[Trial]:
    """Build the planned list of trials: fresh baselines followed by sliced candidates."""
    trials = [
        Trial(f"baseline_{i + 1:02d}", "baseline", None) for i in range(baseline_runs)
    ]
    for i, flag in enumerate(candidates):
        if i < skip_candidates:
            continue
        trials.append(Trial(f"cand_{i + 1:03d}", "candidate", flag))
    return trials


def run_trial(
    trial: Trial,
    cfg: TuningConfig,
    server_mgr: ServerManager,
    bench_runner: Any,
    artifact_dir: str,
    summary_fp,
) -> TrialResult:
    """Execute a single trial locally via ServerManager and bench_runner."""
    t0 = time.time()
    trial_dir = os.path.join(artifact_dir, "trials", trial.trial_id)
    os.makedirs(trial_dir, exist_ok=True)

    trial_json_path = os.path.join(trial_dir, f"{trial.trial_id}.json")
    if os.path.exists(trial_json_path):
        try:
            with open(trial_json_path, encoding="utf-8") as f:
                saved = json.load(f)
            if saved.get("success") and saved.get("target_value") is not None:
                logger.info(
                    "[%s] Trial already completed with %s=%.2f. Skipping re-run.",
                    trial.trial_id,
                    cfg.benchmark.target_metric,
                    saved.get("target_value"),
                )
                return TrialResult(
                    trial_id=saved["trial_id"],
                    kind=saved["kind"],
                    flag=saved.get("flag"),
                    model=saved["model"],
                    target_metric=saved["target_metric"],
                    target_value=saved.get("target_value"),
                    success=saved["success"],
                    metrics=saved.get("metrics", {}),
                    duration_sec=saved.get("duration_sec", 0.0),
                    started_utc=saved.get("started_utc", ""),
                    finished_utc=saved.get("finished_utc", ""),
                    error=saved.get("error", ""),
                )
        except Exception as e:
            logger.debug(
                "[%s] Could not read existing trial JSON: %s", trial.trial_id, e
            )

    result = TrialResult(
        trial_id=trial.trial_id,
        kind=trial.kind,
        flag=trial.flag,
        model=cfg.model.name,
        target_metric=cfg.benchmark.target_metric,
        started_utc=datetime.now(UTC).isoformat(),
        success=False,
    )

    try:
        extra_args = [trial.flag] if trial.flag else []
        env = server_mgr.start_server(extra_args, trial_dir=trial_dir)
        try:
            metrics = bench_runner.run_benchmark(trial_dir=trial_dir, env=env)
            result.success = True
            result.metrics = metrics
        finally:
            server_mgr.stop_server()

        # Extract target metric value
        if result.success and cfg.benchmark.target_metric in result.metrics:
            val = result.metrics[cfg.benchmark.target_metric]
            if isinstance(val, (int, float)):
                result.target_value = float(val)

    except Exception as e:
        result.success = False
        result.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        logger.error("[%s] Trial failed: %s", trial.trial_id, e)

    result.duration_sec = round(time.time() - t0, 2)
    result.finished_utc = datetime.now(UTC).isoformat()

    # Save trial JSON
    trial_json_path = os.path.join(trial_dir, f"{trial.trial_id}.json")
    with open(trial_json_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    # Append to summary.jsonl
    summary_fp.write(json.dumps(result.to_dict()) + "\n")
    summary_fp.flush()

    logger.info(
        "[%s] kind=%s flag=%s success=%s %s=%s dur=%.1fs",
        trial.trial_id,
        trial.kind,
        trial.flag or "(baseline)",
        result.success,
        cfg.benchmark.target_metric,
        result.target_value,
        result.duration_sec,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to YAML or JSON experiment config file")
    parser.add_argument(
        "--model", help="Model name (e.g. deepseek-ai/DeepSeek-V4-Pro-0813)"
    )
    parser.add_argument("--flag-list-file", help="Path to candidate flags.txt")
    parser.add_argument(
        "--slice-index", type=int, help="1-based slice index for sharding"
    )
    parser.add_argument("--slice-count", type=int, help="Total number of shards")
    parser.add_argument("--baseline-runs", type=int, help="Number of baseline runs")
    parser.add_argument(
        "--skip-candidates", type=int, help="Number of candidates to skip (resume)"
    )
    parser.add_argument(
        "--target-metric", help="Metric to optimize (e.g. total_token_throughput)"
    )
    parser.add_argument("--artifact-dir", help="Output directory for results and logs")
    parser.add_argument(
        "--dry-run", action="store_true", help="Simulate without hardware execution"
    )

    args = parser.parse_args()

    # Load base config if provided, or build default
    if args.config:
        cfg = load_tuning_config(args.config)
    else:
        if not args.model:
            parser.error("Either --config or --model must be provided")
        cfg = TuningConfig(
            model=ModelConfig(name=args.model),
            benchmark=BenchmarkConfig(),
        )

    # CLI arguments override config fields
    if args.flag_list_file:
        cfg.flag_list_file = args.flag_list_file
    elif not os.path.isabs(cfg.flag_list_file) and not os.path.exists(
        cfg.flag_list_file
    ):
        # Resolve relative to script dir
        cand_path = os.path.join(os.path.dirname(__file__), cfg.flag_list_file)
        if os.path.exists(cand_path):
            cfg.flag_list_file = cand_path

    if args.slice_index is not None:
        cfg.slice_index = args.slice_index
    if args.slice_count is not None:
        cfg.slice_count = args.slice_count
    if args.baseline_runs is not None:
        cfg.baseline_runs = args.baseline_runs
    if args.skip_candidates is not None:
        cfg.skip_candidates = args.skip_candidates
    if args.target_metric:
        cfg.benchmark.target_metric = args.target_metric
    if args.artifact_dir:
        cfg.artifact_dir = args.artifact_dir
    if args.dry_run:
        cfg.dry_run = True

    os.makedirs(cfg.artifact_dir, exist_ok=True)

    flags = load_flags(cfg.flag_list_file)
    sliced = slice_flags(flags, cfg.slice_index, cfg.slice_count)
    trials = plan_trials(sliced, cfg.baseline_runs, cfg.skip_candidates)

    manifest_path = os.path.join(cfg.artifact_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": cfg.model.name,
                "execution_mode": "local",
                "slice_index": cfg.slice_index,
                "slice_count": cfg.slice_count,
                "total_flags": len(flags),
                "sliced_flags": sliced,
                "baseline_runs": cfg.baseline_runs,
                "skip_candidates": cfg.skip_candidates,
                "target_metric": cfg.benchmark.target_metric,
                "trial_count": len(trials),
                "started_utc": datetime.now(UTC).isoformat(),
            },
            f,
            indent=2,
        )

    logger.info(
        "Starting local sweep: shard %d/%d (%d candidates, %d baselines, %d total trials) -> %s",
        cfg.slice_index,
        cfg.slice_count,
        len(sliced),
        cfg.baseline_runs,
        len(trials),
        cfg.artifact_dir,
    )

    # Initialize local server and benchmark runner
    server_mgr = ServerManager(cfg.model, dry_run=cfg.dry_run)
    bench_runner = get_benchmark_runner(cfg.model, cfg.benchmark, dry_run=cfg.dry_run)

    summary_file = os.path.join(cfg.artifact_dir, "summary.jsonl")
    all_results: list[TrialResult] = []
    failures = 0

    with open(summary_file, "a", encoding="utf-8") as summary_fp:
        for trial in trials:
            res = run_trial(
                trial=trial,
                cfg=cfg,
                server_mgr=server_mgr,
                bench_runner=bench_runner,
                artifact_dir=cfg.artifact_dir,
                summary_fp=summary_fp,
            )
            all_results.append(res)
            if not res.success:
                failures += 1

    ranking_path = os.path.join(cfg.artifact_dir, "ranking.md")
    generate_ranking_report(all_results, cfg.benchmark.target_metric, ranking_path)

    logger.info(
        "Finished shard %d/%d: %d trials, %d failure(s)",
        cfg.slice_index,
        cfg.slice_count,
        len(trials),
        failures,
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
