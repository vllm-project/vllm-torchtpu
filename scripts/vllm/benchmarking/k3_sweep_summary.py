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
"""Result parsing for the Kimi-K3 sweep drivers.

Pure functions plus a small CLI, so the drivers' summary logic can be unit
tested (tests/scripts/test_k3_sweep_summary.py) without a TPU:

  lm-eval-score  --output-dir DIR --task TASK
      -> "<metric> <score> <stderr>" from the newest lm_eval results_*.json
         under DIR, or "none nan nan" (with an ERROR line on stderr).
  question-limit --task TASK --conc N --gsm8k-limit G --mmlu-limit M
      -> "<lm_eval --limit value> <total questions>"; the counts grow with
         concurrency so that N requests are really in flight.
  gsm8k-row      --output-dir DIR --conc N
      -> "| N | <strict-match> | <flexible-extract> |"
  agentx-table   --results-dir DIR --base NAME --concs N...
      -> markdown table over DIR/NAME_c<N>.json, the aggregate JSON that
         InferenceX's process_agentic_result.py writes per point.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Any

MMLU_PRO_SUBJECTS = 14

# lm_eval metric keys, in preference order per task family.
_SCORE_KEYS = (
    "exact_match,custom-extract",  # mmlu_pro
    "exact_match,flexible-extract",  # gsm8k
    "exact_match,strict-match",
)

# (column title, dotted path into the InferenceX aggregate JSON)
AGENTX_COLUMNS = (
    ("out tok/s per chip", "request_metrics.throughput.per_gpu.output_tput_tps"),
    ("out tok/s total", "request_metrics.throughput.output.tokens_per_second"),
    ("requests ok", "num_requests_successful"),
    ("TTFT p50 s", "request_metrics.latency.ttft.p50"),
    ("TTFT p90 s", "request_metrics.latency.ttft.p90"),
    ("TPOT p50 s", "request_metrics.latency.tpot.p50"),
    ("P90 intvty", "request_metrics.latency.intvty.p90"),
    ("P90 E2E-norm intvty", "request_metrics.latency.e2e_norm_intvty.p90"),
    ("e2el p50 s", "request_metrics.latency.e2el.p50"),
    ("cache hit (theory)", "request_metrics.cache.theoretical_cache_hit_rate"),
    ("KV used", "server_metrics.kv_cache.gpu_usage_pct"),
)


def dotted_get(d: Any, path: str) -> float | None:
    """Numeric value at a dotted path, or None when absent / not a number."""
    for key in path.split("."):
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d if isinstance(d, (int, float)) and not isinstance(d, bool) else None


def newest_results_file(output_dir: str) -> str | None:
    files = glob.glob(os.path.join(output_dir, "**", "results_*.json"), recursive=True)
    return max(files, key=os.path.getmtime) if files else None


def lm_eval_score(output_dir: str, task: str) -> tuple[str, float, float]:
    """(metric, score, stderr) for TASK from the newest results file."""
    path = newest_results_file(output_dir)
    if path is None:
        raise FileNotFoundError(f"no results_*.json under {output_dir}")
    with open(path) as f:
        results = json.load(f)["results"]
    if task not in results:
        raise KeyError(f"task {task!r} not in {path} (has {sorted(results)})")
    r = results[task]
    for key in _SCORE_KEYS:
        if key in r:
            stderr = r.get(
                key.replace("exact_match", "exact_match_stderr"), float("nan")
            )
            return key, float(r[key]), float(stderr)
    raise KeyError(f"no exact_match metric for {task!r} in {path}: {sorted(r)}")


def question_limit(
    task: str, conc: int, gsm8k_limit: int, mmlu_limit: int
) -> tuple[int, int]:
    """(lm_eval --limit, total questions) so that 2*conc questions exist.

    gsm8k's --limit is a question count; mmlu_pro's is per subject, and it
    has 14 subjects.
    """
    if task == "gsm8k":
        limit = max(gsm8k_limit, 2 * conc)
        return limit, limit
    if task == "mmlu_pro":
        limit = max(mmlu_limit, math.ceil(2 * conc / MMLU_PRO_SUBJECTS))
        return limit, limit * MMLU_PRO_SUBJECTS
    raise ValueError(f"unknown task {task!r}")


def gsm8k_row(output_dir: str, conc: int) -> str:
    path = newest_results_file(output_dir)
    if path is None:
        return f"| {conc} | (no results file) | |"
    with open(path) as f:
        results = json.load(f)["results"]
    r = next(iter(results.values()), {})
    return (
        f"| {conc} | {r.get('exact_match,strict-match')} | "
        f"{r.get('exact_match,flexible-extract')} |"
    )


def agentx_table(results_dir: str, base: str, concs: list[int]) -> str:
    lines = [
        "| conc | " + " | ".join(t for t, _ in AGENTX_COLUMNS) + " |",
        "|" + "---|" * (len(AGENTX_COLUMNS) + 1),
    ]
    for c in concs:
        path = os.path.join(results_dir, f"{base}_c{c}.json")
        timed_out = os.path.exists(
            os.path.join(results_dir, f"conc_{c}", "HARD_TIMEOUT")
        )
        if not os.path.exists(path):
            why = "hard timeout, " if timed_out else ""
            lines.append(
                f"| {c} | (no aggregate: {why}{os.path.basename(path)} missing) |"
            )
            continue
        with open(path) as f:
            d = json.load(f)
        cells = []
        for _, key in AGENTX_COLUMNS:
            v = dotted_get(d, key)
            cells.append("-" if v is None else f"{v:.4g}")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("lm-eval-score")
    s.add_argument("--output-dir", required=True)
    s.add_argument("--task", required=True)
    s = sub.add_parser("question-limit")
    s.add_argument("--task", required=True)
    s.add_argument("--conc", type=int, required=True)
    s.add_argument("--gsm8k-limit", type=int, required=True)
    s.add_argument("--mmlu-limit", type=int, required=True)
    s = sub.add_parser("gsm8k-row")
    s.add_argument("--output-dir", required=True)
    s.add_argument("--conc", type=int, required=True)
    s = sub.add_parser("agentx-table")
    s.add_argument("--results-dir", required=True)
    s.add_argument("--base", required=True)
    s.add_argument("--concs", type=int, nargs="*", default=[])
    a = p.parse_args(argv)

    if a.cmd == "lm-eval-score":
        try:
            metric, score, stderr = lm_eval_score(a.output_dir, a.task)
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as e:
            print(
                f"[k3-sweep-summary] ERROR: cannot score {a.task} from "
                f"{a.output_dir}: {e}",
                file=sys.stderr,
            )
            print("none nan nan")
            return 0
        print(f"{metric} {score} {stderr}")
    elif a.cmd == "question-limit":
        limit, total = question_limit(a.task, a.conc, a.gsm8k_limit, a.mmlu_limit)
        print(f"{limit} {total}")
    elif a.cmd == "gsm8k-row":
        print(gsm8k_row(a.output_dir, a.conc))
    elif a.cmd == "agentx-table":
        print(agentx_table(a.results_dir, a.base, a.concs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
