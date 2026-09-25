#!/usr/bin/env python3
"""Compare results against a baseline (perf or lm-eval).

Usage:
    check_regression.py --mode <perf|eval> --results-dir <dir> --baseline <baseline.json> [--tolerance 0.05]
    check_regression.py --mode <perf|eval> --calibrate --results-dir <dir> [<dir2> ...]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple


class Metric(NamedTuple):
    result_field: str
    baseline_field: str
    direction: str
    legacy_baseline_fields: tuple[str, ...] = ()


GATED_METRICS = [
    # Gated only when the baseline entry carries the field. Under closed-loop
    # load beyond the prefill-admission capacity (DP x max-num-batched-tokens
    # / prompt-len), median TTFT is queue-position arithmetic, not a latency
    # signal, and swings up to ~40% run to run on identical code — calibrate
    # with --gate-ttft-max-concurrency to omit it for such cells.
    Metric("median_ttft_ms", "median_ttft_ms", "lower_is_better"),
    Metric("median_tpot_ms", "median_tpot_ms", "lower_is_better"),
    Metric("total_token_throughput", "total_token_throughput", "higher_is_better"),
    Metric(
        "output_throughput",
        "output_token_throughput",
        "higher_is_better",
        ("output_throughput",),
    ),
]

COMPLETION_FLOOR_RATIO = 0.95

RESULT_FILENAME_RE = re.compile(r"^isl(\d+)_osl(\d+)_c(\d+)\.json$")

ACC_KEYS = (
    "exact_match,strict_match",
    "exact_match,custom-extract",
    "exact_match,flexible-extract",
    "acc,none",
    "exact_match,none",
    "acc_norm,none",
    "pass@1,create_test",
    "pass_at_1,create_test",
    "pass_at_1,none",
)


def stderr_key(metric: str) -> str:
    base, sep, filter_ = metric.partition(",")
    if base.endswith("_stderr"):
        return metric
    return f"{base}_stderr{sep}{filter_}"


# The gate compares two independent runs, so the noisy quantity is their
# *difference*, whose standard error is hypot(se_baseline, se_run) -- about
# 1.41x a single run's stderr, not 1x. Gating at one raw stderr would still
# fail ~24% of clean reruns on a small benchmark like humaneval_plus (N=164,
# stderr ~2.3pp); scaling the difference's stderr by the one-sided 95%
# critical value puts that at 5% by construction.
Z_ONE_SIDED_95 = 1.645

# Guards the exactly-at-threshold comparison against binary floating-point
# error (e.g. 0.88 - 0.90 == -0.020000000000000018, not -0.02).
FLOAT_EPSILON = 1e-9


def key_from_filename(name: str) -> str | None:
    m = RESULT_FILENAME_RE.match(name)
    return m.group(0).removesuffix(".json") if m else None


def load_perf_results(results_dir: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(results_dir.glob("isl*_osl*_c*.json")):
        key = key_from_filename(f.name)
        if key is None:
            continue
        with f.open() as fh:
            out[key] = json.load(fh)
    return out


def get_baseline_metric(entry: dict, metric: Metric) -> float | None:
    for field in (metric.baseline_field, *metric.legacy_baseline_fields):
        value = entry.get(field)
        if value is not None:
            return float(value)
    return None


def check_perf(
    results: dict[str, dict], baseline: dict[str, dict], tolerance: float
) -> int:
    rows: list[tuple[str, str, float, float, float, str]] = []
    failed = 0

    keys = sorted(set(baseline) | set(results))
    for key in keys:
        if key not in baseline:
            rows.append((key, "(no baseline)", 0.0, 0.0, 0.0, "SKIP"))
            continue
        if key not in results:
            rows.append((key, "(no result)", 0.0, 0.0, 0.0, "FAIL"))
            failed += 1
            continue

        b = baseline[key]
        r = results[key]

        baseline_completed = b.get("completed")
        completed = r.get("completed")
        if (
            baseline_completed is not None
            and completed is not None
            and baseline_completed > 0
        ):
            ratio = completed / baseline_completed
            if ratio < COMPLETION_FLOOR_RATIO:
                rows.append(
                    (key, "completed", completed, baseline_completed, ratio, "FAIL")
                )
                failed += 1

        for metric in GATED_METRICS:
            actual = r.get(metric.result_field)
            base = get_baseline_metric(b, metric)
            if actual is None or base is None or base == 0:
                rows.append(
                    (
                        key,
                        metric.baseline_field,
                        actual or 0.0,
                        base or 0.0,
                        0.0,
                        "SKIP",
                    )
                )
                continue
            actual = float(actual)
            ratio = actual / base
            if metric.direction == "higher_is_better":
                threshold = 1.0 - tolerance
                ok = ratio >= threshold
            else:
                threshold = 1.0 + tolerance
                ok = ratio <= threshold
            status = "PASS" if ok else "FAIL"
            if not ok:
                failed += 1
            rows.append((key, metric.baseline_field, actual, base, ratio, status))

    print(f"\n## Perf regression check (tolerance: {tolerance * 100:.2f}%)\n")
    print("| key | metric | actual | baseline | actual/base | status |")
    print("|---|---|---:|---:|---:|---|")
    for key, metric, actual, base, ratio, status in rows:
        print(
            f"| `{key}` | {metric} | {actual:.4g} | {base:.4g} | {ratio:.4f} | {status} |"
        )
    print()
    return failed


def calibrate_perf(
    results_dirs: list[Path], gate_ttft_max_concurrency: int | None = None
) -> dict[str, dict]:
    collected: dict[str, dict[str, list[float]]] = {}
    num_prompts_by_key: dict[str, int] = {}
    completed_by_key: dict[str, list[int]] = {}
    for d in results_dirs:
        for key, r in load_perf_results(d).items():
            slot = collected.setdefault(key, {})
            for metric in GATED_METRICS:
                if metric.result_field in r:
                    slot.setdefault(metric.baseline_field, []).append(
                        float(r[metric.result_field])
                    )
            if "num_prompts" in r:
                num_prompts_by_key[key] = int(r["num_prompts"])
            if "completed" in r:
                completed_by_key.setdefault(key, []).append(int(r["completed"]))

    baseline = {}
    for key, metrics in collected.items():
        entry = {m: statistics.median(vs) for m, vs in metrics.items()}
        if key in num_prompts_by_key:
            entry["num_prompts"] = num_prompts_by_key[key]
        if key in completed_by_key:
            entry["completed"] = int(statistics.median(completed_by_key[key]))
        # Cells above the concurrency cutoff drop median_ttft_ms so the gate
        # reports SKIP for it (queue-dominated TTFT is not a latency signal).
        if gate_ttft_max_concurrency is not None:
            m = RESULT_FILENAME_RE.match(key + ".json")
            if m and int(m.group(3)) > gate_ttft_max_concurrency:
                entry.pop("median_ttft_ms", None)
        baseline[key] = entry
    return baseline


def find_eval_results_jsons(results_dir: Path) -> list[Path]:
    return list(results_dir.rglob("results_*.json"))


def load_eval_results(results_dir: Path) -> dict[str, dict]:
    jsons = find_eval_results_jsons(results_dir)
    if not jsons:
        return {}
    # One results file per lm-eval task; oldest first so a re-run wins.
    data: dict = {"results": {}}
    for path in sorted(jsons, key=lambda p: p.stat().st_mtime):
        try:
            with path.open() as fh:
                data["results"].update(json.load(fh).get("results") or {})
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: failed to parse {path}: {e}", file=sys.stderr)

    out: dict[str, dict] = {}
    for task, metrics in (data.get("results") or {}).items():
        if not isinstance(metrics, dict):
            continue
        slot = {}
        for k in ACC_KEYS:
            if not isinstance(metrics.get(k), (int, float)):
                continue
            slot[k] = float(metrics[k])
            # Only pair a stderr with an accuracy value we actually recorded;
            # an orphan stderr would otherwise make the task look present but
            # metric-less, turning a missing-result FAIL into a silent SKIP.
            err_k = stderr_key(k)
            if isinstance(metrics.get(err_k), (int, float)):
                slot[err_k] = float(metrics[err_k])
        if slot:
            out[task] = slot
    return out


class PrimaryMetric(NamedTuple):
    name: str
    value: float


def primary_metric(entry: dict) -> PrimaryMetric | None:
    for k in ACC_KEYS:
        if k in entry:
            return PrimaryMetric(k, entry[k])
    return None


def primary_stderr(entry: dict, metric_name: str | None = None) -> float | None:
    if metric_name is None:
        m = primary_metric(entry)
        if m is None:
            return None
        metric_name = m.name
    key = stderr_key(metric_name)
    val = entry.get(key)
    if isinstance(val, (int, float)):
        return float(val)
    return None


def delta_stderr(
    baseline_entry: dict, results_entry: dict | None, metric_name: str
) -> float:
    """Standard error of (results - baseline) for one metric.

    Both runs contribute noise, so the errors add in quadrature. The two
    sides are not treated alike when one is missing: the baseline's stderr
    sets the scale for both sides, the run's stderr for its own side only.

    - Both report: ``hypot(se_baseline, se_run)``.
    - Run reports none: assume it is as noisy as the baseline, recovering the
      ``sqrt(2) * se_baseline`` form.
    - Baseline reports none: the baseline is treated as exact, giving
      ``hypot(0.0, se_run) == se_run``. The gate still widens off the run's
      measured stderr rather than dropping back to the flat tolerance, but by
      ``1 / sqrt(2)`` less than when the same stderr is on the baseline side.
      Regenerating the baseline with ``--calibrate`` restores the full width.
    - Neither reports: 0.0, and the caller falls back to the flat tolerance
      floor.
    """
    se_baseline = primary_stderr(baseline_entry, metric_name) or 0.0
    se_results = primary_stderr(results_entry or {}, metric_name)
    if se_results is None:
        se_results = se_baseline
    return math.hypot(se_baseline, se_results)


class EvalRow(NamedTuple):
    """One line of the eval report table."""

    task: str
    metric: str
    status: str
    allowed_drop: float
    stderr: float = 0.0
    actual: float = 0.0
    baseline: float = 0.0
    delta: float = 0.0
    # False when the baseline entry carried no recognized accuracy key. Such a
    # row's allowed_drop is just the tolerance floor and says nothing about the
    # task, so it is excluded from the reported range.
    gated: bool = True


@dataclasses.dataclass(frozen=True)
class EvalCheckResult:
    """Outcome of an eval check; truthy when at least one task regressed."""

    rows: list[EvalRow]

    @property
    def failed_rows(self) -> list[EvalRow]:
        return [r for r in self.rows if r.status == "FAIL"]

    @property
    def failed_count(self) -> int:
        return len(self.failed_rows)

    @property
    def allowed_drops(self) -> list[float]:
        return [r.allowed_drop for r in self.rows if r.gated]

    def __bool__(self) -> bool:
        return bool(self.failed_rows)


def format_drop_range(drops: Sequence[float], prefix: str = "") -> str:
    """Render a set of allowed drops as `X.XXpp` or `X.XXpp..Y.YYpp`."""
    lo, hi = min(drops), max(drops)
    if lo == hi:
        return f"{prefix}{lo * 100:.2f}pp"
    return f"{prefix}{lo * 100:.2f}pp..{prefix}{hi * 100:.2f}pp"


def check_eval(
    results: dict[str, dict], baseline: dict[str, dict], tolerance: float
) -> EvalCheckResult:
    rows: list[EvalRow] = []

    # Eval result JSONs include every MMLU subtask even when the baseline only
    # gates aggregate rows. Keep check output focused on the baseline contract.
    for task in sorted(baseline):
        b = primary_metric(baseline[task])
        metric_name = b.name if b else "(unknown)"
        se_delta = (
            delta_stderr(baseline[task], results.get(task), metric_name) if b else 0.0
        )
        allowed_drop = max(tolerance, Z_ONE_SIDED_95 * se_delta)
        defaults = {
            "allowed_drop": allowed_drop,
            "stderr": se_delta,
            "gated": b is not None,
        }

        if task not in results:
            rows.append(EvalRow(task, "(no result)", "FAIL", **defaults))
            continue

        r = primary_metric(results[task])
        if b is None or r is None:
            rows.append(EvalRow(task, "(no acc)", "SKIP", **defaults))
            continue

        delta = r.value - b.value
        # Epsilon guards the exactly-at-tolerance case against binary
        # floating-point error (e.g. 0.88 - 0.90 = -0.020000000000000018).
        status = "PASS" if delta >= -allowed_drop - FLOAT_EPSILON else "FAIL"
        rows.append(
            EvalRow(
                task,
                metric_name,
                status,
                actual=r.value,
                baseline=b.value,
                delta=delta,
                **defaults,
            )
        )

    result = EvalCheckResult(rows)
    tol_header = format_drop_range(result.allowed_drops or [tolerance], prefix="-")

    print(f"\n## MMLU/eval regression check (tolerance: {tol_header})\n")
    print(
        "| task | metric | actual | baseline | stderr(diff) | allowed drop |"
        " delta | status |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---|")
    for row in rows:
        print(
            f"| `{row.task}` | {row.metric} | {row.actual:.4f} | "
            f"{row.baseline:.4f} | {row.stderr:.4f} | {row.allowed_drop:.4f} | "
            f"{row.delta:+.4f} | {row.status} |"
        )
    print()
    return result


def calibrate_eval(
    results_dirs: list[Path], task_filter: set[str] | None
) -> dict[str, dict]:
    collected: dict[str, dict[str, list[float]]] = {}
    for d in results_dirs:
        for task, metrics in load_eval_results(d).items():
            if task_filter is not None and task not in task_filter:
                continue
            slot = collected.setdefault(task, {})
            for k, v in metrics.items():
                slot.setdefault(k, []).append(float(v))

    baseline = {}
    for task, metrics in collected.items():
        baseline[task] = {k: statistics.median(vs) for k, vs in metrics.items()}
    return baseline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["perf", "eval"],
        required=True,
        help="Mode of operation: perf or eval.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        action="append",
        required=True,
        help="Directory containing result files. Repeat for calibrate.",
    )
    parser.add_argument(
        "--baseline", type=Path, help="Baseline JSON to compare against."
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        help="Allowed deviation (default 0.05 for perf, 0.015 for eval).",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Emit baseline JSON to stdout instead of checking.",
    )
    parser.add_argument(
        "--task-filter",
        default="",
        help="Comma-separated allow-list of tasks/datasets.",
    )
    parser.add_argument(
        "--gate-ttft-max-concurrency",
        type=int,
        default=None,
        help="Calibrate (perf) only: omit median_ttft_ms from baseline "
        "entries whose concurrency exceeds this, so TTFT is ungated there. "
        "0 ungates TTFT for every cell.",
    )
    args = parser.parse_args()

    tolerance = args.tolerance
    if tolerance is None:
        tolerance = 0.05 if args.mode == "perf" else 0.015

    if args.calibrate:
        if args.mode == "perf":
            baseline = calibrate_perf(args.results_dir, args.gate_ttft_max_concurrency)
        elif args.mode == "eval":
            if args.task_filter == "all":
                task_filter = None
            elif args.task_filter:
                task_filter = set(
                    s.strip() for s in args.task_filter.split(",") if s.strip()
                )
            else:
                task_filter = {
                    "mmlu_llama",
                    "mmlu_llama_humanities",
                    "mmlu_llama_stem",
                    "mmlu_llama_social_sciences",
                    "mmlu_llama_other",
                    "mmlu_pro",
                    "humaneval_plus_tpu",
                    "mbpp_plus_tpu",
                }
            baseline = calibrate_eval(args.results_dir, task_filter)
        json.dump(baseline, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    if args.baseline is None:
        parser.error("--baseline is required unless --calibrate is set")
    if len(args.results_dir) != 1:
        parser.error("--results-dir may only be given once in check mode")

    if args.mode == "perf":
        results = load_perf_results(args.results_dir[0])
        if not results:
            print(
                f"ERROR: no isl*_osl*_c*.json files found in {args.results_dir[0]}",
                file=sys.stderr,
            )
            return 2
    else:
        results = load_eval_results(args.results_dir[0])
        if not results:
            print(
                f"ERROR: no results_*.json found under {args.results_dir[0]}",
                file=sys.stderr,
            )
            return 2

    with args.baseline.open() as fh:
        baseline = json.load(fh)
    baseline = {k: v for k, v in baseline.items() if not k.startswith("_")}

    if args.mode == "perf":
        failed = check_perf(results, baseline, tolerance)
        if failed:
            print(
                f"FAILED: {failed} metric(s) regressed beyond {tolerance * 100:.2f}%",
                file=sys.stderr,
            )
            return 1
        print("PASS: all gated metrics within tolerance.")
    else:
        result = check_eval(results, baseline, tolerance)
        if result:
            tol_str = format_drop_range(
                [row.allowed_drop for row in result.failed_rows]
            )
            print(
                f"FAILED: {result.failed_count} task(s) regressed beyond {tol_str}",
                file=sys.stderr,
            )
            return 1
        print("PASS: all evaluated tasks within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
