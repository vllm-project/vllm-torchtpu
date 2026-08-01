#!/usr/bin/env python3
"""Compare results against a baseline (perf, lm-eval, or EvalPlus).

Usage:
    check_regression.py --mode <perf|eval|evalplus> --results-dir <dir> --baseline <baseline.json> [--tolerance 0.05]
    check_regression.py --mode <perf|eval|evalplus> --calibrate --results-dir <dir> [<dir2> ...]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import NamedTuple


class Metric(NamedTuple):
    result_field: str
    baseline_field: str
    direction: str
    legacy_baseline_fields: tuple[str, ...] = ()


GATED_METRICS = [
    Metric("median_ttft_ms", "median_ttft_ms", "lower_is_better"),
    Metric("median_tpot_ms", "median_tpot_ms", "lower_is_better"),
    Metric("total_token_throughput", "total_token_throughput",
           "higher_is_better"),
    Metric("output_throughput", "output_token_throughput", "higher_is_better",
           ("output_throughput", )),
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
)
STDERR_KEYS = (
    "exact_match_stderr,strict_match",
    "exact_match_stderr,custom-extract",
    "exact_match_stderr,flexible-extract",
    "acc_stderr,none",
)

EVALPLUS_METRICS = ("plus_pass@1", "base_pass@1")


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


def check_perf(results: dict[str, dict], baseline: dict[str, dict],
               tolerance: float) -> int:
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
        if (baseline_completed is not None and completed is not None
                and baseline_completed > 0):
            ratio = completed / baseline_completed
            if ratio < COMPLETION_FLOOR_RATIO:
                rows.append((key, "completed", completed, baseline_completed,
                             ratio, "FAIL"))
                failed += 1

        for metric in GATED_METRICS:
            actual = r.get(metric.result_field)
            base = get_baseline_metric(b, metric)
            if actual is None or base is None or base == 0:
                rows.append((key, metric.baseline_field, actual or 0.0, base
                             or 0.0, 0.0, "SKIP"))
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
            rows.append(
                (key, metric.baseline_field, actual, base, ratio, status))

    print(f"\n## Perf regression check (tolerance: {tolerance*100:.2f}%)\n")
    print("| key | metric | actual | baseline | actual/base | status |")
    print("|---|---|---:|---:|---:|---|")
    for key, metric, actual, base, ratio, status in rows:
        print(
            f"| `{key}` | {metric} | {actual:.4g} | {base:.4g} | {ratio:.4f} | {status} |"
        )
    print()
    return failed


def calibrate_perf(results_dirs: list[Path]) -> dict[str, dict]:
    collected: dict[str, dict[str, list[float]]] = {}
    num_prompts_by_key: dict[str, int] = {}
    completed_by_key: dict[str, list[int]] = {}
    for d in results_dirs:
        for key, r in load_perf_results(d).items():
            slot = collected.setdefault(key, {})
            for metric in GATED_METRICS:
                if metric.result_field in r:
                    slot.setdefault(metric.baseline_field,
                                    []).append(float(r[metric.result_field]))
            if "num_prompts" in r:
                num_prompts_by_key[key] = int(r["num_prompts"])
            if "completed" in r:
                completed_by_key.setdefault(key,
                                            []).append(int(r["completed"]))

    baseline = {}
    for key, metrics in collected.items():
        entry = {m: statistics.median(vs) for m, vs in metrics.items()}
        if key in num_prompts_by_key:
            entry["num_prompts"] = num_prompts_by_key[key]
        if key in completed_by_key:
            entry["completed"] = int(statistics.median(completed_by_key[key]))
        baseline[key] = entry
    return baseline


def find_eval_results_jsons(results_dir: Path) -> list[Path]:
    return sorted(results_dir.rglob("results_*.json"))


def load_eval_results(results_dir: Path) -> dict[str, dict]:
    jsons = find_eval_results_jsons(results_dir)
    if not jsons:
        return {}
    latest = max(jsons, key=lambda p: p.stat().st_mtime)
    with latest.open() as fh:
        data = json.load(fh)

    out: dict[str, dict] = {}
    for task, metrics in (data.get("results") or {}).items():
        if not isinstance(metrics, dict):
            continue
        slot = {}
        for k in ACC_KEYS:
            if k in metrics and isinstance(metrics[k], (int, float)):
                slot[k] = float(metrics[k])
        for k in STDERR_KEYS:
            if k in metrics and isinstance(metrics[k], (int, float)):
                slot[k] = float(metrics[k])
        if slot:
            out[task] = slot
    return out


def primary_metric(entry: dict) -> tuple[str, float] | None:
    for k in ACC_KEYS:
        if k in entry:
            return k, entry[k]
    return None


def check_eval(results: dict[str, dict], baseline: dict[str, dict],
               tolerance: float) -> int:
    rows: list[tuple[str, str, float, float, float, str]] = []
    failed = 0

    # Eval result JSONs include every MMLU subtask even when the baseline only
    # gates aggregate rows. Keep check output focused on the baseline contract.
    keys = sorted(baseline)
    for task in keys:
        if task not in results:
            rows.append((task, "(no result)", 0.0, 0.0, 0.0, "FAIL"))
            failed += 1
            continue

        b = primary_metric(baseline[task])
        r = primary_metric(results[task])
        if b is None or r is None:
            rows.append((task, "(no acc)", 0.0, 0.0, 0.0, "SKIP"))
            continue
        metric_name = b[0]
        baseline_acc = b[1]
        actual_acc = r[1]
        delta = actual_acc - baseline_acc
        status = "PASS" if delta >= -tolerance else "FAIL"
        if status == "FAIL":
            failed += 1
        rows.append(
            (task, metric_name, actual_acc, baseline_acc, delta, status))

    print(
        f"\n## MMLU/eval regression check (tolerance: -{tolerance*100:.2f}pp)\n"
    )
    print("| task | metric | actual | baseline | delta | status |")
    print("|---|---|---:|---:|---:|---|")
    for task, metric, actual, baseline, delta, status in rows:
        print(
            f"| `{task}` | {metric} | {actual:.4f} | {baseline:.4f} | {delta:+.4f} | {status} |"
        )
    print()
    return failed


def calibrate_eval(results_dirs: list[Path],
                   task_filter: set[str] | None) -> dict[str, dict]:
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
        baseline[task] = {
            k: statistics.median(vs)
            for k, vs in metrics.items()
        }
    return baseline


def find_evalplus_result_jsons(results_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*_eval_results.json", "eval_results.json"):
        paths.extend(results_dir.rglob(pattern))
    return sorted(set(paths))


def evalplus_dataset_name(path: Path) -> str:
    if path.name == "eval_results.json":
        return path.parent.parent.name
    return path.parent.name


def evalplus_pass_at_1(result_json: dict) -> dict[str, float]:
    totals: list[int] = []
    base_correct: list[int] = []
    plus_correct: list[int] = []
    saw_plus = False

    for task_results in (result_json.get("eval") or {}).values():
        if not isinstance(task_results, list) or not task_results:
            continue
        totals.append(len(task_results))
        base_correct.append(
            sum(r.get("base_status") == "pass" for r in task_results))
        plus_count = 0
        for result in task_results:
            if result.get("plus_status") is not None:
                saw_plus = True
                if (result.get("base_status") == "pass"
                        and result.get("plus_status") == "pass"):
                    plus_count += 1
        plus_correct.append(plus_count)

    if not totals:
        return {}

    metrics = {
        "base_pass@1":
        statistics.mean(correct / total
                        for correct, total in zip(base_correct, totals))
    }
    if saw_plus:
        metrics["plus_pass@1"] = statistics.mean(
            correct / total for correct, total in zip(plus_correct, totals))
    return metrics


def load_evalplus_results(results_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in find_evalplus_result_jsons(results_dir):
        with path.open() as fh:
            data = json.load(fh)
        metrics = evalplus_pass_at_1(data)
        if metrics:
            out[evalplus_dataset_name(path)] = metrics
    return out


def check_evalplus(results: dict[str, dict], baseline: dict[str, dict],
                   tolerance: float) -> int:
    rows: list[tuple[str, str, float, float, float, str]] = []
    failed = 0

    for dataset in sorted(baseline):
        if dataset not in results:
            rows.append((dataset, "(no result)", 0.0, 0.0, 0.0, "FAIL"))
            failed += 1
            continue

        checked_metric = False
        for metric in EVALPLUS_METRICS:
            if metric not in baseline[dataset]:
                continue
            checked_metric = True
            if metric not in results[dataset]:
                rows.append((dataset, metric, 0.0,
                             float(baseline[dataset][metric]), 0.0, "FAIL"))
                failed += 1
                continue
            baseline_acc = float(baseline[dataset][metric])
            actual_acc = float(results[dataset][metric])
            delta = actual_acc - baseline_acc
            status = "PASS" if delta >= -tolerance else "FAIL"
            if status == "FAIL":
                failed += 1
            rows.append(
                (dataset, metric, actual_acc, baseline_acc, delta, status))

        if not checked_metric:
            rows.append((dataset, "(no pass@1)", 0.0, 0.0, 0.0, "SKIP"))

    print(
        f"\n## EvalPlus regression check (tolerance: -{tolerance*100:.2f}pp)\n"
    )
    print("| dataset | metric | actual | baseline | delta | status |")
    print("|---|---|---:|---:|---:|---|")
    for dataset, metric, actual, baseline_acc, delta, status in rows:
        print(
            f"| `{dataset}` | {metric} | {actual:.4f} | {baseline_acc:.4f} | {delta:+.4f} | {status} |"
        )
    print()
    return failed


def calibrate_evalplus(results_dirs: list[Path],
                       task_filter: set[str] | None) -> dict[str, dict]:
    collected: dict[str, dict[str, list[float]]] = {}
    for d in results_dirs:
        for dataset, metrics in load_evalplus_results(d).items():
            if task_filter is not None and dataset not in task_filter:
                continue
            slot = collected.setdefault(dataset, {})
            for k, v in metrics.items():
                slot.setdefault(k, []).append(float(v))

    baseline = {}
    for dataset, metrics in collected.items():
        baseline[dataset] = {
            k: statistics.median(vs)
            for k, vs in metrics.items()
        }
    return baseline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",
                        choices=["perf", "eval", "evalplus"],
                        required=True,
                        help="Mode of operation: perf, eval, or evalplus.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        action="append",
        required=True,
        help="Directory containing result files. Repeat for calibrate.")
    parser.add_argument("--baseline",
                        type=Path,
                        help="Baseline JSON to compare against.")
    parser.add_argument(
        "--tolerance",
        type=float,
        help=
        "Allowed deviation (default 0.05 for perf, 0.01 for eval and evalplus)."
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Emit baseline JSON to stdout instead of checking.")
    parser.add_argument("--task-filter",
                        default="",
                        help="Comma-separated allow-list of tasks/datasets.")
    args = parser.parse_args()

    tolerance = args.tolerance
    if tolerance is None:
        tolerance = 0.05 if args.mode == "perf" else 0.015

    if args.calibrate:
        if args.mode == "perf":
            baseline = calibrate_perf(args.results_dir)
        elif args.mode == "eval":
            if args.task_filter == "all":
                task_filter = None
            elif args.task_filter:
                task_filter = set(s.strip()
                                  for s in args.task_filter.split(",")
                                  if s.strip())
            else:
                task_filter = {
                    "mmlu_llama",
                    "mmlu_llama_humanities",
                    "mmlu_llama_stem",
                    "mmlu_llama_social_sciences",
                    "mmlu_llama_other",
                    "mmlu_pro",
                }
            baseline = calibrate_eval(args.results_dir, task_filter)
        else:
            if args.task_filter == "all":
                task_filter = None
            elif args.task_filter:
                task_filter = set(s.strip()
                                  for s in args.task_filter.split(",")
                                  if s.strip())
            else:
                task_filter = {"humaneval", "mbpp"}
            baseline = calibrate_evalplus(args.results_dir, task_filter)
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
                file=sys.stderr)
            return 2
    elif args.mode == "eval":
        results = load_eval_results(args.results_dir[0])
        if not results:
            print(
                f"ERROR: no results_*.json found under {args.results_dir[0]}",
                file=sys.stderr)
            return 2
    else:
        results = load_evalplus_results(args.results_dir[0])
        if not results:
            print(
                f"ERROR: no EvalPlus *_eval_results.json files found under {args.results_dir[0]}",
                file=sys.stderr)
            return 2

    with args.baseline.open() as fh:
        baseline = json.load(fh)
    baseline = {k: v for k, v in baseline.items() if not k.startswith("_")}

    if args.mode == "perf":
        failed = check_perf(results, baseline, tolerance)
        if failed:
            print(
                f"FAILED: {failed} metric(s) regressed beyond {tolerance*100:.2f}%",
                file=sys.stderr)
            return 1
        print("PASS: all gated metrics within tolerance.")
    else:
        if args.mode == "eval":
            failed = check_eval(results, baseline, tolerance)
        else:
            failed = check_evalplus(results, baseline, tolerance)
        if failed:
            print(
                f"FAILED: {failed} task(s) regressed beyond {tolerance*100:.2f}pp",
                file=sys.stderr)
            return 1
        print("PASS: all evaluated tasks within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
