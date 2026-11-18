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
"""Compare results against a baseline (perf or lm-eval).

Usage:
    check_regression.py --mode <perf|eval> --results-dir <dir> --baseline <baseline.json> [--tolerance 0.05]
    check_regression.py --mode <perf|eval> --calibrate --results-dir <dir> [<dir2> ...]
    check_regression.py --mode update --baseline <new.json> [--previous-baseline <old.json>]

Eval mode accepts --results-dir several times and compares the mean of the runs.
"""  # noqa: E501

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
from typing import Any, NamedTuple


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
            f"| `{key}` | {metric} | {actual:.4g} | "
            f"{base:.4g} | {ratio:.4f} | {status} |"
        )
    print()
    return failed


ROOFLINE_CELL_STATUSES = ("ok", "unverified", "oom")


class RooflineCellTarget(NamedTuple):
    output_token_throughput_per_chip: float
    status: str = "ok"


class RooflineConfigTarget(NamedTuple):
    num_chips: int | None
    cells: dict[str, RooflineCellTarget]


class NotModeledEntry(NamedTuple):
    reason: str
    detail: str


class RooflineTargets(NamedTuple):
    targets: dict[str, RooflineConfigTarget]
    not_modeled: dict[str, NotModeledEntry]


class RooflineRow(NamedTuple):
    cell: str
    metric: str
    actual_per_chip: float
    target_per_chip: float
    ratio: float
    target_status: str
    status: str


@dataclasses.dataclass(frozen=True)
class RooflineCheckResult:
    """Advisory % of roofline report for one config."""

    config_name: str
    num_chips: int | None
    rows: list[RooflineRow]
    not_modeled: NotModeledEntry | None = None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    val = float(value)
    return val if math.isfinite(val) else None


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def resolve_physical_chips(
    chips: int | None = None,
    prefill_chips: int | None = None,
    decode_chips: int | None = None,
) -> int | None:
    """Resolve total physical chips from single-pool or disaggregated flags."""
    if chips is not None:
        _positive_int(chips, "chips")
    if prefill_chips is not None:
        _positive_int(prefill_chips, "prefill_chips")
    if decode_chips is not None:
        _positive_int(decode_chips, "decode_chips")
    if (prefill_chips is None) != (decode_chips is None):
        raise ValueError("--prefill-chips and --decode-chips must be provided together")
    if chips is not None and prefill_chips is not None:
        raise ValueError(
            "--chips cannot be combined with --prefill-chips/--decode-chips"
        )
    if prefill_chips is not None and decode_chips is not None:
        return prefill_chips + decode_chips
    return chips


def _parse_roofline_config(name: str, raw: Any) -> RooflineConfigTarget:
    if not isinstance(raw, dict):
        raise ValueError(f"roofline config {name!r} must be an object")
    num_chips = None
    if raw.get("num_chips") is not None:
        num_chips = _positive_int(raw["num_chips"], f"{name}.num_chips")
    raw_cells = raw.get("cells")
    if not isinstance(raw_cells, dict) or not raw_cells:
        raise ValueError(f"{name}.cells must be a non-empty object")
    cells: dict[str, RooflineCellTarget] = {}
    for cell_key, cell in raw_cells.items():
        if not isinstance(cell, dict):
            raise ValueError(f"{name}.cells.{cell_key} must be an object")
        status = cell.get("status", "ok")
        if status not in ROOFLINE_CELL_STATUSES:
            raise ValueError(
                f"{name}.cells.{cell_key}.status must be one of "
                f"{ROOFLINE_CELL_STATUSES}, got {status!r}"
            )
        tput = _finite_number(cell.get("output_token_throughput_per_chip"))
        if status != "oom" and (tput is None or tput <= 0):
            raise ValueError(
                f"{name}.cells.{cell_key}.output_token_throughput_per_chip "
                "must be positive"
            )
        cells[cell_key] = RooflineCellTarget(tput or 0.0, status)
    return RooflineConfigTarget(num_chips, cells)


def load_roofline_targets(path: Path) -> RooflineTargets:
    """Load and validate a roofline_targets.json file."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"cannot read roofline targets {path}: {e}") from e
    if not isinstance(raw, dict) or not isinstance(raw.get("targets"), dict):
        raise ValueError(f"{path} must hold an object with a 'targets' map")

    targets: dict[str, RooflineConfigTarget] = {}
    for name, entry in raw["targets"].items():
        targets[name] = _parse_roofline_config(name, entry)

    raw_not_modeled = raw.get("not_modeled", {})
    if not isinstance(raw_not_modeled, dict):
        raise ValueError(f"{path}: 'not_modeled' must be an object")
    not_modeled: dict[str, NotModeledEntry] = {}
    for name, entry in raw_not_modeled.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: not_modeled.{name} must be an object")
        reason = entry.get("reason")
        detail = entry.get("detail", "")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{path}: not_modeled.{name}.reason must be non-empty")
        not_modeled[name] = NotModeledEntry(reason, str(detail))
    return RooflineTargets(targets, not_modeled)


def check_roofline(
    results: dict[str, dict],
    roofline_targets: RooflineTargets,
    config_name: str,
    chips: int | None = None,
) -> RooflineCheckResult:
    """Compare measured output throughput per physical chip against roofline."""
    if config_name in roofline_targets.not_modeled:
        note = roofline_targets.not_modeled[config_name]
        print(f"\n## Roofline speed check (`{config_name}`: not modeled)\n")
        print(
            f"NOTE: `{config_name}` is not modeled in roofline targets "
            f"({note.reason}: {note.detail})\n"
        )
        return RooflineCheckResult(config_name, chips, [], not_modeled=note)

    config_target = roofline_targets.targets[config_name]
    num_chips = chips if chips is not None else config_target.num_chips
    if num_chips is None or num_chips <= 0:
        raise ValueError(f"chip count required for roofline check of {config_name!r}")

    rows: list[RooflineRow] = []
    for key in sorted(results):
        if key not in config_target.cells:
            rows.append(
                RooflineRow(key, "(no roofline cell)", 0.0, 0.0, 0.0, "", "SKIP")
            )
            continue
        cell_target = config_target.cells[key]
        raw_tput = _finite_number(results[key].get("output_throughput"))
        if cell_target.status == "oom":
            skip = "SKIP (oom)"
            rows.append(
                RooflineRow(
                    key,
                    "output_token_throughput_per_chip",
                    0.0,
                    cell_target.output_token_throughput_per_chip,
                    0.0,
                    cell_target.status,
                    skip,
                )
            )
            continue
        if raw_tput is None or raw_tput < 0:
            rows.append(
                RooflineRow(
                    key,
                    "output_token_throughput_per_chip",
                    0.0,
                    cell_target.output_token_throughput_per_chip,
                    0.0,
                    cell_target.status,
                    "SKIP",
                )
            )
            continue
        actual_per_chip = raw_tput / num_chips
        ratio = actual_per_chip / cell_target.output_token_throughput_per_chip
        status = (
            "ADVISORY (unverified)"
            if cell_target.status == "unverified"
            else "ADVISORY"
        )
        rows.append(
            RooflineRow(
                key,
                "output_token_throughput_per_chip",
                actual_per_chip,
                cell_target.output_token_throughput_per_chip,
                ratio,
                cell_target.status,
                status,
            )
        )

    print(
        f"\n## Roofline speed check (`{config_name}`, "
        f"{num_chips} physical chips, advisory)\n"
    )
    print("| key | metric | actual/chip | roofline/chip | % of roofline | status |")
    print("|---|---|---:|---:|---:|---|")
    for row in rows:
        pct = "-" if row.status.startswith("SKIP") else f"{row.ratio * 100:.2f}%"
        print(
            f"| `{row.cell}` | {row.metric} | {row.actual_per_chip:.2f} | "
            f"{row.target_per_chip:.2f} | {pct} | {row.status} |"
        )
    print()
    return RooflineCheckResult(config_name, num_chips, rows)


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


# Baseline files keep metadata under top-level keys that start with "_"; the
# gates ignore them. The update policy lives under this one:
#   "_policy": {
#     "state": "development" | "frozen",   (absent block means development)
#     "owner": "<handle>",                 (required when frozen)
#     "approval": {"approver": "<handle>", "reference": "<PR or issue URL>"}
#   }
POLICY_KEY = "_policy"
DEVELOPMENT = "development"
FROZEN = "frozen"


class Approval(NamedTuple):
    approver: str
    reference: str


@dataclasses.dataclass(frozen=True)
class BaselinePolicy:
    """Who may change a baseline file, and whether it is frozen."""

    state: str = DEVELOPMENT
    owner: str | None = None
    approval: Approval | None = None


def _policy_str(block: dict, field: str, where: str) -> str:
    value = block.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}.{field} must be a non-empty string")
    return value


def parse_policy(baseline: dict) -> BaselinePolicy:
    """Read the `_policy` block; raise ValueError when it is malformed."""
    raw = baseline.get(POLICY_KEY)
    if raw is None:
        return BaselinePolicy()
    if not isinstance(raw, dict):
        raise ValueError(f"{POLICY_KEY} must be an object")
    state = raw.get("state", DEVELOPMENT)
    if state not in (DEVELOPMENT, FROZEN):
        raise ValueError(
            f"{POLICY_KEY}.state must be {DEVELOPMENT!r} or {FROZEN!r}, got {state!r}"
        )
    owner = None
    if state == FROZEN or "owner" in raw:
        owner = _policy_str(raw, "owner", POLICY_KEY)
    approval = None
    if "approval" in raw:
        block = raw["approval"]
        where = f"{POLICY_KEY}.approval"
        if not isinstance(block, dict):
            raise ValueError(f"{where} must be an object")
        approval = Approval(
            _policy_str(block, "approver", where),
            _policy_str(block, "reference", where),
        )
    return BaselinePolicy(state, owner, approval)


def load_baseline_file(path: Path) -> dict:
    """Read a baseline JSON file; raise ValueError when it is not an object."""
    try:
        baseline = json.loads(path.read_text())
    except OSError as e:
        raise ValueError(f"cannot read {path}: {e}") from e
    if not isinstance(baseline, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return baseline


def gated_entries(baseline: dict) -> dict:
    return {k: v for k, v in baseline.items() if not k.startswith("_")}


def check_baseline_update(old: dict | None, new: dict) -> str | None:
    """Return why replacing baseline `old` with `new` is not allowed, or None.

    A development baseline (or a new file, `old` None) changes freely. A frozen
    baseline changes its gated values, state, or owner only when `new` carries
    an approval record that the old file did not have and whose approver is the
    owner named in the old file.
    """
    new_policy = parse_policy(new)
    if old is None:
        return None
    old_policy = parse_policy(old)
    if old_policy.state != FROZEN:
        return None
    if gated_entries(old) == gated_entries(new) and (
        new_policy.state,
        new_policy.owner,
    ) == (old_policy.state, old_policy.owner):
        return None
    approval = new_policy.approval
    if approval is None or approval == old_policy.approval:
        return (
            "frozen baseline changed without a new "
            f"{POLICY_KEY}.approval record from owner {old_policy.owner!r}"
        )
    if approval.approver != old_policy.owner:
        return (
            f"frozen baseline changed with approval from {approval.approver!r}, "
            f"but its owner is {old_policy.owner!r}"
        )
    return None


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


# Fewest runs whose spread replaces the per-run reported stderr. The spread of
# N runs has N - 1 degrees of freedom; below 5 runs it is too rough an estimate
# to narrow a gate with, so fewer runs keep the single-run stderr and only the
# value is averaged.
MIN_RUNS_FOR_EMPIRICAL_STDERR = 5


def aggregate_eval_runs(runs: Sequence[dict[str, dict]]) -> dict[str, dict]:
    """Collapse the results of N runs into one entry per task.

    Each accuracy value becomes the mean over the runs. Its stderr depends on
    N:

    - N < MIN_RUNS_FOR_EMPIRICAL_STDERR: the mean of the stderrs the runs
      reported, present only when every run reported one. The gate is then as
      wide as for a single run.
    - N >= MIN_RUNS_FOR_EMPIRICAL_STDERR: the empirical standard error of the
      mean, ``stdev(values) / sqrt(N)``. Runs that agree exactly give 0.0, so
      a deterministic task falls back to the flat tolerance on its side.

    A metric that any run lacks is left out, so a run that lost a task shows
    up as a missing result instead of a mean over fewer runs.
    """
    if not runs:
        return {}
    n = len(runs)
    out: dict[str, dict] = {}
    for task in sorted(set().union(*runs)):
        entries = [run.get(task, {}) for run in runs]
        entry: dict[str, float] = {}
        for k in ACC_KEYS:
            values = [e[k] for e in entries if k in e]
            if not values:
                continue
            if len(values) < n:
                print(
                    f"Warning: {task} {k} is in {len(values)} of {n} runs; dropped.",
                    file=sys.stderr,
                )
                continue
            entry[k] = statistics.fmean(values)
            reported = [e[stderr_key(k)] for e in entries if stderr_key(k) in e]
            if n >= MIN_RUNS_FOR_EMPIRICAL_STDERR:
                entry[stderr_key(k)] = statistics.stdev(values) / math.sqrt(n)
            elif len(reported) == n:
                entry[stderr_key(k)] = statistics.fmean(reported)
        if entry:
            out[task] = {**entry, "num_runs": n}
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
    """Build an eval baseline as the mean of the runs in `results_dirs`.

    Raises ValueError when a run has no selected task or lacks a metric that
    another run has, so a broken run cannot shrink or empty the baseline.
    """
    runs = []
    for d in results_dirs:
        run = {
            task: metrics
            for task, metrics in load_eval_results(d).items()
            if task_filter is None or task in task_filter
        }
        if not run:
            raise ValueError(f"no selected eval results under {d}")
        runs.append(run)
    baseline = aggregate_eval_runs(runs)
    dropped = sorted(
        {
            f"{task} {k}"
            for run in runs
            for task, metrics in run.items()
            for k in ACC_KEYS
            if k in metrics and k not in baseline.get(task, {})
        }
    )
    if dropped:
        raise ValueError(f"not in every run: {'; '.join(dropped)}")
    return baseline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["perf", "eval", "update"],
        required=True,
        help="perf or eval gate results against a baseline; update checks "
        "that a baseline file change follows its freeze policy.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory containing result files. Repeat for calibrate, or in "
        "eval mode to compare the mean of several runs.",
    )
    parser.add_argument(
        "--baseline", type=Path, help="Baseline JSON to compare against."
    )
    parser.add_argument(
        "--previous-baseline",
        type=Path,
        help="Update mode: the baseline file before the change. Omit for a "
        "new baseline file.",
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
    parser.add_argument(
        "--roofline-targets",
        type=Path,
        help="Perf mode: roofline_targets.json for % of roofline reporting.",
    )
    parser.add_argument(
        "--config-name",
        help="Perf mode: config stem to look up in --roofline-targets.",
    )
    parser.add_argument(
        "--chips",
        type=int,
        default=None,
        help="Perf mode: total physical accelerator chips serving the run.",
    )
    parser.add_argument(
        "--prefill-chips",
        type=int,
        default=None,
        help="Perf mode: physical prefill chips in a disaggregated run.",
    )
    parser.add_argument(
        "--decode-chips",
        type=int,
        default=None,
        help="Perf mode: physical decode chips in a disaggregated run.",
    )
    args = parser.parse_args()

    perf_only = (
        args.roofline_targets,
        args.config_name,
        args.chips,
        args.prefill_chips,
        args.decode_chips,
    )
    if args.mode == "update":
        if any(v is not None for v in perf_only):
            parser.error("speed gate and chip flags are only valid in perf check mode")
        if args.baseline is None:
            parser.error("--baseline is required in update mode")
        try:
            old = None
            if args.previous_baseline is not None:
                old = load_baseline_file(args.previous_baseline)
            violation = check_baseline_update(old, load_baseline_file(args.baseline))
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        if violation:
            print(f"FAILED: {args.baseline}: {violation}", file=sys.stderr)
            return 1
        print("PASS: baseline update follows its freeze policy.")
        return 0
    if not args.results_dir:
        parser.error("--results-dir is required in perf and eval modes")

    tolerance = args.tolerance
    if tolerance is None:
        tolerance = 0.05 if args.mode == "perf" else 0.015

    if (args.mode == "eval" or args.calibrate) and any(
        v is not None for v in perf_only
    ):
        parser.error("speed gate and chip flags are only valid in perf check mode")

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
            try:
                baseline = calibrate_eval(args.results_dir, task_filter)
            except ValueError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                return 2
        json.dump(baseline, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    if args.mode == "eval":
        if args.baseline is None:
            parser.error("--baseline is required unless --calibrate is set")
        runs = [load_eval_results(d) for d in args.results_dir]
        for results_dir, run in zip(args.results_dir, runs):
            if not run:
                print(
                    f"ERROR: no results_*.json found under {results_dir}",
                    file=sys.stderr,
                )
                return 2
        # One run passes through untouched; several are gated on their mean.
        results = runs[0] if len(runs) == 1 else aggregate_eval_runs(runs)
        try:
            baseline = gated_entries(load_baseline_file(args.baseline))
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
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

    if len(args.results_dir) != 1:
        parser.error("--results-dir may only be given once in perf check mode")
    if args.baseline is None and args.roofline_targets is None:
        parser.error(
            "--baseline or --roofline-targets is required unless --calibrate is set"
        )
    if args.roofline_targets is not None and not args.config_name:
        parser.error("--config-name is required with --roofline-targets")

    try:
        chips = resolve_physical_chips(
            args.chips, args.prefill_chips, args.decode_chips
        )
        roofline_db = (
            load_roofline_targets(args.roofline_targets)
            if args.roofline_targets is not None
            else None
        )
        if (
            roofline_db is not None
            and args.config_name not in roofline_db.targets
            and args.config_name not in roofline_db.not_modeled
        ):
            raise ValueError(
                f"config {args.config_name!r} not found in roofline targets"
            )
        baseline_entries = (
            gated_entries(load_baseline_file(args.baseline))
            if args.baseline is not None
            else None
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    roofline_not_modeled = (
        roofline_db is not None and args.config_name in roofline_db.not_modeled
    )
    results = load_perf_results(args.results_dir[0])
    if not results and (baseline_entries is not None or not roofline_not_modeled):
        print(
            f"ERROR: no isl*_osl*_c*.json files found in {args.results_dir[0]}",
            file=sys.stderr,
        )
        return 2

    perf_failed = 0
    if baseline_entries is not None:
        perf_failed = check_perf(results, baseline_entries, tolerance)

    if roofline_db is not None:
        try:
            check_roofline(results, roofline_db, args.config_name, chips=chips)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    if perf_failed:
        print(
            f"FAILED: {perf_failed} metric(s) regressed beyond {tolerance * 100:.2f}%",
            file=sys.stderr,
        )
        return 1
    if baseline_entries is not None:
        print("PASS: all gated metrics within tolerance.")
    else:
        print("PASS: advisory roofline check complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
