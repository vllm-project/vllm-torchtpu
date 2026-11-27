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
"""AgentX result loading and GPU reference curve speed gating."""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import baseline_policy  # noqa: E402
import roofline_gate  # noqa: E402

FLOAT_EPSILON = roofline_gate.FLOAT_EPSILON

AGENTX_AGGREGATE_RE = re.compile(r"^(.+)_c([1-9]\d*)\.json$")
CONC_DIR_RE = re.compile(r"^conc_([1-9]\d*)$")
_PERF_CELL_FILENAME_RE = re.compile(r"^isl(\d+)_osl(\d+)_c(\d+)\.json$")

AGENTX_ARG_NAMES = frozenset({"agentx_gpu_reference", "min_gpu_ratio"})

_LATENCY_UNIT_TO_MS: dict[str, float] = {
    "ms": 1.0,
    "msec": 1.0,
    "millisecond": 1.0,
    "milliseconds": 1.0,
    "s": 1000.0,
    "sec": 1000.0,
    "second": 1000.0,
    "seconds": 1000.0,
    "us": 1e-3,
    "usec": 1e-3,
    "microsecond": 1e-3,
    "microseconds": 1e-3,
    "µs": 1e-3,
    "μs": 1e-3,
    "ns": 1e-6,
    "nsec": 1e-6,
    "nanosecond": 1e-6,
    "nanoseconds": 1e-6,
}


def add_agentx_args(parser: argparse.ArgumentParser) -> None:
    """Register AgentX GPU reference speed gate CLI flags on `parser`."""
    parser.add_argument(
        "--agentx-gpu-reference",
        type=Path,
        default=None,
        help="Perf check mode: JSON of {p90_interactivity, out_tok_s_per_gpu} points.",
    )
    parser.add_argument(
        "--min-gpu-ratio",
        type=float,
        default=None,
        help="Perf check mode: minimum fraction of interpolated GPU AgentX throughput "
        "required per physical chip. Advisory when omitted.",
    )


class AgentxPoint(NamedTuple):
    key: str
    concurrency: int | None
    output_tok_s_total: float
    p90_interactivity: float
    invalid_reason: str | None = None


class GpuReferencePoint(NamedTuple):
    p90_interactivity: float
    out_tok_s_per_gpu: float


class AgentxGpuRow(NamedTuple):
    key: str
    p90_interactivity: float
    actual_per_chip: float
    gpu_ref_per_gpu: float | None
    ratio: float | None
    status: str
    invalid_reason: str | None = None


@dataclasses.dataclass(frozen=True)
class AgentxGpuCheckResult:
    """Outcome of an AgentX GPU reference speed check."""

    config_name: str
    rows: list[AgentxGpuRow]
    ref_range: tuple[float, float]
    out_of_range_failed: bool = False

    @property
    def failed_rows(self) -> list[AgentxGpuRow]:
        return [r for r in self.rows if r.status.startswith("FAIL")]

    def __bool__(self) -> bool:
        return bool(self.failed_rows) or self.out_of_range_failed


def _dotted_number(data: Any, path: str) -> float | None:
    cur = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return roofline_gate.finite_number(cur)


def _aiperf_stat(
    data: dict[str, Any], metric: str, stat: str
) -> tuple[float | None, Any]:
    block = data.get(metric)
    if isinstance(block, dict):
        return roofline_gate.finite_number(block.get(stat)), block.get("unit")
    if stat == "avg":
        return roofline_gate.finite_number(block), None
    return None, None


def _latency_to_ms(value: float, unit: Any, path: Path, metric: str) -> float:
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError(f"{path}: {metric} requires a latency unit")
    scale = _LATENCY_UNIT_TO_MS.get(unit.strip().lower())
    if scale is None:
        raise ValueError(f"{path}: unrecognized latency unit {unit!r} for {metric}")
    return value * scale


def _parse_agentx_metrics(
    path: Path, data: dict[str, Any], *, allow_missing: bool = False
) -> tuple[float, float]:
    output_tput = _dotted_number(
        data, "request_metrics.throughput.output.tokens_per_second"
    )
    if output_tput is None:
        output_tput, _ = _aiperf_stat(data, "output_token_throughput", "avg")
    if output_tput is None:
        output_tput = roofline_gate.finite_number(data.get("output_throughput"))

    p90_intvty = _dotted_number(data, "request_metrics.latency.intvty.p90")
    if p90_intvty is None:
        p90_intvty = roofline_gate.finite_number(data.get("p90_interactivity"))
    if p90_intvty is None:
        tpot_ms = roofline_gate.finite_number(data.get("p90_tpot_ms"))
        if tpot_ms is None:
            for metric in ("inter_token_latency", "time_per_output_token"):
                val, unit = _aiperf_stat(data, metric, "p90")
                if val is not None:
                    tpot_ms = _latency_to_ms(val, unit, path, metric)
                    break
        if tpot_ms is not None and tpot_ms > 0:
            p90_intvty = 1000.0 / tpot_ms

    if output_tput is None or output_tput < 0 or p90_intvty is None or p90_intvty <= 0:
        if allow_missing:
            return max(output_tput or 0.0, 0.0), max(p90_intvty or 0.0, 0.0)
        raise ValueError(
            f"{path}: AgentX result requires non-negative output throughput "
            "and positive P90 interactivity"
        )
    return output_tput, p90_intvty


def _conc_dir_scope(results_dir: Path, path: Path) -> tuple[Path, int] | None:
    """Return `(parent_dir, conc)` for the innermost `conc_<N>` ancestor in root."""
    root = results_dir.resolve()
    cur = path.parent.resolve()
    if not cur.is_relative_to(root):
        return None
    while True:
        m = CONC_DIR_RE.match(cur.name)
        if m and cur != root:
            return cur.parent, int(m.group(1))
        if cur == root or cur.parent == cur:
            return None
        cur = cur.parent


def _aiperf_concurrency(
    results_dir: Path, path: Path, data: dict[str, Any]
) -> int | None:
    input_cfg = data.get("input_config")
    phases = input_cfg.get("phases") if isinstance(input_cfg, dict) else None
    for phase in phases if isinstance(phases, list) else ():
        if isinstance(phase, dict) and phase.get("kind") == "profiling":
            conc = phase.get("concurrency")
            if not isinstance(conc, bool) and isinstance(conc, int) and conc > 0:
                return conc
    scope = _conc_dir_scope(results_dir, path)
    return scope[1] if scope is not None else None


def _has_hard_timeout_ancestor(results_dir: Path, path: Path) -> bool:
    """Check whether any directory from `path.parent` up to `results_dir` timed out."""
    root = results_dir.resolve()
    cur = path.parent.resolve()
    if not cur.is_relative_to(root):
        return False
    while True:
        if (cur / "HARD_TIMEOUT").exists():
            return True
        if cur == root or cur.parent == cur:
            return False
        cur = cur.parent


def _agentx_invalid_reason(*, has_timeout: bool, was_cancelled: bool) -> str | None:
    """Unified precedence rule: HARD_TIMEOUT wins over was_cancelled=true."""
    if has_timeout:
        return "HARD_TIMEOUT"
    if was_cancelled:
        return "was_cancelled=true"
    return None


def _point_key_for_scope(root: Path, parent: Path | None, conc: int | None) -> str:
    base = f"c{conc}" if conc is not None else "profile_export_aiperf"
    if parent is not None and parent != root and parent.is_relative_to(root):
        return f"{parent.relative_to(root).as_posix()}/{base}"
    return base


def _load_json_allowing_timeout(
    path: Path, *, has_timeout: bool
) -> dict[str, Any] | None:
    if not has_timeout:
        return baseline_policy.load_json_object(path)
    try:
        return baseline_policy.load_json_object(path)
    except ValueError:
        return None


def load_agentx_results(results_dir: Path) -> list[AgentxPoint]:
    """Load AgentX points from top-level `<label>_c<N>.json` and aiperf exports."""
    root = results_dir.resolve()

    timeout_scopes: set[tuple[Path, int]] = set()
    for marker in sorted(results_dir.rglob("HARD_TIMEOUT")):
        marker_dir = marker.parent.resolve()
        if not marker_dir.is_relative_to(root) or marker_dir == root:
            continue
        m = CONC_DIR_RE.match(marker_dir.name)
        if m:
            timeout_scopes.add((marker_dir.parent, int(m.group(1))))

    cancelled_scopes: set[tuple[Path, int]] = set()
    aiperf_exports: list[
        tuple[
            Path,
            dict[str, Any] | None,
            int | None,
            tuple[Path, int] | None,
            Path,
            str | None,
        ]
    ] = []
    for path in sorted(results_dir.rglob("profile_export_aiperf.json")):
        scope = _conc_dir_scope(results_dir, path)
        has_timeout = _has_hard_timeout_ancestor(results_dir, path) or (
            scope is not None and scope in timeout_scopes
        )
        data = _load_json_allowing_timeout(path, has_timeout=has_timeout)
        if data is not None:
            conc = _aiperf_concurrency(results_dir, path, data)
        else:
            conc = scope[1] if scope is not None else None
        parent_dir = scope[0] if scope is not None else path.parent.resolve()
        export_scope = (
            scope
            if scope is not None
            else ((parent_dir, conc) if conc is not None else None)
        )
        was_cancelled = bool(data is not None and data.get("was_cancelled") is True)
        if was_cancelled and export_scope is not None:
            cancelled_scopes.add(export_scope)
        reason = _agentx_invalid_reason(
            has_timeout=has_timeout, was_cancelled=was_cancelled
        )
        aiperf_exports.append((path, data, conc, export_scope, parent_dir, reason))

    points: list[AgentxPoint] = []
    seen_keys: set[str] = set()
    covered_scopes: set[tuple[Path, int]] = set()

    # Aggregates live at top level (matching agentx_client.sh).
    for path in sorted(results_dir.glob("*_c*.json")):
        if (
            not path.is_file()
            or _PERF_CELL_FILENAME_RE.match(path.name)
            or path.name.startswith("results_")
            or path.name.endswith(".baseline.json")
        ):
            continue
        m = AGENTX_AGGREGATE_RE.match(path.name)
        if not m:
            continue
        conc = int(m.group(2))
        agg_scope = (path.parent.resolve(), conc)
        has_timeout = agg_scope in timeout_scopes or _has_hard_timeout_ancestor(
            results_dir, path
        )
        data = _load_json_allowing_timeout(path, has_timeout=has_timeout)
        was_cancelled = agg_scope in cancelled_scopes or bool(
            data is not None and data.get("was_cancelled") is True
        )
        reason = _agentx_invalid_reason(
            has_timeout=has_timeout, was_cancelled=was_cancelled
        )
        if data is None:
            output_tput, p90_intvty = 0.0, 0.0
        else:
            output_tput, p90_intvty = _parse_agentx_metrics(
                path, data, allow_missing=reason is not None
            )
        if path.stem in seen_keys:
            raise ValueError(f"duplicate AgentX result {path.stem!r}")
        seen_keys.add(path.stem)
        covered_scopes.add(agg_scope)
        points.append(AgentxPoint(path.stem, conc, output_tput, p90_intvty, reason))

    for path, data, conc, export_scope, parent_dir, reason in aiperf_exports:
        if export_scope is not None and export_scope in covered_scopes:
            continue
        if data is None:
            output_tput, p90_intvty = 0.0, 0.0
        else:
            output_tput, p90_intvty = _parse_agentx_metrics(
                path, data, allow_missing=reason is not None
            )
        key = _point_key_for_scope(root, parent_dir, conc)
        if key in seen_keys:
            raise ValueError(f"duplicate AgentX concurrency point {key!r}")
        seen_keys.add(key)
        if export_scope is not None:
            covered_scopes.add(export_scope)
        points.append(AgentxPoint(key, conc, output_tput, p90_intvty, reason))

    for parent, conc in sorted(timeout_scopes - covered_scopes):
        key = _point_key_for_scope(root, parent, conc)
        points.append(AgentxPoint(key, conc, 0.0, 0.0, "HARD_TIMEOUT"))

    return sorted(
        points, key=lambda p: (p.concurrency is None, p.concurrency or 0, p.key)
    )


def load_agentx_points_checked(results_dir: Path) -> list[AgentxPoint]:
    """Load AgentX points or raise ValueError when none exist under results_dir."""
    points = load_agentx_results(results_dir)
    if not points:
        raise ValueError(
            "no AgentX result files (profile_export_aiperf.json or "
            f"<label>_c<N>.json) found under {results_dir}"
        )
    return points


def load_gpu_reference(path: Path) -> list[GpuReferencePoint]:
    """Load GPU `(p90_interactivity, out_tok_s_per_gpu)` points sorted by x.

    Accepts a list or {"points": [...]} object.
    """
    try:
        raw = json.loads(
            path.read_text(), parse_constant=baseline_policy.reject_json_constant
        )
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot read GPU reference {path}: {e}") from e
    items = raw.get("points") if isinstance(raw, dict) else raw
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path} must be a non-empty list of reference points")
    points: list[GpuReferencePoint] = []
    seen_x: set[float] = set()
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{path}[{idx}] must be an object")
        x = roofline_gate.positive_ratio(
            item.get("p90_interactivity"), f"{path}[{idx}].p90_interactivity"
        )
        y = roofline_gate.positive_ratio(
            item.get("out_tok_s_per_gpu"), f"{path}[{idx}].out_tok_s_per_gpu"
        )
        if x in seen_x:
            raise ValueError(f"{path}: duplicate p90_interactivity {x}")
        seen_x.add(x)
        points.append(GpuReferencePoint(x, y))
    return sorted(points, key=lambda p: p.p90_interactivity)


def interpolate_gpu_reference(
    reference: Sequence[GpuReferencePoint], p90_interactivity: float
) -> float | None:
    """Linearly interpolate `out_tok_s_per_gpu` at `p90_interactivity`, or None."""
    if not reference:
        return None
    if (
        p90_interactivity < reference[0].p90_interactivity - FLOAT_EPSILON
        or p90_interactivity > reference[-1].p90_interactivity + FLOAT_EPSILON
    ):
        return None
    if abs(p90_interactivity - reference[0].p90_interactivity) <= FLOAT_EPSILON:
        return reference[0].out_tok_s_per_gpu
    if abs(p90_interactivity - reference[-1].p90_interactivity) <= FLOAT_EPSILON:
        return reference[-1].out_tok_s_per_gpu
    for lo, hi in zip(reference, reference[1:]):
        if lo.p90_interactivity <= p90_interactivity <= hi.p90_interactivity:
            span = hi.p90_interactivity - lo.p90_interactivity
            weight = (p90_interactivity - lo.p90_interactivity) / span
            return lo.out_tok_s_per_gpu + weight * (
                hi.out_tok_s_per_gpu - lo.out_tok_s_per_gpu
            )
    return None


def check_agentx_gpu(
    points: Sequence[AgentxPoint],
    reference: Sequence[GpuReferencePoint],
    num_chips: int,
    config_name: str,
    *,
    min_ratio: float | None = None,
) -> AgentxGpuCheckResult:
    """Compare AgentX per-chip output throughput against interpolated GPU points."""
    if not reference:
        raise ValueError("GPU reference must be non-empty")

    rows: list[AgentxGpuRow] = []
    in_range_count = 0
    for point in points:
        actual_per_chip = point.output_tok_s_total / num_chips
        if point.invalid_reason is not None:
            gpu_ref = (
                interpolate_gpu_reference(reference, point.p90_interactivity)
                if point.p90_interactivity > 0
                else None
            )
            ratio = actual_per_chip / gpu_ref if gpu_ref is not None else None
            prefix = "ADVISORY" if min_ratio is None else "FAIL"
            rows.append(
                AgentxGpuRow(
                    point.key,
                    point.p90_interactivity,
                    actual_per_chip,
                    gpu_ref,
                    ratio,
                    f"{prefix} ({point.invalid_reason})",
                    point.invalid_reason,
                )
            )
            continue
        gpu_ref = interpolate_gpu_reference(reference, point.p90_interactivity)
        if gpu_ref is None:
            rows.append(
                AgentxGpuRow(
                    point.key,
                    point.p90_interactivity,
                    actual_per_chip,
                    None,
                    None,
                    "SKIP (out of range)",
                )
            )
            continue
        in_range_count += 1
        ratio = actual_per_chip / gpu_ref
        if min_ratio is None:
            status = "ADVISORY"
        else:
            status = "PASS" if ratio + FLOAT_EPSILON >= min_ratio else "FAIL"
        rows.append(
            AgentxGpuRow(
                point.key,
                point.p90_interactivity,
                actual_per_chip,
                gpu_ref,
                ratio,
                status,
            )
        )

    mode_desc = (
        "advisory" if min_ratio is None else f"min ratio: {min_ratio * 100:.2f}%"
    )
    print(
        f"\n## AgentX GPU reference speed check (`{config_name}`, "
        f"{num_chips} physical chips, {mode_desc})\n"
    )
    print("| point | p90 intvty | actual/chip | gpu ref/gpu | % of gpu | status |")
    print("|---|---:|---:|---:|---:|---|")
    for row in rows:
        ref_str = "-" if row.gpu_ref_per_gpu is None else f"{row.gpu_ref_per_gpu:.2f}"
        pct_str = "-" if row.ratio is None else f"{row.ratio * 100:.2f}%"
        print(
            f"| `{row.key}` | {row.p90_interactivity:.2f} | "
            f"{row.actual_per_chip:.2f} | {ref_str} | {pct_str} | {row.status} |"
        )
    print()

    ref_range = (reference[0].p90_interactivity, reference[-1].p90_interactivity)
    has_row_failures = any(r.status.startswith("FAIL") for r in rows)
    out_of_range_failed = (
        min_ratio is not None
        and bool(rows)
        and in_range_count == 0
        and not has_row_failures
    )
    return AgentxGpuCheckResult(config_name, rows, ref_range, out_of_range_failed)


def format_agentx_failures(result: AgentxGpuCheckResult, min_ratio: float) -> list[str]:
    """Build stderr FAILED diagnostic lines for a failing AgentX GPU gate."""
    lines: list[str] = []
    for row in result.failed_rows:
        if row.invalid_reason is not None:
            lines.append(
                f"FAILED: config={result.config_name} point={row.key} "
                f"metric=out_tok_s_per_chip measured=invalid "
                f"({row.invalid_reason}) "
                f"target>={min_ratio * 100:.2f}% of GPU reference"
            )
            continue
        assert row.gpu_ref_per_gpu is not None and row.ratio is not None
        min_target = row.gpu_ref_per_gpu * min_ratio
        lines.append(
            f"FAILED: config={result.config_name} point={row.key} "
            "metric=out_tok_s_per_chip@"
            f"p90_interactivity={row.p90_interactivity:.2f} "
            f"measured={row.actual_per_chip:.2f} tok/s/chip "
            f"({row.ratio * 100:.2f}%) "
            f"target>={min_target:.2f} tok/s/chip "
            f"({min_ratio * 100:.2f}% of {row.gpu_ref_per_gpu:.2f})"
        )
    if result.out_of_range_failed:
        lo_x, hi_x = result.ref_range
        lines.append(
            f"FAILED: config={result.config_name} "
            f"metric=out_tok_s_per_chip measured=0 in-range points "
            f"target=p90_interactivity within [{lo_x:.2f}, {hi_x:.2f}]"
        )
    return lines
