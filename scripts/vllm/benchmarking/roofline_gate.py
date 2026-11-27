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
"""Roofline target loading and speed gating for check_regression.py."""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path
from typing import Any, NamedTuple

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import baseline_policy  # noqa: E402

# Guards exact-threshold ratio checks against binary floating-point error.
FLOAT_EPSILON = 1e-9

ROOFLINE_CELL_STATUSES = ("ok", "unverified", "oom")
ROOFLINE_ARG_NAMES = frozenset(
    {
        "roofline_targets",
        "config_name",
        "chips",
        "prefill_chips",
        "decode_chips",
        "min_roofline_ratio",
        "enforce_unverified_rooflines",
    }
)


def add_roofline_args(parser: argparse.ArgumentParser) -> None:
    """Register roofline and chip-count CLI flags on `parser`."""
    parser.add_argument(
        "--roofline-targets",
        type=Path,
        help="Perf check mode: roofline_targets.json, to print each measured "
        "cell as a percentage of its roofline. Needs --config-name.",
    )
    parser.add_argument(
        "--config-name",
        help="The config to look up in --roofline-targets, or the label for "
        "--agentx-gpu-reference.",
    )
    parser.add_argument(
        "--chips",
        type=int,
        default=None,
        help="Physical chips serving the run, for --roofline-targets or "
        "--agentx-gpu-reference. Default for roofline: the config's num_chips "
        "in that file.",
    )
    parser.add_argument(
        "--prefill-chips",
        type=int,
        default=None,
        help="Physical prefill chips of a disaggregated run; give it with "
        "--decode-chips instead of --chips.",
    )
    parser.add_argument(
        "--decode-chips",
        type=int,
        default=None,
        help="Physical decode chips of a disaggregated run.",
    )
    parser.add_argument(
        "--min-roofline-ratio",
        type=float,
        default=None,
        help="Perf check mode: minimum fraction of roofline throughput required "
        "(for example 0.8). Advisory when omitted.",
    )
    parser.add_argument(
        "--enforce-unverified-rooflines",
        action="store_true",
        help="Perf check mode: enforce --min-roofline-ratio on status=='unverified' "
        "roofline cells too.",
    )


def validate_perf_speed_flags(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Enforce companion-flag requirements for roofline and AgentX speed flags."""
    if args.roofline_targets is None and args.agentx_gpu_reference is None:
        for flag, value in (
            ("--config-name", args.config_name),
            ("--chips", args.chips),
            ("--prefill-chips", args.prefill_chips),
            ("--decode-chips", args.decode_chips),
        ):
            if value is not None:
                parser.error(
                    f"{flag} requires --roofline-targets or --agentx-gpu-reference"
                )
    if args.roofline_targets is not None and not args.config_name:
        parser.error("--config-name is required with --roofline-targets")
    if args.min_roofline_ratio is not None and args.roofline_targets is None:
        parser.error("--min-roofline-ratio requires --roofline-targets")
    if args.enforce_unverified_rooflines and args.min_roofline_ratio is None:
        parser.error("--enforce-unverified-rooflines requires --min-roofline-ratio")
    if args.min_gpu_ratio is not None and args.agentx_gpu_reference is None:
        parser.error("--min-gpu-ratio requires --agentx-gpu-reference")


class RooflineCellTarget(NamedTuple):
    output_token_throughput_per_chip: float
    status: str


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
    status: str
    failure_reason: str | None = None


@dataclasses.dataclass(frozen=True)
class RooflineCheckResult:
    """Outcome of a roofline speed check; truthy when any cell failed."""

    config_name: str
    rows: list[RooflineRow]

    @property
    def failed_rows(self) -> list[RooflineRow]:
        return [r for r in self.rows if r.status.startswith("FAIL")]

    def __bool__(self) -> bool:
        return bool(self.failed_rows)


@dataclasses.dataclass(frozen=True)
class ResolvedRoofline:
    """Validated roofline target config with a concrete physical chip count."""

    config_name: str
    cells: dict[str, RooflineCellTarget]
    num_chips: int
    modeled_num_chips: int | None = None


def finite_number(value: Any) -> float | None:
    """Return float(value) if value is a finite real number (not bool), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    val = float(value)
    return val if math.isfinite(val) else None


def positive_int(value: Any, name: str) -> int:
    """Validate that value is a strictly positive int (not bool)."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def positive_ratio(value: Any, name: str) -> float:
    """Validate that value is a strictly positive finite float (not bool)."""
    num = finite_number(value)
    if num is None or num <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return num


def resolve_physical_chips(
    chips: int | None, prefill_chips: int | None, decode_chips: int | None
) -> int | None:
    """Total physical chips from --chips or --prefill-chips plus --decode-chips."""
    if chips is not None:
        positive_int(chips, "--chips")
    if prefill_chips is not None:
        positive_int(prefill_chips, "--prefill-chips")
    if decode_chips is not None:
        positive_int(decode_chips, "--decode-chips")
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
        num_chips = positive_int(raw["num_chips"], f"{name}.num_chips")
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
        if status == "oom":
            target_tput = 0.0
        else:
            tput = finite_number(cell.get("output_token_throughput_per_chip"))
            if tput is None or tput <= 0:
                raise ValueError(
                    f"{name}.cells.{cell_key}.output_token_throughput_per_chip "
                    "must be positive"
                )
            target_tput = tput
        cells[cell_key] = RooflineCellTarget(target_tput, status)
    return RooflineConfigTarget(num_chips, cells)


def load_roofline_targets(path: Path) -> RooflineTargets:
    """Load and validate a roofline_targets.json file."""
    raw = baseline_policy.load_json_object(path, label="roofline targets")
    if not isinstance(raw.get("targets"), dict):
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
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{path}: not_modeled.{name}.reason must be non-empty")
        raw_detail = entry.get("detail")
        if raw_detail is None:
            detail = ""
        elif isinstance(raw_detail, str):
            detail = raw_detail
        else:
            raise ValueError(f"{path}: not_modeled.{name}.detail must be a string")
        not_modeled[name] = NotModeledEntry(reason, detail)

    overlap = sorted(set(targets) & set(not_modeled))
    if overlap:
        names = ", ".join(repr(k) for k in overlap)
        raise ValueError(
            f"{path}: config(s) listed in both 'targets' and 'not_modeled': {names}"
        )
    return RooflineTargets(targets, not_modeled)


def resolve_roofline_config(
    roofline: RooflineTargets, config_name: str, chips: int | None
) -> tuple[NotModeledEntry | None, ResolvedRoofline | None]:
    """Look up config_name in roofline targets and resolve its physical chip count."""
    not_modeled = roofline.not_modeled.get(config_name)
    if not_modeled is not None:
        return not_modeled, None
    if config_name not in roofline.targets:
        raise ValueError(f"config {config_name!r} not found in roofline targets")
    target = roofline.targets[config_name]
    num_chips = chips if chips is not None else target.num_chips
    if num_chips is None:
        raise ValueError(f"chip count required for roofline check of {config_name!r}")
    return None, ResolvedRoofline(
        config_name, target.cells, num_chips, target.num_chips
    )


def validate_roofline_matches(
    results: dict[str, dict[str, Any]],
    cells: dict[str, RooflineCellTarget],
    config_name: str,
    *,
    min_ratio: float | None = None,
) -> None:
    """Require at least one non-oom matched cell when blocking roofline gating."""
    if min_ratio is not None and not any(
        k in cells and cells[k].status != "oom" for k in results
    ):
        raise ValueError(
            f"no non-oom measured cells match roofline targets for {config_name!r}"
        )


def print_not_modeled_note(config_name: str, entry: NotModeledEntry) -> None:
    """Print the advisory note for a config in `not_modeled`."""
    print(f"\n## Roofline speed check (`{config_name}`: not modeled)\n")
    detail_suffix = f": {entry.detail}" if entry.detail else ""
    print(
        f"NOTE: `{config_name}` is not modeled in roofline targets "
        f"({entry.reason}{detail_suffix})\n"
    )


def warn_if_chip_count_differs(resolved: ResolvedRoofline) -> None:
    """Print a stderr note when explicit chips differ from the modeled chip count."""
    if (
        resolved.modeled_num_chips is not None
        and resolved.num_chips != resolved.modeled_num_chips
    ):
        print(
            f"NOTE: using {resolved.num_chips} chips "
            f"(modeled on {resolved.modeled_num_chips} chips)",
            file=sys.stderr,
        )


def check_roofline(
    results: dict[str, dict[str, Any]],
    cells: dict[str, RooflineCellTarget],
    config_name: str,
    num_chips: int,
    *,
    min_ratio: float | None = None,
    enforce_unverified: bool = False,
) -> RooflineCheckResult:
    """Compare measured output throughput per physical chip against roofline."""
    validate_roofline_matches(results, cells, config_name, min_ratio=min_ratio)

    rows: list[RooflineRow] = []
    for key in sorted(results):
        if key not in cells:
            rows.append(RooflineRow(key, "(no roofline cell)", 0.0, 0.0, 0.0, "SKIP"))
            continue
        cell = cells[key]
        if cell.status == "oom":
            rows.append(
                RooflineRow(
                    key,
                    "output_token_throughput_per_chip",
                    0.0,
                    cell.output_token_throughput_per_chip,
                    0.0,
                    "SKIP (oom)",
                )
            )
            continue
        unverified = cell.status == "unverified"
        gated = min_ratio is not None and (not unverified or enforce_unverified)
        measured = finite_number(results[key].get("output_throughput"))
        if measured is None or measured < 0:
            reason = (
                "missing output_throughput"
                if measured is None
                else "negative output_throughput"
            )
            rows.append(
                RooflineRow(
                    key,
                    "output_token_throughput_per_chip",
                    0.0,
                    cell.output_token_throughput_per_chip,
                    0.0,
                    f"FAIL ({reason})" if gated else "SKIP",
                    reason if gated else None,
                )
            )
            continue
        actual_per_chip = measured / num_chips
        ratio = actual_per_chip / cell.output_token_throughput_per_chip
        if not gated:
            status = "ADVISORY (unverified)" if unverified else "ADVISORY"
        else:
            assert min_ratio is not None
            ok = ratio + FLOAT_EPSILON >= min_ratio
            suffix = " (unverified)" if unverified else ""
            status = f"{'PASS' if ok else 'FAIL'}{suffix}"
        rows.append(
            RooflineRow(
                key,
                "output_token_throughput_per_chip",
                actual_per_chip,
                cell.output_token_throughput_per_chip,
                ratio,
                status,
            )
        )

    mode_desc = (
        "advisory" if min_ratio is None else f"min ratio: {min_ratio * 100:.2f}%"
    )
    print(
        f"\n## Roofline speed check (`{config_name}`, "
        f"{num_chips} physical chips, {mode_desc})\n"
    )
    print("| key | metric | actual/chip | roofline/chip | % of roofline | status |")
    print("|---|---|---:|---:|---:|---|")
    for row in rows:
        pct_str = (
            "-"
            if (
                row.target_per_chip <= 0
                or row.status.startswith("SKIP")
                or row.failure_reason is not None
            )
            else f"{row.ratio * 100:.2f}%"
        )
        print(
            f"| `{row.cell}` | {row.metric} | {row.actual_per_chip:.2f} | "
            f"{row.target_per_chip:.2f} | {pct_str} | {row.status} |"
        )
    print()
    return RooflineCheckResult(config_name, rows)


def format_roofline_failures(
    result: RooflineCheckResult, min_ratio: float
) -> list[str]:
    """Build stderr FAILED diagnostic lines for a failing roofline gate."""
    lines: list[str] = []
    for row in result.failed_rows:
        min_target = row.target_per_chip * min_ratio
        if row.failure_reason is not None:
            lines.append(
                f"FAILED: config={result.config_name} cell={row.cell} "
                f"metric={row.metric} "
                f"measured=invalid ({row.failure_reason}) "
                f"target>={min_target:.2f} tok/s/chip "
                f"({min_ratio * 100:.2f}% of {row.target_per_chip:.2f})"
            )
        else:
            lines.append(
                f"FAILED: config={result.config_name} cell={row.cell} "
                f"metric={row.metric} "
                f"measured={row.actual_per_chip:.2f} tok/s/chip "
                f"({row.ratio * 100:.2f}%) "
                f"target>={min_target:.2f} tok/s/chip "
                f"({min_ratio * 100:.2f}% of {row.target_per_chip:.2f})"
            )
    return lines
