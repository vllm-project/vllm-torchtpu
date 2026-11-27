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
"""Roofline target loading and speed reporting for check_regression.py."""

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

ROOFLINE_CELL_STATUSES = ("ok", "unverified", "oom")
ROOFLINE_ARG_NAMES = frozenset(
    {"roofline_targets", "config_name", "chips", "prefill_chips", "decode_chips"}
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
        help="The config to look up in --roofline-targets.",
    )
    parser.add_argument(
        "--chips",
        type=int,
        default=None,
        help="Physical chips serving the run, for --roofline-targets. "
        "Default: the config's num_chips in that file.",
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
) -> None:
    """Print measured output throughput per physical chip as a % of roofline."""
    rows: list[tuple[str, str, float, float, str, str]] = []
    for key in sorted(results):
        if key not in cells:
            rows.append((key, "(no roofline cell)", 0.0, 0.0, "-", "SKIP"))
            continue
        cell = cells[key]
        if cell.status == "oom":
            rows.append(
                (
                    key,
                    "output_token_throughput_per_chip",
                    0.0,
                    cell.output_token_throughput_per_chip,
                    "-",
                    "SKIP (oom)",
                )
            )
            continue
        measured = finite_number(results[key].get("output_throughput"))
        if measured is None or measured < 0:
            rows.append(
                (
                    key,
                    "output_token_throughput_per_chip",
                    0.0,
                    cell.output_token_throughput_per_chip,
                    "-",
                    "SKIP",
                )
            )
            continue
        actual_per_chip = measured / num_chips
        ratio = actual_per_chip / cell.output_token_throughput_per_chip
        status = "ADVISORY (unverified)" if cell.status == "unverified" else "ADVISORY"
        rows.append(
            (
                key,
                "output_token_throughput_per_chip",
                actual_per_chip,
                cell.output_token_throughput_per_chip,
                f"{ratio * 100:.2f}%",
                status,
            )
        )

    print(
        f"\n## Roofline speed check (`{config_name}`, "
        f"{num_chips} physical chips, advisory)\n"
    )
    print("| key | metric | actual/chip | roofline/chip | % of roofline | status |")
    print("|---|---|---:|---:|---:|---|")
    for cell_key, metric, actual, target, pct, status in rows:
        print(
            f"| `{cell_key}` | {metric} | {actual:.2f} | "
            f"{target:.2f} | {pct} | {status} |"
        )
    print()
