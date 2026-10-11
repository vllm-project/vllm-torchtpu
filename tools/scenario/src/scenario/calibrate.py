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
"""calibrate: baselines from run directories, written back into the scenario.

The values are check_regression.py's own calibration, the medians over the
runs: calibrate_perf for perf cells and calibrate_eval for lm_eval runs.
Writing rewrites only the baseline entries it has values for, in the file
that holds them; every other byte stays, comments included, since ruamel
gives the entries' lines and only those lines are replaced. A baseline
that is `from:` another scenario is refused, naming that scenario.
"""

import json
import pathlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from scenario.plan.build import plan
from scenario.plan.types import EvalTaskPlan, GatePlan, PerfTaskPlan
from scenario.resolve import Resolved
from scenario.runtime.gate import load_check_regression

Baseline = dict[str, dict[str, Any]]


class CalibrateError(ValueError):
    """The scenario or its file cannot be calibrated as asked."""


@dataclass
class Proposal:
    """New entries for one task's baseline."""

    task: str
    current: Baseline
    new: Baseline
    # Keys of the task with no result in any run; their entries are kept.
    missing: list[str] = field(default_factory=list)


def _current(gates: Sequence[GatePlan | None]) -> Baseline:
    found: Baseline = {}
    for gate in gates:
        if gate is not None:
            found |= gate.as_dict()
    return found


def propose(
    resolved: Resolved, run_dirs: Sequence[pathlib.Path], repo_root: pathlib.Path
) -> list[Proposal]:
    """New baselines for the point's perf and lm_eval tasks from `run_dirs`.

    Raises:
        CalibrateError: a baseline is `from:` another scenario, or a task
            cannot be calibrated yet.
        GateError: check_regression.py is not under `repo_root`.
    """
    for task, ref in sorted(resolved.ref_baselines.items()):
        raise CalibrateError(
            f"tasks.{task}.baseline is from: {ref.source}; calibrate {ref.source} "
            "instead"
        )
    check = load_check_regression(repo_root)
    dirs = list(run_dirs)
    proposals = []
    for task in plan(resolved).driver.tasks:
        if isinstance(task, PerfTaskPlan):
            if task.mode == "cells":
                raise CalibrateError("perf cells mode is not calibrated yet")
            keys = [cell.key for cell in task.cells]
            found = check.calibrate_perf(dirs, None)
            proposals.append(
                Proposal(
                    "perf",
                    _current([task.gate]),
                    {key: found[key] for key in keys if key in found},
                    [key for key in keys if key not in found],
                )
            )
        elif isinstance(task, EvalTaskPlan) and task.client == "lm_eval":
            new: Baseline = {}
            for run in task.runs:
                where = dirs if run.key == run.task else [d / run.key for d in dirs]
                found = check.calibrate_eval(
                    [d for d in where if d.is_dir()], {run.task}
                )
                if run.task in found:
                    new[run.key] = found[run.task]
            keys = [run.key for run in task.runs]
            proposals.append(
                Proposal(
                    "eval",
                    _current([run.gate for run in task.runs]),
                    new,
                    [key for key in keys if key not in new],
                )
            )
    return proposals


def _number(value: Any) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def render(proposals: Sequence[Proposal]) -> str:
    """Each entry's change, metric by metric."""
    lines = []
    for proposal in proposals:
        lines.append(f"{proposal.task}:")
        for key, metrics in proposal.new.items():
            old = proposal.current.get(key)
            lines.append(f"  {key}:" + ("" if old is not None else " (new)"))
            for metric, value in sorted(metrics.items()):
                before = (old or {}).get(metric)
                text = f"    {metric}: "
                if before is None:
                    text += _number(value)
                else:
                    text += f"{_number(before)} -> {_number(value)}"
                    if isinstance(before, int | float) and before:
                        text += f" ({(value - before) / before:+.1%})"
                lines.append(text)
        lines += [f"  {key}: no results; kept" for key in proposal.missing]
    return "\n".join(lines) + "\n" if lines else "nothing to calibrate\n"


# Writing ----------------------------------------------------------------------

PLAIN_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_.@-]*")
YAML_WORDS = {"true", "false", "null", "yes", "no", "on", "off", "y", "n", "~"}


def _key(text: str) -> str:
    if PLAIN_KEY.fullmatch(text) and text.lower() not in YAML_WORDS:
        return text
    return json.dumps(text)


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _end(lines: Sequence[str], start: int, column: int) -> int:
    """The line after the node whose key sits at `start`, at `column`.

    The node runs until a line that is not blank and starts at `column` or
    left of it; a comment there belongs to what follows. Blank lines at the
    end stay outside.
    """
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line.strip() and _indent(line) <= column:
            break
        end += 1
    while end - 1 > start and not lines[end - 1].strip():
        end -= 1
    return end


def _entry(key: str, metrics: Mapping[str, Any], column: int, step: int) -> list[str]:
    pad = " " * column
    out = [f"{pad}{_key(key)}:\n"]
    out += [
        f"{pad}{' ' * step}{_key(metric)}: {_value(value)}\n"
        for metric, value in sorted(metrics.items())
    ]
    return out


def _holder(doc: Any, point: str | None, task: str) -> tuple[Any, str]:
    """The mapping that holds the task's baseline in this file, and its key.

    Raises:
        CalibrateError: the file has no place for it.
    """

    def child(node: Any, key: Any) -> Any:
        if isinstance(node, CommentedMap):
            return node.get(key)
        return None

    if point is None:
        holder = child(child(doc, "tasks"), task)
        where, key = f"tasks.{task}", "baseline"
    else:
        sweep = child(doc, "sweep")
        points = child(sweep, "points")
        if isinstance(points, CommentedSeq):
            match = [p for p in points if child(p, "name") == point]
            holder = child(child(match[0], "tasks"), task) if match else None
            where, key = f"sweep point {point}: tasks.{task}", "baseline"
        else:
            holder = child(child(sweep, "baselines"), point)
            where, key = f"sweep.baselines.{point}", task
    if not isinstance(holder, CommentedMap):
        raise CalibrateError(
            f"this file has no {where} to hold the {task} baseline; add it first"
        )
    if holder.fa.flow_style():
        raise CalibrateError(f"{where} is a flow mapping; write it as a block")
    return holder, key


def _create(
    holder: CommentedMap, name: str, entries: Baseline, lines: list[str]
) -> tuple[int, int, list[str]]:
    """The edit that adds `name:` with `entries` to `holder`."""
    column = holder.lc.key(next(iter(holder)))[1]
    new = [f"{' ' * column}{name}:\n"]
    for key, metrics in entries.items():
        new += _entry(key, metrics, column + 2, 2)
    if name in holder:
        # Written with no value: the key's own line is replaced.
        line = holder.lc.key(name)[0]
        return line, line + 1, new
    line, key_column = holder.lc.key(list(holder)[-1])
    at = _end(lines, line, key_column)
    return at, at, new


def write(file: pathlib.Path, point: str | None, proposals: Sequence[Proposal]) -> int:
    """Write the proposals' entries into `file`; how many entries changed.

    An entry with new values replaces the old one's lines, keeping its
    metric indentation; a new key goes after the block's last entry; a task
    without a baseline gets one at the end of its mapping.

    Raises:
        CalibrateError: the file has no place for a baseline, or holds one
            that is not an inline block mapping.
    """
    text = file.read_text()
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    doc = YAML(typ="rt").load(text)
    # (start, end, replacement) line edits, applied bottom up.
    edits: list[tuple[int, int, list[str]]] = []
    for proposal in proposals:
        if not proposal.new:
            continue
        holder, name = _holder(doc, point, proposal.task)
        block = holder.get(name)
        if block is None:
            edits.append(_create(holder, name, proposal.new, lines))
            continue
        if not isinstance(block, CommentedMap) or "from" in block:
            raise CalibrateError(f"the {proposal.task} baseline here is not inline")
        if block.fa.flow_style() or not block:
            raise CalibrateError(f"the {proposal.task} baseline is a flow mapping")
        keys = list(block)
        column = block.lc.key(keys[0])[1]
        block_end = _end(lines, block.lc.key(keys[-1])[0], column)
        for key, metrics in proposal.new.items():
            if key not in block:
                continue
            start = block.lc.key(key)[0]
            end = _end(lines, start, column)
            inner = [line for line in lines[start + 1 : end] if line.strip()]
            step = _indent(inner[0]) - column if inner else 2
            edits.append((start, end, _entry(key, metrics, column, step)))
        added = [key for key in proposal.new if key not in block]
        if added:
            new = []
            for key in added:
                new += _entry(key, proposal.new[key], column, 2)
            edits.append((block_end, block_end, new))
    changed = 0
    for start, end, replacement in sorted(edits, reverse=True):
        if lines[start:end] != replacement:
            changed += 1
        lines[start:end] = replacement
    updated = "".join(lines)
    if updated != text:
        file.write_text(updated)
    return changed
