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

# Source provenance: inferact/k3-megakernel@30c9ba966b
# tools/megakernel/coverage.py
"""Which kernel lines a run executes (``--coverage FILE``).

The kernel builders are Python that runs while JAX traces them, so the lines
a run executes are the branches its ops were built from. Rank 0 records every
executed line of the megakernel kernel packages (``sys.settrace``; frames
elsewhere are skipped at call time) and merges them into ``FILE`` (JSON:
file -> sorted lines), so runs over several commands accumulate.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from types import FrameType

    from _typeshed import TraceFunction

_PKG_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()),
    Path.cwd(),
)
ROOTS = tuple(str(_PKG_ROOT / sub) for sub in ("kernels", "common"))
_SNIPPET_LIMIT = 70


def _write_line(message: str) -> None:
    """Write ``message`` with a trailing newline to standard output."""
    sys.stdout.write(f"{message}\n")


class LineCoverage:
    """Line-level execution tracer for megakernel builder modules."""

    def __init__(self, path: str) -> None:
        """Initialize line coverage recorder writing merged output to ``path``."""
        self.path = path
        self.lines: dict[str, set[int]] = {}

    def _local(
        self,
        frame: FrameType,
        event: str,
        _arg: object,
    ) -> object:
        """Record executed line numbers within traced megakernel files."""
        if event == "line":
            self.lines.setdefault(frame.f_code.co_filename, set()).add(frame.f_lineno)
        return self._local

    def _global(
        self,
        frame: FrameType,
        _event: str,
        _arg: object,
    ) -> object:
        """Enter local line tracing only for files under ``ROOTS``."""
        if frame.f_code.co_filename.startswith(ROOTS):
            self.lines.setdefault(frame.f_code.co_filename, set()).add(frame.f_lineno)
            return self._local
        return None

    def start(self) -> None:
        """Enable thread and process line tracing hooks."""
        trace_fn = cast("TraceFunction", self._global)
        sys.settrace(trace_fn)
        threading.settrace(trace_fn)

    def stop(self) -> None:
        """Disable tracing hooks and merge recorded line numbers into ``self.path``."""
        sys.settrace(None)
        threading.settrace(None)
        target = Path(self.path)
        merged: dict[str, list[int]] = {}
        if target.exists():
            merged = json.loads(target.read_text(encoding="utf-8"))
        for name, lines in self.lines.items():
            key = os.path.relpath(name, _REPO_ROOT)
            merged[key] = sorted(set(merged.get(key, [])) | lines)
        target.write_text(json.dumps(merged), encoding="utf-8")


def _statements(node: ast.AST) -> set[int]:
    """Return all AST statement line numbers inside ``node``."""
    return {n.lineno for n in ast.walk(node) if isinstance(n, ast.stmt)}


def _unentered_functions(
    tree: ast.AST,
    hit: set[int],
) -> list[tuple[str, int, int]]:
    """Return ``(name, lineno, end_lineno)`` for functions with zero hit lines."""
    unentered: list[tuple[str, int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            body = _statements(node) - {node.lineno}
            end_lineno = node.end_lineno or node.lineno
            if (
                body
                and not (body & hit)
                and not any(start < node.lineno <= end for _, start, end in unentered)
            ):
                unentered.append((node.name, node.lineno, end_lineno))
    return unentered


def _untaken_branches(
    tree: ast.AST,
    source: str,
    hit: set[int],
) -> list[tuple[int, str, str]]:
    """Return ``(lineno, part, test_snippet)`` for untaken ``if`` branches."""
    untaken: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and node.lineno in hit:
            for part, block in (("body", node.body), ("else", node.orelse)):
                lines = set().union(*(_statements(b) for b in block))
                is_elif = (
                    part == "else" and len(block) == 1 and isinstance(block[0], ast.If)
                )
                if lines and not (lines & hit) and not is_elif:
                    segment = ast.get_source_segment(source, node.test) or ""
                    untaken.append((node.lineno, part, segment[:_SNIPPET_LIMIT]))
    return untaken


def _collect_rows(
    executed: dict[str, list[int]],
) -> list[
    tuple[
        str, set[int], set[int], list[tuple[str, int, int]], list[tuple[int, str, str]]
    ]
]:
    """Collect per-file coverage statistics across ``ROOTS``."""
    rows: list[
        tuple[
            str,
            set[int],
            set[int],
            list[tuple[str, int, int]],
            list[tuple[int, str, str]],
        ]
    ] = []
    for root in ROOTS:
        for file in sorted(Path(root).glob("*.py")):
            rel = os.path.relpath(file, _REPO_ROOT)
            source = file.read_text(encoding="utf-8")
            tree = ast.parse(source)
            hit = set(executed.get(rel, []))
            statements = {
                n.lineno
                for n in ast.walk(tree)
                if isinstance(n, ast.stmt)
                and not isinstance(n, (ast.FunctionDef, ast.ClassDef))
            }
            unentered = _unentered_functions(tree, hit)
            untaken = _untaken_branches(tree, source, hit)
            rows.append((rel, statements, hit, unentered, untaken))
    return rows


def report(path: str, detail: list[str]) -> None:
    """Format and write kernel line coverage summary for ``path``."""
    executed = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = _collect_rows(executed)
    _write_line(
        f"{'file':56} {'statements run':>15} {'unentered fns':>14} "
        f"{'their lines':>11} {'untaken ifs':>11}"
    )
    for rel, statements, hit, unentered, untaken in sorted(
        rows,
        key=lambda r: -sum(e - s + 1 for _, s, e in r[3]),
    ):
        lines = sum(end - start + 1 for _, start, end in unentered)
        _write_line(
            f"{rel.split('megakernel/')[-1]:56} "
            f"{len(statements & hit):7}/{len(statements):<7} "
            f"{len(unentered):14} {lines:11} {len(untaken):11}"
        )
    for rel, _, _, unentered, untaken in rows:
        if any(name in rel for name in detail):
            _write_line(f"\n== {rel}")
            for name, start, end in unentered:
                _write_line(f"  never entered: {name} (lines {start}-{end})")
            for line, part, test in untaken:
                _write_line(f"  never taken: line {line} {part} of `{test}`")


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(
        description="report a --coverage file's kernel line coverage"
    )
    cli_parser.add_argument("file")
    cli_parser.add_argument("--detail", nargs="*", default=[])
    parsed = cli_parser.parse_args()
    report(parsed.file, parsed.detail)
