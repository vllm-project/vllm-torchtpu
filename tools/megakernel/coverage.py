# SPDX-License-Identifier: Apache-2.0
"""Which kernel lines a run executes (``--coverage FILE``).

The kernel builders are Python that runs while JAX traces them, so the lines
a run executes are the branches its ops were built from. Rank 0 records every
executed line of the megakernel kernel packages (``sys.settrace``; frames
elsewhere are skipped at call time) and merges them into ``FILE`` (JSON:
file -> sorted lines), so runs over several commands accumulate.

    python -m tools.megakernel.coverage FILE [--detail NAME ...]

reports, per kernel file, the statements executed, the functions no run
entered and the branches no run took (``--detail``: list them for the files
whose path contains NAME). Tests, tools and configurations the runs did not
cover may still call an unentered function. Line numbers refer to the sources
the runs traced: record the coverage again after editing a kernel file.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import threading
from pathlib import Path

ROOTS = tuple(
    str(Path(__file__).resolve().parents[2] / "src/vllm_torchtpu/megakernel" / sub)
    for sub in ("kimi_k3/kernels", "common")
)


class LineCoverage:
    def __init__(self, path: str) -> None:
        self.path = path
        self.lines: dict[str, set[int]] = {}

    def _local(self, frame, event, arg):
        if event == "line":
            self.lines.setdefault(frame.f_code.co_filename, set()).add(frame.f_lineno)
        return self._local

    def _global(self, frame, event, arg):
        if frame.f_code.co_filename.startswith(ROOTS):
            self.lines.setdefault(frame.f_code.co_filename, set()).add(frame.f_lineno)
            return self._local
        return None

    def start(self) -> None:
        sys.settrace(self._global)
        threading.settrace(self._global)

    def stop(self) -> None:
        sys.settrace(None)
        threading.settrace(None)
        merged: dict[str, list[int]] = {}
        if os.path.exists(self.path):
            with open(self.path) as handle:
                merged = json.load(handle)
        for name, lines in self.lines.items():
            key = os.path.relpath(name, Path(__file__).resolve().parents[2])
            merged[key] = sorted(set(merged.get(key, [])) | lines)
        with open(self.path, "w") as handle:
            json.dump(merged, handle)


def _statements(node) -> set[int]:
    return {n.lineno for n in ast.walk(node) if isinstance(n, ast.stmt)}


def report(path: str, detail: list[str]) -> None:
    repo = Path(__file__).resolve().parents[2]
    with open(path) as handle:
        executed = json.load(handle)
    rows = []
    for root in ROOTS:
        for file in sorted(Path(root).glob("*.py")):
            rel = os.path.relpath(file, repo)
            source = file.read_text()
            tree = ast.parse(source)
            hit = set(executed.get(rel, []))
            statements = {
                n.lineno
                for n in ast.walk(tree)
                if isinstance(n, ast.stmt)
                and not isinstance(n, (ast.FunctionDef, ast.ClassDef))
            }
            unentered = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    body = _statements(node) - {node.lineno}
                    if (
                        body
                        and not body & hit
                        and not any(
                            start < node.lineno <= end for _, start, end in unentered
                        )
                    ):
                        unentered.append((node.name, node.lineno, node.end_lineno))
            untaken = []
            for node in ast.walk(tree):
                if isinstance(node, ast.If) and node.lineno in hit:
                    for part, block in (("body", node.body), ("else", node.orelse)):
                        lines = set().union(*(_statements(b) for b in block))
                        if (
                            lines
                            and not lines & hit
                            and not (
                                part == "else"
                                and len(block) == 1
                                and isinstance(block[0], ast.If)
                            )
                        ):
                            untaken.append(
                                (
                                    node.lineno,
                                    part,
                                    ast.get_source_segment(source, node.test)[:70],
                                )
                            )
            rows.append((rel, statements, hit, unentered, untaken))
    print(
        f"{'file':56} {'statements run':>15} {'unentered fns':>14} "
        f"{'their lines':>11} {'untaken ifs':>11}"
    )
    for rel, statements, hit, unentered, untaken in sorted(
        rows, key=lambda r: -sum(e - s + 1 for _, s, e in r[3])
    ):
        lines = sum(end - start + 1 for _, start, end in unentered)
        print(
            f"{rel.split('megakernel/')[-1]:56} "
            f"{len(statements & hit):7}/{len(statements):<7} "
            f"{len(unentered):14} {lines:11} {len(untaken):11}"
        )
    for rel, _, _, unentered, untaken in rows:
        if any(name in rel for name in detail):
            print(f"\n== {rel}")
            for name, start, end in unentered:
                print(f"  never entered: {name} (lines {start}-{end})")
            for line, part, test in untaken:
                print(f"  never taken: line {line} {part} of `{test}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="report a --coverage file's kernel line coverage"
    )
    parser.add_argument("file")
    parser.add_argument("--detail", nargs="*", default=[])
    args = parser.parse_args()
    report(args.file, args.detail)
