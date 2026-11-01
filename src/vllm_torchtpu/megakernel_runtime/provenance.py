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
"""Provenance headers of copied model-folder files, and a source's copies.

A Python file copied from another (by `megakernel_runtime.new` or by hand)
carries, right after its license header::

    # Copied from src/vllm_torchtpu/models/<model>/megakernel/<file> @ <commit>

so a change to the source can list its copies. Files moved (not copied) carry
none. `list_copies(source)` scans a checkout for the copies of one file;
`changed_sources_with_copies(base)` maps each file changed since `base` to
its copies. As a command (stdlib only)::

    python -m vllm_torchtpu.megakernel_runtime.provenance [--base REV] [FILE ...]

prints each changed (or named) source that has copies, and the copies.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import NamedTuple

HEADER_RE = re.compile(
    r"^# Copied from (?P<source>\S+) @ (?P<commit>\S+)(?:\s+# noqa: E501)?\s*$"
)
_LINE_LIMIT = 88


class Provenance(NamedTuple):
    """Where a copied file came from: a repo-relative path and a commit."""

    source: str
    commit: str


def header_line(source: str, commit: str) -> str:
    """The provenance header for a copy of `source` (repo-relative) at
    `commit`; an over-long line carries `# noqa: E501`."""
    line = f"# Copied from {source} @ {commit}"
    return line if len(line) <= _LINE_LIMIT else f"{line}  # noqa: E501"


def _leading_comment_end(lines: Sequence[str]) -> int:
    n = 0
    while n < len(lines) and lines[n].startswith("#"):
        n += 1
    return n


def read_provenance(text: str) -> Provenance | None:
    """The provenance header in the leading comment block of `text`."""
    lines = text.splitlines()
    for line in lines[: _leading_comment_end(lines)]:
        if m := HEADER_RE.match(line):
            return Provenance(m["source"], m["commit"])
    return None


def with_header(text: str, source: str, commit: str) -> str:
    """`text` with its provenance header set to `source @ commit`: replaces
    an existing header, else goes right after the leading comment block (the
    license header)."""
    lines = text.splitlines(keepends=True)
    end = _leading_comment_end(lines)
    new = header_line(source, commit) + "\n"
    for i in range(end):
        if HEADER_RE.match(lines[i].rstrip("\n")):
            lines[i] = new
            return "".join(lines)
    if end and not lines[end - 1].endswith("\n"):
        lines[end - 1] += "\n"
    lines.insert(end, new)
    return "".join(lines)


def git_toplevel(path: Path) -> Path | None:
    """The git checkout holding `path`, or `None`."""
    cwd = path if path.is_dir() else path.parent
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return Path(out.stdout.strip()) if out.returncode == 0 else None


def repo_relative(path: Path, root: Path | None = None) -> str:
    """`path` relative to its checkout (`root`, or the git top level). Outside
    a checkout: `src/` + its path from the top-level package's parent."""
    path = path.resolve()
    root = root or git_toplevel(path)
    if root is not None:
        try:
            return path.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    top = path.parent
    while (top.parent / "__init__.py").is_file():
        top = top.parent
    return "src/" + path.relative_to(top.parent).as_posix()


def list_copies(
    source: Path | str,
    roots: Iterable[Path | str] | None = None,
    *,
    repo_root: Path | None = None,
) -> list[tuple[Path, str]]:
    """`(copy, commit)` for every `.py` file under `roots` (default: the
    checkout's `src/` and `tests/`) whose header names `source`."""
    source = Path(source)
    if source.is_absolute():
        repo_root = repo_root or git_toplevel(source)
        rel = repo_relative(source, repo_root)
    else:
        repo_root = repo_root or git_toplevel(Path.cwd())
        rel = source.as_posix()
    if roots is None:
        if repo_root is None:
            raise ValueError(f"{source} is not in a git checkout; pass roots")
        roots = [repo_root / "src", repo_root / "tests"]
    found: list[tuple[Path, str]] = []
    for root in roots:
        for path in sorted(Path(root).rglob("*.py")):
            try:
                head = path.read_text(encoding="utf-8")[:4096]
            except (OSError, UnicodeDecodeError):
                continue
            prov = read_provenance(head)
            if prov is not None and prov.source == rel:
                found.append((path, prov.commit))
    return found


def changed_files(base: str, repo_root: Path) -> list[str]:
    """Repo-relative files changed since `base` (committed or not)."""
    out = subprocess.run(
        ["git", "diff", "--name-only", base, "--"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return [line for line in out.stdout.splitlines() if line.endswith(".py")]


def changed_sources_with_copies(
    base: str, repo_root: Path, roots: Iterable[Path] | None = None
) -> dict[str, list[tuple[Path, str]]]:
    """Each `.py` file changed since `base` that has copies -> its copies."""
    roots = list(roots) if roots is not None else None
    out: dict[str, list[tuple[Path, str]]] = {}
    for rel in changed_files(base, repo_root):
        copies = list_copies(rel, roots, repo_root=repo_root)
        if copies:
            out[rel] = copies
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vllm_torchtpu.megakernel_runtime.provenance",
        description="List the copies of changed (or named) source files.",
    )
    parser.add_argument("files", nargs="*", help="repo-relative source files")
    parser.add_argument("--base", help="list the files changed since this rev")
    parser.add_argument("--root", type=Path, help="the checkout (default: cwd's)")
    args = parser.parse_args(argv)
    root = args.root or git_toplevel(Path.cwd())
    if root is None:
        parser.error("not in a git checkout; pass --root")
    sources = list(args.files)
    if args.base:
        sources += changed_files(args.base, root)
    for rel in dict.fromkeys(sources):
        copies = list_copies(rel, repo_root=root)
        if copies:
            out = sys.stdout
            out.write(f"{rel} has copies; check whether they need the change too:\n")
            for path, commit in copies:
                out.write(
                    f"  {path.relative_to(root).as_posix()} (copied @ {commit})\n"
                )
    return 0


__all__ = [
    "HEADER_RE",
    "Provenance",
    "changed_files",
    "changed_sources_with_copies",
    "git_toplevel",
    "header_line",
    "list_copies",
    "read_provenance",
    "repo_relative",
    "with_header",
]


if __name__ == "__main__":
    raise SystemExit(main())
