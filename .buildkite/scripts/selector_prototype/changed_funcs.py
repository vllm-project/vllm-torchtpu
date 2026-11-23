"""Which functions a diff changed, per file.

PROTOTYPE. Adapted from vllm-project/ci-infra (Apache-2.0),
buildkite/ci_selector/ci_selector/coverage/changed_funcs.py at 04dd673a92:
`hunks`, `attribute` and the code-object helpers are kept as upstream wrote
them. Dropped: platform guards, inert-change detection, and vLLM's scopes.

Names come from CPython (compile the source and read the qualnames and line
tables off the code objects), which is how the recorder spells them.
"""

from __future__ import annotations

import re
import subprocess
import types
from dataclasses import dataclass
from inspect import CO_OPTIMIZED
from pathlib import Path

MODULE = "<module>"
# What the recorder records by function: our package and the tests.
RECORDER_SCOPES = ("src/vllm_torchtpu/", "tests/")

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class FileQuery:
    """One changed file and the functions it changed, on either side."""

    path: str
    status: str  # git's letter: A, M, D, R, C, T
    old_path: str | None = None
    names: frozenset[str] = frozenset()
    # The subset of `names` that runs on import: module and class bodies.
    import_time: frozenset[str] = frozenset()
    # Why the names could not be read. Empty when they could.
    failed: str = ""

    @property
    def python(self) -> bool:
        return self.path.endswith(".py")

    @property
    def in_scope(self) -> bool:
        return self.path.startswith(RECORDER_SCOPES)

    @property
    def answerable(self) -> bool:
        """Whether a recording can say anything about this file."""
        return self.python and self.in_scope and not self.failed

    @property
    def function_names(self) -> frozenset[str]:
        """Changed names that need a real call, not merely an import."""
        return self.names - self.import_time


def code_objects(code: types.CodeType):
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from code_objects(const)


def names_of(source: str, path: str) -> frozenset[str]:
    """Every qualname CPython produces for this source."""
    return frozenset(c.co_qualname for c in code_objects(compile(source, path, "exec")))


def import_time_names(source: str, path: str) -> frozenset[str]:
    """Names whose code runs on import: the module body and every class body."""
    return frozenset(
        c.co_qualname
        for c in code_objects(compile(source, path, "exec"))
        if not c.co_flags & CO_OPTIMIZED
    )


def _spans(code: types.CodeType, total_lines: int) -> dict[str, tuple[int, int]]:
    """qualname -> the inclusive line range it covers, nested scopes included."""
    spans: dict[str, tuple[int, int]] = {}

    def visit(c: types.CodeType) -> tuple[int, int]:
        low = high = c.co_firstlineno
        for _s, _e, lineno in c.co_lines():
            if lineno:
                low = min(low, lineno)
                high = max(high, lineno)
        for const in c.co_consts:
            if isinstance(const, types.CodeType):
                clow, chigh = visit(const)
                low, high = min(low, clow), max(high, chigh)
        prev = spans.get(c.co_qualname)
        spans[c.co_qualname] = (
            (low, high) if prev is None else (min(prev[0], low), max(prev[1], high))
        )
        return low, high

    visit(code)
    spans[MODULE] = (1, max(total_lines, spans.get(MODULE, (1, 1))[1]))
    return spans


def _owners(
    source: str, path: str
) -> tuple[dict[int, set[str]], dict[str, tuple[int, int]]]:
    code = compile(source, path, "exec")
    owners: dict[int, set[str]] = {}
    for c in code_objects(code):
        for _s, _e, lineno in c.co_lines():
            if lineno:
                owners.setdefault(lineno, set()).add(c.co_qualname)
    return owners, _spans(code, source.count("\n") + 1)


def attribute(source: str, path: str, lines: set[int]) -> tuple[frozenset[str], bool]:
    """(names owning those lines, residue). Residue means a changed line sits
    outside every scope, which only happens when the diff and source disagree."""
    if not lines:
        return frozenset(), False
    owners, spans = _owners(source, path)
    found: set[str] = set()
    residue = False
    for line in lines:
        direct = owners.get(line)
        if direct:
            found |= direct
            continue
        containing = [
            (hi - lo, name) for name, (lo, hi) in spans.items() if lo <= line <= hi
        ]
        if not containing:
            residue = True
            continue
        best = min(width for width, _ in containing)
        found |= {name for width, name in containing if width == best}
    return frozenset(found), residue


def _normalize(names: frozenset[str], at_import: frozenset[str]):
    """Map Python 3.14's `__annotate__` functions back to import time.

    CI records on Python 3.12, where annotations are evaluated when the `def`
    or class body runs. 3.14 compiles them into separate `__annotate__`
    functions that 3.12 never runs, so a changed annotation read on 3.14 would
    otherwise look like a function no step calls. Treating it as a module-body
    change runs every step that imports the file: the safe direction.
    """
    annotate = {n for n in names if n.rsplit(".", 1)[-1] == "__annotate__"}
    if not annotate:
        return names, at_import
    return (names - annotate) | {MODULE}, (at_import - annotate) | {MODULE}


def hunks(
    repo: Path, base: str, head: str | None, ignore_cr_at_eol: bool = True
) -> dict[str, tuple[set[int], set[int]]]:
    """path -> (base-side changed lines, head-side changed lines), keyed under
    both sides of a rename."""
    args = [
        "git",
        "-c",
        "core.quotepath=false",
        "-C",
        str(repo),
        "diff",
        "-U0",
        "-M",
        "--no-prefix",
        *(["--ignore-cr-at-eol"] if ignore_cr_at_eol else []),
        base,
    ]
    if head:
        args.append(head)
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout

    found: dict[str, tuple[set[int], set[int]]] = {}
    old_path = new_path = None
    for line in out.splitlines():
        if line.startswith("--- "):
            old_path = None if line[4:] == "/dev/null" else line[4:]
        elif line.startswith("+++ "):
            new_path = None if line[4:] == "/dev/null" else line[4:]
        elif line.startswith("@@"):
            m = _HUNK.match(line)
            if not m:
                continue
            ostart, ocount, nstart, ncount = (
                int(m.group(1)),
                1 if m.group(2) is None else int(m.group(2)),
                int(m.group(3)),
                1 if m.group(4) is None else int(m.group(4)),
            )
            for key in (old_path, new_path):
                if key is None:
                    continue
                base_lines, head_lines = found.setdefault(key, (set(), set()))
                base_lines |= set(range(ostart, ostart + ocount))
                head_lines |= set(range(nstart, nstart + ncount))
    return found


def read_at(repo: Path, ref: str | None, path: str) -> str | None:
    """File content at a ref, or from the working tree when ref is None."""
    if ref is None:
        try:
            return (repo / path).read_text()
        except OSError:
            return None
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:{path}"], capture_output=True
    )
    if proc.returncode != 0:
        return None
    try:
        return proc.stdout.decode()
    except UnicodeDecodeError:
        return None


def diff_files(repo: Path, base: str, head: str | None) -> list[tuple[str, str, str | None]]:
    """(status letter, path, old path) for every changed file."""
    args = ["git", "-c", "core.quotepath=false", "-C", str(repo), "diff",
            "--name-status", "-M", base]
    if head:
        args.append(head)
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    files = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0][:1]
        if status in ("R", "C") and len(parts) == 3:
            files.append((status, parts[2], parts[1]))
        else:
            files.append((status, parts[1], None))
    return files


def build(repo: Path, base: str, head: str | None = None) -> list[FileQuery]:
    """Per changed file, the functions the diff touched."""
    by_path = hunks(repo, base, head)
    queries: list[FileQuery] = []
    for status, path, old_path in diff_files(repo, base, head):
        query = FileQuery(path=path, status=status, old_path=old_path)
        queries.append(query)
        if not query.python:
            continue
        base_lines: set[int] = set()
        head_lines: set[int] = set()
        for key in (path, old_path):
            if key and key in by_path:
                base_lines |= by_path[key][0]
                head_lines |= by_path[key][1]
        if not base_lines and not head_lines:
            # A mode change, a binary blob, or a line-ending-only edit: we
            # cannot tell which, so we do not get to say "nothing changed".
            query.failed = "no hunks parsed"
            continue
        names: set[str] = set()
        at_import: set[str] = set()
        notes: list[str] = []
        sides = (
            (base, None if status == "A" else (old_path or path), base_lines),
            (head, None if status == "D" else path, head_lines),
        )
        for ref, side_path, lines in sides:
            if side_path is None or not lines:
                continue
            source = read_at(repo, ref, side_path)
            if source is None:
                notes.append(f"unreadable at {ref}")
                continue
            try:
                found, residue = attribute(source, side_path, lines)
                names |= found
                at_import |= found & import_time_names(source, side_path)
                if residue:
                    notes.append("a changed line is outside every scope")
            except Exception as exc:  # anything compile() cannot handle
                notes.append(f"{type(exc).__name__}: {exc}")
        query.names, query.import_time = _normalize(
            frozenset(names), frozenset(at_import)
        )
        query.failed = "; ".join(notes)
    return queries
