"""Which functions a diff changed, per file.

PROTOTYPE. Adapted from vllm-project/ci-infra (Apache-2.0),
buildkite/ci_selector/ci_selector/coverage/changed_funcs.py at 04dd673a92:
`hunks`, `attribute` and the code-object helpers are kept as upstream wrote
them. Also kept: `_drop_unchanged` and the helpers it uses (changes that
leave behaviour alone). Dropped: platform guards and vLLM's scopes.

Names come from CPython (compile the source and read the qualnames and line
tables off the code objects), which is how the recorder spells them.
"""

from __future__ import annotations

import ast
import io
import re
import subprocess
import tokenize
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
    # Every change in the file leaves behaviour alone (comments, docstrings,
    # annotations, reformatting), so no step needs it.
    inert: bool = False

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
        base_side = None if status == "A" else (old_path or path)
        head_side = None if status == "D" else path
        sides = ((base, base_side, base_lines), (head, head_side, head_lines))
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
        query.failed = "; ".join(notes)
        if not query.failed and base_side and head_side:
            before = read_at(repo, base, base_side)
            after = read_at(repo, head, head_side)
            if before is not None and after is not None:
                kept, kept_import, query.inert = drop_unchanged(
                    repo, head, before, after, base_side, head_side,
                    frozenset(names), frozenset(at_import),
                )
                names, at_import = set(kept), set(kept_import)
        query.names, query.import_time = _normalize(
            frozenset(names), frozenset(at_import)
        )
    return queries


# ---------------------------------------------------------------------------
# Changes that leave behaviour alone. Kept as upstream wrote them
# (`_equivalent`, `_strip_annotations`, `_strip_docstrings`, `_logical_lines`,
# `_docstring_read`, `_preserving`, `_Shell`, `_shell_equal`,
# `_drop_unchanged`), with the readers of annotations adapted to this repo.
# ---------------------------------------------------------------------------


def _equivalent(a: types.CodeType, b: types.CodeType) -> bool:
    """Same code, line numbers aside: bytecode, names, locals, flags and
    constants, nested code compared the same way."""
    if (
        a.co_code != b.co_code
        or a.co_names != b.co_names
        or a.co_varnames != b.co_varnames
        or a.co_freevars != b.co_freevars
        or a.co_cellvars != b.co_cellvars
        or a.co_flags != b.co_flags
        or a.co_argcount != b.co_argcount
        or a.co_posonlyargcount != b.co_posonlyargcount
        or a.co_kwonlyargcount != b.co_kwonlyargcount
        or a.co_qualname != b.co_qualname
        or len(a.co_consts) != len(b.co_consts)
    ):
        return False
    for x, y in zip(a.co_consts, b.co_consts):
        if isinstance(x, types.CodeType) or isinstance(y, types.CodeType):
            if not (
                isinstance(x, types.CodeType)
                and isinstance(y, types.CodeType)
                and _equivalent(x, y)
            ):
                return False
        elif type(x) is not type(y) or x != y:
            return False
    return True


# A module naming one of these may read function annotations at runtime:
# torch custom op registration infers the op schema from them, and
# support_torch_compile the dynamic dims from forward's.
_ANNOTATION_READERS = (
    "direct_register_custom_op",
    "library.custom_op",
    "infer_schema",
    "get_type_hints",
    "signature(",
    "support_torch_compile",
    "torch.library",
)


def _strip_annotations(tree: ast.AST) -> ast.AST:
    """Remove parameter and return annotations of undecorated functions that
    import defines. Decorated ones keep theirs, as do class-level fields."""
    todo = [(tree, False)]
    while todo:
        node, in_function = todo.pop()
        is_def = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        if is_def and not in_function and not node.decorator_list:
            node.returns = None
            a = node.args
            for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg):
                if arg is not None:
                    arg.annotation = None
        todo.extend((c, in_function or is_def) for c in ast.iter_child_nodes(node))
    return tree


def _strip_docstrings(tree: ast.AST) -> list[tuple[str, str | None]]:
    """Remove the leading docstring of every undecorated function, and return
    (name, docstring) for each in walk order."""
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.decorator_list
        ):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                node.body = node.body[1:]
            found.append((node.name, doc))
    return found


def _logical_lines(text: str) -> list[str]:
    """Each statement line of the source, comments left out."""
    lines, cur = [], []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in (tokenize.NEWLINE, tokenize.ENDMARKER):
                lines.append(" ".join(cur))
                cur = []
            elif tok.type not in (tokenize.COMMENT, tokenize.NL):
                cur.append(tok.string)
    except (tokenize.TokenError, SyntaxError):
        return [text]
    return lines


def _docstring_read(repo: Path, ref: str | None, names: set[str]) -> bool:
    """Whether a statement that reads `__doc__` or `getdoc` names one of these.
    A search that fails counts as a read."""
    proc = subprocess.run(
        ["git", "-C", str(repo), "grep", "-l", "-z", "-I", "-E", "__doc__|getdoc"]
        + ([ref] if ref else [])
        + ["--", "*.py"],
        capture_output=True,
        text=True,
        errors="replace",
    )
    if proc.returncode not in (0, 1):
        return True
    word = re.compile(r"\b(?:" + "|".join(map(re.escape, sorted(names))) + r")\b")
    for hit in filter(None, proc.stdout.split("\0")):
        path = hit[len(ref) + 1 :] if ref else hit
        text = read_at(repo, ref, path)
        if text is None:
            return True
        for line in _logical_lines(text):
            if ("__doc__" in line or "getdoc" in line) and word.search(line):
                return True
    return False


def _preserving(repo: Path, ref: str | None, before: str, after: str) -> bool:
    """Whether the two sources differ only in removable annotations and
    docstrings, with no changed docstring read at runtime."""
    annotations = not any(m in before or m in after for m in _ANNOTATION_READERS)
    try:
        trees = [ast.parse(before), ast.parse(after)]
    except SyntaxError:
        return False
    if annotations:
        trees = [_strip_annotations(t) for t in trees]
    old_docs, new_docs = (_strip_docstrings(t) for t in trees)
    if ast.dump(trees[0]) != ast.dump(trees[1]):
        return False
    edited = {n for (n, a), (_, b) in zip(old_docs, new_docs) if a != b}
    return not edited or not _docstring_read(repo, ref, edited)


class _Shell(ast.NodeTransformer):
    """A module with what import does not run taken out: function bodies,
    parameter lists and annotations, and defaults that are plain literals.
    Decorators stay, and so does any default that is an expression."""

    def _fn(self, node):
        args = node.args
        kept = [
            d
            for d in [*args.defaults, *(k for k in args.kw_defaults if k is not None)]
            if not isinstance(d, ast.Constant)
        ]
        for d in kept:
            self.visit(d)
        node.decorator_list = [self.visit(d) for d in node.decorator_list]
        node.body = [ast.Pass()]
        node.returns = None
        node.args = ast.arguments(
            posonlyargs=[], args=[], vararg=None, kwonlyargs=[],
            kw_defaults=[], kwarg=None, defaults=kept,
        )
        if hasattr(node, "type_params"):
            node.type_params = []
        return node

    visit_FunctionDef = visit_AsyncFunctionDef = _fn


def _shell_equal(before: str, after: str) -> bool:
    """Whether import runs the same code on both sides."""
    try:
        a = _Shell().visit(ast.parse(before))
        b = _Shell().visit(ast.parse(after))
    except SyntaxError:
        return False
    return ast.dump(a) == ast.dump(b)


def drop_unchanged(
    repo: Path, head: str | None, before: str, after: str,
    base_side: str, head_side: str,
    names: frozenset[str], import_time: frozenset[str],
) -> tuple[frozenset[str], frozenset[str], bool]:
    """(names whose code changed, their import-time subset, nothing left).

    A changed line can belong to a function whose code is identical (a split
    signature, a comment). A module body differs whenever a def in it does, so
    if only function bodies, signatures or literal defaults changed, import
    runs the same code and the module body is not a change of its own.
    """
    try:
        old: dict[str, list] = {}
        for c in code_objects(compile(before, base_side, "exec")):
            old.setdefault(c.co_qualname, []).append(c)
        new: dict[str, list] = {}
        for c in code_objects(compile(after, head_side, "exec")):
            new.setdefault(c.co_qualname, []).append(c)
    except Exception:
        return names, import_time, False

    def same(name: str) -> bool:
        a, b = old.get(name), new.get(name)
        return (
            a is not None
            and b is not None
            and len(a) == len(b)
            and all(_equivalent(x, y) for x, y in zip(a, b))
        )

    unchanged = {n for n in names if same(n)}
    left = names - unchanged
    if left and _preserving(repo, head, before, after):
        unchanged |= left
        left = frozenset()
    if left & import_time and left - import_time and _shell_equal(before, after):
        unchanged |= left & import_time
        left = left - import_time
    if left and left <= import_time:
        # Only import-time names would be left: keep the file as it was.
        return names, import_time, False
    return frozenset(names - unchanged), frozenset(import_time - unchanged), not left
