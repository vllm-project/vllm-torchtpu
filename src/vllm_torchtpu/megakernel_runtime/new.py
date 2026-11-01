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
"""Scaffold a new megakernel model folder by copying a template::

    python -m vllm_torchtpu.megakernel_runtime.new --from <template> \\
        --dir src/vllm_torchtpu/models/<model>/megakernel --name <model> \\
        [--model-name <model def name>]

`--from` is a shorthand (`megakernel_runtime.plugin.SHORTHANDS`), a package
name or a folder. The folder imports its own files relatively, so the copy
runs as is under the package its directory gives it. Only the names the
library registries key on are rewritten, so the copy and the template can be
registered side by side: `SPEC.name` -> `--name`, profile names
`<old spec name>-*` -> `<name>-*`, and the `ModelDef` name -> `--model-name`
(default `--name`). Strings are rewritten in code only (not in comments,
docstrings or other files). Every copied `.py` file gets a provenance header
(`megakernel_runtime.provenance`). Stdlib only.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib.metadata
import importlib.util
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from . import provenance
from .plugin import SHORTHANDS

_NAME_RE = re.compile(r"^[a-z][a-z0-9_.]*$")
_DIST_PACKAGE = "vllm_torchtpu"


class ScaffoldError(Exception):
    """The template or the arguments can't be scaffolded."""


@dataclasses.dataclass(frozen=True)
class ScaffoldResult:
    source: Path
    dest: Path
    package: str
    commit: str
    renames: dict[str, str]
    files: tuple[Path, ...]


def resolve_template(ref: str) -> Path:
    """The folder of a shorthand, a package name or a directory."""
    path = Path(ref)
    if path.is_dir():
        return path.resolve()
    module = SHORTHANDS[ref].module if ref in SHORTHANDS else ref
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError) as exc:
        raise ScaffoldError(f"can't find template {ref!r}: {exc}") from exc
    if spec is None or not spec.submodule_search_locations:
        raise ScaffoldError(f"template {ref!r} is not a package or a directory")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def source_commit(source: Path) -> str:
    """The checkout's `HEAD`, else the installed package version, else
    `unknown`."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    for dist in importlib.metadata.packages_distributions().get(_DIST_PACKAGE, []):
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def package_name(dest: Path) -> str:
    """The dotted package `dest` imports as (up its `__init__.py` parents)."""
    parts = [dest.name]
    parent = dest.parent
    while (parent / "__init__.py").is_file():
        parts.append(parent.name)
        parent = parent.parent
    return ".".join(reversed(parts))


# --- finding the names ----------------------------------------------------------


def _str_const(node: ast.AST | None) -> ast.Constant | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node
    return None


def _spec_name_node(tree: ast.Module) -> ast.Constant | None:
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "SPEC" for t in node.targets):
            continue
        call = node.value
        for kw in call.keywords:
            if kw.arg == "name":
                return _str_const(kw.value)
        return _str_const(call.args[0]) if call.args else None
    return None


def _module_str_assigns(tree: ast.Module) -> Iterator[tuple[str, ast.Constant]]:
    for node in tree.body:
        if isinstance(node, ast.Assign) and (const := _str_const(node.value)):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    yield t.id, const


def _model_name_exprs(tree: ast.Module) -> Iterator[ast.expr]:
    """`name = ...` in a class that also sets `hf_architectures` (a ModelDef)."""
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        attrs: dict[str, ast.expr] = {}
        for node in cls.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        attrs[t.id] = node.value
        if "name" in attrs and "hf_architectures" in attrs:
            yield attrs["name"]


def _docstrings(tree: ast.Module) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and _str_const(body[0].value)
        ):
            ids.add(id(body[0].value))
    return ids


def _plain_str_consts(tree: ast.Module) -> Iterator[ast.Constant]:
    """String constants in code: not docstrings, not f-string pieces."""
    skip = _docstrings(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            skip.update(id(v) for v in node.values)
    for node in ast.walk(tree):
        if (const := _str_const(node)) is not None and id(const) not in skip:
            yield const


# --- rewriting ------------------------------------------------------------------


def _replace_consts(text: str, edits: dict[int, tuple[ast.Constant, str]]) -> str:
    """Replace each constant's source span with the new string, same quotes."""
    lines = text.encode("utf-8").splitlines(keepends=True)
    spans = sorted(
        (
            (c.lineno, c.col_offset, c.end_lineno, c.end_col_offset, new)
            for c, new in edits.values()
        ),
        reverse=True,
    )
    for lno, col, end_lno, end_col, new in spans:
        if lno != end_lno:
            raise ScaffoldError(f"line {lno}: a multi-line string can't be renamed")
        line = lines[lno - 1]
        old = line[col:end_col].decode("utf-8")
        quote = old.lstrip("rRuU")[:1]
        lines[lno - 1] = line[:col] + f"{quote}{new}{quote}".encode() + line[end_col:]
    return b"".join(lines).decode("utf-8")


def _plan_renames(
    trees: dict[Path, ast.Module], init: Path, name: str, model_name: str
) -> tuple[dict[Path, dict[int, tuple[ast.Constant, str]]], dict[str, str]]:
    edits: dict[Path, dict[int, tuple[ast.Constant, str]]] = {p: {} for p in trees}
    spec_node = _spec_name_node(trees[init])
    if spec_node is None:
        raise ScaffoldError(f"{init}: no `SPEC = TpuMonoSpec(name=...)` string")
    old_spec = spec_node.value
    renames = {old_spec: name}
    edits[init][id(spec_node)] = (spec_node, name)

    profile_re = re.compile(rf"^{re.escape(old_spec)}-[A-Za-z0-9_.-]+$")
    for path, tree in trees.items():
        for const in _plain_str_consts(tree):
            if profile_re.match(const.value):
                new = name + const.value[len(old_spec) :]
                edits[path][id(const)] = (const, new)
                renames[const.value] = new

    assigns = {
        ident: (path, const)
        for path, tree in trees.items()
        for ident, const in _module_str_assigns(tree)
    }
    for path, tree in trees.items():
        for expr in _model_name_exprs(tree):
            if (const := _str_const(expr)) is not None:
                site = (path, const)
            elif isinstance(expr, ast.Name) and expr.id in assigns:
                site = assigns[expr.id]
            else:
                raise ScaffoldError(f"{path}: can't resolve the ModelDef name")
            renames[site[1].value] = model_name
            edits[site[0]][id(site[1])] = (site[1], model_name)
    old_models = {k for k, v in renames.items() if v == model_name}
    for path, tree in trees.items():
        for kw in (n for n in ast.walk(tree) if isinstance(n, ast.keyword)):
            const = _str_const(kw.value)
            if kw.arg == "model" and const is not None and const.value in old_models:
                edits[path][id(const)] = (const, model_name)
    return edits, renames


def scaffold(
    source: Path,
    dest: Path,
    *,
    name: str,
    model_name: str | None = None,
    commit: str | None = None,
) -> ScaffoldResult:
    """Copy the template folder `source` to `dest` (see the module doc)."""
    model_name = model_name or name
    for what, val in (("--name", name), ("--model-name", model_name)):
        if not _NAME_RE.match(val):
            raise ScaffoldError(f"{what} {val!r}: use lowercase [a-z0-9_.]")
    if not dest.name.isidentifier():
        raise ScaffoldError(f"{dest.name!r} is not a valid package name")
    if dest.exists() and any(dest.iterdir()):
        raise ScaffoldError(f"{dest} exists and is not empty")
    init = source / "__init__.py"
    if not init.is_file():
        raise ScaffoldError(f"{source} is not a package (no __init__.py)")
    root = provenance.git_toplevel(source)
    commit = commit or source_commit(source)

    sources = sorted(
        p
        for p in source.rglob("*.py")
        if "__pycache__" not in p.relative_to(source).parts
    )
    texts = {p: p.read_text(encoding="utf-8") for p in sources}
    trees = {p: ast.parse(t, filename=str(p)) for p, t in texts.items()}
    edits, renames = _plan_renames(trees, init, name, model_name)

    shutil.copytree(
        source,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        dirs_exist_ok=True,
    )
    for path in sources:
        text = _replace_consts(texts[path], edits[path])
        rel = provenance.repo_relative(path, root)
        text = provenance.with_header(text, rel, commit)
        (dest / path.relative_to(source)).write_text(text, encoding="utf-8")
    return ScaffoldResult(
        source=source,
        dest=dest,
        package=package_name(dest.resolve()),
        commit=commit,
        renames=renames,
        files=tuple(dest / p.relative_to(source) for p in sources),
    )


def _next_steps(res: ScaffoldResult) -> str:
    renamed = ", ".join(f"{a} -> {b}" for a, b in sorted(res.renames.items()))
    plugin = f'{{"megakernel": {{"plugin": "{res.package}"}}}}'
    lines = [
        f"Created {res.dest} (package {res.package}) from {res.source} @ {res.commit}.",
        f"Renamed: {renamed}.",
        "",
        "Next steps:",
        "  1. Adapt the copy (its README.md, 'Adapting this folder'): model.py "
        "(geometry, hf_architectures, hf_model_type), profiles/, adapter.py, "
        "SPEC.",
        f"  2. Run its CPU tests: JAX_PLATFORMS=cpu python -m pytest {res.dest}/tests",
        f"  3. Serve it: --additional-config '{plugin}'",
    ]
    parent = res.dest.resolve().parent
    if (
        not (parent / "__init__.py").is_file()
        and (parent.parent / "__init__.py").is_file()
    ):
        lines.insert(3, f"Add {parent / '__init__.py'} to make the package importable.")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vllm_torchtpu.megakernel_runtime.new",
        description="Scaffold a megakernel model folder from a template.",
    )
    parser.add_argument(
        "--from",
        dest="template",
        required=True,
        help="a shorthand, a package name or a folder",
    )
    parser.add_argument("--dir", type=Path, required=True, help="the new folder")
    parser.add_argument("--name", required=True, help="the new SPEC.name")
    parser.add_argument("--model-name", help="the new ModelDef name (default --name)")
    args = parser.parse_args(argv)
    try:
        res = scaffold(
            resolve_template(args.template),
            args.dir,
            name=args.name,
            model_name=args.model_name,
        )
    except ScaffoldError as exc:
        parser.error(str(exc))
    sys.stdout.write(_next_steps(res))
    return 0


__all__ = [
    "ScaffoldError",
    "ScaffoldResult",
    "main",
    "package_name",
    "resolve_template",
    "scaffold",
    "source_commit",
]


if __name__ == "__main__":
    raise SystemExit(main())
