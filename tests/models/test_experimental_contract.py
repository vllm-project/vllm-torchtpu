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
"""Rules every `models/<model>/experimental/` directory follows.

These checks are model-agnostic: they discover every experimental directory
under `vllm_torchtpu/models/` and apply the same rules, so a new model gets
them without adding a test of its own: one-way imports, no global env reads,
namespaced custom ops, and profiles that resolve.

- Shared code never imports (or names) a model's experimental code; a model's
  own outer files (e.g. its package `__init__`) may only import it lazily,
  inside a function or through a `"module:attr"` string target, so plugin load
  never pulls experimental code in. Relative imports are resolved first.
- An experimental directory never imports another model's experimental code.
- Experimental code does not read process environment variables; the model
  plan selects behavior. A line that must read one (deployment config) carries
  `# experimental-env-read: <reason>`.
- Custom ops registered in experimental code carry the model name, so two
  models' ops cannot collide in `torch.ops`.
- If an experimental directory defines `profiles.py` and `variants.py`, every
  profile names existing variants and every `"module:attr"` target imports.
"""

import ast
import dataclasses
import importlib
import re
from pathlib import Path

import pytest

import vllm_torchtpu

SRC = Path(vllm_torchtpu.__file__).resolve().parent
MODELS = SRC / "models"
ENV_READ_MARKER = "# experimental-env-read:"
_EXPERIMENTAL_IMPORT = re.compile(r"vllm_torchtpu\.models\.(\w+)\.experimental\b")
_TARGET = re.compile(r"^(vllm_torchtpu(?:\.\w+)+):(\w+)$")
_OP_REGISTRARS = {"custom_op", "define"}
# Modules whose attribute reads are environment reads.
_ENV_MODULES = {"os.environ", "os.getenv", "vllm.envs", "vllm_torchtpu.envs"}


def _experimental_dirs() -> list[Path]:
    return sorted(
        d / "experimental"
        for d in MODELS.iterdir()
        if d.is_dir() and (d / "experimental").is_dir()
    )


def _py_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _model_of(path: Path) -> str | None:
    """The model directory a source file belongs to, if any."""
    rel = path.relative_to(SRC).parts
    if len(rel) >= 3 and rel[0] == "models" and rel[1] != "vllm":
        return rel[1]
    return None


def _in_experimental(path: Path) -> bool:
    rel = path.relative_to(SRC).parts
    return len(rel) >= 3 and rel[0] == "models" and rel[2] == "experimental"


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC.parent).with_suffix("")
    parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
    return ".".join(parts)


def _package_of(path: Path) -> str:
    """The package relative imports in `path` resolve against."""
    module = _module_name(path)
    return module if path.name == "__init__.py" else module.rpartition(".")[0]


def _absolute_module(node: ast.ImportFrom, package: str) -> str:
    """`node`'s module with a relative import resolved against `package`."""
    if not node.level:
        return node.module or ""
    base = package.split(".")
    base = base[: len(base) - (node.level - 1)]
    return ".".join([*base, node.module] if node.module else base)


def _experimental_refs(tree: ast.AST, package: str = "") -> list[tuple[int, str, bool]]:
    """(line, model, at module level) for each reference to experimental code.

    `package` is the importing module's package, for relative imports. A
    `"module:attr"` string names a lazy target: it counts as a reference, but
    never as a module-level import."""
    refs = []

    def visit(node: ast.AST, top: bool) -> None:
        for child in ast.iter_child_nodes(node):
            nested = isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
            )
            names: list[str] = []
            eager = top
            if isinstance(child, ast.Import):
                names = [a.name for a in child.names]
            elif isinstance(child, ast.ImportFrom):
                module = _absolute_module(child, package)
                names = [module] + [f"{module}.{a.name}" for a in child.names]
            elif isinstance(child, ast.Constant) and isinstance(child.value, str):
                names = [child.value]
                eager = False
            for name in names:
                if m := _EXPERIMENTAL_IMPORT.search(name):
                    refs.append((child.lineno, m.group(1), eager))
                    break
            visit(child, top and not nested)

    visit(tree, True)
    return refs


def _boundary_error(
    model: str | None, in_experimental: bool, ref_model: str, top: bool
) -> str | None:
    """Why a reference to `ref_model`'s experimental code is not allowed from
    a file of `model` (None: shared code), or None if it is."""
    if model is None or ref_model != model:
        return f"imports {ref_model}/experimental"
    if not in_experimental and top:
        return "imports experimental at module level"
    return None


def test_experimental_import_boundaries() -> None:
    bad = []
    for path in _py_files(SRC):
        model = _model_of(path)
        tree = ast.parse(path.read_text())
        for line, ref_model, top in _experimental_refs(tree, _package_of(path)):
            error = _boundary_error(model, _in_experimental(path), ref_model, top)
            if error:
                bad.append(f"{path.relative_to(SRC)}:{line} {error}")
    assert not bad, "\n".join(bad)


def _env_reads(tree: ast.AST) -> list[int]:
    """Lines that read the process environment: imports of `os.environ`,
    `os.getenv` or an `envs` module, and every use of a name bound to one
    (``os.environ[...]``, ``vllm_envs.X``, ``from vllm_torchtpu.envs import X``
    then ``X``), so a marker on the import line does not cover the reads."""
    aliases: dict[str, str] = {"os": "os"}  # local name -> module it binds
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name in _ENV_MODULES:
                    lines.add(node.lineno)
                if a.asname:
                    aliases[a.asname] = a.name
                else:
                    aliases[a.name.partition(".")[0]] = a.name.partition(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                full = f"{node.module}.{a.name}"
                if full in _ENV_MODULES or node.module in _ENV_MODULES:
                    lines.add(node.lineno)
                    aliases[a.asname or a.name] = full
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            dotted = _dotted(node, aliases)
            if dotted and any(
                dotted == m or dotted.startswith(f"{m}.") for m in _ENV_MODULES
            ):
                lines.add(node.lineno)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            bound = aliases.get(node.id, "")
            if any(bound == m or bound.startswith(f"{m}.") for m in _ENV_MODULES):
                lines.add(node.lineno)
    return sorted(lines)


def _dotted(node: ast.AST, aliases: dict[str, str]) -> str | None:
    """``a.b.c`` with its root name replaced by the module it is bound to."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or node.id not in aliases:
        return None
    return ".".join([aliases[node.id], *reversed(parts)])


@pytest.mark.parametrize("exp_dir", _experimental_dirs(), ids=lambda d: d.parent.name)
def test_experimental_reads_no_environment(exp_dir: Path) -> None:
    bad = []
    for path in _py_files(exp_dir):
        text = path.read_text()
        lines = text.splitlines()
        for line in _env_reads(ast.parse(text)):
            if ENV_READ_MARKER not in lines[line - 1]:
                bad.append(f"{path.relative_to(SRC)}:{line}: {lines[line - 1].strip()}")
    assert not bad, (
        "experimental code reads the environment; select behavior through the "
        f"model plan, or mark a deployment-config read with '{ENV_READ_MARKER} "
        "<reason>':\n" + "\n".join(bad)
    )


def _literal_prefix(node: ast.AST) -> str | None:
    """The literal leading text of a string expression: a constant, an
    f-string up to its first placeholder, or such a string's method call
    (e.g. ``f"pallas::op_{x}".replace(".", "p")``)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        prefix = ""
        for part in node.values:
            if not (isinstance(part, ast.Constant) and isinstance(part.value, str)):
                break
            prefix += part.value
        return prefix
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return _literal_prefix(node.func.value)
    return None


def _scopes(tree: ast.AST) -> list[tuple[list[ast.AST], dict[str, list[ast.AST]]]]:
    """Per function (and the module): its own nodes, without nested
    functions, and name -> values assigned to it there or in an enclosing
    scope (module-level constants included; the innermost values first)."""
    scopes = []

    def collect(scope: ast.AST, outer: dict[str, list[ast.AST]]) -> None:
        nodes: list[ast.AST] = []
        own: dict[str, list[ast.AST]] = {}
        functions = []
        stack = list(ast.iter_child_nodes(scope))
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node)
                continue
            nodes.append(node)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        own.setdefault(target.id, []).append(node.value)
            stack.extend(ast.iter_child_nodes(node))
        names = {k: own.get(k, []) + outer.get(k, []) for k in {*own, *outer}}
        scopes.append((nodes, names))
        for function in functions:
            collect(function, names)

    collect(tree, {})
    return scopes


def _op_name_prefix(
    node: ast.AST, names: dict[str, list[ast.AST]], seen: frozenset = frozenset()
) -> str | None:
    """`_literal_prefix`, following a local name (or a method call on one,
    e.g. ``name.replace(...)``) to the values assigned to it in scope."""
    while isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        node = node.func.value
    if isinstance(node, ast.Name):
        if node.id in seen:
            return None
        for value in names.get(node.id, []):
            prefix = _op_name_prefix(value, names, seen | {node.id})
            if prefix is not None:
                return prefix
        return None
    return _literal_prefix(node)


def _op_name_errors(tree: ast.AST, model: str) -> list[tuple[int, str]]:
    """(line, problem) for each custom op registered without the model name.

    Registrars are any callee ending in ``jax_op`` (``pallas.jax_op``,
    ``sharded_jax_op``, wrappers) plus ``custom_op`` / ``define``, and a
    ``Library(ns, ...)`` must use the model's namespace. A ``jax_op`` name
    must resolve to a literal prefix (through local, enclosing or module
    assignments); names of the other registrars are checked when they are
    qualified (``ns::op``). Calls of a name assigned in scope (``jax_op =
    pallas.jax_op(...)`` then ``jax_op(q)``) or of ``self.<x>`` attributes
    call an op object, not a registrar."""
    bad = []
    for nodes, names in _scopes(tree):
        for node in nodes:
            if not isinstance(node, ast.Call) or not node.args:
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id in names:
                continue
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "self"
            ):
                continue
            if getattr(func, "attr", getattr(func, "id", None)) == "Library":
                namespace = _op_name_prefix(node.args[0], names)
                if namespace is not None and namespace != model:
                    bad.append((node.lineno, f"Library({namespace!r})"))
                continue
            callee = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", None)
            ) or ""
            is_jax_op = callee.endswith("jax_op")
            if not is_jax_op and callee not in _OP_REGISTRARS:
                continue
            prefix = _op_name_prefix(node.args[0], names)
            if prefix is None:
                if is_jax_op:
                    bad.append((node.lineno, "op name is not a resolvable literal"))
                continue
            if "::" not in prefix:
                if is_jax_op:
                    bad.append((node.lineno, f"{prefix!r} has no namespace"))
                continue
            namespace, _, op = prefix.partition("::")
            if namespace != model and not op.startswith(f"{model}_"):
                bad.append((node.lineno, repr(prefix)))
    return sorted(set(bad))


@pytest.mark.parametrize("exp_dir", _experimental_dirs(), ids=lambda d: d.parent.name)
def test_experimental_custom_ops_are_namespaced(exp_dir: Path) -> None:
    model = exp_dir.parent.name
    bad = [
        f"{path.relative_to(SRC)}:{line}: {problem}"
        for path in _py_files(exp_dir)
        for line, problem in _op_name_errors(ast.parse(path.read_text()), model)
    ]
    assert not bad, (
        f"custom ops must be named '{model}::<op>' or '<ns>::{model}_<op>':\n"
        + "\n".join(bad)
    )


def _plan_dirs() -> list[Path]:
    return [
        d
        for d in _experimental_dirs()
        if (d / "profiles.py").is_file() and (d / "variants.py").is_file()
    ]


def _targets(value: object):
    if isinstance(value, str):
        if _TARGET.match(value):
            yield value
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            yield from _targets(getattr(value, f.name))
    elif isinstance(value, dict):
        for v in value.values():
            yield from _targets(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _targets(v)


def _profile_errors(profiles, variants) -> list[str]:
    """Problems of a plan: a default profile that is not defined, a profile
    naming an unknown slot or variant, or a ``"module:attr"`` target that
    does not import (on CPU, as this lane runs)."""
    bad = []
    if profiles.DEFAULT_PROFILE not in profiles.PROFILES:
        bad.append(f"default profile {profiles.DEFAULT_PROFILE!r} is not defined")
    for profile, slots in profiles.PROFILES.items():
        for slot, names in slots.items():
            if slot not in variants:
                bad.append(f"{profile}: unknown slot {slot!r}")
                continue
            for name in (names,) if isinstance(names, str) else names:
                if name not in variants[slot]:
                    bad.append(f"{profile}: {slot}={name!r}")
    for slot, by_name in variants.items():
        for name, variant in by_name.items():
            for target in _targets(variant):
                module, _, attr = target.partition(":")
                try:
                    resolved = hasattr(importlib.import_module(module), attr)
                except ImportError:
                    resolved = False
                if not resolved:
                    bad.append(f"{slot}={name}: {target} does not resolve")
    return bad


@pytest.mark.parametrize("exp_dir", _plan_dirs(), ids=lambda d: d.parent.name)
def test_experimental_profiles_resolve(exp_dir: Path) -> None:
    package = f"vllm_torchtpu.models.{exp_dir.parent.name}.experimental"
    profiles = importlib.import_module(f"{package}.profiles")
    variants = importlib.import_module(f"{package}.variants").VARIANTS
    bad = _profile_errors(profiles, variants)
    assert not bad, "\n".join(bad)


# Must-fail controls: the checks above run only over existing experimental
# directories, so the helpers are exercised here on snippets.

_PKG = "vllm_torchtpu.models.m"


@pytest.mark.parametrize(
    ("src", "package", "want"),
    [
        ("from .experimental import k\n", _PKG, [(1, "m", True)]),
        ("from . import experimental\n", _PKG, [(1, "m", True)]),
        ("from ..experimental.ops import k\n", f"{_PKG}.sub", [(1, "m", True)]),
        (
            "import vllm_torchtpu.models.other.experimental.ops\n",
            _PKG,
            [(1, "other", True)],
        ),
        ("def f():\n    from .experimental import k\n", _PKG, [(2, "m", False)]),
        ('T = "vllm_torchtpu.models.m.experimental.ops:fn"\n', _PKG, [(1, "m", False)]),
        ("from .layers import k\n", _PKG, []),
    ],
)
def test_experimental_refs_controls(src, package, want) -> None:
    assert _experimental_refs(ast.parse(src), package) == want


@pytest.mark.parametrize(
    ("src", "want"),
    [
        ('import os\nx = os.environ["X"]\n', [2]),
        ('import os\nx = os.getenv("X")\n', [2]),
        ('from os import environ\nx = environ["X"]\n', [1, 2]),
        ("from vllm import envs as vllm_envs\nx = vllm_envs.VLLM_X\n", [1, 2]),
        (
            "from vllm_torchtpu import envs as tpu_envs\na = 1\nx = tpu_envs.X\n",
            [1, 3],
        ),
        ("from vllm_torchtpu.envs import X\ny = X\n", [1, 2]),
        ("import vllm\nx = vllm.envs.VLLM_X\n", [2]),
        ("import os\np = os.path.join('a', 'b')\n", []),
        ("from vllm_torchtpu import utils\nx = utils.cdiv(3, 2)\n", []),
    ],
)
def test_env_reads_controls(src, want) -> None:
    assert _env_reads(ast.parse(src)) == want


@pytest.mark.parametrize(
    ("src", "bad_lines"),
    [
        ('pallas.jax_op("pallas::foo", f)\n', [1]),
        ('pallas.jax_op("pallas::m_foo", f)\n', []),
        ('pallas.jax_op("m::foo", f)\n', []),
        (
            'def g(x):\n    op_name = f"pallas::gdn_{x}"\n'
            "    pallas.jax_op(op_name, f)\n",
            [3],
        ),
        (
            'def g(x):\n    name = f"pallas::m_draft_{x}".replace(".", "p")\n'
            "    pcp_streaming_jax_op(name, f)\n",
            [],
        ),
        (
            'def g(x):\n    name = f"pallas::draft_{x}"\n'
            '    name = name.replace(".", "p")\n'
            "    sharded_jax_op(name, f)\n",
            [4],
        ),
        ("def g(name):\n    pallas.jax_op(name, f)\n", [2]),
        ('torch.library.custom_op("other::foo")\n', [1]),
        ('lib.define("foo(Tensor x) -> Tensor")\n', []),
        (
            'def g():\n    jax_op = pallas.jax_op("pallas::m_a", f)\n'
            "    def op(q):\n        return jax_op(q)\n",
            [],
        ),
    ],
)
def test_op_name_controls(src, bad_lines) -> None:
    assert [line for line, _ in _op_name_errors(ast.parse(src), "m")] == bad_lines


@pytest.mark.parametrize(
    ("model", "in_experimental", "ref_model", "top", "bad"),
    [
        (None, False, "m", False, True),  # shared code, even lazily
        ("m", False, "other", False, True),  # outer file, another model, lazy
        ("m", True, "other", False, True),  # experimental, another model
        ("m", False, "m", True, True),  # outer file, own, module level
        ("m", False, "m", False, False),  # outer file, own, lazy
        ("m", True, "m", True, False),  # experimental, own
    ],
)
def test_boundary_error_controls(model, in_experimental, ref_model, top, bad) -> None:
    assert (_boundary_error(model, in_experimental, ref_model, top) is not None) == bad


@pytest.mark.parametrize(
    ("src", "bad_lines"),
    [
        ('NAME = "pallas::foo"\ndef g():\n    pallas.jax_op(NAME, f)\n', [3]),
        ('NAME = "pallas::m_foo"\ndef g():\n    pallas.jax_op(NAME, f)\n', []),
        ("def g(self, q):\n    return self._attend_jax_op(q)\n", []),
        ('lib = Library("other", "DEF")\n', [1]),
        ('lib = Library("m", "DEF")\n', []),
        (
            "from torch_tpu._internal.pallas import jax_op\n"
            "def g():\n    jax_op = 1\n"
            'def h():\n    jax_op("pallas::foo", f)\n',
            [5],
        ),
    ],
)
def test_op_name_scope_controls(src, bad_lines) -> None:
    assert [line for line, _ in _op_name_errors(ast.parse(src), "m")] == bad_lines


def test_profile_errors_controls() -> None:
    from types import SimpleNamespace

    variants = {
        "slot": {
            "good": {"impl": "vllm_torchtpu.envs:environment_variables"},
            "missing": {"impl": "vllm_torchtpu.envs:no_such_attr"},
        }
    }
    ok = SimpleNamespace(DEFAULT_PROFILE="p", PROFILES={"p": {"slot": "good"}})
    assert _profile_errors(ok, {"slot": {"good": variants["slot"]["good"]}}) == []
    bad = SimpleNamespace(
        DEFAULT_PROFILE="q", PROFILES={"p": {"slot": "nope", "other": "x"}}
    )
    errors = _profile_errors(bad, variants)
    assert len(errors) == 4, errors  # default, unknown variant, slot, target
