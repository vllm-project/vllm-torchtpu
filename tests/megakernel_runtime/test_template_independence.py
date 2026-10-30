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
"""The runtime is model-independent: no runtime module and no runtime test
imports a model folder (`megakernel_runtime.models.*`). A model folder is
named by string only (a shorthand or a `module:attr` plugin reference); the
toy plugin in `test_program_driver.py` covers load, create, decline, serving
and the fingerprint without one."""

from __future__ import annotations

import ast
from pathlib import Path

import vllm_torchtpu.megakernel_runtime as mkr

_RT = Path(mkr.__file__).resolve().parent
_SRC = _RT.parents[1]
_TESTS = Path(__file__).resolve().parent
_MODELS = "vllm_torchtpu.megakernel_runtime.models"
# The Qwen3.5 template's own tests; they move out with the template.
_TEMPLATE_TESTS = frozenset(
    {
        "test_qwen35_template.py",
        "test_vllm_qwen35_e2e.py",
    }
)


def _imported(path: Path, package: str) -> list[tuple[int, str]]:
    """`(line, dotted name)` of every module (or `from` name) `path` imports;
    relative imports are resolved against `package`."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found += [(node.lineno, a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join(parts + ([base] if base else []))
            found.append((node.lineno, base))
            found += [(node.lineno, f"{base}.{a.name}") for a in node.names]
    return found


def _violations(paths: list[Path], package_of: dict[Path, str]) -> list[str]:
    template_tests = {f"tests.megakernel_runtime.{n[:-3]}" for n in _TEMPLATE_TESTS}
    bad = []
    for path in paths:
        for line, name in _imported(path, package_of[path]):
            if name == _MODELS or name.startswith(_MODELS + "."):
                bad.append(f"{path.name}:{line}: {name}")
            elif name in template_tests:
                bad.append(f"{path.name}:{line}: {name} (a template test)")
    return bad


def test_runtime_modules_import_no_model_folder() -> None:
    paths = [
        p for p in sorted(_RT.rglob("*.py")) if "models" not in p.relative_to(_RT).parts
    ]
    assert len(paths) > 5
    package_of = {
        p: ".".join(p.relative_to(_SRC).with_suffix("").parts[:-1]) for p in paths
    }
    assert _violations(paths, package_of) == []


def test_runtime_tests_import_no_model_folder() -> None:
    paths = [p for p in sorted(_TESTS.glob("*.py")) if p.name not in _TEMPLATE_TESTS]
    assert Path(__file__).resolve() in paths
    package_of = dict.fromkeys(paths, "tests.megakernel_runtime")
    assert _violations(paths, package_of) == []


def test_the_guard_sees_model_folder_imports(tmp_path: Path) -> None:
    src = tmp_path / "x.py"
    src.write_text(
        "import vllm_torchtpu.megakernel_runtime.models.qwen35\n"
        "from vllm_torchtpu.megakernel_runtime import models\n"
        "from .models.qwen35 import adapter\n"
        "from ..models import qwen35\n"
        "from tests.megakernel_runtime.test_vllm_qwen35_e2e import _cpu_mesh\n"
        'SHORTHAND = "vllm_torchtpu.megakernel_runtime.models.qwen35"\n',
        encoding="utf-8",
    )
    pkg = {src: "vllm_torchtpu.megakernel_runtime"}
    lines = {int(v.split(":")[1]) for v in _violations([src], pkg)}
    assert lines == {1, 2, 3, 5}
    pkg = {src: "vllm_torchtpu.megakernel_runtime.sub"}
    assert 4 in {int(v.split(":")[1]) for v in _violations([src], pkg)}
