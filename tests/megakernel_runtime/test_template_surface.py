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
"""Public-surface guard over every model folder
(`src/vllm_torchtpu/models/*/megakernel/**`).

A model folder imports its own files relatively (so a copy under another
package name works), uses only the runtime's public surface (not the driver,
config, fingerprint or salt modules, nor the vLLM runner), imports no private
library names, and registers nothing at import time.
"""

from __future__ import annotations

import ast
from pathlib import Path

import vllm_torchtpu.megakernel_runtime as mkr

_MODELS = Path(mkr.__file__).resolve().parents[1] / "models"
_FORBIDDEN_MODULES = (
    "vllm_torchtpu.megakernel_runtime.driver",
    "vllm_torchtpu.megakernel_runtime.config",
    "vllm_torchtpu.megakernel_runtime.fingerprint",
    "vllm_torchtpu.megakernel_runtime.salt",
    "vllm_torchtpu.runner",
)


def _surface_violations(path: Path, package: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level == 0:
                if mod == package or mod.startswith(package + "."):
                    bad.append(f"absolute sibling import {mod}")
                if any(mod == f or mod.startswith(f + ".") for f in _FORBIDDEN_MODULES):
                    bad.append(f"runtime internal import {mod}")
            for alias in node.names:
                if alias.name.startswith("_") and alias.name != "__future__":
                    bad.append(f"private name {alias.name} from {mod}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == package or alias.name.startswith(package + "."):
                    bad.append(f"absolute sibling import {alias.name}")
                if any(
                    alias.name == f or alias.name.startswith(f + ".")
                    for f in _FORBIDDEN_MODULES
                ):
                    bad.append(f"runtime internal import {alias.name}")
    for node in tree.body:
        call = node.value if isinstance(node, (ast.Expr, ast.Assign)) else None
        if isinstance(call, ast.Call):
            name = ast.unparse(call.func).rsplit(".", 1)[-1]
            if name.startswith("register"):
                bad.append(f"module-level {name}() call")
        for deco in getattr(node, "decorator_list", ()):
            target = deco.func if isinstance(deco, ast.Call) else deco
            if ast.unparse(target).rsplit(".", 1)[-1].startswith("register"):
                bad.append(f"module-level @{ast.unparse(target)}")
    return bad


def _model_folders() -> list[Path]:
    return sorted(p for p in _MODELS.glob("*/megakernel") if p.is_dir())


def test_model_folders_use_only_the_public_surface() -> None:
    folders = _model_folders()
    assert folders, _MODELS
    checked = 0
    for folder in folders:
        package = f"vllm_torchtpu.models.{folder.parent.name}.megakernel"
        for path in sorted(folder.rglob("*.py")):
            rel = path.relative_to(_MODELS)
            assert _surface_violations(path, package) == [], rel
            checked += 1
    assert checked >= 8


def test_public_surface_guard_catches_violations(tmp_path: Path) -> None:
    src = """
from vllm_torchtpu.megakernel_runtime.driver import ProgramDriver
from vllm_torchtpu.models.mine.megakernel.model import Mine
import vllm_torchtpu.models.mine.megakernel.pool
from vllm_torchtpu.megakernel import _private_helper
import vllm_torchtpu.runner.tpu_runner
from vllm_torchtpu.megakernel import register_model
register_model(Mine)
PROFILE = register_profile(Profile("x"))

@register_model(name="mine")
class Mine2:
    pass
"""
    path = tmp_path / "bad.py"
    path.write_text(src, encoding="utf-8")
    bad = _surface_violations(path, "vllm_torchtpu.models.mine.megakernel")
    assert len(bad) == 8, bad
