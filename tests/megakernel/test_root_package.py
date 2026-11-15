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
"""The megakernel root package (`src/vllm_torchtpu/megakernel/__init__.py`) only
groups `core` and `adapter`: it imports nothing, so importing the root loads
neither subpackage, and importing one never loads the other through it."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import vllm_torchtpu

_PKG_DIR = Path(vllm_torchtpu.__file__).resolve().parent
_ROOT_INIT = _PKG_DIR / "megakernel" / "__init__.py"


def test_root_init_is_docstring_only() -> None:
    tree = ast.parse(_ROOT_INIT.read_text(encoding="utf-8"))
    assert len(tree.body) == 1, ast.dump(tree)
    assert isinstance(tree.body[0], ast.Expr), ast.dump(tree)
    assert isinstance(tree.body[0].value, ast.Constant), ast.dump(tree)


def test_importing_the_root_loads_nothing_beyond_stdlib() -> None:
    code = (
        "import importlib, sys, vllm_torchtpu; before = set(sys.modules); "
        "importlib.import_module('.megakernel', package='vllm_torchtpu'); "
        "print(chr(10).join(sorted(set(sys.modules) - before)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": str(_PKG_DIR.parent), "PATH": "/usr/bin:/bin"},
    ).stdout.split()
    root = f"{vllm_torchtpu.__name__}.megakernel"
    assert root in out, out
    assert [m for m in out if m.split(".")[0] == vllm_torchtpu.__name__] == [root], out
    roots = {m.split(".")[0] for m in out} - {vllm_torchtpu.__name__}
    assert roots <= set(sys.stdlib_module_names), roots - set(sys.stdlib_module_names)
