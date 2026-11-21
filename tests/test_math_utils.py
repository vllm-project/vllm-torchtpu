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
"""Unit tests for pure-Python math helpers."""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest

from vllm_torchtpu import math_utils, utils
from vllm_torchtpu.math_utils import largest_divisor


@pytest.mark.parametrize(
    ("x", "cap", "expected"),
    [
        (80, 48, 40),
        (80, 80, 80),
        (80, 128, 80),
        (12, 5, 4),
        (13, 12, 1),
        (1, 1, 1),
        (16, 0, 1),
        (16, -4, 1),
    ],
)
def test_largest_divisor_values(x: int, cap: int, expected: int) -> None:
    assert largest_divisor(x, cap) == expected


def test_utils_reexports_largest_divisor() -> None:
    assert utils.largest_divisor is largest_divisor


def test_math_utils_is_pure_python_and_avoids_torch_and_vllm() -> None:
    assert math_utils.__file__ is not None
    tree = ast.parse(pathlib.Path(math_utils.__file__).read_text(encoding="utf-8"))
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            imported_modules.append(node.module or "")
    assert not imported_modules, f"Unexpected imports in math_utils: {imported_modules}"

    code = (
        "import sys; "
        "from vllm_torchtpu.math_utils import largest_divisor; "
        "assert largest_divisor(80, 48) == 40; "
        "forbidden = {'torch', 'torch_tpu', 'vllm'}; "
        "loaded = forbidden.intersection(sys.modules); "
        "assert not loaded, f'Unexpected heavy imports: {sorted(loaded)}'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
