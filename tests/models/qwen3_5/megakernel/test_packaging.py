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
"""The model folder's non-Python files ship in the wheel: setuptools includes
every git-tracked file under a package (`include-package-data`, with
setuptools_scm's file finder), so the README, the serving scripts and the
recipe go with the code."""

from __future__ import annotations

import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

import vllm_torchtpu.models.qwen3_5.megakernel as folder

_FOLDER = Path(folder.__file__).resolve().parent
_REPO = _FOLDER.parents[4]
_DATA = (
    "README.md",
    "serving/serve.sh",
    "serving/run_real_ckpt_smoke.py",
    "serving/mk-real-ckpt-smoke.yml",
    "serving/acceptance_benchmark.py",
    "serving/BENCH.md",
)


def test_folder_data_files_are_package_data() -> None:
    pyproject = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    st = pyproject["tool"]["setuptools"]
    assert st["include-package-data"] is True
    assert "setuptools_scm[toml]>=8.0" in pyproject["build-system"]["requires"]
    assert st["packages"]["find"]["include"] == ["vllm_torchtpu*"]
    for rel in _DATA:
        assert (_FOLDER / rel).is_file(), rel
    if shutil.which("git") is None or not (_REPO / ".git").exists():
        pytest.skip("not a git checkout")
    tracked = subprocess.run(
        ["git", "ls-files", "--", str(_FOLDER)],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    rels = {Path(t).relative_to(_FOLDER.relative_to(_REPO)).as_posix() for t in tracked}
    assert set(_DATA) <= rels, set(_DATA) - rels
