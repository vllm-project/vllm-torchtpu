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
"""The command line entry points and project wiring."""

import pathlib
import subprocess
import sys
import tomllib

from click.testing import CliRunner
from scenario import PLAN_FORMAT, __version__
from scenario.cli import main

TOOL_DIR = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = TOOL_DIR.parents[1]


def test_version() -> None:
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert result.output == f"scenario {__version__} (plan format {PLAN_FORMAT})\n"


def test_python_m_scenario() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "scenario", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.startswith(f"scenario {__version__} ")


def test_console_script_declared() -> None:
    project = tomllib.loads((TOOL_DIR / "pyproject.toml").read_text())
    assert project["project"]["scripts"] == {"scenario": "scenario.cli:main"}


def test_build_backend_pinned_exactly() -> None:
    project = tomllib.loads((TOOL_DIR / "pyproject.toml").read_text())
    (requirement,) = project["build-system"]["requires"]
    assert "==" in requirement


def test_root_pytest_does_not_collect_tool_tests() -> None:
    root = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    testpaths = root["tool"]["pytest"]["ini_options"]["testpaths"]
    for path in testpaths:
        resolved = (REPO_ROOT / path).resolve()
        assert not TOOL_DIR.is_relative_to(resolved), path
