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
"""Resolution reads only its files and arguments (R-RES-7, R-NF-3)."""

import ast
import os
import pathlib

import pytest
from scenario.resolve import resolve

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "src" / "scenario"
PURE_MODULES = [
    "accel.py",
    "errors.py",
    "keycheck.py",
    "load.py",
    "merge.py",
    "overrides.py",
    "paths.py",
    "reserved.py",
    "resolve.py",
    "model/*.py",
    "plan/*.py",
]
# Environment, clock, randomness, network, subprocesses, and vLLM and its
# stack: none of them may decide what a scenario resolves to.
BANNED_MODULES = {
    "os",
    "time",
    "datetime",
    "random",
    "socket",
    "subprocess",
    "threading",
    "multiprocessing",
    "urllib",
    "http",
    "requests",
    "vllm",
    "torch",
    "ray",
    "transformers",
}
BANNED_ATTRIBUTES = {
    "environ",
    "getenv",
    "putenv",
    "getcwd",
    "cwd",
    "home",
    "expanduser",
}


def violations(source: str) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            names = []
        for name in names:
            if name.split(".")[0] in BANNED_MODULES:
                found.append(f"line {node.lineno}: imports {name}")
        if isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRIBUTES:
            found.append(f"line {node.lineno}: uses .{node.attr}")
    return found


def pure_files() -> list[pathlib.Path]:
    return sorted(path for pattern in PURE_MODULES for path in PACKAGE.glob(pattern))


@pytest.mark.parametrize(
    "path", pure_files(), ids=lambda path: str(path.relative_to(PACKAGE))
)
def test_pure_modules(path: pathlib.Path) -> None:
    assert violations(path.read_text()) == []


@pytest.mark.parametrize(
    "source",
    [
        "import time",
        "import os.path",
        "from os import environ",
        "from datetime import datetime",
        "import subprocess",
        "import vllm.entrypoints",
        "from torch import nn",
        "root = pathlib.Path.cwd()",
        "token = mapping.getenv('HF_TOKEN')",
    ],
)
def test_seeded_violations_are_caught(source: str) -> None:
    assert violations(source)


def test_clean_source_passes() -> None:
    assert (
        violations("import pathlib\nimport re\nfrom scenario.paths import KeyPath")
        == []
    )


def test_resolution_ignores_env_and_cwd(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenarios = tmp_path / "repo" / "scenarios"
    scenarios.mkdir(parents=True)
    (scenarios / "defaults.yaml").write_text(
        "schema: 1\nserver: {image: ci, startup_timeout: 2h}\n"
    )
    scenario = scenarios / "s.yaml"
    scenario.write_text(
        "schema: 1\nname: s\nserver: {model: {name: m}, slice: tpu7x/2x2x1}\n"
    )
    first = resolve(scenario, ["server.args.max-model-len=4096"])
    for key in list(os.environ):
        monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    second = resolve(scenario, ["server.args.max-model-len=4096"])
    assert first == second
    assert first[0].provenance == second[0].provenance
