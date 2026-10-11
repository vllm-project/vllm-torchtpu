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
"""ci/check_freeze_delta.py, which guards the tool's install into an image."""

import importlib.util
import pathlib
import subprocess
import sys

import pytest

TOOL = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = TOOL / "ci" / "check_freeze_delta.py"
DOCKERFILE = TOOL.parents[1] / "docker" / "Dockerfile"

spec = importlib.util.spec_from_file_location("check_freeze_delta", SCRIPT)
assert spec is not None and spec.loader is not None
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

BEFORE = """\
# a comment
click==8.2.1
pydantic==2.12.5
Pydantic_Core==2.41.5
vllm @ git+https://github.com/vllm-project/vllm.git@0123abc
torch-tpu==2.9.0.dev1
"""
AFTER = (
    BEFORE
    + """\
ruamel.yaml==0.18.16
ruamel-yaml-clib==0.2.14
vllm-torchtpu-scenario @ file:///workspace/vllm-torchtpu/tools/scenario
"""
)


def run(
    tmp_path: pathlib.Path, before: str, after: str
) -> subprocess.CompletedProcess[str]:
    (tmp_path / "before.txt").write_text(before)
    (tmp_path / "after.txt").write_text(after)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tmp_path / "before.txt"),
            str(tmp_path / "after.txt"),
        ],
        capture_output=True,
        text=True,
    )


def test_parse() -> None:
    assert check.parse(AFTER) == {
        "click": "8.2.1",
        "pydantic": "2.12.5",
        "pydantic-core": "2.41.5",
        "vllm": "git+https://github.com/vllm-project/vllm.git@0123abc",
        "torch-tpu": "2.9.0.dev1",
        "ruamel-yaml": "0.18.16",
        "ruamel-yaml-clib": "0.2.14",
        "vllm-torchtpu-scenario": "file:///workspace/vllm-torchtpu/tools/scenario",
    }
    with pytest.raises(ValueError, match="not a freeze line"):
        check.parse("pydantic>=2\n")


def test_only_the_tool_and_ruamel_added(tmp_path: pathlib.Path) -> None:
    done = run(tmp_path, BEFORE, AFTER)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith(
        "scenario tool installed; added ruamel-yaml==0.18.16, "
        "ruamel-yaml-clib==0.2.14, vllm-torchtpu-scenario @ file:///workspace/"
    )


@pytest.mark.parametrize(
    ("after", "problem"),
    [
        (
            AFTER.replace("pydantic==2.12.5", "pydantic==2.13.0"),
            "changed pydantic from 2.12.5 to 2.13.0",
        ),
        (AFTER.replace("click==8.2.1\n", ""), "removed click==8.2.1"),
        (
            AFTER + "annotated-types==0.7.0\n",
            "added annotated-types==0.7.0, which only vLLM's",
        ),
        (BEFORE + "ruamel.yaml==0.18.16\n", "vllm-torchtpu-scenario is not installed"),
        (AFTER.replace("0123abc", "4567def"), "changed vllm from git+https"),
    ],
)
def test_any_other_change_fails(
    tmp_path: pathlib.Path, after: str, problem: str
) -> None:
    done = run(tmp_path, BEFORE, after)
    assert done.returncode == 1
    assert f"error: installing the scenario tool {problem}" in done.stderr


def test_usage(tmp_path: pathlib.Path) -> None:
    done = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
    assert done.returncode == 2 and done.stderr.startswith("usage:")


def test_the_image_build_runs_it() -> None:
    text = DOCKERFILE.read_text()
    assert (
        "uv pip install --system -c /tmp/scenario-before.txt ./tools/scenario" in text
    )
    assert "python3 tools/scenario/ci/check_freeze_delta.py" in text
