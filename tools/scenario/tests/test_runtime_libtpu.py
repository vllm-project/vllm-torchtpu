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
"""The libtpu blocker a chipless driver's tasks and router get (P-18)."""

import os
import pathlib
import subprocess
import sys

from plan_fixtures import Data, build
from scenario.runtime.placement import LIBTPU_BLOCK, Placement, Refs

# How vLLM's platform detection decides that a host has TPUs.
DETECT = """
try:
    import libtpu
    print("tpu")
except Exception as error:
    print(type(error).__name__, getattr(error, "name", None))
"""


def detect(pythonpath: str) -> str:
    env = {**os.environ, "PYTHONPATH": pythonpath}
    done = subprocess.run(
        [sys.executable, "-c", DETECT], env=env, capture_output=True, text=True
    )
    return done.stdout.strip()


def test_the_blocker_hides_an_installed_libtpu(tmp_path: pathlib.Path) -> None:
    site = tmp_path / "site"
    (site / "libtpu").mkdir(parents=True)
    (site / "libtpu" / "__init__.py").write_text("")
    assert detect(str(site)) == "tpu"
    # It fails as an absent module does, so detection finds no TPU.
    blocked = os.pathsep.join([str(LIBTPU_BLOCK), str(site)])
    assert detect(blocked) == "ModuleNotFoundError libtpu"


def test_the_ref_puts_the_blocker_first(
    variants: dict[str, Data], tmp_path: pathlib.Path
) -> None:
    plan = build(variants["pd"])
    client = Placement("client", 0)

    def filled(env: dict[str, str]) -> str:
        refs = Refs(
            plan, client, run_dir=tmp_path, server_log=tmp_path / "log", env=env
        )
        return refs.env(plan.driver.task_env)["PYTHONPATH"]

    assert filled({}) == str(LIBTPU_BLOCK)
    inherited = "/workspace/src:/workspace"
    assert filled({"PYTHONPATH": inherited}) == f"{LIBTPU_BLOCK}:{inherited}"


def test_the_blocker_is_a_directory_of_one_module() -> None:
    assert sorted(path.name for path in LIBTPU_BLOCK.glob("*.py")) == ["libtpu.py"]
    assert not (LIBTPU_BLOCK / "__init__.py").exists()
