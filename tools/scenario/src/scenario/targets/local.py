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
"""The local target: this host's part of a Plan, run in this process.

It hands the runtime the same values a pod gets from its env, as flags. It
reserves no env and has no after-tasks, so a local run uploads nothing
(R-TASK-12).
"""

import dataclasses
import pathlib
import subprocess

from scenario.plan.types import Plan
from scenario.runtime import driver
from scenario.target import EnvNames, TargetData, TargetInfo


def git_commit(directory: pathlib.Path) -> str | None:
    """The checked-out commit of `directory`, or None outside a git checkout."""
    try:
        done = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    commit = done.stdout.strip()
    return commit if done.returncode == 0 and commit else None


class LocalTarget:
    def __init__(self, options: driver.RunOptions) -> None:
        self.options = options

    def info(self) -> TargetInfo:
        return TargetInfo("local", git_commit(self.options.repo_root), None)

    def reserved_env(self) -> EnvNames:
        return EnvNames(frozenset())

    def submit(self, plan: Plan) -> int:
        options = dataclasses.replace(self.options, target=TargetData(self.info()))
        return driver.run(plan, options)
