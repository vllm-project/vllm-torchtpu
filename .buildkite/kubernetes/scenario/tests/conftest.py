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
"""The CI fleet target, loaded as --target loads it, and a step's agent env."""

import os
import pathlib
from types import ModuleType

import pytest
from scenario.targets.loader import load_module

TPU_CI = pathlib.Path(__file__).resolve().parents[1] / "tpu_ci.py"
COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def tpu_ci() -> ModuleType:
    return load_module(TPU_CI)


@pytest.fixture
def fleet_env(monkeypatch: pytest.MonkeyPatch, tpu_ci: ModuleType) -> None:
    """A nightly step's env on main, with nothing of this machine's in it."""
    names = {
        *tpu_ci.STEP_ENV,
        "BQ_UPLOAD",
        "CI_RUNNER",
        "CREATED_BY",
        "WORKLOAD_IMAGE",
        "TPU_MAX_RUNTIME_SECONDS",
        "DOCKER_IMAGE_SUFFIX",
        "SCENARIO_TARGET",
    }
    names |= {name for name in os.environ if name.startswith("BUILDKITE_")}
    for name in names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BUILDKITE_COMMIT", COMMIT)
    monkeypatch.setenv("BUILDKITE_BRANCH", "main")
    monkeypatch.setenv("BUILDKITE_PULL_REQUEST", "false")
