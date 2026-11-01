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
"""Pytest configuration for `tests/megakernel_runtime`."""

from __future__ import annotations

import pytest

from tests.megakernel.conftest import register_megakernel_cpu_plugin, route_lazy_loggers
from vllm_torchtpu.megakernel_runtime.testkit import mk_fake_runner as mk_fake_runner
from vllm_torchtpu.megakernel_runtime.testkit import mk_toy_plugin as mk_toy_plugin


def pytest_configure(config: pytest.Config) -> None:
    register_megakernel_cpu_plugin(config)


@pytest.fixture(autouse=True)
def _std_megakernel_loggers(monkeypatch: pytest.MonkeyPatch) -> None:
    route_lazy_loggers(monkeypatch)
