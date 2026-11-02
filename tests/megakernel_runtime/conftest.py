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

import logging
import sys

import pytest

from tests.megakernel.conftest import register_megakernel_cpu_plugin

_RUNTIME = "vllm_torchtpu.megakernel_runtime."


def pytest_configure(config: pytest.Config) -> None:
    register_megakernel_cpu_plugin(config)


@pytest.fixture(autouse=True)
def _std_megakernel_loggers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the lazy vLLM loggers of the imported runtime modules to
    `logging` when vLLM is absent (CPU)."""
    for name in sorted(n for n in sys.modules if n.startswith(_RUNTIME)):
        lazy = getattr(sys.modules[name], "logger", None)
        if not callable(getattr(lazy, "_resolve", None)):
            continue
        try:
            lazy._resolve()
        except ModuleNotFoundError:
            monkeypatch.setattr(lazy, "_logger", logging.getLogger(lazy._name))
