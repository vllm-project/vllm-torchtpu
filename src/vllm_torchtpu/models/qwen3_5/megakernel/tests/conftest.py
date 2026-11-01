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
"""Pytest configuration for this folder's CPU tests."""

from __future__ import annotations

import pytest

from vllm_torchtpu.megakernel_runtime.testkit import mk_fake_runner as mk_fake_runner
from vllm_torchtpu.megakernel_runtime.testkit import route_lazy_loggers


@pytest.fixture(autouse=True)
def _std_loggers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the runtime's and this folder's lazy vLLM loggers to `logging`
    when vLLM is absent (CPU)."""
    route_lazy_loggers(monkeypatch, __package__.rpartition(".")[0])
