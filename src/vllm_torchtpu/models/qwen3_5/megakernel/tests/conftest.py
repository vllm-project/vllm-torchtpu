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

import importlib
import logging
import sys

import pytest


@pytest.fixture(autouse=True)
def _std_loggers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the runtime's and this folder's lazy vLLM loggers to `logging`
    when vLLM is absent (CPU)."""
    # The runtime modules a test will use log through lazy loggers too.
    importlib.import_module("vllm_torchtpu.megakernel_runtime.driver")
    folder = __package__.rpartition(".")[0]
    prefixes = ("vllm_torchtpu.megakernel_runtime.", f"{folder}.")
    for name in sorted(sys.modules):
        if not name.startswith(prefixes):
            continue
        lazy = getattr(sys.modules[name], "logger", None)
        if not callable(getattr(lazy, "_resolve", None)):
            continue
        try:
            lazy._resolve()
        except ModuleNotFoundError:
            monkeypatch.setattr(lazy, "_logger", logging.getLogger(lazy._name))
