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
"""Host-only tests for the Raiden worker-registration timeout default."""

import ast
import logging
import os
import pathlib
import textwrap

SRC = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src/vllm_torchtpu/offload/raiden_store.py"
)
VAR = "RAIDEN_EXPECTED_WORKERS_TIMEOUT_S"


def _fn():
    src = SRC.read_text()
    for node in ast.parse(src).body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_default_worker_registration_timeout"
        ):
            g = {"os": os, "logger": logging.getLogger("t")}
            exec(textwrap.dedent(ast.get_source_segment(src, node)), g)  # noqa: S102
            return g[node.name]
    raise AssertionError("helper missing")


def test_defaults_to_engine_ready_timeout(monkeypatch):
    monkeypatch.delenv(VAR, raising=False)
    monkeypatch.setenv("VLLM_ENGINE_READY_TIMEOUT_S", "7200")
    assert _fn()() == 7200 and os.environ[VAR] == "7200"


def test_floor_3600_without_engine_timeout(monkeypatch):
    monkeypatch.delenv(VAR, raising=False)
    monkeypatch.delenv("VLLM_ENGINE_READY_TIMEOUT_S", raising=False)
    assert _fn()() == 3600 and os.environ[VAR] == "3600"
    monkeypatch.delenv(VAR, raising=False)
    monkeypatch.setenv("VLLM_ENGINE_READY_TIMEOUT_S", "600")
    assert _fn()() == 3600


def test_explicit_value_kept(monkeypatch):
    monkeypatch.setenv(VAR, "1800")
    monkeypatch.setenv("VLLM_ENGINE_READY_TIMEOUT_S", "7200")
    assert _fn()() == 1800 and os.environ[VAR] == "1800"


def test_called_before_store_construction():
    src = SRC.read_text()
    i = src.index("_default_worker_registration_timeout()\n")
    assert i < src.index("store_types.KVCacheStore(")
