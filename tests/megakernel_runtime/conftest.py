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

import importlib.machinery
import logging
import sys

import pytest
import torch

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


@pytest.fixture(autouse=True)
def _mock_uninitialized_distributed(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU test isolation when torch_tpu/vLLM are installed but uninitialized."""
    # 1. PathFinder fix for `monkeypatch.setitem(sys.modules, "torch_tpu", None)`
    orig_path_find_spec = importlib.machinery.PathFinder.find_spec

    def _safe_find_spec(
        fullname: str, path: object = None, target: object = None
    ) -> object:
        try:
            return orig_path_find_spec(fullname, path, target)
        except AttributeError:
            return None

    monkeypatch.setattr(importlib.machinery.PathFinder, "find_spec", _safe_find_spec)

    # 2. When torch.distributed is uninitialized, fallback _get_pallas_mesh_fn to None
    # unless a test explicitly sets a fake pallas mesh.
    dist = torch.distributed
    if not (dist.is_available() and dist.is_initialized()):
        import vllm_torchtpu.megakernel_runtime.driver as driver_mod

        monkeypatch.setattr(driver_mod, "_get_pallas_mesh_fn", lambda: None)

    # 3. When vLLM model parallel is uninitialized, _local_tp_rank returns 0.
    try:
        import vllm.distributed.parallel_state as ps

        orig_tp_rank = ps.get_tensor_model_parallel_rank

        def _safe_tp_rank() -> int:
            if getattr(ps, "_TP", None) is None:
                return 0
            return orig_tp_rank()

        monkeypatch.setattr(ps, "get_tensor_model_parallel_rank", _safe_tp_rank)
        import vllm.distributed as vd

        monkeypatch.setattr(vd, "get_tensor_model_parallel_rank", _safe_tp_rank)
    except (ImportError, AttributeError):
        pass
