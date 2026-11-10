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
"""The cached forward context is off by default, ignores the old environment
switch, and follows ``set_enabled``, which ``runner.host_trims`` calls under
the batch-1 rule. Off, call sites take vLLM's set_forward_context."""

import importlib
import types

import pytest
import torch
import vllm.forward_context as fc

from vllm_torchtpu.runner import fast_forward_context as ffc


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_FAST_FORWARD_CONTEXT", raising=False)
    importlib.reload(ffc)
    yield
    monkeypatch.delenv("VLLM_TPU_FAST_FORWARD_CONTEXT", raising=False)
    importlib.reload(ffc)


def test_off_by_default(fresh):
    assert ffc._ENABLED is False
    assert ffc.enabled() is False


def test_old_environment_switch_is_ignored(fresh, monkeypatch):
    monkeypatch.setenv("VLLM_TPU_FAST_FORWARD_CONTEXT", "1")
    importlib.reload(ffc)
    assert ffc.enabled() is False


def test_off_takes_the_stock_forward_context(fresh, monkeypatch):
    monkeypatch.setattr(
        fc.current_platform,
        "set_additional_forward_context",
        lambda *a, **k: {},
        raising=False,
    )
    ffc._CACHE.clear()
    pc = types.SimpleNamespace(
        data_parallel_size=1,
        use_sequence_parallel_moe=True,
        is_moe_model=True,
        data_parallel_rank=0,
    )
    cc = types.SimpleNamespace(
        static_forward_context={"x": 1},
        fast_moe_cold_start=False,
        static_all_moe_layers=None,
    )
    cfg = types.SimpleNamespace(parallel_config=pc, compilation_config=cc)
    for _ in range(2):
        with ffc.fast_set_forward_context(
            {},
            cfg,
            num_tokens=4,
            num_tokens_across_dp=torch.tensor([4], dtype=torch.int32),
        ):
            pass
    assert len(ffc._CACHE) == 0


def test_enabled_follows_the_switch(fresh):
    assert ffc.enabled() is False
    ffc.set_enabled(True)
    assert ffc.enabled() is True
    ffc._CACHE[("k",)] = object()
    ffc.set_enabled(False)
    assert ffc.enabled() is False
    assert ffc._CACHE == {}
