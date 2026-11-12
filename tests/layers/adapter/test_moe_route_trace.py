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
"""The routing trace writes each step's per-expert counts and logs them back."""

import importlib
import logging
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def trace(monkeypatch):
    monkeypatch.setenv("TPU_MOE_ROUTE_TRACE", "4")
    monkeypatch.setenv("TPU_MOE_ROUTE_TRACE_EVERY", "1")
    from vllm_torchtpu.layers.adapter import moe_route_trace as mod

    mod = importlib.reload(mod)
    monkeypatch.setattr(mod, "_install_runner_hook", lambda: None)
    yield mod
    monkeypatch.delenv("TPU_MOE_ROUTE_TRACE")
    importlib.reload(mod)


def _layer(i):
    cfg = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=True), ep_size=2, ep_rank=0
    )
    return SimpleNamespace(
        layer_name=f"model.layers.{i}.block_sparse_moe.experts",
        global_num_experts=8,
        w13_weight=torch.zeros(1),
        moe_config=cfg,
    )


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("TPU_MOE_ROUTE_TRACE", raising=False)
    from vllm_torchtpu.layers.adapter import moe_route_trace as mod

    mod = importlib.reload(mod)
    layer = _layer(3)
    mod.register(layer)
    assert not mod.ENABLED
    assert getattr(layer, "_route_trace_row", None) is None


def test_ring_round_trip(trace, caplog):
    layers = [_layer(3), _layer(4)]
    for layer in layers:
        trace.register(layer)
    steps = [
        [torch.tensor([[0, 1], [1, 7]]), torch.tensor([[2, 3], [2, 2]])],
        [torch.tensor([[5, 6]]), torch.tensor([[0, 4]])],
    ]
    for ids_per_layer in steps:
        for layer, ids in zip(layers, ids_per_layer):
            trace.record(layer, ids)
    with caplog.at_level(logging.INFO):
        trace.logger.propagate = True
        trace.dump()
    parsed = [trace.parse_line(r.getMessage()) for r in caplog.records]
    parsed = [p for p in parsed if p is not None]
    assert parsed == [
        (0, 2, [{0: 1, 1: 2, 7: 1}, {2: 3, 3: 1}]),
        (1, 1, [{5: 1, 6: 1}, {0: 1, 4: 1}]),
    ]
    # Nothing new: no second copy of the same steps.
    caplog.clear()
    trace.dump()
    assert not [r for r in caplog.records if "s=" in r.getMessage()]


def test_ring_wraps(trace, caplog):
    layer = _layer(3)
    trace.register(layer)
    for s in range(6):
        trace.record(layer, torch.tensor([[s % 8, (s + 1) % 8]]))
    with caplog.at_level(logging.INFO):
        trace.logger.propagate = True
        trace.dump()
    parsed = [trace.parse_line(r.getMessage()) for r in caplog.records]
    steps = [p[0] for p in parsed if p is not None]
    assert steps == [2, 3, 4, 5]
