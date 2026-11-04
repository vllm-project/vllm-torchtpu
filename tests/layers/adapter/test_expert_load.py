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
"""Host-side checks for the routed-row accumulator readback (expert_load.py)."""

import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import expert_load

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(expert_load, "_REGISTERED", [])
    monkeypatch.setattr(expert_load, "_STATE", {"steps": 0})
    monkeypatch.setattr(expert_load, "_READER", None)


def _layer(name):
    layer = torch.nn.Module()
    layer.layer_name = name
    layer.w13_weight = torch.zeros(2)
    return layer


def test_off_registers_nothing_and_leaves_the_operand_absent(monkeypatch):
    monkeypatch.setattr(envs, "TPU_MOE_EPLB_DYNAMIC", False)
    layer = _layer("model.layers.3.ffn.experts")
    expert_load.register(layer, physical=8)
    assert expert_load.operand(layer) is None
    assert not expert_load.enabled()
    expert_load.on_step(1)  # nothing registered: a no-op
    assert expert_load.snapshot() == {}


def test_the_synchronous_path_snapshots_every_layer_at_its_width(monkeypatch):
    monkeypatch.setattr(envs, "TPU_MOE_EPLB_DYNAMIC", True)
    a, b = _layer("model.layers.0.ffn.experts"), _layer("model.layers.1.ffn.experts")
    expert_load.register(a, physical=4)
    expert_load.register(b, physical=2)  # a drafter layer: fewer slots
    a._tpu_layout_key = ("target", 0)
    b._tpu_layout_key = ("draft", 0)
    assert expert_load.operand(a).tolist() == [0, 0, 0, 0]
    expert_load.commit(a, torch.tensor([3, 1, 0, 0], dtype=torch.int32))
    expert_load.commit(b, torch.tensor([5, 6], dtype=torch.int32))
    snap = expert_load.snapshot()
    assert snap[("target", 0)].tolist() == [3, 1, 0, 0]
    assert snap[("draft", 0)].tolist() == [5, 6]


def _fake_events(monkeypatch):
    """A torch.tpu.Event stand-in: a recorded event completes when the test
    lands it; copies are the CPU tensors themselves (already on the host)."""
    import types

    events = []

    class Event:
        def __init__(self):
            self.recorded, self.done = False, False

        def record(self):
            self.recorded = True
            events.append(self)

        def query(self):
            return self.done

        def synchronize(self):
            self.done = True

    monkeypatch.setattr(torch, "tpu", types.SimpleNamespace(Event=Event), raising=False)
    # The CPU guard keeps the suite on the synchronous path; lift it here.
    monkeypatch.setattr(expert_load, "_reader", _bind)
    return events


def _bind():
    """Production's `_reader()` without its CPU guard."""
    if expert_load._READER is not None or expert_load._STATE.get("raw_disabled"):
        return expert_load._READER
    try:
        reader = expert_load._AsyncReader()
        reader.bind()
    except Exception:  # noqa: BLE001
        expert_load._STATE["raw_disabled"] = True
        return None
    expert_load._READER = reader
    return reader


def test_async_reader_issues_on_demand_adopts_at_the_hook_never_waits(monkeypatch):
    monkeypatch.setattr(envs, "TPU_MOE_EPLB_DYNAMIC", True)
    events = _fake_events(monkeypatch)
    a, b = _layer("model.layers.0.ffn.experts"), _layer("model.layers.1.ffn.experts")
    expert_load.register(a, physical=4)
    expert_load.register(b, physical=2)
    a._tpu_layout_key = ("target", 0)
    b._tpu_layout_key = ("draft", 0)
    a._tpu_expert_load.copy_(torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    b._tpu_expert_load.copy_(torch.tensor([5, 6], dtype=torch.int32))
    expert_load.on_step(1)
    reader = expert_load._READER
    assert reader is not None and not reader.polls_block and events == []
    assert expert_load.snapshot() == {}  # nothing landed: no wait, no counts
    expert_load.issue_now()  # after the forward: issue
    assert len(events) == 1 and reader.latest is None
    expert_load.on_step(2)  # not landed yet: nothing adopted, no wait
    assert reader.latest is None
    events[0].done = True
    expert_load.on_step(3)  # adopted at the hook
    assert reader.latest.tolist() == [[1, 2, 3, 4], [5, 6, 0, 0]]
    assert expert_load.snapshot()[("target", 0)].tolist() == [1, 2, 3, 4]
    # A read still in flight is adopted (waited for) by flush().
    a._tpu_expert_load.copy_(torch.tensor([7, 7, 7, 7], dtype=torch.int32))
    expert_load.issue_now()
    assert len(events) == 2 and not events[1].done
    expert_load.flush()
    assert expert_load.snapshot()[("target", 0)].tolist() == [7, 7, 7, 7]


def test_async_reader_without_an_event_falls_back_to_the_synchronous_path(monkeypatch):
    import types

    monkeypatch.setattr(envs, "TPU_MOE_EPLB_DYNAMIC", True)

    def no_event():
        raise RuntimeError("no device")

    monkeypatch.setattr(
        torch, "tpu", types.SimpleNamespace(Event=no_event), raising=False
    )
    monkeypatch.setattr(expert_load, "_reader", _bind)
    a = _layer("model.layers.0.ffn.experts")
    expert_load.register(a, physical=4)
    a._tpu_layout_key = ("target", 0)
    expert_load.on_step(1)
    assert expert_load._STATE.get("raw_disabled") and expert_load._READER is None
