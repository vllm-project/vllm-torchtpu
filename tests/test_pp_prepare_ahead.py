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
"""Capacity, failure wakeup and receive-credit contracts for prepare-ahead."""

import contextlib
import importlib.util
import sys
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def feature(monkeypatch):
    annotation = ModuleType("vllm_torchtpu.tracing.annotation")
    annotation.TraceAnnotation = lambda _: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, annotation.__name__, annotation)
    path = (
        Path(__file__).parents[1] / "src/vllm_torchtpu/distributed/pp_prepare_ahead.py"
    )
    spec = importlib.util.spec_from_file_location("prepare_ahead_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ring(depth=4, carried=1, last=False):
    value = SimpleNamespace(
        depth=depth,
        carried=tuple(range(carried)),
        rank=3 if last else 0,
        last=3,
        _cv=threading.Condition(),
        _sends=deque(),
        _error=None,
        _closing=False,
    )

    def check():
        if value._error is not None:
            raise value._error

    value._raise_send_failure_locked = check
    return value


def test_overlap_keeps_original_live_output_bound(feature):
    value = ring()
    owner = feature.PrepareAhead(value)
    for i in range(value.depth):
        value._sends.append((i, (object(),), object()))
    # The old end_forward would block holding this fifth output locally.
    owner.enqueue(4, (object(),), object())
    assert len(value._sends) == value.depth + 1
    prepared = threading.Event()

    def next_forward():
        prepared.set()
        owner.wait()
        owner.enqueue(5, (object(),), object())

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(next_forward)
        assert prepared.wait(2)
        assert not future.done()
        with value._cv:
            value._sends.popleft()
            value._cv.notify_all()
        # <= depth must admit the forward after exactly ONE completion.
        future.result(timeout=2)
    assert [item[0] for item in value._sends] == [1, 2, 3, 4, 5]
    assert len(value._sends) == value.depth + 1


@pytest.mark.parametrize("closing", [False, True])
def test_credit_wait_wakes_on_failure_or_close(feature, closing):
    value = ring()
    value._sends.extend(range(value.depth + 1))
    owner = feature.PrepareAhead(value)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(owner.wait)
        with value._cv:
            if closing:
                value._closing = True
            else:
                value._error = RuntimeError("send failed")
            value._cv.notify_all()
        with pytest.raises(RuntimeError, match="close|send failed"):
            future.result(timeout=2)


def test_prepare_ahead_selects_only_stages_with_outgoing_tensors(feature, monkeypatch):
    monkeypatch.setattr(feature, "install_worker", lambda: None)
    assert feature.create(ring(depth=5)) is not None
    assert feature.create(ring(last=True)) is None


@pytest.mark.parametrize("tokens", [0, 8192])
def test_pending_metadata_copy_reads_old_contents_before_batch_mutation(
    feature, monkeypatch, tokens
):
    # Model contiguous source storage retained by asynchronous copy_slice.
    source, copied = [0.25], []

    class Worker:
        def execute_model(self, scheduled):
            # add/condense for real work, remove for a zero-token step.
            source[0] = 0.75 if scheduled.total_num_scheduled_tokens else 0.0
            return "done"

    module = ModuleType("vllm_torchtpu.worker.tpu_worker")
    module.TPUWorker = Worker
    monkeypatch.setitem(sys.modules, module.__name__, module)
    feature.install_worker()
    worker = Worker()
    worker.model_runner = SimpleNamespace(
        _wait_input_staging_fence=lambda: copied.append(source[0])
    )
    assert (
        worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=tokens))
        == "done"
    )
    assert copied == [0.25]
    assert source == [0.75 if tokens else 0.0]
