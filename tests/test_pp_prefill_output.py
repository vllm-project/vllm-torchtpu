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
"""Ownership and FIFO contracts for synchronous-prefill overlap."""

import importlib.util
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def output_stub(*, get_output, kv=None):
    return SimpleNamespace(get_output=get_output, get_kv_connector_output=lambda: kv)


@pytest.fixture
def feature(monkeypatch):
    outputs = ModuleType("vllm.v1.outputs")
    outputs.AsyncModelRunnerOutput = type("AsyncModelRunnerOutput", (), {})
    monkeypatch.setitem(sys.modules, "vllm.v1.outputs", outputs)
    path = Path(__file__).parents[1] / "src/vllm_torchtpu/runner/pp_prefill_output.py"
    spec = importlib.util.spec_from_file_location("prefill_output_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_is_single_reader_and_caches_failure(feature):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def materialize():
        calls.append(1)
        entered.set()
        assert release.wait(2)
        return "tokens"

    pending = feature.PendingOutput(
        output_stub(get_output=materialize), object(), ["r"]
    )
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(pending.get_output)
        assert entered.wait(2)
        second = pool.submit(pending.get_output)
        release.set()
        assert first.result() == second.result() == "tokens"
    assert calls == [1]
    assert pending.output is pending.drafts is None
    error = RuntimeError("copy failed")

    def fail():
        raise error

    pending = feature.PendingOutput(output_stub(get_output=fail), object(), ["r"])
    for _ in range(2):
        with pytest.raises(RuntimeError) as caught:
            pending.get_output()
        assert caught.value is error


def scheduled(finished=(), preempted=(), resumed=(), tokens=8192):
    return SimpleNamespace(
        finished_req_ids=set(finished),
        preempted_req_ids=set(preempted),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set(resumed)),
        total_num_scheduled_tokens=tokens,
    )


@pytest.mark.parametrize("kind", ["finished", "preempted", "resumed"])
def test_retirement_precedes_state_reuse_but_not_disjoint_execute(feature, kind):
    order = []
    runner = SimpleNamespace(
        _wait_input_staging_fence=lambda: order.append("upload-fence")
    )
    owner = feature.PrefillOutput(runner)
    assert not owner.active  # Direct warmup calls retain synchronous behavior.
    owner.wrap(
        output_stub(get_output=lambda: order.append("device-done")), None, ["old"]
    )
    owner.begin_execute(scheduled(**{kind: ["unrelated"]}))
    assert order == ["upload-fence"]
    assert owner.pending is not None and owner.active
    owner.begin_execute(scheduled(**{kind: ["old"]}))
    assert order == ["upload-fence", "upload-fence", "device-done"]
    assert owner.pending is None


def test_non_reply_rank_retires_on_next_sample(feature):
    calls = []
    owner = feature.PrefillOutput(None)
    owner.wrap(output_stub(get_output=lambda: calls.append("ready")), object(), ["r"])
    owner.retire()
    owner.retire()
    assert calls == ["ready"]


def test_drain_distinguishes_device_completion_and_reply_enqueue(feature):
    owner = feature.PrefillOutput(None)
    materialized = threading.Event()
    owner.wrap(output_stub(get_output=lambda: materialized.set()), None, ["r"])
    owner.submitted = 1
    with ThreadPoolExecutor(1) as pool:
        drained = pool.submit(owner.drain)
        assert materialized.wait(2)
        assert not drained.done()
        with owner.replies:
            owner.enqueued = 1
            owner.replies.notify_all()
        drained.result(timeout=2)
    owner.dispatch_error = RuntimeError("transport failed")
    with pytest.raises(RuntimeError, match="reply dispatch failed"):
        owner.drain()


def test_create_selects_only_mtp_ring_prefill(feature, monkeypatch):
    monkeypatch.setattr(feature, "install_worker", lambda: None)
    monkeypatch.setenv("TPU_PP_HANDOFF", "ring")
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=4, data_parallel_size=1),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
    )
    runner = SimpleNamespace(
        vllm_config=config,
        _mtp_prefill_only=True,
        is_pooling_model=False,
        _pp_is_last=True,
        _fence_spec_kv_under_token_copy=True,
    )
    assert isinstance(feature.create(runner), feature.PrefillOutput)
    runner._fence_spec_kv_under_token_copy = False
    with pytest.raises(ValueError, match="fence"):
        feature.create(runner)
    runner._fence_spec_kv_under_token_copy = True
    for role in ("kv_consumer", "kv_both"):
        config.kv_transfer_config.kv_role = role
        assert feature.create(runner) is None
    config.kv_transfer_config.kv_role = "kv_producer"
    runner._mtp_prefill_only = False
    assert feature.create(runner) is None
    runner._mtp_prefill_only = True
    runner._pp_is_last = False
    assert feature.create(runner) is None


def test_failure_preserves_kv_output_after_source_is_released(feature):
    kv = object()

    def fail():
        raise RuntimeError("copy failed")

    pending = feature.PendingOutput(output_stub(get_output=fail, kv=kv), None, ["r"])
    assert pending.get_kv_connector_output() is kv
    with pytest.raises(RuntimeError, match="copy failed"):
        pending.get_output()
    assert pending.output is None
    assert pending.get_kv_connector_output() is kv
