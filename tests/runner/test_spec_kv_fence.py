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

"""The sampled-token copy of a Stage-3 KV producer with a drafter is fenced
behind every KV write of the step, so a consumer that pulls on the first
token (the producer registered its blocks ahead of the finish) never reads
deferred draft KV writes."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUCopyState


class _FakeEvent:
    created: list["_FakeEvent"] = []

    def __init__(self):
        self.recorded = False
        self.synchronized = False
        _FakeEvent.created.append(self)

    def record(self):
        self.recorded = True

    def synchronize(self):
        self.synchronized = True


def test_fence_pending_device_work_rearms_ready_event(monkeypatch):
    _FakeEvent.created.clear()
    # torch_tpu registers torch.tpu only on a host with TPU chips; stub the
    # whole namespace so the test also runs on the CPU-only CI pod.
    monkeypatch.setattr(torch, "tpu", SimpleNamespace(Event=_FakeEvent), raising=False)
    first = _FakeEvent()
    first.record()
    state = AsyncTPUCopyState(
        chunks_cpu=[torch.zeros(2, dtype=torch.int32)],
        chunk_real_lens=[1],
        copy_ready_event=first,
        chunks_tpu=[torch.zeros(2, dtype=torch.int32)],
    )
    state.fence_pending_device_work()
    fence = state.copy_ready_event
    assert fence is not first
    assert fence is _FakeEvent.created[-1]
    assert fence.recorded
    state.wait()
    assert fence.synchronized
    assert not first.synchronized
    assert state.completed
    assert state.chunks_tpu is None


def _runner(*, producer: bool, method: str | None) -> TPUModelRunner:
    runner = object.__new__(TPUModelRunner)
    runner.speculative_config = (
        SimpleNamespace(
            use_eagle=lambda: method in ("eagle", "eagle3", "mtp"), method=method
        )
        if method
        else None
    )
    runner.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_producer=producer)
    )
    runner._pp_is_last = True
    return runner


def test_fence_applies_to_producers_with_a_device_drafter_only():
    assert _runner(producer=True, method="mtp")._fence_spec_kv_under_token_copy
    assert _runner(producer=True, method="eagle3")._fence_spec_kv_under_token_copy
    assert _runner(producer=True, method="dflash")._fence_spec_kv_under_token_copy
    assert not _runner(producer=True, method="ngram")._fence_spec_kv_under_token_copy
    assert not _runner(producer=True, method=None)._fence_spec_kv_under_token_copy
    assert not _runner(producer=False, method="mtp")._fence_spec_kv_under_token_copy
    runner = _runner(producer=True, method="mtp")
    runner.vllm_config = SimpleNamespace(kv_transfer_config=None)
    assert not runner._fence_spec_kv_under_token_copy
    # Stages before the last hand their hidden states on and never draft.
    runner = _runner(producer=True, method="mtp")
    runner._pp_is_last = False
    assert not runner._fence_spec_kv_under_token_copy


def test_fence_follows_the_register_ahead_flag():
    runner = _runner(producer=True, method="mtp")
    with patch(
        "vllm_torchtpu.runner.tpu_runner.dist_utils.get_stage3_register_ahead_enabled",
        return_value=False,
    ):
        assert not runner._fence_spec_kv_under_token_copy
    assert runner._fence_spec_kv_under_token_copy
