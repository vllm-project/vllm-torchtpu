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
behind every KV write of the step when the producer registers its blocks
ahead of the finish, so a consumer that pulls on the first token never
reads deferred draft KV writes. Producers that register at the finish sync
the KV caches before registering and are not fenced."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import KVTransferConfig

from vllm_torchtpu import envs as tpu_envs
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


@pytest.fixture
def stage3_env(monkeypatch):
    """Sets envs over a non-GLM Raiden Stage-3 producer's: the Raiden
    transport, register-ahead at its default (on) and no finish-time
    admission (GLM, DSv4, Kimi-K3)."""

    def set_envs(**overrides):
        values = {
            "TPU_KV_RESHARD_TRANSPORT": "raiden",
            "TPU_RAIDEN_STAGE3_REGISTER_AHEAD": True,
            "TPU_USE_RAIDEN_KV_CACHE_MANAGER": False,
            "TPU_RAIDEN_GLM_ADMISSION": False,
            "TPU_RAIDEN_DSV4_ADMISSION": False,
            "TPU_RAIDEN_KIMIK3_ADMISSION": False,
            **overrides,
        }
        for name, value in values.items():
            monkeypatch.setattr(tpu_envs, name, value, raising=False)

    set_envs()
    return set_envs


def _runner(
    *,
    producer: bool = True,
    method: str | None = "mtp",
    pp_size: int = 1,
    architecture: str = "Qwen3NextForCausalLM",
) -> TPUModelRunner:
    runner = object.__new__(TPUModelRunner)
    runner.speculative_config = (
        SimpleNamespace(
            use_eagle=lambda: method in ("eagle", "eagle3", "mtp"), method=method
        )
        if method
        else None
    )
    runner.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_producer=producer),
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
        model_config=SimpleNamespace(architecture=architecture),
    )
    return runner


def test_fence_applies_to_producers_with_a_device_drafter_only(stage3_env):
    assert _runner(method="mtp")._fence_spec_kv_under_token_copy
    assert _runner(method="eagle3")._fence_spec_kv_under_token_copy
    assert _runner(method="dflash")._fence_spec_kv_under_token_copy
    assert not _runner(method="ngram")._fence_spec_kv_under_token_copy
    assert not _runner(method=None)._fence_spec_kv_under_token_copy
    assert not _runner(producer=False)._fence_spec_kv_under_token_copy
    runner = _runner()
    runner.vllm_config.kv_transfer_config = None
    assert not runner._fence_spec_kv_under_token_copy


def test_fence_skips_an_offloading_server(stage3_env):
    # kv_both counts as a KV producer, but without the Raiden Stage-3
    # transport nothing registers ahead of the finish.
    stage3_env(TPU_KV_RESHARD_TRANSPORT="zmq")
    runner = _runner()
    runner.vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="TPURaidenOffloadingConnector", kv_role="kv_both"
    )
    assert runner.vllm_config.kv_transfer_config.is_kv_producer
    assert not runner._fence_spec_kv_under_token_copy


@pytest.mark.parametrize("pp_size", (1, 2))
def test_fence_skips_glm_producers(stage3_env, pp_size):
    stage3_env(TPU_USE_RAIDEN_KV_CACHE_MANAGER=True, TPU_RAIDEN_GLM_ADMISSION=True)
    assert not _runner(pp_size=pp_size)._fence_spec_kv_under_token_copy


def test_fence_skips_dsv4_and_kimi_k3_producers(stage3_env):
    stage3_env(TPU_USE_RAIDEN_KV_CACHE_MANAGER=True, TPU_RAIDEN_DSV4_ADMISSION=True)
    assert not _runner()._fence_spec_kv_under_token_copy
    stage3_env(TPU_USE_RAIDEN_KV_CACHE_MANAGER=True, TPU_RAIDEN_KIMIK3_ADMISSION=True)
    assert _runner()._fence_spec_kv_under_token_copy
    runner = _runner(architecture="KimiK3ForConditionalGeneration")
    assert not runner._fence_spec_kv_under_token_copy


def test_fence_follows_the_register_ahead_flag(stage3_env):
    assert _runner()._fence_spec_kv_under_token_copy
    stage3_env(TPU_RAIDEN_STAGE3_REGISTER_AHEAD=False)
    assert not _runner()._fence_spec_kv_under_token_copy


def test_fence_applies_to_pipeline_producers_without_the_flag(stage3_env):
    # A pipeline producer registers ahead whatever the flag says.
    stage3_env(TPU_RAIDEN_STAGE3_REGISTER_AHEAD=False)
    assert _runner(pp_size=2)._fence_spec_kv_under_token_copy
