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

"""The sampled-token copy of a P/D KV producer with a drafter is fenced
behind every KV write of the step wherever the scheduler may register the
producer's blocks ahead of the finish (TPU_RAIDEN_STAGE3_REGISTER_AHEAD, or
pipeline parallelism), so a consumer that pulls on the first token never
reads deferred draft KV writes. The producer role includes a P/D child's
role under TPUMultiConnector. Producers that register at the finish keep the
fence as well."""

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


def _transfer_config(role: str = "kv_producer") -> KVTransferConfig:
    return KVTransferConfig(kv_connector="TPURaidenConnector", kv_role=role)


def _multi_config(parent_role: str, *children: tuple[str, str]) -> KVTransferConfig:
    """A TPUMultiConnector config; each child is (connector, kv_role)."""
    return KVTransferConfig(
        kv_connector="TPUMultiConnector",
        kv_role=parent_role,
        kv_connector_extra_config={
            "connectors": [
                {"kv_connector": name, "kv_role": role} for name, role in children
            ]
        },
    )


def _runner(
    *,
    transfer_config: KVTransferConfig | None = None,
    method: str | None = "mtp",
    pp_size: int = 1,
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
        kv_transfer_config=transfer_config or _transfer_config(),
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
    )
    return runner


def test_fence_applies_to_producers_with_a_device_drafter_only(stage3_env):
    assert _runner(method="mtp")._fence_spec_kv_under_token_copy
    assert _runner(method="eagle3")._fence_spec_kv_under_token_copy
    assert _runner(method="dflash")._fence_spec_kv_under_token_copy
    assert not _runner(method="ngram")._fence_spec_kv_under_token_copy
    assert not _runner(method=None)._fence_spec_kv_under_token_copy
    consumer = _runner(transfer_config=_transfer_config("kv_consumer"))
    assert not consumer._fence_spec_kv_under_token_copy
    runner = _runner()
    runner.vllm_config.kv_transfer_config = None
    assert not runner._fence_spec_kv_under_token_copy


def test_fence_keeps_finish_time_producers(stage3_env):
    # Producers that register at the finish keep the fence: an offloading
    # kv_both server (no Stage-3 transport) and GLM, DSv4 and Kimi-K3
    # producers. Dropping it is left to a measured change.
    stage3_env(TPU_KV_RESHARD_TRANSPORT="zmq")
    offload = KVTransferConfig(
        kv_connector="TPURaidenOffloadingConnector", kv_role="kv_both"
    )
    assert _runner(transfer_config=offload)._fence_spec_kv_under_token_copy
    stage3_env(TPU_USE_RAIDEN_KV_CACHE_MANAGER=True, TPU_RAIDEN_GLM_ADMISSION=True)
    assert _runner()._fence_spec_kv_under_token_copy
    stage3_env(TPU_USE_RAIDEN_KV_CACHE_MANAGER=True, TPU_RAIDEN_DSV4_ADMISSION=True)
    assert _runner()._fence_spec_kv_under_token_copy


def test_fence_follows_the_register_ahead_flag(stage3_env):
    assert _runner()._fence_spec_kv_under_token_copy
    stage3_env(TPU_RAIDEN_STAGE3_REGISTER_AHEAD=False)
    assert not _runner()._fence_spec_kv_under_token_copy


def test_fence_applies_to_pipeline_producers_without_the_flag(stage3_env):
    # A pipeline producer registers ahead whatever the flag says. On this
    # branch platforms/pp_validation.py rejects speculative decoding with
    # pipeline parallelism, so such an engine cannot start here; the case
    # guards the predicate for branches that allow a drafter on a pipeline
    # producer (main's prefill-only MTP).
    stage3_env(TPU_RAIDEN_STAGE3_REGISTER_AHEAD=False)
    assert _runner(pp_size=2)._fence_spec_kv_under_token_copy


def test_fence_follows_a_pd_child_producer_under_a_multi_connector(stage3_env):
    # vLLM builds each child's KVTransferConfig from its own entry, so a
    # kv_producer Raiden child registers ahead even under a kv_consumer
    # parent.
    child_producer = _multi_config("kv_consumer", ("TPURaidenConnector", "kv_producer"))
    assert _runner(transfer_config=child_producer)._fence_spec_kv_under_token_copy
    # A kv_both parent was fenced before; it still is.
    parent_both = _multi_config("kv_both", ("TPURaidenConnector", "kv_consumer"))
    assert _runner(transfer_config=parent_both)._fence_spec_kv_under_token_copy


def test_offload_child_alone_adds_no_fence(stage3_env):
    # An offload tier child does not send to a P/D peer: a decode engine
    # (kv_consumer parent and Raiden child) with a kv_both offload tier stays
    # unfenced.
    decode = _multi_config(
        "kv_consumer",
        ("TPURaidenConnector", "kv_consumer"),
        ("TPURaidenOffloadingConnector", "kv_both"),
    )
    assert not _runner(transfer_config=decode)._fence_spec_kv_under_token_copy
