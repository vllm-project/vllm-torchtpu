# SPDX-License-Identifier: Apache-2.0
"""Decode KV prefetch (core/kv_prefetch_scheduler.py) on vLLM's real scheduler.

A NixlConnector consumer on CPU, as vLLM's own remote-prefill lifecycle tests
(tests/v1/kv_connector/unit) drive it: requests with do_remote_prefill start
an async load at admission and run once the connector reports them received.
"""

import pytest
import torch
from vllm import SamplingParams
from vllm.config import (
    AttentionConfig,
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu.core.kv_prefetch_scheduler import (
    KvPrefetchAsyncScheduler,
    KvPrefetchScheduler,
)

BLOCK = 16
EOS = 50256
CAP = 2


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def async_scheduling(request):
    return request.param


def _scheduler(
    monkeypatch, prefetch: int, role: str = "kv_consumer", async_scheduling=False
):
    """vLLM's create_vllm_config + create_scheduler (tests/v1/kv_connector/unit)."""
    monkeypatch.setenv("TPU_SCHED_KV_PREFETCH_WAITING", str(prefetch))
    monkeypatch.setenv("TPU_SCHED_KV_PREFETCH_RESERVE_BLOCKS", "4")
    num_blocks = 1000
    cache_config = CacheConfig(
        block_size=BLOCK, gpu_memory_utilization=0.9, enable_prefix_caching=True
    )
    cache_config.kv_cache_layout = "LBNHC"
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=CAP,
            max_num_batched_tokens=256,
            max_model_len=4096,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            async_scheduling=async_scheduling,
        ),
        model_config=ModelConfig(
            model="facebook/opt-125m", trust_remote_code=True, dtype="float16", seed=42
        ),
        cache_config=cache_config,
        device_config=DeviceConfig("cpu"),
        attention_config=AttentionConfig(backend=None),
    )
    # Past VllmConfig validation, which admits only the TPU connectors: the
    # scheduler builds its connector from this field alone.
    vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="NixlConnector", kv_role=role, kv_load_failure_policy="fail"
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            )
        ],
    )
    cls = KvPrefetchAsyncScheduler if async_scheduling else KvPrefetchScheduler
    return cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK,
    )


_hash_ready = False


def _request(rid: int, remote: bool = True, num_tokens: int = 40) -> Request:
    global _hash_ready
    if not _hash_ready:
        init_none_hash(sha256)
        _hash_ready = True
    params = SamplingParams(max_tokens=64)
    params.update_from_generation_config({}, EOS)
    req = Request(
        request_id=f"r{rid}",
        prompt_token_ids=[rid * 1000 + i for i in range(num_tokens)],
        sampling_params=params,
        pooling_params=None,
        mm_features=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )
    if remote:
        req.kv_transfer_params = dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_engine_id="prefill-engine",
            remote_request_id=f"prefill-r{rid}",
            remote_block_ids=list(range(3)),
            remote_host="prefill-host",
            remote_port=1234,
            tp_size=1,
        )
    return req


def _output(reqs, received=(), eos=()) -> ModelRunnerOutput:
    ids = [r.request_id for r in reqs]
    return ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={rid: i for i, rid in enumerate(ids)},
        sampled_token_ids=[[EOS if r in eos else 0] for r in reqs],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        kv_connector_output=(
            KVConnectorOutput(finished_recving={r.request_id for r in received})
            if received
            else None
        ),
    )


def _step(scheduler, received=(), eos=()):
    """One engine step: schedule, then report the running requests' tokens."""
    output = scheduler.schedule()
    ran = [r for r in scheduler.running if r.request_id in output.num_scheduled_tokens]
    scheduler.update_from_output(output, _output(ran, received, eos))
    assert len(scheduler.running) <= CAP
    return output


def _fill(scheduler):
    """Two received remote requests running: the set is at the cap."""
    first = [_request(1), _request(2)]
    for req in first:
        scheduler.add_request(req)
    _step(scheduler)  # both start their loads
    assert all(r.status == RequestStatus.WAITING_FOR_REMOTE_KVS for r in first)
    _step(scheduler, received=first)
    _step(scheduler)  # both promoted and running
    assert [r.status for r in first] == [RequestStatus.RUNNING] * 2
    return first


def test_without_prefetch_no_load_starts_at_the_cap(monkeypatch, async_scheduling):
    scheduler = _scheduler(monkeypatch, prefetch=0, async_scheduling=async_scheduling)
    _fill(scheduler)
    waiting = _request(3)
    scheduler.add_request(waiting)
    _step(scheduler)
    assert waiting.status == RequestStatus.WAITING


def test_prefetch_starts_loads_at_the_cap_up_to_n(monkeypatch, async_scheduling):
    scheduler = _scheduler(monkeypatch, prefetch=2, async_scheduling=async_scheduling)
    running = _fill(scheduler)
    held = [_request(3), _request(4), _request(5)]
    for req in held:
        scheduler.add_request(req)
    output = _step(scheduler)
    # Two loads started, the third refused; nothing joined the running set,
    # and the running requests still decoded this step.
    assert [r.status for r in held] == [
        RequestStatus.WAITING_FOR_REMOTE_KVS,
        RequestStatus.WAITING_FOR_REMOTE_KVS,
        RequestStatus.WAITING,
    ]
    assert scheduler.running == running
    assert set(output.num_scheduled_tokens) == {r.request_id for r in running}
    assert scheduler.max_num_running_reqs == CAP


def test_a_received_request_waits_for_a_slot_then_takes_it(
    monkeypatch, async_scheduling
):
    scheduler = _scheduler(monkeypatch, prefetch=2, async_scheduling=async_scheduling)
    running = _fill(scheduler)
    held = _request(3)
    scheduler.add_request(held)
    _step(scheduler)
    assert held.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    # Its KV arrives while the set is full; a fresh request behind it lifts
    # the next step. The received one stays held, and the scan passes it to
    # start the fresh one's load.
    _step(scheduler, received=[held])
    behind = _request(4)
    scheduler.add_request(behind)
    _step(scheduler)
    assert held.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert behind.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.running == running
    # A running request finishes: the held one takes the slot next step.
    _step(scheduler, eos=[running[0]])
    output = _step(scheduler)
    assert held.status == RequestStatus.RUNNING
    assert held.request_id in output.num_scheduled_tokens
    assert behind.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert len(scheduler.running) == CAP


def test_a_local_request_never_joins_past_the_cap(monkeypatch, async_scheduling):
    scheduler = _scheduler(monkeypatch, prefetch=2, async_scheduling=async_scheduling)
    running = _fill(scheduler)
    local = _request(3, remote=False)
    behind = _request(4)
    scheduler.add_request(local)
    scheduler.add_request(behind)
    _step(scheduler)
    # The local prefill needs a slot, so the scan stops at it.
    assert local.status == RequestStatus.WAITING
    assert behind.status == RequestStatus.WAITING
    assert scheduler.running == running


def test_prefetch_is_off_on_a_producer(monkeypatch, async_scheduling):
    consumer = _scheduler(monkeypatch, prefetch=2, async_scheduling=async_scheduling)
    producer = _scheduler(
        monkeypatch, prefetch=2, role="kv_producer", async_scheduling=async_scheduling
    )
    assert consumer._kv_prefetch_n == 2
    assert producer._kv_prefetch_n == 0
