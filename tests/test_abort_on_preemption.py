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
"""Capacity-preemption abort contracts; no model weights or TPU execution."""

from typing import Literal

import pytest
import torch
from transformers import LlamaConfig
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.engine import FinishReason
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu import _patch_vllm_abort_on_preemption
from vllm_torchtpu.core.abort_on_preemption import (
    _CapacityPreemption,
    _should_enable_abort_on_preemption,
)

BLOCK_SIZE = 16
EOS_TOKEN_ID = 127
_none_hash_initialized = False


def _preemption_state(scheduler: Scheduler) -> _CapacityPreemption:
    state = scheduler.__dict__["_tpu_capacity_preemption"]
    assert isinstance(state, _CapacityPreemption)
    return state


class _HashAdditionalConfig:
    def compute_hash(self) -> str:
        return "scheduler-test-config"


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> str:
    model_dir = tmp_path_factory.mktemp("tiny-llama")
    LlamaConfig(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=256,
    ).save_pretrained(model_dir)
    return str(model_dir)


def _create_scheduler(
    model: str,
    *,
    enabled: bool | None = None,
    kv_load_failure_policy: Literal["recompute", "fail"] = "fail",
    policy: Literal["fcfs", "priority"] = "fcfs",
    num_blocks: int = 11,
    enable_prefix_caching: bool = False,
    hash_additional_config: bool = False,
) -> Scheduler:
    _patch_vllm_abort_on_preemption()
    model_config = ModelConfig(
        model=model,
        tokenizer=model,
        dtype="float16",
        skip_tokenizer_init=True,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=8,
        max_num_batched_tokens=256,
        max_model_len=256,
        enable_chunked_prefill=True,
        is_encoder_decoder=False,
        policy=policy,
        disable_hybrid_kv_cache_manager=True,
        watermark=0.0,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
    )
    cache_config.num_gpu_blocks = num_blocks
    additional_config = {} if enabled is None else {"abort_on_preemption": enabled}
    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
        additional_config=additional_config,
    )
    vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="ExampleConnector",
        kv_role="kv_consumer",
        kv_connector_extra_config={"shared_storage_path": model},
        kv_load_failure_policy=kv_load_failure_policy,
    )
    if hash_additional_config:
        vllm_config.additional_config = _HashAdditionalConfig()
    register_all_kvcache_specs(vllm_config)
    kv_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], kv_spec)],
    )
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )
    # Connector transport is outside this file's contract. Detach the in-process
    # connector after init so local prefix-cache behavior stays authoritative.
    scheduler.connector = None
    scheduler.requires_kv_delivery = False
    return scheduler


def _request(
    request_id: str,
    num_tokens: int,
    *,
    client_index: int = 0,
    priority: int = 0,
    arrival_time: float = 0.0,
    token_id: int = 1,
) -> Request:
    global _none_hash_initialized
    if not _none_hash_initialized:
        init_none_hash(sha256)
        _none_hash_initialized = True
    sampling_params = SamplingParams(ignore_eos=True, max_tokens=32)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=request_id,
        prompt_token_ids=[token_id] * num_tokens,
        sampling_params=sampling_params,
        pooling_params=None,
        client_index=client_index,
        priority=priority,
        arrival_time=arrival_time,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _runner_output(
    request_ids: list[str],
    token: int = 42,
    kv_connector_output: KVConnectorOutput | None = None,
) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=request_ids,
        req_id_to_index={request_id: i for i, request_id in enumerate(request_ids)},
        sampled_token_ids=[[token] for _ in request_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        kv_connector_output=kv_connector_output,
    )


@pytest.mark.parametrize(
    ("kv_load_failure_policy", "enabled"),
    [
        ("recompute", None),
        ("fail", False),
    ],
)
@pytest.mark.parametrize("hash_additional_config", [False, True])
def test_recompute_policy_or_explicit_off_keeps_capacity_preemption_recomputable(
    tiny_model: str,
    kv_load_failure_policy: Literal["recompute", "fail"],
    enabled: bool | None,
    hash_additional_config: bool,
) -> None:
    if enabled is False and hash_additional_config:
        pytest.skip("hash_additional_config replaces the additional_config dict")
    scheduler = _create_scheduler(
        tiny_model,
        enabled=enabled,
        kv_load_failure_policy=kv_load_failure_policy,
        num_blocks=6,
        hash_additional_config=hash_additional_config,
    )
    request = _request("request", 80)
    scheduler.add_request(request)

    first = scheduler.schedule()
    scheduler.update_from_output(first, _runner_output([request.request_id]))
    second = scheduler.schedule()

    assert second.total_num_scheduled_tokens == 0
    assert request.status == RequestStatus.PREEMPTED
    assert request in scheduler.waiting
    assert request.request_id in scheduler.requests


@pytest.mark.parametrize(
    ("kv_load_failure_policy", "enabled"),
    [
        ("fail", None),
        ("recompute", True),
    ],
)
def test_last_request_finishes_from_zero_token_preemption_step(
    tiny_model: str,
    kv_load_failure_policy: Literal["recompute", "fail"],
    enabled: bool | None,
) -> None:
    scheduler = _create_scheduler(
        tiny_model,
        enabled=enabled,
        kv_load_failure_policy=kv_load_failure_policy,
        num_blocks=6,
    )
    request = _request("last", 80, client_index=2)
    scheduler.add_request(request)

    unwritten = scheduler.kv_cache_manager.block_pool.__dict__.setdefault(
        "_tpu_unwritten_block_ids", set()
    )
    unwritten.add(-1)
    first = scheduler.schedule()
    assert not unwritten
    scheduler.update_from_output(first, _runner_output([request.request_id]))
    terminal_step = scheduler.schedule()

    assert terminal_step.total_num_scheduled_tokens == 0
    assert terminal_step.finished_req_ids == {request.request_id}
    outputs = scheduler.update_from_output(terminal_step, _runner_output([]))
    assert len(outputs[2].outputs) == 1
    terminal = outputs[2].outputs[0]
    assert terminal.request_id == request.request_id
    assert terminal.new_token_ids == []
    assert terminal.finish_reason == FinishReason.ERROR
    assert terminal.kv_transfer_params is not None
    assert terminal.kv_transfer_params["kv_transfer_error"]["status_name"] == (
        "model.KVCachePreempt"
    )

    empty_step = scheduler.schedule()
    repeated = scheduler.update_from_output(empty_step, _runner_output([]))
    assert all(not batch.outputs for batch in repeated.values())


def test_decode_load_failure_and_capacity_terminal_compose_once(
    tiny_model: str,
) -> None:
    scheduler = _create_scheduler(tiny_model, num_blocks=6)
    assert type(scheduler) is Scheduler
    capacity_victim = _request("capacity-victim", 80, client_index=2)
    scheduler.add_request(capacity_victim)

    first = scheduler.schedule()
    scheduler.update_from_output(first, _runner_output([capacity_victim.request_id]))
    terminal_step = scheduler.schedule()
    assert terminal_step.finished_req_ids == {capacity_victim.request_id}
    assert set(_preemption_state(scheduler).pending) == {capacity_victim.request_id}

    load_failed = _request("load-failed", 16, client_index=7)
    scheduler.add_request(load_failed)
    load_failed.status = RequestStatus.WAITING_FOR_REMOTE_KVS

    outputs = scheduler.update_from_output(
        terminal_step,
        _runner_output(
            [],
            kv_connector_output=KVConnectorOutput(
                failed_recving={load_failed.request_id}
            ),
        ),
    )

    (capacity_terminal,) = outputs[2].outputs
    (load_terminal,) = outputs[7].outputs
    assert capacity_terminal.request_id == capacity_victim.request_id
    assert capacity_terminal.new_token_ids == []
    assert capacity_terminal.finish_reason is FinishReason.ERROR
    assert capacity_terminal.kv_transfer_params is not None
    assert (
        capacity_terminal.kv_transfer_params["kv_transfer_error"]["status_name"]
        == "model.KVCachePreempt"
    )
    assert load_terminal.request_id == load_failed.request_id
    assert load_terminal.new_token_ids == []
    assert load_terminal.finish_reason is FinishReason.ERROR
    assert _preemption_state(scheduler).pending == {}

    next_step = scheduler.schedule()
    repeated = scheduler.update_from_output(
        next_step,
        _runner_output(
            [],
            kv_connector_output=KVConnectorOutput(
                failed_recving={load_failed.request_id}
            ),
        ),
    )
    assert all(not batch.outputs for batch in repeated.values())


def test_terminal_waits_for_its_finished_id_step(tiny_model: str) -> None:
    scheduler = _create_scheduler(tiny_model, num_blocks=3)
    scheduler.defer_block_free = True
    older = _request("older", 14)
    victim = _request("victim", 16, client_index=4)
    for request in (older, victim):
        scheduler.add_request(request)

    first = scheduler.schedule()
    scheduler.update_from_output(
        first, _runner_output([older.request_id, victim.request_id])
    )
    victim.next_decode_eligible_step = scheduler.current_step + 2

    older_step = scheduler.schedule()
    assert older_step.num_scheduled_tokens == {older.request_id: 1}
    assert older.last_sched_seq > scheduler.processed_step_seq
    assert victim.last_sched_seq <= scheduler.processed_step_seq
    terminal_step = scheduler.schedule()
    assert terminal_step.total_num_scheduled_tokens == 0
    assert terminal_step.finished_req_ids == {victim.request_id}
    assert set(_preemption_state(scheduler).pending) == {victim.request_id}

    older_outputs = scheduler.update_from_output(
        older_step, _runner_output([older.request_id])
    )
    assert all(
        output.request_id != victim.request_id
        for batch in older_outputs.values()
        for output in batch.outputs
    )
    assert set(_preemption_state(scheduler).pending) == {victim.request_id}

    terminal_outputs = scheduler.update_from_output(terminal_step, _runner_output([]))
    assert [output.request_id for output in terminal_outputs[4].outputs] == [
        victim.request_id
    ]
    assert _preemption_state(scheduler).pending == {}

    next_step = scheduler.schedule()
    repeated = scheduler.update_from_output(
        next_step, _runner_output(list(next_step.num_scheduled_tokens))
    )
    assert all(
        output.request_id != victim.request_id
        for batch in repeated.values()
        for output in batch.outputs
    )


@pytest.mark.parametrize("policy", ["fcfs", "priority"])
def test_policy_victim_gets_one_structured_terminal(
    tiny_model: str, policy: Literal["fcfs", "priority"]
) -> None:
    scheduler = _create_scheduler(tiny_model, policy=policy, num_blocks=11)
    active = _request("active", 80, priority=0, arrival_time=0.0)
    victim = _request("victim", 80, client_index=3, priority=5, arrival_time=1.0)

    scheduler.add_request(active)
    active_step = scheduler.schedule()
    scheduler.add_request(victim)
    victim_step = scheduler.schedule()
    scheduler.update_from_output(active_step, _runner_output([active.request_id]))
    scheduler.update_from_output(victim_step, _runner_output([victim.request_id]))

    preemption_step = scheduler.schedule()

    assert victim.status == RequestStatus.FINISHED_ERROR
    assert victim.request_id not in scheduler.requests
    assert victim not in scheduler.waiting
    assert victim not in scheduler.running
    assert preemption_step.finished_req_ids == {victim.request_id}
    assert preemption_step.preempted_req_ids == {victim.request_id}

    outputs = scheduler.update_from_output(
        preemption_step, _runner_output([active.request_id])
    )
    terminal = outputs[3].outputs[0]
    assert terminal.request_id == victim.request_id
    assert terminal.new_token_ids == []
    assert terminal.finish_reason == FinishReason.ERROR
    assert terminal.stop_reason == "kv_cache_preempted"
    assert terminal.kv_transfer_params == {
        "kv_transfer_error": {
            "status_code": 500,
            "status_name": "model.KVCachePreempt",
            "status_message": "D-side request was preempted due to KV cache capacity.",
        }
    }

    next_step = scheduler.schedule()
    next_outputs = scheduler.update_from_output(
        next_step, _runner_output([active.request_id])
    )
    assert all(
        output.request_id != victim.request_id
        for batch in next_outputs.values()
        for output in batch.outputs
    )


@pytest.mark.parametrize("policy", ["fcfs", "priority"])
def test_inflight_victim_is_kept_until_step_drains(
    tiny_model: str, policy: Literal["fcfs", "priority"]
) -> None:
    scheduler = _create_scheduler(tiny_model, policy=policy, num_blocks=4)
    pressure = _request("pressure", 16, priority=0, arrival_time=0.0)
    schedulable = _request("schedulable", 14, priority=1, arrival_time=1.0)
    victim = _request("victim", 16, priority=5, arrival_time=2.0)
    for request in (pressure, schedulable, victim):
        scheduler.add_request(request)

    first = scheduler.schedule()
    scheduler.update_from_output(
        first,
        _runner_output(
            [pressure.request_id, schedulable.request_id, victim.request_id]
        ),
    )
    scheduler.defer_block_free = True
    victim.last_sched_seq = scheduler.processed_step_seq + 1

    guarded = scheduler.schedule()

    assert guarded.num_scheduled_tokens == {}
    assert scheduler.running == [pressure, schedulable, victim]
    assert all(request.status == RequestStatus.RUNNING for request in scheduler.running)
    assert _preemption_state(scheduler).pending == {}

    scheduler.processed_step_seq = victim.last_sched_seq
    resumed = scheduler.schedule()
    assert resumed.num_scheduled_tokens == {
        pressure.request_id: 1,
        schedulable.request_id: 1,
    }
    assert victim.status == RequestStatus.FINISHED_ERROR
    assert set(_preemption_state(scheduler).pending) == {victim.request_id}


def test_management_reset_still_requeues_requests(tiny_model: str) -> None:
    scheduler = _create_scheduler(
        tiny_model,
        num_blocks=20,
        enable_prefix_caching=True,
    )
    requests = [_request("first", 16), _request("second", 16)]
    for request in requests:
        scheduler.add_request(request)
    first = scheduler.schedule()
    scheduler.update_from_output(
        first, _runner_output([request.request_id for request in requests])
    )

    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    assert scheduler.running == []
    assert list(scheduler.waiting) == requests
    assert all(request.status == RequestStatus.PREEMPTED for request in requests)
    assert _preemption_state(scheduler).pending == {}


def test_abort_releases_only_victims_shared_prefix_reference(
    tiny_model: str,
) -> None:
    scheduler = _create_scheduler(
        tiny_model,
        num_blocks=20,
        enable_prefix_caching=True,
    )
    survivor = _request("survivor", 32, token_id=7)
    scheduler.add_request(survivor)
    first = scheduler.schedule()
    scheduler.update_from_output(first, _runner_output([survivor.request_id]))

    victim = _request("victim", 32, token_id=7)
    scheduler.add_request(victim)
    second = scheduler.schedule()
    survivor_blocks = scheduler.kv_cache_manager.get_blocks(
        survivor.request_id
    ).get_block_ids()[0]
    victim_blocks = scheduler.kv_cache_manager.get_blocks(
        victim.request_id
    ).get_block_ids()[0]
    shared_blocks = set(survivor_blocks) & set(victim_blocks)
    assert shared_blocks
    scheduler.update_from_output(
        second, _runner_output([survivor.request_id, victim.request_id])
    )

    _preemption_state(scheduler).abort(victim, 0.0)

    assert (
        scheduler.kv_cache_manager.get_blocks(survivor.request_id).get_block_ids()[0]
        == survivor_blocks
    )
    assert victim.request_id not in scheduler.requests
    assert survivor in scheduler.running

    # The survivor can continue using the shared prefix after victim cleanup.
    survivor_step = scheduler.schedule()
    assert survivor.request_id in survivor_step.num_scheduled_tokens
    scheduler.update_from_output(survivor_step, _runner_output([survivor.request_id]))


def test_invalid_abort_on_preemption_config_raises(tiny_model: str) -> None:
    model_config = ModelConfig(
        model=tiny_model,
        tokenizer=tiny_model,
        dtype="float16",
        skip_tokenizer_init=True,
    )
    non_bool_config = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cpu"),
        additional_config={"abort_on_preemption": "yes"},
    )
    with pytest.raises(ValueError, match="must be a bool"):
        _should_enable_abort_on_preemption(non_bool_config)

    non_consumer_config = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cpu"),
        additional_config={"abort_on_preemption": True},
    )
    with pytest.raises(ValueError, match="requires kv_role='kv_consumer'"):
        _should_enable_abort_on_preemption(non_consumer_config)


def test_abort_rejects_delayed_block_release(tiny_model: str) -> None:
    class _DelayFreeConnector:
        def request_finished(
            self, request: Request, block_ids: list[int]
        ) -> tuple[bool, dict[str, object] | None]:
            return True, None

    scheduler = _create_scheduler(tiny_model, num_blocks=6)
    victim = _request("victim", 32)
    scheduler.add_request(victim)
    first = scheduler.schedule()
    scheduler.update_from_output(first, _runner_output([victim.request_id]))
    scheduler.connector = _DelayFreeConnector()  # type: ignore[assignment]

    with pytest.raises(
        AssertionError,
        match="Capacity preemption abort requires immediate block release",
    ):
        _preemption_state(scheduler).abort(victim, 0.0)
