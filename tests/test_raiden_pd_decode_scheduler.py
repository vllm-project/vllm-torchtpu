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
"""Decode-side Raiden KV load failure handling on real vLLM schedulers."""

import json
from typing import Any

import pytest
import torch
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    SupportsHMA,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.engine import FinishReason
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

BLOCK_SIZE = 16


class _FakeConsumerConnector(KVConnectorBase_V1, SupportsHMA):
    """Minimal consumer connector placing admissions into WAITING_FOR_REMOTE_KVS."""

    def __init__(self):
        self.consumed: set[str] = set()

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        if request.request_id in self.consumed:
            return 0, False
        return BLOCK_SIZE, True

    def update_state_after_alloc(
        self, request: Request, blocks: Any, num_external_tokens: int
    ) -> None:
        if num_external_tokens > 0:
            self.consumed.add(request.request_id)

    def get_loaded_kv_cache_group_ids(self, request: Request) -> tuple[int, ...]:
        return (0,)

    def build_connector_meta(self, scheduler_output: Any) -> KVConnectorMetadata:
        return KVConnectorMetadata()

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        return None

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        return None

    def wait_for_save(self) -> None:
        return None

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None


@pytest.fixture(scope="module")
def shared_model_config(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("opt_decode_config")
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["OPTForCausalLM"],
                "model_type": "opt",
                "hidden_size": 64,
                "num_attention_heads": 2,
                "num_hidden_layers": 2,
                "ffn_dim": 128,
                "max_position_embeddings": 2048,
                "vocab_size": 1000,
            }
        )
    )
    return ModelConfig(
        model=str(model_dir),
        tokenizer=str(model_dir),
        skip_tokenizer_init=True,
        dtype="float16",
        seed=42,
        max_model_len=2048,
    )


@pytest.fixture(params=[Scheduler, AsyncScheduler])
def scheduler_cls(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["single_group", "hybrid_multi_group"])
def hybrid_cache(request):
    return request.param


def _build_decode_scheduler(
    model_config: ModelConfig,
    scheduler_cls: type[Scheduler],
    *,
    policy: str = "fail",
    hybrid: bool = False,
    num_blocks: int = 16,
):
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=256,
            max_model_len=2048,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
            async_scheduling=scheduler_cls is AsyncScheduler,
        ),
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=False,
        ),
        parallel_config=ParallelConfig(),
        kv_transfer_config=KVTransferConfig(
            kv_connector="TPURaidenConnector",
            kv_role="kv_consumer",
            kv_load_failure_policy=policy,
        ),
        device_config=DeviceConfig(device="cpu"),
    )
    groups = [
        KVCacheGroupSpec(
            ["layer0"],
            FullAttentionSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=1,
                head_size=32,
                dtype=torch.float16,
            ),
        )
    ]
    if hybrid:
        groups.append(
            KVCacheGroupSpec(
                ["layer1"],
                MambaSpec(
                    block_size=BLOCK_SIZE,
                    shapes=((2, 16),),
                    dtypes=(torch.float16,),
                    mamba_cache_mode="all",
                ),
            )
        )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vllm_config)
    monkey_init = getattr(Scheduler.__init__, "__wrapped__", Scheduler.__init__)
    scheduler = scheduler_cls.__new__(scheduler_cls)
    monkey_init(
        scheduler,
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK_SIZE,
    )
    scheduler.connector = _FakeConsumerConnector()
    init_none_hash(sha256)
    hasher = get_request_block_hasher(BLOCK_SIZE, sha256)
    return scheduler, hasher


def _admit_remote_kv_requests(scheduler: Scheduler, hasher, *req_ids: str):
    for req_id in req_ids:
        req = Request(
            req_id,
            list(range(1, 20)),
            SamplingParams(max_tokens=4, detokenize=False),
            None,
            block_hasher=hasher,
        )
        scheduler.add_request(req)
    sched_out = scheduler.schedule()
    assert sched_out.total_num_scheduled_tokens == 0
    for req_id in req_ids:
        assert scheduler.requests[req_id].status == RequestStatus.WAITING_FOR_REMOTE_KVS
    return sched_out


def _no_forward_runner_output(kv_output: KVConnectorOutput) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        kv_connector_output=kv_output,
    )


def test_failed_recving_under_fail_policy_terminates_and_frees_blocks_even_for_hybrid(
    shared_model_config, scheduler_cls, hybrid_cache
):
    scheduler, hasher = _build_decode_scheduler(
        shared_model_config, scheduler_cls, policy="fail", hybrid=hybrid_cache
    )
    free_before = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    sched_out = _admit_remote_kv_requests(scheduler, hasher, "bad", "good")
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() < free_before

    # Aggregate across two TP workers where only rank 1 observed the load error.
    aggregator = KVOutputAggregator(2)
    r0 = _no_forward_runner_output(
        KVConnectorOutput(
            finished_recving={"bad", "good"},
            failed_recving=set(),
        )
    )
    r1 = _no_forward_runner_output(
        KVConnectorOutput(
            finished_recving={"bad", "good"},
            failed_recving={"bad"},
        )
    )
    aggregated = aggregator.aggregate([r0, r1], output_rank=0)
    assert aggregated.kv_connector_output.finished_recving == {"bad", "good"}
    assert aggregated.kv_connector_output.failed_recving == {"bad"}

    outputs = scheduler.update_from_output(sched_out, aggregated)
    (terminal,) = outputs[0].outputs
    assert terminal.request_id == "bad"
    assert terminal.new_token_ids == []
    assert terminal.finish_reason is FinishReason.ERROR
    assert "bad" not in scheduler.requests
    assert "good" in scheduler.requests
    assert scheduler.finished_recving_kv_req_ids == {"good"}

    # Next schedule() step promotes "good" from WAITING_FOR_REMOTE_KVS to RUNNING.
    next_sched = scheduler.schedule()
    assert [r.req_id for r in next_sched.scheduled_new_reqs] == ["good"]
    assert scheduler.requests["good"].status == RequestStatus.RUNNING


def test_failed_recving_under_recompute_policy_reschedules_local_prefill(
    shared_model_config, scheduler_cls, hybrid_cache
):
    scheduler, hasher = _build_decode_scheduler(
        shared_model_config, scheduler_cls, policy="recompute", hybrid=hybrid_cache
    )
    sched_out = _admit_remote_kv_requests(scheduler, hasher, "retry-req")
    assert scheduler.requests["retry-req"].num_computed_tokens == BLOCK_SIZE

    outputs = scheduler.update_from_output(
        sched_out,
        _no_forward_runner_output(
            KVConnectorOutput(
                finished_recving={"retry-req"},
                failed_recving={"retry-req"},
            )
        ),
    )
    assert outputs == {}
    assert scheduler.requests["retry-req"].num_computed_tokens == 0
    assert "retry-req" in scheduler.failed_recving_kv_req_ids
    assert "retry-req" in scheduler.finished_recving_kv_req_ids

    # Next schedule() step frees the unwritten remote blocks and recomputes locally.
    next_sched = scheduler.schedule()
    assert [r.req_id for r in next_sched.scheduled_new_reqs] == ["retry-req"]
    assert next_sched.num_scheduled_tokens["retry-req"] == 19
    assert scheduler.failed_recving_kv_req_ids == set()
    assert scheduler.finished_recving_kv_req_ids == set()


def test_cancelled_load_failure_releases_buffers_without_duplicate_error(
    shared_model_config, scheduler_cls, hybrid_cache
):
    scheduler, hasher = _build_decode_scheduler(
        shared_model_config, scheduler_cls, policy="fail", hybrid=hybrid_cache
    )
    free_before = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    sched_out = _admit_remote_kv_requests(scheduler, hasher, "bad")
    scheduler.finish_requests("bad", RequestStatus.FINISHED_ABORTED)
    # Blocks remain pinned until finished_recving arrives.
    assert "bad" in scheduler.requests
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() < free_before

    outputs = scheduler.update_from_output(
        sched_out,
        _no_forward_runner_output(
            KVConnectorOutput(
                finished_recving={"bad"},
                failed_recving={"bad"},
            )
        ),
    )
    assert outputs == {}
    assert "bad" not in scheduler.requests
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free_before
