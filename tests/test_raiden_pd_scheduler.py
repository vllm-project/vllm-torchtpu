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
"""P response lifetime and send-result contracts on real vLLM schedulers."""

import json
from typing import Any

import pytest
import torch
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiKVConnectorWorkerMetadata,
)
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.engine import (
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu.core.raiden_pd import _ProducerCompletion
from vllm_torchtpu.distributed.kv_transfer.raiden_pd_completion import (
    RaidenSendOutcome,
    RaidenSendResults,
)

BLOCK_SIZE = 16


class _FakeProducerConnector(KVConnectorBase_V1, SupportsHMA):
    """Minimal scheduler-side producer connector exercising real delay_free_blocks."""

    def __init__(self, *, pending: bool = True):
        self.pending = pending
        self.updated_outputs: list[KVConnectorOutput] = []

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self, request: Request, blocks: Any, num_external_tokens: int
    ) -> None:
        return None

    def build_connector_meta(self, scheduler_output: Any) -> KVConnectorMetadata:
        return KVConnectorMetadata()

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        if request.status not in (
            RequestStatus.FINISHED_LENGTH_CAPPED,
            RequestStatus.FINISHED_STOPPED,
        ):
            return False, None
        params: dict[str, Any] = {
            "req_id": request.request_id,
            "uuid": 7,
            "num_tokens": request.num_prompt_tokens,
            "src_controller_address": "127.0.0.1:27000",
        }
        if self.pending:
            params["kv_transfer_pending"] = True
        return True, params

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

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        self.updated_outputs.append(connector_output)


@pytest.fixture(scope="module")
def shared_model_config(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("opt_config")
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


def _build_producer_scheduler(
    model_config: ModelConfig,
    scheduler_cls: type[Scheduler],
    *,
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
        device_config=DeviceConfig(device="cpu"),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer0"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=32,
                    dtype=torch.float16,
                ),
            )
        ],
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vllm_config)
    scheduler = scheduler_cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK_SIZE,
    )
    scheduler.connector = _FakeProducerConnector()
    state = _ProducerCompletion(scheduler)
    init_none_hash(sha256)
    hasher = get_request_block_hasher(BLOCK_SIZE, sha256)
    return scheduler, state, hasher


def _add_and_schedule_prefill(
    scheduler: Scheduler,
    hasher,
    *,
    req_id: str = "p-request",
    client_index: int = 3,
    max_tokens: int = 1,
    stop_token_ids: list[int] | None = None,
):
    params = SamplingParams(
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids,
        detokenize=False,
    )
    req = Request(
        req_id,
        list(range(1, 33)),
        params,
        None,
        client_index=client_index,
        block_hasher=hasher,
    )
    scheduler.add_request(req)
    sched_out = scheduler.schedule()
    assert [r.req_id for r in sched_out.scheduled_new_reqs] == [req_id]
    return req, sched_out


def _runner_output_for_step(
    sched_out,
    *,
    token_by_req: dict[str, list[int]] | None = None,
    results: RaidenSendResults | None = None,
    finished_sending: set[str] | None = None,
) -> ModelRunnerOutput:
    req_ids = list(sched_out.num_scheduled_tokens.keys())
    sampled = [(token_by_req or {}).get(rid, [42]) for rid in req_ids]
    kv_output = KVConnectorOutput(
        finished_sending=finished_sending or set(),
        kv_connector_worker_meta=MultiKVConnectorWorkerMetadata(
            metadata=(results, None)
        ),
    )
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: idx for idx, rid in enumerate(req_ids)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        kv_connector_output=kv_output,
    )


def test_prefill_remains_open_and_retains_blocks_until_finished_sending(
    shared_model_config, scheduler_cls
):
    scheduler, state, hasher = _build_producer_scheduler(
        shared_model_config, scheduler_cls
    )
    free_before = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    _, sched_out = _add_and_schedule_prefill(scheduler, hasher)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() < free_before

    first_batch = scheduler.update_from_output(
        sched_out, _runner_output_for_step(sched_out)
    )
    first = first_batch[3].outputs[0]
    assert first.finish_reason is None
    assert first.stop_reason is None
    assert first.new_token_ids == [42]
    assert first.kv_transfer_params["uuid"] == 7
    assert first.kv_transfer_params["kv_transfer_pending"] is True
    # Request stays in scheduler.requests with KV blocks pinned until finished_sending.
    assert "p-request" in scheduler.requests
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() < free_before

    # Partial rank completion without finished_sending keeps blocks pinned.
    empty_sched = scheduler.schedule()
    assert (
        scheduler.update_from_output(
            empty_sched,
            _runner_output_for_step(
                empty_sched,
                results=RaidenSendResults({"p-request": RaidenSendOutcome.SUCCESS}),
            ),
        )
        == {}
    )
    assert "p-request" in scheduler.requests

    # Final finished_sending barrier frees blocks and emits the terminal response.
    empty_sched = scheduler.schedule()
    final_batch = scheduler.update_from_output(
        empty_sched,
        _runner_output_for_step(
            empty_sched,
            results=RaidenSendResults({"p-request": RaidenSendOutcome.SUCCESS}),
            finished_sending={"p-request"},
        ),
    )
    final = final_batch[3].outputs
    assert len(final) == 1
    assert final[0].new_token_ids == []
    assert final[0].finish_reason == FinishReason.LENGTH
    assert final[0].kv_transfer_params == {
        "req_id": "p-request",
        "uuid": 7,
        "num_tokens": 32,
        "src_controller_address": "127.0.0.1:27000",
        "kv_transfer_done": True,
    }
    assert state.pending_responses == {}
    assert "p-request" not in scheduler.requests
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free_before

    # A late worker send outcome after finished_sending must not emit a duplicate
    # response.
    empty_sched = scheduler.schedule()
    assert (
        scheduler.update_from_output(
            empty_sched,
            _runner_output_for_step(
                empty_sched,
                results=RaidenSendResults(
                    {"p-request": RaidenSendOutcome.SEND_FAILURE}
                ),
            ),
        )
        == {}
    )


def test_stop_finish_reason_and_stop_reason_are_preserved(
    shared_model_config, scheduler_cls
):
    scheduler, _, hasher = _build_producer_scheduler(shared_model_config, scheduler_cls)
    _, sched_out = _add_and_schedule_prefill(
        scheduler, hasher, max_tokens=4, stop_token_ids=[99]
    )
    first = scheduler.update_from_output(
        sched_out,
        _runner_output_for_step(sched_out, token_by_req={"p-request": [99]}),
    )[3].outputs[0]
    assert first.finish_reason is None
    assert first.stop_reason is None

    empty_sched = scheduler.schedule()
    final = scheduler.update_from_output(
        empty_sched,
        _runner_output_for_step(
            empty_sched,
            results=RaidenSendResults({"p-request": RaidenSendOutcome.SUCCESS}),
            finished_sending={"p-request"},
        ),
    )[3].outputs[0]
    assert final.finish_reason == FinishReason.STOP
    assert final.stop_reason == 99
    assert final.kv_transfer_params["kv_transfer_done"] is True
    assert final.kv_transfer_params["uuid"] == 7


@pytest.mark.parametrize(
    ("outcome", "status_name", "status_message"),
    [
        (
            RaidenSendOutcome.SEND_FAILURE,
            "model.KVTPrefillWriteFailure",
            "P-side failed to write KV cache data. Category: internal.",
        ),
        (
            RaidenSendOutcome.UNCLAIMED_TIMEOUT,
            "model.KVTPrefillTransferTimeout",
            "P-side timed out waiting for KV cache transfer completion.",
        ),
        (
            None,
            "model.KVTPrefillWriteFailure",
            "P-side KV cache transfer finished without a sender result. "
            "Category: internal.",
        ),
    ],
)
def test_failed_or_missing_result_has_structured_error_and_frees_blocks(
    outcome, status_name, status_message, shared_model_config, scheduler_cls
):
    scheduler, _, hasher = _build_producer_scheduler(shared_model_config, scheduler_cls)
    free_before = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    _, sched_out = _add_and_schedule_prefill(scheduler, hasher)
    scheduler.update_from_output(sched_out, _runner_output_for_step(sched_out))

    if outcome is not None:
        empty_sched = scheduler.schedule()
        scheduler.update_from_output(
            empty_sched,
            _runner_output_for_step(
                empty_sched, results=RaidenSendResults({"p-request": outcome})
            ),
        )
        empty_sched = scheduler.schedule()
        scheduler.update_from_output(
            empty_sched,
            _runner_output_for_step(
                empty_sched,
                results=RaidenSendResults({"p-request": RaidenSendOutcome.SUCCESS}),
            ),
        )
    empty_sched = scheduler.schedule()
    final = scheduler.update_from_output(
        empty_sched,
        _runner_output_for_step(empty_sched, finished_sending={"p-request"}),
    )[3].outputs[0]
    assert final.finish_reason == FinishReason.ERROR
    assert final.stop_reason is None
    assert final.kv_transfer_params == {
        "kv_transfer_error": {
            "status_code": 500,
            "status_name": status_name,
            "status_message": status_message,
        }
    }
    assert "p-request" not in scheduler.requests
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free_before


def test_rank_result_aggregation_preserves_failure_in_both_orders():
    for values in [
        (RaidenSendOutcome.SUCCESS, RaidenSendOutcome.SEND_FAILURE),
        (RaidenSendOutcome.SEND_FAILURE, RaidenSendOutcome.SUCCESS),
    ]:
        result = MultiKVConnectorWorkerMetadata(
            metadata=(RaidenSendResults({"p-request": values[0]}), None)
        )
        result = result.aggregate(
            MultiKVConnectorWorkerMetadata(
                metadata=(RaidenSendResults({"p-request": values[1]}), None)
            )
        )
        assert result.metadata[0].outcomes == {
            "p-request": RaidenSendOutcome.SEND_FAILURE
        }


def test_rank_result_aggregation_prefers_failure_over_timeout():
    result = RaidenSendResults({"p-request": RaidenSendOutcome.UNCLAIMED_TIMEOUT})
    result.aggregate(RaidenSendResults({"p-request": RaidenSendOutcome.SEND_FAILURE}))
    assert result.outcomes == {"p-request": RaidenSendOutcome.SEND_FAILURE}


def test_cancelled_response_suppresses_late_pkg2_but_waits_for_send_barrier_to_free(
    shared_model_config, scheduler_cls
):
    scheduler, state, hasher = _build_producer_scheduler(
        shared_model_config, scheduler_cls
    )
    free_before = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    _, sched_out = _add_and_schedule_prefill(scheduler, hasher)
    scheduler.update_from_output(sched_out, _runner_output_for_step(sched_out))

    # Client aborts after prefill finished while KV transfer is still in flight.
    scheduler.finish_requests("p-request", RequestStatus.FINISHED_ABORTED)
    assert state.pending_responses == {}
    assert "p-request" in scheduler.requests
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() < free_before

    empty_sched = scheduler.schedule()
    assert (
        scheduler.update_from_output(
            empty_sched,
            _runner_output_for_step(
                empty_sched,
                results=RaidenSendResults({"p-request": RaidenSendOutcome.SUCCESS}),
                finished_sending={"p-request"},
            ),
        )
        == {}
    )
    assert "p-request" not in scheduler.requests
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free_before


def test_prefill_and_send_completion_in_same_step(shared_model_config, scheduler_cls):
    scheduler, state, hasher = _build_producer_scheduler(
        shared_model_config, scheduler_cls
    )
    _, sched_out = _add_and_schedule_prefill(scheduler, hasher)
    outputs = scheduler.update_from_output(
        sched_out,
        _runner_output_for_step(
            sched_out,
            results=RaidenSendResults({"p-request": RaidenSendOutcome.SUCCESS}),
            finished_sending={"p-request"},
        ),
    )[3].outputs
    assert len(outputs) == 2
    assert outputs[0].new_token_ids == [42]
    assert outputs[0].finish_reason is None
    assert outputs[1].new_token_ids == []
    assert outputs[1].finish_reason == FinishReason.LENGTH
    assert outputs[1].kv_transfer_params == {
        "req_id": "p-request",
        "uuid": 7,
        "num_tokens": 32,
        "src_controller_address": "127.0.0.1:27000",
        "kv_transfer_done": True,
    }
    assert state.pending_responses == {}


@pytest.mark.parametrize(
    "output_kind", [RequestOutputKind.DELTA, RequestOutputKind.FINAL_ONLY]
)
def test_output_processor_preserves_transfer_params_and_finish_reason(
    output_kind, shared_model_config, scheduler_cls
):
    processor = OutputProcessor(tokenizer=None, log_stats=False)
    processor.add_request(
        EngineCoreRequest(
            request_id="p-request",
            external_req_id="client-request",
            prompt_token_ids=list(range(1, 33)),
            mm_features=None,
            sampling_params=SamplingParams(
                max_tokens=1, detokenize=False, output_kind=output_kind
            ),
            pooling_params=None,
            arrival_time=0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        ),
        prompt=None,
    )
    scheduler, _, hasher = _build_producer_scheduler(shared_model_config, scheduler_cls)
    _, sched_out = _add_and_schedule_prefill(scheduler, hasher)
    first_core_outputs = scheduler.update_from_output(
        sched_out, _runner_output_for_step(sched_out)
    )[3].outputs
    first_processed = processor.process_outputs(
        first_core_outputs, engine_core_timestamp=0
    ).request_outputs
    if output_kind is RequestOutputKind.DELTA:
        assert len(first_processed) == 1
        assert not first_processed[0].finished
        assert first_processed[0].outputs[0].token_ids == [42]
        assert first_processed[0].kv_transfer_params["uuid"] == 7
        assert first_processed[0].kv_transfer_params["kv_transfer_pending"] is True
    else:
        assert first_processed == []
    assert processor.has_unfinished_requests()

    empty_sched = scheduler.schedule()
    final_core_outputs = scheduler.update_from_output(
        empty_sched,
        _runner_output_for_step(
            empty_sched,
            results=RaidenSendResults({"p-request": RaidenSendOutcome.SUCCESS}),
            finished_sending={"p-request"},
        ),
    )[3].outputs
    final = processor.process_outputs(
        final_core_outputs, engine_core_timestamp=0
    ).request_outputs[0]
    assert final.request_id == "client-request"
    assert final.finished
    assert final.outputs[0].finish_reason == "length"
    expected_tokens = [] if output_kind is RequestOutputKind.DELTA else [42]
    assert list(final.outputs[0].token_ids) == expected_tokens
    assert final.kv_transfer_params == {
        "req_id": "p-request",
        "uuid": 7,
        "num_tokens": 32,
        "src_controller_address": "127.0.0.1:27000",
        "kv_transfer_done": True,
    }
    assert not processor.has_unfinished_requests()
