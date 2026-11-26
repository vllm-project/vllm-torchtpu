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
"""Routed experts of blocks hashed in a request's last step are kept.

When a request finishes, vLLM's routed-experts connector keeps the routing of
every block still in the KV cache, so that later prefix hits on those blocks
can read it. `_patch_vllm_same_step_prefix_hits` must not hide the blocks the
last step hashed from that check: later steps hit them. These tests drive
vLLM's real `Scheduler` and `AsyncScheduler` on CPU, without a model.
"""

from collections import deque

import numpy as np
import pytest
import torch
from transformers import LlamaConfig
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.aux_output_connector.connector import AuxRequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from tests.platforms.same_step_prefix_hits_test_utils import apply_patch

BLOCK_SIZE = 16
MAX_MODEL_LEN = 256
EOS_TOKEN_ID = 127


@pytest.fixture(autouse=True)
def patched(monkeypatch):
    apply_patch(monkeypatch)


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
        max_position_embeddings=MAX_MODEL_LEN,
    ).save_pretrained(model_dir)
    init_none_hash(sha256)
    return str(model_dir)


def _create_scheduler(model: str, scheduler_cls: type[Scheduler]) -> Scheduler:
    model_config = ModelConfig(
        model=model,
        tokenizer=model,
        dtype="float16",
        skip_tokenizer_init=True,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=8,
        max_num_batched_tokens=MAX_MODEL_LEN,
        max_model_len=MAX_MODEL_LEN,
        enable_chunked_prefill=True,
        is_encoder_decoder=False,
        disable_hybrid_kv_cache_manager=True,
        watermark=0.0,
        async_scheduling=scheduler_cls is AsyncScheduler,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=True,
    )
    cache_config.num_gpu_blocks = 32
    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    # Enabled after construction because config validation requires an MoE
    # model, and the scheduler side does not depend on the model.
    vllm_config.aux_output_config.enable_return_routed_experts = True
    register_all_kvcache_specs(vllm_config)
    kv_cache_config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    scheduler = scheduler_cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=False,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )
    return scheduler


def _request(request_id: str, num_prompt_tokens: int, max_tokens: int) -> Request:
    sampling_params = SamplingParams(ignore_eos=True, max_tokens=max_tokens)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id,
        list(range(1, num_prompt_tokens + 1)),
        sampling_params,
        None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _runner_output(scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
    request_ids = list(scheduler_output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=request_ids,
        req_id_to_index={request_id: i for i, request_id in enumerate(request_ids)},
        sampled_token_ids=[[42] for _ in request_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        # These tests check which blocks the scheduler keeps, not the routing
        # values, so every request gets zero rows from its first position on.
        aux_output_connector_output={
            request_id: AuxRequestOutput(
                0, np.zeros((MAX_MODEL_LEN, 1, 1), dtype=np.int16)
            )
            for request_id in request_ids
        },
    )


def _retained_hashes(scheduler_output: SchedulerOutput) -> list[bytes]:
    metadata = scheduler_output.aux_output_connector_metadata
    assert metadata is not None
    packed = metadata.retained_block_hashes
    return [] if packed is None else list(packed)


def _run_to_completion(scheduler: Scheduler, request: Request) -> list[bytes]:
    """Run one request to its end and return the block hashes kept for it.

    The async engine schedules step t + 1 before it processes step t's output,
    so `AsyncScheduler` keeps two steps in flight.
    """
    scheduler.add_request(request)
    max_in_flight = 2 if isinstance(scheduler, AsyncScheduler) else 1
    in_flight: deque[SchedulerOutput] = deque()
    retained: list[bytes] = []
    for _ in range(MAX_MODEL_LEN):
        if not (scheduler.has_unfinished_requests() or in_flight):
            break
        if scheduler.has_unfinished_requests():
            scheduler_output = scheduler.schedule()
            retained += _retained_hashes(scheduler_output)
            in_flight.append(scheduler_output)
        if len(in_flight) == max_in_flight or not scheduler.has_unfinished_requests():
            scheduler_output = in_flight.popleft()
            scheduler.update_from_output(
                scheduler_output, _runner_output(scheduler_output)
            )
    assert not scheduler.has_unfinished_requests()
    # The connector sends the hashes kept at a finish with the next step.
    retained += _retained_hashes(scheduler.schedule())
    return retained


@pytest.mark.parametrize(
    ("scheduler_cls", "num_prompt_tokens", "max_tokens", "num_kept_blocks"),
    [
        # The prompt's only full block is hashed in the request's only step.
        (Scheduler, BLOCK_SIZE + 8, 1, 1),
        # The last step's input token completes the second block.
        (Scheduler, 2 * BLOCK_SIZE - 1, 2, 2),
        # The output of the second-to-last step completes the second block.
        # The async scheduler hashes that block when the last step's output
        # arrives, in the same call that finishes the request.
        (AsyncScheduler, BLOCK_SIZE + 8, 9, 2),
    ],
    ids=["sync-prefill-only", "sync-last-decode", "async-last-decode"],
)
def test_finished_request_keeps_blocks_its_last_step_hashed(
    tiny_model: str,
    scheduler_cls: type[Scheduler],
    num_prompt_tokens: int,
    max_tokens: int,
    num_kept_blocks: int,
) -> None:
    scheduler = _create_scheduler(tiny_model, scheduler_cls)
    request = _request("finishing", num_prompt_tokens, max_tokens)

    retained = _run_to_completion(scheduler, request)

    assert request.num_computed_tokens // BLOCK_SIZE == num_kept_blocks
    assert retained == request.block_hashes[:num_kept_blocks]


def test_same_step_request_still_recomputes_shared_prefix(tiny_model: str) -> None:
    # The connector's lookup sees blocks hashed this step, but admission still
    # must not hit a block that the same step is still filling.
    scheduler = _create_scheduler(tiny_model, Scheduler)
    filler = _request("filler", BLOCK_SIZE + 8, 1)
    same_step = _request("same_step", BLOCK_SIZE + 8, 1)
    scheduler.add_request(filler)
    scheduler.add_request(same_step)

    scheduler_output = scheduler.schedule()

    assert scheduler_output.num_scheduled_tokens == {
        "filler": BLOCK_SIZE + 8,
        "same_step": BLOCK_SIZE + 8,
    }
