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
"""DpPipelineScheduler on a real scheduler, through schedule()."""

import pytest
import torch
from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu.core.dp_pipeline_scheduler import DpPipelineScheduler
from vllm_torchtpu.distributed.dp_pipeline.plan import PLAN_KEY, PipelinePlan

BLOCK_SIZE = 16
MODEL = "facebook/opt-125m"
PLAN = PipelinePlan(1, (1, 0), (0, 6, 12), 64)


@pytest.fixture(scope="module")
def model_config():
    return ModelConfig(model=MODEL, dtype="float16", seed=42, max_model_len=2048)


def _build(model_config, *, num_blocks=64, budget=64, threshold=16, full_isl=True):
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=budget,
            max_model_len=2048,
            enable_chunked_prefill=True,
            long_prefill_token_threshold=threshold,
            is_encoder_decoder=False,
            watermark=0.0,
            async_scheduling=True,
            scheduler_reserve_full_isl=full_isl,
        ),
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=False,
        ),
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
                    head_size=64,
                    dtype=torch.float16,
                ),
            )
        ],
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vllm_config)
    return DpPipelineScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK_SIZE,
        log_stats=False,
    )


def _request(
    request_id: str, num_tokens: int, pipelined=False, seed=0, max_tokens=16
) -> Request:
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    if pipelined:
        params.extra_args = {PLAN_KEY: PLAN.to_json()}
    params.update_from_generation_config({}, 2)
    prompt = [seed * 100_000 + i for i in range(num_tokens)]
    return Request(request_id, prompt, params, None)


def test_pipelined_chunks_run_alone_with_the_whole_budget(model_config):
    scheduler = _build(model_config)
    scheduler.add_request(_request("p", 150, pipelined=True))
    scheduler.add_request(_request("late", 8, seed=1))
    # Two requests wait, yet each chunk takes the whole budget: the threshold
    # of 16 tokens would make another engine's chunks differ.
    chunks = [scheduler.schedule().num_scheduled_tokens for _ in range(3)]
    assert chunks == [{"p": 64}, {"p": 64}, {"p": 22}]
    assert [r.request_id for r in scheduler.running] == ["p"]
    assert [r.request_id for r in scheduler.waiting] == ["late"]
    # The prompt is scheduled: the next step is a regular one.
    assert scheduler.schedule().num_scheduled_tokens == {"p": 1, "late": 8}


def test_requests_routed_during_a_pipeline_wait_for_its_prompt(model_config):
    scheduler = _build(model_config)
    scheduler.add_request(_request("p", 100, pipelined=True))
    assert scheduler.schedule().num_scheduled_tokens == {"p": 64}
    scheduler.add_request(_request("late", 8, seed=1))
    assert scheduler.schedule().num_scheduled_tokens == {"p": 36}
    assert scheduler.schedule().num_scheduled_tokens == {"p": 1, "late": 8}


def test_ordinary_requests_keep_upstream_scheduling(model_config):
    scheduler = _build(model_config)
    scheduler.add_request(_request("a", 40, seed=1))
    scheduler.add_request(_request("b", 40, seed=2))
    # Upstream's long-prefill threshold applies with two requests.
    assert scheduler.schedule().num_scheduled_tokens == {"a": 16, "b": 16}


def test_a_pipelined_request_needs_an_idle_engine(model_config):
    scheduler = _build(model_config)
    scheduler.add_request(_request("a", 8, seed=1))
    scheduler.schedule()
    scheduler.add_request(_request("p", 100, pipelined=True))
    with pytest.raises(RuntimeError, match="idle engines only"):
        scheduler.schedule()


def test_a_pipelined_request_is_never_preempted(model_config):
    # Five usable blocks hold 80 tokens: admitted on its first chunk alone,
    # the prompt's second chunk does not fit.
    scheduler = _build(model_config, num_blocks=6, full_isl=False)
    scheduler.add_request(_request("p", 100, pipelined=True))
    assert scheduler.schedule().num_scheduled_tokens == {"p": 64}
    with pytest.raises(RuntimeError, match="was preempted"):
        scheduler.schedule()


def test_after_its_prompt_a_pipelined_request_is_ordinary(model_config):
    scheduler = _build(model_config)
    p = _request("p", 100, pipelined=True)
    scheduler.add_request(p)
    assert scheduler.schedule().num_scheduled_tokens == {"p": 64}
    assert scheduler.schedule().num_scheduled_tokens == {"p": 36}
    scheduler.add_request(_request("a", 40, seed=1))
    # p decodes in an ordinary step beside a.
    assert scheduler.schedule().num_scheduled_tokens == {"p": 1, "a": 16}
