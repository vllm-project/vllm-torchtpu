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
"""Unit tests for PrefillFirstScheduler solo whole-prompt prefill scheduling."""

import random
from collections import deque

import pytest
import torch
from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.core.experimental.prefill_first_scheduler import (
    PrefillFirstScheduler,
)

BLOCK_SIZE = 16
MODEL = "facebook/opt-125m"


@pytest.fixture(scope="module")
def shared_model_config():
    return ModelConfig(model=MODEL, dtype="float16", seed=42, max_model_len=2048)


def _build(
    model_config: ModelConfig,
    monkeypatch,
    *,
    over_tokens: int = 0,
    num_blocks: int = 64,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 128,
    long_prefill_token_threshold: int = 0,
    data_parallel_size: int = 1,
    log_stats: bool = False,
    policy: str = "fcfs",
):
    monkeypatch.setenv("VLLM_TPU_SOLO_PREFILL_OVER_TOKENS", str(over_tokens))
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=2048,
            enable_chunked_prefill=True,
            long_prefill_token_threshold=long_prefill_token_threshold,
            is_encoder_decoder=False,
            watermark=0.0,
            async_scheduling=True,
            policy=policy,
        ),
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=True,
        ),
        parallel_config=ParallelConfig(
            data_parallel_size=data_parallel_size,
            data_parallel_size_local=data_parallel_size,
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
    scheduler = PrefillFirstScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK_SIZE,
        log_stats=log_stats,
    )
    init_none_hash(sha256)
    return scheduler, get_request_block_hasher(BLOCK_SIZE, sha256)


def _request(
    request_id: str,
    num_tokens: int,
    hasher,
    seed: int = 0,
    prefix: list[int] | None = None,
) -> Request:
    params = SamplingParams(max_tokens=16, ignore_eos=True)
    params.update_from_generation_config({}, 2)
    prompt = list(prefix or [])
    prompt += [seed * 100_000 + i for i in range(num_tokens - len(prompt))]
    return Request(request_id, prompt, params, None, block_hasher=hasher)


def _add(scheduler, hasher, *specs: tuple[str, int, int]) -> None:
    for request_id, num_tokens, seed in specs:
        scheduler.add_request(_request(request_id, num_tokens, hasher, seed))


def _ids(scheduler) -> list[str]:
    return [r.request_id for r in scheduler.running]


def test_long_prompt_runs_whole_and_alone(shared_model_config, monkeypatch):
    scheduler, hasher = _build(shared_model_config, monkeypatch, over_tokens=32)
    _add(scheduler, hasher, ("a", 16, 1), ("b", 16, 2))
    assert scheduler.schedule().num_scheduled_tokens == {"a": 16, "b": 16}

    _add(scheduler, hasher, ("long", 100, 3))
    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {"long": 100}
    assert [r.req_id for r in out.scheduled_new_reqs] == ["long"]
    assert _ids(scheduler) == ["a", "b", "long"]
    assert scheduler.max_num_running_reqs == 8
    assert scheduler.get_request_counts() == (3, 0)
    assert scheduler.get_num_unfinished_requests() == 3

    assert scheduler.schedule().num_scheduled_tokens == {"a": 1, "b": 1, "long": 1}


def test_regular_steps_carry_at_most_threshold(shared_model_config, monkeypatch):
    scheduler, hasher = _build(shared_model_config, monkeypatch, over_tokens=32)
    _add(scheduler, hasher, ("s1", 24, 1), ("s2", 24, 2), ("s3", 24, 3))
    totals = [scheduler.schedule().total_num_scheduled_tokens for _ in range(4)]
    assert totals[0] == 32
    assert max(totals) <= 32


def test_regular_step_holds_requests_behind_waiting_solo_prompt(
    shared_model_config, monkeypatch
):
    # Ten usable blocks: running `s1` (1 block) prevents 150-token `L1` (10
    # blocks) from fitting yet; `s2` ahead of `L1` is admitted, while `s3`
    # behind `L1` stays behind it in FCFS order.
    scheduler, hasher = _build(
        shared_model_config,
        monkeypatch,
        over_tokens=32,
        num_blocks=11,
        max_num_batched_tokens=256,
    )
    _add(scheduler, hasher, ("s1", 16, 1))
    scheduler.schedule()
    _add(scheduler, hasher, ("s2", 16, 2), ("L1", 150, 3), ("s3", 16, 4))

    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {"s1": 1, "s2": 16}
    assert [r.request_id for r in scheduler.waiting] == ["L1", "s3"]
    assert scheduler.get_request_counts() == (2, 2)


def test_blocked_kv_holding_request_does_not_stop_solo_steps(
    shared_model_config, monkeypatch
):
    scheduler, hasher = _build(shared_model_config, monkeypatch, over_tokens=32)
    session = _request("session", 16, hasher, seed=1)
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session.num_computed_tokens = 16
    scheduler.kv_holding_waiting.add_request(session)
    _add(scheduler, hasher, ("long", 100, 2))

    assert scheduler.schedule().num_scheduled_tokens == {"long": 100}
    assert list(scheduler.kv_holding_waiting) == [session]


def test_solo_step_skips_prefix_cache_only_for_that_step(
    shared_model_config, monkeypatch
):
    scheduler, hasher = _build(shared_model_config, monkeypatch, over_tokens=32)
    prefix = [900_000 + i for i in range(32)]
    scheduler.add_request(_request("primer", 32, hasher, seed=1, prefix=prefix))
    scheduler.schedule()
    scheduler.schedule()
    scheduler.finish_requests("primer", RequestStatus.FINISHED_ABORTED)
    scheduler.schedule()

    long_request = _request("long", 100, hasher, seed=2, prefix=prefix)
    assert scheduler.kv_cache_manager.get_computed_blocks(long_request)[1] == 32
    scheduler.add_request(long_request)
    assert scheduler.schedule().num_scheduled_tokens == {"long": 100}
    assert long_request.skip_reading_prefix_cache is False


def test_waits_for_kv_blocks_before_admitting_solo_step(
    shared_model_config, monkeypatch
):
    scheduler, hasher = _build(
        shared_model_config,
        monkeypatch,
        over_tokens=32,
        num_blocks=11,
        max_num_batched_tokens=256,
    )
    _add(scheduler, hasher, ("a", 16, 1))
    scheduler.schedule()
    _add(scheduler, hasher, ("long", 150, 2))
    assert scheduler.schedule().num_scheduled_tokens == {"a": 1}
    assert [r.request_id for r in scheduler.waiting] == ["long"]

    scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    assert scheduler.schedule().num_scheduled_tokens == {"long": 150}


def test_true_counts_preserved_when_paused_and_logged(shared_model_config, monkeypatch):
    scheduler, hasher = _build(
        shared_model_config,
        monkeypatch,
        over_tokens=32,
        data_parallel_size=2,
        log_stats=True,
    )
    _add(scheduler, hasher, ("queued", 100, 1))
    stats = scheduler.make_stats()
    assert (stats.num_running_reqs, stats.num_waiting_reqs) == (0, 1)
    assert scheduler.get_request_counts() == (0, 1)

    scheduler.set_pause_state(PauseState.PAUSED_NEW)
    assert scheduler.get_num_unfinished_requests() == 0
    assert not scheduler.has_requests()


def test_rejects_threshold_at_or_above_step_budget(shared_model_config, monkeypatch):
    with pytest.raises(ValueError, match="step budget"):
        _build(shared_model_config, monkeypatch, over_tokens=128)


def test_rejects_incompatible_long_prefill_threshold(shared_model_config, monkeypatch):
    with pytest.raises(ValueError, match="long_prefill_token_threshold"):
        _build(
            shared_model_config,
            monkeypatch,
            over_tokens=32,
            long_prefill_token_threshold=64,
        )


def test_requires_positive_solo_over_tokens(shared_model_config, monkeypatch):
    with pytest.raises(ValueError, match="VLLM_TPU_SOLO_PREFILL_OVER_TOKENS > 0"):
        _build(shared_model_config, monkeypatch, over_tokens=0)


def test_flag_is_excluded_from_compile_key():
    name = "VLLM_TPU_SOLO_PREFILL_OVER_TOKENS"
    assert name in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    assert name not in tpu_compiler._tpu_compile_env_factors()


def test_resumed_requests_carry_only_output_tokens(shared_model_config, monkeypatch):
    scheduler, hasher = _build(shared_model_config, monkeypatch, over_tokens=32)
    _add(scheduler, hasher, ("a", 16, 1))
    scheduler.schedule()
    _add(scheduler, hasher, ("long", 100, 2))
    scheduler.schedule()  # Solo step: `a` sits this step out.

    out = scheduler.schedule()
    cached = out.scheduled_cached_reqs
    num_output_tokens = cached.num_output_tokens[cached.req_ids.index("a")]
    request = scheduler.requests["a"]
    assert num_output_tokens > 0
    assert cached.all_token_ids["a"] == list(request.all_token_ids)[-num_output_tokens:]


def test_random_async_workload_preserves_step_invariants_and_drains(
    shared_model_config, monkeypatch
):
    rng = random.Random(0)
    scheduler, hasher = _build(
        shared_model_config,
        monkeypatch,
        over_tokens=32,
        num_blocks=64,
        max_num_seqs=6,
        max_num_batched_tokens=256,
    )
    in_flight: deque = deque()
    num_added = 0
    for step in range(300):
        if step < 150 and rng.random() < 0.2:
            request = _request(
                f"r{num_added}", rng.choice([8, 20, 40, 90, 150]), hasher, num_added
            )
            request.max_tokens = rng.randint(1, 12)
            scheduler.add_request(request)
            num_added += 1
        fresh = {r.request_id for r in scheduler.waiting if not r.num_computed_tokens}
        out = scheduler.schedule()
        scheduled = out.num_scheduled_tokens
        solo = [r for r, n in scheduled.items() if n > 32]
        if solo:
            assert len(scheduled) == 1
            assert solo[0] in fresh
            assert scheduled[solo[0]] == scheduler.requests[solo[0]].num_tokens
        else:
            assert out.total_num_scheduled_tokens <= 32
        req_ids = list(scheduled)
        in_flight.append(
            (
                out,
                ModelRunnerOutput(
                    req_ids=req_ids,
                    req_id_to_index={r: i for i, r in enumerate(req_ids)},
                    sampled_token_ids=[
                        [] if scheduler.requests[r].is_prefill_chunk else [7]
                        for r in req_ids
                    ],
                    logprobs=None,
                    prompt_logprobs_dict={},
                    pooler_output=[],
                ),
            )
        )
        if len(in_flight) == 2 or not scheduled:
            scheduler.update_from_output(*in_flight.popleft())
    while in_flight:
        scheduler.update_from_output(*in_flight.popleft())
    assert num_added > 20
    assert scheduler.get_num_unfinished_requests() == 0
