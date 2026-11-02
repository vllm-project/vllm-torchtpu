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
"""PrefillFirstScheduler tests on a real scheduler, through schedule()."""

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
    max_consecutive: int = 0,
    decode_steps: int = 1,
    num_blocks: int = 64,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 128,
    long_prefill_token_threshold: int = 0,
    policy: str = "fcfs",
):
    monkeypatch.setenv("VLLM_TPU_SOLO_PREFILL_OVER_TOKENS", str(over_tokens))
    monkeypatch.setenv("VLLM_TPU_SOLO_PREFILL_MAX_CONSECUTIVE", str(max_consecutive))
    monkeypatch.setenv("VLLM_TPU_SOLO_PREFILL_DECODE_STEPS", str(decode_steps))
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

    assert scheduler.schedule().num_scheduled_tokens == {"a": 1, "b": 1, "long": 1}


def test_regular_steps_carry_at_most_the_threshold(shared_model_config, monkeypatch):
    scheduler, hasher = _build(shared_model_config, monkeypatch, over_tokens=32)
    _add(scheduler, hasher, ("s1", 24, 1), ("s2", 24, 2), ("s3", 24, 3))
    totals = [scheduler.schedule().total_num_scheduled_tokens for _ in range(4)]
    assert totals[0] == 32
    assert max(totals) <= 32


@pytest.mark.parametrize(
    ("max_consecutive", "decode_steps", "expected"),
    [
        # No limit: the long prompts back to back.
        (0, 1, ["L1", "L2", "L3", "regular"]),
        # One solo step, then one regular step.
        (1, 1, ["L1", "regular", "L2", "regular"]),
        # Two solo steps, then two regular steps.
        (2, 2, ["L1", "L2", "regular", "regular", "L3", "regular"]),
    ],
)
def test_solo_steps_interleave_with_decode_steps(
    shared_model_config, monkeypatch, max_consecutive, decode_steps, expected
):
    scheduler, hasher = _build(
        shared_model_config,
        monkeypatch,
        over_tokens=32,
        max_consecutive=max_consecutive,
        decode_steps=decode_steps,
    )
    _add(scheduler, hasher, ("a", 16, 1))
    scheduler.schedule()
    _add(scheduler, hasher, ("L1", 100, 2), ("L2", 100, 3), ("L3", 100, 4))

    seen = []
    for _ in expected:
        scheduled = scheduler.schedule().num_scheduled_tokens
        solo = [r for r in scheduled if r.startswith("L") and scheduled[r] == 100]
        if solo:
            assert len(scheduled) == 1
            seen.append(solo[0])
        else:
            # A long prompt is never admitted in pieces.
            assert all(scheduled[r] == 1 for r in scheduled)
            seen.append("regular")
    assert seen == expected


def test_regular_step_admits_only_requests_ahead_of_a_waiting_solo_prompt(
    shared_model_config, monkeypatch
):
    scheduler, hasher = _build(
        shared_model_config, monkeypatch, over_tokens=32, max_consecutive=1
    )
    _add(scheduler, hasher, ("s1", 16, 1))
    scheduler.schedule()
    _add(scheduler, hasher, ("L1", 100, 2), ("s2", 16, 3), ("L2", 100, 4))
    _add(scheduler, hasher, ("s3", 16, 5))
    assert scheduler.schedule().num_scheduled_tokens == {"L1": 100}

    # `s2` was ahead of `L2` and is admitted; `s3` waits behind `L2`.
    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {"s1": 1, "L1": 1, "s2": 16}
    assert [r.request_id for r in scheduler.waiting] == ["L2", "s3"]
    assert scheduler.schedule().num_scheduled_tokens == {"L2": 100}


def test_blocked_kv_holding_request_does_not_stop_solo_steps(
    shared_model_config, monkeypatch
):
    scheduler, hasher = _build(shared_model_config, monkeypatch, over_tokens=32)
    # A streaming session waiting for its next input holds KV blocks.
    session = _request("session", 16, hasher, seed=1)
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session.num_computed_tokens = 16
    scheduler.kv_holding_waiting.add_request(session)
    _add(scheduler, hasher, ("long", 100, 2))

    assert scheduler.schedule().num_scheduled_tokens == {"long": 100}
    assert list(scheduler.kv_holding_waiting) == [session]


def test_solo_step_skips_the_prefix_cache(shared_model_config, monkeypatch):
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
    # Only the solo step skipped the read.
    assert long_request.skip_reading_prefix_cache is False


def test_no_solo_step_until_the_prompt_fits(shared_model_config, monkeypatch):
    # Ten usable blocks (one is the null block); the running request's block
    # keeps the 150-token prompt's ten from fitting until it finishes.
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


def test_rejects_a_threshold_at_the_step_budget(shared_model_config, monkeypatch):
    with pytest.raises(ValueError, match="step budget"):
        _build(shared_model_config, monkeypatch, over_tokens=128)


def test_rejects_a_long_prefill_threshold_that_splits_solo_prompts(
    shared_model_config, monkeypatch
):
    with pytest.raises(ValueError, match="long_prefill_token_threshold"):
        _build(
            shared_model_config,
            monkeypatch,
            over_tokens=32,
            long_prefill_token_threshold=64,
        )


def test_flags_stay_out_of_the_compile_key():
    for name in (
        "VLLM_TPU_SOLO_PREFILL_OVER_TOKENS",
        "VLLM_TPU_SOLO_PREFILL_MAX_CONSECUTIVE",
        "VLLM_TPU_SOLO_PREFILL_DECODE_STEPS",
    ):
        assert name in tpu_compiler._TPU_COMPILE_ENV_IGNORED
        assert name not in tpu_compiler._tpu_compile_env_factors()


def test_resent_requests_carry_only_their_output_tokens(
    shared_model_config, monkeypatch
):
    scheduler, hasher = _build(shared_model_config, monkeypatch, over_tokens=32)
    _add(scheduler, hasher, ("a", 16, 1))
    scheduler.schedule()
    _add(scheduler, hasher, ("long", 100, 2))
    scheduler.schedule()  # Solo: `a` sits this step out.

    out = scheduler.schedule()
    cached = out.scheduled_cached_reqs
    num_output_tokens = cached.num_output_tokens[cached.req_ids.index("a")]
    request = scheduler.requests["a"]
    assert num_output_tokens > 0
    assert cached.all_token_ids["a"] == list(request.all_token_ids)[-num_output_tokens:]


@pytest.mark.parametrize(("max_consecutive", "num_blocks"), [(0, 64), (1, 40)])
def test_random_workload_keeps_step_shapes_and_finishes(
    shared_model_config, monkeypatch, max_consecutive, num_blocks
):
    """Async steps (one in flight) over random arrivals: every step is solo or
    within the threshold, and every request finishes."""
    rng = random.Random(0)
    over_tokens = 32
    scheduler, hasher = _build(
        shared_model_config,
        monkeypatch,
        over_tokens=over_tokens,
        max_consecutive=max_consecutive,
        num_blocks=num_blocks,
        max_num_seqs=6,
        max_num_batched_tokens=256,
    )
    in_flight: deque = deque()
    num_added = 0
    for step in range(400):
        if step < 200 and rng.random() < 0.15:
            request = _request(
                f"r{num_added}", rng.choice([8, 20, 40, 90, 150]), hasher, num_added
            )
            request.max_tokens = rng.randint(1, 12)
            scheduler.add_request(request)
            num_added += 1
        fresh = {r.request_id for r in scheduler.waiting if not r.num_computed_tokens}
        out = scheduler.schedule()
        scheduled = out.num_scheduled_tokens
        solo = [r for r, n in scheduled.items() if n > over_tokens]
        if solo:
            assert len(scheduled) == 1
            assert solo[0] in fresh
            assert scheduled[solo[0]] == scheduler.requests[solo[0]].num_tokens
        else:
            assert out.total_num_scheduled_tokens <= over_tokens
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
