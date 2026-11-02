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

from vllm_torchtpu.core.experimental import prefill_first_scheduler
from vllm_torchtpu.core.experimental.prefill_first_scheduler import (
    PrefillFirstScheduler,
)

BLOCK_SIZE = 16
MODEL = "facebook/opt-125m"


@pytest.fixture(scope="module")
def shared_model_config():
    return ModelConfig(model=MODEL, dtype="float16", seed=42, max_model_len=2048)


class _Clock:
    def __init__(self, step_s: float) -> None:
        self.now = 0.0
        self.step_s = step_s

    def __call__(self) -> float:
        return self.now

    def tick(self) -> None:
        self.now += self.step_s


def _build(
    model_config: ModelConfig,
    monkeypatch,
    *,
    over_tokens: int = 0,
    max_consecutive: int = 0,
    decode_steps: int = 1,
    decode_interval_ms: int = 0,
    num_blocks: int = 64,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 128,
    policy: str = "fcfs",
):
    monkeypatch.setenv("VLLM_TPU_SOLO_PREFILL_OVER_TOKENS", str(over_tokens))
    monkeypatch.setenv("VLLM_TPU_SOLO_PREFILL_MAX_CONSECUTIVE", str(max_consecutive))
    monkeypatch.setenv("VLLM_TPU_SOLO_PREFILL_DECODE_STEPS", str(decode_steps))
    monkeypatch.setenv("VLLM_TPU_DECODE_INTERVAL_MS", str(decode_interval_ms))
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=2048,
            enable_chunked_prefill=True,
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


def test_regular_step_puts_hidden_requests_back_in_order(
    shared_model_config, monkeypatch
):
    scheduler, hasher = _build(
        shared_model_config, monkeypatch, over_tokens=32, max_consecutive=1
    )
    _add(scheduler, hasher, ("s1", 16, 1))
    scheduler.schedule()
    _add(scheduler, hasher, ("L1", 100, 2), ("L2", 100, 3), ("s2", 30, 4))
    _add(scheduler, hasher, ("s3", 16, 5))
    assert scheduler.schedule().num_scheduled_tokens == {"L1": 100}

    # Two decodes and s2's 30 tokens fill the 32-token regular budget.
    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {"s1": 1, "L1": 1, "s2": 30}
    assert [r.request_id for r in scheduler.waiting] == ["L2", "s3"]


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


def test_needs_a_policy(shared_model_config, monkeypatch):
    with pytest.raises(ValueError, match="needs"):
        _build(shared_model_config, monkeypatch)


def test_decodes_are_paced_while_a_prefill_is_pending(shared_model_config, monkeypatch):
    clock = _Clock(step_s=0.010)
    monkeypatch.setattr(prefill_first_scheduler, "monotonic", clock)
    scheduler, hasher = _build(
        shared_model_config,
        monkeypatch,
        decode_interval_ms=30,
        max_num_batched_tokens=32,
    )
    _add(scheduler, hasher, ("d", 16, 1))
    steps = []
    for i in range(9):
        if i == 1:
            _add(scheduler, hasher, ("long", 160, 2))
        steps.append(scheduler.schedule().num_scheduled_tokens)
        clock.tick()
    assert steps == [
        {"d": 16},
        # First step with a prefill pending: due; the next is 30 ms later.
        {"d": 1, "long": 31},
        {"long": 32},
        {"d": 1, "long": 31},
        {"long": 32},
        {"long": 32},
        {"d": 1, "long": 2},
        # Nothing left to prefill: decodes every step.
        {"d": 1, "long": 1},
        {"d": 1, "long": 1},
    ]


def test_decodes_are_not_paused_for_a_prompt_that_does_not_fit(
    shared_model_config, monkeypatch
):
    clock = _Clock(step_s=0.010)
    monkeypatch.setattr(prefill_first_scheduler, "monotonic", clock)
    # Three usable blocks: the 64-token prompt needs four.
    scheduler, hasher = _build(
        shared_model_config, monkeypatch, decode_interval_ms=30, num_blocks=4
    )
    _add(scheduler, hasher, ("d", 16, 1))
    scheduler.schedule()
    _add(scheduler, hasher, ("w", 64, 2))
    for _ in range(3):
        clock.tick()
        assert scheduler.schedule().num_scheduled_tokens == {"d": 1}


def test_paused_step_that_schedules_nothing_forces_decodes(
    shared_model_config, monkeypatch
):
    clock = _Clock(step_s=0.010)
    monkeypatch.setattr(prefill_first_scheduler, "monotonic", clock)
    scheduler, hasher = _build(
        shared_model_config, monkeypatch, decode_interval_ms=30, num_blocks=4
    )
    # The pending-prefill check expects the prompt to fit; admission refuses.
    monkeypatch.setattr(scheduler, "_fits", lambda request, num_tokens: True)
    _add(scheduler, hasher, ("d", 16, 1))
    steps = []
    for i in range(4):
        if i == 1:
            _add(scheduler, hasher, ("w", 64, 2))
        steps.append(scheduler.schedule().num_scheduled_tokens)
        clock.tick()
    # The third step paused the decode and scheduled nothing, so the fourth
    # decodes before the pacing due time.
    assert steps == [{"d": 16}, {"d": 1}, {}, {"d": 1}]


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


@pytest.mark.parametrize(
    ("over_tokens", "max_consecutive", "decode_interval_ms", "num_blocks"),
    [(32, 0, 0, 64), (32, 1, 30, 40), (0, 0, 30, 24)],
)
def test_random_workload_keeps_step_shapes_and_finishes(
    shared_model_config,
    monkeypatch,
    over_tokens,
    max_consecutive,
    decode_interval_ms,
    num_blocks,
):
    """Async steps (one in flight) over random arrivals: every step is solo or
    within the threshold, and every request finishes."""
    rng = random.Random(0)
    clock = _Clock(step_s=0.010)
    monkeypatch.setattr(prefill_first_scheduler, "monotonic", clock)
    scheduler, hasher = _build(
        shared_model_config,
        monkeypatch,
        over_tokens=over_tokens,
        max_consecutive=max_consecutive,
        decode_interval_ms=decode_interval_ms,
        num_blocks=num_blocks,
        max_num_seqs=6,
        max_num_batched_tokens=256 if over_tokens else 48,
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
        solo = [r for r, n in scheduled.items() if over_tokens and n > over_tokens]
        if solo:
            assert len(scheduled) == 1
            assert solo[0] in fresh
            assert scheduled[solo[0]] == scheduler.requests[solo[0]].num_tokens
        elif over_tokens:
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
        clock.tick()
    while in_flight:
        scheduler.update_from_output(*in_flight.popleft())
    assert num_added > 20
    assert scheduler.get_num_unfinished_requests() == 0
