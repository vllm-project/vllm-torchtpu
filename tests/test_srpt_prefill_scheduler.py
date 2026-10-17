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
"""SrptPrefillScheduler tests on a real scheduler, through schedule()."""

import pytest
import torch
from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.core.experimental.srpt_prefill_scheduler import SrptPrefillScheduler
from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler

BLOCK_SIZE = 16
MODEL = "facebook/opt-125m"


@pytest.fixture(scope="module")
def shared_model_config():
    return ModelConfig(model=MODEL, dtype="float16", seed=42, max_model_len=2048)


def _build_real_scheduler(
    model_config: ModelConfig,
    *,
    scheduler_cls=SrptPrefillScheduler,
    num_blocks: int = 32,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 64,
    long_prefill_token_threshold: int = 0,
    scheduler_reserve_full_isl: bool = False,
    policy: str = "fcfs",
    num_draft_tokens: int = 0,
):
    parallel_config = ParallelConfig()
    speculative_config = None
    if num_draft_tokens:
        # Parallel drafting reserves num_draft_tokens slots per request.
        speculative_config = SpeculativeConfig(
            target_model_config=model_config,
            target_parallel_config=parallel_config,
            method="draft_model",
            model=MODEL,
            num_speculative_tokens=num_draft_tokens,
            parallel_drafting=True,
        )
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=2048,
            enable_chunked_prefill=True,
            long_prefill_token_threshold=long_prefill_token_threshold,
            scheduler_reserve_full_isl=scheduler_reserve_full_isl,
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
        parallel_config=parallel_config,
        speculative_config=speculative_config,
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
    scheduler = scheduler_cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK_SIZE,
    )
    init_none_hash(sha256)
    hasher = get_request_block_hasher(BLOCK_SIZE, sha256)
    return scheduler, hasher


def _make_real_request(
    request_id: str,
    num_tokens: int,
    hasher,
    seed: int = 0,
    shared_prefix: list[int] | None = None,
) -> Request:
    params = SamplingParams(max_tokens=8, ignore_eos=True)
    params.update_from_generation_config({}, 2)
    prefix = list(shared_prefix) if shared_prefix is not None else []
    tail_len = num_tokens - len(prefix)
    prompt = prefix + [seed * 100_000 + i for i in range(tail_len)]
    return Request(request_id, prompt, params, None, block_hasher=hasher)


def _running_long_prefill(scheduler, hasher, num_tokens: int = 64) -> Request:
    """Admit `long` and run its first chunk so it is a running prefill."""
    req_long = _make_real_request("long", num_tokens, hasher, seed=1)
    scheduler.add_request(req_long)
    out = scheduler.schedule()
    assert set(out.num_scheduled_tokens) == {"long"}
    assert req_long.is_prefill_chunk is True
    return req_long


@pytest.mark.parametrize(
    ("scheduler_cls", "expected"),
    [
        # Admission order: `long` (48 left) takes its 16-token cap first.
        (AsyncScheduler, {"long": 16, "short": 8}),
        # SRPT order: `short` (40 left) goes first.
        (SrptPrefillScheduler, {"short": 16, "long": 8}),
    ],
)
def test_runs_running_prefill_chunks_fewest_remaining_first(
    shared_model_config, scheduler_cls, expected
):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        scheduler_cls=scheduler_cls,
        max_num_batched_tokens=24,
        long_prefill_token_threshold=16,
    )
    scheduler.add_request(_make_real_request("long", 64, hasher, seed=1))
    scheduler.add_request(_make_real_request("short", 48, hasher, seed=2))
    assert scheduler.schedule().num_scheduled_tokens == {"long": 16, "short": 8}

    assert scheduler.schedule().num_scheduled_tokens == expected


@pytest.mark.parametrize(
    ("scheduler_cls", "expected_step2_scheduled"),
    [
        # Upstream schedules running `long` first, which takes the whole
        # 16-token budget and leaves `short_wait` waiting.
        (PrefixAwareScheduler, {"long"}),
        # SRPT holds `long` (48 left > 16) so `short_wait` is admitted.
        (SrptPrefillScheduler, {"short_wait"}),
    ],
)
def test_holds_back_long_running_chunk_for_shorter_waiting_request(
    shared_model_config,
    scheduler_cls,
    expected_step2_scheduled,
):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        scheduler_cls=scheduler_cls,
        max_num_seqs=4,
        max_num_batched_tokens=16,
    )
    req_long = _running_long_prefill(scheduler, hasher)
    req_short_wait = _make_real_request("short_wait", 16, hasher, seed=2)
    scheduler.add_request(req_short_wait)

    out2 = scheduler.schedule()
    assert set(out2.num_scheduled_tokens) == expected_step2_scheduled
    assert scheduler.max_num_running_reqs == 4

    if scheduler_cls is SrptPrefillScheduler:
        # The held chunk is restored with its KV blocks.
        assert req_long in scheduler.running
        assert req_long.status == RequestStatus.RUNNING
        assert req_short_wait.is_prefill_chunk is False


def test_holds_for_next_waiting_request_without_reordering_waiting(
    shared_model_config,
):
    """SRPT holds for the request upstream admits next (queue order) and does
    not promote a warmer request behind it."""
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        max_num_batched_tokens=16,
    )
    assert isinstance(scheduler, AsyncScheduler)
    assert not isinstance(scheduler, PrefixAwareScheduler)
    # Prime 32 tokens (2 blocks) into the prefix cache.
    shared_prefix = [900_000 + i for i in range(32)]
    primer = _make_real_request(
        "primer", 32, hasher, seed=1, shared_prefix=shared_prefix
    )
    scheduler.add_request(primer)
    scheduler.schedule()
    scheduler.schedule()
    scheduler.finish_requests("primer", RequestStatus.FINISHED_ABORTED)
    scheduler.schedule()

    req_long = _make_real_request("long", 64, hasher, seed=10)
    scheduler.add_request(req_long)
    scheduler.schedule()
    assert req_long.is_prefill_chunk is True

    # `next` (32 uncached) arrives ahead of `warm` (48 tokens, 32 cached).
    scheduler.add_request(_make_real_request("next", 32, hasher, seed=2))
    scheduler.add_request(
        _make_real_request("warm", 48, hasher, seed=3, shared_prefix=shared_prefix)
    )

    out = scheduler.schedule()
    assert set(out.num_scheduled_tokens) == {"next"}
    assert [r.request_id for r in scheduler.waiting] == ["warm"]
    assert req_long in scheduler.running


def test_no_hold_when_waiting_request_is_longer(shared_model_config):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        max_num_batched_tokens=16,
    )
    _running_long_prefill(scheduler, hasher, num_tokens=32)  # 16 left
    scheduler.add_request(_make_real_request("longer", 48, hasher, seed=2))

    assert set(scheduler.schedule().num_scheduled_tokens) == {"long"}


def test_no_hold_at_max_num_seqs(shared_model_config):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        max_num_seqs=2,
        max_num_batched_tokens=32,
        long_prefill_token_threshold=16,
    )
    scheduler.add_request(_make_real_request("long", 64, hasher, seed=1))
    scheduler.add_request(_make_real_request("dec", 16, hasher, seed=2))
    assert set(scheduler.schedule().num_scheduled_tokens) == {"long", "dec"}
    # Both running slots are taken, so holding `long` cannot admit `short`.
    scheduler.add_request(_make_real_request("short", 16, hasher, seed=3))

    out = scheduler.schedule()
    assert "long" in out.num_scheduled_tokens
    assert "short" not in out.num_scheduled_tokens


@pytest.mark.parametrize(
    ("throttle_prefills", "policy"),
    [(True, "fcfs"), (False, "priority")],
)
def test_no_hold_on_throttled_step_or_priority_policy(
    shared_model_config, throttle_prefills, policy
):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        max_num_batched_tokens=16,
        policy=policy,
    )
    _running_long_prefill(scheduler, hasher)
    scheduler.add_request(_make_real_request("short_wait", 16, hasher, seed=2))

    out = scheduler.schedule(throttle_prefills=throttle_prefills)
    assert set(out.num_scheduled_tokens) == {"long"}


def test_holds_only_the_chunks_needed(shared_model_config):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        max_num_batched_tokens=48,
        long_prefill_token_threshold=32,
    )
    scheduler.add_request(_make_real_request("medium", 48, hasher, seed=1))
    scheduler.add_request(_make_real_request("long", 80, hasher, seed=2))
    assert scheduler.schedule().num_scheduled_tokens == {"medium": 32, "long": 16}

    # `medium` (16 left) plus `long` (32-token cap) fill the 48-token budget;
    # holding `long` alone leaves 32 for `short_wait`, so `medium` still runs.
    scheduler.add_request(_make_real_request("short_wait", 16, hasher, seed=3))
    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {"medium": 16, "short_wait": 16}


def test_hold_never_exceeds_max_num_seqs(shared_model_config):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        long_prefill_token_threshold=16,
    )
    for request_id, num_tokens, seed in (("dec_1", 16, 1), ("dec_2", 16, 2)):
        scheduler.add_request(_make_real_request(request_id, num_tokens, hasher, seed))
    scheduler.add_request(_make_real_request("long", 64, hasher, seed=3))
    assert set(scheduler.schedule().num_scheduled_tokens) == {"dec_1", "dec_2", "long"}

    # Holding `long` lowers the admission cap to 3 for the step, so only one of
    # two short requests is admitted and `long` restored makes 4, not 5.
    scheduler.add_request(_make_real_request("short_1", 16, hasher, seed=4))
    scheduler.add_request(_make_real_request("short_2", 16, hasher, seed=5))
    out = scheduler.schedule()
    assert "short_1" in out.num_scheduled_tokens
    assert "short_2" not in out.num_scheduled_tokens
    assert len(scheduler.running) == 4
    assert scheduler.max_num_running_reqs == 4
    assert scheduler.max_num_active_reqs == 4


def test_failed_hold_backs_off_exponentially_per_target(shared_model_config):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        max_num_seqs=4,
        max_num_batched_tokens=16,
    )
    _running_long_prefill(scheduler, hasher, num_tokens=160)
    # Deliverable stale output in flight: upstream skips this request in its
    # WAITING pass, so a hold for it admits nothing.
    req_stuck = _make_real_request("stuck", 16, hasher, seed=2)
    req_stuck.num_stale_output_tokens = 1
    req_stuck.drop_stale_output = False
    scheduler.add_request(req_stuck)

    scheduled = [set(scheduler.schedule().num_scheduled_tokens) for _ in range(5)]
    # Failed hold, 1 step off, failed hold, 2 steps off.
    assert scheduled == [set(), {"long"}, set(), {"long"}, {"long"}]

    req_stuck.num_stale_output_tokens = 0
    assert set(scheduler.schedule().num_scheduled_tokens) == {"stuck"}


@pytest.mark.parametrize(
    ("scheduler_cls", "expected_running", "expected_preempted"),
    [
        # Admission order leaves `dec` at running[-1], so it is preempted.
        (PrefixAwareScheduler, ["long", "short"], "dec"),
        # SRPT order puts `long` at running[-1], so it is preempted instead.
        (SrptPrefillScheduler, ["dec", "short"], "long"),
    ],
)
def test_kv_pressure_preempts_longest_prefill(
    shared_model_config,
    scheduler_cls,
    expected_running,
    expected_preempted,
):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        scheduler_cls=scheduler_cls,
        num_blocks=6,
        max_num_batched_tokens=48,
        long_prefill_token_threshold=BLOCK_SIZE,
    )
    reqs = {
        "long": _make_real_request("long", 64, hasher, seed=1),
        "short": _make_real_request("short", 32, hasher, seed=2),
        "dec": _make_real_request("dec", 16, hasher, seed=3),
    }
    for req in reqs.values():
        scheduler.add_request(req)
    assert set(scheduler.schedule().num_scheduled_tokens) == set(reqs)
    # An older long prefill ahead of newer requests, and no free KV blocks.
    scheduler.running[:] = [reqs["long"], reqs["short"], reqs["dec"]]
    pinned = _make_real_request("pinned", 16, hasher, seed=4)
    scheduler.kv_cache_manager.allocate_slots(pinned, 16)
    # A short waiting request must not trigger a hold under KV pressure.
    scheduler.add_request(_make_real_request("short_wait", 16, hasher, seed=5))

    out = scheduler.schedule()
    assert [r.request_id for r in scheduler.running] == expected_running
    assert set(out.num_scheduled_tokens) == set(expected_running)
    assert reqs[expected_preempted].status == RequestStatus.PREEMPTED


@pytest.mark.parametrize(
    ("scheduler_reserve_full_isl", "expected"), [(False, "wait_32"), (True, "long")]
)
def test_full_isl_reservation_checks_whole_prompt_before_holding(
    shared_model_config, scheduler_reserve_full_isl, expected
):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        num_blocks=16,
        max_num_batched_tokens=16,
        scheduler_reserve_full_isl=scheduler_reserve_full_isl,
    )
    _running_long_prefill(scheduler, hasher)
    # `long` (48 tokens left) reserves 3 more blocks; leave 4 free: room for one
    # 16-token chunk of `wait_32` (1 block) but not its whole prompt (2 blocks).
    pool = scheduler.kv_cache_manager.block_pool
    pin_tokens = (pool.get_num_free_blocks() - 4) * BLOCK_SIZE
    pinned = _make_real_request("pinned", pin_tokens, hasher, seed=6)
    scheduler.kv_cache_manager.allocate_slots(pinned, pin_tokens)
    assert pool.get_num_free_blocks() == 4
    scheduler.add_request(_make_real_request("wait_32", 32, hasher, seed=7))

    assert set(scheduler.schedule().num_scheduled_tokens) == {expected}


def test_hold_with_speculative_decoding_draft_slots(shared_model_config):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        max_num_batched_tokens=40,
        num_draft_tokens=2,
    )
    _running_long_prefill(scheduler, hasher, num_tokens=96)
    scheduler.add_request(_make_real_request("short_wait", 16, hasher, seed=2))

    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {"short_wait": 16}
    # Upstream charges each scheduled request its 2 drafting slots.
    assert out.total_num_scheduled_tokens + 2 * len(out.num_scheduled_tokens) <= 40


def test_consecutive_holds_are_capped(shared_model_config):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        max_num_batched_tokens=16,
    )
    scheduler.max_consecutive_holds = 2
    req_long = _running_long_prefill(scheduler, hasher)

    for i in (1, 2):
        scheduler.add_request(_make_real_request(f"short_{i}", 16, hasher, seed=i + 1))
        out = scheduler.schedule()
        # `long` is held; earlier shorts may decode alongside.
        assert f"short_{i}" in out.num_scheduled_tokens
        assert "long" not in out.num_scheduled_tokens
    # Held twice in a row: `long` is not held again and runs its chunk.
    scheduler.add_request(_make_real_request("short_3", 16, hasher, seed=4))
    out = scheduler.schedule()
    assert "long" in out.num_scheduled_tokens
    assert req_long.num_computed_tokens > 16


def test_progress_is_measured_in_kv_cache_blocks(shared_model_config):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        max_num_batched_tokens=20,
    )
    scheduler.max_consecutive_holds = 2
    # Hybrid models use the LCM of all KV group block sizes here, which can
    # exceed a whole chunk; progress must not be measured against it.
    scheduler.block_size = 64 * BLOCK_SIZE
    req_long = _running_long_prefill(scheduler, hasher, num_tokens=96)

    for i in (1, 2):
        scheduler.add_request(_make_real_request(f"short_{i}", 16, hasher, seed=i + 1))
        assert "long" not in scheduler.schedule().num_scheduled_tokens
    # Stalled: runs a chunk of at least one KV cache block next to decodes.
    scheduler.add_request(_make_real_request("short_3", 16, hasher, seed=4))
    assert "long" in scheduler.schedule().num_scheduled_tokens
    computed = req_long.num_computed_tokens
    # Progress resets the count, so `long` can be held again.
    scheduler.add_request(_make_real_request("short_4", 16, hasher, seed=5))
    out = scheduler.schedule()
    assert "long" not in out.num_scheduled_tokens
    assert req_long.num_computed_tokens == computed


def test_prefill_crowded_out_by_shorter_running_prefill_is_promoted(
    shared_model_config,
):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        max_num_batched_tokens=16,
    )
    scheduler.max_consecutive_holds = 2
    req_long = _running_long_prefill(scheduler, hasher)
    scheduler.add_request(_make_real_request("mid", 40, hasher, seed=2))

    # Step 2: `long` is held so `mid` (40 < 48 tokens left) is admitted.
    assert set(scheduler.schedule().num_scheduled_tokens) == {"mid"}
    # Step 3: no hold, but `mid` (24 left) runs first and takes the budget.
    assert set(scheduler.schedule().num_scheduled_tokens) == {"mid"}
    # Two steps without a block of progress: `long` now runs ahead of `mid`.
    assert scheduler.schedule().num_scheduled_tokens == {"long": 16}
    assert req_long.num_computed_tokens == 32
    # Having advanced, `long` falls back behind the shorter `mid`.
    assert scheduler.schedule().num_scheduled_tokens == {"mid": 8, "long": 8}


def test_max_consecutive_holds_env_var_and_compile_cache(
    shared_model_config, monkeypatch
):
    monkeypatch.setenv("VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS", "16")
    scheduler, _ = _build_real_scheduler(shared_model_config)
    assert scheduler.max_consecutive_holds == 16
    assert (
        "VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS" in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    )
    assert (
        "VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS"
        not in tpu_compiler._tpu_compile_env_factors()
    )
