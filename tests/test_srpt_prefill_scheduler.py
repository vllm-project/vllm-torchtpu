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
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler
from vllm_torchtpu.core.srpt_prefill_scheduler import SrptPrefillScheduler

BLOCK_SIZE = 16


@pytest.fixture(scope="module")
def shared_model_config():
    return ModelConfig(
        model="facebook/opt-125m", dtype="float16", seed=42, max_model_len=2048
    )


def _build_real_scheduler(
    model_config: ModelConfig,
    *,
    scheduler_cls=SrptPrefillScheduler,
    num_blocks: int = 32,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 64,
    long_prefill_token_threshold: int = 0,
):
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=2048,
            enable_chunked_prefill=True,
            long_prefill_token_threshold=long_prefill_token_threshold,
            scheduler_reserve_full_isl=False,
            is_encoder_decoder=False,
            watermark=0.0,
            async_scheduling=True,
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


def test_decodes_first_then_prefill_chunks_by_remaining(shared_model_config):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        num_blocks=32,
        max_num_batched_tokens=64,
        long_prefill_token_threshold=BLOCK_SIZE,
    )
    req_long = _make_real_request("long_prefill", 64, hasher, seed=1)
    req_dec_1 = _make_real_request("dec_1", 16, hasher, seed=2)
    req_short = _make_real_request("short_prefill", 32, hasher, seed=3)
    req_dec_2 = _make_real_request("dec_2", 16, hasher, seed=4)

    for req in (req_long, req_dec_1, req_short, req_dec_2):
        scheduler.add_request(req)

    scheduler.schedule()
    assert req_dec_1.is_prefill_chunk is False
    assert req_dec_2.is_prefill_chunk is False
    assert req_short.is_prefill_chunk is True
    assert req_long.is_prefill_chunk is True

    # Scramble running order and verify _order_running places decodes first
    # (preserving relative order) followed by prefill chunks fewest-remaining first.
    scheduler.running[:] = [req_long, req_dec_1, req_short, req_dec_2]
    scheduler._order_running()
    assert [r.request_id for r in scheduler.running] == [
        "dec_1",
        "dec_2",
        "short_prefill",
        "long_prefill",
    ]


def test_priority_policy_and_throttle_prefills_skip_hold(shared_model_config):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        num_blocks=32,
        max_num_batched_tokens=16,
    )
    req_long = _make_real_request("long_prefill", 64, hasher, seed=1)
    scheduler.add_request(req_long)
    scheduler.schedule()

    req_short_wait = _make_real_request("short_wait", 16, hasher, seed=2)
    scheduler.add_request(req_short_wait)

    # 1. throttle_prefills=True skips holding.
    assert scheduler._chunks_to_hold(throttle_prefills=True) == []

    # 2. SchedulingPolicy.PRIORITY skips _order_running and _chunks_to_hold.
    req_dec = _make_real_request("dec", 16, hasher, seed=3)
    req_dec.num_computed_tokens = 16
    scheduler.running[:] = [req_long, req_dec]
    scheduler.policy = SchedulingPolicy.PRIORITY
    scheduler._order_running()
    assert [r.request_id for r in scheduler.running] == ["long_prefill", "dec"]
    assert scheduler._chunks_to_hold() == []


def test_inherits_prefix_aware_waiting_promotion_and_deduplicates_warmth_lookup(
    shared_model_config, monkeypatch
):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        num_blocks=32,
        max_num_batched_tokens=16,
    )
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

    # Put a long running prefill chunk in self.running so both _chunks_to_hold
    # and _promote_warmest evaluate the waiting queue in the same step.
    req_long = _make_real_request("long", 64, hasher, seed=10)
    scheduler.add_request(req_long)
    scheduler.schedule()
    assert req_long.is_prefill_chunk is True

    # Add a cold 48-token request ahead of a warm 48-token request (32 cached,
    # 16 uncached).
    req_cold = _make_real_request("cold", 48, hasher, seed=2)
    req_warm = _make_real_request(
        "warm", 48, hasher, seed=3, shared_prefix=shared_prefix
    )
    scheduler.add_request(req_cold)
    scheduler.add_request(req_warm)

    lookup_calls = []
    orig_find = scheduler.kv_cache_manager.coordinator.find_longest_cache_hit

    def counting_find(*args, **kwargs):
        lookup_calls.append(args)
        return orig_find(*args, **kwargs)

    monkeypatch.setattr(
        scheduler.kv_cache_manager.coordinator,
        "find_longest_cache_hit",
        counting_find,
    )

    out = scheduler.schedule()
    assert "warm" in out.num_scheduled_tokens
    assert "cold" not in out.num_scheduled_tokens
    # PrefixAware warmth scoring for (cold, warm) is memoized across
    # _chunks_to_hold and _promote_warmest (plus 1 direct get_computed_blocks
    # call inside upstream Scheduler.schedule() when admitting `warm`).
    assert len(lookup_calls) == 3


@pytest.mark.parametrize(
    ("scheduler_cls", "expected_step2_scheduled"),
    [
        # Plain PrefixAwareScheduler schedules running `long` first, which
        # consumes the entire 16-token step budget and leaves `short_wait`
        # unscheduled in self.waiting.
        (PrefixAwareScheduler, {"long"}),
        # SrptPrefillScheduler holds back `long` (48 remaining > 16 waiting)
        # for Step 2 so `short_wait` is admitted immediately, then restores
        # `long` to self.running.
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
        num_blocks=32,
        max_num_seqs=4,
        max_num_batched_tokens=16,
    )
    req_long = _make_real_request("long", 64, hasher, seed=1)
    scheduler.add_request(req_long)

    # Step 1: `long` runs its first 16-token chunk (48 tokens remaining).
    out1 = scheduler.schedule()
    assert set(out1.num_scheduled_tokens.keys()) == {"long"}
    assert req_long.is_prefill_chunk is True
    assert req_long.num_computed_tokens == 16

    # A short 16-token request arrives in self.waiting while `long` is in self.running.
    req_short_wait = _make_real_request("short_wait", 16, hasher, seed=2)
    scheduler.add_request(req_short_wait)

    out2 = scheduler.schedule()
    assert set(out2.num_scheduled_tokens.keys()) == expected_step2_scheduled
    assert scheduler.max_num_running_reqs == 4

    if scheduler_cls is SrptPrefillScheduler:
        # Both `short_wait` (now decoding) and `long` (restored) are in self.running.
        assert [r.request_id for r in scheduler.running] == ["short_wait", "long"]
        assert req_short_wait.is_prefill_chunk is False
        assert req_long.is_prefill_chunk is True


def test_no_hold_when_waiting_is_longer_or_blocked_or_at_capacity(shared_model_config):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        num_blocks=32,
        max_num_seqs=2,
        max_num_batched_tokens=16,
    )
    req_chunk = _make_real_request("chunk", 32, hasher, seed=1)
    scheduler.add_request(req_chunk)
    scheduler.schedule()
    assert req_chunk.is_prefill_chunk is True
    assert req_chunk.num_computed_tokens == 16  # 16 remaining

    # 1. Waiting request has more remaining tokens (48 > 16): do not hold.
    req_longer = _make_real_request("longer", 48, hasher, seed=2)
    scheduler.add_request(req_longer)
    assert scheduler._chunks_to_hold() == []
    scheduler.waiting.clear()

    # 2. Waiting request is blocked on remote KVs: do not hold.
    req_blocked = _make_real_request("blocked", 16, hasher, seed=3)
    req_blocked.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.add_request(req_blocked)
    assert scheduler._chunks_to_hold() == []
    scheduler.waiting.clear()

    # 3. Running queue is already at max_num_seqs capacity (2 == 2): do not hold.
    req_dec = _make_real_request("dec", 16, hasher, seed=4)
    scheduler.running.append(req_dec)
    req_short = _make_real_request("short", 16, hasher, seed=5)
    scheduler.add_request(req_short)
    assert scheduler._chunks_to_hold() == []


def test_holds_only_needed_chunks_and_respects_preempted_tier(shared_model_config):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        num_blocks=32,
        max_num_seqs=8,
        max_num_batched_tokens=48,
        long_prefill_token_threshold=32,
    )
    # Admit two prefill chunks: `medium` (48 tokens -> 32 scheduled, 16 left)
    # and `long` (80 tokens -> 16 scheduled, 64 left).
    req_medium = _make_real_request("medium", 48, hasher, seed=1)
    req_long = _make_real_request("long", 80, hasher, seed=2)
    scheduler.add_request(req_medium)
    scheduler.add_request(req_long)
    out1 = scheduler.schedule()
    assert out1.num_scheduled_tokens == {"medium": 32, "long": 16}
    assert req_medium.is_prefill_chunk is True  # 16 remaining
    assert req_long.is_prefill_chunk is True  # 64 remaining (wants 32 on next step)

    # 1. If a preempted long request (64 remaining) and a normal short request
    # (16 remaining) are both waiting, _promote_warmest prioritizes the
    # preempted request (tier 1), so _chunks_to_hold compares against 64 and
    # does not hold either running chunk.
    req_preempted_long = _make_real_request("preempted_long", 64, hasher, seed=3)
    req_preempted_long.num_preemptions = 1
    req_short_wait = _make_real_request("short_wait", 16, hasher, seed=4)
    scheduler.add_request(req_preempted_long)
    scheduler.add_request(req_short_wait)
    assert scheduler._chunks_to_hold() == []

    # 2. Remove the preempted request so `short_wait` (16 tokens) is the target.
    # Step budget is 48: `medium` needs 16 and `long` wants 32 (total 48).
    # Holding only `long` leaves 48 - 16 = 32 >= 16 tokens for `short_wait`, so
    # `medium` is NOT held and runs alongside `short_wait` in Step 2.
    scheduler.waiting.remove(req_preempted_long)
    held = scheduler._chunks_to_hold()
    assert [r.request_id for r in held] == ["long"]

    out2 = scheduler.schedule()
    assert set(out2.num_scheduled_tokens.keys()) == {"medium", "short_wait"}


def test_near_capacity_hold_never_exceeds_max_num_seqs(shared_model_config):
    """Reviewer Test #1: Near-capacity (max_num_seqs=4, 3 running including a
    long chunk, >=2 short waiting requests that fit the budget)."""
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        num_blocks=32,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        long_prefill_token_threshold=16,
    )
    req_dec_1 = _make_real_request("dec_1", 16, hasher, seed=1)
    req_dec_2 = _make_real_request("dec_2", 16, hasher, seed=2)
    req_long = _make_real_request("long", 64, hasher, seed=3)
    for req in (req_dec_1, req_dec_2, req_long):
        scheduler.add_request(req)

    # Step 1: dec_1 (16), dec_2 (16), long (16 of 64) -> 3 running requests.
    out1 = scheduler.schedule()
    assert set(out1.num_scheduled_tokens.keys()) == {"dec_1", "dec_2", "long"}
    assert len(scheduler.running) == 3

    # Add 2 short waiting requests (16 tokens each) that both fit the 64-token
    # step budget. Because 1 chunk (`long`) is held, active cap is lowered from
    # 4 to 3 during super().schedule(), admitting only `short_1` so restored
    # running length is 4 (never 5 > max_num_seqs).
    req_short_1 = _make_real_request("short_1", 16, hasher, seed=4)
    req_short_2 = _make_real_request("short_2", 16, hasher, seed=5)
    scheduler.add_request(req_short_1)
    scheduler.add_request(req_short_2)

    scheduler.max_num_active_reqs = 4
    out2 = scheduler.schedule()
    assert "short_1" in out2.num_scheduled_tokens
    assert "short_2" not in out2.num_scheduled_tokens
    assert len(scheduler.running) == 4
    assert scheduler.max_num_running_reqs == 4
    assert scheduler.max_num_active_reqs == 4


def test_failed_hold_skips_until_waiting_head_changes_or_blocks_free(
    shared_model_config,
):
    """Reviewer Test #2: When a held step fails to admit the waiting head, sets
    _skip_hold_next_step and suppresses further holds on subsequent steps until
    either free KV blocks increase or the waiting head changes."""
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        num_blocks=32,
        max_num_seqs=4,
        max_num_batched_tokens=16,
    )
    req_long = _make_real_request("long", 96, hasher, seed=1)
    scheduler.add_request(req_long)

    # Step 1: `long` runs first 16-token chunk (80 remaining).
    scheduler.schedule()
    assert req_long.num_computed_tokens == 16

    # Add `short_unadmittable` (16 tokens) with deliverable stale output in
    # flight (`num_stale_output_tokens=1`, `drop_stale_output=False`) so
    # upstream Scheduler.schedule() skips it during waiting admission.
    req_unadmittable = _make_real_request("short_unadmittable", 16, hasher, seed=2)
    req_unadmittable.num_stale_output_tokens = 1
    req_unadmittable.drop_stale_output = False
    scheduler.add_request(req_unadmittable)

    # Step 2: `long` is held back, admission of `short_unadmittable` fails, and
    # `_skip_hold_next_step` + `_blocked_hold_state` are armed.
    out2 = scheduler.schedule()
    assert out2.num_scheduled_tokens == {}
    assert scheduler._skip_hold_next_step is True

    # Step 3: `_skip_hold_next_step` lets `long` run (32 computed).
    out3 = scheduler.schedule()
    assert set(out3.num_scheduled_tokens.keys()) == {"long"}
    assert req_long.num_computed_tokens == 32
    assert scheduler._skip_hold_next_step is False

    # Step 4: `short_unadmittable` is still at the head and free blocks have not
    # increased -> `_blocked_hold_state` keeps suppressing hold so `long` does
    # NOT lose every other step (48 computed).
    out4 = scheduler.schedule()
    assert set(out4.num_scheduled_tokens.keys()) == {"long"}
    assert req_long.num_computed_tokens == 48

    # Step 5: A new admissible request arrives and becomes the best candidate ->
    # hold re-engages and admits `short_fresh` immediately.
    req_unadmittable.num_stale_output_tokens = 0
    req_fresh = _make_real_request("short_fresh", 16, hasher, seed=3)
    scheduler.waiting.clear()
    scheduler.add_request(req_fresh)
    out5 = scheduler.schedule()
    assert set(out5.num_scheduled_tokens.keys()) == {"short_fresh"}


@pytest.mark.parametrize(
    ("scheduler_cls", "expected_running", "expected_preempted"),
    [
        # Plain PrefixAwareScheduler leaves running in [long, short, dec], so
        # long takes the block and dec (at running[-1]) is preempted.
        (PrefixAwareScheduler, ["long", "short"], "dec"),
        # SrptPrefillScheduler reorders running to [dec, short, long], so dec
        # and short proceed and long (at running[-1]) is preempted.
        (SrptPrefillScheduler, ["dec", "short"], "long"),
    ],
)
def test_real_scheduler_orders_running_and_preempts_longest_prefill(
    shared_model_config,
    scheduler_cls,
    expected_running,
    expected_preempted,
):
    """Compare PrefixAwareScheduler vs SrptPrefillScheduler under KV pressure
    on a real Scheduler + KVCacheManager."""
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        scheduler_cls=scheduler_cls,
        num_blocks=6,
        max_num_batched_tokens=48,
        long_prefill_token_threshold=BLOCK_SIZE,
    )

    req_long = _make_real_request("long", 64, hasher, seed=1)
    req_short = _make_real_request("short", 32, hasher, seed=2)
    req_dec = _make_real_request("dec", 16, hasher, seed=3)

    scheduler.add_request(req_long)
    scheduler.add_request(req_short)
    scheduler.add_request(req_dec)

    out1 = scheduler.schedule()
    assert set(out1.num_scheduled_tokens.keys()) == {"dec", "short", "long"}
    assert req_dec.is_prefill_chunk is False
    assert req_short.is_prefill_chunk is True
    assert req_long.is_prefill_chunk is True

    # Place req_long at the front of self.running to simulate FCFS admission order
    # where an older long prefill sits ahead of newer decode / short prefill.
    scheduler.running[:] = [req_long, req_short, req_dec]

    # Pin the 1 remaining free block so 0 blocks are free entering Step 2.
    extra_dummy = _make_real_request("pinned", 16, hasher, seed=4)
    scheduler.kv_cache_manager.allocate_slots(extra_dummy, 16)

    out2 = scheduler.schedule()

    assert [r.request_id for r in scheduler.running] == expected_running
    assert expected_preempted not in out2.num_scheduled_tokens
    reqs = {"long": req_long, "short": req_short, "dec": req_dec}
    assert reqs[expected_preempted].status == RequestStatus.PREEMPTED


def test_preemption_while_waiting_request_present_does_not_hold_or_preempt_decode(
    shared_model_config,
):
    """Reviewer Test #3: With the KV pool full and a short waiting request
    present, SrptPrefillScheduler must NOT hold `long` (which would leave `dec`
    or `short` at `self.running[-1]` to be preempted while `long` keeps its KV
    blocks). Instead, `long` stays at `self.running[-1]` and is preempted."""
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        num_blocks=6,
        max_num_batched_tokens=48,
        long_prefill_token_threshold=BLOCK_SIZE,
    )
    req_long = _make_real_request("long", 64, hasher, seed=1)
    req_short = _make_real_request("short", 32, hasher, seed=2)
    req_dec = _make_real_request("dec", 16, hasher, seed=3)

    for req in (req_long, req_short, req_dec):
        scheduler.add_request(req)

    scheduler.schedule()
    scheduler.running[:] = [req_long, req_short, req_dec]

    # Exhaust the last free block AND add a short waiting request that would
    # normally trigger hold-back if KV blocks were available.
    extra_dummy = _make_real_request("pinned", 16, hasher, seed=4)
    scheduler.kv_cache_manager.allocate_slots(extra_dummy, 16)
    req_wait = _make_real_request("short_wait", 16, hasher, seed=5)
    scheduler.add_request(req_wait)

    guard_results: list[bool] = []
    orig_guard = scheduler._has_kv_blocks_for_hold

    def _spy_guard(*args, **kwargs):
        res = orig_guard(*args, **kwargs)
        guard_results.append(res)
        return res

    scheduler._has_kv_blocks_for_hold = _spy_guard
    assert scheduler._chunks_to_hold() == []
    assert guard_results == [False]
    out2 = scheduler.schedule()
    assert guard_results == [False, False]

    # `dec` and `short` are scheduled; `long` (at running[-1]) is preempted.
    assert [r.request_id for r in scheduler.running] == ["dec", "short"]
    assert set(out2.num_scheduled_tokens.keys()) == {"dec", "short"}
    assert req_long.status == RequestStatus.PREEMPTED
    assert req_dec.status == RequestStatus.RUNNING


def test_spec_decode_draft_slots_and_full_isl_and_backoff(shared_model_config):
    """Reviewer §3.1, §3.3, §3.4:
    1. `draft_slots` is subtracted per scheduled request (both decode and prefill).
    2. `scheduler_reserve_full_isl=True` checks the waiting request's full prompt
       length against free blocks, not only its first step chunk.
    3. Exponential backoff suppresses repeated holds even if free blocks grow.
    """
    from types import SimpleNamespace

    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        num_blocks=32,
        max_num_seqs=8,
        max_num_batched_tokens=64,
    )
    # 1. Configure draft_slots = 6 (like DSpark K=6).
    scheduler.vllm_config.speculative_config = SimpleNamespace(
        max_num_new_slots_for_drafting=6
    )

    req_dec_1 = _make_real_request("dec_1", 16, hasher, seed=1)
    req_dec_2 = _make_real_request("dec_2", 16, hasher, seed=2)
    req_med = _make_real_request("med", 48, hasher, seed=3)
    req_long = _make_real_request("long", 96, hasher, seed=4)
    for req in (req_dec_1, req_dec_2, req_med, req_long):
        scheduler.add_request(req)

    # Step 1: admit dec_1 (16+6=22), dec_2 (16+6=22), med (14+6=20 -> 34 rem).
    # `long` stays in waiting; give it 16 computed tokens manually and place in running
    # so we have 2 decodes (1 tok each) + `med` (32 rem) + `long` (80 rem).
    scheduler.schedule()
    scheduler.waiting.clear()
    req_med.num_computed_tokens = 20
    req_med.is_prefill_chunk = True
    req_long.num_computed_tokens = 16
    req_long.is_prefill_chunk = True
    req_long.status = RequestStatus.RUNNING
    scheduler.running[:] = [req_dec_1, req_dec_2, req_med, req_long]
    scheduler.requests["long"] = req_long

    # With input_budget=64 and draft_slots=6:
    # - 2 decodes consume (1 + 6) * 2 = 14 input_budget -> 50 left.
    # - `med` wants 48 - 20 = 28 tokens, consuming 28 + 6 = 34 input_budget -> 16 left.
    # - `short_wait` (16 tokens) needs 16 + 6 = 22 input_budget!
    # If draft_slots were only subtracted once, 64 - 6 - 2 - 28 = 28 >= 16 would
    # falsely claim `med` leaves enough room. With per-request draft_slots, only
    # 16 - 6 = 10 tokens remain after `med`, so BOTH `long` and `med` must be held.
    req_short_wait = _make_real_request("short_wait", 16, hasher, seed=5)
    scheduler.add_request(req_short_wait)
    held = scheduler._chunks_to_hold()
    assert [r.request_id for r in held] == ["med", "long"]

    # 2. Full-ISL fit check (`scheduler_reserve_full_isl=True`):
    # Reset speculative config and set step budget = lptt = 16 so `req_long`
    # alone consumes the 16-token step budget. Test a waiting request whose step
    # chunk (lptt=16 -> 1 block) fits in free_blocks, but whose full prompt
    # (48 tokens -> 3 blocks) does not fit.
    scheduler.vllm_config.speculative_config = None
    scheduler.scheduler_config.max_num_batched_tokens = 16
    scheduler.scheduler_config.long_prefill_token_threshold = 16
    scheduler.scheduler_reserve_full_isl = True
    scheduler.waiting.clear()
    scheduler.running[:] = [req_long]
    scheduler._inflight_prefills.clear()

    # Leave exactly 2 free blocks: enough for a 1-block step chunk, but < 3 blocks
    # needed for `wait_48`'s full 48-token prompt.
    free_now = scheduler._num_free_blocks()
    if free_now > 2:
        pin_toks = (free_now - 2) * BLOCK_SIZE
        pin_req = _make_real_request("pin_isl", pin_toks, hasher, seed=6)
        scheduler.kv_cache_manager.allocate_slots(pin_req, pin_toks)
    assert scheduler._num_free_blocks() == 2

    req_wait_48 = _make_real_request("wait_48", 48, hasher, seed=7)
    scheduler.add_request(req_wait_48)
    assert scheduler._chunks_to_hold() == []
    scheduler.scheduler_reserve_full_isl = False
    assert [r.request_id for r in scheduler._chunks_to_hold()] == ["long"]

    # 3. Exponential backoff when a non-KV failure blocks `wait_48` across steps
    # even as free blocks increase. Simulate two consecutive failed holds on
    # `wait_48` (`failures=2` -> cooldown=2 steps):
    scheduler._blocked_hold_failures["wait_48"] = 2
    scheduler._blocked_hold_cooldown["wait_48"] = 2
    scheduler._blocked_hold_target_id = "wait_48"
    scheduler._blocked_hold_free_blocks = 1  # free_now=2 > 1, yet cooldown blocks!
    assert scheduler._chunks_to_hold() == []
    assert scheduler._blocked_hold_cooldown["wait_48"] == 1
    assert scheduler._chunks_to_hold() == []
    assert scheduler._blocked_hold_cooldown.get("wait_48", 0) == 0
    assert [r.request_id for r in scheduler._chunks_to_hold()] == ["long"]


def test_starvation_bound_caps_consecutive_holds(shared_model_config):
    """Reviewer Test #4: After MAX_CONSECUTIVE_HOLDS consecutive holds, the
    long prefill chunk is guaranteed a step to advance."""
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        num_blocks=32,
        max_num_seqs=8,
        max_num_batched_tokens=16,
    )
    scheduler.MAX_CONSECUTIVE_HOLDS = 2

    req_long = _make_real_request("long", 64, hasher, seed=1)
    scheduler.add_request(req_long)

    # Step 1: `long` runs its first 16-token chunk (48 remaining).
    out1 = scheduler.schedule()
    assert set(out1.num_scheduled_tokens.keys()) == {"long"}
    assert req_long.num_computed_tokens == 16

    # Step 2 (hold #1): `short_1` arrives and is admitted while `long` is held.
    scheduler.add_request(_make_real_request("short_1", 16, hasher, seed=2))
    out2 = scheduler.schedule()
    assert "short_1" in out2.num_scheduled_tokens
    assert "long" not in out2.num_scheduled_tokens
    assert req_long.num_computed_tokens == 16

    # Step 3 (hold #2): `short_2` arrives and is admitted while `long` is held.
    scheduler.add_request(_make_real_request("short_2", 16, hasher, seed=3))
    out3 = scheduler.schedule()
    assert "short_2" in out3.num_scheduled_tokens
    assert "long" not in out3.num_scheduled_tokens
    assert req_long.num_computed_tokens == 16

    # Step 4: `short_3` arrives, but `long` has reached MAX_CONSECUTIVE_HOLDS=2,
    # so `long` is exempt from holding, runs its next chunk, and resets its
    # consecutive-hold counter to 0.
    scheduler.add_request(_make_real_request("short_3", 16, hasher, seed=4))
    out4 = scheduler.schedule()
    assert "long" in out4.num_scheduled_tokens
    assert req_long.num_computed_tokens > 16
    assert scheduler._consecutive_holds.get("long", 0) == 0

    # Step 5: With its hold counter reset, `long` can be held again for `short_3`.
    out5 = scheduler.schedule()
    assert "short_3" in out5.num_scheduled_tokens


def test_env_var_and_pruning_and_adaptive_lptt(shared_model_config, monkeypatch):
    """Verify:
    1. `VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS` configures `MAX_CONSECUTIVE_HOLDS`.
    2. `_prune_hold_tracking` cleans up `_blocked_hold_failures`,
       `_blocked_hold_cooldown`, and `_blocked_hold_state` for aborted waiting.
    3. `_effective_long_prefill_threshold` returns 0 when only 1 request is
       eligible and applies the adaptive floor when enabled.
    """
    monkeypatch.setenv("VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS", "16")
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        num_blocks=32,
        max_num_seqs=8,
        max_num_batched_tokens=64,
        long_prefill_token_threshold=16,
    )
    assert scheduler.MAX_CONSECUTIVE_HOLDS == 16

    from vllm_torchtpu.compilation import tpu_compiler

    assert (
        "VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS" in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    )
    assert (
        "VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS"
        not in tpu_compiler._tpu_compile_env_factors()
    )

    # 2. Aborted waiting request state is pruned on the next step.
    req_wait = _make_real_request("aborted_wait", 16, hasher, seed=1)
    scheduler.add_request(req_wait)
    scheduler._blocked_hold_failures["aborted_wait"] = 3
    scheduler._blocked_hold_cooldown["aborted_wait"] = 4
    scheduler._blocked_hold_state = ("aborted_wait", 10)
    scheduler._prune_hold_tracking()
    assert "aborted_wait" in scheduler._blocked_hold_failures

    scheduler.waiting.clear()
    scheduler._prune_hold_tracking()
    assert "aborted_wait" not in scheduler._blocked_hold_failures
    assert "aborted_wait" not in scheduler._blocked_hold_cooldown
    assert scheduler._blocked_hold_state is None

    # 3. Runtime-adjusted long_prefill_token_threshold:
    # With only 1 eligible request, threshold becomes 0 (no cap).
    req_1 = _make_real_request("r1", 48, hasher, seed=2)
    scheduler.running[:] = [req_1]
    assert scheduler._effective_long_prefill_threshold() == 0
    assert scheduler._running_step_tokens(req_1) == 48

    # With 2 eligible requests and adaptive_long_prefill_threshold=True:
    # input_budget=64 // 2 = 32 > lptt=16 -> floor is 32.
    req_2 = _make_real_request("r2", 16, hasher, seed=3)
    scheduler.add_request(req_2)
    scheduler.adaptive_long_prefill_threshold = False
    assert scheduler._effective_long_prefill_threshold() == 16
    scheduler.adaptive_long_prefill_threshold = True
    assert scheduler._effective_long_prefill_threshold() == 32
    assert scheduler._running_step_tokens(req_1) == 32
