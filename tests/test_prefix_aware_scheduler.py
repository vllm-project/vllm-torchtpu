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

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy

from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler


def _make_request(
    request_id: str,
    num_tokens: int = 1000,
    arrival_time: float = 0.0,
    num_preemptions: int = 0,
    num_computed_tokens: int = 0,
    skip_reading_prefix_cache: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        num_tokens=num_tokens,
        arrival_time=arrival_time,
        num_preemptions=num_preemptions,
        num_computed_tokens=num_computed_tokens,
        skip_reading_prefix_cache=skip_reading_prefix_cache,
        block_hashes=[object()],
    )


def _make_scheduler(
    requests: list[SimpleNamespace],
    warmth_map: dict[str, int],
    window: int = 64,
    enable_caching: bool = True,
    raise_on_lookup: bool = False,
) -> PrefixAwareScheduler:
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    q = FCFSRequestQueue()
    for r in requests:
        q.add_request(r)
    s.waiting = q
    s.kv_holding_waiting = FCFSRequestQueue()
    s.policy = SchedulingPolicy.FCFS
    s.WINDOW = window
    s._paw_n = 0
    s._paw_cold_tokens = 0
    s._paw_prompt_tokens = 0
    s._paw_scoring_broken = False

    coordinator = MagicMock()

    def find_hit(block_hashes, max_len):
        if raise_on_lookup:
            raise RuntimeError("simulated coordinator failure")
        for r in requests:
            if r.block_hashes is block_hashes:
                return ({}, warmth_map.get(r.request_id, 0), 0)
        return ({}, 0, 0)

    coordinator.find_longest_cache_hit.side_effect = find_hit

    kvm = MagicMock()
    kvm.enable_caching = enable_caching
    kvm.coordinator = coordinator
    s.kv_cache_manager = kvm
    return s


def test_default_knobs():
    assert PrefixAwareScheduler.WINDOW == 64


def test_promotes_warmest_within_window():
    now = time.time()
    r_cold = _make_request("cold", num_tokens=1000, arrival_time=now - 10)
    r_warm = _make_request("warm", num_tokens=1000, arrival_time=now - 5)
    s = _make_scheduler([r_cold, r_warm], {"cold": 0, "warm": 800})

    expected = object()

    def schedule(throttle_prefills):
        assert throttle_prefills is True
        assert [r.request_id for r in s.waiting] == ["warm", "cold"]
        return expected

    with patch.object(AsyncScheduler, "schedule", side_effect=schedule):
        assert s.schedule(throttle_prefills=True) is expected


def test_ranks_by_uncached_tokens_not_hit_ratio():
    now = time.time()
    # 100k tokens, 90% cached -> 10k uncached tokens
    r_long = _make_request("long", num_tokens=100_000, arrival_time=now - 10)
    # 20k tokens, 80% cached -> 4k uncached tokens (fewer uncached tokens wins)
    r_short = _make_request("short", num_tokens=20_000, arrival_time=now - 5)
    s = _make_scheduler([r_long, r_short], {"long": 90_000, "short": 16_000})

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["short", "long"]


def test_fcfs_order_preserved_when_all_cold():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 10)
    r2 = _make_request("r2", num_tokens=1000, arrival_time=now - 5)
    s = _make_scheduler([r1, r2], {"r1": 0, "r2": 0})

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    order = [r.request_id for r in q]
    assert order == ["r1", "r2"]


def test_preempted_tier_outranks_normal():
    now = time.time()
    r_preempted = _make_request(
        "preempted", num_tokens=1000, arrival_time=now - 5, num_preemptions=1
    )
    r_warm = _make_request("warm", num_tokens=1000, arrival_time=now - 10)
    s = _make_scheduler([r_preempted, r_warm], {"preempted": 0, "warm": 900})

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    order = [r.request_id for r in q]
    assert order == ["preempted", "warm"]


def test_window_bounds_evaluation():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 15)
    r2 = _make_request("r2", num_tokens=1000, arrival_time=now - 10)
    r_outside = _make_request("outside", num_tokens=1000, arrival_time=now - 5)
    # window = 2, so r_outside is beyond the inspection window
    s = _make_scheduler(
        [r1, r2, r_outside], {"r1": 100, "r2": 200, "outside": 999}, window=2
    )

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    order = [r.request_id for r in q]
    assert order[0] == "r2"  # r2 was warmest in window [r1, r2]


def test_single_or_empty_queue_is_noop():
    s = _make_scheduler([_make_request("single")], {"single": 500})
    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["single"]


def test_uses_num_computed_tokens_when_present():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 10)
    r2 = _make_request(
        "r2", num_tokens=1000, arrival_time=now - 5, num_computed_tokens=900
    )
    s = _make_scheduler([r1, r2], {"r1": 0, "r2": 0})

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["r2", "r1"]


def test_skip_reading_prefix_cache_or_caching_disabled():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 10)
    r2 = _make_request(
        "r2", num_tokens=1000, arrival_time=now - 5, skip_reading_prefix_cache=True
    )
    s = _make_scheduler([r1, r2], {"r1": 0, "r2": 900})

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["r1", "r2"]

    s_disabled = _make_scheduler(
        [r1, _make_request("r3", num_tokens=1000, arrival_time=now - 5)],
        {"r1": 0, "r3": 900},
        enable_caching=False,
    )
    q_disabled = s_disabled._select_waiting_queue_for_scheduling()
    assert q_disabled is not None
    assert [r.request_id for r in q_disabled] == ["r1", "r3"]


def test_coordinator_failure_falls_back_to_fcfs():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 10)
    r2 = _make_request("r2", num_tokens=1000, arrival_time=now - 5)
    s = _make_scheduler([r1, r2], {"r1": 0, "r2": 500}, raise_on_lookup=True)

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["r1", "r2"]
    assert s._paw_scoring_broken is True


def test_epoch_cache_reuses_longest_cache_hit_and_invalidates_on_epoch_change():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 10)
    r2 = _make_request("r2", num_tokens=1000, arrival_time=now - 5)
    s = _make_scheduler([r1, r2], {"r1": 128, "r2": 640})

    class DummyBlockPool:
        def __init__(self):
            self.cached_block_hash_to_block = {}
            self._tpu_prefix_cache_epoch = 1

    pool = DummyBlockPool()
    s.kv_cache_manager.block_pool = pool
    s.kv_cache_manager.enable_kv_cache_events = False
    s.kv_cache_manager.prefix_cache_lookup_enabled = lambda req: True
    s.kv_cache_manager.create_kv_cache_blocks = lambda blocks: blocks
    s.connector = None

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["r2", "r1"]
    assert s.kv_cache_manager.coordinator.find_longest_cache_hit.call_count == 2

    # _get_local_prefix_cache_hit on the promoted request reuses _paw_hit_cache
    blocks, num_local, boundary, diverged = s._get_local_prefix_cache_hit(r2)
    assert num_local == 640
    assert diverged is False
    assert s.kv_cache_manager.coordinator.find_longest_cache_hit.call_count == 2

    # Repeated scheduling step with unchanged epoch does not re-invoke coordinator
    s._select_waiting_queue_for_scheduling()
    assert s.kv_cache_manager.coordinator.find_longest_cache_hit.call_count == 2

    # Advancing _tpu_prefix_cache_epoch invalidates _paw_hit_cache
    pool._tpu_prefix_cache_epoch += 1
    s._select_waiting_queue_for_scheduling()
    assert s.kv_cache_manager.coordinator.find_longest_cache_hit.call_count == 4


def test_free_block_fit_tier_promotes_fitting_warm_over_stalling_head():
    now = time.time()
    # waiting[0] is preempted but needs ceil(1000 / 16) = 63 blocks > 10 free blocks
    r_preempted = _make_request(
        "preempted_big",
        num_tokens=1000,
        arrival_time=now - 20,
        num_preemptions=1,
    )
    # waiting[1] is warm (cold_tokens=320 -> 20 blocks), still > 10 free blocks
    r_warm_big = _make_request("warm_big", num_tokens=1000, arrival_time=now - 15)
    # waiting[2] is warm (cold_tokens=120 -> ceil(120/16)=8 blocks <= 10 free blocks)
    r_warm_fit = _make_request("warm_fit", num_tokens=1000, arrival_time=now - 10)

    s = _make_scheduler(
        [r_preempted, r_warm_big, r_warm_fit],
        {"preempted_big": 0, "warm_big": 680, "warm_fit": 880},
    )
    s.block_size = 16
    s.kv_cache_manager.block_pool.get_num_free_blocks.return_value = 10

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == [
        "warm_fit",
        "preempted_big",
        "warm_big",
    ]


def test_free_block_fit_tier_picks_warmest_among_multiple_fitting():
    now = time.time()
    r_head = _make_request(
        "head_big", num_tokens=1000, arrival_time=now - 20, num_preemptions=1
    )
    # 128 cold tokens -> 8 blocks <= 10 free blocks
    r_fit_8 = _make_request("fit_8", num_tokens=1000, arrival_time=now - 15)
    # 48 cold tokens -> 3 blocks <= 10 free blocks (fewer cold tokens wins)
    r_fit_3 = _make_request("fit_3", num_tokens=1000, arrival_time=now - 10)

    s = _make_scheduler(
        [r_head, r_fit_8, r_fit_3],
        {"head_big": 200, "fit_8": 872, "fit_3": 952},
    )
    s.block_size = 16
    s.kv_cache_manager.block_pool.get_num_free_blocks.return_value = 10

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["fit_3", "head_big", "fit_8"]


def test_free_block_fit_tier_preserves_order_when_no_warm_request_fits():
    now = time.time()
    r_preempted = _make_request(
        "preempted_big",
        num_tokens=1000,
        arrival_time=now - 20,
        num_preemptions=1,
    )
    # 160 cold tokens -> 10 blocks > 5 free blocks
    r_warm_nonfit = _make_request("warm_nonfit", num_tokens=1000, arrival_time=now - 15)
    # 32 cold tokens -> 2 blocks <= 5 free blocks, but warmth == 0 (cold, not warm)
    r_cold_small = _make_request("cold_small", num_tokens=32, arrival_time=now - 10)

    s = _make_scheduler(
        [r_preempted, r_warm_nonfit, r_cold_small],
        {"preempted_big": 0, "warm_nonfit": 840, "cold_small": 0},
    )
    s.block_size = 16
    s.kv_cache_manager.block_pool.get_num_free_blocks.return_value = 5

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == [
        "preempted_big",
        "warm_nonfit",
        "cold_small",
    ]


def test_free_block_fit_tier_inactive_when_head_fits_in_free_blocks():
    now = time.time()
    # 160 cold tokens -> 10 blocks <= 10 free blocks (head fits!)
    r_preempted = _make_request(
        "preempted_fits",
        num_tokens=1000,
        arrival_time=now - 10,
        num_preemptions=1,
    )
    # 32 cold tokens -> 2 blocks <= 10 free blocks
    r_warm = _make_request("warm", num_tokens=1000, arrival_time=now - 5)

    s = _make_scheduler(
        [r_preempted, r_warm],
        {"preempted_fits": 840, "warm": 968},
    )
    s.block_size = 16
    s.kv_cache_manager.block_pool.get_num_free_blocks.return_value = 10

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["preempted_fits", "warm"]


def _running_request(request_id: str, prompt: int, computed: int) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        num_prompt_tokens=prompt,
        num_computed_tokens=computed,
    )


def _make_step_scheduler(running, monkeypatch, budget=8192):
    """A PrefixAwareScheduler whose vLLM schedule() records what it saw."""
    seen = {}

    def fake_schedule(self, *args, **kwargs):
        seen["order"] = [r.request_id for r in self.running]
        seen["budget"] = self.max_num_scheduled_tokens
        seen["args"] = (args, kwargs)
        return "out"

    monkeypatch.setattr(AsyncScheduler, "schedule", fake_schedule, raising=False)
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s.running = list(running)
    s.max_num_scheduled_tokens = budget
    return s, seen


def test_decoders_first_moves_decoders_ahead_of_partial_prefills(monkeypatch):
    running = [
        _running_request("p1", prompt=100_000, computed=7680),
        _running_request("d1", prompt=500, computed=501),
        _running_request("new", prompt=2000, computed=0),
        _running_request("d2", prompt=800, computed=800),
        _running_request("p2", prompt=50_000, computed=640),
    ]
    s, seen = _make_step_scheduler(running, monkeypatch)
    running_list = s.running

    assert s.schedule(True) == "out"

    # With SRPF enabled by default, decoders come first in stable order and
    # prefilling requests are ordered by fewest remaining prompt tokens.
    assert seen["order"] == ["d1", "d2", "new", "p2", "p1"]
    assert s.running is running_list
    assert seen["args"] == ((True,), {})

    # With TPU_SCHED_SRPF_RUNNING=0, prefilling requests keep FCFS order.
    monkeypatch.setenv("TPU_SCHED_SRPF_RUNNING", "0")
    s_fcfs, seen_fcfs = _make_step_scheduler(running, monkeypatch)
    assert s_fcfs.schedule(True) == "out"
    assert seen_fcfs["order"] == ["d1", "d2", "p1", "new", "p2"]


def test_decoders_first_is_noop_without_a_mix(monkeypatch):
    for running in (
        [_running_request("d1", 10, 10), _running_request("d2", 10, 12)],
        [_running_request("p1", 10, 5), _running_request("p2", 10, 0)],
        [],
    ):
        s, seen = _make_step_scheduler(running, monkeypatch)
        s.schedule()
        assert seen["order"] == [r.request_id for r in running]


def test_no_step_token_cap_with_many_decoders(monkeypatch):
    """The validated policy has no decode-protect cap: the step budget stays
    max_num_batched_tokens however many requests decode."""
    running = [_running_request(f"d{i}", 100, 100) for i in range(24)]
    running.append(_running_request("p", 100_000, 0))
    s, seen = _make_step_scheduler(running, monkeypatch, budget=8192)
    s.schedule()
    assert seen["budget"] == 8192
    assert s.max_num_scheduled_tokens == 8192


def test_cold_prefill_chunk_cap_only_caps_cold_requests_at_high_concurrency():
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s._step_num_decoding = 16
    s._step_decode_tokens = 64
    s.waiting = []
    s.skipped_waiting = []

    # Cold request (remaining = 90,000 > 8192) is capped to 2048 - 64 = 1984
    # so decode_tokens + cold_chunk <= 2048 XLA compile bucket.
    cold = _running_request("cold", prompt=90_000, computed=0)
    assert s._cap_cold_prefill_tokens(cold, 0, 8128) == 2048 - 64

    # At 32 decoders (128 decode tokens), cap is 2048 - 128 = 1920 (exact 3 * 640 Mamba
    # blocks).
    s._step_num_decoding = 32
    s._step_decode_tokens = 128
    assert s._cap_cold_prefill_tokens(cold, 0, 8064) == 1920

    # Warm delta turn (remaining = 4,000 <= 8192) is NEVER capped, even at 30 decoders.
    s._step_num_decoding = 30
    s._step_decode_tokens = 120
    warm = _running_request("warm", prompt=50_000, computed=0)
    assert s._cap_cold_prefill_tokens(warm, 46_000, 4000) == 4000

    # At c=8 / c=12 (num_decoding < 12 and waiting empty), cold request is NOT capped.
    s._step_num_decoding = 11
    s._step_decode_tokens = 44
    assert s._cap_cold_prefill_tokens(cold, 0, 8128) == 8128

    # When waiting queue is non-empty, cold request in running IS capped to unlock
    # waiting.
    s._step_num_decoding = 4
    s._step_decode_tokens = 16
    s.waiting = [warm]
    s._step_running_prefill_ids = {id(cold)}
    assert s._cap_cold_prefill_tokens(cold, 2048, 8128) == 2048 - 16

    # First-step admission from waiting only triggers on decoder count (not
    # self-triggering).
    s._step_running_prefill_ids = set()
    assert s._cap_cold_prefill_tokens(cold, 0, 8128) == 8128


def test_step_total_cap_prevents_multi_prefill_8192_bucket_spill():
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s.block_size = 640
    s._step_num_decoding = 28
    s._step_decode_tokens = 112
    s._step_prefill_tokens_by_req = {}
    s.waiting = []
    s.skipped_waiting = []

    c1 = _running_request("c1", prompt=90_000, computed=0)
    c2 = _running_request("c2", prompt=90_000, computed=0)
    c3 = _running_request("c3", prompt=90_000, computed=0)

    # First cold chunk: 2048 - 112 = 1936 (-> 1920 after Mamba 640-block align)
    assert s._cap_cold_prefill_tokens(c1, 0, 8080) == 1936
    s._step_prefill_tokens_by_req[id(c1)] = 1920

    # Second cold chunk: 112 + 1920 = 2032 already scheduled; fits within 4096 cap
    # (1936 -> 1920)
    assert s._cap_cold_prefill_tokens(c2, 0, 6160) == 1936
    s._step_prefill_tokens_by_req[id(c2)] = 1920

    # Third prefill request: 112 + 1920 + 1920 = 3952 already scheduled ->
    # remaining under 4096 cap is 144 < 640 block_size, so it is deferred (0 tokens)
    # rather than spilling the step into the 8192 XLA bucket.
    assert s._cap_cold_prefill_tokens(c3, 0, 4240) == 0

    # Strict 2048 step-total cap (TPU_SCHED_STEP_TOTAL_CAP=2048) defers 2nd prefill
    # immediately:
    s._step_total_cap = 2048
    s._step_prefill_tokens_by_req = {}
    assert s._cap_cold_prefill_tokens(c1, 0, 8080) == 1936
    s._step_prefill_tokens_by_req[id(c1)] = 1920
    assert s._cap_cold_prefill_tokens(c2, 0, 6160) == 0


def test_warm_streak_cadence_gating_defers_prefills_until_k_decode_steps():
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s.block_size = 640
    s._step_num_decoding = 28
    s._step_decode_tokens = 112
    s._warm_streak = 2
    s._warm_min_decoders = 24
    s._pure_decode_streak = 0
    s.waiting = []
    s.skipped_waiting = []

    warm = _running_request("warm", prompt=20_000, computed=18_000)
    # Streak = 0 < 2 -> prefill deferred
    assert s._cap_cold_prefill_tokens(warm, 18_000, 2000) == 0
    # Streak = 1 < 2 -> prefill deferred
    s._pure_decode_streak = 1
    assert s._cap_cold_prefill_tokens(warm, 18_000, 2000) == 0
    # Streak = 2 >= 2 -> prefill admitted
    s._pure_decode_streak = 2
    assert s._cap_cold_prefill_tokens(warm, 18_000, 2000) == 2000


def test_mk_max_running_caps_running_seqs_when_megakernel_active(monkeypatch):
    seen = {}

    def fake_schedule(self, *args, **kwargs):
        seen["max_running"] = self.max_num_running_reqs
        return "out"

    monkeypatch.setattr(AsyncScheduler, "schedule", fake_schedule, raising=False)
    monkeypatch.setenv("VLLM_TPU_QWEN35_MEGAKERNEL", "1")
    # Auto mode is opt-in (the default 0 leaves max_num_running_reqs alone).
    monkeypatch.setenv("TPU_SCHED_MK_MAX_RUNNING", "-1")

    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s.running = [_running_request(f"d{i}", 100, 100) for i in range(16)]
    s.max_num_scheduled_tokens = 8192
    s.max_num_running_reqs = 32
    s.dcp_world_size = 1
    s.num_spec_tokens = 3

    assert s._resolve_mk_max_running() == 16
    assert s.schedule() == "out"
    assert seen["max_running"] == 16
    # Restored after schedule() returns
    assert s.max_num_running_reqs == 32

    # TPU_SCHED_MK_MAX_RUNNING=0 (also the default) preserves max_num_running_reqs=32
    monkeypatch.setenv("TPU_SCHED_MK_MAX_RUNNING", "0")
    s._mk_max_running = None
    assert s.schedule() == "out"
    assert seen["max_running"] == 32
    monkeypatch.delenv("TPU_SCHED_MK_MAX_RUNNING")
    s._mk_max_running = None
    assert s._resolve_mk_max_running() == 0
