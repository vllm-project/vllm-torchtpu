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
"""TPU_PP_WARM_FIRST: the pipeline chunk scheduler's waiting order.

Ported from Benliang Wang's branch benliangw/q38fh-integ-pin-5b44702aea
(d523c2ce0, tests/test_pp_chunk_warm_first.py) by q38 HIGHC-WARMFIRST. The
host-tier test is not ported: on this tree no KV connector implements
host_tier_hit_tokens (see tests/test_pp_chunk_warm_first_q38.py for the
tier fallback).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu.core import pp_chunk_scheduler
from vllm_torchtpu.core.pp_chunk_scheduler import TpuPipelineChunkScheduler

# Prompt tokens a second the served Qwen3.8-Flash PP8 prefill clears.
PREFILL_RATE = 97_000.0
_EMPTY_STEP = SimpleNamespace(total_num_scheduled_tokens=0)


class _Cache:
    """Local prefix cache keyed by request id; counts lookups."""

    def __init__(self, warmth: dict[str, int]):
        self.warmth = dict(warmth)
        self.lookups = 0

    def find_longest_cache_hit(self, block_hashes, max_len):
        self.lookups += 1
        owner = getattr(block_hashes, "owner", block_hashes)
        return ((), min(self.warmth.get(owner, 0), max_len), 0)


def _request(
    req_id: str,
    num_tokens: int,
    arrival: float,
    computed: int = 0,
    preemptions: int = 0,
    skip_cache: bool = False,
):
    return SimpleNamespace(
        request_id=req_id,
        num_tokens=num_tokens,
        num_prompt_tokens=num_tokens,
        num_computed_tokens=computed,
        num_preemptions=preemptions,
        arrival_time=arrival,
        skip_reading_prefix_cache=skip_cache,
        # The cache stub looks hits up by request id.
        block_hashes=req_id,
    )


def _scheduler(
    requests=(),
    warmth=None,
    aging=0.0,
    enabled=True,
    enable_caching=True,
    policy=SchedulingPolicy.FCFS,
    connector=None,
    tier=False,
):
    scheduler = TpuPipelineChunkScheduler.__new__(TpuPipelineChunkScheduler)
    scheduler.connector = connector
    scheduler._init_warm_first(enabled, aging, tier)
    scheduler.policy = policy
    scheduler.waiting = FCFSRequestQueue()
    for request in requests:
        scheduler.waiting.add_request(request)
    scheduler.kv_holding_waiting = FCFSRequestQueue()
    scheduler.running = []
    cache = _Cache(warmth or {})
    scheduler.kv_cache_manager = SimpleNamespace(
        enable_caching=enable_caching, coordinator=cache
    )
    # schedule() state the warm-first order does not read.
    scheduler._cost = None
    scheduler._pp_size = 8
    scheduler._mamba_split = False
    scheduler.need_mamba_block_aligned_split = True
    scheduler._clips = 0
    scheduler._waits = 0
    return scheduler, cache


def _ids(queue) -> list[str]:
    return [r.request_id for r in queue]


def test_warm_requests_go_ahead_of_cold_ones():
    requests = [
        _request("cold", 100_000, 0.0),
        _request("warm3k", 100_000, 0.1),
        _request("warm2k", 50_000, 0.2),
    ]
    warmth = {"warm3k": 97_000, "warm2k": 48_000}
    scheduler, _ = _scheduler(requests, warmth)
    scheduler._order_waiting(now=1.0)
    assert _ids(scheduler.waiting) == ["warm2k", "warm3k", "cold"]
    assert scheduler._wf_reorders == 1


def test_equal_uncached_tokens_go_by_arrival_then_queue_order():
    requests = [
        _request("late", 4_000, 2.0),
        _request("early", 4_000, 1.0),
        _request("same_a", 4_000, 3.0),
        _request("same_b", 4_000, 3.0),
    ]
    scheduler, _ = _scheduler(requests)
    scheduler._order_waiting(now=5.0)
    assert _ids(scheduler.waiting) == ["early", "late", "same_a", "same_b"]


def test_without_aging_the_order_is_by_uncached_tokens_alone():
    # A cold request that has waited a minute is still behind a warm one
    # that just arrived.
    requests = [_request("cold", 100_000, 0.0), _request("warm", 100_000, 60.0)]
    scheduler, _ = _scheduler(requests, {"warm": 96_000}, aging=0.0)
    scheduler._order_waiting(now=60.0)
    assert _ids(scheduler.waiting) == ["warm", "cold"]


def test_a_cold_request_is_passed_only_by_requests_arriving_within_its_bound():
    # At 10,000 tokens/s a 100,000-token cold prompt is passed by a 3,000-token
    # warm request only if the warm one arrives less than (100,000 - 3,000) /
    # 10,000 = 9.7 s after it.
    aging = 10_000.0
    for arrival, passes in ((9.6, True), (9.8, False), (30.0, False)):
        requests = [_request("cold", 100_000, 0.0), _request("warm", 100_000, arrival)]
        scheduler, _ = _scheduler(requests, {"warm": 97_000}, aging=aging)
        scheduler._order_waiting(now=arrival + 0.01)
        expected = ["warm", "cold"] if passes else ["cold", "warm"]
        assert _ids(scheduler.waiting) == expected, arrival
    # Once it has waited 100,000 / 10,000 = 10 s, nothing that arrives later
    # passes it, however few tokens it needs.
    requests = [_request("cold", 100_000, 0.0), _request("tiny", 2, 10.0)]
    scheduler, _ = _scheduler(requests, {"tiny": 1}, aging=aging)
    scheduler._order_waiting(now=10.0)
    assert _ids(scheduler.waiting) == ["cold", "tiny"]


def _drain(aging: float, horizon_s: float = 60.0, warm_per_s: int = 33):
    """Serves the queue head one request at a time at PREFILL_RATE, ordering
    the waiting queue before each admission. A 100,000-token cold prompt
    arrives at t=0 into a stream of 3,000-token warm requests at warm_per_s
    (99,000 tokens/s, just above the prefill's rate) for horizon_s. Returns
    the cold prompt's wait in seconds."""
    arrivals = [(0.0, "cold", 100_000)]
    step = 1.0 / warm_per_s
    arrivals += [(i * step, f"w{i}", 3_000) for i in range(int(horizon_s * warm_per_s))]
    arrivals.sort(key=lambda a: a[0])
    scheduler, _ = _scheduler(aging=aging)
    now, nxt = 0.0, 0
    while True:
        while nxt < len(arrivals) and arrivals[nxt][0] <= now:
            t, req_id, tokens = arrivals[nxt]
            scheduler.waiting.add_request(_request(req_id, tokens, t))
            nxt += 1
        if not scheduler.waiting:
            now = arrivals[nxt][0]
            continue
        scheduler._order_waiting(now)
        head = scheduler.waiting.pop_request()
        if head.request_id == "cold":
            return now
        now += head.num_tokens / PREFILL_RATE


def test_aging_bounds_the_wait_of_a_cold_prompt_at_the_measured_prefill_rate():
    # Warm first without aging starves the cold prompt while warm work keeps
    # arriving; at 10,000 tokens/s it waits about its 10 s bound (9.7 s for
    # the warm requests that may pass it, plus the backlog left at 97,000
    # tokens/s).
    starved = _drain(aging=0.0)
    bounded = _drain(aging=10_000.0)
    assert starved > 55.0
    assert 9.0 < bounded <= 100_000 / 10_000 + 1.0


def test_started_requests_keep_the_head_in_their_order():
    resumed = _request("resumed", 60_000, 5.0, computed=6_400)
    preempted = _request("preempted", 80_000, 4.0, preemptions=1)
    requests = [_request("cold", 100_000, 0.0), resumed, preempted]
    requests.append(_request("warm", 100_000, 6.0))
    scheduler, cache = _scheduler(requests, {"warm": 99_000})
    running = [_request("run_a", 100_000, 0.0, computed=3_200)]
    running.append(_request("run_b", 2_000, 0.0, computed=1_000))
    scheduler.running = list(running)
    holding = _request("holding", 100_000, 0.0, computed=90_000)
    scheduler.kv_holding_waiting.add_request(holding)
    scheduler._order_waiting(now=7.0)
    assert _ids(scheduler.waiting) == ["resumed", "preempted", "warm", "cold"]
    # Running requests and kv_holding_waiting are untouched, and requests
    # that resume work need no lookup.
    assert [r.request_id for r in scheduler.running] == ["run_a", "run_b"]
    assert all(a is b for a, b in zip(scheduler.running, running))
    assert _ids(scheduler.kv_holding_waiting) == ["holding"]
    assert cache.lookups == 2


def test_only_the_window_is_ordered():
    window = pp_chunk_scheduler._WARM_FIRST_WINDOW
    # The window holds prompts from large to small; two tiny prompts sit
    # past it.
    requests = [_request(f"r{i}", 200_000 - i * 100, float(i)) for i in range(window)]
    requests += [_request("tiny_a", 10, 1000.0), _request("tiny_b", 20, 1001.0)]
    scheduler, cache = _scheduler(requests)
    scheduler._order_waiting(now=2000.0)
    ids = _ids(scheduler.waiting)
    assert ids[:window] == [f"r{i}" for i in reversed(range(window))]
    assert ids[window:] == ["tiny_a", "tiny_b"]
    assert cache.lookups == window


def test_a_window_longer_than_the_queue_adds_and_drops_nothing():
    requests = [_request("b", 9_000, 0.0), _request("a", 1_000, 1.0)]
    scheduler, _ = _scheduler(requests)
    scheduler._order_waiting(now=2.0)
    assert _ids(scheduler.waiting) == ["a", "b"]
    assert len(scheduler.waiting) == 2


def test_empty_and_single_request_queues_are_left_alone():
    scheduler, cache = _scheduler([])
    scheduler._order_waiting(now=1.0)
    assert len(scheduler.waiting) == 0
    scheduler, cache = _scheduler([_request("only", 100_000, 0.0)])
    scheduler._order_waiting(now=1.0)
    assert _ids(scheduler.waiting) == ["only"]
    assert cache.lookups == 0
    assert scheduler._wf_sorts == 0


def test_cache_bypass_and_full_hits_count_as_the_base_scheduler_counts_them():
    # A full hit leaves the last prompt token to compute; a request that skips
    # the prefix cache is cold whatever the cache holds; caching off makes
    # every prompt cold, so shorter prompts go first.
    requests = [
        _request("skip", 50_000, 0.0, skip_cache=True),
        _request("full", 50_000, 1.0),
        _request("part", 50_000, 2.0),
    ]
    warmth = {"skip": 50_000, "full": 50_000, "part": 40_000}
    scheduler, _ = _scheduler(requests, warmth)
    scheduler._order_waiting(now=3.0)
    assert _ids(scheduler.waiting) == ["full", "part", "skip"]
    assert scheduler._uncached["full"][0] == 1

    requests = [_request("long", 90_000, 0.0), _request("short", 30_000, 1.0)]
    scheduler, cache = _scheduler(requests, {"long": 89_000}, enable_caching=False)
    scheduler._order_waiting(now=2.0)
    assert _ids(scheduler.waiting) == ["short", "long"]
    assert cache.lookups == 0


def test_a_lookup_is_reused_until_it_is_stale():
    requests = [_request("a", 100_000, 0.0), _request("b", 100_000, 0.5)]
    scheduler, cache = _scheduler(requests, {"a": 99_000, "b": 0})
    scheduler._order_waiting(now=1.0)
    assert _ids(scheduler.waiting) == ["a", "b"]
    assert cache.lookups == 2
    # "a" loses its prefix and "b" gains one; within the TTL the order holds.
    cache.warmth = {"a": 0, "b": 99_000}
    scheduler._order_waiting(now=5.0)
    assert cache.lookups == 2
    assert _ids(scheduler.waiting) == ["a", "b"]
    ttl = pp_chunk_scheduler._WARM_FIRST_LOOKUP_TTL_S
    scheduler._order_waiting(now=1.0 + ttl)
    assert cache.lookups == 4
    assert _ids(scheduler.waiting) == ["b", "a"]


def test_lookups_of_requests_that_left_the_window_are_dropped():
    requests = [_request("a", 5_000, 0.0), _request("b", 6_000, 0.1)]
    scheduler, _ = _scheduler(requests)
    scheduler._order_waiting(now=1.0)
    assert set(scheduler._uncached) == {"a", "b"}
    scheduler.waiting.pop_request()
    scheduler.waiting.add_request(_request("c", 7_000, 1.5))
    scheduler._order_waiting(now=2.0)
    assert set(scheduler._uncached) == {"b", "c"}


def test_a_priority_queue_is_left_alone():
    requests = [_request("cold", 100_000, 0.0), _request("warm", 100_000, 1.0)]
    scheduler, cache = _scheduler(
        requests, {"warm": 99_000}, policy=SchedulingPolicy.PRIORITY
    )
    scheduler._order_waiting(now=2.0)
    assert _ids(scheduler.waiting) == ["cold", "warm"]
    assert cache.lookups == 0


def test_schedule_orders_the_queue_before_the_base_scan():
    requests = [_request("cold", 100_000, 0.0), _request("warm", 100_000, 1.0)]
    scheduler, _ = _scheduler(requests, {"warm": 99_000})
    seen = []

    def base_schedule(self, throttle_prefills=False):
        seen.append(_ids(self.waiting))
        return _EMPTY_STEP

    with patch.object(Scheduler, "schedule", base_schedule):
        scheduler.schedule()
    assert seen == [["warm", "cold"]]


def test_off_by_default_keeps_first_come_first_served(monkeypatch):
    monkeypatch.delenv("TPU_PP_WARM_FIRST", raising=False)
    assert pp_chunk_scheduler.envs.TPU_PP_WARM_FIRST is False
    requests = [_request("cold", 100_000, 0.0), _request("warm", 100_000, 1.0)]
    scheduler, cache = _scheduler(requests, {"warm": 99_000}, enabled=False)
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert _ids(scheduler.waiting) == ["cold", "warm"]
    assert cache.lookups == 0


def test_a_negative_aging_rate_is_refused():
    scheduler = TpuPipelineChunkScheduler.__new__(TpuPipelineChunkScheduler)
    with pytest.raises(ValueError, match="TPU_PP_WARM_FIRST_AGING"):
        scheduler._init_warm_first(True, -1.0)
