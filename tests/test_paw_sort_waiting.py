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
"""TPU_SCHED_PAW_SORT_WAITING: the waiting queue is sorted inside schedule().

vLLM v0.31.0 no longer calls ``_select_waiting_queue_for_scheduling`` from its
waiting loop, so these tests check the order the base scheduler sees when it
starts admitting, i.e. inside ``AsyncScheduler.schedule``.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import vllm
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy

from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler


def _req(request_id, num_tokens, arrival_time, num_computed_tokens=0, num_preemptions=0):
    return SimpleNamespace(
        request_id=request_id,
        num_tokens=num_tokens,
        num_prompt_tokens=num_tokens,
        arrival_time=arrival_time,
        num_preemptions=num_preemptions,
        num_computed_tokens=num_computed_tokens,
        skip_reading_prefix_cache=False,
        block_hashes=[object()],
    )


def _scheduler(monkeypatch, requests, warmth, sort="1", aging="0", window=None):
    monkeypatch.setenv("TPU_SCHED_PAW_SORT_WAITING", sort)
    monkeypatch.setenv("TPU_SCHED_PAW_AGING_TOK_PER_S", aging)
    if window is not None:
        monkeypatch.setenv("TPU_SCHED_PAW_SORT_WINDOW", str(window))
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    q = FCFSRequestQueue()
    for r in requests:
        q.add_request(r)
    s.waiting = q
    s.kv_holding_waiting = FCFSRequestQueue()
    s.running = []
    s.policy = SchedulingPolicy.FCFS
    s._paw_n = 0
    s._paw_cold_tokens = 0
    s._paw_prompt_tokens = 0
    s._paw_scoring_broken = False
    s._mk_max_running = 0
    s.block_size = 2176

    coordinator = MagicMock()

    def find_hit(block_hashes, max_len):
        for r in requests:
            if r.block_hashes is block_hashes:
                return ({}, warmth.get(r.request_id, 0), 0)
        return ({}, 0, 0)

    coordinator.find_longest_cache_hit.side_effect = find_hit
    kvm = MagicMock()
    kvm.enable_caching = True
    kvm.coordinator = coordinator
    s.kv_cache_manager = kvm
    s._init_paw_sort()
    return s


def _order_seen_by_base_schedule(s):
    seen = {}

    def base_schedule(*args, **kwargs):
        seen["order"] = [r.request_id for r in s.waiting]
        return SimpleNamespace()

    with patch.object(AsyncScheduler, "schedule", side_effect=base_schedule):
        s.schedule()
    return seen["order"]


def test_flag_off_leaves_waiting_fcfs(monkeypatch):
    now = time.time()
    reqs = [
        _req("big_cold", 90_000, now - 3),
        _req("small_warm", 92_000, now - 2),
        _req("mid_warm", 60_000, now - 1),
    ]
    warmth = {"small_warm": 90_000, "mid_warm": 50_000}
    s = _scheduler(monkeypatch, reqs, warmth, sort="0")
    assert _order_seen_by_base_schedule(s) == ["big_cold", "small_warm", "mid_warm"]


def test_sorts_by_uncached_tokens_before_base_schedule(monkeypatch):
    now = time.time()
    reqs = [
        _req("big_cold", 90_000, now - 3),  # 90,000 uncached
        _req("small_warm", 92_000, now - 2),  # 2,000 uncached
        _req("mid_warm", 60_000, now - 1),  # 10,000 uncached
    ]
    warmth = {"small_warm": 90_000, "mid_warm": 50_000}
    s = _scheduler(monkeypatch, reqs, warmth)
    assert _order_seen_by_base_schedule(s) == ["small_warm", "mid_warm", "big_cold"]
    assert s._paw_sort_reordered == 1
    # Already sorted: the next step leaves the queue alone.
    assert _order_seen_by_base_schedule(s) == ["small_warm", "mid_warm", "big_cold"]
    assert s._paw_sort_reordered == 1 and s._paw_sort_steps == 2


def test_aging_bounds_how_long_a_large_prompt_is_overtaken(monkeypatch):
    now = time.time()
    reqs = [
        _req("old_big", 150_000, now - 60),  # 150,000 - 3,000 x 60 s < 0
        _req("fresh_small", 52_000, now - 1),  # 2,000 - 3,000 x 1 s
    ]
    warmth = {"fresh_small": 50_000}
    s = _scheduler(monkeypatch, reqs, warmth, aging="0")
    assert _order_seen_by_base_schedule(s) == ["fresh_small", "old_big"]
    reqs = [
        _req("old_big", 150_000, now - 60),
        _req("fresh_small", 52_000, now - 1),
    ]
    s = _scheduler(monkeypatch, reqs, warmth, aging="3000")
    assert _order_seen_by_base_schedule(s) == ["old_big", "fresh_small"]


def test_started_requests_go_first(monkeypatch):
    now = time.time()
    reqs = [
        _req("small", 12_000, now - 2),
        _req("resumed", 200_000, now - 1, num_computed_tokens=4_352),
    ]
    warmth = {"small": 10_000}
    s = _scheduler(monkeypatch, reqs, warmth)
    assert _order_seen_by_base_schedule(s) == ["resumed", "small"]


def test_requests_that_do_not_fit_free_blocks_go_last(monkeypatch):
    now = time.time()
    reqs = [
        _req("huge", 200_000, now - 100),  # 92 blocks of 2,176
        _req("small", 12_000, now - 1),  # 1 block
    ]
    s = _scheduler(monkeypatch, reqs, {"small": 10_000}, aging="3000")
    s._get_num_free_blocks = lambda: 50
    assert _order_seen_by_base_schedule(s) == ["small", "huge"]


def test_external_hit_counts_when_it_fits(monkeypatch):
    now = time.time()
    reqs = [
        _req("hbm_mid", 30_000, now - 2),  # 8,000 uncached
        _req("dram_small", 100_000, now - 1),  # 2,000 uncached after a DRAM load
    ]
    s = _scheduler(monkeypatch, reqs, {"hbm_mid": 22_000, "dram_small": 10_000})
    s._is_kv_consumer = False
    s.connector = SimpleNamespace(
        estimate_external_hit_tokens=lambda r, local: 88_000
        if r.request_id == "dram_small"
        else 0
    )
    assert _order_seen_by_base_schedule(s) == ["dram_small", "hbm_mid"]
    # Without room for the loaded blocks the DRAM hit does not count.
    reqs = [_req("hbm_mid", 30_000, now - 2), _req("dram_small", 100_000, now - 1)]
    s = _scheduler(monkeypatch, reqs, {"hbm_mid": 22_000, "dram_small": 10_000})
    s._is_kv_consumer = False
    s._is_kv_producer = True
    s._cold_threshold = 8192
    s.connector = SimpleNamespace(
        estimate_external_hit_tokens=lambda r, local: 88_000
        if r.request_id == "dram_small"
        else 0
    )
    s._get_num_free_blocks = lambda: 45
    s._inflight_prefill_reserved_blocks = lambda: 30  # 15 left < 41 to load
    assert _order_seen_by_base_schedule(s) == ["hbm_mid", "dram_small"]


def test_window_bounds_the_sort(monkeypatch):
    now = time.time()
    reqs = [
        _req("a", 50_000, now - 4),
        _req("b", 3_000, now - 3),
        _req("c", 1_000, now - 2),  # outside a window of 2
    ]
    s = _scheduler(monkeypatch, reqs, {}, window=2)
    assert _order_seen_by_base_schedule(s) == ["b", "a", "c"]


def test_sort_failure_disables_the_flag_and_schedule_still_runs(monkeypatch):
    now = time.time()
    reqs = [_req("a", 50_000, now - 2), _req("b", 3_000, now - 1)]
    s = _scheduler(monkeypatch, reqs, {})

    def boom(*args, **kwargs):
        raise RuntimeError("simulated scoring failure")

    s._paw_sort_key = boom
    assert _order_seen_by_base_schedule(s) == ["a", "b"]
    assert s._paw_sort is False


@pytest.mark.skipif(vllm.__version__ != "0.31.0", reason="pins v0.31.0 behaviour")
def test_vllm_0_31_waiting_loop_skips_the_selection_hook():
    # The reason the flag exists: on v0.31.0 _promote_warmest is never reached.
    assert PrefixAwareScheduler._base_calls_waiting_hook() is False
