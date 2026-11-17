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
"""The waiting-queue sort helpers (``_init_paw_sort``, ``_paw_sort_key``,
``_sort_waiting_queue``).

The pipeline chunk scheduler's prefix-aware admission (#3090) borrows them and
sorts the waiting queue at the start of its schedule(); these tests check the
order one sort leaves. PrefixAwareScheduler itself does not sort.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy

from vllm_torchtpu.core.prefix_aware_scheduler import (
    ExternalHitEstimator,
    PrefixAwareScheduler,
)


def _req(
    request_id, num_tokens, arrival_time, num_computed_tokens=0, num_preemptions=0
):
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
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    # The sort is on for a KV producer; the aging rate is set by the
    # scheduler that borrows the sort.
    s._paw_aging_tok_per_s = float(aging)
    if window is not None:
        s._PAW_SORT_WINDOW = window
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
    s._is_kv_producer = sort == "1"
    s._is_kv_consumer = False
    s._hit_estimator = None
    s._kv_prefetch_n = 0
    s._kv_prefetch_cap = None
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
    """The waiting order the base scheduler admits from: sorted first when
    the sort is on, as the borrowing scheduler's schedule() does."""
    if s._paw_sort:
        s._sort_waiting_queue()
    return [r.request_id for r in s.waiting]


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
    assert s._paw_sort_steps == 0


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
    s._hit_estimator = SimpleNamespace(
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
    s._hit_estimator = SimpleNamespace(
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


def test_prefix_aware_scheduler_itself_does_not_sort(monkeypatch):
    # A KV producer naming PrefixAwareScheduler gets the warmest-first
    # admission alone: the sort runs only where a scheduler borrows it.
    now = time.time()
    reqs = [_req("big_cold", 90_000, now - 2), _req("small_warm", 92_000, now - 1)]
    s = _scheduler(monkeypatch, reqs, {"small_warm": 90_000})
    s._sort_waiting_queue = MagicMock()
    seen = {}

    def base_schedule(*args, **kwargs):
        seen["order"] = [r.request_id for r in s.waiting]
        return SimpleNamespace()

    with patch.object(AsyncScheduler, "schedule", side_effect=base_schedule):
        s.schedule()
    # The promotion moved the warmest request to the head; nothing sorted.
    assert seen["order"] == ["small_warm", "big_cold"]
    s._sort_waiting_queue.assert_not_called()


# ---------------- the connector's host-tier estimate ----------------


class _Estimator:
    def __init__(self, answer=88_000, fail=False):
        self.answer = answer
        self.fail = fail
        self.calls = []

    def estimate_external_hit_tokens(self, request, num_local_tokens):
        self.calls.append((request.request_id, num_local_tokens))
        if self.fail:
            raise RuntimeError("store unreachable")
        return self.answer


def _warmth_scheduler(estimator, consumer=False):
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s._is_kv_consumer = consumer
    s._hit_estimator = estimator
    s.block_size = 2176
    return s


def test_the_estimate_needs_the_named_interface(monkeypatch):
    assert isinstance(_Estimator(), ExternalHitEstimator)
    assert not isinstance(SimpleNamespace(), ExternalHitEstimator)

    def fake_init(self, *args, **kwargs):
        self.vllm_config = SimpleNamespace(kv_transfer_config=None)
        self.connector = self._test_connector
        self.kv_cache_manager = MagicMock()

    monkeypatch.setattr(AsyncScheduler, "__init__", fake_init)
    for connector, expected in ((_Estimator(), True), (SimpleNamespace(), False)):
        monkeypatch.setattr(
            PrefixAwareScheduler, "_test_connector", connector, raising=False
        )
        s = PrefixAwareScheduler()
        assert (s._hit_estimator is connector) is expected


def test_the_estimate_is_asked_past_the_local_hit():
    estimator = _Estimator()
    s = _warmth_scheduler(estimator)
    req = _req("r", 100_000, time.time())
    assert s._external_warmth(req, 10_000) == 88_000
    assert estimator.calls == [("r", 10_000)]
    # Less than a block past the local hit, or a KV consumer: not asked.
    assert s._external_warmth(req, 100_000 - 2_000) == 0
    assert _warmth_scheduler(estimator, consumer=True)._external_warmth(req, 0) == 0
    assert _warmth_scheduler(None)._external_warmth(req, 0) == 0
    assert estimator.calls == [("r", 10_000)]


def test_a_failing_estimate_is_logged_and_counts_as_zero():
    from vllm_torchtpu.core import prefix_aware_scheduler

    s = _warmth_scheduler(_Estimator(fail=True))
    # The real warning_once runs; only the logger's output is captured.
    with patch.object(prefix_aware_scheduler.logger, "warning") as warn:
        assert s._external_warmth(_req("r-fails", 100_000, time.time()), 0) == 0
    warn.assert_called_once()
    assert "r-fails" in warn.call_args.args
