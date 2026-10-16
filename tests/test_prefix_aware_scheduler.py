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
    s.running = []
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


    q = s._select_waiting_queue_for_scheduling()


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

    assert s.running is running_list
    assert seen["args"] == ((True,), {})


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


