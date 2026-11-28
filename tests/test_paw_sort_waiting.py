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
"""The pipeline chunk scheduler's waiting-queue sort (``_init_paw_sort``,
``_paw_sort_key``, ``_sort_waiting_queue``) and the host-tier hit it counts
(``_external_warmth``, the connector's ``ExternalHitEstimator`` interface).

The prefix-aware admission sorts the waiting queue of a KV producer at the
start of its schedule(); these tests check the order one sort leaves, with the
quadratic pricing (option quadratic_cost) off and on. PrefixAwareScheduler
does not sort.
"""

import logging
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy

from vllm_torchtpu.core import pp_chunk_scheduler
from vllm_torchtpu.core.pp_chunk_scheduler import (
    ExternalHitEstimator,
    TpuPipelineChunkScheduler,
    _wait_seconds,
)
from vllm_torchtpu.core.pp_chunks import StepCostModel
from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler

BUCKETS = [4096, 8192, 12288, 16384]


def _cost_model() -> StepCostModel:
    """A stage cost model with an attention slope, for the quadratic pricing."""
    linear = {b: 12.0 + 0.0109 * b for b in BUCKETS}
    attn = 1.5e-7
    full = linear[16384] + attn * 16384**2 / 2
    return StepCostModel(BUCKETS, linear, attn, full * 1.1, 4096)


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


def _waiting(s, requests, warmth):
    """Fills ``s.waiting`` with ``requests`` and answers each one's local hit
    from ``warmth``."""
    q = FCFSRequestQueue()
    for r in requests:
        q.add_request(r)
    s.waiting = q
    s.kv_holding_waiting = FCFSRequestQueue()
    coordinator = MagicMock()

    def find_hit(block_hashes, max_len):
        for r in requests:
            if r.block_hashes is block_hashes:
                return ({}, warmth.get(r.request_id, 0), 0)
        return ({}, 0, 0)

    coordinator.find_longest_cache_hit.side_effect = find_hit
    # One full-attention group: the blocks of the uncached tokens. (The
    # real coordinator's count is tested on a hybrid manager in
    # test_pp_chunk_scheduler_memo.py.)
    coordinator.get_num_blocks_to_allocate.side_effect = lambda **kw: -(
        -max(kw["num_tokens"] - kw["total_computed_tokens"], 0) // 2176
    )
    kvm = MagicMock()
    kvm.enable_caching = True
    kvm.coordinator = coordinator
    kvm.watermark_blocks = 0
    # Plenty of free blocks: every request fits unless a test sets fewer.
    kvm.block_pool = SimpleNamespace(get_num_free_blocks=lambda: 1 << 20)
    s.kv_cache_manager = kvm


def _scheduler(
    requests, warmth, producer=True, aging=0.0, window=None, quadratic=False
):
    """A pipeline chunk scheduler built without __init__, with the
    prefix-aware admission on; the sort runs on a KV producer."""
    s = TpuPipelineChunkScheduler.__new__(TpuPipelineChunkScheduler)
    s._paw_aging_tok_per_s = float(aging)
    s._step_wall_time = None
    if window is not None:
        s._PAW_SORT_WINDOW = window
    _waiting(s, requests, warmth)
    s.running = []
    s.policy = SchedulingPolicy.FCFS
    s._paw_scoring_broken = False
    s._is_kv_producer = producer
    s._is_kv_consumer = False
    s._hit_estimator = None
    s._retention = None
    s._inflight_prefills = set()
    s._cost = _cost_model() if quadratic else None
    s._quadratic_cost = quadratic
    s.block_size = 2176
    s.max_model_len = 262_144
    s.num_lookahead_tokens = 0
    s._init_paw_sort()
    return s


def _prefix_aware_scheduler(requests, warmth):
    """A PrefixAwareScheduler built without __init__, off a decode leg."""
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    _waiting(s, requests, warmth)
    s.running = []
    s.policy = SchedulingPolicy.FCFS
    s._paw_n = 0
    s._paw_cold_tokens = 0
    s._paw_prompt_tokens = 0
    s._paw_scoring_broken = False
    s._is_kv_consumer = False
    s._kv_prefetch_n = 0
    s._kv_prefetch_cap = None
    s._promote_warmest_switch = True
    s.block_size = 2176
    return s


def _order_seen_by_base_schedule(s):
    """The waiting order the base scheduler admits from: sorted first when
    the sort is on, as the chunk scheduler's schedule() does."""
    if s._paw_sort:
        s._sort_waiting_queue()
    return [r.request_id for r in s.waiting]


def test_flag_off_leaves_waiting_fcfs():
    now = time.time()
    reqs = [
        _req("big_cold", 90_000, now - 3),
        _req("small_warm", 92_000, now - 2),
        _req("mid_warm", 60_000, now - 1),
    ]
    warmth = {"small_warm": 90_000, "mid_warm": 50_000}
    s = _scheduler(reqs, warmth, producer=False)
    assert _order_seen_by_base_schedule(s) == ["big_cold", "small_warm", "mid_warm"]
    assert s._paw_sort_steps == 0


def test_sorts_by_uncached_tokens_before_base_schedule():
    now = time.time()
    reqs = [
        _req("big_cold", 90_000, now - 3),  # 90,000 uncached
        _req("small_warm", 92_000, now - 2),  # 2,000 uncached
        _req("mid_warm", 60_000, now - 1),  # 10,000 uncached
    ]
    warmth = {"small_warm": 90_000, "mid_warm": 50_000}
    s = _scheduler(reqs, warmth)
    assert _order_seen_by_base_schedule(s) == ["small_warm", "mid_warm", "big_cold"]
    assert s._paw_sort_reordered == 1
    # Already sorted: the next step leaves the queue alone.
    assert _order_seen_by_base_schedule(s) == ["small_warm", "mid_warm", "big_cold"]
    assert s._paw_sort_reordered == 1 and s._paw_sort_steps == 2


def test_aging_bounds_how_long_a_large_prompt_is_overtaken():
    now = time.time()
    reqs = [
        _req("old_big", 150_000, now - 60),  # 150,000 - 3,000 x 60 s < 0
        _req("fresh_small", 52_000, now - 1),  # 2,000 - 3,000 x 1 s
    ]
    warmth = {"fresh_small": 50_000}
    s = _scheduler(reqs, warmth)
    assert _order_seen_by_base_schedule(s) == ["fresh_small", "old_big"]
    reqs = [
        _req("old_big", 150_000, now - 60),
        _req("fresh_small", 52_000, now - 1),
    ]
    s = _scheduler(reqs, warmth, aging=3000)
    assert _order_seen_by_base_schedule(s) == ["old_big", "fresh_small"]


def test_started_requests_go_first():
    now = time.time()
    reqs = [
        _req("small", 12_000, now - 2),
        _req("resumed", 200_000, now - 1, num_computed_tokens=4_352),
    ]
    warmth = {"small": 10_000}
    s = _scheduler(reqs, warmth)
    assert _order_seen_by_base_schedule(s) == ["resumed", "small"]


def test_requests_that_do_not_fit_free_blocks_go_last():
    now = time.time()
    reqs = [
        _req("huge", 200_000, now - 100),  # 92 blocks of 2,176
        _req("small", 12_000, now - 1),  # 1 block
    ]
    s = _scheduler(reqs, {"small": 10_000}, aging=3000)
    s._get_num_free_blocks = lambda: 50
    assert _order_seen_by_base_schedule(s) == ["small", "huge"]


def test_the_fit_tier_counts_blocks_as_allocate_slots_does():
    """The fit tier compares the free blocks with the blocks vLLM's
    allocate_slots would take for the request (``_blocks_to_admit``: every KV
    cache group, the hit's free blocks, the lookahead, the watermark), not
    its uncached tokens over the block size."""
    now = time.time()
    reqs = [
        _req("small", 12_000, now - 2),  # 2,000 uncached: 1 block of 2,176
        _req("mid", 30_000, now - 1),  # 8,000 uncached
    ]
    s = _scheduler(reqs, {"small": 10_000, "mid": 22_000})
    s._get_num_free_blocks = lambda: 10
    # By token count both fit; by allocate_slots' count "small" does not.
    s._blocks_to_admit = lambda r: {"small": 11, "mid": 6}[r.request_id]
    assert _order_seen_by_base_schedule(s) == ["mid", "small"]


def _dram_small_estimator():
    """A connector that expects to load 88,000 tokens of "dram_small" from its
    host tier and nothing of any other request."""
    return SimpleNamespace(
        estimate_external_hit_tokens=lambda r, local: 88_000
        if r.request_id == "dram_small"
        else 0
    )


def test_external_hit_counts_when_it_fits():
    now = time.time()
    reqs = [
        _req("hbm_mid", 30_000, now - 2),  # 8,000 uncached
        _req("dram_small", 100_000, now - 1),  # 2,000 uncached after a DRAM load
    ]
    s = _scheduler(reqs, {"hbm_mid": 22_000, "dram_small": 10_000})
    s._hit_estimator = _dram_small_estimator()
    assert _order_seen_by_base_schedule(s) == ["dram_small", "hbm_mid"]
    # Without room for the loaded blocks the DRAM hit does not count.
    reqs = [_req("hbm_mid", 30_000, now - 2), _req("dram_small", 100_000, now - 1)]
    s = _scheduler(reqs, {"hbm_mid": 22_000, "dram_small": 10_000})
    s._hit_estimator = _dram_small_estimator()
    s.kv_cache_manager.block_pool = SimpleNamespace(get_num_free_blocks=lambda: 45)
    s._inflight_prefill_reserved_blocks = lambda: 30  # 15 left < 41 to load
    assert _order_seen_by_base_schedule(s) == ["hbm_mid", "dram_small"]


def test_the_blocks_retention_pins_are_no_room_for_a_host_tier_load():
    """vLLM refuses an async host-tier load for the async-load reservation
    before retention evicts (only a real shortage evicts), so the pinned
    blocks, which the HBM fit counts as free, do not count as room for the
    load: here the pool has 45 free blocks, 30 more are pinned, 10 are
    reserved, and the 41-block load does not fit the 35 left."""
    now = time.time()
    reqs = [_req("hbm_mid", 30_000, now - 2), _req("dram_small", 100_000, now - 1)]
    s = _scheduler(reqs, {"hbm_mid": 22_000, "dram_small": 10_000})
    s._hit_estimator = _dram_small_estimator()
    s.kv_cache_manager.block_pool = SimpleNamespace(get_num_free_blocks=lambda: 45)
    s._retention = SimpleNamespace(num_pinned_blocks=lambda: 30)
    s._inflight_prefill_reserved_blocks = lambda: 10
    assert s._get_num_free_blocks() == 75
    assert _order_seen_by_base_schedule(s) == ["hbm_mid", "dram_small"]
    # With the 30 blocks free instead of pinned the load fits and counts.
    s.kv_cache_manager.block_pool = SimpleNamespace(get_num_free_blocks=lambda: 75)
    s._retention = None
    assert _order_seen_by_base_schedule(s) == ["dram_small", "hbm_mid"]


def test_the_aging_reads_the_arrival_time_directly():
    """The sort key and the promotion's ranking both age a request by
    _wait_seconds. vLLM's Request always has arrival_time: a request without
    it is an error, not a request that never ages."""
    assert _wait_seconds(SimpleNamespace(arrival_time=10.0), 12.5) == 2.5
    assert _wait_seconds(SimpleNamespace(arrival_time=20.0), 12.5) == 0.0
    with pytest.raises(AttributeError, match="arrival_time"):
        _wait_seconds(SimpleNamespace(), 12.5)


def test_window_bounds_the_sort():
    now = time.time()
    reqs = [
        _req("a", 50_000, now - 4),
        _req("b", 3_000, now - 3),
        _req("c", 1_000, now - 2),  # outside a window of 2
    ]
    s = _scheduler(reqs, {}, window=2)
    assert _order_seen_by_base_schedule(s) == ["b", "a", "c"]


@pytest.mark.parametrize("quadratic", [False, True])
def test_quadratic_cost_prices_a_deep_prefix(quadratic):
    # 8,000 uncached tokens behind a 100,000-token hit attend over that
    # prefix: quadratic pricing makes them dearer than 9,000 cold tokens.
    now = time.time()
    reqs = [
        _req("deep_warm", 108_000, now - 2),  # 8,000 uncached behind 100,000
        _req("short_cold", 9_000, now - 1),  # 9,000 uncached, no prefix
    ]
    s = _scheduler(reqs, {"deep_warm": 100_000}, quadratic=quadratic)
    expected = ["short_cold", "deep_warm"] if quadratic else ["deep_warm", "short_cold"]
    assert _order_seen_by_base_schedule(s) == expected


def test_prefix_aware_scheduler_itself_does_not_sort():
    # A KV producer naming PrefixAwareScheduler gets the warmest-first
    # admission alone: only the pipeline chunk scheduler sorts.
    assert not hasattr(PrefixAwareScheduler, "_sort_waiting_queue")
    now = time.time()
    reqs = [
        _req("big_cold", 90_000, now - 3),
        _req("mid_warm", 91_000, now - 2),
        _req("small_warm", 92_000, now - 1),
    ]
    s = _prefix_aware_scheduler(reqs, {"mid_warm": 45_000, "small_warm": 90_000})
    seen = {}

    def base_schedule(*args, **kwargs):
        seen["order"] = [r.request_id for r in s.waiting]
        return SimpleNamespace()

    with patch.object(AsyncScheduler, "schedule", side_effect=base_schedule):
        s.schedule()
    # The promotion moved the warmest request to the head and left the rest
    # in arrival order; a sort would have put mid_warm before big_cold.
    assert seen["order"] == ["small_warm", "big_cold", "mid_warm"]


# ---------------- the connector's host-tier estimate ----------------


class _Estimator:
    def __init__(self, answer=88_000, fail=False, error=RuntimeError):
        self.answer = answer
        self.fail = fail
        self.error = error
        self.calls = []

    def estimate_external_hit_tokens(self, request, num_local_tokens):
        self.calls.append((request.request_id, num_local_tokens))
        if self.fail:
            raise self.error(f"store unreachable for {request.request_id}")
        return self.answer


def _warmth_scheduler(estimator, consumer=False):
    s = TpuPipelineChunkScheduler.__new__(TpuPipelineChunkScheduler)
    s._is_kv_consumer = consumer
    s._hit_estimator = estimator
    s.block_size = 2176
    return s


def test_the_estimate_needs_the_named_interface():
    # The pipeline chunk scheduler keeps a connector as its estimator only
    # through this interface (test_pp_scheduler_options checks its __init__).
    assert isinstance(_Estimator(), ExternalHitEstimator)
    assert not isinstance(SimpleNamespace(), ExternalHitEstimator)


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
    estimator = _Estimator(fail=True)
    s = _warmth_scheduler(estimator)
    # The module's logger resolves lazily to vLLM's logger of this name.
    log = logging.getLogger("vllm." + pp_chunk_scheduler.__name__)
    # The real warning_once runs; only the logger's output is captured. Its
    # process-wide cache starts empty: tests/conftest.py's autouse
    # _reset_warning_once clears it before and after every test.
    with (
        patch.object(log, "warning") as warn,
        patch.object(log, "debug") as debug,
    ):
        for rid in ("r-fails", "r-fails-too"):
            assert s._external_warmth(_req(rid, 100_000, time.time()), 0) == 0
        # Two failures of one kind, with different requests and messages:
        # one warning, naming the exception type; each failure in the debug
        # log with its request.
        warn.assert_called_once()
        assert "RuntimeError" in warn.call_args.args
        assert [c.args[1] for c in debug.call_args_list] == ["r-fails", "r-fails-too"]
        # A different exception type is a different failure: warned once more.
        estimator.error = ValueError
        assert s._external_warmth(_req("r-other", 100_000, time.time()), 0) == 0
        assert warn.call_count == 2 and "ValueError" in warn.call_args.args
