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

import pytest
from vllm.config import KVTransferConfig
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.request import RequestStatus

from vllm_torchtpu.core import prefix_change_log
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
    s._is_kv_consumer = False
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


def test_promotes_waiting_while_kv_holding_waiting_is_not_empty():
    now = time.time()
    r_cold = _make_request("cold", num_tokens=1000, arrival_time=now - 10)
    r_warm = _make_request("warm", num_tokens=1000, arrival_time=now - 5)
    s = _make_scheduler([r_cold, r_warm], {"cold": 0, "warm": 800})
    loading = _make_request("loading", num_computed_tokens=900)
    s.kv_holding_waiting.add_request(loading)

    q = s._select_waiting_queue_for_scheduling()
    assert q is s.kv_holding_waiting
    assert [r.request_id for r in s.waiting] == ["warm", "cold"]


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


def test_a_failing_memo_recheck_falls_back_to_fcfs(monkeypatch):
    """A cross-step memo reuse recomputes the lookup to verify it; if that
    lookup fails, warmth scoring falls back to FCFS as a fresh lookup does."""
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 10)
    r2 = _make_request("r2", num_tokens=1000, arrival_time=now - 5)
    s = _make_scheduler([r1, r2], {"r1": 0, "r2": 500}, raise_on_lookup=True)
    monkeypatch.setattr(s, "_get_prefix_cache_epoch", lambda: 2)
    monkeypatch.setattr(s, "_memo_still_exact", lambda cached, epoch: True)
    s._memo_xstep = True
    s._memo_verify_left = 64
    for r in (r1, r2):
        # A still-exact memo from epoch 1 for this request's lookup length.
        r._paw_hit_cache = (1, 999, ({}, 500, 0), 500, frozenset(), 1)

    q = s._select_waiting_queue_for_scheduling()
    assert [r.request_id for r in q] == ["r1", "r2"]
    assert s._paw_scoring_broken is True


def test_epoch_cache_reuses_longest_cache_hit_and_invalidates_on_epoch_change(
    monkeypatch,
):
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 10)
    r2 = _make_request("r2", num_tokens=1000, arrival_time=now - 5)
    s = _make_scheduler([r1, r2], {"r1": 128, "r2": 640})
    monkeypatch.setattr(prefix_change_log, "epoch_installed", lambda: True)

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
    s._is_kv_consumer = True
    s.running = list(running)
    s.waiting = FCFSRequestQueue()
    s.kv_holding_waiting = FCFSRequestQueue()
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

    # Decoders come first in stable order; prefilling requests keep FCFS order.
    assert seen["order"] == ["d1", "d2", "p1", "new", "p2"]
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
    """Decoders first leaves the step budget at max_num_scheduled_tokens
    however many requests decode."""
    running = [_running_request(f"d{i}", 100, 100) for i in range(24)]
    running.append(_running_request("p", 100_000, 0))
    s, seen = _make_step_scheduler(running, monkeypatch, budget=8192)
    s.schedule()
    assert seen["budget"] == 8192
    assert s.max_num_scheduled_tokens == 8192


def test_waiting_loop_stop_reason_and_admission_telemetry():
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s._is_kv_consumer = True
    s.running = []
    s.waiting = [object()]
    s.kv_holding_waiting = []
    s.max_num_scheduled_tokens = 8192
    s.max_num_active_reqs = 32
    s.num_waiting_for_streaming_input = 0
    s.deferred_waiting = set()
    s.block_size = 640
    out = SimpleNamespace(total_num_scheduled_tokens=640, num_scheduled_tokens={})

    s._step_wait_event = None
    assert s._waiting_loop_stop_reason(out, 1) == "other"

    # The first stop of a waiting request wins; running requests are ignored.
    s._note_waiting_stop(SimpleNamespace(status=RequestStatus.RUNNING), "align0")
    assert s._step_wait_event is None
    s._note_waiting_stop(SimpleNamespace(status=RequestStatus.WAITING), "lookahead0")
    s._note_waiting_stop(SimpleNamespace(status=RequestStatus.WAITING), "align0")
    assert s._waiting_loop_stop_reason(out, 1) == "lookahead0"

    s._step_wait_event = None
    assert s._waiting_loop_stop_reason(out, 1) == "other"
    full = SimpleNamespace(total_num_scheduled_tokens=8192, num_scheduled_tokens={})
    assert s._waiting_loop_stop_reason(full, 1) == "budget"
    # A skipped request back in `waiting` is deferred, not unreached.
    s.deferred_waiting = set(s.waiting)
    assert s._waiting_loop_stop_reason(out, 1) == "drained"
    s.deferred_waiting = set()
    s.running = [object()] * 32
    assert s._waiting_loop_stop_reason(out, 1) == "max_running"
    s.running = []
    s.waiting = []
    assert s._waiting_loop_stop_reason(out, 0) == "idle"
    s.kv_holding_waiting = [object()]
    assert s._waiting_loop_stop_reason(out, 0) == "drained"
    s.kv_holding_waiting = []

    s.waiting = [object()]
    s._record_step_telemetry(out, 3, "lookahead0")
    s._record_step_telemetry(out, 3, "lookahead0")
    s._record_step_telemetry(full, 0, "idle")
    # Every waiting request got in: a short step is not a small step then.
    s.waiting = []
    s._record_step_telemetry(out, 2, "drained")
    text = s._admission_telemetry("lookahead0")
    assert "stop=lookahead0" in text
    assert "min_steps=4 min_small_steps_waiting=2" in text
    assert "min_stops=lookahead0:2,drained:1" in text
    # The per-minute counters restart after each telemetry line.
    text = s._admission_telemetry("idle")
    assert "min_steps=0" in text and "min_stops=-" in text


def test_local_prefix_hit_forwards_max_cache_hit_length(monkeypatch):
    """vLLM 0.33 passes ``max_cache_hit_length`` (an aux-output connector's
    bound); the override must take it, pass it on, and never reuse a memo
    computed for a different bound."""
    from vllm.v1.core.sched.scheduler import Scheduler

    seen = []

    def base_hit(self, request, max_cache_hit_length=None):
        seen.append(max_cache_hit_length)
        return "blocks", 7, 0, False

    monkeypatch.setattr(Scheduler, "_get_local_prefix_cache_hit", base_hit)
    monkeypatch.setattr(prefix_change_log, "epoch_installed", lambda: True)
    r = _make_request("r", num_tokens=1000)
    s = _make_scheduler([r], {"r": 640})
    s.connector = None
    pool = SimpleNamespace(cached_block_hash_to_block={}, _tpu_prefix_cache_epoch=3)
    s.kv_cache_manager.block_pool = pool
    s.kv_cache_manager.enable_kv_cache_events = False
    s.kv_cache_manager.prefix_cache_lookup_enabled = lambda req: True
    s.kv_cache_manager.create_kv_cache_blocks = lambda blocks: blocks

    # A memo for the default bound (num_tokens - 1) exists ...
    assert s._prefix_warmth(r) == 640
    # ... so the default lookup reuses it without calling the base ...
    assert s._get_local_prefix_cache_hit(r)[1] == 640
    assert seen == []
    # ... but a lookup with another bound goes to the base with that bound.
    assert s._get_local_prefix_cache_hit(r, 500)[1] == 7
    assert seen == [500]


def _init_with(monkeypatch, kv_transfer_config, calls=None):
    def fake_init(self, *args, **kwargs):
        self.vllm_config = SimpleNamespace(kv_transfer_config=kv_transfer_config)
        self.connector = None if kv_transfer_config is None else object()
        self.kv_cache_manager = MagicMock()

    calls = [] if calls is None else calls
    monkeypatch.setattr(AsyncScheduler, "__init__", fake_init)
    monkeypatch.setattr(
        PrefixAwareScheduler, "_enable_memo_xstep", lambda self: calls.append("memo")
    )
    return PrefixAwareScheduler()


def _kv_transfer_config(role):
    if role is None:
        return None
    return KVTransferConfig(kv_connector="TPURaidenConnector", kv_role=role)


@pytest.mark.parametrize(
    ("role", "consumer"),
    [(None, False), ("kv_producer", False), ("kv_both", False), ("kv_consumer", True)],
)
def test_decode_leg_levers_only_on_a_kv_consumer(monkeypatch, role, consumer):
    def fail():
        raise AssertionError("the scheduler must not install patches")

    monkeypatch.setattr(prefix_change_log, "install_epoch", fail)
    monkeypatch.setattr(prefix_change_log, "install", fail)
    calls = []
    s = _init_with(monkeypatch, _kv_transfer_config(role), calls)
    assert s._is_kv_consumer is consumer
    assert calls == (["memo"] if consumer else [])


@pytest.mark.parametrize("role", [None, "kv_producer", "kv_both"])
def test_off_a_decode_leg_schedule_is_the_warmest_first_admission(monkeypatch, role):
    """No decoders first, only the first non-empty waiting queue promoted, no
    packed prompt ids and no resend trim."""
    from vllm.v1.core.sched.output import CachedRequestData

    s = _init_with(monkeypatch, _kv_transfer_config(role))
    now = time.time()
    queued = [
        _make_request("cold", arrival_time=now - 10),
        _make_request("warm", arrival_time=now - 5),
    ]
    holding = [
        _make_request("hold_cold", arrival_time=now - 10),
        _make_request("hold_warm", arrival_time=now - 5),
    ]
    template = _make_scheduler(queued + holding, {"warm": 800, "hold_warm": 800})
    s.kv_cache_manager = template.kv_cache_manager
    s.waiting = template.waiting
    s.kv_holding_waiting = FCFSRequestQueue()
    for r in holding:
        s.waiting.remove(r)
        s.kv_holding_waiting.add_request(r)
    s.policy = SchedulingPolicy.FCFS
    s._paw_scoring_broken = False
    s._paw_n = s._paw_cold_tokens = s._paw_prompt_tokens = 0
    s.running = [
        _running_request("p1", prompt=10_000, computed=640),
        _running_request("d1", prompt=500, computed=501),
    ]
    prompt = list(range(4096))
    new_req = SimpleNamespace(
        prompt_token_ids=prompt,
        mm_features=None,
        prompt_embeds=None,
        prompt_is_token_ids=None,
        sampling_params=SimpleNamespace(
            thinking_token_budget=None, prompt_logprobs=None, extra_args=None
        ),
    )
    out = SimpleNamespace(scheduled_new_reqs=[new_req])
    seen = {}

    def fake_schedule(self, *args, **kwargs):
        seen["running"] = [r.request_id for r in self.running]
        return out

    def fake_cached(self, *args, **kwargs):
        return CachedRequestData(
            req_ids=["a"],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={"a": [1, 2, 3, 4, 5]},
            new_block_ids=[None],
            num_computed_tokens=[3],
            num_output_tokens=[2],
        )

    monkeypatch.setattr(AsyncScheduler, "schedule", fake_schedule, raising=False)
    monkeypatch.setattr(AsyncScheduler, "_make_cached_request_data", fake_cached)
    assert s.schedule() is out
    assert seen["running"] == ["p1", "d1"]
    assert [r.request_id for r in s.kv_holding_waiting] == ["hold_warm", "hold_cold"]
    assert [r.request_id for r in s.waiting] == ["cold", "warm"]
    assert new_req.prompt_token_ids is prompt
    cached = s._make_cached_request_data([], [], {}, {}, {})
    assert cached.all_token_ids == {"a": [1, 2, 3, 4, 5]}
