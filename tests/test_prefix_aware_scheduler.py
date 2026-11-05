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


class _Probe:
    """An ExternalPrefixProbe connector with a fixed host-resident map."""

    def __init__(self, external: dict[str, int]):
        self.external = external
        self.calls: list[tuple[str, int]] = []

    def peek_external_prefix_tokens(self, request, num_local_tokens):
        self.calls.append((request.request_id, num_local_tokens))
        return self.external.get(request.request_id, 0)


def test_probe_protocol_recognized():
    from vllm_torchtpu.core.external_prefix_probe import ExternalPrefixProbe

    assert isinstance(_Probe({}), ExternalPrefixProbe)
    assert not isinstance(object(), ExternalPrefixProbe)


def test_connector_resident_prefix_counts_as_warm():
    now = time.time()
    r_local = _make_request("local", num_tokens=10_000, arrival_time=now - 10)
    r_host = _make_request("host", num_tokens=10_000, arrival_time=now - 5)
    s = _make_scheduler([r_local, r_host], {"local": 5_000, "host": 1_000})
    probe = _Probe({"host": 8_000})
    s._paw_probe = probe

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    # host: 1k local + 8k loadable = 9k warm beats local's 5k.
    assert [r.request_id for r in q] == ["host", "local"]
    assert ("host", 1_000) in probe.calls


def test_connector_probe_is_cached_and_clamped():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1_000, arrival_time=now - 10)
    r2 = _make_request("r2", num_tokens=1_000, arrival_time=now - 5)
    s = _make_scheduler([r1, r2], {"r1": 100, "r2": 0})
    probe = _Probe({"r2": 50_000})
    s._paw_probe = probe

    assert s._prefix_warmth(r2) == 1_000  # clamped to the prompt
    s._prefix_warmth(r2)
    assert len(probe.calls) == 1
    s._paw_step += 100  # refresh window passed
    s._prefix_warmth(r2)
    assert len(probe.calls) == 2


def test_connector_probe_failure_disables_probe():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1_000, arrival_time=now - 10)
    s = _make_scheduler([r1], {"r1": 100})

    class _Broken:
        def peek_external_prefix_tokens(self, request, num_local_tokens):
            raise RuntimeError("boom")

    s._paw_probe = _Broken()
    assert s._prefix_warmth(r1) == 100
    assert s._paw_probe is None


def test_aged_request_outranks_warmer_ones():
    now = time.time()
    r_old_cold = _make_request("old", num_tokens=100_000, arrival_time=now - 120)
    r_warm = _make_request("warm", num_tokens=1_000, arrival_time=now - 5)
    r_older_cold = _make_request("older", num_tokens=100_000, arrival_time=now - 300)
    s = _make_scheduler(
        [r_warm, r_old_cold, r_older_cold], {"warm": 1_000, "old": 0, "older": 0}
    )
    s._paw_max_wait_s = 60.0

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q][0] == "older"  # oldest aged first
    assert s._paw_aged == 1


def test_aging_disabled_keeps_warmest_first():
    now = time.time()
    r_old_cold = _make_request("old", num_tokens=100_000, arrival_time=now - 600)
    r_warm = _make_request("warm", num_tokens=1_000, arrival_time=now - 5)
    s = _make_scheduler([r_old_cold, r_warm], {"warm": 1_000, "old": 0})
    s._paw_max_wait_s = 0.0

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["warm", "old"]


def test_preempted_still_outranks_aged():
    now = time.time()
    r_aged = _make_request("aged", num_tokens=1_000, arrival_time=now - 600)
    r_pre = _make_request(
        "pre", num_tokens=1_000, arrival_time=now - 1, num_preemptions=1
    )
    s = _make_scheduler([r_aged, r_pre], {"aged": 0, "pre": 0})
    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["pre", "aged"]


def test_max_wait_from_additional_config():
    import pytest

    from vllm_torchtpu.core.prefix_aware_scheduler import (
        DEFAULT_PAW_MAX_WAIT_S,
        _max_wait_from_config,
    )

    assert _max_wait_from_config(None) == DEFAULT_PAW_MAX_WAIT_S
    cfg = SimpleNamespace(additional_config={})
    assert _max_wait_from_config(cfg) == DEFAULT_PAW_MAX_WAIT_S
    cfg = SimpleNamespace(additional_config={"paw_max_wait_s": 30})
    assert _max_wait_from_config(cfg) == 30.0
    cfg = SimpleNamespace(additional_config={"paw_max_wait_s": 0})
    assert _max_wait_from_config(cfg) == 0.0
    for bad in (-1, "x"):
        with pytest.raises(ValueError):
            _max_wait_from_config(
                SimpleNamespace(additional_config={"paw_max_wait_s": bad})
            )


def test_max_wait_key_excluded_from_compile_hash():
    from vllm_torchtpu.runner.utils import HASH_IGNORED_ADDITIONAL_CONFIG_KEYS

    assert "paw_max_wait_s" in HASH_IGNORED_ADDITIONAL_CONFIG_KEYS
