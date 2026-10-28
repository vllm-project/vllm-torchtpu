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
from unittest.mock import MagicMock

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
    tier_warmth: bool = False,
    connector: object | None = None,
) -> PrefixAwareScheduler:
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    q = FCFSRequestQueue()
    for r in requests:
        q.add_request(r)
    s.waiting = q
    s.skipped_waiting = FCFSRequestQueue()
    s.policy = SchedulingPolicy.FCFS
    s.WINDOW = window
    s._paw_n = 0
    s._paw_cold_tokens = 0
    s._paw_prompt_tokens = 0
    s._paw_scoring_broken = False
    s._init_tier_warmth(tier_warmth)
    s.connector = connector

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


# --- Offload-tier warmth (TPU_PREFIX_AWARE_TIER_WARMTH) ---------------------

_CHUNK_TOKENS = 4096
_HASHES_PER_CHUNK = 16
_HASH_TOKENS = _CHUNK_TOKENS // _HASHES_PER_CHUNK


def _hashed_request(request_id: str, num_tokens: int, arrival_time: float):
    r = _make_request(request_id, num_tokens=num_tokens, arrival_time=arrival_time)
    r.block_hashes = [
        f"{request_id}:{i}".encode() for i in range(num_tokens // _HASH_TOKENS)
    ]
    return r


def _chunk_key(request: SimpleNamespace, chunk: int) -> bytes:
    from vllm.v1.kv_offload.base import make_offload_key

    return make_offload_key(
        request.block_hashes[(chunk + 1) * _HASHES_PER_CHUNK - 1], 0
    )


class _FakeTierManager:
    """Answers lookups from a key set; records how often it was asked."""

    def __init__(self, keys=(), pending=(), fail=False):
        self.keys = set(keys)
        self.pending = set(pending)
        self.fail = fail
        self.calls = 0
        self.local_only_calls = 0

    def lookup(self, key, req_context=None, *, local_only: bool = False):
        from vllm.v1.kv_offload.base import LookupResult

        self.calls += 1
        if local_only:
            self.local_only_calls += 1
        if self.fail:
            raise RuntimeError("simulated store failure")
        if key in self.pending:
            return LookupResult.HIT_PENDING
        return LookupResult.HIT if key in self.keys else LookupResult.MISS


def _offload_connector(
    manager,
    num_groups: int = 1,
    multi: bool = True,
    is_eagle_group: bool = False,
):
    from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingScheduler

    group = SimpleNamespace(
        group_idx=0,
        tokens_per_chunk=_CHUNK_TOKENS,
        hashes_per_chunk=_HASHES_PER_CHUNK,
        is_eagle_group=is_eagle_group,
    )
    sched = TPURaidenOffloadingScheduler.__new__(TPURaidenOffloadingScheduler)
    sched._raiden_manager = manager
    sched.config = SimpleNamespace(kv_group_configs=[group] * num_groups)
    offload = SimpleNamespace(connector_scheduler=sched)
    if not multi:
        return offload
    # A P/D producer child first, as in TPUMultiConnector configs.
    return SimpleNamespace(_connectors=[SimpleNamespace(), offload])


def test_default_knobs():
    assert PrefixAwareScheduler.WINDOW == 64


def test_promotes_warmest_within_window():
    now = time.time()
    r_cold = _make_request("cold", num_tokens=1000, arrival_time=now - 10)
    r_warm = _make_request("warm", num_tokens=1000, arrival_time=now - 5)
    s = _make_scheduler([r_cold, r_warm], {"cold": 0, "warm": 800})

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    order = [r.request_id for r in q]
    assert order == ["warm", "cold"]


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


def _tier_resident_and_warm():
    now = time.time()
    # HBM evicted this 25-chunk session; the tier holds its first 24 chunks,
    # so it needs ~4k tokens of compute.
    resident = _hashed_request("resident", num_tokens=25 * 4096, arrival_time=now - 10)
    # 20k tokens, 16k of them in HBM: 4k uncached, a bit more than `resident`
    # once the tier counts.
    warm = _hashed_request("warm", num_tokens=20_480 + 512, arrival_time=now - 5)
    manager = _FakeTierManager(keys=[_chunk_key(resident, c) for c in range(24)])
    return resident, warm, manager, {"resident": 0, "warm": 16_384}


def test_tier_warmth_off_ranks_on_hbm_only():
    resident, warm, manager, hbm = _tier_resident_and_warm()
    s = _make_scheduler([resident, warm], hbm, connector=_offload_connector(manager))

    q = s._select_waiting_queue_for_scheduling()
    assert [r.request_id for r in q] == ["warm", "resident"]
    assert manager.calls == 0


def test_tier_warmth_promotes_tier_resident_request():
    resident, warm, manager, hbm = _tier_resident_and_warm()
    s = _make_scheduler(
        [warm, resident], hbm, tier_warmth=True, connector=_offload_connector(manager)
    )

    q = s._select_waiting_queue_for_scheduling()
    assert [r.request_id for r in q] == ["resident", "warm"]
    assert s._prefix_warmth(resident) == 24 * 4096


def test_tier_probe_stops_at_first_miss_and_counts_pending_stores():
    now = time.time()
    r = _hashed_request("r", num_tokens=10 * 4096, arrival_time=now)
    manager = _FakeTierManager(
        keys=[_chunk_key(r, 0), _chunk_key(r, 1), _chunk_key(r, 4)],
        pending=[_chunk_key(r, 2)],
    )
    s = _make_scheduler(
        [r], {"r": 0}, tier_warmth=True, connector=_offload_connector(manager)
    )

    # Chunks 0-2 (one still being stored), then chunk 3 misses.
    assert s._prefix_warmth(r) == 3 * 4096


def test_tier_probe_starts_at_the_hbm_hit():
    now = time.time()
    r = _hashed_request("r", num_tokens=10 * 4096, arrival_time=now)
    manager = _FakeTierManager(keys=[_chunk_key(r, c) for c in (2, 3, 4)])
    s = _make_scheduler(
        [r], {"r": 2 * 4096}, tier_warmth=True, connector=_offload_connector(manager)
    )

    assert s._prefix_warmth(r) == 5 * 4096


def test_tier_probe_needs_a_chunk_aligned_hbm_hit():
    now = time.time()
    r = _hashed_request("r", num_tokens=10 * 4096, arrival_time=now)
    manager = _FakeTierManager(keys=[_chunk_key(r, c) for c in range(10)])
    s = _make_scheduler(
        [r], {"r": 4096 + 256}, tier_warmth=True, connector=_offload_connector(manager)
    )

    assert s._prefix_warmth(r) == 4096 + 256
    assert manager.calls == 0


def test_tier_probe_skips_nearly_warm_requests():
    now = time.time()
    r = _hashed_request("r", num_tokens=10 * 4096, arrival_time=now)
    manager = _FakeTierManager(keys=[_chunk_key(r, c) for c in range(10)])
    s = _make_scheduler(
        [r], {"r": 8 * 4096}, tier_warmth=True, connector=_offload_connector(manager)
    )

    # One whole chunk left past the hit: below TIER_PROBE_MIN_CHUNKS.
    assert s._prefix_warmth(r) == 8 * 4096
    assert manager.calls == 0


def test_tier_probe_is_cached_until_the_hbm_hit_changes():
    now = time.time()
    r = _hashed_request("r", num_tokens=10 * 4096, arrival_time=now)
    manager = _FakeTierManager(keys=[_chunk_key(r, c) for c in range(4)])
    hbm = {"r": 0}
    s = _make_scheduler(
        [r], hbm, tier_warmth=True, connector=_offload_connector(manager)
    )

    assert s._prefix_warmth(r) == 4 * 4096
    calls = manager.calls
    assert s._prefix_warmth(r) == 4 * 4096
    assert manager.calls == calls
    hbm["r"] = 4096
    assert s._prefix_warmth(r) == 4 * 4096
    assert manager.calls > calls


def test_tier_probe_failure_falls_back_to_hbm_only():
    now = time.time()
    r = _hashed_request("r", num_tokens=10 * 4096, arrival_time=now)
    manager = _FakeTierManager(fail=True)
    s = _make_scheduler(
        [r], {"r": 4096}, tier_warmth=True, connector=_offload_connector(manager)
    )

    assert s._prefix_warmth(r) == 4096
    assert s._tier_probe_broken is True
    assert s._prefix_warmth(r) == 4096
    assert manager.calls == 1


def test_tier_warmth_needs_a_single_group_offloading_connector():
    now = time.time()
    r = _hashed_request("r", num_tokens=10 * 4096, arrival_time=now)
    keys = [_chunk_key(r, c) for c in range(10)]

    no_connector = _make_scheduler([r], {"r": 0}, tier_warmth=True)
    assert no_connector._prefix_warmth(r) == 0

    plain = _make_scheduler(
        [r], {"r": 0}, tier_warmth=True, connector=SimpleNamespace()
    )
    assert plain._prefix_warmth(r) == 0

    two_groups = _make_scheduler(
        [r],
        {"r": 0},
        tier_warmth=True,
        connector=_offload_connector(_FakeTierManager(keys=keys), num_groups=2),
    )
    assert two_groups._prefix_warmth(r) == 0

    single = _make_scheduler(
        [r],
        {"r": 0},
        tier_warmth=True,
        connector=_offload_connector(_FakeTierManager(keys=keys), multi=False),
    )
    # A hit covers at most num_tokens - 1 tokens: 9 whole chunks.
    assert single._prefix_warmth(r) == 9 * 4096


def test_tier_probe_accounts_for_flagged_eagle_group_tail_pop():
    now = time.time()
    r = _hashed_request("r", num_tokens=10 * 4096, arrival_time=now)
    # 3 chunks in tier -> 2 chunks usable when is_eagle_group=True.
    manager = _FakeTierManager(keys=[_chunk_key(r, c) for c in range(3)])
    s = _make_scheduler(
        [r],
        {"r": 0},
        tier_warmth=True,
        connector=_offload_connector(manager, is_eagle_group=True),
    )
    assert s._prefix_warmth(r) == 2 * 4096

    # 1 chunk in tier -> dropped to 0 when is_eagle_group=True.
    one_chunk = _FakeTierManager(keys=[_chunk_key(r, 0)])
    s_one = _make_scheduler(
        [r],
        {"r": 0},
        tier_warmth=True,
        connector=_offload_connector(one_chunk, is_eagle_group=True),
    )
    assert s_one._prefix_warmth(r) == 0


def test_tier_probe_caps_chunks_and_requests_local_only():
    now = time.time()
    r = _hashed_request("r", num_tokens=100 * 4096, arrival_time=now)
    manager = _FakeTierManager(keys=[_chunk_key(r, c) for c in range(100)])
    s = _make_scheduler(
        [r], {"r": 0}, tier_warmth=True, connector=_offload_connector(manager)
    )
    assert s._prefix_warmth(r) == s.TIER_PROBE_MAX_CHUNKS * 4096
    assert manager.calls == s.TIER_PROBE_MAX_CHUNKS
    assert manager.local_only_calls == s.TIER_PROBE_MAX_CHUNKS


def test_paw_tier_tokens_excludes_already_computed_requests():
    now = time.time()
    r1 = _hashed_request("r1", num_tokens=10 * 4096, arrival_time=now - 10)
    r2 = _hashed_request("r2", num_tokens=10 * 4096, arrival_time=now - 5)
    manager = _FakeTierManager(keys=[_chunk_key(r1, c) for c in range(4)])
    s = _make_scheduler(
        [r1, r2],
        {"r1": 0, "r2": 0},
        tier_warmth=True,
        connector=_offload_connector(manager),
    )
    # First step probes r1 (4 chunks = 16384 tier tokens).
    s._select_waiting_queue_for_scheduling()
    assert s._paw_tier_tokens == 4 * 4096

    # Once r1 has num_computed_tokens > 0, it does not re-credit cached tier tokens.
    r1.num_computed_tokens = 9 * 4096
    s._select_waiting_queue_for_scheduling()
    assert s._paw_tier_tokens == 4 * 4096
