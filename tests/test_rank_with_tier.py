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

"""TPU_PREFIX_AWARE_RANK_WITH_TIER: the waiting queue counts a prompt's
prefix in the KV connector's tier as cached, as admission does."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.kv_offload.base import LookupResult, make_offload_key
from vllm.v1.request import RequestStatus

from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler

BLOCK = 16
HASHES_PER_CHUNK = 4
CHUNK = BLOCK * HASHES_PER_CHUNK


class FakeTier:
    """The parts of an offloading connector's scheduler side the ranking
    reads: its request states, its groups and its directory."""

    def __init__(self, stored_chunks: dict[str, int], sliding_window=None, eagle=False):
        self.stored_chunks = stored_chunks
        self.config = SimpleNamespace(
            kv_group_configs=[
                SimpleNamespace(
                    group_idx=0,
                    tokens_per_chunk=CHUNK,
                    hashes_per_chunk=HASHES_PER_CHUNK,
                    sliding_window_size_in_chunks=sliding_window,
                    is_eagle_group=eagle,
                )
            ]
        )
        self._req_status = {
            request_id: SimpleNamespace(req_context=request_id)
            for request_id in stored_chunks
        }
        self.lookups = 0
        self.manager = SimpleNamespace(lookup=self._lookup)

    def _lookup(self, key, req_context):
        self.lookups += 1
        request_id, index = key[:-4].decode().rsplit(":", 1)
        index = int(index)
        chunk = (index + 1) // HASHES_PER_CHUNK - 1
        if chunk < self.stored_chunks[request_id]:
            return LookupResult.HIT
        return LookupResult.MISS


def request(
    request_id: str,
    num_tokens: int,
    arrival_time: float,
    skip_reading_prefix_cache: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        num_tokens=num_tokens,
        arrival_time=arrival_time,
        num_preemptions=0,
        num_computed_tokens=0,
        status=RequestStatus.WAITING,
        skip_reading_prefix_cache=skip_reading_prefix_cache,
        block_hashes=[f"{request_id}:{i}".encode() for i in range(num_tokens // BLOCK)],
    )


def scheduler(
    requests: list[SimpleNamespace],
    hbm_tokens: dict[str, int],
    tier: FakeTier | None,
) -> PrefixAwareScheduler:
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    queue = FCFSRequestQueue()
    for r in requests:
        queue.add_request(r)
    s.waiting = queue
    s.policy = SchedulingPolicy.FCFS
    s.WINDOW = 64
    s.block_size = BLOCK
    s._paw_n = 0
    s._paw_cold_tokens = 0
    s._paw_prompt_tokens = 0
    s._paw_scoring_broken = False
    s._init_tier_warmth(False)
    s._uncached_by_request = {}
    s._tier = tier
    s._tier_prefix_by_request = {}

    by_hashes = {id(r.block_hashes): r.request_id for r in requests}
    coordinator = MagicMock()
    coordinator.find_longest_cache_hit.side_effect = lambda hashes, max_len: (
        {},
        hbm_tokens.get(by_hashes.get(id(hashes)), 0),
        0,
    )
    kvm = MagicMock()
    kvm.prefix_cache_lookup_enabled.side_effect = (
        lambda r: not r.skip_reading_prefix_cache
    )
    kvm.coordinator = coordinator
    s.kv_cache_manager = kvm
    return s


def returning_turn_and_short_prompt():
    returning = request("returning", 100 * CHUNK, arrival_time=0.0)
    short = request("short", 4 * CHUNK, arrival_time=1.0)
    return [returning, short], {"returning": 98, "short": 0}


def test_a_turn_whose_prefix_is_in_the_tier_ranks_as_cached():
    requests, stored = returning_turn_and_short_prompt()
    s = scheduler(requests, {}, FakeTier(stored))
    s._promote_warmest(s.waiting)
    assert s.waiting.peek_request().request_id == "returning"
    assert s._prefix_warmth(requests[0]) == 98 * CHUNK


def test_off_a_turn_whose_prefix_is_in_the_tier_ranks_as_uncached():
    requests, _ = returning_turn_and_short_prompt()
    s = scheduler(requests, {}, None)
    s._promote_warmest(s.waiting)
    assert s.waiting.peek_request().request_id == "short"
    assert s._prefix_warmth(requests[0]) == 0


def test_the_tier_extends_a_chunk_aligned_hbm_prefix():
    r = request("r", 10 * CHUNK, arrival_time=0.0)
    s = scheduler([r], {"r": 3 * CHUNK}, FakeTier({"r": 7}))
    assert s._prefix_warmth(r) == 7 * CHUNK


def test_an_hbm_prefix_ending_inside_a_chunk_reads_no_tier():
    r = request("r", 10 * CHUNK, arrival_time=0.0)
    tier = FakeTier({"r": 7})
    s = scheduler([r], {"r": 3 * CHUNK + BLOCK}, tier)
    assert s._prefix_warmth(r) == 3 * CHUNK + BLOCK
    assert tier.lookups == 0


def test_a_tier_hit_shorter_than_one_chunk_past_the_hbm_prefix_counts_nothing():
    r = request("r", 10 * CHUNK, arrival_time=0.0)
    s = scheduler([r], {"r": 3 * CHUNK}, FakeTier({"r": 3}))
    assert s._prefix_warmth(r) == 3 * CHUNK


def test_a_flagged_drafter_group_drops_its_last_tier_chunk():
    r = request("r", 10 * CHUNK, arrival_time=0.0)
    s = scheduler([r], {}, FakeTier({"r": 6}, eagle=True))
    assert s._prefix_warmth(r) == 5 * CHUNK


def test_the_raiden_store_is_read_with_one_side_effect_free_probe():
    r = request("r", 10 * CHUNK, arrival_time=0.0)
    probes = []
    tier = FakeTier({"r": 9})
    tier.manager = SimpleNamespace(
        lookup=lambda key, ctx: pytest.fail("the probe replaces per-key lookups"),
        probe_warm_chunks=lambda keys: probes.append(list(keys)) or 6,
    )
    s = scheduler([r], {"r": CHUNK}, tier)
    assert s._prefix_warmth(r) == 7 * CHUNK
    assert probes == [
        [
            make_offload_key(r.block_hashes[(chunk + 1) * HASHES_PER_CHUNK - 1], 0)
            for chunk in range(1, 9)
        ]
    ]


def test_a_tier_shorter_than_the_hbm_prefix_keeps_the_hbm_prefix():
    r = request("r", 10 * CHUNK, arrival_time=0.0)
    s = scheduler([r], {"r": 5 * CHUNK}, FakeTier({"r": 2}))
    assert s._prefix_warmth(r) == 5 * CHUNK


def test_the_tier_never_counts_the_last_prompt_token():
    r = request("r", 4 * CHUNK, arrival_time=0.0)
    s = scheduler([r], {}, FakeTier({"r": 4}))
    assert s._look_up_uncached(r) == CHUNK


def test_the_tier_is_read_once_per_stay_in_the_queue():
    requests, stored = returning_turn_and_short_prompt()
    tier = FakeTier(stored)
    s = scheduler(requests, {}, tier)
    s._promote_warmest(s.waiting)
    lookups = tier.lookups
    s._promote_warmest(s.waiting)
    s._look_up_uncached(requests[0])
    assert lookups > 0 and tier.lookups == lookups
    s.waiting.remove_request(requests[0])
    s._forget_requests_that_left_waiting()
    assert set(s._tier_prefix_by_request) == {"short"}


def test_a_prompt_that_skips_the_prefix_cache_reads_no_tier():
    r = request("r", 10 * CHUNK, arrival_time=0.0, skip_reading_prefix_cache=True)
    tier = FakeTier({"r": 9})
    s = scheduler([r], {}, tier)
    assert s._look_up_uncached(r) == r.num_tokens
    assert tier.lookups == 0


def test_a_request_the_connector_does_not_know_reads_no_tier():
    r = request("r", 10 * CHUNK, arrival_time=0.0)
    tier = FakeTier({})
    s = scheduler([r], {}, tier)
    assert s._look_up_uncached(r) == r.num_tokens


def test_the_lookup_keys_are_the_connectors_chunk_keys():
    r = request("r", 3 * CHUNK, arrival_time=0.0)
    keys = []
    tier = FakeTier({"r": 3})
    tier.manager = SimpleNamespace(
        lookup=lambda key, ctx: keys.append(key) or LookupResult.HIT
    )
    s = scheduler([r], {}, tier)
    s._look_up_uncached(r)
    assert keys == [
        make_offload_key(r.block_hashes[HASHES_PER_CHUNK - 1], 0),
        make_offload_key(r.block_hashes[2 * HASHES_PER_CHUNK - 1], 0),
    ]


def test_the_tier_is_found_among_a_multi_connectors_children():
    tier = FakeTier({})
    tier.__class__ = type("Tier", (FakeTier, OffloadingConnectorScheduler), {})
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s.connector = SimpleNamespace(
        _connectors=[
            SimpleNamespace(connector_scheduler=SimpleNamespace()),
            SimpleNamespace(connector_scheduler=tier),
        ]
    )
    assert s._connector_tier() is tier


@pytest.mark.parametrize("connector", [None, SimpleNamespace()])
def test_ranking_with_the_tier_needs_an_offloading_connector(connector):
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s.connector = (
        None if connector is None else SimpleNamespace(connector_scheduler=connector)
    )
    with pytest.raises(NotImplementedError, match="offloading KV connector"):
        s._connector_tier()


def short_prompts_first(tier: FakeTier | None) -> PrefixAwareScheduler:
    returning = request("returning", 100 * CHUNK, arrival_time=0.0)
    s = scheduler([returning], {}, tier)
    s.short_prompt_tokens = 4 * CHUNK
    s.shorts_take_oldest_chunk = True
    s.scheduler_config = SimpleNamespace(long_prefill_token_threshold=2 * CHUNK)
    s._pause_state = PauseState.UNPAUSED
    s.running = []
    s.max_num_running_reqs = 4
    s.num_waiting_for_streaming_input = 0
    return s


def test_a_turn_whose_prefix_is_in_the_tier_counts_as_a_short_prompt():
    s = short_prompts_first(FakeTier({"returning": 98}))
    assert s._short_waiting_chunks() == [2 * CHUNK]
    assert s._has_short_prompts()


def test_off_a_turn_whose_prefix_is_in_the_tier_counts_as_a_long_prompt():
    s = short_prompts_first(None)
    assert s._short_waiting_chunks() == []
    assert not s._has_short_prompts()
