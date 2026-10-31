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
"""Tests for the HBM KV cache memory metrics.

See `vllm_torchtpu.distributed.kv_transfer.kv_cache_memory_stats`. The
collector runs against a real `KVCacheManager` and a scheduler stand-in that
has the tables and queues it reads.
"""

from collections import deque
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from prometheus_client import CollectorRegistry, Counter, Gauge
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu.distributed.kv_transfer.kv_cache_memory_stats import (
    ACTIVE_PRIVATE,
    ACTIVE_SHARED,
    BLOCK_STATES,
    CACHED_EVICTABLE,
    EXCEEDS_POOL_CAPACITY,
    FREE_UNHASHED,
    HEAD_OF_LINE,
    HELD_FOR_SEND,
    INFLIGHT_LOAD_RESERVE,
    INSUFFICIENT_FREE_BLOCKS,
    LOADING_KV,
    ORPHAN_SUSPECT,
    STATS_BLOCKS_BY_STATE,
    STATS_ORPHAN_BLOCKS,
    STATS_REFUSALS,
    STATS_RESERVED_BLOCKS,
    STATS_TOKEN_SLOTS,
    TRANSIENT_PIN,
    WATERMARK,
    KVAdmissionRefusalTracker,
    KVCacheMemoryCollector,
    KVCacheMemoryProm,
    classify_refusal,
)
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPUConnector
from vllm_torchtpu.distributed.kv_transfer.tpu_connector_stats import (
    TpuKVConnectorStats,
)

BLOCK_SIZE = 16


@pytest.mark.parametrize(
    ("num_required", "num_free", "watermark", "reserved", "reason"),
    [
        (20, 30, 0, 0, EXCEEDS_POOL_CAPACITY),  # capacity is 10
        (8, 5, 0, 0, INSUFFICIENT_FREE_BLOCKS),
        (4, 5, 2, 0, WATERMARK),
        (4, 9, 2, 4, INFLIGHT_LOAD_RESERVE),
    ],
)
def test_classify_refusal(num_required, num_free, watermark, reserved, reason):
    assert (
        classify_refusal(num_required, num_free, watermark, reserved, capacity=10)
        == reason
    )


def test_refusals_count_once_per_episode():
    tracker = KVAdmissionRefusalTracker()
    tracker.record_refusal("a", INSUFFICIENT_FREE_BLOCKS, num_queued_behind=2)
    tracker.record_refusal("a", INSUFFICIENT_FREE_BLOCKS, num_queued_behind=3)
    assert tracker.take_counts() == {INSUFFICIENT_FREE_BLOCKS: 1, HEAD_OF_LINE: 2}

    # A new reason for the same request is a new episode, not a new head.
    tracker.record_refusal("a", WATERMARK, num_queued_behind=3)
    assert tracker.take_counts() == {WATERMARK: 1}

    tracker.record_admission("a")
    tracker.record_refusal("a", WATERMARK, num_queued_behind=1)
    assert tracker.take_counts() == {WATERMARK: 1, HEAD_OF_LINE: 1}

    # A request that is gone ends its episode.
    tracker.prune(live_req_ids=())
    tracker.record_refusal("a", WATERMARK, num_queued_behind=0)
    assert tracker.take_counts() == {WATERMARK: 1}


class _Scheduler:
    """The scheduler tables and queues `KVCacheMemoryCollector` reads."""

    def __init__(self, num_blocks: int):
        init_none_hash(sha256)
        spec = FullAttentionSpec(
            block_size=BLOCK_SIZE, num_kv_heads=1, head_size=64, dtype=torch.float32
        )
        config = KVCacheConfig(
            num_blocks=num_blocks,  # Block 0 is the null block.
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
        )
        self.kv_cache_manager = KVCacheManager(
            config,
            max_model_len=1024,
            scheduler_block_size=BLOCK_SIZE,
            hash_block_size=BLOCK_SIZE,
            enable_caching=True,
        )
        self.requests: dict[str, Request] = {}
        self.running: list[Request] = []
        self.waiting: list[Request] = []
        self.kv_holding_waiting: list[Request] = []
        self.deferred_frees: deque = deque()
        self.finished_recving_kv_req_ids: set[str] = set()
        self.inflight_reserved_blocks = 0

    def _inflight_prefill_reserved_blocks(self) -> int:
        return self.inflight_reserved_blocks

    def finish_requests(self, request_ids, finished_status):
        for req_id in request_ids:
            self.requests[req_id].status = finished_status
        return []

    def add(self, req_id: str, num_tokens: int, start: int = 0) -> Request:
        request = Request(
            req_id,
            list(range(start, start + num_tokens)),
            SamplingParams(max_tokens=1),
            None,
            block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
        )
        self.requests[req_id] = request
        self.waiting.append(request)
        return request

    def schedule(self, request: Request) -> bool:
        """Admits `request` and computes its prompt, as one scheduler step."""
        manager = self.kv_cache_manager
        blocks, num_hits = manager.get_computed_blocks(request)[:2]
        num_new = request.num_tokens - num_hits
        if manager.allocate_slots(request, num_new, num_hits, blocks) is None:
            return False
        self.waiting.remove(request)
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = request.num_tokens
        self.running.append(request)
        return True


def _collector(scheduler: _Scheduler) -> KVCacheMemoryCollector:
    collector = KVCacheMemoryCollector(scheduler, interval_s=0)  # type: ignore[arg-type]
    return collector


def _record(collector: KVCacheMemoryCollector) -> dict:
    stats = TpuKVConnectorStats()
    collector.record(stats)
    return stats.data


def _states(data: dict, scheduler: _Scheduler) -> dict[str, int]:
    counts = dict(data[STATS_BLOCKS_BY_STATE][-1])
    usable = scheduler.kv_cache_manager.block_pool.num_gpu_blocks - 1
    assert sum(counts.values()) == usable
    return {state: n for state, n in counts.items() if n}


def test_admission_refusals_and_head_of_line():
    # 3 usable blocks; each request needs 2.
    scheduler = _Scheduler(num_blocks=4)
    collector = _collector(scheduler)
    first, second, _ = (scheduler.add(f"r{i}", 32, start=100 * i) for i in range(3))
    assert scheduler.schedule(first)
    assert not scheduler.schedule(second)
    data = _record(collector)
    # `second` heads the queue with one request behind it.
    assert dict(data[STATS_REFUSALS]) == {INSUFFICIENT_FREE_BLOCKS: 1, HEAD_OF_LINE: 1}

    # Retrying the head is the same episode.
    assert not scheduler.schedule(second)
    assert _record(collector)[STATS_REFUSALS] == []


def test_request_larger_than_the_pool():
    scheduler = _Scheduler(num_blocks=4)
    collector = _collector(scheduler)
    assert not scheduler.schedule(scheduler.add("big", 4 * BLOCK_SIZE))
    assert dict(_record(collector)[STATS_REFUSALS]) == {EXCEEDS_POOL_CAPACITY: 1}


def test_running_requests_are_not_counted():
    scheduler = _Scheduler(num_blocks=4)
    collector = _collector(scheduler)
    request = scheduler.add("r", 32)
    request.status = RequestStatus.RUNNING  # Refusals here are preemptions.
    scheduler.kv_cache_manager.allocate_slots(request, 4 * BLOCK_SIZE)
    assert _record(collector)[STATS_REFUSALS] == []


def test_reserved_blocks():
    scheduler = _Scheduler(num_blocks=10)
    collector = _collector(scheduler)
    scheduler.inflight_reserved_blocks = 3
    assert _record(collector)[STATS_RESERVED_BLOCKS] == [
        [["watermark", 0], ["inflight_load", 3]]
    ]


def test_token_slots():
    scheduler = _Scheduler(num_blocks=10)
    collector = _collector(scheduler)
    assert scheduler.schedule(scheduler.add("r", 20))
    slots = {kind: n for _, kind, n in _record(collector)[STATS_TOKEN_SLOTS][-1]}
    assert slots == {"filled": 20, "tail_slack": 12, "reserved": 0}


def test_block_states():
    scheduler = _Scheduler(num_blocks=20)
    collector = _collector(scheduler)
    first = scheduler.add("first", 64)
    second = scheduler.add("second", 64)  # Same prompt.
    assert scheduler.schedule(first)
    assert _states(_record(collector), scheduler) == {
        ACTIVE_PRIVATE: 4,
        FREE_UNHASHED: 15,
    }

    # The second request shares the first one's cached prefix blocks.
    assert scheduler.schedule(second)
    assert _states(_record(collector), scheduler)[ACTIVE_SHARED] > 0

    # Finished requests whose blocks are kept, e.g. for a P2P send, hold them.
    first.status = RequestStatus.FINISHED_LENGTH_CAPPED
    second.status = RequestStatus.FINISHED_LENGTH_CAPPED
    assert set(_states(_record(collector), scheduler)) == {
        HELD_FOR_SEND,
        FREE_UNHASHED,
    }

    # Once freed, full blocks stay cached.
    for request in (first, second):
        scheduler.kv_cache_manager.free(request)
    assert set(_states(_record(collector), scheduler)) == {
        CACHED_EVICTABLE,
        FREE_UNHASHED,
    }


def test_loading_blocks_stay_loading_after_an_abort():
    scheduler = _Scheduler(num_blocks=10)
    collector = _collector(scheduler)
    request = scheduler.add("r", 32)
    assert scheduler.schedule(request)
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    assert LOADING_KV in _states(_record(collector), scheduler)

    # Aborted mid-load: its blocks stay held until the load ends.
    scheduler.finish_requests(["r"], RequestStatus.FINISHED_ABORTED)
    assert LOADING_KV in _states(_record(collector), scheduler)


def test_deferred_frees_are_transient_pins():
    scheduler = _Scheduler(num_blocks=10)
    collector = _collector(scheduler)
    blocks = scheduler.kv_cache_manager.block_pool.get_new_blocks(2)
    scheduler.deferred_frees.append((0, blocks))
    assert _states(_record(collector), scheduler)[TRANSIENT_PIN] == 2


def test_orphan_reported_after_two_scans():
    scheduler = _Scheduler(num_blocks=20)
    collector = _collector(scheduler)
    pool = scheduler.kv_cache_manager.block_pool
    leaked = pool.get_new_blocks(1)[0]

    data = _record(collector)
    assert _states(data, scheduler)[ORPHAN_SUSPECT] == 1
    assert data[STATS_ORPHAN_BLOCKS] == [[]]
    assert _record(collector)[STATS_ORPHAN_BLOCKS] == [[[leaked.block_id, "none"]]]

    pool.free_blocks([leaked])
    assert _record(collector)[STATS_ORPHAN_BLOCKS] == [[]]


def test_scan_waits_for_the_interval():
    scheduler = _Scheduler(num_blocks=10)
    collector = KVCacheMemoryCollector(scheduler, interval_s=3600)  # type: ignore[arg-type]
    assert _record(collector)[STATS_BLOCKS_BY_STATE] != []
    assert _record(collector)[STATS_BLOCKS_BY_STATE] == []


def test_connector_drains_the_collector():
    scheduler = _Scheduler(num_blocks=10)
    connector = SimpleNamespace()
    TPUConnector.bind_scheduler(connector, scheduler)  # type: ignore[arg-type]
    connector._kv_cache_memory._scanner.interval_s = 0

    stats = TPUConnector._take_scheduler_stats(connector)  # type: ignore[arg-type]

    assert stats is not None
    assert stats.data[STATS_RESERVED_BLOCKS] == [
        [["watermark", 0], ["inflight_load", 0]]
    ]
    assert stats.data[STATS_BLOCKS_BY_STATE] != []


def test_prometheus_metrics():
    registry = CollectorRegistry()
    prom = KVCacheMemoryProm(
        partial(Gauge, registry=registry),
        partial(Counter, registry=registry),
        ["model_name", "engine"],
        {0: ["m", "0"]},
    )
    labels = {"model_name": "m", "engine": "0"}

    def value(name: str, **extra: str) -> float | None:
        return registry.get_sample_value(name, {**labels, **extra})

    # Fixed series exist before any event.
    assert value("vllm:kv_admission_refusals_total", reason=HEAD_OF_LINE) == 0
    assert all(
        value("vllm:kv_cache_blocks_by_state", kv_group="all", state=state) == 0
        for state in BLOCK_STATES
    )

    # As shipped from the engine: lists, the newest snapshot last.
    prom.observe(
        {
            STATS_REFUSALS: [[WATERMARK, 2]],
            STATS_RESERVED_BLOCKS: [
                [["watermark", 1], ["inflight_load", 2]],
                [["watermark", 3], ["inflight_load", 5]],
            ],
            STATS_TOKEN_SLOTS: [[["0", "filled", 70], ["0", "tail_slack", 10]]],
            STATS_BLOCKS_BY_STATE: [[[ACTIVE_PRIVATE, 4], [FREE_UNHASHED, 5]]],
            STATS_ORPHAN_BLOCKS: [[[7, "req-1"]]],
        }
    )
    prom.observe({STATS_REFUSALS: [[WATERMARK, 1]]})

    assert value("vllm:kv_admission_reserved_blocks", kind="inflight_load") == 5
    assert value("vllm:kv_admission_refusals_total", reason=WATERMARK) == 3
    assert value("vllm:kv_token_slots", kv_group="0", kind="tail_slack") == 10
    # Block states keep the last scan between scans.
    assert (
        value("vllm:kv_cache_blocks_by_state", kv_group="all", state=ACTIVE_PRIVATE)
        == 4
    )
    orphan = {"kv_group": "all", "block_id": "7", "last_owner": "req-1"}
    assert value("vllm:kv_orphan_block_info", **orphan) == 1

    prom.observe({STATS_ORPHAN_BLOCKS: [[]]})
    assert value("vllm:kv_orphan_block_info", **orphan) is None


def test_patch_binds_the_scheduler_to_connectors(monkeypatch):
    from vllm.v1.core.sched.scheduler import Scheduler

    from vllm_torchtpu import _patch_vllm_scheduler_binds_kv_connector

    bound = []
    child = SimpleNamespace(bind_scheduler=bound.append)
    other = SimpleNamespace()  # A connector without bind_scheduler.

    def init(self):
        self.connector = SimpleNamespace(_connectors=[other, child])

    monkeypatch.setattr(Scheduler, "__init__", init)
    _patch_vllm_scheduler_binds_kv_connector()
    scheduler = object.__new__(Scheduler)
    Scheduler.__init__(scheduler)

    assert bound == [scheduler]
