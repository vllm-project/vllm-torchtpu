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
"""vllm:kv_cache_blocks_by_state over a real vLLM KVCacheManager."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from prometheus_client import REGISTRY, Counter, Gauge, Histogram
from vllm import SamplingParams
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.metrics.prometheus import unregister_vllm_metrics
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from vllm_torchtpu.distributed.kv_transfer import kv_cache_memory_stats
from vllm_torchtpu.distributed.kv_transfer.kv_cache_memory_stats import (
    BLOCK_STATES,
    KV_CACHE_BLOCKS_BY_STATE_METRIC_NAME,
    SCAN_INTERVAL_S,
    STATS_KEY_BLOCKS_BY_STATE,
    KVBlockStateTracker,
)
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPUConnector
from vllm_torchtpu.distributed.kv_transfer.tpu_connector_stats import (
    TpuKVConnectorPromMetrics,
    TpuKVConnectorStats,
)
from vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector import TPUMultiConnector
from vllm_torchtpu.offload.raiden_connector import (
    TpuOffloadPromMetrics,
    TPURaidenOffloadingConnector,
)

BLOCK_SIZE = 4
NUM_BLOCKS = 16  # 15 usable; block 0 is the null block.
KV_CACHE_CONFIG = KVCacheConfig(
    num_blocks=NUM_BLOCKS,
    kv_cache_tensors=[],
    kv_cache_groups=[
        KVCacheGroupSpec(
            ["layer"],
            FullAttentionSpec(
                block_size=BLOCK_SIZE, num_kv_heads=1, head_size=8, dtype=torch.float32
            ),
        )
    ],
)


def _manager() -> KVCacheManager:
    init_none_hash(sha256)
    return KVCacheManager(
        KV_CACHE_CONFIG,
        max_model_len=64,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
    )


def _allocate(manager: KVCacheManager, req_id: str, prompt: list[int]) -> Request:
    request = Request(
        req_id,
        prompt,
        SamplingParams(max_tokens=1),
        None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    blocks, num_hit_tokens = manager.get_computed_blocks(request)[:2]
    assert manager.allocate_slots(
        request, len(prompt) - num_hit_tokens, num_hit_tokens, blocks
    )
    manager.cache_blocks(request, len(prompt))
    return request


def _counts(manager: KVCacheManager, tracker: KVBlockStateTracker) -> dict:
    counts = kv_cache_memory_stats.count_blocks_by_state(
        manager, tracker.loading_req_ids, tracker.held_req_ids
    )
    assert sum(counts) == NUM_BLOCKS - 1
    return dict(zip(BLOCK_STATES, counts))


def test_each_block_counts_once_in_one_state():
    manager, tracker = _manager(), KVBlockStateTracker()
    _allocate(manager, "running", list(range(8)))
    # Prefix hit on both blocks of "running", plus one new block.
    _allocate(manager, "sharing", list(range(12)))
    manager.free(_allocate(manager, "finished", list(range(100, 108))))
    loading = _allocate(manager, "loading", list(range(200, 208)))
    aborted = _allocate(manager, "aborted", list(range(300, 304)))
    sending = _allocate(manager, "sending", list(range(400, 412)))
    tracker.on_alloc("loading", num_external_tokens=8)
    tracker.on_alloc("aborted", num_external_tokens=4)
    tracker.on_alloc("running", num_external_tokens=0)
    tracker.on_finished("aborted", delay_free=False)
    tracker.on_finished("sending", delay_free=True)
    assert _counts(manager, tracker) == {
        "active": 3,
        "loading_kv": 3,
        "held_for_send": 3,
        "cached_evictable": 2,
        "empty_free": 4,
    }

    # An aborted request keeps its blocks until its load reports; the
    # scheduler then frees them, as it does after a finished send.
    tracker.on_connector_output(
        KVConnectorOutput(finished_recving={"loading", "aborted"})
    )
    tracker.on_connector_output(KVConnectorOutput(finished_sending={"sending"}))
    manager.free(aborted)
    manager.free(sending)
    assert loading.request_id not in tracker.loading_req_ids
    assert _counts(manager, tracker) == {
        "active": 5,
        "loading_kv": 0,
        "held_for_send": 0,
        "cached_evictable": 6,
        "empty_free": 4,
    }


def test_samples_after_bind_then_every_interval_and_once_when_idle(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(
        kv_cache_memory_stats, "time", SimpleNamespace(monotonic=lambda: now[0])
    )
    manager, tracker = _manager(), KVBlockStateTracker()
    assert tracker.maybe_count() is None
    tracker.bind(manager)
    request = _allocate(manager, "r", list(range(8)))
    assert tracker.maybe_count() == [2, 0, 0, 0, 13]
    assert tracker.maybe_count() is None
    manager.free(request)
    assert tracker.maybe_count() == [0, 0, 0, 2, 13]
    assert tracker.maybe_count() is None
    # A burst within the interval, then idle again: the new idle state is
    # published too.
    burst = _allocate(manager, "burst", list(range(100, 116)))
    assert tracker.maybe_count() is None
    manager.free(burst)
    assert tracker.maybe_count() == [0, 0, 0, 6, 9]
    now[0] += SCAN_INTERVAL_S
    assert tracker.maybe_count() == [0, 0, 0, 6, 9]


def _tpu_connector(delay_free: bool) -> TPUConnector:
    cfg = MagicMock()
    cfg.speculative_config = None
    cfg.max_concurrent_batches = 1
    cfg.kv_transfer_config.kv_connector_extra_config = {}
    cfg.cache_config.block_size = BLOCK_SIZE
    with patch(
        "vllm_torchtpu.distributed.kv_transfer.tpu_connector.TPUConnectorScheduler"
    ) as scheduler_cls:
        scheduler_cls.return_value.request_finished.return_value = (delay_free, None)
        return TPUConnector(cfg, KVConnectorRole.SCHEDULER, KV_CACHE_CONFIG)


def _offload_connector() -> TPURaidenOffloadingConnector:
    # Building the store needs TPU resources; these hooks only use the
    # connector scheduler's request_finished and get_stats.
    connector = object.__new__(TPURaidenOffloadingConnector)
    connector.connector_scheduler = MagicMock()
    connector.connector_scheduler.request_finished.return_value = (False, None)
    connector.connector_scheduler.get_stats.return_value = None
    connector.kv_block_states = KVBlockStateTracker()
    return connector


def test_tpu_connector_hooks_drive_the_snapshot():
    manager, connector = _manager(), _tpu_connector(delay_free=True)
    connector.bind_kv_cache_manager(manager)
    for req_id, prompt in (("load", range(8)), ("sync", range(100, 104))):
        request = _allocate(manager, req_id, list(prompt))
        connector.update_state_after_alloc(request, manager.get_blocks(req_id), 4)
    connector.build_connector_meta(SimpleNamespace(num_scheduled_tokens={"sync": 4}))
    send = _allocate(manager, "send", list(range(200, 204)))
    connector.request_finished_all_groups(send, manager.get_block_ids("send"))

    stats = connector.get_kv_connector_stats()
    assert stats is not None
    assert stats.data[STATS_KEY_BLOCKS_BY_STATE] == [1, 2, 1, 0, 11]

    connector.update_connector_output(
        KVConnectorOutput(finished_recving={"load"}, finished_sending={"send"})
    )
    assert not connector.kv_block_states.loading_req_ids
    assert not connector.kv_block_states.held_req_ids
    assert connector.get_kv_connector_stats() is None


def _sample(engine: str, state: str) -> float | None:
    return REGISTRY.get_sample_value(
        KV_CACHE_BLOCKS_BY_STATE_METRIC_NAME,
        {"model_name": "m", "engine": engine, "state": state},
    )


def test_both_connector_types_share_one_gauge():
    unregister_vllm_metrics()
    try:
        metric_types = {Gauge: Gauge, Counter: Counter, Histogram: Histogram}
        args = (["model_name", "engine"], {0: ["m", "0"], 1: ["m", "1"]})
        cfg = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                kv_connector="TPUMultiConnector",
                kv_connector_extra_config={},
                kv_load_failure_policy="fail",
            )
        )
        tpu_prom = TpuKVConnectorPromMetrics(MagicMock(), metric_types, *args)
        offload_prom = TpuOffloadPromMetrics(cfg, metric_types, *args)
        assert _sample("0", "active") is None  # absent until the first sample

        tpu_stats = TpuKVConnectorStats()
        tpu_stats.record_blocks_by_state([9, 9, 9, 9, 9])
        tpu_stats.record_blocks_by_state([5, 4, 3, 2, 1])  # the last one wins
        tpu_prom.observe(tpu_stats.data, engine_idx=0)
        assert [_sample("0", s) for s in BLOCK_STATES] == [5, 4, 3, 2, 1]

        manager, connector = _manager(), _offload_connector()
        connector.bind_kv_cache_manager(manager)
        offload_stats = connector.get_kv_connector_stats()
        assert offload_stats is not None
        offload_prom.observe(offload_stats.data, engine_idx=1)
        assert [_sample("1", s) for s in BLOCK_STATES] == [0, 0, 0, 0, 15]
    finally:
        unregister_vllm_metrics()


def test_multi_connector_samples_all_children_once():
    manager = _manager()
    tpu_child, offload_child = _tpu_connector(delay_free=True), _offload_connector()
    multi = object.__new__(TPUMultiConnector)
    multi._connectors = [tpu_child, offload_child]
    multi.bind_kv_cache_manager(manager)
    assert tpu_child.kv_block_states is offload_child.kv_block_states

    load = _allocate(manager, "load", list(range(8)))
    offload_child.update_state_after_alloc(load, manager.get_blocks("load"), 8)
    tpu_child.update_state_after_alloc(load, manager.get_blocks("load"), 0)
    send = _allocate(manager, "send", list(range(100, 104)))
    for child in (tpu_child, offload_child):
        child.request_finished(send, manager.get_block_ids("send")[0])

    stats = multi.get_kv_connector_stats()
    assert stats is not None
    assert list(stats.data) == ["TPUConnector"]
    assert stats.data["TPUConnector"].data[STATS_KEY_BLOCKS_BY_STATE] == [
        0,
        2,
        1,
        0,
        12,
    ]
