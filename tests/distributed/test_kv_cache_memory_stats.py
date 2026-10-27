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
"""Unit tests for scheduler-side KV-cache block lifecycle state metrics."""

from __future__ import annotations

import sys
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.request import RequestStatus

import vllm_torchtpu
from vllm_torchtpu import envs
from vllm_torchtpu.distributed.kv_transfer.kv_cache_memory_stats import (
    ALL_KV_GROUPS,
    BLOCK_STATES,
    KV_CACHE_BLOCKS_BY_STATE_METRIC_NAME,
    STATS_BLOCKS_BY_STATE,
    KVBlockStateScanner,
    KVCacheMemoryCollector,
    KVCacheMemoryProm,
    get_or_create_kv_cache_blocks_by_state_gauge,
)
from vllm_torchtpu.distributed.kv_transfer.tpu_connector_stats import (
    TpuKVConnectorPromMetrics,
    TpuKVConnectorStats,
)
from vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector import (
    TPUMultiConnector,
)
from vllm_torchtpu.offload.raiden_connector import (
    TpuOffloadPromMetrics,
    TPURaidenOffloadingConnector,
    TPURaidenOffloadingScheduler,
)


def _get_tpu_connector_cls():
    """Import TPUConnector with mock torch_tpu/tpu_sync modules if absent."""
    mod_name = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"
    if mod_name in sys.modules:
        return sys.modules[mod_name].TPUConnector
    mock_mods = {}
    for name in (
        "jax",
        "jax.numpy",
        "torch_tpu",
        "torch_tpu._internal",
        "torch_tpu._internal.batch_transfer",
        "torch_tpu._internal.sync",
        "tpu_sync",
        "tpu_sync.api",
        "tpu_sync.api.torch",
        "tpu_sync.api.torch.kv_cache_manager",
        "vllm_torchtpu.runner.tpu_runner",
        "vllm_torchtpu.runner.pcp_utils",
        "vllm_torchtpu.utils",
        "vllm_torchtpu.kernels.gdn.head_geometry",
    ):
        if name not in sys.modules:
            mock_mods[name] = MagicMock()
    with patch.dict(sys.modules, mock_mods):
        from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
            TPUConnector,
        )

        return TPUConnector


def _make_real_blocks(num_blocks: int) -> list[KVCacheBlock]:
    blocks = [KVCacheBlock(block_id=i) for i in range(num_blocks)]
    if blocks:
        blocks[0].is_null = True
    return blocks


def _make_mock_scheduler(
    num_blocks: int = 12,
    num_managers: int = 1,
    use_real_blocks: bool = True,
    connector=None,
):
    if use_real_blocks:
        blocks = _make_real_blocks(num_blocks)
    else:
        blocks = [
            SimpleNamespace(
                block_id=i,
                ref_cnt=0,
                block_hash=None,
                is_null=(i == 0),
                pool=None,
            )
            for i in range(num_blocks)
        ]
    block_pool = SimpleNamespace(
        blocks=blocks,
        cached_block_hashes_by_block={},
    )
    managers = [SimpleNamespace(req_to_blocks={}) for _ in range(num_managers)]
    coordinator = SimpleNamespace(
        block_pool=block_pool,
        single_type_managers=managers,
    )
    kv_cache_manager = SimpleNamespace(
        block_pool=block_pool,
        coordinator=coordinator,
    )
    scheduler = SimpleNamespace(
        kv_cache_manager=kv_cache_manager,
        running=[],
        waiting=[],
        requests={},
        deferred_frees=[],
        connector=connector,
        finish_requests=MagicMock(),
    )
    return scheduler, blocks, managers


def _make_req(
    req_id: str,
    status: RequestStatus = RequestStatus.WAITING,
    finished: bool = False,
):
    return SimpleNamespace(
        request_id=req_id,
        status=status,
        is_finished=lambda: finished,
    )


class TestKVBlockStateScanner:
    """Tests for KVBlockStateScanner classification, priority, and cadence."""

    def test_scan_empty_or_missing_scheduler_structures(self):
        assert KVBlockStateScanner.scan(SimpleNamespace()) is None
        assert (
            KVBlockStateScanner.scan(
                SimpleNamespace(kv_cache_manager=SimpleNamespace(coordinator=None))
            )
            is None
        )
        assert (
            KVBlockStateScanner.scan(
                SimpleNamespace(
                    kv_cache_manager=SimpleNamespace(
                        block_pool=SimpleNamespace(blocks=None),
                        coordinator=SimpleNamespace(single_type_managers=[object()]),
                    )
                )
            )
            is None
        )

    def test_scan_all_nine_states_with_real_kv_cache_blocks_and_conservation(
        self,
    ):
        num_gpu_blocks = 12
        scheduler, blocks, managers = _make_mock_scheduler(
            num_blocks=num_gpu_blocks, use_real_blocks=True
        )
        # Verify real KVCacheBlock from vLLM uses slots=True and has no .pool
        assert not hasattr(blocks[1], "pool")

        # Block 1: active_shared (referenced by 2 running requests)
        blocks[1].ref_cnt = 2
        # Block 2: active_private (referenced by 1 running request)
        blocks[2].ref_cnt = 1
        # Block 3: held_for_send (finished request still holding blocks)
        blocks[3].ref_cnt = 1
        # Block 4: loading_kv (active WAITING_FOR_REMOTE_KVS request)
        blocks[4].ref_cnt = 1
        # Block 5: loading_kv (finished/aborted request in finished_loading_req_ids)
        blocks[5].ref_cnt = 1
        # Block 6: waiting_hold (WAITING request holding blocks)
        blocks[6].ref_cnt = 1
        # Block 7: transient_pin (in scheduler.deferred_frees)
        blocks[7].ref_cnt = 1
        # Block 8: orphan_suspect (ref_cnt > 0, no holder in req_to_blocks)
        blocks[8].ref_cnt = 1
        # Block 9: cached_evictable via block._block_hash
        blocks[9].ref_cnt = 0
        blocks[9]._block_hash = MagicMock()
        # Block 10: cached_evictable via cached_block_hashes_by_block
        blocks[10].ref_cnt = 0
        scheduler.kv_cache_manager.block_pool.cached_block_hashes_by_block[10] = (
            "hash_10"
        )
        # Block 11: free_unhashed (ref_cnt == 0, no hash)
        blocks[11].ref_cnt = 0

        r_run1 = _make_req("r_run1", RequestStatus.RUNNING)
        r_run2 = _make_req("r_run2", RequestStatus.RUNNING)
        r_send = _make_req("r_send", RequestStatus.FINISHED_STOPPED, finished=True)
        r_load_active = _make_req("r_load_active", RequestStatus.WAITING_FOR_REMOTE_KVS)
        r_load_aborted = _make_req(
            "r_load_aborted", RequestStatus.FINISHED_ABORTED, finished=True
        )
        r_wait = _make_req("r_wait", RequestStatus.WAITING)

        scheduler.running = [r_run1, r_run2]
        scheduler.requests = {
            "r_run1": r_run1,
            "r_run2": r_run2,
            "r_send": r_send,
            "r_load_active": r_load_active,
            "r_load_aborted": r_load_aborted,
            "r_wait": r_wait,
        }
        managers[0].req_to_blocks = {
            "r_run1": [blocks[0], blocks[1], blocks[2]],
            "r_run2": [blocks[1]],
            "r_send": [blocks[3]],
            "r_load_active": [blocks[4]],
            "r_load_aborted": [blocks[5]],
            "r_wait": [blocks[6]],
        }
        scheduler.deferred_frees = [(1, [blocks[7]])]

        counts = KVBlockStateScanner.scan(
            scheduler, finished_loading_req_ids={"r_load_aborted"}
        )
        assert counts is not None
        assert counts == {
            "active_shared": 1,
            "active_private": 1,
            "held_for_send": 1,
            "loading_kv": 2,
            "waiting_hold": 1,
            "transient_pin": 1,
            "orphan_suspect": 1,
            "cached_evictable": 2,
            "free_unhashed": 1,
        }
        # Conservation invariant: sum of all 9 states == num_gpu_blocks - 1
        assert sum(counts.values()) == num_gpu_blocks - 1

    def test_scan_priority_ordering_and_synthetic_pool_guards(self):
        scheduler, blocks, managers = _make_mock_scheduler(
            num_blocks=6, use_real_blocks=False
        )
        for i in range(1, 6):
            blocks[i].ref_cnt = 2

        r_run = _make_req("r_run", RequestStatus.RUNNING)
        r_send = _make_req("r_send", RequestStatus.FINISHED_STOPPED, finished=True)
        r_load = _make_req("r_load", RequestStatus.WAITING_FOR_REMOTE_KVS)
        r_wait = _make_req("r_wait", RequestStatus.WAITING)

        scheduler.running = [r_run]
        scheduler.requests = {
            "r_run": r_run,
            "r_send": r_send,
            "r_load": r_load,
            "r_wait": r_wait,
        }
        # Synthetic block from a foreign pool and out-of-range block must be ignored
        foreign_block = SimpleNamespace(
            block_id=5,
            ref_cnt=1,
            block_hash=None,
            is_null=False,
            pool=object(),
        )
        out_of_range_block = SimpleNamespace(
            block_id=99,
            ref_cnt=1,
            block_hash=None,
            is_null=False,
            pool=None,
        )

        # Block 1: r_run (active_private) + r_send (held_for_send) -> active_private
        # Block 2: r_send (held_for_send) + r_load (loading_kv) -> held_for_send
        # Block 3: r_load (loading_kv) + r_wait (waiting_hold) -> loading_kv
        # Block 4: r_wait (waiting_hold) + deferred_frees -> waiting_hold
        # Block 5: only referenced via foreign_block -> falls through to orphan_suspect
        managers[0].req_to_blocks = {
            "r_run": [blocks[1], foreign_block, out_of_range_block],
            "r_send": [blocks[1], blocks[2]],
            "r_load": [blocks[2], blocks[3]],
            "r_wait": [blocks[3], blocks[4], foreign_block, out_of_range_block],
        }
        scheduler.deferred_frees = [(1, [blocks[4]])]

        counts = KVBlockStateScanner.scan(scheduler)
        assert counts == {
            "active_shared": 0,
            "active_private": 1,
            "held_for_send": 1,
            "loading_kv": 1,
            "waiting_hold": 1,
            "transient_pin": 0,
            "orphan_suspect": 1,
            "cached_evictable": 0,
            "free_unhashed": 0,
        }

    def test_scan_multi_group_managers_sharing_one_block_pool(self):
        scheduler, blocks, managers = _make_mock_scheduler(
            num_blocks=6, num_managers=2, use_real_blocks=True
        )
        for i in range(1, 5):
            blocks[i].ref_cnt = 1

        r_run1 = _make_req("r_run1", RequestStatus.RUNNING)
        r_run2 = _make_req("r_run2", RequestStatus.RUNNING)
        r_wait = _make_req("r_wait", RequestStatus.WAITING)
        scheduler.running = [r_run1, r_run2]
        scheduler.requests = {
            "r_run1": r_run1,
            "r_run2": r_run2,
            "r_wait": r_wait,
        }
        # Manager 0: attention group; Manager 1: sliding-window/Mamba group
        managers[0].req_to_blocks = {
            "r_run1": [blocks[1], blocks[2]],
            "r_run2": [blocks[1]],
        }
        managers[1].req_to_blocks = {
            "r_run1": [blocks[3]],
            "r_wait": [blocks[4]],
        }

        counts = KVBlockStateScanner.scan(scheduler)
        assert counts == {
            "active_shared": 1,  # block 1
            "active_private": 2,  # blocks 2, 3
            "held_for_send": 0,
            "loading_kv": 0,
            "waiting_hold": 1,  # block 4
            "transient_pin": 0,
            "orphan_suspect": 0,
            "cached_evictable": 0,
            "free_unhashed": 1,  # block 5
        }
        assert sum(counts.values()) == 5

    def test_maybe_scan_cadence_gating_and_nonpositive_interval(self):
        scheduler, _, _ = _make_mock_scheduler(num_blocks=4)

        scanner = KVBlockStateScanner(interval_s=10.0)
        first = scanner.maybe_scan(scheduler, now=100.0)
        assert first is not None
        assert first["free_unhashed"] == 3

        # Within interval -> throttled
        assert scanner.maybe_scan(scheduler, now=105.0) is None
        assert scanner.maybe_scan(scheduler, now=109.9) is None

        # At or past interval -> scans again
        second = scanner.maybe_scan(scheduler, now=110.0)
        assert second is not None

        # interval_s <= 0 scans on every poll
        every_poll_scanner = KVBlockStateScanner(interval_s=0.0)
        assert every_poll_scanner.maybe_scan(scheduler, now=100.0) is not None
        assert every_poll_scanner.maybe_scan(scheduler, now=100.0) is not None
        neg_scanner = KVBlockStateScanner(interval_s=-1.0)
        assert neg_scanner.maybe_scan(scheduler, now=100.0) is not None
        assert neg_scanner.maybe_scan(scheduler, now=100.0) is not None


class TestFinishRequestsAndAbortedLoadingTracking:
    """Tests for finish_requests([], ...) fix and _prune_finished_loading_req_ids."""

    def test_finish_requests_empty_list_vs_specific_vs_none_and_pruning(self):
        scheduler, blocks, managers = _make_mock_scheduler(num_blocks=5)
        for i in range(1, 5):
            blocks[i].ref_cnt = 1

        r_load_1 = _make_req("r_load_1", RequestStatus.WAITING_FOR_REMOTE_KVS)
        r_load_2 = _make_req("r_load_2", RequestStatus.WAITING_FOR_REMOTE_KVS)
        r_load_3 = _make_req("r_load_3", RequestStatus.WAITING_FOR_REMOTE_KVS)
        r_send = _make_req("r_send", RequestStatus.FINISHED_STOPPED, finished=True)

        scheduler.requests = {
            "r_load_1": r_load_1,
            "r_load_2": r_load_2,
            "r_load_3": r_load_3,
            "r_send": r_send,
        }
        managers[0].req_to_blocks = {
            "r_load_1": [blocks[1]],
            "r_load_2": [blocks[2]],
            "r_load_3": [blocks[3]],
            "r_send": [blocks[4]],
        }

        collector = KVCacheMemoryCollector(scheduler, interval_s=0.0)
        assert (
            getattr(scheduler.finish_requests, "_tpu_kv_memory_wrapped", None) is True
        )

        # Calling finish_requests([], ...) with an empty list MUST NOT mark any
        # WAITING_FOR_REMOTE_KVS request as aborted!
        scheduler.finish_requests([], RequestStatus.FINISHED_ABORTED)
        assert collector._finished_loading_req_ids == set()

        # Aborting a single WAITING_FOR_REMOTE_KVS request records only that request
        scheduler.finish_requests(["r_load_1"], RequestStatus.FINISHED_ABORTED)
        r_load_1.status = RequestStatus.FINISHED_ABORTED
        r_load_1.is_finished = lambda: True
        assert collector._finished_loading_req_ids == {"r_load_1"}

        # Even though r_load_1 is now is_finished() == True, its block is still
        # classified as loading_kv (not held_for_send) while in req_to_blocks
        data = collector.collect(now=10.0)
        counts = data[STATS_BLOCKS_BY_STATE]
        assert counts["loading_kv"] == 3  # r_load_1, r_load_2, r_load_3
        assert counts["held_for_send"] == 1  # r_send

        # Aborting all requests with request_ids=None marks all remaining
        # WAITING_FOR_REMOTE_KVS requests (r_load_2, r_load_3) as aborted
        scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
        r_load_2.status = RequestStatus.FINISHED_ABORTED
        r_load_2.is_finished = lambda: True
        r_load_3.status = RequestStatus.FINISHED_ABORTED
        r_load_3.is_finished = lambda: True
        assert collector._finished_loading_req_ids == {
            "r_load_1",
            "r_load_2",
            "r_load_3",
        }

        # Once r_load_1 and r_load_2 free their blocks from req_to_blocks,
        # _prune_finished_loading_req_ids cleans them up on the next collect()
        del managers[0].req_to_blocks["r_load_1"]
        del managers[0].req_to_blocks["r_load_2"]
        blocks[1].ref_cnt = 0
        blocks[2].ref_cnt = 0

        data2 = collector.collect(now=20.0)
        assert collector._finished_loading_req_ids == {"r_load_3"}
        counts2 = data2[STATS_BLOCKS_BY_STATE]
        assert counts2["loading_kv"] == 1  # r_load_3
        assert counts2["held_for_send"] == 1  # r_send
        assert counts2["free_unhashed"] == 2  # blocks 1, 2


class TestPrometheusAndConnectorIntegration:
    """Tests for Prometheus gauge and KV connector integration."""

    def test_shared_gauge_deduplication_and_initialization(self):
        registry = CollectorRegistry()
        gauge_cls = partial(Gauge, registry=registry)
        labelnames = ["model_name", "engine"]
        per_engine = {0: ["test-model", "0"]}

        prom1 = KVCacheMemoryProm(gauge_cls, labelnames, per_engine)
        for state in BLOCK_STATES:
            val = registry.get_sample_value(
                KV_CACHE_BLOCKS_BY_STATE_METRIC_NAME,
                {
                    "model_name": "test-model",
                    "engine": "0",
                    "kv_group": ALL_KV_GROUPS,
                    "state": state,
                },
            )
            assert val == 0.0

        prom1.observe(
            {
                STATS_BLOCKS_BY_STATE: {
                    "active_shared": 2,
                    "active_private": 5,
                    "free_unhashed": 10,
                }
            },
            engine_idx=0,
        )
        assert (
            registry.get_sample_value(
                KV_CACHE_BLOCKS_BY_STATE_METRIC_NAME,
                {
                    "model_name": "test-model",
                    "engine": "0",
                    "kv_group": ALL_KV_GROUPS,
                    "state": "active_private",
                },
            )
            == 5.0
        )

    def test_global_registry_duplicate_gauge_reuses_collector(self):
        labelnames = ["model_name", "engine"]
        g1 = get_or_create_kv_cache_blocks_by_state_gauge(Gauge, labelnames)
        g2 = get_or_create_kv_cache_blocks_by_state_gauge(Gauge, labelnames)
        assert g1 is g2

    def test_tpu_kv_connector_prom_metrics_and_stats_aggregation(self):
        registry = CollectorRegistry()
        metric_types = {
            Gauge: partial(Gauge, registry=registry),
            Counter: partial(Counter, registry=registry),
            Histogram: partial(Histogram, registry=registry),
        }
        vllm_config = MagicMock()
        labelnames = ["model_name", "engine"]
        per_engine = {0: ["m", "0"]}

        prom = TpuKVConnectorPromMetrics(
            vllm_config, metric_types, labelnames, per_engine
        )
        s1 = TpuKVConnectorStats()
        assert s1.is_empty()

        s2 = TpuKVConnectorStats()
        s2.data[STATS_BLOCKS_BY_STATE] = {s: i for i, s in enumerate(BLOCK_STATES)}
        assert not s2.is_empty()

        s1.aggregate(s2)
        assert s1.data[STATS_BLOCKS_BY_STATE] == s2.data[STATS_BLOCKS_BY_STATE]

        prom.observe(s1.data, engine_idx=0)
        for i, state in enumerate(BLOCK_STATES):
            assert registry.get_sample_value(
                KV_CACHE_BLOCKS_BY_STATE_METRIC_NAME,
                {
                    "model_name": "m",
                    "engine": "0",
                    "kv_group": ALL_KV_GROUPS,
                    "state": state,
                },
            ) == float(i)

    def test_tpu_offload_prom_metrics_and_scheduler_get_stats(self):
        registry = CollectorRegistry()
        metric_types = {
            Gauge: partial(Gauge, registry=registry),
            Counter: partial(Counter, registry=registry),
            Histogram: partial(Histogram, registry=registry),
        }
        vllm_config = MagicMock()
        vllm_config.kv_transfer_config.kv_connector_extra_config = {}
        labelnames = ["model_name", "engine"]
        per_engine = {0: ["m", "0"]}

        prom = TPURaidenOffloadingConnector.build_prom_metrics(
            vllm_config, metric_types, labelnames, per_engine
        )
        assert isinstance(prom, TpuOffloadPromMetrics)

        # Build a lightweight TPURaidenOffloadingScheduler without heavy manager init
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
            OffloadingConnectorStats,
        )

        offload_sched = TPURaidenOffloadingScheduler.__new__(
            TPURaidenOffloadingScheduler
        )
        offload_sched._connector_stats = OffloadingConnectorStats()
        offload_sched.manager = SimpleNamespace(get_stats=lambda: None)
        offload_sched._kv_cache_memory = None

        scheduler, blocks, managers = _make_mock_scheduler(num_blocks=5)
        blocks[1].ref_cnt = 1
        r_run = _make_req("r_run", RequestStatus.RUNNING)
        scheduler.running = [r_run]
        managers[0].req_to_blocks = {"r_run": [blocks[1]]}
        offload_sched.bind_scheduler(scheduler)

        stats = offload_sched.get_stats()
        assert stats is not None
        prom.observe(stats.data, engine_idx=0)

        assert (
            registry.get_sample_value(
                KV_CACHE_BLOCKS_BY_STATE_METRIC_NAME,
                {
                    "model_name": "m",
                    "engine": "0",
                    "kv_group": ALL_KV_GROUPS,
                    "state": "active_private",
                },
            )
            == 1.0
        )
        assert (
            registry.get_sample_value(
                KV_CACHE_BLOCKS_BY_STATE_METRIC_NAME,
                {
                    "model_name": "m",
                    "engine": "0",
                    "kv_group": ALL_KV_GROUPS,
                    "state": "free_unhashed",
                },
            )
            == 3.0
        )

    def test_tpu_multi_connector_deduplicates_binding_and_gauge_registration(
        self,
    ):
        # Verify both TpuKVConnectorPromMetrics and TpuOffloadPromMetrics can be
        # constructed against the same CollectorRegistry without duplicate error
        registry = CollectorRegistry()
        metric_types = {
            Gauge: partial(Gauge, registry=registry),
            Counter: partial(Counter, registry=registry),
            Histogram: partial(Histogram, registry=registry),
        }
        vllm_config = MagicMock()
        vllm_config.kv_transfer_config.kv_connector_extra_config = {}
        labelnames = ["model_name", "engine"]
        per_engine = {0: ["m", "0"]}

        tpu_prom = TpuKVConnectorPromMetrics(
            vllm_config, metric_types, labelnames, per_engine
        )
        offload_prom = TpuOffloadPromMetrics(
            vllm_config, metric_types, labelnames, per_engine
        )
        assert (
            tpu_prom._kv_cache_memory_prom.blocks_by_state
            is offload_prom.blocks_by_state
        )

        TPUConnector = _get_tpu_connector_cls()
        tpu_conn = TPUConnector.__new__(TPUConnector)
        tpu_conn.connector_worker = None
        tpu_conn.connector_scheduler = None
        tpu_conn._kv_cache_memory = None

        offload_sched = TPURaidenOffloadingScheduler.__new__(
            TPURaidenOffloadingScheduler
        )
        offload_sched._kv_cache_memory = None
        offload_conn = TPURaidenOffloadingConnector.__new__(
            TPURaidenOffloadingConnector
        )
        offload_conn.connector_scheduler = offload_sched

        multi = TPUMultiConnector.__new__(TPUMultiConnector)
        multi._connectors = [tpu_conn, offload_conn]

        scheduler, _, _ = _make_mock_scheduler(num_blocks=4, connector=multi)
        multi.bind_scheduler(scheduler)

        assert getattr(scheduler, "_tpu_kv_block_state_collector_bound", None) is True
        assert tpu_conn._kv_cache_memory is not None
        assert offload_sched._kv_cache_memory is None

        stats = tpu_conn.get_kv_connector_stats()
        assert stats is not None
        assert STATS_BLOCKS_BY_STATE in stats.data
        assert stats.data[STATS_BLOCKS_BY_STATE]["free_unhashed"] == 3

        # Immediate second call is throttled and returns None when no worker stats exist
        assert tpu_conn.get_kv_connector_stats() is None

        # When _xfer_stats is present on the connector, sched_stats merges into it
        tpu_conn._kv_cache_memory._scanner._last_scan_ts = None
        xfer = TpuKVConnectorStats()
        xfer.record_d2h_transfer(12.5)
        tpu_conn._xfer_stats = xfer
        merged = tpu_conn.get_kv_connector_stats()
        assert merged is not None
        assert merged.data["d2h_transfer_time"] == [12.5]
        assert merged.data[STATS_BLOCKS_BY_STATE]["free_unhashed"] == 3

    def test_patch_vllm_scheduler_binds_kv_connector_is_idempotent(self, monkeypatch):
        from vllm.v1.core.sched.scheduler import Scheduler

        called_with = []

        def fake_orig_init(self, *args, **kwargs):
            self.connector = SimpleNamespace(
                bind_scheduler=lambda sched: called_with.append(sched)
            )

        monkeypatch.setattr(Scheduler, "__init__", fake_orig_init)
        vllm_torchtpu._patch_vllm_scheduler_binds_kv_connector()
        patched_once = Scheduler.__init__
        vllm_torchtpu._patch_vllm_scheduler_binds_kv_connector()
        assert Scheduler.__init__ is patched_once

        dummy = Scheduler.__new__(Scheduler)
        Scheduler.__init__(dummy)
        assert called_with == [dummy]

    def test_env_default_interval(self, monkeypatch):
        monkeypatch.delenv("TPU_KV_BLOCK_STATE_INTERVAL_S", raising=False)
        val = envs.environment_variables["TPU_KV_BLOCK_STATE_INTERVAL_S"]()
        assert val == pytest.approx(10.0)

    def test_interval_env_is_excluded_from_tpu_compile_cache_key(self):
        mock_mods = {}
        for name in (
            "torch_tpu",
            "torch_tpu._internal",
            "torch_tpu._internal.compile",
            "torch_tpu._internal.compile._backend",
            "vllm_torchtpu.utils",
        ):
            if name not in sys.modules:
                mock_mods[name] = MagicMock()
        with patch.dict(sys.modules, mock_mods):
            from vllm_torchtpu.compilation.tpu_compiler import (
                _TPU_COMPILE_ENV_IGNORED,
                _tpu_compile_env_factors,
            )

            assert "TPU_KV_BLOCK_STATE_INTERVAL_S" in _TPU_COMPILE_ENV_IGNORED
            assert "TPU_KV_BLOCK_STATE_INTERVAL_S" not in set(
                _tpu_compile_env_factors()
            )
