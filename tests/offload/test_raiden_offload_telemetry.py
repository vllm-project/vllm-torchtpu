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
"""Unit tests for buffered Raiden telemetry in TPURaidenOffloadingConnector."""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_torchtpu.distributed import utils as dist_utils
from vllm_torchtpu.distributed.kv_transfer.tpu_connector_stats import (
    TpuKVConnectorPromMetrics,
    TpuKVConnectorStats,
)
from vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector import TPUMultiConnector
from vllm_torchtpu.offload.metrics import (
    RaidenOffloadingConnectorStats,
    TPURaidenOffloadingPromMetrics,
)
from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector


def _make_offload_vllm_config() -> SimpleNamespace:
    return SimpleNamespace(
        kv_transfer_config=KVTransferConfig(
            kv_connector="TPURaidenOffloadingConnector",
            kv_connector_module_path="vllm_torchtpu.offload.raiden_connector",
            kv_role="kv_both",
            kv_connector_extra_config={"cpu_bytes_to_use": 1 << 20},
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
    )


def _make_metric_types(registry: CollectorRegistry) -> dict[type, object]:
    return {
        Gauge: partial(Gauge, registry=registry),
        Counter: partial(Counter, registry=registry),
        Histogram: partial(Histogram, registry=registry),
    }


def _sample_raiden_descriptors() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            name="kv_recall_blocks_total",
            type="COUNTER",
            label_names=["dp_rank", "tier", "outcome"],
            description="Total KV blocks recalled",
        ),
        SimpleNamespace(
            name="kv_recall_duration_ms",
            type="HISTOGRAM",
            buckets=[1.0, 10.0, 100.0],
            label_names=["dp_rank", "tier", "outcome"],
            description="KV recall duration in ms",
        ),
        SimpleNamespace(
            name="host_pool_blocks",
            type="GAUGE",
            label_names=["dp_rank", "state"],
            description="Host pool block counts",
        ),
        SimpleNamespace(
            name="d2h_bytes_total",
            type="COUNTER",
            label_names=["host_ip", "local_rank"],
            description="Total D2H DMA bytes",
        ),
        SimpleNamespace(
            name="storage_lookup_duration_ms",
            type="HISTOGRAM",
            buckets=[0.5, 5.0, 50.0],
            label_names=["dp_rank"],
            description="Secondary storage lookup duration",
        ),
        SimpleNamespace(
            name="buffer_allocated_bytes",
            type="GAUGE",
            label_names=[],
            description="Allocated host staging bytes",
        ),
    ]


def _bare_offload_connector(
    *,
    scheduler: object | None = None,
    worker: object | None = None,
    worker_rank: int | None = None,
) -> TPURaidenOffloadingConnector:
    conn = object.__new__(TPURaidenOffloadingConnector)
    conn.connector_scheduler = scheduler  # type: ignore[assignment]
    conn.connector_worker = worker  # type: ignore[assignment]
    conn._worker_rank = (
        0 if (worker is not None and worker_rank is None) else worker_rank
    )
    return conn


class TestTPURaidenOffloadingConnectorStatsFlow:
    def test_worker_role_drains_samples_and_returns_none_when_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = _bare_offload_connector(worker=MagicMock(), worker_rank=1)

        mock_telemetry = MagicMock()
        mock_telemetry.get_metric_metadata.return_value = _sample_raiden_descriptors()
        mock_telemetry.get_and_reset_metric_samples.return_value = {}
        monkeypatch.setattr(
            dist_utils, "get_raiden_telemetry_module", lambda: mock_telemetry
        )
        assert conn.get_kv_connector_stats() is None

        mock_telemetry.get_and_reset_metric_samples.return_value = {
            'tpu_raiden_d2h_bytes_total{host_ip="10.0.0.2",local_rank="1"}': [4096.0],
            "tpu_raiden_buffer_allocated_bytes": [1048576.0],
        }
        stats = conn.get_kv_connector_stats()
        assert isinstance(stats, RaidenOffloadingConnectorStats)
        assert not stats.is_empty()
        assert stats.data["raiden_samples"] == {
            'tpu_raiden_d2h_bytes_total{host_ip="10.0.0.2",local_rank="1"}': [4096.0],
            'tpu_raiden_buffer_allocated_bytes{local_rank="1"}': [1048576.0],
        }

    def test_scheduler_role_merges_offloading_stats_and_kv_cache_store_samples(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base_stats = OffloadingConnectorStats()
        base_stats.increase_counter("vllm:kv_offload_store_bytes", 8192)
        base_stats.observe_histogram("vllm:kv_offload_lookup_sync_delay_seconds", 0.002)
        base_stats.set_info([{"backend": "raiden"}])

        sched = MagicMock()
        sched.get_stats.return_value = base_stats
        conn = _bare_offload_connector(scheduler=sched)

        mock_telemetry = MagicMock()
        recall_local_key = (
            'tpu_raiden_kv_recall_blocks_total{dp_rank="0",outcome="ok",'
            'tier="local_dram"}'
        )
        mock_telemetry.get_and_reset_metric_samples.return_value = {
            recall_local_key: [12.0],
            'tpu_raiden_host_pool_blocks{dp_rank="0",state="used"}': [64.0],
        }
        monkeypatch.setattr(
            dist_utils, "get_raiden_telemetry_module", lambda: mock_telemetry
        )

        stats = conn.get_kv_connector_stats()
        assert isinstance(stats, RaidenOffloadingConnectorStats)
        assert not stats.is_empty()
        assert stats.data["info"] == [{"backend": "raiden"}]
        assert stats.data["raiden_samples"] == {
            recall_local_key: [12.0],
            'tpu_raiden_host_pool_blocks{dp_rank="0",state="used"}': [64.0],
        }
        assert stats.reduce()["vllm:kv_offload_store_bytes"] == 8192
        assert stats.reduce()["vllm:kv_offload_lookup_sync_delay_seconds_count"] == 1
        assert "raiden_samples" not in stats.reduce()

    def test_worker_and_scheduler_aggregation_and_prom_observation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate 2 TP workers draining through `get_kv_connector_stats()`,
        # aggregated by KVOutputAggregator, then merged with EngineCore
        # scheduler stats in Scheduler.update_from_output().
        mock_telemetry = MagicMock()
        mock_telemetry.get_metric_metadata.return_value = _sample_raiden_descriptors()
        monkeypatch.setattr(
            dist_utils, "get_raiden_telemetry_module", lambda: mock_telemetry
        )

        mock_telemetry.get_and_reset_metric_samples.return_value = {
            'tpu_raiden_d2h_bytes_total{host_ip="10.0.0.1",local_rank="0"}': [1024.0],
            "tpu_raiden_buffer_allocated_bytes": [2048.0],
        }
        w0 = _bare_offload_connector(
            worker=MagicMock(), worker_rank=0
        ).get_kv_connector_stats()

        mock_telemetry.get_and_reset_metric_samples.return_value = {
            'tpu_raiden_d2h_bytes_total{host_ip="10.0.0.1",local_rank="1"}': [3072.0],
            "tpu_raiden_buffer_allocated_bytes": [4096.0],
        }
        w1 = _bare_offload_connector(
            worker=MagicMock(), worker_rank=1
        ).get_kv_connector_stats()

        assert isinstance(w0, RaidenOffloadingConnectorStats)
        assert isinstance(w1, RaidenOffloadingConnectorStats)
        agg_workers = w0.aggregate(w1)

        base_sched = OffloadingConnectorStats()
        base_sched.increase_counter("vllm:kv_offload_load_bytes", 4096)
        recall_blocks_key = (
            'tpu_raiden_kv_recall_blocks_total{dp_rank="0",outcome="ok",'
            'tier="shared_storage"}'
        )
        recall_dur_key = (
            'tpu_raiden_kv_recall_duration_ms{dp_rank="0",outcome="ok",'
            'tier="shared_storage"}'
        )
        sched_stats = RaidenOffloadingConnectorStats.from_base_stats(
            base_sched,
            {
                recall_blocks_key: [8.0],
                recall_dur_key: [2.5, 7.5],
                'tpu_raiden_host_pool_blocks{dp_rank="0",state="total"}': [512.0],
                'tpu_raiden_storage_lookup_duration_ms{dp_rank="0"}': [1.2],
            },
        )
        assert sched_stats is not None
        combined = agg_workers.aggregate(sched_stats)

        # Round-trip through serialized dict (ZMQ IPC payload).
        wire_payload = combined.to_dict()
        rebuilt = TPURaidenOffloadingConnector.build_kv_connector_stats(wire_payload)
        assert isinstance(rebuilt, RaidenOffloadingConnectorStats)
        assert rebuilt.reduce() == {"vllm:kv_offload_load_bytes": 4096}

        registry = CollectorRegistry()
        with patch("vllm_torchtpu.distributed.utils.configure_raiden_telemetry"):
            prom = TPURaidenOffloadingConnector.build_prom_metrics(
                _make_offload_vllm_config(),  # type: ignore[arg-type]
                _make_metric_types(registry),  # type: ignore[arg-type]
                ["model_name", "engine"],
                {0: ["qwen3", "0"]},
            )
        assert isinstance(prom, TPURaidenOffloadingPromMetrics)
        prom.observe(rebuilt.to_dict(), engine_idx=0)

        assert (
            registry.get_sample_value(
                "vllm:kv_offload_load_bytes_total",
                {"model_name": "qwen3", "engine": "0"},
            )
            == 4096.0
        )
        assert (
            registry.get_sample_value(
                "tpu_raiden_d2h_bytes_total",
                {
                    "model_name": "qwen3",
                    "engine": "0",
                    "host_ip": "10.0.0.1",
                    "local_rank": "0",
                },
            )
            == 1024.0
        )
        assert (
            registry.get_sample_value(
                "tpu_raiden_d2h_bytes_total",
                {
                    "model_name": "qwen3",
                    "engine": "0",
                    "host_ip": "10.0.0.1",
                    "local_rank": "1",
                },
            )
            == 3072.0
        )
        assert (
            registry.get_sample_value(
                "tpu_raiden_kv_recall_blocks_total",
                {
                    "model_name": "qwen3",
                    "engine": "0",
                    "dp_rank": "0",
                    "tier": "shared_storage",
                    "outcome": "ok",
                },
            )
            == 8.0
        )
        assert (
            registry.get_sample_value(
                "tpu_raiden_kv_recall_duration_ms_sum",
                {
                    "model_name": "qwen3",
                    "engine": "0",
                    "dp_rank": "0",
                    "tier": "shared_storage",
                    "outcome": "ok",
                },
            )
            == 10.0
        )
        assert (
            registry.get_sample_value(
                "tpu_raiden_host_pool_blocks",
                {
                    "model_name": "qwen3",
                    "engine": "0",
                    "dp_rank": "0",
                    "state": "total",
                    "local_rank": "",
                },
            )
            == 512.0
        )
        # Each TP worker keeps its own gauge series via `local_rank`.
        assert (
            registry.get_sample_value(
                "tpu_raiden_buffer_allocated_bytes",
                {"model_name": "qwen3", "engine": "0", "local_rank": "0"},
            )
            == 2048.0
        )
        assert (
            registry.get_sample_value(
                "tpu_raiden_buffer_allocated_bytes",
                {"model_name": "qwen3", "engine": "0", "local_rank": "1"},
            )
            == 4096.0
        )

    @pytest.mark.parametrize("worker_first", [True, False])
    def test_single_worker_uniproc_worker_and_scheduler_share_gauge_series(
        self, monkeypatch: pytest.MonkeyPatch, worker_first: bool
    ) -> None:
        # On the uniproc executor (world_size=1), worker and scheduler share
        # one process and one C++ sample buffer. Both roles must pass
        # worker_rank=None so gauges always land in the same series regardless
        # of which role drains first.
        uniproc_cfg = SimpleNamespace(
            parallel_config=SimpleNamespace(world_size=1, rank=0)
        )
        worker_rank = dist_utils.get_raiden_worker_telemetry_rank(uniproc_cfg)
        assert worker_rank is None

        mock_telemetry = MagicMock()
        mock_telemetry.get_metric_metadata.return_value = _sample_raiden_descriptors()
        # The single process buffer resets on read: first reader gets the
        # sample, second reader sees an empty buffer.
        mock_telemetry.get_and_reset_metric_samples.side_effect = [
            {"tpu_raiden_buffer_allocated_bytes": [8192.0]},
            {},
        ]
        monkeypatch.setattr(
            dist_utils, "get_raiden_telemetry_module", lambda: mock_telemetry
        )

        worker_conn = _bare_offload_connector(worker=MagicMock(), worker_rank=None)
        worker_conn._worker_rank = worker_rank
        sched = MagicMock()
        sched.get_stats.return_value = None
        sched_conn = _bare_offload_connector(scheduler=sched)

        if worker_first:
            first_stats = worker_conn.get_kv_connector_stats()
            second_stats = sched_conn.get_kv_connector_stats()
        else:
            first_stats = sched_conn.get_kv_connector_stats()
            second_stats = worker_conn.get_kv_connector_stats()

        assert isinstance(first_stats, RaidenOffloadingConnectorStats)
        assert second_stats is None
        assert first_stats.data["raiden_samples"] == {
            "tpu_raiden_buffer_allocated_bytes": [8192.0]
        }

        registry = CollectorRegistry()
        with patch("vllm_torchtpu.distributed.utils.configure_raiden_telemetry"):
            prom = TPURaidenOffloadingConnector.build_prom_metrics(
                _make_offload_vllm_config(),  # type: ignore[arg-type]
                _make_metric_types(registry),  # type: ignore[arg-type]
                ["model_name", "engine"],
                {0: ["qwen3", "0"]},
            )
        prom.observe(first_stats.to_dict(), engine_idx=0)
        assert (
            registry.get_sample_value(
                "tpu_raiden_buffer_allocated_bytes",
                {"model_name": "qwen3", "engine": "0", "local_rank": ""},
            )
            == 8192.0
        )
        assert (
            registry.get_sample_value(
                "tpu_raiden_buffer_allocated_bytes",
                {"model_name": "qwen3", "engine": "0", "local_rank": "0"},
            )
            is None
        )


class TestTPUMultiConnectorSharedRegistryAndDrain:
    def test_multi_connector_shares_raiden_collectors_without_duplicate_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Register on a copy so the process-wide factory registry is restored
        # after the test; tpu_platform registers these names itself later.
        monkeypatch.setattr(
            KVConnectorFactory, "_registry", dict(KVConnectorFactory._registry)
        )
        for name, mod_path in (
            (
                "TPURaidenConnector",
                "vllm_torchtpu.distributed.kv_transfer.tpu_connector",
            ),
            ("TPURaidenOffloadingConnector", "vllm_torchtpu.offload.raiden_connector"),
        ):
            if name not in KVConnectorFactory._registry:
                KVConnectorFactory.register_connector(name, mod_path, name)

        multi_cfg = SimpleNamespace(
            kv_transfer_config=KVTransferConfig(
                kv_connector="TPUMultiConnector",
                kv_connector_module_path=(
                    "vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector"
                ),
                kv_role="kv_consumer",
                kv_connector_extra_config={
                    "connectors": [
                        {
                            "kv_connector": "TPURaidenConnector",
                            "kv_connector_module_path": (
                                "vllm_torchtpu.distributed.kv_transfer.tpu_connector"
                            ),
                            "kv_role": "kv_consumer",
                        },
                        {
                            "kv_connector": "TPURaidenOffloadingConnector",
                            "kv_connector_module_path": (
                                "vllm_torchtpu.offload.raiden_connector"
                            ),
                            "kv_role": "kv_both",
                            "kv_connector_extra_config": {"cpu_bytes_to_use": 1 << 20},
                        },
                    ]
                },
            ),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
        )

        registry = CollectorRegistry()
        mock_telemetry = MagicMock()
        mock_telemetry.get_metric_metadata.return_value = _sample_raiden_descriptors()
        with (
            patch(
                "vllm_torchtpu.distributed.utils.get_raiden_telemetry_module",
                return_value=mock_telemetry,
            ),
            patch("vllm_torchtpu.distributed.utils.configure_raiden_telemetry"),
        ):
            multi_prom = TPUMultiConnector.build_prom_metrics(
                multi_cfg,  # type: ignore[arg-type]
                _make_metric_types(registry),  # type: ignore[arg-type]
                ["model_name", "engine"],
                {0: ["qwen3", "0"]},
            )

        p2p_prom = multi_prom._prom_metrics["TPURaidenConnector"]
        offload_prom = multi_prom._prom_metrics["TPURaidenOffloadingConnector"]
        assert isinstance(p2p_prom, TpuKVConnectorPromMetrics)
        assert isinstance(offload_prom, TPURaidenOffloadingPromMetrics)
        assert (
            p2p_prom.dynamic_metrics["tpu_raiden_d2h_bytes_total"][1]
            is offload_prom.dynamic_metrics["tpu_raiden_d2h_bytes_total"][1]
        )

        # Simulate worker P2P draining DMA bytes while scheduler offload drains
        # KVCacheStore recall blocks in the same step, via MultiConnector.
        p2p_stats = TpuKVConnectorStats()
        p2p_stats.data[
            'tpu_raiden_d2h_bytes_total{host_ip="10.0.0.9",local_rank="0"}'
        ] = [2048.0]
        recall_local_key = (
            'tpu_raiden_kv_recall_blocks_total{dp_rank="0",outcome="ok",'
            'tier="local_dram"}'
        )
        offload_stats = RaidenOffloadingConnectorStats.from_base_stats(
            None,
            {recall_local_key: [4.0]},
        )
        assert offload_stats is not None

        rebuilt_multi = MultiConnector.build_kv_connector_stats(
            {
                "TPURaidenConnector": p2p_stats.to_dict(),
                "TPURaidenOffloadingConnector": offload_stats.to_dict(),
            }
        )
        assert rebuilt_multi is not None
        multi_prom.observe(rebuilt_multi.to_dict(), engine_idx=0)

        assert (
            registry.get_sample_value(
                "tpu_raiden_d2h_bytes_total",
                {
                    "model_name": "qwen3",
                    "engine": "0",
                    "host_ip": "10.0.0.9",
                    "local_rank": "0",
                },
            )
            == 2048.0
        )
        assert (
            registry.get_sample_value(
                "tpu_raiden_kv_recall_blocks_total",
                {
                    "model_name": "qwen3",
                    "engine": "0",
                    "dp_rank": "0",
                    "tier": "local_dram",
                    "outcome": "ok",
                },
            )
            == 4.0
        )


class TestRaidenPromMetricsDynamicMetricsProperty:
    def test_dynamic_metrics_delegates_to_shared_raiden_metrics(self) -> None:
        registry = CollectorRegistry()
        mock_telemetry = MagicMock()
        mock_telemetry.get_metric_metadata.return_value = _sample_raiden_descriptors()
        with (
            patch(
                "vllm_torchtpu.distributed.utils.get_raiden_telemetry_module",
                return_value=mock_telemetry,
            ),
            patch("vllm_torchtpu.distributed.utils.configure_raiden_telemetry"),
        ):
            prom = TPURaidenOffloadingConnector.build_prom_metrics(
                _make_offload_vllm_config(),  # type: ignore[arg-type]
                _make_metric_types(registry),  # type: ignore[arg-type]
                ["model_name", "engine"],
                {0: ["qwen3", "0"]},
            )
        assert isinstance(prom, TPURaidenOffloadingPromMetrics)
        assert prom.dynamic_metrics is prom._raiden_metrics.dynamic_metrics

        prom.dynamic_metrics.clear()
        assert prom._raiden_metrics.dynamic_metrics == {}
        # With no collectors, observing samples is a no-op instead of an error.
        prom.observe(
            {"raiden_samples": {"tpu_raiden_buffer_allocated_bytes": [1.0]}},
            engine_idx=0,
        )


class TestRaidenStatsIpcRoundTrip:
    @pytest.mark.parametrize("wrap_in_multi", [False, True])
    def test_stats_survive_engine_core_outputs_msgpack(
        self, wrap_in_multi: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # MultiConnector rebuilds child stats via the factory registry.
        monkeypatch.setattr(
            KVConnectorFactory, "_registry", dict(KVConnectorFactory._registry)
        )
        if "TPURaidenOffloadingConnector" not in KVConnectorFactory._registry:
            KVConnectorFactory.register_connector(
                "TPURaidenOffloadingConnector",
                "vllm_torchtpu.offload.raiden_connector",
                "TPURaidenOffloadingConnector",
            )
        base = OffloadingConnectorStats()
        base.increase_counter("vllm:kv_offload_load_bytes", 4096)
        recall_key = (
            'tpu_raiden_kv_recall_blocks_total{dp_rank="0",outcome="ok",'
            'tier="local_dram"}'
        )
        stats = RaidenOffloadingConnectorStats.from_base_stats(
            base, {recall_key: [3.0, 5.0]}
        )
        assert stats is not None

        payload: dict[str, Any]
        if wrap_in_multi:
            payload = MultiKVConnectorStats(
                data={"TPURaidenOffloadingConnector": stats}
            ).to_dict()
        else:
            payload = stats.to_dict()

        # Scheduler.make_stats() ships `to_dict()` inside SchedulerStats on the
        # EngineCoreOutputs msgpack channel to the frontend.
        outputs = EngineCoreOutputs(
            scheduler_stats=SchedulerStats(kv_connector_stats=payload)
        )
        decoded = MsgpackDecoder(EngineCoreOutputs).decode(
            MsgpackEncoder().encode(outputs)
        )
        assert decoded.scheduler_stats is not None
        wire = decoded.scheduler_stats.kv_connector_stats
        assert wire is not None

        if wrap_in_multi:
            rebuilt_multi = MultiConnector.build_kv_connector_stats(wire)
            assert isinstance(rebuilt_multi, MultiKVConnectorStats)
            rebuilt = rebuilt_multi["TPURaidenOffloadingConnector"]
        else:
            rebuilt = TPURaidenOffloadingConnector.build_kv_connector_stats(wire)

        assert isinstance(rebuilt, RaidenOffloadingConnectorStats)
        assert rebuilt.reduce() == {"vllm:kv_offload_load_bytes": 4096}
        assert rebuilt.data["raiden_samples"] == {recall_key: [3.0, 5.0]}
