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
"""Unit tests for TPU Raiden telemetry and KV connector stats/metrics integration."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from vllm_torchtpu.distributed import utils as dist_utils
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPURaidenConnectorWorker
from vllm_torchtpu.distributed.kv_transfer.tpu_connector_stats import (
    TpuKVConnectorPromMetrics,
    TpuKVConnectorStats,
)


class TestRaidenTelemetryUtils:
    """Tests for dist_utils.get_raiden_telemetry_module and
    configure_raiden_telemetry.
    """

    def test_get_raiden_telemetry_module_cached(self, monkeypatch):
        monkeypatch.setattr(dist_utils, "_RAIDEN_TELEMETRY_MODULE", None)
        monkeypatch.setattr(dist_utils, "_RAIDEN_TELEMETRY_IMPORT_ATTEMPTED", False)
        mock_impl = MagicMock()
        mock_kcm = MagicMock()
        mock_kcm._torch_impl.return_value = mock_impl
        mock_torch = MagicMock()
        mock_torch.kv_cache_manager = mock_kcm
        mock_api = MagicMock()
        mock_api.torch = mock_torch
        mock_tpu_sync = MagicMock()
        mock_tpu_sync.api = mock_api

        with patch.dict(
            "sys.modules",
            {
                "tpu_sync": mock_tpu_sync,
                "tpu_sync.api": mock_api,
                "tpu_sync.api.torch": mock_torch,
                "tpu_sync.api.torch.kv_cache_manager": mock_kcm,
            },
        ):
            mod1 = dist_utils.get_raiden_telemetry_module()
            assert mod1 is mock_impl
            mock_kcm._torch_impl.assert_called_once()

            # Second call should use cached module
            mod2 = dist_utils.get_raiden_telemetry_module()
            assert mod2 is mock_impl
            mock_kcm._torch_impl.assert_called_once()

    def test_get_raiden_telemetry_module_import_error(self, monkeypatch):
        monkeypatch.setattr(dist_utils, "_RAIDEN_TELEMETRY_MODULE", None)
        with patch.dict(
            "sys.modules",
            {"tpu_sync": None, "tpu_sync.api.torch.kv_cache_manager": None},
        ):
            assert dist_utils.get_raiden_telemetry_module() is None

    def test_get_raiden_telemetry_module_impl_exception(self, monkeypatch):
        monkeypatch.setattr(dist_utils, "_RAIDEN_TELEMETRY_MODULE", None)
        mock_kcm = MagicMock()
        mock_kcm._torch_impl.side_effect = RuntimeError(
            "Extension initialization error"
        )
        mock_torch = MagicMock()
        mock_torch.kv_cache_manager = mock_kcm
        mock_api = MagicMock()
        mock_api.torch = mock_torch
        mock_tpu_sync = MagicMock()
        mock_tpu_sync.api = mock_api

        with patch.dict(
            "sys.modules",
            {
                "tpu_sync": mock_tpu_sync,
                "tpu_sync.api": mock_api,
                "tpu_sync.api.torch": mock_torch,
                "tpu_sync.api.torch.kv_cache_manager": mock_kcm,
            },
        ):
            assert dist_utils.get_raiden_telemetry_module() is None

    def test_configure_raiden_telemetry_default_backends(self, monkeypatch):
        monkeypatch.setattr(dist_utils, "_RAIDEN_TELEMETRY_CONFIGURED", False)
        mock_telemetry = MagicMock()
        monkeypatch.setattr(
            dist_utils, "get_raiden_telemetry_module", lambda: mock_telemetry
        )

        dist_utils.configure_raiden_telemetry()
        mock_telemetry.configure_telemetry.assert_called_once_with(None)
        assert dist_utils._RAIDEN_TELEMETRY_CONFIGURED is True

    def test_configure_raiden_telemetry_explicit_backends(self, monkeypatch):
        monkeypatch.setattr(dist_utils, "_RAIDEN_TELEMETRY_CONFIGURED", False)
        mock_telemetry = MagicMock()
        monkeypatch.setattr(
            dist_utils, "get_raiden_telemetry_module", lambda: mock_telemetry
        )

        dist_utils.configure_raiden_telemetry(backends=["buffered", "custom_exporter"])
        mock_telemetry.configure_telemetry.assert_called_with(
            ["buffered", "custom_exporter"]
        )

        dist_utils.configure_raiden_telemetry(backends={"single_backend"})
        mock_telemetry.configure_telemetry.assert_called_with({"single_backend"})

    def test_configure_raiden_telemetry_idempotent_for_same_backends(self, monkeypatch):
        monkeypatch.setattr(dist_utils, "_RAIDEN_TELEMETRY_CONFIGURED", False)
        monkeypatch.setattr(dist_utils, "_RAIDEN_TELEMETRY_CONFIGURED_BACKENDS", None)
        mock_telemetry = MagicMock()
        monkeypatch.setattr(
            dist_utils, "get_raiden_telemetry_module", lambda: mock_telemetry
        )

        dist_utils.configure_raiden_telemetry(backends=["buffered"])
        dist_utils.configure_raiden_telemetry(backends=[" BUFFERED "])
        dist_utils.configure_raiden_telemetry()
        mock_telemetry.configure_telemetry.assert_called_once_with(["buffered"])

    def test_configure_raiden_telemetry_telemetry_none(self, monkeypatch):
        monkeypatch.setattr(dist_utils, "_RAIDEN_TELEMETRY_CONFIGURED", False)
        monkeypatch.setattr(dist_utils, "get_raiden_telemetry_module", lambda: None)

        dist_utils.configure_raiden_telemetry()
        assert dist_utils._RAIDEN_TELEMETRY_CONFIGURED is False

    def test_configure_raiden_telemetry_no_configure_method(self, monkeypatch):
        monkeypatch.setattr(dist_utils, "_RAIDEN_TELEMETRY_CONFIGURED", False)
        monkeypatch.setattr(dist_utils, "get_raiden_telemetry_module", lambda: object())

        dist_utils.configure_raiden_telemetry()
        assert dist_utils._RAIDEN_TELEMETRY_CONFIGURED is False

    def test_configure_raiden_telemetry_exception_handled(self, monkeypatch):
        monkeypatch.setattr(dist_utils, "_RAIDEN_TELEMETRY_CONFIGURED", False)
        mock_telemetry = MagicMock()
        mock_telemetry.configure_telemetry.side_effect = RuntimeError(
            "Failed to configure"
        )
        monkeypatch.setattr(
            dist_utils, "get_raiden_telemetry_module", lambda: mock_telemetry
        )

        # Should not raise exception
        dist_utils.configure_raiden_telemetry()
        assert dist_utils._RAIDEN_TELEMETRY_CONFIGURED is False


class TestTpuKVConnectorStatsDynamicMetrics:
    """Tests for TpuKVConnectorStats handling dynamic Raiden telemetry metrics."""

    def test_is_empty_fresh_and_reset(self):
        stats = TpuKVConnectorStats()
        assert stats.is_empty() is True

        stats.data = {}
        assert stats.is_empty() is True

        stats.reset()
        assert stats.is_empty() is True

    def test_is_empty_with_standard_metrics(self):
        stats = TpuKVConnectorStats()
        stats.record_prefill_queue_length(15)
        assert stats.is_empty() is False

    def test_is_empty_with_dynamic_telemetry_metrics(self):
        stats = TpuKVConnectorStats()
        stats.data["tpu_raiden_transfer_duration_ms"] = [1.2, 3.4]
        assert stats.is_empty() is False

        stats.data["tpu_raiden_transfer_duration_ms"] = []
        assert stats.is_empty() is True

    def test_aggregate_dynamic_metrics(self):
        stats1 = TpuKVConnectorStats()
        stats2 = TpuKVConnectorStats()

        stats1.data["tpu_raiden_transfer_duration_ms"] = [1.0]
        stats2.data["tpu_raiden_transfer_duration_ms"] = [2.0, 3.0]
        stats2.data["tpu_raiden_sent_bytes_total"] = [100]

        stats1.aggregate(stats2)
        assert stats1.data["tpu_raiden_transfer_duration_ms"] == [1.0, 2.0, 3.0]
        assert stats1.data["tpu_raiden_sent_bytes_total"] == [100]

    def test_clone_and_reset_dynamic_metrics(self):
        stats = TpuKVConnectorStats()
        stats.data["tpu_raiden_sent_bytes_total"] = [1, 2]
        stats.record_prefill_queue_length(10)

        cloned = stats.clone_and_reset()
        assert cloned.data["tpu_raiden_sent_bytes_total"] == [1, 2]
        assert cloned.data["prefill_queue_length"] == [10]

        assert stats.is_empty() is True
        assert "tpu_raiden_sent_bytes_total" not in stats.data


class TestTpuKVConnectorPromMetricsDynamicMetrics:
    """Tests for TpuKVConnectorPromMetrics dynamic metric
    registration and observations.
    """

    def _setup_prom_metrics(self, descriptors):
        registry = CollectorRegistry()
        metric_types = {
            Gauge: partial(Gauge, registry=registry),
            Counter: partial(Counter, registry=registry),
            Histogram: partial(Histogram, registry=registry),
        }
        labelnames = ["model_name", "engine"]
        per_engine_labelvalues = {0: ["my_model", "0"]}

        mock_telemetry = MagicMock()
        mock_telemetry.get_metric_metadata.return_value = descriptors

        with (
            patch(
                "vllm_torchtpu.distributed.utils.get_raiden_telemetry_module",
                return_value=mock_telemetry,
            ),
            patch(
                "vllm_torchtpu.distributed.utils.configure_raiden_telemetry"
            ) as mock_conf,
        ):
            metrics = TpuKVConnectorPromMetrics(
                vllm_config=MagicMock(),
                metric_types=metric_types,
                labelnames=labelnames,
                per_engine_labelvalues=per_engine_labelvalues,
            )
            mock_conf.assert_called_once()
        return metrics

    def test_dynamic_metric_registration(self):
        descriptors = [
            SimpleNamespace(
                name="transfer_duration_ms",
                type="HISTOGRAM",
                buckets=[0.5, 1.0, 5.0],
                description="Transfer duration in milliseconds",
            ),
            SimpleNamespace(
                name="sent_bytes_total",
                type="COUNTER",
                description="Total number of sent bytes",
            ),
            SimpleNamespace(
                name="buffer_allocated_bytes",
                type="GAUGE",
                description="Buffer allocated bytes",
            ),
            SimpleNamespace(
                name="unsupported_summary",
                type="SUMMARY",
                description="Unsupported metric type",
            ),
        ]

        metrics = self._setup_prom_metrics(descriptors)

        assert "tpu_raiden_transfer_duration_ms" in metrics.dynamic_metrics
        assert "tpu_raiden_sent_bytes_total" in metrics.dynamic_metrics
        assert "tpu_raiden_buffer_allocated_bytes" in metrics.dynamic_metrics
        assert "tpu_raiden_unsupported_summary" not in metrics.dynamic_metrics

        # Verify metric types
        assert (
            metrics.dynamic_metrics["tpu_raiden_transfer_duration_ms"][0] == "histogram"
        )
        assert metrics.dynamic_metrics["tpu_raiden_sent_bytes_total"][0] == "counter"
        assert (
            metrics.dynamic_metrics["tpu_raiden_buffer_allocated_bytes"][0] == "gauge"
        )

    def test_dynamic_metric_observe(self):
        descriptors = [
            SimpleNamespace(
                name="transfer_duration_ms",
                type="histogram",
                buckets=[1.0, 10.0, 100.0],
                description="Transfer duration",
            ),
            SimpleNamespace(
                name="sent_bytes_total",
                type="counter",
                description="Sent bytes total",
            ),
            SimpleNamespace(
                name="buffer_allocated_bytes",
                type="gauge",
                description="Buffer allocated bytes",
            ),
        ]

        metrics = self._setup_prom_metrics(descriptors)

        data = {
            "prefill_queue_length": [],
            "decode_queue_length": [],
            "tpu_raiden_transfer_duration_ms": [0.5, 5.0, 50.0],
            "tpu_raiden_sent_bytes_total": [10, 20],
            "tpu_raiden_buffer_allocated_bytes": [4, 7],
        }

        metrics.observe(data, engine_idx=0)

        # Verify histogram observation
        hist = metrics.dynamic_metrics["tpu_raiden_transfer_duration_ms"][1][0]
        assert hist._sum.get() == 55.5
        assert [b.get() for b in hist._buckets] == [1.0, 1.0, 1.0, 0.0]

        # Verify counter observation
        counter = metrics.dynamic_metrics["tpu_raiden_sent_bytes_total"][1][0]
        assert counter._value.get() == 30.0

        # Verify gauge observation (last value wins: set(4) then set(7))
        gauge = metrics.dynamic_metrics["tpu_raiden_buffer_allocated_bytes"][1].labels(
            model_name="my_model", engine="0", local_rank=""
        )
        assert gauge._value.get() == 7.0

    def test_dynamic_metric_telemetry_exception_handled(self):
        registry = CollectorRegistry()
        metric_types = {
            Gauge: partial(Gauge, registry=registry),
            Counter: partial(Counter, registry=registry),
            Histogram: partial(Histogram, registry=registry),
        }
        labelnames = ["model_name", "engine"]
        per_engine_labelvalues = {0: ["my_model", "0"]}

        mock_telemetry = MagicMock()
        mock_telemetry.get_metric_metadata.side_effect = RuntimeError(
            "Failed to read metadata"
        )

        with patch(
            "vllm_torchtpu.distributed.utils.get_raiden_telemetry_module",
            return_value=mock_telemetry,
        ):
            metrics = TpuKVConnectorPromMetrics(
                vllm_config=MagicMock(),
                metric_types=metric_types,
                labelnames=labelnames,
                per_engine_labelvalues=per_engine_labelvalues,
            )

        assert metrics.dynamic_metrics == {}
        # Observe should still succeed without crashing
        metrics.observe({"prefill_queue_length": [], "decode_queue_length": []})

    def test_configure_exception_does_not_raise_unbound_local(self):
        registry = CollectorRegistry()
        metric_types = {
            Gauge: partial(Gauge, registry=registry),
            Counter: partial(Counter, registry=registry),
            Histogram: partial(Histogram, registry=registry),
        }
        with patch(
            "vllm_torchtpu.distributed.utils.configure_raiden_telemetry",
            side_effect=RuntimeError("configure boom"),
        ):
            metrics = TpuKVConnectorPromMetrics(
                vllm_config=MagicMock(),
                metric_types=metric_types,
                labelnames=["model_name", "engine"],
                per_engine_labelvalues={0: ["my_model", "0"]},
            )
        assert metrics.dynamic_metrics == {}

    def test_prometheus_only_mode_skips_dynamic_collector_registration(self):
        registry = CollectorRegistry()
        metric_types = {
            Gauge: partial(Gauge, registry=registry),
            Counter: partial(Counter, registry=registry),
            Histogram: partial(Histogram, registry=registry),
        }
        vllm_config = MagicMock()
        vllm_config.additional_config = {"raiden_telemetry_backends": ["prometheus"]}
        with patch(
            "vllm_torchtpu.distributed.utils.configure_raiden_telemetry"
        ) as mock_conf:
            metrics = TpuKVConnectorPromMetrics(
                vllm_config=vllm_config,
                metric_types=metric_types,
                labelnames=["model_name", "engine"],
                per_engine_labelvalues={0: ["my_model", "0"]},
            )
            mock_conf.assert_not_called()
        assert metrics.dynamic_metrics == {}


class TestRaidenBufferedTelemetryGating:
    """Tests for additional_config Raiden telemetry gating and worker rank tagging."""

    @pytest.mark.parametrize(
        ("additional_config", "expected"),
        [
            (None, ["buffered"]),
            ({}, ["buffered"]),
            ({"raiden_telemetry_backends": None}, ["buffered"]),
            ({"raiden_telemetry_backends": ["buffered"]}, ["buffered"]),
            ({"raiden_telemetry_backends": [" BUFFERED "]}, ["buffered"]),
            (
                {"raiden_telemetry_backends": ["prometheus", "buffered"]},
                ["prometheus", "buffered"],
            ),
            ({"raiden_telemetry_backends": []}, []),
        ],
    )
    def test_get_raiden_telemetry_backends(self, additional_config, expected):
        cfg = MagicMock()
        cfg.additional_config = additional_config
        assert dist_utils.get_raiden_telemetry_backends(cfg) == expected

    @pytest.mark.parametrize("bad_value", ["buffered", [123], [""], ["  "]])
    def test_get_raiden_telemetry_backends_rejects_invalid_values(self, bad_value):
        cfg = MagicMock()
        cfg.additional_config = {"raiden_telemetry_backends": bad_value}
        with pytest.raises(ValueError, match="raiden_telemetry_backends"):
            dist_utils.get_raiden_telemetry_backends(cfg)

    @pytest.mark.parametrize(
        ("backends", "expected"),
        [
            (None, True),
            (["buffered"], True),
            (["BUFFERED"], True),
            (["prometheus", "buffered"], True),
            ([], False),
            (["prometheus"], False),
            (["cloud_logging"], False),
            # cloud_logging drains the same C++ buffer, so vLLM must not.
            (["buffered", "cloud_logging"], False),
        ],
    )
    def test_is_raiden_buffered_telemetry_enabled(self, backends, expected):
        assert dist_utils.is_raiden_buffered_telemetry_enabled(backends) is expected

    def test_drain_skips_non_buffered_modes(self, monkeypatch):
        mock_telemetry = MagicMock()
        monkeypatch.setattr(
            dist_utils, "get_raiden_telemetry_module", lambda: mock_telemetry
        )
        for backends in (
            ["prometheus"],
            ["cloud_logging"],
            ["buffered", "cloud_logging"],
        ):
            assert dist_utils.drain_raiden_metric_samples(backends=backends) == {}
        mock_telemetry.get_and_reset_metric_samples.assert_not_called()

    def test_drain_tags_worker_gauges_without_rank_label(self, monkeypatch):
        mock_telemetry = MagicMock()
        mock_telemetry.get_metric_metadata.return_value = [
            SimpleNamespace(name="buffer_allocated_bytes", type="GAUGE"),
            SimpleNamespace(name="host_memory_breakdown_bytes", type="GAUGE"),
            SimpleNamespace(name="host_numa_memory_bytes", type="GAUGE"),
            SimpleNamespace(name="host_pool_blocks", type="GAUGE"),
            SimpleNamespace(name="d2h_bytes_total", type="COUNTER"),
        ]
        mock_telemetry.get_and_reset_metric_samples.return_value = {
            "tpu_raiden_buffer_allocated_bytes": [2048.0],
            'tpu_raiden_host_memory_breakdown_bytes{component="kv_pool"}': [4096.0],
            'tpu_raiden_host_numa_memory_bytes{local_rank="1",numa_node="0"}': [8192.0],
            # `dp_rank` must not be mistaken for `rank`.
            'tpu_raiden_host_pool_blocks{dp_rank="0",state="used"}': [16.0],
            'tpu_raiden_d2h_bytes_total{host_ip="10.0.0.1"}': [512.0],
        }
        monkeypatch.setattr(
            dist_utils, "get_raiden_telemetry_module", lambda: mock_telemetry
        )

        samples = dist_utils.drain_raiden_metric_samples(
            backends=["buffered"], worker_rank=1
        )
        breakdown_key = (
            'tpu_raiden_host_memory_breakdown_bytes{component="kv_pool",local_rank="1"}'
        )
        pool_key = (
            'tpu_raiden_host_pool_blocks{dp_rank="0",state="used",local_rank="1"}'
        )
        assert samples == {
            'tpu_raiden_buffer_allocated_bytes{local_rank="1"}': [2048.0],
            breakdown_key: [4096.0],
            'tpu_raiden_host_numa_memory_bytes{local_rank="1",numa_node="0"}': [8192.0],
            pool_key: [16.0],
            'tpu_raiden_d2h_bytes_total{host_ip="10.0.0.1"}': [512.0],
        }

    @pytest.mark.parametrize(
        ("parallel_config", "expected"),
        [
            (None, None),
            (SimpleNamespace(world_size=1, rank=0), None),
            (
                SimpleNamespace(
                    tensor_parallel_size=1,
                    pipeline_parallel_size=1,
                    prefill_context_parallel_size=1,
                    rank=0,
                ),
                None,
            ),
            (SimpleNamespace(world_size=2, rank=1), 1),
            (
                SimpleNamespace(
                    tensor_parallel_size=1,
                    pipeline_parallel_size=1,
                    prefill_context_parallel_size=2,
                    rank=1,
                ),
                1,
            ),
        ],
    )
    def test_get_raiden_worker_telemetry_rank(self, parallel_config, expected):
        cfg = SimpleNamespace(parallel_config=parallel_config)
        assert dist_utils.get_raiden_worker_telemetry_rank(cfg) == expected

    def test_pcp_workers_with_same_tp_rank_emit_distinct_gauge_series(
        self, monkeypatch
    ):
        # Two PCP ranks both have tp_rank=0, so tagging must use
        # parallel_config.rank rather than tp_rank.
        mock_telemetry = MagicMock()
        mock_telemetry.get_metric_metadata.return_value = [
            SimpleNamespace(
                name="buffer_allocated_bytes",
                type="GAUGE",
                label_names=[],
                description="Allocated host staging bytes",
            )
        ]
        monkeypatch.setattr(
            dist_utils, "get_raiden_telemetry_module", lambda: mock_telemetry
        )

        def make_pcp_worker(global_rank: int) -> TPURaidenConnectorWorker:
            w = object.__new__(TPURaidenConnectorWorker)
            w.vllm_config = SimpleNamespace(
                parallel_config=SimpleNamespace(
                    world_size=2,
                    tensor_parallel_size=1,
                    pipeline_parallel_size=1,
                    prefill_context_parallel_size=2,
                    rank=global_rank,
                )
            )
            w.tp_rank = 0
            w.owns_raiden_telemetry_drain = True
            w._raiden_telemetry_backends = ["buffered"]
            w._worker_telemetry_rank = dist_utils.get_raiden_worker_telemetry_rank(
                w.vllm_config
            )
            w.transfer_stats = TpuKVConnectorStats()
            return w

        w0 = make_pcp_worker(0)
        w1 = make_pcp_worker(1)

        mock_telemetry.get_and_reset_metric_samples.return_value = {
            "tpu_raiden_buffer_allocated_bytes": [1024.0]
        }
        w0._get_raiden_stats()

        mock_telemetry.get_and_reset_metric_samples.return_value = {
            "tpu_raiden_buffer_allocated_bytes": [2048.0]
        }
        w1._get_raiden_stats()

        agg = w0.transfer_stats.aggregate(w1.transfer_stats)
        registry = CollectorRegistry()
        metric_types = {
            Gauge: partial(Gauge, registry=registry),
            Counter: partial(Counter, registry=registry),
            Histogram: partial(Histogram, registry=registry),
        }
        with patch("vllm_torchtpu.distributed.utils.configure_raiden_telemetry"):
            prom = TpuKVConnectorPromMetrics(
                vllm_config=MagicMock(),
                metric_types=metric_types,
                labelnames=["model_name", "engine"],
                per_engine_labelvalues={0: ["my_model", "0"]},
            )
        prom.observe(agg.to_dict(), engine_idx=0)
        assert (
            registry.get_sample_value(
                "tpu_raiden_buffer_allocated_bytes",
                {"model_name": "my_model", "engine": "0", "local_rank": "0"},
            )
            == 1024.0
        )
        assert (
            registry.get_sample_value(
                "tpu_raiden_buffer_allocated_bytes",
                {"model_name": "my_model", "engine": "0", "local_rank": "1"},
            )
            == 2048.0
        )


class TestTPURaidenConnectorWorkerTelemetry:
    """Tests for TPURaidenConnectorWorker telemetry collection and stats reporting."""

    def _create_mock_worker(self, is_producer=True, tp_rank=0):
        worker = MagicMock(spec=TPURaidenConnectorWorker)
        worker.owns_raiden_telemetry_drain = True
        worker._get_raiden_stats = TPURaidenConnectorWorker._get_raiden_stats.__get__(
            worker
        )
        worker.tp_rank = tp_rank
        worker.tp_size = 1
        worker.dp_rank = 0
        worker.kv_transfer_port = 9100
        worker.is_producer = is_producer
        worker._stage3_registered_sends = {1: MagicMock(), 2: MagicMock()}
        worker._stage3_submitted_loads = {
            1: MagicMock(),
            2: MagicMock(),
            3: MagicMock(),
        }
        worker._stage3_terminal_loads = {1: MagicMock()}
        worker.transfer_stats = TpuKVConnectorStats()
        return worker

    def test_worker_get_kv_connector_stats_collects_raiden_metrics(self):
        worker = self._create_mock_worker(is_producer=True, tp_rank=0)

        mock_telemetry = MagicMock()
        mock_telemetry.get_and_reset_metric_samples.return_value = {
            "tpu_raiden_transfer_duration_ms": [1.5, 3.5],
            "tpu_raiden_sent_bytes_total": [1024, 2048],
            "tpu_raiden_buffer_allocated_bytes": [4096],
        }

        with patch(
            "vllm_torchtpu.distributed.utils.get_raiden_telemetry_module",
            return_value=mock_telemetry,
        ):
            stats = TPURaidenConnectorWorker.get_kv_connector_stats(worker)

        assert stats is not None
        assert stats.data["tpu_raiden_transfer_duration_ms"] == [1.5, 3.5]
        assert stats.data["tpu_raiden_sent_bytes_total"] == [1024, 2048]
        assert stats.data["tpu_raiden_buffer_allocated_bytes"] == [4096]
        assert stats.data["prefill_queue_length"] == [2]

        # Worker's stats should have been cloned and reset
        assert worker.transfer_stats.is_empty() is True

    def test_worker_get_kv_connector_stats_consumer_queue_len(self):
        worker = self._create_mock_worker(is_producer=False, tp_rank=0)

        mock_telemetry = MagicMock()
        mock_telemetry.get_and_reset_metric_samples.return_value = {
            "tpu_raiden_transfer_duration_ms": [10.0],
        }

        with patch(
            "vllm_torchtpu.distributed.utils.get_raiden_telemetry_module",
            return_value=mock_telemetry,
        ):
            stats = TPURaidenConnectorWorker.get_kv_connector_stats(worker)

        assert stats is not None
        assert stats.data["tpu_raiden_transfer_duration_ms"] == [10.0]
        # 3 submitted - 1 terminal = 2
        assert stats.data["decode_queue_length"] == [2]

    def test_worker_get_kv_connector_stats_non_zero_rank_no_queue_len(self):
        worker = self._create_mock_worker(is_producer=True, tp_rank=1)

        mock_telemetry = MagicMock()
        mock_telemetry.get_and_reset_metric_samples.return_value = {
            "tpu_raiden_transfer_duration_ms": [2.0],
        }

        with patch(
            "vllm_torchtpu.distributed.utils.get_raiden_telemetry_module",
            return_value=mock_telemetry,
        ):
            stats = TPURaidenConnectorWorker.get_kv_connector_stats(worker)

        assert stats is not None
        assert stats.data["tpu_raiden_transfer_duration_ms"] == [2.0]
        assert (
            "prefill_queue_length" not in stats.data
            or stats.data["prefill_queue_length"] == []
        )

    def test_worker_get_kv_connector_stats_telemetry_exception_graceful(self):
        worker = self._create_mock_worker(is_producer=True, tp_rank=0)

        mock_telemetry = MagicMock()
        mock_telemetry.get_and_reset_metric_samples.side_effect = RuntimeError(
            "Sampling error"
        )

        with patch(
            "vllm_torchtpu.distributed.utils.get_raiden_telemetry_module",
            return_value=mock_telemetry,
        ):
            stats = TPURaidenConnectorWorker.get_kv_connector_stats(worker)

        # Should still record queue lengths on tp_rank 0 without crashing
        assert stats is not None
        assert stats.data["prefill_queue_length"] == [2]

    def test_worker_get_kv_connector_stats_empty_returns_none(self):
        worker = self._create_mock_worker(is_producer=True, tp_rank=1)

        mock_telemetry = MagicMock()
        mock_telemetry.get_and_reset_metric_samples.return_value = {}

        with patch(
            "vllm_torchtpu.distributed.utils.get_raiden_telemetry_module",
            return_value=mock_telemetry,
        ):
            stats = TPURaidenConnectorWorker.get_kv_connector_stats(worker)

        assert stats is None
