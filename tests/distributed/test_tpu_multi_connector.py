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
"""Unit tests for TPUMultiConnector.

TPUMultiConnector adds TPU-specific method fan-out on top of upstream
MultiConnector. Children come entirely from the explicit
``kv_connector_extra_config["connectors"]`` list. The tests here run the
real upstream __init__, with fake child connector classes resolved from this
module via kv_connector_module_path (the same mechanism the recipe uses for
the real children). No TPU or network access is required.
"""

from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorPromMetrics,
)

from vllm_torchtpu.distributed import utils as dist_utils
from vllm_torchtpu.distributed.kv_transfer import tpu_multi_connector
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
    TPUConnector,
    TPUConnectorWorker,
    TPURaidenConnector,
    TPURaidenConnectorWorker,
)
from vllm_torchtpu.distributed.kv_transfer.tpu_connector_stats import (
    TpuKVConnectorPromMetrics,
    TpuKVConnectorStats,
)
from vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector import TPUMultiConnector
from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector


class _FakeRaidenChild:
    """Stands in for TPURaidenConnector: accepts a runner."""

    def __init__(self, vllm_config, role, kv_cache_config):
        self.vllm_config = vllm_config
        self.role = role
        self.kv_cache_config = kv_cache_config
        self.runner = None

    def register_runner(self, runner):
        self.runner = runner

    def get_block_ids_with_load_errors_group_index(self):
        return 2


class _FakeOffloadChild:
    """Stands in for OffloadingConnector: no register_runner, like
    upstream connectors."""

    def __init__(self, vllm_config, role, kv_cache_config):
        self.vllm_config = vllm_config
        self.role = role
        self.kv_cache_config = kv_cache_config


class _CapableChild(_FakeOffloadChild):
    """Completes divergent local hybrid hits, like a prefix-aware consumer."""

    supports_divergent_local_hybrid_hits = True


class _IncapableChild(_FakeOffloadChild):
    supports_divergent_local_hybrid_hits = False


class _DrainingRaidenChild(TPUConnector):
    """Real TPUConnector subclass without TPU/Raiden setup."""

    def __init__(self, vllm_config, role, kv_cache_config):
        self.vllm_config = vllm_config
        self._role = role
        self.kv_cache_config = kv_cache_config
        if role == KVConnectorRole.WORKER:
            worker = MagicMock(spec=TPURaidenConnectorWorker)
            worker.owns_raiden_telemetry_drain = True
            self.connector_worker = worker
        else:
            self.connector_worker = None


class _DrainingOffloadChild(TPURaidenOffloadingConnector):
    """Real TPURaidenOffloadingConnector subclass without TPU/store setup."""

    def __init__(self, vllm_config, role, kv_cache_config):
        self.vllm_config = vllm_config
        self._role = role
        self.kv_cache_config = kv_cache_config


def _make_vllm_config(
    kv_role="kv_consumer", children=("_FakeRaidenChild", "_FakeOffloadChild")
):
    # Mirrors the disagg-offloading recipe shape (raiden child inheriting the
    # outer role + offloading child as kv_both), with the child classes
    # pointed at this module instead of the real connectors.
    first, second = children
    return SimpleNamespace(
        kv_transfer_config=KVTransferConfig(
            kv_connector="TPUMultiConnector",
            kv_connector_module_path=(
                "vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector"
            ),
            kv_role=kv_role,
            kv_connector_extra_config={
                "connectors": [
                    {
                        "kv_connector": first,
                        "kv_connector_module_path": __name__,
                        "kv_role": kv_role,
                    },
                    {
                        "kv_connector": second,
                        "kv_connector_module_path": __name__,
                        "kv_role": "kv_both",
                    },
                ],
            },
        ),
        # The fake children don't implement SupportsHMA, so hybrid KV cache
        # management must be off for upstream __init__'s HMA assert.
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
    )


def _build(role=KVConnectorRole.WORKER):
    return TPUMultiConnector(_make_vllm_config(), role, MagicMock())


class TestTPUMultiConnectorConstruction:
    def test_children_built_from_explicit_connectors_config(self):
        connector = _build()

        raiden, offload = connector._connectors
        assert isinstance(raiden, _FakeRaidenChild)
        assert isinstance(offload, _FakeOffloadChild)

    def test_each_child_gets_its_own_kv_transfer_config(self):
        connector = _build()

        raiden, offload = connector._connectors
        assert raiden.vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        assert offload.vllm_config.kv_transfer_config.kv_role == "kv_both"

    def test_role_and_kv_cache_config_forwarded_to_children(self):
        kv_cache_config = MagicMock()
        connector = TPUMultiConnector(
            _make_vllm_config(), KVConnectorRole.WORKER, kv_cache_config
        )

        for child in connector._connectors:
            assert child.role == KVConnectorRole.WORKER
            assert child.kv_cache_config is kv_cache_config


class TestTPUMultiConnectorRegisterRunner:
    def test_runner_reaches_children_that_accept_it(self):
        # tpu_runner dispatches register_runner via hasattr; upstream
        # MultiConnector doesn't forward it, so without this fan-out the
        # Raiden child fails with "register_runner must be called before
        # transfer" on the first load. The offload child has no
        # register_runner and must be skipped without raising.
        connector = _build()
        runner = MagicMock()

        connector.register_runner(runner)

        raiden, _ = connector._connectors
        assert raiden.runner is runner

    def test_upstream_still_lacks_register_runner(self):
        # The override exists only because upstream MultiConnector doesn't
        # forward register_runner; if this starts failing, upstream grew
        # native support and the override should be dropped.
        assert not hasattr(MultiConnector, "register_runner")


def test_failed_load_group_index_reaches_child_connector():
    assert _build().get_block_ids_with_load_errors_group_index() == 2


def test_conflicting_group_indices_across_children_raise():
    # A single scalar can't scope errors from two groups; mis-scoping would
    # make the scheduler recompute the wrong group's blocks, so it must fail
    # loudly instead.
    connector = _build()
    conflicting = MagicMock()
    conflicting.get_block_ids_with_load_errors_group_index.return_value = 3
    connector._connectors.append(conflicting)

    with pytest.raises(RuntimeError, match="Conflicting"):
        connector.get_block_ids_with_load_errors_group_index()


class TestTPUMultiConnectorDivergentHitWarning:
    @pytest.mark.parametrize(
        ("children", "warns"),
        [
            # One child turns suffix loads off for the whole engine.
            (("_CapableChild", "_IncapableChild"), True),
            # A child without the property reads as incapable, as upstream.
            (("_CapableChild", "_FakeOffloadChild"), True),
            (("_CapableChild", "_CapableChild"), False),
            # Nothing asked for suffix loads, so nothing was lost.
            (("_IncapableChild", "_FakeOffloadChild"), False),
        ],
    )
    def test_warns_when_a_child_disables_divergent_hits(
        self, children, warns, monkeypatch
    ):
        log = MagicMock()
        monkeypatch.setattr(tpu_multi_connector, "logger", log)

        TPUMultiConnector(
            _make_vllm_config(children=children),
            KVConnectorRole.SCHEDULER,
            MagicMock(),
        )

        assert log.warning.called is warns

    def test_workers_do_not_warn(self, monkeypatch):
        # Only the scheduler consults the capability.
        log = MagicMock()
        monkeypatch.setattr(tpu_multi_connector, "logger", log)

        TPUMultiConnector(
            _make_vllm_config(children=("_CapableChild", "_IncapableChild")),
            KVConnectorRole.WORKER,
            MagicMock(),
        )

        assert not log.warning.called

    def test_producers_do_not_warn(self, monkeypatch):
        # Prefill wraps a producer transfer child, which always reports False,
        # with an offload child that reports True when the flag is on. Only a
        # consumer starts from a decode-local hit, so that mix loses nothing.
        log = MagicMock()
        monkeypatch.setattr(tpu_multi_connector, "logger", log)

        TPUMultiConnector(
            _make_vllm_config(
                kv_role="kv_producer", children=("_IncapableChild", "_CapableChild")
            ),
            KVConnectorRole.SCHEDULER,
            MagicMock(),
        )

        assert not log.warning.called


class TestTPUMultiConnectorTelemetryDrainOwnership:
    @pytest.mark.parametrize(
        "children",
        [
            ("_DrainingRaidenChild", "_DrainingOffloadChild"),
            ("_DrainingOffloadChild", "_DrainingRaidenChild"),
        ],
    )
    def test_init_assigns_worker_drain_owner_to_raiden_child(self, children):
        connector = TPUMultiConnector(
            _make_vllm_config(children=children),
            KVConnectorRole.WORKER,
            MagicMock(),
        )

        offload = next(
            c for c in connector._connectors if isinstance(c, _DrainingOffloadChild)
        )
        raiden = next(
            c for c in connector._connectors if isinstance(c, _DrainingRaidenChild)
        )
        assert raiden.owns_raiden_telemetry_drain is True
        assert offload.owns_raiden_telemetry_drain is False

    @pytest.mark.parametrize(
        "children",
        [
            ("_DrainingRaidenChild", "_DrainingOffloadChild"),
            ("_DrainingOffloadChild", "_DrainingRaidenChild"),
        ],
    )
    def test_init_assigns_scheduler_drain_owner_to_offload_child(self, children):
        connector = TPUMultiConnector(
            _make_vllm_config(children=children),
            KVConnectorRole.SCHEDULER,
            MagicMock(),
        )

        offload = next(
            c for c in connector._connectors if isinstance(c, _DrainingOffloadChild)
        )
        raiden = next(
            c for c in connector._connectors if isinstance(c, _DrainingRaidenChild)
        )
        assert raiden.owns_raiden_telemetry_drain is False
        assert offload.owns_raiden_telemetry_drain is True


def _bare_raiden_connector(
    worker: TPURaidenConnectorWorker | None,
) -> TPURaidenConnector:
    conn = object.__new__(TPURaidenConnector)
    conn.connector_worker = worker  # type: ignore[assignment]
    conn.connector_scheduler = None  # type: ignore[assignment]
    return conn


def _bare_raiden_worker() -> TPURaidenConnectorWorker:
    worker = object.__new__(TPURaidenConnectorWorker)
    worker.transfer_stats = TpuKVConnectorStats()
    # tp_rank != 0 skips the queue-length gauges, which need full worker state.
    worker.tp_rank = 1
    return worker


def _bare_offload_connector(
    *, scheduler: object | None = None, worker: object | None = None
) -> TPURaidenOffloadingConnector:
    conn = object.__new__(TPURaidenOffloadingConnector)
    conn.connector_scheduler = scheduler  # type: ignore[assignment]
    conn.connector_worker = worker  # type: ignore[assignment]
    return conn


def _bare_multi_connector(children: list[object]) -> TPUMultiConnector:
    multi = object.__new__(TPUMultiConnector)
    multi._connectors = children  # type: ignore[assignment]
    multi._assign_raiden_telemetry_drain_owner()
    return multi


class TestRaidenTelemetryDrainOwnership:
    def test_non_owner_offload_connector_does_not_drain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_telemetry = MagicMock()
        monkeypatch.setattr(
            dist_utils, "get_raiden_telemetry_module", lambda: mock_telemetry
        )
        conn = _bare_offload_connector(worker=MagicMock())
        conn.owns_raiden_telemetry_drain = False

        assert conn.get_kv_connector_stats() is None
        mock_telemetry.get_and_reset_metric_samples.assert_not_called()

    def test_non_owner_raiden_worker_does_not_drain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_telemetry = MagicMock()
        monkeypatch.setattr(
            dist_utils, "get_raiden_telemetry_module", lambda: mock_telemetry
        )
        conn = _bare_raiden_connector(_bare_raiden_worker())
        conn.owns_raiden_telemetry_drain = False

        assert conn.get_kv_connector_stats() is None
        mock_telemetry.get_and_reset_metric_samples.assert_not_called()

    def test_raiden_connector_without_worker_never_owns_drain(self) -> None:
        conn = _bare_raiden_connector(None)
        assert conn.owns_raiden_telemetry_drain is False
        # The setter is a no-op without a worker.
        conn.owns_raiden_telemetry_drain = True
        assert conn.owns_raiden_telemetry_drain is False

    def test_zmq_worker_never_owns_raiden_drain(self) -> None:
        zmq_worker = MagicMock(spec=TPUConnectorWorker)
        conn = _bare_raiden_connector(zmq_worker)  # type: ignore[arg-type]
        assert conn.owns_raiden_telemetry_drain is False
        conn.owns_raiden_telemetry_drain = True
        assert conn.owns_raiden_telemetry_drain is False
        assert not hasattr(zmq_worker, "owns_raiden_telemetry_drain")

    @pytest.mark.parametrize("offload_first", [False, True])
    def test_multi_connector_keeps_worker_drain_on_raiden_connector(
        self, offload_first: bool
    ) -> None:
        raiden = _bare_raiden_connector(_bare_raiden_worker())
        offload = _bare_offload_connector(worker=MagicMock())
        children = [offload, raiden] if offload_first else [raiden, offload]
        _bare_multi_connector(children)

        assert raiden.owns_raiden_telemetry_drain is True
        assert offload.owns_raiden_telemetry_drain is False

    def test_multi_connector_scheduler_role_keeps_offload_owner(self) -> None:
        # The scheduler-side TPURaidenConnector has no worker and never drains.
        raiden = _bare_raiden_connector(None)
        offload = _bare_offload_connector(scheduler=MagicMock())
        _bare_multi_connector([raiden, offload])

        assert offload.owns_raiden_telemetry_drain is True
        assert raiden.owns_raiden_telemetry_drain is False

    @pytest.mark.parametrize("offload_first", [False, True])
    def test_multi_connector_observes_each_worker_sample_once(
        self, monkeypatch: pytest.MonkeyPatch, offload_first: bool
    ) -> None:
        d2h_key = 'tpu_raiden_d2h_bytes_total{host_ip="10.0.0.9",local_rank="0"}'
        mock_telemetry = MagicMock()
        # The C++ buffer resets on read: one batch, then nothing.
        mock_telemetry.get_and_reset_metric_samples.side_effect = [
            {d2h_key: [2048.0]},
            {},
        ]
        mock_telemetry.get_metric_metadata.return_value = [
            SimpleNamespace(
                name="d2h_bytes_total",
                type="COUNTER",
                label_names=["host_ip", "local_rank"],
                description="Total D2H DMA bytes",
            ),
        ]
        monkeypatch.setattr(
            dist_utils, "get_raiden_telemetry_module", lambda: mock_telemetry
        )

        raiden = _bare_raiden_connector(_bare_raiden_worker())
        offload = _bare_offload_connector(worker=MagicMock())
        children = [offload, raiden] if offload_first else [raiden, offload]
        multi = _bare_multi_connector(children)

        stats = MultiConnector.get_kv_connector_stats(multi)
        assert stats is not None
        assert set(stats.data) == {"TPURaidenConnector"}
        assert isinstance(stats["TPURaidenConnector"], TpuKVConnectorStats)
        assert stats["TPURaidenConnector"].data[d2h_key] == [2048.0]
        assert mock_telemetry.get_and_reset_metric_samples.call_count == 1

        registry = CollectorRegistry()
        metric_types = {
            Gauge: partial(Gauge, registry=registry),
            Counter: partial(Counter, registry=registry),
            Histogram: partial(Histogram, registry=registry),
        }
        offload_cfg = SimpleNamespace(
            kv_transfer_config=KVTransferConfig(
                kv_connector="TPURaidenOffloadingConnector",
                kv_connector_module_path="vllm_torchtpu.offload.raiden_connector",
                kv_role="kv_both",
                kv_connector_extra_config={"cpu_bytes_to_use": 1 << 20},
            ),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
        )
        with patch("vllm_torchtpu.distributed.utils.configure_raiden_telemetry"):
            raiden_prom = TpuKVConnectorPromMetrics(
                vllm_config=MagicMock(),
                metric_types=metric_types,  # type: ignore[arg-type]
                labelnames=["model_name", "engine"],
                per_engine_labelvalues={0: ["qwen3", "0"]},
            )
            offload_prom = TPURaidenOffloadingConnector.build_prom_metrics(
                offload_cfg,  # type: ignore[arg-type]
                metric_types,  # type: ignore[arg-type]
                ["model_name", "engine"],
                {0: ["qwen3", "0"]},
            )
        multi_prom = MultiKVConnectorPromMetrics(
            MagicMock(),
            metric_types,  # type: ignore[arg-type]
            ["model_name", "engine"],
            {0: ["qwen3", "0"]},
            {
                "TPURaidenConnector": raiden_prom,
                "TPURaidenOffloadingConnector": offload_prom,
            },
        )
        multi_prom.observe(stats.to_dict(), engine_idx=0)

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
