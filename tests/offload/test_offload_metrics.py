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
"""Unit tests for Raiden KV offload Prometheus metrics and stats collectors."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import prometheus_client
import pytest
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import PromMetric
from vllm.v1.kv_offload.base import OffloadKey, make_offload_key
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.metrics.cache_hit_source import CacheHitSource
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector import TPUMultiConnector
from vllm_torchtpu.offload.metrics import (
    SOURCE_LABELNAMES,
    RaidenOffloadMetricName,
    RaidenOffloadPromMetrics,
    RaidenOffloadStatsCollector,
    SubBlockSourceCounts,
    build_raiden_offload_metric_definitions,
    compute_load_tokens_by_source,
    external_sources_with_tokens,
    split_external_hit,
)
from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector


def _make_prom_metric_types(
    registry: prometheus_client.CollectorRegistry,
) -> dict[type[PromMetric], type[PromMetric]]:
    class _Counter(prometheus_client.Counter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, registry=registry, **kwargs)

    class _Gauge(prometheus_client.Gauge):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, registry=registry, **kwargs)

    class _Histogram(prometheus_client.Histogram):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, registry=registry, **kwargs)

    return {
        prometheus_client.Counter: _Counter,
        prometheus_client.Gauge: _Gauge,
        prometheus_client.Histogram: _Histogram,
    }


def test_metric_definitions_have_source_labels() -> None:
    defs = build_raiden_offload_metric_definitions()
    assert set(defs.keys()) == {m.value for m in RaidenOffloadMetricName}
    assert defs[RaidenOffloadMetricName.HIT_TOKENS].labelnames == SOURCE_LABELNAMES


def test_external_sources_with_tokens() -> None:
    raw = {
        CacheHitSource.DEVICE: 16,
        CacheHitSource.HOST: 64,
        CacheHitSource.P2P: 0,
        CacheHitSource.DISK: 128,
        CacheHitSource.EXTERNAL_UNSPECIFIED: 32,
    }
    assert external_sources_with_tokens(raw) == {
        CacheHitSource.HOST: 64,
        CacheHitSource.DISK: 128,
        CacheHitSource.EXTERNAL_UNSPECIFIED: 32,
    }


def test_collector_labels_disk_without_storage_backend_as_unknown() -> None:
    collector = RaidenOffloadStatsCollector(storage_backend=None)
    assert collector.source_labels(CacheHitSource.DISK) == ("disk", "unknown")


@pytest.mark.parametrize(
    "source",
    [
        CacheHitSource.DEVICE,
        CacheHitSource.EXTERNAL_UNSPECIFIED,
        pytest.param("host", id="raw str"),
    ],
)
def test_collector_rejects_unsupported_sources(source: Any) -> None:
    collector = RaidenOffloadStatsCollector(storage_backend=None)
    with pytest.raises(ValueError, match="Unsupported offload source"):
        collector.source_labels(source)


_KEY_0 = make_offload_key(b"key0", 0)
_KEY_1 = make_offload_key(b"key1", 0)


@pytest.mark.parametrize(
    ("end_token", "tokens_per_chunk", "load_key_ranges", "key_sources", "expected"),
    [
        # A 64-token key of four 16-token sub-blocks: 3 HOST + 1 DISK sub-blocks
        # form 2 runs -> 48 HOST + 16 DISK.
        pytest.param(
            64,
            64,
            [(0, 64, _KEY_0)],
            {_KEY_0: ((CacheHitSource.HOST, 3), (CacheHitSource.DISK, 1))},
            {CacheHitSource.HOST: 48, CacheHitSource.DISK: 16},
            id="one key in two tiers",
        ),
        # The range [0, 48) of a 64-token chunk excludes its last sub-block,
        # [48, 64).
        pytest.param(
            48,
            64,
            [(0, 48, _KEY_0)],
            {
                _KEY_0: (
                    (CacheHitSource.HOST, 2),
                    (CacheHitSource.P2P, 1),
                    (CacheHitSource.DISK, 1),
                )
            },
            {CacheHitSource.HOST: 32, CacheHitSource.P2P: 16},
            id="partial tail",
        ),
        # 8192-token chunks of 512 sub-blocks: a uniform HOST key, and a key
        # of 384 HOST + 128 DISK sub-blocks in 2 runs.
        pytest.param(
            16384,
            8192,
            [(0, 8192, _KEY_0), (8192, 16384, _KEY_1)],
            {
                _KEY_0: ((CacheHitSource.HOST, 512),),
                _KEY_1: ((CacheHitSource.HOST, 384), (CacheHitSource.DISK, 128)),
            },
            {CacheHitSource.HOST: 8192 + 6144, CacheHitSource.DISK: 2048},
            id="512 sub-blocks per key",
        ),
    ],
)
def test_compute_load_tokens_by_source_splits_mixed_key_by_sub_block(
    end_token: int,
    tokens_per_chunk: int,
    load_key_ranges: list[tuple[int, int, OffloadKey]],
    key_sources: dict[OffloadKey, SubBlockSourceCounts],
    expected: dict[CacheHitSource, int],
) -> None:
    assert (
        compute_load_tokens_by_source(
            start_token=0,
            end_token=end_token,
            load_key_ranges=load_key_ranges,
            key_sources=key_sources,
            tokens_per_chunk_by_group={0: tokens_per_chunk},
        )
        == expected
    )


@pytest.mark.parametrize(
    ("mamba_sources", "expected"),
    [
        pytest.param(
            ((CacheHitSource.HOST, 4),),
            {CacheHitSource.HOST: 24, CacheHitSource.P2P: 8},
            id="recurrent state in host DRAM",
        ),
        # Mamba's indivisible checkpoint has one byte slice in DISK, so the
        # whole prefix [0, 32) depends on that DISK checkpoint.
        pytest.param(
            ((CacheHitSource.HOST, 3), (CacheHitSource.DISK, 1)),
            {CacheHitSource.DISK: 32},
            id="recurrent state partly on disk",
        ),
    ],
)
def test_compute_load_tokens_by_source_hybrid_overlap_uses_slowest_tier(
    mamba_sources: SubBlockSourceCounts,
    expected: dict[CacheHitSource, int],
) -> None:
    # Hybrid Attention (group 0) + Mamba (group 1) over [0, 32):
    # k_a0=(HOST*4), k_a1=(HOST*2, P2P*2), and k_m1 covers the whole prefix.
    k_a0 = make_offload_key(b"a0", 0)
    k_a1 = make_offload_key(b"a1", 0)
    k_m1 = make_offload_key(b"m1", 1)
    assert (
        compute_load_tokens_by_source(
            start_token=0,
            end_token=32,
            load_key_ranges=[(0, 16, k_a0), (16, 32, k_a1), (0, 32, k_m1)],
            key_sources={
                k_a0: ((CacheHitSource.HOST, 4),),
                k_a1: ((CacheHitSource.HOST, 2), (CacheHitSource.P2P, 2)),
                k_m1: mamba_sources,
            },
            tokens_per_chunk_by_group={0: 16, 1: 16},
            mamba_group_ids={1},
        )
        == expected
    )


@pytest.mark.parametrize(
    ("start_token", "end_token", "key_range", "expected"),
    [
        # The HBM hit ends inside the chunk, so the load starts mid-chunk.
        (16, 64, (16, 64), {CacheHitSource.HOST: 16, CacheHitSource.DISK: 32}),
        (16, 64, (0, 64), {CacheHitSource.HOST: 16, CacheHitSource.DISK: 32}),
        # The hit starts and ends inside sub-blocks.
        (8, 40, (0, 64), {CacheHitSource.HOST: 24, CacheHitSource.DISK: 8}),
        (40, 64, (0, 64), {CacheHitSource.DISK: 24}),
        # The second chunk of a prompt: tokens [64, 128).
        (80, 128, (64, 128), {CacheHitSource.HOST: 16, CacheHitSource.DISK: 32}),
    ],
)
def test_compute_load_tokens_by_source_clips_mixed_key_to_hit_range(
    start_token: int,
    end_token: int,
    key_range: tuple[int, int],
    expected: dict[CacheHitSource, int],
) -> None:
    # A 64-token key with four 16-token sub-blocks: two in host DRAM, two on disk.
    key = make_offload_key(b"mixed", 0)
    assert (
        compute_load_tokens_by_source(
            start_token=start_token,
            end_token=end_token,
            load_key_ranges=[(*key_range, key)],
            key_sources={key: ((CacheHitSource.HOST, 2), (CacheHitSource.DISK, 2))},
            tokens_per_chunk_by_group={0: 64},
        )
        == expected
    )


@pytest.mark.parametrize(
    ("tokens_per_chunk_by_group", "mamba_group_ids", "num_warnings"),
    [
        pytest.param({}, (), 1, id="group without a chunk size"),
        pytest.param({0: 62}, (), 1, id="chunk size not divisible by sub-blocks"),
        pytest.param({0: 64}, (0,), 0, id="mamba checkpoint"),
    ],
)
def test_compute_load_tokens_by_source_unsplit_key_uses_slowest_tier(
    tokens_per_chunk_by_group: dict[int, int],
    mamba_group_ids: tuple[int, ...],
    num_warnings: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        "vllm_torchtpu.offload.metrics.logger.warning_once",
        lambda *args: warnings.append(args),
    )
    key = make_offload_key(b"mixed", 0)
    assert compute_load_tokens_by_source(
        start_token=0,
        end_token=64,
        load_key_ranges=[(0, 64, key)],
        key_sources={key: ((CacheHitSource.HOST, 3), (CacheHitSource.DISK, 1))},
        tokens_per_chunk_by_group=tokens_per_chunk_by_group,
        mamba_group_ids=mamba_group_ids,
    ) == {CacheHitSource.DISK: 64}
    assert len(warnings) == num_warnings


def test_split_external_hit_sums_to_hit_count() -> None:
    assert split_external_hit({CacheHitSource.HOST: 32}, 48) == {
        CacheHitSource.HOST: 32,
        CacheHitSource.EXTERNAL_UNSPECIFIED: 16,
    }
    assert split_external_hit(
        {CacheHitSource.DISK: 8, CacheHitSource.EXTERNAL_UNSPECIFIED: 8}, 32
    ) == {CacheHitSource.DISK: 8, CacheHitSource.EXTERNAL_UNSPECIFIED: 24}
    assert split_external_hit({CacheHitSource.P2P: 16}, 16) == {CacheHitSource.P2P: 16}
    assert split_external_hit({}, 16) == {CacheHitSource.EXTERNAL_UNSPECIFIED: 16}


def test_prom_metrics_registration_and_end_to_end_observation() -> None:
    registry = prometheus_client.CollectorRegistry()
    extra = {"cpu_bytes_to_use": 1 << 20}
    kv_transfer_config = KVTransferConfig(
        kv_connector="TPURaidenOffloadingConnector",
        kv_connector_module_path="vllm_torchtpu.offload.raiden_connector",
        kv_role="kv_both",
        kv_connector_extra_config=dict(extra),
    )
    vllm_config = cast(
        VllmConfig, SimpleNamespace(kv_transfer_config=kv_transfer_config)
    )

    prom = TPURaidenOffloadingConnector.build_prom_metrics(
        vllm_config,
        _make_prom_metric_types(registry),
        ["model_name", "engine"],
        {0: ["test-model", "0"]},
    )
    assert isinstance(prom, RaidenOffloadPromMetrics)
    assert not prom._observe_deprecated_metrics
    # The spec is named on a config copy: neither the caller's extra_config
    # nor the upstream spec registry is mutated.
    assert "spec_name" not in kv_transfer_config.kv_connector_extra_config
    assert "spec_module_path" not in kv_transfer_config.kv_connector_extra_config
    assert "TPURaidenStoreOffloadingSpec" not in OffloadingSpecFactory._registry
    # Raiden's definitions are registered, and CPUOffloadingSpec's are not.
    families = {metric.name for metric in registry.collect()}
    assert RaidenOffloadMetricName.HIT_TOKENS in families
    assert not any(name.startswith("vllm:kv_offload_cpu_") for name in families)

    collector = RaidenOffloadStatsCollector(
        storage_backend="posix",
    )
    collector.record_admission_hits({CacheHitSource.DISK: 32})

    raw_stats = collector.pop_stats()
    assert raw_stats is not None
    # The scheduler adds the manager's config info to an engine's first
    # payload. The Raiden manager reports none.
    raw_stats.set_info([{}])
    # Ship the payload through the real EngineCore -> API server codec, which
    # must preserve the label-value tuple keys.
    ipc_stats = MsgpackDecoder(SchedulerStats).decode(
        MsgpackEncoder().encode(SchedulerStats(kv_connector_stats=raw_stats.to_dict()))
    )
    assert ipc_stats.kv_connector_stats is not None
    prom.observe(ipc_stats.kv_connector_stats, engine_idx=0)

    base_labels = {"model_name": "test-model", "engine": "0"}
    # Raiden's spec declares no config info labels.
    assert registry.get_sample_value("vllm:kv_offload_config_info", base_labels) == 1.0
    disk_load_labels = {**base_labels, "source": "disk", "backend": "posix"}

    assert (
        registry.get_sample_value(
            "vllm:kv_offload_raiden_hit_tokens_total", disk_load_labels
        )
        == 32.0
    )
    # Deprecated CPUOffloadingSpec metric stays at its initial 0.0 value.
    assert (
        registry.get_sample_value(
            "vllm:kv_offload_total_bytes_total",
            {**base_labels, "transfer_type": "CPU_to_GPU"},
        )
        == 0.0
    )


@pytest.mark.parametrize(
    ("tier_config", "env", "configured_tiers"),
    [
        ({}, {}, {("host", "dram")}),
        ({"raiden_storage_path": "/mnt/kv"}, {}, {("host", "dram"), ("disk", "posix")}),
        (
            {},
            {"RAIDEN_STORAGE_PATH": "/mnt/kv", "RAIDEN_STORAGE_TYPE": "POSIX"},
            {("host", "dram"), ("disk", "posix")},
        ),
        (
            {"global_registry_address": "registry:2379"},
            {},
            {("host", "dram"), ("p2p", "rdma")},
        ),
    ],
)
def test_prom_metrics_start_each_configured_tier_hit_series_at_zero(
    monkeypatch: pytest.MonkeyPatch,
    tier_config: dict[str, Any],
    env: dict[str, str],
    configured_tiers: set[tuple[str, str]],
) -> None:
    """`rate()` misses the first hit of a series that only appears with it."""
    for name in ("RAIDEN_STORAGE_PATH", "RAIDEN_STORAGE_TYPE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    registry = prometheus_client.CollectorRegistry()
    kv_transfer_config = KVTransferConfig(
        kv_connector="TPURaidenOffloadingConnector",
        kv_connector_module_path="vllm_torchtpu.offload.raiden_connector",
        kv_role="kv_both",
        kv_connector_extra_config={"cpu_bytes_to_use": 1 << 20, **tier_config},
    )

    TPURaidenOffloadingConnector.build_prom_metrics(
        cast(VllmConfig, SimpleNamespace(kv_transfer_config=kv_transfer_config)),
        _make_prom_metric_types(registry),
        ["model_name", "engine"],
        {0: ["test-model", "0"], 1: ["test-model", "1"]},
    )

    for engine in ("0", "1"):
        for source, backend in (("host", "dram"), ("p2p", "rdma"), ("disk", "posix")):
            value = registry.get_sample_value(
                "vllm:kv_offload_raiden_hit_tokens_total",
                {
                    "model_name": "test-model",
                    "engine": engine,
                    "source": source,
                    "backend": backend,
                },
            )
            assert value == (0.0 if (source, backend) in configured_tiers else None)


def test_tpu_multi_connector_builds_and_observes_raiden_offload_prom_metrics() -> None:
    registry = prometheus_client.CollectorRegistry()
    kv_transfer_config = KVTransferConfig(
        kv_connector="TPUMultiConnector",
        kv_connector_module_path=(
            "vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector"
        ),
        kv_role="kv_both",
        kv_connector_extra_config={
            "connectors": [
                {
                    "kv_connector": "TPURaidenConnector",
                    "kv_connector_module_path": (
                        "vllm_torchtpu.distributed.kv_transfer.tpu_connector"
                    ),
                    "kv_role": "kv_both",
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
    )
    vllm_config = cast(
        VllmConfig, SimpleNamespace(kv_transfer_config=kv_transfer_config)
    )
    multi_prom = TPUMultiConnector.build_prom_metrics(
        vllm_config,
        _make_prom_metric_types(registry),
        ["model_name"],
        {0: ["disagg-model"]},
    )
    assert multi_prom is not None

    collector = RaidenOffloadStatsCollector(storage_backend="posix")
    collector.record_admission_hits({CacheHitSource.DISK: 64})
    offload_stats = collector.pop_stats()
    assert offload_stats is not None

    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiKVConnectorStats,
    )

    multi_stats = MultiKVConnectorStats(
        data={"TPURaidenOffloadingConnector": offload_stats}
    )
    multi_prom.observe(multi_stats.to_dict(), engine_idx=0)
    assert (
        registry.get_sample_value(
            "vllm:kv_offload_raiden_hit_tokens_total",
            {"model_name": "disagg-model", "source": "disk", "backend": "posix"},
        )
        == 64.0
    )
