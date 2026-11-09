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

import enum
from types import SimpleNamespace
from typing import Any, cast

import prometheus_client
import pytest
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import PromMetric
from vllm.v1.kv_offload.base import make_offload_key
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.metrics.cache_hit_source import CacheHitSource
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector import TPUMultiConnector
from vllm_torchtpu.offload.metrics import (
    HOST_OFFLOAD_LABELS,
    OFFLOAD_LOAD_LABELNAMES,
    UNSPECIFIED_OFFLOAD_LABELS,
    RaidenOffloadMetricName,
    RaidenOffloadPromMetrics,
    RaidenOffloadStatsCollector,
    build_raiden_offload_metric_definitions,
    external_sources_with_tokens,
    split_external_hit,
    status_to_hit_source,
)
from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector


class _FakeBlockStatus(enum.Enum):
    INIT = 0
    REMOTE = 1
    HOST = 2
    HBM = 3
    HOST_AND_HBM = 4
    SHARED_STORAGE = 5
    CXL = 6


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


def test_metric_definitions_have_directional_labels() -> None:
    assert UNSPECIFIED_OFFLOAD_LABELS == ("external_unspecified", "unknown")
    defs = build_raiden_offload_metric_definitions({})
    assert set(defs.keys()) == {m.value for m in RaidenOffloadMetricName}
    load_metrics = {
        RaidenOffloadMetricName.HIT_TOKENS,
    }
    for metric in load_metrics:
        assert defs[metric].labelnames == OFFLOAD_LOAD_LABELNAMES


@pytest.mark.parametrize(
    ("status", "expected_hit"),
    [
        (_FakeBlockStatus.HOST, CacheHitSource.HOST),
        (_FakeBlockStatus.HOST_AND_HBM, CacheHitSource.HOST),
        (_FakeBlockStatus.REMOTE, CacheHitSource.P2P),
        (_FakeBlockStatus.SHARED_STORAGE, CacheHitSource.DISK),
        (_FakeBlockStatus.HBM, CacheHitSource.EXTERNAL_UNSPECIFIED),
        (_FakeBlockStatus.INIT, CacheHitSource.EXTERNAL_UNSPECIFIED),
        (_FakeBlockStatus.CXL, CacheHitSource.EXTERNAL_UNSPECIFIED),
    ],
)
def test_status_to_hit_source(
    status: _FakeBlockStatus,
    expected_hit: CacheHitSource,
) -> None:
    assert status_to_hit_source(status, _FakeBlockStatus) == expected_hit


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


def test_collector_unknown_sources_and_missing_key_handling() -> None:
    collector = RaidenOffloadStatsCollector(
        storage_backend=None,
    )

    # 1. Disk source without configured storage_backend uses ("disk", "unknown"),
    # EXTERNAL_UNSPECIFIED uses ("external_unspecified", "unknown"), and DEVICE
    # or non-CacheHitSource values raise.
    assert collector.source_labels(CacheHitSource.DISK) == ("disk", "unknown")
    assert (
        collector.source_labels(CacheHitSource.EXTERNAL_UNSPECIFIED)
        == UNSPECIFIED_OFFLOAD_LABELS
    )
    for invalid in (CacheHitSource.DEVICE, "host"):
        with pytest.raises(ValueError, match="Unsupported offload source"):
            collector.source_labels(cast(Any, invalid))

    # 2. Missing key in _step_key_sources returns None from get_load_sub_sources
    # and is skipped by compute_load_tokens_by_source rather than mislabeled.
    missing_key = make_offload_key(b"missing", 0)
    assert collector.get_load_sub_sources(missing_key) is None
    assert (
        collector.compute_load_tokens_by_source(
            start_token=0,
            end_token=16,
            load_key_ranges=[(0, 16, missing_key)],
            tokens_per_chunk_by_group={0: 16},
        )
        == {}
    )
    assert collector.pop_stats() is None

    # 3. Unrecognized BlockStatus (CXL) in prepare_load records under
    # ("external_unspecified", "unknown") instead of ("p2p", "rdma").
    k0 = make_offload_key(b"k0", 0)
    cxl_block = SimpleNamespace(status=_FakeBlockStatus.CXL)
    collector.record_prepared_load(
        [k0],
        1,
        {b"s0": cxl_block},
        [b"s0"],
        _FakeBlockStatus,
    )
    assert collector.get_load_sub_sources(k0) == (
        (CacheHitSource.EXTERNAL_UNSPECIFIED, 1),
    )

    collector.record_prepared_load(
        [k0],
        2,
        {b"s1": cxl_block},
        [b"s0", b"s1"],
        _FakeBlockStatus,
    )
    assert collector.get_load_sub_sources(k0) == (
        (CacheHitSource.HOST, 1),
        (CacheHitSource.EXTERNAL_UNSPECIFIED, 1),
    )

    tokens_by_source = collector.compute_load_tokens_by_source(
        start_token=0,
        end_token=16,
        load_key_ranges=[(0, 16, k0)],
        tokens_per_chunk_by_group={0: 16},
    )
    assert tokens_by_source == {
        CacheHitSource.HOST: 8,
        CacheHitSource.EXTERNAL_UNSPECIFIED: 8,
    }
    collector.record_admission_hits(tokens_by_source)

    stats = collector.pop_stats()
    assert stats is not None
    values = stats.data["data"]
    assert values[RaidenOffloadMetricName.HIT_TOKENS] == {
        HOST_OFFLOAD_LABELS: 8,
        UNSPECIFIED_OFFLOAD_LABELS: 8,
    }


def test_compute_load_tokens_by_source_sub_block_and_mamba() -> None:
    collector = RaidenOffloadStatsCollector(storage_backend="posix")
    peer_blk = SimpleNamespace(status=_FakeBlockStatus.REMOTE)
    disk_blk = SimpleNamespace(status=_FakeBlockStatus.SHARED_STORAGE)

    # 1. Single attention key (64 tokens, 4 x 16-token sub-blocks):
    # 3 HOST + 1 DISK sub-blocks form 2 runs -> 48 HOST + 16 DISK.
    k_dense = make_offload_key(b"dense0", 0)
    collector.record_prepared_load(
        [k_dense],
        4,
        {b"s3": disk_blk},
        [b"s0", b"s1", b"s2", b"s3"],
        _FakeBlockStatus,
    )
    assert collector.get_load_sub_sources(k_dense) == (
        (CacheHitSource.HOST, 3),
        (CacheHitSource.DISK, 1),
    )
    assert collector.compute_load_tokens_by_source(
        start_token=0,
        end_token=64,
        load_key_ranges=[(0, 64, k_dense)],
        tokens_per_chunk_by_group={0: 64},
    ) == {
        CacheHitSource.HOST: 48,
        CacheHitSource.DISK: 16,
    }

    # 2. Partial-tail range [0, 48) on a 64-token chunk excludes t3 ([48, 64)).
    k_tail = make_offload_key(b"tail0", 0)
    collector.record_prepared_load(
        [k_tail],
        4,
        {b"t2": peer_blk, b"t3": disk_blk},
        [b"t0", b"t1", b"t2", b"t3"],
        _FakeBlockStatus,
    )
    assert collector.get_load_sub_sources(k_tail) == (
        (CacheHitSource.HOST, 2),
        (CacheHitSource.P2P, 1),
        (CacheHitSource.DISK, 1),
    )
    assert collector.compute_load_tokens_by_source(
        start_token=0,
        end_token=48,
        load_key_ranges=[(0, 48, k_tail)],
        tokens_per_chunk_by_group={0: 64},
    ) == {
        CacheHitSource.HOST: 32,
        CacheHitSource.P2P: 16,
    }

    # 3. Large sub-block factor (factor=512, 8192-token chunks):
    # - Uniform HOST batch shares a single 1-element tuple across keys.
    # - Contiguous 384 HOST + 128 DISK collapses to 2 runs instead of 512.
    k_big0 = make_offload_key(b"big0", 0)
    k_big1 = make_offload_key(b"big1", 0)
    collector.record_prepared_load([k_big0, k_big1], 512, {}, [], _FakeBlockStatus)
    runs_big0 = collector.get_load_sub_sources(k_big0)
    runs_big1 = collector.get_load_sub_sources(k_big1)
    assert runs_big0 == ((CacheHitSource.HOST, 512),)
    assert runs_big0 is runs_big1

    k_big_mixed = make_offload_key(b"big_mixed", 0)
    big_hashes = [f"bh_{i}".encode() for i in range(512)]
    collector.record_prepared_load(
        [k_big_mixed],
        512,
        {h: disk_blk for h in big_hashes[384:]},
        big_hashes,
        _FakeBlockStatus,
    )
    assert collector.get_load_sub_sources(k_big_mixed) == (
        (CacheHitSource.HOST, 384),
        (CacheHitSource.DISK, 128),
    )
    assert collector.compute_load_tokens_by_source(
        start_token=0,
        end_token=16384,
        load_key_ranges=[(0, 8192, k_big0), (8192, 16384, k_big_mixed)],
        tokens_per_chunk_by_group={0: 8192},
    ) == {
        CacheHitSource.HOST: 8192 + 6144,
        CacheHitSource.DISK: 2048,
    }

    # 4. Hybrid Attention (group 0) + Mamba (group 1) over [0, 32):
    # k_a0=(HOST*4), k_a1=(HOST*2, P2P*2), k_m1=(HOST*4) -> 24 HOST + 8 P2P.
    k_a0 = make_offload_key(b"a0", 0)
    k_a1 = make_offload_key(b"a1", 0)
    k_m1 = make_offload_key(b"m1", 1)
    collector.record_prepared_load([k_a0, k_m1], 4, {}, [], _FakeBlockStatus)
    collector.record_prepared_load(
        [k_a1],
        4,
        {b"a1_2": peer_blk, b"a1_3": peer_blk},
        [b"a1_0", b"a1_1", b"a1_2", b"a1_3"],
        _FakeBlockStatus,
    )
    assert collector.compute_load_tokens_by_source(
        start_token=0,
        end_token=32,
        load_key_ranges=[(0, 16, k_a0), (16, 32, k_a1), (0, 32, k_m1)],
        tokens_per_chunk_by_group={0: 16, 1: 16},
        mamba_group_ids={1},
    ) == {
        CacheHitSource.HOST: 24,
        CacheHitSource.P2P: 8,
    }

    # If Mamba's indivisible checkpoint has one byte slice in DISK, the whole
    # prefix [0, 32) depends on that DISK checkpoint -> {DISK: 32}.
    collector.record_prepared_load(
        [k_m1],
        4,
        {b"m1_3": disk_blk},
        [b"m1_0", b"m1_1", b"m1_2", b"m1_3"],
        _FakeBlockStatus,
    )
    assert collector.compute_load_tokens_by_source(
        start_token=0,
        end_token=32,
        load_key_ranges=[(0, 16, k_a0), (16, 32, k_a1), (0, 32, k_m1)],
        tokens_per_chunk_by_group={0: 16, 1: 16},
        mamba_group_ids={1},
    ) == {
        CacheHitSource.DISK: 32,
    }


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
    collector = RaidenOffloadStatsCollector(storage_backend="posix")
    disk_blk = SimpleNamespace(status=_FakeBlockStatus.SHARED_STORAGE)
    # A 64-token key with four 16-token sub-blocks: two in host DRAM, two on disk.
    key = make_offload_key(b"mixed", 0)
    collector.record_prepared_load(
        [key],
        4,
        {b"s2": disk_blk, b"s3": disk_blk},
        [b"s0", b"s1", b"s2", b"s3"],
        _FakeBlockStatus,
    )
    assert (
        collector.compute_load_tokens_by_source(
            start_token=start_token,
            end_token=end_token,
            load_key_ranges=[(*key_range, key)],
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
    collector = RaidenOffloadStatsCollector(storage_backend="posix")
    disk_blk = SimpleNamespace(status=_FakeBlockStatus.SHARED_STORAGE)
    key = make_offload_key(b"mixed", 0)
    collector.record_prepared_load(
        [key],
        4,
        {b"s3": disk_blk},
        [b"s0", b"s1", b"s2", b"s3"],
        _FakeBlockStatus,
    )
    assert collector.compute_load_tokens_by_source(
        start_token=0,
        end_token=64,
        load_key_ranges=[(0, 64, key)],
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
        registry.get_sample_value("vllm:kv_offload_hit_tokens_total", disk_load_labels)
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
    ("tier_config", "configured_tiers"),
    [
        ({}, {("host", "dram")}),
        ({"raiden_storage_path": "/mnt/kv"}, {("host", "dram"), ("disk", "posix")}),
        (
            {"global_registry_address": "registry:2379"},
            {("host", "dram"), ("p2p", "rdma")},
        ),
    ],
)
def test_prom_metrics_start_each_configured_tier_hit_series_at_zero(
    monkeypatch: pytest.MonkeyPatch,
    tier_config: dict[str, Any],
    configured_tiers: set[tuple[str, str]],
) -> None:
    """`rate()` misses the first hit of a series that only appears with it."""
    for name in ("RAIDEN_STORAGE_PATH", "RAIDEN_STORAGE_TYPE"):
        monkeypatch.delenv(name, raising=False)
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
                "vllm:kv_offload_hit_tokens_total",
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
            "vllm:kv_offload_hit_tokens_total",
            {"model_name": "disagg-model", "source": "disk", "backend": "posix"},
        )
        == 64.0
    )
