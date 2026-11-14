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
"""Forward chunks, telemetry transport, and asynchronous observation boundaries."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

from vllm_torchtpu.distributed.kv_transfer.tpu_connector_stats import (
    TpuKVConnectorPromMetrics,
    TpuKVConnectorStats,
    attach_forward_metrics,
)
from vllm_torchtpu.runner import forward_timing as stage_timing
from vllm_torchtpu.runner.forward_metrics import ForwardBatch


@pytest.mark.parametrize("tokens", [(3072, 3071), (4096, 4096), (239, 867), (1,)])
def test_counts_are_per_chunk_global_queries_not_cached_or_padded_tokens(tokens):
    values = {
        name: value
        for _, _, name, value in ForwardBatch(1024, 8192, tokens).samples("8")
    }
    assert values["valid_tokens"] == sum(tokens)
    assert values["capacity_tokens"] == 8192
    assert values["utilization_ratio"] == sum(tokens) / 8192
    assert values["request_min_tokens"] == min(tokens)
    assert values["request_max_tokens"] == max(tokens)
    assert values["requests"] == len(tokens)


def test_empty_and_overfull_batches_are_rejected():
    for tokens in ((), (0,), (8193,)):
        with pytest.raises(AssertionError):
            ForwardBatch(1024, 8192, tokens).samples("0")


def test_transfer_summary_remains_scalar_with_forward_samples():
    stats = TpuKVConnectorStats()
    stats.data["forward_samples"].extend(ForwardBatch(128, 128, (1,)).samples("0"))
    assert stats.reduce()["Num failed transfers"] == 0
    stats.record_failed_transfer()
    stats.record_failed_transfer()
    assert stats.reduce()["Num failed transfers"] == 2


def test_idle_multi_connector_preserves_samples_through_serialization(monkeypatch):
    class Leaf:
        @classmethod
        def build_kv_connector_stats(cls):
            return TpuKVConnectorStats()

    multi = object.__new__(MultiConnector)
    multi._connectors = [Leaf(), Leaf()]
    samples = ForwardBatch(1024, 8192, (4096, 2047)).samples("8")
    result = attach_forward_metrics(multi, None, samples)
    result = attach_forward_metrics(multi, result, samples)
    leaf = result.data["Leaf"]
    data = msgspec.msgpack.decode(msgspec.msgpack.encode(leaf.clone_and_reset().data))
    assert leaf.is_empty()
    assert len(data["forward_samples"]) == 2 * len(samples)

    registry = CollectorRegistry()
    monkeypatch.setattr(
        TpuKVConnectorPromMetrics, "_get_raiden_metric_definitions", lambda *_: {}
    )
    metrics = TpuKVConnectorPromMetrics(
        SimpleNamespace(
            kv_transfer_config=None,
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1, prefill_context_parallel_size=8
            ),
        ),
        {
            kind: partial(kind, registry=registry)
            for kind in (Gauge, Counter, Histogram)
        },
        ["model_name", "engine"],
        {0: ["model", "0"]},
    )
    metrics.observe(data)
    labels = {"model_name": "model", "engine": "0", "rank": "8", "bucket": "1024"}
    assert (
        registry.get_sample_value("vllm:tpu_forward_valid_tokens_sum", labels) == 12286
    )
    assert (
        registry.get_sample_value("vllm:tpu_forward_capacity_tokens_sum", labels)
        == 16384
    )
    assert registry.get_sample_value("vllm:tpu_forward_requests_count", labels) == 2


def test_host_gap_and_completion_gap_have_no_first_sample(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(stage_timing.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(stage_timing.threading.Thread, "start", lambda _: None)

    class Event:
        def record(self):
            self.timestamp = clock[0]

        def synchronize(self):
            clock[0] = max(clock[0], self.timestamp)

    monkeypatch.setattr(stage_timing, "TpuEvent", Event)
    timing = stage_timing.ForwardTiming(8)
    batch = ForwardBatch(1024, 8192, (4096, 2047))
    with timing.stage(batch):
        clock[0] = 0.2
    clock[0] = 0.3
    with timing.stage(batch):
        clock[0] = 0.55
    samples = timing.take_forward_samples()

    def values(key):
        return [v for _, _, name, v in samples if name == key]

    assert values("host_issue_seconds") == pytest.approx([0.2, 0.25])
    assert values("host_gap_seconds") == pytest.approx([0.1])
    assert timing.take_forward_samples() == []

    pending = [timing._pending.get_nowait(), timing._pending.get_nowait()]
    timing._pending.get = Mock(side_effect=[*pending, StopIteration])
    clock[0] = 0
    with pytest.raises(StopIteration):
        timing._collect_device_time()
    samples = timing.take_forward_samples()
    assert values("completion_span_seconds") == pytest.approx([0.2, 0.25])
    assert values("completion_gap_seconds") == pytest.approx([0.1])


def test_full_observer_queue_keeps_cpu_samples_without_waiting(monkeypatch):
    monkeypatch.setattr(stage_timing.threading.Thread, "start", lambda _: None)
    monkeypatch.setattr(stage_timing, "TpuEvent", Mock)
    timing = stage_timing.ForwardTiming(0)
    timing._slots = Mock()
    timing._slots.acquire.return_value = False
    with timing.stage(ForwardBatch(256, 2048, (239,))):
        pass
    timing._slots.acquire.assert_called_once_with(blocking=False)
    assert timing._pending.empty()
    samples = timing.take_forward_samples()
    assert any(name == "host_issue_seconds" for _, _, name, _ in samples)
    assert not any(name == "completion_span_seconds" for _, _, name, _ in samples)


def test_skipped_completion_breaks_gap_chain_and_exception_closes_events(monkeypatch):
    monkeypatch.setattr(stage_timing.threading.Thread, "start", lambda _: None)
    monkeypatch.setattr(stage_timing, "TpuEvent", Mock)
    monkeypatch.setattr(stage_timing.ForwardTiming, "MAX_IN_FLIGHT_DEVICE_TIMINGS", 1)
    timing = stage_timing.ForwardTiming(0)
    batch = ForwardBatch(128, 128, (1,))
    with timing.stage(batch):
        pass
    with timing.stage(batch):
        pass  # Full observer: this forward has no completion sample.

    def drain():
        pending = timing._pending.get_nowait()
        assert pending.end_recorded.is_set()
        pending.end.record.assert_called_once_with()
        with monkeypatch.context() as collector_patch:
            collector_patch.setattr(
                timing._pending, "get", Mock(side_effect=[pending, StopIteration])
            )
            with pytest.raises(StopIteration):
                timing._collect_device_time()

    drain()
    with pytest.raises(ValueError, match="forward failed"), timing.stage(batch):
        raise ValueError("forward failed")
    drain()
    samples = timing.take_forward_samples()
    names = [name for _, _, name, _ in samples]
    assert names.count("valid_tokens") == 3
    assert names.count("host_issue_seconds") == 3
    assert names.count("completion_span_seconds") == 2
    assert "completion_gap_seconds" not in names


def test_non_tpu_sink_preserves_existing_stats():
    stats = TpuKVConnectorStats()
    stats.record_failed_transfer()
    connector = SimpleNamespace(build_kv_connector_stats=lambda: None)
    samples = ForwardBatch(128, 128, (1,)).samples("0")
    assert attach_forward_metrics(connector, stats, samples) is stats
    assert stats.data["num_failed_transfers"] == [1]
    assert stats.data["forward_samples"] == []
