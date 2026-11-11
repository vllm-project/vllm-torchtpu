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
"""Stats and Prometheus metrics for the TPU KV connector."""

import copy
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from prometheus_client import Gauge
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.v1.metrics.utils import create_metric_per_engine

import vllm_torchtpu.distributed.utils as dist_utils
from vllm_torchtpu.distributed.kv_transfer.raiden.telemetry_utils import (
    normalize_raiden_metric_name,
    parse_raiden_metric_key,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Prometheus metric type constants
METRIC_TYPE_HISTOGRAM = "histogram"
METRIC_TYPE_COUNTER = "counter"
METRIC_TYPE_GAUGE = "gauge"

# Default fallback value for missing label dimensions
DEFAULT_LABEL_VALUE = ""
RUNNER_TIMING_BUCKETS_SECONDS = (
    0.00001,
    0.000025,
    0.00005,
    0.0001,
    0.00025,
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)


@dataclass
class TpuKVConnectorStats(KVConnectorStats):
    """Container for transfer performance metrics"""

    def __post_init__(self):
        if not self.data:
            # Empty container init, no data is passed in.
            self.reset()

    def record_prefill_queue_length(self, prefill_queue_length: int):
        """Record the prefill (send) queue length."""
        self.data["prefill_queue_length"].append(prefill_queue_length)

    def record_decode_queue_length(self, decode_queue_length: int):
        """Record the decode (load) queue length."""
        self.data["decode_queue_length"].append(decode_queue_length)

    def record_runner_timings(
        self,
        cpu_samples: list[tuple[str, str, float]],
        device_samples: list[tuple[str, str, float]],
    ) -> None:
        self.data["runner_cpu_issue_seconds"].extend(cpu_samples)
        self.data["runner_device_elapsed_seconds"].extend(device_samples)

    def reset(self):
        # Must be serializable
        self.data: dict[str, list[float | int | tuple[str, str, float]]] = {
            "prefill_queue_length": [],
            "decode_queue_length": [],
            "runner_cpu_issue_seconds": [],
            "runner_device_elapsed_seconds": [],
        }

    def clone_and_reset(self) -> "TpuKVConnectorStats":
        old = copy.copy(self)
        self.reset()
        return old

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        if not other.is_empty():
            for k, v in other.data.items():
                if k not in self.data:
                    self.data[k] = []
                accumulator = self.data[k]
                assert isinstance(accumulator, list)
                accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        prefill_queue_length = np.asarray(self.data.get("prefill_queue_length", []))
        decode_queue_length = np.asarray(self.data.get("decode_queue_length", []))
        return {
            "Prefill queue length": int(prefill_queue_length.max())
            if prefill_queue_length.size > 0
            else 0,
            "Decode queue length": int(decode_queue_length.max())
            if decode_queue_length.size > 0
            else 0,
        }

    def is_empty(self) -> bool:
        if not self.data:
            return True
        return not any(self.data.values())


class TpuKVConnectorPromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)

        runner_labels = [*labelnames, "rank", "stage"]
        self.runner_cpu_issue = self._histogram_cls(
            name="vllm:tpu_runner_cpu_issue_seconds",
            documentation=(
                "Host time spent issuing a TPU runner stage, including Python work."
            ),
            buckets=RUNNER_TIMING_BUCKETS_SECONDS,
            labelnames=runner_labels,
        )
        self.runner_device_elapsed = self._histogram_cls(
            name="vllm:tpu_runner_device_elapsed_seconds",
            documentation="Host-observed time between TPU stage Event completions.",
            buckets=RUNNER_TIMING_BUCKETS_SECONDS,
            labelnames=runner_labels,
        )
        gauge_tpu_prefill_queue_length = self._gauge_cls(
            name="vllm:tpu_prefill_kv_queue_length",
            documentation="Current length of the TPU KV transfer prefill queue.",
            labelnames=labelnames,
        )
        self.gauge_tpu_prefill_queue_length = create_metric_per_engine(
            gauge_tpu_prefill_queue_length, self.per_engine_labelvalues
        )

        gauge_tpu_decode_queue_length = self._gauge_cls(
            name="vllm:tpu_decode_kv_queue_length",
            documentation="Current length of the TPU KV transfer decode queue.",
            labelnames=labelnames,
        )
        self.gauge_tpu_decode_queue_length = create_metric_per_engine(
            gauge_tpu_decode_queue_length, self.per_engine_labelvalues
        )

        self.dynamic_metrics: dict[str, tuple[str, Any, list[str]]] = (
            self._get_raiden_metric_definitions(labelnames)
        )

    def _get_raiden_metric_definitions(self, labelnames: list[str] | None = None):
        if labelnames is None:
            labelnames = getattr(self, "labelnames", None) or getattr(
                self, "_labelnames", None
            )
        base_labels = list(labelnames) if labelnames is not None else []
        # Dynamically load TPU Raiden C++ metric schema descriptors
        metric_defs = {}
        try:
            # Without this call, get_metric_metadata won't return the correct
            # metrics definitions
            dist_utils.configure_raiden_telemetry()
            telemetry = dist_utils.get_raiden_telemetry_module()
            descriptors = []
            if telemetry is not None and hasattr(telemetry, "get_metric_metadata"):
                descriptors = telemetry.get_metric_metadata()
        except Exception as e:
            logger.warning("Failed to load TPU Raiden telemetry descriptors: %s", e)

        for desc in descriptors:
            # Construct prometheus metric objects based on the definition.
            name = normalize_raiden_metric_name(desc.name)
            help_str = getattr(desc, "description", getattr(desc, "help", ""))
            mtype = desc.type
            mtype_str = getattr(mtype, "name", str(mtype)).split(".")[-1].lower()
            metric_label_names = list(getattr(desc, "label_names", None) or [])
            combined_labels = base_labels + metric_label_names

            if mtype_str == METRIC_TYPE_HISTOGRAM:
                buckets_val = desc.buckets if desc.buckets else [1.0, 10.0, 100.0]
                metric_obj = self._histogram_cls(
                    name=name,
                    documentation=help_str,
                    buckets=buckets_val,
                    labelnames=combined_labels,
                )
            elif mtype_str == METRIC_TYPE_COUNTER:
                metric_obj = self._counter_cls(
                    name=name,
                    documentation=help_str,
                    labelnames=combined_labels,
                )
            elif mtype_str == METRIC_TYPE_GAUGE:
                metric_obj = self._gauge_cls(
                    name=name,
                    documentation=help_str,
                    labelnames=combined_labels,
                )
            else:
                continue

            if metric_label_names:
                per_engine_obj = metric_obj
            elif base_labels and self.per_engine_labelvalues:
                per_engine_obj = create_metric_per_engine(
                    metric_obj, self.per_engine_labelvalues
                )
            else:
                per_engine_obj = {
                    idx: metric_obj for idx in (self.per_engine_labelvalues or [0])
                }
            metric_defs[name] = (mtype_str, per_engine_obj, metric_label_names)
        return metric_defs

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        engine_labels = self.per_engine_labelvalues[engine_idx]
        for rank, stage, seconds in transfer_stats_data.get(
            "runner_cpu_issue_seconds", []
        ):
            self.runner_cpu_issue.labels(*engine_labels, rank, stage).observe(seconds)
        for rank, stage, seconds in transfer_stats_data.get(
            "runner_device_elapsed_seconds", []
        ):
            self.runner_device_elapsed.labels(*engine_labels, rank, stage).observe(
                seconds
            )

        for gauge_obj, gauge_item_key in zip(
            [
                self.gauge_tpu_prefill_queue_length,
                self.gauge_tpu_decode_queue_length,
            ],
            [
                "prefill_queue_length",
                "decode_queue_length",
            ],
        ):
            for list_item in transfer_stats_data.get(gauge_item_key, []):
                cast(Gauge, gauge_obj[engine_idx]).set(list_item)

        # Dynamic metric observations
        base_labels = (
            getattr(self, "labelnames", None)
            or getattr(self, "_labelnames", None)
            or []
        )
        engine_labels = dict(
            zip(base_labels, (self.per_engine_labelvalues or {}).get(engine_idx, ()))
        )

        for raw_key, values in transfer_stats_data.items():
            if not values:
                continue
            base_name, parsed_labels = parse_raiden_metric_key(raw_key)
            info = self.dynamic_metrics.get(normalize_raiden_metric_name(base_name))
            if info is None:
                continue

            mtype, per_engine_obj, metric_label_names = info
            if metric_label_names:
                label_names = getattr(per_engine_obj, "_labelnames", None)
                if not isinstance(label_names, (tuple, list)):
                    label_names = list(base_labels) + metric_label_names
                call_labels = {
                    k: parsed_labels.get(k, engine_labels.get(k, DEFAULT_LABEL_VALUE))
                    for k in label_names
                }
                target_obj = per_engine_obj.labels(**call_labels)
            else:
                target_obj = (
                    per_engine_obj.get(engine_idx)
                    if isinstance(per_engine_obj, dict)
                    else per_engine_obj
                )

            if target_obj is None:
                continue

            for val in values:
                if mtype == METRIC_TYPE_HISTOGRAM:
                    target_obj.observe(val)
                elif mtype == METRIC_TYPE_COUNTER:
                    target_obj.inc(val)
                elif mtype == METRIC_TYPE_GAUGE:
                    target_obj.set(val)
