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
"""Telemetry utility functions and dynamic Prometheus collectors for TPU Raiden."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any
from weakref import WeakKeyDictionary

import prometheus_client
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from vllm.v1.metrics.utils import create_metric_per_engine

from vllm_torchtpu.distributed import utils as dist_utils
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Metric name prefix
RAIDEN_METRIC_PREFIX = "tpu_raiden_"

# Prometheus metric type constants
METRIC_TYPE_HISTOGRAM = "histogram"
METRIC_TYPE_COUNTER = "counter"
METRIC_TYPE_GAUGE = "gauge"

# Default fallback value for missing label dimensions
DEFAULT_LABEL_VALUE = ""

# Per-registry cache of registered dynamic Raiden collectors so multiple
# connectors (e.g. TPURaidenConnector + TPURaidenOffloadingConnector under
# TPUMultiConnector) share collectors instead of raising duplicate-timeseries
# ValueError on the same CollectorRegistry.
_COLLECTOR_CACHE: WeakKeyDictionary[
    CollectorRegistry, dict[str, tuple[str, Any, tuple[str, ...]]]
] = WeakKeyDictionary()

_LABELED_METRIC_KEY_REGEX = re.compile(r"([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?")
_LABEL_PAIR_REGEX = re.compile(
    r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
)
_UNESCAPE_PATTERN = re.compile(r'\\([\\n"tr])')


def _unescape_match(match: re.Match) -> str:
    escaped = match.group(1)
    return {"n": "\n", "t": "\t", "r": "\r"}.get(escaped, escaped)


def parse_raiden_metric_key(raw_key: str) -> tuple[str, dict[str, str]]:
    """Parses Prometheus text-formatted keys emitted by Raiden telemetry, e.g.:

    tpu_raiden_sent_bytes_total{direction="push"} into base name and labels
    dict.
    """
    match = _LABELED_METRIC_KEY_REGEX.fullmatch(raw_key.strip())
    if not match:
        return raw_key.strip(), {}

    base_name = match.group(1)
    raw_labels = match.group(2)
    if not raw_labels:
        return base_name, {}

    labels = {}
    for pair_match in _LABEL_PAIR_REGEX.finditer(raw_labels):
        k, raw_val = pair_match.group(1), pair_match.group(2)
        labels[k] = _UNESCAPE_PATTERN.sub(_unescape_match, raw_val)
    return base_name, labels


def normalize_raiden_metric_name(name: str) -> str:
    """Ensures a Raiden metric name starts with RAIDEN_METRIC_PREFIX."""
    return (
        name
        if name.startswith(RAIDEN_METRIC_PREFIX)
        else f"{RAIDEN_METRIC_PREFIX}{name}"
    )


def _resolve_collector_registry(metric_cls: Any) -> CollectorRegistry | None:
    """Extracts the target CollectorRegistry for a Prometheus metric factory."""
    if isinstance(metric_cls, partial):
        reg = (metric_cls.keywords or {}).get("registry")
        if isinstance(reg, CollectorRegistry):
            return reg
        if metric_cls.func in (Counter, Gauge, Histogram):
            return prometheus_client.REGISTRY
        return None
    if metric_cls in (Counter, Gauge, Histogram):
        return prometheus_client.REGISTRY
    return None


def _get_or_create_collector(
    *,
    metric_cls: Any,
    mtype_str: str,
    name: str,
    documentation: str,
    labelnames: list[str],
    buckets: Sequence[float] | None = None,
) -> Any | None:
    """Creates or reuses a Prometheus collector without duplicate registration."""
    label_tuple = tuple(labelnames)
    registry = _resolve_collector_registry(metric_cls)
    reg_cache: dict[str, tuple[str, Any, tuple[str, ...]]] | None = None
    if registry is not None:
        reg_cache = _COLLECTOR_CACHE.setdefault(registry, {})
        cached = reg_cache.get(name)
        if cached is not None:
            cached_type, cached_obj, cached_labels = cached
            if cached_type == mtype_str and cached_labels == label_tuple:
                return cached_obj
            logger.warning(
                "Skipping incompatible cached Raiden collector %s "
                "(existing type=%s labels=%s, requested type=%s labels=%s)",
                name,
                cached_type,
                cached_labels,
                mtype_str,
                label_tuple,
            )
            return None

    kwargs: dict[str, Any] = {
        "name": name,
        "documentation": documentation,
        "labelnames": labelnames,
    }
    if mtype_str == METRIC_TYPE_GAUGE:
        kwargs["multiprocess_mode"] = "mostrecent"
    if buckets is not None:
        kwargs["buckets"] = buckets

    collector = metric_cls(**kwargs)

    if reg_cache is not None:
        reg_cache[name] = (mtype_str, collector, label_tuple)
    return collector


class RaidenDynamicPromMetrics:
    """Shared dynamic Prometheus metric registration and observation for Raiden."""

    def __init__(
        self,
        metric_types: Mapping[type[Any], Any],
        labelnames: Sequence[str] | None,
        per_engine_labelvalues: dict[int, list[object]] | None,
        backends: Sequence[str] | None = None,
    ) -> None:
        self._gauge_cls = metric_types[Gauge]
        self._counter_cls = metric_types[Counter]
        self._histogram_cls = metric_types[Histogram]
        self.base_labels: list[str] = list(labelnames) if labelnames else []
        self.per_engine_labelvalues: dict[int, list[object]] = (
            per_engine_labelvalues or {}
        )
        self._backends: list[str] = (
            list(dist_utils.DEFAULT_RAIDEN_TELEMETRY_BACKENDS)
            if backends is None
            else list(backends)
        )
        self.dynamic_metrics: dict[str, tuple[str, Any, list[str]]] = (
            self._build_metric_definitions()
        )

    def _build_metric_definitions(
        self,
    ) -> dict[str, tuple[str, Any, list[str]]]:
        if not dist_utils.is_raiden_buffered_telemetry_enabled(self._backends):
            return {}

        metric_defs: dict[str, tuple[str, Any, list[str]]] = {}
        descriptors: list[Any] = []
        try:
            # Ensure C++ telemetry metadata is initialized before reading
            # descriptors.
            dist_utils.configure_raiden_telemetry(backends=self._backends)
            telemetry = dist_utils.get_raiden_telemetry_module()
            if telemetry is not None and hasattr(telemetry, "get_metric_metadata"):
                descriptors = list(telemetry.get_metric_metadata() or [])
        except Exception as e:
            logger.warning("Failed to load TPU Raiden telemetry descriptors: %s", e)

        for desc in descriptors:
            name = normalize_raiden_metric_name(desc.name)
            help_str = getattr(desc, "description", getattr(desc, "help", ""))
            mtype = desc.type
            mtype_str = getattr(mtype, "name", str(mtype)).split(".")[-1].lower()
            raw_metric_labels = list(getattr(desc, "label_names", None) or [])
            metric_label_names = [
                lbl for lbl in raw_metric_labels if lbl not in self.base_labels
            ]
            # Worker-side gauges without a rank label receive `local_rank` at
            # drain time so TP > 1 workers export distinct series instead of
            # overwriting one another.
            if (
                mtype_str == METRIC_TYPE_GAUGE
                and "local_rank" not in metric_label_names
                and "rank" not in metric_label_names
                and "local_rank" not in self.base_labels
                and "rank" not in self.base_labels
            ):
                metric_label_names.append("local_rank")
            combined_labels = self.base_labels + metric_label_names

            buckets_val: Sequence[float] | None = None
            if mtype_str == METRIC_TYPE_HISTOGRAM:
                metric_cls = self._histogram_cls
                buckets_val = desc.buckets if desc.buckets else [1.0, 10.0, 100.0]
            elif mtype_str == METRIC_TYPE_COUNTER:
                metric_cls = self._counter_cls
            elif mtype_str == METRIC_TYPE_GAUGE:
                metric_cls = self._gauge_cls
            else:
                continue

            metric_obj = _get_or_create_collector(
                metric_cls=metric_cls,
                mtype_str=mtype_str,
                name=name,
                documentation=help_str,
                labelnames=combined_labels,
                buckets=buckets_val,
            )
            if metric_obj is None:
                continue

            if metric_label_names:
                per_engine_obj: Any = metric_obj
            elif self.base_labels and self.per_engine_labelvalues:
                per_engine_obj = create_metric_per_engine(
                    metric_obj, self.per_engine_labelvalues
                )
            else:
                per_engine_obj = {
                    idx: metric_obj for idx in (self.per_engine_labelvalues or [0])
                }
            metric_defs[name] = (mtype_str, per_engine_obj, metric_label_names)
        return metric_defs

    def observe(
        self,
        samples_by_key: Mapping[str, Sequence[float | int]],
        engine_idx: int = 0,
    ) -> None:
        if not self.dynamic_metrics or not samples_by_key:
            return
        engine_labels = dict(
            zip(
                self.base_labels,
                self.per_engine_labelvalues.get(engine_idx, ()),
            )
        )
        for raw_key, values in samples_by_key.items():
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
                    label_names = self.base_labels + metric_label_names
                # A label that C++ emits wins over the vLLM engine label of the
                # same name. Base labels a descriptor also declares are dropped
                # from `metric_label_names`, so this only matters when the
                # sample key carries such a label anyway.
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
