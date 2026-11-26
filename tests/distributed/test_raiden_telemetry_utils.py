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
"""Unit tests for TPU Raiden telemetry utility functions."""

from functools import partial

import pytest
from prometheus_client import CollectorRegistry, Gauge, Histogram

from vllm_torchtpu.distributed.kv_transfer.raiden import telemetry_utils
from vllm_torchtpu.distributed.kv_transfer.raiden.telemetry_utils import (
    _UNESCAPE_PATTERN,
    RAIDEN_METRIC_PREFIX,
    _unescape_match,
    normalize_raiden_metric_name,
    parse_raiden_metric_key,
)


def test_parse_raiden_metric_key_unlabeled():
    base, labels = parse_raiden_metric_key("tpu_raiden_h2d_transfer_time_ms")
    assert base == "tpu_raiden_h2d_transfer_time_ms"
    assert labels == {}

    base, labels = parse_raiden_metric_key("tpu_raiden:custom_latency:seconds")
    assert base == "tpu_raiden:custom_latency:seconds"
    assert labels == {}

    base, labels = parse_raiden_metric_key("   tpu_raiden_h2d_transfer_time_ms   ")
    assert base == "tpu_raiden_h2d_transfer_time_ms"
    assert labels == {}

    base, labels = parse_raiden_metric_key("tpu_raiden_metric{}")
    assert base == "tpu_raiden_metric"
    assert labels == {}

    base, labels = parse_raiden_metric_key("tpu_raiden_metric{   }")
    assert base == "tpu_raiden_metric"
    assert labels == {}


def test_parse_raiden_metric_key_single_label():
    base, labels = parse_raiden_metric_key(
        'tpu_raiden_sent_bytes_total{direction="push"}'
    )
    assert base == "tpu_raiden_sent_bytes_total"
    assert labels == {"direction": "push"}


def test_parse_raiden_metric_key_multiple_labels():
    base, labels = parse_raiden_metric_key(
        'tpu_raiden_transfer_duration_ms{direction="pull",mode="direct"}'
    )
    assert base == "tpu_raiden_transfer_duration_ms"
    assert labels == {"direction": "pull", "mode": "direct"}

    base, labels = parse_raiden_metric_key(
        'tpu_raiden_transfer_duration_ms{ direction = "pull" , mode = "direct" }'
    )
    assert base == "tpu_raiden_transfer_duration_ms"
    assert labels == {"direction": "pull", "mode": "direct"}


def test_parse_raiden_metric_key_escaping():
    base, labels = parse_raiden_metric_key(
        'tpu_raiden_failures{error="msg=\\"timeout\\""}'
    )
    assert base == "tpu_raiden_failures"
    assert labels == {"error": 'msg="timeout"'}

    raw_escape = r'tpu_raiden_failures{error="line1\nline2",path="C:\\dir"}'
    base, labels = parse_raiden_metric_key(raw_escape)
    assert base == "tpu_raiden_failures"
    assert labels == {"error": "line1\nline2", "path": "C:\\dir"}

    raw_escape_tab = r'tpu_raiden_metric{tab="col1\tcol2",cr="a\rb"}'
    base, labels = parse_raiden_metric_key(raw_escape_tab)
    assert base == "tpu_raiden_metric"
    assert labels == {"tab": "col1\tcol2", "cr": "a\rb"}


def test_parse_raiden_metric_key_special_characters():
    base, labels = parse_raiden_metric_key(
        'tpu_raiden_metric{pattern="{foo}", key = "val" }'
    )
    assert base == "tpu_raiden_metric"
    assert labels == {"pattern": "{foo}", "key": "val"}

    base, labels = parse_raiden_metric_key(
        'tpu_raiden_metric{tags="a,b,c",stage="decode"}'
    )
    assert base == "tpu_raiden_metric"
    assert labels == {"tags": "a,b,c", "stage": "decode"}

    base, labels = parse_raiden_metric_key('tpu_raiden_metric{expr="k=v",flag="true"}')
    assert base == "tpu_raiden_metric"
    assert labels == {"expr": "k=v", "flag": "true"}

    base, labels = parse_raiden_metric_key('tpu_raiden_metric{empty="",nonempty="val"}')
    assert base == "tpu_raiden_metric"
    assert labels == {"empty": "", "nonempty": "val"}


def test_parse_raiden_metric_key_malformed_and_edge_cases():
    base, labels = parse_raiden_metric_key("  not_a_valid{unclosed  ")
    assert base == "not_a_valid{unclosed"
    assert labels == {}

    base, labels = parse_raiden_metric_key('123_invalid{foo="bar"}')
    assert base == '123_invalid{foo="bar"}'
    assert labels == {}

    base, labels = parse_raiden_metric_key("tpu_raiden_metric{unquoted=val}")
    assert base == "tpu_raiden_metric"
    assert labels == {}

    base, labels = parse_raiden_metric_key("")
    assert base == ""
    assert labels == {}

    base, labels = parse_raiden_metric_key("   ")
    assert base == ""
    assert labels == {}

    base, labels = parse_raiden_metric_key("invalid metric name")
    assert base == "invalid metric name"
    assert labels == {}


def test_unescape_match_fallback():
    # Directly test _unescape_match for quote and backslash pass-through
    match_quote = _UNESCAPE_PATTERN.search(r"\"")
    assert match_quote is not None
    assert _unescape_match(match_quote) == '"'

    match_slash = _UNESCAPE_PATTERN.search(r"\\")
    assert match_slash is not None
    assert _unescape_match(match_slash) == "\\"


def test_constants():
    assert RAIDEN_METRIC_PREFIX == "tpu_raiden_"


def test_normalize_raiden_metric_name():
    # Prefixed name returns unchanged
    assert (
        normalize_raiden_metric_name("tpu_raiden_sent_bytes_total")
        == "tpu_raiden_sent_bytes_total"
    )

    # Unprefixed name gets prefixed
    assert (
        normalize_raiden_metric_name("sent_bytes_total")
        == "tpu_raiden_sent_bytes_total"
    )


class TestRaidenCollectorCache:
    def test_invalid_collector_definition_is_not_swallowed(self) -> None:
        registry = CollectorRegistry()
        # `le` is reserved for histograms, so construction raises ValueError.
        with pytest.raises(ValueError):
            telemetry_utils._get_or_create_collector(
                metric_cls=partial(Histogram, registry=registry),
                mtype_str="histogram",
                name="tpu_raiden_bad_histogram",
                documentation="reserved label",
                labelnames=["le"],
            )

    def test_gauge_uses_mostrecent_multiprocess_mode(self) -> None:
        registry = CollectorRegistry()
        gauge = telemetry_utils._get_or_create_collector(
            metric_cls=partial(Gauge, registry=registry),
            mtype_str="gauge",
            name="tpu_raiden_test_gauge",
            documentation="test gauge",
            labelnames=["model_name"],
        )
        assert gauge is not None
        assert gauge._multiprocess_mode == "mostrecent"

    def test_cached_collector_reused_on_same_registry(self) -> None:
        registry = CollectorRegistry()
        first = telemetry_utils._get_or_create_collector(
            metric_cls=partial(Gauge, registry=registry),
            mtype_str="gauge",
            name="tpu_raiden_cached_gauge",
            documentation="cached gauge",
            labelnames=["x"],
        )
        second = telemetry_utils._get_or_create_collector(
            metric_cls=partial(Gauge, registry=registry),
            mtype_str="gauge",
            name="tpu_raiden_cached_gauge",
            documentation="cached gauge",
            labelnames=["x"],
        )
        assert first is second

    def test_foreign_duplicate_raises_value_error(self) -> None:
        registry = CollectorRegistry()
        Gauge("tpu_raiden_foreign", "foreign", ["x"], registry=registry)
        with pytest.raises(ValueError):
            telemetry_utils._get_or_create_collector(
                metric_cls=partial(Gauge, registry=registry),
                mtype_str="gauge",
                name="tpu_raiden_foreign",
                documentation="foreign",
                labelnames=["x"],
            )
