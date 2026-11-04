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
"""Tests for KV transfer labels exported on ``vllm:cache_config_info``."""

import inspect
from types import SimpleNamespace

import pytest
from prometheus_client import REGISTRY
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.prometheus import unregister_vllm_metrics

from vllm_torchtpu.platforms.tpu_platform import (
    _get_kv_transfer_metrics_info,
    _register_cache_config_kv_metrics_info,
)


@pytest.fixture(autouse=True)
def registered_prom_logger(monkeypatch):
    monkeypatch.setattr(
        PrometheusStatLogger,
        "log_metrics_info",
        inspect.unwrap(PrometheusStatLogger.log_metrics_info),
    )
    unregister_vllm_metrics()
    _register_cache_config_kv_metrics_info()
    yield PrometheusStatLogger
    unregister_vllm_metrics()


class _StubCacheConfig:
    def metrics_info(self) -> dict[str, str]:
        return {
            "block_size": "128",
            "num_gpu_blocks": "2048",
            "cache_dtype": "auto",
            "enable_prefix_caching": "True",
        }


def _collect_cache_config_samples() -> list[tuple[dict[str, str], float]]:
    for family in REGISTRY.collect():
        if family.name == "vllm:cache_config_info":
            return [(dict(s.labels), s.value) for s in family.samples]
    return []


def test_cache_config_info_without_kv_transfer(registered_prom_logger):
    stat_logger = object.__new__(registered_prom_logger)
    stat_logger.engine_indexes = [0]
    stat_logger.vllm_config = SimpleNamespace(
        cache_config=_StubCacheConfig(),
        kv_transfer_config=None,
    )

    stat_logger.log_engine_initialized()

    samples = _collect_cache_config_samples()
    assert len(samples) == 1
    labels, value = samples[0]
    assert value == 1.0
    assert labels["engine"] == "0"
    assert labels["block_size"] == "128"
    assert labels["num_gpu_blocks"] == "2048"
    assert labels["kv_connector"] == ""
    assert labels["kv_connectors"] == ""
    assert labels["kv_role"] == ""
    assert labels["kv_load_failure_policy"] == ""


def test_cache_config_info_with_single_kv_connector(registered_prom_logger):
    stat_logger = object.__new__(registered_prom_logger)
    stat_logger.engine_indexes = [0]
    stat_logger.vllm_config = SimpleNamespace(
        cache_config=_StubCacheConfig(),
        kv_transfer_config=SimpleNamespace(
            kv_connector="TPURaidenConnector",
            kv_connector_extra_config={"backend": "raiden"},
            kv_role="kv_consumer",
            kv_load_failure_policy="recompute",
        ),
    )

    stat_logger.log_engine_initialized()

    samples = _collect_cache_config_samples()
    assert len(samples) == 1
    labels, value = samples[0]
    assert value == 1.0
    assert labels["engine"] == "0"
    assert labels["kv_connector"] == "TPURaidenConnector"
    assert labels["kv_connectors"] == ""
    assert labels["kv_role"] == "kv_consumer"
    assert labels["kv_load_failure_policy"] == "recompute"


def test_cache_config_info_with_multi_connector_and_multiple_engines(
    registered_prom_logger,
):
    stat_logger = object.__new__(registered_prom_logger)
    stat_logger.engine_indexes = [0, 1]
    stat_logger.vllm_config = SimpleNamespace(
        cache_config=_StubCacheConfig(),
        kv_transfer_config=SimpleNamespace(
            kv_connector="TPUMultiConnector",
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "TPURaidenConnector"},
                    {"kv_connector": "TPURaidenOffloadingConnector"},
                ]
            },
            kv_role="kv_both",
            kv_load_failure_policy="fail",
        ),
    )

    stat_logger.log_engine_initialized()

    samples = _collect_cache_config_samples()
    assert len(samples) == 2
    by_engine = {labels["engine"]: (labels, val) for labels, val in samples}
    assert set(by_engine) == {"0", "1"}
    for engine_str in ("0", "1"):
        labels, value = by_engine[engine_str]
        assert value == 1.0
        assert labels["kv_connector"] == "TPUMultiConnector"
        assert (
            labels["kv_connectors"] == "TPURaidenConnector,TPURaidenOffloadingConnector"
        )
        assert labels["kv_role"] == "kv_both"
        assert labels["kv_load_failure_policy"] == "fail"


def test_get_kv_transfer_metrics_info_handles_empty_or_malformed_specs():
    assert _get_kv_transfer_metrics_info(None) == {
        "kv_connector": "",
        "kv_connectors": "",
        "kv_role": "",
        "kv_load_failure_policy": "",
    }
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="TPUMultiConnector",
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "TPURaidenConnector"},
                    "not-a-dict",
                    {"kv_connector": ""},
                ]
            },
            kv_role=None,
            kv_load_failure_policy=None,
        )
    )
    assert _get_kv_transfer_metrics_info(vllm_config) == {
        "kv_connector": "TPUMultiConnector",
        "kv_connectors": "TPURaidenConnector",
        "kv_role": "",
        "kv_load_failure_policy": "",
    }


def test_register_cache_config_kv_metrics_info_is_idempotent(registered_prom_logger):
    first = registered_prom_logger.log_metrics_info
    _register_cache_config_kv_metrics_info()
    assert registered_prom_logger.log_metrics_info is first
