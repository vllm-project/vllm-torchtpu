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
"""`vllm:kv_load_failures` counts requests whose external KV load failed.

See `TPUConnector._count_load_failures`.
"""

from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import pytest
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.v1.outputs import KVConnectorOutput

from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPUConnector
from vllm_torchtpu.distributed.kv_transfer.tpu_connector_stats import (
    KV_LOAD_FAILURES_METRIC_NAME,
    TpuKVConnectorPromMetrics,
    TpuKVConnectorStats,
)
from vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector import (
    TPUMultiConnector,
)
from vllm_torchtpu.offload.raiden_connector import (
    TpuOffloadPromMetrics,
    TPURaidenOffloadingScheduler,
)
from vllm_torchtpu.offload.raiden_store import (
    AdmissionOp,
    FinishedJob,
    RaidenOffloadingManager,
)

_MOD = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"


def _vllm_config(connector="TPUConnector", policy="recompute", **kwargs):
    return SimpleNamespace(
        kv_transfer_config=KVTransferConfig(
            kv_connector=connector,
            kv_role="kv_consumer",
            kv_load_failure_policy=policy,
            **kwargs,
        ),
        max_concurrent_batches=1,
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
    )


def _connector(policy="recompute", max_concurrent_batches=1):
    cfg = _vllm_config(policy=policy)
    cfg.max_concurrent_batches = max_concurrent_batches
    with patch(f"{_MOD}.TPUConnectorScheduler") as scheduler_cls:
        scheduler_cls.return_value.request_finished.return_value = (False, None)
        return TPUConnector(cfg, KVConnectorRole.SCHEDULER, MagicMock())


def _blocks(*block_ids):
    return SimpleNamespace(get_block_ids=lambda: (list(block_ids),))


def _admit(connector, req_id, *block_ids):
    request = SimpleNamespace(request_id=req_id)
    connector.update_state_after_alloc(request, _blocks(*block_ids), 16)


def _schedule(connector, *req_ids):
    scheduler_output = SimpleNamespace(num_scheduled_tokens=dict.fromkeys(req_ids, 16))
    connector.build_connector_meta(scheduler_output)


def _output(connector, *invalid_block_ids, finished_recving=None):
    connector.update_connector_output(
        KVConnectorOutput(
            invalid_block_ids=set(invalid_block_ids), finished_recving=finished_recving
        )
    )


def _failures(connector):
    stats = connector.get_kv_connector_stats()
    return None if stats is None else stats.data["kv_load_failures"]


def test_counts_a_failed_load_once_and_drains():
    connector = _connector()
    _admit(connector, "R", 3, 4)
    _admit(connector, "S", 5)
    _output(connector, 4)
    _output(connector, 4)  # Reported again.
    assert _failures(connector) == [("TPUConnector", "recompute", 1)]
    assert _failures(connector) is None


def test_fail_policy_finishes_the_request_before_the_output():
    connector = _connector(policy="fail")
    _admit(connector, "R", 3)
    connector.request_finished(SimpleNamespace(request_id="R"), [3])
    _output(connector, 3)
    assert _failures(connector) == [("TPUConnector", "fail", 1)]


@pytest.mark.parametrize("load_done_by", ["finish", "inline_load", "async_load"])
def test_reused_blocks_only_count_for_the_new_load(load_done_by):
    connector = _connector()
    _admit(connector, "R", 3)
    if load_done_by == "finish":
        connector.request_finished(SimpleNamespace(request_id="R"), [3])
        _output(connector)
    elif load_done_by == "inline_load":
        _schedule(connector, "R")
        _output(connector)
    else:
        _output(connector, finished_recving={"R"})
    # Block 3 is freed and reused by T, whose load fails.
    _admit(connector, "T", 3)
    _output(connector, 3)
    assert _failures(connector) == [("TPUConnector", "recompute", 1)]


def test_async_scheduling_counts_inline_load_failures():
    # vLLM schedules step N+1 before it processes the output of step N.
    connector = _connector(max_concurrent_batches=2)
    _admit(connector, "R", 7)
    _schedule(connector, "R")  # Step N+1 loads R inline.
    _output(connector)  # Output of step N.
    _output(connector, 7)  # Output of step N+1.
    assert _failures(connector) == [("TPUConnector", "recompute", 1)]

    _admit(connector, "S", 8)
    _schedule(connector, "S")
    _output(connector)
    _output(connector)
    _admit(connector, "T", 8)  # Block 8 is reused.
    _output(connector, 8)
    assert _failures(connector) == [("TPUConnector", "recompute", 1)]


def test_readmitted_request_gets_a_new_claim():
    # With pipeline parallelism, a preempted request can be admitted again
    # with a new load before its old claim is dropped.
    connector = _connector(max_concurrent_batches=4)
    _admit(connector, "R", 3)
    _schedule(connector, "R")
    _output(connector)
    _output(connector)
    _admit(connector, "R", 9)
    _schedule(connector, "R")
    for _ in range(3):
        _output(connector)
    _output(connector, 9)  # Fourth output after the new load was scheduled.
    assert _failures(connector) == [("TPUConnector", "recompute", 1)]


def test_offload_connector_counts_requests_with_failed_load_jobs():
    manager = MagicMock(spec=RaidenOffloadingManager)
    manager.get_stats.return_value = None
    manager.poll_finished_jobs.return_value = [
        FinishedJob(1, AdmissionOp.LOAD, False, "R", failed_device_block_ids=[3]),
        FinishedJob(2, AdmissionOp.LOAD, False, "R", failed_device_block_ids=[4]),
        FinishedJob(3, AdmissionOp.LOAD, True, "S"),
    ]

    def base_init(self, spec, vllm_config, kv_cache_config):
        self.manager = manager
        self._connector_stats = OffloadingConnectorStats()

    with patch.object(OffloadingConnectorScheduler, "__init__", base_init):
        scheduler = TPURaidenOffloadingScheduler(
            MagicMock(), _vllm_config("TPURaidenOffloadingConnector"), MagicMock()
        )
    stock_meta = OffloadingConnectorMetadata(
        load_jobs={}, store_jobs={}, jobs_to_flush=set()
    )
    with patch.object(
        OffloadingConnectorScheduler, "build_connector_meta", return_value=stock_meta
    ):
        scheduler.build_connector_meta(MagicMock())
    assert scheduler.get_stats().data["data"][KV_LOAD_FAILURES_METRIC_NAME] == {
        ("TPURaidenOffloadingConnector", "recompute"): 1
    }


def test_multi_connector_counts_on_the_loading_child_with_the_parent_policy():
    child = {"kv_connector_module_path": _MOD, "kv_role": "kv_consumer"}
    cfg = _vllm_config(
        "TPUMultiConnector",
        policy="fail",
        kv_connector_extra_config={
            "connectors": [
                {**child, "kv_connector": "TPUConnector"},
                # The scheduler acts on the parent's policy, not this one.
                {**child, "kv_connector": "TPURaidenConnector"}
                | {"kv_load_failure_policy": "recompute"},
            ]
        },
    )
    with (
        patch(f"{_MOD}.TPUConnectorScheduler") as scheduler_cls,
        patch(f"{_MOD}.TPURaidenConnectorScheduler") as raiden_scheduler_cls,
    ):
        scheduler_cls.return_value.get_num_new_matched_tokens.return_value = (0, False)
        raiden_scheduler_cls.return_value.get_num_new_matched_tokens.return_value = (
            16,
            False,
        )
        multi = TPUMultiConnector(cfg, KVConnectorRole.SCHEDULER, MagicMock())
    request = SimpleNamespace(request_id="R")
    multi.get_num_new_matched_tokens(request, 0)
    multi.update_state_after_alloc(request, _blocks(3), 16)
    multi.update_connector_output(KVConnectorOutput(invalid_block_ids={3}))

    stats = multi.get_kv_connector_stats()
    assert list(stats.data) == ["TPURaidenConnector"]
    assert stats["TPURaidenConnector"].data["kv_load_failures"] == [
        ("TPURaidenConnector", "fail", 1)
    ]


def test_prometheus_export():
    registry = CollectorRegistry()
    metric_types = {
        cls: partial(cls, registry=registry) for cls in (Counter, Gauge, Histogram)
    }
    labelnames, per_engine_labelvalues = ["model_name", "engine"], {0: ["m", "0"]}
    # Under TPUMultiConnector both children's metrics register the counter.
    tpu = TpuKVConnectorPromMetrics(
        _vllm_config("TPURaidenConnector"),
        metric_types,
        labelnames,
        per_engine_labelvalues,
    )
    offload = TpuOffloadPromMetrics(
        _vllm_config("TPURaidenOffloadingConnector"),
        metric_types,
        labelnames,
        per_engine_labelvalues,
    )

    def value(connector):
        labels = {"model_name": "m", "engine": "0", "policy": "recompute"}
        return registry.get_sample_value(
            "vllm:kv_load_failures_total", {**labels, "connector": connector}
        )

    assert value("TPURaidenConnector") == value("TPURaidenOffloadingConnector") == 0

    stats = TpuKVConnectorStats()
    stats.record_kv_load_failures("TPURaidenConnector", "recompute", 2)
    # The engine core sends stats to the API server process with msgpack.
    tpu.observe(msgspec.msgpack.decode(msgspec.msgpack.encode(stats.data)))
    offload_stats = OffloadingConnectorStats()
    offload_stats.increase_counter(
        KV_LOAD_FAILURES_METRIC_NAME,
        labelvalues=("TPURaidenOffloadingConnector", "recompute"),
    )
    offload.observe(offload_stats.data)
    assert value("TPURaidenConnector") == 2
    assert value("TPURaidenOffloadingConnector") == 1
