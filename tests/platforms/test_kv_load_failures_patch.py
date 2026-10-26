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
"""Tests for ``vllm:kv_load_failures_total`` and retry cache-stat deduplication."""

import inspect
from types import SimpleNamespace

import pytest
import torch
import vllm.v1.metrics.loggers as loggers_mod
from prometheus_client import REGISTRY
from vllm import SamplingParams
from vllm.config.kv_transfer import KVTransferConfig
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.metrics.prometheus import unregister_vllm_metrics
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_torchtpu import _patch_vllm_hybrid_kv_load_failure_recovery
from vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector import (
    TPUMultiConnector,
)
from vllm_torchtpu.offload.raiden_connector import (
    TPURaidenOffloadingConnector,
    TPURaidenOffloadingScheduler,
)

BLOCK_SIZE = 16


def _prom_logger_cls():
    cls = getattr(loggers_mod, "PrometheusStatLogger", None) or getattr(
        loggers_mod, "PromStatLogger", None
    )
    assert cls is not None
    return cls


@pytest.fixture(autouse=True)
def patched(monkeypatch):
    from vllm_torchtpu.platforms import tpu_platform

    prom_cls = _prom_logger_cls()
    for name in (
        "_update_requests_with_invalid_blocks",
        "_handle_invalid_blocks",
        "_handle_failed_recving",
        "_get_local_prefix_cache_hit",
        "schedule",
    ):
        monkeypatch.setattr(Scheduler, name, inspect.unwrap(getattr(Scheduler, name)))
    monkeypatch.setattr(
        PrefixCacheStats, "record", inspect.unwrap(PrefixCacheStats.record)
    )
    monkeypatch.setattr(
        KVCacheManager,
        "record_prefix_cache_stats",
        inspect.unwrap(KVCacheManager.record_prefix_cache_stats),
    )
    unregister_vllm_metrics()
    tpu_platform._tpu_kv_connectors_registered = False
    tpu_platform._register_tpu_kv_connectors()
    _patch_vllm_hybrid_kv_load_failure_recovery()
    yield prom_cls
    unregister_vllm_metrics()


def _make_raiden_scheduler_connector():
    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPURaidenConnector

    conn = object.__new__(TPURaidenConnector)
    conn.use_raiden = False
    conn.connector_scheduler = SimpleNamespace()
    conn.connector_worker = None
    return conn


def _make_offloading_scheduler_connector():
    sched_impl = object.__new__(TPURaidenOffloadingScheduler)
    sched_impl._connector_stats = (
        TPURaidenOffloadingConnector.build_kv_connector_stats()
    )
    sched_impl.manager = SimpleNamespace(get_stats=lambda: None)
    conn = object.__new__(TPURaidenOffloadingConnector)
    conn.connector_scheduler = sched_impl
    conn.connector_worker = None
    return conn


def _make_multi_scheduler_connector(children, reqs_to_conn):
    conn = object.__new__(TPUMultiConnector)
    conn._connectors = list(children)
    conn._requests_to_connector = dict(reqs_to_conn)
    conn._extra_async_saves = {}
    return conn


def _make_kv_cache_manager(num_groups: int = 1) -> tuple[KVCacheManager, KVCacheConfig]:
    init_none_hash(sha256)
    groups = [
        KVCacheGroupSpec(
            [f"layer_{i}"],
            FullAttentionSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=1,
                head_size=64,
                dtype=torch.float32,
            ),
        )
        for i in range(num_groups)
    ]
    config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )
    manager = KVCacheManager(
        config,
        max_model_len=256,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        enable_caching=True,
        log_stats=True,
    )
    return manager, config


def _make_request(req_id: str, num_tokens: int = 32) -> Request:
    return Request(
        req_id,
        list(range(num_tokens)),
        SamplingParams(max_tokens=4),
        None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _make_scheduler_stub(
    *,
    num_groups: int = 1,
    recompute: bool = True,
    connector=None,
) -> Scheduler:
    manager, config = _make_kv_cache_manager(num_groups=num_groups)
    sched = object.__new__(Scheduler)
    sched.kv_cache_manager = manager
    sched.kv_cache_config = config
    sched.block_size = BLOCK_SIZE
    sched.recompute_kv_load_failures = recompute
    sched.connector = (
        connector if connector is not None else _make_raiden_scheduler_connector()
    )
    sched.ec_connector = None
    sched.requests = {}
    sched.waiting = FCFSRequestQueue()
    sched.kv_holding_waiting = FCFSRequestQueue()
    sched.deferred_waiting = set()
    sched.running = []
    sched.failed_recving_kv_req_ids = set()
    sched.finished_recving_kv_req_ids = set()
    sched._kv_fetch_counts = {}
    sched.log_stats = True
    sched.connector_prefix_cache_stats = PrefixCacheStats()
    sched.kv_metrics_collector = None
    sched.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector=type(sched.connector).__name__,
            kv_load_failure_policy="recompute" if recompute else "fail",
        )
    )
    return sched


def _collect_kv_load_failure_samples() -> list[tuple[dict[str, str], float]]:
    for family in REGISTRY.collect():
        if family.name == "vllm:kv_load_failures":
            return [
                (dict(s.labels), s.value)
                for s in family.samples
                if s.name == "vllm:kv_load_failures_total"
            ]
    return []


def _make_prom_vllm_config(kv_transfer_config: KVTransferConfig):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            served_model_name="qwen3-32b",
            max_model_len=256,
            is_diffusion=False,
        ),
        observability_config=SimpleNamespace(
            show_hidden_metrics=False,
            kv_cache_metrics=False,
            custom_histogram_buckets=None,
            cudagraph_metrics=False,
        ),
        cache_config=SimpleNamespace(
            block_size=16,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=True,
            prefix_caching_hash_algo="sha256",
            cpu_offload_gb=0,
            calculate_kv_scales=False,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        parallel_config=SimpleNamespace(data_parallel_size=1),
        speculative_config=None,
        ec_transfer_config=None,
        lora_config=None,
        kv_transfer_config=kv_transfer_config,
    )


def _collect_scheduler_stats(sched: Scheduler) -> SchedulerStats:
    kv_connector_stats = (
        sched.connector.get_kv_connector_stats()
        if sched.connector is not None
        else None
    )
    if kv_connector_stats is not None and kv_connector_stats.is_empty():
        kv_connector_stats = None
    stats = sched.make_stats(kv_connector_stats=kv_connector_stats)
    assert stats is not None
    return stats


def test_single_connector_invalid_blocks_and_make_stats_drain():
    sched = _make_scheduler_stub(recompute=True)
    req = _make_request("req_async")
    blocks = sched.kv_cache_manager.allocate_slots(req, BLOCK_SIZE)
    assert blocks is not None
    req.num_computed_tokens = BLOCK_SIZE
    req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    sched.requests[req.request_id] = req
    sched.kv_holding_waiting.add_request(req)

    invalid_block_id = sched.kv_cache_manager.get_block_ids(req.request_id)[0][0]
    skipped = sched._handle_invalid_blocks({invalid_block_id}, {})

    assert skipped == set()
    assert req.request_id in sched.failed_recving_kv_req_ids
    assert req.num_computed_tokens == 0
    assert getattr(req, "_tpu_kv_load_recompute_pending", False) is True

    stats = _collect_scheduler_stats(sched)
    assert isinstance(stats.kv_connector_stats, dict)
    assert stats.kv_connector_stats["kv_load_failures"] == [
        ("TPURaidenConnector", "recompute", 1)
    ]

    # Subsequent collection must return None for kv_connector_stats after draining.
    next_stats = _collect_scheduler_stats(sched)
    assert next_stats.kv_connector_stats is None


def test_multi_connector_child_attribution_for_invalid_blocks_and_failed_recving():
    multi_conn = _make_multi_scheduler_connector(
        children=[
            _make_raiden_scheduler_connector(),
            _make_offloading_scheduler_connector(),
        ],
        reqs_to_conn={"req_p2p": 0, "req_offload": 1},
    )
    sched = _make_scheduler_stub(recompute=False, connector=multi_conn)

    req_p2p = _make_request("req_p2p")
    assert sched.kv_cache_manager.allocate_slots(req_p2p, BLOCK_SIZE) is not None
    req_p2p.num_computed_tokens = BLOCK_SIZE
    req_p2p.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    sched.requests[req_p2p.request_id] = req_p2p
    sched.kv_holding_waiting.add_request(req_p2p)

    req_offload = _make_request("req_offload")
    assert sched.kv_cache_manager.allocate_slots(req_offload, BLOCK_SIZE) is not None
    req_offload.num_computed_tokens = BLOCK_SIZE
    req_offload.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    sched.requests[req_offload.request_id] = req_offload
    sched.kv_holding_waiting.add_request(req_offload)

    failed = sched._handle_failed_recving({"req_p2p"})
    assert failed == {"req_p2p"}

    offload_block_id = sched.kv_cache_manager.get_block_ids("req_offload")[0][0]
    failed_invalid = sched._handle_invalid_blocks({offload_block_id}, {})
    assert failed_invalid == {"req_offload"}

    stats = _collect_scheduler_stats(sched)
    assert isinstance(stats.kv_connector_stats, dict)
    raiden_stats = stats.kv_connector_stats["TPURaidenConnector"]
    offload_stats = stats.kv_connector_stats["TPURaidenOffloadingConnector"]
    assert raiden_stats["kv_load_failures"] == [("TPURaidenConnector", "fail", 1)]
    assert offload_stats["data"]["vllm:kv_load_failures"] == {
        ("TPURaidenOffloadingConnector", "fail"): 1
    }


def test_hybrid_multi_group_invalid_blocks_does_not_raise():
    sched = _make_scheduler_stub(num_groups=2, recompute=True)
    req = _make_request("req_hybrid")
    assert sched.kv_cache_manager.allocate_slots(req, BLOCK_SIZE) is not None
    req.num_computed_tokens = BLOCK_SIZE
    req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    sched.requests[req.request_id] = req
    sched.kv_holding_waiting.add_request(req)

    group1_block_id = sched.kv_cache_manager.get_block_ids(req.request_id)[1][0]
    skipped = sched._handle_invalid_blocks({group1_block_id}, {})

    assert skipped == set()
    assert req.num_computed_tokens == 0
    assert req.request_id in sched.failed_recving_kv_req_ids
    stats = _collect_scheduler_stats(sched)
    assert stats.kv_connector_stats is not None
    assert stats.kv_connector_stats["kv_load_failures"] == [
        ("TPURaidenConnector", "recompute", 1)
    ]


def test_recompute_retry_skips_duplicate_external_prefix_cache_stats():
    sched = _make_scheduler_stub(recompute=True)
    req_retry = _make_request("req_retry", num_tokens=32)
    req_fresh = _make_request("req_fresh", num_tokens=32)
    sched.requests[req_retry.request_id] = req_retry
    sched.requests[req_fresh.request_id] = req_fresh

    # Initial admission of req_retry records 32 queried tokens and 16 hit tokens.
    sched._get_local_prefix_cache_hit(req_retry)
    sched.connector_prefix_cache_stats.record(
        num_tokens=32, num_hits=16, preempted=False
    )
    sched.kv_cache_manager.record_prefix_cache_stats(req_retry, 0)
    assert sched.connector_prefix_cache_stats.requests == 1
    assert sched.connector_prefix_cache_stats.queries == 32
    assert sched.connector_prefix_cache_stats.hits == 16

    # Async load fails under recompute policy and resets num_computed_tokens to 0.
    req_retry.num_computed_tokens = 16
    req_retry.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    sched._handle_failed_recving({req_retry.request_id})
    assert req_retry.num_computed_tokens == 0
    assert getattr(req_retry, "_tpu_kv_load_recompute_pending", False) is True

    # Simulate a step where _get_local_prefix_cache_hit runs for req_retry,
    # allocate_slots fails (record not called), and a fresh request is admitted.
    sched._get_local_prefix_cache_hit(req_retry)
    sched._get_local_prefix_cache_hit(req_fresh)
    sched.connector_prefix_cache_stats.record(
        num_tokens=32, num_hits=0, preempted=False
    )
    sched.kv_cache_manager.record_prefix_cache_stats(req_fresh, 0)
    # req_fresh was counted; req_retry is still pending its retry admission.
    assert sched.connector_prefix_cache_stats.requests == 2
    assert sched.connector_prefix_cache_stats.queries == 64
    assert sched.connector_prefix_cache_stats.hits == 16
    assert getattr(req_retry, "_tpu_kv_load_recompute_pending", False) is True

    # Next step: req_retry is admitted on its recompute retry pass.
    sched._get_local_prefix_cache_hit(req_retry)
    sched.connector_prefix_cache_stats.record(
        num_tokens=32, num_hits=0, preempted=False
    )
    sched.kv_cache_manager.record_prefix_cache_stats(req_retry, 0)

    # Duplicate external prefix cache query/hit count was suppressed!
    assert sched.connector_prefix_cache_stats.requests == 2
    assert sched.connector_prefix_cache_stats.queries == 64
    assert sched.connector_prefix_cache_stats.hits == 16
    assert getattr(req_retry, "_tpu_kv_load_recompute_pending", False) is False


def test_prometheus_stat_logger_exports_kv_load_failures_across_msgspec_ipc(
    patched,
):
    prom_cls = patched
    vllm_config = _make_prom_vllm_config(
        KVTransferConfig(
            kv_connector="TPUMultiConnector",
            kv_role="kv_consumer",
            kv_load_failure_policy="recompute",
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "TPURaidenConnector", "kv_role": "kv_consumer"},
                    {
                        "kv_connector": "TPURaidenOffloadingConnector",
                        "kv_role": "kv_both",
                    },
                ]
            },
        )
    )
    logger = prom_cls(vllm_config, engine_indexes=[0, 1])

    # Verify zero-initialized series exist before any failure occurs.
    initial = {
        (labels["engine"], labels["connector"], labels["policy"]): val
        for labels, val in _collect_kv_load_failure_samples()
    }
    assert initial == {
        ("0", "TPURaidenConnector", "recompute"): 0.0,
        ("0", "TPURaidenOffloadingConnector", "recompute"): 0.0,
        ("1", "TPURaidenConnector", "recompute"): 0.0,
        ("1", "TPURaidenOffloadingConnector", "recompute"): 0.0,
    }

    multi_conn = _make_multi_scheduler_connector(
        children=[
            _make_raiden_scheduler_connector(),
            _make_offloading_scheduler_connector(),
        ],
        reqs_to_conn={"req_p2p_1": 0, "req_p2p_2": 0, "req_offload": 1},
    )
    sched = _make_scheduler_stub(recompute=True, connector=multi_conn)
    for req_id in ("req_p2p_1", "req_p2p_2", "req_offload"):
        req = _make_request(req_id)
        req.num_computed_tokens = BLOCK_SIZE
        req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        sched.requests[req_id] = req

    sched._handle_failed_recving({"req_p2p_1", "req_p2p_2", "req_offload"})
    sched_stats = _collect_scheduler_stats(sched)

    # Round-trip EngineCoreOutputs through vLLM's MsgpackEncoder/MsgpackDecoder
    # to verify SchedulerStats.kv_connector_stats survives multi-process IPC.
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(EngineCoreOutputs)
    wire_chunks = encoder.encode(EngineCoreOutputs(scheduler_stats=sched_stats))
    decoded_outputs = decoder.decode(wire_chunks)
    assert decoded_outputs.scheduler_stats is not None
    assert isinstance(decoded_outputs.scheduler_stats.kv_connector_stats, dict)

    logger.record(decoded_outputs.scheduler_stats, None, engine_idx=0)

    after = {
        (labels["engine"], labels["connector"], labels["policy"]): val
        for labels, val in _collect_kv_load_failure_samples()
    }
    assert after[("0", "TPURaidenConnector", "recompute")] == 2.0
    assert after[("0", "TPURaidenOffloadingConnector", "recompute")] == 1.0
    assert after[("1", "TPURaidenConnector", "recompute")] == 0.0


def test_tpu_connector_drains_evictions_and_load_failures_together():
    conn = _make_raiden_scheduler_connector()
    conn._num_evicted_blocks = 3
    conn.record_kv_load_failure("TPURaidenConnector", "recompute", count=2)

    stats = conn.get_kv_connector_stats()
    assert stats is not None
    assert stats.data["num_evicted_blocks"] == [3]
    assert stats.data["kv_load_failures"] == [("TPURaidenConnector", "recompute", 2)]
    assert conn.get_kv_connector_stats() is None
