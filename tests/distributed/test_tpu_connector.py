# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import contextlib
import dataclasses
import json
import logging
import math
import pickle
from collections.abc import Iterator, Mapping, Sequence
from vllm.config import VllmConfig
from vllm.v1.core.sched.output import SchedulerOutput
"""
Unit tests for TPUConnector, TPUConnectorScheduler, and TPUConnectorWorker.

Structure mirrors tests/distributed/test_tpu_connector.py in the upstream
tpu-inference repo, adapted for the torch-based coordinator-mode connector
used here.  Heavy dependencies (ZMQ sockets, shared-memory pool, TPU device
queries) are mocked at construction time so that all tests run without a real
TPU.
"""
import threading
import time
import unittest
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, MagicMock, patch

import pytest
import torch
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, MambaSpec)
from vllm.v1.request import RequestStatus

from vllm_torchtpu.distributed.kv_transfer.tpu_connector_stats import (
    TpuKVConnectorPromMetrics, TpuKVConnectorStats)

from vllm_torchtpu.distributed.kv_transfer.connector_metadata import SendMeta
from vllm_torchtpu.distributed.kv_transfer.raiden import byte_spans
from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import PoolByteSpan, PoolSpanRegistration
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
    CHUNK_TRANSFER_WINDOW,
    _Stage3SendState,
    _compute_boundary_skip_bytes,
    _compute_cached_prefix_progress,
    _get_chunk_transfer_window,
    _validate_interval_tiling,
    chunk_uuid,  # isort: skip
    LoadMeta, TPUConnector, TPUConnectorMetadata, TPUConnectorScheduler,
    TPUConnectorWorker, TPURaidenConnector, TPURaidenConnectorScheduler,
    TPURaidenConnectorWorker, _CoordRecvEntry, _CoordSendEntry,
    _Stage3LoadMeta, _Stage3RegisteredSend, _select_committed_mamba_blocks,
    stage3_fa_raiden_id_fields)

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

_MOD = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"
_BASE = "vllm_torchtpu.distributed.kv_transfer.zmq_shm_base"


def _make_test_kv_cache_config() -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["model.layers.0.self_attn"],
                kv_cache_spec=FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=8,
                    dtype=torch.bfloat16,
                ),
            )
        ],
    )


def _make_stage3_hybrid_kv_cache_config(
    *,
    fa_group_index: int = 2,
    num_speculative_blocks: int = 0,
) -> KVCacheConfig:
    fa_group = KVCacheGroupSpec(
        layer_names=["model.layers.2.self_attn"],
        kv_cache_spec=FullAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.bfloat16,
        ),
    )
    mamba_groups = [
        KVCacheGroupSpec(
            layer_names=[f"model.layers.{index}.linear_attn"],
            kv_cache_spec=MambaSpec(
                block_size=16,
                shapes=((1, ), ),
                dtypes=(torch.bfloat16, ),
                num_speculative_blocks=num_speculative_blocks,
            ),
        ) for index in range(3)
    ]
    groups = list(mamba_groups)
    groups.insert(fa_group_index, fa_group)
    return KVCacheConfig(num_blocks=0,
                         kv_cache_tensors=[],
                         kv_cache_groups=groups)


@pytest.mark.parametrize(("num_speculative_blocks", "expected"),
                         ((0, 14), (3, 11)))
def test_select_committed_mamba_blocks(num_speculative_blocks, expected):
    assert _select_committed_mamba_blocks(
        [[10, 11, 12, 13, 14]],
        num_speculative_blocks,
    ) == [expected]


def _make_vllm_config(*,
                      is_producer: bool = True,
                      block_size: int = 16,
                      enable_prefix_caching: bool = False,
                      dp_rank: int = 0,
                      dp_size: int = 1,
                      tp_size: int = 1,
                      pcp_size: int = 1,
                      interleave_size: int = 1):
    cfg = MagicMock()
    cfg.kv_transfer_config.is_kv_producer = is_producer
    cfg.kv_transfer_config.kv_connector_extra_config = {}
    cfg.cache_config.block_size = block_size
    cfg.cache_config.enable_prefix_caching = enable_prefix_caching
    cfg.model_config.max_model_len = 64
    cfg.model_config.hf_config = SimpleNamespace(
        num_key_value_heads=2,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
    )
    cfg.parallel_config.data_parallel_rank = dp_rank
    cfg.parallel_config.data_parallel_size = dp_size
    cfg.parallel_config.tensor_parallel_size = tp_size
    cfg.parallel_config.prefill_context_parallel_size = pcp_size
    cfg.parallel_config.cp_kv_cache_interleave_size = interleave_size
    cfg.parallel_config.cp_kv_cache_interleave_size = interleave_size
    return cfg


def _make_scheduler(*,
                    is_producer: bool = False,
                    dp_rank: int = 0,
                    tp_size: int = 1,
                    kv_ips: Any = "127.0.0.1",
                    kv_ports: Any = 9100):
    """Construct a TPUConnectorScheduler with network calls patched out."""
    cfg = _make_vllm_config(is_producer=is_producer,
                            dp_rank=dp_rank,
                            tp_size=tp_size)
    with patch(f"{_MOD}.dist_utils.get_kv_ips", return_value=kv_ips), \
         patch(f"{_MOD}.dist_utils.get_kv_ports", return_value=kv_ports), \
         patch(f"{_MOD}.dist_utils.get_side_channel_port", return_value="9600"):
        return TPUConnectorScheduler(cfg)


def _make_raiden_scheduler(*,
                           is_producer: bool = False,
                           block_size: int = 16,
                           enable_prefix_caching: bool = False,
                           dp_rank: int = 0,
                           tp_size: int = 1,
                           pcp_size: int = 1,
                           interleave_size: int = 256,
                           chunk_transfer_window: int = 2,
                           kv_ips: Any = "127.0.0.1",
                           kv_ports: Any = 9100):
    """Construct a TPURaidenConnectorScheduler with network calls patched."""
    cfg = _make_vllm_config(is_producer=is_producer,
                            block_size=block_size,
                            enable_prefix_caching=enable_prefix_caching,
                            dp_rank=dp_rank,
                            tp_size=tp_size,
                            pcp_size=pcp_size,
                            interleave_size=interleave_size)
    cfg.kv_transfer_config.chunk_transfer_window = chunk_transfer_window
    with patch(f"{_MOD}.dist_utils.get_kv_ips", return_value=kv_ips), \
         patch(f"{_MOD}.dist_utils.get_kv_ports", return_value=kv_ports):
        sched = TPURaidenConnectorScheduler(cfg)
        sched.chunk_transfer_window = chunk_transfer_window
        return sched


def _make_worker(*,
                 tp_rank: int = 0,
                 tp_size: int = 1,
                 is_producer: bool = True,
                 dp_rank: int = 0,
                 kv_ips: Any = "127.0.0.1",
                 kv_ports: Any = 9100) -> TPUConnectorWorker:
    """Construct a TPUConnectorWorker with all I/O and device calls mocked.

    Patches are applied only during __init__; the returned object has real
    Python state (locks, queues, dicts) but no live ZMQ or TPU handles.
    register_runner() is intentionally NOT called — tests that need it
    should mock _coord_setup and call it separately.
    """
    cfg = _make_vllm_config(is_producer=is_producer, dp_rank=dp_rank)
    with patch(f"{_BASE}.get_tensor_model_parallel_rank", return_value=tp_rank), \
         patch(f"{_BASE}.get_tensor_model_parallel_world_size", return_value=tp_size), \
         patch(f"{_BASE}.dist_utils.get_node_id", return_value=0), \
         patch(f"{_BASE}.dist_utils.get_host_ip", return_value=kv_ips), \
         patch(f"{_BASE}.dist_utils.get_kv_transfer_port", return_value=kv_ports), \
         patch(f"{_BASE}.dist_utils.get_side_channel_port", return_value="9600"), \
         patch(f"{_BASE}.dist_utils.get_transfer_channel_number", return_value=0), \
         patch(f"{_BASE}.dist_utils.get_kv_latency_log_interval", return_value=0.0), \
         patch(f"{_BASE}.zmq.Context"):
        return TPUConnectorWorker(cfg)


def _make_raiden_worker(*,
                        tp_rank: int = 1,
                        tp_size: int = 4,
                        is_producer: bool = True,
                        dp_rank: int = 0,
                        dp_size: int = 1,
                        pcp_size: int = 1,
                        interleave_size: int = 1,
                        block_size: int = 16,
                        kv_ips: Any = "127.0.0.1",
                        kv_ports: Any = 9100) -> TPURaidenConnectorWorker:
    cfg = _make_vllm_config(is_producer=is_producer,
                            block_size=block_size,
                            dp_rank=dp_rank,
                            dp_size=dp_size,
                            tp_size=tp_size,
                            pcp_size=pcp_size,
                            interleave_size=interleave_size)
    with patch(f"{_MOD}.get_tensor_model_parallel_rank", return_value=tp_rank), \
         patch(f"{_MOD}.get_tensor_model_parallel_world_size", return_value=tp_size), \
         patch(f"{_MOD}.dist_utils.get_node_id", return_value=0), \
         patch(f"{_MOD}.dist_utils.get_host_ip", return_value=kv_ips), \
         patch(f"{_MOD}.dist_utils.get_kv_transfer_port", return_value=kv_ports):
        return TPURaidenConnectorWorker(cfg)


# ---------------------------------------------------------------------------
# TestTPUConnector — role initialisation and method delegation
# ---------------------------------------------------------------------------


class TestTPUConnector:

    @patch(f"{_MOD}.TPUConnectorWorker")
    @patch(f"{_MOD}.TPUConnectorScheduler")
    def test_init_scheduler_role(self, mock_sched_cls, mock_worker_cls):
        cfg = _make_vllm_config()
        connector = TPUConnector(cfg, KVConnectorRole.SCHEDULER,
                                 _make_test_kv_cache_config())
        mock_sched_cls.assert_called_once_with(cfg)
        mock_worker_cls.assert_not_called()
        assert connector.connector_scheduler is not None
        assert connector.connector_worker is None

    @patch(f"{_MOD}.TPUConnectorWorker")
    @patch(f"{_MOD}.TPUConnectorScheduler")
    def test_init_worker_role(self, mock_sched_cls, mock_worker_cls):
        cfg = _make_vllm_config()
        connector = TPUConnector(cfg, KVConnectorRole.WORKER,
                                 _make_test_kv_cache_config())
        mock_worker_cls.assert_called_once_with(cfg)
        mock_sched_cls.assert_not_called()
        assert connector.connector_scheduler is None
        assert connector.connector_worker is not None

    @patch(f"{_MOD}.TPUConnectorWorker")
    @patch(f"{_MOD}.TPUConnectorScheduler")
    def test_scheduler_method_delegation(self, mock_sched_cls,
                                         mock_worker_cls):
        cfg = _make_vllm_config()
        connector = TPUConnector(cfg, KVConnectorRole.SCHEDULER,
                                 _make_test_kv_cache_config())
        sched = mock_sched_cls.return_value
        req, blocks, sched_out = MagicMock(), MagicMock(), MagicMock()

        connector.get_num_new_matched_tokens(req, 16)
        sched.get_num_new_matched_tokens.assert_called_once_with(req, 16)

        connector.update_state_after_alloc(req, blocks, 16)
        sched.update_state_after_alloc.assert_called_once_with(req, blocks, 16)

        # build_connector_meta receives scheduler_output but strips it before
        # delegating — verify the downstream call has no arguments.
        connector.build_connector_meta(sched_out)
        sched.build_connector_meta.assert_called_once_with(sched_out)

        connector.request_finished(req, [1, 2])
        sched.request_finished.assert_called_once_with(req, [1, 2])

        connector.get_finished_count()
        sched.get_finished_count.assert_called_once_with()

    @patch(f"{_MOD}.TPUConnectorWorker")
    @patch(f"{_MOD}.TPUConnectorScheduler")
    def test_worker_method_delegation(self, mock_sched_cls, mock_worker_cls):
        cfg = _make_vllm_config()
        connector = TPUConnector(cfg, KVConnectorRole.WORKER,
                                 _make_test_kv_cache_config())
        worker = mock_worker_cls.return_value
        runner, meta = MagicMock(), TPUConnectorMetadata()

        connector.register_runner(runner)
        worker.register_runner.assert_called_once_with(runner)

        connector._connector_metadata = meta
        connector.start_load_kv(None)
        worker.process_send_load.assert_called_once_with(
            meta, wait_for_completion=False, report_completion=True)

        connector.get_finished(set())
        worker.get_finished.assert_called_once_with(set())

        connector.get_kv_connector_stats()
        worker.get_kv_connector_stats.assert_called_once_with()

    @patch(f"{_MOD}.TPURaidenConnectorWorker")
    @patch(f"{_MOD}.TPURaidenConnectorScheduler")
    @patch(f"{_MOD}.TPUConnectorWorker")
    @patch(f"{_MOD}.TPUConnectorScheduler")
    def test_init_defaults_to_zmq_backend(self, mock_sched_cls,
                                          mock_worker_cls,
                                          mock_raiden_sched_cls,
                                          mock_raiden_worker_cls):
        cfg = _make_vllm_config()
        TPUConnector(cfg, KVConnectorRole.SCHEDULER,
                     _make_test_kv_cache_config())
        TPUConnector(cfg, KVConnectorRole.WORKER, _make_test_kv_cache_config())

        mock_sched_cls.assert_called_once_with(cfg)
        mock_worker_cls.assert_called_once_with(cfg)
        mock_raiden_sched_cls.assert_not_called()
        mock_raiden_worker_cls.assert_not_called()

    @patch(f"{_MOD}.TPURaidenConnectorWorker")
    @patch(f"{_MOD}.TPURaidenConnectorScheduler")
    @patch(f"{_MOD}.TPUConnectorWorker")
    @patch(f"{_MOD}.TPUConnectorScheduler")
    def test_init_uses_raiden_backend_when_flag_enabled(
            self, mock_sched_cls, mock_worker_cls, mock_raiden_sched_cls,
            mock_raiden_worker_cls):
        cfg = _make_vllm_config()
        cfg.kv_transfer_config.kv_connector_extra_config = {
            "use_raiden_connector": True,
        }

        TPUConnector(cfg, KVConnectorRole.SCHEDULER,
                     _make_test_kv_cache_config())
        TPUConnector(cfg, KVConnectorRole.WORKER, _make_test_kv_cache_config())

        mock_raiden_sched_cls.assert_called_once_with(cfg)
        mock_raiden_worker_cls.assert_called_once_with(cfg)
        mock_sched_cls.assert_not_called()
        mock_worker_cls.assert_not_called()

    @patch(f"{_MOD}.TPURaidenConnectorWorker")
    @patch(f"{_MOD}.TPURaidenConnectorScheduler")
    @patch(f"{_MOD}.TPUConnectorWorker")
    @patch(f"{_MOD}.TPUConnectorScheduler")
    def test_stage3_transport_env_selects_raiden_backend(
            self, mock_sched_cls, mock_worker_cls, mock_raiden_sched_cls,
            mock_raiden_worker_cls):
        cfg = _make_vllm_config()

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True):
            TPUConnector(cfg, KVConnectorRole.SCHEDULER,
                         _make_test_kv_cache_config())
            TPUConnector(cfg, KVConnectorRole.WORKER,
                         _make_test_kv_cache_config())

        mock_raiden_sched_cls.assert_called_once_with(cfg)
        mock_raiden_worker_cls.assert_called_once_with(cfg)
        mock_sched_cls.assert_not_called()
        mock_worker_cls.assert_not_called()

    @patch(f"{_MOD}.TPURaidenConnectorWorker")
    @patch(f"{_MOD}.TPURaidenConnectorScheduler")
    def test_stage3_hybrid_wrapper_routes_request_finish_to_fa_group(
            self, mock_raiden_sched_cls, mock_raiden_worker_cls):
        cfg = _make_vllm_config(is_producer=True, pcp_size=8)
        cache_config = _make_stage3_hybrid_kv_cache_config(
            fa_group_index=2, num_speculative_blocks=3)

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True):
            connector = TPUConnector(cfg, KVConnectorRole.SCHEDULER,
                                     cache_config)
            request = MagicMock()
            block_ids = ([10], [20], [30, 31], [40])
            expected = (True, {"uuid": 7})
            mock_raiden_sched_cls.return_value.request_finished.return_value = (
                expected)

            assert connector.request_finished_all_groups(request,
                                                         block_ids) == expected
            assert (
                connector.get_block_ids_with_load_errors_group_index() == 2)

        mock_raiden_sched_cls.return_value.request_finished.assert_called_once_with(
            request, [30, 31], mamba_block_ids=[[10], [20], [40]])
        assert (mock_raiden_sched_cls.return_value.
                _stage3_mamba_num_speculative_blocks == 3)
        mock_raiden_worker_cls.assert_not_called()

    @patch(f"{_MOD}.TPURaidenConnectorWorker")
    @patch(f"{_MOD}.TPURaidenConnectorScheduler")
    @patch(f"{_MOD}.TPUConnectorWorker")
    @patch(f"{_MOD}.TPUConnectorScheduler")
    def test_explicit_raiden_connector_forces_raiden_backend(
            self, mock_sched_cls, mock_worker_cls, mock_raiden_sched_cls,
            mock_raiden_worker_cls):
        cfg = _make_vllm_config()

        TPURaidenConnector(cfg, KVConnectorRole.SCHEDULER,
                           _make_test_kv_cache_config())
        TPURaidenConnector(cfg, KVConnectorRole.WORKER,
                           _make_test_kv_cache_config())

        mock_raiden_sched_cls.assert_called_once_with(cfg)
        mock_raiden_worker_cls.assert_called_once_with(cfg)
        mock_sched_cls.assert_not_called()
        mock_worker_cls.assert_not_called()

    @patch(f"{_MOD}.TPURaidenConnectorWorker")
    @patch(f"{_MOD}.TPURaidenConnectorScheduler")
    def test_raiden_connector_registers_named_kv_caches(
            self, mock_sched_cls, mock_worker_cls):
        cfg = _make_vllm_config()
        connector = TPURaidenConnector(cfg, KVConnectorRole.WORKER,
                                       _make_test_kv_cache_config())
        named = {"model.layers.0.self_attn": object()}

        connector.register_kv_caches(named)

        assert connector.use_raiden
        assert mock_worker_cls.return_value.named_kv_caches is named
        mock_sched_cls.assert_not_called()

    @patch(f"{_MOD}.TPUConnectorWorker")
    @patch(f"{_MOD}.TPUConnectorScheduler")
    def test_request_finished_all_groups_routes_to_flat_in_non_hma(
            self, mock_sched_cls, mock_worker_cls):
        """vLLM routes every SupportsHMA connector through
        request_finished_all_groups. The default (non-HMA) connector must
        translate the single-group tuple back to the flat request_finished."""
        cfg = _make_vllm_config()
        connector = TPUConnector(cfg, KVConnectorRole.SCHEDULER,
                                 _make_test_kv_cache_config())
        sched = mock_sched_cls.return_value
        req = MagicMock()

        connector.request_finished_all_groups(req, ([1, 2, 3], ))
        sched.request_finished.assert_called_once_with(req, [1, 2, 3])
        sched.request_finished_all_groups.assert_not_called()


# ---------------------------------------------------------------------------
# TestTPUConnectorScheduler — arithmetic and state-transition logic
# ---------------------------------------------------------------------------


class TestTPUConnectorScheduler:

    def setup_method(self):
        self.consumer = _make_scheduler(is_producer=False)
        self.producer = _make_scheduler(is_producer=True)

    # ---- get_num_new_matched_tokens ----------------------------------------

    def test_producer_always_returns_zero(self):
        req = MagicMock()
        req.kv_transfer_params = {"uuid": 1}
        n, is_async = self.producer.get_num_new_matched_tokens(req, 0)
        assert n == 0
        assert not is_async

    def test_no_kv_transfer_params_returns_zero(self):
        req = MagicMock()
        req.kv_transfer_params = None
        n, is_async = self.consumer.get_num_new_matched_tokens(req, 0)
        assert n == 0
        assert not is_async

    def test_consumer_partial_cache_hit(self):
        # prompt=35 tokens, block_size=16 → round_down=32
        # num_computed=16  → 32-16=16 tokens to load
        req = MagicMock()
        req.prompt_token_ids = [0] * 35
        req.kv_transfer_params = {"uuid": 1}
        n, is_async = self.consumer.get_num_new_matched_tokens(req, 16)
        assert n == 16
        assert is_async

    def test_consumer_full_cache_hit_returns_zero(self):
        # prompt=31 → round_down=16; computed=32 → max(16-32, 0)=0
        req = MagicMock()
        req.prompt_token_ids = [0] * 31
        req.kv_transfer_params = {"uuid": 1}
        n, is_async = self.consumer.get_num_new_matched_tokens(req, 32)
        assert n == 0
        assert not is_async

    # ---- update_state_after_alloc ------------------------------------------

    def test_update_producer_is_noop(self):
        self.producer.update_state_after_alloc(MagicMock(), MagicMock(), 32)
        assert len(self.producer.reqs_to_load) == 0

    def test_update_no_kv_params_is_noop(self):
        req = MagicMock()
        req.kv_transfer_params = None
        self.consumer.update_state_after_alloc(req, MagicMock(), 32)
        assert len(self.consumer.reqs_to_load) == 0

    def test_update_consumer_external_tokens_populates_full_meta(self):
        req = MagicMock()
        req.request_id = "req-1"
        req.kv_transfer_params = {
            "uuid": 42,
            "remote_block_ids": [10, 11],
            "remote_host": "2.2.2.2",
            "remote_port": 9200,
        }
        blocks = MagicMock()
        blocks.get_block_ids.return_value = [[1, 2]]

        self.consumer.update_state_after_alloc(req, blocks, 32)

        assert "req-1" in self.consumer.reqs_to_load
        meta = self.consumer.reqs_to_load["req-1"]
        assert meta.uuid == 42
        assert meta.local_block_ids == [1, 2]
        assert meta.remote_block_ids == [10, 11]
        assert meta.remote_host == "2.2.2.2"
        assert meta.remote_port == 9200

    def test_update_consumer_zero_external_tokens_sets_nones(self):
        """Cache hit / async drain: both local and remote block ids are None."""
        req = MagicMock()
        req.request_id = "req-2"
        req.kv_transfer_params = {
            "uuid": 99,
            "remote_block_ids": [5, 6],
            "remote_host": "3.3.3.3",
            "remote_port": 9300,
        }

        self.consumer.update_state_after_alloc(req, MagicMock(), 0)

        assert "req-2" in self.consumer.reqs_to_load
        meta = self.consumer.reqs_to_load["req-2"]
        assert meta.uuid == 99
        assert meta.local_block_ids is None
        assert meta.remote_block_ids is None

    # ---- build_connector_meta ----------------------------------------------

    def test_build_meta_drains_reqs_to_send(self):
        self.producer.reqs_to_send = {"r1": "m1"}
        meta = self.producer.build_connector_meta()
        assert meta.reqs_to_send == {"r1": "m1"}
        assert len(self.producer.reqs_to_send) == 0

    def test_build_meta_drains_reqs_to_load(self):
        self.consumer.reqs_to_load = {"r2": "m2"}
        meta = self.consumer.build_connector_meta()
        assert meta.reqs_to_load == {"r2": "m2"}
        assert len(self.consumer.reqs_to_load) == 0

    # ---- request_finished --------------------------------------------------

    def test_consumer_request_finished_is_noop(self):
        delay, params = self.consumer.request_finished(MagicMock(), [])
        assert not delay
        assert params is None

    def test_producer_not_length_capped_returns_no_delay(self):
        req = MagicMock()
        req.status = RequestStatus.RUNNING
        delay, params = self.producer.request_finished(req, [1, 2])
        assert not delay
        assert params is None

    @patch(f"{_MOD}.get_uuid", return_value=777)
    @patch(f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout", return_value=30)
    def test_producer_finished_all_full_blocks(self, _timeout, _uuid):
        req = MagicMock()
        req.request_id = "p-full"
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        req.num_computed_tokens = 32  # 32 % 16 == 0 → all_full=True

        delay, params = self.producer.request_finished(req, [3, 4])

        assert delay
        send = self.producer.reqs_to_send["p-full"]
        assert send.uuid == 777
        assert send.local_block_ids == [3, 4]
        assert params["uuid"] == 777
        assert params["remote_block_ids"] == [3, 4]
        assert params["remote_host"] == "127.0.0.1"
        assert params["remote_port"] == 9100

    @patch(f"{_MOD}.get_uuid", return_value=888)
    @patch(f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout", return_value=30)
    def test_producer_finished_trailing_partial_block_is_dropped(
            self, _timeout, _uuid):
        """Trailing partial block must not be transferred: last id excluded."""
        req = MagicMock()
        req.request_id = "p-partial"
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        req.num_computed_tokens = 33  # 33 % 16 != 0 → last block is partial

        delay, params = self.producer.request_finished(req, [5, 6, 7])

        assert delay
        send = self.producer.reqs_to_send["p-partial"]
        assert send.local_block_ids == [5, 6]  # [7] dropped
        assert params["remote_block_ids"] == [5, 6]

    def test_producer_finished_only_partial_block_returns_no_delay(self):
        """When the sole block is partial, no transfer is triggered."""
        req = MagicMock()
        req.request_id = "p-tiny"
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        req.num_computed_tokens = 5  # 5 % 16 != 0

        delay, params = self.producer.request_finished(req, [8])

        assert not delay
        assert params == {}
        assert "p-tiny" not in self.producer.reqs_to_send

    # ---- get_finished_count ------------------------------------------------

    def test_get_finished_count_single_host(self):
        self.consumer.kv_ip = "1.1.1.1"
        assert self.consumer.get_finished_count() == 1

    def test_get_finished_count_multi_host(self):
        self.consumer.kv_ip = ["1.1.1.1", "2.2.2.2", "3.3.3.3"]
        assert self.consumer.get_finished_count() == 3

    # ---- test DP configurations --------------------------------------------
    def test_dp_port_configurations_singlehost(self):
        scheduler = _make_scheduler(dp_rank=0, tp_size=1)
        assert scheduler.kv_port == 9100
        assert scheduler.side_channel_port == 9600

        scheduler = _make_scheduler(dp_rank=1, tp_size=1)
        assert scheduler.kv_port == 9101
        assert scheduler.side_channel_port == 9601

        scheduler = _make_scheduler(dp_rank=0, tp_size=4)
        assert scheduler.kv_port == 9100
        assert scheduler.side_channel_port == 9600

        scheduler = _make_scheduler(dp_rank=1, tp_size=4)
        assert scheduler.kv_port == 9104
        assert scheduler.side_channel_port == 9601

    def test_dp_port_configurations_multihost(self):
        scheduler = _make_scheduler(dp_rank=0,
                                    tp_size=1,
                                    kv_ports=[9100, 9200])
        assert scheduler.kv_port == [9100, 9200]
        assert scheduler.side_channel_port == 9600

        scheduler = _make_scheduler(dp_rank=1,
                                    tp_size=1,
                                    kv_ports=[9100, 9200])
        assert scheduler.kv_port == [9101, 9201]
        assert scheduler.side_channel_port == 9601

        scheduler = _make_scheduler(dp_rank=0,
                                    tp_size=4,
                                    kv_ports=[9100, 9200])
        assert scheduler.kv_port == [9100, 9200]
        assert scheduler.side_channel_port == 9600

        scheduler = _make_scheduler(dp_rank=1,
                                    tp_size=4,
                                    kv_ports=[9100, 9200])
        assert scheduler.kv_port == [9104, 9204]
        assert scheduler.side_channel_port == 9601


# ---------------------------------------------------------------------------
# TestTPURaidenConnectorScheduler — Raiden-specific scheduler behavior
# ---------------------------------------------------------------------------


class TestTPURaidenConnectorScheduler:

    def setup_method(self):
        self.consumer = _make_raiden_scheduler(is_producer=False)
        self.producer = _make_raiden_scheduler(is_producer=True)

    def test_stage3_mamba_truncation_happens_at_admission(self):
        producer = _make_raiden_scheduler(is_producer=True)
        producer._stage3_mamba_group_indices = [1]
        req = MagicMock()
        req.request_id = "req-mamba-p"
        req.prompt_token_ids = list(range(33))
        req._all_token_ids = list(range(33))
        req.num_prompt_tokens = 33
        req.kv_transfer_params = {"do_remote_decode": True}

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True):
            producer.on_new_request(req)
            assert len(req.prompt_token_ids) == 32
            assert req.num_prompt_tokens == 32
            assert req.kv_transfer_params["_p_side_truncated"] is True

            # Re-admission after preemption must not truncate again.
            producer.on_new_request(req)
            assert len(req.prompt_token_ids) == 32

            # The lookup path runs once the prefix-cache hit is fixed, so
            # it must leave the length alone.
            assert producer.get_num_new_matched_tokens(req, 0) == (0, False)
            assert len(req.prompt_token_ids) == 32

    def test_stage3_truncation_skipped_without_mamba_groups(self):
        producer = _make_raiden_scheduler(is_producer=True)
        req = MagicMock()
        req.request_id = "req-fa-p"
        req.prompt_token_ids = list(range(33))
        req._all_token_ids = list(range(33))
        req.num_prompt_tokens = 33
        req.kv_transfer_params = {"do_remote_decode": True}

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True):
            producer.on_new_request(req)

        assert len(req.prompt_token_ids) == 33
        assert "_p_side_truncated" not in req.kv_transfer_params

    @patch(f"{_MOD}.dist_utils.get_raiden_inline_load", return_value=True)
    def test_inline_full_hit_recomputes_last_token(self, _inline):
        req = MagicMock()
        req.request_id = "req-inline"
        req.prompt_token_ids = [0] * 32
        req.kv_transfer_params = {"uuid": 1}

        n, is_async = self.consumer.get_num_new_matched_tokens(req, 0)

        assert n == 31
        assert not is_async

    @patch(f"{_MOD}.dist_utils.get_raiden_inline_load", return_value=True)
    def test_inline_partial_hit_stays_block_aligned(self, _inline):
        req = MagicMock()
        req.request_id = "req-inline-partial"
        req.prompt_token_ids = [0] * 35
        req.kv_transfer_params = {"uuid": 1}

        n, is_async = self.consumer.get_num_new_matched_tokens(req, 0)

        assert n == 32
        assert not is_async

    def test_update_consumer_uses_unhashed_blocks(self):
        req = MagicMock()
        req.request_id = "req-1"
        req.kv_transfer_params = {
            "uuid": 42,
            "remote_block_ids": [10, 11],
            "remote_host": "2.2.2.2",
            "remote_port": 9200,
        }
        blocks = MagicMock()
        blocks.get_block_ids.return_value = ([1, 2], )

        self.consumer.update_state_after_alloc(req, blocks, 32)

        meta = self.consumer.reqs_to_load["req-1"]
        assert meta.uuid == 42
        assert meta.local_block_ids == [1, 2]
        assert meta.remote_block_ids == [10, 11]
        assert meta.remote_host == "2.2.2.2"
        assert meta.remote_port == 9200
        blocks.get_unhashed_block_ids.assert_not_called()

    def test_update_consumer_partial_prefix_hit_loads_only_suffix(self):
        req = MagicMock()
        req.request_id = "req-prefix"
        req.kv_transfer_params = {
            "uuid": 43,
            "remote_block_ids": [10, 11, 12, 13],
            "remote_host": "2.2.2.2",
            "remote_port": 9200,
        }
        blocks = MagicMock()
        blocks.get_block_ids.return_value = ([4], )

        self.consumer.update_state_after_alloc(req, blocks, 16)

        meta = self.consumer.reqs_to_load["req-prefix"]
        assert meta.local_block_ids == [4]
        assert meta.remote_block_ids == [13]

    def test_update_consumer_empty_unhashed_blocks_releases_remote_send(self):
        req = MagicMock()
        req.request_id = "req-hit"
        req.kv_transfer_params = {
            "uuid": 44,
            "remote_block_ids": [10, 11],
            "remote_host": "2.2.2.2",
            "remote_port": 9200,
        }
        blocks = MagicMock()
        blocks.get_block_ids.return_value = ([], )

        self.consumer.update_state_after_alloc(req, blocks, 16)

        meta = self.consumer.reqs_to_load["req-hit"]
        assert meta.local_block_ids is None
        assert meta.remote_block_ids is None
        # The request IS waiting on this connector (num_external_tokens > 0),
        # so the empty release read must still report finished_recving.
        assert meta.report_completion is True

    def test_update_consumer_zero_external_tokens_suppresses_report(self):
        # num_external_tokens == 0: nothing is pulled through this connector
        # (full local cache hit, or another MultiConnector child owns the
        # load). The release read must still be enrolled, but its completion
        # must not surface as finished_recving.
        req = MagicMock()
        req.request_id = "req-release"
        req.kv_transfer_params = {
            "uuid": 45,
            "remote_block_ids": [10, 11],
            "remote_host": "2.2.2.2",
            "remote_port": 9200,
        }
        blocks = MagicMock()
        blocks.get_block_ids.return_value = ([4], )

        self.consumer.update_state_after_alloc(req, blocks, 0)

        meta = self.consumer.reqs_to_load["req-release"]
        assert meta.local_block_ids is None
        assert meta.remote_block_ids is None
        assert meta.report_completion is False

    def test_get_finished_count_uses_vllm_world_size(self):
        assert self.consumer.get_finished_count() == 0

    def test_stage3_producer_transfers_committed_mamba_checkpoint(self):
        producer = _make_raiden_scheduler(is_producer=True,
                                          block_size=16,
                                          pcp_size=1)
        producer._stage3_mamba_group_indices = [1]
        producer._stage3_mamba_num_speculative_blocks = 3
        req = MagicMock()
        req.request_id = "mtp-producer"
        req.num_prompt_tokens = 17
        req.num_computed_tokens = 16
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        req.kv_transfer_params = None

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), patch(
                       f"{_MOD}.tpu_envs.TPU_RAIDEN_CONTROLLER_ADDRESS",
                       "prefill-controller.test:27000",
                       create=True), patch(
                           f"{_MOD}.tpu_envs.TPU_RAIDEN_JOB_NAME",
                           "prefill",
                           create=True), patch(
                               f"{_MOD}.tpu_envs.TPU_RAIDEN_ENGINE_ID",
                               "producer-engine",
                               create=True), patch(
                                   f"{_MOD}.tpu_envs."
                                   "TPU_RAIDEN_TRANSFER_PARALLELISM",
                                   1,
                                   create=True), patch(
                                       f"{_MOD}.get_uuid",
                                       return_value=777), patch(
                                           f"{_MOD}.dist_utils."
                                           "get_p2p_wait_pull_timeout",
                                           return_value=30.0):
            delay, _ = producer.request_finished(
                req,
                [100],
                mamba_block_ids=[[10, 11, 12, 13, 14]],
            )
            meta = producer.build_connector_meta()

        assert delay
        assert meta.reqs_to_send[req.request_id].mamba_state_block_ids == [11]

    @pytest.mark.parametrize("num_computed_tokens", (65_023, 65_024))
    def test_v3_stage3_finish_keeps_partial_page_and_exact_token_count(
            self, num_computed_tokens):
        producer = _make_raiden_scheduler(is_producer=True,
                                          block_size=4096,
                                          pcp_size=8)
        req = MagicMock()
        req.request_id = f"tail-{num_computed_tokens}"
        # num_computed_tokens is scheduler-owned transfer extent. It may be
        # prompt_tokens - 1 or the full prompt under the one-token prefill
        # proxy; the wire extent excludes the final prompt token either way.
        req.prompt_token_ids = [0] * 65_024
        req.num_prompt_tokens = 65_024
        req.num_computed_tokens = num_computed_tokens
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        # Under PCP8 each scheduler block represents 8 physical 4096-token
        # pages. Every PCP rank sees this same two-ID block-table prefix.
        block_ids = [100, 101]

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), patch(
                       f"{_MOD}.tpu_envs.TPU_RAIDEN_CONTROLLER_ADDRESS",
                       "prefill-controller.test:27000",
                       create=True), patch(
                           f"{_MOD}.tpu_envs.TPU_RAIDEN_JOB_NAME",
                           "custom-prefill-job",
                           create=True), patch(
                               f"{_MOD}.tpu_envs.TPU_RAIDEN_ENGINE_ID",
                               "producer-engine-9",
                               create=True), patch(
                                   f"{_MOD}.tpu_envs."
                                   "TPU_RAIDEN_TRANSFER_PARALLELISM",
                                   8,
                                   create=True), patch(
                                       f"{_MOD}.get_uuid",
                                       return_value=777), patch(
                                           f"{_MOD}.dist_utils."
                                           "get_p2p_wait_pull_timeout",
                                           return_value=30.0):
            delay, params = producer.request_finished(req, block_ids)
            first_meta = producer.build_connector_meta()
            duplicate_delay, duplicate_params = producer.request_finished(
                req, block_ids)
            duplicate_meta = producer.build_connector_meta()

        assert delay and duplicate_delay
        assert params == duplicate_params == {
            "req_id": req.request_id,
            "uuid": 777,
            "num_tokens": 65_023,
            "src_controller_address": "prefill-controller.test:27000",
            "src_job_name": "custom-prefill-job",
            "src_engine_id": "producer-engine-9",
            "src_data_replica_idx": 0,
            "src_parallelism": 8,
        }
        assert "remote_block_ids" not in params
        assert first_meta.reqs_to_send[req.request_id].local_block_ids == (
            block_ids)
        assert first_meta.reqs_to_send[req.request_id].num_tokens == (65_023)
        assert duplicate_meta.reqs_to_send == {}

    @pytest.mark.parametrize("num_computed_tokens", [32_775, 32_768])
    def test_v3_stage3_finish_trims_blocks_past_the_transfer_prefix(
            self, num_computed_tokens):
        """Blocks opened by tokens kept on the producer are trimmed."""
        producer = _make_raiden_scheduler(is_producer=True,
                                          block_size=4096,
                                          pcp_size=8)
        req = MagicMock()
        req.request_id = f"trim-{num_computed_tokens}"
        req.prompt_token_ids = [0] * 32_768
        req.num_prompt_tokens = 32_768
        req.num_computed_tokens = num_computed_tokens
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        # Generation crossed the 32768-token scheduler block, so the producer
        # table carries a second ID that is outside the transfer prefix.
        block_ids = [100, 101]

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), patch(
                       f"{_MOD}.tpu_envs.TPU_RAIDEN_CONTROLLER_ADDRESS",
                       "prefill-controller.test:27000",
                       create=True), patch(f"{_MOD}.get_uuid",
                                           return_value=778), patch(
                                               f"{_MOD}.dist_utils."
                                               "get_p2p_wait_pull_timeout",
                                               return_value=30.0):
            delay, params = producer.request_finished(req, block_ids)
            meta = producer.build_connector_meta()

        assert delay
        assert params["num_tokens"] == 32_767
        # Only the block covering the transferred prefix is published.
        assert list(meta.reqs_to_send[req.request_id].local_block_ids) == [100]

    def test_v3_stage3_finish_rejects_exceeding_max_transfer_tokens(self):
        """Tokens exceeding TPU_RAIDEN_MAX_TRANSFER_TOKENS must bypass transfer and free blocks."""
        producer = _make_raiden_scheduler(is_producer=True,
                                          block_size=4096,
                                          pcp_size=8)
        req = MagicMock()
        req.request_id = "too-long"
        req.prompt_token_ids = [0] * 10_000
        req.num_prompt_tokens = 10_000
        req.num_computed_tokens = 10_000
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        block_ids = [100]

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), patch(
                       f"{_MOD}.tpu_envs.TPU_RAIDEN_CONTROLLER_ADDRESS",
                       "prefill-controller.test:27000",
                       create=True), patch(
                           f"{_MOD}.tpu_envs.TPU_RAIDEN_MAX_TRANSFER_TOKENS",
                           8192,
                           create=True):
            delay, params = producer.request_finished(req, block_ids)
            meta = producer.build_connector_meta()

        assert not delay
        assert params == {}
        assert req.request_id not in meta.reqs_to_send

    def test_v3_stage3_finish_still_rejects_too_few_blocks(self):
        """Trimming the tail must not mask a genuinely short block table."""
        producer = _make_raiden_scheduler(is_producer=True,
                                          block_size=4096,
                                          pcp_size=8)
        req = MagicMock()
        req.request_id = "short"
        req.prompt_token_ids = [0] * 65_536
        req.num_prompt_tokens = 65_536
        req.num_computed_tokens = 65_536
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        # 65535 transferred tokens need 2 scheduler blocks; only one given.
        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), patch(
                       f"{_MOD}.tpu_envs.TPU_RAIDEN_CONTROLLER_ADDRESS",
                       "prefill-controller.test:27000",
                       create=True):
            with pytest.raises(ValueError, match="must cover every PCP"):
                producer.request_finished(req, [100])

    def test_v3_stage3_finish_dedup_refuses_unsafe_inflight_eviction(self):
        producer = _make_raiden_scheduler(is_producer=True,
                                          block_size=4096,
                                          pcp_size=8)
        requests = []
        for index in range(3):
            req = MagicMock()
            req.request_id = f"bounded-{index}"
            req.num_prompt_tokens = 32_769
            req.num_computed_tokens = 32_768
            req.status = RequestStatus.FINISHED_LENGTH_CAPPED
            requests.append(req)

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), patch(
                       f"{_MOD}.tpu_envs.TPU_RAIDEN_CONTROLLER_ADDRESS",
                       "prefill-controller.test:27000",
                       create=True), patch(
                           f"{_MOD}.tpu_envs.TPU_RAIDEN_JOB_NAME",
                           "prefill",
                           create=True), patch(
                               f"{_MOD}.tpu_envs.TPU_RAIDEN_ENGINE_ID",
                               "0",
                               create=True), patch(
                                   f"{_MOD}.tpu_envs."
                                   "TPU_RAIDEN_TRANSFER_PARALLELISM",
                                   8,
                                   create=True), patch(
                                       f"{_MOD}.get_uuid",
                                       side_effect=(11, 12, 13)), patch(
                                           f"{_MOD}.dist_utils."
                                           "get_p2p_wait_pull_timeout",
                                           return_value=30.0), patch(
                                               f"{_MOD}."
                                               "_STAGE3_FINISH_DEDUP_LIMIT",
                                               2):
            producer.request_finished(requests[0], [0])
            producer.request_finished(requests[1], [0])
            with pytest.raises(RuntimeError, match="capacity exhausted"):
                producer.request_finished(requests[2], [0])
            producer.update_connector_output(
                SimpleNamespace(finished_sending={"bounded-0"}))
            producer.request_finished(requests[2], [0])

        assert list(
            producer._stage3_finished_sends) == ["bounded-1", "bounded-2"]

    @pytest.mark.parametrize("num_computed_tokens", (32_768, 32_769))
    def test_stage3_finish_trims_last_token_only_scheduler_block(
            self, num_computed_tokens):
        producer = _make_raiden_scheduler(is_producer=True,
                                          block_size=4096,
                                          pcp_size=8)
        req = MagicMock()
        req.request_id = "boundary-tail"
        req.num_prompt_tokens = 32_769
        req.num_computed_tokens = num_computed_tokens
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), patch(
                       f"{_MOD}.tpu_envs.TPU_RAIDEN_CONTROLLER_ADDRESS",
                       "prefill-controller.test:27000",
                       create=True), patch(
                           f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM",
                           8,
                           create=True), patch(f"{_MOD}.get_uuid",
                                               return_value=778), patch(
                                                   f"{_MOD}.dist_utils."
                                                   "get_p2p_wait_pull_timeout",
                                                   return_value=30.0):
            delay, params = producer.request_finished(req, [100, 101])
            meta = producer.build_connector_meta()

        assert delay
        assert params["num_tokens"] == 32_768
        assert meta.reqs_to_send["boundary-tail"].local_block_ids == [100]
        assert meta.reqs_to_send["boundary-tail"].num_tokens == 32_768

    def test_stage3_completion_aggregation_is_role_aware(self):
        producer = _make_raiden_scheduler(is_producer=True, pcp_size=8)
        consumer = _make_raiden_scheduler(is_producer=False, pcp_size=1)

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True):
            assert producer.get_finished_count() == 8
            assert consumer.get_finished_count() == 1

    def test_v4_stage3_consumer_uses_declared_transfer_extent(self):
        consumer = _make_raiden_scheduler(is_producer=False, block_size=1024)
        req = MagicMock()
        req.request_id = "tail-load"
        req.prompt_token_ids = [0] * 65_024
        req.kv_transfer_params = {
            "req_id": "tail-load",
            "uuid": 991,
            "num_tokens": 65_023,
            "src_controller_address": "prefill-controller.test:27000",
            "src_job_name": "nondefault-prefill",
            "src_engine_id": "producer-engine",
            "src_data_replica_idx": 0,
            "src_parallelism": 8,
        }
        blocks = MagicMock()
        blocks.get_block_ids.return_value = (list(range(200, 264)), )

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), patch(
                       f"{_MOD}.dist_utils.get_raiden_inline_load",
                       return_value=False):
            matched, is_async = consumer.get_num_new_matched_tokens(req, 0)
            consumer.update_state_after_alloc(req, blocks, matched)

        assert matched == 65_023
        assert is_async
        load = consumer.reqs_to_load["tail-load"]
        assert load.num_tokens == 65_023
        assert load.local_block_ids == list(range(200, 264))
        assert load.src_job_name == "nondefault-prefill"
        assert load.src_engine_id == "producer-engine"
        assert not hasattr(load, "remote_block_ids")

    def test_stage3_consumer_writes_committed_mamba_checkpoint(self):
        consumer = _make_raiden_scheduler(is_producer=False, block_size=16)
        consumer._stage3_fa_group_index = 0
        consumer._stage3_mamba_group_indices = [1]
        consumer._stage3_mamba_num_speculative_blocks = 3
        req = MagicMock()
        req.request_id = "mtp-consumer"
        req.kv_transfer_params = {
            "req_id": "mtp-producer",
            "uuid": 991,
            "num_tokens": 16,
            "src_controller_address": "prefill-controller.test:27000",
            "src_job_name": "prefill",
            "src_engine_id": "producer-engine",
            "src_data_replica_idx": 0,
            "src_parallelism": 1,
        }
        blocks = MagicMock()
        blocks.get_block_ids.return_value = ([200], [10, 11, 12, 13, 14])

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True):
            consumer.update_state_after_alloc(req, blocks, 16)

        load = consumer.reqs_to_load[req.request_id]
        assert load.mamba_state_block_ids == [11]

    def test_stage3_consumer_prefix_hit_pulls_suffix_pages_only(self):
        req = MagicMock()
        req.request_id = "suffix-load"
        req.num_computed_tokens = 0
        req.prompt_token_ids = [0] * 4096
        req.kv_transfer_params = {
            "req_id": "suffix-load-src",
            "uuid": 993,
            "num_tokens": 4095,
            "src_controller_address": "prefill-controller.test:27000",
            "src_job_name": "prefill",
            "src_engine_id": "producer-engine",
            "src_data_replica_idx": 0,
            "src_parallelism": 8,
        }
        blocks = MagicMock()
        blocks.get_block_ids.return_value = ([50, 51, 52, 53], )

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), \
             patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_PREFIX_AWARE_LOAD",
                   True,
                   create=True), \
             patch(f"{_MOD}._prefix_aware_load_supported",
                   return_value=True), \
             patch(f"{_MOD}.dist_utils.get_raiden_inline_load",
                   return_value=False):
            consumer = _make_raiden_scheduler(is_producer=False,
                                              block_size=1024,
                                              enable_prefix_caching=True)
            matched, is_async = consumer.get_num_new_matched_tokens(req, 2048)
            consumer.update_state_after_alloc(req, blocks, matched)

        assert matched == 2047
        assert is_async
        load = consumer.reqs_to_load["suffix-load"]
        assert load.skip_tokens == 2048
        assert load.local_block_ids == [52, 53]
        assert load.num_tokens == 4095
        assert not load.release_only

    @pytest.mark.parametrize(("requested", "supported"),
                             ((False, True), (True, False)))
    def test_stage3_consumer_partial_hit_falls_back_to_local_compute(
            self, requested, supported):
        req = MagicMock()
        req.request_id = "local-suffix-fallback"
        # vLLM assigns the admitted local hit only after the connector update;
        # zero here makes the unused producer registration release promptly.
        req.num_computed_tokens = 0
        req.prompt_token_ids = [0] * 4096
        req.kv_transfer_params = {
            "req_id": "local-suffix-fallback-src",
            "uuid": 994,
            "num_tokens": 4095,
            "src_controller_address": "prefill-controller.test:27000",
            "src_job_name": "prefill",
            "src_engine_id": "producer-engine",
            "src_data_replica_idx": 0,
            "src_parallelism": 8,
        }
        blocks = MagicMock()

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), \
             patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_PREFIX_AWARE_LOAD",
                   requested,
                   create=True), \
             patch(f"{_MOD}._prefix_aware_load_supported",
                   return_value=supported), \
             patch(f"{_MOD}.dist_utils.get_raiden_inline_load",
                   return_value=False):
            consumer = _make_raiden_scheduler(is_producer=False,
                                              block_size=1024,
                                              enable_prefix_caching=True)
            matched, is_async = consumer.get_num_new_matched_tokens(req, 2048)
            consumer.update_state_after_alloc(req, blocks, matched)

        assert matched == 0
        assert not is_async
        blocks.get_block_ids.assert_not_called()
        release = consumer.reqs_to_load[req.request_id]
        assert release.release_only
        assert release.source_req_id == "local-suffix-fallback-src"
        assert release.local_block_ids == []

    def test_stage3_consumer_full_hit_enqueues_release_only(self):
        consumer = _make_raiden_scheduler(is_producer=False, block_size=1024)
        req = MagicMock()
        req.request_id = "full-hit"
        req.num_computed_tokens = 0
        req.prompt_token_ids = [0] * 2048
        req.kv_transfer_params = {
            "req_id": "full-hit-src",
            "uuid": 995,
            "num_tokens": 2047,
            "src_controller_address": "prefill-controller.test:27000",
            "src_job_name": "prefill",
            "src_engine_id": "producer-engine",
            "src_data_replica_idx": 0,
            "src_parallelism": 8,
        }
        blocks = MagicMock()

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True):
            matched, is_async = consumer.get_num_new_matched_tokens(req, 2048)
            consumer.update_state_after_alloc(req, blocks, matched)

        assert matched == 0
        assert not is_async
        load = consumer.reqs_to_load["full-hit"]
        assert load.release_only
        assert load.local_block_ids == []
        assert load.uuid == 995
        assert load.source_req_id == "full-hit-src"

    def test_stage3_consumer_post_load_resume_does_not_release(self):
        consumer = _make_raiden_scheduler(is_producer=False, block_size=1024)
        req = MagicMock()
        req.request_id = "resumed"
        # The async-load completion set num_computed_tokens before this
        # re-admission; a release here would spuriously cancel the claimed
        # producer registration on every resumed request.
        req.num_computed_tokens = 2047
        req.kv_transfer_params = {
            "req_id": "resumed-src",
            "uuid": 996,
            "num_tokens": 2047,
            "src_controller_address": "prefill-controller.test:27000",
        }
        blocks = MagicMock()

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True):
            consumer.update_state_after_alloc(req, blocks, 0)

        assert consumer.reqs_to_load == {}

    def test_stage3_consumer_preserves_distinct_source_request_id(self):
        consumer = _make_raiden_scheduler(is_producer=False, block_size=1024)
        req = MagicMock()
        req.request_id = "proxy-id-decode5678"
        req.kv_transfer_params = {
            "req_id": "proxy-id-prefill1234",
            "uuid": 992,
            "num_tokens": 1536,
            "src_controller_address": "prefill-controller.test:27000",
            "src_job_name": "prefill",
            "src_engine_id": "producer-engine",
            "src_data_replica_idx": 0,
            "src_parallelism": 8,
        }
        blocks = MagicMock()
        blocks.get_block_ids.return_value = ([200, 201], )

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True):
            consumer.update_state_after_alloc(req, blocks, 1536)

        load = consumer.reqs_to_load["proxy-id-decode5678"]
        assert load.source_req_id == "proxy-id-prefill1234"
        assert load.local_block_ids == [200, 201]


# ---- test DP configurations --------------------------------------------

    def test_dp_port_configurations_singlehost(self):
        scheduler = _make_raiden_scheduler(dp_rank=0, tp_size=1)
        assert scheduler.kv_port == 9100

        scheduler = _make_raiden_scheduler(dp_rank=1, tp_size=1)
        assert scheduler.kv_port == 9102

        scheduler = _make_raiden_scheduler(dp_rank=0, tp_size=4)
        assert scheduler.kv_port == 9100

        scheduler = _make_raiden_scheduler(dp_rank=1, tp_size=4)
        assert scheduler.kv_port == 9108

    def test_dp_port_configurations_multihost(self):
        scheduler = _make_raiden_scheduler(dp_rank=0,
                                           tp_size=1,
                                           kv_ports=[9100, 9200])
        assert scheduler.kv_port == [9100, 9200]

        scheduler = _make_raiden_scheduler(dp_rank=1,
                                           tp_size=1,
                                           kv_ports=[9100, 9200])
        assert scheduler.kv_port == [9102, 9202]

        scheduler = _make_raiden_scheduler(dp_rank=0,
                                           tp_size=4,
                                           kv_ports=[9100, 9200])
        assert scheduler.kv_port == [9100, 9200]

        scheduler = _make_raiden_scheduler(dp_rank=1,
                                           tp_size=4,
                                           kv_ports=[9100, 9200])
        assert scheduler.kv_port == [9108, 9208]


class _FakeRaidenEngine:

    def __init__(self):
        self.calls = []
        self.poll_results = [(["sent"], ["recv"], [])]

    def register_read(self, req_id, uuid, block_ids):
        self.calls.append(("register_read", req_id, uuid, block_ids))
        return 1

    def start_read(self, req_id, uuid, endpoint, remote_blocks, local_blocks):
        self.calls.append(("start_read", req_id, uuid, endpoint, remote_blocks,
                           local_blocks))
        return 2

    def poll_stats(self):
        self.calls.append(("poll_stats", ))
        if self.poll_results:
            return self.poll_results.pop(0)
        return [], [], []


class _FakeAdmissionRaidenEngine:

    def __init__(self,
                 *,
                 transfer_address="10.20.0.1:24001",
                 listener_address="10.20.0.1:24002"):
        self.registered_pools = []
        self.transfer_address = transfer_address
        self.listener_address = listener_address

    def register_pools(self, pools):
        self.registered_pools = list(pools)
        storage_indices = {
            pool["storage_index"]
            for pool in self.registered_pools
        }
        return {
            "admitted": True,
            "pools": len(self.registered_pools),
            "storages": len(storage_indices),
        }


class _FakeRaidenControllerFacade:

    def __init__(self):
        self.register_work_unit_calls = []
        self.register_request_blocks_calls = []
        self.complete_request_blocks_calls = []
        self.cancel_request_blocks_calls = []
        self.start_transfer_calls = []
        self.start_transfer_result = True
        self.cancel_request_blocks_result = True

    def register_work_unit(self, **kwargs):
        self.register_work_unit_calls.append(kwargs)

    def register_request_blocks(self, **kwargs):
        self.register_request_blocks_calls.append(kwargs)

    def complete_request_blocks(self, **kwargs):
        self.complete_request_blocks_calls.append(kwargs)

    def cancel_request_blocks_if_unclaimed(self, **kwargs):
        self.cancel_request_blocks_calls.append(kwargs)
        return self.cancel_request_blocks_result

    def start_transfer(self, **kwargs):
        self.start_transfer_calls.append(kwargs)
        return self.start_transfer_result


def _seed_stateful_stage3_producer(worker,
                                   facade,
                                   *,
                                   req_id="stateful",
                                   uuid=123):
    """Registers one tiny conv/SSM sibling pair plus its scheduler base."""
    from vllm_torchtpu.distributed.kv_transfer.raiden.pool_manifest import (  # noqa: E501
        BINDING_ALIASED_RAW, PoolEntry, PoolManifest, RegionSpec)

    worker._raiden_manifest = PoolManifest(
        binding=BINDING_ALIASED_RAW,
        storages=[],
        pools=[
            PoolEntry(
                "gdn.conv.g0",
                "linear.0",
                0,
                64,
                4096,
                32,
                (
                    RegionSpec("gdn_conv_q", 0, 32, 4, 3, 2),
                    RegionSpec("gdn_conv_k", 8, 32, 4, 3, 2),
                    RegionSpec("gdn_conv_v", 16, 32, 4, 3, 4),
                ),
                "bfloat16",
            ),
            PoolEntry(
                "gdn.ssm.g0",
                "linear.0",
                0,
                0,
                4096,
                32,
                (RegionSpec("gdn_ssm", 0, 16, 16, 4), ),
                "float32",
            ),
        ],
    )
    worker._stage3_state_group_count = 1
    worker._stage3_register_state_blocks(facade, req_id, uuid, 0, [17])
    worker._stage3_registered_sends[req_id] = _Stage3RegisteredSend(
        uuid=uuid,
        local_block_ids=(11, ),
        num_tokens=1,
        expiration_time=1e20,
    )
    return {
        call["req_id"]: call["uuid"]
        for call in facade.register_request_blocks_calls
    }


def _synthetic_qwen35_materialization(*, tp_size: int, pcp_size: int = 1):
    """Builds the 15-FA/45-GDN Qwen3.5 live-cache shape at tiny scale."""
    key_heads = 16 // tp_size // pcp_size
    value_heads = 32 // tp_size // pcp_size
    conv_dim = 2 * key_heads * 2 + value_heads * 2
    fa_kv_heads = max(2 // tp_size, 1)

    fa_layers = tuple(range(3, 60, 4))
    named = {}
    for idx in range(60):
        if idx in fa_layers:
            named[f"model.layers.{idx}.self_attn.attn"] = torch.empty(
                (4, 8, 1, 4, 4), dtype=torch.uint8)
        else:
            named[f"model.layers.{idx}.linear_attn"] = (
                torch.empty((4, 3, conv_dim), dtype=torch.bfloat16),
                torch.empty((4, value_heads, 2, 2), dtype=torch.float32),
            )

    fa_names = [name for name in named if "self_attn" in name]
    gdn_names = [name for name in named if "linear_attn" in name]
    groups = (
        SimpleNamespace(
            layer_names=tuple(fa_names),
            kv_cache_spec=SimpleNamespace(block_size=8,
                                          num_kv_heads=fa_kv_heads,
                                          head_size=4),
        ),
        SimpleNamespace(layer_names=tuple(gdn_names[:15]),
                        kv_cache_spec=SimpleNamespace()),
        SimpleNamespace(layer_names=tuple(gdn_names[15:30]),
                        kv_cache_spec=SimpleNamespace()),
        SimpleNamespace(layer_names=tuple(gdn_names[30:]),
                        kv_cache_spec=SimpleNamespace()),
    )
    return named, groups


def _synthetic_qwen35_unified_pool_materialization(*,
                                                   tp_size: int,
                                                   pcp_size: int = 1):
    """PR #106 materialization: 105 logical caches over 15 raw pools."""
    key_heads = 16 // tp_size // pcp_size
    value_heads = 32 // tp_size // pcp_size
    conv_dim = 2 * key_heads * 2 + value_heads * 2
    fa_kv_heads = max(2 // tp_size, 1)
    manager_page_bytes = 4096
    manager_block_tokens = 256

    fa_layers = tuple(range(3, 60, 4))
    fa_names = [f"model.layers.{idx}.self_attn.attn" for idx in fa_layers]
    gdn_names = [
        f"model.layers.{idx}.linear_attn" for idx in range(60)
        if idx not in fa_layers
    ]
    raw_tensors = [
        torch.empty((4, manager_block_tokens, 1, 4, 4), dtype=torch.uint8)
        for _ in fa_names
    ]
    named = {}
    for pool_index, pool in enumerate(raw_tensors):
        named[fa_names[pool_index]] = pool
        for group_ordinal in range(3):
            gdn_name = gdn_names[group_ordinal * len(fa_names) + pool_index]
            named[gdn_name] = [pool]

    fa_spec = FullAttentionSpec(block_size=manager_block_tokens,
                                num_kv_heads=fa_kv_heads,
                                head_size=4,
                                dtype=torch.uint8,
                                page_size_padded=manager_page_bytes)
    mamba_specs = [
        MambaSpec(block_size=manager_block_tokens,
                  shapes=((3, conv_dim), (value_heads, 2, 2)),
                  dtypes=(torch.bfloat16, torch.float32),
                  page_size_padded=manager_page_bytes) for _ in range(3)
    ]
    groups = (
        KVCacheGroupSpec(layer_names=fa_names, kv_cache_spec=fa_spec),
        *(KVCacheGroupSpec(
            layer_names=gdn_names[index * len(fa_names):(index + 1) *
                                  len(fa_names)],
            kv_cache_spec=mamba_specs[index]) for index in range(3)),
    )
    return named, groups, raw_tensors


class TestTPURaidenConnectorWorker:

    @pytest.fixture(autouse=True)
    def _byte_lowering_defaults(self):
        # Every producer worker gets the measured FA token bytes required by
        # registration-time lowering. (T3.4: destination page geometry is
        # controller-derived; no producer env exists.)
        with patch.object(TPURaidenConnectorWorker,
                          "_stage3_fa_token_bytes",
                          return_value=1024):
            yield

    def setup_method(self):
        self.worker = _make_raiden_worker(is_producer=True)
        self.engine = _FakeRaidenEngine()
        self.worker._raiden_transfer_engine = self.engine

    def test_stage3_request_id_binding_is_bijective(self):
        worker = _make_raiden_worker(is_producer=False)
        worker._bind_stage3_request_ids("decode-a", "prefill-a")

        with pytest.raises(ValueError, match="Conflicting Stage-3 source"):
            worker._bind_stage3_request_ids("decode-a", "prefill-b")
        with pytest.raises(ValueError, match="already bound"):
            worker._bind_stage3_request_ids("decode-b", "prefill-a")

    def test_v1_qwen35_explicit_pool_admission_golden(self):
        cases = (
            (True, "pcp8_prefill", 8, 1),
            (False, "dp8_decode", 1, 8),
        )
        for is_producer, expected_topology, pcp_size, dp_size in cases:
            worker = _make_raiden_worker(tp_rank=0,
                                         tp_size=1,
                                         is_producer=is_producer,
                                         dp_size=dp_size,
                                         pcp_size=pcp_size)
            named, groups = _synthetic_qwen35_materialization(
                tp_size=1, pcp_size=pcp_size)
            raw_tensors = [
                torch.empty((8, ), dtype=torch.uint8) for _ in range(60)
            ]
            runner = SimpleNamespace(
                kv_caches=list(named.values()),
                kv_cache_raw_tensors=raw_tensors,
                kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
            )
            worker.named_kv_caches = named
            engine = _FakeAdmissionRaidenEngine()
            worker._construct_raiden_transfer_engine = MagicMock(
                return_value=engine)

            with patch(f"{_MOD}.tpu_envs.TPU_USE_RAIDEN_KV_CACHE_MANAGER",
                       True,
                       create=True), patch(
                           f"{_MOD}.tpu_envs.TPU_RAIDEN_QWEN35_ADMISSION",
                           True,
                           create=True), patch(f"{_MOD}.logger.info") as log:
                worker.register_runner(runner)

            construct_call = worker._construct_raiden_transfer_engine.call_args
            wrapped_storages = construct_call.args[0]
            assert construct_call.kwargs == {"num_slots": worker.chunk_transfer_window}
            typed_storages = set()
            for cache in named.values():
                tensors = cache if isinstance(cache, tuple) else (cache, )
                typed_storages.update(id(tensor) for tensor in tensors)
            wrapped_storage_ids = {id(tensor) for tensor in wrapped_storages}
            raw_storage_ids = {id(tensor) for tensor in raw_tensors}
            assert wrapped_storage_ids == typed_storages
            assert not wrapped_storage_ids & raw_storage_ids

            assert len(engine.registered_pools) == 105
            tag_counts = {}
            for pool in engine.registered_pools:
                tag_counts[pool["tag"]] = tag_counts.get(pool["tag"], 0) + 1
            assert tag_counts == {
                "fa": 15,
                "gdn.conv": 45,
                "gdn.ssm": 45,
            }
            summary = worker.raiden_admission_summary()
            assert summary["admitted"] is True
            assert summary["topology"] == expected_topology
            assert summary["model_server_role"] == (
                "kv_producer" if is_producer else "kv_consumer")
            assert summary["binding"] == "private_typed"
            assert summary["pools"] == 105
            assert summary["storages"] == 105
            assert summary["tag_counts"] == tag_counts
            assert summary["geometry"]["fa"] == {
                "num_blocks": 4,
                "block_stride_bytes": 128,
                "live_bytes_per_block": 128,
            }

            messages = [
                call.args[0] % call.args[1:] for call in log.call_args_list
            ]
            assert any(
                "Raiden pool admission complete "
                f"topology={expected_topology}" in message
                and "pools=105 storages=105 fa=15 gdn.conv=45 gdn.ssm=45" in
                message for message in messages)
            assert any(
                "Raiden pool binding verified: 105/105 pool storages matched "
                "typed KV cache storages" in message for message in messages)

    def test_v1_qwen35_unified_pool_admission_golden(self):
        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=1,
                                     is_producer=True,
                                     pcp_size=8)
        named, groups, raw_tensors = \
            _synthetic_qwen35_unified_pool_materialization(tp_size=1,
                                                           pcp_size=8)
        runner = SimpleNamespace(
            kv_caches=list(named.values()),
            kv_cache_raw_tensors=raw_tensors,
            kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        )
        worker.named_kv_caches = named
        engine = _FakeAdmissionRaidenEngine()
        worker._construct_raiden_transfer_engine = MagicMock(
            return_value=engine)

        with patch(f"{_MOD}.tpu_envs.TPU_USE_RAIDEN_KV_CACHE_MANAGER",
                   True,
                   create=True), patch(
                       f"{_MOD}.tpu_envs.TPU_RAIDEN_QWEN35_ADMISSION",
                       True,
                       create=True):
            worker.register_runner(runner)

        construct_call = worker._construct_raiden_transfer_engine.call_args
        assert construct_call.args[0] == raw_tensors
        assert construct_call.kwargs == {"num_slots": worker.chunk_transfer_window}
        assert len(engine.registered_pools) == 105
        assert len({pool["storage_index"]
                    for pool in engine.registered_pools}) == 15
        summary = worker.raiden_admission_summary()
        assert summary["binding"] == "aliased_raw"
        assert summary["pools"] == 105
        assert summary["storages"] == 15
        assert summary["tag_counts"] == {
            "fa": 15,
            "gdn.conv": 45,
            "gdn.ssm": 45,
        }
        assert summary["geometry"]["fa"] == {
            "num_blocks": 4,
            "block_stride_bytes": 4096,
            "live_bytes_per_block": 4096,
        }
        conv_pool = next(pool for pool in engine.registered_pools
                         if pool["tag"] == "gdn.conv")
        assert [region["units_per_stride"]
                for region in conv_pool["regions"]] == [2, 2, 4]

    def test_v2_stage3_startup_registration_payload_golden(self):
        cases = (
            # role, dp rank, transfer rank, unit replica, data/listener hosts
            (True, 0, 3, "engine7-rank3", "10.30.0.3"),
            (False, 5, 0, "engine7", "10.40.0.5"),
        )
        for is_producer, dp_rank, transfer_rank, replica_id, host in cases:
            worker = _make_raiden_worker(
                tp_rank=0,
                tp_size=1,
                is_producer=is_producer,
                dp_rank=dp_rank,
                dp_size=1 if is_producer else 8,
                pcp_size=8 if is_producer else 1,
                interleave_size=4,
            )
            named, groups = _synthetic_qwen35_materialization(
                tp_size=1, pcp_size=8 if is_producer else 1)
            runner = SimpleNamespace(
                kv_caches=list(named.values()),
                kv_cache_raw_tensors=[],
                kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
            )
            worker.named_kv_caches = named
            engine = _FakeAdmissionRaidenEngine(
                transfer_address=f"{host}:31001",
                listener_address=f"{host}:31002",
            )
            facade = _FakeRaidenControllerFacade()
            worker._construct_raiden_transfer_engine = MagicMock(
                return_value=engine)
            worker._measure_raiden_fa_layout = MagicMock(return_value=(
                "layout-fingerprint-golden",
                {
                    "schema": "qwen35-fa-raw-layout-fingerprint-v1",
                    "torch_tpu": "test-torch-tpu",
                    "libtpu": "test-libtpu",
                    "minor_to_major": [4, 3, 2, 1, 0],
                    "tiles": [[4, 128], [4, 1]],
                    "element_size_in_bits": 8,
                },
            ))
            worker._local_raiden_transfer_rank = MagicMock(
                return_value=transfer_rank)
            worker._new_raiden_controller_facade = MagicMock(
                return_value=facade)
            worker._new_raiden_id = MagicMock(
                side_effect=lambda fields: SimpleNamespace(**fields))

            with patch(
                    f"{_MOD}.tpu_envs.TPU_USE_RAIDEN_KV_CACHE_MANAGER",
                    True,
                    create=True), patch(
                        f"{_MOD}.tpu_envs.TPU_RAIDEN_QWEN35_ADMISSION",
                        True,
                        create=True), patch(
                            f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                            "raiden",
                            create=True), patch(
                                f"{_MOD}.tpu_envs."
                                "TPU_RAIDEN_CONTROLLER_ADDRESS",
                                "controller.test:27000",
                                create=True), patch(
                                    f"{_MOD}.tpu_envs.TPU_RAIDEN_JOB_NAME",
                                    "prefill-job"
                                    if is_producer else "decode-job",
                                    create=True), patch(
                                        f"{_MOD}.tpu_envs."
                                        "TPU_RAIDEN_ENGINE_ID",
                                        "engine7",
                                        create=True), patch(
                                            f"{_MOD}.tpu_envs."
                                            "TPU_RAIDEN_TRANSFER_PARALLELISM",
                                            8,
                                            create=True):
                worker.register_runner(runner)

            assert len(facade.register_work_unit_calls) == 1
            call = facade.register_work_unit_calls[0]
            unit = call["unit"]
            assert vars(unit) == {
                "job_name": "prefill-job" if is_producer else "decode-job",
                "job_replica_id": replica_id,
                "data_name": "kv.fa",
                "data_replica_idx": dp_rank,
            }
            assert call["shards"] == [f"{host}:31001"]
            assert call["control_plane_rpc_address"] == f"{host}:31002"
            assert call["layout_fingerprint"] == "layout-fingerprint-golden"
            assert call["page_tokens"] == 8
            assert "page_slice_tokens" not in call
            assert call["transfer_parallelism"] == 8
            assert call["transfer_rank"] == transfer_rank
            assert call["pool_manifest"] == engine.registered_pools
            assert len(call["pool_manifest"]) == 105
            assert [pool["tag"]
                    for pool in call["pool_manifest"]].count("fa") == 15

            registration = worker.raiden_admission_summary(
            )["stage3_registration"]
            assert registration["unit"] == vars(unit)
            assert registration["shards"] == [f"{host}:31001"]
            assert registration["control_plane_rpc_address"] == f"{host}:31002"
            assert registration["interleave_tokens"] == (4
                                                         if is_producer else 8)
            assert worker._raiden_controller_facade is facade
            assert worker._raiden_work_unit is unit

    def test_stage3_producer_unit_ids_are_unique_and_enumerable(self):
        fields = [
            stage3_fa_raiden_id_fields(
                job_name="prefill",
                engine_id="engine0",
                dp_rank=0,
                transfer_rank=rank,
                is_producer=True,
            ) for rank in range(8)
        ]
        assert [field["job_replica_id"] for field in fields
                ] == [f"engine0-rank{rank}" for rank in range(8)]
        assert len({tuple(field.items()) for field in fields}) == 8

    def test_stage3_registration_requires_explicit_controller_address(self):
        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=1,
                                     is_producer=True,
                                     dp_size=1,
                                     pcp_size=8,
                                     interleave_size=256,
                                     block_size=4096)
        runner = SimpleNamespace(kv_caches=[])
        with patch(
                f"{_MOD}.tpu_envs.TPU_USE_RAIDEN_KV_CACHE_MANAGER",
                True,
                create=True), patch(
                    f"{_MOD}.tpu_envs.TPU_RAIDEN_QWEN35_ADMISSION",
                    True,
                    create=True), patch(
                        f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                        "raiden",
                        create=True), patch(
                            f"{_MOD}.tpu_envs.TPU_RAIDEN_CONTROLLER_ADDRESS",
                            "",
                            create=True):
            with pytest.raises(ValueError,
                               match="TPU_RAIDEN_CONTROLLER_ADDRESS"):
                worker.register_runner(runner)

    def test_stage3_manager_owns_listener_and_uses_pcp_rank_endpoint(self):
        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=1,
                                     is_producer=True,
                                     dp_size=1,
                                     pcp_size=8)
        worker._local_raiden_transfer_rank = MagicMock(return_value=3)
        engine = _FakeAdmissionRaidenEngine()
        worker._new_raiden_manager = MagicMock(return_value=engine)
        storage = object()

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), patch(
                       f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM",
                       8,
                       create=True), patch(
                           f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout",
                           return_value=12.0):
            result = worker._construct_raiden_transfer_engine([storage],
                                                              num_slots=1)

        assert result is engine
        worker._new_raiden_manager.assert_called_once_with(
            kv_caches=[storage],
            node_id=3,
            local_control_port=9106,
            max_blocks=4,
            num_slots=1,
            timeout_s=12.0,
            listener_port=0,
            parallelism=8,
        )

    @pytest.mark.parametrize("transfer_rank", (0, 3, 7))
    def test_stage3_gdn_registration_declares_every_pcp_head_shard(
            self, transfer_rank):
        from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import \
            PoolByteSpan
        from vllm_torchtpu.distributed.kv_transfer.raiden.pool_manifest import (  # noqa: E501
            BINDING_ALIASED_RAW, PoolEntry, PoolManifest, RegionSpec)

        # Qwen3.5-35B TP8 shard geometry with the QK pair-blocked layout:
        # every raw extent is a whole 1024-byte physical pool token.
        conv_regions = (
            RegionSpec("gdn_conv_qk", 0, 2048, 1024, 3, 1),
            RegionSpec("gdn_conv_v", 1024, 2048, 256, 3, 4),
        )
        ssm_regions = (RegionSpec("gdn_ssm", 0, 1024, 1024, 4), )
        manifest = PoolManifest(
            binding=BINDING_ALIASED_RAW,
            storages=[],
            pools=[
                PoolEntry("gdn.conv.g0", "linear.0", 0, 2048, 8192, 32,
                          conv_regions, "bfloat16"),
                PoolEntry("gdn.ssm.g0", "linear.0", 0, 0, 4096, 32,
                          ssm_regions, "float32"),
            ],
        )
        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=1,
                                     is_producer=True,
                                     dp_size=1,
                                     pcp_size=8)
        worker._raiden_manifest = manifest
        worker._raiden_layout_fingerprint_payload = {
            "minor_to_major": [4, 3, 2, 1, 0],
            "tiles": [[4, 128], [4, 1]],
            "element_size_in_bits": 8,
        }
        worker._raiden_work_unit = SimpleNamespace(job_name="prefill")
        worker._stage3_state_group_count = 1
        facade = _FakeRaidenControllerFacade()

        registrations = worker._stage3_state_pool_spans([17], transfer_rank, 8)

        # T3.1 collapse: the state classes ride the base registration as
        # additional PoolSpanRegistrations — no derived req-ids, no
        # sibling facade calls.
        assert len(facade.register_request_blocks_calls) == 0
        assert len(registrations) == 2
        conv, ssm = registrations
        assert conv.tag == "gdn.conv.g0"
        assert conv.block_ids == (17, )
        assert conv.declared_bytes == 6144
        assert conv.spans == (
            PoolByteSpan(0, 0, 0, transfer_rank * 1024, 1024, 2048, 16384, 3),
            PoolByteSpan(0, 1024, 0, 8192 + transfer_rank * 1024, 1024, 2048,
                         16384, 3),
        )
        assert ssm.tag == "gdn.ssm.g0"
        assert ssm.block_ids == (17, )
        assert ssm.declared_bytes == 4096
        assert ssm.spans == (PoolByteSpan(0, 0, 0, transfer_rank * 4096,
                                          4096), )

    def test_v3_stage3_producer_registers_rank_stripe_and_releases_on_done(
            self):
        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=1,
                                     is_producer=True,
                                     dp_size=1,
                                     pcp_size=8,
                                     interleave_size=256,
                                     block_size=4096)
        engine = _FakeRaidenEngine()
        engine.poll_results = [([], [], []), (["striped"], [], [])]
        facade = _FakeRaidenControllerFacade()
        unit = SimpleNamespace(job_name="custom-prefill",
                               job_replica_id="producer-engine-rank3",
                               data_name="kv.fa",
                               data_replica_idx=0)
        worker._raiden_transfer_engine = engine
        worker._raiden_controller_facade = facade
        worker._raiden_controller_address = "prefill-controller.test:27000"
        worker._raiden_work_unit = unit
        worker._local_raiden_transfer_rank = MagicMock(return_value=3)
        meta = TPUConnectorMetadata()
        meta.reqs_to_send["striped"] = MagicMock(uuid=1234,
                                                 local_block_ids=[100, 101],
                                                 num_tokens=65_023,
                                                 expiration_time=1e20)

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), patch(
                       f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM",
                       8,
                       create=True):
            worker.process_send_load(meta)
            worker.process_send_load(meta)
            assert worker.get_finished() == (set(), set())
            assert facade.complete_request_blocks_calls == []
            assert worker.get_finished() == ({"striped"}, set())

        assert len(facade.register_request_blocks_calls) == 1
        registration_call = facade.register_request_blocks_calls[0]
        assert registration_call["req_id"] == "striped"
        assert registration_call["uuid"] == 1234
        assert registration_call["unit"] is unit
        assert registration_call["block_ids"] == [100, 101]
        # The declared source map rides the registration and is generated by
        # the kernel's own layout function, lowered to the byte-span IR
        # (the default plan vocabulary since the M4 cutover).
        from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import \
            lower_fa_spans
        expected_registration = lower_fa_spans(
            num_tokens=65_023,
            transfer_rank=3,
            parallelism=8,
            interleave_tokens=256,
            page_tokens=4096,
            token_bytes=1024,
            block_ids=[100, 101],
        )
        assert registration_call["pool_spans"] == [expected_registration]
        assert expected_registration.declared_bytes <= 2 * 4096 * 1024
        assert all(span.src_offset_bytes + span.size_bytes <= 4096 * 1024
                   for span in expected_registration.spans)
        assert facade.complete_request_blocks_calls == [{
            "req_id": "striped",
            "uuid": 1234,
            "unit": unit,
        }]
        assert worker._stage3_registered_sends == {}
        worker.process_send_load(meta)
        assert worker.get_finished() == (set(), set())
        assert len(facade.register_request_blocks_calls) == 1

    def test_v3_stage3_late_registration_cancellation_is_local_terminal(self):
        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=1,
                                     is_producer=True,
                                     dp_size=1,
                                     pcp_size=8,
                                     block_size=4096)
        engine = _FakeRaidenEngine()
        engine.poll_results = [([], [], []), ([], [], [])]
        facade = _FakeRaidenControllerFacade()
        facade.register_request_blocks = MagicMock(side_effect=RuntimeError(
            "Remote Controller Server execution failed: Request block "
            "registration was cancelled for req_id=late-rank, uuid=2468"))
        worker._raiden_transfer_engine = engine
        worker._raiden_controller_facade = facade
        worker._raiden_controller_address = "prefill-controller.test:27000"
        worker._raiden_work_unit = SimpleNamespace(job_name="prefill")
        worker._local_raiden_transfer_rank = MagicMock(return_value=7)
        meta = TPUConnectorMetadata()
        meta.reqs_to_send["late-rank"] = MagicMock(
            uuid=2468,
            local_block_ids=[90, 91],
            num_tokens=65_023,
            expiration_time=time.perf_counter() - 1.0,
        )

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), patch(
                       f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM",
                       8,
                       create=True), patch(
                           f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout",
                           return_value=30.0):
            worker.process_send_load(meta)
            # Replays both before and after reporting the terminal vote are
            # local no-ops and never retry the rejected controller RPC.
            worker.process_send_load(meta)
            assert worker.get_finished() == ({"late-rank"}, set())
            worker.process_send_load(meta)
            assert worker.get_finished() == (set(), set())

        facade.register_request_blocks.assert_called_once_with(
            req_id="late-rank",
            uuid=2468,
            unit=worker._raiden_work_unit,
            block_ids=[90, 91],
            pool_spans=ANY,
        )
        assert worker._stage3_registered_sends == {}
        tombstone = worker._stage3_terminal_sends["late-rank"]
        assert tombstone.uuid == 2468
        assert tombstone.local_block_ids == (90, 91)
        assert tombstone.num_tokens == 65_023
        assert tombstone.expiration_time > time.perf_counter()
        assert worker._stage3_terminal_cleanup == {}
        assert facade.complete_request_blocks_calls == []
        assert facade.cancel_request_blocks_calls == []

    @pytest.mark.parametrize(
        "registration_error",
        (
            RuntimeError("controller unavailable"),
            ValueError("Request block registration is already claimed"),
        ),
    )
    def test_v3_stage3_non_cancellation_registration_error_propagates(
            self, registration_error):
        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=1,
                                     is_producer=True,
                                     dp_size=1,
                                     pcp_size=8,
                                     block_size=4096)
        worker._raiden_transfer_engine = _FakeRaidenEngine()
        facade = _FakeRaidenControllerFacade()
        facade.register_request_blocks = MagicMock(
            side_effect=registration_error)
        worker._raiden_controller_facade = facade
        worker._raiden_controller_address = "prefill-controller.test:27000"
        worker._raiden_work_unit = SimpleNamespace(job_name="prefill")
        worker._local_raiden_transfer_rank = MagicMock(return_value=7)
        meta = TPUConnectorMetadata()
        meta.reqs_to_send["registration-error"] = MagicMock(
            uuid=2469,
            local_block_ids=[92, 93],
            num_tokens=65_023,
            expiration_time=1e20,
        )

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), patch(
                       f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM",
                       8,
                       create=True), pytest.raises(
                           type(registration_error),
                           match=str(registration_error)):
            worker.process_send_load(meta)

        assert worker._stage3_registered_sends == {}
        assert worker._stage3_terminal_sends == {}
        assert worker._done_sending == set()
        assert facade.complete_request_blocks_calls == []
        assert facade.cancel_request_blocks_calls == []

    @pytest.mark.parametrize(
        ("transfer_rank", "expected_ids"),
        ((0, [77]), (1, [77]), (2, [77]), (7, [77])),
    )
    def test_v3_stage3_partial_pcp_group_registers_rank_prefix(
            self, transfer_rank, expected_ids):
        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=1,
                                     is_producer=True,
                                     dp_size=1,
                                     pcp_size=8,
                                     block_size=4096)
        engine = _FakeRaidenEngine()
        engine.poll_results = [([], [], [])]
        worker._raiden_transfer_engine = engine
        facade = _FakeRaidenControllerFacade()
        worker._raiden_controller_facade = facade
        worker._raiden_controller_address = "prefill-controller.test:27000"
        worker._raiden_work_unit = SimpleNamespace(job_name="prefill")
        worker._local_raiden_transfer_rank = MagicMock(
            return_value=transfer_rank)
        meta = TPUConnectorMetadata()
        # 4,097 logical tokens span two complete 2,048-token interleave cycles
        # plus rank 0's first token in cycle 2. Every PCP rank therefore owns
        # live bytes in the shared scheduler block.
        meta.reqs_to_send["partial"] = MagicMock(uuid=4321,
                                                 local_block_ids=[77],
                                                 num_tokens=4097,
                                                 expiration_time=1e20)

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), patch(
                       f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM",
                       8,
                       create=True):
            worker.process_send_load(meta)
            done_sending, done_recving = worker.get_finished()
            worker.process_send_load(meta)
            replay_done = worker.get_finished()

        assert len(facade.register_request_blocks_calls) == 1
        registration_call = facade.register_request_blocks_calls[0]
        assert registration_call["req_id"] == "partial"
        assert registration_call["uuid"] == 4321
        assert registration_call["unit"] is worker._raiden_work_unit
        assert registration_call["block_ids"] == expected_ids
        declared = sum(span.size_bytes // 1024
                       for entry in registration_call["pool_spans"]
                       for span in entry.spans)
        assert declared <= len(expected_ids) * 4096
        if expected_ids:
            assert declared > (len(expected_ids) - 1) * 4096
        else:
            assert registration_call["pool_spans"] == []
        assert done_sending == ({"partial"} if not expected_ids else set())
        assert done_recving == set()
        assert replay_done == (set(), set())
        assert facade.complete_request_blocks_calls == (
            [{
                "req_id": "partial",
                "uuid": 4321,
                "unit": worker._raiden_work_unit,
            }] if not expected_ids else [])

    def test_v3_stage3_unconsumed_send_cancels_only_if_unclaimed(self):
        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=1,
                                     is_producer=True,
                                     dp_size=1,
                                     pcp_size=8,
                                     block_size=4096)
        engine = _FakeRaidenEngine()
        engine.poll_results = [([], [], [])]
        facade = _FakeRaidenControllerFacade()
        worker._raiden_transfer_engine = engine
        worker._raiden_controller_facade = facade
        worker._raiden_controller_address = "prefill-controller.test:27000"
        worker._raiden_work_unit = SimpleNamespace(job_name="prefill")
        worker._local_raiden_transfer_rank = MagicMock(return_value=0)
        meta = TPUConnectorMetadata()
        meta.reqs_to_send["unconsumed"] = MagicMock(
            uuid=765,
            local_block_ids=[12],
            num_tokens=1,
            expiration_time=time.perf_counter() - 1.0,
        )

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), patch(
                       f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM",
                       8,
                       create=True):
            worker.process_send_load(meta)
            assert worker.get_finished() == ({"unconsumed"}, set())

        assert facade.cancel_request_blocks_calls == [{
            "req_id": "unconsumed",
            "uuid": 765,
        }]
        assert facade.complete_request_blocks_calls == []
        assert worker._stage3_registered_sends == {}

    def test_v3_stage3_claimed_send_cannot_expire_under_active_transfer(self):
        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=1,
                                     is_producer=True,
                                     dp_size=1,
                                     pcp_size=8,
                                     block_size=4096)
        engine = _FakeRaidenEngine()
        engine.poll_results = [([], [], [])]
        facade = _FakeRaidenControllerFacade()
        facade.cancel_request_blocks_result = False
        worker._raiden_transfer_engine = engine
        worker._raiden_controller_facade = facade
        worker._raiden_controller_address = "prefill-controller.test:27000"
        worker._raiden_work_unit = SimpleNamespace(job_name="prefill")
        worker._local_raiden_transfer_rank = MagicMock(return_value=0)
        meta = TPUConnectorMetadata()
        meta.reqs_to_send["claimed"] = MagicMock(
            uuid=766,
            local_block_ids=[13],
            num_tokens=1,
            expiration_time=time.perf_counter() - 1.0,
        )

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), patch(
                       f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM",
                       8,
                       create=True):
            worker.process_send_load(meta)
            assert worker.get_finished() == (set(), set())

        assert facade.cancel_request_blocks_calls == [{
            "req_id": "claimed",
            "uuid": 766,
        }]
        assert "claimed" in worker._stage3_registered_sends
        assert facade.complete_request_blocks_calls == []

    def test_v4_stage3_retries_only_missing_d5_registration(self):
        facade = MagicMock()
        facade.start_transfer.side_effect = (
            RuntimeError("Missing producer block registration for rank 7"),
            True,
        )

        with patch(f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=30.0), patch(f"{_MOD}.time.sleep") as sleep:
            accepted = TPURaidenConnectorWorker._start_stage3_transfer_with_d5_retry(
                facade, req_id="d5-race", uuid=987)

        assert accepted is True
        assert facade.start_transfer.call_count == 2
        sleep.assert_called_once_with(0.01)

        facade.start_transfer.reset_mock()
        facade.start_transfer.side_effect = RuntimeError(
            "fingerprint mismatch")
        with patch(f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=30.0):
            with pytest.raises(RuntimeError, match="fingerprint mismatch"):
                TPURaidenConnectorWorker._start_stage3_transfer_with_d5_retry(
                    facade, req_id="not-retryable", uuid=988)
        facade.start_transfer.assert_called_once()

    def test_stage3_consumer_clip_kwarg_reflects_skip_tokens(self):
        worker = _make_raiden_worker(is_producer=False, block_size=1024)
        worker._raiden_work_unit = MagicMock()
        facade = MagicMock()
        facade.start_transfer.return_value = True
        worker._stage3_source_facades["prefill-controller.test:27000"] = facade
        common = dict(
            num_tokens=4095,
            src_controller_address="prefill-controller.test:27000",
            src_job_name="prefill",
            src_engine_id="producer-engine",
            src_data_replica_idx=0,
            src_parallelism=8,
        )
        meta = TPUConnectorMetadata()
        meta.reqs_to_load["dst-suffix"] = _Stage3LoadMeta(
            uuid=997,
            source_req_id="src-suffix",
            local_block_ids=[52, 53],
            skip_tokens=2048,
            **common,
        )
        meta.reqs_to_load["dst-full"] = _Stage3LoadMeta(
            uuid=998,
            source_req_id="src-full",
            local_block_ids=[50, 51, 52, 53],
            **common,
        )

        with patch.object(worker,
                          "_require_stage3_controller",
                          return_value=(MagicMock(), "10.0.0.2:28000")), \
             patch.object(worker,
                          "_stage3_source_work_units",
                          return_value=[MagicMock()]), \
             patch.object(worker,
                          "_raiden_hbm_memory_type",
                          MagicMock(return_value=3)), \
             patch.object(worker, "_stage3_fa_token_bytes", return_value=64):
            worker._submit_stage3_loads(meta, MagicMock())

        assert facade.start_transfer.call_count == 2
        clipped = facade.start_transfer.call_args_list[0].kwargs
        assert clipped["dst_skip_bytes"] == [2048 * 64]
        assert clipped["dst_device_block_ids"] == [52, 53]
        assert clipped["dst_block_counts"] == [2]
        assert clipped["transfer_pool_tags"] == ["fa"]
        assert clipped["num_tokens"] == 4095
        assert worker._stage3_submitted_loads["dst-suffix"] == 997
        assert worker._load_block_ids["dst-suffix"] == [52, 53]
        # Skip-free submissions must stay byte-identical on the wire: the
        # kwarg is omitted entirely rather than passed as zeros.
        unclipped = facade.start_transfer.call_args_list[1].kwargs
        assert "dst_skip_bytes" not in unclipped
        assert unclipped["dst_device_block_ids"] == [50, 51, 52, 53]

    def test_stage3_release_only_meta_cancels_producer_registration(self):
        worker = _make_raiden_worker(is_producer=False, block_size=1024)
        worker._raiden_work_unit = MagicMock()
        facade = MagicMock()
        facade.cancel_request_blocks_if_unclaimed.return_value = True
        worker._stage3_source_facades["prefill-controller.test:27000"] = facade
        meta = TPUConnectorMetadata()
        meta.reqs_to_load["dst-hit"] = _Stage3LoadMeta(
            uuid=999,
            source_req_id="src-hit",
            local_block_ids=[],
            num_tokens=2047,
            src_controller_address="prefill-controller.test:27000",
            src_job_name="prefill",
            src_engine_id="producer-engine",
            src_data_replica_idx=0,
            src_parallelism=8,
            release_only=True,
        )

        with patch.object(worker,
                          "_require_stage3_controller",
                          return_value=(MagicMock(), "10.0.0.2:28000")):
            worker._submit_stage3_loads(meta, MagicMock())

        facade.cancel_request_blocks_if_unclaimed.assert_called_once_with(
            req_id="src-hit", uuid=999)
        facade.start_transfer.assert_not_called()
        assert worker._stage3_submitted_loads == {}
        assert "dst-hit" not in worker._load_block_ids

    def test_v3_stage3_sender_failure_is_terminal_and_releases_d5(self):
        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=1,
                                     is_producer=True,
                                     dp_size=1,
                                     pcp_size=8,
                                     block_size=4096)
        engine = _FakeRaidenEngine()
        # Native ReshardPush failures use poll_stats' historical third tuple.
        engine.poll_results = [([], [], ["failed-send"])]
        facade = _FakeRaidenControllerFacade()
        unit = SimpleNamespace(job_name="prefill",
                               job_replica_id="engine-rank0",
                               data_name="kv.fa",
                               data_replica_idx=0)
        worker._raiden_transfer_engine = engine
        worker._raiden_controller_facade = facade
        worker._raiden_controller_address = "prefill-controller.test:27000"
        worker._raiden_work_unit = unit
        worker._local_raiden_transfer_rank = MagicMock(return_value=0)
        meta = TPUConnectorMetadata()
        meta.reqs_to_send["failed-send"] = MagicMock(uuid=55,
                                                     local_block_ids=[0, 1],
                                                     num_tokens=65_023,
                                                     expiration_time=1e20)

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), patch(
                       f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM",
                       8,
                       create=True):
            worker.process_send_load(meta)
            done_sending, done_recving = worker.get_finished()

        assert done_sending == {"failed-send"}
        assert done_recving == set()
        assert facade.complete_request_blocks_calls == [{
            "req_id": "failed-send",
            "uuid": 55,
            "unit": unit,
        }]
        assert worker._stage3_registered_sends == {}

    def test_v3_stage3_d5_release_failure_retries_without_ttl_leak(self):
        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=1,
                                     is_producer=True,
                                     dp_size=1,
                                     pcp_size=8,
                                     block_size=4096)
        engine = _FakeRaidenEngine()
        engine.poll_results = [(["release-retry"], [], []), ([], [], [])]
        facade = _FakeRaidenControllerFacade()
        facade.complete_request_blocks = MagicMock(
            side_effect=(RuntimeError("controller unavailable"), None))
        worker._raiden_transfer_engine = engine
        worker._raiden_controller_facade = facade
        worker._raiden_controller_address = "prefill-controller.test:27000"
        worker._raiden_work_unit = SimpleNamespace(
            job_name="prefill",
            job_replica_id="engine-rank0",
            data_name="kv.fa",
            data_replica_idx=0,
        )
        worker._local_raiden_transfer_rank = MagicMock(return_value=0)
        meta = TPUConnectorMetadata()
        meta.reqs_to_send["release-retry"] = MagicMock(uuid=66,
                                                       local_block_ids=[0, 1],
                                                       num_tokens=65_023,
                                                       expiration_time=1e20)

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), patch(
                       f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM",
                       8,
                       create=True):
            worker.process_send_load(meta)
            with pytest.raises(RuntimeError, match="controller unavailable"):
                worker.get_finished()
            assert "release-retry" in worker._stage3_registered_sends
            assert worker.get_finished() == ({"release-retry"}, set())

        assert facade.complete_request_blocks.call_count == 2
        facade.complete_request_blocks.assert_called_with(
            req_id="release-retry",
            uuid=66,
            unit=worker._raiden_work_unit,
        )
        assert worker._stage3_registered_sends == {}

    def test_v4_stage3_consumer_maps_source_id_to_local_lifecycle(self):
        scheduler = _make_raiden_scheduler(is_producer=False, block_size=1024)
        req = MagicMock()
        req.request_id = "proxy-id-decode5678"
        req.kv_transfer_params = {
            "req_id": "proxy-id-prefill1234",
            "uuid": 808,
            "num_tokens": 65_023,
            "src_controller_address": "source-controller.test:27000",
            "src_job_name": "custom-source-job",
            "src_engine_id": "producer-engine-9",
            "src_data_replica_idx": 0,
            "src_parallelism": 8,
        }
        blocks = MagicMock()
        blocks.get_block_ids.return_value = (list(range(300, 364)), )
        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True):
            scheduler.update_state_after_alloc(req, blocks, 65_023)
            meta = scheduler.build_connector_meta()

        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=1,
                                     is_producer=False,
                                     dp_rank=5,
                                     dp_size=8,
                                     pcp_size=1)
        engine = _FakeRaidenEngine()
        # A stale native ID equal to the active destination ID must not be
        # accepted; only the explicitly bound producer/controller ID may
        # complete the decode-local lifecycle.
        engine.poll_results = [([], ["proxy-id-decode5678"], []),
                               ([], ["proxy-id-prefill1234"], []),
                               ([], [], [])]
        facade = _FakeRaidenControllerFacade()
        destination_unit = SimpleNamespace(job_name="decode-job",
                                           job_replica_id="decode-engine-4",
                                           data_name="kv.fa",
                                           data_replica_idx=5)
        worker._raiden_transfer_engine = engine
        worker._raiden_controller_facade = facade
        worker._raiden_controller_address = "dest-controller.test:28000"
        # P2: load submissions go to the SOURCE controller facade.
        worker._new_raiden_controller_facade = MagicMock(return_value=facade)
        worker._raiden_work_unit = destination_unit
        worker._raiden_manifest = SimpleNamespace(tag_counts=lambda: {
            "fa": 15,
            "gdn.conv": 45,
            "gdn.ssm": 45,
        })
        worker._new_raiden_id = MagicMock(
            side_effect=lambda fields: SimpleNamespace(**fields))
        worker._raiden_hbm_memory_type = MagicMock(return_value="HBM")

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), patch(
                       f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM",
                       8,
                       create=True), patch(
                           f"{_MOD}.tpu_envs.TPU_RAIDEN_ENGINE_ID",
                           "different-decode-engine",
                           create=True), patch(f"{_MOD}.logger.info") as log:
            worker.process_send_load(meta)
            worker.process_send_load(meta)
            assert worker.get_finished() == (set(), set())
            assert worker.get_finished() == (set(), {"proxy-id-decode5678"})
            # _reported_recving preserves the existing at-most-once contract.
            assert worker.get_finished() == (set(), set())
            assert worker.get_finished({"proxy-id-decode5678"}) == (set(),
                                                                    set())

        assert worker._stage3_source_req_ids == {}
        assert worker._stage3_destination_req_ids == {}

        assert len(facade.start_transfer_calls) == 1
        call = facade.start_transfer_calls[0]
        assert [unit.job_replica_id for unit in call["src_units"]
                ] == [f"producer-engine-9-rank{rank}" for rank in range(8)]
        assert {unit.job_name
                for unit in call["src_units"]} == {"custom-source-job"}
        assert {unit.data_replica_idx for unit in call["src_units"]} == {0}
        assert call["dst_units"] == [destination_unit]
        # Controller/native identity must match the producer's D5 key, while
        # vLLM completion above remains keyed by the decode-local request ID.
        assert call["req_id"] == "proxy-id-prefill1234"
        assert call["uuid"] == 808
        assert call["dst_device_block_ids"] == list(range(300, 364))
        assert call["num_tokens"] == 65_023
        assert call["src_controller_address"] == (
            "source-controller.test:27000")
        assert call["dst_controller_address"] == ("dest-controller.test:28000")
        assert call["is_sender"] is True
        assert call["use_block_chunks"] is True
        assert call["dst_mem_type"] == "HBM"
        # Pool selection is request data: the connector names manifest tags;
        # parallelism comes from the producer's registration, not this wire.
        assert call["transfer_pool_tags"] == ["fa"]
        assert "parallelism" not in call
        assert "src_block_ids" not in call
        assert worker.get_block_ids_with_load_errors() == set()
        info_messages = [
            record.args[0] % record.args[1:] for record in log.call_args_list
        ]
        assert any('"event": "raiden_stage3_transfer_submitted"' in message
                   and '"req_id": "proxy-id-prefill1234"' in message
                   and '"destination_req_id": "proxy-id-decode5678"' in message
                   for message in info_messages)
        assert any('"event": "raiden_stage3_receiver_complete"' in message
                   and '"req_id": "proxy-id-prefill1234"' in message
                   and '"destination_req_id": "proxy-id-decode5678"' in message
                   for message in info_messages)
        assert any("recv_armed_before_push=1 state_groups=0" in message
                   for message in info_messages)

    def test_v4_stage3_native_failure_surfaces_exact_destination_blocks(self):
        scheduler = _make_raiden_scheduler(is_producer=False, block_size=1024)
        req = MagicMock()
        req.request_id = "failed-load-decode"
        req.kv_transfer_params = {
            "req_id": "failed-load-prefill",
            "uuid": 909,
            "num_tokens": 1536,
            "src_controller_address": "source-controller.test:27000",
            "src_job_name": "prefill",
            "src_engine_id": "source-engine",
            "src_data_replica_idx": 0,
            "src_parallelism": 8,
        }
        blocks = MagicMock()
        blocks.get_block_ids.return_value = ([41, 43], )
        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True):
            scheduler.update_state_after_alloc(req, blocks, 1536)
            meta = scheduler.build_connector_meta()

        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=1,
                                     is_producer=False,
                                     dp_size=8,
                                     pcp_size=1)
        engine = _FakeRaidenEngine()
        engine.poll_results = [([], [], ["failed-load-prefill"]), ([], [], [])]
        facade = _FakeRaidenControllerFacade()
        worker._raiden_transfer_engine = engine
        worker._raiden_controller_facade = facade
        worker._raiden_controller_address = "dest-controller.test:28000"
        # P2: load submissions go to the SOURCE controller facade.
        worker._new_raiden_controller_facade = MagicMock(return_value=facade)
        worker._raiden_work_unit = SimpleNamespace(job_name="decode",
                                                   job_replica_id="decode",
                                                   data_name="kv.fa",
                                                   data_replica_idx=0)
        worker._raiden_manifest = SimpleNamespace(tag_counts=lambda: {
            "fa": 15,
            "gdn.conv": 45,
            "gdn.ssm": 45,
        })
        worker._new_raiden_id = MagicMock(
            side_effect=lambda fields: SimpleNamespace(**fields))
        worker._raiden_hbm_memory_type = MagicMock(return_value="HBM")

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), patch(
                       f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM",
                       8,
                       create=True):
            worker.process_send_load(meta)
            worker.process_send_load(meta)
            assert worker.get_finished() == (set(), {"failed-load-decode"})
            assert worker.get_finished() == (set(), set())

        assert len(facade.start_transfer_calls) == 1
        assert facade.start_transfer_calls[0]["req_id"] == (
            "failed-load-prefill")
        assert worker.get_block_ids_with_load_errors() == {41, 43}
        assert worker.get_block_ids_with_load_errors() == set()

    @pytest.mark.parametrize("terminal_result", ("failure", "success"))
    def test_v4_stage3_post_arm_rpc_error_waits_for_native_terminal(
            self, terminal_result):
        scheduler = _make_raiden_scheduler(is_producer=False, block_size=1024)
        req = MagicMock()
        req.request_id = "uncertain-load"
        req.kv_transfer_params = {
            "req_id": "uncertain-load",
            "uuid": 910,
            "num_tokens": 1536,
            "src_controller_address": "source-controller.test:27000",
            "src_job_name": "prefill",
            "src_engine_id": "source-engine",
            "src_data_replica_idx": 0,
            "src_parallelism": 8,
        }
        blocks = MagicMock()
        blocks.get_block_ids.return_value = ([41, 43], )
        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True):
            scheduler.update_state_after_alloc(req, blocks, 1536)
            meta = scheduler.build_connector_meta()

        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=1,
                                     is_producer=False,
                                     dp_size=8,
                                     pcp_size=1)
        engine = _FakeRaidenEngine()
        terminal_poll = (([], [], ["uncertain-load"]) if terminal_result
                         == "failure" else ([], ["uncertain-load"], []))
        engine.poll_results = [([], [], []), terminal_poll]
        facade = _FakeRaidenControllerFacade()
        facade.start_transfer = MagicMock(
            side_effect=RuntimeError("sender dispatch failed after arm"))
        worker._raiden_transfer_engine = engine
        worker._raiden_controller_facade = facade
        worker._raiden_controller_address = "dest-controller.test:28000"
        # P2: load submissions go to the SOURCE controller facade.
        worker._new_raiden_controller_facade = MagicMock(return_value=facade)
        worker._raiden_work_unit = SimpleNamespace(job_name="decode")
        worker._raiden_manifest = SimpleNamespace(tag_counts=lambda: {
            "fa": 15,
            "gdn.conv": 45,
            "gdn.ssm": 45,
        })
        worker._new_raiden_id = MagicMock(
            side_effect=lambda fields: SimpleNamespace(**fields))
        worker._raiden_hbm_memory_type = MagicMock(return_value="HBM")

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), patch(
                       f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM",
                       8,
                       create=True), patch(
                           f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout",
                           return_value=30.0):
            worker.process_send_load(meta)
            assert worker.get_finished() == (set(), set())
            assert worker.get_block_ids_with_load_errors() == set()
            assert worker.get_finished() == (set(), {"uncertain-load"})

        expected_errors = {41, 43} if terminal_result == "failure" else set()
        assert worker.get_block_ids_with_load_errors() == expected_errors

    def test_v4_stage3_uncertain_rpc_times_out_before_recompute(self):
        # Exercise the no-native-terminal fallback without duplicating the
        # full scheduler setup used above.
        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=1,
                                     is_producer=False,
                                     dp_size=8,
                                     pcp_size=1)
        engine = _FakeRaidenEngine()
        engine.poll_results = [([], [], [])]
        worker._raiden_transfer_engine = engine
        worker._bind_stage3_request_ids("timed-out-load", "timed-out-source")
        worker._stage3_controller_accepted.add("timed-out-load")
        worker._stage3_pending_controller_failures["timed-out-load"] = 0.0
        worker._load_block_ids["timed-out-load"] = [51, 53]

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True):
            assert worker.get_finished() == (set(), {"timed-out-load"})

        assert worker.get_block_ids_with_load_errors() == {51, 53}

    @pytest.mark.parametrize(
        ("terminal_poll", "expected_errors"),
        [
            (([], ["aborted-load-prefill"], []), set()),
            (([], [], ["aborted-load-prefill"]), {61, 63}),
        ],
        ids=("success", "failure"),
    )
    def test_v4_stage3_aborted_load_retains_state_until_native_terminal(
            self, terminal_poll, expected_errors):
        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=1,
                                     is_producer=False,
                                     dp_size=8,
                                     pcp_size=1)
        engine = _FakeRaidenEngine()
        engine.poll_results = [([], [], []), terminal_poll]
        worker._raiden_transfer_engine = engine
        worker._stage3_submitted_loads["aborted-load-decode"] = 911
        worker._stage3_submitted_load_tokens["aborted-load-decode"] = 1536
        worker._bind_stage3_request_ids("aborted-load-decode",
                                        "aborted-load-prefill")
        worker._stage3_controller_accepted.add("aborted-load-decode")
        worker._load_block_ids["aborted-load-decode"] = [61, 63]

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True):
            # vLLM has marked the request finished, but its delayed block free
            # still needs a connector terminal. The first manager poll has no
            # result, so every field required to recognize a late terminal is
            # retained.
            assert worker.get_finished({"aborted-load-decode"}) == (set(),
                                                                    set())
            assert worker._stage3_submitted_loads == {
                "aborted-load-decode": 911
            }
            assert worker._stage3_controller_accepted == {
                "aborted-load-decode"
            }
            assert worker._load_block_ids == {"aborted-load-decode": [61, 63]}
            assert worker._stage3_source_req_ids == {
                "aborted-load-decode": "aborted-load-prefill"
            }
            assert worker._stage3_destination_req_ids == {
                "aborted-load-prefill": "aborted-load-decode"
            }
            assert worker._stage3_finished_loads_pending_cleanup == {
                "aborted-load-decode"
            }
            assert worker.get_block_ids_with_load_errors() == set()

            # Native completion is accepted despite the earlier abort, is
            # reported exactly once so the scheduler frees delayed blocks,
            # and only then retires the retained state.
            assert worker.get_finished() == (set(), {"aborted-load-decode"})

        assert worker._stage3_submitted_loads == {}
        assert worker._stage3_submitted_load_tokens == {}
        assert worker._stage3_controller_accepted == set()
        assert worker._load_block_ids == {}
        assert worker._stage3_finished_loads_pending_cleanup == set()
        assert worker._stage3_terminal_loads == set()
        assert worker._stage3_source_req_ids == {}
        assert worker._stage3_destination_req_ids == {}
        assert worker.get_block_ids_with_load_errors() == expected_errors
        assert worker.get_finished() == (set(), set())

    def test_register_runner_keeps_legacy_path_when_admission_disabled(self):
        worker = _make_raiden_worker(tp_rank=0, tp_size=1)
        runner = SimpleNamespace(
            kv_caches=[torch.empty((4, 8), dtype=torch.bfloat16)])
        engine = _FakeRaidenEngine()
        worker._construct_raiden_transfer_engine = MagicMock(
            return_value=engine)

        with patch(f"{_MOD}.tpu_envs.TPU_USE_RAIDEN_KV_CACHE_MANAGER",
                   False,
                   create=True), patch(
                       f"{_MOD}.tpu_envs.TPU_RAIDEN_QWEN35_ADMISSION",
                       False,
                       create=True):
            worker.register_runner(runner)

        call = worker._construct_raiden_transfer_engine.call_args
        assert len(call.args[0]) == 1
        assert call.args[0][0] is runner.kv_caches[0]
        assert call.kwargs == {}
        assert worker._raiden_transfer_engine is engine
        assert worker.raiden_admission_summary() == {"admitted": False}

    def test_v1_admission_rejects_the_v2_tp2dp4_decode_topology(self):
        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=2,
                                     is_producer=False,
                                     dp_size=4,
                                     pcp_size=1)

        with unittest.TestCase().assertRaisesRegex(ValueError,
                                                   "dp8_decode.*requires"):
            worker._raiden_qwen35_admission_topology()

    @pytest.mark.parametrize("dp_size", [4, 8])
    def test_v1_admission_accepts_dp_prefill_topology(self, dp_size):
        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=1,
                                     is_producer=True,
                                     dp_size=dp_size,
                                     pcp_size=1)

        assert worker._raiden_qwen35_admission_topology() == (
            f"dp{dp_size}_prefill")

    def test_producer_registers_sends_with_raiden(self):
        meta = TPUConnectorMetadata()
        meta.reqs_to_send["req"] = MagicMock(uuid=123, local_block_ids=[7, 8])

        self.worker.process_send_load(meta)

        assert self.engine.calls == [("register_read", "req", 123, [7, 8])]

    def test_consumer_submits_loads_to_rank_endpoint(self):
        worker = _make_raiden_worker(is_producer=False)
        worker._raiden_transfer_engine = self.engine
        meta = TPUConnectorMetadata()
        meta.reqs_to_load["req"] = LoadMeta(uuid=5,
                                            local_block_ids=[1],
                                            remote_block_ids=[9],
                                            remote_host="10.1.2.3",
                                            remote_port=9200)

        worker.process_send_load(meta)

        assert self.engine.calls == [("start_read", "req", 5, "10.1.2.3:9202",
                                      [9], [1])]

    def test_consumer_releases_cleared_remote_metadata(self):
        worker = _make_raiden_worker(is_producer=False)
        worker._raiden_transfer_engine = self.engine
        meta = TPUConnectorMetadata()
        meta.reqs_to_load["req"] = LoadMeta(uuid=5,
                                            local_block_ids=None,
                                            remote_block_ids=None,
                                            remote_host="10.1.2.3",
                                            remote_port=9200)

        worker.process_send_load(meta)

        assert self.engine.calls == [("start_read", "req", 5, "10.1.2.3:9202",
                                      [], [])]

    def test_consumer_release_read_reports_completion_by_default(self):
        # A release-only read whose request IS waiting on this connector
        # (prefix hit with num_external_tokens > 0) keeps reporting.
        worker = _make_raiden_worker(is_producer=False)
        engine = _FakeRaidenEngine()
        engine.poll_results = [([], ["req"], [])]
        worker._raiden_transfer_engine = engine
        meta = TPUConnectorMetadata()
        meta.reqs_to_load["req"] = LoadMeta(uuid=5,
                                            local_block_ids=None,
                                            remote_block_ids=None,
                                            remote_host="10.1.2.3",
                                            remote_port=9200)

        worker.process_send_load(meta)

        assert worker.get_finished() == (set(), {"req"})

    def test_consumer_suppressed_release_read_not_reported(self):
        # report_completion=False (another MultiConnector child owns the
        # load): the empty read is still issued to release P's payload, but
        # its completion must never surface as finished_recving — vLLM's
        # scheduler asserts on a finished_recving req that isn't
        # WAITING_FOR_REMOTE_KVS.
        worker = _make_raiden_worker(is_producer=False)
        engine = _FakeRaidenEngine()
        engine.poll_results = [([], ["req"], [])]
        worker._raiden_transfer_engine = engine
        meta = TPUConnectorMetadata()
        meta.reqs_to_load["req"] = LoadMeta(uuid=5,
                                            local_block_ids=None,
                                            remote_block_ids=None,
                                            remote_host="10.1.2.3",
                                            remote_port=9200,
                                            report_completion=False)

        worker.process_send_load(meta)

        assert engine.calls == [("start_read", "req", 5, "10.1.2.3:9202", [],
                                 [])]
        assert worker.get_finished() == (set(), set())
        # The suppression entry is consumed once the recv completion lands.
        assert worker._suppress_done_recving == set()

    def test_finished_req_ids_prune_suppression_state(self):
        # A suppressed request whose recv never surfaces (or that finished
        # generating first) must not leak bookkeeping forever.
        worker = _make_raiden_worker(is_producer=False)
        engine = _FakeRaidenEngine()
        engine.poll_results = [([], [], [])]
        worker._raiden_transfer_engine = engine
        worker._suppress_done_recving.add("req")

        worker.get_finished(finished_req_ids={"req"})

        assert worker._suppress_done_recving == set()

    def test_get_finished_returns_engine_sets(self):
        assert self.worker.get_finished() == ({"sent"}, {"recv"})
        assert self.engine.calls == [("poll_stats", )]

    def test_consumer_waits_for_submitted_load_completion(self):
        worker = _make_raiden_worker(is_producer=False)
        engine = _FakeRaidenEngine()
        engine.poll_results = [([], [], []), ([], ["req"], [])]
        worker._raiden_transfer_engine = engine
        meta = TPUConnectorMetadata()
        meta.reqs_to_load["req"] = LoadMeta(uuid=5,
                                            local_block_ids=[1],
                                            remote_block_ids=[9],
                                            remote_host="10.1.2.3",
                                            remote_port=9200)

        worker.process_send_load(meta, wait_for_completion=True)

        assert engine.calls == [
            ("start_read", "req", 5, "10.1.2.3:9202", [9], [1]),
            ("poll_stats", ),
            ("poll_stats", ),
        ]
        assert worker.get_finished() == (set(), {"req"})

    def test_consumer_wait_can_suppress_completion_report(self):
        worker = _make_raiden_worker(is_producer=False)
        engine = _FakeRaidenEngine()
        engine.poll_results = [([], [], []), ([], ["req"], [])]
        worker._raiden_transfer_engine = engine
        meta = TPUConnectorMetadata()
        meta.reqs_to_load["req"] = LoadMeta(uuid=5,
                                            local_block_ids=[1],
                                            remote_block_ids=[9],
                                            remote_host="10.1.2.3",
                                            remote_port=9200)

        worker.process_send_load(meta,
                                 wait_for_completion=True,
                                 report_completion=False)

        assert worker.get_finished() == (set(), set())

    def test_num_slots_can_be_overridden(self):
        runner = MagicMock()
        runner.kv_caches = [torch.empty((128, 2), dtype=torch.bfloat16)]
        self.worker.runner = runner
        with patch(f"{_MOD}.dist_utils.get_raiden_transfer_num_slots",
                   return_value=3):
            assert self.worker._num_raiden_slots(max_blocks=4) == 3

    def test_num_slots_autosizes_sparse_mla_tuples(self):
        runner = MagicMock()
        nope = torch.empty((4, 8, 4, 128), dtype=torch.uint8)
        rope = torch.empty((4, 2, 4, 128), dtype=torch.uint8)
        idx = torch.empty((4, 2, 4, 256), dtype=torch.uint8)
        runner.kv_caches = [(nope, rope), idx]
        self.worker.runner = runner
        with patch(f"{_MOD}.dist_utils.get_raiden_transfer_num_slots",
                   return_value=0), \
             patch(f"{_MOD}.dist_utils.get_kv_shm_pool_gb",
                   return_value=1.0):
            # Per-block bytes: nope 8*4*128=4096, rope 2*4*128=1024,
            # idx 2*4*256=2048 -> 7168; x 4 blocks = 28672 per slot;
            # budget 1 GiB / tp_size 4.
            assert self.worker._num_raiden_slots(
                max_blocks=4) == ((1024**3 // 4) // 28672) * self.worker.chunk_transfer_window

    def test_dp_port_configurations(self):
        worker = _make_raiden_worker(dp_rank=0, tp_size=1)
        assert worker.kv_transfer_port == 9100

        worker = _make_raiden_worker(dp_rank=1, tp_size=1)
        assert worker.kv_transfer_port == 9102

        worker = _make_raiden_worker(dp_rank=0, tp_size=4)
        assert worker.kv_transfer_port == 9100

        worker = _make_raiden_worker(dp_rank=1, tp_size=4)
        assert worker.kv_transfer_port == 9108

    def test_raiden_worker_queue_length_stats_producer_stage3(self):
        worker = _make_raiden_worker(tp_rank=0, is_producer=True)
        worker._stage3_registered_sends = {
            "req1": MagicMock(),
            "req2": MagicMock(),
            "req3": MagicMock(),
        }
        with patch.object(worker, "_raiden_stage3_enabled", return_value=True):
            stats = worker.get_kv_connector_stats()
            assert stats is not None
            assert stats.data["prefill_queue_length"] == [3]

    def test_raiden_worker_queue_length_stats_producer_legacy(self):
        worker = _make_raiden_worker(tp_rank=0, is_producer=True)
        worker._legacy_registered_sends = {"req1", "req2"}
        with patch.object(worker, "_raiden_stage3_enabled",
                          return_value=False):
            stats = worker.get_kv_connector_stats()
            assert stats is not None
            assert stats.data["prefill_queue_length"] == [2]

    def test_raiden_worker_queue_length_stats_consumer_stage3(self):
        worker = _make_raiden_worker(tp_rank=0, is_producer=False)
        worker._stage3_submitted_loads = {
            "req1": 1,
            "req2": 1,
            "req3": 1,
            "req4": 1,
            "req5": 1,
        }
        worker._stage3_terminal_loads = {"req1", "req2"}
        with patch.object(worker, "_raiden_stage3_enabled", return_value=True):
            stats = worker.get_kv_connector_stats()
            assert stats is not None
            assert stats.data["decode_queue_length"] == [3]

    def test_raiden_worker_queue_length_stats_consumer_legacy(self):
        worker = _make_raiden_worker(tp_rank=0, is_producer=False)
        worker._legacy_submitted_loads = {"req1", "req2", "req3", "req4"}
        with patch.object(worker, "_raiden_stage3_enabled",
                          return_value=False):
            stats = worker.get_kv_connector_stats()
            assert stats is not None
            assert stats.data["decode_queue_length"] == [4]

    def test_raiden_worker_queue_length_stats_non_zero_tp_rank(self):
        worker = _make_raiden_worker(tp_rank=1, is_producer=True)
        worker._stage3_registered_sends = {"req1": MagicMock()}
        with patch.object(worker, "_raiden_stage3_enabled", return_value=True):
            stats = worker.get_kv_connector_stats()
            assert stats is None

    def test_poll_finished_cleans_legacy_registered_sends_producer(self):
        worker = _make_raiden_worker(is_producer=True)
        worker._legacy_registered_sends = {"req1", "req2", "req3", "req4"}
        engine = MagicMock()
        engine.poll_stats.return_value = (["req1"], [], ["req2"])
        with patch.object(worker, "_raiden_stage3_enabled",
                          return_value=False):
            worker._poll_finished(engine)
            assert worker._legacy_registered_sends == {"req3", "req4"}

    def test_poll_finished_cleans_legacy_submitted_loads_consumer(self):
        worker = _make_raiden_worker(is_producer=False)
        worker._legacy_submitted_loads = {"req1", "req2", "req3", "req4"}
        engine = MagicMock()
        engine.poll_stats.return_value = ([], ["req1"], ["req2"])
        with patch.object(worker, "_raiden_stage3_enabled",
                          return_value=False):
            worker._poll_finished(engine)
            assert worker._legacy_submitted_loads == {"req3", "req4"}


# ---------------------------------------------------------------------------
# TestTPUConnectorWorkerInit — construction-time state
# ---------------------------------------------------------------------------


class TestTPUConnectorWorkerInit:
    """Verify the coordinator state set up by __init__ / _init_coord_state.

    register_runner() is not called; all shm/ZMQ/thread bringup is skipped.
    """

    def test_tp1_coord_workers_ready_is_set_eagerly(self):
        # With a single rank there are no peers to wait for.
        worker = _make_worker(tp_rank=0, tp_size=1)
        assert worker._coord_workers_ready.is_set()

    def test_tp2_coord_workers_ready_not_yet_set(self):
        worker = _make_worker(tp_rank=0, tp_size=2)
        assert not worker._coord_workers_ready.is_set()

    def test_n_channels_auto_equals_tp_size(self):
        # get_transfer_channel_number returns 0 → auto → n_channels == tp_size
        worker = _make_worker(tp_rank=0, tp_size=4)
        assert worker._n_channels == 4

    def test_n_channels_override_clamped_to_tp_size(self):
        # Requesting 16 channels on a TP=4 worker is clamped to 4.
        with patch(f"{_BASE}.get_tensor_model_parallel_rank", return_value=0), \
             patch(f"{_BASE}.get_tensor_model_parallel_world_size", return_value=4), \
             patch(f"{_BASE}.dist_utils.get_node_id", return_value=0), \
             patch(f"{_BASE}.dist_utils.get_host_ip", return_value="127.0.0.1"), \
             patch(f"{_BASE}.dist_utils.get_kv_transfer_port", return_value="9100"), \
             patch(f"{_BASE}.dist_utils.get_side_channel_port", return_value="9600"), \
             patch(f"{_BASE}.dist_utils.get_transfer_channel_number", return_value=16), \
             patch(f"{_BASE}.dist_utils.get_kv_latency_log_interval", return_value=0.0), \
             patch(f"{_BASE}.zmq.Context"):
            worker = TPUConnectorWorker(_make_vllm_config())
        assert worker._n_channels == 4

    def test_kv_scatter_disabled_until_smoke_test(self):
        # Scatter is only enabled after register_runner runs the smoke test.
        worker = _make_worker()
        assert not worker._kv_scatter_enabled

    def test_kv_transfer_ports_span_n_channels(self):
        worker = _make_worker(tp_rank=0, tp_size=3)
        base = int(worker.kv_transfer_port)
        assert worker._kv_transfer_ports == [base, base + 1, base + 2]

    def test_dp_port_configurations(self):
        worker = _make_worker(dp_rank=0, tp_size=1)
        assert worker.kv_transfer_port == 9100
        assert worker.side_channel_port == 9600

        worker = _make_worker(dp_rank=1, tp_size=1)
        assert worker.kv_transfer_port == 9101
        assert worker.side_channel_port == 9601

        worker = _make_worker(dp_rank=0, tp_size=4)
        assert worker.kv_transfer_port == 9100
        assert worker.side_channel_port == 9600

        worker = _make_worker(dp_rank=1, tp_size=4)
        assert worker.kv_transfer_port == 9104
        assert worker.side_channel_port == 9601


# ---------------------------------------------------------------------------
# TestKVCacheReplacement — runner / forward-context consistency
# ---------------------------------------------------------------------------


class TestKVCacheReplacement:

    def test_replace_runner_kv_cache_updates_bound_attention_layers(self):
        worker = _make_worker(tp_rank=0, tp_size=1, is_producer=False)

        old_cache0 = torch.empty(1)
        old_cache1 = torch.empty(1)
        new_cache0 = torch.ones(1)

        runner = MagicMock()
        runner.kv_caches = [old_cache0, old_cache0, old_cache1]
        worker.runner = runner

        layer0 = MagicMock()
        layer0.kv_cache = old_cache0
        shared_layer0 = MagicMock()
        shared_layer0.kv_cache = old_cache0
        layer1 = MagicMock()
        layer1.kv_cache = old_cache1
        worker.vllm_config.compilation_config.static_forward_context = {
            "layer.0": layer0,
            "layer.0.shared": shared_layer0,
            "layer.1": layer1,
            "non_attention": object(),
        }

        worker._replace_runner_kv_cache(0, new_cache0)

        assert runner.kv_caches[0] is new_cache0
        assert runner.kv_caches[1] is new_cache0
        assert layer0.kv_cache is new_cache0
        assert shared_layer0.kv_cache is new_cache0
        assert runner.kv_caches[2] is old_cache1
        assert layer1.kv_cache is old_cache1


# ---------------------------------------------------------------------------
# TestResolveRemoteHostPort — multi-host routing helper
# ---------------------------------------------------------------------------


class TestResolveRemoteHostPort:
    """_resolve_remote_host_port picks the right host/port for this node_id."""

    def setup_method(self):
        self.worker = _make_worker(tp_rank=0, tp_size=1)

    def test_single_host_passthrough(self):
        meta = LoadMeta(uuid=1,
                        local_block_ids=[0],
                        remote_block_ids=[0],
                        remote_host="1.2.3.4",
                        remote_port=9100)
        host, port, side_channel_port = self.worker._resolve_remote_host_port(
            meta)
        assert host == "1.2.3.4"
        assert port == 9100
        assert side_channel_port is None

    def test_multi_host_selects_node_id_entry(self):
        # node_id=0 (set in _make_worker) → picks index 0
        meta = LoadMeta(uuid=2,
                        local_block_ids=[0],
                        remote_block_ids=[0],
                        remote_host=["10.0.0.1", "10.0.0.2"],
                        remote_port=[9100, 9101])
        host, port, side_channel_port = self.worker._resolve_remote_host_port(
            meta)
        assert host == "10.0.0.1"
        assert port == 9100
        assert side_channel_port is None

    def test_multi_host_side_channel_port(self):
        meta = LoadMeta(uuid=3,
                        local_block_ids=[0],
                        remote_block_ids=[0],
                        remote_host=["10.0.0.1", "10.0.0.2"],
                        remote_port=[9100, 9101],
                        remote_side_channel_port=9200)
        host, port, side_channel_port = self.worker._resolve_remote_host_port(
            meta)
        assert host == "10.0.0.1"
        assert port == 9100
        assert side_channel_port == 9200


# ---------------------------------------------------------------------------
# TestCoordGetFinished — _coord_get_finished state-machine tests
# ---------------------------------------------------------------------------


class TestCoordGetFinished:
    """Directly inject coordinator state and verify the returned finished sets.

    These tests cover the six state transitions in _coord_get_finished without
    needing a running ZMQ socket, shm pool, or real TPU.
    """

    def setup_method(self):
        self.worker = _make_worker(tp_rank=0, tp_size=1, is_producer=True)
        # _coord_pool is None until register_runner; mock it so release_slot
        # calls don't crash.
        self.worker._coord_pool = MagicMock()

    # ---- non-zero rank short-circuit ---------------------------------------

    def test_non_zero_rank_returns_empty_sets(self):
        worker = _make_worker(tp_rank=1, tp_size=2)
        done_sending, done_recving = worker._coord_get_finished()
        assert done_sending == set()
        assert done_recving == set()

    # ---- producer / done_sending -------------------------------------------

    def test_pull_acked_entry_moves_to_done_sending(self):
        entry = _CoordSendEntry(req_id="req-acked",
                                slot_idx=0,
                                num_blocks=4,
                                expiration_time=1e9,
                                pull_acked=True)
        self.worker._coord_send[10] = entry

        done_sending, done_recving = self.worker._coord_get_finished()

        assert done_sending == {"req-acked"}
        assert done_recving == set()
        assert 10 not in self.worker._coord_send
        self.worker._coord_pool.release_slot.assert_called_once_with(0)

    def test_unacked_entry_stays_in_coord_send(self):
        entry = _CoordSendEntry(req_id="req-pending",
                                slot_idx=1,
                                num_blocks=4,
                                expiration_time=1e9,
                                pull_acked=False)
        self.worker._coord_send[20] = entry

        done_sending, _ = self.worker._coord_get_finished()

        assert done_sending == set()
        assert 20 in self.worker._coord_send
        self.worker._coord_pool.release_slot.assert_not_called()

    def test_expiry_sweep_results_drained_once(self):
        """Entries written by the expiration-sweeper thread are merged in and
        cleared so they aren't double-reported on the next tick."""
        self.worker._coord_done_sending = {"swept-req"}

        done_sending, _ = self.worker._coord_get_finished()

        assert done_sending == {"swept-req"}
        assert len(self.worker._coord_done_sending) == 0

    # ---- consumer / done_recving -------------------------------------------

    def test_load_complete_pull_ok_moves_to_done_recving(self):
        ev = threading.Event()
        ev.set()
        entry = _CoordRecvEntry(req_id="req-pulled",
                                uuid=55,
                                slot_idx=2,
                                num_blocks=4,
                                local_blocks=[0, 1, 2, 3],
                                remote_blocks=[10, 11, 12, 13],
                                remote_host="1.2.3.4",
                                remote_port=9100,
                                load_complete=ev,
                                pull_ok=True,
                                reported_done=False)
        self.worker._coord_recv[55] = entry

        _, done_recving = self.worker._coord_get_finished()

        assert done_recving == {"req-pulled"}
        assert entry.reported_done is True

    def test_incomplete_pull_not_reported(self):
        ev = threading.Event()  # not set
        entry = _CoordRecvEntry(req_id="req-in-flight",
                                uuid=56,
                                slot_idx=3,
                                num_blocks=4,
                                local_blocks=[],
                                remote_blocks=[],
                                remote_host="1.2.3.4",
                                remote_port=9100,
                                load_complete=ev,
                                pull_ok=False,
                                reported_done=False)
        self.worker._coord_recv[56] = entry

        _, done_recving = self.worker._coord_get_finished()

        assert done_recving == set()
        assert not entry.reported_done

    def test_reported_done_prevents_duplicate_reporting(self):
        ev = threading.Event()
        ev.set()
        entry = _CoordRecvEntry(
            req_id="req-already-reported",
            uuid=57,
            slot_idx=4,
            num_blocks=4,
            local_blocks=[],
            remote_blocks=[],
            remote_host="1.2.3.4",
            remote_port=9100,
            load_complete=ev,
            pull_ok=True,
            reported_done=True)  # already reported on a prior tick
        self.worker._coord_recv[57] = entry

        _, done_recving = self.worker._coord_get_finished()

        assert done_recving == set()

    def test_failure_recving_surfaced_and_cleared(self):
        """Requests in _coord_done_recving (pool exhausted / pull failed) are
        surfaced exactly once so the scheduler doesn't hang waiting for them."""
        self.worker._coord_done_recving = {"failed-req"}

        _, done_recving = self.worker._coord_get_finished()

        assert done_recving == {"failed-req"}
        assert len(self.worker._coord_done_recving) == 0

    # ---- combined tick -----------------------------------------------------

    def test_send_and_recv_reported_in_same_tick(self):
        entry_send = _CoordSendEntry(req_id="sending-req",
                                     slot_idx=5,
                                     num_blocks=2,
                                     expiration_time=1e9,
                                     pull_acked=True)
        self.worker._coord_send[100] = entry_send

        ev = threading.Event()
        ev.set()
        entry_recv = _CoordRecvEntry(req_id="recving-req",
                                     uuid=101,
                                     slot_idx=6,
                                     num_blocks=2,
                                     local_blocks=[0, 1],
                                     remote_blocks=[10, 11],
                                     remote_host="1.2.3.4",
                                     remote_port=9100,
                                     load_complete=ev,
                                     pull_ok=True,
                                     reported_done=False)
        self.worker._coord_recv[101] = entry_recv

        done_sending, done_recving = self.worker._coord_get_finished()

        assert done_sending == {"sending-req"}
        assert done_recving == {"recving-req"}


class TestTPUConnectorStats(unittest.TestCase):

    def setUp(self):
        self.registry = CollectorRegistry()
        metric_types = {
            Gauge: partial(Gauge, registry=self.registry),
            Counter: partial(Counter, registry=self.registry),
            Histogram: partial(Histogram, registry=self.registry),
        }
        labelnames = ["model_name", "engine"]
        per_engine_labelvalues = {0: ["my_model", "0"]}
        self.metrics = TpuKVConnectorPromMetrics(
            vllm_config=MagicMock(),
            metric_types=metric_types,
            labelnames=labelnames,
            per_engine_labelvalues=per_engine_labelvalues)

        mock_data = {
            "d2h_transfer_time": [100.0, 200.0, 300.0],
            "h2d_transfer_time": [202.0, 303.0, 404.0],
            "kv_pull_time": [1200.0, 5400.0, 12000.0],
            "mb_transferred": [128.0, 256.0, 2048.0],
            "num_failed_transfers": [0, 1, 2],
        }

        self.metrics.observe(mock_data, engine_idx=0)

    def validate_prometheus_histogram_buckets(self, hist, num_buckets,
                                              non_zero_buckets):
        assert len(
            hist._buckets
        ) == num_buckets, f"Incorrect number of buckets returned: expected {num_buckets} actual {len(hist._buckets)}"
        for i in range(num_buckets):
            if i in non_zero_buckets:
                assert hist._buckets[i].get() == non_zero_buckets[
                    i], f"Incorrect value for bucket {i}: expected {non_zero_buckets[i]} actual: {hist._buckets[i].get()}"
            else:
                assert hist._buckets[i].get(
                ) == 0, f"Incorrect value for bucket {i}: expected 0 actual: {hist._buckets[i].get()}"

    def test_tpu_stats_aggregation_d2h_transfer(self):
        stats = TpuKVConnectorStats()

        reduced = stats.reduce()
        assert reduced["Avg D2H transfer time (ms)"] == 0.0
        assert reduced["P90 D2H transfer time (ms)"] == 0.0
        assert stats.is_empty() is True

        for i in range(10):
            stats.record_d2h_transfer(d2h_transfer_time=200.0 + i)
        reduced = stats.reduce()

        assert reduced["Avg D2H transfer time (ms)"] == 204.5
        assert reduced["P90 D2H transfer time (ms)"] == 208.1
        assert stats.is_empty() is False

    def test_tpu_stats_aggregation_h2d_transfer(self):
        stats = TpuKVConnectorStats()

        reduced = stats.reduce()
        assert reduced["Avg H2D transfer time (ms)"] == 0.0
        assert reduced["P90 H2D transfer time (ms)"] == 0.0
        assert stats.is_empty() is True

        for i in range(10):
            stats.record_h2d_transfer(h2d_transfer_time=300.0 + i)
        reduced = stats.reduce()

        assert reduced["Avg H2D transfer time (ms)"] == 304.5
        assert reduced["P90 H2D transfer time (ms)"] == 308.1
        assert stats.is_empty() is False

    def test_tpu_stats_aggregation_kv_pull(self):
        stats = TpuKVConnectorStats()

        reduced = stats.reduce()
        assert reduced["Avg KV pull time (ms)"] == 0.0
        assert reduced["P90 KV pull time (ms)"] == 0.0
        assert stats.is_empty() is True

        for i in range(10):
            stats.record_kv_pull(kv_pull_time=300.0 + i)
        reduced = stats.reduce()

        assert reduced["Avg KV pull time (ms)"] == 304.5
        assert reduced["P90 KV pull time (ms)"] == 308.1
        assert stats.is_empty() is False

    def test_tpu_stats_aggregation_mb_transferred(self):
        stats = TpuKVConnectorStats()

        reduced = stats.reduce()
        assert reduced["Avg MB per transfer"] == 0.0
        assert stats.is_empty() is True

        for i in range(10):
            stats.record_mb_transferred(mb_transferred=20 + i)
        reduced = stats.reduce()

        assert reduced["Avg MB per transfer"] == 24.5
        assert stats.is_empty() is False

    def test_tpu_stats_aggregation_failed_transfer(self):
        stats = TpuKVConnectorStats()

        reduced = stats.reduce()
        assert sum(reduced["Num failed transfers"]) == 0
        assert stats.is_empty() is True

        for i in range(10):
            stats.record_failed_transfer()
        reduced = stats.reduce()

        assert sum(reduced["Num failed transfers"]) == 10
        assert stats.is_empty() is False

    def test_prometheus_histogram_d2h_transfer_time(self):
        hist = self.metrics.tpu_histogram_d2h_transfer_time[0]
        assert hist._sum.get() == 600.0
        num_buckets = 14
        non_zero_buckets = {
            2: 1.0,
            3: 1.0,
            4: 1.0,
        }
        self.validate_prometheus_histogram_buckets(hist, num_buckets,
                                                   non_zero_buckets)

    def test_prometheus_histogram_h2d_transfer_time(self):
        hist = self.metrics.tpu_histogram_h2d_transfer_time[0]
        assert hist._sum.get() == 909.0
        num_buckets = 14
        non_zero_buckets = {
            3: 1.0,
            4: 2.0,
        }
        self.validate_prometheus_histogram_buckets(hist, num_buckets,
                                                   non_zero_buckets)

    def test_prometheus_histogram_kv_pull_time(self):
        hist = self.metrics.tpu_histogram_kv_pull_time[0]
        assert hist._sum.get() == 18600.0
        num_buckets = 14
        non_zero_buckets = {
            7: 1.0,
            9: 1.0,
            11: 1.0,
        }
        self.validate_prometheus_histogram_buckets(hist, num_buckets,
                                                   non_zero_buckets)

    def test_prometheus_histogram_kv_megabytes_transferred(self):
        hist = self.metrics.tpu_histogram_kv_megabytes_transferred[0]
        assert hist._sum.get() == 2432.0
        num_buckets = 9
        non_zero_buckets = {
            2: 1.0,
            3: 1.0,
            6: 1.0,
        }
        self.validate_prometheus_histogram_buckets(hist, num_buckets,
                                                   non_zero_buckets)

    def test_prometheus_counter_num_failed_transfers(self):
        counter = self.metrics.counter_tpu_num_failed_transfers[0]
        assert counter._value.get() == 3.0

    def test_tpu_stats_aggregation_queue_lengths(self):
        stats = TpuKVConnectorStats()
        reduced = stats.reduce()
        assert reduced["Prefill queue length"] == 0
        assert reduced["Decode queue length"] == 0
        assert stats.is_empty() is True

        stats.record_prefill_queue_length(5)
        stats.record_decode_queue_length(3)
        reduced = stats.reduce()
        assert reduced["Prefill queue length"] == 5
        assert reduced["Decode queue length"] == 3
        assert stats.is_empty() is False

    def test_prometheus_gauge_queue_lengths(self):
        mock_data = {
            "d2h_transfer_time": [],
            "h2d_transfer_time": [],
            "kv_pull_time": [],
            "mb_transferred": [],
            "num_failed_transfers": [],
            "prefill_queue_length": [4],
            "decode_queue_length": [2],
        }
        self.metrics.observe(mock_data, engine_idx=0)
        prefill_gauge = self.metrics.gauge_tpu_prefill_queue_length[0]
        decode_gauge = self.metrics.gauge_tpu_decode_queue_length[0]
        assert prefill_gauge._value.get() == 4.0
        assert decode_gauge._value.get() == 2.0

    def test_worker_in_flight_queue_length_stats(self):
        worker = MagicMock()
        worker._is_host_coordinator = True
        worker.is_producer = False
        worker._coord_recv = {"uuid1": MagicMock(), "uuid2": MagicMock()}
        worker._coord_lock = MagicMock()
        worker.transfer_stats = TpuKVConnectorStats()

        stats = TPUConnectorWorker.get_kv_connector_stats(worker)
        assert stats is not None
        assert stats.data["decode_queue_length"] == [2]

        worker.is_producer = True
        worker._coord_send = {"uuid1": MagicMock()}
        worker.transfer_stats = TpuKVConnectorStats()

        stats = TPUConnectorWorker.get_kv_connector_stats(worker)
        assert stats is not None
        assert stats.data["prefill_queue_length"] == [1]


# ---------------------------------------------------------------------------
# GLM-5.2 admission topology: prefill tp{N}/dp1 -> decode tp1/dp{M} only.
# ---------------------------------------------------------------------------


def test_glm_tag_geometry_measures_manifest_in_fixed_tag_order():
    from vllm_torchtpu.distributed.kv_transfer.raiden import \
        pool_manifest as rpm

    from .raiden_test_utils import glm_named_kv_caches
    manifest = rpm.build_glm_mla_pool_manifest(
        named_kv_caches=glm_named_kv_caches(),
        raw_tensors=(),
        block_size_tokens=1024)
    geometry = TPURaidenConnectorWorker._measure_glm_tag_geometry(manifest)
    assert list(geometry) == ["mla.nope", "mla.rope", "dsa.idx"]
    assert geometry["mla.nope"] == (1024 * 512, 512)
    assert geometry["mla.rope"] == (256 * 512, 512)
    assert geometry["dsa.idx"] == (256 * 1024, 1024)


@pytest.mark.parametrize(("is_producer", "tp_size", "dp_size", "expected"), [
    (True, 8, 1, "tp8_prefill"),
    (False, 1, 8, "dp8_decode"),
    (False, 1, 1, "dp1_decode"),
])
def test_glm_admission_topology_accepts_supported_shapes(
        is_producer, tp_size, dp_size, expected):
    worker = _make_raiden_worker(tp_rank=0,
                                 tp_size=tp_size,
                                 is_producer=is_producer,
                                 dp_size=dp_size,
                                 pcp_size=1)

    assert worker._raiden_glm_admission_topology() == expected


@pytest.mark.parametrize(
    ("is_producer", "tp_size", "dp_size", "pcp_size", "message"),
    [
        # A TP>1 decode engine would need one plan per destination worker
        (False, 4, 2, 1, "consumer tensor_parallel_size=4"),
        # A DP>1 prefill engine has no single source unit set.
        (True, 8, 2, 1, "producer tensor_parallel_size=8"),
        # PCP is not supported on either role.
        (False, 1, 8, 2, "prefill_context_parallel_size=2"),
    ])
def test_glm_admission_topology_rejects_unsupported_shapes(
        is_producer, tp_size, dp_size, pcp_size, message):
    worker = _make_raiden_worker(tp_rank=0,
                                 tp_size=tp_size,
                                 is_producer=is_producer,
                                 dp_size=dp_size,
                                 pcp_size=pcp_size)

    with pytest.raises(ValueError, match=message):
        worker._raiden_glm_admission_topology()


# ---------------------------------------------------------------------------
# Shared Chunked Transfer Test Fixtures
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _patch_raiden_stage3_env(
    *,
    base_uuid: int = 12345,
    parallelism: int = 4,
    timeout_s: float = 30.0,
    controller_address: str = "127.0.0.1:27000",
    job_name: str = "prefill",
    engine_id: str = "producer-0",
) -> Iterator[None]:
    with (
        patch(f"{_MOD}._use_raiden_stage3_transport", return_value=True),
        patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT", "raiden"),
        patch(
            f"{_MOD}.tpu_envs.TPU_RAIDEN_CONTROLLER_ADDRESS",
            controller_address,
        ),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_JOB_NAME", job_name),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_ENGINE_ID", engine_id),
        patch(
            f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM",
            parallelism,
        ),
        patch(
            f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout",
            return_value=timeout_s,
        ),
        patch(f"{_MOD}.get_uuid", return_value=base_uuid),
    ):
        yield

def _make_mock_vllm_config(
    *,
    is_producer: bool = True,
    block_size: int = 1024,
    tp_size: int = 1,
    pcp_size: int = 4,
    interleave_size: int = 256,
) -> MagicMock:
    return _make_vllm_config(
        is_producer=is_producer,
        block_size=block_size,
        tp_size=tp_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    )


# ---------------------------------------------------------------------------
# Metadata & Serialization Schema for Chunked Transfers
# ---------------------------------------------------------------------------

def test_send_meta_defaults() -> None:
    """Verifies SendMeta defaults for legacy compatibility."""
    meta = SendMeta(uuid=12345, local_block_ids=[10, 11, 12], expiration_time=100.0)
    assert meta.uuid == 12345
    assert meta.local_block_ids == [10, 11, 12]
    assert meta.expiration_time == 100.0
    assert meta.num_tokens is None
    assert meta.mamba_state_block_ids is None
    assert meta.token_base == 0
    assert meta.is_final is True

def test_send_meta_subchunk_fields() -> None:
    """Verifies SendMeta instantiation with sub-chunk coordinates."""
    meta = SendMeta(uuid=67890, local_block_ids=[20, 21], expiration_time=200.0, num_tokens=1024, mamba_state_block_ids=None, token_base=1024, is_final=False)
    assert meta.uuid == 67890
    assert meta.local_block_ids == [20, 21]
    assert meta.expiration_time == 200.0
    assert meta.num_tokens == 1024
    assert meta.mamba_state_block_ids is None
    assert meta.token_base == 1024
    assert meta.is_final is False

def test_load_meta_defaults_and_subchunk_fields() -> None:
    """Verifies LoadMeta defaults and sub-chunk coordinates."""
    meta_default = LoadMeta(uuid=11111, local_block_ids=[1, 2], remote_block_ids=[3, 4], remote_host='10.0.0.1', remote_port=5000)
    assert meta_default.uuid == 11111
    assert meta_default.local_block_ids == [1, 2]
    assert meta_default.remote_block_ids == [3, 4]
    assert meta_default.remote_host == '10.0.0.1'
    assert meta_default.remote_port == 5000
    assert meta_default.remote_side_channel_port is None
    assert meta_default.report_completion is True
    assert meta_default.token_base == 0
    assert meta_default.is_final is True
    meta_chunked = LoadMeta(uuid=22222, local_block_ids=[5, 6], remote_block_ids=[7, 8], remote_host='10.0.0.2', remote_port=5001, remote_side_channel_port=6000, report_completion=False, token_base=2048, is_final=False)
    assert meta_chunked.token_base == 2048
    assert meta_chunked.is_final is False
    assert meta_chunked.report_completion is False

def test_stage3_registered_send_schema() -> None:
    """Verifies _Stage3RegisteredSend chunk coordinates and defaults."""
    reg_default = _Stage3RegisteredSend(uuid=33333, local_block_ids=(100, 101), num_tokens=4096, expiration_time=500.0)
    assert reg_default.uuid == 33333
    assert reg_default.local_block_ids == (100, 101)
    assert reg_default.num_tokens == 4096
    assert reg_default.expiration_time == 500.0
    assert reg_default.token_base == 0
    assert reg_default.is_final is True
    reg_chunked = _Stage3RegisteredSend(uuid=44444, local_block_ids=(102, 103), num_tokens=1024, expiration_time=600.0, token_base=1024, is_final=False)
    assert reg_chunked.token_base == 1024
    assert reg_chunked.is_final is False

def test_metadata_serialization_roundtrip() -> None:
    """Verifies serialization roundtrip for TPUConnectorMetadata."""
    send_meta_1 = SendMeta(uuid=101, local_block_ids=[1, 2], expiration_time=150.0, num_tokens=1024, token_base=0, is_final=False)
    send_meta_2 = SendMeta(uuid=102, local_block_ids=[3, 4], expiration_time=160.0, num_tokens=1024, token_base=1024, is_final=True)
    load_meta_1 = LoadMeta(uuid=201, local_block_ids=[10, 11], remote_block_ids=[1, 2], remote_host='10.0.0.5', remote_port=8000, token_base=0, is_final=False)
    load_meta_2 = LoadMeta(uuid=202, local_block_ids=[12, 13], remote_block_ids=[3, 4], remote_host='10.0.0.5', remote_port=8000, token_base=1024, is_final=True)
    tpu_meta = TPUConnectorMetadata(reqs_to_send={'req-1-chunk-0': send_meta_1, 'req-1-chunk-1': send_meta_2}, reqs_to_load={'req-2-chunk-0': load_meta_1, 'req-2-chunk-1': load_meta_2})
    dict_data = dataclasses.asdict(tpu_meta)
    json_str = json.dumps(dict_data)
    recovered_dict = json.loads(json_str)
    send_dict = recovered_dict['reqs_to_send']
    load_dict = recovered_dict['reqs_to_load']
    assert send_dict['req-1-chunk-0']['token_base'] == 0
    assert send_dict['req-1-chunk-0']['is_final'] is False
    assert send_dict['req-1-chunk-1']['token_base'] == 1024
    assert send_dict['req-1-chunk-1']['is_final'] is True
    assert load_dict['req-2-chunk-0']['token_base'] == 0
    assert load_dict['req-2-chunk-0']['is_final'] is False
    assert load_dict['req-2-chunk-1']['token_base'] == 1024
    assert load_dict['req-2-chunk-1']['is_final'] is True
    reconstructed_sends = {k: SendMeta(**v) for k, v in send_dict.items()}
    reconstructed_loads = {k: LoadMeta(**v) for k, v in load_dict.items()}
    reconstructed_meta = TPUConnectorMetadata(reqs_to_send=reconstructed_sends, reqs_to_load=reconstructed_loads)
    assert reconstructed_meta.reqs_to_send['req-1-chunk-0'].token_base == 0
    assert reconstructed_meta.reqs_to_send['req-1-chunk-0'].is_final is False
    assert reconstructed_meta.reqs_to_send['req-1-chunk-1'].token_base == 1024
    assert reconstructed_meta.reqs_to_send['req-1-chunk-1'].is_final is True
    assert reconstructed_meta.reqs_to_load['req-2-chunk-0'].token_base == 0
    assert reconstructed_meta.reqs_to_load['req-2-chunk-0'].is_final is False
    assert reconstructed_meta.reqs_to_load['req-2-chunk-1'].token_base == 1024
    assert reconstructed_meta.reqs_to_load['req-2-chunk-1'].is_final is True
    pickled = pickle.dumps(tpu_meta)
    unpickled_meta = pickle.loads(pickled)
    assert unpickled_meta.reqs_to_send['req-1-chunk-0'].token_base == 0
    assert unpickled_meta.reqs_to_send['req-1-chunk-0'].is_final is False
    assert unpickled_meta.reqs_to_send['req-1-chunk-1'].token_base == 1024
    assert unpickled_meta.reqs_to_send['req-1-chunk-1'].is_final is True
    assert unpickled_meta.reqs_to_load['req-2-chunk-0'].token_base == 0
    assert unpickled_meta.reqs_to_load['req-2-chunk-0'].is_final is False
    assert unpickled_meta.reqs_to_load['req-2-chunk-1'].token_base == 1024
    assert unpickled_meta.reqs_to_load['req-2-chunk-1'].is_final is True

# ---------------------------------------------------------------------------
# Scheduler Stride-Aligned Chunk Emission & Short-Prompt Bypass
# ---------------------------------------------------------------------------

_MOD = 'vllm_torchtpu.distributed.kv_transfer.tpu_connector'

def test_chunk_uuid_determinism_and_properties() -> None:
    """Verifies chunk_uuid determinism, collision resistance, and range."""
    base_uuid = 987654321
    uuid_0 = chunk_uuid(base_uuid, 0)
    uuid_0_dup = chunk_uuid(base_uuid, 0)
    uuid_1 = chunk_uuid(base_uuid, 1)
    uuid_2 = chunk_uuid(base_uuid, 2)
    other_base_uuid = 987654322
    other_uuid_0 = chunk_uuid(other_base_uuid, 0)
    assert uuid_0 == uuid_0_dup
    assert uuid_0 != uuid_1
    assert uuid_1 != uuid_2
    assert uuid_0 != other_uuid_0
    assert 0 <= uuid_0 < 1 << 50
    assert 0 <= uuid_1 < 1 << 50
    assert 0 <= uuid_2 < 1 << 50

def test_stage3_send_state_dataclass_defaults() -> None:
    """Verifies default values and field assignment in _Stage3SendState."""
    state = _Stage3SendState(base_uuid=12345, block_ids=[1, 2, 3])
    assert state.base_uuid == 12345
    assert state.block_ids == [1, 2, 3]
    assert state.num_computed_tokens == 0
    assert state.tokens_sent == 0
    assert state.next_chunk_idx == 0
    assert state.total_prompt_tokens == 0

def test_scheduler_step_chunk_emission() -> None:
    """Simulate a 4096-token prefill executed in 1024-token steps on PCP=4.

    Verify that build_connector_meta() emits SendMeta([0, 1024), is_final=False)
    on step 1, SendMeta([1024, 2048), is_final=False) on step 2, etc.
    """
    scheduler = _make_raiden_scheduler(is_producer=True, pcp_size=4, interleave_size=256, chunk_transfer_window=4)
    req_id = 'req-4096'
    base_uuid = 123456789
    block_ids = [100, 101, 102, 103, 104, 105, 106, 107]
    scheduler._stage3_send_requests[req_id] = _Stage3SendState(base_uuid=base_uuid, block_ids=block_ids, num_computed_tokens=0, tokens_sent=0, next_chunk_idx=0, total_prompt_tokens=4096)
    sched_out_1 = MagicMock(spec=SchedulerOutput)
    sched_out_1.num_scheduled_tokens = {req_id: 1024}
    meta_1 = scheduler.build_connector_meta(sched_out_1)
    assert req_id in meta_1.reqs_to_send
    send_meta_1 = meta_1.reqs_to_send[req_id]
    assert send_meta_1.uuid == chunk_uuid(base_uuid, 0)
    assert send_meta_1.token_base == 0
    assert send_meta_1.num_tokens == 1024
    assert send_meta_1.is_final is False
    assert send_meta_1.local_block_ids == block_ids
    assert send_meta_1.expiration_time > 0
    req_state = scheduler._stage3_send_requests[req_id]
    assert req_state.tokens_sent == 1024
    assert req_state.num_computed_tokens == 1024
    assert req_state.next_chunk_idx == 1
    sched_out_2 = MagicMock(spec=SchedulerOutput)
    sched_out_2.num_scheduled_tokens = {req_id: 1024}
    meta_2 = scheduler.build_connector_meta(sched_out_2)
    assert req_id in meta_2.reqs_to_send
    send_meta_2 = meta_2.reqs_to_send[req_id]
    assert send_meta_2.uuid == chunk_uuid(base_uuid, 1)
    assert send_meta_2.token_base == 1024
    assert send_meta_2.num_tokens == 1024
    assert send_meta_2.is_final is False
    assert send_meta_2.local_block_ids == block_ids
    assert req_state.tokens_sent == 2048
    assert req_state.num_computed_tokens == 2048
    assert req_state.next_chunk_idx == 2
    sched_out_3 = MagicMock(spec=SchedulerOutput)
    sched_out_3.num_scheduled_tokens = {req_id: 1024}
    meta_3 = scheduler.build_connector_meta(sched_out_3)
    assert req_id in meta_3.reqs_to_send
    send_meta_3 = meta_3.reqs_to_send[req_id]
    assert send_meta_3.uuid == chunk_uuid(base_uuid, 2)
    assert send_meta_3.token_base == 2048
    assert send_meta_3.num_tokens == 1024
    assert send_meta_3.is_final is False
    assert req_state.tokens_sent == 3072
    assert req_state.num_computed_tokens == 3072
    assert req_state.next_chunk_idx == 3
    sched_out_4 = MagicMock(spec=SchedulerOutput)
    sched_out_4.num_scheduled_tokens = {req_id: 1024}
    meta_4 = scheduler.build_connector_meta(sched_out_4)
    assert req_id in meta_4.reqs_to_send
    send_meta_4 = meta_4.reqs_to_send[req_id]
    assert send_meta_4.uuid == chunk_uuid(base_uuid, 3)
    assert send_meta_4.token_base == 3072
    assert send_meta_4.num_tokens == 1024
    assert send_meta_4.is_final is False
    assert req_state.tokens_sent == 4096
    assert req_state.num_computed_tokens == 4096
    assert req_state.next_chunk_idx == 4
    sched_out_5 = MagicMock(spec=SchedulerOutput)
    sched_out_5.num_scheduled_tokens = {req_id: 0}
    meta_5 = scheduler.build_connector_meta(sched_out_5)
    assert len(meta_5.reqs_to_send) == 0

def test_scheduler_unaligned_remainder_carry() -> None:
    """Simulate prefill steps of 768 tokens.

    Verify that step 1 emits nothing (carries 768 tokens) and step 2 emits
    SendMeta([0, 1024)) once 1536 tokens are computed.
    """
    scheduler = _make_raiden_scheduler(is_producer=True, pcp_size=4, interleave_size=256, chunk_transfer_window=8)
    req_id = 'req-768'
    base_uuid = 54321
    block_ids = [200, 201, 202, 203]
    scheduler._stage3_send_requests[req_id] = _Stage3SendState(base_uuid=base_uuid, block_ids=block_ids, num_computed_tokens=0, tokens_sent=0, next_chunk_idx=0, total_prompt_tokens=3072)
    sched_out_1 = MagicMock(spec=SchedulerOutput)
    sched_out_1.num_scheduled_tokens = {req_id: 768}
    meta_1 = scheduler.build_connector_meta(sched_out_1)
    assert len(meta_1.reqs_to_send) == 0
    req_state = scheduler._stage3_send_requests[req_id]
    assert req_state.tokens_sent == 0
    assert req_state.num_computed_tokens == 768
    assert req_state.next_chunk_idx == 0
    sched_out_2 = MagicMock(spec=SchedulerOutput)
    sched_out_2.num_scheduled_tokens = {req_id: 768}
    meta_2 = scheduler.build_connector_meta(sched_out_2)
    assert req_id in meta_2.reqs_to_send
    send_meta_2 = meta_2.reqs_to_send[req_id]
    assert send_meta_2.uuid == chunk_uuid(base_uuid, 0)
    assert send_meta_2.token_base == 0
    assert send_meta_2.num_tokens == 1024
    assert send_meta_2.is_final is False
    assert send_meta_2.local_block_ids == block_ids
    assert req_state.tokens_sent == 1024
    assert req_state.num_computed_tokens == 1536
    assert req_state.next_chunk_idx == 1
    sched_out_3 = MagicMock(spec=SchedulerOutput)
    sched_out_3.num_scheduled_tokens = {req_id: 768}
    meta_3 = scheduler.build_connector_meta(sched_out_3)
    assert req_id in meta_3.reqs_to_send
    send_meta_3 = meta_3.reqs_to_send[req_id]
    assert send_meta_3.uuid == chunk_uuid(base_uuid, 1)
    assert send_meta_3.token_base == 1024
    assert send_meta_3.num_tokens == 1024
    assert send_meta_3.is_final is False
    assert req_state.tokens_sent == 2048
    assert req_state.num_computed_tokens == 2304
    assert req_state.next_chunk_idx == 2

def test_update_state_after_alloc_stage3_producer() -> None:
    """Verifies update_state_after_alloc initializes producer send state."""
    scheduler = _make_raiden_scheduler(is_producer=True, pcp_size=4, interleave_size=256)
    req = MagicMock()
    req.request_id = 'req-alloc-1'
    req.num_computed_tokens = 0
    req.num_prompt_tokens = 2048
    req.prompt_token_ids = [0] * 2048
    req.kv_transfer_params = None
    blocks = MagicMock()
    blocks.get_block_ids.return_value = [[50, 51, 52, 53]]
    with patch(f'{_MOD}._use_raiden_stage3_transport', return_value=True):
        scheduler.update_state_after_alloc(req, blocks, 0)
    assert 'req-alloc-1' in scheduler._stage3_send_requests
    state = scheduler._stage3_send_requests['req-alloc-1']
    assert state.block_ids == [50, 51, 52, 53]
    assert state.tokens_sent == 0
    assert state.next_chunk_idx == 0
    assert state.total_prompt_tokens == 2048

def test_update_state_after_alloc_existing_request_preserves_progress() -> None:
    """Verifies updating block_ids on existing request preserves progress."""
    scheduler = _make_raiden_scheduler(is_producer=True, pcp_size=4, interleave_size=256)
    req_id = 'req-existing-1'
    scheduler._stage3_send_requests[req_id] = _Stage3SendState(base_uuid=99999, block_ids=[10, 11], num_computed_tokens=1024, tokens_sent=1024, next_chunk_idx=1, total_prompt_tokens=4096)
    req = MagicMock()
    req.request_id = req_id
    req.num_computed_tokens = 1024
    req.num_prompt_tokens = 4096
    req.kv_transfer_params = None
    new_blocks = MagicMock()
    new_blocks.get_block_ids.return_value = [[10, 11, 12, 13]]
    with patch(f'{_MOD}._use_raiden_stage3_transport', return_value=True):
        scheduler.update_state_after_alloc(req, new_blocks, 0)
    state = scheduler._stage3_send_requests[req_id]
    assert state.block_ids == [10, 11, 12, 13]
    assert state.tokens_sent == 1024
    assert state.num_computed_tokens == 1024
    assert state.next_chunk_idx == 1
    assert state.base_uuid == 99999
    assert state.total_prompt_tokens == 4096

def test_update_state_after_alloc_block_format_variants() -> None:
    """Verifies flash-attention block IDs parse from various formats."""
    scheduler = _make_raiden_scheduler(is_producer=True, pcp_size=4, interleave_size=256)
    req_a = MagicMock()
    req_a.request_id = 'req-flat-list'
    req_a.num_computed_tokens = 0
    req_a.num_prompt_tokens = 1024
    req_a.kv_transfer_params = None
    with patch(f'{_MOD}._use_raiden_stage3_transport', return_value=True):
        scheduler.update_state_after_alloc(req_a, [100, 101, 102], 0)
    state_a = scheduler._stage3_send_requests['req-flat-list']
    assert state_a.block_ids == [100, 101, 102]
    req_b = MagicMock()
    req_b.request_id = 'req-nested-tuple'
    req_b.num_computed_tokens = 0
    req_b.num_prompt_tokens = 1024
    req_b.kv_transfer_params = None
    with patch(f'{_MOD}._use_raiden_stage3_transport', return_value=True):
        scheduler.update_state_after_alloc(req_b, ((200, 201), (300, 301)), 0)
    state_b = scheduler._stage3_send_requests['req-nested-tuple']
    assert state_b.block_ids == [200, 201]
    req_c = MagicMock()
    req_c.request_id = 'req-token-fallback'
    req_c.num_computed_tokens = 0
    req_c.num_prompt_tokens = None
    req_c.prompt_token_ids = [1, 2, 3, 4, 5]
    req_c.kv_transfer_params = None
    with patch(f'{_MOD}._use_raiden_stage3_transport', return_value=True):
        scheduler.update_state_after_alloc(req_c, [300], 0)
    state_c = scheduler._stage3_send_requests['req-token-fallback']
    assert state_c.total_prompt_tokens == 5

def test_update_state_after_alloc_non_stage3_or_non_producer() -> None:
    """Verifies update_state_after_alloc ignores calls when Stage-3 disabled."""
    scheduler = _make_raiden_scheduler(is_producer=True, pcp_size=4, interleave_size=256)
    req = MagicMock()
    req.request_id = 'req-no-stage3'
    req.num_computed_tokens = 0
    req.num_prompt_tokens = 1024
    req.kv_transfer_params = None
    with patch(f'{_MOD}._use_raiden_stage3_transport', return_value=False):
        scheduler.update_state_after_alloc(req, [1, 2], 0)
    assert 'req-no-stage3' not in scheduler._stage3_send_requests

def test_multi_request_interleaved_chunk_emission() -> None:
    """Verifies that multiple concurrent requests emit chunks independently."""
    scheduler = _make_raiden_scheduler(is_producer=True, pcp_size=4, interleave_size=256)
    req_a = 'req-A'
    req_b = 'req-B'
    base_uuid_a = 111111
    base_uuid_b = 222222
    scheduler._stage3_send_requests[req_a] = _Stage3SendState(base_uuid=base_uuid_a, block_ids=[1, 2], num_computed_tokens=0, tokens_sent=0, next_chunk_idx=0, total_prompt_tokens=2048)
    scheduler._stage3_send_requests[req_b] = _Stage3SendState(base_uuid=base_uuid_b, block_ids=[3, 4], num_computed_tokens=0, tokens_sent=0, next_chunk_idx=0, total_prompt_tokens=2048)
    sched_out = MagicMock(spec=SchedulerOutput)
    sched_out.num_scheduled_tokens = {req_a: 1024, req_b: 512}
    meta = scheduler.build_connector_meta(sched_out)
    assert req_a in meta.reqs_to_send
    assert req_b not in meta.reqs_to_send
    assert meta.reqs_to_send[req_a].num_tokens == 1024
    assert meta.reqs_to_send[req_a].token_base == 0
    sched_out_2 = MagicMock(spec=SchedulerOutput)
    sched_out_2.num_scheduled_tokens = {req_a: 512, req_b: 512}
    meta_2 = scheduler.build_connector_meta(sched_out_2)
    assert req_a not in meta_2.reqs_to_send
    assert req_b in meta_2.reqs_to_send
    assert meta_2.reqs_to_send[req_b].num_tokens == 1024
    assert meta_2.reqs_to_send[req_b].token_base == 0

def test_build_connector_meta_no_scheduled_tokens_or_empty() -> None:
    """Verifies build_connector_meta handles empty or None output safely."""
    scheduler = _make_raiden_scheduler(is_producer=True, pcp_size=4, interleave_size=256)
    meta_none = scheduler.build_connector_meta(None)
    assert isinstance(meta_none, TPUConnectorMetadata)
    assert len(meta_none.reqs_to_send) == 0
    sched_out_none_tokens = MagicMock(spec=SchedulerOutput)
    sched_out_none_tokens.num_scheduled_tokens = None
    meta_none_tokens = scheduler.build_connector_meta(sched_out_none_tokens)
    assert len(meta_none_tokens.reqs_to_send) == 0
    sched_out_empty = MagicMock(spec=SchedulerOutput)
    sched_out_empty.num_scheduled_tokens = {}
    meta_empty = scheduler.build_connector_meta(sched_out_empty)
    assert len(meta_empty.reqs_to_send) == 0

def test_build_connector_meta_untracked_request_ignored() -> None:
    """Verifies requests not in _stage3_send_requests are ignored."""
    scheduler = _make_raiden_scheduler(is_producer=True, pcp_size=4, interleave_size=256)
    sched_out = MagicMock(spec=SchedulerOutput)
    sched_out.num_scheduled_tokens = {'unknown-req-id': 1024}
    meta = scheduler.build_connector_meta(sched_out)
    assert len(meta.reqs_to_send) == 0

def test_build_connector_meta_multi_stride_single_step() -> None:
    """Verifies multi-stride step emits a chunk covering full extent."""
    scheduler = _make_raiden_scheduler(is_producer=True, pcp_size=4, interleave_size=256)
    req_id = 'req-multi-stride'
    base_uuid = 777777
    scheduler._stage3_send_requests[req_id] = _Stage3SendState(base_uuid=base_uuid, block_ids=[1, 2, 3, 4], num_computed_tokens=0, tokens_sent=0, next_chunk_idx=0, total_prompt_tokens=4096)
    sched_out = MagicMock(spec=SchedulerOutput)
    sched_out.num_scheduled_tokens = {req_id: 2048}
    meta = scheduler.build_connector_meta(sched_out)
    assert req_id in meta.reqs_to_send
    send_meta = meta.reqs_to_send[req_id]
    assert send_meta.token_base == 0
    assert send_meta.num_tokens == 2048
    assert send_meta.is_final is False
    assert send_meta.uuid == chunk_uuid(base_uuid, 0)
    req_state = scheduler._stage3_send_requests[req_id]
    assert req_state.tokens_sent == 2048
    assert req_state.num_computed_tokens == 2048
    assert req_state.next_chunk_idx == 1

def test_build_connector_meta_zero_min_transfer_tokens() -> None:
    """Verifies zero min_transfer_tokens emits computed tokens immediately."""
    scheduler = _make_raiden_scheduler(is_producer=True, pcp_size=4, interleave_size=256)
    scheduler.pcp_size = 0
    scheduler.interleave_tokens = 0
    req_id = 'req-zero-min'
    base_uuid = 888888
    scheduler._stage3_send_requests[req_id] = _Stage3SendState(base_uuid=base_uuid, block_ids=[10, 11], num_computed_tokens=0, tokens_sent=0, next_chunk_idx=0, total_prompt_tokens=500)
    sched_out = MagicMock(spec=SchedulerOutput)
    sched_out.num_scheduled_tokens = {req_id: 128}
    meta = scheduler.build_connector_meta(sched_out)
    assert req_id in meta.reqs_to_send
    send_meta = meta.reqs_to_send[req_id]
    assert send_meta.token_base == 0
    assert send_meta.num_tokens == 128
    assert send_meta.is_final is False

def test_build_connector_meta_consumer_mode_no_chunk_emission() -> None:
    """Verifies consumer mode does not emit chunks in build_connector_meta."""
    scheduler = _make_raiden_scheduler(is_producer=False, pcp_size=4, interleave_size=256)
    req_id = 'req-consumer'
    scheduler._stage3_send_requests[req_id] = _Stage3SendState(base_uuid=999, block_ids=[1, 2], num_computed_tokens=0, tokens_sent=0, next_chunk_idx=0, total_prompt_tokens=2048)
    sched_out = MagicMock(spec=SchedulerOutput)
    sched_out.num_scheduled_tokens = {req_id: 1024}
    meta = scheduler.build_connector_meta(sched_out)
    assert len(meta.reqs_to_send) == 0
    assert scheduler._stage3_send_requests[req_id].tokens_sent == 0

def test_connector_wrapper_pass_through() -> None:
    """Verifies that TPUConnector passes scheduler_output downstream."""
    mock_sched = MagicMock()
    mock_sched.build_connector_meta.return_value = TPUConnectorMetadata()
    connector = TPUConnector.__new__(TPUConnector)
    connector.connector_scheduler = mock_sched
    sched_out = MagicMock(spec=SchedulerOutput)
    result = connector.build_connector_meta(sched_out)
    mock_sched.build_connector_meta.assert_called_once_with(sched_out)
    assert isinstance(result, TPUConnectorMetadata)

def test_raiden_connector_wrapper_pass_through() -> None:
    """Verifies TPURaidenConnector passes scheduler_output to scheduler."""
    mock_sched = MagicMock()
    mock_sched.build_connector_meta.return_value = TPUConnectorMetadata()
    connector = TPURaidenConnector.__new__(TPURaidenConnector)
    connector.connector_scheduler = mock_sched
    sched_out = MagicMock(spec=SchedulerOutput)
    result = connector.build_connector_meta(sched_out)
    mock_sched.build_connector_meta.assert_called_once_with(sched_out)
    assert isinstance(result, TPUConnectorMetadata)

def test_short_prompt_bypass() -> None:
    """Verifies 512-token prompt bypasses intermediate chunking.

    Under PCP4 with stride 1024, a 512-token prompt is shorter than the minimum
    transfer threshold. build_connector_meta() must emit zero intermediate
    chunks, and request_finished() must emit a single terminal SendMeta
    descriptor covering [0, 512) with is_final=True.
    """
    scheduler = _make_raiden_scheduler(is_producer=True, pcp_size=4, interleave_size=256)
    req_id = 'req-short-512'
    base_uuid = 555555
    block_ids = list(range(100, 108))
    req = MagicMock()
    req.request_id = req_id
    req.num_prompt_tokens = 512
    req.num_computed_tokens = 512
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    req.kv_transfer_params = {'_p_side_truncated': True}
    with _patch_raiden_stage3_env(base_uuid=base_uuid):
        scheduler.update_state_after_alloc(req, block_ids, 0)
        sched_out = MagicMock(spec=SchedulerOutput)
        sched_out.num_scheduled_tokens = {req_id: 512}
        meta_step = scheduler.build_connector_meta(sched_out)
        assert req_id not in meta_step.reqs_to_send
        assert len(meta_step.reqs_to_send) == 0
        req_state = scheduler._stage3_send_requests[req_id]
        assert req_state.tokens_sent == 0
        assert req_state.next_chunk_idx == 0
        delay, params = scheduler.request_finished(req, block_ids)
        meta_fin = scheduler.build_connector_meta()
        assert delay is True
        assert req_id in meta_fin.reqs_to_send
        send_meta = meta_fin.reqs_to_send[req_id]
        assert send_meta.token_base == 0
        assert send_meta.num_tokens == 512
        assert send_meta.is_final is True
        assert send_meta.uuid == chunk_uuid(base_uuid, 0)
        assert send_meta.local_block_ids == block_ids
        assert req_id not in scheduler._stage3_send_requests

def test_terminal_remainder_emission() -> None:
    """Verifies intermediate chunks and terminal remainder on 3500-token prompt.

    Intermediate execution steps emit chunks for [0, 1024), [1024, 2048),
    and [2048, 3072). Upon completion, request_finished() emits the terminal
    remainder [3072, 3500) marked with is_final=True.
    """
    scheduler = _make_raiden_scheduler(is_producer=True, pcp_size=4, interleave_size=256, chunk_transfer_window=4)
    req_id = 'req-long-3500'
    base_uuid = 999111
    block_ids = list(range(100, 155))
    req = MagicMock()
    req.request_id = req_id
    req.num_prompt_tokens = 3500
    req.num_computed_tokens = 0
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    req.kv_transfer_params = {'_p_side_truncated': True}
    with _patch_raiden_stage3_env(base_uuid=base_uuid):
        scheduler.update_state_after_alloc(req, block_ids, 0)
        sched_out_1 = MagicMock(spec=SchedulerOutput)
        sched_out_1.num_scheduled_tokens = {req_id: 1024}
        meta_1 = scheduler.build_connector_meta(sched_out_1)
        assert req_id in meta_1.reqs_to_send
        s1 = meta_1.reqs_to_send[req_id]
        assert s1.token_base == 0
        assert s1.num_tokens == 1024
        assert s1.is_final is False
        assert s1.uuid == chunk_uuid(base_uuid, 0)
        sched_out_2 = MagicMock(spec=SchedulerOutput)
        sched_out_2.num_scheduled_tokens = {req_id: 1024}
        meta_2 = scheduler.build_connector_meta(sched_out_2)
        assert req_id in meta_2.reqs_to_send
        s2 = meta_2.reqs_to_send[req_id]
        assert s2.token_base == 1024
        assert s2.num_tokens == 1024
        assert s2.is_final is False
        assert s2.uuid == chunk_uuid(base_uuid, 1)
        sched_out_3 = MagicMock(spec=SchedulerOutput)
        sched_out_3.num_scheduled_tokens = {req_id: 1024}
        meta_3 = scheduler.build_connector_meta(sched_out_3)
        assert req_id in meta_3.reqs_to_send
        s3 = meta_3.reqs_to_send[req_id]
        assert s3.token_base == 2048
        assert s3.num_tokens == 1024
        assert s3.is_final is False
        assert s3.uuid == chunk_uuid(base_uuid, 2)
        sched_out_4 = MagicMock(spec=SchedulerOutput)
        sched_out_4.num_scheduled_tokens = {req_id: 428}
        meta_4 = scheduler.build_connector_meta(sched_out_4)
        assert req_id not in meta_4.reqs_to_send
        req.num_computed_tokens = 3500
        delay, params = scheduler.request_finished(req, block_ids)
        meta_fin = scheduler.build_connector_meta()
        assert delay is True
        assert req_id in meta_fin.reqs_to_send
        s_fin = meta_fin.reqs_to_send[req_id]
        assert s_fin.token_base == 3072
        assert s_fin.num_tokens == 428
        assert s_fin.is_final is True
        assert s_fin.uuid == chunk_uuid(base_uuid, 3)
        assert s_fin.local_block_ids == block_ids
        assert req_id not in scheduler._stage3_send_requests

def test_fail_safe_fallback_degradation() -> None:
    """Verifies monolithic fallback when scheduler emits no intermediate chunks.

    If intermediate chunk emission is skipped (e.g. window capacity limit),
    request_finished() must emit the full range [0, 3500) with is_final=True,
    matching baseline monolithic behavior.
    """
    scheduler = _make_raiden_scheduler(is_producer=True, pcp_size=4, interleave_size=256)
    req_id = 'req-fallback-3500'
    base_uuid = 888222
    block_ids = list(range(100, 155))
    req = MagicMock()
    req.request_id = req_id
    req.num_prompt_tokens = 3500
    req.num_computed_tokens = 3500
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    req.kv_transfer_params = {'_p_side_truncated': True}
    with _patch_raiden_stage3_env(base_uuid=base_uuid):
        scheduler.update_state_after_alloc(req, block_ids, 0)
        req_state = scheduler._stage3_send_requests[req_id]
        assert req_state.tokens_sent == 0
        assert req_state.next_chunk_idx == 0
        delay, params = scheduler.request_finished(req, block_ids)
        meta_fin = scheduler.build_connector_meta()
        assert delay is True
        assert req_id in meta_fin.reqs_to_send
        s_fin = meta_fin.reqs_to_send[req_id]
        assert s_fin.token_base == 0
        assert s_fin.num_tokens == 3500
        assert s_fin.is_final is True
        assert s_fin.uuid == chunk_uuid(base_uuid, 0)
        assert s_fin.local_block_ids == block_ids
        assert req_id not in scheduler._stage3_send_requests

def test_request_finished_duplicate_idempotency_after_chunking() -> None:
    """Verifies duplicate request_finished calls return cached parameters."""
    scheduler = _make_raiden_scheduler(is_producer=True, pcp_size=4, interleave_size=256)
    req_id = 'req-dedup-test'
    base_uuid = 333444
    block_ids = list(range(100, 108))
    req = MagicMock()
    req.request_id = req_id
    req.num_prompt_tokens = 512
    req.num_computed_tokens = 512
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    req.kv_transfer_params = {'_p_side_truncated': True}
    with _patch_raiden_stage3_env(base_uuid=base_uuid):
        scheduler.update_state_after_alloc(req, block_ids, 0)
        delay1, params1 = scheduler.request_finished(req, block_ids)
        meta1 = scheduler.build_connector_meta()
        delay2, params2 = scheduler.request_finished(req, block_ids)
        meta2 = scheduler.build_connector_meta()
        assert delay1 is True and delay2 is True
        assert params1 == params2
        assert req_id in meta1.reqs_to_send
        assert len(meta2.reqs_to_send) == 0

def test_terminal_emission_exact_stride_aligned_all_tokens_sent() -> None:
    """Verifies request_finished() when all prompt tokens were already emitted.

    For an exact stride multiple (e.g. 4096 tokens on PCP=4), all tokens are
    emitted across intermediate steps. request_finished() must clean up
    the send state, return (True, params), and NOT emit an empty remainder into
    reqs_to_send.
    """
    scheduler = _make_raiden_scheduler(is_producer=True, pcp_size=4, interleave_size=256, chunk_transfer_window=4)
    req_id = 'req-exact-4096'
    base_uuid = 777111
    block_ids = list(range(100, 164))
    req = MagicMock()
    req.request_id = req_id
    req.num_prompt_tokens = 4096
    req.num_computed_tokens = 0
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    req.kv_transfer_params = {'_p_side_truncated': True}
    with _patch_raiden_stage3_env(base_uuid=base_uuid):
        scheduler.update_state_after_alloc(req, block_ids, 0)
        for step in range(4):
            sched_out = MagicMock(spec=SchedulerOutput)
            sched_out.num_scheduled_tokens = {req_id: 1024}
            meta = scheduler.build_connector_meta(sched_out)
            assert req_id in meta.reqs_to_send
            assert meta.reqs_to_send[req_id].token_base == step * 1024
            assert meta.reqs_to_send[req_id].num_tokens == 1024
            assert meta.reqs_to_send[req_id].is_final is False
            assert meta.reqs_to_send[req_id].uuid == chunk_uuid(base_uuid, step)
        req.num_computed_tokens = 4096
        delay, params = scheduler.request_finished(req, block_ids)
        meta_fin = scheduler.build_connector_meta()
        assert delay is True
        assert params['req_id'] == req_id
        assert params['uuid'] == chunk_uuid(base_uuid, 4)
        assert params['num_tokens'] == 4096
        assert req_id not in meta_fin.reqs_to_send
        assert len(meta_fin.reqs_to_send) == 0
        assert req_id not in scheduler._stage3_send_requests

def test_untracked_request_terminal_fallback() -> None:
    """Verifies request_finished() fallback when request was not tracked.

    If update_state_after_alloc was not called or request is not in
    _stage3_send_requests (req_state is None), request_finished() must
    gracefully emit a single monolithic SendMeta([0, num_tokens), is_final=True)
    with a freshly generated UUID.
    """
    scheduler = _make_raiden_scheduler(is_producer=True, pcp_size=4, interleave_size=256)
    req_id = 'req-untracked-2048'
    base_uuid = 666777
    block_ids = list(range(100, 132))
    req = MagicMock()
    req.request_id = req_id
    req.num_prompt_tokens = 2048
    req.num_computed_tokens = 2048
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    req.kv_transfer_params = {'_p_side_truncated': True}
    assert req_id not in scheduler._stage3_send_requests
    with _patch_raiden_stage3_env(base_uuid=base_uuid):
        delay, params = scheduler.request_finished(req, block_ids)
        meta_fin = scheduler.build_connector_meta()
        assert delay is True
        assert params['req_id'] == req_id
        assert params['uuid'] == base_uuid
        assert params['num_tokens'] == 2048
        assert req_id in meta_fin.reqs_to_send
        s_fin = meta_fin.reqs_to_send[req_id]
        assert s_fin.token_base == 0
        assert s_fin.num_tokens == 2048
        assert s_fin.is_final is True
        assert s_fin.uuid == base_uuid
        assert s_fin.local_block_ids == block_ids

def test_short_prompt_multi_step_bypass() -> None:
    """Verifies that multi-step execution for short prompts stays bypassed.

    A 512-token prompt executed in two 256-token steps must emit zero
    intermediate chunks on both steps, and emit a single terminal chunk on
    request_finished().
    """
    scheduler = _make_raiden_scheduler(is_producer=True, pcp_size=4, interleave_size=256)
    req_id = 'req-short-multistep-512'
    base_uuid = 444333
    block_ids = list(range(100, 108))
    req = MagicMock()
    req.request_id = req_id
    req.num_prompt_tokens = 512
    req.num_computed_tokens = 0
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    req.kv_transfer_params = {'_p_side_truncated': True}
    with _patch_raiden_stage3_env(base_uuid=base_uuid):
        scheduler.update_state_after_alloc(req, block_ids, 0)
        sched_out_1 = MagicMock(spec=SchedulerOutput)
        sched_out_1.num_scheduled_tokens = {req_id: 256}
        meta_1 = scheduler.build_connector_meta(sched_out_1)
        assert len(meta_1.reqs_to_send) == 0
        req_state = scheduler._stage3_send_requests[req_id]
        assert req_state.tokens_sent == 0
        assert req_state.next_chunk_idx == 0
        sched_out_2 = MagicMock(spec=SchedulerOutput)
        sched_out_2.num_scheduled_tokens = {req_id: 256}
        meta_2 = scheduler.build_connector_meta(sched_out_2)
        assert len(meta_2.reqs_to_send) == 0
        assert req_state.tokens_sent == 0
        req.num_computed_tokens = 512
        delay, params = scheduler.request_finished(req, block_ids)
        meta_fin = scheduler.build_connector_meta()
        assert delay is True
        assert req_id in meta_fin.reqs_to_send
        s_fin = meta_fin.reqs_to_send[req_id]
        assert s_fin.token_base == 0
        assert s_fin.num_tokens == 512
        assert s_fin.is_final is True
        assert s_fin.uuid == chunk_uuid(base_uuid, 0)
        assert s_fin.local_block_ids == block_ids
        assert req_id not in scheduler._stage3_send_requests

def test_request_finished_duplicate_idempotency_multi_chunk() -> None:
    """Verifies duplicate request_finished calls on multi-chunk requests.

    A multi-chunk request with intermediate emissions and a terminal remainder
    must return identical cached parameters and avoid duplicate emissions on
    subsequent request_finished() calls.
    """
    scheduler = _make_raiden_scheduler(is_producer=True, pcp_size=4, interleave_size=256, chunk_transfer_window=4)
    req_id = 'req-dedup-multichunk'
    base_uuid = 123987
    block_ids = list(range(100, 155))
    req = MagicMock()
    req.request_id = req_id
    req.num_prompt_tokens = 3500
    req.num_computed_tokens = 0
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    req.kv_transfer_params = {'_p_side_truncated': True}
    with _patch_raiden_stage3_env(base_uuid=base_uuid):
        scheduler.update_state_after_alloc(req, block_ids, 0)
        for _ in range(3):
            sched_out = MagicMock(spec=SchedulerOutput)
            sched_out.num_scheduled_tokens = {req_id: 1024}
            scheduler.build_connector_meta(sched_out)
        req.num_computed_tokens = 3500
        delay1, params1 = scheduler.request_finished(req, block_ids)
        meta1 = scheduler.build_connector_meta()
        assert delay1 is True
        assert req_id in meta1.reqs_to_send
        assert meta1.reqs_to_send[req_id].token_base == 3072
        assert meta1.reqs_to_send[req_id].num_tokens == 428
        assert meta1.reqs_to_send[req_id].is_final is True
        delay2, params2 = scheduler.request_finished(req, block_ids)
        meta2 = scheduler.build_connector_meta()
        assert delay2 is True
        assert params1 == params2
        assert len(meta2.reqs_to_send) == 0

# ---------------------------------------------------------------------------
# Producer Worker Sub-Chunk Registration & Buffer Keep-Alives
# ---------------------------------------------------------------------------

_MOD = 'vllm_torchtpu.distributed.kv_transfer.tpu_connector'

class _FakeProducerRaidenEngine:
    """Mock low-level Raiden transfer engine for worker testing."""

    def __init__(self) -> None:
        """Initializes call tracking lists and queued completion results."""
        self.calls: list[Any] = []
        self.poll_results: list[tuple[list[str], list[str], list[str]]] = []
        self.complete_read_results: list[tuple[list[str], list[str], list[str]]] = []

    def register_read(self, req_id: str, uuid: int, block_ids: list[int]) -> int:
        """Records a register_read invocation."""
        self.calls.append(('register_read', req_id, uuid, block_ids))
        return 1

    def start_read(self, req_id: str, uuid: int, endpoint: str, remote_blocks: list[int], local_blocks: list[int]) -> int:
        """Records a start_read invocation."""
        self.calls.append(('start_read', req_id, uuid, endpoint, remote_blocks, local_blocks))
        return 2

    def complete_read(self) -> tuple[list[str], list[str], list[str]]:
        """Returns queued complete_read completion triples."""
        self.calls.append(('complete_read',))
        if self.complete_read_results:
            return self.complete_read_results.pop(0)
        if self.poll_results:
            return self.poll_results.pop(0)
        return ([], [], [])

    def poll_stats(self) -> tuple[list[str], list[str], list[str]]:
        """Returns queued poll_stats completion triples."""
        self.calls.append(('poll_stats',))
        if self.poll_results:
            return self.poll_results.pop(0)
        if self.complete_read_results:
            return self.complete_read_results.pop(0)
        return ([], [], [])

class _FakeProducerRaidenControllerFacade:
    """Mock controller client recording RPC invocations."""

    def __init__(self) -> None:
        """Initializes lists to record controller RPC calls."""
        self.register_work_unit_calls: list[dict[str, Any]] = []
        self.register_request_blocks_calls: list[dict[str, Any]] = []
        self.complete_request_blocks_calls: list[dict[str, Any]] = []
        self.cancel_request_blocks_calls: list[dict[str, Any]] = []
        self.cancel_request_blocks_result: bool = True

    def register_work_unit(self, **kwargs: Any) -> None:
        """Records a register_work_unit RPC call."""
        self.register_work_unit_calls.append(kwargs)

    def register_request_blocks(self, **kwargs: Any) -> None:
        """Records a register_request_blocks RPC call."""
        self.register_request_blocks_calls.append(kwargs)

    def complete_request_blocks(self, **kwargs: Any) -> None:
        """Records a complete_request_blocks RPC call."""
        self.complete_request_blocks_calls.append(kwargs)

    def cancel_request_blocks_if_unclaimed(self, **kwargs: Any) -> bool:
        """Records a cancel_request_blocks_if_unclaimed RPC call."""
        self.cancel_request_blocks_calls.append(kwargs)
        return self.cancel_request_blocks_result

def _make_producer_test_worker(*, tp_rank: int=0, tp_size: int=1, is_producer: bool=True, pcp_size: int=4, interleave_size: int=256, block_size: int=1024, token_bytes: int=1024, transfer_rank: int=0) -> tuple[TPURaidenConnectorWorker, _FakeRaidenControllerFacade, _FakeRaidenEngine]:
    """Constructs a TPURaidenConnectorWorker with mocked dependencies."""
    cfg = _make_mock_vllm_config(is_producer=is_producer, block_size=block_size, tp_size=tp_size, pcp_size=pcp_size, interleave_size=interleave_size)
    with patch(f'{_MOD}.get_tensor_model_parallel_rank', return_value=tp_rank), patch(f'{_MOD}.get_tensor_model_parallel_world_size', return_value=tp_size), patch(f'{_MOD}.dist_utils.get_node_id', return_value=0), patch(f'{_MOD}.dist_utils.get_host_ip', return_value='127.0.0.1'), patch(f'{_MOD}.dist_utils.get_kv_transfer_port', return_value=9100):
        worker = TPURaidenConnectorWorker(cfg)
    engine = _FakeProducerRaidenEngine()
    facade = _FakeProducerRaidenControllerFacade()
    unit = SimpleNamespace(job_name='prefill-worker', job_replica_id='producer-engine-rank0', data_name='kv.fa', data_replica_idx=0)
    worker._raiden_transfer_engine = engine
    worker._raiden_controller_facade = facade
    worker._raiden_controller_address = 'prefill-controller.test:27000'
    worker._raiden_work_unit = unit
    worker._local_raiden_transfer_rank = MagicMock(return_value=transfer_rank)
    worker._raiden_manifest = SimpleNamespace(geometry_by_tag=lambda: {'fa': {'live_bytes_per_block': block_size * token_bytes}}, tag_counts=lambda: {'fa': 15})
    return (worker, facade, engine)

@contextlib.contextmanager
def _patch_raiden_worker_env(parallelism: int=4, wait_pull_timeout: float | None=None) -> Iterator[None]:
    """Patches runtime environment variables for worker testing."""
    patches = [patch(f'{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT', 'raiden', create=True), patch(f'{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM', parallelism, create=True)]
    if wait_pull_timeout is not None:
        patches.append(patch(f'{_MOD}.dist_utils.get_p2p_wait_pull_timeout', return_value=wait_pull_timeout))
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield

def test_worker_registers_subchunk_work_unit() -> None:
    """Verifies worker registers sub-chunk work unit with range-addressed spans.

    Feeds sub-chunk SendMeta(token_base=1024, num_tokens=1024) into
    process_send_load(). Asserts that the controller client receives a
    registration containing pool spans offset at token 1024 in the destination
    byte space.
    """
    worker, facade, _ = _make_producer_test_worker(pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0)
    req_id = 'req-chunk-001'
    sub_uuid = 54321
    token_base = 1024
    num_tokens = 1024
    local_block_ids = [200, 201]
    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(uuid=sub_uuid, local_block_ids=local_block_ids, expiration_time=time.perf_counter() + 30.0, num_tokens=num_tokens, token_base=token_base, is_final=False)
    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta)
    assert len(facade.register_request_blocks_calls) == 1
    call = facade.register_request_blocks_calls[0]
    assert call['req_id'] == req_id
    assert call['uuid'] == sub_uuid
    assert call['unit'] is worker._raiden_work_unit
    assert call['block_ids'] == [200]
    pool_spans: list[PoolSpanRegistration] = call['pool_spans']
    assert len(pool_spans) == 1
    fa_reg = pool_spans[0]
    assert fa_reg.tag == 'fa'
    assert fa_reg.dst_space_version == 1
    token_bytes = worker._stage3_fa_token_bytes(1024)
    assert fa_reg.declared_bytes == 256 * token_bytes
    assert len(fa_reg.spans) == 1
    span: PoolByteSpan = fa_reg.spans[0]
    assert span.src_block_ordinal == 0
    assert span.src_offset_bytes == 0
    assert span.dst_offset_bytes == token_base * token_bytes
    assert span.size_bytes == 256 * token_bytes
    assert req_id in worker._stage3_registered_sends
    reg_entry: _Stage3RegisteredSend = worker._stage3_registered_sends[req_id]
    assert reg_entry.uuid == sub_uuid
    assert reg_entry.token_base == token_base
    assert reg_entry.num_tokens == num_tokens
    assert reg_entry.is_final is False
    assert reg_entry.local_block_ids == (200,)

def test_worker_registration_idempotency() -> None:
    """Verifies duplicate sub-chunk SendMeta registrations are ignored.

    Feeds identical sub-chunk SendMeta twice into process_send_load(). Asserts
    that the controller facade receives only one registration RPC and no
    duplicate errors are raised.
    """
    worker, facade, _ = _make_producer_test_worker(pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0)
    req_id = 'req-idempotent-002'
    sub_uuid = 98765
    token_base = 1024
    num_tokens = 1024
    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(uuid=sub_uuid, local_block_ids=[100, 101], expiration_time=time.perf_counter() + 30.0, num_tokens=num_tokens, token_base=token_base, is_final=False)
    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta)
        worker.process_send_load(meta)
    assert len(facade.register_request_blocks_calls) == 1
    assert req_id in worker._stage3_registered_sends

def test_worker_registration_conflicting_duplicate_rejected() -> None:
    """Verifies that duplicate registrations with conflicting parameters fail.

    Registers a sub-chunk with token_base=1024, then attempts to register
    conflicting metadata under the same req_id. Asserts that ValueError is
    raised with an explicit conflict message.
    """
    worker, _, _ = _make_producer_test_worker(pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0)
    req_id = 'req-conflict-003'
    meta1 = TPUConnectorMetadata()
    meta1.reqs_to_send[req_id] = SendMeta(uuid=11111, local_block_ids=[100, 101], expiration_time=time.perf_counter() + 30.0, num_tokens=1024, token_base=1024, is_final=False)
    meta_conflict = TPUConnectorMetadata()
    meta_conflict.reqs_to_send[req_id] = SendMeta(uuid=11111, local_block_ids=[100, 101], expiration_time=time.perf_counter() + 30.0, num_tokens=1024, token_base=2048, is_final=False)
    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta1)
        with pytest.raises(ValueError, match='Conflicting duplicate Stage-3 producer registration'):
            worker.process_send_load(meta_conflict)

def test_worker_registration_terminal_idempotency_and_conflict() -> None:
    """Verifies replay behavior against terminal tombstone entries.

    Simulates a sub-chunk registration that has already completed and moved
    to _stage3_terminal_sends. Identical replays should be ignored, while
    conflicting replays must raise ValueError.
    """
    worker, facade, _ = _make_producer_test_worker(pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0)
    req_id = 'req-terminal-004'
    sub_uuid = 22222
    token_base = 2048
    num_tokens = 1024
    local_ids = (100,)
    worker._stage3_terminal_sends[req_id] = _Stage3RegisteredSend(uuid=sub_uuid, local_block_ids=local_ids, num_tokens=num_tokens, expiration_time=time.perf_counter() + 30.0, token_base=token_base, is_final=False)
    meta_identical = TPUConnectorMetadata()
    meta_identical.reqs_to_send[req_id] = SendMeta(uuid=sub_uuid, local_block_ids=[100, 101, 102], expiration_time=time.perf_counter() + 30.0, num_tokens=num_tokens, token_base=token_base, is_final=False)
    meta_conflict = TPUConnectorMetadata()
    meta_conflict.reqs_to_send[req_id] = SendMeta(uuid=sub_uuid, local_block_ids=[100, 101, 102], expiration_time=time.perf_counter() + 30.0, num_tokens=num_tokens, token_base=1024, is_final=False)
    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta_identical)
        assert len(facade.register_request_blocks_calls) == 0
        with pytest.raises(ValueError, match='Conflicting replay after terminal Stage-3 producer registration'):
            worker.process_send_load(meta_conflict)

@pytest.mark.parametrize('transfer_rank', [0, 1, 2, 3])
def test_worker_registration_multi_rank_offsets(transfer_rank: int) -> None:
    """Verifies sub-chunk span destination offsets across all PCP ranks.

    For chunk [1024, 2048) on PCP=4 with interleave 256:
    - Rank 0 owns [1024, 1280) -> dst_offset = 1024 * token_bytes
    - Rank 1 owns [1280, 1536) -> dst_offset = 1280 * token_bytes
    - Rank 2 owns [1536, 1792) -> dst_offset = 1536 * token_bytes
    - Rank 3 owns [1792, 2048) -> dst_offset = 1792 * token_bytes
    """
    worker, facade, _ = _make_producer_test_worker(pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=transfer_rank)
    req_id = f'req-rank-{transfer_rank}'
    sub_uuid = 30000 + transfer_rank
    token_base = 1024
    num_tokens = 1024
    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(uuid=sub_uuid, local_block_ids=[10, 11], expiration_time=time.perf_counter() + 30.0, num_tokens=num_tokens, token_base=token_base, is_final=False)
    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta)
    assert len(facade.register_request_blocks_calls) == 1
    call = facade.register_request_blocks_calls[0]
    fa_reg = call['pool_spans'][0]
    token_bytes = worker._stage3_fa_token_bytes(1024)
    expected_offset = (token_base + transfer_rank * 256) * token_bytes
    assert len(fa_reg.spans) == 1
    assert fa_reg.spans[0].dst_offset_bytes == expected_offset
    assert fa_reg.spans[0].size_bytes == 256 * token_bytes

def test_worker_registration_unaligned_token_base_rejected() -> None:
    """Verifies unaligned token_base raises ValueError during registration."""
    worker, _, _ = _make_producer_test_worker(pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0)
    meta = TPUConnectorMetadata()
    meta.reqs_to_send['req-unaligned'] = SendMeta(uuid=44444, local_block_ids=[100, 101], expiration_time=time.perf_counter() + 30.0, num_tokens=1024, token_base=512, is_final=False)
    with _patch_raiden_worker_env(parallelism=4):
        with pytest.raises(ValueError, match='must be aligned to PCP cycle'):
            worker.process_send_load(meta)

def test_worker_registration_negative_token_base_rejected() -> None:
    """Verifies negative token_base raises ValueError during registration."""
    worker, _, _ = _make_producer_test_worker(pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0)
    meta = TPUConnectorMetadata()
    meta.reqs_to_send['req-negative'] = SendMeta(uuid=55555, local_block_ids=[100, 101], expiration_time=time.perf_counter() + 30.0, num_tokens=1024, token_base=-1024, is_final=False)
    with _patch_raiden_worker_env(parallelism=4):
        with pytest.raises(ValueError, match='Stage-3 producer send metadata requires a non-negative token_base'):
            worker.process_send_load(meta)

def test_worker_registration_legacy_backward_compatibility() -> None:
    """Verifies legacy SendMeta registers properly with default base 0."""
    worker, facade, _ = _make_producer_test_worker(pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0)
    req_id = 'req-legacy-001'
    uuid_val = 66001
    num_tokens = 1024
    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(uuid=uuid_val, local_block_ids=[100], expiration_time=time.perf_counter() + 30.0, num_tokens=num_tokens)
    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta)
    assert len(facade.register_request_blocks_calls) == 1
    call = facade.register_request_blocks_calls[0]
    fa_reg = call['pool_spans'][0]
    token_bytes = worker._stage3_fa_token_bytes(1024)
    assert fa_reg.spans[0].dst_offset_bytes == 0
    assert fa_reg.spans[0].size_bytes == 256 * token_bytes
    assert req_id in worker._stage3_registered_sends
    reg = worker._stage3_registered_sends[req_id]
    assert reg.token_base == 0
    assert reg.is_final is True

def test_worker_registration_insufficient_scheduler_blocks_rejected() -> None:
    """Verifies insufficient scheduler blocks for window raises ValueError."""
    worker, _, _ = _make_producer_test_worker(pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0)
    req_id = 'req-insufficient-blocks'
    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(uuid=66002, local_block_ids=[100], expiration_time=time.perf_counter() + 30.0, num_tokens=1024, token_base=4096, is_final=False)
    with _patch_raiden_worker_env(parallelism=4):
        with pytest.raises(ValueError, match='Stage-3 producer scheduler block count is inconsistent with PCP geometry'):
            worker.process_send_load(meta)

def test_worker_registration_cancellation_race_records_tombstone() -> None:
    """Verifies cancellation errors record tombstone with sub-chunk fields."""
    worker, facade, _ = _make_producer_test_worker(pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0)
    req_id = 'req-cancelled-001'
    sub_uuid = 66003
    token_base = 1024
    num_tokens = 1024
    facade.register_request_blocks = MagicMock(side_effect=RuntimeError('Request block registration was cancelled: generation expired'))
    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(uuid=sub_uuid, local_block_ids=[100, 101], expiration_time=time.perf_counter() + 30.0, num_tokens=num_tokens, token_base=token_base, is_final=False)
    with _patch_raiden_worker_env(parallelism=4, wait_pull_timeout=5.0):
        worker.process_send_load(meta)
    assert req_id in worker._stage3_terminal_sends
    assert req_id not in worker._stage3_registered_sends
    assert req_id in worker._done_sending
    tombstone = worker._stage3_terminal_sends[req_id]
    assert tombstone.uuid == sub_uuid
    assert tombstone.token_base == token_base
    assert tombstone.num_tokens == num_tokens
    assert tombstone.is_final is False

def test_worker_registration_multiple_sequential_subchunks() -> None:
    """Verifies multiple sequential sub-chunks register independently."""
    worker, facade, _ = _make_producer_test_worker(pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0)
    token_bytes = worker._stage3_fa_token_bytes(1024)
    meta0 = TPUConnectorMetadata()
    meta0.reqs_to_send['req-multi:0'] = SendMeta(uuid=70000, local_block_ids=[100, 101], expiration_time=time.perf_counter() + 30.0, num_tokens=1024, token_base=0, is_final=False)
    meta1 = TPUConnectorMetadata()
    meta1.reqs_to_send['req-multi:1'] = SendMeta(uuid=70001, local_block_ids=[100, 101], expiration_time=time.perf_counter() + 30.0, num_tokens=1024, token_base=1024, is_final=True)
    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta0)
        worker.process_send_load(meta1)
    assert len(facade.register_request_blocks_calls) == 2
    call0 = facade.register_request_blocks_calls[0]
    call1 = facade.register_request_blocks_calls[1]
    assert call0['req_id'] == 'req-multi:0'
    assert call0['uuid'] == 70000
    assert call0['pool_spans'][0].spans[0].dst_offset_bytes == 0
    assert call1['req_id'] == 'req-multi:1'
    assert call1['uuid'] == 70001
    assert call1['pool_spans'][0].spans[0].dst_offset_bytes == 1024 * token_bytes
    assert 'req-multi:0' in worker._stage3_registered_sends
    assert 'req-multi:1' in worker._stage3_registered_sends
    assert worker._stage3_registered_sends['req-multi:0'].is_final is False
    assert worker._stage3_registered_sends['req-multi:1'].is_final is True

def test_worker_registration_conflicting_num_tokens_rejected() -> None:
    """Verifies registration with mismatched num_tokens raises ValueError."""
    worker, _, _ = _make_producer_test_worker(pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0)
    req_id = 'req-conflict-num-tokens'
    meta1 = TPUConnectorMetadata()
    meta1.reqs_to_send[req_id] = SendMeta(uuid=80001, local_block_ids=[100, 101], expiration_time=time.perf_counter() + 30.0, num_tokens=1024, token_base=1024, is_final=False)
    meta_conflict = TPUConnectorMetadata()
    meta_conflict.reqs_to_send[req_id] = SendMeta(uuid=80001, local_block_ids=[100, 101], expiration_time=time.perf_counter() + 30.0, num_tokens=2048, token_base=1024, is_final=False)
    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta1)
        with pytest.raises(ValueError, match='Conflicting duplicate Stage-3 producer registration'):
            worker.process_send_load(meta_conflict)

def test_worker_registration_conflicting_local_block_ids_rejected() -> None:
    """Verifies registration with mismatched block_ids raises ValueError."""
    worker, _, _ = _make_producer_test_worker(pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0)
    req_id = 'req-conflict-blocks'
    meta1 = TPUConnectorMetadata()
    meta1.reqs_to_send[req_id] = SendMeta(uuid=80002, local_block_ids=[100, 101], expiration_time=time.perf_counter() + 30.0, num_tokens=1024, token_base=1024, is_final=False)
    meta_conflict = TPUConnectorMetadata()
    meta_conflict.reqs_to_send[req_id] = SendMeta(uuid=80002, local_block_ids=[102, 103], expiration_time=time.perf_counter() + 30.0, num_tokens=1024, token_base=1024, is_final=False)
    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta1)
        with pytest.raises(ValueError, match='Conflicting duplicate Stage-3 producer registration'):
            worker.process_send_load(meta_conflict)

def test_device_buffer_retained_during_dma() -> None:
    """Verifies device buffer handles are retained during DMA."""
    worker, facade, _ = _make_producer_test_worker(pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0)
    mock_tensor_k = MagicMock(name='mock_kv_tensor_k')
    mock_tensor_v = MagicMock(name='mock_kv_tensor_v')
    mock_runner = MagicMock()
    mock_runner.kv_caches = [(mock_tensor_k, mock_tensor_v)]
    worker.runner = mock_runner
    req_id = 'req-dma-001'
    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(uuid=90001, local_block_ids=[100, 101], expiration_time=time.perf_counter() + 30.0, num_tokens=1024, token_base=1024, is_final=False)
    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta)
    assert req_id in worker._active_tensor_refs
    assert req_id in worker._stage3_registered_sends
    ref = worker._active_tensor_refs[req_id]
    assert isinstance(ref, _Stage3RegisteredSend)
    assert ref.uuid == 90001
    assert ref.token_base == 1024
    assert ref.num_tokens == 1024
    assert ref.local_block_ids == (100,)
    assert ref.device_buffers == (mock_tensor_k, mock_tensor_v)

def test_device_buffer_released_on_dma_completion() -> None:
    """Verifies device buffer handles are released when DMA completes."""
    worker, facade, engine = _make_producer_test_worker(pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0)
    mock_tensor = MagicMock(name='mock_tensor')
    mock_runner = MagicMock()
    mock_runner.kv_caches = [mock_tensor]
    worker.runner = mock_runner
    req_id = 'req-dma-002'
    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(uuid=90002, local_block_ids=[100, 101], expiration_time=time.perf_counter() + 30.0, num_tokens=1024, token_base=0, is_final=False)
    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta)
    assert req_id in worker._active_tensor_refs
    assert worker._active_tensor_refs[req_id].device_buffers == (mock_tensor,)
    engine.complete_read_results = [([req_id], [], [])]
    with _patch_raiden_worker_env(parallelism=4):
        done_sending, _ = worker.get_finished()
    assert req_id in done_sending
    assert req_id not in worker._active_tensor_refs
    assert req_id in worker._stage3_terminal_sends
    assert worker._stage3_terminal_sends[req_id].device_buffers == ()
    assert req_id not in worker._stage3_registered_sends

def test_device_buffer_multiple_subchunks_independent_lifecycle() -> None:
    """Verifies sub-chunks have independent buffer retention lifecycles."""
    worker, facade, engine = _make_producer_test_worker(pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0)
    mock_tensor = MagicMock(name='mock_tensor')
    mock_runner = MagicMock()
    mock_runner.kv_caches = [mock_tensor]
    worker.runner = mock_runner
    meta0 = TPUConnectorMetadata()
    meta0.reqs_to_send['req-sub:0'] = SendMeta(uuid=91000, local_block_ids=[100, 101], expiration_time=time.perf_counter() + 30.0, num_tokens=1024, token_base=0, is_final=False)
    meta1 = TPUConnectorMetadata()
    meta1.reqs_to_send['req-sub:1'] = SendMeta(uuid=91001, local_block_ids=[100, 101], expiration_time=time.perf_counter() + 30.0, num_tokens=1024, token_base=1024, is_final=True)
    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta0)
        worker.process_send_load(meta1)
    assert 'req-sub:0' in worker._active_tensor_refs
    assert 'req-sub:1' in worker._active_tensor_refs
    engine.complete_read_results = [(['req-sub:0'], [], [])]
    with _patch_raiden_worker_env(parallelism=4):
        done_sending, _ = worker.get_finished()
    assert 'req-sub:0' in done_sending
    assert 'req-sub:0' not in worker._active_tensor_refs
    assert 'req-sub:1' in worker._active_tensor_refs
    engine.complete_read_results = [(['req-sub:1'], [], [])]
    with _patch_raiden_worker_env(parallelism=4):
        done_sending2, _ = worker.get_finished()
    assert 'req-sub:1' in done_sending2
    assert 'req-sub:1' not in worker._active_tensor_refs

def test_device_buffer_unregistered_on_cancellation_race() -> None:
    """Verifies buffer handles are dropped when registration is cancelled."""
    worker, facade, _ = _make_producer_test_worker(pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0)
    mock_tensor = MagicMock(name='mock_tensor')
    mock_runner = MagicMock()
    mock_runner.kv_caches = [mock_tensor]
    worker.runner = mock_runner
    facade.register_request_blocks = MagicMock(side_effect=RuntimeError('Request block registration was cancelled: generation expired'))
    req_id = 'req-cancel-race'
    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(uuid=92000, local_block_ids=[100, 101], expiration_time=time.perf_counter() + 30.0, num_tokens=1024, token_base=1024, is_final=False)
    with _patch_raiden_worker_env(parallelism=4, wait_pull_timeout=5.0):
        worker.process_send_load(meta)
    assert req_id not in worker._active_tensor_refs
    assert req_id in worker._stage3_terminal_sends
    assert worker._stage3_terminal_sends[req_id].device_buffers == ()

def test_flattened_kv_cache_tensors_variations() -> None:
    """Verifies _flattened_kv_cache_tensors handles varied cache formats."""
    worker, _, _ = _make_producer_test_worker()
    worker.runner = None
    assert worker._flattened_kv_cache_tensors() == []
    worker.runner = SimpleNamespace()
    assert worker._flattened_kv_cache_tensors() == []
    worker.runner = SimpleNamespace(kv_caches=None)
    assert worker._flattened_kv_cache_tensors() == []
    t1, t2 = (MagicMock(name='t1'), MagicMock(name='t2'))
    worker.runner = SimpleNamespace(kv_caches=[t1, t2])
    assert worker._flattened_kv_cache_tensors() == [t1, t2]
    k1, v1 = (MagicMock(name='k1'), MagicMock(name='v1'))
    k2, v2 = (MagicMock(name='k2'), MagicMock(name='v2'))
    worker.runner = SimpleNamespace(kv_caches=[(k1, v1), (k2, v2)])
    assert worker._flattened_kv_cache_tensors() == [k1, v1, k2, v2]
    t3 = MagicMock(name='t3')
    worker.runner = SimpleNamespace(kv_caches=[(k1, v1), t3])
    assert worker._flattened_kv_cache_tensors() == [k1, v1, t3]

def test_poll_finished_engine_interface_polymorphism() -> None:
    """Verifies _poll_finished handles polymorphic engine interfaces."""
    worker, _, _ = _make_producer_test_worker(is_producer=True)
    mock_tensor = MagicMock(name='mock_tensor')
    worker.runner = SimpleNamespace(kv_caches=[mock_tensor])
    req_id_1 = 'req-poly-complete-read'
    meta1 = TPUConnectorMetadata()
    meta1.reqs_to_send[req_id_1] = SendMeta(uuid=93001, local_block_ids=[100, 101], expiration_time=time.perf_counter() + 30.0, num_tokens=1024, token_base=0, is_final=True)
    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta1)
    assert req_id_1 in worker._active_tensor_refs

    class EngineWithCompleteRead:
        """Mock transfer engine supporting complete_read interface."""

        def complete_read(self) -> tuple[list[str], list[str], list[str]]:
            """Returns completed read requests."""
            return ([req_id_1], [], [])
    with _patch_raiden_worker_env(parallelism=4):
        worker._poll_finished(EngineWithCompleteRead())
    assert req_id_1 not in worker._active_tensor_refs
    req_id_2 = 'req-poly-poll-stats'
    meta2 = TPUConnectorMetadata()
    meta2.reqs_to_send[req_id_2] = SendMeta(uuid=93002, local_block_ids=[100, 101], expiration_time=time.perf_counter() + 30.0, num_tokens=1024, token_base=0, is_final=True)
    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta2)
    assert req_id_2 in worker._active_tensor_refs

    class EngineWithPollStats:
        """Mock transfer engine supporting poll_stats interface."""

        def poll_stats(self) -> tuple[list[str], list[str], list[str]]:
            """Returns polled transfer completion stats."""
            return ([req_id_2], [], [])
    with _patch_raiden_worker_env(parallelism=4):
        worker._poll_finished(EngineWithPollStats())
    assert req_id_2 not in worker._active_tensor_refs

    class EngineWithoutPolling:
        """Mock transfer engine without completion query methods."""
    with _patch_raiden_worker_env(parallelism=4):
        worker._poll_finished(EngineWithoutPolling())

def test_device_buffer_released_on_terminal_cleanup_force_release() -> None:
    """Verifies buffer handles are purged during terminal cleanup."""
    worker, facade, _ = _make_producer_test_worker(pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0)
    mock_tensor = MagicMock(name='mock_tensor')
    worker.runner = SimpleNamespace(kv_caches=[mock_tensor])
    req_id = 'req-term-cleanup'
    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(uuid=94001, local_block_ids=[100, 101], expiration_time=time.perf_counter() + 30.0, num_tokens=1024, token_base=0, is_final=True)
    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta)
    assert req_id in worker._active_tensor_refs
    assert worker._active_tensor_refs[req_id].device_buffers == (mock_tensor,)
    worker._stage3_terminal_cleanup[req_id] = True
    with _patch_raiden_worker_env(parallelism=4):
        worker.get_finished()
    assert req_id not in worker._active_tensor_refs
    assert req_id not in worker._stage3_registered_sends
    assert req_id in worker._stage3_terminal_sends
    assert worker._stage3_terminal_sends[req_id].device_buffers == ()
    assert len(facade.complete_request_blocks_calls) == 1
    assert facade.complete_request_blocks_calls[0]['req_id'] == req_id
    assert facade.complete_request_blocks_calls[0]['uuid'] == 94001

def test_glm_device_buffer_retained_and_released_on_cancellation_race() -> None:
    """Verifies GLM producer registration retains and releases buffers."""
    worker, facade, _ = _make_producer_test_worker(tp_rank=0, tp_size=4, pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0)
    mock_tensor_k = MagicMock(name='mock_kv_tensor_k')
    mock_tensor_v = MagicMock(name='mock_kv_tensor_v')
    worker.runner = SimpleNamespace(kv_caches=[(mock_tensor_k, mock_tensor_v)])
    worker._stage3_glm_geometry = {'mla.nope': (1024 * 128, 128), 'mla.rope': (1024 * 64, 64), 'dsa.idx': (1024 * 16, 16)}
    req_id_ok = 'req-glm-ok'
    meta_ok = TPUConnectorMetadata()
    meta_ok.reqs_to_send[req_id_ok] = SendMeta(uuid=95001, local_block_ids=[200], expiration_time=time.perf_counter() + 30.0, num_tokens=1024, token_base=0, is_final=True)
    with patch(f'{_MOD}._use_raiden_glm_admission', return_value=True), _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta_ok)
    assert req_id_ok in worker._active_tensor_refs
    assert req_id_ok in worker._stage3_registered_sends
    assert worker._active_tensor_refs[req_id_ok].device_buffers == (mock_tensor_k, mock_tensor_v)
    facade.register_request_blocks = MagicMock(side_effect=RuntimeError('Request block registration was cancelled: generation expired'))
    req_id_cancel = 'req-glm-cancel'
    meta_cancel = TPUConnectorMetadata()
    meta_cancel.reqs_to_send[req_id_cancel] = SendMeta(uuid=95002, local_block_ids=[202], expiration_time=time.perf_counter() + 30.0, num_tokens=1024, token_base=0, is_final=True)
    with patch(f'{_MOD}._use_raiden_glm_admission', return_value=True), _patch_raiden_worker_env(parallelism=4, wait_pull_timeout=5.0):
        worker.process_send_load(meta_cancel)
    assert req_id_cancel not in worker._active_tensor_refs
    assert req_id_cancel in worker._stage3_terminal_sends
    assert worker._stage3_terminal_sends[req_id_cancel].device_buffers == ()

# ---------------------------------------------------------------------------
# Consumer Worker Staging, Range Accumulation & Fail-Closed Interval Tiling
# ---------------------------------------------------------------------------

_MOD = 'vllm_torchtpu.distributed.kv_transfer.tpu_connector'

class _FakeConsumerRaidenEngine:
    """Mock low-level Raiden transfer engine for consumer worker testing."""

    def __init__(self) -> None:
        """Initializes call tracking lists and queued completion results."""
        self.calls: list[Any] = []
        self.poll_results: list[tuple[list[str], list[str], list[str]]] = []
        self.complete_read_results: list[tuple[list[str], list[str], list[str]]] = []

    def register_read(self, req_id: str, uuid: int, block_ids: list[int]) -> int:
        """Records a register_read invocation."""
        self.calls.append(('register_read', req_id, uuid, block_ids))
        return 1

    def start_read(self, req_id: str, uuid: int, endpoint: str, remote_blocks: list[int], local_blocks: list[int]) -> int:
        """Records a start_read invocation."""
        self.calls.append(('start_read', req_id, uuid, endpoint, remote_blocks, local_blocks))
        return 2

    def complete_read(self) -> tuple[list[str], list[str], list[str]]:
        """Returns queued complete_read completion triples."""
        self.calls.append(('complete_read',))
        if self.complete_read_results:
            return self.complete_read_results.pop(0)
        if self.poll_results:
            return self.poll_results.pop(0)
        return ([], [], [])

    def poll_stats(self) -> tuple[list[str], list[str], list[str]]:
        """Returns queued poll_stats completion triples."""
        self.calls.append(('poll_stats',))
        if self.poll_results:
            return self.poll_results.pop(0)
        if self.complete_read_results:
            return self.complete_read_results.pop(0)
        return ([], [], [])

class _FakeConsumerRaidenControllerFacade:
    """Mock controller client recording RPC invocations."""

    def __init__(self) -> None:
        """Initializes lists to record controller RPC calls."""
        self.register_work_unit_calls: list[dict[str, Any]] = []
        self.register_request_blocks_calls: list[dict[str, Any]] = []
        self.complete_request_blocks_calls: list[dict[str, Any]] = []
        self.cancel_request_blocks_calls: list[dict[str, Any]] = []
        self.start_transfer_calls: list[dict[str, Any]] = []
        self.cancel_request_blocks_result: bool = True
        self.start_transfer_result: bool = True

    def register_work_unit(self, **kwargs: Any) -> None:
        """Records a register_work_unit invocation."""
        self.register_work_unit_calls.append(kwargs)

    def register_request_blocks(self, **kwargs: Any) -> None:
        """Records a register_request_blocks invocation."""
        self.register_request_blocks_calls.append(kwargs)

    def complete_request_blocks(self, **kwargs: Any) -> None:
        """Records a complete_request_blocks invocation."""
        self.complete_request_blocks_calls.append(kwargs)

    def cancel_request_blocks_if_unclaimed(self, **kwargs: Any) -> bool:
        """Records a cancel_request_blocks_if_unclaimed invocation."""
        self.cancel_request_blocks_calls.append(kwargs)
        return self.cancel_request_blocks_result

    def start_transfer(self, **kwargs: Any) -> bool:
        """Records a start_transfer invocation."""
        self.start_transfer_calls.append(kwargs)
        return self.start_transfer_result

def _make_mock_consumer_vllm_config(*, block_size: int=1024, dp_rank: int=0, dp_size: int=4, tp_size: int=1) -> Any:
    """Creates mock VllmConfig tailored for Stage-3 consumer decode tests."""
    cfg = MagicMock()
    cfg.kv_transfer_config.is_kv_producer = False
    cfg.kv_transfer_config.kv_connector_extra_config = {}
    cfg.cache_config.block_size = block_size
    cfg.cache_config.enable_prefix_caching = False
    cfg.model_config.max_model_len = 16384
    cfg.model_config.hf_config = SimpleNamespace(num_key_value_heads=8, head_dim=128)
    cfg.parallel_config.data_parallel_rank = dp_rank
    cfg.parallel_config.data_parallel_size = dp_size
    cfg.parallel_config.tensor_parallel_size = tp_size
    cfg.parallel_config.prefill_context_parallel_size = 1
    cfg.parallel_config.cp_kv_cache_interleave_size = 256
    return cfg

def _make_consumer_test_worker(*, block_size: int=1024, dp_rank: int=0, dp_size: int=4, tp_size: int=1) -> tuple[TPURaidenConnectorWorker, _FakeRaidenControllerFacade, _FakeRaidenEngine]:
    """Constructs consumer TPURaidenConnectorWorker with mocked state."""
    cfg = _make_mock_consumer_vllm_config(block_size=block_size, dp_rank=dp_rank, dp_size=dp_size, tp_size=tp_size)
    with patch(f'{_MOD}.get_tensor_model_parallel_rank', return_value=0), patch(f'{_MOD}.get_tensor_model_parallel_world_size', return_value=tp_size), patch(f'{_MOD}.dist_utils.get_node_id', return_value=0), patch(f'{_MOD}.dist_utils.get_host_ip', return_value='127.0.0.1'), patch(f'{_MOD}.dist_utils.get_kv_transfer_port', return_value=9100):
        worker = TPURaidenConnectorWorker(cfg)
    engine = _FakeConsumerRaidenEngine()
    facade = _FakeConsumerRaidenControllerFacade()
    unit = SimpleNamespace(job_name='decode-worker', job_replica_id='consumer-engine-rank0', data_name='kv.fa', data_replica_idx=dp_rank)
    worker._raiden_transfer_engine = engine
    worker._raiden_controller_facade = facade
    worker._raiden_controller_address = 'decode-controller.test:27000'
    worker._raiden_work_unit = unit
    worker._local_raiden_transfer_rank = MagicMock(return_value=0)
    worker._raiden_manifest = SimpleNamespace(geometry_by_tag=lambda: {'fa': {'live_bytes_per_block': block_size * 1024}}, tag_counts=lambda: {'fa': 15})
    return (worker, facade, engine)

def test_consumer_accumulates_ranges() -> None:
    """Simulates out-of-order chunk arrivals for chunks 1 and 0.

    Verifies that self._stage3_recv_intervals[req_id] accumulates both
    intervals accurately regardless of arrival order.
    """
    worker, _, engine = _make_consumer_test_worker(block_size=1024)
    req_id = 'req-out-of-order'
    worker._update_stage3_state_after_alloc(req_id=req_id, expected_tokens=2048)
    assert worker._stage3_recv_intervals[req_id] == []
    worker._start_stage3_recv_if_ready(req_id='chunk-1', uuid=101, endpoint='10.0.0.1:9000', remote_blocks=[10, 11], local_blocks=[0, 1], token_base=1024, num_tokens=1024, engine=engine)
    worker._stage3_in_flight_recv_chunks['chunk-1'] = (req_id, 1024, 1024)
    engine.complete_read_results = [([], ['chunk-1'], [])]
    worker._poll_finished(engine)
    assert (1024, 2048) in worker._stage3_recv_intervals[req_id]
    assert len(worker._stage3_recv_intervals[req_id]) == 1
    worker._start_stage3_recv_if_ready(req_id='chunk-0', uuid=100, endpoint='10.0.0.1:9000', remote_blocks=[8, 9], local_blocks=[2, 3], token_base=0, num_tokens=1024, engine=engine)
    worker._stage3_in_flight_recv_chunks['chunk-0'] = (req_id, 0, 1024)
    engine.complete_read_results = [([], ['chunk-0'], [])]
    worker._poll_finished(engine)
    assert (1024, 2048) in worker._stage3_recv_intervals[req_id]
    assert (0, 1024) in worker._stage3_recv_intervals[req_id]
    assert len(worker._stage3_recv_intervals[req_id]) == 2

def test_destination_pages_preallocated() -> None:
    """Verifies allocation size matches ceil(total_tokens / block_size).

    Asserts that destination pages are calculated identically regardless of the
    number of sub-chunks the prompt will be delivered across.
    """
    worker, _, _ = _make_consumer_test_worker(block_size=1024)
    test_cases = [(512, 1024, 1), (1024, 1024, 1), (1025, 1024, 2), (2048, 1024, 2), (3500, 1024, 4), (4096, 1024, 4), (4097, 1024, 5), (8192, 1024, 8)]
    for total_tokens, block_size, expected_pages in test_cases:
        worker.vllm_config.cache_config.block_size = block_size
        req = SimpleNamespace(request_id=f'req-alloc-{total_tokens}', kv_transfer_params={'num_tokens': total_tokens})
        pages = worker._update_stage3_state_after_alloc(req)
        assert pages == expected_pages
        assert pages == math.ceil(total_tokens / block_size)
        assert worker._stage3_recv_intervals[f'req-alloc-{total_tokens}'] == []

def test_subchunk_start_read_dispatch() -> None:
    """Verifies start_read dispatch with 50-bit UUID and block IDs."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)
    worker._start_stage3_recv_if_ready(req_id='chunk-test-dispatch', uuid=54321, endpoint='10.0.0.2:9100', remote_blocks=[5, 6], local_blocks=[12, 13], token_base=2048, num_tokens=1024, engine=engine)
    assert len(engine.calls) == 1
    call = engine.calls[0]
    assert call[0] == 'start_read'
    assert call[1] == 'chunk-test-dispatch'
    assert call[2] == 54321
    assert call[3] == '10.0.0.2:9100'
    assert call[4] == [5, 6]
    assert call[5] == [12, 13]
    assert worker._stage3_in_flight_recv_chunks['chunk-test-dispatch'] == ('chunk-test-dispatch', 2048, 1024)

def test_consumer_accumulates_three_sequential_chunks() -> None:
    """Verifies sequential accumulation across 3 sub-chunks in sequence."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)
    req_id = 'req-3-chunks'
    worker._update_stage3_state_after_alloc(req_id=req_id, expected_tokens=3072)
    chunks = [('c0', 0, 1024), ('c1', 1024, 1024), ('c2', 2048, 1024)]
    for chunk_id, base, num in chunks:
        worker._start_stage3_recv_if_ready(req_id=chunk_id, uuid=200 + base, endpoint='10.0.0.1:9000', token_base=base, num_tokens=num, engine=engine)
        worker._stage3_in_flight_recv_chunks[chunk_id] = (req_id, base, num)
        engine.complete_read_results = [([], [chunk_id], [])]
        worker._poll_finished(engine)
    intervals = worker._stage3_recv_intervals[req_id]
    assert len(intervals) == 3
    assert (0, 1024) in intervals
    assert (1024, 2048) in intervals
    assert (2048, 3072) in intervals

def test_legacy_consumer_process_send_load_accumulates_intervals() -> None:
    """Verifies non-Stage-3 process_send_load path registers intervals."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)
    with patch.object(worker, '_raiden_stage3_enabled', return_value=False):
        meta = TPUConnectorMetadata(reqs_to_load={'legacy-req': LoadMeta(uuid=777, local_block_ids=[0, 1], remote_block_ids=[10, 11], remote_host='10.0.0.1', remote_port=9000, token_base=0, is_final=True)})
        meta.reqs_to_load['legacy-req'].num_tokens = 2048
        worker.process_send_load(meta)
        engine.complete_read_results = [([], ['legacy-req'], [])]
        worker._poll_finished(engine)
        assert (0, 2048) in worker._stage3_recv_intervals['legacy-req']

def test_consumer_cleanup_purges_recv_intervals() -> None:
    """Verifies get_finished cleans up _stage3_recv_intervals on retirement."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)
    req_id = 'req-clean'
    worker._stage3_recv_intervals[req_id] = [(0, 1024), (1024, 2048)]
    worker._stage3_in_flight_recv_chunks[req_id] = (req_id, 0, 1024)
    worker._stage3_recv_chunk_meta[req_id] = (0, 1024)
    worker.get_finished(finished_req_ids={req_id})
    assert req_id not in worker._stage3_recv_intervals
    assert req_id not in worker._stage3_in_flight_recv_chunks
    assert req_id not in worker._stage3_recv_chunk_meta

def test_update_stage3_state_after_alloc_calling_conventions() -> None:
    """Verifies _update_stage3_state_after_alloc polymorphic calling styles."""
    worker, _, _ = _make_consumer_test_worker(block_size=1024)
    req_obj = SimpleNamespace(request_id='req-dict', kv_transfer_params={'num_tokens': 4096})
    pages = worker._update_stage3_state_after_alloc(req_obj)
    assert pages == 4
    assert worker._stage3_recv_intervals['req-dict'] == []
    req_none_params = SimpleNamespace(request_id='req-none-params', kv_transfer_params=None)
    pages = worker._update_stage3_state_after_alloc(req_none_params, expected_tokens=3072)
    assert pages == 3
    assert worker._stage3_recv_intervals['req-none-params'] == []
    pages = worker._update_stage3_state_after_alloc('req-str-pos', expected_tokens=2048)
    assert pages == 2
    assert worker._stage3_recv_intervals['req-str-pos'] == []
    pages = worker._update_stage3_state_after_alloc(req_id='req-kwarg', expected_tokens=1024)
    assert pages == 1
    assert worker._stage3_recv_intervals['req-kwarg'] == []
    pages = worker._update_stage3_state_after_alloc('req-extra-args', [1, 2], 'extra1', 'extra2', req_id='req-extra-args', expected_tokens=1024)
    assert pages == 1
    assert worker._stage3_recv_intervals['req-extra-args'] == []

def test_update_stage3_state_after_alloc_malformed_and_empty_params() -> None:
    """Verifies allocation update handles zero or negative tokens."""
    worker, _, _ = _make_consumer_test_worker(block_size=1024)
    pages = worker._update_stage3_state_after_alloc(req_id='req-zero', expected_tokens=0)
    assert pages == 0
    assert worker._stage3_recv_intervals['req-zero'] == []
    pages = worker._update_stage3_state_after_alloc(req_id='req-neg', expected_tokens=-500)
    assert pages == 0
    assert worker._stage3_recv_intervals['req-neg'] == []
    pages = worker._update_stage3_state_after_alloc(req_id='req-none-tok')
    assert pages == 0
    assert worker._stage3_recv_intervals['req-none-tok'] == []
    pages = worker._update_stage3_state_after_alloc(req_id='req-bad-str', expected_tokens='not_an_int')
    assert pages == 0
    assert worker._stage3_recv_intervals['req-bad-str'] == []

def test_register_in_flight_recv_chunk_coordinates_and_aliases() -> None:
    """Verifies coordinate tracking and alias mapping in _register_in_flight."""
    worker, _, _ = _make_consumer_test_worker(block_size=1024)
    worker._register_in_flight_recv_chunk(transfer_id='transfer-chunk-1', destination_req_id='dest-req-1', token_base=1024, num_tokens=1024, alias_id='alias-chunk-1')
    assert 'dest-req-1' in worker._stage3_recv_intervals
    assert worker._stage3_recv_intervals['dest-req-1'] == []
    assert worker._stage3_in_flight_recv_chunks['transfer-chunk-1'] == ('dest-req-1', 1024, 1024)
    assert worker._stage3_in_flight_recv_chunks['alias-chunk-1'] == ('dest-req-1', 1024, 1024)
    assert worker._stage3_recv_chunk_meta['dest-req-1'] == (1024, 1024)

def test_register_in_flight_recv_chunk_invalid_inputs_ignored() -> None:
    """Verifies invalid coordinates are ignored during chunk registration."""
    worker, _, _ = _make_consumer_test_worker(block_size=1024)
    worker._register_in_flight_recv_chunk(transfer_id='chunk-zero-tok', destination_req_id='req-dest-1', token_base=0, num_tokens=0)
    assert 'chunk-zero-tok' not in worker._stage3_in_flight_recv_chunks
    worker._register_in_flight_recv_chunk(transfer_id='chunk-neg-tok', destination_req_id='req-dest-2', token_base=0, num_tokens=-100)
    assert 'chunk-zero-tok' not in worker._stage3_in_flight_recv_chunks
    worker._register_in_flight_recv_chunk(transfer_id='chunk-neg-base', destination_req_id='req-dest-3', token_base=-10, num_tokens=1024)
    assert 'chunk-neg-base' not in worker._stage3_in_flight_recv_chunks

def test_subchunk_start_read_with_destination_alias_resolution() -> None:
    """Verifies _start_stage3_recv_if_ready resolves destination alias."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)
    worker._stage3_destination_req_ids['prod-chunk-1'] = 'consumer-req-42'
    worker._start_stage3_recv_if_ready(req_id='prod-chunk-1', uuid=88888, endpoint='10.0.0.5:9100', remote_blocks=[20, 21], local_blocks=[30, 31], token_base=0, num_tokens=1024, engine=engine)
    assert len(engine.calls) == 1
    assert engine.calls[0] == ('start_read', 'prod-chunk-1', 88888, '10.0.0.5:9100', [20, 21], [30, 31])
    assert worker._stage3_in_flight_recv_chunks['prod-chunk-1'] == ('consumer-req-42', 0, 1024)
    assert worker._stage3_in_flight_recv_chunks['consumer-req-42'] == ('consumer-req-42', 0, 1024)

def test_consumer_duplicate_chunk_completion_idempotent() -> None:
    """Verifies duplicate chunk completions do not insert duplicate ranges."""
    worker, _, _ = _make_consumer_test_worker(block_size=1024)
    req_id = 'req-idempotent'
    worker._stage3_recv_intervals[req_id] = []
    worker._stage3_in_flight_recv_chunks['chunk-dup'] = (req_id, 0, 1024)
    worker._record_stage3_recv_chunk_completion('chunk-dup')
    assert worker._stage3_recv_intervals[req_id] == [(0, 1024)]
    worker._stage3_in_flight_recv_chunks['chunk-dup-2'] = (req_id, 0, 1024)
    worker._record_stage3_recv_chunk_completion('chunk-dup-2')
    assert worker._stage3_recv_intervals[req_id] == [(0, 1024)]
    assert len(worker._stage3_recv_intervals[req_id]) == 1

def test_poll_finished_failed_chunk_purged_without_recording_interval() -> None:
    """Verifies failed chunks are purged without recording partial ranges."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)
    req_id = 'req-fail-purge'
    worker._update_stage3_state_after_alloc(req_id=req_id, expected_tokens=2048)
    with patch.object(worker, '_raiden_stage3_enabled', return_value=True):
        worker._start_stage3_recv_if_ready(req_id='chunk-failing', uuid=99999, endpoint='10.0.0.1:9000', token_base=0, num_tokens=1024, engine=engine, destination_req_id=req_id)
        assert 'chunk-failing' in worker._stage3_in_flight_recv_chunks
        engine.complete_read_results = [([], [], ['chunk-failing'])]
        worker._poll_finished(engine)
        assert 'chunk-failing' not in worker._stage3_in_flight_recv_chunks
        assert worker._stage3_recv_intervals.get(req_id, []) == []

def test_process_send_load_fallback_num_tokens_from_block_count() -> None:
    """Verifies fallback token count derivation from local_block_ids."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)
    with patch.object(worker, '_raiden_stage3_enabled', return_value=False):
        meta = TPUConnectorMetadata(reqs_to_load={'legacy-fallback-req': LoadMeta(uuid=12345, local_block_ids=[0, 1, 2], remote_block_ids=[10, 11, 12], remote_host='10.0.0.3', remote_port=9100, token_base=0, is_final=True)})
        worker.process_send_load(meta)
        assert worker._stage3_in_flight_recv_chunks['legacy-fallback-req'] == ('legacy-fallback-req', 0, 3072)
        engine.complete_read_results = [([], ['legacy-fallback-req'], [])]
        worker._poll_finished(engine)
        assert worker._stage3_recv_intervals['legacy-fallback-req'] == [(0, 3072)]

def test_consumer_cleanup_with_source_alias_mapping() -> None:
    """Verifies get_finished cleans up both destination and source aliases."""
    worker, _, _ = _make_consumer_test_worker(block_size=1024)
    dest_id = 'dest-req-cleanup'
    source_id = 'source-req-cleanup'
    worker._stage3_source_req_ids[dest_id] = source_id
    worker._stage3_destination_req_ids[source_id] = dest_id
    worker._stage3_recv_intervals[dest_id] = [(0, 1024)]
    worker._stage3_recv_intervals[source_id] = [(0, 1024)]
    worker._stage3_in_flight_recv_chunks[dest_id] = (dest_id, 0, 1024)
    worker._stage3_in_flight_recv_chunks[source_id] = (dest_id, 0, 1024)
    worker._stage3_recv_chunk_meta[dest_id] = (0, 1024)
    worker._stage3_recv_chunk_meta[source_id] = (0, 1024)
    worker.get_finished(finished_req_ids={dest_id})
    assert dest_id not in worker._stage3_recv_intervals
    assert dest_id not in worker._stage3_in_flight_recv_chunks
    assert dest_id not in worker._stage3_recv_chunk_meta
    assert source_id not in worker._stage3_recv_intervals
    assert source_id not in worker._stage3_in_flight_recv_chunks
    assert source_id not in worker._stage3_recv_chunk_meta
    assert source_id not in worker._stage3_destination_req_ids
    assert dest_id not in worker._stage3_source_req_ids

def test_consumer_accumulates_non_contiguous_intervals() -> None:
    """Verifies accumulator records disjoint [0, 1024) and [2048, 3072)."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)
    req_id = 'req-disjoint'
    worker._update_stage3_state_after_alloc(req_id=req_id, expected_tokens=3072)
    worker._start_stage3_recv_if_ready(req_id='chunk-0', uuid=10, token_base=0, num_tokens=1024, engine=engine, destination_req_id=req_id)
    worker._start_stage3_recv_if_ready(req_id='chunk-2', uuid=12, token_base=2048, num_tokens=1024, engine=engine, destination_req_id=req_id)
    engine.complete_read_results = [([], ['chunk-0', 'chunk-2'], [])]
    worker._poll_finished(engine)
    intervals = worker._stage3_recv_intervals[req_id]
    assert len(intervals) == 2
    assert (0, 1024) in intervals
    assert (2048, 3072) in intervals

def test_record_stage3_recv_chunk_completion_fallback_to_chunk_meta() -> None:
    """Verifies completion falls back to _stage3_recv_chunk_meta if needed."""
    worker, _, _ = _make_consumer_test_worker(block_size=1024)
    req_id = 'req-meta-fallback'
    worker._stage3_recv_intervals[req_id] = []
    worker._stage3_recv_chunk_meta[req_id] = (1024, 1024)
    worker._record_stage3_recv_chunk_completion(req_id)
    assert worker._stage3_recv_intervals[req_id] == [(1024, 2048)]
    assert req_id not in worker._stage3_recv_chunk_meta

def test_tiling_check_contiguous_success() -> None:
    """Tests complete tiling of [0, 4096) via contiguous chunks returns True."""
    intervals = [(0, 1024), (1024, 2048), (2048, 3072), (3072, 4096)]
    assert _validate_interval_tiling(intervals, 4096) is True

def test_tiling_check_gap_rejection() -> None:
    """Tests that intervals missing [1024, 2048) returns False due to gap."""
    intervals = [(0, 1024), (2048, 3072), (3072, 4096)]
    assert _validate_interval_tiling(intervals, 4096) is False

def test_tiling_check_overlapping_success() -> None:
    """Tests that overlapping intervals correctly merge and return True."""
    intervals = [(0, 1536), (1024, 3072), (3072, 4096)]
    assert _validate_interval_tiling(intervals, 4096) is True
    assert _validate_interval_tiling([(0, 1536), (1024, 3072)], 3072) is True

def test_tiling_check_shortfall_rejection() -> None:
    """Tests that intervals with shortfall tokens return False."""
    intervals = [(0, 2000), (2000, 4000)]
    assert _validate_interval_tiling(intervals, 4096) is False

def test_tiling_check_empty_and_invalid_inputs() -> None:
    """Tests that empty, None, or offset intervals return False."""
    assert _validate_interval_tiling([], 4096) is False
    assert _validate_interval_tiling(None, 4096) is False
    assert _validate_interval_tiling([(0, 1024)], 0) is False
    assert _validate_interval_tiling([(0, 1024)], -100) is False
    assert _validate_interval_tiling([(100, 4096)], 4096) is False
    assert _validate_interval_tiling([(1024, 2048), (0, 1024), (0, 1024)], 2048) is True
    assert _validate_interval_tiling([(3072, 4096), (0, 1024), (2048, 3072), (1024, 2048)], 4096) is True

def test_get_finished_validates_interval_tiling_admission() -> None:
    """Verifies interval tiling admission during get_finished."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)
    req_complete = 'req-complete'
    worker._update_stage3_state_after_alloc(req_id=req_complete, expected_tokens=2048)
    worker._stage3_recv_intervals[req_complete] = [(0, 1024), (1024, 2048)]
    worker._done_recving.add(req_complete)
    req_gap = 'req-gap'
    worker._update_stage3_state_after_alloc(req_id=req_gap, expected_tokens=3072)
    worker._stage3_recv_intervals[req_gap] = [(0, 1024), (2048, 3072)]
    worker._done_recving.add(req_gap)
    req_shortfall = 'req-shortfall'
    worker._update_stage3_state_after_alloc(req_id=req_shortfall, expected_tokens=2048)
    worker._stage3_recv_intervals[req_shortfall] = [(0, 1024)]
    worker._done_recving.add(req_shortfall)
    _, done_recving = worker.get_finished()
    assert req_complete in done_recving
    assert req_gap not in done_recving
    assert req_shortfall not in done_recving

def test_validate_interval_tiling_advanced_boundaries() -> None:
    """Tests comprehensive boundary conditions for _validate_interval_tiling."""
    assert _validate_interval_tiling([(0, 2048)], 2048) is True
    assert _validate_interval_tiling([(0, 3000)], 2048) is True
    assert _validate_interval_tiling([(0, 2048), (512, 1024)], 2048) is True
    assert _validate_interval_tiling([(0, 512), (1024, 1536), (2048, 2560)], 2560) is False
    assert _validate_interval_tiling([(1, 1024)], 1024) is False
    assert _validate_interval_tiling([(1024, 512)], 1024) is False
    assert _validate_interval_tiling([(0, 0), (0, 1024)], 1024) is True
    assert _validate_interval_tiling([(-50, 1024)], 1024) is True

def test_is_stage3_recv_complete_fallback_and_resolution_paths() -> None:
    """Verifies token resolution paths for _is_stage3_recv_complete."""
    worker, _, _ = _make_consumer_test_worker(block_size=1024)
    assert worker._is_stage3_recv_complete('untracked-req') is True
    req_direct = 'req-direct-tokens'
    worker._stage3_recv_intervals[req_direct] = [(0, 1024), (1024, 2048)]
    worker._stage3_submitted_load_tokens[req_direct] = 2048
    assert worker._is_stage3_recv_complete(req_direct) is True
    req_direct_incomplete = 'req-direct-incomplete'
    worker._stage3_recv_intervals[req_direct_incomplete] = [(0, 1024)]
    worker._stage3_submitted_load_tokens[req_direct_incomplete] = 2048
    assert worker._is_stage3_recv_complete(req_direct_incomplete) is False
    dest_alias_req = 'dest-alias-req'
    source_alias_req = 'source-alias-req'
    worker._stage3_source_req_ids[dest_alias_req] = source_alias_req
    worker._stage3_submitted_load_tokens[source_alias_req] = 2048
    worker._stage3_recv_intervals[dest_alias_req] = [(0, 1024), (1024, 2048)]
    assert worker._is_stage3_recv_complete(dest_alias_req) is True
    dest_alias_incomplete = 'dest-alias-incomplete'
    source_alias_incomplete = 'source-alias-incomplete'
    worker._stage3_source_req_ids[dest_alias_incomplete] = source_alias_incomplete
    worker._stage3_submitted_load_tokens[source_alias_incomplete] = 3072
    worker._stage3_recv_intervals[dest_alias_incomplete] = [(0, 1024), (1024, 2048)]
    assert worker._is_stage3_recv_complete(dest_alias_incomplete) is False
    req_block_fallback = 'req-block-fallback'
    worker._load_block_ids[req_block_fallback] = [10, 11, 12]
    worker._stage3_recv_intervals[req_block_fallback] = [(0, 1024), (1024, 2048), (2048, 3072)]
    assert worker._is_stage3_recv_complete(req_block_fallback) is True
    req_block_shortfall = 'req-block-shortfall'
    worker._load_block_ids[req_block_shortfall] = [10, 11, 12]
    worker._stage3_recv_intervals[req_block_shortfall] = [(0, 2048)]
    assert worker._is_stage3_recv_complete(req_block_shortfall) is False

def test_tpu_connector_worker_staticmethod_binding() -> None:
    """Verifies _validate_interval_tiling binding on connector worker class."""
    intervals = [(0, 1024), (1024, 2048)]
    assert TPURaidenConnectorWorker._validate_interval_tiling(intervals, 2048) is True
    assert TPURaidenConnectorWorker._validate_interval_tiling(intervals, 3072) is False
    worker, _, _ = _make_consumer_test_worker(block_size=1024)
    assert worker._validate_interval_tiling(intervals, 2048) is True
    assert worker._validate_interval_tiling(intervals, 3072) is False

def test_get_finished_preserves_unadmitted_requests_until_completion() -> None:
    """Verifies unadmitted requests persist across passes until completed."""
    worker, _, _ = _make_consumer_test_worker(block_size=1024)
    req_id = 'req-two-step-arrival'
    worker._update_stage3_state_after_alloc(req_id=req_id, expected_tokens=2048)
    worker._stage3_recv_intervals[req_id] = [(0, 1024)]
    worker._done_recving.add(req_id)
    _, done_recving_step1 = worker.get_finished()
    assert req_id not in done_recving_step1
    worker._stage3_recv_intervals[req_id].append((1024, 2048))
    worker._done_recving.add(req_id)
    _, done_recving_step2 = worker.get_finished()
    assert req_id in done_recving_step2

def test_failed_chunk_triggers_recomputation() -> None:
    """Simulates chunk 2 failing in _failed_recving and verifies recomputation.

    Registers 3 sub-chunks for a 3072-token prompt across blocks [101, 102,
    103]. Chunks 0 and 1 succeed, accumulating intervals [0, 1024) and [1024,
    2048). Chunk 2 fails in engine completion polling. Verifies that the base
    request is placed into _failed_recving, partial interval state is cleared,
    get_finished() reports the request as completed, and
    get_block_ids_with_load_errors() returns the full allocated block set
    {101, 102, 103} for local recomputation.
    """
    worker, _, engine = _make_consumer_test_worker(block_size=1024)
    req_id = 'req-subchunk-fail'
    blocks = [101, 102, 103]
    worker._update_stage3_state_after_alloc(req_id=req_id, expected_tokens=3072)
    worker._load_block_ids[req_id] = list(blocks)
    worker._stage3_controller_accepted.add(req_id)
    with patch.object(worker, '_raiden_stage3_enabled', return_value=True):
        worker._start_stage3_recv_if_ready(req_id='chunk-0', uuid=10001, endpoint='10.0.0.1:9000', token_base=0, num_tokens=1024, engine=engine, destination_req_id=req_id)
        worker._start_stage3_recv_if_ready(req_id='chunk-1', uuid=10002, endpoint='10.0.0.1:9000', token_base=1024, num_tokens=1024, engine=engine, destination_req_id=req_id)
        worker._start_stage3_recv_if_ready(req_id='chunk-2', uuid=10003, endpoint='10.0.0.1:9000', token_base=2048, num_tokens=1024, engine=engine, destination_req_id=req_id)
        engine.complete_read_results = [([], ['chunk-0', 'chunk-1'], [])]
        worker._poll_finished(engine)
        assert set(worker._stage3_recv_intervals[req_id]) == {(0, 1024), (1024, 2048)}
        assert req_id not in worker._failed_recving
        engine.complete_read_results = [([], [], ['chunk-2'])]
        worker._poll_finished(engine)
        assert req_id in worker._failed_recving
        assert worker._stage3_recv_intervals.get(req_id) is None
        assert 'chunk-2' not in worker._stage3_in_flight_recv_chunks
        done_sending, done_recving = worker.get_finished()
        assert req_id in done_recving
        failed_blocks = worker.get_block_ids_with_load_errors()
        assert failed_blocks == {101, 102, 103}
        assert worker.get_block_ids_with_load_errors() == set()

def test_tiling_timeout_triggers_error() -> None:
    """Simulates missing final chunk causing timeout without deadlock.

    Registers 2 sub-chunks for a 2048-token prompt across blocks [201, 202].
    Chunk 0 completes, while Chunk 1 is lost / delayed. Simulates timeout
    expiration in _poll_finished / _wait_for_recving. Verifies that the request
    enters error recovery, is marked in _failed_recving, reports done_recving
    in get_finished(), and surfaces block IDs {201, 202} for local
    recomputation.
    """
    worker, _, engine = _make_consumer_test_worker(block_size=1024)
    req_id = 'req-tiling-timeout'
    blocks = [201, 202]
    worker._update_stage3_state_after_alloc(req_id=req_id, expected_tokens=2048)
    worker._load_block_ids[req_id] = list(blocks)
    worker._stage3_controller_accepted.add(req_id)
    with patch.object(worker, '_raiden_stage3_enabled', return_value=True):
        worker._start_stage3_recv_if_ready(req_id='chunk-0-ok', uuid=20001, endpoint='10.0.0.1:9000', token_base=0, num_tokens=1024, engine=engine, destination_req_id=req_id)
        worker._start_stage3_recv_if_ready(req_id='chunk-1-lost', uuid=20002, endpoint='10.0.0.1:9000', token_base=1024, num_tokens=1024, engine=engine, destination_req_id=req_id)
        engine.complete_read_results = [([], ['chunk-0-ok'], [])]
        worker._poll_finished(engine)
        assert worker._stage3_recv_intervals[req_id] == [(0, 1024)]
        worker._stage3_load_start_times[req_id] = time.perf_counter() - 10000.0
        engine.complete_read_results = [([], [], [])]
        worker._poll_finished(engine)
        assert req_id in worker._failed_recving
        assert worker._stage3_recv_intervals.get(req_id) is None
        done_sending, done_recving = worker.get_finished()
        assert req_id in done_recving
        assert worker.get_block_ids_with_load_errors() == {201, 202}

def test_wait_for_recving_timeout_recovers_without_deadlock() -> None:
    """Verifies that _wait_for_recving times out safely on incomplete chunks."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)
    req_id = 'req-wait-timeout'
    blocks = [301, 302]
    worker._update_stage3_state_after_alloc(req_id=req_id, expected_tokens=2048)
    worker._load_block_ids[req_id] = list(blocks)
    worker._stage3_controller_accepted.add(req_id)
    with patch.object(worker, '_raiden_stage3_enabled', return_value=True), patch(f'{_MOD}.dist_utils.get_p2p_wait_pull_timeout', return_value=0.01):
        worker._start_stage3_recv_if_ready(req_id='chunk-wait-0', uuid=30001, endpoint='10.0.0.1:9000', token_base=0, num_tokens=1024, engine=engine, destination_req_id=req_id)
        engine.complete_read_results = [([], ['chunk-wait-0'], [])]
        worker._wait_for_recving({req_id})
        assert req_id in worker._failed_recving
        assert worker._stage3_recv_intervals.get(req_id) is None
        _, done_recving = worker.get_finished()
        assert req_id in done_recving
        assert worker.get_block_ids_with_load_errors() == {301, 302}

def test_legacy_consumer_chunk_failure_triggers_recomputation() -> None:
    """Verifies sub-chunk failure in non-Stage3 legacy path."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)
    req_id = 'req-legacy-fail'
    blocks = [401, 402]
    worker._load_block_ids[req_id] = list(blocks)
    worker._register_in_flight_recv_chunk(transfer_id='legacy-chunk-0', destination_req_id=req_id, token_base=0, num_tokens=1024)
    worker._register_in_flight_recv_chunk(transfer_id='legacy-chunk-1', destination_req_id=req_id, token_base=1024, num_tokens=1024)
    with patch.object(worker, '_raiden_stage3_enabled', return_value=False):
        engine.poll_results = [([], ['legacy-chunk-0'], []), ([], [], ['legacy-chunk-1'])]
        worker._poll_finished(engine)
        worker._poll_finished(engine)
        assert req_id in worker._failed_recving
        assert worker._stage3_recv_intervals.get(req_id) is None
        _, done_recving = worker.get_finished()
        assert req_id in done_recving
        assert worker.get_block_ids_with_load_errors() == {401, 402}

def test_absorbing_failure_state_ignores_late_chunks() -> None:
    """Verifies that late-arriving chunks do not resurrect a failed request."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)
    req_id = 'req-absorbing-fail'
    blocks = [501, 502]
    worker._update_stage3_state_after_alloc(req_id=req_id, expected_tokens=2048)
    worker._load_block_ids[req_id] = list(blocks)
    worker._stage3_controller_accepted.add(req_id)
    with patch.object(worker, '_raiden_stage3_enabled', return_value=True):
        worker._start_stage3_recv_if_ready(req_id='chunk-0-fails', uuid=50001, endpoint='10.0.0.1:9000', token_base=0, num_tokens=1024, engine=engine, destination_req_id=req_id)
        worker._start_stage3_recv_if_ready(req_id='chunk-1-late', uuid=50002, endpoint='10.0.0.1:9000', token_base=1024, num_tokens=1024, engine=engine, destination_req_id=req_id)
        engine.complete_read_results = [([], [], ['chunk-0-fails'])]
        worker._poll_finished(engine)
        assert req_id in worker._failed_recving
        assert worker._stage3_recv_intervals.get(req_id) is None
        engine.complete_read_results = [([], ['chunk-1-late'], [])]
        worker._poll_finished(engine)
        assert req_id in worker._failed_recving
        assert worker._stage3_recv_intervals.get(req_id) is None
        assert worker._is_stage3_recv_complete(req_id) is False
        _, done_recving = worker.get_finished()
        assert req_id in done_recving
        assert worker.get_block_ids_with_load_errors() == {501, 502}

def test_mixed_concurrent_requests_isolate_failure_and_success() -> None:
    """Verifies error isolation across concurrent requests."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)
    req_ok = 'req-ok'
    req_fail = 'req-fail'
    blocks_ok = [601, 602]
    blocks_fail = [701, 702]
    worker._update_stage3_state_after_alloc(req_id=req_ok, expected_tokens=2048)
    worker._load_block_ids[req_ok] = list(blocks_ok)
    worker._stage3_controller_accepted.add(req_ok)
    worker._update_stage3_state_after_alloc(req_id=req_fail, expected_tokens=2048)
    worker._load_block_ids[req_fail] = list(blocks_fail)
    worker._stage3_controller_accepted.add(req_fail)
    with patch.object(worker, '_raiden_stage3_enabled', return_value=True):
        worker._start_stage3_recv_if_ready(req_id='ok-0', uuid=60001, endpoint='10.0.0.1:9000', token_base=0, num_tokens=1024, engine=engine, destination_req_id=req_ok)
        worker._start_stage3_recv_if_ready(req_id='ok-1', uuid=60002, endpoint='10.0.0.1:9000', token_base=1024, num_tokens=1024, engine=engine, destination_req_id=req_ok)
        worker._start_stage3_recv_if_ready(req_id='fail-0', uuid=70001, endpoint='10.0.0.1:9000', token_base=0, num_tokens=1024, engine=engine, destination_req_id=req_fail)
        worker._start_stage3_recv_if_ready(req_id='fail-1', uuid=70002, endpoint='10.0.0.1:9000', token_base=1024, num_tokens=1024, engine=engine, destination_req_id=req_fail)
        engine.complete_read_results = [([], ['ok-0', 'ok-1', 'fail-0'], ['fail-1'])]
        worker._poll_finished(engine)
        assert req_ok not in worker._failed_recving
        assert req_fail in worker._failed_recving
        assert worker._stage3_recv_intervals.get(req_fail) is None
        _, done_recving = worker.get_finished()
        assert req_ok in done_recving
        assert req_fail in done_recving
        assert worker.get_block_ids_with_load_errors() == {701, 702}

def test_uncertain_controller_failure_timeout_triggers_recomputation() -> None:
    """Verifies expired pending controller failures trigger recomputation."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)
    req_id = 'req-uncertain-timeout'
    blocks = [801, 802]
    worker._update_stage3_state_after_alloc(req_id=req_id, expected_tokens=2048)
    worker._load_block_ids[req_id] = list(blocks)
    worker._stage3_controller_accepted.add(req_id)
    worker._stage3_pending_controller_failures[req_id] = time.perf_counter() - 5.0
    with patch.object(worker, '_raiden_stage3_enabled', return_value=True):
        engine.complete_read_results = [([], [], [])]
        worker._poll_finished(engine)
        assert req_id in worker._failed_recving
        assert req_id not in worker._stage3_pending_controller_failures
        _, done_recving = worker.get_finished()
        assert req_id in done_recving
        assert worker.get_block_ids_with_load_errors() == {801, 802}

def test_incomplete_multichunk_does_not_retire_to_terminal_loads() -> None:
    """Verifies that partial multi-chunk recvs do not enter terminal loads."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)
    req_id = 'req-multichunk-progression'
    blocks = [901, 902]
    worker._update_stage3_state_after_alloc(req_id=req_id, expected_tokens=2048)
    worker._load_block_ids[req_id] = list(blocks)
    worker._stage3_controller_accepted.add(req_id)
    with patch.object(worker, '_raiden_stage3_enabled', return_value=True):
        worker._start_stage3_recv_if_ready(req_id='chunk-step0', uuid=90001, endpoint='10.0.0.1:9000', token_base=0, num_tokens=1024, engine=engine, destination_req_id=req_id)
        worker._start_stage3_recv_if_ready(req_id='chunk-step1', uuid=90002, endpoint='10.0.0.1:9000', token_base=1024, num_tokens=1024, engine=engine, destination_req_id=req_id)
        engine.complete_read_results = [([], ['chunk-step0'], [])]
        worker._poll_finished(engine)
        assert req_id not in worker._stage3_terminal_loads
        _, done_recving = worker.get_finished()
        assert req_id not in done_recving
        engine.complete_read_results = [([], ['chunk-step1'], [])]
        worker._poll_finished(engine)
        assert req_id in worker._stage3_terminal_loads
        _, done_recving = worker.get_finished()
        assert req_id in done_recving

def test_tpu_connector_load_errors_group_index() -> None:
    """Verifies get_block_ids_with_load_errors_group_index returns FA group."""
    cfg = _make_mock_consumer_vllm_config()
    cfg.kv_transfer_config.kv_connector_extra_config = {'use_raiden_connector': True}
    kv_cache_config = MagicMock()
    kv_cache_config.kv_cache_groups = [SimpleNamespace(kv_cache_spec=MagicMock())]
    with patch(f'{_MOD}._use_raiden_stage3_transport', return_value=True), patch(f'{_MOD}.get_tensor_model_parallel_rank', return_value=0), patch(f'{_MOD}.get_tensor_model_parallel_world_size', return_value=1), patch(f'{_MOD}.dist_utils.get_node_id', return_value=0), patch(f'{_MOD}.dist_utils.get_host_ip', return_value='127.0.0.1'), patch(f'{_MOD}.dist_utils.get_kv_transfer_port', return_value=9100):
        connector = TPUConnector(cfg, role=MagicMock(), kv_cache_config=kv_cache_config)
        connector._stage3_fa_group_index = 0
        assert connector.get_block_ids_with_load_errors_group_index() == 0
        with patch(f'{_MOD}._use_raiden_stage3_transport', return_value=False):
            assert connector.get_block_ids_with_load_errors_group_index() is None

# ---------------------------------------------------------------------------
# Sliding Window Flow Control & Host Memory Slot Scaling
# ---------------------------------------------------------------------------

_MOD = 'vllm_torchtpu.distributed.kv_transfer.tpu_connector'

def _make_slot_raiden_worker(*, is_producer: bool=True, tp_rank: int=0, tp_size: int=4, dp_rank: int=0, dp_size: int=1, pcp_size: int=4, interleave_size: int=256, block_size: int=16, chunk_transfer_window: int | None=None, extra_config_window: int | None=None) -> TPURaidenConnectorWorker:
    """Creates a mock-configured TPURaidenConnectorWorker for testing.

    Args:
        is_producer: Whether the worker operates in producer role.
        tp_rank: Tensor model parallel rank.
        tp_size: Tensor model parallel world size.
        dp_rank: Data parallel rank.
        dp_size: Data parallel world size.
        pcp_size: Prefill context parallelism size.
        interleave_size: Token chunk stride per context parallel rank.
        block_size: Number of tokens per physical KV cache block.
        chunk_transfer_window: Direct window size attribute on transfer config.
        extra_config_window: Window size entry in extra config dictionary.

    Returns:
        A configured instance of TPURaidenConnectorWorker.
    """
    cfg = MagicMock(spec=VllmConfig)
    cfg.kv_transfer_config = MagicMock()
    cfg.kv_transfer_config.is_kv_producer = is_producer
    extra_cfg: dict[str, Any] = {}
    if extra_config_window is not None:
        extra_cfg['chunk_transfer_window'] = extra_config_window
    cfg.kv_transfer_config.kv_connector_extra_config = extra_cfg
    if chunk_transfer_window is not None:
        cfg.kv_transfer_config.chunk_transfer_window = chunk_transfer_window
    else:
        cfg.kv_transfer_config.chunk_transfer_window = None
    cfg.cache_config = MagicMock()
    cfg.cache_config.block_size = block_size
    cfg.cache_config.enable_prefix_caching = False
    cfg.model_config = MagicMock()
    cfg.model_config.max_model_len = 8192
    cfg.parallel_config = MagicMock()
    cfg.parallel_config.data_parallel_rank = dp_rank
    cfg.parallel_config.data_parallel_size = dp_size
    cfg.parallel_config.tensor_parallel_size = tp_size
    cfg.parallel_config.prefill_context_parallel_size = pcp_size
    cfg.parallel_config.cp_kv_cache_interleave_size = interleave_size
    with patch(f'{_MOD}.get_tensor_model_parallel_rank', return_value=tp_rank), patch(f'{_MOD}.get_tensor_model_parallel_world_size', return_value=tp_size), patch(f'{_MOD}.dist_utils.get_node_id', return_value=0), patch(f'{_MOD}.dist_utils.get_host_ip', return_value='127.0.0.1'), patch(f'{_MOD}.dist_utils.get_kv_transfer_port', return_value=9100):
        worker = TPURaidenConnectorWorker(cfg)
    return worker

def _make_req(req_id: str, prompt_len: int=4096) -> SimpleNamespace:
    """Creates a lightweight mock Request structure for unit testing.

    Args:
        req_id: Unique string identifier for the mock request.
        prompt_len: Total prompt token length for the mock request.

    Returns:
        SimpleNamespace simulating a vLLM Request object.
    """
    return SimpleNamespace(request_id=req_id, num_prompt_tokens=prompt_len, num_computed_tokens=0, prompt_token_ids=[0] * prompt_len, kv_transfer_params=None, status=RequestStatus.FINISHED_LENGTH_CAPPED)

def test_default_chunk_transfer_window_constant():
    """Verify that default CHUNK_TRANSFER_WINDOW constant is 2."""
    assert CHUNK_TRANSFER_WINDOW == 2

def test_get_chunk_transfer_window_config_precedence():
    """Verify hierarchy, precedence, and clamping for window size (W)."""
    assert _get_chunk_transfer_window(None) == 2
    cfg_empty = MagicMock(spec=VllmConfig)
    cfg_empty.kv_transfer_config = None
    assert _get_chunk_transfer_window(cfg_empty) == 2
    cfg_direct = MagicMock(spec=VllmConfig)
    cfg_direct.kv_transfer_config = MagicMock()
    cfg_direct.kv_transfer_config.chunk_transfer_window = 4
    cfg_direct.kv_transfer_config.kv_connector_extra_config = {}
    assert _get_chunk_transfer_window(cfg_direct) == 4
    cfg_extra = MagicMock(spec=VllmConfig)
    cfg_extra.kv_transfer_config = MagicMock()
    cfg_extra.kv_transfer_config.chunk_transfer_window = None
    cfg_extra.kv_transfer_config.kv_connector_extra_config = {'chunk_transfer_window': 3}
    assert _get_chunk_transfer_window(cfg_extra) == 3
    cfg_both = MagicMock(spec=VllmConfig)
    cfg_both.kv_transfer_config = MagicMock()
    cfg_both.kv_transfer_config.chunk_transfer_window = 5
    cfg_both.kv_transfer_config.kv_connector_extra_config = {'chunk_transfer_window': 10}
    assert _get_chunk_transfer_window(cfg_both) == 5
    cfg_zero = MagicMock(spec=VllmConfig)
    cfg_zero.kv_transfer_config = MagicMock()
    cfg_zero.kv_transfer_config.chunk_transfer_window = 0
    assert _get_chunk_transfer_window(cfg_zero) == 1
    cfg_neg = MagicMock(spec=VllmConfig)
    cfg_neg.kv_transfer_config = MagicMock()
    cfg_neg.kv_transfer_config.chunk_transfer_window = -4
    assert _get_chunk_transfer_window(cfg_neg) == 1
    cfg_str = MagicMock(spec=VllmConfig)
    cfg_str.kv_transfer_config = MagicMock()
    cfg_str.kv_transfer_config.chunk_transfer_window = '6'
    assert _get_chunk_transfer_window(cfg_str) == 6

def test_in_flight_window_limiting():
    """Attempt to emit 3 chunks for a request with W=2.

    Verify first 2 chunks are emitted and 3rd chunk is deferred until chunk 0
    completes.
    """
    with _patch_raiden_stage3_env(base_uuid=100001):
        scheduler = _make_raiden_scheduler(is_producer=True, pcp_size=4, interleave_size=256, chunk_transfer_window=2)
        assert scheduler.chunk_transfer_window == 2
        req = _make_req('req-window-test', prompt_len=4096)
        block_ids = list(range(256))
        scheduler.update_state_after_alloc(req, block_ids, 0)
        req_state = scheduler._stage3_send_requests['req-window-test']
        assert req_state.tokens_sent == 0
        assert scheduler._in_flight_chunks.get('req-window-test', 0) == 0
        assert scheduler._can_emit_chunk('req-window-test') is True
        sched_out1 = MagicMock(spec=SchedulerOutput)
        sched_out1.num_scheduled_tokens = {'req-window-test': 1024}
        meta1 = scheduler.build_connector_meta(sched_out1)
        assert 'req-window-test' in meta1.reqs_to_send
        send0: SendMeta = meta1.reqs_to_send['req-window-test']
        assert send0.token_base == 0
        assert send0.num_tokens == 1024
        assert send0.is_final is False
        assert scheduler._in_flight_chunks['req-window-test'] == 1
        assert scheduler._can_emit_chunk('req-window-test') is True
        assert req_state.tokens_sent == 1024
        sched_out2 = MagicMock(spec=SchedulerOutput)
        sched_out2.num_scheduled_tokens = {'req-window-test': 1024}
        meta2 = scheduler.build_connector_meta(sched_out2)
        assert 'req-window-test' in meta2.reqs_to_send
        send1: SendMeta = meta2.reqs_to_send['req-window-test']
        assert send1.token_base == 1024
        assert send1.num_tokens == 1024
        assert send1.is_final is False
        assert scheduler._in_flight_chunks['req-window-test'] == 2
        assert scheduler._can_emit_chunk('req-window-test') is False
        assert req_state.tokens_sent == 2048
        sched_out3 = MagicMock(spec=SchedulerOutput)
        sched_out3.num_scheduled_tokens = {'req-window-test': 1024}
        meta3 = scheduler.build_connector_meta(sched_out3)
        assert 'req-window-test' not in meta3.reqs_to_send
        assert scheduler._in_flight_chunks['req-window-test'] == 2
        assert req_state.tokens_sent == 2048
        assert req_state.num_computed_tokens == 3072
        conn_out = SimpleNamespace(finished_sending=['req-window-test'])
        scheduler.update_connector_output(conn_out)
        assert scheduler._in_flight_chunks['req-window-test'] == 1
        assert scheduler._can_emit_chunk('req-window-test') is True
        sched_out4 = MagicMock(spec=SchedulerOutput)
        sched_out4.num_scheduled_tokens = {'req-window-test': 0}
        meta4 = scheduler.build_connector_meta(sched_out4)
        assert 'req-window-test' in meta4.reqs_to_send
        send2: SendMeta = meta4.reqs_to_send['req-window-test']
        assert send2.token_base == 2048
        assert send2.num_tokens == 1024
        assert send2.is_final is False
        assert scheduler._in_flight_chunks['req-window-test'] == 2
        assert req_state.tokens_sent == 3072

def test_slot_pool_scaling():
    """Verify _num_raiden_slots calculation scales linearly with window size."""
    window_sizes = [1, 2, 3, 4, 8]
    slot_counts = {}
    for window_size in window_sizes:
        worker = _make_slot_raiden_worker(is_producer=True, tp_size=4, chunk_transfer_window=window_size)
        assert worker.chunk_transfer_window == window_size
        runner = MagicMock()
        nope = torch.empty((4, 8, 4, 128), dtype=torch.uint8)
        rope = torch.empty((4, 2, 4, 128), dtype=torch.uint8)
        idx = torch.empty((4, 2, 4, 256), dtype=torch.uint8)
        runner.kv_caches = [(nope, rope), idx]
        worker.runner = runner
        with patch(f'{_MOD}.dist_utils.get_raiden_transfer_num_slots', return_value=0), patch(f'{_MOD}.dist_utils.get_kv_shm_pool_gb', return_value=1.0):
            slots = worker._num_raiden_slots(max_blocks=4)
            slot_counts[window_size] = slots
    base_slots = slot_counts[1]
    assert base_slots > 0
    for window_size in window_sizes:
        assert slot_counts[window_size] == base_slots * window_size
    worker_override = _make_slot_raiden_worker(is_producer=True, chunk_transfer_window=4)
    worker_override.runner = MagicMock()
    with patch(f'{_MOD}.dist_utils.get_raiden_transfer_num_slots', return_value=7):
        assert worker_override._num_raiden_slots(max_blocks=4) == 7

def test_multi_request_independent_window_accounting():
    """Verify independent in-flight tracking across distinct requests."""
    with _patch_raiden_stage3_env(base_uuid=200001):
        scheduler = _make_raiden_scheduler(is_producer=True, pcp_size=4, interleave_size=256, chunk_transfer_window=2)
        req_a = _make_req('req-A', prompt_len=4096)
        req_b = _make_req('req-B', prompt_len=4096)
        scheduler.update_state_after_alloc(req_a, list(range(256)), 0)
        scheduler.update_state_after_alloc(req_b, list(range(256)), 0)
        for _ in range(2):
            sched_out = MagicMock(spec=SchedulerOutput)
            sched_out.num_scheduled_tokens = {'req-A': 1024}
            scheduler.build_connector_meta(sched_out)
        assert scheduler._in_flight_chunks['req-A'] == 2
        assert scheduler._can_emit_chunk('req-A') is False
        assert scheduler._in_flight_chunks.get('req-B', 0) == 0
        assert scheduler._can_emit_chunk('req-B') is True
        sched_out_b = MagicMock(spec=SchedulerOutput)
        sched_out_b.num_scheduled_tokens = {'req-B': 1024}
        meta_b = scheduler.build_connector_meta(sched_out_b)
        assert 'req-B' in meta_b.reqs_to_send
        assert scheduler._in_flight_chunks['req-B'] == 1

def test_on_chunk_completed_helper():
    """Verify on_chunk_completed decrements and removes tracked request IDs."""
    scheduler = _make_raiden_scheduler(chunk_transfer_window=2)
    scheduler._in_flight_chunks['req-1'] = 2
    scheduler.on_chunk_completed('req-1')
    assert scheduler._in_flight_chunks['req-1'] == 1
    scheduler.on_chunk_completed('req-1')
    assert 'req-1' not in scheduler._in_flight_chunks
    scheduler.on_chunk_completed('req-1')
    assert 'req-1' not in scheduler._in_flight_chunks

def test_request_finished_cleans_in_flight_state():
    """Verify request_finished purges in-flight tracking state."""
    req_id = 'req-finish-test'
    block_ids = list(range(32))
    req = MagicMock()
    req.request_id = req_id
    req.num_prompt_tokens = 2048
    req.num_computed_tokens = 2048
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    req.kv_transfer_params = {'_p_side_truncated': True}
    with _patch_raiden_stage3_env(base_uuid=300001):
        scheduler = _make_raiden_scheduler(is_producer=True, pcp_size=4, interleave_size=256, chunk_transfer_window=2)
        scheduler.update_state_after_alloc(req, block_ids, 0)
        sched_out = MagicMock(spec=SchedulerOutput)
        sched_out.num_scheduled_tokens = {req_id: 1024}
        scheduler.build_connector_meta(sched_out)
        assert scheduler._in_flight_chunks[req_id] == 1
        with patch(f'{_MOD}._resolved_reshard_controller_address', return_value='127.0.0.1:27000'):
            ok, params = scheduler.request_finished(req, block_ids)
            assert ok is True
        assert req_id not in scheduler._in_flight_chunks

def test_can_emit_chunk_custom_window_sizes():
    """Verify _can_emit_chunk across window sizes W=1, W=3, W=4."""
    sched_w1 = _make_raiden_scheduler(chunk_transfer_window=1)
    assert sched_w1._can_emit_chunk('req-w1') is True
    sched_w1._in_flight_chunks['req-w1'] = 1
    assert sched_w1._can_emit_chunk('req-w1') is False
    sched_w3 = _make_raiden_scheduler(chunk_transfer_window=3)
    assert sched_w3._can_emit_chunk('req-w3') is True
    sched_w3._in_flight_chunks['req-w3'] = 2
    assert sched_w3._can_emit_chunk('req-w3') is True
    sched_w3._in_flight_chunks['req-w3'] = 3
    assert sched_w3._can_emit_chunk('req-w3') is False

def test_update_connector_output_none_and_untracked_safe():
    """Verify update_connector_output with None, empty, or untracked inputs."""
    scheduler = _make_raiden_scheduler(chunk_transfer_window=2)
    scheduler._stage3_finished_sends['req-tracked'] = (1, (0,), 1024, {})
    scheduler._in_flight_chunks['req-tracked'] = 1
    scheduler.update_connector_output(None)
    assert scheduler._in_flight_chunks['req-tracked'] == 1
    scheduler.update_connector_output(SimpleNamespace(finished_sending=None))
    assert scheduler._in_flight_chunks['req-tracked'] == 1
    scheduler.update_connector_output(SimpleNamespace(finished_sending=['req-untracked']))
    assert scheduler._in_flight_chunks['req-tracked'] == 1
    scheduler.update_connector_output(SimpleNamespace(finished_sending=['req-tracked']))
    assert 'req-tracked' not in scheduler._stage3_finished_sends
    assert 'req-tracked' not in scheduler._in_flight_chunks

def test_on_chunk_completed_non_string_id_interop():
    """Verify on_chunk_completed and _can_emit_chunk with non-string IDs."""
    scheduler = _make_raiden_scheduler(chunk_transfer_window=2)
    numeric_id = 98765
    scheduler._in_flight_chunks[str(numeric_id)] = 2
    assert scheduler._can_emit_chunk(str(numeric_id)) is False
    assert scheduler._can_emit_chunk(numeric_id) is False
    scheduler.on_chunk_completed(numeric_id)
    assert scheduler._in_flight_chunks[str(numeric_id)] == 1
    assert scheduler._can_emit_chunk(numeric_id) is True
    scheduler.on_chunk_completed(numeric_id)
    assert str(numeric_id) not in scheduler._in_flight_chunks
    assert scheduler._can_emit_chunk(numeric_id) is True

# ---------------------------------------------------------------------------
# Prefix Caching Suffix Transfers & Hybrid Recurrent State Isolation
# ---------------------------------------------------------------------------

_MOD = 'vllm_torchtpu.distributed.kv_transfer.tpu_connector'

class _FakeHybridRaidenControllerFacade:
    """Mock controller client recording RPC invocations for testing."""

    def __init__(self) -> None:
        """Initializes recorded call lists and default returns."""
        self.register_work_unit_calls: list[dict[str, Any]] = []
        self.register_request_blocks_calls: list[dict[str, Any]] = []
        self.complete_request_blocks_calls: list[dict[str, Any]] = []
        self.cancel_request_blocks_calls: list[dict[str, Any]] = []
        self.cancel_request_blocks_result: bool = True

    def register_work_unit(self, **kwargs: Any) -> None:
        """Records a register_work_unit call."""
        self.register_work_unit_calls.append(kwargs)

    def register_request_blocks(self, **kwargs: Any) -> None:
        """Records a register_request_blocks call."""
        self.register_request_blocks_calls.append(kwargs)

    def complete_request_blocks(self, **kwargs: Any) -> None:
        """Records a complete_request_blocks call."""
        self.complete_request_blocks_calls.append(kwargs)

    def cancel_request_blocks_if_unclaimed(self, **kwargs: Any) -> bool:
        """Records a cancel_request_blocks_if_unclaimed call."""
        self.cancel_request_blocks_calls.append(kwargs)
        return self.cancel_request_blocks_result

def _make_hybrid_test_worker(*, tp_rank: int=0, tp_size: int=1, is_producer: bool=True, pcp_size: int=4, interleave_size: int=256, block_size: int=1024, token_bytes: int=64, transfer_rank: int=0) -> tuple[TPURaidenConnectorWorker, _FakeRaidenControllerFacade]:
    """Creates a TPURaidenConnectorWorker wired with fake controllers."""
    cfg = MagicMock(spec=VllmConfig)
    cfg.kv_transfer_config = MagicMock()
    cfg.kv_transfer_config.is_kv_producer = is_producer
    cfg.kv_transfer_config.kv_connector_extra_config = {}
    cfg.cache_config = MagicMock()
    cfg.cache_config.block_size = block_size
    cfg.cache_config.enable_prefix_caching = True
    cfg.model_config = MagicMock()
    cfg.model_config.max_model_len = 16384
    cfg.model_config.num_hidden_layers = 32
    cfg.model_config.get_num_layers = MagicMock(return_value=32)
    cfg.model_config.get_num_kv_heads = MagicMock(return_value=8)
    cfg.model_config.get_head_size = MagicMock(return_value=128)
    cfg.model_config.dtype = torch.bfloat16
    cfg.parallel_config = MagicMock()
    cfg.parallel_config.data_parallel_rank = 0
    cfg.parallel_config.data_parallel_size = 1
    cfg.parallel_config.tensor_parallel_size = tp_size
    cfg.parallel_config.prefill_context_parallel_size = pcp_size
    cfg.parallel_config.cp_kv_cache_interleave_size = interleave_size
    with patch(f'{_MOD}.dist_utils.get_kv_ips', return_value='127.0.0.1'), patch(f'{_MOD}.dist_utils.get_kv_ports', return_value=9100), patch(f'{_MOD}.dist_utils.get_side_channel_port', return_value='9600'), patch(f'{_MOD}.get_tensor_model_parallel_rank', return_value=tp_rank), patch(f'{_MOD}.get_tensor_model_parallel_world_size', return_value=tp_size):
        worker = TPURaidenConnectorWorker(cfg)
    facade = _FakeHybridRaidenControllerFacade()
    unit = SimpleNamespace(job_name='prefill', job_replica_id='producer-0', data_name='kv.fa', data_replica_idx=0)
    worker._raiden_controller_facade = facade
    worker._raiden_controller_address = '127.0.0.1:27000'
    worker._raiden_work_unit = unit
    worker._local_raiden_transfer_rank = MagicMock(return_value=transfer_rank)
    worker._raiden_transfer_parallelism = MagicMock(return_value=pcp_size)
    worker._stage3_fa_token_bytes = MagicMock(return_value=token_bytes)
    worker._flattened_kv_cache_tensors = MagicMock(return_value=[])
    return (worker, facade)

def test_prefix_cache_chunk_skipping() -> None:
    """Verifies that cached prefix chunks are skipped when prefix is cached."""
    scheduler = _make_raiden_scheduler(is_producer=True, block_size=16, pcp_size=4, interleave_size=256)
    req_id = 'req-prefix-skip-test'
    mock_req = MagicMock()
    mock_req.request_id = req_id
    mock_req.num_computed_tokens = 2048
    mock_req.num_prompt_tokens = 4096
    mock_req.prompt_token_ids = list(range(4096))
    mock_req.status = RequestStatus.RUNNING
    mock_req.kv_transfer_params = {'_p_side_truncated': True}
    base_uuid = 555123
    with _patch_raiden_stage3_env(base_uuid=base_uuid):
        scheduler._update_stage3_producer_state_after_alloc(mock_req, blocks=list(range(64)))
        req_state = scheduler._stage3_send_requests[req_id]
        assert req_state.cached_prefix_tokens == 2048
        assert req_state.tokens_sent == 2048
        assert req_state.next_chunk_idx == 2
        sched_output = MagicMock(spec=SchedulerOutput)
        sched_output.num_scheduled_tokens = {req_id: 1024}
        meta = scheduler.build_connector_meta(sched_output)
        assert req_id in meta.reqs_to_send
        chunk = meta.reqs_to_send[req_id]
        assert chunk.token_base == 2048
        assert chunk.num_tokens == 1024
        assert chunk.is_final is False
        assert chunk.dst_skip_bytes is None
        assert req_state.tokens_sent == 3072
        assert req_state.next_chunk_idx == 3
        mock_req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        mock_req.num_computed_tokens = 4096
        scheduler.request_finished(mock_req, block_ids=list(range(64)))
        assert req_id in scheduler.reqs_to_send
        final_chunk = scheduler.reqs_to_send[req_id]
        assert final_chunk.token_base == 3072
        assert final_chunk.num_tokens == 1024
        assert final_chunk.is_final is True
        assert final_chunk.dst_skip_bytes is None

def test_prefix_cache_boundary_chunk_dst_skip_bytes() -> None:
    """Verifies unaligned cached prefixes produce correct dst_skip_bytes."""
    scheduler = _make_raiden_scheduler(is_producer=True, block_size=16, pcp_size=4, interleave_size=256)
    scheduler._stage3_fa_token_bytes = MagicMock(return_value=64)
    req_id = 'req-prefix-boundary-test'
    mock_req = MagicMock()
    mock_req.request_id = req_id
    mock_req.num_computed_tokens = 2560
    mock_req.num_prompt_tokens = 4096
    mock_req.prompt_token_ids = list(range(4096))
    mock_req.status = RequestStatus.RUNNING
    mock_req.kv_transfer_params = {'_p_side_truncated': True}
    base_uuid = 777123
    with _patch_raiden_stage3_env(base_uuid=base_uuid):
        scheduler._update_stage3_producer_state_after_alloc(mock_req, blocks=list(range(64)))
        req_state = scheduler._stage3_send_requests[req_id]
        assert req_state.cached_prefix_tokens == 2560
        assert req_state.tokens_sent == 2048
        assert req_state.next_chunk_idx == 2
        sched_output = MagicMock(spec=SchedulerOutput)
        sched_output.num_scheduled_tokens = {req_id: 512}
        meta = scheduler.build_connector_meta(sched_output)
        assert req_id in meta.reqs_to_send
        boundary_chunk = meta.reqs_to_send[req_id]
        assert boundary_chunk.token_base == 2048
        assert boundary_chunk.num_tokens == 1024
        assert boundary_chunk.is_final is False
        assert boundary_chunk.dst_skip_bytes == 32768
        assert req_state.tokens_sent == 3072
        assert req_state.next_chunk_idx == 3
        mock_req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        mock_req.num_computed_tokens = 4096
        scheduler.request_finished(mock_req, block_ids=list(range(64)))
        assert req_id in scheduler.reqs_to_send
        final_chunk = scheduler.reqs_to_send[req_id]
        assert final_chunk.token_base == 3072
        assert final_chunk.num_tokens == 1024
        assert final_chunk.is_final is True
        assert final_chunk.dst_skip_bytes is None

def test_prefix_cache_terminal_remainder_dst_skip_bytes() -> None:
    """Verifies terminal remainder chunks carry dst_skip_bytes if boundary."""
    scheduler = _make_raiden_scheduler(is_producer=True, block_size=16, pcp_size=4, interleave_size=256)
    scheduler._stage3_fa_token_bytes = MagicMock(return_value=64)
    req_id = 'req-prefix-terminal-boundary-test'
    mock_req = MagicMock()
    mock_req.request_id = req_id
    mock_req.num_computed_tokens = 2560
    mock_req.num_prompt_tokens = 3000
    mock_req.prompt_token_ids = list(range(3000))
    mock_req.status = RequestStatus.RUNNING
    mock_req.kv_transfer_params = {'_p_side_truncated': True}
    base_uuid = 888123
    with _patch_raiden_stage3_env(base_uuid=base_uuid):
        scheduler._update_stage3_producer_state_after_alloc(mock_req, blocks=list(range(64)))
        req_state = scheduler._stage3_send_requests[req_id]
        assert req_state.tokens_sent == 2048
        mock_req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        mock_req.num_computed_tokens = 3000
        scheduler.request_finished(mock_req, block_ids=list(range(64)))
        assert req_id in scheduler.reqs_to_send
        terminal_chunk = scheduler.reqs_to_send[req_id]
        assert terminal_chunk.token_base == 2048
        assert terminal_chunk.num_tokens == 952
        assert terminal_chunk.is_final is True
        assert terminal_chunk.dst_skip_bytes == 32768

def test_recurrent_states_on_terminal_chunk_only() -> None:
    """Verifies that recurrent states are only sent on the terminal chunk."""
    worker, facade = _make_hybrid_test_worker(pcp_size=4, interleave_size=256)
    worker._stage3_state_group_count = 2
    worker._stage3_state_pool_spans = MagicMock(return_value=[SimpleNamespace(tag='gdn.conv.g0', spans=[]), SimpleNamespace(tag='gdn.ssm.g0', spans=[])])
    intermediate_req_id = 'req-hybrid-intermediate'
    intermediate_meta = TPUConnectorMetadata()
    intermediate_meta.reqs_to_send[intermediate_req_id] = SendMeta(uuid=1001, local_block_ids=[1, 2, 3, 4], expiration_time=time.perf_counter() + 60.0, num_tokens=1024, token_base=0, is_final=False, mamba_state_block_ids=[10, 11])
    with patch.object(worker, '_require_stage3_controller', return_value=(facade, '127.0.0.1:27000')):
        worker._register_stage3_request_blocks(intermediate_meta)
    assert len(facade.register_request_blocks_calls) == 1
    call1 = facade.register_request_blocks_calls[0]
    assert call1['req_id'] == intermediate_req_id
    assert call1['uuid'] == 1001
    pool_spans1 = call1['pool_spans']
    tags1 = [span.tag for span in pool_spans1]
    assert 'gdn.conv.g0' not in tags1
    assert 'gdn.ssm.g0' not in tags1
    terminal_req_id = 'req-hybrid-terminal'
    terminal_meta = TPUConnectorMetadata()
    terminal_meta.reqs_to_send[terminal_req_id] = SendMeta(uuid=1002, local_block_ids=[1, 2, 3, 4], expiration_time=time.perf_counter() + 60.0, num_tokens=1024, token_base=1024, is_final=True, mamba_state_block_ids=[10, 11])
    with patch.object(worker, '_require_stage3_controller', return_value=(facade, '127.0.0.1:27000')):
        worker._register_stage3_request_blocks(terminal_meta)
    assert len(facade.register_request_blocks_calls) == 2
    call2 = facade.register_request_blocks_calls[1]
    assert call2['req_id'] == terminal_req_id
    assert call2['uuid'] == 1002
    pool_spans2 = call2['pool_spans']
    tags2 = [span.tag for span in pool_spans2]
    assert 'gdn.conv.g0' in tags2
    assert 'gdn.ssm.g0' in tags2

def test_worker_forwards_dst_skip_bytes() -> None:
    """Verifies that TPURaidenConnectorWorker forwards dst_skip_bytes."""
    worker, facade = _make_hybrid_test_worker(pcp_size=4, interleave_size=256)
    req_id = 'req-worker-skip-bytes'
    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(uuid=2001, local_block_ids=[1, 2, 3, 4], expiration_time=time.perf_counter() + 60.0, num_tokens=1024, token_base=0, is_final=False, dst_skip_bytes=32768)
    with patch.object(worker, '_require_stage3_controller', return_value=(facade, '127.0.0.1:27000')):
        worker._register_stage3_request_blocks(meta)
    assert len(facade.register_request_blocks_calls) == 1
    call = facade.register_request_blocks_calls[0]
    assert call['dst_skip_bytes'] == [32768]
    assert worker._active_tensor_refs[req_id].dst_skip_bytes == 32768

def test_worker_rejects_negative_dst_skip_bytes() -> None:
    """Verifies TPURaidenConnectorWorker raises on negative dst_skip_bytes."""
    worker, facade = _make_hybrid_test_worker(pcp_size=4, interleave_size=256)
    req_id = 'req-negative-skip'
    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(uuid=2002, local_block_ids=[1, 2, 3, 4], expiration_time=time.perf_counter() + 60.0, num_tokens=1024, token_base=0, is_final=False, dst_skip_bytes=-100)
    with patch.object(worker, '_require_stage3_controller', return_value=(facade, '127.0.0.1:27000')), pytest.raises(ValueError, match='Stage-3 producer send metadata requires a non-negative dst_skip_bytes'):
        worker._register_stage3_request_blocks(meta)

def test_worker_duplicate_and_replay_dst_skip_bytes_validation() -> None:
    """Verifies duplicate and replay dst_skip_bytes validation."""
    worker, facade = _make_hybrid_test_worker(pcp_size=4, interleave_size=256)
    req_id = 'req-dup-test'
    meta1 = TPUConnectorMetadata()
    meta1.reqs_to_send[req_id] = SendMeta(uuid=2003, local_block_ids=[1, 2, 3, 4], expiration_time=time.perf_counter() + 60.0, num_tokens=1024, token_base=0, is_final=False, dst_skip_bytes=1024)
    with patch.object(worker, '_require_stage3_controller', return_value=(facade, '127.0.0.1:27000')):
        worker._register_stage3_request_blocks(meta1)
    assert len(facade.register_request_blocks_calls) == 1
    with patch.object(worker, '_require_stage3_controller', return_value=(facade, '127.0.0.1:27000')):
        worker._register_stage3_request_blocks(meta1)
    assert len(facade.register_request_blocks_calls) == 1
    meta_conflict = TPUConnectorMetadata()
    meta_conflict.reqs_to_send[req_id] = SendMeta(uuid=2003, local_block_ids=[1, 2, 3, 4], expiration_time=time.perf_counter() + 60.0, num_tokens=1024, token_base=0, is_final=False, dst_skip_bytes=2048)
    with patch.object(worker, '_require_stage3_controller', return_value=(facade, '127.0.0.1:27000')), pytest.raises(ValueError, match='Conflicting duplicate Stage-3 producer registration'):
        worker._register_stage3_request_blocks(meta_conflict)
    terminal_reg = worker._stage3_registered_sends.pop(req_id)
    worker._stage3_terminal_sends[req_id] = terminal_reg
    worker._active_tensor_refs.pop(req_id, None)
    with patch.object(worker, '_require_stage3_controller', return_value=(facade, '127.0.0.1:27000')), pytest.raises(ValueError, match='Conflicting replay after terminal Stage-3 producer registration'):
        worker._register_stage3_request_blocks(meta_conflict)

def test_fa_only_model_terminal_no_mamba_spans() -> None:
    """Verifies that FA-only models register no recurrent state spans."""
    worker, facade = _make_hybrid_test_worker(pcp_size=4, interleave_size=256)
    worker._stage3_state_group_count = 0
    req_id = 'req-fa-only-terminal'
    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(uuid=3001, local_block_ids=[1, 2, 3, 4], expiration_time=time.perf_counter() + 60.0, num_tokens=1024, token_base=0, is_final=True)
    with patch.object(worker, '_require_stage3_controller', return_value=(facade, '127.0.0.1:27000')):
        worker._register_stage3_request_blocks(meta)
    assert len(facade.register_request_blocks_calls) == 1
    call = facade.register_request_blocks_calls[0]
    tags = [span.tag for span in call['pool_spans']]
    assert 'gdn.conv.g0' not in tags
    assert 'gdn.ssm.g0' not in tags

@pytest.mark.parametrize('cached_prefix,token_base,token_bytes,expected_skip', [(2560, 2048, 64, 32768), (2048, 2048, 64, None), (1024, 2048, 64, None), (2560, 2048, 0, None), (2560, 2048, -10, None), (0, 0, 64, None)])
def test_compute_boundary_skip_bytes_contract(cached_prefix: int, token_base: int, token_bytes: int, expected_skip: int | None) -> None:
    """Verifies calculation and edge-case contracts of boundary skip bytes."""
    skip = _compute_boundary_skip_bytes(cached_prefix, token_base, token_bytes)
    assert skip == expected_skip

@pytest.mark.parametrize('cached_prefix,total_prompt,min_transfer,expected_progress', [(2560, 4096, 1024, (2048, 2)), (4096, 4096, 1024, (0, 0)), (5000, 4096, 1024, (0, 0)), (0, 4096, 1024, (0, 0)), (2560, 4096, 0, (0, 0))])
def test_compute_cached_prefix_progress_contract(cached_prefix: int, total_prompt: int, min_transfer: int, expected_progress: tuple[int, int]) -> None:
    """Verifies calculation and edge-case contracts of prefix progress."""
    progress = _compute_cached_prefix_progress(cached_prefix, total_prompt, min_transfer)
    assert progress == expected_progress

def test_stage3_fa_token_bytes_calculation_and_fallback(caplog: pytest.LogCaptureFixture) -> None:
    """Verifies _stage3_fa_token_bytes computes bytes or returns fallback."""
    scheduler = _make_raiden_scheduler()
    scheduler.vllm_config.model_config.num_hidden_layers = 32
    scheduler.vllm_config.model_config.get_num_layers = MagicMock(return_value=32)
    scheduler.vllm_config.model_config.get_num_kv_heads = MagicMock(return_value=8)
    scheduler.vllm_config.model_config.get_head_size = MagicMock(return_value=128)
    scheduler.vllm_config.model_config.dtype = torch.bfloat16
    token_bytes = scheduler._stage3_fa_token_bytes()
    assert token_bytes == 131072
    scheduler.vllm_config.model_config.num_hidden_layers = None
    scheduler.vllm_config.model_config.get_num_layers = MagicMock(side_effect=AttributeError('broken'))
    with caplog.at_level(logging.WARNING):
        fallback_bytes = scheduler._stage3_fa_token_bytes()
    assert fallback_bytes == 64
    assert any(('failed to compute FA token bytes' in record.message for record in caplog.records))
