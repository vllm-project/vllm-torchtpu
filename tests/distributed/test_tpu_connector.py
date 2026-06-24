# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for TPUConnector, TPUConnectorScheduler, and TPUConnectorWorker.

Structure mirrors tests/distributed/test_tpu_connector.py in the upstream
tpu-inference repo, adapted for the torch-based coordinator-mode connector
used here.  Heavy dependencies (ZMQ sockets, shared-memory pool, TPU device
queries) are mocked at construction time so that all tests run without a real
TPU.
"""
import threading
import unittest
from functools import partial
from typing import Any
from unittest.mock import MagicMock, patch

import torch
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import RequestStatus

from vllm_torchtpu.distributed.kv_transfer.tpu_connector_stats import (
    TpuKVConnectorPromMetrics, TpuKVConnectorStats)

from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (  # isort: skip
    LoadMeta, TPUConnector, TPUConnectorMetadata, TPUConnectorScheduler,
    TPUConnectorWorker, TPURaidenConnector, TPURaidenConnectorScheduler,
    TPURaidenConnectorWorker, _CoordRecvEntry, _CoordSendEntry)

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

_MOD = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"
_BASE = "vllm_torchtpu.distributed.kv_transfer.zmq_shm_base"


def _make_test_kv_cache_config() -> KVCacheConfig:
    return KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[])


def _make_vllm_config(*,
                      is_producer: bool = True,
                      block_size: int = 16,
                      dp_rank: int = 0,
                      tp_size: int = 1):
    cfg = MagicMock()
    cfg.kv_transfer_config.is_kv_producer = is_producer
    cfg.kv_transfer_config.kv_connector_extra_config = {}
    cfg.cache_config.block_size = block_size
    cfg.model_config.max_model_len = 64
    cfg.parallel_config.data_parallel_rank = dp_rank
    cfg.parallel_config.tensor_parallel_size = tp_size
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
                           dp_rank: int = 0,
                           tp_size: int = 1):
    """Construct a TPURaidenConnectorScheduler with network calls patched."""
    cfg = _make_vllm_config(is_producer=is_producer,
                            dp_rank=dp_rank,
                            tp_size=tp_size)
    with patch(f"{_MOD}.dist_utils.get_kv_ips", return_value="127.0.0.1"), \
         patch(f"{_MOD}.dist_utils.get_kv_ports", return_value=9100):
        return TPURaidenConnectorScheduler(cfg)


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
                        is_producer: bool = True) -> TPURaidenConnectorWorker:
    cfg = _make_vllm_config(is_producer=is_producer)
    with patch(f"{_MOD}.get_tensor_model_parallel_rank", return_value=tp_rank), \
         patch(f"{_MOD}.get_tensor_model_parallel_world_size", return_value=tp_size), \
         patch(f"{_MOD}.dist_utils.get_node_id", return_value=0), \
         patch(f"{_MOD}.dist_utils.get_host_ip", return_value="127.0.0.1"), \
         patch(f"{_MOD}.dist_utils.get_kv_transfer_port", return_value="9100"):
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
        sched.build_connector_meta.assert_called_once_with()

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

    @patch(f"{_MOD}.TPUConnectorHMAWorker")
    @patch(f"{_MOD}.TPUConnectorHMAScheduler")
    @patch(f"{_MOD}.TPUConnectorWorker")
    @patch(f"{_MOD}.TPUConnectorScheduler")
    def test_init_uses_hma_backend_when_flag_enabled(self, mock_sched_cls,
                                                     mock_worker_cls,
                                                     mock_hma_sched_cls,
                                                     mock_hma_worker_cls):
        cfg = _make_vllm_config()
        cfg.kv_transfer_config.kv_connector_extra_config = {
            "use_hma_connector": True,
        }

        TPUConnector(cfg, KVConnectorRole.SCHEDULER,
                     _make_test_kv_cache_config())
        TPUConnector(cfg, KVConnectorRole.WORKER, _make_test_kv_cache_config())

        mock_hma_sched_cls.assert_called_once_with(cfg)
        mock_hma_worker_cls.assert_called_once_with(cfg)
        mock_sched_cls.assert_not_called()
        mock_worker_cls.assert_not_called()

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

    def test_get_finished_count_uses_vllm_world_size(self):
        assert self.consumer.get_finished_count() == 0


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


class TestTPURaidenConnectorWorker:

    def setup_method(self):
        self.worker = _make_raiden_worker(is_producer=True)
        self.engine = _FakeRaidenEngine()
        self.worker._raiden_transfer_engine = self.engine

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
