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

from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (  # isort: skip
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
                      interleave_size: int = 1,
                      pp_size: int = 1):
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
    cfg.parallel_config.pipeline_parallel_size = pp_size
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
                           kv_ips: Any = "127.0.0.1",
                           kv_ports: Any = 9100):
    """Construct a TPURaidenConnectorScheduler with network calls patched."""
    cfg = _make_vllm_config(is_producer=is_producer,
                            block_size=block_size,
                            enable_prefix_caching=enable_prefix_caching,
                            dp_rank=dp_rank,
                            tp_size=tp_size,
                            pcp_size=pcp_size)
    with patch(f"{_MOD}.dist_utils.get_kv_ips", return_value=kv_ips), \
         patch(f"{_MOD}.dist_utils.get_kv_ports", return_value=kv_ports):
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
                        is_producer: bool = True,
                        dp_rank: int = 0,
                        dp_size: int = 1,
                        pcp_size: int = 1,
                        interleave_size: int = 1,
                        block_size: int = 16,
                        pp_size: int = 1,
                        kv_ips: Any = "127.0.0.1",
                        kv_ports: Any = 9100) -> TPURaidenConnectorWorker:
    cfg = _make_vllm_config(is_producer=is_producer,
                            block_size=block_size,
                            dp_rank=dp_rank,
                            dp_size=dp_size,
                            tp_size=tp_size,
                            pcp_size=pcp_size,
                            interleave_size=interleave_size,
                            pp_size=pp_size)
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
    def test_stage3_consumer_partial_hit_without_feature_pulls_full_payload(
            self, requested, supported):
        """Feature off: legacy full pull into every page, no release."""
        req = MagicMock()
        req.request_id = "legacy-partial-hit"
        req.num_computed_tokens = 0
        req.prompt_token_ids = [0] * 4096
        req.kv_transfer_params = {
            "req_id": "legacy-partial-hit-src",
            "uuid": 994,
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

        assert matched == 2047
        assert is_async
        load = consumer.reqs_to_load[req.request_id]
        assert not load.release_only
        assert load.skip_tokens == 0
        assert load.local_block_ids == [50, 51, 52, 53]
        assert load.num_tokens == 4095
        assert load.source_req_id == "legacy-partial-hit-src"

    @staticmethod
    def _full_hit_request():
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
        return req

    def test_stage3_consumer_full_hit_enqueues_release_only(self):
        req = self._full_hit_request()
        blocks = MagicMock()

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), \
             patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_PREFIX_AWARE_LOAD",
                   True,
                   create=True), \
             patch(f"{_MOD}._prefix_aware_load_supported",
                   return_value=True):
            consumer = _make_raiden_scheduler(is_producer=False,
                                              block_size=1024,
                                              enable_prefix_caching=True)
            matched, is_async = consumer.get_num_new_matched_tokens(req, 2048)
            consumer.update_state_after_alloc(req, blocks, matched)

        assert matched == 0
        assert not is_async
        load = consumer.reqs_to_load["full-hit"]
        assert load.release_only
        assert load.local_block_ids == []
        assert load.uuid == 995
        assert load.source_req_id == "full-hit-src"

    @pytest.mark.parametrize(("requested", "supported"),
                             ((False, True), (True, False)))
    def test_stage3_consumer_full_hit_without_feature_emits_no_release(
            self, requested, supported):
        """Feature off: a full hit leaves the registration to the TTL."""
        req = self._full_hit_request()
        blocks = MagicMock()

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), \
             patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_PREFIX_AWARE_LOAD",
                   requested,
                   create=True), \
             patch(f"{_MOD}._prefix_aware_load_supported",
                   return_value=supported):
            consumer = _make_raiden_scheduler(is_producer=False,
                                              block_size=1024,
                                              enable_prefix_caching=True)
            matched, is_async = consumer.get_num_new_matched_tokens(req, 2048)
            consumer.update_state_after_alloc(req, blocks, matched)

        assert matched == 0
        assert not is_async
        assert consumer.reqs_to_load == {}
        blocks.get_block_ids.assert_not_called()

    def test_stage3_consumer_post_load_resume_does_not_release(self):
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
                   create=True), \
             patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_PREFIX_AWARE_LOAD",
                   True,
                   create=True), \
             patch(f"{_MOD}._prefix_aware_load_supported",
                   return_value=True):
            consumer = _make_raiden_scheduler(is_producer=False,
                                              block_size=1024,
                                              enable_prefix_caching=True)
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

    def register_pools(self, pools, **kwargs):
        self.registered_pools = list(pools)
        # Bounded host staging kwargs (RESHARD_BOUNDED_STAGING_DESIGN.md).
        self.register_pools_kwargs = dict(kwargs)
        storage_indices = {
            pool["storage_index"]
            for pool in self.registered_pools
        }
        return {
            "admitted": True,
            "pools": len(self.registered_pools),
            "storages": len(storage_indices),
            "host_staging": {
                "mode": "bounded" if kwargs.get("staging_leases") else "full",
                "leases": kwargs.get("staging_leases", 0),
            },
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
        # (req_id, uuid) -> GetRequestBlockStatusResponse.Status; keys not
        # listed read as registered (2).
        self.request_block_statuses = {}
        self.get_request_block_status_calls = []

    def get_request_block_status(self, keys):
        self.get_request_block_status_calls.append(list(keys))
        return [self.request_block_statuses.get(tuple(key), 2) for key in keys]

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


def _flush_stage3_submits(worker):
    """Waits until every queued Stage-3 submission has posted its outcome.

    Coordination RPCs run on background submit workers; joining the task
    queue fences the outcome mailbox so the next poll drains it
    deterministically.
    """
    worker._stage3_submit_queue.join()


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
            assert construct_call.kwargs == {"num_slots": 1}
            # Bounded host staging: the connector passes the lease count and
            # one hint per pool: the distinct blocks one transfer touches on
            # the pool's storage. Every typed cache is its own storage here,
            # so FA pools report pages per max-length request and GDN state
            # pools report their single block.
            staging_kwargs = engine.register_pools_kwargs
            assert staging_kwargs["staging_leases"] == 8
            hints = staging_kwargs["staging_blocks_per_pool"]
            assert len(hints) == len(engine.registered_pools)
            for pool_dict, hint in zip(engine.registered_pools, hints):
                tag = str(pool_dict["tag"])
                if tag.startswith("fa"):
                    assert hint == worker._max_request_blocks()
                else:
                    assert hint == 1, (tag, hint)
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
        assert construct_call.kwargs == {"num_slots": 1}
        assert len(engine.registered_pools) == 105
        assert len({pool["storage_index"]
                    for pool in engine.registered_pools}) == 15
        # Every raw storage hosts one FA layer plus one layer of each of the
        # three GDN groups (conv + ssm). Raiden sizes a storage's staging
        # arena from max(hint) over its pools, so each pool reports the
        # storage-wide union: FA pages plus one block per GDN group.
        staging_kwargs = engine.register_pools_kwargs
        assert staging_kwargs["staging_leases"] == 8
        hints = staging_kwargs["staging_blocks_per_pool"]
        assert len(hints) == 105
        assert set(hints) == {worker._max_request_blocks() + 3}
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

    def test_raiden_staging_hints_cover_every_block_table_on_a_storage(self):
        worker = _make_raiden_worker(tp_rank=0, tp_size=1, is_producer=True)
        fa_pages = worker._max_request_blocks()
        groups = (
            SimpleNamespace(layer_names=("fa.0", "fa.1")),
            SimpleNamespace(layer_names=("gdn.0", )),
            SimpleNamespace(layer_names=("gdn.1", )),
            SimpleNamespace(layer_names=("gdn.2", )),
        )

        def pool(tag, layer_name, storage_index):
            return SimpleNamespace(tag=tag,
                                   layer_name=layer_name,
                                   storage_index=storage_index)

        manifest = SimpleNamespace(pools=[
            # Storage 0: FA pages plus one block per GDN group; a group's
            # conv and ssm pools share that block.
            pool("fa", "fa.0", 0),
            pool("gdn.conv.g0", "gdn.0", 0),
            pool("gdn.ssm.g0", "gdn.0", 0),
            pool("gdn.conv.g1", "gdn.1", 0),
            pool("gdn.ssm.g1", "gdn.1", 0),
            pool("gdn.conv.g2", "gdn.2", 0),
            pool("gdn.ssm.g2", "gdn.2", 0),
            # Storage 1: only full attention.
            pool("fa", "fa.1", 1),
            # Storage 2: an unknown pool kind keeps the whole storage on the
            # full host mirror.
            pool("fa", "fa.1", 2),
            pool("opaque", "fa.1", 2),
        ])

        hints = worker._raiden_staging_blocks_per_pool(manifest, groups)

        assert hints == [fa_pages + 3] * 7 + [fa_pages] + [0, 0]

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
        stage = stage3_fa_raiden_id_fields(job_name="decode",
                                           engine_id="engine1",
                                           dp_rank=0,
                                           transfer_rank=3,
                                           is_producer=False,
                                           per_rank_unit=True)
        assert stage["job_replica_id"] == "engine1-rank3"

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
            _flush_stage3_submits(worker)

        assert facade.start_transfer.call_count == 2
        # Submit workers run concurrently, so match the calls by uuid rather
        # than by execution order.
        by_uuid = {
            call.kwargs["uuid"]: call.kwargs
            for call in facade.start_transfer.call_args_list
        }
        clipped = by_uuid[997]
        assert clipped["dst_skip_bytes"] == [2048 * 64]
        assert clipped["dst_device_block_ids"] == [52, 53]
        assert clipped["dst_block_counts"] == [2]
        assert clipped["transfer_pool_tags"] == ["fa"]
        assert clipped["num_tokens"] == 4095
        assert worker._stage3_submitted_loads["dst-suffix"] == 997
        assert worker._load_block_ids["dst-suffix"] == [52, 53]
        # Skip-free submissions must stay byte-identical on the wire: the
        # kwarg is omitted entirely rather than passed as zeros.
        unclipped = by_uuid[998]
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
            _flush_stage3_submits(worker)

        facade.cancel_request_blocks_if_unclaimed.assert_called_once_with(
            req_id="src-hit", uuid=999)
        facade.start_transfer.assert_not_called()
        assert worker._stage3_submitted_loads == {}
        assert "dst-hit" not in worker._load_block_ids

    @staticmethod
    def _make_stage3_load_meta(destination_req_id, source_req_id, uuid,
                               local_block_ids):
        meta = TPUConnectorMetadata()
        meta.reqs_to_load[destination_req_id] = _Stage3LoadMeta(
            uuid=uuid,
            source_req_id=source_req_id,
            local_block_ids=local_block_ids,
            num_tokens=2048,
            src_controller_address="prefill-controller.test:27000",
            src_job_name="prefill",
            src_engine_id="producer-engine",
            src_data_replica_idx=0,
            src_parallelism=8,
        )
        return meta

    def test_stage3_async_submit_defers_native_terminal_until_outcome(self):
        worker = _make_raiden_worker(is_producer=False, block_size=1024)
        worker._raiden_work_unit = MagicMock()
        engine = _FakeRaidenEngine()
        # The armed receiver completes while the coordination RPC is still
        # blocked on the source controller.
        engine.poll_results = [([], ["src-slow"], []), ([], [], [])]
        worker._raiden_transfer_engine = engine
        rpc_gate = threading.Event()
        facade = MagicMock()
        facade.start_transfer.side_effect = (
            lambda **kwargs: rpc_gate.wait(10.0))
        worker._stage3_source_facades["prefill-controller.test:27000"] = facade
        meta = self._make_stage3_load_meta("dst-slow", "src-slow", 1001,
                                           [7, 8])

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), \
             patch.object(worker,
                          "_require_stage3_controller",
                          return_value=(MagicMock(), "10.0.0.2:28000")), \
             patch.object(worker,
                          "_stage3_source_work_units",
                          return_value=[MagicMock()]), \
             patch.object(worker,
                          "_raiden_hbm_memory_type",
                          MagicMock(return_value=3)):
            worker._submit_stage3_loads(meta, MagicMock())
            # The step thread is not blocked on the in-flight RPC.
            assert "dst-slow" in worker._stage3_inflight_submits
            # The native terminal is parked, not dropped and not reported.
            assert worker.get_finished() == (set(), set())
            assert worker._stage3_deferred_native_failures == {
                "dst-slow": False
            }
            rpc_gate.set()
            _flush_stage3_submits(worker)
            assert worker.get_finished() == (set(), {"dst-slow"})
            assert worker._stage3_deferred_native_failures == {}
            assert worker._stage3_inflight_submits == {}
        assert worker.get_block_ids_with_load_errors() == set()

    def test_stage3_async_submit_abandons_overdue_rpc(self):
        worker = _make_raiden_worker(is_producer=False, block_size=1024)
        worker._raiden_work_unit = MagicMock()
        engine = _FakeRaidenEngine()
        engine.poll_results = []
        worker._raiden_transfer_engine = engine
        rpc_gate = threading.Event()
        facade = MagicMock()
        facade.start_transfer.side_effect = (
            lambda **kwargs: rpc_gate.wait(10.0))
        worker._stage3_source_facades["prefill-controller.test:27000"] = facade
        meta = self._make_stage3_load_meta("dst-hang", "src-hang", 1002,
                                           [9, 10])

        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), \
             patch(f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=0.0), \
             patch.object(worker,
                          "_require_stage3_controller",
                          return_value=(MagicMock(), "10.0.0.2:28000")), \
             patch.object(worker,
                          "_stage3_source_work_units",
                          return_value=[MagicMock()]), \
             patch.object(worker,
                          "_raiden_hbm_memory_type",
                          MagicMock(return_value=3)):
            worker._submit_stage3_loads(meta, MagicMock())
            # A zero deadline abandons the still-blocked RPC on the next
            # poll, and the zero uncertainty window expires in the same
            # pass: the request surfaces as a failed load with its exact
            # destination blocks staged for recompute.
            assert worker.get_finished() == (set(), {"dst-hang"})
            assert worker.get_block_ids_with_load_errors() == {9, 10}
            assert "dst-hang" in worker._stage3_abandoned_submits
            rpc_gate.set()
            _flush_stage3_submits(worker)
            # The late outcome is discarded; nothing is re-reported and the
            # abandonment marker is consumed.
            assert worker.get_finished() == (set(), set())
            assert "dst-hang" not in worker._stage3_abandoned_submits
        assert facade.start_transfer.call_count == 1

    def _park_missing_registration_load(self, worker, facade_side_effect,
                                        registration_wait_s):
        """Submits one load whose first coordination attempt reports a
        missing producer registration and returns the facade."""
        worker._raiden_work_unit = MagicMock()
        engine = _FakeRaidenEngine()
        engine.poll_results = []
        worker._raiden_transfer_engine = engine
        facade = MagicMock()
        facade.start_transfer.side_effect = facade_side_effect
        worker._stage3_source_facades["prefill-controller.test:27000"] = facade
        meta = self._make_stage3_load_meta("dst-late", "src-late", 1003,
                                           [11, 12])
        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), \
             patch(f"{_MOD}.dist_utils.get_stage3_deferred_submit_enabled",
                   return_value=True), \
             patch(f"{_MOD}.dist_utils.get_stage3_registration_wait_s",
                   return_value=registration_wait_s), \
             patch.object(worker,
                          "_require_stage3_controller",
                          return_value=(MagicMock(), "10.0.0.2:28000")), \
             patch.object(worker,
                          "_stage3_source_work_units",
                          return_value=[MagicMock()]), \
             patch.object(worker,
                          "_raiden_hbm_memory_type",
                          MagicMock(return_value=3)):
            worker._submit_stage3_loads(meta, MagicMock())
            _flush_stage3_submits(worker)
        return engine, facade

    def test_stage3_async_submit_parks_missing_registration_then_retries(self):
        worker = _make_raiden_worker(is_producer=False, block_size=1024)
        engine, facade = self._park_missing_registration_load(
            worker, [
                RuntimeError("Missing producer block registration for rank 3"),
                True,
            ],
            registration_wait_s=30.0)
        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True):
            # The missing-registration outcome parks the request instead of
            # failing it; nothing is reported and no block is invalidated.
            assert worker.get_finished() == (set(), set())
            assert "dst-late" in worker._stage3_pending_submits
            assert "dst-late" not in worker._stage3_inflight_submits
            assert worker.get_block_ids_with_load_errors() == set()
            # Re-attempts are paced by the interval; once it has elapsed the
            # next step re-dispatches the prepared call to a submit worker.
            with patch(f"{_MOD}._STAGE3_REGISTRATION_REATTEMPT_MIN_INTERVAL_S",
                       0.0):
                assert worker.get_finished() == (set(), set())
            assert "dst-late" not in worker._stage3_pending_submits
            assert "dst-late" in worker._stage3_inflight_submits
            _flush_stage3_submits(worker)
            engine.poll_results = [([], ["src-late"], [])]
            assert worker.get_finished() == (set(), {"dst-late"})
            assert worker._stage3_inflight_submits == {}
        assert facade.start_transfer.call_count == 2
        assert worker.get_block_ids_with_load_errors() == set()

    def test_stage3_async_submit_parked_load_aborted_by_scheduler(self):
        worker = _make_raiden_worker(is_producer=False, block_size=1024)
        _, facade = self._park_missing_registration_load(
            worker,
            [RuntimeError("Missing producer block registration for rank 3")],
            registration_wait_s=30.0)
        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True):
            assert worker.get_finished() == (set(), set())
            assert "dst-late" in worker._stage3_pending_submits
            # The scheduler finishes (aborts) the request while it is parked:
            # the load resolves as a pre-arm failure with its destination
            # blocks staged, so vLLM's delayed block free completes.
            assert worker.get_finished({"dst-late"}) == (set(), {"dst-late"})
            assert worker.get_block_ids_with_load_errors() == {11, 12}
            assert worker._stage3_pending_submits == {}
        assert facade.start_transfer.call_count == 1

    def test_stage3_async_submit_registration_wait_deadline_fails(self):
        worker = _make_raiden_worker(is_producer=False, block_size=1024)
        _, facade = self._park_missing_registration_load(
            worker,
            [RuntimeError("Missing producer block registration for rank 3")],
            registration_wait_s=0.0)
        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True):
            # The wait budget is already exhausted when the outcome lands, so
            # the same missing-registration error is the pre-arm failure
            # terminal.
            assert worker.get_finished() == (set(), {"dst-late"})
            assert worker.get_block_ids_with_load_errors() == {11, 12}
            assert worker._stage3_pending_submits == {}
        assert facade.start_transfer.call_count == 1

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

    @staticmethod
    def _registered_stage3_producer(req_id="cancelled-by-consumer", uuid=77):
        """A producer rank with one registered send and no native terminal:
        the state a consumer's release-only cancel leaves behind."""
        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=1,
                                     is_producer=True,
                                     dp_size=1,
                                     pcp_size=8,
                                     block_size=4096)
        engine = _FakeRaidenEngine()
        engine.poll_results = []
        facade = _FakeRaidenControllerFacade()
        worker._raiden_transfer_engine = engine
        worker._raiden_controller_facade = facade
        worker._raiden_controller_address = "prefill-controller.test:27000"
        worker._raiden_work_unit = SimpleNamespace(job_name="prefill",
                                                   job_replica_id="rank0",
                                                   data_name="kv.fa",
                                                   data_replica_idx=0)
        worker._local_raiden_transfer_rank = MagicMock(return_value=0)
        meta = TPUConnectorMetadata()
        meta.reqs_to_send[req_id] = MagicMock(uuid=uuid,
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
        assert req_id in worker._stage3_registered_sends
        return worker, facade

    def test_v3_stage3_status_probe_reports_consumer_cancel_as_terminal(self):
        """The PR #487 producer leak: a release-only cancel retires only the
        registry row. The probe turns it into a send terminal within one
        interval instead of p2p_wait_pull_timeout."""
        worker, facade = self._registered_stage3_producer()
        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), patch(
                       f"{_MOD}.dist_utils.get_stage3_status_probe_s",
                       return_value=1e-9):
            # First poll only arms the probe (grace period); the row is
            # still registered, so nothing is terminal.
            assert worker.get_finished() == (set(), set())
            assert worker.get_finished() == (set(), set())
            assert facade.get_request_block_status_calls == [[
                ("cancelled-by-consumer", 77)
            ]]
            # The consumer's cancel lands at the store.
            facade.request_block_statuses[("cancelled-by-consumer", 77)] = 4
            done_sending, done_recving = worker.get_finished()

        assert done_sending == {"cancelled-by-consumer"}
        assert done_recving == set()
        # Cancelled, not completed: no completion vote, no force release.
        assert facade.complete_request_blocks_calls == []
        assert facade.cancel_request_blocks_calls == []
        assert worker._stage3_registered_sends == {}
        assert "cancelled-by-consumer" in worker._stage3_terminal_sends
        # Reported exactly once.
        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True):
            assert worker.get_finished() == (set(), set())

    def test_v3_stage3_status_probe_is_rate_limited_and_optional(self):
        worker, facade = self._registered_stage3_producer()
        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), patch(
                       f"{_MOD}.dist_utils.get_stage3_status_probe_s",
                       return_value=3600.0):
            for _ in range(3):
                assert worker.get_finished() == (set(), set())
        # Within the interval the registry is never queried.
        assert facade.get_request_block_status_calls == []

        # Older clients without the probe degrade to the TTL backstop.
        facade.get_request_block_status = None
        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True), patch(
                       f"{_MOD}.dist_utils.get_stage3_status_probe_s",
                       return_value=1e-9):
            assert worker.get_finished() == (set(), set())
            assert worker.get_finished() == (set(), set())
        assert "cancelled-by-consumer" in worker._stage3_registered_sends

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
            _flush_stage3_submits(worker)
            assert worker.get_finished() == (set(), set())
            assert worker.get_finished() == (set(), {"proxy-id-decode5678"})
            # _reported_recving preserves the existing at-most-once contract.
            assert worker.get_finished() == (set(), set())
            assert worker.get_finished({"proxy-id-decode5678"}) == (set(),
                                                                    set())

        assert worker._stage3_source_req_ids == {}
        assert worker._stage3_destination_req_ids == {}
        assert worker._stage3_submitted_loads == {}
        assert worker._stage3_submitted_load_tokens == {}
        assert worker._stage3_load_start_times == {}

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
            _flush_stage3_submits(worker)
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
            _flush_stage3_submits(worker)
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
        worker._stage3_load_start_times["aborted-load-decode"] = 100.0
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
            assert worker._stage3_load_start_times == {
                "aborted-load-decode": 100.0
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
        assert worker._stage3_load_start_times == {}
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
                max_blocks=4) == ((1024**3 // 4) // 28672)

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


class TestPipelineParallelProducer:
    """Pipeline-parallel prefill (PP x TP1) as a Stage-3 source: each stage
    transfers every token of its own layers and pools pair up by layer."""

    @staticmethod
    def _stage_manifest():
        from vllm_torchtpu.distributed.kv_transfer.raiden.pool_manifest import (
            PoolEntry, PoolManifest, RegionSpec)

        def pool(tag, layer):
            return PoolEntry(tag=tag,
                             layer_name=f"model.layers.{layer}.x",
                             storage_index=0,
                             base_offset_bytes=0,
                             block_stride_bytes=1024,
                             num_blocks=4,
                             regions=(RegionSpec("live", 0, 1024, 1024, 1), ),
                             dtype_tag="uint8")

        # Stage holding layers 0..4: one FA layer (3), four GDN layers.
        return PoolManifest(binding="private_typed",
                            storages=[object()],
                            pools=[
                                pool("gdn.conv.g0.l0", 0),
                                pool("gdn.ssm.g0.l0", 0),
                                pool("gdn.conv.g0.l1", 1),
                                pool("gdn.ssm.g0.l1", 1),
                                pool("fa.l3", 3),
                                pool("gdn.conv.g0.l4", 4),
                                pool("gdn.ssm.g0.l4", 4),
                            ])

    def test_topology_names_the_stage_count(self):
        worker = _make_raiden_worker(tp_rank=0, tp_size=1, pp_size=8)
        assert worker._raiden_qwen35_admission_topology() == "pp8_prefill"
        consumer = _make_raiden_worker(tp_rank=0,
                                       tp_size=1,
                                       is_producer=False,
                                       dp_size=8)
        assert consumer._raiden_qwen35_admission_topology() == "dp8_decode"

    def test_transfer_rank_is_the_stage_and_spans_are_contiguous(self):
        worker = _make_raiden_worker(tp_rank=0, tp_size=1, pp_size=4)
        group = SimpleNamespace(rank_in_group=2)
        with patch("vllm.distributed.parallel_state.get_pp_group",
                   return_value=group):
            assert worker._local_raiden_transfer_rank() == 2
        # A stage owns every token: no PCP interleave applies.
        assert worker._raiden_interleave_tokens(4096, 4) == 4096

    def test_parallelism_must_be_the_stage_count_with_per_layer_tags(self):
        worker = _make_raiden_worker(tp_rank=0, tp_size=1, pp_size=4)
        with patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_POOL_TAGS_PER_LAYER",
                   True,
                   create=True):
            worker._validate_stage3_transfer_parallelism(4)
            with pytest.raises(ValueError, match="PP stage count"):
                worker._validate_stage3_transfer_parallelism(8)
        with patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_POOL_TAGS_PER_LAYER",
                   False,
                   create=True), pytest.raises(
                       ValueError, match="TPU_RAIDEN_POOL_TAGS_PER_LAYER"):
            worker._validate_stage3_transfer_parallelism(4)

    def test_registration_tags_follow_the_stage_manifest(self):
        worker = _make_raiden_worker(tp_rank=0, tp_size=1, pp_size=8)
        worker._raiden_manifest = self._stage_manifest()
        worker._stage3_state_group_count = 1
        with patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_POOL_TAGS_PER_LAYER",
                   True,
                   create=True):
            assert worker._stage3_fa_registration_tags() == ["fa.l3"]
            assert worker._stage3_state_registration_tags() == [
                ("gdn.conv.g0.l0", 0),
                ("gdn.conv.g0.l1", 0),
                ("gdn.conv.g0.l4", 0),
                ("gdn.ssm.g0.l0", 0),
                ("gdn.ssm.g0.l1", 0),
                ("gdn.ssm.g0.l4", 0),
            ]
        with patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_POOL_TAGS_PER_LAYER",
                   False,
                   create=True):
            assert worker._stage3_fa_registration_tags() == ["fa"]
            assert worker._stage3_state_registration_tags() == [
                ("gdn.conv.g0", 0),
                ("gdn.ssm.g0", 0),
            ]

    def test_scheduler_expects_one_report_per_stage(self):
        cfg = _make_vllm_config(is_producer=True, pp_size=8)
        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True):
            scheduler = TPURaidenConnectorScheduler(cfg)
            assert scheduler.get_finished_count() == 8


class TestPipelineParallelConsumer:
    """Pipeline-parallel decode (PP x TP1) as a Stage-3 destination."""

    def test_topology_and_transfer_rank(self):
        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=1,
                                     is_producer=False,
                                     pp_size=8)
        assert worker._raiden_qwen35_admission_topology() == "pp8_decode"
        group = SimpleNamespace(rank_in_group=5)
        with patch("vllm.distributed.parallel_state.get_pp_group",
                   return_value=group):
            assert worker._local_raiden_transfer_rank() == 5

    def test_stage_names_every_prefill_stage_as_source(self):
        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=1,
                                     is_producer=False,
                                     pp_size=8)
        meta = SimpleNamespace(src_parallelism=8,
                               src_job_name="prefill",
                               src_engine_id="p",
                               src_data_replica_idx=0)
        group = SimpleNamespace(rank_in_group=3)
        with patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM",
                   8,
                   create=True), patch(
                       "vllm.distributed.parallel_state.get_pp_group",
                       return_value=group):
            units = worker._stage3_source_work_units(meta)
        assert [unit.job_replica_id
                for unit in units] == [f"p-rank{rank}" for rank in range(8)]

    def test_each_stage_is_its_own_endpoint_and_unit(self):
        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=1,
                                     is_producer=False,
                                     pp_size=8)
        group = SimpleNamespace(rank_in_group=5)
        with patch.object(worker, "_raiden_stage3_enabled",
                          return_value=True), patch(
                              "vllm.distributed.parallel_state.get_pp_group",
                              return_value=group):
            assert worker._raiden_endpoint_identity() == (5, 5)
            fields = worker._raiden_work_unit_fields(5)
        assert fields["job_replica_id"].endswith("-rank5")
        assert fields["data_replica_idx"] == worker.dp_rank

    def test_non_pipelined_consumer_is_one_endpoint_per_engine(self):
        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=1,
                                     is_producer=False,
                                     pp_size=1)
        with patch.object(worker, "_raiden_stage3_enabled", return_value=True):
            assert worker._raiden_endpoint_identity() == (0, worker.dp_rank)
            fields = worker._raiden_work_unit_fields(0)
        assert "-rank" not in fields["job_replica_id"]

    def test_stage_count_mismatch_is_rejected(self):
        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=1,
                                     is_producer=False,
                                     pp_size=4)
        meta = SimpleNamespace(src_parallelism=8,
                               src_job_name="prefill",
                               src_engine_id="p",
                               src_data_replica_idx=0)
        with patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM",
                   8,
                   create=True), pytest.raises(ValueError,
                                               match="same stage count"):
            worker._stage3_source_work_units(meta)

    def test_scheduler_expects_one_report_per_decode_stage(self):
        cfg = _make_vllm_config(is_producer=False, pp_size=8)
        with patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
                   "raiden",
                   create=True):
            scheduler = TPURaidenConnectorScheduler(cfg)
            assert scheduler.get_finished_count() == 8
