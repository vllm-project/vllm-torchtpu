# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the HMA (hybrid memory allocator) KV connector:
TPUConnectorHMA, TPUConnectorHMAScheduler, and TPUConnectorHMAWorker.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.request import RequestStatus

from vllm_torchtpu.distributed.kv_transfer.host_kv_shm_hma import PoolSpecHMA

from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (  # isort: skip
    TPUConnectorHMA, TPUConnectorHMAScheduler, TPUConnectorHMAWorker)

_BASE_CONN = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"
# The HMA classes now live in tpu_connector.py alongside the base connector.
_HMA_MOD = _BASE_CONN
_BASE = "vllm_torchtpu.distributed.kv_transfer.zmq_shm_base"


def _make_vllm_config(*, is_producer: bool = True, block_size: int = 16):
    cfg = MagicMock()
    cfg.kv_transfer_config.is_kv_producer = is_producer
    cfg.cache_config.block_size = block_size
    cfg.parallel_config.data_parallel_rank = 0
    cfg.parallel_config.tensor_parallel_size = 1
    return cfg


def _make_scheduler(*, is_producer: bool = False) -> TPUConnectorHMAScheduler:
    cfg = _make_vllm_config(is_producer=is_producer)
    with patch(f"{_BASE_CONN}.dist_utils.get_kv_ips", return_value="127.0.0.1"), \
         patch(f"{_BASE_CONN}.dist_utils.get_kv_ports", return_value=9100):
        return TPUConnectorHMAScheduler(cfg)


def _make_worker(*,
                 tp_rank: int = 0,
                 tp_size: int = 1,
                 is_producer: bool = True) -> TPUConnectorHMAWorker:
    """Construct a TPUConnectorHMAWorker with all I/O and device calls mocked.
    """
    cfg = _make_vllm_config(is_producer=is_producer)
    with patch(f"{_BASE}.get_tensor_model_parallel_rank", return_value=tp_rank), \
         patch(f"{_BASE}.get_tensor_model_parallel_world_size", return_value=tp_size), \
         patch(f"{_BASE}.dist_utils.get_node_id", return_value=0), \
         patch(f"{_BASE}.dist_utils.get_host_ip", return_value="127.0.0.1"), \
         patch(f"{_BASE}.dist_utils.get_kv_transfer_port", return_value="9100"), \
         patch(f"{_BASE}.dist_utils.get_side_channel_port", return_value="9600"), \
         patch(f"{_BASE}.dist_utils.get_transfer_channel_number", return_value=0), \
         patch(f"{_BASE}.dist_utils.get_kv_latency_log_interval", return_value=0.0), \
         patch(f"{_BASE}.zmq.Context"):
        return TPUConnectorHMAWorker(cfg)


def _attn_group(layer_names):
    g = MagicMock()
    g.kv_cache_spec = MagicMock()  # not a MambaSpec
    g.layer_names = layer_names
    return g


def _mamba_group(layer_names):
    g = MagicMock()
    g.kv_cache_spec = MagicMock(spec=MambaSpec)
    g.layer_names = layer_names
    return g


def _build_hybrid_runner():
    """One attention layer + one mamba layer (conv + ssm tuple). """
    attn_cache = torch.zeros(8, 2, 4)
    mamba_conv = torch.zeros(8, 3, 4)
    mamba_ssm = torch.zeros(8, 5, 4)
    mamba_cache = (mamba_conv, mamba_ssm)
    kv_caches = [attn_cache, mamba_cache]

    groups = [_attn_group(["attn0"]), _mamba_group(["mamba0"])]

    runner = MagicMock()
    runner.kv_caches = kv_caches
    runner.kv_cache_config.kv_cache_groups = groups
    runner.device = torch.device("cpu")

    sfc = {
        "attn0": SimpleNamespace(kv_cache=attn_cache),
        "mamba0": SimpleNamespace(kv_cache=mamba_cache),
    }
    return runner, {
        "attn": attn_cache,
        "conv": mamba_conv,
        "ssm": mamba_ssm,
        "groups": groups,
        "sfc": sfc,
    }


def _attach_runner(worker, runner, sfc):
    worker.runner = runner
    worker.device = torch.device("cpu")
    worker.vllm_config.compilation_config.static_forward_context = sfc
    worker.named_kv_caches = {name: ctx.kv_cache for name, ctx in sfc.items()}


# ---------------------------------------------------------------------------
# TestTPUConnectorHMA
# ---------------------------------------------------------------------------


class TestTPUConnectorHMA:

    @patch(f"{_HMA_MOD}.TPUConnectorHMAWorker")
    @patch(f"{_HMA_MOD}.TPUConnectorHMAScheduler")
    def test_init_scheduler_role_builds_hma_scheduler(self, mock_sched_cls,
                                                      mock_worker_cls):
        cfg = _make_vllm_config()
        connector = TPUConnectorHMA(cfg, KVConnectorRole.SCHEDULER, None)
        mock_sched_cls.assert_called_once_with(cfg)
        mock_worker_cls.assert_not_called()
        assert connector.connector_scheduler is not None
        assert connector.connector_worker is None

    @patch(f"{_HMA_MOD}.TPUConnectorHMAWorker")
    @patch(f"{_HMA_MOD}.TPUConnectorHMAScheduler")
    def test_init_worker_role_builds_hma_worker(self, mock_sched_cls,
                                                mock_worker_cls):
        cfg = _make_vllm_config()
        connector = TPUConnectorHMA(cfg, KVConnectorRole.WORKER, None)
        mock_worker_cls.assert_called_once_with(cfg)
        mock_sched_cls.assert_not_called()
        assert connector.connector_scheduler is None
        assert connector.connector_worker is not None

    @patch(f"{_HMA_MOD}.TPUConnectorHMAWorker")
    @patch(f"{_HMA_MOD}.TPUConnectorHMAScheduler")
    def test_request_finished_all_groups_delegates(self, mock_sched_cls,
                                                   mock_worker_cls):
        cfg = _make_vllm_config()
        connector = TPUConnectorHMA(cfg, KVConnectorRole.SCHEDULER, None)
        sched = mock_sched_cls.return_value
        req = MagicMock()
        block_ids = ([1, 2], [3])

        connector.request_finished_all_groups(req, block_ids)
        sched.request_finished_all_groups.assert_called_once_with(
            req, block_ids)


# ---------------------------------------------------------------------------
# TestHMAScheduler — HMA-specific scheduler arithmetic & state
# ---------------------------------------------------------------------------


class TestHMAScheduler:

    def setup_method(self):
        self.consumer = _make_scheduler(is_producer=False)
        self.producer = _make_scheduler(is_producer=True)

    # ---- get_num_new_matched_tokens ---------------------------------------

    def test_producer_truncates_then_returns_zero(self):
        req = SimpleNamespace(num_prompt_tokens=5,
                              prompt_token_ids=[1, 2, 3, 4, 5],
                              _all_token_ids=[1, 2, 3, 4, 5],
                              max_tokens=16,
                              kv_transfer_params={"uuid": 1})
        n, is_async = self.producer.get_num_new_matched_tokens(req, 0)
        assert (n, is_async) == (0, False)
        # The producer path truncates as a side effect.
        assert req.prompt_token_ids == [1, 2, 3, 4]
        assert req.kv_transfer_params["_p_side_truncated"] is True

    def test_consumer_pulls_all_but_last_token_without_rounding(self):
        """Key divergence from the uniform connector: no block-alignment
        round_down. prompt=35, computed=16 -> (35-1)-16 = 18 (NOT 16)."""
        req = MagicMock()
        req.prompt_token_ids = [0] * 35
        req.kv_transfer_params = {"uuid": 1}
        n, is_async = self.consumer.get_num_new_matched_tokens(req, 16)
        assert n == 18
        assert is_async

    # ---- update_state_after_alloc -----------------------------------------

    def test_update_consumer_stores_per_group_block_ids(self):
        req = MagicMock()
        req.request_id = "req-1"
        req.kv_transfer_params = {
            "uuid": 42,
            "remote_block_ids": [[10, 11], [12]],
            "remote_host": "2.2.2.2",
            "remote_port": 9200,
        }
        blocks = MagicMock()
        blocks.get_block_ids.return_value = [[1, 2], [3]]

        self.consumer.update_state_after_alloc(req, blocks, 32)

        meta = self.consumer.reqs_to_load["req-1"]
        assert meta.uuid == 42
        assert meta.local_block_ids == [[1, 2], [3]]
        assert meta.remote_block_ids == [[10, 11], [12]]

    # ---- request_finished_all_groups --------------------------------------

    def test_consumer_request_finished_all_groups_is_noop(self):
        delay, params = self.consumer.request_finished_all_groups(
            MagicMock(), ([1], [2]))
        assert not delay
        assert params is None

    @patch(f"{_HMA_MOD}.get_uuid", return_value=777)
    @patch(f"{_HMA_MOD}.dist_utils.get_p2p_wait_pull_timeout", return_value=30)
    def test_producer_keeps_all_groups_without_trimming(self, _timeout, _uuid):
        """Unlike the uniform connector, no trailing partial block is dropped:
        every group's block ids are shipped verbatim (the last partial block
        is recomputed on D from the loaded mamba state)."""
        req = MagicMock()
        req.request_id = "p-hma"
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        req.prompt_token_ids = [0] * 40

        delay, params = self.producer.request_finished_all_groups(
            req, ([5, 6, 7], [8]))

        assert delay
        send = self.producer.reqs_to_send["p-hma"]
        assert send.uuid == 777
        assert send.local_block_ids == [[5, 6, 7], [8]]
        assert params["uuid"] == 777
        assert params["remote_block_ids"] == [[5, 6, 7], [8]]
        assert params["remote_host"] == "127.0.0.1"
        assert params["remote_port"] == 9100


# ---------------------------------------------------------------------------
# TestHMAWorkerLayout — flattening kv_caches into per-rank arrays
# ---------------------------------------------------------------------------


class TestHMAWorkerLayout:

    def test_extract_kv_layout_flattens_attention_and_mamba(self):
        worker = _make_worker(tp_rank=0, tp_size=1)
        runner, c = _build_hybrid_runner()
        _attach_runner(worker, runner, c["sfc"])

        worker._extract_kv_layout()

        # attention -> 1 array; mamba (conv, ssm) -> 2 arrays.
        assert worker.num_arrays == 3
        assert worker.num_layers == 3  # wire-frame count == array count
        assert worker.array_to_runner == [(0, None), (1, 0), (1, 1)]
        assert worker.array_to_group == [0, 1, 1]
        assert worker.array_inner_shape == [(2, 4), (3, 4), (5, 4)]
        assert worker.group_is_mamba == [False, True]
        # Representative shape/dtype used only for base-class logging.
        assert worker.shape == [0, 2, 4]
        assert worker.dtype == torch.float32

    def test_map_layers_to_groups_identity_match(self):
        worker = _make_worker(tp_rank=0, tp_size=1)
        runner, c = _build_hybrid_runner()
        _attach_runner(worker, runner, c["sfc"])
        mapping = worker._map_layers_to_groups(runner.kv_caches, c["groups"])
        assert mapping == [0, 1]


# ---------------------------------------------------------------------------
# TestHMAWorkerPoolSpec — per-array shm pool spec
# ---------------------------------------------------------------------------


class TestHMAWorkerPoolSpec:

    def _ready_worker(self):
        worker = _make_worker(tp_rank=0, tp_size=1)
        runner, c = _build_hybrid_runner()
        _attach_runner(worker, runner, c["sfc"])
        worker._extract_kv_layout()
        worker.vllm_config.model_config.max_model_len = 64
        return worker

    def test_build_pool_spec_mamba_gets_one_block_attention_gets_many(self):
        worker = self._ready_worker()
        with patch(f"{_HMA_MOD}.dist_utils.get_kv_shm_pool_gb",
                   return_value=1.0):
            spec = worker._build_pool_spec()

        assert isinstance(spec, PoolSpecHMA)
        # block_size=16, max_model_len=64 -> attn_max_blocks = 4.
        # array_to_group = [0(attn), 1(mamba), 1(mamba)].
        assert spec.array_max_blocks == (4, 1, 1)
        assert spec.array_to_group == (0, 1, 1)
        assert spec.num_arrays == 3
        assert spec.array_inner_shape == ((2, 4), (3, 4), (5, 4))

    def test_blocks_token_is_per_group_counts(self):
        worker = _make_worker(tp_rank=0, tp_size=1)
        # blocks token is just per-group lengths; layout not needed.
        assert worker._blocks_token([[1, 2, 3], [4]]) == (3, 1)
        assert worker._blocks_token([[], [9, 9]]) == (0, 2)


# ---------------------------------------------------------------------------
# TestHMAWorkerStaging — D2H gather (producer) & H2D insert (consumer)
# ---------------------------------------------------------------------------


class TestHMAWorkerStaging:

    def _ready_worker(self, *, is_producer=True):
        worker = _make_worker(tp_rank=0, tp_size=1, is_producer=is_producer)
        runner, c = _build_hybrid_runner()
        _attach_runner(worker, runner, c["sfc"])
        worker._extract_kv_layout()
        return worker, runner, c

    def test_build_d2h_views_gathers_per_array(self):
        worker, runner, c = self._ready_worker()
        # Seed distinct values so we can verify the gather picks the right
        # blocks for each array.
        c["attn"][2] = 1.0
        c["attn"][5] = 2.0
        c["conv"][3] = 3.0
        c["ssm"][3] = 4.0

        # Per-group block ids: group0(attn) pulls blocks [2,5];
        # group1(mamba) pulls block [3].
        block_ids = [[2, 5], [3]]
        blocks = (2, 1)  # per-group counts token

        # Pool layer_view just hands back a correctly-shaped CPU scratch
        # tensor for each array; the copy itself isn't under test here.
        def fake_layer_view(slot, rank, a, blk):
            shape = (worker._blocks_token(block_ids)[worker.array_to_group[a]],
                     ) + worker.array_inner_shape[a]
            return torch.empty(shape, dtype=worker.array_dtype[a])

        worker._coord_pool = MagicMock()
        worker._coord_pool.layer_view.side_effect = fake_layer_view

        tpu_tensors, cpu_tensors, total_bytes = worker._build_d2h_views(
            0, blocks, block_ids)

        assert len(tpu_tensors) == 3
        assert len(cpu_tensors) == 3
        # array0: attention gather of blocks [2, 5].
        assert tpu_tensors[0].shape == (2, 2, 4)
        assert tpu_tensors[0][0, 0, 0].item() == 1.0
        assert tpu_tensors[0][1, 0, 0].item() == 2.0
        # array1/2: mamba conv/ssm gather of block [3].
        assert tpu_tensors[1].shape == (1, 3, 4)
        assert tpu_tensors[1][0, 0, 0].item() == 3.0
        assert tpu_tensors[2].shape == (1, 5, 4)
        assert tpu_tensors[2][0, 0, 0].item() == 4.0
        # total_bytes is summed over the destination views.
        expected = sum(v.numel() * v.element_size() for v in cpu_tensors)
        assert total_bytes == expected

    def test_coord_scatter_shard_inserts_attention_and_mamba(self):
        worker, runner, c = self._ready_worker(is_producer=False)
        worker._synchronize_device = MagicMock()  # no real TPU sync

        block_ids = [[0, 1], [2]]  # group0 attn -> 2 blocks; group1 mamba -> 1
        blocks = (2, 1)

        # Source shards arriving from shm, distinct per array.
        attn_src = torch.full((2, 2, 4), 7.0)
        conv_src = torch.full((1, 3, 4), 8.0)
        ssm_src = torch.full((1, 5, 4), 9.0)
        shards = {0: attn_src, 1: conv_src, 2: ssm_src}
        worker._coord_pool = MagicMock()
        worker._coord_pool.layer_view.side_effect = (
            lambda slot, rank, a, blk: shards[a])

        worker._coord_scatter_shard(0, blocks, block_ids)

        new_attn = runner.kv_caches[0]
        new_conv, new_ssm = runner.kv_caches[1]
        # Attention blocks 0 and 1 written; others untouched.
        assert torch.all(new_attn[0] == 7.0)
        assert torch.all(new_attn[1] == 7.0)
        assert torch.all(new_attn[2] == 0.0)
        # Mamba block 2 written for both conv and ssm states.
        assert torch.all(new_conv[2] == 8.0)
        assert torch.all(new_ssm[2] == 9.0)
        assert torch.all(new_conv[0] == 0.0)
        # The mamba layer stays a tuple after replacement.
        assert isinstance(runner.kv_caches[1], tuple)
        # Sync was called once per replaced state tensor (attn + conv + ssm).
        assert worker._synchronize_device.call_count == 3
