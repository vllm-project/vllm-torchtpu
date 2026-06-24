# SPDX-License-Identifier: Apache-2.0
"""
TPUConnector: KV-cache connector for P/D disaggregated serving on TPU.

Ported from vllm_torchtpu/distributed/tpu_connector.py in the tpu-inference
(JAX) project, with the JAX transfer-server data plane replaced by a ZMQ-based
pull-on-demand transport that moves torch tensors over the TPU host network.

The default generic shm-staged ZMQ machinery lives in
``vllm_torchtpu.distributed.kv_transfer.zmq_shm_base.ZmqShmKvConnectorBase``;
this file holds the TPU-specific transport hooks plus the scheduler half
of the connector. A Raiden C++ transfer backend is available as an opt-in
alternative via ``kv_connector_extra_config.use_raiden_connector``.

P/D disagg of GDN models which mixing attention and Mamba layers is available
either via the dedicated ``TPUConnectorHMA`` connector class or via
``kv_connector_extra_config.use_hma_connector``.
"""

import time
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

import torch
from torch_tpu._internal.batch_transfer import (batch_transfer_d2h,
                                                batch_transfer_d2h_sync,
                                                batch_transfer_h2d,
                                                batch_transfer_h2d_sync)
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorRole, SupportsHMA)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics, KVConnectorStats, PromMetric, PromMetricT)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.utils.math_utils import round_down
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from tpu_raiden.api.torch.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

import vllm_torchtpu.distributed.utils as dist_utils
from vllm_torchtpu.distributed.kv_transfer import kv_scatter
from vllm_torchtpu.distributed.kv_transfer.host_kv_shm_hma import (
    HostKVShmPoolHMA, PoolSpecHMA)
from vllm_torchtpu.distributed.kv_transfer.tpu_connector_stats import (
    TpuKVConnectorPromMetrics, TpuKVConnectorStats)
from vllm_torchtpu.distributed.kv_transfer.zmq_shm_base import (
    LoadMeta, ReqId, SendMeta, TPUConnectorMetadata, ZmqShmKvConnectorBase,
    _CoordRecvEntry, _CoordSendEntry)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)

__all__ = [
    "TPUConnector",
    "TPUConnectorScheduler",
    "TPUConnectorWorker",
    "TPURaidenConnector",
    "TPURaidenConnectorScheduler",
    "TPURaidenConnectorWorker",
    "TPUConnectorHMA",
    "TPUConnectorHMAScheduler",
    "TPUConnectorHMAWorker",
    "TPUConnectorMetadata",
    "SendMeta",
    "LoadMeta",
    "_CoordSendEntry",
    "_CoordRecvEntry",
]


class _DoneFuture:

    def wait(self) -> None:
        return


_DONE_FUTURE = _DoneFuture()


def _get_extra_config(vllm_config: VllmConfig) -> dict[str, Any]:
    config = vllm_config.kv_transfer_config
    if config is None:
        return {}
    return getattr(config, "kv_connector_extra_config", None) or {}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("1", "true", "yes", "on"):
            return True
        if value in ("0", "false", "no", "off"):
            return False
    return bool(value)


def _use_raiden_connector(vllm_config: VllmConfig) -> bool:
    extra_config = _get_extra_config(vllm_config)
    if "use_raiden_connector" in extra_config:
        return _as_bool(extra_config["use_raiden_connector"])
    return dist_utils.get_use_raiden_connector()


def _use_hma_connector(vllm_config: VllmConfig) -> bool:
    extra_config = _get_extra_config(vllm_config)
    if "use_hma_connector" in extra_config:
        return _as_bool(extra_config["use_hma_connector"])
    return False


class TPUConnector(KVConnectorBase_V1, SupportsHMA):
    force_raiden_connector = False
    force_hma_connector = False

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole,
                 kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        self._connector_metadata: Optional[TPUConnectorMetadata] = None
        use_hma = self.force_hma_connector or _use_hma_connector(vllm_config)
        use_raiden = self.force_raiden_connector or _use_raiden_connector(
            vllm_config)
        self.use_hma = use_hma
        if use_hma:
            scheduler_cls = TPUConnectorHMAScheduler
            worker_cls = TPUConnectorHMAWorker
            backend = "HMA"
        elif use_raiden:
            scheduler_cls = TPURaidenConnectorScheduler
            worker_cls = TPURaidenConnectorWorker
            backend = "Raiden"
        else:
            scheduler_cls = TPUConnectorScheduler
            worker_cls = TPUConnectorWorker
            backend = "ZMQ-shm"
        logger.info("TPUConnector --> using %s backend", backend)

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = scheduler_cls(vllm_config)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = worker_cls(vllm_config)

    # ---- Scheduler-side methods -----------------------------------------
    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> TPUConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta()

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        assert self.connector_scheduler is not None
        if self.use_hma:
            return self.connector_scheduler.request_finished_all_groups(
                request, block_ids)
        assert len(block_ids) == 1, (
            "Non-HMA TPUConnector expects a single kv-cache group; got "
            f"{len(block_ids)} groups")
        return self.connector_scheduler.request_finished(request, block_ids[0])

    def get_finished_count(self) -> int:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_finished_count()

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """
        Get the KV transfer stats for the connector.
        """
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_kv_connector_stats()

    @classmethod
    def build_kv_connector_stats(
            cls,
            data: dict[str, Any] | None = None) -> KVConnectorStats | None:
        return (TpuKVConnectorStats(
            data=data) if data is not None else TpuKVConnectorStats())

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        return TpuKVConnectorPromMetrics(vllm_config, metric_types, labelnames,
                                         per_engine_labelvalues)

    # ---- Worker-side methods --------------------------------------------
    def register_kv_caches(self, kv_caches: dict[str, Any]):
        """For non-HMA connectors,this is a no-op: we call
        register_runner() from the runner after bind_kv_cache and read
        runner.kv_caches lazily in the worker."""
        if self.use_hma and self.connector_worker is not None:
            self.connector_worker.named_kv_caches = kv_caches

    def register_runner(self, runner: TPUModelRunner) -> None:
        assert self.connector_worker is not None
        self.connector_worker.register_runner(runner)

    def start_load_kv(self,
                      _,
                      wait_for_completion: bool = False,
                      report_completion: bool = True,
                      **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, TPUConnectorMetadata)
        self.connector_worker.process_send_load(
            self._connector_metadata,
            wait_for_completion=wait_for_completion,
            report_completion=report_completion)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Layer-wise load is not supported on TPU."""
        pass

    def save_kv_layer(self, *args, **kwargs) -> None:
        """Layer-wise save is not supported on TPU."""
        pass

    def wait_for_save(self):
        """No-op. See comment in vllm_torchtpu's TPUConnector: reqs_to_send is
        only populated after the request finishes prefill, at which point
        total_num_scheduled_tokens may be 0 and wait_for_save is not called
        by the KVConnectorModelRunnerMixin. We run the send from
        start_load_kv -> process_send_load instead."""
        pass

    def get_finished(self,
                     finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        # Surface KV-load failures so the scheduler recomputes the affected
        # blocks instead of running with absent KV. Only the Raiden worker
        # tracks these; other backends fall back to the base (no errors).
        worker = self.connector_worker
        fn = getattr(worker, "get_block_ids_with_load_errors", None)
        return fn() if fn is not None else set()


class TPURaidenConnector(TPUConnector):
    """Explicit Raiden connector class for callers that prefer connector name
    selection over ``kv_connector_extra_config.use_raiden_connector``."""

    force_raiden_connector = True


class TPUConnectorScheduler:

    def __init__(self, vllm_config: "VllmConfig"):
        self.vllm_config = vllm_config
        self.config = vllm_config.kv_transfer_config
        self.is_producer = self.config.is_kv_producer

        self.block_size = vllm_config.cache_config.block_size

        # Populated by request_finished() on P.
        self.reqs_to_send: dict[ReqId, SendMeta] = {}
        # Populated by update_state_after_alloc() on D.
        self.reqs_to_load: dict[ReqId, LoadMeta] = {}

        self.kv_ip = dist_utils.get_kv_ips()
        # Get DP rank and TP size from config and stagger kv_port and side_channel_port
        dp_rank = vllm_config.parallel_config.data_parallel_rank if vllm_config.parallel_config else 0
        tp_size = vllm_config.parallel_config.tensor_parallel_size if vllm_config.parallel_config else 1
        port_base = dist_utils.get_kv_ports()
        if isinstance(port_base, list):
            self.kv_port = [int(p) + dp_rank * tp_size for p in port_base]
        else:
            self.kv_port = int(port_base) + dp_rank * tp_size
        self.side_channel_port = int(
            dist_utils.get_side_channel_port()) + dp_rank
        logger.info(
            "TPUConnectorScheduler --> kv_ip=%s | kv_port=%s | side_channel_port=%s",
            self.kv_ip, self.kv_port, self.side_channel_port)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        D workers use this to get the number of new tokens
        that can be loaded from remote P workers.
        No-op for P workers.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that will be loaded from the
                  external KV cache.
                - If async loading. Must be 'False' for TPU connector
                  because TPU pulls KV cache in a blocking way.

        """
        if self.is_producer or not request.kv_transfer_params:
            return 0, False

        assert num_computed_tokens % self.block_size == 0
        # Rounding must match request_finished()'s remote_block_ids computation.
        rounded_num_prompt_tokens = round_down(len(request.prompt_token_ids),
                                               self.block_size)
        count = max(rounded_num_prompt_tokens - num_computed_tokens, 0)
        # The pull is blocking at the ZMQ layer, but we wrap it in a thread
        # pool so from the scheduler's perspective it's async.
        if count > 0:
            return count, True
        return 0, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        if self.is_producer or not request.kv_transfer_params:
            return

        params = request.kv_transfer_params
        if num_external_tokens > 0:
            local_block_ids = blocks.get_block_ids()[0]
            # D must pull the whole prefill blocks regardless of partial
            # prefix-cache hits, because the transport has no RDMA-style
            # partial-pull and P publishes the full payload under a uuid.
            self.reqs_to_load[request.request_id] = LoadMeta(
                uuid=params["uuid"],
                local_block_ids=local_block_ids,
                remote_block_ids=params["remote_block_ids"],
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
                remote_side_channel_port=params["remote_side_channel_port"]
                if "remote_side_channel_port" in params else None,
            )
        else:
            # Full prefix-cache hit or async pull done -- we still need to
            # notify P so it can free the pending buffer.
            self.reqs_to_load[request.request_id] = LoadMeta(
                uuid=params["uuid"],
                local_block_ids=None,
                remote_block_ids=None,
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
                remote_side_channel_port=params["remote_side_channel_port"]
                if "remote_side_channel_port" in params else None,
            )
        logger.info(
            "TPUConnectorScheduler update_state_after_alloc --> reqs_to_load=%s",
            self.reqs_to_load)

    def build_connector_meta(self) -> TPUConnectorMetadata:
        meta = TPUConnectorMetadata()
        if self.is_producer:
            meta.reqs_to_send = self.reqs_to_send
            self.reqs_to_send = {}
        else:
            meta.reqs_to_load = self.reqs_to_load
            self.reqs_to_load = {}
        return meta

    def get_finished_count(self) -> int:
        return len(self.kv_ip) if isinstance(self.kv_ip, list) else 1

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        if not self.is_producer:
            return False, None

        # max_tokens is forced to 1 by the proxy on P, so the only way the
        # prefill finishes cleanly is length cap.
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            return False, None

        # Only transfer full blocks; let D re-prefill the trailing partial
        # block locally.
        all_full = request.num_computed_tokens % self.block_size == 0
        computed_block_ids = block_ids if all_full else block_ids[:-1]

        delay_free_blocks = len(computed_block_ids) > 0
        if delay_free_blocks:
            uuid = get_uuid()
            expiration_time = (time.perf_counter() +
                               dist_utils.get_p2p_wait_pull_timeout())
            self.reqs_to_send[request.request_id] = SendMeta(
                uuid=uuid,
                local_block_ids=computed_block_ids,
                expiration_time=expiration_time)
            kv_transfer_params = dict(
                uuid=uuid,
                remote_block_ids=computed_block_ids,
                remote_host=self.kv_ip,
                remote_port=self.kv_port,
                remote_side_channel_port=self.side_channel_port)
            logger.info(
                "TPUConnectorScheduler --> reqs_to_send=%s | kv_transfer_params=%s",
                self.reqs_to_send, kv_transfer_params)
        else:
            kv_transfer_params = {}

        return delay_free_blocks, kv_transfer_params


class TPUConnectorWorker(ZmqShmKvConnectorBase):
    """TPU-specific transport hooks for the generic ZmqShmKvConnectorBase.
    """

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """
        Get the KV transfer stats for the worker.
        """
        # Clear stats for next iteration
        if not self.transfer_stats.is_empty():
            return self.transfer_stats.clone_and_reset()
        return None

    def process_send_load(self,
                          metadata: TPUConnectorMetadata,
                          wait_for_completion: bool = False,
                          report_completion: bool = True) -> None:
        del wait_for_completion, report_completion
        super().process_send_load(metadata)

    def _build_d2h_views(self, slot_idx: int, num_blocks: int,
                         block_ids: list[int]) -> tuple[list, list, int]:
        indices = torch.tensor(block_ids,
                               dtype=torch.int64,
                               device=self.device)
        kv_caches = self.runner.kv_caches
        tpu_tensors: list = []
        cpu_tensors: list = []
        d2h_total_bytes = 0
        for layer_idx, cache in enumerate(kv_caches):
            src_shard = torch.index_select(cache, 0, indices)
            dest_view = self._coord_pool.layer_view(slot_idx,
                                                    self.local_tp_rank,
                                                    layer_idx, num_blocks)
            tpu_tensors.append(src_shard)
            cpu_tensors.append(dest_view)
            d2h_total_bytes += dest_view.numel() * dest_view.element_size()
        return tpu_tensors, cpu_tensors, d2h_total_bytes

    def _stage_d2h(self, slot_idx: int, num_blocks: int,
                   block_ids: list[int]) -> tuple[Any, list, list, int]:
        tpu_tensors, cpu_tensors, total_bytes = self._build_d2h_views(
            slot_idx, num_blocks, block_ids)
        future = batch_transfer_d2h(tpu_tensors, cpu_tensors)
        return future, tpu_tensors, cpu_tensors, total_bytes

    def _stage_d2h_sync(self, slot_idx: int, num_blocks: int,
                        block_ids: list[int]) -> None:
        tpu_tensors, cpu_tensors, _ = self._build_d2h_views(
            slot_idx, num_blocks, block_ids)
        batch_transfer_d2h_sync(tpu_tensors, cpu_tensors)

    def _wait_stage(self, future: Any) -> None:
        future.wait()

    def _h2d_into_device(self, src_views: list) -> list[torch.Tensor]:
        device_shards = [
            torch.empty(v.shape, dtype=v.dtype, device=self.device)
            for v in src_views
        ]
        batch_transfer_h2d_sync(src_views, device_shards)
        return device_shards

    def _h2d_into_device_async(
            self, src_views: list) -> tuple[Any, list[torch.Tensor]]:
        device_shards = [
            torch.empty(v.shape, dtype=v.dtype, device=self.device)
            for v in src_views
        ]
        future = batch_transfer_h2d(src_views, device_shards)
        return future, device_shards

    def _synchronize_device(self, tensor: torch.Tensor) -> None:
        from torch_tpu._internal.sync import sync as tpu_sync
        tpu_sync.synchronize(tensor)

    def _try_fast_scatter(
            self, device_shards: list[torch.Tensor],
            kv_caches: list[torch.Tensor],
            local_blocks: list[int]) -> Optional[list[torch.Tensor]]:
        if not self._kv_scatter_enabled:
            return None
        dest_blocks_dev = torch.tensor(local_blocks,
                                       dtype=torch.int32,
                                       device=self.device)
        prebuilt = kv_scatter.prepare_scatter_args(dest_blocks_dev,
                                                   self.device)
        return kv_scatter.multi_layer_scatter_into(device_shards,
                                                   kv_caches,
                                                   dest_blocks_dev,
                                                   prebuilt_args=prebuilt)

    def _maybe_enable_kv_scatter(self) -> None:
        """Run the startup smoke test for the multi-layer scatter kernel
        (``kv_scatter``) and flip it on if it passes. Consumer-only. On
        by default; any failure falls back permanently to ``index_put_``
        with a warning. Smoke-tests with a small num_layers; compile for
        the real per-worker num_layers happens on first invocation
        (typically during warmup)."""
        if self.is_producer:
            return
        if not kv_scatter.scatter_available():
            logger.warning(
                "TPUConnectorWorker %s rank%d --> kv_scatter kernel "
                "unavailable; staying on index_put_", self.node_id,
                self.tp_rank)
            return
        trailing = tuple(self.shape[1:]) if len(self.shape) > 1 else (128, )
        ok = kv_scatter.smoke_test_multi_layer_scatter(
            device=self.device,
            num_layers=2,
            num_blocks=4,
            trailing_shape=trailing,
            dtype=self.dtype,
        )
        if ok:
            self._kv_scatter_enabled = True
            logger.info(
                "TPUConnectorWorker %s rank%d --> multi-layer scatter "
                "enabled for insert path", self.node_id, self.tp_rank)
        else:
            logger.warning(
                "TPUConnectorWorker %s rank%d --> multi-layer scatter "
                "smoke test failed; falling back to index_put_", self.node_id,
                self.tp_rank)


class TPURaidenConnectorScheduler(TPUConnectorScheduler):
    """Scheduler half for the opt-in Raiden transfer backend."""

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if self.is_producer or not request.kv_transfer_params:
            return 0, False

        assert num_computed_tokens % self.block_size == 0
        rounded_num_prompt_tokens = round_down(len(request.prompt_token_ids),
                                               self.block_size)
        count = max(rounded_num_prompt_tokens - num_computed_tokens, 0)
        if count > 0:
            if dist_utils.get_raiden_inline_load():
                total_external_tokens = num_computed_tokens + count
                if total_external_tokens >= len(request.prompt_token_ids):
                    count = max(count - 1, 0)
                logger.info(
                    "TPURaidenConnectorScheduler inline load req_id=%s "
                    "external_tokens=%d", request.request_id, count)
                return count, False
            return count, True
        return 0, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        if self.is_producer or not request.kv_transfer_params:
            return

        params = request.kv_transfer_params
        if num_external_tokens > 0:
            # D must pull the WHOLE prefill payload: P publishes it under one
            # uuid with no partial-pull, so load into every allocated block.
            local_block_ids = blocks.get_block_ids()[0]
            if not local_block_ids:
                self.reqs_to_load[request.request_id] = LoadMeta(
                    uuid=params["uuid"],
                    local_block_ids=None,
                    remote_block_ids=None,
                    remote_host=params["remote_host"],
                    remote_port=params["remote_port"],
                )
                logger.info(
                    "TPURaidenConnectorScheduler prefix hit req_id=%s "
                    "releases remote send uuid=%s", request.request_id,
                    params["uuid"])
                return

            remote_block_ids = params["remote_block_ids"]
            if len(local_block_ids) > len(remote_block_ids):
                raise ValueError(
                    "TPURaidenConnector cannot pull more local blocks than "
                    "the producer published: "
                    f"local={len(local_block_ids)} remote="
                    f"{len(remote_block_ids)}")
            remote_block_ids = remote_block_ids[-len(local_block_ids):]
            self.reqs_to_load[request.request_id] = LoadMeta(
                uuid=params["uuid"],
                local_block_ids=local_block_ids,
                remote_block_ids=remote_block_ids,
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
            )
        else:
            self.reqs_to_load[request.request_id] = LoadMeta(
                uuid=params["uuid"],
                local_block_ids=None,
                remote_block_ids=None,
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
            )
        logger.info(
            "TPURaidenConnectorScheduler update_state_after_alloc --> "
            "reqs_to_load=%s", self.reqs_to_load)

    def get_finished_count(self) -> int:
        # Raiden reports completion from every local worker rank. Returning 0
        # asks vLLM's KVOutputAggregator to use the model runner world size.
        return 0


class TPURaidenConnectorWorker:
    """Worker-side integration for the opt-in Raiden transfer backend.

    Python owns vLLM request metadata only. Raiden C++ owns host slots, D2H,
    H2H, H2D, readiness, and completion state.
    """

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.config = vllm_config.kv_transfer_config
        self.is_producer = self.config.is_kv_producer
        self.runner: Optional[TPUModelRunner] = None
        self.node_id = dist_utils.get_node_id()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.host_ip = dist_utils.get_host_ip()
        self.kv_transfer_port = int(dist_utils.get_kv_transfer_port())
        self._raiden_transfer_engine: Optional["KVCacheManager"] = None
        self._done_sending: set[str] = set()
        self._done_recving: set[str] = set()
        self._failed_recving: set[str] = set()
        self._suppress_done_recving: set[str] = set()
        # Report each req's recv-completion to the scheduler at most once. vLLM's
        # _update_from_kv_xfer_finished asserts a finished_recving req is still
        # WAITING_FOR_REMOTE_KVS (or finished); a duplicate/late report (after
        # the req was admitted -> RUNNING) trips that assert.
        self._reported_recving: set[str] = set()
        # req_id -> local (device) block ids, retained while a load is in flight
        # so a failed recv can surface the affected blocks for recompute. Pruned
        # once the request finishes generating (see get_finished).
        self._load_block_ids: dict[str, list[int]] = {}
        # Block ids of failed loads, drained by get_block_ids_with_load_errors().
        self._failed_block_ids: set[int] = set()
        logger.info(
            "TPURaidenConnectorWorker --> init | ip=%s | base_port=%s | "
            "is_producer=%s | node_id=%s | tp_rank=%d | tp_size=%d",
            self.host_ip, self.kv_transfer_port, self.is_producer,
            self.node_id, self.tp_rank, self.tp_size)

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        return None

    def register_runner(self, runner: TPUModelRunner) -> None:
        self.runner = runner
        self._ensure_raiden_transfer_engine()

    def process_send_load(self,
                          metadata: TPUConnectorMetadata,
                          wait_for_completion: bool = False,
                          report_completion: bool = True) -> None:
        engine = self._ensure_raiden_transfer_engine()
        if self.is_producer:
            for req_id, req_meta in metadata.reqs_to_send.items():
                engine.register_read(req_id, req_meta.uuid,
                                     req_meta.local_block_ids)
                logger.debug(
                    "TPURaidenConnectorWorker rank%d --> registered send "
                    "req_id=%s uuid=%s blocks=%d", self.tp_rank, req_id,
                    req_meta.uuid, len(req_meta.local_block_ids))
            return

        submitted_loads: set[str] = set()
        for req_id, req_meta in metadata.reqs_to_load.items():
            endpoint = self._resolve_remote_endpoint(req_meta)
            if (req_meta.remote_block_ids is None
                    and req_meta.local_block_ids is None):
                engine.start_read(req_id, req_meta.uuid, endpoint, [], [])
                logger.debug(
                    "TPURaidenConnectorWorker rank%d --> released remote "
                    "send req_id=%s uuid=%s endpoint=%s", self.tp_rank, req_id,
                    req_meta.uuid, endpoint)
                continue
            if (req_meta.remote_block_ids is None
                    or req_meta.local_block_ids is None):
                raise ValueError(
                    "TPURaidenConnector load metadata must contain both "
                    "remote and local block ids, or neither")

            remote_blocks = req_meta.remote_block_ids
            local_blocks = req_meta.local_block_ids
            engine.start_read(req_id, req_meta.uuid, endpoint, remote_blocks,
                              local_blocks)
            logger.debug(
                "TPURaidenConnectorWorker rank%d --> submitted load "
                "req_id=%s uuid=%s endpoint=%s remote_blocks=%d "
                "local_blocks=%d", self.tp_rank, req_id, req_meta.uuid,
                endpoint, len(remote_blocks), len(local_blocks))
            submitted_loads.add(req_id)
            self._load_block_ids[req_id] = list(local_blocks)
        if wait_for_completion:
            self._wait_for_recving(submitted_loads)
            if not report_completion:
                self._suppress_done_recving.update(submitted_loads)

    def get_finished(
            self,
            finished_req_ids: set[str] | None = None
    ) -> tuple[set[str], set[str]]:
        engine = self._ensure_raiden_transfer_engine()
        self._poll_finished(engine)
        done_sending = self._done_sending
        # Report failed recvs as finished too, so the scheduler stops waiting in
        # WAITING_FOR_REMOTE_KVS until the API timeout; the affected blocks are
        # surfaced via get_block_ids_with_load_errors() in the same pass so vLLM
        # recomputes them rather than running with absent KV.
        recv_finished = self._done_recving | self._failed_recving
        done_recving = (recv_finished - self._suppress_done_recving -
                        self._reported_recving)
        self._suppress_done_recving.difference_update(recv_finished)
        self._reported_recving.update(done_recving)
        if done_recving:
            logger.debug(
                "TPURaidenConnectorWorker rank%d --> reporting done_recving=%s",
                self.tp_rank, done_recving)
        # Bound the bookkeeping: once a request has finished generating it can no
        # longer be re-reported, so drop its dedup / failure / block-id state
        # (without this these sets grow for the lifetime of the process).
        if finished_req_ids:
            self._reported_recving -= finished_req_ids
            self._failed_recving -= finished_req_ids
            for req_id in finished_req_ids:
                self._load_block_ids.pop(req_id, None)
        self._done_sending = set()
        self._done_recving = set()
        return done_sending, done_recving

    def _poll_finished(self, engine: "KVCacheManager") -> None:
        done_sending, done_recving, failed_recving = engine.poll_stats()
        self._done_sending.update(done_sending)
        self._done_recving.update(done_recving)
        newly_failed = set(failed_recving) - self._failed_recving
        self._failed_recving.update(failed_recving)
        # Stage the affected device blocks so get_block_ids_with_load_errors()
        # can report them no later than the pass that reports the req finished.
        for req_id in newly_failed:
            blocks = self._load_block_ids.get(req_id)
            if blocks:
                self._failed_block_ids.update(blocks)
        if failed_recving:
            logger.error(
                "TPURaidenConnectorWorker rank%d --> failed_recving=%s",
                self.tp_rank, failed_recving)

    def get_block_ids_with_load_errors(self) -> set[int]:
        # Drain the failed-load block ids for the scheduler to recompute. Paired
        # with reporting the req as finished in get_finished() in the same pass.
        failed = self._failed_block_ids
        self._failed_block_ids = set()
        return failed

    def _wait_for_recving(self, req_ids: set[str]) -> None:
        if not req_ids:
            return
        engine = self._ensure_raiden_transfer_engine()
        deadline = (time.perf_counter() +
                    float(dist_utils.get_p2p_wait_pull_timeout()))
        while True:
            finished = self._done_recving | self._failed_recving
            if req_ids <= finished:
                return
            self._poll_finished(engine)
            finished = self._done_recving | self._failed_recving
            if req_ids <= finished:
                return
            if time.perf_counter() >= deadline:
                pending = sorted(req_ids - finished)
                logger.warning(
                    "TPURaidenConnectorWorker rank%d --> timed out waiting "
                    "for Raiden load completion for req_ids=%s", self.tp_rank,
                    pending)
                return
            time.sleep(0.001)

    def _ensure_raiden_transfer_engine(self) -> "KVCacheManager":
        if self._raiden_transfer_engine is not None:
            return self._raiden_transfer_engine
        if self.runner is None:
            raise RuntimeError(
                "register_runner must be called before transfer")
        max_blocks = self._max_request_blocks()
        num_slots = self._num_raiden_slots(max_blocks)
        local_control_port = self._rank_control_port(self.kv_transfer_port)
        from tpu_raiden.api.torch.kv_cache_manager import KVCacheManager
        engine = KVCacheManager(
            kv_caches=list(self.runner.kv_caches),
            node_id=self.tp_rank,
            local_control_port=local_control_port,
            max_blocks=max_blocks,
            num_slots=num_slots,
            timeout_s=float(dist_utils.get_p2p_wait_pull_timeout()),
        )
        self._raiden_transfer_engine = engine
        logger.info(
            "TPURaidenConnectorWorker rank%d --> Raiden engine enabled | "
            "control_port=%d data_port=%d max_blocks=%d num_slots=%d",
            self.tp_rank, local_control_port, local_control_port + 1,
            max_blocks, num_slots)
        return engine

    def _rank_control_port(self, base_port: int) -> int:
        return int(base_port) + 2 * self.tp_rank

    def _resolve_remote_endpoint(self, req_meta: LoadMeta) -> str:
        if isinstance(req_meta.remote_host, list):
            host = req_meta.remote_host[self.node_id]
            base_port = int(req_meta.remote_port[self.node_id])
        else:
            host = req_meta.remote_host
            base_port = int(req_meta.remote_port)
        return f"{host}:{self._rank_control_port(base_port)}"

    def _max_request_blocks(self) -> int:
        block_size = self.vllm_config.cache_config.block_size
        max_model_len = self.vllm_config.model_config.max_model_len
        return max(1, (max_model_len + block_size - 1) // block_size)

    def _num_raiden_slots(self, max_blocks: int) -> int:
        override = dist_utils.get_raiden_transfer_num_slots()
        if override > 0:
            return override
        assert self.runner is not None
        kv_layer = self.runner.kv_caches[0]
        dtype_bytes = torch.tensor([], dtype=kv_layer.dtype).element_size()
        bytes_per_layer = dtype_bytes * max_blocks
        for dim in kv_layer.shape[1:]:
            bytes_per_layer *= int(dim)
        bytes_per_slot = bytes_per_layer * len(self.runner.kv_caches)
        per_rank_budget = int(dist_utils.get_kv_shm_pool_gb() * (1024**3))
        per_rank_budget //= max(1, self.tp_size)
        return max(1, per_rank_budget // max(1, bytes_per_slot))


class TPUConnectorHMA(TPUConnector):
    """Explicit HMA connector class for callers that prefer connector name
    selection over ``kv_connector_extra_config.use_hma_connector``. Supports
    P/D disagg of hybrid attention+Mamba models."""

    force_hma_connector = True


class TPUConnectorHMAScheduler(TPUConnectorScheduler):

    def _maybe_truncate_for_mamba(self, request: "Request") -> None:
        """P-side: drop the last prompt token so the prefiller computes the
        Mamba recurrent state h(N-1) (state before the last token) instead
        of h(N).

        For attention that recompute is idempotent, but a Mamba layer would
        fold the last token through the conv1d/SSM recurrence a second time
        (P's transferred state already includes it), corrupting the first
        decode logit. By having P prefill shipping h(N-1),
        the decoder's recompute of the last token reproduces h(N) exactly.
        """
        if request.num_prompt_tokens <= 1:
            return
        params = request.kv_transfer_params
        if params is not None and params.get("_p_side_truncated"):
            return
        if request.prompt_token_ids is not None:
            request.prompt_token_ids.pop()
        else:
            return
        request._all_token_ids.pop()
        request.num_prompt_tokens -= 1
        if request.kv_transfer_params is None:
            request.kv_transfer_params = {}
        request.kv_transfer_params["_p_side_truncated"] = True

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if self.is_producer:
            self._maybe_truncate_for_mamba(request)
            return 0, False
        if not request.kv_transfer_params:
            return 0, False

        # Pull all prompt tokens except the last one. vLLM forces D to
        # recompute the last prompt token (see _maybe_truncate_for_mamba).
        count = max((len(request.prompt_token_ids) - 1) - num_computed_tokens,
                    0)
        if count > 0:
            return count, True
        return 0, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        if self.is_producer or not request.kv_transfer_params:
            return

        params = request.kv_transfer_params
        if num_external_tokens > 0:
            local_block_ids = list(blocks.get_block_ids())
            assert all(isinstance(g, list) for g in local_block_ids), (
                f"Expected list[list[int]] from blocks.get_block_ids() in "
                f"HMA mode; got {local_block_ids}")
            self.reqs_to_load[request.request_id] = LoadMeta(
                uuid=params["uuid"],
                local_block_ids=local_block_ids,
                remote_block_ids=params["remote_block_ids"],
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
            )
        else:
            self.reqs_to_load[request.request_id] = LoadMeta(
                uuid=params["uuid"],
                local_block_ids=None,
                remote_block_ids=None,
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
            )
        load_meta = self.reqs_to_load[request.request_id]
        logger.info(
            "TPUConnectorHMAScheduler Decode --> load queued | req_id=%s | "
            "uuid=%s | remote_host=%s | remote_port=%s | pending_loads=%d",
            request.request_id, load_meta.uuid, load_meta.remote_host,
            load_meta.remote_port, len(self.reqs_to_load))

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        if not self.is_producer:
            return False, None

        # max_tokens is forced to 1 by the proxy on P, so the only clean
        # finish is length cap.
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            return False, None

        # Per-group, no trim/rounding for HMA.
        computed_per_group: list[list[int]] = [
            list(one_group) for one_group in block_ids
        ]
        delay_free_blocks = any(len(g) > 0 for g in computed_per_group)

        if delay_free_blocks:
            uuid = get_uuid()
            expiration_time = (time.perf_counter() +
                               dist_utils.get_p2p_wait_pull_timeout())
            self.reqs_to_send[request.request_id] = SendMeta(
                uuid=uuid,
                local_block_ids=computed_per_group,
                expiration_time=expiration_time)
            kv_transfer_params = dict(uuid=uuid,
                                      remote_block_ids=computed_per_group,
                                      remote_host=self.kv_ip,
                                      remote_port=self.kv_port)
            logger.info(
                "TPUConnectorHMAScheduler Prefill --> send queued | "
                "req_id=%s | uuid=%s | num_prompt_tokens=%d | "
                "pending_sends=%d", request.request_id, uuid,
                len(request.prompt_token_ids), len(self.reqs_to_send))
        else:
            kv_transfer_params = {}
        return delay_free_blocks, kv_transfer_params

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        raise AssertionError(
            "This scheduler only expects `request_finished_all_groups`.")


class TPUConnectorHMAWorker(ZmqShmKvConnectorBase):
    """Worker half of the HMA connector."""

    # layer_name -> kv cache dict used to resolve each positional kv cache's
    # group by object id in _map_layers_to_groups.
    named_kv_caches: Optional[dict[str, Any]] = None

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """Get the KV transfer stats for the worker."""
        # Clear stats for next iteration
        if not self.transfer_stats.is_empty():
            return self.transfer_stats.clone_and_reset()
        return None

    def process_send_load(self,
                          metadata: TPUConnectorMetadata,
                          wait_for_completion: bool = False,
                          report_completion: bool = True) -> None:
        del wait_for_completion, report_completion
        super().process_send_load(metadata)

    def _synchronize_device(self, tensor: torch.Tensor) -> None:
        from torch_tpu._internal.sync import sync as tpu_sync
        tpu_sync.synchronize(tensor)

    def _extract_kv_layout(self) -> None:
        runner = self.runner
        kv_caches = runner.kv_caches
        groups = runner.kv_cache_config.kv_cache_groups
        self.group_is_mamba: list[bool] = [
            isinstance(g.kv_cache_spec, MambaSpec) for g in groups
        ]
        layer_to_group = self._map_layers_to_groups(kv_caches, groups)

        # Flatten into per-rank arrays. A mamba layer (tuple) contributes one
        # array per state tensor; an attention layer contributes one array.
        self.array_to_runner: list[tuple[int, Optional[int]]] = []
        self.array_to_group: list[int] = []
        self.array_inner_shape: list[tuple] = []
        self.array_dtype: list = []
        for layer_idx, cache in enumerate(kv_caches):
            gid = layer_to_group[layer_idx]
            if isinstance(cache, tuple):
                for sub, t in enumerate(cache):
                    self.array_to_runner.append((layer_idx, sub))
                    self.array_to_group.append(gid)
                    self.array_inner_shape.append(tuple(t.shape[1:]))
                    self.array_dtype.append(t.dtype)
            else:
                self.array_to_runner.append((layer_idx, None))
                self.array_to_group.append(gid)
                self.array_inner_shape.append(tuple(cache.shape[1:]))
                self.array_dtype.append(cache.dtype)

        self.num_arrays = len(self.array_to_runner)
        # num_layers is the per-rank wire frame count in the base class.
        self.num_layers = self.num_arrays
        # Representative shape/dtype for the base's logging only.
        self.shape = [0] + list(self.array_inner_shape[0])
        self.dtype = self.array_dtype[0]
        logger.info(
            "TPUConnectorHMAWorker %s rank%d --> layout | num_groups=%d | "
            "group_is_mamba=%s | num_layers=%d | num_arrays=%d | ",
            self.node_id, self.tp_rank, len(groups), self.group_is_mamba,
            len(kv_caches), self.num_arrays)

    def _map_layers_to_groups(self, kv_caches: list, groups: list) -> list:
        """Map each entry in ``runner.kv_caches`` to its kv-cache-group id.
        """
        named = self.named_kv_caches
        name_to_group: dict[str, int] = {
            name: gid
            for gid, g in enumerate(groups)
            for name in g.layer_names
        }
        id_to_name = {id(cache): name for name, cache in named.items()}
        mapping: list[Optional[int]] = []
        for cache in kv_caches:
            name = id_to_name.get(id(cache))
            assert name is not None, (
                "a runner kv cache was not found by identity in the "
                "registered named_kv_caches dict.")
            mapping.append(name_to_group[name])
        return mapping  # type: ignore[return-value]

    def _build_pool_spec(self) -> PoolSpecHMA:
        block_size = self.vllm_config.cache_config.block_size
        max_model_len = self.vllm_config.model_config.max_model_len
        attn_max_blocks = (max_model_len + block_size - 1) // block_size
        # Mamba groups hold a single recurrent state per sequence.
        array_max_blocks = tuple(
            1 if self.group_is_mamba[self.
                                     array_to_group[a]] else attn_max_blocks
            for a in range(self.num_arrays))

        def make_spec(num_slots: int) -> PoolSpecHMA:
            return PoolSpecHMA(
                num_slots=num_slots,
                tp_size=self.tp_size,
                num_arrays=self.num_arrays,
                array_inner_shape=tuple(self.array_inner_shape),
                array_dtype=tuple(self.array_dtype),
                array_max_blocks=array_max_blocks,
                array_to_group=tuple(self.array_to_group),
            )

        probe = make_spec(1)
        per_slot_bytes = probe.per_slot_bytes
        budget_bytes = int(dist_utils.get_kv_shm_pool_gb() * (1024**3))
        num_slots = max(1, budget_bytes // per_slot_bytes)
        logger.info(
            "TPUConnectorHMAWorker %s rank%d --> shm pool budget=%.2fGB "
            "per_slot=%.2fMB -> num_slots=%d", self.node_id, self.tp_rank,
            budget_bytes / (1024**3), per_slot_bytes / (1024**2), num_slots)
        return make_spec(num_slots)

    def _pool_create(self, spec: PoolSpecHMA,
                     shm_name: str) -> HostKVShmPoolHMA:
        return HostKVShmPoolHMA.create(spec, shm_name)

    def _pool_attach(self, spec: PoolSpecHMA,
                     shm_name: str) -> HostKVShmPoolHMA:
        return HostKVShmPoolHMA.attach(spec, shm_name)

    def _blocks_token(self, local_block_ids: Any) -> tuple:
        """Per-kv-cache-group block counts. Indexed by group id so the pool
        can resolve each array's runtime block count via array_to_group."""
        return tuple(len(g) for g in local_block_ids)

    def _pull_response_header(self, entry) -> dict:
        return {
            "tp_size": self.tp_size,
            "num_layers": self.num_layers,
            "num_blocks": entry.num_blocks,
        }

    def _maybe_enable_kv_scatter(self) -> None:
        """The fused kv_scatter kernel is not available for HMA yet."""
        return

    def _warmup_block_sizes(self) -> list[int]:
        """ Skip the kv ops warmup path for now. """
        return []

    # ---- D2H staging (producer) -----------------------------------------
    def _build_d2h_views(self, slot_idx: int, blocks: Any,
                         block_ids: Any) -> tuple[list, list, int]:
        """``blocks`` is the per-group count token; ``block_ids`` is the
        per-group list[list[int]] to gather. One (src, dst) pair per flat
        array."""
        kv_caches = self.runner.kv_caches
        tpu_tensors: list = []
        cpu_tensors: list = []
        total_bytes = 0
        for a in range(self.num_arrays):
            gid = self.array_to_group[a]
            ids = block_ids[gid]
            indices = torch.tensor(ids, dtype=torch.int64, device=self.device)
            layer_idx, sub = self.array_to_runner[a]
            src = kv_caches[layer_idx] if sub is None else kv_caches[
                layer_idx][sub]
            src_shard = torch.index_select(src, 0, indices)
            dest_view = self._coord_pool.layer_view(slot_idx, self.tp_rank, a,
                                                    blocks)
            tpu_tensors.append(src_shard)
            cpu_tensors.append(dest_view)
            total_bytes += dest_view.numel() * dest_view.element_size()
        return tpu_tensors, cpu_tensors, total_bytes

    # The batched DMA primitives (batch_transfer_d2h/h2d) assume a *uniform*
    # batch. This is not working for mamba+attention arrays yet. Use ordinary
    # shape-aware torch copies instead until DMA primitives are available.
    def _stage_d2h(self, slot_idx: int, blocks: Any,
                   block_ids: Any) -> tuple[Any, list, list, int]:
        tpu_tensors, cpu_tensors, total_bytes = self._build_d2h_views(
            slot_idx, blocks, block_ids)
        for src_shard, dest_view in zip(tpu_tensors, cpu_tensors):
            dest_view.copy_(src_shard.cpu())
        return _DONE_FUTURE, [], [], total_bytes

    def _wait_stage(self, future: Any) -> None:
        # D2H/H2D completed synchronously in the helpers above.
        return

    def _h2d_into_device_async(
            self, src_views: list) -> tuple[Any, list[torch.Tensor]]:
        return _DONE_FUTURE, [v.to(self.device) for v in src_views]

    # ---- H2D + insert (consumer) ----------------------------------------
    def _coord_scatter_shard(self, slot_idx: int, blocks: Any,
                             local_blocks: Any) -> None:
        """H2D each array's shm shard into HBM and insert it into the kv
        cache at the per-group local block ids. ``blocks`` is the per-group
        count token; ``local_blocks`` is the per-group list[list[int]].

        This will be replaced by KV scatter Pallas kernel implementation."""
        if not any(len(g) > 0 for g in local_blocks):
            return
        kv_caches = self.runner.kv_caches

        alloc_t0 = time.perf_counter()
        src_views = [
            self._coord_pool.layer_view(slot_idx, self.tp_rank, a, blocks)
            for a in range(self.num_arrays)
        ]
        h2d_total_bytes = sum(v.numel() * v.element_size() for v in src_views)
        alloc_t1 = time.perf_counter()

        h2d_t0 = time.perf_counter()
        device_shards = [v.to(self.device) for v in src_views]
        h2d_t1 = time.perf_counter()

        insert_t0 = time.perf_counter()
        # Insert each array's shard into its kv cache via functional index_put.
        group_indices: dict[int, torch.Tensor] = {}
        for gid, ids in enumerate(local_blocks):
            if ids:
                group_indices[gid] = torch.tensor(ids,
                                                  dtype=torch.int64,
                                                  device=self.device)

        new_attn: dict[int, torch.Tensor] = {}
        new_mamba: dict[int, list] = {}
        for a in range(self.num_arrays):
            indices = group_indices.get(self.array_to_group[a])
            if indices is None:
                continue
            li, sub = self.array_to_runner[a]
            if sub is None:
                new_attn[li] = kv_caches[li].index_put((indices, ),
                                                       device_shards[a])
            else:
                lst = new_mamba.setdefault(li, list(kv_caches[li]))
                lst[sub] = lst[sub].index_put((indices, ), device_shards[a])

        for layer_idx, new_cache in new_attn.items():
            self._synchronize_device(new_cache)
            self._replace_runner_kv_cache(layer_idx, new_cache)
        for layer_idx, lst in new_mamba.items():
            for t in lst:
                self._synchronize_device(t)
            self._replace_runner_kv_cache(layer_idx, tuple(lst))
        insert_t1 = time.perf_counter()

        h2d_ms = (h2d_t1 - h2d_t0) * 1000.0
        insert_ms = (insert_t1 - insert_t0) * 1000.0
        h2d_mb = h2d_total_bytes / (1024 * 1024)
        logger.info(
            "TPUConnectorHMAWorker %s rank%d --> insert slot=%d "
            "blocks_per_group=%s arrays=%d alloc=%.2fms h2d=%.2fms "
            "insert=%.2fms h2d_MiB=%.2f", self.node_id, self.tp_rank, slot_idx,
            list(blocks), self.num_arrays, (alloc_t1 - alloc_t0) * 1000.0,
            h2d_ms, insert_ms, h2d_mb)


def get_uuid() -> int:
    int128 = uuid4().int
    # Stay under 64-bit so JSON-encoded responses through the proxy are safe.
    return int128 >> 78
