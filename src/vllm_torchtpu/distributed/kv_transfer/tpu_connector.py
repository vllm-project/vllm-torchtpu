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

import json
import time
from collections import OrderedDict
from dataclasses import dataclass
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
from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.distributed.kv_transfer import kv_scatter
from vllm_torchtpu.distributed.kv_transfer.connector_metadata import (
    LoadMeta, ReqId, SendMeta, TPUConnectorMetadata)
from vllm_torchtpu.distributed.kv_transfer.host_kv_shm_hma import (
    HostKVShmPoolHMA, PoolSpecHMA)
from vllm_torchtpu.distributed.kv_transfer.tpu_connector_stats import (
    TpuKVConnectorPromMetrics, TpuKVConnectorStats)
# zmq/shm transport pieces: used only by the zmq-stack workers below
# (TPUConnectorWorker, TPUConnectorHMAWorker), not by the Raiden connector.
from vllm_torchtpu.distributed.kv_transfer.zmq_shm_base import (
    ZmqShmKvConnectorBase, _CoordRecvEntry, _CoordSendEntry)
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
    "stage3_fa_raiden_id_fields",
]


class _DoneFuture:

    def wait(self) -> None:
        return


_DONE_FUTURE = _DoneFuture()
_STAGE3_FINISH_DEDUP_LIMIT = 4096
_STAGE3_D5_REGISTRATION_WAIT_S = 5.0
_STAGE3_REGISTRATION_CANCELLED_ERROR = (
    "Request block registration was cancelled")
# Pool selection is vLLM policy expressed as request data: the connector names
# the manifest tags to move and raiden resolves them against both peers'
# registered manifests. Each GDN state class rides a sibling transfer with a
# T3.1: one transfer carries the FA payload plus every GDN state class
# under the base req_id/uuid; the tags below define the canonical order
# (FA first = H2D order rank 0).
_STAGE3_TRANSFER_POOL_TAGS = ("fa", )
_STAGE3_STATE_CLASS_TAGS = ("gdn.conv", "gdn.ssm")


def _select_committed_mamba_blocks(
    block_tables: list[list[int]],
    num_speculative_blocks: int,
) -> list[int]:
    """Select each Mamba group's last committed state checkpoint.

    The final ``K`` entries are speculative, so the committed checkpoint is
    the ``-(K + 1)`` entry for a group with speculative depth ``K``.
    """
    required_blocks = num_speculative_blocks + 1
    committed_blocks: list[int] = []
    for ordinal, block_ids in enumerate(block_tables):
        assert len(block_ids) >= required_blocks, (
            "Mamba block table cannot contain the committed checkpoint and "
            f"speculative checkpoints: ordinal={ordinal}, "
            f"blocks={len(block_ids)}, required={required_blocks}")
        committed_blocks.append(int(block_ids[-required_blocks]))
    return committed_blocks


@dataclass
class _Stage3LoadMeta:
    """Decode-local declaration for one controller-driven FA transfer.

    Source physical block IDs deliberately do not appear here.  They stay in
    the source controller's D5 registry; the only source-side routing datum
    crossing the P/D boundary is its controller address.
    """

    uuid: int
    source_req_id: str
    local_block_ids: list[int]
    num_tokens: int
    src_controller_address: str
    src_job_name: str
    src_engine_id: str
    src_data_replica_idx: int
    src_parallelism: int
    # Uniform-mamba-layout destination state slots, one per mamba kv-cache
    # group ordinal: the block holding the recurrent state for the resumed
    # request (state follows each group's block table; None for FA-only
    # models).
    mamba_state_block_ids: Optional[list[int]] = None


@dataclass(frozen=True)
class _Stage3RegisteredSend:
    """Worker-local D5 registration retained until terminal send state."""

    uuid: int
    local_block_ids: tuple[int, ...]
    num_tokens: int
    expiration_time: float


def _get_extra_config(vllm_config: VllmConfig) -> dict[str, Any]:
    config = vllm_config.kv_transfer_config
    if config is None:
        return {}
    return config.kv_connector_extra_config or {}


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


def stage3_fa_raiden_id_fields(
    *,
    job_name: str,
    engine_id: str,
    dp_rank: int,
    transfer_rank: int,
    is_producer: bool,
) -> dict[str, Any]:
    """Build the canonical Stage-3 FA work-unit identity.

    PCP ranks share a vLLM DP identity, so the transfer rank is part of the
    producer's engine replica label.  Decode engines are already distinguished
    by ``data_replica_idx``.  Keeping this in one helper lets the VB3 consumer
    enumerate source units with exactly the same scheme.
    """
    job_name = str(job_name).strip()
    engine_id = str(engine_id).strip()
    dp_rank = int(dp_rank)
    transfer_rank = int(transfer_rank)
    if not job_name:
        raise ValueError("Stage-3 Raiden job_name must not be empty")
    if not engine_id:
        raise ValueError("Stage-3 Raiden engine_id must not be empty")
    if dp_rank < 0:
        raise ValueError("Stage-3 Raiden dp_rank must be non-negative")
    if transfer_rank < 0:
        raise ValueError("Stage-3 Raiden transfer_rank must be non-negative")
    replica_id = (f"{engine_id}-rank{transfer_rank}"
                  if is_producer else engine_id)
    return {
        "job_name": job_name,
        "job_replica_id": replica_id,
        "data_name": "kv.fa",
        "data_replica_idx": dp_rank,
    }


def _use_raiden_stage3_transport() -> bool:
    return str(tpu_envs.TPU_KV_RESHARD_TRANSPORT).strip().lower() == "raiden"


def _reshard_store_mode() -> bool:
    """In-engine hosting: the reshard service lives in each engine's rank-0
    worker (an in-process KVCacheStore), not an external controller."""
    return str(tpu_envs.TPU_RAIDEN_RESHARD_IMPL).strip().lower() == "store"


def _reshard_advertise_host() -> str:
    host = str(tpu_envs.TPU_RAIDEN_ADVERTISE_HOST).strip()
    if not host:
        raise ValueError("TPU_RAIDEN_ADVERTISE_HOST is required when "
                         "TPU_RAIDEN_RESHARD_IMPL=store")
    return host


def _reshard_service_port(dp_rank: int) -> int:
    return int(tpu_envs.TPU_RAIDEN_RESHARD_PORT_BASE) + int(dp_rank)


def _reshard_service_address(dp_rank: int) -> str:
    return f"{_reshard_advertise_host()}:{_reshard_service_port(dp_rank)}"


def _reshard_dispatch_address(dp_rank: int) -> str:
    port = int(tpu_envs.TPU_RAIDEN_STORE_DISPATCH_PORT_BASE) + int(dp_rank)
    return f"{_reshard_advertise_host()}:{port}"


def _resolved_reshard_controller_address(dp_rank: int) -> str:
    """The address the facade dials and the handoff advertises: the
    per-engine in-process store under store mode, else the role-level
    external controller from the environment."""
    if _reshard_store_mode():
        return _reshard_service_address(dp_rank)
    return str(tpu_envs.TPU_RAIDEN_CONTROLLER_ADDRESS).strip()


def _use_raiden_connector(vllm_config: VllmConfig) -> bool:
    if _use_raiden_stage3_transport():
        return True
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
        self.use_raiden = use_raiden
        self._stage3_fa_group_index = 0
        if use_raiden and _use_raiden_stage3_transport():
            fa_groups = [
                index
                for index, group in enumerate(kv_cache_config.kv_cache_groups)
                if not isinstance(group.kv_cache_spec, MambaSpec)
            ]
            if len(fa_groups) != 1:
                raise ValueError(
                    "Stage-3 Qwen3.5 resharding requires exactly one "
                    f"full-attention KV cache group, got {fa_groups}")
            self._stage3_fa_group_index = fa_groups[0]
        self._stage3_mamba_group_indices: list[int] = []
        stage3_mamba_num_speculative_blocks = 0
        if use_raiden and _use_raiden_stage3_transport():
            speculative_depths: set[int] = set()
            for index, group in enumerate(kv_cache_config.kv_cache_groups):
                spec = group.kv_cache_spec
                if isinstance(spec, MambaSpec):
                    self._stage3_mamba_group_indices.append(index)
                    speculative_depths.add(spec.num_speculative_blocks)
            assert len(speculative_depths) <= 1, (
                "Stage-3 requires a uniform speculative depth across Mamba "
                f"KV cache groups: depths={speculative_depths}")
            stage3_mamba_num_speculative_blocks = next(
                iter(speculative_depths), 0)
            assert stage3_mamba_num_speculative_blocks >= 0, (
                "vLLM produced a MambaSpec with a negative speculative "
                f"depth: value={stage3_mamba_num_speculative_blocks}")
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
            if use_raiden and _use_raiden_stage3_transport():
                self.connector_scheduler._stage3_fa_group_index = (
                    self._stage3_fa_group_index)
                self.connector_scheduler._stage3_mamba_group_indices = (
                    self._stage3_mamba_group_indices)
                self.connector_scheduler._stage3_mamba_num_speculative_blocks = (
                    stage3_mamba_num_speculative_blocks)
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
        if self.use_raiden and _use_raiden_stage3_transport():
            if self._stage3_fa_group_index >= len(block_ids):
                raise ValueError(
                    "Stage-3 FA KV cache group is absent from request block "
                    f"tables: index={self._stage3_fa_group_index}, "
                    f"groups={len(block_ids)}")
            mamba_block_ids = None
            if self._stage3_mamba_group_indices:
                mamba_block_ids = []
                for mamba_gid in self._stage3_mamba_group_indices:
                    if mamba_gid >= len(block_ids):
                        raise ValueError(
                            "GDN state reshard: mamba KV cache group is "
                            "absent from request block tables: "
                            f"index={mamba_gid}, groups={len(block_ids)}")
                    mamba_block_ids.append(list(block_ids[mamba_gid]))
            # The use_hma branch returned above, so use_raiden here means
            # __init__ selected TPURaidenConnectorScheduler -- the only one
            # of the three schedulers accepting mamba block ids. The checker
            # cannot narrow the union to that subclass.
            return self.connector_scheduler.request_finished(
                request,
                block_ids[self._stage3_fa_group_index],
                mamba_block_ids=mamba_block_ids)  # type: ignore
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
        # `data` is inherited from vLLM's KVConnectorStats dataclass, which a
        # static checker cannot see through vllm's editable install.
        if data is not None:
            return TpuKVConnectorStats(
                data=data)  # type: ignore[unexpected-keyword]
        return TpuKVConnectorStats()

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
        """Registers the named cache materialization when the backend needs it.

        ZMQ reads ``runner.kv_caches`` positionally. HMA needs layer-to-group
        identity and Raiden explicit-pool admission needs the layer names and
        group specs to derive its canonical pool manifest.
        """
        if ((self.use_hma or self.use_raiden)
                and self.connector_worker is not None):
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

    def get_block_ids_with_load_errors_group_index(self) -> int | None:
        """Scopes Stage-3 load errors to the one transferred FA group."""
        if self.use_raiden and _use_raiden_stage3_transport():
            return self._stage3_fa_group_index
        return None

    def update_connector_output(self, connector_output: Any) -> None:
        """Retires scheduler-side Stage-3 dedup state on real completion."""
        scheduler = self.connector_scheduler
        if scheduler is not None and hasattr(scheduler,
                                             "update_connector_output"):
            scheduler.update_connector_output(connector_output)


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

    def _maybe_truncate_for_mamba(self, request: "Request") -> None:
        """P-side: drop the last prompt token so prefill ships the mamba
        state h(N-1); D's recompute of token N then reproduces h(N) instead
        of advancing the recurrence a second time."""
        if request.num_prompt_tokens <= 1:
            return
        params = request.kv_transfer_params
        if params is not None and params.get("_p_side_truncated"):
            return
        if request.prompt_token_ids is None:
            return
        request.prompt_token_ids.pop()
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

    def __init__(self, vllm_config: "VllmConfig"):
        super().__init__(vllm_config)
        # req_id -> (uuid, global block ids, exact token count, wire params).
        # This survives build_connector_meta() so a duplicate request_finished
        # callback cannot mint a conflicting UUID or enqueue a second D5
        # registration.
        self._stage3_finished_sends: OrderedDict[str, tuple[int, tuple[
            int, ...], int, dict[str, Any]]] = OrderedDict()
        self._stage3_fa_group_index = 0
        self._stage3_mamba_group_indices: list[int] = []
        self._stage3_mamba_num_speculative_blocks: int = 0
        # Get DP rank and TP size from config and stagger kv_port and side_channel_port
        dp_rank = vllm_config.parallel_config.data_parallel_rank if vllm_config.parallel_config else 0
        tp_size = vllm_config.parallel_config.tensor_parallel_size if vllm_config.parallel_config else 1
        port_base = dist_utils.get_kv_ports()
        if isinstance(port_base, list):
            self.kv_port = [int(p) + 2 * dp_rank * tp_size for p in port_base]
        else:
            self.kv_port = int(port_base) + 2 * dp_rank * tp_size
        logger.info("TPURaidenConnectorScheduler --> kv_ip=%s | kv_port=%s",
                    self.kv_ip, self.kv_port)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if self.is_producer:
            if (_use_raiden_stage3_transport()
                    and self._stage3_mamba_group_indices):
                self._maybe_truncate_for_mamba(request)
            return 0, False
        if not request.kv_transfer_params:
            return 0, False

        assert num_computed_tokens % self.block_size == 0
        if _use_raiden_stage3_transport():
            # Unlike the legacy equal-page pull, the controller planner has a
            # byte-precise partial-tail entry. The producer's scheduler-owned
            # num_computed_tokens is the transfer extent: in the deployed
            # proxy's one-token-prefill flow this can be prompt_tokens - 1.
            # Do not replace it with prompt length here.
            transfer_tokens = int(
                request.kv_transfer_params.get("num_tokens", 0))
            if transfer_tokens <= 0:
                raise ValueError(
                    "Stage-3 metadata requires a positive num_tokens")
            count = max(transfer_tokens - num_computed_tokens, 0)
            if count > 0 and dist_utils.get_raiden_inline_load():
                return count, False
            return count, count > 0

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

        if _use_raiden_stage3_transport():
            self._update_stage3_state_after_alloc(request, blocks,
                                                  num_external_tokens)
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
            # If num_external_tokens is 0, we don't need to pull any data through
            # this connector (either due to a full local cache hit, or because
            # another MultiConnector child owns the load). But the producer
            # (prefill node) doesn't know that: it has the prefill KV blocks
            # registered under this uuid and keeps them pinned until they are
            # pulled or p2p_wait_pull_timeout expires. So we still enqueue an
            # empty read (LoadMeta with no block ids), which the worker turns
            # into a "release now" signal to the producer instead of a pull.
            # TODO: Remove this branch once the producer releases
            # unpulled sends by another path (e.g. a direct scheduler-side
            # release, or the unified Raiden connector); otherwise its blocks
            # stay pinned until p2p_wait_pull_timeout.
            self.reqs_to_load[request.request_id] = LoadMeta(
                uuid=params["uuid"],
                local_block_ids=None,
                remote_block_ids=None,
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
                report_completion=False,
            )
            logger.info(
                "TPURaidenConnectorScheduler no-load release req_id=%s "
                "uuid=%s", request.request_id, params["uuid"])
        logger.info(
            "TPURaidenConnectorScheduler update_state_after_alloc --> "
            "reqs_to_load=%s", self.reqs_to_load)

    def _update_stage3_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Declares a destination transfer without source physical IDs."""
        if num_external_tokens <= 0:
            return
        params = request.kv_transfer_params
        assert params is not None
        forbidden = {"remote_block_ids", "remote_host", "remote_port"}
        leaked = sorted(forbidden.intersection(params))
        if leaked:
            raise ValueError(
                "Stage-3 cross-server metadata must not carry legacy "
                f"source routing/block fields: {leaked}")

        source_req_id = params.get("req_id")
        if not isinstance(source_req_id, str) or not source_req_id:
            raise ValueError(
                "Stage-3 metadata requires a non-empty source request ID")
        uuid = int(params.get("uuid", 0))
        num_tokens = int(params.get("num_tokens", 0))
        src_controller_address = str(params.get("src_controller_address",
                                                "")).strip()
        src_job_name = str(params.get("src_job_name", "")).strip()
        src_engine_id = str(params.get("src_engine_id", "")).strip()
        src_data_replica_idx = int(params.get("src_data_replica_idx", -1))
        src_parallelism = int(params.get("src_parallelism", 0))
        if uuid <= 0:
            raise ValueError("Stage-3 request UUID must be positive")
        if num_tokens <= 0:
            raise ValueError("Stage-3 transfer token count must be positive")
        if not src_controller_address:
            raise ValueError(
                "Stage-3 metadata requires an explicit source controller "
                "address")
        if not src_job_name or not src_engine_id:
            raise ValueError(
                "Stage-3 metadata requires explicit source job and engine "
                "identity")
        if src_data_replica_idx < 0:
            raise ValueError(
                "Stage-3 source data replica identity must be non-negative")
        if src_parallelism <= 0:
            raise ValueError(
                "Stage-3 source transfer parallelism must be positive")

        grouped_block_ids = blocks.get_block_ids()
        if self._stage3_fa_group_index >= len(grouped_block_ids):
            raise ValueError(
                "Stage-3 FA KV cache group is absent from decode allocation: "
                f"index={self._stage3_fa_group_index}, "
                f"groups={len(grouped_block_ids)}")
        local_block_ids = list(grouped_block_ids[self._stage3_fa_group_index])
        expected_pages = ((num_tokens + self.block_size - 1) //
                          self.block_size)
        if len(local_block_ids) != expected_pages:
            raise ValueError(
                "Stage-3 FA resharding requires the complete destination "
                "page set (prefix-suffix pulls are unsupported): "
                f"blocks={len(local_block_ids)}, expected={expected_pages}, "
                f"num_tokens={num_tokens}, page_tokens={self.block_size}")
        mamba_state_block_ids: Optional[list[int]] = None
        if self._stage3_mamba_group_indices:
            mamba_block_ids: list[list[int]] = []
            for mamba_gid in self._stage3_mamba_group_indices:
                if mamba_gid >= len(grouped_block_ids):
                    raise ValueError(
                        "GDN state reshard: mamba KV cache group is absent "
                        f"from decode allocation: index={mamba_gid}, "
                        f"groups={len(grouped_block_ids)}")
                mamba_block_ids.append(list(grouped_block_ids[mamba_gid]))
            mamba_state_block_ids = _select_committed_mamba_blocks(
                mamba_block_ids, self._stage3_mamba_num_speculative_blocks)
        self.reqs_to_load[request.request_id] = _Stage3LoadMeta(
            uuid=uuid,
            source_req_id=source_req_id,
            local_block_ids=local_block_ids,
            num_tokens=num_tokens,
            src_controller_address=src_controller_address,
            src_job_name=src_job_name,
            src_engine_id=src_engine_id,
            src_data_replica_idx=src_data_replica_idx,
            src_parallelism=src_parallelism,
            mamba_state_block_ids=mamba_state_block_ids,
        )
        logger.info(
            "TPURaidenConnectorScheduler Stage-3 load req_id=%s "
            "source_req_id=%s uuid=%d num_tokens=%d destination_pages=%d "
            "source_controller=%s",
            request.request_id,
            source_req_id,
            uuid,
            num_tokens,
            len(local_block_ids),
            src_controller_address,
        )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
        mamba_block_ids: Optional[list[list[int]]] = None,
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        if not _use_raiden_stage3_transport():
            return super().request_finished(request, block_ids)
        if not self.is_producer:
            return False, None
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            return False, None

        # Decode deliberately computes the final prompt token locally to
        # produce its first logits. Publish no more than the preceding prompt
        # prefix even when vLLM reports the producer's full prompt as computed.
        # The controller emits a short final chunk, so retain that partial
        # source page instead of inheriting the legacy full-block-only trim.
        computed_tokens = int(request.num_computed_tokens)
        prompt_tokens = int(request.num_prompt_tokens)
        finish_params = request.kv_transfer_params
        if isinstance(finish_params, dict) and bool(
                finish_params.get("_p_side_truncated")):
            # The producer already dropped one token; N-1 is already applied.
            num_tokens = min(computed_tokens, prompt_tokens)
        else:
            num_tokens = min(computed_tokens, prompt_tokens - 1)
        if num_tokens <= 0 or not block_ids:
            return False, {}
        parallel_config = self.vllm_config.parallel_config
        pcp_size = int(parallel_config.prefill_context_parallel_size or 1)
        interleave_size = int(
            getattr(parallel_config, "cp_kv_cache_interleave_size", 256)
            or 256)
        min_transfer_tokens = pcp_size * interleave_size
        if (num_tokens < min_transfer_tokens):
            return False, {}

        scheduler_block_tokens = self.block_size * pcp_size
        expected_scheduler_blocks = (
            (num_tokens + scheduler_block_tokens - 1) //
            scheduler_block_tokens)
        normalized_block_ids = tuple(int(block_id) for block_id in block_ids)
        # The producer may allocate trailing blocks for its excluded final
        # prompt token or locally generated tokens. They are outside the
        # transfer prefix; too few blocks remains an error.
        if len(normalized_block_ids) > expected_scheduler_blocks:
            normalized_block_ids = (
                normalized_block_ids[:expected_scheduler_blocks])
        if len(normalized_block_ids) != expected_scheduler_blocks:
            raise ValueError(
                "Stage-3 producer block IDs must cover every PCP scheduler "
                "block, including the partial tail: "
                f"blocks={len(block_ids)}, transfer_blocks="
                f"{len(normalized_block_ids)}, expected="
                f"{expected_scheduler_blocks}, num_tokens={num_tokens}, "
                f"page_tokens={self.block_size}, pcp_size={pcp_size}")

        producer_dp_rank = (self.vllm_config.parallel_config.data_parallel_rank
                            if self.vllm_config.parallel_config else 0)
        controller_address = _resolved_reshard_controller_address(
            producer_dp_rank)
        if not controller_address:
            raise ValueError("Stage-3 producer metadata requires "
                             "TPU_RAIDEN_CONTROLLER_ADDRESS")
        normalized_ids = normalized_block_ids
        src_job_name = str(tpu_envs.TPU_RAIDEN_JOB_NAME).strip() or "prefill"
        src_engine_id = str(tpu_envs.TPU_RAIDEN_ENGINE_ID).strip()
        src_parallelism = int(tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM)
        parallel_config = self.vllm_config.parallel_config
        src_data_replica_idx = int(parallel_config.data_parallel_rank or 0)
        if not src_engine_id:
            raise ValueError("TPU_RAIDEN_ENGINE_ID must not be empty")
        if src_parallelism <= 0:
            raise ValueError(
                "TPU_RAIDEN_TRANSFER_PARALLELISM must be positive")
        if src_parallelism != pcp_size:
            raise ValueError(
                "Stage-3 producer transfer parallelism must equal PCP size: "
                f"parallelism={src_parallelism}, pcp_size={pcp_size}")
        if src_data_replica_idx < 0:
            raise ValueError(
                "Producer data-parallel rank must be non-negative")
        existing = self._stage3_finished_sends.get(request.request_id)
        if existing is not None:
            uuid, old_ids, old_num_tokens, params = existing
            if old_ids != normalized_ids or old_num_tokens != num_tokens:
                raise ValueError(
                    "Conflicting duplicate Stage-3 request finish for "
                    f"req_id={request.request_id}")
            self._stage3_finished_sends.move_to_end(request.request_id)
            return True, dict(params)

        # Never evict an in-flight UUID merely to satisfy a memory bound: a
        # delayed duplicate callback could otherwise mint a conflicting UUID
        # while native/D5 state still exists. Scheduler completion removes
        # records through update_connector_output(). Refuse excess concurrency
        # rather than weakening idempotency.
        if len(self._stage3_finished_sends) >= _STAGE3_FINISH_DEDUP_LIMIT:
            raise RuntimeError(
                "Stage-3 in-flight finish dedup capacity exhausted: "
                f"limit={_STAGE3_FINISH_DEDUP_LIMIT}")

        mamba_state_block_ids: Optional[list[int]] = None
        if self._stage3_mamba_group_indices:
            if not mamba_block_ids:
                raise ValueError("GDN state is present but the producer's "
                                 "mamba block tables were not provided")
            mamba_state_block_ids = _select_committed_mamba_blocks(
                mamba_block_ids, self._stage3_mamba_num_speculative_blocks)
        uuid = get_uuid()
        now = time.perf_counter()
        expiration_time = (now + dist_utils.get_p2p_wait_pull_timeout())
        self.reqs_to_send[request.request_id] = SendMeta(
            uuid=uuid,
            local_block_ids=list(normalized_ids),
            expiration_time=expiration_time,
            num_tokens=num_tokens,
            mamba_state_block_ids=mamba_state_block_ids,
        )
        params: dict[str, Any] = {
            "req_id": request.request_id,
            "uuid": uuid,
            "num_tokens": num_tokens,
            "src_controller_address": controller_address,
            "src_job_name": src_job_name,
            "src_engine_id": src_engine_id,
            "src_data_replica_idx": src_data_replica_idx,
            "src_parallelism": src_parallelism,
        }
        self._stage3_finished_sends[request.request_id] = (
            uuid,
            normalized_ids,
            num_tokens,
            dict(params),
        )
        logger.info(
            "TPURaidenConnectorScheduler Stage-3 send req_id=%s uuid=%d "
            "num_tokens=%d source_pages=%d source_controller=%s",
            request.request_id,
            uuid,
            num_tokens,
            len(normalized_ids),
            controller_address,
        )
        return True, params

    def update_connector_output(self, connector_output: Any) -> None:
        for req_id in connector_output.finished_sending or ():
            self._stage3_finished_sends.pop(req_id, None)

    def get_finished_count(self) -> int:
        if not _use_raiden_stage3_transport():
            # Preserve the legacy connector's model-runner-world aggregation.
            return 0
        if not self.is_producer:
            # DP8 routes one request to one decode engine.
            return 1
        parallel_config = self.vllm_config.parallel_config
        return int(parallel_config.prefill_context_parallel_size or 1)


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
        self.dp_rank: int = vllm_config.parallel_config.data_parallel_rank if vllm_config.parallel_config else 0
        self.host_ip = dist_utils.get_host_ip()
        self.kv_transfer_port = int(dist_utils.get_kv_transfer_port()) + (
            2 * self.dp_rank * self.tp_size)
        self._raiden_transfer_engine: Optional["KVCacheManager"] = None
        self.named_kv_caches: Optional[dict[str, Any]] = None
        self._raiden_admission_summary: dict[str, Any] | None = None
        self._raiden_manifest: Any | None = None
        self._raiden_layout_fingerprint: str | None = None
        self._raiden_layout_fingerprint_payload: dict[str, Any] | None = None
        self._raiden_controller_facade: Any | None = None
        self._raiden_controller_address: str | None = None
        # In-engine store hosting: set only on the engine's rank-0 worker.
        self._reshard_store: Any | None = None
        self._raiden_work_unit: Any | None = None
        # Producer request registrations are retained until Raiden itself
        # reports done_sending. Consumer submissions are keyed by UUID so a
        # duplicate connector-metadata delivery cannot launch a second plan.
        self._stage3_registered_sends: dict[str, _Stage3RegisteredSend] = {}
        self._stage3_terminal_sends: dict[str, _Stage3RegisteredSend] = {}
        self._stage3_terminal_cleanup: dict[str, bool] = {}
        self._stage3_reported_sends: set[str] = set()
        self._stage3_submitted_loads: dict[str, int] = {}
        self._stage3_submitted_load_tokens: dict[str, int] = {}
        self._stage3_load_start_times: dict[str, float] = {}
        self._legacy_registered_sends: set[str] = set()
        self._legacy_submitted_loads: set[str] = set()
        # Consumer-side facades for the producers' controllers, keyed by
        # controller address: the consumer coordinates each load with the
        # SOURCE controller directly (it plans, arms this worker, and
        # dispatches its senders); the local controller is only the
        # admission/metadata directory.
        self._stage3_source_facades: dict[str, Any] = {}
        self._stage3_state_group_count = 0
        # GDN state-class sibling transfers: derived source req_id -> base
        # destination req_id; destination -> derived ids still pending; FA
        # completions parked until every state class lands.
        # Producer-side terminal outcomes, keyed by base request id. A
        # request appears here once its single multi-tag transfer reached a
        # native terminal (or was cancelled unclaimed) and is drained into
        # the scheduler report exactly once.
        self._stage3_send_outcomes: dict[str, str] = {}
        # vLLM independently randomizes the internal request ID on the
        # prefill and decode engines. D5 and the controller/native transfer
        # use the producer ID, while every scheduler-facing lifecycle table
        # must remain keyed by the decode-local ID.
        self._stage3_source_req_ids: dict[str, str] = {}
        self._stage3_destination_req_ids: dict[str, str] = {}
        self._stage3_controller_accepted: set[str] = set()
        # A scheduler-finished request can still have an armed H2D in flight.
        # Keep its load metadata pinned until the native manager reports a
        # terminal state; only then may the scheduler's delayed block free be
        # completed safely.
        self._stage3_terminal_loads: set[str] = set()
        self._stage3_finished_loads_pending_cleanup: set[str] = set()
        # A controller RPC can fail after the receiver has been armed and
        # sibling pushes have started. Such requests stay pending until the
        # native manager reports a terminal state, or until a full manager
        # timeout has elapsed after the RPC returned.
        self._stage3_pending_controller_failures: dict[str, float] = {}
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
        # so a failed recv can surface the affected blocks for recompute. A
        # scheduler-finished load is pruned only after its native terminal.
        self._load_block_ids: dict[str, list[int]] = {}
        # Block ids of failed loads, drained by get_block_ids_with_load_errors().
        self._failed_block_ids: set[int] = set()
        self.transfer_stats = TpuKVConnectorStats()
        logger.info(
            "TPURaidenConnectorWorker --> init | ip=%s | base_port=%s | "
            "is_producer=%s | node_id=%s | tp_rank=%d | tp_size=%d | dp_rank=%d",
            self.host_ip, self.kv_transfer_port, self.is_producer,
            self.node_id, self.tp_rank, self.tp_size, self.dp_rank)

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        if self.tp_rank == 0:
            if self.is_producer:
                prefill_queue_len = (len(self._stage3_registered_sends)
                                     if self._raiden_stage3_enabled() else len(
                                         self._legacy_registered_sends))
                self.transfer_stats.record_prefill_queue_length(
                    prefill_queue_len)
            else:
                decode_queue_len = (len(self._stage3_submitted_loads) -
                                    len(self._stage3_terminal_loads)
                                    if self._raiden_stage3_enabled() else len(
                                        self._legacy_submitted_loads))
                self.transfer_stats.record_decode_queue_length(
                    decode_queue_len)

        if not self.transfer_stats.is_empty():
            return self.transfer_stats.clone_and_reset()
        return None

    def register_runner(self, runner: TPUModelRunner) -> None:
        self.runner = runner
        if (self._raiden_stage3_enabled()
                and not self._raiden_qwen35_admission_enabled()):
            raise RuntimeError(
                "TPU_KV_RESHARD_TRANSPORT=raiden requires explicit Qwen3.5 "
                "pool admission (TPU_USE_RAIDEN_KV_CACHE_MANAGER=1 and "
                "TPU_RAIDEN_QWEN35_ADMISSION=1)")
        if self._raiden_qwen35_admission_enabled():
            self._admit_raiden_qwen35_kv_cache(runner)
            return
        self._ensure_raiden_transfer_engine()

    @staticmethod
    def _raiden_stage3_enabled() -> bool:
        return _use_raiden_stage3_transport()

    @staticmethod
    def _raiden_qwen35_admission_enabled() -> bool:
        return bool(tpu_envs.TPU_USE_RAIDEN_KV_CACHE_MANAGER
                    and tpu_envs.TPU_RAIDEN_QWEN35_ADMISSION)

    def _admit_raiden_qwen35_kv_cache(self, runner: TPUModelRunner) -> None:
        """Constructs the v1 Raiden engine over the explicit Qwen3.5 pools."""
        if self._raiden_transfer_engine is not None:
            return

        stage3_enabled = self._raiden_stage3_enabled()
        controller_address = ""
        if stage3_enabled:
            self._maybe_host_reshard_store()
            controller_address = _resolved_reshard_controller_address(
                self.dp_rank)
            if not controller_address:
                raise ValueError(
                    "TPU_RAIDEN_CONTROLLER_ADDRESS is required when "
                    "TPU_KV_RESHARD_TRANSPORT=raiden")

        from vllm_torchtpu.distributed.kv_transfer.v2 import \
            raiden_pool_manifest as rpm

        topology = self._raiden_qwen35_admission_topology()
        role = "kv_producer" if self.is_producer else "kv_consumer"
        named_kv_caches = self.named_kv_caches
        if not named_kv_caches:
            raise ValueError("Raiden pool admission requires "
                             "register_kv_caches() before register_runner()")
        kv_cache_groups = tuple(runner.kv_cache_config.kv_cache_groups or ())
        raw_tensors = tuple(runner.kv_cache_raw_tensors or ())

        mamba_group_ordinal_by_layer = None
        if stage3_enabled:
            group_ordinals: dict[str, int] = {}
            ordinal = 0
            for group in kv_cache_groups:
                if not isinstance(group.kv_cache_spec, MambaSpec):
                    continue
                for layer_name in group.layer_names:
                    group_ordinals[layer_name] = ordinal
                ordinal += 1
            if group_ordinals:
                mamba_group_ordinal_by_layer = group_ordinals
                self._stage3_state_group_count = ordinal
        manifest = rpm.build_qwen35_pool_manifest(
            named_kv_caches=named_kv_caches,
            kv_cache_groups=kv_cache_groups,
            raw_tensors=raw_tensors,
            mamba_group_ordinal_by_layer=mamba_group_ordinal_by_layer,
            gdn_geometry=rpm.GdnHeadGeometry(
                local_key_heads=self._local_gdn_head_count(
                    self._model_config_int("linear_num_key_heads", 0)),
                local_value_heads=self._local_gdn_head_count(
                    self._model_config_int("linear_num_value_heads", 0)),
                key_head_dim=self._model_config_int("linear_key_head_dim", 1),
                value_head_dim=self._model_config_int(
                    "linear_value_head_dim",
                    self._model_config_int("linear_key_head_dim", 1)),
            ),
        )
        # Hard-fail before manager construction if the pools point at storage
        # that the model kernels do not actually use.
        verified_storages = rpm.verify_storage_binding(
            manifest=manifest,
            named_kv_caches=named_kv_caches,
            raw_tensors=raw_tensors,
        )
        rpm.materialize_storages(manifest)

        storages = list(manifest.storages)
        engine = self._construct_raiden_transfer_engine(storages, num_slots=1)
        summary = dict(engine.register_pools(manifest.pool_dicts()))

        if stage3_enabled:
            registration = self._register_raiden_stage3_work_unit(
                engine=engine,
                manifest=manifest,
                controller_address=controller_address,
            )
            summary["stage3_registration"] = registration

        self._raiden_transfer_engine = engine
        self._raiden_manifest = manifest

        counts = manifest.tag_counts()
        geometry = manifest.geometry_by_tag()
        gdn_conv_count = sum(count for tag, count in counts.items()
                             if str(tag).startswith(rpm.TAG_GDN_CONV))
        gdn_ssm_count = sum(count for tag, count in counts.items()
                            if str(tag).startswith(rpm.TAG_GDN_SSM))
        summary_geometry = {tag: dict(geo) for tag, geo in geometry.items()}
        summary.update({
            "topology": topology,
            "model_server_role": role,
            "binding": manifest.binding,
            "tag_counts": dict(counts),
            "geometry": summary_geometry,
        })
        self._raiden_admission_summary = dict(summary)

        logger.info(
            "Raiden pool admission complete topology=%s role=%s binding=%s "
            "pools=%d storages=%d fa=%d gdn.conv=%d gdn.ssm=%d",
            topology,
            role,
            manifest.binding,
            len(manifest.pools),
            len(manifest.storages),
            counts.get(rpm.TAG_FA, 0),
            gdn_conv_count,
            gdn_ssm_count,
        )
        for tag, geo in geometry.items():
            logger.info(
                "Raiden pool geometry tag=%s num_blocks=%d "
                "block_stride_bytes=%d live_bytes_per_block=%d",
                tag,
                int(geo["num_blocks"]),
                int(geo["block_stride_bytes"]),
                int(geo["live_bytes_per_block"]),
            )
        logger.info(
            "Raiden pool binding verified: %d/%d pool storages matched "
            "typed KV cache storages",
            verified_storages,
            len(manifest.storages),
        )

    @staticmethod
    def _measure_raiden_fa_layout(
        manifest: Any, ) -> tuple[str, dict[str, Any]]:
        from vllm_torchtpu.distributed.kv_transfer.v2.raiden_layout_fingerprint import \
            measured_fa_layout_fingerprint

        return measured_fa_layout_fingerprint(manifest)

    @staticmethod
    def _new_raiden_controller_facade(controller_address: str) -> Any:
        # The reshard client surface is C++-owned by default —
        # identical signatures, return shapes, and exception text; the
        # Python facade remains as the env-gated rollback for one release.
        if str(tpu_envs.TPU_RAIDEN_CLIENT_IMPL).strip().lower() == "cpp":
            from tpu_raiden.api.torch.reshard_client import ReshardClient

            return ReshardClient(controller_address)
        from tpu_sync.rpc.raiden_controller import RaidenControllerClientFacade

        return RaidenControllerClientFacade(controller_address)

    @staticmethod
    def _new_raiden_id(fields: dict[str, Any]) -> Any:
        from tpu_sync.rpc.raiden_controller import RaidenId

        return RaidenId(**fields)

    @staticmethod
    def _new_raiden_manager(**kwargs: Any) -> "KVCacheManager":
        from tpu_raiden.api.torch.kv_cache_manager import KVCacheManager

        return KVCacheManager(**kwargs)

    def _raiden_transfer_parallelism(self) -> int:
        parallelism = int(tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM)
        if parallelism <= 0:
            raise ValueError(
                "TPU_RAIDEN_TRANSFER_PARALLELISM must be positive")
        return parallelism

    def _local_raiden_transfer_rank(self) -> int:
        if not self.is_producer:
            return 0
        from vllm_torchtpu.distributed.pcp import get_pcp_rank

        return int(get_pcp_rank())

    def _maybe_host_reshard_store(self) -> None:
        """In-engine hosting (zero sidecars): each engine's transfer-rank-0
        worker owns the in-process reshard store — the framed reshard
        service plus the dispatch controller its local workers register
        with. Peer ranks only register into it, with bounded retries."""
        if not _reshard_store_mode() or self._reshard_store is not None:
            return
        if self._local_raiden_transfer_rank() != 0:
            return
        from tpu_raiden.api.torch.kv_cache_store import \
            RaidenId as _StoreRaidenId
        from tpu_raiden.api.torch.reshard_store import ReshardStore

        default_job = "prefill" if self.is_producer else "decode"
        job_name = str(tpu_envs.TPU_RAIDEN_JOB_NAME).strip() or default_job
        self._reshard_store = ReshardStore(
            raiden_id=_StoreRaidenId(
                job_name=job_name,
                job_replica_id=str(tpu_envs.TPU_RAIDEN_ENGINE_ID).strip(),
                data_name="reshard_store",
                data_replica_idx=self.dp_rank,
            ),
            store_server_ip=_reshard_advertise_host(),
            raiden_controller_port=(
                int(tpu_envs.TPU_RAIDEN_STORE_DISPATCH_PORT_BASE) +
                int(self.dp_rank)),
            reshard_service_port=_reshard_service_port(self.dp_rank),
        )
        logger.info(
            "TPURaidenConnectorWorker rank%d --> hosting reshard store "
            "service=%s dispatch=%s", self.tp_rank,
            _reshard_service_address(self.dp_rank),
            _reshard_dispatch_address(self.dp_rank))

    def _raiden_interleave_tokens(self, page_tokens: int,
                                  transfer_parallelism: int) -> int:
        """Returns the kernel interleave driving byte-span lowering."""
        page_tokens = int(page_tokens)
        transfer_parallelism = int(transfer_parallelism)
        if page_tokens <= 0:
            raise ValueError("Raiden page_tokens must be positive")
        if transfer_parallelism <= 0:
            raise ValueError("Raiden transfer_parallelism must be positive")
        # A TP1 destination is logically contiguous even if an unrelated PCP
        # option remains present in its shared ParallelConfig.
        if not self.is_producer or transfer_parallelism == 1:
            return page_tokens
        parallel_config = self.vllm_config.parallel_config
        interleave_tokens = int(parallel_config.cp_kv_cache_interleave_size
                                or 0)
        if interleave_tokens <= 0:
            raise ValueError("Stage-3 PCP source requires a positive "
                             "cp_kv_cache_interleave_size")
        if page_tokens % interleave_tokens:
            raise ValueError(
                "Stage-3 PCP page geometry requires page_tokens divisible by "
                "cp_kv_cache_interleave_size: "
                f"page_tokens={page_tokens}, "
                f"cp_kv_cache_interleave_size={interleave_tokens}")
        return interleave_tokens

    def _raiden_work_unit_fields(self, transfer_rank: int) -> dict[str, Any]:
        default_job = "prefill" if self.is_producer else "decode"
        job_name = str(tpu_envs.TPU_RAIDEN_JOB_NAME).strip()
        engine_id = str(tpu_envs.TPU_RAIDEN_ENGINE_ID).strip()
        return stage3_fa_raiden_id_fields(
            job_name=job_name or default_job,
            engine_id=engine_id,
            dp_rank=self.dp_rank,
            transfer_rank=transfer_rank,
            is_producer=self.is_producer,
        )

    def _register_raiden_stage3_work_unit(
        self,
        *,
        engine: Any,
        manifest: Any,
        controller_address: str,
    ) -> dict[str, Any]:
        """Measure and register this worker with its cluster controller."""
        from vllm_torchtpu.distributed.kv_transfer.v2.raiden_layout_fingerprint import \
            fa_page_tokens

        data_address = str(getattr(engine, "transfer_address", "")).strip()
        listener_address = str(getattr(engine, "listener_address", "")).strip()
        if not data_address:
            raise RuntimeError(
                "Stage-3 Raiden manager did not advertise a data endpoint")
        if not listener_address:
            raise RuntimeError(
                "Stage-3 Raiden manager did not advertise a listener endpoint")

        fingerprint, fingerprint_payload = self._measure_raiden_fa_layout(
            manifest)
        page_tokens = fa_page_tokens(manifest)
        transfer_parallelism = self._raiden_transfer_parallelism()
        transfer_rank = self._local_raiden_transfer_rank()
        interleave_tokens = self._raiden_interleave_tokens(
            page_tokens, transfer_parallelism)
        if self.is_producer:
            pcp_size = int(
                self.vllm_config.parallel_config.prefill_context_parallel_size
                or 1)
            if transfer_parallelism != pcp_size:
                raise ValueError(
                    "Producer transfer parallelism must equal the complete "
                    "PCP rank count: "
                    f"parallelism={transfer_parallelism}, pcp_size={pcp_size}")
        if transfer_rank < 0 or transfer_rank >= transfer_parallelism:
            raise ValueError(
                "Raiden transfer rank is outside the admitted parallelism: "
                f"rank={transfer_rank}, parallelism={transfer_parallelism}")

        unit = self._new_raiden_id(
            self._raiden_work_unit_fields(transfer_rank))
        facade = self._new_raiden_controller_facade(controller_address)
        # Store mode: the store is brought up by this engine's rank-0
        # worker in parallel with the peer ranks' init — bounded retry
        # instead of the sidecar era's launcher-ordered readiness.
        registration_deadline = time.perf_counter() + (
            120.0 if _reshard_store_mode() else 0.0)
        while True:
            try:
                facade.register_work_unit(
                    unit=unit,
                    shards=[data_address],
                    control_plane_rpc_address=listener_address,
                    pool_manifest=manifest.pool_dicts(),
                    layout_fingerprint=fingerprint,
                    page_tokens=page_tokens,
                    transfer_parallelism=transfer_parallelism,
                    transfer_rank=transfer_rank,
                )
                break
            except Exception as exc:  # noqa: BLE001 — retry-or-reraise
                if time.perf_counter() >= registration_deadline:
                    raise
                logger.info(
                    "Raiden work-unit registration to %s not accepted yet "
                    "(%s); retrying", controller_address, exc)
                time.sleep(0.5)

        self._raiden_controller_facade = facade
        self._raiden_controller_address = controller_address
        self._raiden_work_unit = unit
        self._raiden_layout_fingerprint = fingerprint
        self._raiden_layout_fingerprint_payload = dict(fingerprint_payload)
        logger.info(
            "Raiden work unit registered controller=%s unit=%s "
            "data=%s listener=%s page_tokens=%d transfer_rank=%d/%d "
            "interleave_tokens=%d layout_fingerprint=%s",
            controller_address,
            unit,
            data_address,
            listener_address,
            page_tokens,
            transfer_rank,
            transfer_parallelism,
            interleave_tokens,
            fingerprint,
        )
        return {
            "controller_address": controller_address,
            "unit": dict(self._raiden_work_unit_fields(transfer_rank)),
            "shards": [data_address],
            "control_plane_rpc_address": listener_address,
            "layout_fingerprint": fingerprint,
            "layout_fingerprint_payload": dict(fingerprint_payload),
            "page_tokens": page_tokens,
            "interleave_tokens": interleave_tokens,
            "transfer_parallelism": transfer_parallelism,
            "transfer_rank": transfer_rank,
        }

    def raiden_admission_summary(self) -> dict[str, Any]:
        if self._raiden_admission_summary is None:
            return {"admitted": False}
        return dict(self._raiden_admission_summary)

    def _raiden_qwen35_admission_topology(self) -> str:
        parallel_config = self.vllm_config.parallel_config
        pcp_size = int(parallel_config.prefill_context_parallel_size or 1)
        tp_size = int(parallel_config.tensor_parallel_size or self.tp_size)
        dp_size = int(parallel_config.data_parallel_size or 1)
        if self.is_producer:
            if tp_size == 1 and pcp_size in (4, 8) and dp_size == 1:
                return f"pcp{pcp_size}_prefill"
            if tp_size == 1 and pcp_size == 1 and dp_size in (4, 8):
                return f"dp{dp_size}_prefill"
            raise ValueError(
                "Raiden Qwen3.5 admission topology pcp8_prefill, "
                "pcp4_prefill, dp8_prefill, or dp4_prefill requires "
                "kv_producer with tensor_parallel_size=1 and either "
                "prefill_context_parallel_size in (4, 8), "
                "data_parallel_size=1 or "
                "prefill_context_parallel_size=1, "
                "data_parallel_size in (4, 8); got "
                f"prefill_context_parallel_size={pcp_size}, "
                f"tensor_parallel_size={tp_size}, "
                f"data_parallel_size={dp_size}")
        if pcp_size != 1 or tp_size != 1 or dp_size not in (4, 8):
            raise ValueError(
                "Raiden Qwen3.5 admission topology dp8_decode or dp4_decode requires "
                "kv_consumer with prefill_context_parallel_size=1, "
                "tensor_parallel_size=1, data_parallel_size in (4, 8); got "
                f"prefill_context_parallel_size={pcp_size}, "
                f"tensor_parallel_size={tp_size}, "
                f"data_parallel_size={dp_size}")
        return f"dp{dp_size}_decode"

    def _model_config_int(self, name: str, default: int) -> int:
        model_config = self.vllm_config.model_config
        hf_config = model_config.hf_config
        if hf_config is None:
            hf_config = model_config
        text_config = getattr(hf_config, "text_config", hf_config)
        value = getattr(text_config, name, None)
        return int(default if value is None else value)

    def _local_head_count(self, total_heads: int) -> int:
        total_heads = int(total_heads)
        if total_heads <= 0:
            return 0
        parallel_config = self.vllm_config.parallel_config
        tp_size = int(parallel_config.tensor_parallel_size or self.tp_size)
        if total_heads < tp_size:
            return 1
        if total_heads % tp_size:
            raise ValueError(f"total_heads={total_heads} must be divisible by "
                             f"tp_size={tp_size}")
        return total_heads // tp_size

    def _local_gdn_head_count(self, total_heads: int) -> int:
        """Return the GDN head count stored by this TP/PCP worker.

        GDN cache specs are localized across PCP as well as TP.  Full-
        attention head geometry is only TP-local, so keep this adjustment
        specific to GDN admission.
        """
        tp_local_heads = self._local_head_count(total_heads)
        if tp_local_heads <= 0:
            return 0
        parallel_config = getattr(self.vllm_config, "parallel_config", None)
        pcp_size = int(
            getattr(parallel_config, "prefill_context_parallel_size", 1) or 1)
        if tp_local_heads % pcp_size:
            raise ValueError(
                f"TP-local GDN heads={tp_local_heads} must be divisible by "
                f"pcp_size={pcp_size}")
        return tp_local_heads // pcp_size

    def process_send_load(self,
                          metadata: TPUConnectorMetadata,
                          wait_for_completion: bool = False,
                          report_completion: bool = True) -> None:
        engine = self._ensure_raiden_transfer_engine()
        if self.is_producer:
            if self._raiden_stage3_enabled():
                self._register_stage3_request_blocks(metadata)
                return
            for req_id, req_meta in metadata.reqs_to_send.items():
                engine.register_read(req_id, req_meta.uuid,
                                     req_meta.local_block_ids)
                self._legacy_registered_sends.add(str(req_id))
                logger.debug(
                    "TPURaidenConnectorWorker rank%d --> registered send "
                    "req_id=%s uuid=%s blocks=%d", self.tp_rank, req_id,
                    req_meta.uuid, len(req_meta.local_block_ids))
            return

        if self._raiden_stage3_enabled():
            self._submit_stage3_loads(metadata, engine)
            submitted_loads = set(metadata.reqs_to_load)
            if wait_for_completion:
                self._wait_for_recving(submitted_loads)
                if not report_completion:
                    self._suppress_done_recving.update(submitted_loads)
            return

        submitted_loads: set[str] = set()
        for req_id, req_meta in metadata.reqs_to_load.items():
            endpoint = self._resolve_remote_endpoint(req_meta)
            if (req_meta.remote_block_ids is None
                    and req_meta.local_block_ids is None):
                engine.start_read(req_id, req_meta.uuid, endpoint, [], [])
                if not req_meta.report_completion:
                    # The request is not WAITING_FOR_REMOTE_KVS on this
                    # connector (another MultiConnector child may own its
                    # load); reporting this empty read as finished_recving
                    # trips the scheduler's is_finished assert or prematurely
                    # resumes a request whose offload load is still in flight.
                    self._suppress_done_recving.add(req_id)
                logger.debug(
                    "TPURaidenConnectorWorker rank%d --> released remote "
                    "send req_id=%s uuid=%s endpoint=%s report_completion=%s",
                    self.tp_rank, req_id, req_meta.uuid, endpoint,
                    req_meta.report_completion)
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
            self._legacy_submitted_loads.add(str(req_id))
            self._load_block_ids[req_id] = list(local_blocks)
        if wait_for_completion:
            self._wait_for_recving(submitted_loads)
            if not report_completion:
                self._suppress_done_recving.update(submitted_loads)

    def _require_stage3_controller(self) -> tuple[Any, str]:
        facade = self._raiden_controller_facade
        address = str(self._raiden_controller_address or "").strip()
        if facade is None or not address:
            raise RuntimeError(
                "Stage-3 request flow requires successful work-unit "
                "registration with an explicit destination controller")
        return facade, address

    def _register_stage3_request_blocks(
            self, metadata: TPUConnectorMetadata) -> None:
        """Registers only this PCP rank's interleaved physical source blocks."""
        facade, _ = self._require_stage3_controller()
        if self._raiden_work_unit is None:
            raise RuntimeError("Stage-3 producer work unit is not registered")
        parallelism = self._raiden_transfer_parallelism()
        transfer_rank = self._local_raiden_transfer_rank()
        if transfer_rank < 0 or transfer_rank >= parallelism:
            raise ValueError(
                "Stage-3 producer transfer rank is outside parallelism: "
                f"rank={transfer_rank}, parallelism={parallelism}")

        now = time.perf_counter()
        expired_terminal_req_ids = [
            req_id
            for req_id, registration in self._stage3_terminal_sends.items()
            if registration.expiration_time <= now
        ]
        for req_id in expired_terminal_req_ids:
            self._stage3_terminal_sends.pop(req_id, None)
            self._stage3_reported_sends.discard(req_id)
            self._stage3_prune_state_send_tracking(req_id)

        for req_id, req_meta in metadata.reqs_to_send.items():
            uuid = int(req_meta.uuid)
            num_tokens = int(req_meta.num_tokens or 0)
            expiration_time = float(req_meta.expiration_time)
            if num_tokens <= 0:
                raise ValueError(
                    "Stage-3 producer send metadata requires num_tokens")
            if expiration_time <= 0:
                raise ValueError(
                    "Stage-3 producer send metadata requires a positive "
                    "expiration_time")
            page_tokens = int(self.vllm_config.cache_config.block_size)
            interleave_tokens = self._raiden_interleave_tokens(
                page_tokens, parallelism)
            scheduler_block_tokens = page_tokens * parallelism
            expected_scheduler_blocks = (
                (num_tokens + scheduler_block_tokens - 1) //
                scheduler_block_tokens)
            scheduler_ids = tuple(
                int(block_id) for block_id in req_meta.local_block_ids)
            if len(scheduler_ids) != expected_scheduler_blocks:
                raise ValueError(
                    "Stage-3 producer scheduler block count is inconsistent "
                    f"with PCP geometry: blocks={len(scheduler_ids)}, "
                    f"expected={expected_scheduler_blocks}, "
                    f"num_tokens={num_tokens}, page_tokens={page_tokens}, "
                    f"interleave_tokens={interleave_tokens}, "
                    f"parallelism={parallelism}")
            # This rank's declared source map, lowered from the kernel's own
            # layout package to the byte-span IR: owned interleave slices
            # packed dense rank-major over this rank's blocks (a prefix of
            # the shared scheduler BlockTable ids), split at destination
            # page boundaries, scaled to bytes. The producer-side semantic
            # checks live in this construction and Raiden's byte-level
            # registration/planning validation.
            from vllm_torchtpu.distributed.kv_transfer.v2.pool_byte_spans import (  # noqa: E501
                lower_fa_spans, owned_token_ranges)
            owned_tokens = sum(end - start
                               for start, end in owned_token_ranges(
                                   num_tokens=num_tokens,
                                   transfer_rank=transfer_rank,
                                   parallelism=parallelism,
                                   interleave_tokens=interleave_tokens,
                               ))
            owned_blocks = (owned_tokens + page_tokens - 1) // page_tokens
            local_ids = scheduler_ids[:owned_blocks]
            fa_pool_spans = []
            token_bytes = self._stage3_fa_token_bytes(page_tokens)
            fa_registration = lower_fa_spans(
                num_tokens=num_tokens,
                transfer_rank=transfer_rank,
                parallelism=parallelism,
                interleave_tokens=interleave_tokens,
                page_tokens=page_tokens,
                token_bytes=token_bytes,
                block_ids=list(local_ids),
            )
            if fa_registration.spans:
                fa_pool_spans = [fa_registration]
            terminal = self._stage3_terminal_sends.get(req_id)
            if terminal is not None:
                if (terminal.uuid != uuid
                        or terminal.local_block_ids != local_ids
                        or terminal.num_tokens != num_tokens):
                    raise ValueError(
                        "Conflicting replay after terminal Stage-3 producer "
                        f"registration for req_id={req_id}")
                continue
            existing = self._stage3_registered_sends.get(req_id)
            if existing is not None:
                if (existing.uuid != uuid
                        or existing.local_block_ids != local_ids
                        or existing.num_tokens != num_tokens):
                    raise ValueError(
                        "Conflicting duplicate Stage-3 producer registration "
                        f"for req_id={req_id}")
                continue
            try:
                # T3.1 sibling collapse: one registration carries the FA
                # spans plus every GDN state class's spans under the SAME
                # req_id/uuid — one D5 row per rank, one claim lifecycle.
                # The controller filters registrations per requested tag at
                # plan build.
                pool_spans = list(fa_pool_spans)
                pool_spans.extend(
                    self._stage3_state_pool_spans(
                        getattr(req_meta, "mamba_state_block_ids", None),
                        transfer_rank, parallelism))
                facade.register_request_blocks(
                    req_id=req_id,
                    uuid=uuid,
                    unit=self._raiden_work_unit,
                    block_ids=list(local_ids),
                    pool_spans=pool_spans,
                )
            except (RuntimeError, ValueError) as exc:
                if _STAGE3_REGISTRATION_CANCELLED_ERROR not in str(exc):
                    raise
                # Another rank won the expiry race and atomically sealed this
                # request generation before this slow rank reached D5. There
                # is no local registration to complete or release. Retain a
                # fresh local tombstone so metadata replay is idempotent, and
                # contribute this rank's terminal vote to PCP aggregation.
                tombstone_deadline = (
                    time.perf_counter() +
                    float(dist_utils.get_p2p_wait_pull_timeout()))
                self._stage3_terminal_sends[req_id] = _Stage3RegisteredSend(
                    uuid=uuid,
                    local_block_ids=local_ids,
                    num_tokens=num_tokens,
                    expiration_time=tombstone_deadline,
                )
                self._done_sending.add(req_id)
                logger.warning(
                    "TPURaidenConnectorWorker rank%d --> Stage-3 request "
                    "registration was already cancelled req_id=%s uuid=%d",
                    self.tp_rank,
                    req_id,
                    uuid,
                )
                continue
            self._stage3_registered_sends[req_id] = _Stage3RegisteredSend(
                uuid=uuid,
                local_block_ids=local_ids,
                num_tokens=num_tokens,
                expiration_time=expiration_time,
            )
            if not local_ids:
                # This PCP rank owns no physical page for the request. It has
                # nevertheless published an empty D5 entry so controller
                # lookup sees the complete source unit set, and is terminal
                # immediately: there is no native transfer to wait for.
                self._stage3_send_outcomes.setdefault(req_id, "done")
            logger.info(
                "%s",
                json.dumps(
                    {
                        "event": ("raiden_stage3_request_blocks_registered"),
                        "req_id":
                        req_id,
                        "uuid":
                        uuid,
                        "num_tokens":
                        num_tokens,
                        "transfer_rank":
                        transfer_rank,
                        "parallelism":
                        parallelism,
                        "page_tokens":
                        page_tokens,
                        "interleave_tokens":
                        interleave_tokens,
                        "scheduler_blocks":
                        expected_scheduler_blocks,
                        "owned_blocks":
                        len(local_ids),
                        "owned_tokens":
                        owned_tokens,
                        "declared_spans":
                        (len(fa_pool_spans[0].spans) if fa_pool_spans else 0),
                    },
                    sort_keys=True,
                ),
            )
            logger.debug(
                "TPURaidenConnectorWorker rank%d --> registered Stage-3 "
                "send req_id=%s uuid=%d transfer_rank=%d/%d blocks=%s",
                self.tp_rank,
                req_id,
                uuid,
                transfer_rank,
                parallelism,
                list(local_ids),
            )

    @staticmethod
    def _raiden_hbm_memory_type() -> Any:
        from tpu_sync.rpc.raiden_controller import RaidenMemoryType

        return RaidenMemoryType.HBM

    def _stage3_source_work_units(self,
                                  req_meta: _Stage3LoadMeta) -> list[Any]:
        """Enumerates the canonical complete PCP producer unit set."""
        parallelism = int(req_meta.src_parallelism)
        if parallelism != self._raiden_transfer_parallelism():
            raise ValueError(
                "Stage-3 source parallelism does not match destination "
                f"configuration: source={parallelism}, destination="
                f"{self._raiden_transfer_parallelism()}")
        return [
            self._new_raiden_id(
                stage3_fa_raiden_id_fields(
                    job_name=req_meta.src_job_name,
                    engine_id=req_meta.src_engine_id,
                    dp_rank=req_meta.src_data_replica_idx,
                    transfer_rank=rank,
                    is_producer=True,
                )) for rank in range(parallelism)
        ]

    def _record_stage3_load_failure(self, req_id: str,
                                    block_ids: list[int]) -> None:
        self._failed_recving.add(req_id)
        self._failed_block_ids.update(block_ids)
        self._stage3_terminal_loads.add(req_id)

    def _stage3_fa_token_bytes(self, page_tokens: int) -> int:
        """Bytes per token of the admitted FA pools, measured from the
        manifest (live bytes per block over the page geometry)."""
        manifest = self._raiden_manifest
        if manifest is None or not hasattr(manifest, "geometry_by_tag"):
            raise RuntimeError(
                "Stage-3 byte-span lowering requires the admitted pool "
                "manifest")
        geometry = manifest.geometry_by_tag().get("fa")
        if not geometry:
            raise RuntimeError("Stage-3 admitted manifest contains no FA "
                               "pools")
        live_bytes = int(geometry["live_bytes_per_block"])
        if page_tokens <= 0 or live_bytes <= 0 or live_bytes % page_tokens:
            raise RuntimeError(
                "Stage-3 FA pool live bytes are not token-aligned: "
                f"live={live_bytes}, page_tokens={page_tokens}")
        return live_bytes // page_tokens

    def _stage3_state_live_bytes(self, tag: str) -> int:
        """Return the admitted whole-slot byte size for an exact state tag."""
        manifest = self._raiden_manifest
        if manifest is None or not hasattr(manifest, "geometry_by_tag"):
            raise RuntimeError(
                "GDN state reshard requires the admitted pool manifest")
        geometry = manifest.geometry_by_tag().get(tag)
        if not geometry:
            raise RuntimeError(
                f"GDN state tag {tag!r} is absent from the pool manifest")
        live_bytes = int(geometry["live_bytes_per_block"])
        if live_bytes <= 0:
            raise RuntimeError(
                f"GDN state tag {tag!r} has invalid live bytes {live_bytes}")
        return live_bytes

    def _stage3_state_regions(self, tag: str) -> tuple[Any, ...]:
        """Return one uniform admitted live-region map for a state tag."""
        manifest = self._raiden_manifest
        pools = tuple(getattr(manifest, "pools", ()) or ())
        matching = tuple(pool for pool in pools
                         if str(getattr(pool, "tag", "")) == tag)
        if not matching:
            raise RuntimeError(
                f"GDN state tag {tag!r} is absent from the pool manifest")

        def signature(pool: Any) -> tuple[tuple[Any, ...], ...]:
            return tuple((str(region.name), int(region.offset_bytes),
                          int(region.stride_bytes), int(region.unit_bytes),
                          int(region.num_units), int(region.units_per_stride))
                         for region in pool.regions)

        expected = signature(matching[0])
        if any(signature(pool) != expected for pool in matching[1:]):
            raise RuntimeError(
                f"GDN state pools tagged {tag!r} disagree on live regions")
        return tuple(matching[0].regions)

    def _stage3_state_pool_spans(self, mamba_state_block_ids,
                                 transfer_rank: int, parallelism: int) -> list:
        """Producer: the GDN state classes' span registrations.

        PCP GDN execution exchanges token shards for head shards.  Every PCP
        rank therefore owns a distinct q/k/v and SSM head slice.  Each rank
        declares its slice in compact-live byte space, and the controller
        combines the complete rank set into the full non-PCP destination
        state. T3.1: the registrations ride the base request's single
        register_request_blocks call (same req_id/uuid as FA) instead of
        derived sibling registrations.
        """
        if not self._stage3_state_group_count:
            return []
        if not mamba_state_block_ids:
            raise RuntimeError(
                "GDN state registration requires the producer's per-group "
                "mamba state block ids")
        if len(mamba_state_block_ids) != self._stage3_state_group_count:
            raise RuntimeError(
                "GDN state registration group count disagrees with the "
                f"manifest: blocks={len(mamba_state_block_ids)}, "
                f"groups={self._stage3_state_group_count}")
        from vllm_torchtpu.distributed.kv_transfer.v2.pool_byte_spans import \
            lower_gdn_state_shard_spans
        from vllm_torchtpu.distributed.kv_transfer.v2.raiden_layout_fingerprint import (
            EXPECTED_FA_MINOR_TO_MAJOR, EXPECTED_FA_TILES)
        from vllm_torchtpu.distributed.kv_transfer.v2.raiden_pool_manifest import \
            BINDING_ALIASED_RAW

        aliased_raw = self._raiden_manifest.binding == BINDING_ALIASED_RAW
        if aliased_raw:
            # Raw transfers move physical bytes; admit only the fingerprinted
            # tiled layout.  The whole-token QK pair-blocked conv layout is
            # enforced downstream: the span lowering rejects any other conv
            # region vocabulary, and the layout version rides the compared
            # fingerprint payload.
            payload = self._raiden_layout_fingerprint_payload
            if not payload:
                raise RuntimeError(
                    "aliased raw GDN lowering requires an admitted physical "
                    "layout fingerprint")
            measured_minor_to_major = tuple(
                int(value) for value in payload.get("minor_to_major", ()))
            measured_tiles = tuple(
                tuple(int(value) for value in tile)
                for tile in payload.get("tiles", ()))
            measured_element_bits = int(payload.get("element_size_in_bits", 0))
            if (measured_minor_to_major != EXPECTED_FA_MINOR_TO_MAJOR
                    or measured_tiles != EXPECTED_FA_TILES
                    or measured_element_bits != 8):
                raise RuntimeError(
                    "aliased raw GDN lowering is unsupported for the "
                    "admitted physical layout")

        registrations = []
        for tag in _STAGE3_STATE_CLASS_TAGS:
            for ordinal, block_id in enumerate(mamba_state_block_ids):
                exact_tag = f"{tag}.g{ordinal}"
                if aliased_raw:
                    matching_pools = tuple(
                        pool for pool in self._raiden_manifest.pools
                        if str(getattr(pool, "tag", "")) == exact_tag)
                    if any(
                            int(pool.base_offset_bytes) %
                            1024 or int(pool.block_stride_bytes) % 1024
                            for pool in matching_pools):
                        raise RuntimeError(
                            "aliased raw GDN pool base and block stride must "
                            f"be physical-token aligned ({exact_tag})")
                registration = lower_gdn_state_shard_spans(
                    tag=exact_tag,
                    block_id=int(block_id),
                    transfer_rank=transfer_rank,
                    parallelism=parallelism,
                    regions=self._stage3_state_regions(exact_tag),
                )
                admitted_live_bytes = self._stage3_state_live_bytes(exact_tag)
                if registration.declared_bytes != admitted_live_bytes:
                    raise RuntimeError(
                        "GDN state byte lowering disagrees with the admitted "
                        f"manifest for {exact_tag}: lowered="
                        f"{registration.declared_bytes}, admitted="
                        f"{admitted_live_bytes}")
                registrations.append(registration)
        return registrations

    def _stage3_prune_state_send_tracking(self, base_req_id: str) -> None:
        """Drops producer terminal tracking with the base tombstone."""
        self._stage3_send_outcomes.pop(base_req_id, None)

    def _stage3_record_producer_send_terminals(self, done_sending,
                                               failed_sending) -> None:
        """Records native terminals (T3.1: one uuid, base req ids only)."""
        for req_id, failed in [
            (str(req_id), False) for req_id in done_sending
        ] + [(str(req_id), True) for req_id in failed_sending]:
            if req_id in self._stage3_registered_sends:
                if failed:
                    self._stage3_send_outcomes[req_id] = "failed"
                else:
                    self._stage3_send_outcomes.setdefault(req_id, "done")
                continue
            # Replayed native terminals after the base moved to its tombstone
            # are harmless and must not become duplicate scheduler reports.
            if (req_id not in self._stage3_terminal_sends
                    and req_id not in self._stage3_reported_sends):
                logger.warning(
                    "Ignoring unmapped Stage-3 producer terminal req_id=%s",
                    req_id)

    def _stage3_ready_producer_send_terminals(
            self) -> tuple[set[str], set[str], set[str]]:
        """Returns terminal base requests (T3.1: no sibling aggregation)."""
        outcomes, self._stage3_send_outcomes = self._stage3_send_outcomes, {}
        done = {r for r, outcome in outcomes.items() if outcome == "done"}
        failed = {r for r, outcome in outcomes.items() if outcome == "failed"}
        cancelled = {
            r
            for r, outcome in outcomes.items() if outcome == "cancelled"
        }
        return done, failed, cancelled

    @staticmethod
    def _start_stage3_transfer_with_d5_retry(facade: Any,
                                             **kwargs: Any) -> bool:
        """Bridges the one-step P scheduler-to-worker registration window.

        vLLM returns producer transfer parameters from update_from_output(),
        while the D5 block registrations reach producer workers in the next
        scheduler step. The source controller fails before receiver arming
        when those registrations are incomplete, so retry only that precise,
        side-effect-free error for a short bounded interval.
        """
        configured_timeout = float(dist_utils.get_p2p_wait_pull_timeout())
        deadline = time.perf_counter() + min(max(configured_timeout, 0.0),
                                             _STAGE3_D5_REGISTRATION_WAIT_S)
        attempts = 0
        while True:
            attempts += 1
            try:
                return facade.start_transfer(**kwargs)
            except RuntimeError as exc:
                if ("Missing producer block registration" not in str(exc)
                        or time.perf_counter() >= deadline):
                    raise
                if attempts == 1:
                    logger.info(
                        "Stage-3 waiting for producer D5 registrations "
                        "req_id=%s uuid=%s", kwargs.get("req_id"),
                        kwargs.get("uuid"))
                time.sleep(0.01)

    def _bind_stage3_request_ids(self, destination_req_id: str,
                                 source_req_id: str) -> None:
        """Binds the native/controller identity to the local scheduler ID."""
        existing_source = self._stage3_source_req_ids.get(destination_req_id)
        if existing_source is not None and existing_source != source_req_id:
            raise ValueError(
                "Conflicting Stage-3 source request ID for destination "
                f"request {destination_req_id!r}: existing="
                f"{existing_source!r}, new={source_req_id!r}")
        existing_destination = self._stage3_destination_req_ids.get(
            source_req_id)
        if (existing_destination is not None
                and existing_destination != destination_req_id):
            raise ValueError(
                "Stage-3 source request ID is already bound to a different "
                f"destination request: source={source_req_id!r}, existing="
                f"{existing_destination!r}, new={destination_req_id!r}")
        self._stage3_source_req_ids[destination_req_id] = source_req_id
        self._stage3_destination_req_ids[source_req_id] = destination_req_id

    def _stage3_source_request_id(self, destination_req_id: str) -> str:
        source_req_id = self._stage3_source_req_ids.get(destination_req_id)
        if source_req_id is None:
            raise RuntimeError(
                "Stage-3 destination request has no source identity binding: "
                f"destination={destination_req_id!r}")
        return source_req_id

    def _stage3_destination_request_id(self,
                                       source_req_id: str) -> Optional[str]:
        # Native completion IDs are controller plan IDs. Never treat an
        # unknown ID as a decode-local ID: it may be a stale terminal record
        # that happens to collide with an active destination request.
        return self._stage3_destination_req_ids.get(source_req_id)

    def _submit_stage3_loads(self, metadata: TPUConnectorMetadata,
                             engine: "KVCacheManager") -> None:
        """Starts exactly one source-controller transfer per request."""
        del engine  # Completion is observed through the manager in poll_stats.
        _, dst_controller_address = self._require_stage3_controller()
        if self._raiden_work_unit is None:
            raise RuntimeError(
                "Stage-3 destination work unit is not registered")

        for destination_req_id, req_meta in metadata.reqs_to_load.items():
            if not isinstance(req_meta, _Stage3LoadMeta):
                raise TypeError(
                    "Stage-3 consumer requires controller load metadata, got "
                    f"{type(req_meta).__name__}")
            uuid = int(req_meta.uuid)
            num_tokens = int(req_meta.num_tokens)
            source_req_id = req_meta.source_req_id
            self._bind_stage3_request_ids(destination_req_id, source_req_id)
            existing_uuid = self._stage3_submitted_loads.get(
                destination_req_id)
            if existing_uuid is not None:
                if (existing_uuid != uuid
                        or self._stage3_submitted_load_tokens.get(
                            destination_req_id) != num_tokens):
                    raise ValueError(
                        "Conflicting duplicate Stage-3 load submission for "
                        f"req_id={destination_req_id}")
                continue
            self._stage3_submitted_loads[destination_req_id] = uuid
            self._stage3_submitted_load_tokens[destination_req_id] = num_tokens
            self._stage3_load_start_times[
                destination_req_id] = time.perf_counter()
            local_blocks = list(req_meta.local_block_ids)
            self._load_block_ids[destination_req_id] = local_blocks
            controller_contacted = False
            fa_accepted = False
            submit_ms = None
            try:
                src_units = self._stage3_source_work_units(req_meta)
                src_controller_address = str(
                    req_meta.src_controller_address).strip()
                if not src_controller_address:
                    raise ValueError("empty Stage-3 source controller address")
                source_facade = self._stage3_source_facades.get(
                    src_controller_address)
                if source_facade is None:
                    source_facade = self._new_raiden_controller_facade(
                        src_controller_address)
                    self._stage3_source_facades[src_controller_address] = (
                        source_facade)
                # The facade RPC returns only after the source controller
                # server has awaited its RaidenFuture (receiver arm + sender
                # dispatch). Actual byte completion is deliberately *not*
                # inferred from that acknowledgement; get_finished polls the
                # local manager.
                controller_contacted = True
                # T3.1 sibling collapse: ONE transfer carries the FA
                # payload plus every GDN state class. Tag order fixes the
                # H2D order ranks executor-side (FA = group 0 uploads
                # first; state classes land after it on aliased arena
                # pages, replacing the old deferred-sibling submission).
                transfer_tags = list(_STAGE3_TRANSFER_POOL_TAGS)
                dst_blocks = list(local_blocks)
                dst_counts = [len(local_blocks)]
                mamba_state_block_ids = getattr(req_meta,
                                                "mamba_state_block_ids", None)
                if self._stage3_state_group_count:
                    if not mamba_state_block_ids or len(
                            mamba_state_block_ids
                    ) != self._stage3_state_group_count:
                        raise RuntimeError(
                            "GDN state load requires one destination state "
                            "block per mamba group")
                    for tag in _STAGE3_STATE_CLASS_TAGS:
                        for ordinal, slot in enumerate(mamba_state_block_ids):
                            transfer_tags.append(f"{tag}.g{ordinal}")
                            dst_blocks.append(int(slot))
                            dst_counts.append(1)
                start_submit = time.perf_counter()
                accepted = self._start_stage3_transfer_with_d5_retry(
                    source_facade,
                    src_units=src_units,
                    dst_units=[self._raiden_work_unit],
                    # D5 is keyed by the producer's internal request ID. The
                    # destination's independently randomized ID is strictly a
                    # local scheduler lifecycle key.
                    req_id=source_req_id,
                    dst_device_block_ids=dst_blocks,
                    dst_mem_type=self._raiden_hbm_memory_type(),
                    use_block_chunks=True,
                    src_controller_address=src_controller_address,
                    dst_controller_address=dst_controller_address,
                    uuid=uuid,
                    is_sender=True,
                    num_tokens=num_tokens,
                    transfer_pool_tags=transfer_tags,
                    dst_block_counts=dst_counts,
                )
                submit_ms = (time.perf_counter() - start_submit) * 1000
                if accepted is not True:
                    raise RuntimeError(
                        "source controller rejected Stage-3 transfer")
                fa_accepted = True
            except Exception as exc:  # pylint: disable=broad-except
                missing_d5 = "Missing producer block registration" in str(exc)
                if not controller_contacted or (missing_d5
                                                and not fa_accepted):
                    # FA D5 lookup and local validation fail before receiver
                    # arming, so no late H2D is possible.
                    self._record_stage3_load_failure(destination_req_id,
                                                     local_blocks)
                    logger.error(
                        "Stage-3 controller transfer failed before receiver "
                        "arming req_id=%s destination_req_id=%s uuid=%d: %s",
                        source_req_id, destination_req_id, uuid, exc)
                else:
                    # Generic RPC rejection can occur after receiver arming.
                    # Keep the request blocked and accept native terminal
                    # records; only surface recompute after manager failure or
                    # a full post-RPC manager timeout.
                    self._stage3_controller_accepted.add(destination_req_id)
                    deadline = (time.perf_counter() +
                                float(dist_utils.get_p2p_wait_pull_timeout()))
                    self._stage3_pending_controller_failures[
                        destination_req_id] = deadline
                    logger.error(
                        "Stage-3 controller transfer outcome uncertain; "
                        "waiting for native terminal state req_id=%s "
                        "destination_req_id=%s uuid=%d: %s", source_req_id,
                        destination_req_id, uuid, exc)
                continue
            self._stage3_controller_accepted.add(destination_req_id)
            logger.info(
                "%s",
                json.dumps(
                    {
                        "event": "raiden_stage3_transfer_submitted",
                        "req_id": source_req_id,
                        "destination_req_id": destination_req_id,
                        "uuid": uuid,
                        "num_tokens": num_tokens,
                        "recv_armed_before_push": True,
                        "state_group_count": self._stage3_state_group_count,
                        "destination_pages": len(local_blocks),
                        "controller_submit_ms": submit_ms,
                    },
                    sort_keys=True,
                ),
            )
            logger.info(
                "Raiden Stage 3 reshard: req_id=%s "
                "destination_req_id=%s "
                "recv_armed_before_push=1 state_groups=%d",
                source_req_id,
                destination_req_id,
                self._stage3_state_group_count,
            )
            logger.debug(
                "TPURaidenConnectorWorker rank%d --> submitted Stage-3 load "
                "req_id=%s destination_req_id=%s uuid=%d src_units=%d "
                "destination_pages=%d num_tokens=%d",
                self.tp_rank,
                source_req_id,
                destination_req_id,
                uuid,
                len(src_units),
                len(local_blocks),
                int(req_meta.num_tokens),
            )

    def get_finished(
            self,
            finished_req_ids: set[str] | None = None
    ) -> tuple[set[str], set[str]]:
        engine = self._ensure_raiden_transfer_engine()
        self._poll_finished(engine)
        done_sending = self._done_sending
        if self.is_producer and self._raiden_stage3_enabled():
            done_sending = done_sending - self._stage3_reported_sends
            self._stage3_reported_sends.update(done_sending)
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
        # A request aborted in WAITING_FOR_REMOTE_KVS is scheduler-finished,
        # but vLLM deliberately delays freeing its blocks until this connector
        # reports a receive terminal. Retain accepted Stage-3 state across that
        # abort so a later manager success/failure is not filtered as stale.
        cleanup_req_ids: set[str] = set()
        if finished_req_ids:
            for req_id in finished_req_ids:
                if (req_id in self._stage3_submitted_loads
                        and req_id not in self._stage3_terminal_loads):
                    self._stage3_finished_loads_pending_cleanup.add(req_id)
                else:
                    cleanup_req_ids.add(req_id)
        cleanup_req_ids.update(
            self._stage3_finished_loads_pending_cleanup.intersection(
                self._stage3_terminal_loads))
        for req_id in cleanup_req_ids:
            source_req_id = self._stage3_source_req_ids.pop(req_id, None)
            if (source_req_id is not None
                    and self._stage3_destination_req_ids.get(source_req_id)
                    == req_id):
                self._stage3_destination_req_ids.pop(source_req_id, None)
            self._reported_recving.discard(req_id)
            self._failed_recving.discard(req_id)
            self._suppress_done_recving.discard(req_id)
            self._load_block_ids.pop(req_id, None)
            self._stage3_submitted_loads.pop(req_id, None)
            self._stage3_submitted_load_tokens.pop(req_id, None)
            self._stage3_controller_accepted.discard(req_id)
            self._stage3_pending_controller_failures.pop(req_id, None)
            self._stage3_terminal_loads.discard(req_id)
            self._stage3_finished_loads_pending_cleanup.discard(req_id)
        self._done_sending = set()
        self._done_recving = set()
        return done_sending, done_recving

    def _poll_finished(self, engine: "KVCacheManager") -> None:
        done_sending, done_recving, failed_recving = engine.poll_stats()
        sender_failures: set[str] = set()
        cancelled_sends: set[str] = set()
        if self.is_producer and not self._raiden_stage3_enabled():
            for req_id in done_sending:
                self._legacy_registered_sends.discard(str(req_id))
            for req_id in failed_recving:
                self._legacy_registered_sends.discard(str(req_id))
        elif not self.is_producer and not self._raiden_stage3_enabled():
            for req_id in done_recving:
                self._legacy_submitted_loads.discard(str(req_id))
            for req_id in failed_recving:
                self._legacy_submitted_loads.discard(str(req_id))
        if self.is_producer and self._raiden_stage3_enabled():
            # The native manager's third tuple is named failed_recving for
            # historical pull semantics, but ReshardPush sender failures are
            # reported there too. Native completion is per transfer plan, so
            # record the connector-private state siblings and collapse them
            # with FA onto the one base ID known to vLLM.
            self._stage3_record_producer_send_terminals(
                done_sending, failed_recving)
            failed_recving = []
        elif self._raiden_stage3_enabled():
            # Stage-3 success requires both halves: the synchronous facade RPC
            # has observed the controller's RaidenFuture, and the local manager
            # has observed byte completion. Ignore stale/unrelated manager
            # records that have no accepted controller submission.
            # Reshard plans carry the producer ID so D5 lookup and all four
            # role logs share one correlation key. Translate native terminal
            # records back to decode-local IDs before touching scheduler
            # state. (T3.1: one transfer per request — no derived sibling
            # IDs exist.)
            native_done_recving = list(done_recving)
            native_failed_recving = list(failed_recving)
            done_recving = [
                destination_req_id for req_id in native_done_recving
                if (destination_req_id := self._stage3_destination_request_id(
                    req_id)) is not None
            ]
            failed_recving = [
                destination_req_id for req_id in native_failed_recving
                if (destination_req_id := self._stage3_destination_request_id(
                    req_id)) is not None
            ]
            unmapped_terminal_ids = (
                set(native_done_recving) | set(native_failed_recving)
            ) - self._stage3_destination_req_ids.keys()
            if unmapped_terminal_ids:
                logger.warning(
                    "Ignoring unmapped Stage-3 native terminal request IDs: "
                    "%s", sorted(unmapped_terminal_ids))
            done_recving = [
                req_id for req_id in done_recving
                if req_id in self._stage3_controller_accepted
            ]
            failed_recving = [
                req_id for req_id in failed_recving
                if req_id in self._stage3_controller_accepted
            ]
            terminal_recvs = set(done_recving) | set(failed_recving)
            for req_id in terminal_recvs:
                self._stage3_pending_controller_failures.pop(req_id, None)
            now = time.perf_counter()
            expired_uncertain = {
                req_id
                for req_id, deadline in
                self._stage3_pending_controller_failures.items()
                if deadline <= now
            }
            if expired_uncertain:
                failed_recving = list(set(failed_recving) | expired_uncertain)
                for req_id in expired_uncertain:
                    self._stage3_pending_controller_failures.pop(req_id, None)
            self._stage3_terminal_loads.update(done_recving)
            self._stage3_terminal_loads.update(failed_recving)
        if self.is_producer and self._raiden_stage3_enabled():
            facade, _ = self._require_stage3_controller()
            now = time.perf_counter()
            expired_candidates = {
                req_id
                for req_id, registration in
                self._stage3_registered_sends.items()
                if registration.expiration_time <= now
                and req_id not in self._stage3_terminal_cleanup
                and req_id not in self._stage3_send_outcomes
            }
            if expired_candidates:
                for req_id in expired_candidates:
                    registration = self._stage3_registered_sends[req_id]
                    if facade.cancel_request_blocks_if_unclaimed(
                            req_id=req_id, uuid=registration.uuid):
                        # T3.1: one registration per request — the base
                        # cancellation is the whole cancellation.
                        self._stage3_send_outcomes[req_id] = "cancelled"
            done_sending, sender_failures, cancelled_sends = (
                self._stage3_ready_producer_send_terminals())
            if cancelled_sends:
                logger.warning(
                    "TPURaidenConnectorWorker rank%d --> safely cancelled "
                    "unclaimed Stage-3 sends=%s", self.tp_rank,
                    sorted(cancelled_sends))
        native_terminal_sends = set(done_sending) | sender_failures
        terminal_sends = native_terminal_sends | cancelled_sends
        self._done_sending.update(terminal_sends)
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
        if sender_failures:
            logger.error(
                "TPURaidenConnectorWorker rank%d --> failed_sending=%s",
                self.tp_rank, sender_failures)
        if self.is_producer and self._raiden_stage3_enabled():
            transfer_rank = self._local_raiden_transfer_rank()
            for req_id in done_sending:
                registration = self._stage3_registered_sends.get(req_id)
                if (req_id not in self._stage3_reported_sends
                        and registration is not None):
                    logger.info(
                        "%s",
                        json.dumps(
                            {
                                "event": "raiden_stage3_sender_complete",
                                "req_id": req_id,
                                "uuid": registration.uuid,
                                "num_tokens": registration.num_tokens,
                                "transfer_rank": transfer_rank,
                            },
                            sort_keys=True,
                        ),
                    )
            for req_id in sender_failures:
                registration = self._stage3_registered_sends.get(req_id)
                logger.error(
                    "Stage-3 producer native transfer failed req_id=%s "
                    "uuid=%s transfer_rank=%d",
                    req_id,
                    registration.uuid
                    if registration is not None else "unknown",
                    transfer_rank,
                )
        elif self._raiden_stage3_enabled():
            for req_id in done_recving:
                uuid = self._stage3_submitted_loads.get(req_id)
                num_tokens = self._stage3_submitted_load_tokens.get(req_id)
                source_req_id = self._stage3_source_request_id(req_id)
                if (req_id not in self._reported_recving and uuid is not None
                        and num_tokens is not None):
                    start_time = self._stage3_load_start_times.get(req_id)
                    latency_ms = None
                    if start_time is not None:
                        latency_ms = (time.perf_counter() - start_time) * 1000
                    logger.info(
                        "%s",
                        json.dumps(
                            {
                                "event": "raiden_stage3_receiver_complete",
                                "req_id": source_req_id,
                                "destination_req_id": req_id,
                                "uuid": uuid,
                                "num_tokens": num_tokens,
                                "reshard_e2e_latency_ms": latency_ms,
                            },
                            sort_keys=True,
                        ),
                    )
            for req_id in failed_recving:
                source_req_id = self._stage3_source_request_id(req_id)
                logger.error(
                    "Stage-3 receiver native transfer failed req_id=%s "
                    "destination_req_id=%s uuid=%s",
                    source_req_id,
                    req_id,
                    self._stage3_submitted_loads.get(req_id, "unknown"),
                )
        if self.is_producer and self._raiden_stage3_enabled():
            for req_id in native_terminal_sends:
                self._stage3_terminal_cleanup[req_id] = True
            for req_id in cancelled_sends:
                self._stage3_terminal_cleanup[req_id] = False
        if (self._stage3_terminal_cleanup and self.is_producer
                and self._raiden_stage3_enabled()):
            facade, _ = self._require_stage3_controller()
            # Iterate accumulated terminal sends, not only this poll's delta.
            # If the controller RPC fails, get_finished propagates the error
            # and the retained entry is retried on the next poll instead of
            # leaking until registry TTL.
            for req_id, force_release in list(
                    self._stage3_terminal_cleanup.items()):
                registration = self._stage3_registered_sends.get(req_id)
                if registration is None:
                    self._stage3_terminal_cleanup.pop(req_id, None)
                    continue
                # Do not retire D5 state on request finish/submission. Only the
                # manager's real transfer completion may force-release it. An
                # expiry is terminal only after the controller atomically
                # confirms lookup never claimed the request and seals out late
                # registration. Empty ranks do not issue an early global
                # release. Active ranks may race the idempotent aggregate
                # cleanup RPC after genuine native completion.
                if force_release:
                    facade.complete_request_blocks(
                        req_id=req_id,
                        uuid=registration.uuid,
                        unit=self._raiden_work_unit,
                    )
                tombstone_deadline = (
                    time.perf_counter() +
                    float(dist_utils.get_p2p_wait_pull_timeout()))
                self._stage3_terminal_sends[req_id] = _Stage3RegisteredSend(
                    uuid=registration.uuid,
                    local_block_ids=registration.local_block_ids,
                    num_tokens=registration.num_tokens,
                    expiration_time=tombstone_deadline,
                )
                del self._stage3_registered_sends[req_id]
                del self._stage3_terminal_cleanup[req_id]

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
        engine = self._construct_raiden_transfer_engine(
            list(self.runner.kv_caches))
        self._raiden_transfer_engine = engine
        return engine

    def _construct_raiden_transfer_engine(
            self,
            kv_caches: list[Any],
            *,
            num_slots: int | None = None) -> "KVCacheManager":
        max_blocks = self._max_request_blocks()
        if num_slots is None:
            num_slots = self._num_raiden_slots(max_blocks)
        stage3_enabled = self._raiden_stage3_enabled()
        endpoint_rank = (self._local_raiden_transfer_rank() if stage3_enabled
                         and self.is_producer else self.tp_rank)
        node_id = (endpoint_rank if stage3_enabled and self.is_producer else
                   self.dp_rank if stage3_enabled else self.tp_rank)
        local_control_port = self._rank_control_port(self.kv_transfer_port,
                                                     rank=endpoint_rank)
        manager_kwargs: dict[str, Any] = dict(
            kv_caches=kv_caches,
            node_id=node_id,
            local_control_port=local_control_port,
            max_blocks=max_blocks,
            num_slots=num_slots,
            timeout_s=float(dist_utils.get_p2p_wait_pull_timeout()),
        )
        if stage3_enabled:
            manager_kwargs.update(
                listener_port=0,
                parallelism=self._raiden_transfer_parallelism(),
            )
            if _reshard_store_mode():
                # Register this worker with its engine store's dispatch
                # controller: the reshard coordinator submits transfer
                # programs over the persistent channel this creates.
                manager_kwargs.update(
                    raiden_worker_port=0,
                    raiden_controller_address=_reshard_dispatch_address(
                        self.dp_rank),
                    worker_id=(f"worker_{self._local_raiden_transfer_rank()}"),
                )
        engine = self._new_raiden_manager(**manager_kwargs)
        logger.info(
            "TPURaidenConnectorWorker rank%d --> Raiden engine enabled | "
            "legacy_control_port=%d data_endpoint=%s listener_endpoint=%s "
            "max_blocks=%d num_slots=%d",
            self.tp_rank,
            local_control_port,
            str(getattr(engine, "transfer_address", "")),
            str(getattr(engine, "listener_address", "")),
            max_blocks,
            num_slots,
        )
        return engine

    def _rank_control_port(self,
                           base_port: int,
                           *,
                           rank: int | None = None) -> int:
        if rank is None:
            rank = self.tp_rank
        return int(base_port) + 2 * int(rank)

    def _resolve_remote_endpoint(self, req_meta: LoadMeta) -> str:
        if isinstance(req_meta.remote_host, list):
            host = req_meta.remote_host[self.node_id]
            base_port = int(req_meta.remote_port[self.node_id])
            # Per-node bases already carry the node offset (each entry is that
            # node's first-rank control port), so dial with the node-local rank.
            ranks_per_node = max(1, self.tp_size // len(req_meta.remote_host))
            local_rank = self.tp_rank - self.node_id * ranks_per_node
            return f"{host}:{self._rank_control_port(base_port, rank=local_rank)}"
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
                remote_side_channel_port=params.get(
                    "remote_side_channel_port"),
            )
        else:
            self.reqs_to_load[request.request_id] = LoadMeta(
                uuid=params["uuid"],
                local_block_ids=None,
                remote_block_ids=None,
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
                remote_side_channel_port=params.get(
                    "remote_side_channel_port"),
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
            kv_transfer_params = dict(
                uuid=uuid,
                remote_block_ids=computed_per_group,
                remote_host=self.kv_ip,
                remote_port=self.kv_port,
                remote_side_channel_port=self.side_channel_port)
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
