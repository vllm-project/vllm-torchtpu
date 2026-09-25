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

import dataclasses
import importlib
import json
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import torch
from torch_tpu._internal.batch_transfer import (
    batch_transfer_d2h,
    batch_transfer_d2h_sync,
    batch_transfer_h2d,
    batch_transfer_h2d_sync,
)
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.utils.math_utils import round_down
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.request import RequestStatus

from vllm_torchtpu.tracing.annotation import TraceAnnotation

if TYPE_CHECKING:
    from tpu_sync.api.torch.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

import vllm_torchtpu.distributed.utils as dist_utils
from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.distributed.kv_transfer import kv_scatter
from vllm_torchtpu.distributed.kv_transfer.connector_metadata import (
    LoadMeta,
    ReqId,
    SendMeta,
    TPUConnectorMetadata,
)
from vllm_torchtpu.distributed.kv_transfer.tpu_connector_stats import (
    TpuKVConnectorPromMetrics,
    TpuKVConnectorStats,
)

# zmq/shm transport pieces: used only by the zmq-stack workers below
# (TPUConnectorWorker), not by the Raiden connector.
from vllm_torchtpu.distributed.kv_transfer.zmq_shm_base import (
    ZmqShmKvConnectorBase,
    _CoordRecvEntry,
    _CoordSendEntry,
)
from vllm_torchtpu.kernels.gdn.head_geometry import (
    GdnHeadGeometry,
    derive_gdn_head_geometry,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)

__all__ = [
    "TPUConnector",
    "TPUConnectorScheduler",
    "TPUConnectorWorker",
    "TPURaidenConnector",
    "TPURaidenConnectorScheduler",
    "TPURaidenConnectorWorker",
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
# tpu_sync GetRequestBlockStatusResponse.Status values reported by the
# reshard client's get_request_block_status probe.
_STAGE3_REGISTRY_STATUS_UNKNOWN = 1
_STAGE3_REGISTRY_STATUS_CANCELLED = 4
# Deferred-submit re-attempts ride cheap no-forward steps, which can cycle at
# kHz while a load waits; rate-limit the controller RPCs per parked request.
_STAGE3_REGISTRATION_REATTEMPT_MIN_INTERVAL_S = 0.05
_STAGE3_REGISTRATION_CANCELLED_ERROR = "Request block registration was cancelled"
# Pool selection is vLLM policy expressed as request data: the connector names
# the manifest tags to move and raiden resolves them against both peers'
# registered manifests. Each GDN state class rides a sibling transfer with a
# T3.1: one transfer carries the FA payload plus every GDN state class
# under the base req_id/uuid; the tags below define the canonical order
# (FA first = H2D order rank 0).
_STAGE3_STATE_CLASS_TAGS = ("gdn.conv", "gdn.ssm")
_STAGE3_GLM_TRANSFER_POOL_TAGS = ("mla.nope", "mla.rope", "dsa.idx")
# Concurrent Stage-3 coordination RPCs per worker. The source controller
# handles concurrent coordinations (plan claims are per request), so a small
# pool keeps an admission burst from queueing behind one slow peer.
_STAGE3_SUBMIT_WORKERS = 4

_KV_PARAMS_ADMITTED = "_tpu_kv_params_admitted"
_KV_PARAMS_REJECTED = "_tpu_kv_params_rejected"
_STAGE3_LEGACY_ROUTING_FIELDS = frozenset(
    {"remote_block_ids", "remote_host", "remote_port"}
)


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
            f"blocks={len(block_ids)}, required={required_blocks}"
        )
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
    mamba_state_block_ids: list[int] | None = None
    # Prefix-aware load: tokens satisfied by the decode-local prefix cache
    # and excluded from the transfer. local_block_ids then holds only the
    # suffix pages; the source store clips its plan at
    # skip_tokens * token_bytes of the FA global destination byte space.
    skip_tokens: int = 0
    # Full local hit: nothing to transfer; the worker only releases the
    # producer's request-block registration instead of pulling.
    release_only: bool = False
    # When True on a release_only entry (e.g. a consumer request aborted while
    # queued in reqs_to_load before reaching the worker), also record the
    # request in _done_recving so get_finished() surfaces it and the vLLM
    # scheduler frees the request's delayed local KV blocks.
    report_completion: bool = False
    # The scheduler rejected this load after allocation.
    fail_only: bool = False


@dataclass
class _Stage3PendingSubmit:
    """One consumer load parked because the producer's request-block
    registrations were not yet visible at the source controller.

    The registration travels out-of-band and structurally lags the producer's
    finish response by at least one producer scheduler step, so a consumer
    query can legitimately arrive first. Instead of spinning the worker step
    in an inline poll, the request is re-attempted once per scheduler step
    from get_finished() until defer_deadline.
    """

    req_meta: Any
    source_req_id: str
    destination_req_id: str
    uuid: int
    num_tokens: int
    local_blocks: list
    dst_controller_address: str
    dst_units: list
    # None = legacy inline-poll mode (the entry is then never parked).
    defer_deadline: float | None
    first_attempt_s: float
    wait_logged: bool = False
    # RPC rate limiting for re-attempts.
    last_attempt_s: float = 0.0
    # The prepared coordination call, staged once per submission and reused
    # by every re-attempt (_Stage3SubmitTask).
    task: Any = None


@dataclass(frozen=True)
class _Stage3SubmitTask:
    """One staged source-controller coordination call.

    Everything scheduler-visible about the request is recorded in the worker
    tables before the task is created, so executing the RPC needs no
    connector state beyond the facade and the prepared call arguments.
    """

    pending: _Stage3PendingSubmit
    skip_tokens: int
    fa_skip_bytes: int
    source_units: int
    facade: Any
    call_kwargs: dict[str, Any]
    # Legacy inline-poll mode: the RPC call itself retries a missing producer
    # registration for a short bounded interval instead of reporting it.
    retry_inline: bool


@dataclass(frozen=True)
class _Stage3SubmitOutcome:
    """Result of one coordination RPC, applied on the model-runner thread."""

    task: _Stage3SubmitTask
    error: BaseException | None
    submit_ms: float
    # perf_counter() when the RPC was issued; paces re-attempts.
    attempted_at: float


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


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _as_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _as_int_list(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)):
        return None
    ints = [_as_int(item) for item in value]
    if any(item is None for item in ints):
        return None
    return ints  # type: ignore[return-value]


def stage3_fa_raiden_id_fields(
    *,
    job_name: str,
    engine_id: str,
    dp_rank: int,
    transfer_rank: int = 0,
    is_producer: bool = True,
    per_rank_unit: bool | None = None,
) -> dict[str, Any]:
    """Build the canonical Stage-3 FA work-unit identity.

    `per_rank_unit` names one unit per transfer rank (`<engine>-rank<k>`);
    it defaults to `is_producer`, and a pipeline-parallel consumer sets it
    because each of its stages is its own destination unit. A TP-sharded
    consumer also sets it because every shard is an independent destination
    unit.
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
    if per_rank_unit is None:
        per_rank_unit = is_producer
    replica_id = f"{engine_id}-rank{transfer_rank}" if per_rank_unit else engine_id
    return {
        "job_name": job_name,
        "job_replica_id": replica_id,
        "data_name": "kv.fa",
        "data_replica_idx": dp_rank,
    }


def _use_raiden_stage3_transport() -> bool:
    return str(tpu_envs.TPU_KV_RESHARD_TRANSPORT).strip().lower() == "raiden"


def _use_per_layer_pool_tags() -> bool:
    return bool(tpu_envs.TPU_RAIDEN_POOL_TAGS_PER_LAYER)


def _pipeline_parallel_size(vllm_config: VllmConfig) -> int:
    parallel_config = vllm_config.parallel_config
    if parallel_config is None:
        return 1
    return int(parallel_config.pipeline_parallel_size or 1)


def _engine_is_pipeline(vllm_config: VllmConfig) -> bool:
    """A pipeline-parallel engine: each stage transfers its own layers,
    every token of them, so the transfer ranks are the pipeline stages."""
    return _pipeline_parallel_size(vllm_config) > 1


def _reshard_store_mode() -> bool:
    """In-engine hosting: the reshard service lives in each engine's rank-0
    worker (an in-process KVCacheStore), not an external controller."""
    return str(tpu_envs.TPU_RAIDEN_RESHARD_IMPL).strip().lower() == "store"


def _raiden_seq_on_lane_layout() -> bool:
    """Decision D1: the batched-RPA seq-along-lane FA pool (either kernel
    flag) is declared and lowered by the stand-alone ``raiden/seq_on_lane.py``
    module; the token-major path is untouched."""
    from vllm import envs as vllm_envs

    return vllm_envs.VLLM_KV_CACHE_LAYOUT == "HND"


def _raiden_dst_shards() -> int:
    configured = tpu_envs.TPU_RAIDEN_DST_SHARDS
    shards = 1 if configured is None else int(configured)
    if shards <= 0:
        raise ValueError("TPU_RAIDEN_DST_SHARDS must be positive")
    return shards


def _raiden_dst_dcp_size(vllm_config: VllmConfig) -> int:
    """Destination geometry needed before the producer publishes its spans."""
    degree = int(
        _get_extra_config(vllm_config).get(
            "destination_decode_context_parallel_size", 1
        )
    )
    if degree <= 0 or _raiden_dst_shards() % degree:
        raise ValueError("destination DCP size must divide destination TP size")
    return degree


def _reshard_advertise_host() -> str:
    host = str(tpu_envs.TPU_RAIDEN_ADVERTISE_HOST).strip()
    if not host:
        raise ValueError(
            "TPU_RAIDEN_ADVERTISE_HOST is required when TPU_RAIDEN_RESHARD_IMPL=store"
        )
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


def _use_raiden_glm_admission() -> bool:
    return bool(
        tpu_envs.TPU_USE_RAIDEN_KV_CACHE_MANAGER and tpu_envs.TPU_RAIDEN_GLM_ADMISSION
    )


def _is_kimi(vllm_config: VllmConfig) -> bool:
    return getattr(vllm_config.model_config, "architecture", None) in (
        "KimiK3ForConditionalGeneration",
        "KimiLinearForCausalLM",
    )


def _uses_tp_stage3_source(vllm_config: VllmConfig) -> bool:
    """Whether producer cache ownership follows TP ranks rather than PCP."""
    return _use_raiden_glm_admission() or (
        bool(tpu_envs.TPU_USE_RAIDEN_KV_CACHE_MANAGER)
        and bool(tpu_envs.TPU_RAIDEN_KIMIK3_ADMISSION)
        and _is_kimi(vllm_config)
    )


def _prefix_aware_load_requested() -> bool:
    return bool(tpu_envs.TPU_RAIDEN_PREFIX_AWARE_LOAD)


@lru_cache(maxsize=1)
def _prefix_aware_load_supported() -> bool:
    """Whether the installed reshard client can encode suffix clips.

    Capability/version skew is a performance concern, not a serving-fatal
    condition: callers fall back to decode-local computation on partial hits.
    """
    try:
        from tpu_sync.api.torch import reshard_client as _reshard_client

        return bool(getattr(_reshard_client, "SUPPORTS_DST_SKIP_BYTES", False))
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "Could not probe tpu-sync dst_skip_bytes support; prefix-aware "
            "loads will fall back to decode-local computation: %s",
            exc,
        )
        return False


def _prefix_aware_load_enabled() -> bool:
    """Whether suffix-only Stage-3 loads are requested and supported."""
    return _prefix_aware_load_requested() and _prefix_aware_load_supported()


def _use_raiden_connector(vllm_config: VllmConfig) -> bool:
    if _use_raiden_stage3_transport():
        return True
    extra_config = _get_extra_config(vllm_config)
    if "use_raiden_connector" in extra_config:
        return _as_bool(extra_config["use_raiden_connector"])
    return dist_utils.get_use_raiden_connector()


class TPUConnector(KVConnectorBase_V1, SupportsHMA):
    force_raiden_connector = False

    @property
    def supports_divergent_local_hybrid_hits(self) -> bool:
        """Stage-3 suffix loads also restore the committed Mamba state."""
        return (
            self.use_raiden
            and not self._kv_transfer_config.is_kv_producer
            and _use_raiden_stage3_transport()
            and _prefix_aware_load_enabled()
        )

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        self._connector_metadata: TPUConnectorMetadata | None = None
        use_raiden = self.force_raiden_connector or _use_raiden_connector(vllm_config)
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
                    f"full-attention KV cache group, got {fa_groups}"
                )
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
                f"KV cache groups: depths={speculative_depths}"
            )
            stage3_mamba_num_speculative_blocks = next(iter(speculative_depths), 0)
            assert stage3_mamba_num_speculative_blocks >= 0, (
                "vLLM produced a MambaSpec with a negative speculative "
                f"depth: value={stage3_mamba_num_speculative_blocks}"
            )
        if use_raiden:
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
                    self._stage3_fa_group_index
                )
                self.connector_scheduler._stage3_mamba_group_indices = (
                    self._stage3_mamba_group_indices
                )
                self.connector_scheduler._stage3_mamba_num_speculative_blocks = (
                    stage3_mamba_num_speculative_blocks
                )
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = worker_cls(vllm_config)

    # ---- Scheduler-side methods -----------------------------------------
    def on_new_request(self, request: "Request") -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.on_new_request(request)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

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
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None

        if self.use_raiden and _use_raiden_stage3_transport():
            if self._stage3_fa_group_index >= len(block_ids):
                raise ValueError(
                    "Stage-3 FA KV cache group is absent from request block "
                    f"tables: index={self._stage3_fa_group_index}, "
                    f"groups={len(block_ids)}"
                )
            mamba_block_ids = None
            if self._stage3_mamba_group_indices:
                mamba_block_ids = []
                for mamba_gid in self._stage3_mamba_group_indices:
                    if mamba_gid >= len(block_ids):
                        raise ValueError(
                            "GDN state reshard: mamba KV cache group is "
                            "absent from request block tables: "
                            f"index={mamba_gid}, groups={len(block_ids)}"
                        )
                    mamba_block_ids.append(list(block_ids[mamba_gid]))
            # use_raiden here means __init__ selected
            # TPURaidenConnectorScheduler -- the only one of the two
            # schedulers accepting mamba block ids. The checker cannot
            # narrow the union to that subclass.
            return self.connector_scheduler.request_finished(
                request,
                block_ids[self._stage3_fa_group_index],
                mamba_block_ids=mamba_block_ids,  # type: ignore
            )
        assert len(block_ids) == 1, (
            "Non-HMA TPUConnector expects a single kv-cache group; got "
            f"{len(block_ids)} groups"
        )
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
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        # `data` is inherited from vLLM's KVConnectorStats dataclass, which a
        # static checker cannot see through vllm's editable install.
        if data is not None:
            return TpuKVConnectorStats(data=data)  # type: ignore[unexpected-keyword]
        return TpuKVConnectorStats()

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        return TpuKVConnectorPromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )

    # ---- Worker-side methods --------------------------------------------
    def register_kv_caches(self, kv_caches: dict[str, Any]):
        """Registers the named cache materialization when the backend needs it.

        ZMQ reads ``runner.kv_caches`` positionally. Raiden explicit-pool
        admission needs the layer names and group specs to derive its
        canonical pool manifest.
        """
        if self.use_raiden and self.connector_worker is not None:
            self.connector_worker.named_kv_caches = kv_caches

    def register_runner(self, runner: TPUModelRunner) -> None:
        assert self.connector_worker is not None
        self.connector_worker.register_runner(runner)

    def start_load_kv(
        self,
        _,
        wait_for_completion: bool = False,
        report_completion: bool = True,
        **kwargs,
    ) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, TPUConnectorMetadata)
        self.connector_worker.process_send_load(
            self._connector_metadata,
            wait_for_completion=wait_for_completion,
            report_completion=report_completion,
        )

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

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
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
        if scheduler is not None and hasattr(scheduler, "update_connector_output"):
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
        dp_rank = (
            vllm_config.parallel_config.data_parallel_rank
            if vllm_config.parallel_config
            else 0
        )
        tp_size = (
            vllm_config.parallel_config.tensor_parallel_size
            if vllm_config.parallel_config
            else 1
        )
        port_base = dist_utils.get_kv_ports()
        if isinstance(port_base, list):
            self.kv_port = [int(p) + dp_rank * tp_size for p in port_base]
        else:
            self.kv_port = int(port_base) + dp_rank * tp_size
        self.side_channel_port = int(dist_utils.get_side_channel_port()) + dp_rank
        logger.info(
            "TPUConnectorScheduler --> kv_ip=%s | kv_port=%s | side_channel_port=%s",
            self.kv_ip,
            self.kv_port,
            self.side_channel_port,
        )

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

    def on_new_request(self, request: "Request") -> None:
        """Validates decoder kv_transfer_params once at admission. The Raiden
        stage-3 subclass also truncates producer Mamba prompts here."""
        self._admit_kv_transfer_params(request)

    def _admit_kv_transfer_params(self, request: "Request") -> bool:
        """Returns whether a decoder request carries usable remote-KV params."""
        params = request.kv_transfer_params
        if self.is_producer or not params:
            return False
        if not isinstance(params, dict):
            logger.warning(
                "Ignoring kv_transfer_params of type %s for req_id=%s; "
                "the request prefills locally",
                type(params).__name__,
                request.request_id,
            )
            request.kv_transfer_params = None
            return False
        if params.get(_KV_PARAMS_REJECTED):
            return False
        if params.get(_KV_PARAMS_ADMITTED):
            return True
        reason = self._validate_load_params(request, params)
        if reason is not None:
            self._reject_kv_transfer_params(request, params, reason)
            return False
        params[_KV_PARAMS_ADMITTED] = True
        return True

    def _validate_load_params(
        self, request: "Request", params: dict[str, Any]
    ) -> str | None:
        """Checks the ZMQ/legacy-Raiden routing fields."""
        del request
        uuid = _as_int(params.get("uuid"))
        if uuid is None:
            return f"uuid must be an integer, got {params.get('uuid')!r}"
        remote_block_ids = _as_int_list(params.get("remote_block_ids"))
        if remote_block_ids is None:
            return "remote_block_ids must be a list of integers"
        remote_host: Any = params.get("remote_host")
        remote_port: Any = params.get("remote_port")
        if isinstance(remote_host, list):
            hosts = [_as_str(host) for host in remote_host]
            ports = _as_int_list(remote_port)
            if (
                not hosts
                or any(host is None for host in hosts)
                or ports is None
                or len(ports) != len(hosts)
            ):
                return (
                    "a remote_host list needs non-empty hosts and a "
                    "remote_port list of the same length"
                )
            remote_host, remote_port = hosts, ports
        else:
            remote_host = _as_str(remote_host)
            remote_port = _as_int(remote_port)
            if remote_host is None or remote_port is None:
                return "remote_host and remote_port must be a host and a port"
        if params.get("remote_side_channel_port") is not None:
            side_channel_port = _as_int(params["remote_side_channel_port"])
            if side_channel_port is None:
                return "remote_side_channel_port must be an integer"
            params["remote_side_channel_port"] = side_channel_port
        params["uuid"] = uuid
        params["remote_block_ids"] = remote_block_ids
        params["remote_host"] = remote_host
        params["remote_port"] = remote_port
        return None

    def _reject_kv_transfer_params(
        self, request: "Request", params: dict[str, Any], reason: str
    ) -> None:
        """Marks the params unusable so every hook skips the remote load."""
        logger.warning(
            "Ignoring invalid kv_transfer_params for req_id=%s: %s; "
            "the request prefills locally",
            request.request_id,
            reason,
        )
        params[_KV_PARAMS_REJECTED] = reason
        params["_remote_kv_processed"] = True

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
        if not self._admit_kv_transfer_params(
            request
        ) or request.kv_transfer_params.get("_remote_kv_processed"):
            return 0, False

        assert num_computed_tokens % self.block_size == 0
        # Rounding must match request_finished()'s remote_block_ids computation.
        rounded_num_prompt_tokens = round_down(
            len(request.prompt_token_ids), self.block_size
        )
        count = max(rounded_num_prompt_tokens - num_computed_tokens, 0)
        # The pull is blocking at the ZMQ layer, but we wrap it in a thread
        # pool so from the scheduler's perspective it's async.
        if count > 0:
            return count, True
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        if not self._admit_kv_transfer_params(request):
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
                remote_side_channel_port=params.get("remote_side_channel_port"),
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
                remote_side_channel_port=params.get("remote_side_channel_port"),
            )
        params["_remote_kv_processed"] = True
        logger.info(
            "TPUConnectorScheduler update_state_after_alloc --> reqs_to_load=%s",
            self.reqs_to_load,
        )

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
    ) -> tuple[bool, dict[str, Any] | None]:
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
            expiration_time = (
                time.perf_counter() + dist_utils.get_p2p_wait_pull_timeout()
            )
            self.reqs_to_send[request.request_id] = SendMeta(
                uuid=uuid,
                local_block_ids=computed_block_ids,
                expiration_time=expiration_time,
            )
            kv_transfer_params = dict(
                uuid=uuid,
                remote_block_ids=computed_block_ids,
                remote_host=self.kv_ip,
                remote_port=self.kv_port,
                remote_side_channel_port=self.side_channel_port,
            )
            logger.info(
                "TPUConnectorScheduler --> reqs_to_send=%s | kv_transfer_params=%s",
                self.reqs_to_send,
                kv_transfer_params,
            )
        else:
            kv_transfer_params = {}

        return delay_free_blocks, kv_transfer_params


class TPUConnectorWorker(ZmqShmKvConnectorBase):
    """TPU-specific transport hooks for the generic ZmqShmKvConnectorBase."""

    def process_send_load(
        self,
        metadata: TPUConnectorMetadata,
        wait_for_completion: bool = False,
        report_completion: bool = True,
    ) -> None:
        del wait_for_completion, report_completion
        super().process_send_load(metadata)

    def _build_d2h_views(
        self, slot_idx: int, num_blocks: int, block_ids: list[int]
    ) -> tuple[list, list, int]:
        indices = torch.tensor(block_ids, dtype=torch.int64, device=self.device)
        kv_caches = self.runner.kv_caches
        tpu_tensors: list = []
        cpu_tensors: list = []
        d2h_total_bytes = 0
        for layer_idx, cache in enumerate(kv_caches):
            src_shard = torch.index_select(cache, 0, indices)
            dest_view = self._coord_pool.layer_view(
                slot_idx, self.local_tp_rank, layer_idx, num_blocks
            )
            tpu_tensors.append(src_shard)
            cpu_tensors.append(dest_view)
            d2h_total_bytes += dest_view.numel() * dest_view.element_size()
        return tpu_tensors, cpu_tensors, d2h_total_bytes

    def _stage_d2h(
        self, slot_idx: int, num_blocks: int, block_ids: list[int]
    ) -> tuple[Any, list, list, int]:
        tpu_tensors, cpu_tensors, total_bytes = self._build_d2h_views(
            slot_idx, num_blocks, block_ids
        )
        future = batch_transfer_d2h(tpu_tensors, cpu_tensors)
        return future, tpu_tensors, cpu_tensors, total_bytes

    def _stage_d2h_sync(
        self, slot_idx: int, num_blocks: int, block_ids: list[int]
    ) -> None:
        tpu_tensors, cpu_tensors, _ = self._build_d2h_views(
            slot_idx, num_blocks, block_ids
        )
        batch_transfer_d2h_sync(tpu_tensors, cpu_tensors)

    def _wait_stage(self, future: Any) -> None:
        future.wait()

    def _h2d_into_device(self, src_views: list) -> list[torch.Tensor]:
        device_shards = [
            torch.empty(v.shape, dtype=v.dtype, device=self.device) for v in src_views
        ]
        batch_transfer_h2d_sync(src_views, device_shards)
        return device_shards

    def _h2d_into_device_async(self, src_views: list) -> tuple[Any, list[torch.Tensor]]:
        device_shards = [
            torch.empty(v.shape, dtype=v.dtype, device=self.device) for v in src_views
        ]
        future = batch_transfer_h2d(src_views, device_shards)
        return future, device_shards

    def _synchronize_device(self, tensor: torch.Tensor) -> None:
        synchronize_tensors(tensor, wait=False)

    def _try_fast_scatter(
        self,
        device_shards: list[torch.Tensor],
        kv_caches: list[torch.Tensor],
        local_blocks: list[int],
    ) -> list[torch.Tensor] | None:
        if not self._kv_scatter_enabled:
            return None
        dest_blocks_dev = torch.tensor(
            local_blocks, dtype=torch.int32, device=self.device
        )
        prebuilt = kv_scatter.prepare_scatter_args(dest_blocks_dev, self.device)
        return kv_scatter.multi_layer_scatter_into(
            device_shards, kv_caches, dest_blocks_dev, prebuilt_args=prebuilt
        )

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
                "unavailable; staying on index_put_",
                self.node_id,
                self.tp_rank,
            )
            return
        trailing = tuple(self.shape[1:]) if len(self.shape) > 1 else (128,)
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
                "enabled for insert path",
                self.node_id,
                self.tp_rank,
            )
        else:
            logger.warning(
                "TPUConnectorWorker %s rank%d --> multi-layer scatter "
                "smoke test failed; falling back to index_put_",
                self.node_id,
                self.tp_rank,
            )


class TPURaidenConnectorScheduler(TPUConnectorScheduler):
    """Scheduler half for the opt-in Raiden transfer backend."""

    def __init__(self, vllm_config: "VllmConfig"):
        super().__init__(vllm_config)
        stage3_consumer = not self.is_producer and _use_raiden_stage3_transport()
        self._stage3_prefix_aware_load_enabled = (
            stage3_consumer and _prefix_aware_load_enabled()
        )
        if (
            stage3_consumer
            and bool(vllm_config.cache_config.enable_prefix_caching)
            and not self._stage3_prefix_aware_load_enabled
        ):
            log = logger.warning if _prefix_aware_load_requested() else logger.info
            log(
                "Prefix-aware Stage-3 loads are unavailable; decode-local "
                "prefix hits are ignored and the full producer payload is "
                "pulled into every destination page"
            )
        # req_id -> (uuid, global block ids, exact token count, wire params).
        # This survives build_connector_meta() so a duplicate request_finished
        # callback cannot mint a conflicting UUID or enqueue a second D5
        # registration.
        self._stage3_finished_sends: OrderedDict[
            str, tuple[int, tuple[int, ...], int, dict[str, Any]]
        ] = OrderedDict()
        self._stage3_fa_group_index = 0
        self._stage3_mamba_group_indices: list[int] = []
        self._stage3_mamba_num_speculative_blocks: int = 0
        # Decoder: prefill (source) req_id -> the admitted destination
        # req_id.
        self._stage3_active_source_req_ids: dict[str, str] = {}
        # Get DP rank and TP size from config and stagger kv_port and side_channel_port
        dp_rank = (
            vllm_config.parallel_config.data_parallel_rank
            if vllm_config.parallel_config
            else 0
        )
        tp_size = (
            vllm_config.parallel_config.tensor_parallel_size
            if vllm_config.parallel_config
            else 1
        )
        port_base = dist_utils.get_kv_ports()
        if isinstance(port_base, list):
            self.kv_port = [int(p) + 2 * dp_rank * tp_size for p in port_base]
        else:
            self.kv_port = int(port_base) + 2 * dp_rank * tp_size
        logger.info(
            "TPURaidenConnectorScheduler --> kv_ip=%s | kv_port=%s",
            self.kv_ip,
            self.kv_port,
        )

    def on_new_request(self, request: "Request") -> None:
        """Truncates a P-side Mamba prompt at admission, ahead of the
        scheduler's prefix-cache lookup. The lookup caps its hit at
        num_tokens - 1 of whatever length it sees, so shortening the prompt
        any later can leave zero new tokens to schedule."""
        if (
            self.is_producer
            and _use_raiden_stage3_transport()
            and self._stage3_mamba_group_indices
        ):
            self._maybe_truncate_for_mamba(request)
        self._admit_kv_transfer_params(request)

    def _validate_load_params(
        self, request: "Request", params: dict[str, Any]
    ) -> str | None:
        if not _use_raiden_stage3_transport():
            return super()._validate_load_params(request, params)
        leaked = sorted(_STAGE3_LEGACY_ROUTING_FIELDS.intersection(params))
        if leaked:
            return (
                "Stage-3 metadata must not carry legacy source routing/block "
                f"fields: {leaked}"
            )
        source_req_id = _as_str(params.get("req_id"))
        if source_req_id is None:
            return "Stage-3 metadata requires a non-empty source request ID"
        uuid = _as_int(params.get("uuid"))
        if uuid is None or uuid <= 0:
            return (
                f"Stage-3 uuid must be a positive integer, got {params.get('uuid')!r}"
            )
        # The producer publishes at most its prompt prefix, so a larger
        # extent would size the destination page set past the prompt.
        num_tokens = _as_int(params.get("num_tokens"))
        num_prompt_tokens = int(request.num_prompt_tokens)
        if num_tokens is None or not 0 < num_tokens <= num_prompt_tokens:
            return (
                "Stage-3 num_tokens must be an integer in "
                f"[1, {num_prompt_tokens}], got {params.get('num_tokens')!r}"
            )
        src_controller_address = _as_str(params.get("src_controller_address"))
        if src_controller_address is None:
            return "Stage-3 metadata requires an explicit source controller address"
        src_job_name = _as_str(params.get("src_job_name"))
        src_engine_id = _as_str(params.get("src_engine_id"))
        if src_job_name is None or src_engine_id is None:
            return "Stage-3 metadata requires explicit source job and engine identity"
        src_data_replica_idx = _as_int(params.get("src_data_replica_idx"))
        if src_data_replica_idx is None or src_data_replica_idx < 0:
            return "Stage-3 source data replica index must be a non-negative integer"
        src_parallelism = _as_int(params.get("src_parallelism"))
        if src_parallelism is None or src_parallelism <= 0:
            return "Stage-3 source transfer parallelism must be a positive integer"
        tp_size = int(self.vllm_config.parallel_config.tensor_parallel_size)
        dcp_size = int(self.vllm_config.parallel_config.decode_context_parallel_size)
        if not _uses_tp_stage3_source(self.vllm_config) and (
            _as_int(params.get("dst_tp_size")) != tp_size
            or _as_int(params.get("dst_dcp_size")) != dcp_size
        ):
            return (
                "Stage-3 metadata requires explicit matching destination "
                "TP/DCP geometry for the producer's registered spans"
            )
        bound_req_id = self._stage3_active_source_req_ids.get(source_req_id)
        if bound_req_id is not None and bound_req_id != request.request_id:
            return (
                f"Stage-3 source request ID {source_req_id!r} is already "
                f"bound to req_id={bound_req_id}"
            )
        self._stage3_active_source_req_ids[source_req_id] = request.request_id
        params.update(
            req_id=source_req_id,
            uuid=uuid,
            num_tokens=num_tokens,
            src_controller_address=src_controller_address,
            src_job_name=src_job_name,
            src_engine_id=src_engine_id,
            src_data_replica_idx=src_data_replica_idx,
            src_parallelism=src_parallelism,
        )
        return None

    def _reject_kv_transfer_params(
        self, request: "Request", params: dict[str, Any], reason: str
    ) -> None:
        if _use_raiden_stage3_transport():
            # Release the prefill's pinned registration when the
            # params still identify it, unless another admitted request
            # owns that source.
            source_req_id = _as_str(params.get("req_id"))
            owner = self._stage3_active_source_req_ids.get(source_req_id or "")
            if owner is None or owner == request.request_id:
                self._enqueue_stage3_release(request)
        super()._reject_kv_transfer_params(request, params, reason)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if not self._admit_kv_transfer_params(
            request
        ) or request.kv_transfer_params.get("_remote_kv_processed"):
            return 0, False

        assert num_computed_tokens % self.block_size == 0
        if _use_raiden_stage3_transport():
            # Unlike the legacy equal-page pull, the controller planner has a
            # byte-precise partial-tail entry. The producer's scheduler-owned
            # num_computed_tokens is the transfer extent: in the deployed
            # proxy's one-token-prefill flow this can be prompt_tokens - 1.
            # Do not replace it with prompt length here.
            transfer_tokens = request.kv_transfer_params["num_tokens"]
            count = max(transfer_tokens - num_computed_tokens, 0)
            if count > 0 and dist_utils.get_raiden_inline_load():
                return count, False
            return count, count > 0

        rounded_num_prompt_tokens = round_down(
            len(request.prompt_token_ids), self.block_size
        )
        count = max(rounded_num_prompt_tokens - num_computed_tokens, 0)
        if count > 0:
            if dist_utils.get_raiden_inline_load():
                total_external_tokens = num_computed_tokens + count
                if total_external_tokens >= len(request.prompt_token_ids):
                    count = max(count - 1, 0)
                logger.info(
                    "TPURaidenConnectorScheduler inline load req_id=%s "
                    "external_tokens=%d",
                    request.request_id,
                    count,
                )
                return count, False
            return count, True
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        if not self._admit_kv_transfer_params(request):
            return

        if _use_raiden_stage3_transport():
            self._update_stage3_state_after_alloc(request, blocks, num_external_tokens)
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
                params["_remote_kv_processed"] = True
                logger.info(
                    "TPURaidenConnectorScheduler prefix hit req_id=%s "
                    "releases remote send uuid=%s",
                    request.request_id,
                    params["uuid"],
                )
                return

            remote_block_ids = params["remote_block_ids"]
            if len(local_block_ids) > len(remote_block_ids):
                self._fail_load(
                    request,
                    local_block_ids,
                    "TPURaidenConnector cannot pull more local blocks than "
                    "the producer published: "
                    f"local={len(local_block_ids)} remote="
                    f"{len(remote_block_ids)}",
                )
                return
            remote_block_ids = remote_block_ids[-len(local_block_ids) :]
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
                "TPURaidenConnectorScheduler no-load release req_id=%s uuid=%s",
                request.request_id,
                params["uuid"],
            )
        params["_remote_kv_processed"] = True
        logger.info(
            "TPURaidenConnectorScheduler update_state_after_alloc --> reqs_to_load=%s",
            self.reqs_to_load,
        )

    def _update_stage3_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Declares a destination transfer without source physical IDs."""
        if num_external_tokens <= 0:
            # First admission with a full local prefix hit: nothing to pull,
            # but the producer holds a request-block registration pinned until
            # unless released. request.num_computed_tokens is still 0 here
            # only on first admission — the post-async-load resume re-enters
            # with the loaded count set, and must not re-release.
            if (
                self._stage3_prefix_aware_load_enabled
                and int(request.num_computed_tokens) == 0
            ):
                self._enqueue_stage3_release(request)
                if request.kv_transfer_params is not None:
                    request.kv_transfer_params["_remote_kv_processed"] = True
            return
        params = request.kv_transfer_params
        assert params is not None
        source_req_id = params["req_id"]
        uuid = params["uuid"]
        num_tokens = params["num_tokens"]
        src_controller_address = params["src_controller_address"]

        dcp_size = int(self.vllm_config.parallel_config.decode_context_parallel_size)
        # One scheduler block covers this many global tokens, although each
        # worker allocates only block_size local token slots.
        logical_block_tokens = self.block_size * dcp_size

        grouped_block_ids = blocks.get_block_ids()
        if self._stage3_fa_group_index >= len(grouped_block_ids):
            raise ValueError(
                "Stage-3 FA KV cache group is absent from decode allocation: "
                f"index={self._stage3_fa_group_index}, "
                f"groups={len(grouped_block_ids)}"
            )
        local_block_ids = list(grouped_block_ids[self._stage3_fa_group_index])
        expected_pages = (num_tokens + logical_block_tokens - 1) // logical_block_tokens
        if len(local_block_ids) != expected_pages:
            self._fail_load(
                request,
                local_block_ids,
                "Stage-3 FA resharding requires the complete destination "
                "page set (prefix-suffix pulls are unsupported): "
                f"blocks={len(local_block_ids)}, expected={expected_pages}, "
                f"num_tokens={num_tokens}, page_tokens={logical_block_tokens}",
            )
            return
        external_tokens = int(num_external_tokens)
        if external_tokens > num_tokens:
            self._fail_load(
                request,
                local_block_ids,
                "External token count exceeds the published reshard payload: "
                f"external={external_tokens}, num_tokens={num_tokens}",
            )
            return
        skip_tokens = (
            num_tokens - external_tokens
            if self._stage3_prefix_aware_load_enabled
            else 0
        )
        if skip_tokens:
            # Partial local prefix hit: pull only the suffix, and only into
            # the trailing (newly allocated) pages — the leading adopted
            # cache pages are shared and must never be transfer targets.
            prefix_pages = skip_tokens // logical_block_tokens
            suffix_pages = (
                external_tokens + logical_block_tokens - 1
            ) // logical_block_tokens
            if (
                skip_tokens % logical_block_tokens != 0
                or prefix_pages + suffix_pages != expected_pages
            ):
                self._fail_load(
                    request,
                    local_block_ids,
                    "Prefix hits must be destination-page aligned: "
                    f"skip_tokens={skip_tokens}, page_tokens={logical_block_tokens}, "
                    f"prefix_pages={prefix_pages}, suffix_pages={suffix_pages}, "
                    f"expected_pages={expected_pages}",
                )
                return
            local_block_ids = local_block_ids[prefix_pages:]
        mamba_state_block_ids: list[int] | None = None
        if self._stage3_mamba_group_indices:
            mamba_block_ids: list[list[int]] = []
            for mamba_gid in self._stage3_mamba_group_indices:
                if mamba_gid >= len(grouped_block_ids):
                    self._fail_load(
                        request,
                        local_block_ids,
                        "GDN state reshard: mamba KV cache group is absent "
                        f"from decode allocation: index={mamba_gid}, "
                        f"groups={len(grouped_block_ids)}",
                    )
                    return
                mamba_block_ids.append(list(grouped_block_ids[mamba_gid]))
            mamba_state_block_ids = _select_committed_mamba_blocks(
                mamba_block_ids, self._stage3_mamba_num_speculative_blocks
            )
        self.reqs_to_load[request.request_id] = _Stage3LoadMeta(
            uuid=uuid,
            source_req_id=source_req_id,
            local_block_ids=local_block_ids,
            num_tokens=num_tokens,
            src_controller_address=src_controller_address,
            src_job_name=params["src_job_name"],
            src_engine_id=params["src_engine_id"],
            src_data_replica_idx=params["src_data_replica_idx"],
            src_parallelism=params["src_parallelism"],
            mamba_state_block_ids=mamba_state_block_ids,
            skip_tokens=skip_tokens,
        )
        params["_remote_kv_processed"] = True
        logger.info(
            "TPURaidenConnectorScheduler Stage-3 load req_id=%s "
            "source_req_id=%s uuid=%d num_tokens=%d skip_tokens=%d "
            "destination_pages=%d source_controller=%s",
            request.request_id,
            source_req_id,
            uuid,
            num_tokens,
            skip_tokens,
            len(local_block_ids),
            src_controller_address,
        )

    def _raiden_engine_id(self) -> str:
        env_id = str(tpu_envs.TPU_RAIDEN_ENGINE_ID).strip()
        if env_id:
            return env_id
        config_id = getattr(
            getattr(self.vllm_config, "kv_transfer_config", None),
            "engine_id",
            None,
        )
        if config_id:
            return str(config_id).strip()
        default_job = "prefill" if self.is_producer else "decode"
        return f"{default_job}-engine"

    def _fail_load(
        self, request: "Request", local_block_ids: list[int], reason: str
    ) -> None:
        """Fails one admitted request's remote load."""
        logger.error(
            "Failing the remote KV load for req_id=%s: %s", request.request_id, reason
        )
        params = request.kv_transfer_params
        assert isinstance(params, dict)
        if _use_raiden_stage3_transport():
            self.reqs_to_load[request.request_id] = _Stage3LoadMeta(
                uuid=params["uuid"],
                source_req_id=params["req_id"],
                local_block_ids=list(local_block_ids),
                num_tokens=params["num_tokens"],
                src_controller_address=params["src_controller_address"],
                src_job_name=params["src_job_name"],
                src_engine_id=params["src_engine_id"],
                src_data_replica_idx=params["src_data_replica_idx"],
                src_parallelism=params["src_parallelism"],
                fail_only=True,
            )
        else:
            self.reqs_to_load[request.request_id] = LoadMeta(
                uuid=params["uuid"],
                local_block_ids=list(local_block_ids),
                remote_block_ids=None,
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
                fail_only=True,
            )
        params["_remote_kv_processed"] = True

    def _enqueue_stage3_release(
        self, request: "Request", *, report_completion: bool = False
    ) -> None:
        """Full local hit, pre-pull abort or rejected params: nothing to pull;
        ask the worker to promptly release the producer's request-block
        registration instead of leaving until p2p_wait_pull_timeout.
        Best-effort: the TTL remains the backstop, so malformed params degrade
        to a warning, not a failure."""
        params = request.kv_transfer_params
        if not isinstance(params, dict):
            params = {}
        source_req_id = _as_str(params.get("req_id"))
        uuid = _as_int(params.get("uuid"))
        src_controller_address = _as_str(params.get("src_controller_address"))
        if (
            source_req_id is None
            or uuid is None
            or uuid <= 0
            or not src_controller_address
        ):
            logger.warning(
                "Stage-3 release for req_id=%s carries incomplete "
                "source metadata; leaving producer release to the TTL",
                request.request_id,
            )
            return
        self.reqs_to_load[request.request_id] = _Stage3LoadMeta(
            uuid=uuid,
            source_req_id=source_req_id,
            local_block_ids=[],
            num_tokens=_as_int(params.get("num_tokens")) or 0,
            src_controller_address=src_controller_address,
            src_job_name=_as_str(params.get("src_job_name")) or "",
            src_engine_id=_as_str(params.get("src_engine_id")) or "",
            src_data_replica_idx=_as_int(params.get("src_data_replica_idx")) or 0,
            src_parallelism=_as_int(params.get("src_parallelism")) or 0,
            release_only=True,
            report_completion=report_completion,
        )
        logger.info(
            "TPURaidenConnectorScheduler release-only req_id=%s "
            "source_req_id=%s uuid=%d releases the producer registration",
            request.request_id,
            source_req_id,
            uuid,
        )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
        mamba_block_ids: list[list[int]] | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        if not _use_raiden_stage3_transport():
            return super().request_finished(request, block_ids)
        if not self.is_producer:
            params = request.kv_transfer_params
            if isinstance(params, dict) and params.get(_KV_PARAMS_ADMITTED):
                source_req_id = params["req_id"]
                if (
                    self._stage3_active_source_req_ids.get(source_req_id)
                    == request.request_id
                ):
                    del self._stage3_active_source_req_ids[source_req_id]
            if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
                if (
                    isinstance(params, dict)
                    and params.get("uuid")
                    and not params.get(_KV_PARAMS_REJECTED)
                ):
                    in_flight_load = self.reqs_to_load.get(request.request_id)
                    if in_flight_load is not None or not params.get(
                        "_remote_kv_processed"
                    ):
                        self._enqueue_stage3_release(
                            request, report_completion=in_flight_load is not None
                        )
                        params["_remote_kv_processed"] = True
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
            finish_params.get("_p_side_truncated")
        ):
            # The producer already dropped one token; N-1 is already applied.
            num_tokens = min(computed_tokens, prompt_tokens)
        else:
            num_tokens = min(computed_tokens, prompt_tokens - 1)
        if num_tokens <= 0 or not block_ids:
            return False, {}
        parallel_config = self.vllm_config.parallel_config
        pcp_size = int(parallel_config.prefill_context_parallel_size or 1)
        if not _uses_tp_stage3_source(self.vllm_config) and pcp_size > 1:
            # The interleave minimum applies to PCP sources only.
            interleave_size = int(parallel_config.cp_kv_cache_interleave_size)
            min_transfer_tokens = pcp_size * interleave_size
            if num_tokens < min_transfer_tokens:
                return False, {}

        max_transfer_tokens = tpu_envs.TPU_RAIDEN_MAX_TRANSFER_TOKENS
        if max_transfer_tokens is not None and max_transfer_tokens > 0:
            if num_tokens > max_transfer_tokens:
                logger.warning(
                    "Stage-3 producer token count (%d) exceeds "
                    "TPU_RAIDEN_MAX_TRANSFER_TOKENS (%d) for req_id=%s; "
                    "skipping transfer and freeing blocks immediately.",
                    num_tokens,
                    max_transfer_tokens,
                    request.request_id,
                )
                return False, {}

        scheduler_block_tokens = self.block_size * pcp_size
        expected_scheduler_blocks = (
            num_tokens + scheduler_block_tokens - 1
        ) // scheduler_block_tokens
        normalized_block_ids = tuple(int(block_id) for block_id in block_ids)
        # The producer may allocate trailing blocks for its excluded final
        # prompt token or locally generated tokens. They are outside the
        # transfer prefix; too few blocks remains an error.
        if len(normalized_block_ids) > expected_scheduler_blocks:
            normalized_block_ids = normalized_block_ids[:expected_scheduler_blocks]
        if len(normalized_block_ids) != expected_scheduler_blocks:
            logger.error(
                "Stage-3 producer block IDs must cover every PCP scheduler "
                "block, including the partial tail: req_id=%s blocks=%d, "
                "expected=%d, num_tokens=%d, page_tokens=%d, pcp_size=%d; "
                "skipping the transfer",
                request.request_id,
                len(block_ids),
                expected_scheduler_blocks,
                num_tokens,
                self.block_size,
                pcp_size,
            )
            return False, {}

        producer_dp_rank = (
            self.vllm_config.parallel_config.data_parallel_rank
            if self.vllm_config.parallel_config
            else 0
        )
        controller_address = _resolved_reshard_controller_address(producer_dp_rank)
        if not controller_address:
            raise ValueError(
                "Stage-3 producer metadata requires TPU_RAIDEN_CONTROLLER_ADDRESS"
            )
        normalized_ids = normalized_block_ids
        src_job_name = str(tpu_envs.TPU_RAIDEN_JOB_NAME).strip() or "prefill"
        src_engine_id = self._raiden_engine_id()
        src_parallelism = int(tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM)
        parallel_config = self.vllm_config.parallel_config
        src_data_replica_idx = int(parallel_config.data_parallel_rank or 0)
        if not src_engine_id:
            raise ValueError("TPU_RAIDEN_ENGINE_ID must not be empty")
        if src_parallelism <= 0:
            raise ValueError("TPU_RAIDEN_TRANSFER_PARALLELISM must be positive")
        if _uses_tp_stage3_source(self.vllm_config):
            tp_size = int(parallel_config.tensor_parallel_size or 1)
            if src_parallelism != tp_size:
                raise ValueError(
                    "Stage-3 TP producer transfer parallelism must equal "
                    f"TP size: parallelism={src_parallelism}, "
                    f"tp_size={tp_size}"
                )
        elif _engine_is_pipeline(self.vllm_config):
            pp_size = _pipeline_parallel_size(self.vllm_config)
            if src_parallelism != pp_size:
                raise ValueError(
                    "Stage-3 pipeline producer transfer parallelism must "
                    f"equal PP size: parallelism={src_parallelism}, "
                    f"pp_size={pp_size}"
                )
        elif src_parallelism != pcp_size * int(parallel_config.tensor_parallel_size):
            raise ValueError(
                "Stage-3 producer transfer parallelism must equal PCP * TP size: "
                f"parallelism={src_parallelism}, pcp_size={pcp_size}"
            )
        if src_data_replica_idx < 0:
            raise ValueError("Producer data-parallel rank must be non-negative")
        existing = self._stage3_finished_sends.get(request.request_id)
        if existing is not None:
            uuid, old_ids, old_num_tokens, params = existing
            if old_ids != normalized_ids or old_num_tokens != num_tokens:
                logger.error(
                    "Conflicting duplicate Stage-3 request finish for "
                    "req_id=%s; keeping the first registration uuid=%d",
                    request.request_id,
                    uuid,
                )
            self._stage3_finished_sends.move_to_end(request.request_id)
            return True, dict(params)

        # Never evict an in-flight UUID merely to satisfy a memory bound: a
        # delayed duplicate callback could otherwise mint a conflicting UUID
        # while native/D5 state still exists. Scheduler completion removes
        # records through update_connector_output(). Refuse excess concurrency
        # rather than weakening idempotency.
        if len(self._stage3_finished_sends) >= _STAGE3_FINISH_DEDUP_LIMIT:
            logger.warning(
                "Stage-3 in-flight finish dedup capacity exhausted: "
                "limit=%d; skipping the transfer for req_id=%s",
                _STAGE3_FINISH_DEDUP_LIMIT,
                request.request_id,
            )
            return False, {}

        mamba_state_block_ids: list[int] | None = None
        if self._stage3_mamba_group_indices:
            if not mamba_block_ids:
                raise ValueError(
                    "GDN state is present but the producer's "
                    "mamba block tables were not provided"
                )
            mamba_state_block_ids = _select_committed_mamba_blocks(
                mamba_block_ids, self._stage3_mamba_num_speculative_blocks
            )
        uuid = get_uuid()
        now = time.perf_counter()
        expiration_time = now + dist_utils.get_p2p_wait_pull_timeout()
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
            "dst_tp_size": _raiden_dst_shards(),
            "dst_dcp_size": _raiden_dst_dcp_size(self.vllm_config),
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
        parallel_config = self.vllm_config.parallel_config
        if not self.is_producer:
            return int(
                parallel_config.tensor_parallel_size or 1
            ) * _pipeline_parallel_size(self.vllm_config)
        if _uses_tp_stage3_source(self.vllm_config):
            return int(parallel_config.tensor_parallel_size or 1)
        if _engine_is_pipeline(self.vllm_config):
            return _pipeline_parallel_size(self.vllm_config)
        return int(parallel_config.prefill_context_parallel_size or 1) * int(
            parallel_config.tensor_parallel_size or 1
        )


class TPURaidenConnectorWorker:
    """Worker-side integration for the opt-in Raiden transfer backend.

    Python owns vLLM request metadata only. Raiden C++ owns host slots, D2H,
    H2H, H2D, readiness, and completion state.
    """

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.config = vllm_config.kv_transfer_config
        self.is_producer = self.config.is_kv_producer
        self._stage3_kimi = _is_kimi(vllm_config)
        self.runner: TPUModelRunner | None = None
        self.node_id = dist_utils.get_node_id()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.dp_rank: int = (
            vllm_config.parallel_config.data_parallel_rank
            if vllm_config.parallel_config
            else 0
        )
        self.host_ip = dist_utils.get_host_ip()
        self.kv_transfer_port = int(dist_utils.get_kv_transfer_port()) + (
            2 * self.dp_rank * self.tp_size
        )
        self._raiden_transfer_engine: KVCacheManager | None = None
        self.named_kv_caches: dict[str, Any] | None = None
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
        self._stage3_submitted_load_metas: dict[str, _Stage3LoadMeta] = {}
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
        # req_id -> next registry status probe time (perf_counter seconds).
        self._stage3_status_probe_next: dict[str, float] = {}
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
        # Loads parked for per-step re-attempt while their producer
        # request-block registrations are in flight (_Stage3PendingSubmit).
        self._stage3_pending_submits: dict[str, _Stage3PendingSubmit] = {}
        # Collective membership stays identical on every TP rank, including
        # while the leader is between registration retries. Only the leader
        # owns submit tasks, retry clocks and controller RPC deadlines.
        self._stage3_tp_submits: set[str] = set()
        # Stage-3 coordination runs on background submit threads so the
        # multi-hop source-controller RPC never blocks a model-runner step.
        # The model-runner thread stages every scheduler-visible table before
        # enqueueing a task; the workers touch nothing but the facade RPC and
        # hand results back through _stage3_submit_outcomes, drained under
        # _stage3_submit_lock at the top of each poll.
        # queue.Queue for its task tracking: join() fences "every enqueued
        # submission has appended its outcome".
        self._stage3_submit_queue: queue.Queue = queue.Queue()
        self._stage3_submit_lock = threading.Lock()
        self._stage3_submit_outcomes: list[_Stage3SubmitOutcome] = []
        self._stage3_submit_threads: list[threading.Thread] = []
        # Model-runner-thread-only submission lifecycle: deadline per
        # RPC-in-flight request, requests abandoned at that deadline (their
        # late outcomes are discarded), and native terminal records that
        # arrived while the submission RPC was still in flight (replayed once
        # the outcome resolves). Values in the deferred map record whether
        # the parked terminal was a failure.
        self._stage3_inflight_submits: dict[str, float] = {}
        self._stage3_abandoned_submits: set[str] = set()
        self._stage3_deferred_native_failures: dict[str, bool] = {}
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
        dist_utils.configure_raiden_telemetry()
        logger.info(
            "TPURaidenConnectorWorker --> init | ip=%s | base_port=%s | "
            "is_producer=%s | node_id=%s | tp_rank=%d | tp_size=%d | dp_rank=%d",
            self.host_ip,
            self.kv_transfer_port,
            self.is_producer,
            self.node_id,
            self.tp_rank,
            self.tp_size,
            self.dp_rank,
        )

    def _get_raiden_stats(self):
        # Get metrics from raiden library
        try:
            telemetry = dist_utils.get_raiden_telemetry_module()
            raiden_samples = {}
            if telemetry is not None:
                if hasattr(telemetry, "get_and_reset_metric_samples"):
                    raiden_samples = telemetry.get_and_reset_metric_samples()

                for metric_name, values in raiden_samples.items():
                    if metric_name not in self.transfer_stats.data:
                        self.transfer_stats.data[metric_name] = []
                    self.transfer_stats.data[metric_name].extend(values)
        except Exception as e:
            logger.warning("Failed to collect TPU Raiden C++ telemetry: %s", e)

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        self._get_raiden_stats()

        # Instrument corresponding queue lengths
        if self.tp_rank == 0:
            if self.is_producer:
                prefill_queue_len = (
                    len(self._stage3_registered_sends)
                    if self._raiden_stage3_enabled()
                    else len(self._legacy_registered_sends)
                )
                self.transfer_stats.record_prefill_queue_length(prefill_queue_len)
            else:
                decode_queue_len = (
                    len(self._stage3_submitted_loads) - len(self._stage3_terminal_loads)
                    if self._raiden_stage3_enabled()
                    else len(self._legacy_submitted_loads)
                )
                self.transfer_stats.record_decode_queue_length(decode_queue_len)

        if not self.transfer_stats.is_empty():
            return self.transfer_stats.clone_and_reset()
        return None

    def register_runner(self, runner: TPUModelRunner) -> None:
        self.runner = runner
        manager_enabled = bool(tpu_envs.TPU_USE_RAIDEN_KV_CACHE_MANAGER)
        qwen_admission = manager_enabled and tpu_envs.TPU_RAIDEN_QWEN35_ADMISSION
        kimi_admission = manager_enabled and tpu_envs.TPU_RAIDEN_KIMIK3_ADMISSION
        glm_admission = _use_raiden_glm_admission()
        if sum((qwen_admission, kimi_admission, glm_admission)) > 1:
            raise RuntimeError(
                "TPU_RAIDEN_QWEN35_ADMISSION, TPU_RAIDEN_KIMIK3_ADMISSION, "
                "and TPU_RAIDEN_GLM_ADMISSION are mutually exclusive"
            )
        if self._raiden_stage3_enabled() and not (
            qwen_admission or kimi_admission or glm_admission
        ):
            raise RuntimeError(
                "TPU_KV_RESHARD_TRANSPORT=raiden requires explicit pool "
                "admission (TPU_USE_RAIDEN_KV_CACHE_MANAGER=1 and one of "
                "TPU_RAIDEN_KIMIK3_ADMISSION=1, "
                "TPU_RAIDEN_QWEN35_ADMISSION=1, or TPU_RAIDEN_GLM_ADMISSION=1)"
            )
        if kimi_admission and not self._stage3_kimi:
            raise ValueError(
                "TPU_RAIDEN_KIMIK3_ADMISSION requires "
                "KimiK3ForConditionalGeneration or "
                "KimiLinearForCausalLM"
            )
        if kimi_admission and _use_per_layer_pool_tags():
            raise ValueError("Kimi K3 does not support TPU_RAIDEN_POOL_TAGS_PER_LAYER")
        if self._stage3_kimi and (qwen_admission or glm_admission):
            raise ValueError(
                "Kimi K3 requires TPU_RAIDEN_KIMIK3_ADMISSION=1; "
                "Qwen and GLM admission cannot admit Kimi"
            )
        if qwen_admission or kimi_admission:
            self._admit_raiden_hybrid_kv_cache(runner)
            return
        if glm_admission:
            self._admit_raiden_glm_kv_cache(runner)
            return
        self._ensure_raiden_transfer_engine()

    @staticmethod
    def _raiden_stage3_enabled() -> bool:
        return _use_raiden_stage3_transport()

    def _admit_raiden_hybrid_kv_cache(self, runner: TPUModelRunner) -> None:
        """Construct the Raiden engine over the model's hybrid pools."""
        if self._raiden_transfer_engine is not None:
            return

        stage3_enabled = self._raiden_stage3_enabled()
        topology = self._raiden_hybrid_admission_topology()
        head_geometry = None if self._stage3_kimi else self._gdn_head_geometry()
        if stage3_enabled and self.is_producer and head_geometry is not None:
            self._validate_gdn_transfer_geometry(head_geometry)
            _raiden_dst_dcp_size(self.vllm_config)
        controller_address = ""
        if stage3_enabled:
            self._maybe_host_reshard_store()
            controller_address = _resolved_reshard_controller_address(self.dp_rank)
            if not controller_address:
                raise ValueError(
                    "TPU_RAIDEN_CONTROLLER_ADDRESS is required when "
                    "TPU_KV_RESHARD_TRANSPORT=raiden"
                )

        from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm

        role = "kv_producer" if self.is_producer else "kv_consumer"
        named_kv_caches = self.named_kv_caches
        if not named_kv_caches:
            raise ValueError(
                "Raiden pool admission requires "
                "register_kv_caches() before register_runner()"
            )
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
        manifest_args = dict(
            named_kv_caches=named_kv_caches,
            kv_cache_groups=kv_cache_groups,
            raw_tensors=raw_tensors,
            mamba_group_ordinal_by_layer=mamba_group_ordinal_by_layer,
        )
        if self._stage3_kimi:
            manifest = rpm.build_kimi_k3_pool_manifest(**manifest_args)
        else:
            assert head_geometry is not None
            gdn_geometry = rpm.GdnHeadGeometry(
                local_key_heads=head_geometry.local_num_kq_heads,
                local_value_heads=head_geometry.local_num_v_heads,
                key_head_dim=self._model_config_int("linear_key_head_dim", 1),
                value_head_dim=self._model_config_int(
                    "linear_value_head_dim",
                    self._model_config_int("linear_key_head_dim", 1),
                ),
                conv_kernel_size=self._model_config_int("linear_conv_kernel_dim", 4),
            )
            build_manifest = rpm.build_qwen35_pool_manifest
            if _raiden_seq_on_lane_layout():
                from vllm_torchtpu.distributed.kv_transfer.raiden.seq_on_lane import (
                    build_qwen35_pool_manifest_sol,
                )

                build_manifest = build_qwen35_pool_manifest_sol
            manifest = build_manifest(
                **manifest_args,
                per_layer_tags=_use_per_layer_pool_tags(),
                gdn_geometry=gdn_geometry,
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
        logger.info(
            "TPURaidenConnectorWorker rank%d --> registering %d pools with transfer engine",
            self.tp_rank,
            len(manifest.pools),
        )
        summary = self._register_raiden_pools(engine, manifest, kv_cache_groups)
        logger.info(
            "TPURaidenConnectorWorker rank%d --> transfer engine pools registered",
            self.tp_rank,
        )

        if stage3_enabled:
            logger.info(
                "TPURaidenConnectorWorker rank%d --> registering stage3 work unit with %s",
                self.tp_rank,
                controller_address,
            )
            registration = self._register_raiden_stage3_work_unit(
                engine=engine,
                manifest=manifest,
                controller_address=controller_address,
            )
            summary["stage3_registration"] = registration
            logger.info(
                "TPURaidenConnectorWorker rank%d --> stage3 work unit registered successfully",
                self.tp_rank,
            )

        self._raiden_transfer_engine = engine
        self._raiden_manifest = manifest
        counts = manifest.tag_counts()
        geometry = manifest.geometry_by_tag()
        if self._stage3_kimi:
            fa_pool = next(pool for pool in manifest.pools if pool.tag == rpm.TAG_FA)
            self._stage3_row_geometry = {
                rpm.TAG_FA: (
                    int(geometry[rpm.TAG_FA]["live_bytes_per_block"]),
                    int(fa_pool.regions[0].unit_bytes),
                )
            }

        gdn_conv_count = sum(
            count
            for tag, count in counts.items()
            if str(tag).startswith(rpm.TAG_GDN_CONV)
        )
        gdn_ssm_count = sum(
            count
            for tag, count in counts.items()
            if str(tag).startswith(rpm.TAG_GDN_SSM)
        )
        summary_geometry = {tag: dict(geo) for tag, geo in geometry.items()}
        summary.update(
            {
                "topology": topology,
                "model_server_role": role,
                "binding": manifest.binding,
                "tag_counts": dict(counts),
                "geometry": summary_geometry,
            }
        )
        self._raiden_admission_summary = dict(summary)

        logger.info(
            "Raiden pool admission complete topology=%s role=%s binding=%s "
            "pools=%d storages=%d fa=%d gdn.conv=%d gdn.ssm=%d",
            topology,
            role,
            manifest.binding,
            len(manifest.pools),
            len(manifest.storages),
            len(manifest.pools_of_class(rpm.TAG_FA)),
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

    def _admit_raiden_glm_kv_cache(self, runner: TPUModelRunner) -> None:
        """Constructs Raiden engine over the GLM MLA cache classes
        (MLA latent + DSA indexer)."""
        if self._raiden_transfer_engine is not None:
            return
        if not self._raiden_stage3_enabled():
            raise RuntimeError(
                "TPU_RAIDEN_GLM_ADMISSION requires TPU_KV_RESHARD_TRANSPORT=raiden"
            )
        self._maybe_host_reshard_store()
        controller_address = _resolved_reshard_controller_address(self.dp_rank)
        if not controller_address:
            raise ValueError(
                "TPU_RAIDEN_CONTROLLER_ADDRESS is required when "
                "TPU_KV_RESHARD_TRANSPORT=raiden"
            )

        from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm

        topology = self._raiden_glm_admission_topology()
        role = "kv_producer" if self.is_producer else "kv_consumer"
        named_kv_caches = self.named_kv_caches
        if not named_kv_caches:
            raise ValueError(
                "Raiden pool admission requires "
                "register_kv_caches() before register_runner()"
            )
        raw_tensors = tuple(runner.kv_cache_raw_tensors or ())

        manifest = rpm.build_glm_mla_pool_manifest(
            named_kv_caches=named_kv_caches,
            raw_tensors=raw_tensors,
            block_size_tokens=int(self.vllm_config.cache_config.block_size),
        )
        rpm.verify_storage_binding(
            manifest=manifest,
            named_kv_caches=named_kv_caches,
            raw_tensors=raw_tensors,
        )
        rpm.materialize_storages(manifest)

        storages = list(manifest.storages)
        engine = self._construct_raiden_transfer_engine(storages, num_slots=1)
        summary = self._register_raiden_pools(
            engine,
            manifest,
            getattr(getattr(runner, "kv_cache_config", None), "kv_cache_groups", None),
        )

        registration = self._register_raiden_stage3_work_unit(
            engine=engine,
            manifest=manifest,
            controller_address=controller_address,
        )
        summary["stage3_registration"] = registration

        self._raiden_transfer_engine = engine
        self._raiden_manifest = manifest
        # Per-tag (live_bytes_per_block, row_bytes) is fixed by the admitted
        # manifest; cache it here because the producer's span lowering reads it every step.
        self._stage3_row_geometry = self._measure_glm_tag_geometry(manifest)

        counts = manifest.tag_counts()
        geometry = manifest.geometry_by_tag()
        summary_geometry = {tag: dict(geo) for tag, geo in geometry.items()}
        summary.update(
            {
                "topology": topology,
                "model_server_role": role,
                "binding": manifest.binding,
                "tag_counts": dict(counts),
                "geometry": summary_geometry,
            }
        )
        self._raiden_admission_summary = dict(summary)
        logger.info(
            "Raiden GLM pool admission complete topology=%s role=%s "
            "binding=%s pools=%d storages=%d tag_counts=%s",
            topology,
            role,
            manifest.binding,
            len(manifest.pools),
            len(manifest.storages),
            dict(counts),
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

    def _raiden_glm_admission_topology(self) -> str:
        """Validates the one supported topology. Anything else fails closed here."""
        parallel_config = self.vllm_config.parallel_config
        pcp_size = int(parallel_config.prefill_context_parallel_size or 1)
        tp_size = int(self.tp_size)
        dp_size = int(parallel_config.data_parallel_size or 1)
        supported = (
            "Raiden GLM admission supports prefill tp{N}/dp1 -> decode tp1/dp{M} only"
        )
        if pcp_size != 1:
            raise ValueError(
                f"{supported}; got prefill_context_parallel_size={pcp_size}"
            )
        if self.is_producer:
            if dp_size != 1:
                raise ValueError(
                    f"{supported}; got producer "
                    f"tensor_parallel_size={tp_size}, "
                    f"data_parallel_size={dp_size}"
                )
            return f"tp{tp_size}_prefill"
        if tp_size != 1:
            raise ValueError(
                f"{supported}; got consumer "
                f"tensor_parallel_size={tp_size}, "
                f"data_parallel_size={dp_size}"
            )
        return f"dp{dp_size}_decode"

    @staticmethod
    def _measure_raiden_fa_layout(
        manifest: Any,
    ) -> tuple[str, dict[str, Any]]:
        from vllm_torchtpu.distributed.kv_transfer.raiden.layout_fingerprint import (
            measured_fa_layout_fingerprint,
        )

        return measured_fa_layout_fingerprint(manifest)

    @staticmethod
    def _measure_raiden_glm_layout(
        manifest: Any, page_tokens: int
    ) -> tuple[str, dict[str, Any]]:
        from vllm_torchtpu.distributed.kv_transfer.raiden.layout_fingerprint import (
            measured_glm_layout_fingerprint,
        )

        return measured_glm_layout_fingerprint(manifest, page_tokens=page_tokens)

    def _measure_stage3_layout(self, manifest: Any) -> tuple[str, dict[str, Any], int]:
        """Layout identity and page geometry of the admitted model."""
        if self._stage3_kimi:
            from vllm_torchtpu.distributed.kv_transfer.raiden.layout_fingerprint import (
                measured_kimi_k3_layout_fingerprint,
            )

            page_tokens = int(self.vllm_config.cache_config.block_size)
            config = self.vllm_config.model_config.hf_text_config
            total_heads = int(config.linear_attn_config.get("num_heads", 0))
            source_tp = int(tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM)
            if (
                total_heads <= 0
                or source_tp <= 0
                or total_heads % source_tp
                or source_tp % self.tp_size
            ):
                raise ValueError(
                    "Kimi source and destination TP must form complete KDA "
                    f"head shards: heads={total_heads}, "
                    f"source_tp={source_tp}, destination_tp={self.tp_size}"
                )
            fingerprint, payload = measured_kimi_k3_layout_fingerprint(
                manifest,
                page_tokens=page_tokens,
                state_fragment_heads=total_heads // source_tp,
                state_fragments=source_tp // self.tp_size,
            )
            return fingerprint, payload, page_tokens
        if _use_raiden_glm_admission():
            page_tokens = int(self.vllm_config.cache_config.block_size)
            fingerprint, payload = self._measure_raiden_glm_layout(
                manifest, page_tokens
            )
            return fingerprint, payload, page_tokens
        if _raiden_seq_on_lane_layout():
            from vllm_torchtpu.distributed.kv_transfer.raiden.seq_on_lane import (  # noqa: E501
                measured_sol_layout_fingerprint,
                sol_fa_page_tokens,
            )

            fingerprint, payload = measured_sol_layout_fingerprint(manifest)
            return fingerprint, payload, sol_fa_page_tokens(manifest)
        from vllm_torchtpu.distributed.kv_transfer.raiden.layout_fingerprint import (
            fa_page_tokens,
        )

        fingerprint, payload = self._measure_raiden_fa_layout(manifest)
        return fingerprint, payload, fa_page_tokens(manifest)

    def _validate_stage3_transfer_parallelism(self, transfer_parallelism: int) -> None:
        """The producer must shard the transfer over exactly the rank set its
        byte-span lowering assumes: TP ranks for GLM, PCP ranks otherwise."""
        if _uses_tp_stage3_source(self.vllm_config):
            if transfer_parallelism != self.tp_size:
                raise ValueError(
                    "TP producer transfer parallelism must equal the "
                    "complete TP rank count: "
                    f"parallelism={transfer_parallelism}, "
                    f"tp_size={self.tp_size}"
                )
            return
        if _engine_is_pipeline(self.vllm_config):
            pp_size = _pipeline_parallel_size(self.vllm_config)
            if transfer_parallelism != pp_size:
                raise ValueError(
                    "Pipeline producer transfer parallelism must equal the "
                    "complete PP stage count: "
                    f"parallelism={transfer_parallelism}, pp_size={pp_size}"
                )
            if not _use_per_layer_pool_tags():
                raise ValueError(
                    "A pipeline-parallel producer pairs pools by layer; set "
                    "TPU_RAIDEN_POOL_TAGS_PER_LAYER=1 on both peers"
                )
            return
        pcp_size = int(
            self.vllm_config.parallel_config.prefill_context_parallel_size or 1
        )
        if transfer_parallelism != pcp_size * self.tp_size:
            raise ValueError(
                "Producer transfer parallelism must equal the complete "
                "PCP * TP rank count: "
                f"parallelism={transfer_parallelism}, pcp_size={pcp_size}"
            )

    @staticmethod
    def _new_raiden_controller_facade(controller_address: str) -> Any:
        # The reshard surface moved off the Python controller and behind this
        # client, which fronts the native implementation. The work-unit and
        # request-block calls below keep the names and arguments they had.
        from tpu_sync.api.torch.reshard_client import ReshardClient

        return ReshardClient(controller_address)

    @staticmethod
    def _new_raiden_id(fields: dict[str, Any]) -> Any:
        rpc = importlib.import_module("tpu_sync.rpc.raiden_controller")
        return rpc.RaidenId(**fields)

    @staticmethod
    def _new_raiden_manager(**kwargs: Any) -> "KVCacheManager":
        from tpu_sync.api.torch.kv_cache_manager import KVCacheManager

        return KVCacheManager(**kwargs)

    def _raiden_transfer_parallelism(self) -> int:
        parallelism = int(tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM)
        if parallelism <= 0:
            raise ValueError("TPU_RAIDEN_TRANSFER_PARALLELISM must be positive")
        return parallelism

    def _local_raiden_transfer_rank(self) -> int:
        if _engine_is_pipeline(self.vllm_config):
            from vllm.distributed.parallel_state import get_pp_group

            return int(get_pp_group().rank_in_group)
        if not self.is_producer:
            return int(self.tp_rank)
        if _uses_tp_stage3_source(self.vllm_config):
            return int(self.tp_rank)
        from vllm_torchtpu.distributed.pcp import get_pcp_cache_rank, get_pcp_rank

        if self.tp_size == 1:
            return int(get_pcp_cache_rank())
        # A work-unit identity enumerates workers, not GDN head shards. FA
        # lowering consumes PCP and TP coordinates separately; GDN uses the
        # transposed (TP, PCP) head order. Include both axes here so TP peers
        # cannot overwrite each other's source registration.
        return int(get_pcp_rank()) * self.tp_size + self.tp_rank

    def _raiden_engine_id(self) -> str:
        env_id = str(tpu_envs.TPU_RAIDEN_ENGINE_ID).strip()
        if env_id:
            return env_id
        config_id = getattr(
            getattr(self.vllm_config, "kv_transfer_config", None),
            "engine_id",
            None,
        )
        if config_id:
            return str(config_id).strip()
        default_job = "prefill" if self.is_producer else "decode"
        return f"{default_job}-engine"

    def _maybe_host_reshard_store(self) -> None:
        """In-engine hosting (zero sidecars): each engine's transfer-rank-0
        worker owns the in-process reshard store — the framed reshard
        service plus the dispatch controller its local workers register
        with. Peer ranks only register into it, with bounded retries."""
        if not _reshard_store_mode() or self._reshard_store is not None:
            return
        if self._local_raiden_transfer_rank() != 0:
            return
        from tpu_sync.api.torch import reshard_store as _reshard_store_mod

        default_job = "prefill" if self.is_producer else "decode"
        job_name = str(tpu_envs.TPU_RAIDEN_JOB_NAME).strip() or default_job
        store_kwargs: dict[str, Any] = dict(
            raiden_id=_reshard_store_mod.RaidenId(
                job_name=job_name,
                job_replica_id=self._raiden_engine_id(),
                data_name="reshard_store",
                data_replica_idx=self.dp_rank,
            ),
            store_server_ip=_reshard_advertise_host(),
            raiden_controller_port=(
                int(tpu_envs.TPU_RAIDEN_STORE_DISPATCH_PORT_BASE) + int(self.dp_rank)
            ),
            reshard_service_port=_reshard_service_port(self.dp_rank),
        )
        # The registry must outlive the prefill lease. A producer pins its
        # KV blocks for TPU_P2P_WAIT_PULL_TIMEOUT, and a registration purged
        # before that will cause every later decoder pull fail.
        registry_ttl_s = float(dist_utils.get_raiden_registry_ttl_s())
        lease_s = int(dist_utils.get_p2p_wait_pull_timeout())
        if getattr(_reshard_store_mod, "SUPPORTS_REQUEST_REGISTRY_TTL", False):
            self._reshard_store = _reshard_store_mod.ReshardStore(
                request_registry_ttl_s=registry_ttl_s, **store_kwargs
            )
        else:
            # For older tpu_sync wheels the registry TTL is hard-coded at 600s.
            self._reshard_store = _reshard_store_mod.ReshardStore(**store_kwargs)
            registry_ttl_s = float(
                getattr(self._reshard_store, "request_registry_ttl_s", 600.0) or 600.0
            )
            if registry_ttl_s < lease_s:
                logger.warning(
                    "Installed tpu_sync ReshardStore does not accept "
                    "request_registry_ttl_s; its registry TTL stays at "
                    "%.0fs while TPU_P2P_WAIT_PULL_TIMEOUT=%ds. Consumer "
                    "pulls issued more than %.0fs after prefill will fail."
                    " Consider upgrading tpu_sync to set the registry TTL.",
                    registry_ttl_s,
                    lease_s,
                    registry_ttl_s,
                )
        logger.info(
            "TPURaidenConnectorWorker rank%d --> hosting reshard store "
            "service=%s dispatch=%s registry_ttl_s=%.0f (lease %ds)",
            self.tp_rank,
            _reshard_service_address(self.dp_rank),
            _reshard_dispatch_address(self.dp_rank),
            registry_ttl_s,
            lease_s,
        )

    def _raiden_interleave_tokens(
        self, page_tokens: int, transfer_parallelism: int
    ) -> int:
        """Returns the kernel interleave driving byte-span lowering."""
        page_tokens = int(page_tokens)
        transfer_parallelism = int(transfer_parallelism)
        if page_tokens <= 0:
            raise ValueError("Raiden page_tokens must be positive")
        if transfer_parallelism <= 0:
            raise ValueError("Raiden transfer_parallelism must be positive")
        # A TP1 destination is logically contiguous even if an unrelated PCP
        # option remains present in its shared ParallelConfig, and so is a
        # pipeline stage, which holds every token of its own layers.
        if (
            not self.is_producer
            or transfer_parallelism == 1
            or _engine_is_pipeline(self.vllm_config)
        ):
            return page_tokens
        parallel_config = self.vllm_config.parallel_config
        if self._stage3_kimi:
            return page_tokens
        interleave_tokens = int(parallel_config.cp_kv_cache_interleave_size or 0)
        if interleave_tokens <= 0:
            raise ValueError(
                "Stage-3 PCP source requires a positive cp_kv_cache_interleave_size"
            )
        if page_tokens % interleave_tokens:
            raise ValueError(
                "Stage-3 PCP page geometry requires page_tokens divisible by "
                "cp_kv_cache_interleave_size: "
                f"page_tokens={page_tokens}, "
                f"cp_kv_cache_interleave_size={interleave_tokens}"
            )
        return interleave_tokens

    def _raiden_work_unit_fields(self, transfer_rank: int) -> dict[str, Any]:
        default_job = "prefill" if self.is_producer else "decode"
        job_name = str(tpu_envs.TPU_RAIDEN_JOB_NAME).strip() or default_job
        engine_id = self._raiden_engine_id()
        # A TP>1 decode engine registers one unit per TP worker (transfer
        # rank = tp_rank); the replica label must tell them apart or the
        # coordinator rejects the leader's dst_units as duplicates.
        return stage3_fa_raiden_id_fields(
            job_name=job_name,
            engine_id=engine_id,
            dp_rank=self.dp_rank,
            transfer_rank=transfer_rank,
            is_producer=self.is_producer,
            per_rank_unit=(
                self.is_producer
                or int(self.tp_size) > 1
                or _engine_is_pipeline(self.vllm_config)
            ),
        )

    def _register_raiden_stage3_work_unit(
        self,
        *,
        engine: Any,
        manifest: Any,
        controller_address: str,
    ) -> dict[str, Any]:
        """Measure and register this worker with its cluster controller."""
        data_address = str(getattr(engine, "transfer_address", "")).strip()
        listener_address = str(getattr(engine, "listener_address", "")).strip()
        if not data_address:
            raise RuntimeError(
                "Stage-3 Raiden manager did not advertise a data endpoint"
            )
        if not listener_address:
            raise RuntimeError(
                "Stage-3 Raiden manager did not advertise a listener endpoint"
            )

        fingerprint, fingerprint_payload, page_tokens = self._measure_stage3_layout(
            manifest
        )
        transfer_parallelism = self._raiden_transfer_parallelism()
        transfer_rank = self._local_raiden_transfer_rank()
        interleave_tokens = self._raiden_interleave_tokens(
            page_tokens, transfer_parallelism
        )
        if self.is_producer:
            self._validate_stage3_transfer_parallelism(transfer_parallelism)
        if transfer_rank < 0 or transfer_rank >= transfer_parallelism:
            raise ValueError(
                "Raiden transfer rank is outside the admitted parallelism: "
                f"rank={transfer_rank}, parallelism={transfer_parallelism}"
            )

        unit = self._new_raiden_id(self._raiden_work_unit_fields(transfer_rank))
        if _reshard_store_mode():
            self._wait_for_address_ready(controller_address, timeout_s=60.0)
        facade = self._new_raiden_controller_facade(controller_address)
        # Store mode: the store is brought up by this engine's rank-0
        # worker in parallel with the peer ranks' init — bounded retry
        # instead of the sidecar era's launcher-ordered readiness.
        registration_deadline = time.perf_counter() + (
            120.0 if _reshard_store_mode() else 0.0
        )
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
                    "(%s); retrying",
                    controller_address,
                    exc,
                )
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

    def _raiden_hybrid_admission_topology(self) -> str:
        if self._stage3_kimi:
            return self._raiden_kimi_k3_admission_topology()
        return self._raiden_qwen35_admission_topology()

    def _raiden_kimi_k3_admission_topology(self) -> str:
        dst_shards = tpu_envs.TPU_RAIDEN_DST_SHARDS
        if dst_shards is None:
            raise ValueError(
                "TPU_RAIDEN_DST_SHARDS must be explicitly set to the decode "
                "TP size on both prefill and decode workers for Kimi Stage-3"
            )
        parallel_config = self.vllm_config.parallel_config
        pcp_size = int(parallel_config.prefill_context_parallel_size or 1)
        tp_size = int(parallel_config.tensor_parallel_size or self.tp_size)
        dp_size = int(parallel_config.data_parallel_size or 1)
        pp_size = _pipeline_parallel_size(self.vllm_config)
        source_parallelism = self._raiden_transfer_parallelism()
        expected_tp = source_parallelism if self.is_producer else dst_shards
        if (
            dst_shards <= 0
            or pcp_size != 1
            or tp_size != expected_tp
            or pp_size != 1
            or (self.is_producer and dp_size != 1)
            or source_parallelism % dst_shards
        ):
            raise ValueError(
                "Kimi Stage-3 requires PCP1, producer TP equal to "
                "TPU_RAIDEN_TRANSFER_PARALLELISM, destination TP equal "
                "to TPU_RAIDEN_DST_SHARDS, PP1, and producer DP1: "
                f"tp={tp_size}, pcp={pcp_size}, pp={pp_size}, dp={dp_size}, "
                f"source_parallelism={source_parallelism}, "
                f"dst_shards={dst_shards}"
            )
        role = "prefill" if self.is_producer else "decode"
        return f"tp{tp_size}dp{dp_size}_{role}"

    def _raiden_qwen35_admission_topology(self) -> str:
        parallel_config = self.vllm_config.parallel_config
        pcp_size = int(parallel_config.prefill_context_parallel_size or 1)
        tp_size = int(parallel_config.tensor_parallel_size or self.tp_size)
        dp_size = int(parallel_config.data_parallel_size or 1)
        pp_size = _pipeline_parallel_size(self.vllm_config)
        if _raiden_seq_on_lane_layout() and dp_size == pp_size == 1:
            dcp_size = int(parallel_config.decode_context_parallel_size)
            if dcp_size <= 0 or tp_size % dcp_size:
                raise ValueError("DCP size must divide TP size")
            if self.is_producer and tp_size > 1 and dcp_size == 1:
                return f"pcp{pcp_size}tp{tp_size}_prefill"
            if not self.is_producer and pcp_size == 1 and tp_size > 1:
                return f"tp{tp_size}dcp{dcp_size}_decode"
        if self.is_producer:
            if tp_size == 1 and pcp_size == 1 and dp_size == 1 and pp_size in (2, 4, 8):
                return f"pp{pp_size}_prefill"
            if tp_size == 1 and pcp_size in (4, 8) and dp_size == 1:
                return f"pcp{pcp_size}_prefill"
            if tp_size == 1 and pcp_size == 1 and dp_size in (4, 8, 16):
                return f"dp{dp_size}_prefill"
            raise ValueError(
                "Raiden Qwen3.5 admission topology pcp8_prefill, "
                "pcp4_prefill, dp8_prefill, dp4_prefill, dp16_prefill, or "
                "pp{2,4,8}_prefill requires kv_producer with "
                "tensor_parallel_size=1 and either "
                "prefill_context_parallel_size in (4, 8), "
                "data_parallel_size=1; or "
                "prefill_context_parallel_size=1, "
                "data_parallel_size in (4, 8, 16); or "
                "pipeline_parallel_size in (2, 4, 8) alone; got "
                f"prefill_context_parallel_size={pcp_size}, "
                f"tensor_parallel_size={tp_size}, "
                f"data_parallel_size={dp_size}, "
                f"pipeline_parallel_size={pp_size}"
            )
        if tp_size == 1 and pcp_size == 1 and dp_size == 1 and pp_size in (2, 4, 8):
            return f"pp{pp_size}_decode"
        if pcp_size != 1 or tp_size not in (1, 2) or dp_size not in (2, 4, 8, 16):
            raise ValueError(
                "Raiden Qwen3.5 admission topology dp{N}_decode or "
                "dp{N}tp2_decode requires kv_consumer with "
                "prefill_context_parallel_size=1, tensor_parallel_size in "
                "(1, 2), data_parallel_size in (2, 4, 8, 16); got "
                f"prefill_context_parallel_size={pcp_size}, "
                f"tensor_parallel_size={tp_size}, "
                f"data_parallel_size={dp_size}"
            )
        if tp_size > 1:
            # Head-sharded decode engine: every TP worker is its own
            # destination unit (transfer_rank = tp_rank) and the producer
            # routes each KV head shard to it (TPU_RAIDEN_DST_SHARDS).
            return f"dp{dp_size}tp{tp_size}_decode"
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
            raise ValueError(
                f"total_heads={total_heads} must be divisible by tp_size={tp_size}"
            )
        return total_heads // tp_size

    def _gdn_head_geometry(self) -> GdnHeadGeometry:
        """Use the compute path's QK replication and V-head ownership."""
        parallel_config = self.vllm_config.parallel_config
        degree = int(parallel_config.prefill_context_parallel_size or 1) * int(
            parallel_config.tensor_parallel_size or self.tp_size
        )
        return derive_gdn_head_geometry(
            self._model_config_int("linear_num_key_heads", 0),
            self._model_config_int("linear_num_value_heads", 0),
            degree,
        )

    def _validate_gdn_transfer_geometry(self, source: GdnHeadGeometry) -> None:
        """Reject state mappings the byte-span lowering cannot represent."""
        destination = derive_gdn_head_geometry(
            source.num_kq_heads, source.num_v_heads, _raiden_dst_shards()
        )
        # The current lowerer concatenates source regions for fan-in, or
        # splits a full-width source for fan-out. Neither operation removes
        # or creates Q/K replicas, unlike logical head-based resharding.
        if source.kq_replication_factor != destination.kq_replication_factor:
            raise ValueError(
                "GDN state transfer cannot change Q/K replication: "
                f"source={source.kq_replication_factor}, "
                f"destination={destination.kq_replication_factor}"
            )
        if (
            source.parallel_size != 1
            and source.parallel_size % destination.parallel_size
        ):
            raise ValueError(
                "GDN destination shards must divide producer parallelism "
                "unless the producer holds the full state: "
                f"source={source.parallel_size}, "
                f"destination={destination.parallel_size}"
            )

    def process_send_load(
        self,
        metadata: TPUConnectorMetadata,
        wait_for_completion: bool = False,
        report_completion: bool = True,
    ) -> None:
        engine = self._ensure_raiden_transfer_engine()
        if self.is_producer:
            if self._raiden_stage3_enabled():
                if _uses_tp_stage3_source(self.vllm_config):
                    self._register_stage3_request_row_spans(metadata)
                else:
                    self._register_stage3_request_blocks(metadata)
                return
            for req_id, req_meta in metadata.reqs_to_send.items():
                engine.register_read(req_id, req_meta.uuid, req_meta.local_block_ids)
                self._legacy_registered_sends.add(str(req_id))
                logger.debug(
                    "TPURaidenConnectorWorker rank%d --> registered send "
                    "req_id=%s uuid=%s blocks=%d",
                    self.tp_rank,
                    req_id,
                    req_meta.uuid,
                    len(req_meta.local_block_ids),
                )
            return

        if self._raiden_stage3_enabled():
            # Inline mode waits for the loads below, so the coordination RPCs
            # must have run before the wait loop starts polling.
            self._submit_stage3_loads(metadata, engine, synchronous=wait_for_completion)
            submitted_loads = set(metadata.reqs_to_load)
            if wait_for_completion:
                self._wait_for_recving(submitted_loads)
                if not report_completion:
                    self._suppress_done_recving.update(submitted_loads)
            return

        submitted_loads: set[str] = set()
        for req_id, req_meta in metadata.reqs_to_load.items():
            if req_meta.fail_only:
                local_blocks = list(req_meta.local_block_ids or ())
                self._load_block_ids[req_id] = local_blocks
                self._failed_recving.add(req_id)
                self._failed_block_ids.update(local_blocks)
                submitted_loads.add(req_id)
                continue
            endpoint = self._resolve_remote_endpoint(req_meta)
            if req_meta.remote_block_ids is None and req_meta.local_block_ids is None:
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
                    self.tp_rank,
                    req_id,
                    req_meta.uuid,
                    endpoint,
                    req_meta.report_completion,
                )
                continue
            if req_meta.remote_block_ids is None or req_meta.local_block_ids is None:
                raise ValueError(
                    "TPURaidenConnector load metadata must contain both "
                    "remote and local block ids, or neither"
                )

            remote_blocks = req_meta.remote_block_ids
            local_blocks = req_meta.local_block_ids
            engine.start_read(
                req_id, req_meta.uuid, endpoint, remote_blocks, local_blocks
            )
            logger.debug(
                "TPURaidenConnectorWorker rank%d --> submitted load "
                "req_id=%s uuid=%s endpoint=%s remote_blocks=%d "
                "local_blocks=%d",
                self.tp_rank,
                req_id,
                req_meta.uuid,
                endpoint,
                len(remote_blocks),
                len(local_blocks),
            )
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
                "registration with an explicit destination controller"
            )
        return facade, address

    @staticmethod
    def _maybe_delay_stage3_registration_for_test(
        metadata: TPUConnectorMetadata,
    ) -> None:
        """TEST-ONLY fault injection (TPU_RAIDEN_TEST_REGISTRATION_DELAY_S):
        widen the out-of-band registration race by delaying the producer's
        registration step, reproducing a saturated producer whose
        registration-carrying step runs long after the finish response."""
        delay_s = dist_utils.get_stage3_test_registration_delay_s()
        if delay_s > 0 and metadata.reqs_to_send:
            logger.warning(
                "TEST INJECTION: delaying Stage-3 request-block registration by "
                "%.1fs for req_ids=%s",
                delay_s,
                sorted(metadata.reqs_to_send),
            )
            time.sleep(delay_s)

    def _register_stage3_request_blocks(self, metadata: TPUConnectorMetadata) -> None:
        """Registers only this PCP rank's interleaved physical source blocks."""
        facade, _ = self._require_stage3_controller()
        if self._raiden_work_unit is None:
            raise RuntimeError("Stage-3 producer work unit is not registered")
        parallelism = self._raiden_transfer_parallelism()
        transfer_rank = self._local_raiden_transfer_rank()
        if transfer_rank < 0 or transfer_rank >= parallelism:
            raise ValueError(
                "Stage-3 producer transfer rank is outside parallelism: "
                f"rank={transfer_rank}, parallelism={parallelism}"
            )
        self._maybe_delay_stage3_registration_for_test(metadata)
        # A pipeline stage holds every token of its own layers: its byte
        # spans are lowered as a single contiguous owner, and the layer
        # subset is expressed through per-layer pool tags instead.
        pipeline = _engine_is_pipeline(self.vllm_config)
        lowering_parallelism = 1 if pipeline else parallelism // self.tp_size
        lowering_rank = 0 if pipeline else transfer_rank // self.tp_size

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
                raise ValueError("Stage-3 producer send metadata requires num_tokens")
            if expiration_time <= 0:
                raise ValueError(
                    "Stage-3 producer send metadata requires a positive expiration_time"
                )
            page_tokens = int(self.vllm_config.cache_config.block_size)
            interleave_tokens = self._raiden_interleave_tokens(
                page_tokens, lowering_parallelism
            )
            scheduler_block_tokens = page_tokens * lowering_parallelism
            expected_scheduler_blocks = (
                num_tokens + scheduler_block_tokens - 1
            ) // scheduler_block_tokens
            scheduler_ids = tuple(
                int(block_id) for block_id in req_meta.local_block_ids
            )
            if len(scheduler_ids) != expected_scheduler_blocks:
                raise ValueError(
                    "Stage-3 producer scheduler block count is inconsistent "
                    f"with PCP geometry: blocks={len(scheduler_ids)}, "
                    f"expected={expected_scheduler_blocks}, "
                    f"num_tokens={num_tokens}, page_tokens={page_tokens}, "
                    f"interleave_tokens={interleave_tokens}, "
                    f"parallelism={parallelism}"
                )
            # This rank's declared source map, lowered from the kernel's own
            # layout package to the byte-span IR: owned interleave slices
            # packed dense rank-major over this rank's blocks (a prefix of
            # the shared scheduler BlockTable ids), split at destination
            # page boundaries, scaled to bytes. The producer-side semantic
            # checks live in this construction and Raiden's byte-level
            # registration/planning validation.
            from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import (  # noqa: E501
                lower_fa_spans,
                owned_token_ranges,
            )

            owned_tokens = sum(
                end - start
                for start, end in owned_token_ranges(
                    num_tokens=num_tokens,
                    transfer_rank=lowering_rank,
                    parallelism=lowering_parallelism,
                    interleave_tokens=interleave_tokens,
                )
            )
            owned_blocks = (owned_tokens + page_tokens - 1) // page_tokens
            local_ids = scheduler_ids[:owned_blocks]
            fa_pool_spans = []
            dst_shards = _raiden_dst_shards()
            if _raiden_seq_on_lane_layout():
                from vllm_torchtpu.distributed.kv_transfer.raiden.seq_on_lane import (  # noqa: E501
                    lower_fa_spans_sol,
                    sol_fa_geometry_from_manifest,
                )

                fa_registration = lower_fa_spans_sol(
                    num_tokens=num_tokens,
                    transfer_rank=lowering_rank,
                    parallelism=lowering_parallelism,
                    interleave_tokens=interleave_tokens,
                    page_tokens=page_tokens,
                    geometry=sol_fa_geometry_from_manifest(self._raiden_manifest),
                    dst_shards=dst_shards,
                    block_ids=list(local_ids),
                    producer_tp_size=self.tp_size,
                    producer_tp_rank=self.tp_rank,
                    total_kv_heads=self._model_config_int("num_key_value_heads", 0),
                    dst_dcp_size=_raiden_dst_dcp_size(self.vllm_config),
                )
            else:
                if dst_shards != 1 or self.tp_size != 1:
                    raise ValueError(
                        "TPU_RAIDEN_DST_SHARDS > 1 requires the seq-along-lane "
                        "FA layout (head shards are not placement-exact on "
                        "the token-major layout)"
                    )
                token_bytes = self._stage3_fa_token_bytes(page_tokens)
                fa_registration = lower_fa_spans(
                    num_tokens=num_tokens,
                    transfer_rank=lowering_rank,
                    parallelism=lowering_parallelism,
                    interleave_tokens=interleave_tokens,
                    page_tokens=page_tokens,
                    token_bytes=token_bytes,
                    block_ids=list(local_ids),
                )
            if fa_registration.spans:
                # One identical declaration per FA tag this rank holds
                # (one per FA layer under per-layer tags).
                fa_pool_spans = [
                    dataclasses.replace(fa_registration, tag=tag)
                    for tag in self._stage3_fa_registration_tags()
                ]
            terminal = self._stage3_terminal_sends.get(req_id)
            if terminal is not None:
                if (
                    terminal.uuid != uuid
                    or terminal.local_block_ids != local_ids
                    or terminal.num_tokens != num_tokens
                ):
                    raise ValueError(
                        "Conflicting replay after terminal Stage-3 producer "
                        f"registration for req_id={req_id}"
                    )
                continue
            existing = self._stage3_registered_sends.get(req_id)
            if existing is not None:
                if (
                    existing.uuid != uuid
                    or existing.local_block_ids != local_ids
                    or existing.num_tokens != num_tokens
                ):
                    raise ValueError(
                        "Conflicting duplicate Stage-3 producer registration "
                        f"for req_id={req_id}"
                    )
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
                        lowering_rank,
                        lowering_parallelism,
                    )
                )
                with TraceAnnotation(
                    "KV_Cache_Prefill_Register_Blocks",
                    uuid=uuid,
                    request_id=str(req_id),
                    num_tokens=num_tokens,
                    num_blocks=len(local_ids),
                ):
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
                tombstone_deadline = time.perf_counter() + float(
                    dist_utils.get_p2p_wait_pull_timeout()
                )
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
            if not any(reg.spans for reg in pool_spans):
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
                        "req_id": req_id,
                        "uuid": uuid,
                        "num_tokens": num_tokens,
                        "transfer_rank": transfer_rank,
                        "parallelism": parallelism,
                        "page_tokens": page_tokens,
                        "interleave_tokens": interleave_tokens,
                        "scheduler_blocks": expected_scheduler_blocks,
                        "owned_blocks": len(local_ids),
                        "owned_tokens": owned_tokens,
                        "declared_spans": (
                            len(fa_pool_spans[0].spans) if fa_pool_spans else 0
                        ),
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
    def _measure_glm_tag_geometry(manifest: Any) -> dict[str, tuple[int, int]]:
        """Per-tag (live_bytes_per_block, row_bytes) of an admitted GLM
        manifest."""
        geometry = manifest.geometry_by_tag()
        result: dict[str, tuple[int, int]] = {}
        for tag in _STAGE3_GLM_TRANSFER_POOL_TAGS:
            geo = geometry.get(tag)
            if not geo:
                raise RuntimeError(f"Stage-3 GLM manifest is missing tag {tag!r}")
            pool = next(p for p in manifest.pools if p.tag == tag)
            (region,) = pool.regions
            result[tag] = (int(geo["live_bytes_per_block"]), int(region.unit_bytes))
        return result

    def _register_stage3_request_row_spans(
        self, metadata: TPUConnectorMetadata
    ) -> None:
        """Register this producer rank's row-granular span stripe.

        Kimi reuses the GLM lowering for its replicated MLA cache and adds
        one complete source fragment for each KDA state group.
        """
        from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import (
            lower_glm_row_spans,
        )

        facade, _ = self._require_stage3_controller()
        if self._raiden_work_unit is None:
            raise RuntimeError("Stage-3 producer work unit is not registered")
        parallelism = self._raiden_transfer_parallelism()
        transfer_rank = self._local_raiden_transfer_rank()
        self._maybe_delay_stage3_registration_for_test(metadata)

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

        tag_geometry = self._stage3_row_geometry
        page_tokens = int(self.vllm_config.cache_config.block_size)
        for req_id, req_meta in metadata.reqs_to_send.items():
            uuid = int(req_meta.uuid)
            num_tokens = int(req_meta.num_tokens or 0)
            expiration_time = float(req_meta.expiration_time)
            if num_tokens <= 0:
                raise ValueError("Stage-3 producer send metadata requires num_tokens")
            if expiration_time <= 0:
                raise ValueError(
                    "Stage-3 producer send metadata requires a positive expiration_time"
                )
            scheduler_ids = tuple(
                int(block_id) for block_id in req_meta.local_block_ids
            )
            transfer_tags = (
                ("fa",) if self._stage3_kimi else _STAGE3_GLM_TRANSFER_POOL_TAGS
            )
            pool_spans = [
                lower_glm_row_spans(
                    tag=tag,
                    num_tokens=num_tokens,
                    transfer_rank=transfer_rank,
                    parallelism=parallelism,
                    page_tokens=page_tokens,
                    live_bytes_per_block=tag_geometry[tag][0],
                    row_bytes=tag_geometry[tag][1],
                    block_ids=list(scheduler_ids),
                )
                for tag in transfer_tags
            ]
            if self._stage3_kimi:
                pool_spans.extend(
                    self._stage3_state_pool_spans(
                        getattr(req_meta, "mamba_state_block_ids", None),
                        transfer_rank,
                        parallelism,
                    )
                )
            # Every tag stripes identically, so one tag's span count is this
            # rank's page count.
            owned_pages = len(pool_spans[0].spans)
            has_work = any(registration.spans for registration in pool_spans)
            if not has_work:
                # Ranks owning no page publish a fully empty registration
                # (only empty block_ids may complete before the claim) and
                # self-complete below.
                pool_spans = []
                registered_ids: tuple[int, ...] = ()
            else:
                registered_ids = scheduler_ids
            terminal = self._stage3_terminal_sends.get(req_id)
            if terminal is not None:
                if (
                    terminal.uuid != uuid
                    or terminal.local_block_ids != registered_ids
                    or terminal.num_tokens != num_tokens
                ):
                    raise ValueError(
                        "Conflicting replay after terminal Stage-3 producer "
                        f"registration for req_id={req_id}"
                    )
                continue
            existing = self._stage3_registered_sends.get(req_id)
            if existing is not None:
                if (
                    existing.uuid != uuid
                    or existing.local_block_ids != registered_ids
                    or existing.num_tokens != num_tokens
                ):
                    raise ValueError(
                        "Conflicting duplicate Stage-3 producer registration "
                        f"for req_id={req_id}"
                    )
                continue
            try:
                with TraceAnnotation(
                    "KV_Cache_Prefill_Register_Blocks",
                    uuid=uuid,
                    request_id=str(req_id),
                    num_tokens=num_tokens,
                    num_blocks=len(registered_ids),
                ):
                    facade.register_request_blocks(
                        req_id=req_id,
                        uuid=uuid,
                        unit=self._raiden_work_unit,
                        block_ids=list(registered_ids),
                        pool_spans=pool_spans,
                    )
            except (RuntimeError, ValueError) as exc:
                if _STAGE3_REGISTRATION_CANCELLED_ERROR not in str(exc):
                    raise
                tombstone_deadline = time.perf_counter() + float(
                    dist_utils.get_p2p_wait_pull_timeout()
                )
                self._stage3_terminal_sends[req_id] = _Stage3RegisteredSend(
                    uuid=uuid,
                    local_block_ids=registered_ids,
                    num_tokens=num_tokens,
                    expiration_time=tombstone_deadline,
                )
                self._done_sending.add(req_id)
                logger.warning(
                    "TPURaidenConnectorWorker rank%d --> Stage-3 row-span "
                    "registration was already cancelled req_id=%s uuid=%d",
                    self.tp_rank,
                    req_id,
                    uuid,
                )
                continue
            self._stage3_registered_sends[req_id] = _Stage3RegisteredSend(
                uuid=uuid,
                local_block_ids=registered_ids,
                num_tokens=num_tokens,
                expiration_time=expiration_time,
            )
            if not has_work:
                # No pages owned: terminal immediately, no transfer.
                self._stage3_send_outcomes.setdefault(req_id, "done")
            event = {
                "event": "raiden_stage3_request_blocks_registered",
                "req_id": req_id,
                "uuid": uuid,
                "num_tokens": num_tokens,
                "transfer_rank": transfer_rank,
                "parallelism": parallelism,
                "page_tokens": page_tokens,
                "request_pages": len(scheduler_ids),
                "owned_pages": owned_pages,
                "declared_bytes": {reg.tag: reg.declared_bytes for reg in pool_spans},
            }
            logger.info("%s", json.dumps(event, sort_keys=True))

    @staticmethod
    def _raiden_hbm_memory_type() -> Any:
        rpc = importlib.import_module("tpu_sync.rpc.raiden_controller")
        return rpc.RaidenMemoryType.HBM

    def _stage3_source_work_units(self, req_meta: _Stage3LoadMeta) -> list[Any]:
        """Enumerates the producer units this destination pulls from: the
        complete rank set. A pipeline decode stage names them all too (the
        planner keeps the prefill stage registering its layers, which needs
        equal stage counts on both sides); the coordinator requires the
        source ranks to be contiguous from zero."""
        parallelism = int(req_meta.src_parallelism)
        expected = int(tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM)
        if parallelism != expected:
            raise ValueError(
                "Stage-3 source parallelism does not match destination "
                f"configuration: source={parallelism}, destination={expected}"
            )
        ranks = range(parallelism)
        if _engine_is_pipeline(self.vllm_config):
            pp_size = _pipeline_parallel_size(self.vllm_config)
            if parallelism != pp_size:
                raise ValueError(
                    "A pipeline decode stage pulls from the prefill stage "
                    "with the same layers, so both engines need the same "
                    f"stage count: source={parallelism}, decode={pp_size}"
                )
        return [
            self._new_raiden_id(
                stage3_fa_raiden_id_fields(
                    job_name=req_meta.src_job_name,
                    engine_id=req_meta.src_engine_id,
                    dp_rank=req_meta.src_data_replica_idx,
                    transfer_rank=rank,
                    is_producer=True,
                )
            )
            for rank in ranks
        ]

    def _record_stage3_load_failure(self, req_id: str, block_ids: list[int]) -> None:
        self._failed_recving.add(req_id)
        self._failed_block_ids.update(block_ids)
        self._stage3_terminal_loads.add(req_id)

    def _stage3_fa_token_bytes(self, page_tokens: int) -> int:
        """Bytes per token of the admitted FA pools, measured from the
        manifest (live bytes per block over the page geometry)."""
        manifest = self._raiden_manifest
        if manifest is None or not hasattr(manifest, "geometry_by_tag"):
            raise RuntimeError(
                "Stage-3 byte-span lowering requires the admitted pool manifest"
            )
        geometry = manifest.geometry_by_tag().get("fa")
        if not geometry:
            raise RuntimeError("Stage-3 admitted manifest contains no FA pools")
        live_bytes = int(geometry["live_bytes_per_block"])
        if page_tokens <= 0 or live_bytes <= 0 or live_bytes % page_tokens:
            raise RuntimeError(
                "Stage-3 FA pool live bytes are not token-aligned: "
                f"live={live_bytes}, page_tokens={page_tokens}"
            )
        return live_bytes // page_tokens

    def _stage3_state_live_bytes(self, tag: str) -> int:
        """Return the admitted whole-slot byte size for an exact state tag."""
        manifest = self._raiden_manifest
        if manifest is None or not hasattr(manifest, "geometry_by_tag"):
            raise RuntimeError("GDN state reshard requires the admitted pool manifest")
        from vllm_torchtpu.distributed.kv_transfer.raiden.tags import class_tag

        geometry = manifest.geometry_by_tag().get(class_tag(tag))
        if not geometry:
            raise RuntimeError(
                f"GDN state tag {tag!r} is absent from the pool manifest"
            )
        live_bytes = int(geometry["live_bytes_per_block"])
        if live_bytes <= 0:
            raise RuntimeError(
                f"GDN state tag {tag!r} has invalid live bytes {live_bytes}"
            )
        return live_bytes

    def _stage3_state_regions(self, tag: str) -> tuple[Any, ...]:
        """Return one uniform admitted live-region map for a state tag."""
        manifest = self._raiden_manifest
        pools = tuple(getattr(manifest, "pools", ()) or ())
        matching = tuple(pool for pool in pools if str(getattr(pool, "tag", "")) == tag)
        if not matching:
            raise RuntimeError(
                f"GDN state tag {tag!r} is absent from the pool manifest"
            )

        def signature(pool: Any) -> tuple[tuple[Any, ...], ...]:
            return tuple(
                (
                    str(region.name),
                    int(region.offset_bytes),
                    int(region.stride_bytes),
                    int(region.unit_bytes),
                    int(region.num_units),
                    int(region.units_per_stride),
                )
                for region in pool.regions
            )

        expected = signature(matching[0])
        if any(signature(pool) != expected for pool in matching[1:]):
            raise RuntimeError(
                f"GDN state pools tagged {tag!r} disagree on live regions"
            )
        return tuple(matching[0].regions)

    def _stage3_fa_registration_tags(self) -> list[str]:
        """FA pool tags this rank declares spans for: the class tag, or one
        tag per FA layer under per-layer tags."""
        if not _use_per_layer_pool_tags():
            return ["fa"]
        manifest = self._raiden_manifest
        if manifest is None:
            raise RuntimeError(
                "Stage-3 registration requires the admitted pool manifest"
            )
        tags = [pool.tag for pool in manifest.pools_of_class("fa")]
        if not tags:
            raise RuntimeError("Stage-3 admitted manifest contains no FA pools")
        return tags

    def _stage3_state_registration_tags(self) -> list[tuple[str, int]]:
        """(exact state tag, mamba group ordinal) pairs this rank declares:
        one per state class and group, or one per state pool under
        per-layer tags. Class order first, then manifest order."""
        from vllm_torchtpu.distributed.kv_transfer.raiden.tags import split_layer_tag

        if not _use_per_layer_pool_tags():
            return [
                (f"{tag}.g{ordinal}", ordinal)
                for tag in _STAGE3_STATE_CLASS_TAGS
                for ordinal in range(self._stage3_state_group_count)
            ]
        manifest = self._raiden_manifest
        if manifest is None:
            raise RuntimeError(
                "Stage-3 registration requires the admitted pool manifest"
            )
        pairs = []
        for tag in _STAGE3_STATE_CLASS_TAGS:
            for pool in manifest.pools:
                class_value, layer = split_layer_tag(pool.tag)
                if layer is None or not class_value.startswith(f"{tag}.g"):
                    continue
                ordinal_text = class_value[len(tag) + 2 :]
                if not ordinal_text.isdigit():
                    raise RuntimeError(
                        f"GDN state tag {pool.tag!r} carries no mamba group ordinal"
                    )
                pairs.append((pool.tag, int(ordinal_text)))
        return pairs

    def _stage3_state_pool_spans(
        self, mamba_state_block_ids, transfer_rank: int, parallelism: int
    ) -> list:
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
                "mamba state block ids"
            )
        if len(mamba_state_block_ids) != self._stage3_state_group_count:
            raise RuntimeError(
                "GDN state registration group count disagrees with the "
                f"manifest: blocks={len(mamba_state_block_ids)}, "
                f"groups={self._stage3_state_group_count}"
            )
        from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import (
            lower_kda_state_shard_spans,
        )
        from vllm_torchtpu.distributed.kv_transfer.raiden.layout_fingerprint import (
            EXPECTED_FA_MINOR_TO_MAJOR,
            EXPECTED_FA_TILES,
        )
        from vllm_torchtpu.distributed.kv_transfer.raiden.pool_manifest import (
            BINDING_ALIASED_RAW,
        )
        from vllm_torchtpu.distributed.kv_transfer.raiden.seq_on_lane import (
            SOL_LAYOUT_TOKEN,
            lower_gdn_state_shard_spans_sol,
            sol_fa_geometry_from_manifest,
        )

        aliased_raw = self._raiden_manifest.binding == BINDING_ALIASED_RAW
        if aliased_raw and not self._stage3_kimi:
            # Raw transfers move physical bytes; admit only the fingerprinted
            # tiled layout.  The whole-token QK pair-blocked conv layout is
            # enforced downstream: the span lowering rejects any other conv
            # region vocabulary, and the layout version rides the compared
            # fingerprint payload.
            payload = self._raiden_layout_fingerprint_payload
            if not payload:
                raise RuntimeError(
                    "aliased raw GDN lowering requires an admitted physical "
                    "layout fingerprint"
                )
            measured_minor_to_major = tuple(
                int(value) for value in payload.get("minor_to_major", ())
            )
            measured_tiles = tuple(
                tuple(int(value) for value in tile) for tile in payload.get("tiles", ())
            )
            measured_element_bits = int(payload.get("element_size_in_bits", 0))
            if (
                measured_minor_to_major != EXPECTED_FA_MINOR_TO_MAJOR
                or measured_tiles != EXPECTED_FA_TILES
                or measured_element_bits != 8
            ):
                raise RuntimeError(
                    "aliased raw GDN lowering is unsupported for the "
                    "admitted physical layout"
                )

        physical_granule = 1024
        if _raiden_seq_on_lane_layout() and not self._stage3_kimi:
            if aliased_raw and payload.get("fa_kv_layout") != SOL_LAYOUT_TOKEN:
                raise RuntimeError("HND GDN spans require an HND layout fingerprint")
            geometry = sol_fa_geometry_from_manifest(self._raiden_manifest)
            physical_granule = geometry.packing * geometry.page_tokens

        registrations = []
        for exact_tag, ordinal in self._stage3_state_registration_tags():
            block_id = mamba_state_block_ids[ordinal]
            if aliased_raw and not self._stage3_kimi:
                matching_pools = tuple(
                    pool
                    for pool in self._raiden_manifest.pools
                    if str(getattr(pool, "tag", "")) == exact_tag
                )
                if any(
                    int(pool.base_offset_bytes) % physical_granule
                    or int(pool.block_stride_bytes) % physical_granule
                    for pool in matching_pools
                ):
                    raise RuntimeError(
                        "aliased raw GDN pool base and block stride must "
                        f"be physical-token aligned ({exact_tag})"
                    )
            admitted_live_bytes = self._stage3_state_live_bytes(exact_tag)
            if self._stage3_kimi:
                dst_shards = tpu_envs.TPU_RAIDEN_DST_SHARDS
                assert dst_shards is not None
                registration = lower_kda_state_shard_spans(
                    tag=exact_tag,
                    block_id=int(block_id),
                    transfer_rank=transfer_rank,
                    parallelism=parallelism,
                    dst_shards=dst_shards,
                    regions=self._stage3_state_regions(exact_tag),
                )
            else:
                registration = lower_gdn_state_shard_spans_sol(
                    tag=exact_tag,
                    block_id=int(block_id),
                    transfer_rank=transfer_rank,
                    parallelism=parallelism,
                    regions=self._stage3_state_regions(exact_tag),
                    dst_shards=_raiden_dst_shards(),
                    producer_tp_size=self.tp_size,
                    producer_tp_rank=self.tp_rank,
                    physical_granule_bytes=physical_granule,
                )
            if registration.declared_bytes != admitted_live_bytes:
                raise RuntimeError(
                    "GDN state byte lowering disagrees with the admitted "
                    f"manifest for {exact_tag}: lowered="
                    f"{registration.declared_bytes}, admitted="
                    f"{admitted_live_bytes}"
                )
            registrations.append(registration)
        return registrations

    def _stage3_prune_state_send_tracking(self, base_req_id: str) -> None:
        """Drops producer terminal tracking with the base tombstone."""
        self._stage3_send_outcomes.pop(base_req_id, None)

    def _stage3_record_producer_send_terminals(
        self, done_sending, failed_sending
    ) -> None:
        """Records native terminals (T3.1: one uuid, base req ids only)."""
        for req_id, failed in [(str(req_id), False) for req_id in done_sending] + [
            (str(req_id), True) for req_id in failed_sending
        ]:
            if req_id in self._stage3_registered_sends:
                if failed:
                    self._stage3_send_outcomes[req_id] = "failed"
                else:
                    self._stage3_send_outcomes.setdefault(req_id, "done")
                continue
            # Replayed native terminals after the base moved to its tombstone
            # are harmless and must not become duplicate scheduler reports.
            if (
                req_id not in self._stage3_terminal_sends
                and req_id not in self._stage3_reported_sends
            ):
                logger.warning(
                    "Ignoring unmapped Stage-3 producer terminal req_id=%s", req_id
                )

    def _stage3_ready_producer_send_terminals(
        self,
    ) -> tuple[set[str], set[str], set[str]]:
        """Returns terminal base requests (T3.1: no sibling aggregation)."""
        outcomes, self._stage3_send_outcomes = self._stage3_send_outcomes, {}
        done = {r for r, outcome in outcomes.items() if outcome == "done"}
        failed = {r for r, outcome in outcomes.items() if outcome == "failed"}
        cancelled = {r for r, outcome in outcomes.items() if outcome == "cancelled"}
        return done, failed, cancelled

    def _probe_stage3_registrations(self, facade: Any, now: float) -> None:
        """Observes consumer-side cancellations of this rank's unclaimed
        request-block registrations.

        cancel_request_blocks_if_unclaimed retires only the source store's
        registry row and no native transfer terminal follows, so without this
        probe the registration (and vLLM's delayed block free that waits on
        every rank's send terminal) would sit until p2p_wait_pull_timeout.
        Registrations older than the probe interval and still without a
        terminal are batch-queried at most once per interval; a cancelled
        (or vanished) row is recorded as a cancelled terminal, exactly what
        the TTL sweep records for its own cancellation."""
        probe_s = dist_utils.get_stage3_status_probe_s()
        probe = getattr(facade, "get_request_block_status", None)
        if probe_s <= 0 or probe is None:
            return
        for req_id in list(self._stage3_status_probe_next):
            if req_id not in self._stage3_registered_sends:
                del self._stage3_status_probe_next[req_id]
        due: list[str] = []
        for req_id in self._stage3_registered_sends:
            if (
                req_id in self._stage3_terminal_cleanup
                or req_id in self._stage3_send_outcomes
            ):
                continue
            next_probe = self._stage3_status_probe_next.get(req_id)
            if next_probe is None:
                # Grace period: a healthy registration is claimed and
                # completes natively well within one interval.
                self._stage3_status_probe_next[req_id] = now + probe_s
            elif next_probe <= now:
                due.append(req_id)
        if not due:
            return
        for req_id in due:
            self._stage3_status_probe_next[req_id] = now + probe_s
        keys = [
            (req_id, int(self._stage3_registered_sends[req_id].uuid)) for req_id in due
        ]
        try:
            statuses = [int(status) for status in probe(keys)]
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Stage-3 registry status probe failed for %d registrations; "
                "p2p_wait_pull_timeout remains the backstop: %s",
                len(keys),
                exc,
            )
            return
        if len(statuses) != len(keys):
            raise RuntimeError(
                "Stage-3 registry status probe returned "
                f"{len(statuses)} statuses for {len(keys)} keys"
            )
        for req_id, status in zip(due, statuses):
            if status not in (
                _STAGE3_REGISTRY_STATUS_CANCELLED,
                _STAGE3_REGISTRY_STATUS_UNKNOWN,
            ):
                continue
            self._stage3_send_outcomes[req_id] = "cancelled"
            logger.info(
                "%s",
                json.dumps(
                    {
                        "event": "raiden_stage3_registration_cancelled",
                        "req_id": req_id,
                        "uuid": int(self._stage3_registered_sends[req_id].uuid),
                        "registry_status": status,
                        "transfer_rank": self._local_raiden_transfer_rank(),
                    },
                    sort_keys=True,
                ),
            )

    @staticmethod
    def _start_stage3_transfer_with_d5_retry(facade: Any, **kwargs: Any) -> bool:
        """Bridges the one-step P scheduler-to-worker registration window.

        vLLM returns producer transfer parameters from update_from_output(),
        while the D5 block registrations reach producer workers in the next
        scheduler step. The source controller fails before receiver arming
        when those registrations are incomplete, so retry only that precise,
        side-effect-free error for a short bounded interval.
        """
        configured_timeout = float(dist_utils.get_p2p_wait_pull_timeout())
        deadline = time.perf_counter() + min(
            max(configured_timeout, 0.0), _STAGE3_D5_REGISTRATION_WAIT_S
        )
        attempts = 0
        while True:
            attempts += 1
            try:
                return facade.start_transfer(**kwargs)
            except RuntimeError as exc:
                if (
                    "Missing producer block registration" not in str(exc)
                    or time.perf_counter() >= deadline
                ):
                    raise
                if attempts == 1:
                    logger.info(
                        "Stage-3 waiting for producer D5 registrations "
                        "req_id=%s uuid=%s",
                        kwargs.get("req_id"),
                        kwargs.get("uuid"),
                    )
                time.sleep(0.01)

    def _bind_stage3_request_ids(
        self, destination_req_id: str, source_req_id: str
    ) -> None:
        """Binds the native/controller identity to the local scheduler ID."""
        existing_source = self._stage3_source_req_ids.get(destination_req_id)
        if existing_source is not None and existing_source != source_req_id:
            raise ValueError(
                "Conflicting Stage-3 source request ID for destination "
                f"request {destination_req_id!r}: existing="
                f"{existing_source!r}, new={source_req_id!r}"
            )
        existing_destination = self._stage3_destination_req_ids.get(source_req_id)
        if (
            existing_destination is not None
            and existing_destination != destination_req_id
        ):
            raise ValueError(
                "Stage-3 source request ID is already bound to a different "
                f"destination request: source={source_req_id!r}, existing="
                f"{existing_destination!r}, new={destination_req_id!r}"
            )
        self._stage3_source_req_ids[destination_req_id] = source_req_id
        self._stage3_destination_req_ids[source_req_id] = destination_req_id

    def _stage3_source_request_id(self, destination_req_id: str) -> str:
        source_req_id = self._stage3_source_req_ids.get(destination_req_id)
        if source_req_id is None:
            raise RuntimeError(
                "Stage-3 destination request has no source identity binding: "
                f"destination={destination_req_id!r}"
            )
        return source_req_id

    def _stage3_destination_request_id(self, source_req_id: str) -> str | None:
        # Native completion IDs are controller plan IDs. Never treat an
        # unknown ID as a decode-local ID: it may be a stale terminal record
        # that happens to collide with an active destination request.
        return self._stage3_destination_req_ids.get(source_req_id)

    def _stage3_release_producer_registration(
        self,
        destination_req_id: str,
        req_meta: _Stage3LoadMeta,
        synchronous: bool = False,
    ) -> None:
        """Full local hit: cancel the producer's unclaimed request-block
        registration.

        cancel_request_blocks_if_unclaimed is claim-safe — a concurrently
        claimed (in-flight) registration refuses cancellation — and the TTL
        remains the backstop, so this is best-effort fire-and-forget and
        rides the submit workers instead of the model-runner thread."""
        address = str(req_meta.src_controller_address).strip()
        facade = self._stage3_source_facades.get(address)
        if facade is None:
            try:
                facade = self._new_raiden_controller_facade(address)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "Release-only cancellation skipped req_id=%s: cannot "
                    "reach source controller %r: %s",
                    req_meta.source_req_id,
                    address,
                    exc,
                )
                if req_meta.report_completion:
                    self._done_recving.add(destination_req_id)
                return
            self._stage3_source_facades[address] = facade
        source_req_id = req_meta.source_req_id
        uuid = int(req_meta.uuid)

        def _release() -> None:
            cancelled = False
            try:
                cancelled = bool(
                    facade.cancel_request_blocks_if_unclaimed(
                        req_id=source_req_id, uuid=uuid
                    )
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "Release-only cancellation failed req_id=%s "
                    "uuid=%d: %s (p2p_wait_pull_timeout remains the backstop)",
                    source_req_id,
                    uuid,
                    exc,
                )
            logger.info(
                "%s",
                json.dumps(
                    {
                        "event": "raiden_stage3_release_only",
                        "req_id": source_req_id,
                        "destination_req_id": destination_req_id,
                        "uuid": uuid,
                        "cancelled": cancelled,
                    },
                    sort_keys=True,
                ),
            )

        if req_meta.report_completion:
            self._done_recving.add(destination_req_id)
        if synchronous:
            _release()
            return
        self._stage3_submit_queue.put(_release)
        self._ensure_stage3_submit_workers()

    def _submit_stage3_loads(
        self,
        metadata: TPUConnectorMetadata,
        engine: "KVCacheManager",
        synchronous: bool = False,
    ) -> None:
        """Starts exactly one source-controller transfer per request.

        Each request's scheduler-visible tables are staged here, on the
        model-runner thread; the coordination RPC itself then runs on a
        background submit worker (or inline when ``synchronous``) and its
        outcome is applied when the poll drains the mailbox.
        """
        del engine  # Completion is observed through the manager in poll_stats.
        _, dst_controller_address = self._require_stage3_controller()
        if self._raiden_work_unit is None:
            raise RuntimeError("Stage-3 destination work unit is not registered")
        tp_group = None
        if self._stage3_sharded_consumer():
            tp_group = get_tp_group() if self._stage3_kimi else self._stage3_tp_group()

        for destination_req_id, req_meta in metadata.reqs_to_load.items():
            if not isinstance(req_meta, _Stage3LoadMeta):
                raise TypeError(
                    "Stage-3 consumer requires controller load metadata, got "
                    f"{type(req_meta).__name__}"
                )
            if req_meta.release_only:
                if tp_group is None or self.tp_rank == 0:
                    self._stage3_release_producer_registration(
                        destination_req_id, req_meta, synchronous=synchronous
                    )
                elif req_meta.report_completion:
                    self._done_recving.add(destination_req_id)
                continue
            if req_meta.fail_only:
                # Every rank records the failure (the scheduler aggregates
                # one receive terminal per rank); one rank releases.
                local_blocks = list(req_meta.local_block_ids)
                self._load_block_ids[destination_req_id] = local_blocks
                self._record_stage3_load_failure(destination_req_id, local_blocks)
                if tp_group is None or self.tp_rank == 0:
                    self._stage3_release_producer_registration(
                        destination_req_id, req_meta, synchronous=synchronous
                    )
                continue
            uuid = int(req_meta.uuid)
            num_tokens = int(req_meta.num_tokens)
            source_req_id = req_meta.source_req_id
            try:
                self._bind_stage3_request_ids(destination_req_id, source_req_id)
            except ValueError as exc:
                logger.error(
                    "Stage-3 load for destination_req_id=%s rejected: %s",
                    destination_req_id,
                    exc,
                )
                local_blocks = list(req_meta.local_block_ids)
                self._load_block_ids[destination_req_id] = local_blocks
                self._record_stage3_load_failure(destination_req_id, local_blocks)
                continue
            existing_uuid = self._stage3_submitted_loads.get(destination_req_id)
            if existing_uuid is not None:
                if (
                    existing_uuid != uuid
                    or self._stage3_submitted_load_tokens.get(destination_req_id)
                    != num_tokens
                ):
                    raise ValueError(
                        "Conflicting duplicate Stage-3 load submission for "
                        f"req_id={destination_req_id}"
                    )
                continue
            self._stage3_submitted_loads[destination_req_id] = uuid
            self._stage3_submitted_load_tokens[destination_req_id] = num_tokens
            self._stage3_submitted_load_metas[destination_req_id] = req_meta
            self._stage3_load_start_times[destination_req_id] = time.perf_counter()
            local_blocks = list(req_meta.local_block_ids)
            self._load_block_ids[destination_req_id] = local_blocks
            if self._stage3_sharded_consumer() and not self._stage3_kimi:
                # P3 (tpu-sync #744 + per-span routing): ONE coordination per
                # request names every TP worker of this engine as a destination
                # unit. The tp_rank-0 worker submits; the outcome is broadcast so
                # every worker keeps identical bookkeeping for the terminal it
                # will observe on its own native manager.
                # GLM (TP ranks are independent transfer units) and TP1 keep
                # the historical per-worker submit with the worker's own unit.
                outcome = None
                if self.tp_rank == 0 or tp_group is None:
                    outcome = self._submit_stage3_load_as_leader(
                        req_meta=req_meta,
                        source_req_id=source_req_id,
                        destination_req_id=destination_req_id,
                        uuid=uuid,
                        num_tokens=num_tokens,
                        local_blocks=local_blocks,
                        dst_controller_address=dst_controller_address,
                        dst_units=self._raiden_destination_work_units(),
                    )
                if tp_group is not None:
                    outcome = tp_group.broadcast_object(outcome, src=0)
                status, submit_ms, error_text, skip_tokens, fa_skip_bytes = outcome
                if status == "failed":
                    # FA D5 lookup and local validation fail before receiver
                    # arming, so no late H2D is possible.
                    self._record_stage3_load_failure(destination_req_id, local_blocks)
                    logger.error(
                        "Stage-3 controller transfer failed before receiver "
                        "arming req_id=%s destination_req_id=%s uuid=%d: %s",
                        source_req_id,
                        destination_req_id,
                        uuid,
                        error_text,
                    )
                    continue
                if status == "uncertain":
                    # Generic RPC rejection can occur after receiver arming.
                    # Keep the request blocked and accept native terminal
                    # records; only surface recompute after manager failure or
                    # a full post-RPC manager timeout.
                    self._stage3_controller_accepted.add(destination_req_id)
                    deadline = time.perf_counter() + float(
                        dist_utils.get_p2p_wait_pull_timeout()
                    )
                    self._stage3_pending_controller_failures[destination_req_id] = (
                        deadline
                    )
                    logger.error(
                        "Stage-3 controller transfer outcome uncertain; "
                        "waiting for native terminal state req_id=%s "
                        "destination_req_id=%s uuid=%d: %s",
                        source_req_id,
                        destination_req_id,
                        uuid,
                        error_text,
                    )
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
                            "skip_tokens": skip_tokens,
                            "fa_skip_bytes": fa_skip_bytes,
                            "recv_armed_before_push": True,
                            "state_group_count": self._stage3_state_group_count,
                            "destination_pages": len(local_blocks),
                            "destination_units": int(self.tp_size),
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
                    "req_id=%s destination_req_id=%s uuid=%d "
                    "destination_units=%d destination_pages=%d num_tokens=%d",
                    self.tp_rank,
                    source_req_id,
                    destination_req_id,
                    uuid,
                    int(self.tp_size),
                    len(local_blocks),
                    int(req_meta.num_tokens),
                )

                continue
            now = time.perf_counter()
            defer_wait_s = (
                dist_utils.get_stage3_registration_wait_s()
                if not synchronous and dist_utils.get_stage3_deferred_submit_enabled()
                else None
            )
            pending = _Stage3PendingSubmit(
                req_meta=req_meta,
                source_req_id=source_req_id,
                destination_req_id=destination_req_id,
                uuid=uuid,
                num_tokens=num_tokens,
                local_blocks=local_blocks,
                dst_controller_address=dst_controller_address,
                dst_units=(
                    self._raiden_destination_work_units()
                    if tp_group is not None
                    else [self._raiden_work_unit]
                ),
                defer_deadline=(None if defer_wait_s is None else now + defer_wait_s),
                first_attempt_s=now,
            )
            if tp_group is not None:
                pending.task = self._stage_stage3_submit_task(pending, synchronous)
                ready: list[bool | None] = [None] * int(tp_group.world_size)
                torch.distributed.all_gather_object(
                    ready, pending.task is not None, group=tp_group.cpu_group
                )
                if not all(ready):
                    self._record_stage3_load_failure(destination_req_id, local_blocks)
                    continue
            with TraceAnnotation(
                "KV_Cache_Decode_Submit_Load",
                uuid=uuid,
                request_id=str(destination_req_id),
                source_req_id=str(source_req_id),
                num_tokens=num_tokens,
                num_blocks=len(local_blocks),
            ):
                self._dispatch_stage3_load_submit(pending, synchronous)

        if tp_group is not None and synchronous:
            self._drain_stage3_submit_outcomes()

    def _stage_stage3_submit_task(
        self, pending: _Stage3PendingSubmit, synchronous: bool
    ) -> _Stage3SubmitTask | None:
        """Prepares the coordination call for one staged load.

        Model-runner thread only. A failure here precedes any controller
        contact, so the load resolves to the pre-arm failure terminal and
        None is returned.
        """
        req_meta = pending.req_meta
        source_req_id = pending.source_req_id
        destination_req_id = pending.destination_req_id
        uuid = pending.uuid
        local_blocks = pending.local_blocks
        try:
            src_units = self._stage3_source_work_units(req_meta)
            src_controller_address = str(req_meta.src_controller_address).strip()
            if not src_controller_address:
                raise ValueError("empty Stage-3 source controller address")
            source_facade = self._stage3_source_facades.get(src_controller_address)
            if source_facade is None:
                source_facade = self._new_raiden_controller_facade(
                    src_controller_address
                )
                self._stage3_source_facades[src_controller_address] = source_facade
            # ONE transfer carries every replicated cache class; tag
            # order fixes the H2D order ranks executor-side (the first
            # tag = group 0 uploads first). Each tag replays over the
            # same destination pages.
            if _use_raiden_glm_admission():
                transfer_tags = list(_STAGE3_GLM_TRANSFER_POOL_TAGS)
            else:
                transfer_tags = self._stage3_fa_registration_tags()
            fa_tag_count = len(transfer_tags)
            dst_blocks = list(local_blocks) * len(transfer_tags)
            dst_counts = [len(local_blocks)] * len(transfer_tags)
            mamba_state_block_ids = req_meta.mamba_state_block_ids
            if self._stage3_state_group_count:
                if (
                    not mamba_state_block_ids
                    or len(mamba_state_block_ids) != self._stage3_state_group_count
                ):
                    raise RuntimeError(
                        "GDN state load requires one destination state "
                        "block per mamba group"
                    )
                for tag, ordinal in self._stage3_state_registration_tags():
                    transfer_tags.append(tag)
                    dst_blocks.append(int(mamba_state_block_ids[ordinal]))
                    dst_counts.append(1)
            # Prefix-aware suffix pull: clip the FA global destination
            # byte space at the locally-cached prefix; GDN state classes
            # are not prefix-decomposable and always transfer whole
            # (skip 0). Omit the kwarg entirely when there is no clip so
            # skip-free requests stay byte-identical on the wire.
            skip_tokens = req_meta.skip_tokens
            fa_skip_bytes = 0
            clip_kwargs: dict[str, Any] = {}
            if skip_tokens > 0:
                token_bytes = self._stage3_fa_token_bytes(
                    int(self.vllm_config.cache_config.block_size)
                )
                fa_skip_bytes = skip_tokens * token_bytes
                clip_kwargs["dst_skip_bytes"] = [fa_skip_bytes] * fa_tag_count + [0] * (
                    len(transfer_tags) - fa_tag_count
                )
            dst_mem_type = self._raiden_hbm_memory_type()
        except Exception as exc:  # pylint: disable=broad-except
            # Local staging failed before any controller contact, so no
            # receiver can have been armed and no late H2D is possible.
            self._record_stage3_load_failure(destination_req_id, local_blocks)
            logger.error(
                "Stage-3 controller transfer failed before receiver "
                "arming req_id=%s destination_req_id=%s uuid=%d: %s",
                source_req_id,
                destination_req_id,
                uuid,
                exc,
            )
            return None
        return _Stage3SubmitTask(
            pending=pending,
            skip_tokens=skip_tokens,
            fa_skip_bytes=fa_skip_bytes,
            source_units=len(src_units),
            facade=source_facade,
            call_kwargs=dict(
                src_units=src_units,
                dst_units=list(pending.dst_units),
                # The registration is keyed by the producer's internal
                # request ID. The destination's independently randomized
                # ID is strictly a local scheduler lifecycle key.
                req_id=source_req_id,
                dst_device_block_ids=dst_blocks,
                dst_mem_type=dst_mem_type,
                use_block_chunks=True,
                src_controller_address=src_controller_address,
                dst_controller_address=pending.dst_controller_address,
                uuid=uuid,
                is_sender=True,
                num_tokens=pending.num_tokens,
                transfer_pool_tags=transfer_tags,
                dst_block_counts=dst_counts,
                **clip_kwargs,
            ),
            retry_inline=pending.defer_deadline is None or synchronous,
        )

    def _dispatch_stage3_load_submit(
        self, pending: _Stage3PendingSubmit, synchronous: bool = False
    ) -> None:
        """Issues one coordination attempt for a staged load.

        Synchronous (inline-load) mode runs the RPC on the calling thread and
        applies its outcome at once; otherwise the RPC rides a submit worker
        and the outcome is applied by the poll that drains the mailbox.
        """
        task = pending.task
        if task is None:
            task = self._stage_stage3_submit_task(pending, synchronous)
            if task is None:
                return
            pending.task = task
        if self._stage3_sharded_consumer():
            self._stage3_tp_submits.add(pending.destination_req_id)
            if self.tp_rank != 0:
                # Park native terminals until the leader resolves submission.
                # A follower never expires a controller RPC independently.
                self._stage3_inflight_submits[pending.destination_req_id] = float("inf")
                return
        if synchronous:
            self._apply_stage3_submit_outcome(self._execute_stage3_submit(task))
            return
        # The uncertainty window opens at enqueue: past this deadline a
        # still-unresolved RPC is treated exactly like a post-arm failure
        # (blocks held, request abandoned) so one black-holed peer cannot
        # pin a request forever.
        self._stage3_inflight_submits[pending.destination_req_id] = (
            time.perf_counter() + float(dist_utils.get_p2p_wait_pull_timeout())
        )
        self._stage3_submit_queue.put(task)
        self._ensure_stage3_submit_workers()

    def _ensure_stage3_submit_workers(self) -> None:
        if self._stage3_submit_threads:
            return
        for index in range(_STAGE3_SUBMIT_WORKERS):
            thread = threading.Thread(
                target=self._stage3_submit_worker_loop,
                name=f"raiden-stage3-submit-r{self.tp_rank}-{index}",
                daemon=True,
            )
            thread.start()
            self._stage3_submit_threads.append(thread)

    def _stage3_submit_worker_loop(self) -> None:
        while True:
            task = self._stage3_submit_queue.get()
            try:
                # Release-only cancellations ride the queue as bare
                # callables; they have no scheduler-visible outcome to hand
                # back.
                if callable(task):
                    task()
                    continue
                outcome = self._execute_stage3_submit(task)
                with self._stage3_submit_lock:
                    self._stage3_submit_outcomes.append(outcome)
            finally:
                self._stage3_submit_queue.task_done()

    def _execute_stage3_submit(self, task: _Stage3SubmitTask) -> _Stage3SubmitOutcome:
        """Runs one coordination RPC; safe on any thread.

        The facade RPC returns only after the source controller has awaited
        its transfer future (receiver arm + sender dispatch). Actual byte
        completion is deliberately *not* inferred from that acknowledgement;
        get_finished polls the local manager.
        """
        start_submit = time.perf_counter()
        if (
            task.pending.destination_req_id
            in self._stage3_finished_loads_pending_cleanup
        ):
            return _Stage3SubmitOutcome(
                task=task,
                error=RuntimeError(
                    "Missing producer block registration (aborted before submit)"
                ),
                submit_ms=0.0,
                attempted_at=start_submit,
            )
        error: BaseException | None = None
        try:
            if task.retry_inline:
                accepted = self._start_stage3_transfer_with_d5_retry(
                    task.facade, **task.call_kwargs
                )
            else:
                # Deferred mode: one attempt. A missing registration is
                # handed back as an outcome and the request is re-attempted
                # from a later step instead of spinning here.
                accepted = task.facade.start_transfer(**task.call_kwargs)
            if accepted is not True:
                raise RuntimeError("source controller rejected Stage-3 transfer")
        except Exception as exc:  # pylint: disable=broad-except
            error = exc
        return _Stage3SubmitOutcome(
            task=task,
            error=error,
            submit_ms=(time.perf_counter() - start_submit) * 1000,
            attempted_at=start_submit,
        )

    def _apply_stage3_submit_outcome(self, outcome: _Stage3SubmitOutcome) -> None:
        """Applies one RPC outcome to the scheduler-visible tables.

        Model-runner thread only: every table it touches pairs with the
        get_finished/get_block_ids_with_load_errors drain of the same pass.
        """
        task = outcome.task
        pending = task.pending
        destination_req_id = pending.destination_req_id
        source_req_id = pending.source_req_id
        uuid = pending.uuid
        self._stage3_inflight_submits.pop(destination_req_id, None)
        if destination_req_id in self._stage3_abandoned_submits:
            self._stage3_abandoned_submits.discard(destination_req_id)
            logger.warning(
                "Discarding Stage-3 submission outcome for an abandoned "
                "request req_id=%s destination_req_id=%s uuid=%d error=%s",
                source_req_id,
                destination_req_id,
                uuid,
                outcome.error,
            )
            return
        if destination_req_id not in self._stage3_submitted_loads:
            logger.warning(
                "Discarding Stage-3 submission outcome for an already "
                "released request req_id=%s destination_req_id=%s uuid=%d "
                "error=%s",
                source_req_id,
                destination_req_id,
                uuid,
                outcome.error,
            )
            return
        if outcome.error is not None:
            exc = outcome.error
            missing_registration = "Missing producer block registration" in str(exc)
            cancelled_registration = "Request block registration was cancelled" in str(
                exc
            )
            aborted = destination_req_id in self._stage3_finished_loads_pending_cleanup
            if (
                missing_registration
                and pending.defer_deadline is not None
                and not aborted
                and time.perf_counter() < pending.defer_deadline
            ):
                # The precise pre-arm missing-registration result is
                # side-effect free (registrations ride a later producer
                # scheduler step): park for a per-step re-attempt until the
                # deadline.
                pending.last_attempt_s = outcome.attempted_at
                if not pending.wait_logged:
                    pending.wait_logged = True
                    logger.info(
                        "Stage-3 deferring load until producer request-"
                        "block registrations arrive req_id=%s "
                        "destination_req_id=%s uuid=%d wait_budget_s=%.1f",
                        source_req_id,
                        destination_req_id,
                        uuid,
                        dist_utils.get_stage3_registration_wait_s(),
                    )
                self._stage3_pending_submits[destination_req_id] = pending
                return
            if missing_registration or cancelled_registration:
                # The FA registration lookup fails before receiver arming,
                # so no late H2D is possible. A request the scheduler already
                # finished resolves here as well, so its delayed block-free
                # path completes.
                self._record_stage3_load_failure(
                    destination_req_id, pending.local_blocks
                )
                logger.error(
                    "Stage-3 controller transfer failed before receiver "
                    "arming req_id=%s destination_req_id=%s uuid=%d: %s",
                    source_req_id,
                    destination_req_id,
                    uuid,
                    exc,
                )
                return
            # Generic RPC rejection can occur after receiver arming.
            # Keep the request blocked and accept native terminal
            # records; only surface recompute after manager failure or
            # a full post-RPC manager timeout.
            self._stage3_controller_accepted.add(destination_req_id)
            deadline = time.perf_counter() + float(
                dist_utils.get_p2p_wait_pull_timeout()
            )
            self._stage3_pending_controller_failures[destination_req_id] = deadline
            logger.error(
                "Stage-3 controller transfer outcome uncertain; "
                "waiting for native terminal state req_id=%s "
                "destination_req_id=%s uuid=%d: %s",
                source_req_id,
                destination_req_id,
                uuid,
                exc,
            )
            return
        self._stage3_controller_accepted.add(destination_req_id)
        logger.info(
            "%s",
            json.dumps(
                {
                    "event": "raiden_stage3_transfer_submitted",
                    "req_id": source_req_id,
                    "destination_req_id": destination_req_id,
                    "uuid": uuid,
                    "num_tokens": pending.num_tokens,
                    "skip_tokens": task.skip_tokens,
                    "fa_skip_bytes": task.fa_skip_bytes,
                    "recv_armed_before_push": True,
                    "state_group_count": self._stage3_state_group_count,
                    "destination_pages": len(pending.local_blocks),
                    "controller_submit_ms": outcome.submit_ms,
                },
                sort_keys=True,
            ),
        )
        if pending.wait_logged:
            logger.info(
                "Stage-3 deferred load submitted after %.2fs wait "
                "req_id=%s destination_req_id=%s uuid=%d",
                time.perf_counter() - pending.first_attempt_s,
                source_req_id,
                destination_req_id,
                uuid,
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
            task.source_units,
            len(pending.local_blocks),
            pending.num_tokens,
        )

    def _drain_stage3_submit_outcomes(self) -> None:
        """Applies queued RPC outcomes and abandons overdue submissions."""
        if not self._stage3_sharded_consumer() or self.tp_rank == 0:
            with self._stage3_submit_lock:
                outcomes = self._stage3_submit_outcomes
                self._stage3_submit_outcomes = []
            for outcome in outcomes:
                self._apply_stage3_submit_outcome(outcome)
            now = time.perf_counter()
            for req_id, deadline in list(self._stage3_inflight_submits.items()):
                if deadline > now:
                    continue
                # Receiver arming may have happened before the RPC stalled.
                # Hold its blocks until a native terminal or uncertainty timeout.
                self._stage3_inflight_submits.pop(req_id)
                self._stage3_abandoned_submits.add(req_id)
                self._stage3_controller_accepted.add(req_id)
                self._stage3_pending_controller_failures[req_id] = now + float(
                    dist_utils.get_p2p_wait_pull_timeout()
                )
                logger.error(
                    "Stage-3 controller submission RPC exceeded its deadline and "
                    "was abandoned; waiting for native terminal state "
                    "destination_req_id=%s",
                    req_id,
                )

        if self._stage3_tp_submits:
            self._sync_stage3_tp_submit_states()

    def _sync_stage3_tp_submit_states(self) -> None:
        """Fan out final submission decisions, never rank-local timestamps."""
        states = {}
        if self.tp_rank == 0:
            for req_id in self._stage3_tp_submits:
                if req_id in self._stage3_terminal_loads:
                    states[req_id] = "failed"
                elif req_id in self._stage3_pending_controller_failures:
                    states[req_id] = "uncertain"
                elif req_id in self._stage3_controller_accepted:
                    states[req_id] = "accepted"
        states = get_tp_group().broadcast_object(states, src=0)
        for req_id, state in states.items():
            self._stage3_tp_submits.remove(req_id)
            if self.tp_rank == 0:
                continue
            self._stage3_inflight_submits.pop(req_id)
            if state == "failed":
                self._record_stage3_load_failure(req_id, self._load_block_ids[req_id])
            else:
                self._stage3_controller_accepted.add(req_id)
                if state == "uncertain":
                    self._stage3_pending_controller_failures[req_id] = (
                        time.perf_counter()
                        + float(dist_utils.get_p2p_wait_pull_timeout())
                    )

    def _drain_stage3_pending_submits(self, finished_req_ids: set[str] | None) -> None:
        """Re-attempts parked Stage-3 loads once per scheduler step, paced
        by the re-attempt interval."""
        if self._stage3_sharded_consumer() and self.tp_rank != 0:
            return
        aborted_req_ids = (
            set(finished_req_ids or ()) | self._stage3_finished_loads_pending_cleanup
        )
        if aborted_req_ids:
            for req_id in aborted_req_ids:
                pending = self._stage3_pending_submits.pop(req_id, None)
                if pending is None:
                    continue
                # Scheduler-finished (aborted) before the producer
                # registered: cancel producer's registration if it arrives and
                # resolve to the pre-arm failure terminal so the delayed
                # block-free path completes.
                req_meta = self._stage3_submitted_load_metas.pop(
                    req_id, pending.req_meta
                )
                self._stage3_release_producer_registration(req_id, req_meta)
                self._record_stage3_load_failure(req_id, pending.local_blocks)
                logger.warning(
                    "Stage-3 deferred load aborted before producer "
                    "registration req_id=%s destination_req_id=%s uuid=%d",
                    pending.source_req_id,
                    req_id,
                    pending.uuid,
                )
        now = time.perf_counter()
        for destination_req_id in list(self._stage3_pending_submits):
            pending = self._stage3_pending_submits[destination_req_id]
            if (
                now - pending.last_attempt_s
                < _STAGE3_REGISTRATION_REATTEMPT_MIN_INTERVAL_S
            ):
                continue
            del self._stage3_pending_submits[destination_req_id]
            self._dispatch_stage3_load_submit(pending)

    def _raiden_destination_work_units(self) -> list:
        """Every TP worker of this decode engine, in tp_rank order (the
        sibling RaidenIds differ only in transfer_rank = tp_rank)."""
        if self.tp_size <= 1:
            return [self._raiden_work_unit]
        return [
            self._new_raiden_id(self._raiden_work_unit_fields(rank))
            for rank in range(int(self.tp_size))
        ]

    def _stage3_sharded_consumer(self) -> bool:
        """A head-sharded Qwen3.5 decode engine (dp{N}tp{tp}_decode): its TP
        workers are sibling destination units of ONE coordination, submitted
        by tp_rank 0 and broadcast (P3). GLM TP consumers are independent
        transfer units and keep per-worker submits."""
        if self._stage3_kimi:
            return int(self.tp_size) > 1 and not self.is_producer
        return (
            int(self.tp_size) > 1
            and not self.is_producer
            and not _use_raiden_glm_admission()
        )

    @staticmethod
    def _stage3_tp_group() -> Any:
        """The TP cpu group for the leader broadcast, or None when no
        torch.distributed TP group exists (single-process tests)."""
        from vllm.distributed.parallel_state import get_tp_group

        try:
            return get_tp_group()
        except AssertionError:
            return None

    def _submit_stage3_load_as_leader(
        self,
        *,
        req_meta: Any,
        source_req_id: str,
        destination_req_id: str,
        uuid: int,
        num_tokens: int,
        local_blocks: list,
        dst_controller_address: str,
        dst_units: list,
    ) -> tuple[str, float | None, str, int, int]:
        """Runs the source-store coordination for one request and classifies
        the result: ``("accepted", submit_ms, "", skip_tokens, fa_skip_bytes)``
        | ``("failed", None, text, 0, 0)`` (before receiver arming)
        | ``("uncertain", None, text, 0, 0)`` (after)."""
        controller_contacted = False
        fa_accepted = False
        submit_ms = None
        skip_tokens = 0
        fa_skip_bytes = 0
        try:
            src_units = self._stage3_source_work_units(req_meta)
            src_controller_address = str(req_meta.src_controller_address).strip()
            if not src_controller_address:
                raise ValueError("empty Stage-3 source controller address")
            source_facade = self._stage3_source_facades.get(src_controller_address)
            if source_facade is None:
                source_facade = self._new_raiden_controller_facade(
                    src_controller_address
                )
                self._stage3_source_facades[src_controller_address] = source_facade
            # The facade RPC returns only after the source controller
            # server has awaited its RaidenFuture (receiver arm + sender
            # dispatch). Actual byte completion is deliberately *not*
            # inferred from that acknowledgement; get_finished polls the
            # local manager.
            # ONE transfer carries every replicated cache class; tag
            # order fixes the H2D order ranks executor-side (the first
            # tag = group 0 uploads first). Each tag replays over the
            # same destination pages.
            transfer_tags = (
                list(_STAGE3_GLM_TRANSFER_POOL_TAGS)
                if _use_raiden_glm_admission()
                else self._stage3_fa_registration_tags()
            )
            fa_tag_count = len(transfer_tags)
            dst_blocks = list(local_blocks) * len(transfer_tags)
            dst_counts = [len(local_blocks)] * len(transfer_tags)
            mamba_state_block_ids = getattr(req_meta, "mamba_state_block_ids", None)
            if self._stage3_state_group_count:
                if (
                    not mamba_state_block_ids
                    or len(mamba_state_block_ids) != self._stage3_state_group_count
                ):
                    raise RuntimeError(
                        "GDN state load requires one destination state "
                        "block per mamba group"
                    )
                for tag, ordinal in self._stage3_state_registration_tags():
                    transfer_tags.append(tag)
                    dst_blocks.append(int(mamba_state_block_ids[ordinal]))
                    dst_counts.append(1)
            # Prefix-aware suffix pull: clip the FA global destination
            # byte space at the locally-cached prefix; GDN state classes
            # are not prefix-decomposable and always transfer whole
            # (skip 0). Omit the kwarg entirely when there is no clip so
            # skip-free requests stay byte-identical on the wire.
            skip_tokens = int(getattr(req_meta, "skip_tokens", 0) or 0)
            fa_skip_bytes = 0
            clip_kwargs: dict[str, Any] = {}
            if skip_tokens > 0:
                if _raiden_seq_on_lane_layout():
                    from vllm_torchtpu.distributed.kv_transfer.raiden.seq_on_lane import (  # noqa: E501
                        sol_fa_geometry_from_manifest,
                        sol_fa_skip_bytes,
                    )

                    dcp_size = int(
                        self.vllm_config.parallel_config.decode_context_parallel_size
                    )
                    if skip_tokens % (
                        int(self.vllm_config.cache_config.block_size) * dcp_size
                    ):
                        raise ValueError("DCP prefix must cover whole scheduler blocks")
                    fa_skip_bytes = sol_fa_skip_bytes(
                        skip_tokens // dcp_size,
                        geometry=sol_fa_geometry_from_manifest(self._raiden_manifest),
                        # This manifest already describes local TP heads.
                        dst_shards=1,
                    )
                else:
                    token_bytes = self._stage3_fa_token_bytes(
                        int(self.vllm_config.cache_config.block_size)
                    )
                    fa_skip_bytes = skip_tokens * token_bytes
                clip_kwargs["dst_skip_bytes"] = [fa_skip_bytes] * fa_tag_count + [0] * (
                    len(transfer_tags) - fa_tag_count
                )
            start_submit = time.perf_counter()
            controller_contacted = True
            accepted = self._start_stage3_transfer_with_d5_retry(
                source_facade,
                src_units=src_units,
                dst_units=dst_units,
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
                **clip_kwargs,
            )
            submit_ms = (time.perf_counter() - start_submit) * 1000
            if accepted is not True:
                raise RuntimeError("source controller rejected Stage-3 transfer")
            fa_accepted = True
        except Exception as exc:  # pylint: disable=broad-except
            missing_d5 = "Missing producer block registration" in str(exc)
            if not controller_contacted or (missing_d5 and not fa_accepted):
                return ("failed", None, str(exc), 0, 0)
            return ("uncertain", None, str(exc), 0, 0)
        return ("accepted", submit_ms, "", skip_tokens, fa_skip_bytes)

    def get_finished(
        self, finished_req_ids: set[str] | None = None
    ) -> tuple[set[str], set[str]]:
        engine = self._ensure_raiden_transfer_engine()
        # A request aborted in WAITING_FOR_REMOTE_KVS is scheduler-finished,
        # but vLLM deliberately delays freeing its blocks until this connector
        # reports a receive terminal. Record finished_req_ids upfront so
        # _drain_stage3_pending_submits and _poll_finished immediately see the
        # abort in the same pass.
        cleanup_req_ids: set[str] = set()
        if finished_req_ids:
            for req_id in finished_req_ids:
                if (
                    req_id in self._stage3_submitted_loads
                    and req_id not in self._stage3_terminal_loads
                ):
                    self._stage3_finished_loads_pending_cleanup.add(req_id)
                else:
                    cleanup_req_ids.add(req_id)
        # Deferred Stage-3 submits are re-driven here, once per step, before
        # terminals are read so a deadline failure surfaces in this pass.
        if self._stage3_pending_submits:
            self._drain_stage3_pending_submits(finished_req_ids)
        if self._stage3_sharded_consumer():
            # This is a TP-wide scheduler-step boundary. The native polling
            # loop in _wait_for_recving can run different counts on each rank
            # and must never contain a collective.
            self._drain_stage3_submit_outcomes()
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
        done_recving = (
            recv_finished - self._suppress_done_recving - self._reported_recving
        )
        self._suppress_done_recving.difference_update(recv_finished)
        self._reported_recving.update(done_recving)
        if done_recving:
            logger.debug(
                "TPURaidenConnectorWorker rank%d --> reporting done_recving=%s",
                self.tp_rank,
                done_recving,
            )
        cleanup_req_ids.update(
            self._stage3_finished_loads_pending_cleanup.intersection(
                self._stage3_terminal_loads
            )
        )
        for req_id in cleanup_req_ids:
            req_meta = self._stage3_submitted_load_metas.pop(req_id, None)
            if (
                not self.is_producer
                and req_id in self._load_block_ids
                and req_id not in self._reported_recving
            ):
                done_recving.add(req_id)
                if req_meta is not None:
                    self._stage3_release_producer_registration(req_id, req_meta)
            elif (
                not self.is_producer
                and req_meta is not None
                and (
                    req_id in self._failed_recving
                    or req_id not in self._stage3_controller_accepted
                )
            ):
                self._stage3_release_producer_registration(req_id, req_meta)
            source_req_id = self._stage3_source_req_ids.pop(req_id, None)
            if (
                source_req_id is not None
                and self._stage3_destination_req_ids.get(source_req_id) == req_id
            ):
                self._stage3_destination_req_ids.pop(source_req_id, None)
            self._reported_recving.discard(req_id)
            self._failed_recving.discard(req_id)
            self._suppress_done_recving.discard(req_id)
            self._load_block_ids.pop(req_id, None)
            self._stage3_submitted_loads.pop(req_id, None)
            self._stage3_submitted_load_tokens.pop(req_id, None)
            self._stage3_load_start_times.pop(req_id, None)
            self._stage3_controller_accepted.discard(req_id)
            self._stage3_pending_controller_failures.pop(req_id, None)
            self._stage3_pending_submits.pop(req_id, None)
            self._stage3_tp_submits.discard(req_id)
            self._stage3_deferred_native_failures.pop(req_id, None)
            self._stage3_terminal_loads.discard(req_id)
            self._stage3_finished_loads_pending_cleanup.discard(req_id)
        self._done_sending = set()
        self._done_recving = set()
        return done_sending, done_recving

    def _poll_finished(self, engine: "KVCacheManager") -> None:
        if (
            not self.is_producer
            and self._raiden_stage3_enabled()
            and not self._stage3_sharded_consumer()
        ):
            self._drain_stage3_submit_outcomes()
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
            self._stage3_record_producer_send_terminals(done_sending, failed_recving)
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
                destination_req_id
                for req_id in native_done_recving
                if (destination_req_id := self._stage3_destination_request_id(req_id))
                is not None
            ]
            failed_recving = [
                destination_req_id
                for req_id in native_failed_recving
                if (destination_req_id := self._stage3_destination_request_id(req_id))
                is not None
            ]
            unmapped_terminal_ids = (
                set(native_done_recving) | set(native_failed_recving)
            ) - self._stage3_destination_req_ids.keys()
            if unmapped_terminal_ids:
                logger.warning(
                    "Ignoring unmapped Stage-3 native terminal request IDs: %s",
                    sorted(unmapped_terminal_ids),
                )
            # An armed receiver can complete while its coordination RPC is
            # still in flight on a submit worker. Park such terminals and
            # replay them through the acceptance filter once the submission
            # outcome has resolved.
            if self._stage3_deferred_native_failures:
                replayed = [
                    req_id
                    for req_id in self._stage3_deferred_native_failures
                    if req_id not in self._stage3_inflight_submits
                ]
                for req_id in replayed:
                    if self._stage3_deferred_native_failures.pop(req_id):
                        failed_recving.append(req_id)
                    else:
                        done_recving.append(req_id)
            for req_id in done_recving:
                if req_id in self._stage3_inflight_submits:
                    self._stage3_deferred_native_failures[req_id] = False
            for req_id in failed_recving:
                if req_id in self._stage3_inflight_submits:
                    self._stage3_deferred_native_failures[req_id] = True
            done_recving = [
                req_id
                for req_id in done_recving
                if req_id in self._stage3_controller_accepted
                and req_id not in self._stage3_inflight_submits
            ]
            failed_recving = [
                req_id
                for req_id in failed_recving
                if req_id in self._stage3_controller_accepted
                and req_id not in self._stage3_inflight_submits
            ]
            terminal_recvs = set(done_recving) | set(failed_recving)
            for req_id in terminal_recvs:
                self._stage3_pending_controller_failures.pop(req_id, None)
            now = time.perf_counter()
            expired_uncertain = {
                req_id
                for req_id, deadline in self._stage3_pending_controller_failures.items()
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
            self._probe_stage3_registrations(facade, now)
            expired_candidates = {
                req_id
                for req_id, registration in self._stage3_registered_sends.items()
                if registration.expiration_time <= now
                and req_id not in self._stage3_terminal_cleanup
                and req_id not in self._stage3_send_outcomes
            }
            if expired_candidates:
                for req_id in expired_candidates:
                    registration = self._stage3_registered_sends[req_id]
                    if facade.cancel_request_blocks_if_unclaimed(
                        req_id=req_id, uuid=registration.uuid
                    ):
                        # T3.1: one registration per request — the base
                        # cancellation is the whole cancellation.
                        self._stage3_send_outcomes[req_id] = "cancelled"
            done_sending, sender_failures, cancelled_sends = (
                self._stage3_ready_producer_send_terminals()
            )
            if cancelled_sends:
                logger.warning(
                    "TPURaidenConnectorWorker rank%d --> safely cancelled "
                    "unclaimed Stage-3 sends=%s",
                    self.tp_rank,
                    sorted(cancelled_sends),
                )
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
                self.tp_rank,
                failed_recving,
            )
        if sender_failures:
            logger.error(
                "TPURaidenConnectorWorker rank%d --> failed_sending=%s",
                self.tp_rank,
                sender_failures,
            )
        if self.is_producer and self._raiden_stage3_enabled():
            transfer_rank = self._local_raiden_transfer_rank()
            for req_id in done_sending:
                registration = self._stage3_registered_sends.get(req_id)
                if (
                    req_id not in self._stage3_reported_sends
                    and registration is not None
                ):
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
                    registration.uuid if registration is not None else "unknown",
                    transfer_rank,
                )
        elif self._raiden_stage3_enabled():
            for req_id in done_recving:
                uuid = self._stage3_submitted_loads.get(req_id)
                num_tokens = self._stage3_submitted_load_tokens.get(req_id)
                source_req_id = self._stage3_source_request_id(req_id)
                if (
                    req_id not in self._reported_recving
                    and uuid is not None
                    and num_tokens is not None
                ):
                    start_time = self._stage3_load_start_times.get(req_id)
                    latency_ms = None
                    if start_time is not None:
                        latency_ms = (time.perf_counter() - start_time) * 1000
                    with TraceAnnotation(
                        "KV_Cache_Decode_Recv_Complete",
                        uuid=uuid,
                        request_id=str(req_id),
                        source_req_id=str(source_req_id),
                        num_tokens=num_tokens,
                        reshard_e2e_latency_ms=(
                            round(latency_ms, 3) if latency_ms is not None else 0.0
                        ),
                    ):
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
        if (
            self._stage3_terminal_cleanup
            and self.is_producer
            and self._raiden_stage3_enabled()
        ):
            facade, _ = self._require_stage3_controller()
            # Iterate accumulated terminal sends, not only this poll's delta.
            # If the controller RPC fails, get_finished propagates the error
            # and the retained entry is retried on the next poll instead of
            # leaking until registry TTL.
            for req_id, force_release in list(self._stage3_terminal_cleanup.items()):
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
                    with TraceAnnotation(
                        "KV_Cache_Prefill_Send_Complete",
                        uuid=registration.uuid,
                        request_id=str(req_id),
                        num_tokens=registration.num_tokens,
                    ):
                        facade.complete_request_blocks(
                            req_id=req_id,
                            uuid=registration.uuid,
                            unit=self._raiden_work_unit,
                        )
                tombstone_deadline = time.perf_counter() + float(
                    dist_utils.get_p2p_wait_pull_timeout()
                )
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
        deadline = time.perf_counter() + float(dist_utils.get_p2p_wait_pull_timeout())
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
                    "for Raiden load completion for req_ids=%s",
                    self.tp_rank,
                    pending,
                )
                return
            time.sleep(0.001)

    def _flattened_kv_cache_tensors(self) -> list[Any]:
        assert self.runner is not None
        flat: list[Any] = []
        for entry in self.runner.kv_caches:
            if isinstance(entry, (tuple, list)):
                flat.extend(entry)
            else:
                flat.append(entry)
        return flat

    def _ensure_raiden_transfer_engine(self) -> "KVCacheManager":
        if self._raiden_transfer_engine is not None:
            return self._raiden_transfer_engine
        if self.runner is None:
            raise RuntimeError("register_runner must be called before transfer")
        engine = self._construct_raiden_transfer_engine(
            self._flattened_kv_cache_tensors()
        )
        self._raiden_transfer_engine = engine
        return engine

    def _construct_raiden_transfer_engine(
        self, kv_caches: list[Any], *, num_slots: int | None = None
    ) -> "KVCacheManager":
        max_blocks = self._max_request_blocks()
        if num_slots is None:
            num_slots = self._num_raiden_slots(max_blocks)
        stage3_enabled = self._raiden_stage3_enabled()
        endpoint_rank, node_id = self._raiden_endpoint_identity()
        local_control_port = self._rank_control_port(
            self.kv_transfer_port, rank=endpoint_rank
        )
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
                parallelism=(
                    self._raiden_transfer_parallelism() if self.is_producer else 1
                ),
            )
            if _reshard_store_mode():
                # Register this worker with its engine store's dispatch
                # controller: the reshard coordinator submits transfer
                # programs over the persistent channel this creates.
                dispatch_address = _reshard_dispatch_address(self.dp_rank)
                self._wait_for_address_ready(dispatch_address, timeout_s=30.0)
                manager_kwargs.update(
                    raiden_worker_port=0,
                    raiden_controller_address=dispatch_address,
                    worker_id=f"worker_{self._local_raiden_transfer_rank()}",
                )
            else:
                controller_address = str(tpu_envs.TPU_RAIDEN_CONTROLLER_ADDRESS).strip()
                if controller_address:
                    manager_kwargs.update(
                        raiden_worker_port=0,
                        raiden_controller_address=controller_address,
                        worker_id=f"worker_{self._local_raiden_transfer_rank()}",
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

    @staticmethod
    def _wait_for_address_ready(address: str, timeout_s: float = 30.0) -> None:
        import socket

        host, port_str = address.rsplit(":", 1)
        port = int(port_str)
        deadline = time.perf_counter() + timeout_s
        while True:
            try:
                with socket.create_connection((host, port), timeout=0.5):
                    return
            except OSError as err:
                if time.perf_counter() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for address {address} to be ready"
                    ) from err
                time.sleep(0.05)

    def _raiden_endpoint_identity(self) -> tuple[int, int]:
        """(endpoint rank, node id) of this worker's transfer engine.

        The endpoint rank offsets the control port. A Stage-3 producer and
        every pipeline stage are their own endpoint and node (the transfer
        rank); a non-pipelined consumer is one endpoint per engine, its
        node being the data-parallel replica.
        """
        stage3_enabled = self._raiden_stage3_enabled()
        if stage3_enabled and (
            self.is_producer or _engine_is_pipeline(self.vllm_config)
        ):
            rank = self._local_raiden_transfer_rank()
            return rank, rank
        if stage3_enabled:
            return self.tp_rank, self.dp_rank
        return self.tp_rank, self.tp_rank

    def _rank_control_port(self, base_port: int, *, rank: int | None = None) -> int:
        if rank is None:
            rank = self.tp_rank
        return int(base_port) + 2 * int(rank)

    def _resolve_remote_endpoint(self, req_meta: LoadMeta) -> str:
        if isinstance(req_meta.remote_host, list):
            # If the producer is sharded across nodes, fetch the shard from
            # the same counterpart node. Otherwise, fetch everything from the one node.
            node_idx = self.node_id if len(req_meta.remote_host) > 1 else 0
            host = req_meta.remote_host[node_idx]
            base_port = int(req_meta.remote_port[node_idx])
            # Per-node bases already carry the node offset (each entry is that
            # node's first-rank control port), so dial with the node-local rank.
            ranks_per_node = max(1, self.tp_size // len(req_meta.remote_host))
            local_rank = self.tp_rank - node_idx * ranks_per_node
            return f"{host}:{self._rank_control_port(base_port, rank=local_rank)}"
        host = req_meta.remote_host
        base_port = int(req_meta.remote_port)
        return f"{host}:{self._rank_control_port(base_port)}"

    def _max_request_blocks(self) -> int:
        block_size = self.vllm_config.cache_config.block_size
        max_model_len = self.vllm_config.model_config.max_model_len
        return max(1, (max_model_len + block_size - 1) // block_size)

    def _raiden_staging_blocks_per_pool(
        self, manifest: Any, kv_cache_groups: Any
    ) -> list[int]:
        """Bounded host staging hints, one per pool in manifest order.

        Raiden sizes each storage's staging arena as leases x max(hint) over
        the pools on that storage and leases one slot per distinct device
        block id a transfer touches there. One transfer carries a request's
        full-attention pages plus one state block per GDN group under a
        single uuid, and pools of different kv-cache groups draw block ids
        from different block tables, so pools sharing a storage touch
        disjoint ids. Every pool therefore reports the union for its whole
        storage: pages per max-length request for each full-attention group
        present plus one per GDN group present (a group's conv and ssm pools
        address the same block). A storage holding a pool of unknown kind
        reports 0, which keeps it on the full host mirror (see
        RESHARD_BOUNDED_STAGING_DESIGN.md).
        """
        from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm

        group_of_layer: dict[str, int] = {}
        for index, group in enumerate(kv_cache_groups or ()):
            for layer_name in getattr(group, "layer_names", ()):
                group_of_layer[str(layer_name)] = index
        fa_pages = int(self._max_request_blocks())
        # storage -> {block table: blocks one transfer touches from it}
        tables_by_storage: dict[int, dict[Any, int]] = {}
        full_mirror: set[int] = set()
        for pool in manifest.pools:
            tag = str(pool.tag)
            storage = int(pool.storage_index)
            if tag.startswith(rpm.TAG_FA):
                blocks = fa_pages
            elif tag.startswith(rpm.TAG_GDN_CONV) or tag.startswith(rpm.TAG_GDN_SSM):
                blocks = 1
            else:
                full_mirror.add(storage)
                continue
            # Layers of one kv-cache group share a block table; a layer
            # outside every group is its own table.
            layer_name = str(pool.layer_name)
            table = group_of_layer.get(layer_name, layer_name)
            tables = tables_by_storage.setdefault(storage, {})
            tables[table] = max(tables.get(table, 0), blocks)
        hints: list[int] = []
        for pool in manifest.pools:
            storage = int(pool.storage_index)
            if storage in full_mirror:
                hints.append(0)
            else:
                hints.append(sum(tables_by_storage[storage].values()))
        return hints

    def _register_raiden_pools(
        self, engine: Any, manifest: Any, kv_cache_groups: Any
    ) -> dict:
        """Registers the manifest's pools, with bounded host staging when the
        installed tpu_sync supports it (TPU_RAIDEN_POOL_STAGING_LEASES > 0).

        The full host shadow of the device KV pool does not fit libtpu's
        premapped (pinned) pool once the pool is auto-sized, which silently
        routes every D2H/H2D through the staged-copy slow path; bounded
        staging leases N transfers' worth of pages instead.
        """
        pool_dicts = manifest.pool_dicts()
        leases = int(dist_utils.get_raiden_pool_staging_leases())
        summary: dict
        if leases > 0:
            hints = self._raiden_staging_blocks_per_pool(manifest, kv_cache_groups)
            try:
                summary = dict(
                    engine.register_pools(
                        pool_dicts, staging_leases=leases, staging_blocks_per_pool=hints
                    )
                )
            except TypeError as exc:
                # Older tpu_sync wheels predate bounded staging.
                logger.warning(
                    "Raiden bounded host staging unavailable in the installed "
                    "tpu_sync (%s); falling back to the full host mirror "
                    "(make sure TPU_PREMAPPED_BUFFER_SIZE covers it)",
                    exc,
                )
                summary = dict(engine.register_pools(pool_dicts))
        else:
            summary = dict(engine.register_pools(pool_dicts))
        staging = summary.get("host_staging")
        if isinstance(staging, dict):
            logger.info(
                "Raiden host staging mode=%s leases=%s "
                "bounded_bytes_per_shard=%s full_mirror_bytes_per_shard=%s "
                "storages=%d",
                staging.get("mode"),
                staging.get("leases"),
                staging.get("bounded_storage_bytes_per_shard"),
                staging.get("full_storage_bytes_per_shard"),
                len(staging.get("storages") or ()),
            )
        else:
            logger.info("Raiden host staging mode=full (legacy tpu_sync)")
        return summary

    def _num_raiden_slots(self, max_blocks: int) -> int:
        override = dist_utils.get_raiden_transfer_num_slots()
        if override > 0:
            return override
        assert self.runner is not None
        # Sum per-block bytes over every flattened tensor: cache entries
        # can be heterogeneous e.g. split MLA nope/rope tesnsor list
        # and DSA indexer tensor.
        bytes_per_slot = 0
        for kv_tensor in self._flattened_kv_cache_tensors():
            tensor_bytes = kv_tensor.element_size() * max_blocks
            for dim in kv_tensor.shape[1:]:
                tensor_bytes *= int(dim)
            bytes_per_slot += tensor_bytes
        per_rank_budget = int(dist_utils.get_kv_shm_pool_gb() * (1024**3))
        per_rank_budget //= max(1, self.tp_size)
        return max(1, per_rank_budget // max(1, bytes_per_slot))


def get_uuid() -> int:
    int128 = uuid4().int
    # Stay under 64-bit so JSON-encoded responses through the proxy are safe.
    return int128 >> 78
