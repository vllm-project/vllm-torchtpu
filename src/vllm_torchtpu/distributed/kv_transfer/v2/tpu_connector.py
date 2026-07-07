from __future__ import annotations

import os
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole, SupportsHMA)

import vllm_torchtpu.distributed.utils as dist_utils
from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.distributed.kv_transfer.v2.strided_bridge import \
    TPUConnectorV2StridedBridge
from vllm_torchtpu.distributed.kv_transfer.v2.strided_transfer import (
    RegisteredMemoryRegion, RemoteWorkerMetadata, StridedKVTransferEngine)
from vllm_torchtpu.logger import init_logger

from . import zmq_side_channel
from .common import LayerType, TensorLayout, linear_rank
from .layout import HeadSegment, KVCacheRegion, TokenFirstLayoutSpec
from .metadata import (ConnectorMetadataV2, HeadMapping, KVParallelLayout,
                       LocalDecodeAllocation, PullMeta, RankTransferPlan,
                       SourceBlockRef, StridedSegmentOp, TpKVTopology)
from .pcp_policy import PcpReshardingPolicy
from .planner import ContiguousHeadTPTransferPlanner, TPTransferPlanner

logger = init_logger(__name__)

ReqId = str
V2LifecycleTarget = tuple[str, int]


def get_uuid() -> int:
    int128 = uuid4().int
    # Stay under 64-bit so JSON-encoded responses through the proxy are safe.
    return int128 >> 78


@dataclass
class TPUConnectorV2SendMeta:
    uuid: int
    local_block_ids: tuple[tuple[int, ...], ...]
    expiration_time: float


@dataclass
class TPUConnectorV2LoadMeta:
    uuid: int
    local_block_ids: tuple[tuple[int, ...], ...] | None
    remote_block_ids: tuple[tuple[int, ...], ...] | None
    remote_host: str | list[str]
    remote_port: int | list[int]
    fa_token_offset: int = 0


@dataclass
class TPUConnectorV2Metadata(KVConnectorMetadata):
    reqs_to_send: dict[ReqId, TPUConnectorV2SendMeta]
    reqs_to_load: dict[ReqId, TPUConnectorV2LoadMeta]

    def __init__(self) -> None:
        self.reqs_to_send = {}
        self.reqs_to_load = {}


class _NoopTransferStats:

    def record_failed_transfer(self) -> None:
        return

    def is_empty(self) -> bool:
        return True

    def clone_and_reset(self) -> None:
        return None


def _build_transfer_stats() -> Any:
    try:
        from vllm_torchtpu.distributed.kv_transfer.tpu_connector_stats import \
            TpuKVConnectorStats
    except Exception:
        return _NoopTransferStats()
    return TpuKVConnectorStats()


def _config_int(config: Any, name: str, default: int) -> int:
    value = getattr(config, name, default) if config is not None else default
    return int(default if value is None else value)


def _dist_utils_value(name: str, default: Any) -> Any:
    fn = getattr(dist_utils, name, None)
    if fn is None:
        return default
    value = fn()
    return default if value is None else value


def _new_zmq_context(io_threads: int) -> Any:
    import zmq

    return zmq.Context(io_threads=io_threads)


def _resolve_v2_lifecycle_targets(host: Any, port: Any, *,
                                  label: str) -> tuple[V2LifecycleTarget, ...]:
    host_is_sequence = isinstance(host, (list, tuple))
    port_is_sequence = isinstance(port, (list, tuple))
    if host_is_sequence != port_is_sequence:
        raise ValueError("TPUConnectorV2 ACK metadata must use per-host ACK "
                         "port values when ACK host is per-host")
    if host_is_sequence:
        if not host:
            raise ValueError(f"{label} host must be non-empty")
        if len(host) != len(port):
            raise ValueError(f"{label} host and port must have the same "
                             "number of entries")
        targets = tuple((str(item_host), int(item_port))
                        for item_host, item_port in zip(host, port))
    else:
        targets = ((str(host), int(port)), )
    for target_host, _ in targets:
        if not target_host:
            raise ValueError(f"{label} host must be non-empty")
    return targets


def _get_native_tp_rank() -> int:
    try:
        from vllm.distributed.parallel_state import \
            get_tensor_model_parallel_rank
    except Exception:
        return 0
    try:
        return int(get_tensor_model_parallel_rank())
    except Exception:
        return 0


def _get_native_tp_size() -> int:
    try:
        from vllm.distributed.parallel_state import \
            get_tensor_model_parallel_world_size
    except Exception:
        return 1
    try:
        return int(get_tensor_model_parallel_world_size())
    except Exception:
        return 1


def _get_native_pcp_rank() -> int:
    try:
        from vllm_torchtpu.distributed.pcp import get_pcp_rank
    except Exception:
        return 0
    try:
        return int(get_pcp_rank())
    except Exception:
        return 0


def _is_finished_length_capped(request: Any) -> bool:
    status = getattr(request, "status", None)
    if getattr(status, "name", None) == "FINISHED_LENGTH_CAPPED":
        return True
    if status == "FINISHED_LENGTH_CAPPED":
        return True
    try:
        from vllm.v1.request import RequestStatus
    except Exception:
        return False
    return status == RequestStatus.FINISHED_LENGTH_CAPPED


class _V2SendLifecycleEntry:
    """Producer-side V2 send lifecycle without HMA shared-memory staging."""

    def __init__(self, *, req_id: Any, expiration_time: float) -> None:
        self.req_id = req_id
        self.slot_idx = -1
        self.num_blocks = 0
        self.expiration_time = expiration_time
        self.tp_size = 1
        self.staged = {0}
        staged_event = threading.Event()
        staged_event.set()
        self.staged_events = [staged_event]
        self.stage_complete = threading.Event()
        self.stage_complete.set()
        self.pull_acked = False
        self.pull_started = False
        self.stage_failed = True


@dataclass
class _V2RecvLifecycleEntry:
    req_id: Any
    uuid: int
    status: str
    req_meta: Any
    expected_tp_ranks: set[int]
    completed_tp_ranks: set[int]
    failed_tp_ranks: set[int]
    acked: bool = False


@dataclass(frozen=True)
class TPUConnectorV2WorkerCompletion:
    req_id: Any
    uuid: int
    dp_rank: int
    tp_rank: int
    success: bool
    error: str

    def __post_init__(self) -> None:
        if self.req_id is None:
            raise ValueError("req_id must be non-empty")
        object.__setattr__(self, "uuid", int(self.uuid))
        object.__setattr__(self, "dp_rank", int(self.dp_rank))
        object.__setattr__(self, "tp_rank", int(self.tp_rank))
        object.__setattr__(self, "success", bool(self.success))
        object.__setattr__(self, "error", str(self.error))
        if self.uuid < 0:
            raise ValueError("uuid must be non-negative")
        if self.dp_rank < 0:
            raise ValueError("dp_rank must be non-negative")
        if self.tp_rank < 0:
            raise ValueError("tp_rank must be non-negative")


@dataclass(frozen=True)
class TPUConnectorV2WorkerMeta:
    completions: tuple[TPUConnectorV2WorkerCompletion, ...]

    def __post_init__(self) -> None:
        if isinstance(self.completions, TPUConnectorV2WorkerCompletion):
            raise TypeError("completions must be a sequence")
        result = tuple(self.completions)
        for completion in result:
            if not isinstance(completion, TPUConnectorV2WorkerCompletion):
                raise TypeError("completions must contain only "
                                "TPUConnectorV2WorkerCompletion")
        object.__setattr__(self, "completions", result)

    def aggregate(
            self,
            other: "TPUConnectorV2WorkerMeta") -> "TPUConnectorV2WorkerMeta":
        if not isinstance(other, TPUConnectorV2WorkerMeta):
            raise TypeError("other must be TPUConnectorV2WorkerMeta")
        return TPUConnectorV2WorkerMeta(completions=self.completions +
                                        other.completions)


@dataclass(frozen=True)
class TPUConnectorV2HandshakeMetadata:
    """Per-worker metadata published through the vLLM connector handshake."""

    remote_metadata: RemoteWorkerMetadata
    kv_source_layout: KVParallelLayout
    kv_caches: dict[str, KVCacheRegion]
    topology: TpKVTopology
    fa_group_indices: tuple[int, ...]
    mamba_group_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.remote_metadata, RemoteWorkerMetadata):
            raise TypeError("remote_metadata must be RemoteWorkerMetadata")
        if not isinstance(self.kv_source_layout, KVParallelLayout):
            raise TypeError("kv_source_layout must be KVParallelLayout")
        if not isinstance(self.topology, TpKVTopology):
            raise TypeError("topology must be TpKVTopology")
        if not isinstance(self.kv_caches, dict):
            raise TypeError("kv_caches must be a dict")
        for region in self.kv_caches.values():
            if not isinstance(region, KVCacheRegion):
                raise TypeError("kv_caches values must be KVCacheRegion")
        object.__setattr__(
            self,
            "fa_group_indices",
            tuple(int(index) for index in self.fa_group_indices),
        )
        object.__setattr__(
            self,
            "mamba_group_indices",
            tuple(int(index) for index in self.mamba_group_indices),
        )


class TPUConnectorV2Scheduler:
    """Scheduler half for HMA-shaped V2 strided-transfer metadata."""

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.config = getattr(vllm_config, "kv_transfer_config", None)
        self.is_producer = bool(getattr(self.config, "is_kv_producer", False))
        self.block_size = int(
            getattr(getattr(vllm_config, "cache_config", None), "block_size",
                    1) or 1)
        self.reqs_to_send: dict[ReqId, TPUConnectorV2SendMeta] = {}
        self.reqs_to_load: dict[ReqId, TPUConnectorV2LoadMeta] = {}

        self.kv_ip = _dist_utils_value("get_kv_ips", "127.0.0.1")
        parallel_config = getattr(vllm_config, "parallel_config", None)
        dp_rank = _config_int(parallel_config, "data_parallel_rank", 0)
        tp_size = _config_int(parallel_config, "tensor_parallel_size", 1)
        port_base = _dist_utils_value("get_kv_ports", 9100)
        if isinstance(port_base, list):
            self.kv_port = [
                int(port) + dp_rank * tp_size for port in port_base
            ]
        else:
            self.kv_port = int(port_base) + dp_rank * tp_size
        self.side_channel_port = int(
            _dist_utils_value("get_side_channel_port", 9600)) + dp_rank

        self.strided_source_metadata: ConnectorMetadataV2 | None = None
        self.strided_decode_topology: TpKVTopology | None = None
        self.strided_decode_destination: LocalDecodeAllocation | None = None
        self.strided_transfer_planner: TPTransferPlanner = (
            ContiguousHeadTPTransferPlanner())
        self.remote_metadata: tuple[RemoteWorkerMetadata, ...] = ()
        self._handshake_metadata_by_tp_rank: dict[
            int, TPUConnectorV2HandshakeMetadata] = {}
        self._recv_lifecycle_by_uuid: dict[int, _V2RecvLifecycleEntry] = {}
        self._recv_uuid_by_req_id: dict[Any, int] = {}

    def _maybe_truncate_for_mamba(self, request: Any) -> None:
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
        request: Any,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if self.is_producer:
            self._maybe_truncate_for_mamba(request)
            return 0, False
        if not request.kv_transfer_params:
            return 0, False

        count = max((len(request.prompt_token_ids) - 1) - num_computed_tokens,
                    0)
        if count > 0:
            return count, True
        return 0, False

    def set_strided_source_metadata(
        self,
        metadata: ConnectorMetadataV2,
    ) -> None:
        if not isinstance(metadata, ConnectorMetadataV2):
            raise TypeError("metadata must be ConnectorMetadataV2")
        self.strided_source_metadata = metadata

    def set_strided_decode_metadata(
        self,
        *,
        topology: TpKVTopology,
        destination: LocalDecodeAllocation,
        planner: TPTransferPlanner | None = None,
    ) -> None:
        if not isinstance(topology, TpKVTopology):
            raise TypeError("topology must be TpKVTopology")
        if not isinstance(destination, LocalDecodeAllocation):
            raise TypeError("destination must be LocalDecodeAllocation")
        self.strided_decode_topology = topology
        self.strided_decode_destination = destination
        if planner is not None:
            self.strided_transfer_planner = planner

    def set_remote_metadata(
            self, remote_metadata: Sequence[RemoteWorkerMetadata]) -> None:
        self.remote_metadata = self._require_remote_metadata_sequence(
            remote_metadata)

    def set_xfer_handshake_metadata(
            self, metadata: Mapping[int,
                                    TPUConnectorV2HandshakeMetadata]) -> None:
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping from TP rank to "
                            "TPUConnectorV2HandshakeMetadata")
        parsed: dict[int, TPUConnectorV2HandshakeMetadata] = {}
        for rank, item in metadata.items():
            if not isinstance(item, TPUConnectorV2HandshakeMetadata):
                raise TypeError("metadata values must be "
                                "TPUConnectorV2HandshakeMetadata")
            parsed[int(rank)] = item
        if not parsed:
            raise ValueError("metadata must contain at least one TP rank")
        self._handshake_metadata_by_tp_rank = parsed
        self.remote_metadata = tuple(item.remote_metadata
                                     for _, item in sorted(parsed.items()))

    def _attach_remote_metadata(self, kv_transfer_params: dict[str,
                                                               Any]) -> None:
        remote_metadata = getattr(self, "remote_metadata", ())
        if remote_metadata:
            kv_transfer_params["remote_metadata"] = [
                item.to_dict() for item in remote_metadata
            ]

    def request_finished_all_groups(self, request: Any,
                                    block_ids: Any) -> tuple[bool, Any]:
        if not self.is_producer:
            return False, None
        if not _is_finished_length_capped(request):
            return False, None

        computed_per_group = self._require_grouped_block_ids(block_ids)
        delay_free_blocks = any(group for group in computed_per_group)
        if delay_free_blocks:
            uuid = get_uuid()
            expiration_time = (
                time.perf_counter() +
                float(_dist_utils_value("get_p2p_wait_pull_timeout", 120.0)))
            self.reqs_to_send[request.request_id] = TPUConnectorV2SendMeta(
                uuid=uuid,
                local_block_ids=computed_per_group,
                expiration_time=expiration_time,
            )
            kv_transfer_params = dict(uuid=uuid,
                                      remote_block_ids=computed_per_group,
                                      remote_host=self.kv_ip,
                                      remote_port=self.kv_port)
        else:
            kv_transfer_params = {}

        if "remote_metadata" not in kv_transfer_params:
            self._attach_remote_metadata(kv_transfer_params)
        if self._is_kv_producer() and delay_free_blocks:
            self._attach_v2_side_channel_endpoint(kv_transfer_params)
            source_metadata = self._producer_source_metadata(
                block_ids, kv_transfer_params)
            kv_transfer_params["strided_source_metadata"] = (
                source_metadata.to_dict())
            kv_transfer_params["remote_dp_rank"] = (
                self._producer_remote_dp_rank())
        return delay_free_blocks, kv_transfer_params

    def update_state_after_alloc(self, request: Any, blocks: Any,
                                 num_external_tokens: int) -> Any:
        if self.is_producer or not request.kv_transfer_params:
            return None

        params = getattr(request, "kv_transfer_params", None) or {}
        if num_external_tokens > 0:
            fa_token_offset = self._fa_token_offset_from_request(
                request, num_external_tokens)
            if (self.strided_decode_destination is not None
                    and self.strided_decode_destination.block_ids_by_group):
                local_block_ids = (
                    self.strided_decode_destination.block_ids_by_group)
            else:
                local_block_ids = self._require_grouped_block_ids(
                    blocks.get_block_ids())
            local_block_ids = self._require_grouped_block_ids(local_block_ids)
            self.reqs_to_load[request.request_id] = TPUConnectorV2LoadMeta(
                uuid=params["uuid"],
                local_block_ids=local_block_ids,
                remote_block_ids=self._require_grouped_block_ids(
                    params["remote_block_ids"]),
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
                fa_token_offset=fa_token_offset,
            )
        else:
            self.reqs_to_load[request.request_id] = TPUConnectorV2LoadMeta(
                uuid=params["uuid"],
                local_block_ids=None,
                remote_block_ids=None,
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
            )
        req_meta = self._req_meta_for_request(request)
        if req_meta is None:
            return None

        req_id = getattr(request, "request_id", None)
        uuid = self._metadata_value(req_meta, "uuid")
        remote_block_ids = self._metadata_value(req_meta, "remote_block_ids")
        if remote_block_ids is None:
            self._drop_req_to_load(req_id)
            return None
        if int(num_external_tokens) <= 0:
            raise RuntimeError("TPUConnectorV2 real load requires positive "
                               "num_external_tokens")
        if uuid is None:
            raise RuntimeError("TPUConnectorV2 load metadata requires uuid")
        lifecycle = self._recv_lifecycle_by_uuid.get(int(uuid))
        if lifecycle is not None:
            if lifecycle.req_id != req_id:
                raise RuntimeError("TPUConnectorV2 recv uuid was reused by a "
                                   "different request")
            if lifecycle.status in ("dispatched", "done"):
                self._drop_req_to_load(req_id)
                return None

        if "remote_metadata" in params:
            self._set_metadata_value(
                req_meta,
                "remote_metadata",
                self._remote_metadata_from_wire(params["remote_metadata"]),
            )
        if "remote_rank_ops" in params and params["remote_rank_ops"]:
            raise RuntimeError("TPUConnectorV2 no longer accepts "
                               "remote_rank_ops; use "
                               "remote_rank_ops_by_decode_rank")
        if ("remote_rank_ops_by_decode_rank" in params
                and params["remote_rank_ops_by_decode_rank"]):
            self._set_metadata_value(
                req_meta,
                "remote_rank_ops_by_decode_rank",
                TPUConnectorV2Worker.remote_rank_ops_by_decode_rank_from_wire(
                    params["remote_rank_ops_by_decode_rank"]),
            )
        if "remote_dp_rank" in params:
            self._set_metadata_value(req_meta, "remote_dp_rank",
                                     params["remote_dp_rank"])
        if "v2_ack_host" in params:
            self._set_metadata_value(req_meta, "v2_ack_host",
                                     params["v2_ack_host"])
        if "v2_ack_port" in params:
            self._set_metadata_value(req_meta, "v2_ack_port",
                                     params["v2_ack_port"])

        self._maybe_generate_strided_rank_ops(req_meta, params,
                                              num_external_tokens)
        self._mark_recv_lifecycle_planned(
            req_id,
            int(uuid),
            req_meta=req_meta,
            expected_tp_ranks=self._expected_decode_tp_ranks_from_req_meta(
                req_meta),
        )
        return None

    def build_connector_meta(self) -> Any:
        meta = TPUConnectorV2Metadata()
        if self.is_producer:
            meta.reqs_to_send = self.reqs_to_send
            self.reqs_to_send = {}
        else:
            meta.reqs_to_load = self.reqs_to_load
            self.reqs_to_load = {}
        reqs_to_load = getattr(meta, "reqs_to_load", {}) or {}
        for req_id, req_meta in tuple(reqs_to_load.items()):
            if self._metadata_value(req_meta, "remote_block_ids") is None:
                del reqs_to_load[req_id]
                continue
            uuid = self._metadata_value(req_meta, "uuid")
            if uuid is None:
                raise RuntimeError(
                    "TPUConnectorV2 load metadata requires uuid")
            self._mark_recv_lifecycle_planned(
                req_id,
                int(uuid),
                req_meta=req_meta,
                expected_tp_ranks=self._expected_decode_tp_ranks_from_req_meta(
                    req_meta),
            )
            self._mark_recv_lifecycle_dispatched(req_id, int(uuid))
        return meta

    def request_finished(self, request: Any,
                         block_ids: list[int]) -> tuple[bool, Any]:
        raise AssertionError(
            "TPUConnectorV2 implements SupportsHMA and expects "
            "`request_finished_all_groups` to be invoked.")

    def get_finished_count(self) -> int:
        return len(self.kv_ip) if isinstance(self.kv_ip, list) else 1

    def update_connector_output(self, connector_output: Any) -> None:
        worker_meta = getattr(connector_output, "kv_connector_worker_meta",
                              None)
        if worker_meta is not None:
            self._process_v2_worker_meta(worker_meta, connector_output)
        for req_id in getattr(connector_output, "finished_recving", ()) or ():
            uuid = self._recv_uuid_by_req_id.get(req_id)
            if uuid is None:
                continue
            lifecycle = self._recv_lifecycle_by_uuid.get(uuid)
            if lifecycle is not None:
                lifecycle.status = "done"

    def _drop_req_to_load(self, req_id: Any) -> None:
        reqs_to_load = getattr(self, "reqs_to_load", None)
        if reqs_to_load is not None and req_id is not None:
            reqs_to_load.pop(req_id, None)

    def _mark_recv_lifecycle_planned(self, req_id: Any, uuid: int, *,
                                     req_meta: Any,
                                     expected_tp_ranks: set[int]) -> None:
        if req_id is None:
            raise RuntimeError("TPUConnectorV2 load metadata requires req_id")
        expected = {int(rank) for rank in expected_tp_ranks}
        if not expected:
            raise RuntimeError("TPUConnectorV2 load metadata has no expected "
                               "decode TP ranks")
        existing = self._recv_lifecycle_by_uuid.get(uuid)
        if existing is not None:
            if existing.req_id != req_id:
                raise RuntimeError("TPUConnectorV2 recv uuid was reused by a "
                                   "different request")
            if existing.expected_tp_ranks != expected:
                raise RuntimeError("TPUConnectorV2 recv expected TP ranks "
                                   "changed for a uuid")
            if existing.status == "done":
                return
            existing.req_meta = req_meta
            existing.status = "planned"
        else:
            self._recv_lifecycle_by_uuid[uuid] = _V2RecvLifecycleEntry(
                req_id=req_id,
                uuid=uuid,
                status="planned",
                req_meta=req_meta,
                expected_tp_ranks=expected,
                completed_tp_ranks=set(),
                failed_tp_ranks=set(),
            )
        self._recv_uuid_by_req_id[req_id] = uuid
        logger.info(
            "TPUConnectorV2Scheduler --> START planned | req_id=%s | "
            "uuid=%s | expected_tp_ranks=%s", req_id, uuid,
            tuple(sorted(expected)))

    def _mark_recv_lifecycle_dispatched(self, req_id: Any, uuid: int) -> None:
        lifecycle = self._recv_lifecycle_by_uuid.get(uuid)
        if lifecycle is None:
            raise RuntimeError("TPUConnectorV2 recv lifecycle must be planned "
                               "before dispatch")
        if lifecycle.req_id != req_id:
            raise RuntimeError("TPUConnectorV2 recv uuid was reused by a "
                               "different request")
        if lifecycle.status == "done":
            return
        lifecycle.status = "dispatched"
        logger.info(
            "TPUConnectorV2Scheduler --> START dispatched | req_id=%s | "
            "uuid=%s | expected_tp_ranks=%s", req_id, uuid,
            tuple(sorted(lifecycle.expected_tp_ranks)))

    @staticmethod
    def _expected_decode_tp_ranks_from_req_meta(req_meta: Any) -> set[int]:
        by_decode_rank = (
            TPUConnectorV2Worker._remote_rank_ops_by_decode_rank_from_req_meta(
                req_meta))
        if by_decode_rank:
            return {int(rank) for rank in by_decode_rank}
        return {0}

    def _process_v2_worker_meta(self, worker_meta: Any,
                                connector_output: Any) -> None:
        if not isinstance(worker_meta, TPUConnectorV2WorkerMeta):
            raise TypeError("kv_connector_worker_meta must be "
                            "TPUConnectorV2WorkerMeta")
        first_error: Exception | None = None
        for completion in worker_meta.completions:
            try:
                self._record_v2_worker_completion(completion, connector_output)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def _record_v2_worker_completion(
            self, completion: TPUConnectorV2WorkerCompletion,
            connector_output: Any) -> None:
        lifecycle = self._recv_lifecycle_by_uuid.get(completion.uuid)
        if lifecycle is None:
            raise RuntimeError("TPUConnectorV2 received worker completion for "
                               f"unknown uuid {completion.uuid}")
        if lifecycle.req_id != completion.req_id:
            raise RuntimeError("TPUConnectorV2 worker completion req_id does "
                               "not match lifecycle")
        if completion.tp_rank not in lifecycle.expected_tp_ranks:
            raise RuntimeError("TPUConnectorV2 worker completion TP rank "
                               f"{completion.tp_rank} is not expected")
        completed_tp_ranks = set(lifecycle.completed_tp_ranks)
        if completion.success:
            completed_tp_ranks.add(completion.tp_rank)
        logger.info(
            "TPUConnectorV2Scheduler <-- recv END completion | req_id=%s | "
            "uuid=%s | dp_rank=%s | tp_rank=%s | success=%s | "
            "completed_tp_ranks=%s | expected_tp_ranks=%s", completion.req_id,
            completion.uuid, completion.dp_rank, completion.tp_rank,
            completion.success, tuple(sorted(completed_tp_ranks)),
            tuple(sorted(lifecycle.expected_tp_ranks)))
        if not completion.success:
            lifecycle.failed_tp_ranks.add(completion.tp_rank)
            lifecycle.status = "failed"
            raise RuntimeError("TPUConnectorV2 strided KV pull failed on "
                               f"TP rank {completion.tp_rank}: "
                               f"{completion.error}")
        lifecycle.completed_tp_ranks.add(completion.tp_rank)
        if lifecycle.expected_tp_ranks <= lifecycle.completed_tp_ranks:
            self._complete_v2_recv_lifecycle(lifecycle, connector_output)

    def _complete_v2_recv_lifecycle(self, lifecycle: _V2RecvLifecycleEntry,
                                    connector_output: Any) -> None:
        if lifecycle.acked:
            return
        try:
            self._send_v2_end(lifecycle.req_meta)
        except Exception:
            lifecycle.status = "failed"
            raise
        lifecycle.acked = True
        lifecycle.status = "done"
        logger.info(
            "TPUConnectorV2Scheduler --> recv END complete | req_id=%s | "
            "uuid=%s | completed_tp_ranks=%s", lifecycle.req_id,
            lifecycle.uuid, tuple(sorted(lifecycle.completed_tp_ranks)))
        finished_recving = getattr(connector_output, "finished_recving", None)
        if finished_recving is None:
            finished_recving = set()
            setattr(connector_output, "finished_recving", finished_recving)
        finished_recving.add(lifecycle.req_id)
        self._recv_lifecycle_by_uuid.pop(lifecycle.uuid, None)
        self._recv_uuid_by_req_id.pop(lifecycle.req_id, None)

    def _attach_v2_side_channel_endpoint(
            self, kv_transfer_params: dict[str, Any]) -> None:
        if "remote_host" not in kv_transfer_params:
            raise RuntimeError("TPUConnectorV2 send metadata requires "
                               "remote_host before attaching side-channel "
                               "endpoint")
        from vllm_torchtpu.distributed import utils as dist_utils

        ack_host = kv_transfer_params["remote_host"]
        base_port = int(dist_utils.get_side_channel_port())
        dp_rank = self._local_dp_rank()
        kv_transfer_params["v2_ack_host"] = ack_host
        if isinstance(ack_host, (list, tuple)):
            kv_transfer_params["v2_ack_port"] = [
                base_port + node_id + dp_rank
                for node_id in range(len(ack_host))
            ]
        else:
            kv_transfer_params["v2_ack_port"] = (base_port +
                                                 self._local_node_id() +
                                                 dp_rank)

    def _send_v2_end(self, req_meta: Any) -> None:
        uuid = self._metadata_value(req_meta, "uuid")
        if uuid is None:
            raise RuntimeError("TPUConnectorV2 END metadata requires uuid")
        for host, port in self._resolve_v2_side_channel_targets(req_meta):
            logger.info(
                "TPUConnectorV2 strided lifecycle send END | uuid=%s | "
                "remote=%s:%s", uuid, host, port)
            frames = zmq_side_channel.send_request(
                host=host,
                port=port,
                tag=zmq_side_channel.PULL_END,
                uuid=int(uuid),
                timeout_s=self._v2_side_channel_timeout_s(),
                action="pull-end",
            )
            if frames == [zmq_side_channel.MSG_OK]:
                logger.info(
                    "TPUConnectorV2 strided lifecycle END ack OK | uuid=%s | "
                    "remote=%s:%s", uuid, host, port)
                continue
            if len(frames) == 2 and frames[0] == zmq_side_channel.MSG_ERR:
                message = frames[1].decode("utf-8", errors="replace")
                raise RuntimeError("producer rejected strided KV pull end: "
                                   f"{message}")
            raise RuntimeError("malformed producer strided pull-end response: "
                               f"{frames!r}")

    @staticmethod
    def _v2_side_channel_timeout_s() -> float:
        try:
            from vllm_torchtpu.distributed import utils as dist_utils

            return float(dist_utils.get_p2p_wait_pull_timeout())
        except Exception:
            return 120.0

    def _resolve_v2_side_channel_endpoint(self,
                                          req_meta: Any) -> tuple[str, int]:
        targets = self._resolve_v2_side_channel_targets(req_meta)
        if len(targets) != 1:
            raise ValueError("TPUConnectorV2 ACK metadata contains multiple "
                             "side-channel targets")
        return targets[0]

    def _resolve_v2_side_channel_targets(
            self, req_meta: Any) -> tuple[V2LifecycleTarget, ...]:
        host = self._metadata_value(req_meta, "v2_ack_host")
        port = self._metadata_value(req_meta, "v2_ack_port")
        if host is None or port is None:
            raise RuntimeError("TPUConnectorV2 load metadata requires "
                               "v2_ack_host and v2_ack_port")
        return _resolve_v2_lifecycle_targets(host, port, label="v2_ack")

    @staticmethod
    def _local_node_id() -> int:
        from vllm_torchtpu.distributed import utils as dist_utils

        return int(dist_utils.get_node_id())

    def _local_dp_rank(self) -> int:
        config = getattr(self, "vllm_config", None)
        parallel_config = getattr(config, "parallel_config", None)
        return int(getattr(parallel_config, "data_parallel_rank", 0) or 0)

    def _maybe_generate_strided_rank_ops(
        self,
        req_meta: Any,
        params: Mapping[str, Any],
        num_external_tokens: int,
    ) -> None:
        if TPUConnectorV2Worker._remote_rank_ops_by_decode_rank_from_req_meta(
                req_meta):
            return
        remote_block_ids = self._metadata_value(req_meta, "remote_block_ids")
        if remote_block_ids is None:
            return
        if (self.strided_decode_destination is None
                and self._metadata_value(req_meta, "local_block_ids") is None):
            raise RuntimeError("TPUConnectorV2 load metadata requires "
                               "local_block_ids when remote_block_ids is "
                               "present")

        source_value = params.get("strided_source_metadata",
                                  params.get("source_metadata"))
        if not source_value:
            raise RuntimeError("TPUConnectorV2 load metadata requires "
                               "strided_source_metadata")

        source_metadata = (source_value
                           if isinstance(source_value, ConnectorMetadataV2)
                           else ConnectorMetadataV2.from_mapping(source_value))
        fa_token_offset = int(
            self._metadata_value(req_meta, "fa_token_offset", 0) or 0)
        source_metadata = self._source_metadata_for_external_tokens(
            source_metadata, num_external_tokens, fa_token_offset)
        remote_metadata = self._metadata_value(req_meta, "remote_metadata", ())
        dp_rank = TPUConnectorV2Worker._require_remote_dp_rank_from_req_meta(
            req_meta)
        source_metadata = TPUConnectorV2Worker.apply_remote_source_region_metadata(
            source_metadata, remote_metadata, dp_rank=dp_rank)
        rank_ops_by_decode_rank = self._lower_decode_rank_ops(
            req_meta, source_metadata)
        self._set_metadata_value(req_meta, "remote_rank_ops_by_decode_rank",
                                 rank_ops_by_decode_rank)
        self._set_metadata_value(req_meta, "strided_source_metadata",
                                 source_metadata)

    def _lower_decode_rank_ops(
        self,
        req_meta: Any,
        source_metadata: ConnectorMetadataV2,
    ) -> dict[int, dict[int, tuple[StridedSegmentOp, ...]]]:
        lowered: dict[int, dict[int, tuple[StridedSegmentOp, ...]]] = {}
        for decode_tp_rank, topology, destination in (
                self._decode_metadata_by_tp_rank(req_meta, source_metadata)):
            pull_meta = self.strided_transfer_planner.build_pull_meta(
                source_metadata, topology)
            plans = self.strided_transfer_planner.lower(
                source_metadata,
                topology,
                destination,
                pull_meta,
            )
            lowered[int(decode_tp_rank)] = {
                int(rank): tuple(plan.ops)
                for rank, plan in plans.items() if plan.ops
            }
        if not lowered:
            raise RuntimeError("TPUConnectorV2 decode handshake metadata did "
                               "not produce any decode rank plans")
        return lowered

    @staticmethod
    def _source_metadata_for_external_tokens(
        source_metadata: ConnectorMetadataV2,
        num_external_tokens: int,
        fa_token_offset: int = 0,
    ) -> ConnectorMetadataV2:
        if not source_metadata.fa_block_ids:
            return source_metadata
        num_tokens = int(num_external_tokens)
        fa_token_offset = int(fa_token_offset)
        if num_tokens <= 0:
            raise RuntimeError("TPUConnectorV2 load metadata requires "
                               "positive num_external_tokens when full "
                               "attention blocks are present")
        source_capacity_tokens = (len(source_metadata.fa_block_ids) *
                                  source_metadata.block_size)
        if source_metadata.fa_num_tokens is not None:
            source_capacity_tokens = min(source_capacity_tokens,
                                         source_metadata.fa_num_tokens)
        if fa_token_offset > source_capacity_tokens:
            raise RuntimeError(
                "TPUConnectorV2 load metadata has fa_token_offset beyond "
                f"source tokens: {fa_token_offset} > {source_capacity_tokens}")
        source_capacity_tokens -= fa_token_offset
        num_tokens = min(num_tokens, source_capacity_tokens)
        if (source_metadata.fa_num_tokens == num_tokens
                and source_metadata.fa_token_offset == fa_token_offset):
            return source_metadata
        return replace(source_metadata,
                       fa_num_tokens=num_tokens,
                       fa_token_offset=fa_token_offset)

    @staticmethod
    def _fa_token_offset_from_request(request: Any,
                                      num_external_tokens: int) -> int:
        prompt_transfer_tokens = max(len(request.prompt_token_ids) - 1, 0)
        fa_token_offset = prompt_transfer_tokens - int(num_external_tokens)
        if fa_token_offset < 0:
            raise RuntimeError(
                "TPUConnectorV2 num_external_tokens exceeds prompt transfer "
                f"tokens: {num_external_tokens} > {prompt_transfer_tokens}")
        return fa_token_offset

    def _producer_source_metadata(
        self,
        block_ids: Any,
        kv_transfer_params: Mapping[str, Any],
    ) -> ConnectorMetadataV2:
        if self.strided_source_metadata is not None:
            return self.strided_source_metadata
        if not self._handshake_metadata_by_tp_rank:
            raise RuntimeError("TPUConnectorV2 producer requires worker "
                               "handshake metadata")
        first = self._first_handshake_metadata()
        grouped_block_ids = self._require_grouped_block_ids(block_ids)
        grouped_block_ids = self._normalize_mamba_block_id_groups(
            grouped_block_ids, first.mamba_group_indices)
        fa_block_ids = self._block_ids_for_fa_groups(grouped_block_ids,
                                                     first.fa_group_indices)
        mamba_block_ids = self._block_ids_for_mamba_groups(
            grouped_block_ids, first.mamba_group_indices)
        source_block_size = self._source_block_size_from_handshake()
        req_id = int(kv_transfer_params["uuid"])
        return ConnectorMetadataV2(
            req_id=req_id,
            block_size=source_block_size,
            kv_source_layout=first.kv_source_layout,
            kv_caches={
                int(rank): dict(item.kv_caches)
                for rank, item in self._handshake_metadata_by_tp_rank.items()
            },
            fa_block_ids=fa_block_ids,
            mamba_block_ids=mamba_block_ids,
            block_ids_by_group=grouped_block_ids,
            fa_num_tokens=(len(fa_block_ids) *
                           source_block_size if fa_block_ids else None),
            mamba_num_tokens=None,
        )

    def _producer_remote_dp_rank(self) -> int:
        if self.remote_metadata:
            dp_ranks = {int(item.dp_rank) for item in self.remote_metadata}
            if len(dp_ranks) != 1:
                raise ValueError("producer remote_metadata must contain one "
                                 f"DP rank, got {sorted(dp_ranks)}")
            return next(iter(dp_ranks))
        return int(self._first_handshake_metadata().remote_metadata.dp_rank)

    def _decode_metadata(
        self,
        req_meta: Any,
        source_metadata: ConnectorMetadataV2,
    ) -> tuple[TpKVTopology, LocalDecodeAllocation]:
        if (self.strided_decode_topology is not None
                and self.strided_decode_destination is not None):
            return (self.strided_decode_topology,
                    self._destination_with_fa_token_offset(
                        self.strided_decode_destination, source_metadata))
        decoded = self._decode_metadata_by_tp_rank(req_meta, source_metadata)
        if len(decoded) != 1:
            raise RuntimeError("TPUConnectorV2 decode metadata contains "
                               "multiple TP ranks; use per-rank lowering")
        _, topology, destination = decoded[0]
        return topology, destination

    def _decode_metadata_by_tp_rank(
        self,
        req_meta: Any,
        source_metadata: ConnectorMetadataV2,
    ) -> tuple[tuple[int, TpKVTopology, LocalDecodeAllocation], ...]:
        if (self.strided_decode_topology is not None
                and self.strided_decode_destination is not None):
            return ((int(self.strided_decode_topology.tp_rank),
                     self.strided_decode_topology,
                     self._destination_with_fa_token_offset(
                         self.strided_decode_destination, source_metadata)), )
        if not self._handshake_metadata_by_tp_rank:
            raise RuntimeError("TPUConnectorV2 decode requires worker "
                               "handshake metadata")
        local_block_ids = self._metadata_value(req_meta, "local_block_ids")
        grouped_block_ids = self._require_grouped_block_ids(local_block_ids)
        result: list[tuple[int, TpKVTopology, LocalDecodeAllocation]] = []
        for decode_tp_rank, item in sorted(
                self._handshake_metadata_by_tp_rank.items()):
            local_grouped_block_ids = self._normalize_mamba_block_id_groups(
                grouped_block_ids, item.mamba_group_indices)
            fa_block_ids = self._block_ids_for_fa_groups(
                local_grouped_block_ids, item.fa_group_indices)
            mamba_block_ids = self._block_ids_for_mamba_groups(
                local_grouped_block_ids, item.mamba_group_indices)
            destination = LocalDecodeAllocation(
                rank=item.topology.tp_rank,
                block_size=item.topology.block_size,
                kv_caches=dict(item.kv_caches),
                fa_block_ids=fa_block_ids,
                mamba_block_ids=mamba_block_ids,
                block_ids_by_group=local_grouped_block_ids,
                fa_num_tokens=source_metadata.fa_num_tokens,
                mamba_num_tokens=source_metadata.mamba_num_tokens,
                fa_token_offset=source_metadata.fa_token_offset,
            )
            result.append((int(decode_tp_rank), item.topology, destination))
        return tuple(result)

    @staticmethod
    def _destination_with_fa_token_offset(
        destination: LocalDecodeAllocation,
        source_metadata: ConnectorMetadataV2,
    ) -> LocalDecodeAllocation:
        if (destination.fa_num_tokens == source_metadata.fa_num_tokens
                and destination.fa_token_offset
                == source_metadata.fa_token_offset):
            return destination
        return replace(destination,
                       fa_num_tokens=source_metadata.fa_num_tokens,
                       fa_token_offset=source_metadata.fa_token_offset)

    def _first_handshake_metadata(self) -> TPUConnectorV2HandshakeMetadata:
        if not self._handshake_metadata_by_tp_rank:
            raise RuntimeError("TPUConnectorV2 worker handshake metadata is "
                               "not registered")
        return self._handshake_metadata_by_tp_rank[min(
            self._handshake_metadata_by_tp_rank)]

    def _source_block_size_from_handshake(self) -> int:
        for item in self._handshake_metadata_by_tp_rank.values():
            for region in item.kv_caches.values():
                if region.layer_type == LayerType.FULL_ATTN:
                    if region.block_size is None:
                        raise ValueError("full attention region must have "
                                         "block_size")
                    return int(region.block_size)
        block_size = getattr(getattr(self.vllm_config, "cache_config", None),
                             "block_size", None)
        if block_size is None:
            raise ValueError("source block_size cannot be inferred")
        return int(block_size)

    @staticmethod
    def _require_grouped_block_ids(
            block_ids: Any) -> tuple[tuple[int, ...], ...]:
        if block_ids is None:
            return ()
        if isinstance(block_ids, (str, bytes, bytearray)):
            raise TypeError("block_ids must be a sequence of int sequences")
        block_id_items = tuple(block_ids)
        if not block_id_items:
            return ()
        if isinstance(block_id_items[0], int):
            return (tuple(int(block_id) for block_id in block_id_items), )
        groups = []
        for group in block_id_items:
            if isinstance(group, (str, bytes, bytearray)):
                raise TypeError("block_ids groups must be int sequences")
            groups.append(tuple(int(block_id) for block_id in group))
        return tuple(groups)

    @staticmethod
    def _block_ids_for_fa_groups(
        block_ids: tuple[tuple[int, ...], ...],
        group_indices: Sequence[int],
    ) -> tuple[int, ...]:
        groups = tuple(int(index) for index in group_indices)
        if not groups:
            return ()
        selected = TPUConnectorV2Scheduler._select_block_id_groups(
            block_ids, groups)
        first = selected[0]
        for group in selected[1:]:
            if group != first:
                raise ValueError("multiple full-attention KV cache groups "
                                 "must have identical block ids")
        return first

    @staticmethod
    def _block_ids_for_mamba_groups(
        block_ids: tuple[tuple[int, ...], ...],
        group_indices: Sequence[int],
    ) -> tuple[int, ...]:
        groups = tuple(int(index) for index in group_indices)
        if not groups:
            return ()
        selected = TPUConnectorV2Scheduler._select_block_id_groups(
            block_ids, groups)
        return tuple(group[-1] for group in selected)

    @staticmethod
    def _normalize_mamba_block_id_groups(
        block_ids: tuple[tuple[int, ...], ...],
        group_indices: Sequence[int],
    ) -> tuple[tuple[int, ...], ...]:
        groups = {int(index) for index in group_indices}
        return tuple((group[-1], ) if index in groups else group
                     for index, group in enumerate(block_ids))

    @staticmethod
    def _select_block_id_groups(
        block_ids: tuple[tuple[int, ...], ...],
        group_indices: Sequence[int],
    ) -> tuple[tuple[int, ...], ...]:
        result = []
        for group_index in group_indices:
            if group_index < 0 or group_index >= len(block_ids):
                raise ValueError(f"block_ids has no group {group_index}")
            result.append(block_ids[group_index])
        return tuple(result)

    def _req_meta_for_request(self, request: Any) -> Any | None:
        reqs_to_load = getattr(self, "reqs_to_load", None)
        if not reqs_to_load:
            return None
        request_id = getattr(request, "request_id", None)
        if request_id is None:
            return None
        if isinstance(reqs_to_load, Mapping):
            return reqs_to_load.get(request_id)
        return getattr(reqs_to_load, request_id, None)

    def _is_kv_producer(self) -> bool:
        config = getattr(self, "vllm_config", None)
        kv_transfer_config = getattr(config, "kv_transfer_config", None)
        return bool(getattr(kv_transfer_config, "is_kv_producer", False))

    @staticmethod
    def _require_remote_metadata_sequence(
        remote_metadata: Sequence[RemoteWorkerMetadata]
    ) -> tuple[RemoteWorkerMetadata, ...]:
        if isinstance(remote_metadata, RemoteWorkerMetadata):
            raise TypeError("remote_metadata must be a sequence of "
                            "RemoteWorkerMetadata")
        if isinstance(remote_metadata, (str, bytes, bytearray)):
            raise TypeError("remote_metadata must be a sequence of "
                            "RemoteWorkerMetadata")
        result = tuple(remote_metadata)
        for item in result:
            if not isinstance(item, RemoteWorkerMetadata):
                raise TypeError("remote_metadata must contain only "
                                "RemoteWorkerMetadata")
        return result

    @staticmethod
    def _remote_metadata_from_wire(
        remote_metadata: Sequence[Mapping[str, Any]]
    ) -> tuple[RemoteWorkerMetadata, ...]:
        if isinstance(remote_metadata, Mapping):
            raise TypeError("remote_metadata wire value must be a sequence of "
                            "mappings")
        if isinstance(remote_metadata, (str, bytes, bytearray)):
            raise TypeError("remote_metadata wire value must be a sequence of "
                            "mappings")
        result = []
        for item in remote_metadata:
            if not isinstance(item, Mapping):
                raise TypeError("remote_metadata wire value must contain only "
                                "mappings")
            result.append(RemoteWorkerMetadata.from_mapping(item))
        return tuple(result)

    @staticmethod
    def _set_metadata_value(metadata: Any, key: str, value: Any) -> None:
        if isinstance(metadata, dict):
            metadata[key] = value
        else:
            setattr(metadata, key, value)

    @staticmethod
    def _metadata_value(metadata: Any, key: str, default: Any = None) -> Any:
        if isinstance(metadata, Mapping):
            return metadata.get(key, default)
        return getattr(metadata, key, default)


class TPUConnectorV2Worker:
    """Worker half for V2 strided-transfer bridge."""

    named_kv_caches: dict[str, Any] | None = None

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.config = getattr(vllm_config, "kv_transfer_config", None)
        self.is_producer = bool(getattr(self.config, "is_kv_producer", False))
        self.runner: Any | None = None
        self.device: Any | None = None
        self.node_id = int(_dist_utils_value("get_node_id", 0))

        parallel_config = getattr(vllm_config, "parallel_config", None)
        self.dp_rank = _config_int(parallel_config, "data_parallel_rank", 0)
        self.model_tp_rank = _get_native_tp_rank()
        self.model_tp_size = _get_native_tp_size()
        self.pcp_size = _config_int(parallel_config,
                                    "prefill_context_parallel_size", 1)
        self.pcp_rank = (_get_native_pcp_rank()
                         if self.is_producer and self.pcp_size > 1 else 0)
        if self.is_producer and self.pcp_size > 1:
            self.tp_rank = (self.pcp_rank * self.model_tp_size +
                            self.model_tp_rank)
            self.tp_size = self.pcp_size * self.model_tp_size
        else:
            self.tp_rank = self.model_tp_rank
            self.tp_size = self.model_tp_size
        self.num_hosts = self._configured_num_hosts()
        self.ranks_per_host = self._ranks_per_host(self.tp_size,
                                                   self.num_hosts)
        self.local_tp_rank = self.tp_rank % self.ranks_per_host
        self._is_host_coordinator = self.local_tp_rank == 0

        self.host_ip = str(_dist_utils_value("get_host_ip", "127.0.0.1"))
        self.kv_transfer_port = (
            int(_dist_utils_value("get_kv_transfer_port", 9100)) +
            self.dp_rank * self.tp_size)
        self.side_channel_port = int(
            _dist_utils_value("get_side_channel_port",
                              9600)) + self.node_id + self.dp_rank
        self.zmq_cxt = _new_zmq_context(max(1, int(self.tp_size)))
        self._stop_event = threading.Event()
        self._coord_lock = threading.Lock()
        self._coord_threads: list[threading.Thread] = []
        self._coord_send: dict[int, _V2SendLifecycleEntry] = {}
        self._coord_done_sending: set[ReqId] = set()
        self._coord_recv: dict[int, Any] = {}
        self._coord_done_recving: set[ReqId] = set()
        self._coord_pool: Any | None = None
        self._coord_executor: ThreadPoolExecutor | None = None
        self._v2_lifecycle_threads_started = False
        self.transfer_stats = _build_transfer_stats()
        self.strided_bridge: TPUConnectorV2StridedBridge | None = None
        self.strided_transfer_engine: Any | None = None
        self._strided_recv_started_uuids: set[int] = set()
        self._strided_recv_done_uuids: set[int] = set()
        self._pending_v2_worker_completions: list[
            TPUConnectorV2WorkerCompletion] = []
        self._strided_lifecycle_lock = threading.Lock()
        self._handshake_metadata: TPUConnectorV2HandshakeMetadata | None = None
        self._registered_region_metadata: dict[str,
                                               RegisteredMemoryRegion] = {}
        self._kv_cache_region_templates: dict[str, KVCacheRegion] = {}

    @staticmethod
    def _configured_num_hosts() -> int:
        num_hosts = int(os.getenv("TPU_NUM_HOSTS", "1"))
        if num_hosts <= 0:
            raise ValueError("TPU_NUM_HOSTS must be positive")
        return num_hosts

    @staticmethod
    def _ranks_per_host(tp_size: int, num_hosts: int) -> int:
        tp_size = int(tp_size)
        num_hosts = int(num_hosts)
        if tp_size <= 0:
            raise ValueError("tp_size must be positive")
        if tp_size % num_hosts != 0:
            raise ValueError("tp_size must be divisible by TPU_NUM_HOSTS: "
                             f"tp_size={tp_size} num_hosts={num_hosts}")
        return tp_size // num_hosts

    def register_runner(self, runner: Any) -> Any:
        self.runner = runner
        self.device = getattr(runner, "device", None)
        self._extract_kv_layout()
        self._ensure_coord_executor()
        self._maybe_install_strided_transfer_engine(runner)
        self._ensure_v2_lifecycle_threads()
        return None

    def _ensure_coord_executor(self) -> None:
        if self._coord_executor is not None:
            return
        max_workers = int(
            _dist_utils_value("get_kv_coord_executor_max_workers", 0) or 0)
        if max_workers <= 0:
            max_workers = max(1, min(2, int(self.tp_size)))
        self._coord_executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="tpu-kv-v2-coord")

    def _ensure_v2_lifecycle_threads(self) -> None:
        if self._v2_lifecycle_threads_started:
            return
        self._v2_lifecycle_threads_started = True
        if not self.is_producer or not self._is_host_coordinator:
            return
        t_exp = threading.Thread(target=self._coord_rank0_expire_loop,
                                 name="tpu_conn_v2_expire",
                                 daemon=True)
        t_exp.start()
        self._coord_threads.append(t_exp)
        t_notif = threading.Thread(
            target=self._coord_rank0_external_notif_loop,
            name="tpu_conn_v2_notif",
            daemon=True)
        t_notif.start()
        self._coord_threads.append(t_notif)

    def _extract_kv_layout(self) -> None:
        runner = self.runner
        if runner is None:
            return
        groups = getattr(getattr(runner, "kv_cache_config", None),
                         "kv_cache_groups", ())
        self.group_is_mamba = [
            type(getattr(group, "kv_cache_spec", None)).__name__ == "MambaSpec"
            for group in groups
        ]

    def _map_layers_to_groups(self, kv_caches: list[Any],
                              groups: list[Any]) -> list[int]:
        named = self.named_kv_caches
        if named is None:
            raise AssertionError(
                "named_kv_caches must be registered before mapping KV groups")
        name_to_group: dict[str, int] = {
            name: gid
            for gid, group in enumerate(groups)
            for name in group.layer_names
        }
        id_to_name = {id(cache): name for name, cache in named.items()}
        result = []
        for cache in kv_caches:
            name = id_to_name.get(id(cache))
            if name is None:
                raise AssertionError(
                    "a runner kv cache was not found by identity in the "
                    "registered named_kv_caches dict.")
            result.append(name_to_group[name])
        return result

    def get_finished(self) -> tuple[set[str], set[str]]:
        return self._coord_get_finished()

    def get_kv_connector_stats(self) -> Any | None:
        if not self.transfer_stats.is_empty():
            return self.transfer_stats.clone_and_reset()
        return None

    def __del__(self) -> None:
        stop_event = getattr(self, "_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        executor = getattr(self, "_coord_executor", None)
        if executor is not None and hasattr(executor, "shutdown"):
            executor.shutdown(wait=False)
        engine = getattr(self, "strided_transfer_engine", None)
        if engine is not None and hasattr(engine, "stop"):
            engine.stop()
        for thread in getattr(self, "_coord_threads", ()):
            thread.join(timeout=2)
        zmq_context = getattr(self, "zmq_cxt", None)
        if zmq_context is not None:
            zmq_context.destroy(linger=0)

    def set_strided_transfer_bridge(
            self, bridge: TPUConnectorV2StridedBridge | Any) -> None:
        self.strided_bridge = bridge
        self.strided_transfer_engine = getattr(bridge, "transfer_engine", None)

    def _maybe_install_strided_transfer_engine(self, runner: Any) -> None:
        if self.strided_bridge is not None:
            return
        engine = StridedKVTransferEngine(
            local_dp_rank=self._local_dp_rank(),
            local_tp_rank=self._local_transfer_rank(),
            local_worker_id=self._local_worker_id(),
            listen_host=self._local_host_for_metadata(),
            listen_port=self._strided_listen_port(),
            transport="zmq",
            timeout_s=self._strided_timeout_s(),
        )
        self._register_named_kv_cache_regions(engine, runner)
        self.strided_transfer_engine = engine
        self.strided_bridge = TPUConnectorV2StridedBridge(engine)
        if self._is_kv_producer():
            engine.start()
        self._handshake_metadata = self._build_handshake_metadata(engine)

    def _register_named_kv_cache_regions(self, engine: Any,
                                         runner: Any) -> None:
        named_kv_caches = getattr(self, "named_kv_caches", None)
        if not named_kv_caches:
            return
        raw_tensors = tuple(getattr(runner, "kv_cache_raw_tensors", ()) or ())
        for layer_name, cache in named_kv_caches.items():
            if isinstance(cache, (list, tuple)):
                for idx, tensor in enumerate(cache):
                    region_id = f"{layer_name}.state{idx}"
                    self._register_named_mamba_region(engine, tensor,
                                                      region_id, raw_tensors)
            else:
                self._register_named_region(engine, cache, layer_name,
                                            raw_tensors)

    def _register_named_region(self, engine: Any, tensor: Any, region_id: str,
                               raw_tensors: Sequence[Any]) -> None:
        nbytes = self._buffer_nbytes(tensor)
        page_bytes = self._infer_page_bytes_from_shape(tensor, nbytes)
        template = self._infer_kv_cache_region(tensor=tensor,
                                               region_id=region_id,
                                               nbytes=nbytes,
                                               page_bytes=page_bytes)
        raw_index = self._raw_tensor_index(tensor, raw_tensors)
        if raw_index is not None:
            raw = raw_tensors[raw_index]
            raw_nbytes = self._buffer_nbytes(raw)
            num_blocks = self._num_blocks_from_shape(tensor)
            if raw_nbytes % num_blocks != 0:
                raise ValueError(
                    "raw KV cache tensor nbytes must be divisible "
                    f"by tensor num_blocks: nbytes={raw_nbytes} "
                    f"num_blocks={num_blocks}")
            raw_page_bytes = raw_nbytes // num_blocks
            raw_region_id = self._raw_region_id(raw_index)
            self._register_raw_region_once(engine, raw, raw_region_id,
                                           raw_nbytes, raw_page_bytes)
            region_base_offset = (self._tensor_storage_offset_bytes(tensor) -
                                  self._tensor_storage_offset_bytes(raw))
            if region_base_offset < 0:
                raise ValueError(
                    "KV cache tensor storage offset is before raw "
                    "KV cache tensor storage")
            if region_base_offset >= raw_page_bytes:
                raise ValueError(
                    "KV cache tensor storage offset must be inside "
                    "one raw KV cache page")
            block_stride_bytes = max(raw_page_bytes,
                                     template.physical_block_stride_bytes)
            token_first_layout = template.token_first_layout
            if token_first_layout is not None:
                token_first_layout = replace(
                    token_first_layout,
                    block_stride_bytes=block_stride_bytes,
                )
            self._kv_cache_region_templates[region_id] = replace(
                template,
                physical_region_id=raw_region_id,
                region_base_offset_bytes=region_base_offset,
                block_stride_bytes=block_stride_bytes,
                token_first_layout=token_first_layout,
            )
            return

        region = RegisteredMemoryRegion(region_id=region_id,
                                        nbytes=nbytes,
                                        page_bytes=page_bytes)
        engine.register_local_region(buffer=tensor, region=region)
        self._registered_region_metadata[region_id] = region
        self._kv_cache_region_templates[region_id] = template

    def _register_named_mamba_region(self, engine: Any, tensor: Any,
                                     region_id: str,
                                     raw_tensors: Sequence[Any]) -> None:
        nbytes = self._buffer_nbytes(tensor)
        state_page_bytes = self._infer_page_bytes_from_shape(tensor, nbytes)
        template = self._infer_kv_cache_region(tensor=tensor,
                                               region_id=region_id,
                                               nbytes=nbytes,
                                               page_bytes=state_page_bytes)
        raw_index = self._raw_tensor_index(tensor, raw_tensors)
        if raw_index is None:
            region = RegisteredMemoryRegion(region_id=region_id,
                                            nbytes=nbytes,
                                            page_bytes=state_page_bytes)
            engine.register_local_region(buffer=tensor, region=region)
            self._registered_region_metadata[region_id] = region
            self._kv_cache_region_templates[region_id] = template
            return
        raw = raw_tensors[raw_index]
        raw_nbytes = self._buffer_nbytes(raw)
        num_blocks = self._num_blocks_from_shape(tensor)
        if raw_nbytes % num_blocks != 0:
            raise ValueError("raw KV cache tensor nbytes must be divisible by "
                             f"state num_blocks: nbytes={raw_nbytes} "
                             f"num_blocks={num_blocks}")
        raw_page_bytes = raw_nbytes // num_blocks
        raw_region_id = self._raw_region_id(raw_index)
        self._register_raw_region_once(engine, raw, raw_region_id, raw_nbytes,
                                       raw_page_bytes)

        state_base_offset = (self._tensor_storage_offset_bytes(tensor) -
                             self._tensor_storage_offset_bytes(raw))
        if state_base_offset < 0:
            raise ValueError("GDN/Mamba state storage offset is before raw "
                             "KV cache tensor storage")
        if state_base_offset >= raw_page_bytes:
            raise ValueError("GDN/Mamba state storage offset must be inside "
                             "one raw KV cache page")
        self._kv_cache_region_templates[region_id] = replace(
            template,
            physical_region_id=raw_region_id,
            region_base_offset_bytes=state_base_offset,
            block_stride_bytes=raw_page_bytes,
        )

    def _register_raw_region_once(self, engine: Any, raw: Any, region_id: str,
                                  nbytes: int, page_bytes: int) -> None:
        existing = self._registered_region_metadata.get(region_id)
        if existing is not None:
            if existing.nbytes != nbytes or existing.page_bytes != page_bytes:
                raise ValueError("raw KV cache region metadata changed for "
                                 f"{region_id}")
            return
        region = RegisteredMemoryRegion(
            region_id=region_id,
            nbytes=nbytes,
            page_bytes=page_bytes,
        )
        engine.register_local_region(buffer=raw, region=region)
        self._registered_region_metadata[region_id] = region

    @staticmethod
    def _raw_region_id(raw_index: int) -> str:
        return f"__raw_kv_cache_{raw_index}"

    @staticmethod
    def _num_blocks_from_shape(buffer: Any) -> int:
        shape = getattr(buffer, "shape", None)
        if not shape:
            raise ValueError("GDN/Mamba state tensor shape must include "
                             "num_blocks")
        num_blocks = int(shape[0])
        if num_blocks <= 0:
            raise ValueError("GDN/Mamba state tensor num_blocks must be "
                             "positive")
        return num_blocks

    @staticmethod
    def _tensor_storage_offset_bytes(tensor: Any) -> int:
        if hasattr(tensor, "storage_offset") and hasattr(
                tensor, "element_size"):
            return int(tensor.storage_offset()) * int(tensor.element_size())
        return 0

    @classmethod
    def _raw_tensor_index(cls, tensor: Any,
                          raw_tensors: Sequence[Any]) -> int | None:
        tensor_storage = cls._storage_data_ptr(tensor)
        for idx, raw in enumerate(raw_tensors):
            if raw is tensor:
                return idx
            raw_storage = cls._storage_data_ptr(raw)
            if (tensor_storage is not None and raw_storage is not None
                    and tensor_storage == raw_storage):
                return idx
        return None

    @staticmethod
    def _storage_data_ptr(tensor: Any) -> int | None:
        try:
            storage = tensor.untyped_storage()
            return int(storage.data_ptr())
        except Exception:
            return None

    @staticmethod
    def _buffer_nbytes(buffer: Any) -> int:
        nbytes = getattr(buffer, "nbytes", None)
        if nbytes is not None:
            return int(nbytes)
        if hasattr(buffer, "numel") and hasattr(buffer, "element_size"):
            return int(buffer.numel()) * int(buffer.element_size())
        return len(memoryview(buffer).cast("B"))

    @staticmethod
    def _infer_page_bytes_from_shape(buffer: Any, nbytes: int) -> int:
        shape = getattr(buffer, "shape", None)
        if not shape:
            return nbytes
        num_blocks = int(shape[0])
        if num_blocks <= 0:
            return nbytes
        if nbytes % num_blocks != 0:
            raise ValueError("KV cache tensor nbytes must be divisible by "
                             f"num_blocks: nbytes={nbytes} "
                             f"num_blocks={num_blocks}")
        return nbytes // num_blocks

    def _is_kv_producer(self) -> bool:
        config = getattr(self, "vllm_config", None)
        kv_transfer_config = getattr(config, "kv_transfer_config", None)
        return bool(getattr(kv_transfer_config, "is_kv_producer", False))

    def _local_dp_rank(self) -> int:
        config = getattr(self, "vllm_config", None)
        parallel_config = getattr(config, "parallel_config", None)
        return int(getattr(parallel_config, "data_parallel_rank", 0) or 0)

    def _local_tp_rank(self) -> int:
        tp_rank = int(getattr(self, "tp_rank", 0) or 0)
        parallel_config = getattr(getattr(self, "vllm_config", None),
                                  "parallel_config", None)
        configured_tp_size = getattr(parallel_config, "tensor_parallel_size",
                                     None)
        if configured_tp_size:
            return tp_rank % int(configured_tp_size)
        return tp_rank

    def _local_tp_size(self) -> int:
        config = getattr(self, "vllm_config", None)
        parallel_config = getattr(config, "parallel_config", None)
        configured_tp_size = getattr(parallel_config, "tensor_parallel_size",
                                     None)
        if configured_tp_size:
            return int(configured_tp_size)
        tp_size = int(getattr(self, "tp_size", 1) or 1)
        tp_rank = int(getattr(self, "tp_rank", 0) or 0)
        return max(tp_size, tp_rank + 1)

    def _local_pcp_size(self) -> int:
        config = getattr(self, "vllm_config", None)
        parallel_config = getattr(config, "parallel_config", None)
        return int(
            getattr(parallel_config, "prefill_context_parallel_size", 1) or 1)

    def _local_pcp_rank(self) -> int:
        pcp_rank = getattr(self, "pcp_rank", None)
        if pcp_rank is not None:
            return int(pcp_rank)
        if self._local_pcp_size() <= 1:
            return 0
        try:
            from vllm_torchtpu.distributed.pcp import get_pcp_rank

            return int(get_pcp_rank())
        except Exception:
            return 0

    def _local_transfer_rank(self) -> int:
        return linear_rank(self._local_pcp_rank(), self._local_tp_rank(),
                           self._local_tp_size())

    def _local_transfer_world_size(self) -> int:
        return self._local_pcp_size() * self._local_tp_size()

    def _local_worker_id(self) -> str:
        if self._local_pcp_size() > 1:
            return (f"dp{self._local_dp_rank()}-pcp{self._local_pcp_rank()}"
                    f"-tp{self._local_tp_rank()}")
        return f"dp{self._local_dp_rank()}-tp{self._local_tp_rank()}"

    def _local_host_for_metadata(self) -> str:
        host = getattr(self, "host_ip", None)
        if host:
            return str(host)
        config = getattr(self, "vllm_config", None)
        kv_transfer_config = getattr(config, "kv_transfer_config", None)
        kv_ip = getattr(kv_transfer_config, "kv_ip", None)
        if isinstance(kv_ip, (list, tuple)):
            return str(kv_ip[0]) if kv_ip else "127.0.0.1"
        if kv_ip:
            return str(kv_ip)
        return "127.0.0.1"

    def _strided_listen_port(self) -> int:
        base_port = int(getattr(self, "kv_transfer_port", 9100)) + 10000
        return (base_port +
                self._local_dp_rank() * self._local_transfer_world_size() +
                self._local_transfer_rank())

    def _strided_timeout_s(self) -> float:
        try:
            from vllm_torchtpu.distributed import utils as dist_utils

            return float(dist_utils.get_p2p_wait_pull_timeout())
        except Exception:
            return 120.0

    def get_handshake_metadata(self) -> TPUConnectorV2HandshakeMetadata | None:
        return self._handshake_metadata

    def _build_handshake_metadata(
            self, engine: Any) -> TPUConnectorV2HandshakeMetadata | None:
        if not self._kv_cache_region_templates:
            return None
        remote_metadata = self._engine_remote_metadata(engine)
        layout = self._local_kv_source_layout()
        topology = self._local_topology(layout)
        fa_groups, mamba_groups = self._kv_cache_group_indices()
        return TPUConnectorV2HandshakeMetadata(
            remote_metadata=remote_metadata,
            kv_source_layout=layout,
            kv_caches=dict(self._kv_cache_region_templates),
            topology=topology,
            fa_group_indices=fa_groups,
            mamba_group_indices=mamba_groups,
        )

    def _engine_remote_metadata(self, engine: Any) -> RemoteWorkerMetadata:
        metadata = engine.local_metadata()
        if isinstance(metadata, RemoteWorkerMetadata):
            remote = metadata
        elif hasattr(metadata, "to_dict"):
            remote = RemoteWorkerMetadata.from_mapping(metadata.to_dict())
        elif isinstance(metadata, Mapping):
            remote = RemoteWorkerMetadata.from_mapping(metadata)
        else:
            raise TypeError("transfer engine local_metadata() must return "
                            "RemoteWorkerMetadata")
        if remote.regions:
            return remote
        return RemoteWorkerMetadata(
            dp_rank=remote.dp_rank,
            tp_rank=remote.tp_rank,
            worker_id=remote.worker_id,
            tcp_host=remote.tcp_host,
            tcp_port=remote.tcp_port,
            transport=remote.transport,
            regions=tuple(self._registered_region_metadata.values()),
        )

    def _local_kv_source_layout(self) -> KVParallelLayout:
        parallel_config = getattr(self.vllm_config, "parallel_config", None)
        full_attn_pcp_size = int(
            getattr(parallel_config, "prefill_context_parallel_size", 1) or 1)
        tp_size = self._local_tp_size()
        return KVParallelLayout(
            full_attn_pcp_size=full_attn_pcp_size,
            full_attn_tp_size=tp_size,
            linear_attn_pcp_size=1,
            linear_attn_tp_size=tp_size,
        )

    def _local_topology(self, layout: KVParallelLayout) -> TpKVTopology:
        return TpKVTopology(
            local_layout=layout,
            block_size=self._local_full_attention_block_size(),
            tp_rank=self._local_tp_rank(),
            total_num_kv_heads=self._total_num_kv_heads(),
            total_num_mamba_key_heads=self._linear_num_key_heads(),
            total_num_mamba_heads=self._total_num_mamba_heads(),
        )

    def _local_full_attention_block_size(self) -> int:
        for region in self._kv_cache_region_templates.values():
            if region.layer_type == LayerType.FULL_ATTN:
                if region.block_size is None:
                    raise ValueError("full attention region must have "
                                     "block_size")
                return int(region.block_size)
        block_size = getattr(getattr(self.vllm_config, "cache_config", None),
                             "block_size", None)
        if block_size is not None:
            return int(block_size)
        return 1

    def _kv_cache_group_indices(
            self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        group_is_mamba = self._require_group_is_mamba()
        fa_groups = tuple(index
                          for index, is_mamba in enumerate(group_is_mamba)
                          if not is_mamba)
        mamba_groups = tuple(index
                             for index, is_mamba in enumerate(group_is_mamba)
                             if is_mamba)
        return fa_groups, mamba_groups

    def _require_group_is_mamba(self) -> Sequence[bool]:
        group_is_mamba = getattr(self, "group_is_mamba", None)
        if group_is_mamba is None:
            raise RuntimeError("TPUConnectorV2 requires group_is_mamba; "
                               "register_runner must run before building KV "
                               "cache metadata")
        return group_is_mamba

    def _infer_kv_cache_region(
        self,
        *,
        tensor: Any,
        region_id: str,
        nbytes: int,
        page_bytes: int,
    ) -> KVCacheRegion:
        shape = tuple(int(dim) for dim in (getattr(tensor, "shape", ()) or ()))
        group_index = self._group_index_for_region(region_id)
        if group_index is None:
            raise ValueError("TPUConnectorV2 requires KV cache group index "
                             f"for region {region_id}")
        if self._is_mamba_region(region_id, group_index):
            return self._infer_mamba_region(
                region_id=region_id,
                shape=shape,
                page_bytes=page_bytes,
                group_index=group_index,
                itemsize=self._element_size(tensor))
        return self._infer_full_attention_region(
            region_id=region_id,
            shape=shape,
            page_bytes=page_bytes,
            group_index=group_index,
            itemsize=self._element_size(tensor))

    def _infer_full_attention_region(
        self,
        *,
        region_id: str,
        shape: tuple[int, ...],
        page_bytes: int,
        group_index: int,
        itemsize: int,
    ) -> KVCacheRegion:
        physical_block_size = int(shape[1]) if len(shape) >= 2 else 1
        block_size = physical_block_size
        block_bytes = int(page_bytes)
        block_stride_bytes = int(page_bytes)
        num_heads = self._local_full_attention_heads()
        token_first_layout: TokenFirstLayoutSpec | None = None
        token_stride_bytes: int | None = None
        head_stride_bytes: int | None = None
        live_head_bytes: int | None = None
        if len(shape) >= 5:
            block_size = self._logical_full_attention_block_size(
                group_index, physical_block_size)
            if int(page_bytes) % physical_block_size != 0:
                raise ValueError("full attention page_bytes must be divisible "
                                 "by physical block_size")
            token_stride_bytes = int(page_bytes) // physical_block_size
            block_bytes = block_size * token_stride_bytes
            block_stride_bytes = block_bytes
            head_dim = int(shape[-1])
            head_stride_bytes = 2 * head_dim * itemsize
            live_head_bytes = head_stride_bytes
            token_first_layout = TokenFirstLayoutSpec(
                block_size=block_size,
                block_bytes=block_bytes,
                block_stride_bytes=block_stride_bytes,
                token_stride_bytes=token_stride_bytes,
                head_stride_bytes=head_stride_bytes,
                live_head_bytes=live_head_bytes,
                num_heads=num_heads,
            )
        return KVCacheRegion(
            layer_name=region_id,
            layer_type=LayerType.FULL_ATTN,
            block_size=block_size,
            block_bytes=block_bytes,
            block_stride_bytes=block_stride_bytes,
            layout=TensorLayout.TOKEN_FIRST,
            num_heads=num_heads,
            head_bytes=None,
            token_first_layout=token_first_layout,
            token_stride_bytes=token_stride_bytes,
            head_stride_bytes=head_stride_bytes,
            live_head_bytes=live_head_bytes,
            block_id_group_index=group_index,
            head_segments=(),
            physical_region_id=region_id,
            region_base_offset_bytes=0,
        )

    def _logical_full_attention_block_size(
        self,
        group_index: int | None,
        physical_block_size: int,
    ) -> int:
        if group_index is None:
            return physical_block_size
        logical_block_size = int(
            self.runner.kv_cache_config.kv_cache_groups[group_index].
            kv_cache_spec.block_size)
        if logical_block_size < physical_block_size:
            raise ValueError("full attention logical block_size must be >= "
                             "physical block_size")
        if logical_block_size % physical_block_size != 0:
            raise ValueError("full attention logical block_size must be "
                             "divisible by physical block_size")
        return logical_block_size

    def _infer_mamba_region(
        self,
        *,
        region_id: str,
        shape: tuple[int, ...],
        page_bytes: int,
        group_index: int,
        itemsize: int,
    ) -> KVCacheRegion:
        if region_id.endswith(".state0") and len(shape) >= 3:
            return self._infer_mamba_conv_region(
                region_id=region_id,
                shape=shape,
                page_bytes=page_bytes,
                itemsize=itemsize,
                block_id_group_index=group_index)
        if region_id.endswith(".state1") and len(shape) >= 4:
            return self._infer_mamba_recurrent_region(
                region_id=region_id,
                shape=shape,
                page_bytes=page_bytes,
                itemsize=itemsize,
                block_id_group_index=group_index,
            )
        return KVCacheRegion(
            layer_name=region_id,
            layer_type=LayerType.MAMBA_STATE,
            block_size=None,
            block_bytes=page_bytes,
            block_stride_bytes=page_bytes,
            layout=TensorLayout.BLOCKS_FIRST,
            num_heads=1,
            head_bytes=None,
            token_first_layout=None,
            token_stride_bytes=None,
            head_stride_bytes=None,
            live_head_bytes=None,
            block_id_group_index=group_index,
            head_segments=(),
            physical_region_id=region_id,
            region_base_offset_bytes=0,
        )

    def _infer_mamba_conv_region(
        self,
        *,
        region_id: str,
        shape: tuple[int, ...],
        page_bytes: int,
        itemsize: int,
        block_id_group_index: int,
    ) -> KVCacheRegion:
        kernel_size_minus_1 = int(shape[1])
        local_key_heads = self._local_mamba_key_heads()
        local_value_heads = self._local_mamba_value_heads()
        key_head_dim = self._linear_key_head_dim()
        value_head_dim = self._linear_value_head_dim()
        local_dim = int(shape[2])
        slot_stride_bytes = local_dim * itemsize
        q_offset = 0
        k_offset = local_key_heads * key_head_dim * itemsize
        v_offset = 2 * local_key_heads * key_head_dim * itemsize
        return KVCacheRegion(
            layer_name=region_id,
            layer_type=LayerType.MAMBA_STATE,
            block_size=None,
            block_bytes=page_bytes,
            block_stride_bytes=page_bytes,
            layout=TensorLayout.BLOCKS_FIRST,
            num_heads=local_key_heads * 2 + local_value_heads,
            head_bytes=None,
            token_first_layout=None,
            token_stride_bytes=None,
            head_stride_bytes=None,
            live_head_bytes=None,
            head_segments=(
                HeadSegment(name="q",
                            global_heads=self._local_mamba_keyhead_range(),
                            local_head_start=0,
                            local_head_count=local_key_heads,
                            head_bytes=key_head_dim * itemsize,
                            base_offset_bytes=q_offset,
                            stride_bytes=slot_stride_bytes,
                            num_segments=kernel_size_minus_1),
                HeadSegment(name="k",
                            global_heads=self._local_mamba_keyhead_range(),
                            local_head_start=local_key_heads,
                            local_head_count=local_key_heads,
                            head_bytes=key_head_dim * itemsize,
                            base_offset_bytes=k_offset,
                            stride_bytes=slot_stride_bytes,
                            num_segments=kernel_size_minus_1),
                HeadSegment(name="v",
                            global_heads=self._local_mamba_valuehead_range(),
                            local_head_start=local_key_heads * 2,
                            local_head_count=local_value_heads,
                            head_bytes=value_head_dim * itemsize,
                            base_offset_bytes=v_offset,
                            stride_bytes=slot_stride_bytes,
                            num_segments=kernel_size_minus_1),
            ),
            block_id_group_index=block_id_group_index,
            physical_region_id=region_id,
            region_base_offset_bytes=0,
        )

    def _infer_mamba_recurrent_region(
        self,
        *,
        region_id: str,
        shape: tuple[int, ...],
        page_bytes: int,
        itemsize: int,
        block_id_group_index: int,
    ) -> KVCacheRegion:
        local_value_heads = int(shape[1])
        head_bytes = int(shape[2]) * int(shape[3]) * itemsize
        return KVCacheRegion(
            layer_name=region_id,
            layer_type=LayerType.MAMBA_STATE,
            block_size=None,
            block_bytes=page_bytes,
            block_stride_bytes=page_bytes,
            layout=TensorLayout.BLOCKS_FIRST,
            num_heads=max(local_value_heads, 1),
            head_bytes=head_bytes,
            token_first_layout=None,
            token_stride_bytes=None,
            head_stride_bytes=None,
            live_head_bytes=None,
            block_id_group_index=block_id_group_index,
            head_segments=(),
            physical_region_id=region_id,
            region_base_offset_bytes=0,
        )

    def _group_index_for_region(self, region_id: str) -> int | None:
        runner = getattr(self, "runner", None)
        kv_cache_config = getattr(runner, "kv_cache_config", None)
        groups = getattr(kv_cache_config, "kv_cache_groups", None)
        if not groups:
            return None
        layer_name = self._base_layer_name(region_id)
        for group_index, group in enumerate(groups):
            if layer_name in getattr(group, "layer_names", ()):
                return group_index
        return None

    def _is_mamba_region(self, region_id: str, group_index: int) -> bool:
        group_is_mamba = self._require_group_is_mamba()
        if group_index < 0 or group_index >= len(group_is_mamba):
            raise ValueError("KV cache group index out of range: "
                             f"region={region_id} group={group_index} "
                             f"num_groups={len(group_is_mamba)}")
        return bool(group_is_mamba[group_index])

    @staticmethod
    def _base_layer_name(region_id: str) -> str:
        if region_id.endswith(".state0") or region_id.endswith(".state1"):
            return region_id.rsplit(".", 1)[0]
        return region_id

    @staticmethod
    def _element_size(tensor: Any) -> int:
        if hasattr(tensor, "element_size"):
            return int(tensor.element_size())
        dtype = getattr(tensor, "dtype", None)
        itemsize = getattr(dtype, "itemsize", None)
        if itemsize is not None:
            return int(itemsize)
        return 1

    def _model_text_config(self) -> Any:
        model_config = getattr(self.vllm_config, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        if hf_config is None:
            hf_config = model_config
        return getattr(hf_config, "text_config", hf_config)

    def _config_int(self, name: str, default: int) -> int:
        value = getattr(self._model_text_config(), name, None)
        return int(default if value is None else value)

    def _total_num_kv_heads(self) -> int:
        return self._config_int("num_key_value_heads", 1)

    def _total_num_mamba_heads(self) -> int:
        return self._config_int("linear_num_value_heads", 0)

    def _linear_num_key_heads(self) -> int:
        return self._config_int("linear_num_key_heads",
                                self._total_num_mamba_heads())

    def _linear_num_value_heads(self) -> int:
        return self._config_int("linear_num_value_heads",
                                self._total_num_mamba_heads())

    def _linear_key_head_dim(self) -> int:
        return self._config_int("linear_key_head_dim", 1)

    def _linear_value_head_dim(self) -> int:
        return self._config_int("linear_value_head_dim",
                                self._linear_key_head_dim())

    def _local_full_attention_heads(self) -> int:
        return len(self._localhead_range(self._total_num_kv_heads()))

    def _local_mamba_key_heads(self) -> int:
        return len(self._local_mamba_keyhead_range())

    def _local_mamba_value_heads(self) -> int:
        return len(self._local_mamba_valuehead_range())

    def _local_mamba_keyhead_range(self) -> tuple[int, ...]:
        return self._localhead_range(self._linear_num_key_heads())

    def _local_mamba_valuehead_range(self) -> tuple[int, ...]:
        return self._localhead_range(self._linear_num_value_heads())

    def _localhead_range(self, total_heads: int) -> tuple[int, ...]:
        total = int(total_heads)
        tp_size = self._local_tp_size()
        tp_rank = self._local_tp_rank()
        if total <= 0:
            return ()
        if total < tp_size:
            return (tp_rank % total, )
        per_rank = total // tp_size
        if per_rank * tp_size != total:
            raise ValueError(f"total_heads={total} must be divisible by "
                             f"tp_size={tp_size}")
        start = tp_rank * per_rank
        return tuple(range(start, start + per_rank))

    def register_remote_metadata_from_connector_metadata(
            self, metadata: Any) -> None:
        if self.strided_bridge is None:
            return
        transfer_engine = self.strided_bridge.transfer_engine
        reqs_to_load = getattr(metadata, "reqs_to_load", {})
        for req_meta in reqs_to_load.values():
            remote_metadata = self._metadata_value(req_meta, "remote_metadata")
            if remote_metadata:
                transfer_engine.register_other_remote_metadata(
                    self._require_remote_metadata_sequence(remote_metadata))

    def process_send_load(self,
                          metadata: Any,
                          wait_for_completion: bool = False,
                          report_completion: bool = True) -> Any:
        if wait_for_completion:
            raise NotImplementedError(
                "TPUConnectorV2 strided loads use asynchronous completion "
                "reporting; synchronous wait_for_completion is not supported")
        if not report_completion:
            raise NotImplementedError(
                "TPUConnectorV2 requires completion reporting for strided "
                "loads")
        self.register_remote_metadata_from_connector_metadata(metadata)
        self._process_v2_sends(metadata)
        copied, handled_load_ids = self._process_strided_loads(metadata)
        if len(handled_load_ids) != len(
                getattr(metadata, "reqs_to_load", {}) or {}):
            raise RuntimeError(
                "TPUConnectorV2 requires every load to use "
                "remote_rank_ops_by_decode_rank; HMA fallback is disabled")
        return copied

    def _process_v2_sends(self, metadata: Any) -> None:
        reqs_to_send = getattr(metadata, "reqs_to_send", {}) or {}
        if not reqs_to_send:
            return
        if not self._is_host_coordinator:
            return
        coord_send = getattr(self, "_coord_send", None)
        if coord_send is None:
            self._coord_send = {}
            coord_send = self._coord_send
        for req_id, req_meta in reqs_to_send.items():
            self._register_v2_send_lifecycle(req_id, req_meta)

    def _register_v2_send_lifecycle(self, req_id: Any, req_meta: Any) -> None:
        uuid = self._metadata_value(req_meta, "uuid")
        if uuid is None:
            raise RuntimeError("TPUConnectorV2 send metadata requires uuid")
        expiration_time = self._metadata_value(req_meta, "expiration_time")
        if expiration_time is None:
            raise RuntimeError(
                "TPUConnectorV2 send metadata requires expiration_time")
        entry = _V2SendLifecycleEntry(
            req_id=req_id,
            expiration_time=float(expiration_time),
        )
        lock = getattr(self, "_coord_lock", None)
        if lock is None:
            self._coord_send[int(uuid)] = entry
        else:
            with lock:
                self._coord_send[int(uuid)] = entry
        logger.info(
            "TPUConnectorV2Worker(%d) rank0 <-- START registered | "
            "req_id=%s | uuid=%s | expiration_time=%.6f", self.node_id, req_id,
            uuid, float(expiration_time))

    def _process_strided_loads(self, metadata: Any) -> tuple[int, set[Any]]:
        reqs_to_load = getattr(metadata, "reqs_to_load", {}) or {}
        self._prune_strided_recv_lifecycle(reqs_to_load)
        total_copied = 0
        handled: set[Any] = set()
        for req_id, req_meta in reqs_to_load.items():
            if not hasattr(req_meta, "remote_block_ids"):
                raise RuntimeError(
                    "TPUConnectorV2 load metadata requires remote_block_ids")
            if getattr(req_meta, "remote_block_ids") is None:
                handled.add(req_id)
                continue
            has_decode_rank_ops, rank_ops = (
                self._remote_rank_ops_for_local_decode_rank(req_meta))
            if not rank_ops and not has_decode_rank_ops:
                raise RuntimeError(
                    "TPUConnectorV2 load metadata requires "
                    "remote_rank_ops_by_decode_rank; HMA fallback is disabled")
            uuid = self._require_strided_uuid(req_meta)
            if not self._try_mark_strided_recv_started(uuid):
                handled.add(req_id)
                continue
            logger.info(
                "TPUConnectorV2Worker(%d) tp_rank=%d <-- local START | "
                "req_id=%s | uuid=%s | has_decode_rank_ops=%s | "
                "remote_tp_ranks=%s | op_count=%d", self.node_id,
                self._local_tp_rank(), req_id, uuid, has_decode_rank_ops,
                tuple(sorted(rank_ops)),
                sum(len(ops) for ops in rank_ops.values()))
            if not rank_ops:
                self._finish_strided_recv(req_id,
                                          req_meta,
                                          success=True,
                                          error="")
                handled.add(req_id)
                continue
            executor = getattr(self, "_coord_executor", None)
            if executor is not None:
                executor.submit(self._process_one_strided_load_fail_forward,
                                req_id, req_meta, rank_ops)
            else:
                total_copied += self._process_one_strided_load_fail_forward(
                    req_id, req_meta, rank_ops)
            handled.add(req_id)
        return total_copied, handled

    def _process_one_strided_load_fail_forward(
        self,
        req_id: Any,
        req_meta: Any,
        rank_ops: dict[int, tuple[StridedSegmentOp, ...]],
    ) -> int:
        try:
            return self._process_one_strided_load(req_id, req_meta, rank_ops)
        except Exception as exc:
            transfer_stats = getattr(self, "transfer_stats", None)
            if transfer_stats is not None and hasattr(
                    transfer_stats, "record_failed_transfer"):
                transfer_stats.record_failed_transfer()
            try:
                self._finish_strided_recv(req_id,
                                          req_meta,
                                          success=False,
                                          error=str(exc))
            except Exception:
                pass
            raise

    def _process_one_strided_load(
        self,
        req_id: Any,
        req_meta: Any,
        rank_ops: dict[int, tuple[StridedSegmentOp, ...]],
    ) -> int:
        if getattr(req_meta, "remote_block_ids", None) is None:
            return 0

        if self.strided_bridge is None:
            raise RuntimeError("strided transfer bridge is not registered")

        dp_rank = self._require_remote_dp_rank_from_req_meta(req_meta)
        rank_ops = self._apply_region_metadata_to_rank_ops(rank_ops,
                                                           req_meta,
                                                           dp_rank=dp_rank)
        copied = 0
        self._request_v2_pull_start(req_meta)
        write_session = self.new_destination_write_session()
        try:
            for remote_tp_rank in sorted(rank_ops):
                copied += self.pull_rank_ops_into_session(
                    remote_tp_rank=remote_tp_rank,
                    ops=rank_ops[remote_tp_rank],
                    write_session=write_session,
                    dp_rank=dp_rank)
            write_session.flush_all()
        except Exception:
            write_session.discard()
            raise
        self._finish_strided_recv(req_id, req_meta, success=True, error="")
        return copied

    def _finish_strided_recv(self, req_id: Any, req_meta: Any, *,
                             success: bool, error: str) -> None:
        if not success and not error:
            raise ValueError("failed strided recv completion requires error")
        uuid = self._require_strided_uuid(req_meta)
        tp_rank = self._local_tp_rank()
        expected_tp_ranks = self._expected_strided_decode_tp_ranks(req_meta)
        if tp_rank not in expected_tp_ranks:
            raise RuntimeError("TPUConnectorV2 load metadata has no entry for "
                               f"local decode TP rank {tp_rank}")
        completion = TPUConnectorV2WorkerCompletion(
            req_id=req_id,
            uuid=uuid,
            dp_rank=self._local_dp_rank(),
            tp_rank=tp_rank,
            success=success,
            error=error,
        )
        with self._strided_lifecycle_lock:
            if uuid in self._strided_recv_done_uuids:
                return
            self._strided_recv_done_uuids.add(uuid)
            self._pending_v2_worker_completions.append(completion)
        logger.info(
            "TPUConnectorV2Worker(%d) tp_rank=%d --> local END queued | "
            "req_id=%s | uuid=%s | success=%s | error=%s", self.node_id,
            tp_rank, req_id, uuid, success, error)

    @staticmethod
    def _require_strided_uuid(req_meta: Any) -> int:
        uuid = TPUConnectorV2Worker._metadata_value(req_meta, "uuid")
        if uuid is None:
            raise RuntimeError("TPUConnectorV2 load metadata requires uuid")
        return int(uuid)

    def _try_mark_strided_recv_started(self, uuid: int) -> bool:
        with self._strided_lifecycle_lock:
            if uuid in self._strided_recv_started_uuids:
                return False
            if uuid in self._strided_recv_done_uuids:
                return False
            self._strided_recv_started_uuids.add(uuid)
            return True

    def _request_v2_pull_start(self, req_meta: Any) -> None:
        uuid = self._require_strided_uuid(req_meta)
        for host, port in self._resolve_v2_pull_start_targets(req_meta):
            logger.info(
                "TPUConnectorV2 strided lifecycle send START | uuid=%s | "
                "remote=%s:%s", uuid, host, port)
            frames = zmq_side_channel.send_request(
                host=host,
                port=port,
                tag=zmq_side_channel.PULL_START,
                uuid=uuid,
                timeout_s=self._strided_timeout_s(),
                action="pull-start",
            )
            if frames == [zmq_side_channel.MSG_OK]:
                logger.info(
                    "TPUConnectorV2 strided lifecycle START ack OK | uuid=%s | "
                    "remote=%s:%s", uuid, host, port)
                continue
            if len(frames) == 2 and frames[0] == zmq_side_channel.MSG_ERR:
                message = frames[1].decode("utf-8", errors="replace")
                raise RuntimeError("producer rejected strided KV pull start: "
                                   f"{message}")
            raise RuntimeError(
                "malformed producer strided pull-start response: "
                f"{frames!r}")

    def _resolve_v2_pull_start_endpoint(self,
                                        req_meta: Any) -> tuple[str, int]:
        targets = self._resolve_v2_pull_start_targets(req_meta)
        if len(targets) != 1:
            raise ValueError("TPUConnectorV2 ACK metadata contains multiple "
                             "side-channel targets")
        return targets[0]

    def _resolve_v2_pull_start_targets(
            self, req_meta: Any) -> tuple[V2LifecycleTarget, ...]:
        host = self._metadata_value(req_meta, "v2_ack_host")
        port = self._metadata_value(req_meta, "v2_ack_port")
        if host is None or port is None:
            raise RuntimeError("TPUConnectorV2 load metadata requires both "
                               "v2_ack_host and v2_ack_port")
        return _resolve_v2_lifecycle_targets(host, port, label="v2_ack")

    @staticmethod
    def _local_node_id() -> int:
        from vllm_torchtpu.distributed import utils as dist_utils

        return int(dist_utils.get_node_id())

    def _coord_rank0_external_notif_loop(self) -> None:
        zmq_side_channel.serve_lifecycle_requests(
            node_id=int(self.node_id),
            zmq_context=self.zmq_cxt,
            side_channel_port=int(self.side_channel_port),
            stop_event=self._stop_event,
            handle_pull_start=self._coord_rank0_handle_v2_pull_start,
            handle_pull_end=self._coord_rank0_handle_v2_pull_end,
            log=logger,
        )

    def _coord_rank0_handle_v2_pull_start(
            self,
            uuid: int,
            wait_for_register: bool = True) -> tuple[bool, str]:
        timeout = float(self._strided_timeout_s())
        deadline = time.perf_counter() + timeout
        logger.info("TPUConnectorV2Worker(%d) rank0 <-- recv START | uuid=%s",
                    self.node_id, uuid)
        while True:
            now = time.perf_counter()
            message: str | None = None
            expired = False
            with self._coord_lock:
                entry = self._coord_send.get(uuid)
                if entry is None:
                    message = f"unknown uuid={uuid}"
                elif now > entry.expiration_time:
                    self._coord_send.pop(uuid, None)
                    self._coord_done_sending.add(entry.req_id)
                    message = f"expired uuid={uuid}"
                    expired = True
                else:
                    entry.pull_started = True
                    entry.expiration_time = max(entry.expiration_time,
                                                now + timeout)
                    logger.info(
                        "TPUConnectorV2Worker(%d) rank0 --> accept START | "
                        "uuid=%s | req_id=%s | lease_extended_by=%.1fs",
                        self.node_id, uuid, entry.req_id, timeout)
                    return True, "ok"
            if (not wait_for_register or expired
                    or time.perf_counter() >= deadline):
                logger.warning(
                    "TPUConnectorV2Worker(%d) rank0 --> reject START | %s",
                    self.node_id, message)
                return False, str(message)
            time.sleep(0.05)

    def _coord_rank0_handle_v2_pull_end(self, uuid: int) -> tuple[bool, str]:
        now = time.perf_counter()
        logger.info("TPUConnectorV2Worker(%d) rank0 <-- recv END | uuid=%s",
                    self.node_id, uuid)
        with self._coord_lock:
            entry = self._coord_send.get(uuid)
            if entry is None:
                message = f"unknown uuid={uuid}"
            elif now > entry.expiration_time:
                self._coord_send.pop(uuid, None)
                self._coord_done_sending.add(entry.req_id)
                message = f"expired uuid={uuid}"
            elif not entry.pull_started:
                message = f"end before start uuid={uuid}"
            else:
                entry.pull_acked = True
                logger.info(
                    "TPUConnectorV2Worker(%d) rank0 --> accept END | uuid=%s | "
                    "req_id=%s | lease_valid=True", self.node_id, uuid,
                    entry.req_id)
                return True, "ok"
        logger.warning("TPUConnectorV2Worker(%d) rank0 --> reject END | %s",
                       self.node_id, message)
        return False, message

    def _mark_strided_recv_done(self, uuid: int) -> bool:
        with self._strided_lifecycle_lock:
            if uuid in self._strided_recv_done_uuids:
                return False
            self._strided_recv_done_uuids.add(uuid)
            return True

    def _strided_recv_is_done(self, uuid: int) -> bool:
        with self._strided_lifecycle_lock:
            return uuid in self._strided_recv_done_uuids

    def _prune_strided_recv_lifecycle(self,
                                      reqs_to_load: Mapping[Any, Any]) -> None:
        active_uuids: set[int] = set()
        for req_meta in reqs_to_load.values():
            if getattr(req_meta, "remote_block_ids", None) is None:
                continue
            uuid = self._metadata_value(req_meta, "uuid")
            if uuid is not None:
                active_uuids.add(int(uuid))
        with self._strided_lifecycle_lock:
            self._strided_recv_started_uuids.intersection_update(active_uuids)
            self._strided_recv_done_uuids.intersection_update(active_uuids)

    def _expected_strided_decode_tp_ranks(self, req_meta: Any) -> set[int]:
        by_decode_rank = self._remote_rank_ops_by_decode_rank_from_req_meta(
            req_meta)
        if by_decode_rank:
            return set(by_decode_rank)
        return {0}

    def _apply_region_metadata_to_rank_ops(
        self,
        rank_ops: dict[int, tuple[StridedSegmentOp, ...]],
        req_meta: Any,
        *,
        dp_rank: int,
    ) -> dict[int, tuple[StridedSegmentOp, ...]]:
        remote_metadata = self._metadata_value(req_meta, "remote_metadata", ())
        source_region_ids_by_rank = self._remote_region_ids_by_rank(
            remote_metadata, dp_rank=dp_rank)
        destination_region_ids = self._local_destination_region_ids()
        if not source_region_ids_by_rank and not destination_region_ids:
            return rank_ops

        self._validate_region_metadata_for_rank_ops(
            rank_ops,
            source_region_ids_by_rank,
            destination_region_ids,
        )
        return rank_ops

    @classmethod
    def _validate_region_metadata_for_rank_ops(
        cls,
        rank_ops: dict[int, tuple[StridedSegmentOp, ...]],
        source_region_ids_by_rank: Mapping[int, set[str]],
        destination_region_ids: set[str],
    ) -> None:
        for remote_tp_rank, ops in rank_ops.items():
            source_region_ids = source_region_ids_by_rank.get(
                int(remote_tp_rank), set())
            for op in ops:
                cls._validate_region_metadata_for_op(
                    op,
                    source_region_ids,
                    destination_region_ids,
                    remote_tp_rank=int(remote_tp_rank),
                )

    @classmethod
    def _validate_region_metadata_for_op(
        cls,
        op: StridedSegmentOp,
        source_region_ids: set[str],
        destination_region_ids: set[str],
        *,
        remote_tp_rank: int,
    ) -> None:
        if source_region_ids and op.source_region_id not in source_region_ids:
            raise ValueError("remote metadata for TP rank "
                             f"{remote_tp_rank} has no region for "
                             f"{op.source_region_id!r}")
        if (destination_region_ids
                and op.destination_region_id not in destination_region_ids):
            raise ValueError("local metadata has no destination region for "
                             f"{op.destination_region_id!r}")

    def _local_destination_region_ids(self) -> set[str]:
        engine = getattr(self, "strided_transfer_engine", None)
        if engine is None:
            return set()
        if hasattr(engine, "local_regions_metadata"):
            return self._region_ids_by_id(engine.local_regions_metadata())
        if hasattr(engine, "local_metadata"):
            return self._region_ids_from_metadata(engine.local_metadata())
        return set()

    @staticmethod
    def apply_remote_source_region_metadata(
        metadata: ConnectorMetadataV2,
        remote_metadata: Any,
        *,
        dp_rank: int,
    ) -> ConnectorMetadataV2:
        source_region_ids_by_rank = (
            TPUConnectorV2Worker._remote_region_ids_by_rank(remote_metadata,
                                                            dp_rank=dp_rank))
        if not source_region_ids_by_rank:
            return metadata
        for rank, regions in metadata.kv_caches.items():
            source_region_ids = source_region_ids_by_rank.get(int(rank))
            if not source_region_ids:
                continue
            for layer_name, region in regions.items():
                physical_region_id = str(region.physical_region_id)
                if physical_region_id not in source_region_ids:
                    raise ValueError("remote metadata for TP rank "
                                     f"{rank} has no region for "
                                     f"{physical_region_id!r} "
                                     f"required by {layer_name!r}")
        return metadata

    @staticmethod
    def apply_local_destination_region_metadata(
        destination: LocalDecodeAllocation,
        local_regions: Any,
    ) -> LocalDecodeAllocation:
        destination_region_ids = TPUConnectorV2Worker._region_ids_by_id(
            local_regions)
        if not destination_region_ids:
            return destination
        for layer_name, region in destination.kv_caches.items():
            physical_region_id = str(region.physical_region_id)
            if physical_region_id not in destination_region_ids:
                raise ValueError(
                    "local metadata has no destination region for "
                    f"{physical_region_id!r} required by "
                    f"{layer_name!r}")
        return destination

    def _remote_rank_ops_for_local_decode_rank(
        self,
        req_meta: Any,
    ) -> tuple[bool, dict[int, tuple[StridedSegmentOp, ...]]]:
        by_decode_rank = self._remote_rank_ops_by_decode_rank_from_req_meta(
            req_meta)
        if not by_decode_rank:
            return False, {}
        return True, by_decode_rank.get(self._local_tp_rank(), {})

    @classmethod
    def _remote_rank_ops_by_decode_rank_from_req_meta(
        cls,
        req_meta: Any,
    ) -> dict[int, dict[int, tuple[StridedSegmentOp, ...]]]:
        value = cls._metadata_value(req_meta, "remote_rank_ops_by_decode_rank")
        if not value:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("remote_rank_ops_by_decode_rank must be a mapping "
                            "from decode TP rank to rank ops")
        return {
            int(decode_rank): cls._require_rank_ops(rank_ops)
            for decode_rank, rank_ops in value.items()
        }

    @classmethod
    def remote_rank_ops_by_decode_rank_from_wire(
        cls,
        value: Mapping[Any, Mapping[Any, Sequence[Mapping[str, Any]]]],
    ) -> dict[int, dict[int, tuple[StridedSegmentOp, ...]]]:
        if not isinstance(value, Mapping):
            raise TypeError("remote_rank_ops_by_decode_rank must be a mapping "
                            "from decode TP rank to rank ops")
        return {
            int(decode_rank): cls._rank_ops_from_wire(rank_ops)
            for decode_rank, rank_ops in value.items() if rank_ops
        }

    @classmethod
    def _rank_ops_from_wire(
        cls,
        value: Mapping[Any, Sequence[Mapping[str, Any]]],
    ) -> dict[int, tuple[StridedSegmentOp, ...]]:
        if not isinstance(value, Mapping):
            raise TypeError("rank ops must be a mapping from remote TP rank "
                            "to an op list")
        return {
            int(rank): cls._strided_segment_ops_from_wire(rank_value)
            for rank, rank_value in value.items() if rank_value
        }

    @staticmethod
    def _strided_segment_ops_from_wire(
        ops: Sequence[Mapping[str, Any]], ) -> tuple[StridedSegmentOp, ...]:
        if isinstance(ops, Mapping):
            raise TypeError("rank ops values must be an op sequence")
        if isinstance(ops, (str, bytes, bytearray)):
            raise TypeError("rank ops values must be an op sequence")
        result = []
        for op in ops:
            if not isinstance(op, Mapping):
                raise TypeError("wire op list must contain only mappings")
            result.append(StridedSegmentOp.from_mapping(op))
        return tuple(result)

    @staticmethod
    def _require_strided_segment_ops(
        ops: Sequence[StridedSegmentOp], ) -> tuple[StridedSegmentOp, ...]:
        if isinstance(ops, Mapping):
            raise TypeError("rank ops values must be an op sequence")
        if isinstance(ops, (str, bytes, bytearray)):
            raise TypeError("rank ops values must be an op sequence")
        result = tuple(ops)
        for op in result:
            if not isinstance(op, StridedSegmentOp):
                raise TypeError("rank ops values must contain only "
                                "StridedSegmentOp")
        return result

    @staticmethod
    def _require_remote_metadata_sequence(
        remote_metadata: Sequence[RemoteWorkerMetadata],
    ) -> tuple[RemoteWorkerMetadata, ...]:
        if isinstance(remote_metadata, RemoteWorkerMetadata):
            raise TypeError("remote_metadata must be a sequence of "
                            "RemoteWorkerMetadata")
        if isinstance(remote_metadata, (str, bytes, bytearray)):
            raise TypeError("remote_metadata must be a sequence of "
                            "RemoteWorkerMetadata")
        result = tuple(remote_metadata)
        for item in result:
            if not isinstance(item, RemoteWorkerMetadata):
                raise TypeError("remote_metadata must contain only "
                                "RemoteWorkerMetadata")
        return result

    @classmethod
    def _require_rank_ops(
        cls, value: Mapping[Any, Sequence[StridedSegmentOp]]
    ) -> dict[int, tuple[StridedSegmentOp, ...]]:
        if not isinstance(value, Mapping):
            raise TypeError("rank ops must be a mapping from remote "
                            "TP rank to an op list")
        return {
            int(rank): cls._require_strided_segment_ops(rank_value)
            for rank, rank_value in value.items() if rank_value
        }

    @staticmethod
    def _metadata_value(metadata: Any, key: str, default: Any = None) -> Any:
        if isinstance(metadata, Mapping):
            return metadata.get(key, default)
        return getattr(metadata, key, default)

    @classmethod
    def _remote_region_ids_by_rank(
        cls,
        remote_metadata: Sequence[RemoteWorkerMetadata],
        *,
        dp_rank: int,
    ) -> dict[int, set[str]]:
        if not remote_metadata:
            return {}
        items = cls._require_remote_metadata_sequence(remote_metadata)
        result: dict[int, set[str]] = {}
        for item in items:
            if int(item.dp_rank) != int(dp_rank):
                continue
            regions = cls._region_ids_by_id(item.regions)
            if regions:
                result[int(item.tp_rank)] = regions
        return result

    @staticmethod
    def _region_ids_by_id(
        regions: Sequence[RegisteredMemoryRegion], ) -> set[str]:
        if not regions:
            return set()
        if isinstance(regions, RegisteredMemoryRegion):
            raise TypeError("regions must be a sequence of "
                            "RegisteredMemoryRegion")
        if isinstance(regions, (str, bytes, bytearray)):
            raise TypeError("regions must be a sequence of "
                            "RegisteredMemoryRegion")
        result: set[str] = set()
        for region in regions:
            if not isinstance(region, RegisteredMemoryRegion):
                raise TypeError("regions must contain only "
                                "RegisteredMemoryRegion")
            region_key = region.region_id
            if region_key in result:
                raise ValueError(f"duplicate memory region id: {region_key!r}")
            result.add(region_key)
        return result

    @classmethod
    def _region_ids_from_metadata(cls,
                                  metadata: RemoteWorkerMetadata) -> set[str]:
        if not isinstance(metadata, RemoteWorkerMetadata):
            raise TypeError("metadata must be RemoteWorkerMetadata")
        return cls._region_ids_by_id(metadata.regions)

    @classmethod
    def _remote_dp_rank_from_req_meta(cls, req_meta: Any) -> int | None:
        remote_dp_rank = cls._metadata_value(req_meta, "remote_dp_rank")
        if remote_dp_rank is not None:
            return int(remote_dp_rank)
        remote_metadata = cls._metadata_value(req_meta, "remote_metadata", ())
        if not remote_metadata:
            return None
        items = cls._require_remote_metadata_sequence(remote_metadata)
        dp_ranks = {int(item.dp_rank) for item in items}
        return next(iter(dp_ranks)) if len(dp_ranks) == 1 else None

    @classmethod
    def _require_remote_dp_rank_from_req_meta(cls, req_meta: Any) -> int:
        dp_rank = cls._remote_dp_rank_from_req_meta(req_meta)
        if dp_rank is None:
            raise ValueError(
                "remote_dp_rank must be provided or uniquely inferable from "
                "remote_metadata before strided KV transfer")
        return dp_rank

    def _coord_rank0_expire_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._stop_event.wait(timeout=1.0):
                return
            now = time.perf_counter()
            expired: list[int] = []
            with self._coord_lock:
                for uuid, entry in list(self._coord_send.items()):
                    if not entry.pull_acked and now > entry.expiration_time:
                        expired.append(uuid)
            for uuid in expired:
                with self._coord_lock:
                    entry = self._coord_send.pop(uuid, None)
                    if entry is None:
                        continue
                    self._coord_done_sending.add(entry.req_id)
                if entry.slot_idx >= 0:
                    self._coord_pool.release_slot(entry.slot_idx)

    def _coord_get_finished(self) -> tuple[set[str], set[str]]:
        if not self._is_host_coordinator:
            return set(), set()
        done_sending: set[str] = set()
        done_recving: set[str] = set()
        released_slots: list[int] = []
        with self._coord_lock:
            for uuid in list(self._coord_send.keys()):
                entry = self._coord_send[uuid]
                if entry.pull_acked:
                    done_sending.add(entry.req_id)
                    del self._coord_send[uuid]
                    if entry.slot_idx >= 0:
                        released_slots.append(entry.slot_idx)
            if self._coord_done_sending:
                done_sending |= self._coord_done_sending
                self._coord_done_sending.clear()
            if self._coord_done_recving:
                done_recving |= self._coord_done_recving
                self._coord_done_recving.clear()
        for slot_idx in released_slots:
            self._coord_pool.release_slot(slot_idx)
        return done_sending, done_recving

    def pull_rank_ops_into_session(
        self,
        *,
        remote_tp_rank: int,
        ops: Sequence[StridedSegmentOp],
        write_session: Any,
        dp_rank: int,
    ) -> int:
        if not ops:
            return 0
        if self.strided_bridge is None:
            raise RuntimeError("strided transfer bridge is not registered")
        return self.strided_bridge.pull_rank_ops_into_session(
            remote_tp_rank=remote_tp_rank,
            ops=ops,
            write_session=write_session,
            dp_rank=dp_rank,
        )

    def new_destination_write_session(self) -> Any:
        if self.strided_bridge is None:
            raise RuntimeError("strided transfer bridge is not registered")
        return self.strided_bridge.new_destination_write_session()

    def build_connector_worker_meta(self) -> TPUConnectorV2WorkerMeta | None:
        with self._strided_lifecycle_lock:
            if not self._pending_v2_worker_completions:
                return None
            completions = tuple(self._pending_v2_worker_completions)
            self._pending_v2_worker_completions.clear()
        logger.info(
            "TPUConnectorV2Worker(%d) --> END completion meta | "
            "completions=%s", self.node_id,
            tuple((completion.req_id, completion.uuid, completion.tp_rank,
                   completion.success) for completion in completions))
        return TPUConnectorV2WorkerMeta(completions=completions)


class TPUConnectorV2(KVConnectorBase_V1, SupportsHMA):
    """V2 connector backed by unified-block-pool strided transfer."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Any = None,
        **kwargs: Any,
    ):
        del kwargs
        super().__init__(vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        if not tpu_envs.TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL:
            raise ValueError("TPUConnectorV2 requires "
                             "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1")
        self._connector_metadata = None

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = TPUConnectorV2Scheduler(vllm_config)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = TPUConnectorV2Worker(vllm_config)
        else:
            raise ValueError(f"Invalid role: {role}")

    def get_num_new_matched_tokens(
        self,
        request: Any,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens)

    def update_state_after_alloc(self, request: Any, blocks: Any,
                                 num_external_tokens: int) -> Any:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: Any = None) -> Any:
        del scheduler_output
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta()

    def request_finished_all_groups(
        self,
        request: Any,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Any]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished_all_groups(
            request, block_ids)

    def request_finished(self, request: Any,
                         block_ids: list[int]) -> tuple[bool, Any]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def get_finished_count(self) -> int:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_finished_count()

    def get_kv_connector_stats(self) -> Any | None:
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_kv_connector_stats()

    @classmethod
    def build_kv_connector_stats(cls,
                                 data: dict[str, Any] | None = None) -> Any:
        from vllm_torchtpu.distributed.kv_transfer.tpu_connector_stats import \
            TpuKVConnectorStats

        return TpuKVConnectorStats(
            data=data) if data is not None else (TpuKVConnectorStats())

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[Any], type[Any]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> Any:
        from vllm_torchtpu.distributed.kv_transfer.tpu_connector_stats import \
            TpuKVConnectorPromMetrics

        return TpuKVConnectorPromMetrics(vllm_config, metric_types, labelnames,
                                         per_engine_labelvalues)

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        if self.connector_worker is not None:
            self.connector_worker.named_kv_caches = kv_caches

    def register_runner(self, runner: Any) -> None:
        assert self.connector_worker is not None
        self.connector_worker.register_runner(runner)

    def start_load_kv(self,
                      _,
                      wait_for_completion: bool = False,
                      report_completion: bool = True,
                      **kwargs: Any) -> None:
        del kwargs
        assert self.connector_worker is not None
        assert self._connector_metadata is not None
        self.connector_worker.process_send_load(
            self._connector_metadata,
            wait_for_completion=wait_for_completion,
            report_completion=report_completion)

    def wait_for_layer_load(self, layer_name: str) -> None:
        del layer_name

    def save_kv_layer(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def wait_for_save(self) -> None:
        return

    def get_finished(
            self,
            finished_req_ids: set[str] | None = None
    ) -> tuple[set[str], set[str]]:
        del finished_req_ids
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def get_handshake_metadata(self) -> TPUConnectorV2HandshakeMetadata | None:
        assert self.connector_worker is not None
        return self.connector_worker.get_handshake_metadata()

    def set_xfer_handshake_metadata(
        self,
        metadata: Mapping[int, TPUConnectorV2HandshakeMetadata],
    ) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.set_xfer_handshake_metadata(metadata)

    def update_connector_output(self, connector_output: Any) -> None:
        if self.connector_scheduler is not None:
            self.connector_scheduler.update_connector_output(connector_output)

    def build_connector_worker_meta(self) -> TPUConnectorV2WorkerMeta | None:
        if self.connector_worker is None:
            return None
        return self.connector_worker.build_connector_worker_meta()


TPUConnector = TPUConnectorV2
TPUConnectorScheduler = TPUConnectorV2Scheduler
TPUConnectorWorker = TPUConnectorV2Worker

__all__ = [
    "ConnectorMetadataV2",
    "ContiguousHeadTPTransferPlanner",
    "HeadSegment",
    "HeadMapping",
    "KVCacheRegion",
    "KVParallelLayout",
    "LayerType",
    "LocalDecodeAllocation",
    "PullMeta",
    "PcpReshardingPolicy",
    "RankTransferPlan",
    "RegisteredMemoryRegion",
    "SourceBlockRef",
    "StridedSegmentOp",
    "StridedKVTransferEngine",
    "TPTransferPlanner",
    "TPUConnector",
    "TPUConnectorScheduler",
    "TPUConnectorV2",
    "TPUConnectorV2HandshakeMetadata",
    "TPUConnectorV2Scheduler",
    "TPUConnectorV2StridedBridge",
    "TPUConnectorV2WorkerCompletion",
    "TPUConnectorV2WorkerMeta",
    "TPUConnectorV2Worker",
    "TPUConnectorWorker",
    "TensorLayout",
    "TokenFirstLayoutSpec",
    "TpKVTopology",
]
