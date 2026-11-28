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
"""Direct P2P disaggregated Prefill/Decode KV connector using Mooncake TENT.

`TPUMooncakeConnector` transfers paged KV blocks directly between Prefill
(`kv_producer`) and Decode (`kv_consumer`) TPU HBM pools over Mooncake TENT's
`TransferEngine` (`P2PHANDSHAKE` mode, requiring no external `mooncake_master`).

Enable via:

    # Prefill engine
    --kv-transfer-config '{
        "kv_connector": "TPUMooncakeConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "protocol": "hp_tcp",
            "ctrl_port": 27600
        }
    }'

    # Decode engine
    --kv-transfer-config '{
        "kv_connector": "TPUMooncakeConnector",
        "kv_role": "kv_consumer",
        "kv_connector_extra_config": {
            "protocol": "hp_tcp"
        }
    }'

Transfer lifecycle:
1. Worker init (`register_kv_caches`): each rank deduplicates its TPU unified
   block pool (UBP) storages via `canonicalize_tpu_kv_pools`, exports them to
   `mooncake.tpu` using `tpu_sync.solutions.mooncake.torch.mooncake_provider`,
   initializes a TENT `TransferEngine`, and registers the shard fake addresses.
   Producer ranks also start a lightweight per-rank ZMQ ROUTER control thread.
2. Prefill finish (`request_finished`): holds source KV blocks (`delay_free=True`)
   and returns `kv_transfer_params` containing `uuid`, `remote_block_ids`,
   `remote_ctrl_endpoints`, and `remote_tp_size`.
3. Prefill write fence (`start_load_kv` / `get_transfer_results`): records a
   non-blocking TorchTPU stream event on the worker main thread for each new
   send entry and marks the entry readable only once that event retires.
4. Decode pull (`start_load_kv`): skips locally prefix-cached blocks, queries
   its matching Prefill rank's control endpoint (`ready`), coalesces consecutive
   `(remote_block_id, local_block_id)` pairs into contiguous transfers, executes
   `TransferEngine.batch_transfer_sync_read`, and sends `done` so Prefill frees
   its delayed HBM blocks.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import torch
import zmq
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorTransferResults,
    SupportsHMA,
)
from vllm.utils.math_utils import round_down
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import RequestStatus

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.distributed import utils as dist_utils
from vllm_torchtpu.distributed.kv_transfer import mooncake_tpu_utils as mc_utils
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)

ReqId = str

_KV_PARAMS_ADMITTED = "_tpu_mooncake_kv_params_admitted"
_KV_PARAMS_REJECTED = "_tpu_mooncake_kv_params_rejected"
_KV_PARAMS_PROCESSED = "_tpu_mooncake_kv_params_processed"


def _new_uuid() -> int:
    return uuid4().int >> 64


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _as_int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    out: list[int] = []
    for item in value:
        parsed = _as_int(item)
        if parsed is None:
            return None
        out.append(parsed)
    return out


def _as_str_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not value:
        return None
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return None
        out.append(item.strip())
    return out


@dataclass(frozen=True)
class MooncakeP2PConfig:
    """Resolved runtime configuration for `TPUMooncakeConnector`."""

    protocol: str
    metadata_server: str
    device_name: str
    host_ip: str
    ctrl_port_base: int
    ctrl_bind_host: str
    wait_pull_timeout_s: float
    ready_timeout_s: float
    tent_connections: int
    transfer_threads: int
    remote_hosts: tuple[str, ...]
    tent_extra_config: Mapping[str, Any] | None = None


def parse_mooncake_p2p_config(
    vllm_config: VllmConfig,
    extra_config: Mapping[str, Any],
) -> MooncakeP2PConfig:
    """Resolve `TPUMooncakeConnector` settings from `kv_connector_extra_config`
    and `vllm_torchtpu.envs`."""
    protocol = (
        str(extra_config.get("protocol") or tpu_envs.TPU_MOONCAKE_PROTOCOL)
        .strip()
        .lower()
    )
    metadata_server = str(extra_config.get("metadata_server") or "P2PHANDSHAKE").strip()
    device_name = str(
        extra_config.get("device_name") or tpu_envs.TPU_MOONCAKE_DEVICE_NAME
    ).strip()

    host_ip = str(extra_config.get("local_hostname") or "").strip()
    if not host_ip:
        host_ip = dist_utils.get_host_ip()

    ctrl_port_base = int(extra_config.get("ctrl_port", tpu_envs.TPU_MOONCAKE_CTRL_PORT))
    if ctrl_port_base < 0 or ctrl_port_base > 65535:
        raise ValueError(f"ctrl_port must be in [0, 65535], got {ctrl_port_base}")
    ctrl_bind_host = str(extra_config.get("ctrl_bind_host") or "*").strip()

    wait_pull_timeout_s = float(
        extra_config.get("wait_pull_timeout_s", dist_utils.get_p2p_wait_pull_timeout())
    )
    if wait_pull_timeout_s <= 0:
        raise ValueError(
            f"wait_pull_timeout_s must be positive, got {wait_pull_timeout_s}"
        )
    default_ready_timeout = min(30.0, wait_pull_timeout_s)
    ready_timeout_s = float(extra_config.get("ready_timeout_s", default_ready_timeout))
    if ready_timeout_s <= 0:
        raise ValueError(f"ready_timeout_s must be positive, got {ready_timeout_s}")

    tent_connections = int(
        extra_config.get("tent_connections", tpu_envs.TPU_MOONCAKE_TENT_CONNECTIONS)
    )
    transfer_threads = max(
        1,
        int(
            extra_config.get("transfer_threads", tpu_envs.TPU_MOONCAKE_TRANSFER_THREADS)
        ),
    )

    tp_size = (
        int(vllm_config.parallel_config.tensor_parallel_size)
        if vllm_config.parallel_config is not None
        else 1
    )
    raw_hosts = extra_config.get("remote_hosts")
    if raw_hosts is not None:
        if isinstance(raw_hosts, str):
            hosts = [h.strip() for h in raw_hosts.split(",") if h.strip()]
        elif isinstance(raw_hosts, Sequence):
            hosts = [str(h).strip() for h in raw_hosts if str(h).strip()]
        else:
            raise ValueError("remote_hosts must be a sequence or comma-separated str")
        if len(hosts) == 1:
            remote_hosts = tuple(hosts * tp_size)
        elif len(hosts) == tp_size:
            remote_hosts = tuple(hosts)
        else:
            raise ValueError(
                f"remote_hosts length ({len(hosts)}) must be 1 or tp_size ({tp_size})"
            )
    else:
        kv_ips = dist_utils.get_kv_ips()
        if isinstance(kv_ips, list) and len(kv_ips) == tp_size:
            remote_hosts = tuple(str(ip) for ip in kv_ips)
        elif isinstance(kv_ips, list) and kv_ips:
            # Spread TP ranks evenly across hosts in multi-host slices.
            ranks_per_host = max(1, tp_size // len(kv_ips))
            remote_hosts = tuple(
                str(kv_ips[min(r // ranks_per_host, len(kv_ips) - 1)])
                for r in range(tp_size)
            )
        else:
            remote_hosts = tuple([host_ip] * tp_size)

    tent_extra = extra_config.get("tent_config")
    if tent_extra is not None and not isinstance(tent_extra, Mapping):
        raise ValueError("kv_connector_extra_config['tent_config'] must be a mapping")

    return MooncakeP2PConfig(
        protocol=protocol,
        metadata_server=metadata_server,
        device_name=device_name,
        host_ip=host_ip,
        ctrl_port_base=ctrl_port_base,
        ctrl_bind_host=ctrl_bind_host,
        wait_pull_timeout_s=wait_pull_timeout_s,
        ready_timeout_s=ready_timeout_s,
        tent_connections=tent_connections,
        transfer_threads=transfer_threads,
        remote_hosts=remote_hosts,
        tent_extra_config=tent_extra,
    )


@dataclass
class MooncakeHandshakeMetadata(KVConnectorHandshakeMetadata):
    """Per-rank worker handshake metadata exposed via `get_handshake_metadata`."""

    transfer_rank: int
    session_id: str
    ctrl_endpoint: str
    pool_addrs: list[int]
    pool_page_sizes: list[int]
    tp_size: int


@dataclass
class MooncakeP2PSendMeta:
    """Producer-side metadata for one finished prompt awaiting remote pull."""

    uuid: int
    local_block_ids: list[int]
    expiration_time: float


@dataclass
class MooncakeP2PLoadMeta:
    """Consumer-side metadata for one remote KV pull (or release-only notify)."""

    uuid: int
    local_block_ids: list[int] | None
    remote_block_ids: list[int] | None
    remote_ctrl_endpoints: list[str]
    release_only: bool = False
    report_completion: bool = True


@dataclass
class MooncakeP2PMetadata(KVConnectorMetadata):
    """Scheduler -> worker per-step metadata for `TPUMooncakeConnector`."""

    reqs_to_send: dict[ReqId, MooncakeP2PSendMeta] = field(default_factory=dict)
    reqs_to_load: dict[ReqId, MooncakeP2PLoadMeta] = field(default_factory=dict)


def _create_default_transfer_engine() -> Any:
    try:
        from mooncake.engine import TransferEngine
    except ImportError as exc:
        raise ImportError(
            "TPUMooncakeConnector requires `mooncake.engine.TransferEngine` "
            "(Mooncake built with `-DUSE_TENT=ON -DUSE_TPU=ON`)."
        ) from exc
    return TransferEngine()


def _resolve_tp_rank_and_size(vllm_config: VllmConfig) -> tuple[int, int]:
    tp_size = (
        int(vllm_config.parallel_config.tensor_parallel_size)
        if vllm_config.parallel_config is not None
        else 1
    )
    try:
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        return (
            int(get_tensor_model_parallel_rank()),
            int(get_tensor_model_parallel_world_size()),
        )
    except Exception:
        return 0, tp_size


class TPUMooncakeConnectorScheduler:
    """Scheduler half of `TPUMooncakeConnector` (Prefill and Decode roles)."""

    def __init__(self, vllm_config: VllmConfig, config: MooncakeP2PConfig):
        self.vllm_config = vllm_config
        assert vllm_config.kv_transfer_config is not None
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.config = config
        self.is_producer = bool(self.kv_transfer_config.is_kv_producer)
        self.block_size = int(vllm_config.cache_config.block_size)

        pc = vllm_config.parallel_config
        self.dp_rank = int(pc.data_parallel_rank) if pc is not None else 0
        self.tp_size = int(pc.tensor_parallel_size) if pc is not None else 1

        self.ctrl_endpoints: list[str] = [
            f"tcp://{config.remote_hosts[r]}:"
            f"{config.ctrl_port_base + self.dp_rank * self.tp_size + r}"
            for r in range(self.tp_size)
        ]

        self.reqs_to_send: dict[ReqId, MooncakeP2PSendMeta] = {}
        self.reqs_to_load: dict[ReqId, MooncakeP2PLoadMeta] = {}

    def on_new_request(self, request: Request) -> None:
        self._admit_kv_transfer_params(request)

    def _reject_params(
        self, request: Request, params: dict[str, Any], reason: str
    ) -> None:
        logger.warning(
            "TPUMooncakeConnector: ignoring invalid kv_transfer_params for "
            "req_id=%s (%s); prefilling locally",
            request.request_id,
            reason,
        )
        params[_KV_PARAMS_REJECTED] = reason
        params[_KV_PARAMS_PROCESSED] = True

    def _validate_load_params(self, params: dict[str, Any]) -> str | None:
        uuid = _as_int(params.get("uuid"))
        if uuid is None:
            return f"uuid must be an int, got {params.get('uuid')!r}"
        remote_block_ids = _as_int_list(params.get("remote_block_ids"))
        if remote_block_ids is None:
            return "remote_block_ids must be a list[int]"
        endpoints = _as_str_list(params.get("remote_ctrl_endpoints"))
        if endpoints is None:
            return "remote_ctrl_endpoints must be a non-empty list[str]"
        remote_tp_size = _as_int(params.get("remote_tp_size"))
        if remote_tp_size is None or remote_tp_size <= 0:
            return f"remote_tp_size must be a positive int, got {remote_tp_size!r}"
        if remote_tp_size != self.tp_size:
            return (
                f"TP size mismatch: remote_tp_size={remote_tp_size}, "
                f"local_tp_size={self.tp_size}"
            )
        if len(endpoints) != self.tp_size:
            return (
                f"remote_ctrl_endpoints length ({len(endpoints)}) != "
                f"local_tp_size ({self.tp_size})"
            )
        params["uuid"] = uuid
        params["remote_block_ids"] = remote_block_ids
        params["remote_ctrl_endpoints"] = endpoints
        params["remote_tp_size"] = remote_tp_size
        return None

    def _admit_kv_transfer_params(self, request: Request) -> bool:
        params = request.kv_transfer_params
        if self.is_producer or not params:
            return False
        if not isinstance(params, dict):
            logger.warning(
                "TPUMooncakeConnector: ignoring non-dict kv_transfer_params "
                "for req_id=%s",
                request.request_id,
            )
            request.kv_transfer_params = None
            return False
        if params.get(_KV_PARAMS_REJECTED):
            return False
        if params.get(_KV_PARAMS_ADMITTED):
            return True
        reason = self._validate_load_params(params)
        if reason is not None:
            self._reject_params(request, params, reason)
            return False
        params[_KV_PARAMS_ADMITTED] = True
        return True

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        if not self._admit_kv_transfer_params(request):
            return 0, False
        params = request.kv_transfer_params
        assert isinstance(params, dict)
        if params.get(_KV_PARAMS_PROCESSED):
            return 0, False
        assert num_computed_tokens % self.block_size == 0
        assert request.prompt_token_ids is not None

        remote_blocks = len(params["remote_block_ids"])
        rounded_prompt_tokens = min(
            round_down(len(request.prompt_token_ids), self.block_size),
            remote_blocks * self.block_size,
        )
        count = max(rounded_prompt_tokens - num_computed_tokens, 0)
        if count > 0:
            return count, True
        return 0, False

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        if not self._admit_kv_transfer_params(request):
            return
        params = request.kv_transfer_params
        assert isinstance(params, dict)
        if params.get(_KV_PARAMS_PROCESSED):
            return

        remote_endpoints: list[str] = params["remote_ctrl_endpoints"]
        if num_external_tokens > 0:
            assert num_external_tokens % self.block_size == 0
            local_all = list(blocks.get_block_ids()[0])
            remote_all: list[int] = list(params["remote_block_ids"])
            num_ext_blocks = num_external_tokens // self.block_size
            # Skip any locally cached prefix blocks already resident on Decode.
            prefix_total = min(len(local_all), len(remote_all))
            start_block = max(0, prefix_total - num_ext_blocks)
            local_slice = local_all[start_block : start_block + num_ext_blocks]
            remote_slice = remote_all[start_block : start_block + num_ext_blocks]
            if (
                len(local_slice) != num_ext_blocks
                or len(remote_slice) != num_ext_blocks
            ):
                raise ValueError(
                    "Block slice mismatch in TPUMooncakeConnector: "
                    f"num_ext_blocks={num_ext_blocks}, "
                    f"local={len(local_all)}, remote={len(remote_all)}"
                )
            self.reqs_to_load[request.request_id] = MooncakeP2PLoadMeta(
                uuid=params["uuid"],
                local_block_ids=local_slice,
                remote_block_ids=remote_slice,
                remote_ctrl_endpoints=remote_endpoints,
                release_only=False,
                report_completion=True,
            )
        else:
            # Full local prefix hit: release producer blocks without transferring.
            self.reqs_to_load[request.request_id] = MooncakeP2PLoadMeta(
                uuid=params["uuid"],
                local_block_ids=None,
                remote_block_ids=None,
                remote_ctrl_endpoints=remote_endpoints,
                release_only=True,
                report_completion=False,
            )
        params[_KV_PARAMS_PROCESSED] = True

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput | None = None
    ) -> MooncakeP2PMetadata:
        del scheduler_output
        meta = MooncakeP2PMetadata()
        if self.is_producer:
            meta.reqs_to_send = self.reqs_to_send
            self.reqs_to_send = {}
        else:
            meta.reqs_to_load = self.reqs_to_load
            self.reqs_to_load = {}
        return meta

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        if not self.is_producer:
            return False, None
        if request.status not in (
            RequestStatus.FINISHED_LENGTH_CAPPED,
            RequestStatus.FINISHED_STOPPED,
        ):
            return False, None

        all_full = request.num_computed_tokens % self.block_size == 0
        computed_block_ids = list(block_ids) if all_full else list(block_ids[:-1])
        if not computed_block_ids:
            return False, {}

        uuid = _new_uuid()
        expiration_time = time.time() + self.config.wait_pull_timeout_s
        self.reqs_to_send[request.request_id] = MooncakeP2PSendMeta(
            uuid=uuid,
            local_block_ids=computed_block_ids,
            expiration_time=expiration_time,
        )
        kv_transfer_params: dict[str, Any] = {
            "uuid": uuid,
            "remote_block_ids": computed_block_ids,
            "remote_ctrl_endpoints": list(self.ctrl_endpoints),
            "remote_tp_size": self.tp_size,
        }
        return True, kv_transfer_params


@dataclass
class _ProducerSendEntry:
    req_id: ReqId
    uuid: int
    expiration_time: float
    event: Any
    fenced: bool = False


class TPUMooncakeConnectorWorker:
    """Worker half of `TPUMooncakeConnector` owning the rank's TENT engine."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        config: MooncakeP2PConfig,
        *,
        tp_rank: int | None = None,
        tp_size: int | None = None,
        engine_factory: Callable[[], Any] | None = None,
        provider: Any = None,
        mc_tpu_module: Any = None,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        assert vllm_config.kv_transfer_config is not None
        self.is_producer = bool(vllm_config.kv_transfer_config.is_kv_producer)
        self.config = config
        self._engine_factory = engine_factory
        self._provider = provider
        self._mc_tpu_module = mc_tpu_module

        resolved_rank, resolved_size = _resolve_tp_rank_and_size(vllm_config)
        self.tp_rank = resolved_rank if tp_rank is None else tp_rank
        self.tp_size = resolved_size if tp_size is None else tp_size
        pc = vllm_config.parallel_config
        self.dp_rank = int(pc.data_parallel_rank) if pc is not None else 0

        self._pool_tensors: list[torch.Tensor] = []
        self._page_sizes: list[int] = []
        self._shards: list[Any] = []
        self._shard_addrs: list[int] = []
        self._engine: Any = None
        self._registered_with_engine = False
        self.session_id: str = ""
        self.ctrl_endpoint: str = ""

        self._zmq_ctx = zmq.Context()
        self._ctrl_stop = threading.Event()
        self._ctrl_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._sends: dict[int, _ProducerSendEntry] = {}
        self._done_sending: set[ReqId] = set()
        self._done_recving: set[ReqId] = set()
        self._failed_recving: set[ReqId] = set()
        self._executor = ThreadPoolExecutor(
            max_workers=config.transfer_threads,
            thread_name_prefix=f"mc-p2p-r{self.tp_rank}",
        )
        self._closed = False

    def register_kv_caches(
        self, kv_caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]]
    ) -> None:
        num_blocks = int(self.kv_cache_config.num_blocks)
        pools = mc_utils.canonicalize_tpu_kv_pools(kv_caches, num_blocks)
        self._pool_tensors = [p.tensor for p in pools]
        self._page_sizes = [p.page_size_bytes for p in pools]

        mc_utils.configure_tent_tpu_env(
            protocol=self.config.protocol,
            connections=self.config.tent_connections,
            extra_tent_config=self.config.tent_extra_config,
        )
        # Export TPU buffers BEFORE initializing TransferEngine.
        self._shards = mc_utils.export_tpu_kv_pools(
            self._pool_tensors,
            provider=self._provider,
            mc_tpu_module=self._mc_tpu_module,
        )
        self._shard_addrs = [int(s.fake_addr) for s in self._shards]

        try:
            factory = self._engine_factory or _create_default_transfer_engine
            engine = factory()
            init_proto = self.config.protocol
            rc = engine.initialize(
                self.config.host_ip,
                self.config.metadata_server,
                init_proto,
                self.config.device_name,
            )
            if rc != 0:
                raise RuntimeError(
                    "TPUMooncakeConnectorWorker failed to initialize "
                    f"TransferEngine on rank {self.tp_rank} (rc={rc})"
                )
            self._engine = engine
            rpc_port = int(engine.get_rpc_port())
            self.session_id = f"{self.config.host_ip}:{rpc_port}"

            mc_tpu = (
                self._mc_tpu_module
                if self._mc_tpu_module is not None
                else mc_utils._import_mooncake_tpu()
            )
            mc_tpu.register_with_engine(engine, [[s] for s in self._shards])
            self._registered_with_engine = True

            if self.is_producer:
                self._start_producer_control_server()
        except Exception:
            self.shutdown()
            raise

        logger.info(
            "TPUMooncakeConnectorWorker rank %d registered %d pool(s) "
            "(session_id=%s, ctrl_endpoint=%s)",
            self.tp_rank,
            len(self._shards),
            self.session_id,
            self.ctrl_endpoint or "none",
        )

    def _start_producer_control_server(self) -> None:
        bind_port = (
            0
            if self.config.ctrl_port_base == 0
            else self.config.ctrl_port_base + self.dp_rank * self.tp_size + self.tp_rank
        )
        sock = self._zmq_ctx.socket(zmq.ROUTER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(f"tcp://{self.config.ctrl_bind_host}:{bind_port}")
        bound_endpoint = sock.getsockopt_string(zmq.LAST_ENDPOINT)
        if self.config.ctrl_bind_host == "*":
            actual_port = int(bound_endpoint.rsplit(":", 1)[1])
            self.ctrl_endpoint = f"tcp://{self.config.host_ip}:{actual_port}"
        else:
            self.ctrl_endpoint = bound_endpoint

        self._ctrl_thread = threading.Thread(
            target=self._control_loop,
            args=(sock,),
            name=f"mc-p2p-ctrl-r{self.tp_rank}",
            daemon=True,
        )
        self._ctrl_thread.start()

    def _control_loop(self, sock: zmq.Socket) -> None:
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        # uuid -> list of (client_identity, deadline)
        waiters: dict[int, list[tuple[bytes, float]]] = {}
        try:
            while not self._ctrl_stop.is_set():
                events = dict(poller.poll(timeout=5))
                if sock in events:
                    self._handle_control_message(sock, waiters)
                if waiters:
                    self._flush_waiters(sock, waiters)
        finally:
            for uuid, pending in list(waiters.items()):
                for ident, _ in pending:
                    self._send_router_reply(sock, ident, {"status": "shutdown"})
                waiters.pop(uuid, None)
            sock.close(linger=0)

    def _send_router_reply(
        self, sock: zmq.Socket, ident: bytes, payload: dict[str, Any]
    ) -> None:
        try:
            sock.send_multipart([ident, b"", json.dumps(payload).encode("utf-8")])
        except zmq.ZMQError as exc:
            logger.debug("TPUMooncakeConnector control reply failed: %s", exc)

    def _ready_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "session_id": self.session_id,
            "pool_addrs": list(self._shard_addrs),
            "pool_page_sizes": list(self._page_sizes),
        }

    def _handle_control_message(
        self,
        sock: zmq.Socket,
        waiters: dict[int, list[tuple[bytes, float]]],
    ) -> None:
        frames = sock.recv_multipart()
        if len(frames) < 3:
            return
        ident, raw_payload = frames[0], frames[-1]
        try:
            msg = json.loads(raw_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_router_reply(sock, ident, {"status": "bad_request"})
            return

        op = msg.get("op")
        uuid = _as_int(msg.get("uuid"))
        if uuid is None:
            self._send_router_reply(sock, ident, {"status": "bad_request"})
            return

        if op == "ready":
            with self._lock:
                entry = self._sends.get(uuid)
                is_ready = entry is not None and entry.fenced
            if is_ready:
                self._send_router_reply(sock, ident, self._ready_payload())
            else:
                deadline = time.monotonic() + self.config.ready_timeout_s
                waiters.setdefault(uuid, []).append((ident, deadline))
        elif op == "done":
            with self._lock:
                entry = self._sends.pop(uuid, None)
                if entry is not None:
                    self._done_sending.add(entry.req_id)
            for waiter_ident, _ in waiters.pop(uuid, []):
                self._send_router_reply(sock, waiter_ident, {"status": "released"})
            self._send_router_reply(sock, ident, {"status": "ok"})
        else:
            self._send_router_reply(sock, ident, {"status": "unknown_op"})

    def _flush_waiters(
        self,
        sock: zmq.Socket,
        waiters: dict[int, list[tuple[bytes, float]]],
    ) -> None:
        now = time.monotonic()
        with self._lock:
            ready_uuids = [
                uuid
                for uuid in waiters
                if (entry := self._sends.get(uuid)) is not None and entry.fenced
            ]
        for uuid in ready_uuids:
            payload = self._ready_payload()
            for ident, _ in waiters.pop(uuid, []):
                self._send_router_reply(sock, ident, payload)

        for uuid in list(waiters.keys()):
            remaining: list[tuple[bytes, float]] = []
            for ident, deadline in waiters[uuid]:
                if now >= deadline:
                    self._send_router_reply(sock, ident, {"status": "timeout"})
                else:
                    remaining.append((ident, deadline))
            if remaining:
                waiters[uuid] = remaining
            else:
                del waiters[uuid]

    def _control_rpc(
        self, endpoint: str, payload: dict[str, Any], timeout_s: float
    ) -> dict[str, Any]:
        timeout_ms = max(100, int(timeout_s * 1000))
        sock = self._zmq_ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        try:
            sock.connect(endpoint)
            sock.send(json.dumps(payload).encode("utf-8"))
            raw = sock.recv()
            return json.loads(raw.decode("utf-8"))
        finally:
            sock.close(linger=0)

    def get_handshake_metadata(self) -> MooncakeHandshakeMetadata | None:
        if not self.session_id:
            return None
        return MooncakeHandshakeMetadata(
            transfer_rank=self.tp_rank,
            session_id=self.session_id,
            ctrl_endpoint=self.ctrl_endpoint,
            pool_addrs=list(self._shard_addrs),
            pool_page_sizes=list(self._page_sizes),
            tp_size=self.tp_size,
        )

    def _poll_producer_fences_and_expirations(self) -> None:
        """Advance TPU fence events on the worker main thread and expire
        abandoned sends past their TTL."""
        now_wall = time.time()
        with self._lock:
            entries = list(self._sends.values())

        retired_uuids: list[int] = []
        expired_uuids: list[int] = []
        for entry in entries:
            if not entry.fenced and mc_utils.event_is_retired(entry.event):
                retired_uuids.append(entry.uuid)
            if now_wall >= entry.expiration_time:
                expired_uuids.append(entry.uuid)

        if not retired_uuids and not expired_uuids:
            return
        with self._lock:
            for uuid in retired_uuids:
                entry = self._sends.get(uuid)
                if entry is not None:
                    entry.fenced = True
            for uuid in expired_uuids:
                entry = self._sends.pop(uuid, None)
                if entry is not None:
                    logger.warning(
                        "TPUMooncakeConnectorWorker rank %d expired send for "
                        "req_id=%s (uuid=%d)",
                        self.tp_rank,
                        entry.req_id,
                        uuid,
                    )
                    self._done_sending.add(entry.req_id)

    def _run_pull(
        self, req_id: ReqId, meta: MooncakeP2PLoadMeta, report_completion: bool
    ) -> None:
        endpoint = meta.remote_ctrl_endpoints[self.tp_rank]
        if meta.release_only:
            try:
                self._control_rpc(
                    endpoint,
                    {"op": "done", "uuid": meta.uuid},
                    timeout_s=min(5.0, self.config.ready_timeout_s),
                )
            except Exception as exc:
                logger.warning(
                    "TPUMooncakeConnectorWorker rank %d release-only notify "
                    "failed for req_id=%s: %s",
                    self.tp_rank,
                    req_id,
                    exc,
                )
            if report_completion and meta.report_completion:
                with self._lock:
                    self._done_recving.add(req_id)
            return

        success = False
        ready_ok = False
        try:
            resp = self._control_rpc(
                endpoint,
                {"op": "ready", "uuid": meta.uuid},
                timeout_s=self.config.ready_timeout_s + 5.0,
            )
            if resp.get("status") != "ok":
                raise RuntimeError(
                    f"Producer {endpoint} rejected ready for uuid={meta.uuid}: {resp}"
                )
            ready_ok = True
            remote_session_id = str(resp["session_id"])
            remote_addrs = [int(a) for a in resp["pool_addrs"]]
            remote_page_sizes = [int(s) for s in resp["pool_page_sizes"]]
            if remote_page_sizes != self._page_sizes:
                raise ValueError(
                    "Pool page size mismatch between producer and consumer: "
                    f"remote={remote_page_sizes}, local={self._page_sizes}"
                )
            assert meta.local_block_ids is not None
            assert meta.remote_block_ids is not None
            local_ptrs, remote_ptrs, lengths = mc_utils.build_p2p_transfer_plan(
                self._shard_addrs,
                remote_addrs,
                self._page_sizes,
                meta.local_block_ids,
                meta.remote_block_ids,
            )
            if local_ptrs:
                rc = self._engine.batch_transfer_sync_read(
                    remote_session_id,
                    local_ptrs,
                    remote_ptrs,
                    lengths,
                )
                if rc != 0:
                    raise RuntimeError(
                        f"TransferEngine.batch_transfer_sync_read failed (rc={rc}, "
                        f"remote={remote_session_id}, spans={len(local_ptrs)})"
                    )
            success = True
        except Exception as exc:
            logger.warning(
                "TPUMooncakeConnectorWorker rank %d pull failed for req_id=%s "
                "(uuid=%d): %s",
                self.tp_rank,
                req_id,
                meta.uuid,
                exc,
            )
        finally:
            if ready_ok:
                try:
                    self._control_rpc(
                        endpoint,
                        {"op": "done", "uuid": meta.uuid},
                        timeout_s=min(5.0, self.config.ready_timeout_s),
                    )
                except Exception as exc:
                    logger.warning(
                        "TPUMooncakeConnectorWorker rank %d done notify failed "
                        "for req_id=%s (uuid=%d): %s",
                        self.tp_rank,
                        req_id,
                        meta.uuid,
                        exc,
                    )
            if report_completion and meta.report_completion:
                with self._lock:
                    self._done_recving.add(req_id)
                    if not success:
                        self._failed_recving.add(req_id)

    def process_send_load(
        self,
        metadata: MooncakeP2PMetadata,
        *,
        wait_for_completion: bool = False,
        report_completion: bool = True,
    ) -> None:
        if metadata.reqs_to_send:
            assert self._pool_tensors, (
                "register_kv_caches must be called before sending KV blocks"
            )
            mc_utils.enqueue_pool_writes(self._pool_tensors)
            event = mc_utils.record_fence_event()
            already_retired = mc_utils.event_is_retired(event)
            with self._lock:
                for req_id, send_meta in metadata.reqs_to_send.items():
                    self._sends[send_meta.uuid] = _ProducerSendEntry(
                        req_id=req_id,
                        uuid=send_meta.uuid,
                        expiration_time=send_meta.expiration_time,
                        event=event,
                        fenced=already_retired,
                    )

        futures: list[Future[None]] = []
        for req_id, load_meta in metadata.reqs_to_load.items():
            futures.append(
                self._executor.submit(
                    self._run_pull, req_id, load_meta, report_completion
                )
            )
        if wait_for_completion:
            for fut in futures:
                fut.result()

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        del finished_req_ids
        if self.is_producer:
            self._poll_producer_fences_and_expirations()
        with self._lock:
            done_sending = set(self._done_sending)
            done_recving = set(self._done_recving)
            failed_recving = set(self._failed_recving)
            self._done_sending.clear()
            self._done_recving.clear()
            self._failed_recving.clear()
        return KVConnectorTransferResults(
            finished_sending=done_sending,
            finished_recving=done_recving,
            failed_recving=failed_recving,
        )

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._ctrl_stop.set()
        if self._ctrl_thread is not None:
            self._ctrl_thread.join(timeout=2.0)
            self._ctrl_thread = None
        self._executor.shutdown(wait=True)

        if self._engine is not None and self._registered_with_engine:
            try:
                batch_unreg = getattr(self._engine, "batch_unregister_memory", None)
                if callable(batch_unreg):
                    batch_unreg(list(self._shard_addrs))
                else:
                    for addr in self._shard_addrs:
                        self._engine.unregister_memory(addr)
            except Exception as exc:
                logger.warning("Failed to unregister Mooncake engine memory: %s", exc)
            self._registered_with_engine = False

        if self._shards:
            try:
                mc_utils.unregister_tpu_kv_pools(
                    self._shards, mc_tpu_module=self._mc_tpu_module
                )
            except Exception as exc:
                logger.warning("Failed to unregister Mooncake TPU shards: %s", exc)
            self._shards = []
            self._shard_addrs = []

        self._zmq_ctx.term()


class TPUMooncakeConnector(KVConnectorBase_V1, SupportsHMA):
    """Direct P2P disaggregated Prefill/Decode connector backed by Mooncake TENT."""

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        del vllm_config
        return None

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return bool(tpu_envs.VLLM_TPU_BLOCK_MAJOR_KV)

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self.mc_config = parse_mooncake_p2p_config(vllm_config, extra_config)

        self.connector_scheduler: TPUMooncakeConnectorScheduler | None = None
        self.connector_worker: TPUMooncakeConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = TPUMooncakeConnectorScheduler(
                vllm_config, self.mc_config
            )
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = TPUMooncakeConnectorWorker(
                vllm_config, kv_cache_config, self.mc_config
            )

    # -- Scheduler-side hooks ----------------------------------------------
    def on_new_request(self, request: Request) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.on_new_request(request)

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> MooncakeP2PMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        if len(block_ids) != 1:
            raise NotImplementedError(
                "TPUMooncakeConnector currently supports single-group KV cache "
                f"layouts; got {len(block_ids)} KV cache groups"
            )
        return self.connector_scheduler.request_finished(request, block_ids[0])

    # -- Worker-side hooks -------------------------------------------------
    def register_kv_caches(
        self, kv_caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]]
    ) -> None:
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_handshake_metadata()

    def start_load_kv(
        self,
        forward_context: Any,
        wait_for_completion: bool = False,
        report_completion: bool = True,
        **kwargs: Any,
    ) -> None:
        del forward_context, kwargs
        assert self.connector_worker is not None
        meta = self._get_connector_metadata()
        assert isinstance(meta, MooncakeP2PMetadata)
        self.connector_worker.process_send_load(
            meta,
            wait_for_completion=wait_for_completion,
            report_completion=report_completion,
        )

    def wait_for_layer_load(self, layer_name: str) -> None:
        del layer_name

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        del layer_name, kv_layer, attn_metadata, kwargs

    def wait_for_save(self) -> None:
        pass

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        assert self.connector_worker is not None
        return self.connector_worker.get_transfer_results(finished_req_ids)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        results = self.get_transfer_results(finished_req_ids)
        if results.failed_recving:
            raise RuntimeError(
                "TPUMooncakeConnector.get_finished() cannot report failed loads "
                f"for requests {sorted(results.failed_recving)}; call "
                "get_transfer_results() instead"
            )
        return results.finished_sending, results.finished_recving

    def shutdown(self) -> None:
        if self.connector_worker is not None:
            self.connector_worker.shutdown()
