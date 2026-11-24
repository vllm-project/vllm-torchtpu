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
"""MooncakeDistributedStore-backed KV cache offloading spec, manager, and worker.

Scheduler-side (`MooncakeOffloadingManager`) connects a metadata-only
`MooncakeDistributedStore` client (`local_buffer_size=0`,
`global_segment_size=0`) to probe block presence across all worker ranks via
`batch_is_exist`, enabling both single-host and multi-host shared prefix
caching.

Worker-side (`MooncakeStoreOffloadingWorker`) runs on each TPU worker rank,
exports its canonicalized unified block pool (UBP) storages via `mooncake.tpu`
and `tpu_sync.solutions.mooncake.torch.mooncake_provider`, registers their
TENT fake addresses with `MooncakeDistributedStore`, and executes asynchronous
D2H PUTs and H2D GETs on a bounded worker thread pool. Store jobs are gated by
a non-blocking TorchTPU stream fence so Mooncake never reads HBM before the
producing forward pass retires.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable, Collection, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    Locality,
    LookupResult,
    OffloadingEvent,
    OffloadingManager,
    OffloadingSpec,
    OffloadingWorker,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    TransferResult,
    get_offload_block_hash,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.config import OffloadingConfig

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.distributed import utils as dist_utils
from vllm_torchtpu.distributed.kv_transfer import mooncake_tpu_utils as mc_utils
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_MIB = 1024 * 1024
_REMOTE_MEDIUM = "MOONCAKE"


def _parse_size_bytes(raw: Any, default_bytes: int) -> int:
    """Parse an integer byte count or human-readable size string (e.g. '4GB')."""
    if raw is None or raw == "":
        return default_bytes
    if isinstance(raw, int | float):
        val = int(raw)
        if val < 0:
            raise ValueError(f"Size must be non-negative, got {raw!r}")
        return val
    text = str(raw).strip().upper()
    multipliers = (
        ("TIB", 1024**4),
        ("TB", 1024**4),
        ("GIB", 1024**3),
        ("GB", 1024**3),
        ("MIB", 1024**2),
        ("MB", 1024**2),
        ("KIB", 1024),
        ("KB", 1024),
        ("B", 1),
    )
    for suffix, mult in multipliers:
        if text.endswith(suffix):
            num_str = text[: -len(suffix)].strip()
            val = int(float(num_str) * mult)
            if val < 0:
                raise ValueError(f"Size must be non-negative, got {raw!r}")
            return val
    val = int(text)
    if val < 0:
        raise ValueError(f"Size must be non-negative, got {raw!r}")
    return val


@dataclass(frozen=True)
class MooncakeOffloadConfig:
    """Resolved runtime configuration for `TPUMooncakeOffloadingConnector`."""

    master_server_address: str
    metadata_server: str
    protocol: str
    device_name: str
    global_segment_size: int
    local_buffer_size: int
    tent_connections: int
    transfer_threads: int
    namespace: str
    host_ip: str
    tent_extra_config: Mapping[str, Any] | None = None


def parse_mooncake_offload_config(
    extra_config: Mapping[str, Any],
) -> MooncakeOffloadConfig:
    """Resolve Mooncake offloading config from `kv_connector_extra_config` and
    `vllm_torchtpu.envs` defaults."""
    master_addr = str(
        extra_config.get("master_server_address")
        or extra_config.get("mooncake_master_server_address")
        or tpu_envs.TPU_MOONCAKE_MASTER_SERVER_ADDRESS
    ).strip()
    metadata_server = str(
        extra_config.get("metadata_server")
        or extra_config.get("mooncake_metadata_server")
        or tpu_envs.TPU_MOONCAKE_METADATA_SERVER
    ).strip()
    if not master_addr:
        raise ValueError(
            "TPUMooncakeOffloadingConnector requires `master_server_address` "
            "in kv_connector_extra_config or TPU_MOONCAKE_MASTER_SERVER_ADDRESS"
        )
    if not metadata_server:
        raise ValueError(
            "TPUMooncakeOffloadingConnector requires `metadata_server` "
            "in kv_connector_extra_config or TPU_MOONCAKE_METADATA_SERVER"
        )

    protocol = (
        str(extra_config.get("protocol") or tpu_envs.TPU_MOONCAKE_PROTOCOL)
        .strip()
        .lower()
    )
    device_name = str(
        extra_config.get("device_name") or tpu_envs.TPU_MOONCAKE_DEVICE_NAME
    ).strip()

    default_global_bytes = tpu_envs.TPU_MOONCAKE_GLOBAL_SEGMENT_SIZE_MB * _MIB
    default_local_bytes = tpu_envs.TPU_MOONCAKE_LOCAL_BUFFER_SIZE_MB * _MIB
    global_segment_size = _parse_size_bytes(
        extra_config.get("global_segment_size"), default_global_bytes
    )
    local_buffer_size = _parse_size_bytes(
        extra_config.get("local_buffer_size"), default_local_bytes
    )

    tent_connections = int(
        extra_config.get("tent_connections", tpu_envs.TPU_MOONCAKE_TENT_CONNECTIONS)
    )
    transfer_threads = max(
        1,
        int(
            extra_config.get("transfer_threads", tpu_envs.TPU_MOONCAKE_TRANSFER_THREADS)
        ),
    )
    namespace = str(extra_config.get("namespace", "")).strip()
    host_ip = str(extra_config.get("local_hostname") or "").strip()
    if not host_ip:
        host_ip = dist_utils.get_host_ip()
    tent_extra = extra_config.get("tent_config")
    if tent_extra is not None and not isinstance(tent_extra, Mapping):
        raise ValueError("kv_connector_extra_config['tent_config'] must be a mapping")

    return MooncakeOffloadConfig(
        master_server_address=master_addr,
        metadata_server=metadata_server,
        protocol=protocol,
        device_name=device_name,
        global_segment_size=global_segment_size,
        local_buffer_size=local_buffer_size,
        tent_connections=tent_connections,
        transfer_threads=transfer_threads,
        namespace=namespace,
        host_ip=host_ip,
        tent_extra_config=tent_extra,
    )


def build_mooncake_key_prefix(
    vllm_config: VllmConfig,
    *,
    namespace: str = "",
) -> str:
    """Build a deterministic compatibility prefix isolating incompatible KV
    layouts across models, dtypes, block sizes, and parallelism topologies."""
    pc = vllm_config.parallel_config
    mc = vllm_config.model_config
    cc = vllm_config.cache_config
    model_name = getattr(mc, "model", "")
    dtype = str(getattr(mc, "dtype", ""))
    kv_dtype = str(getattr(cc, "cache_dtype", "auto"))
    block_size = int(getattr(cc, "block_size", 0))
    tp = int(getattr(pc, "tensor_parallel_size", 1))
    pp = int(getattr(pc, "pipeline_parallel_size", 1))
    pcp = int(getattr(pc, "prefill_context_parallel_size", 1))
    dcp = int(getattr(pc, "decode_context_parallel_size", 1))
    block_major = int(bool(tpu_envs.VLLM_TPU_BLOCK_MAJOR_KV))

    fingerprint_src = (
        f"v1|model={model_name}|dtype={dtype}|kv={kv_dtype}|bs={block_size}|"
        f"tp={tp}|pp={pp}|pcp={pcp}|dcp={dcp}|bm={block_major}|ns={namespace}"
    )
    digest = hashlib.blake2b(fingerprint_src.encode("utf-8"), digest_size=8).hexdigest()
    return f"vllm_tpu_mc:{digest}"


def format_mooncake_block_key(prefix: str, rank: int, key: OffloadKey) -> str:
    """Format one worker rank's Mooncake Store object key for an `OffloadKey`."""
    group_idx = get_offload_group_idx(key)
    block_hash = get_offload_block_hash(key).hex()
    return f"{prefix}:r{rank}:g{group_idx}:{block_hash}"


def _resolve_worker_rank(vllm_config: VllmConfig) -> int:
    """Resolve this worker's deterministic ordinal within the engine's
    `(pp, pcp, tp)` rank grid."""
    pc = vllm_config.parallel_config
    tp_size = int(getattr(pc, "tensor_parallel_size", 1))
    pcp_size = int(getattr(pc, "prefill_context_parallel_size", 1))
    try:
        from vllm.distributed.parallel_state import (
            get_pp_group,
            get_tensor_model_parallel_rank,
        )

        tp_rank = int(get_tensor_model_parallel_rank())
        pp_rank = int(get_pp_group().rank_in_group)
    except Exception:
        tp_rank = 0
        pp_rank = 0

    pcp_rank = 0
    if pcp_size > 1:
        try:
            from vllm.distributed.parallel_state import get_pcp_group

            pcp_rank = int(get_pcp_group().rank_in_group)
        except Exception:
            pcp_rank = 0

    return (pp_rank * pcp_size + pcp_rank) * tp_size + tp_rank


def _create_default_store() -> Any:
    try:
        from mooncake.store import MooncakeDistributedStore
    except ImportError as exc:
        raise ImportError(
            "TPUMooncakeOffloadingConnector requires `mooncake.store` "
            "(Mooncake built with `-DWITH_STORE=ON -DUSE_TENT=ON -DUSE_TPU=ON`)."
        ) from exc
    return MooncakeDistributedStore()


class MooncakeLoadStoreSpec(LoadStoreSpec):
    """Offload-side spec listing ordered `OffloadKey`s for a load or store job."""

    def __init__(self, keys: list[OffloadKey]):
        self.keys = keys

    def __repr__(self) -> str:
        return f"MooncakeLoadStoreSpec({len(self.keys)} keys)"


class MooncakeOffloadingManager(OffloadingManager):
    """Scheduler-side offloading manager backed by `MooncakeDistributedStore`."""

    def __init__(
        self,
        *,
        config: MooncakeOffloadConfig,
        key_prefix: str,
        num_workers: int,
        enable_kv_cache_events: bool = False,
        store: Any = None,
        store_factory: Callable[[], Any] | None = None,
    ):
        if num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {num_workers}")
        self._config = config
        self._key_prefix = key_prefix
        self._num_workers = num_workers
        self._enable_kv_cache_events = enable_kv_cache_events
        self._inflight_stores: set[OffloadKey] = set()
        self._pending_events: list[OffloadingEvent] = []
        self._closed = False

        if store is not None:
            self._store = store
        else:
            mc_utils.configure_tent_tpu_env(
                protocol=config.protocol,
                connections=config.tent_connections,
                extra_tent_config=config.tent_extra_config,
            )
            factory = store_factory or _create_default_store
            self._store = factory()
            setup_proto = mc_utils.store_protocol_for_tent(config.protocol)
            rc = self._store.setup(
                config.host_ip,
                config.metadata_server,
                0,
                0,
                setup_proto,
                config.device_name,
                config.master_server_address,
            )
            if rc != 0:
                raise RuntimeError(
                    "MooncakeOffloadingManager failed to initialize "
                    f"MooncakeDistributedStore (rc={rc}, "
                    f"master={config.master_server_address}, "
                    f"metadata={config.metadata_server})"
                )

    def _all_rank_keys(self, key: OffloadKey) -> list[str]:
        return [
            format_mooncake_block_key(self._key_prefix, rank, key)
            for rank in range(self._num_workers)
        ]

    def _batch_exists(self, store_keys: list[str]) -> list[bool]:
        if not store_keys:
            return []
        try:
            results = self._store.batch_is_exist(store_keys)
        except Exception as exc:
            logger.warning(
                "MooncakeOffloadingManager batch_is_exist failed (%s); "
                "treating %d key(s) as misses",
                exc,
                len(store_keys),
            )
            return [False] * len(store_keys)
        if len(results) != len(store_keys):
            logger.warning(
                "MooncakeOffloadingManager batch_is_exist length mismatch: "
                "expected %d, got %d",
                len(store_keys),
                len(results),
            )
            return [False] * len(store_keys)
        return [int(val) == 1 for val in results]

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        del req_context
        return RequestOffloadingContext()

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        del req_context
        if key in self._inflight_stores:
            return LookupResult.HIT_PENDING
        exists = self._batch_exists(self._all_rank_keys(key))
        if exists and all(exists):
            return LookupResult.HIT
        return LookupResult.MISS

    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        del req_context
        return MooncakeLoadStoreSpec(list(keys))

    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        del req_context
        candidates = [k for k in keys if k not in self._inflight_stores]
        if not candidates:
            return None

        # Check rank 0 to skip blocks already persisted by earlier requests or
        # peer instances sharing the same Mooncake master.
        rank0_keys = [
            format_mooncake_block_key(self._key_prefix, 0, k) for k in candidates
        ]
        present = self._batch_exists(rank0_keys)
        to_store = [k for k, exists in zip(candidates, present) if not exists]
        if not to_store:
            return None

        self._inflight_stores.update(to_store)
        return PrepareStoreOutput(
            keys_to_store=to_store,
            store_spec=MooncakeLoadStoreSpec(to_store),
            evicted_keys=[],
        )

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        del req_context
        key_list = list(keys)
        for key in key_list:
            self._inflight_stores.discard(key)
        if success and self._enable_kv_cache_events and key_list:
            self._pending_events.append(
                OffloadingEvent(
                    keys=key_list,
                    medium=_REMOTE_MEDIUM,
                    removed=False,
                    locality=Locality.REMOTE,
                )
            )

    def take_events(self) -> Iterable[OffloadingEvent]:
        events = self._pending_events
        self._pending_events = []
        return events

    def reset_cache(self) -> None:
        self._inflight_stores.clear()
        self._pending_events.clear()
        remove_all = getattr(self._store, "remove_all", None)
        if callable(remove_all):
            remove_all()

    def has_pending_work(self) -> bool:
        return bool(self._inflight_stores)

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._inflight_stores.clear()
        close_fn = getattr(self._store, "close", None)
        if callable(close_fn):
            close_fn()


@dataclass
class _PendingFenceStore:
    job_id: int
    keys: list[OffloadKey]
    block_ids: list[int]
    event: Any


class MooncakeStoreOffloadingWorker(OffloadingWorker):
    """Per-rank worker executing async TPU <-> Mooncake Store transfers.

    Store jobs record a non-blocking TorchTPU stream fence before launching
    Mooncake PUTs on a background thread pool so HBM writes from the preceding
    forward pass always retire before Mooncake reads the device buffers.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        *,
        config: MooncakeOffloadConfig,
        key_prefix: str,
        rank: int,
        blocks_per_chunk: int = 1,
        store: Any = None,
        store_factory: Callable[[], Any] | None = None,
        provider: Any = None,
        mc_tpu_module: Any = None,
    ):
        if blocks_per_chunk <= 0:
            raise ValueError(
                f"blocks_per_chunk must be positive, got {blocks_per_chunk}"
            )
        if not kv_caches.tensors:
            raise ValueError("CanonicalKVCaches must contain at least one pool tensor")

        self._config = config
        self._key_prefix = key_prefix
        self._rank = rank
        self._blocks_per_chunk = blocks_per_chunk
        self._mc_tpu_module = mc_tpu_module
        self._pool_tensors: list[torch.Tensor] = [
            entry.tensor for entry in kv_caches.tensors
        ]
        self._page_sizes: list[int] = [
            int(entry.page_size_bytes) for entry in kv_caches.tensors
        ]
        self._bytes_per_block = sum(self._page_sizes)

        # Configure TENT and export TPU KV pools BEFORE constructing the store.
        mc_utils.configure_tent_tpu_env(
            protocol=config.protocol,
            connections=config.tent_connections,
            extra_tent_config=config.tent_extra_config,
        )
        self._shards = mc_utils.export_tpu_kv_pools(
            self._pool_tensors,
            provider=provider,
            mc_tpu_module=mc_tpu_module,
        )
        self._shard_addrs: list[int] = [int(s.fake_addr) for s in self._shards]
        self._registered_addrs: list[int] = []
        self._store: Any = None
        self._closed = False

        try:
            if store is not None:
                active_store: Any = store
            else:
                factory = store_factory or _create_default_store
                active_store = factory()
                setup_proto = mc_utils.store_protocol_for_tent(config.protocol)
                rc = active_store.setup(
                    config.host_ip,
                    config.metadata_server,
                    config.global_segment_size,
                    config.local_buffer_size,
                    setup_proto,
                    config.device_name,
                    config.master_server_address,
                )
                if rc != 0:
                    raise RuntimeError(
                        "MooncakeStoreOffloadingWorker failed to setup "
                        f"MooncakeDistributedStore on rank {rank} (rc={rc})"
                    )
            self._store = active_store

            for shard in self._shards:
                rc = active_store.register_buffer(int(shard.fake_addr), int(shard.size))
                if rc != 0:
                    raise RuntimeError(
                        "MooncakeStoreOffloadingWorker failed to register "
                        f"TPU shard {int(shard.fake_addr):#x} (rc={rc})"
                    )
                self._registered_addrs.append(int(shard.fake_addr))
        except Exception:
            self._cleanup_resources()
            raise

        self._executor = ThreadPoolExecutor(
            max_workers=config.transfer_threads,
            thread_name_prefix=f"mc-offload-r{rank}",
        )
        self._lock = threading.Lock()
        self._pending_fences: dict[int, _PendingFenceStore] = {}
        self._inflight: dict[int, Future[TransferResult]] = {}
        self._completed: list[TransferResult] = []
        self._prerecorded_fences: dict[int, Any] = {}

    def record_step_store_fence(self, job_ids: Collection[int]) -> None:
        """Record one shared TPU stream event for store jobs created in the
        just-finished forward step so fencing starts immediately."""
        if not job_ids:
            return
        mc_utils.enqueue_pool_writes(self._pool_tensors)
        event = mc_utils.record_fence_event()
        with self._lock:
            for job_id in job_ids:
                self._prerecorded_fences[job_id] = event

    def _build_chunk_buffers(
        self,
        keys: list[OffloadKey],
        block_ids: list[int],
    ) -> tuple[list[str], list[list[int]], list[list[int]], int]:
        expected_blocks = len(keys) * self._blocks_per_chunk
        if len(block_ids) != expected_blocks:
            raise ValueError(
                f"Block count mismatch: got {len(block_ids)} block_ids for "
                f"{len(keys)} keys with blocks_per_chunk={self._blocks_per_chunk}"
            )
        store_keys: list[str] = []
        all_ptrs: list[list[int]] = []
        all_sizes: list[list[int]] = []
        for idx, key in enumerate(keys):
            store_keys.append(
                format_mooncake_block_key(self._key_prefix, self._rank, key)
            )
            chunk_blocks = block_ids[
                idx * self._blocks_per_chunk : (idx + 1) * self._blocks_per_chunk
            ]
            chunk_ptrs, chunk_sizes = mc_utils.build_store_block_buffers(
                self._shard_addrs, self._page_sizes, chunk_blocks
            )
            flat_ptrs = [ptr for per_block in chunk_ptrs for ptr in per_block]
            flat_sizes = [sz for per_block in chunk_sizes for sz in per_block]
            all_ptrs.append(flat_ptrs)
            all_sizes.append(flat_sizes)

        total_bytes = len(block_ids) * self._bytes_per_block
        return store_keys, all_ptrs, all_sizes, total_bytes

    def _run_store(
        self, job_id: int, keys: list[OffloadKey], block_ids: list[int]
    ) -> TransferResult:
        t0 = time.monotonic()
        try:
            store_keys, all_ptrs, all_sizes, total_bytes = self._build_chunk_buffers(
                keys, block_ids
            )
            if not store_keys:
                return TransferResult(
                    job_id=job_id,
                    success=True,
                    transfer_size=0,
                    transfer_time=0.0,
                )
            rcs = self._store.batch_put_from_multi_buffers(
                store_keys, all_ptrs, all_sizes
            )
            ok = len(rcs) == len(store_keys) and all(int(rc) == 0 for rc in rcs)
            if not ok:
                logger.warning(
                    "MooncakeStoreOffloadingWorker rank %d store job %d failed: rcs=%s",
                    self._rank,
                    job_id,
                    rcs,
                )
            return TransferResult(
                job_id=job_id,
                success=ok,
                transfer_size=total_bytes if ok else 0,
                transfer_time=time.monotonic() - t0,
            )
        except Exception as exc:
            logger.warning(
                "MooncakeStoreOffloadingWorker rank %d store job %d raised %s",
                self._rank,
                job_id,
                exc,
            )
            return TransferResult(
                job_id=job_id,
                success=False,
                transfer_size=0,
                transfer_time=time.monotonic() - t0,
            )

    def _run_load(
        self, job_id: int, keys: list[OffloadKey], block_ids: list[int]
    ) -> TransferResult:
        t0 = time.monotonic()
        try:
            store_keys, all_ptrs, all_sizes, total_bytes = self._build_chunk_buffers(
                keys, block_ids
            )
            if not store_keys:
                return TransferResult(
                    job_id=job_id,
                    success=True,
                    transfer_size=0,
                    transfer_time=0.0,
                )
            rcs = self._store.batch_get_into_multi_buffers(
                store_keys, all_ptrs, all_sizes
            )
            ok = len(rcs) == len(store_keys) and all(int(rc) >= 0 for rc in rcs)
            if not ok:
                logger.warning(
                    "MooncakeStoreOffloadingWorker rank %d load job %d failed: rcs=%s",
                    self._rank,
                    job_id,
                    rcs,
                )
            return TransferResult(
                job_id=job_id,
                success=ok,
                transfer_size=total_bytes if ok else 0,
                transfer_time=time.monotonic() - t0,
            )
        except Exception as exc:
            logger.warning(
                "MooncakeStoreOffloadingWorker rank %d load job %d raised %s",
                self._rank,
                job_id,
                exc,
            )
            return TransferResult(
                job_id=job_id,
                success=False,
                transfer_size=0,
                transfer_time=time.monotonic() - t0,
            )

    def _launch_fenced_store(self, entry: _PendingFenceStore) -> None:
        fut = self._executor.submit(
            self._run_store, entry.job_id, entry.keys, entry.block_ids
        )
        with self._lock:
            self._inflight[entry.job_id] = fut

    def _poll_fences(self) -> None:
        with self._lock:
            pending = list(self._pending_fences.values())
        retired: list[_PendingFenceStore] = []
        for entry in pending:
            if mc_utils.event_is_retired(entry.event):
                retired.append(entry)
        if not retired:
            return
        with self._lock:
            to_launch: list[_PendingFenceStore] = []
            for entry in retired:
                if self._pending_fences.pop(entry.job_id, None) is not None:
                    to_launch.append(entry)
        for entry in to_launch:
            self._launch_fenced_store(entry)

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        assert isinstance(dst_spec, MooncakeLoadStoreSpec)
        block_ids = [int(b) for b in src_spec.block_ids.tolist()]
        with self._lock:
            event = self._prerecorded_fences.pop(job_id, None)
        if event is None:
            mc_utils.enqueue_pool_writes(self._pool_tensors)
            event = mc_utils.record_fence_event()

        entry = _PendingFenceStore(
            job_id=job_id,
            keys=list(dst_spec.keys),
            block_ids=block_ids,
            event=event,
        )
        if mc_utils.event_is_retired(event):
            self._launch_fenced_store(entry)
        else:
            with self._lock:
                self._pending_fences[job_id] = entry
        return True

    def submit_load(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
        expedited: bool = False,
    ) -> bool:
        del expedited
        assert isinstance(src_spec, MooncakeLoadStoreSpec)
        block_ids = [int(b) for b in dst_spec.block_ids.tolist()]
        fut = self._executor.submit(
            self._run_load, job_id, list(src_spec.keys), block_ids
        )
        with self._lock:
            self._inflight[job_id] = fut
        return True

    def get_finished(self) -> list[TransferResult]:
        self._poll_fences()
        finished: list[TransferResult] = []
        with self._lock:
            if self._completed:
                finished.extend(self._completed)
                self._completed.clear()
            done_ids = [jid for jid, fut in self._inflight.items() if fut.done()]
            done_futs = [self._inflight.pop(jid) for jid in done_ids]
        for fut in done_futs:
            finished.append(fut.result())
        return finished

    def wait(self, job_ids: set[int]) -> None:
        """Drain fence-pending and in-flight jobs before vLLM reuses their HBM
        blocks (called from `handle_preemptions`)."""
        if not job_ids:
            return
        # Any job still waiting for its TPU fence must finish its forward write
        # and complete its Mooncake PUT before returning so HBM isn't torn.
        to_fence: list[_PendingFenceStore] = []
        with self._lock:
            for job_id in job_ids:
                entry = self._pending_fences.pop(job_id, None)
                if entry is not None:
                    to_fence.append(entry)
        for entry in to_fence:
            mc_utils.synchronize_fence_event(entry.event)
            self._launch_fenced_store(entry)

        futs_to_wait: list[tuple[int, Future[TransferResult]]] = []
        with self._lock:
            for job_id in job_ids:
                fut = self._inflight.pop(job_id, None)
                if fut is not None:
                    futs_to_wait.append((job_id, fut))
        for _, fut in futs_to_wait:
            result = fut.result()
            with self._lock:
                self._completed.append(result)

    def _cleanup_resources(self) -> None:
        if self._store is not None:
            for addr in self._registered_addrs:
                try:
                    self._store.unregister_buffer(addr)
                except Exception as exc:
                    logger.warning(
                        "Failed to unregister Mooncake store buffer %#x: %s",
                        addr,
                        exc,
                    )
            self._registered_addrs.clear()
            close_fn = getattr(self._store, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception as exc:
                    logger.warning("Failed to close MooncakeDistributedStore: %s", exc)
            self._store = None

        if self._shards:
            try:
                mc_utils.unregister_tpu_kv_pools(
                    self._shards, mc_tpu_module=self._mc_tpu_module
                )
            except Exception as exc:
                logger.warning("Failed to unregister Mooncake TPU shards: %s", exc)
            self._shards = []

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._lock:
            self._pending_fences.clear()
            self._prerecorded_fences.clear()
            inflight = list(self._inflight.values())
            self._inflight.clear()
        for fut in inflight:
            try:
                fut.result(timeout=tpu_envs.VLLM_TPU_OFFLOAD_WAIT_TIMEOUT_S)
            except Exception as exc:
                logger.warning("Mooncake transfer failed during shutdown: %s", exc)
        self._executor.shutdown(wait=True)
        self._cleanup_resources()


class TPUMooncakeStoreOffloadingSpec(OffloadingSpec):
    """OffloadingSpec wiring `MooncakeOffloadingManager` and per-rank
    `MooncakeStoreOffloadingWorker` instances."""

    def __init__(
        self,
        config: OffloadingConfig,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        *,
        manager_store: Any = None,
        store_factory: Callable[[], Any] | None = None,
        provider: Any = None,
        mc_tpu_module: Any = None,
    ):
        super().__init__(config)
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.mc_config = parse_mooncake_offload_config(self.extra_config)
        self.key_prefix = build_mooncake_key_prefix(
            vllm_config, namespace=self.mc_config.namespace
        )
        self.num_workers = int(config.parallel.world_size)
        self._manager_store = manager_store
        self._store_factory = store_factory
        self._provider = provider
        self._mc_tpu_module = mc_tpu_module

    def get_manager(self) -> OffloadingManager:
        return MooncakeOffloadingManager(
            config=self.mc_config,
            key_prefix=self.key_prefix,
            num_workers=self.num_workers,
            enable_kv_cache_events=self.kv_events_config.enable_kv_cache_events,
            store=self._manager_store,
            store_factory=self._store_factory,
        )

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        rank = _resolve_worker_rank(self.vllm_config)
        return MooncakeStoreOffloadingWorker(
            kv_caches,
            config=self.mc_config,
            key_prefix=self.key_prefix,
            rank=rank,
            blocks_per_chunk=self.blocks_per_chunk,
            store_factory=self._store_factory,
            provider=self._provider,
            mc_tpu_module=self._mc_tpu_module,
        )
