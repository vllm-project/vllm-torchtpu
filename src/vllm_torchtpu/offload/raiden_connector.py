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
"""Connector wiring for the store-backed Raiden KV offload path.

`TPURaidenOffloadingConnector` subclasses vLLM's `OffloadingConnector` and
orchestrates controller-driven KV offloading via Raiden's C++ `KVCacheStore`.
Enable it via:

    --kv-transfer-config '{
        "kv_connector": "TPURaidenOffloadingConnector",
        "kv_connector_module_path": "vllm_torchtpu.offload.raiden_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"cpu_bytes_to_use": ...}
    }'

Core mechanisms:

- Save fence: Guarantees background store jobs read HBM only after the forward
  pass writing those KV blocks completes, without inducing device idle bubbles.
  Store jobs ship as `fence_job_ids`. Each rank enqueues KV pool writes
  asynchronously, records a TPU event covering them, and polls for device
  retirement across engine steps before acknowledging via `fenced_jobs`.
  The scheduler launches `save()` only after all ranks have acknowledged.

- Completion echoes: Terminal jobs discovered by the scheduler manager ship
  down to workers, which echo them back as `completed_jobs` / `finished_recving`
  to unblock waiting requests and maintain reference counting. Failed loads
  are also reported by request as `failed_recving`, so the scheduler
  recomputes or fails them, per `kv_load_failure_policy`. Block ids would
  be ambiguous here: each KV cache group numbers its blocks independently,
  and vLLM rejects block-level failures on multi-group layouts.

- Memory reuse (jobs_to_flush): Handled entirely scheduler-side. Fence-pending
  stores are cancelled before launching, while active transfers are drained
  synchronously before shipping metadata, preventing torn reads upon block reuse.
"""

from __future__ import annotations

import dataclasses
import os
import time
from collections import Counter, OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
    config as offloading_config_module,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    OffloadKey,
    make_offload_key,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.offload import contiguous_alloc
from vllm_torchtpu.offload.raiden_store import (
    AdmissionOp,
    RaidenLoadStoreSpec,
    RaidenOffloadingManager,
    TPURaidenStoreOffloadingSpec,
    draft_kv_layer_names,
)

if TYPE_CHECKING:
    from vllm.distributed.kv_events import KVCacheEvent

logger = init_logger(__name__)


def _enqueue_pool_writes(pool_tensors: list[torch.Tensor]) -> None:
    """Asynchronously dispatches graph execution for KV pool buffers without blocking.

    Triggers compilation and submission of any deferred operations targeting the
    pool buffers so that subsequent stream events order behind them. Performs a
    non-blocking device synchronization (`wait=False`), which dispatches no new
    work when the writes are already enqueued.
    """
    from torch_tpu._internal import sync

    sync.synchronize(pool_tensors, wait=False)


def _record_fence_event() -> torch.Event:
    """Records a TPU stream event to track completion of enqueued device operations.

    The event orders after work already submitted to the stream, so callers
    enqueue the writes they need fenced (via `_enqueue_pool_writes`) before
    recording.
    """
    event = torch.tpu.Event()
    event.record()
    return event


def _event_is_retired(event: torch.Event) -> bool:
    """Non-blockingly verifies fence event completion and surfaces deferred errors.

    `event.query()` polls device status without blocking the host thread. Once
    execution terminates, calling `event.synchronize()` returns immediately and
    raises any deferred asynchronous runtime errors before acknowledging the fence.
    """
    if not event.query():
        return False
    event.synchronize()
    return True


@dataclass
class RaidenOffloadingConnectorMetadata(OffloadingConnectorMetadata):
    """Scheduler -> worker metadata.

    Transfers are controller-driven (load_jobs and store_jobs are empty);
    extra fields carry fence requests and completion echoes.
    """

    # Store jobs awaiting device fence from all ranks before launching save.
    fence_job_ids: set[int] = field(default_factory=set)
    # Completed store jobs to echo back to the scheduler state machine.
    finished_store_job_ids: list[int] = field(default_factory=list)
    # Completed load jobs (job_id -> req_id) to echo back and unblock requests.
    finished_load_jobs: dict[int, str] = field(default_factory=dict)
    # Requests whose load failed, a subset of finished_load_jobs' values,
    # reported as failed_recving; the scheduler recomputes or fails them per
    # kv_load_failure_policy.
    failed_load_req_ids: set[str] = field(default_factory=set)


@dataclass
class RaidenOffloadingWorkerMetadata(OffloadingWorkerMetadata):
    """Worker -> scheduler metadata: contains completed-job echoes and
    per-rank fence acks."""

    fenced_jobs: dict[int, int] = field(default_factory=dict)

    def aggregate(self, other):
        assert isinstance(other, RaidenOffloadingWorkerMetadata)
        merged_completed = dict(self.completed_jobs)
        for job_id, v in other.completed_jobs.items():
            merged_completed[job_id] = merged_completed.get(job_id, 0) + v
        merged_fenced = dict(self.fenced_jobs)
        for job_id, v in other.fenced_jobs.items():
            merged_fenced[job_id] = merged_fenced.get(job_id, 0) + v
        # completed_jobs and transfer_stats are inherited from vLLM base dataclass.
        return RaidenOffloadingWorkerMetadata(
            # pyrefly: ignore[unexpected-keyword]
            completed_jobs=merged_completed,
            # pyrefly: ignore[unexpected-keyword]
            transfer_stats=self.transfer_stats.aggregate(other.transfer_stats),
            fenced_jobs=merged_fenced,
        )


@dataclass
class _ParkedStore:
    """A store job parked until every worker rank acknowledges the device fence."""

    keys: list[OffloadKey]
    device_block_ids: list[int]
    acks: int = 0


@dataclass(frozen=True)
class WriteBackPolicy:
    """``kv_connector_extra_config["store_policy"]`` in write-back mode.

    The default policy writes every computed chunk through to the host, so
    the host tier mostly mirrors HBM. Write-back stores a finished request's
    chunks only when they near eviction: once one of their blocks is within
    ``window_blocks`` of the head of the free queue (the next blocks to be
    reused), at most ``max_requests_per_step`` requests per step.
    """

    window_blocks: int = 1024
    max_requests_per_step: int = 8

    @classmethod
    def from_extra_config(cls, extra_config: Mapping) -> WriteBackPolicy | None:
        """``None`` for the default write-through policy."""
        policy = extra_config.get("store_policy")
        if policy is None or policy == "write_through":
            return None
        if policy == "write_back":
            policy = {"mode": "write_back"}
        if not isinstance(policy, Mapping) or policy.get("mode") != "write_back":
            raise ValueError(
                f"store_policy must be 'write_through', 'write_back' or "
                f"{{'mode': 'write_back', ...}}, got {policy!r}"
            )
        options = {k: v for k, v in policy.items() if k != "mode"}
        unknown = set(options) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown store_policy options: {sorted(unknown)}")
        written = cls(**options)
        if written.window_blocks <= 0 or written.max_requests_per_step <= 0:
            raise ValueError(f"store_policy options must be positive: {policy!r}")
        return written


def _allocated_block_ids(scheduler_output: SchedulerOutput) -> set[int]:
    """Blocks this step allocates, by the rule of `_update_req_states`
    (which computes the same set inside the base build_connector_meta)."""
    allocated: set[int] = set()
    for _, new_block_id_groups, _ in yield_req_data(scheduler_output):
        for group in new_block_id_groups or ():
            allocated.update(bid for bid in group if bid != 0)
    for copy in scheduler_output.kv_cache_block_copies or ():
        allocated.add(copy.dst_block_id)
    return allocated


def _prefix_hit_block_ids(scheduler_output: SchedulerOutput, config) -> set[int]:
    """Blocks new requests this step reuse as prefix-cache hits (stats only):
    the leading blocks covering each new request's computed tokens."""
    hits: set[int] = set()
    tokens_per_block = {
        g.group_idx: g.tokens_per_block for g in config.kv_group_configs
    }
    for req in scheduler_output.scheduled_new_reqs:
        for group_idx, group in enumerate(req.block_ids or ()):
            n = req.num_computed_tokens // tokens_per_block.get(group_idx, 1 << 62)
            hits.update(bid for bid in group[:n] if bid != 0)
    return hits


class TPURaidenOffloadingScheduler(OffloadingConnectorScheduler):
    """Scheduler for the store-backed offload path.

    Strips per-rank transfer jobs and orchestrates the store engine directly
    via RaidenOffloadingManager.
    """

    # Write-through unless the connector config asks for write-back
    # (class defaults also cover instances built without __init__).
    _write_back: WriteBackPolicy | None = None
    _block_pool = None
    _contiguous_alloc_scan_factor = 3
    # TPU_RAIDEN_LRU_REFRESH == 2: refresh a finished request's resident chain.
    _lru_refresh_on_finish = False
    _offload_stats = False
    # TPU_RAIDEN_FLUSH_REAL_REUSE / TPU_RAIDEN_LOSS_TRACE; `_fresh_block_ids`
    # collects the blocks the pool hands out between two connector metas.
    _flush_real_reuse = False
    _loss_trace = False
    _fresh_block_ids: set[int] | None = None

    def __init__(
        self,
        spec: TPURaidenStoreOffloadingSpec,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(spec, vllm_config, kv_cache_config)
        assert isinstance(self.manager, RaidenOffloadingManager)
        self._raiden_manager: RaidenOffloadingManager = self.manager
        self._fence_pending: dict[int, _ParkedStore] = {}
        extra_config = getattr(spec, "extra_config", None) or {}
        self._write_back = WriteBackPolicy.from_extra_config(extra_config)
        # How deep into the oldest free blocks an allocation of n blocks may
        # look for consecutive ids (`contiguous_alloc`); <= 1 turns it off.
        self._contiguous_alloc_scan_factor = int(
            extra_config.get("contiguous_alloc_scan_factor", 3)
        )
        # Write-back state: finished requests whose chunks are not stored yet
        # (their finish is deferred), the blocks each one would store, and
        # each such block's owners (shared prefix blocks have several).
        self._block_pool = None
        self._wb_blocks: OrderedDict[str, set[int]] = OrderedDict()
        self._wb_owners: dict[int, set[str]] = {}
        self.wb_counters: Counter[str] = Counter()
        lru_refresh = tpu_envs.TPU_RAIDEN_LRU_REFRESH
        self._lru_refresh_on_finish = lru_refresh >= 2
        self._offload_stats = bool(tpu_envs.TPU_RAIDEN_OFFLOAD_STATS)
        self._stats_last_log: float | None = None
        if lru_refresh or self._offload_stats:
            logger.info(
                "TPURaidenOffloadingConnector: TPU_RAIDEN_LRU_REFRESH=%d "
                "(refresh resident keys on store: %s, on request finish: %s), "
                "TPU_RAIDEN_OFFLOAD_STATS=%s",
                lru_refresh,
                lru_refresh >= 1,
                lru_refresh >= 2,
                self._offload_stats,
            )
        self._flush_real_reuse = bool(tpu_envs.TPU_RAIDEN_FLUSH_REAL_REUSE)
        self._loss_trace = bool(tpu_envs.TPU_RAIDEN_LOSS_TRACE)
        self._traced_req_ids: set[str] = set()
        if self._loss_trace:
            self._raiden_manager.enable_loss_trace()
        if self._flush_real_reuse or self._loss_trace:
            logger.info(
                "TPURaidenOffloadingConnector: TPU_RAIDEN_FLUSH_REAL_REUSE=%s "
                "(flush stores only for blocks the pool hands out fresh, not "
                "prefix hits), TPU_RAIDEN_LOSS_TRACE=%s",
                self._flush_real_reuse,
                self._loss_trace,
            )

    def attach_block_pool(self, block_pool) -> None:
        """`EvictionAwareConnector`: the scheduler's block pool, whose free
        queue tells write-back which finished requests near eviction. Its
        fresh blocks come out as consecutive runs (`contiguous_alloc`)."""
        self._block_pool = block_pool
        # No deeper than the write-back window looks for requests to flush, so
        # a held request's blocks are reused no earlier than write-back can see
        # them coming (one it has not flushed yet is dropped, as before).
        contiguous_alloc.install(
            block_pool,
            scan_factor=self._contiguous_alloc_scan_factor,
            max_scan=(
                self._write_back.window_blocks
                if self._write_back
                else WriteBackPolicy.window_blocks
            ),
        )
        if self._flush_real_reuse and not getattr(
            block_pool, "_tpu_raiden_fresh_watch", False
        ):
            fresh: set[int] = set()
            pool_get_new_blocks = block_pool.get_new_blocks

            def get_new_blocks(num_blocks: int) -> list:
                blocks = pool_get_new_blocks(num_blocks)
                fresh.update(b.block_id for b in blocks)
                return blocks

            block_pool.get_new_blocks = get_new_blocks
            block_pool._tpu_raiden_fresh_watch = True
            self._fresh_block_ids = fresh

    def _update_req_states(self, scheduler_output: SchedulerOutput) -> None:
        """TPU_RAIDEN_FLUSH_REAL_REUSE: the blocks whose pending stores must be
        flushed this step are the blocks the pool handed out fresh (their
        contents are about to be overwritten), not every block a scheduled
        request lists: a new request's prefix-hit blocks keep their contents,
        and flushing on them cancels a parked store of the very chain that is
        being reused, or drains a launched one synchronously."""
        super()._update_req_states(scheduler_output)
        fresh = self._fresh_block_ids
        if fresh is None:
            return
        allocated = self._current_batch_allocated_block_ids
        if self._offload_stats and self._block_id_to_pending_jobs:
            pending = self._block_id_to_pending_jobs
            real = {j for b in fresh if b in pending for j in pending[b]}
            hit_only = {
                j for b in allocated - fresh if b in pending for j in pending[b]
            } - real
            for job_id in hit_only:
                parked = self._fence_pending.get(job_id)
                if parked is not None:
                    self.wb_counters["hitflush_skipped_parked"] += 1
                    self.wb_counters["hitflush_skipped_parked_keys"] += len(
                        parked.keys
                    )
                elif self._raiden_manager.is_job_launched(job_id):
                    self.wb_counters["hitflush_skipped_launched"] += 1
        # Every fresh block, including any the scheduler output does not list.
        self._current_batch_allocated_block_ids = set(fresh)
        fresh.clear()

    def _wb_unregister(self, req_id: str) -> None:
        for bid in self._wb_blocks.pop(req_id, ()):
            owners = self._wb_owners.get(bid)
            if owners is not None:
                owners.discard(req_id)
                if not owners:
                    del self._wb_owners[bid]

    def _wb_drop(self, req_id: str) -> None:
        """Finish a held request without storing it (a block was reused)."""
        self._wb_unregister(req_id)
        req_status = self._req_status.get(req_id)
        if req_status is None:
            return
        self.wb_counters["dropped_requests"] += 1
        req_status.finished_signaled = True
        self.manager.on_request_finished(req_status.req_context)
        if not req_status.transfer_jobs:
            del self._req_status[req_id]

    def _wb_register(self, req_id: str) -> bool:
        """Hold a finished request for write-back; False if it has nothing
        left to store."""
        req_status = self._req_status.get(req_id)
        if req_status is None:
            return False
        blocks_per_chunk = self.config.blocks_per_chunk
        blocks: set[int] = set()
        for group_state in req_status.group_states:
            # Only blocks of keyed (full) chunks can be stored; the partial
            # tail block has no hash, so the pool reuses it first.
            start = group_state.next_stored_chunk_idx * blocks_per_chunk
            end = len(group_state.offload_keys) * blocks_per_chunk
            blocks.update(b for b in group_state.block_ids[start:end] if b != 0)
        if not blocks:
            return False
        self._wb_blocks[req_id] = blocks
        for bid in blocks:
            self._wb_owners.setdefault(bid, set()).add(req_id)
        return True

    def _wb_near_eviction(self) -> list[str]:
        """Held requests with a block among the next ``window_blocks`` the
        free queue hands out, oldest first, at most ``max_requests_per_step``.
        Without a block pool, every held request (store at finish)."""
        policy = self._write_back
        assert policy is not None
        if self._block_pool is None:
            return list(self._wb_blocks)
        queue = self._block_pool.free_block_queue
        picked: dict[str, None] = {}
        block = queue.fake_free_list_head.next_free_block
        for _ in range(policy.window_blocks):
            if block is None or block is queue.fake_free_list_tail:
                break
            for req_id in self._wb_owners.get(block.block_id, ()):
                picked.setdefault(req_id)
            if len(picked) >= policy.max_requests_per_step:
                break
            block = block.next_free_block
        return list(picked)[: policy.max_requests_per_step]

    def _maybe_log_write_back(self, interval_s: float = 30.0) -> None:
        """One log line per interval. Not on the KV Transfer stats line: the
        Prometheus logger asserts every stats key is registered, and with
        no ``spec_name`` in the connector config its spec lookup resolves to
        the CPU spec, so names added by this spec would not be registered.
        ``allocated_blocks / steps`` (blocks handed out per step, i.e. the
        eviction rate at steady state) is what ``window_blocks`` should
        cover for a few steps."""
        now = time.monotonic()
        last = getattr(self, "_wb_last_log", None)
        if last is None:
            self._wb_last_log = now
            return
        if now - last < interval_s:
            return
        c = self.wb_counters
        logger.info(
            "KV write-back: %d steps, %.1f blocks allocated/step, %d requests "
            "stored, %d dropped (blocks reused before the flush), %d held "
            "(%d blocks; free and still cached: holding defers only the "
            "connector's finish, never the blocks)",
            c["steps"],
            c["allocated_blocks"] / max(c["steps"], 1),
            c["flushed_requests"],
            c["dropped_requests"],
            c["held_requests"],
            len(self._wb_owners),
        )
        if self._offload_stats:
            mgr_stats = getattr(self._raiden_manager, "_stats", None) or {}
            logger.info(
                "Raiden offload stats: drop blocks hit=%d fresh=%d | lookups "
                "partial=%d stale_head=%d (%d tokens behind a missing head) | "
                "store keys offered=%d resident=%d new=%d | jobs ok=%d failed=%d "
                "cancelled=%d (%d keys) | refreshed entries=%d",
                c["drop_blocks_hit"],
                c["drop_blocks_fresh"],
                c["lookup_partial"],
                c["lookup_stale_head"],
                c["lookup_stale_head_tokens"],
                mgr_stats.get("store_keys_offered", 0),
                mgr_stats.get("store_keys_resident", 0),
                mgr_stats.get("store_keys_new", 0),
                mgr_stats.get("store_jobs_ok", 0),
                mgr_stats.get("store_jobs_failed", 0),
                mgr_stats.get("store_jobs_cancelled", 0),
                mgr_stats.get("store_keys_cancelled", 0),
                mgr_stats.get("refreshed_entries", 0),
            )
            logger.info(
                "Raiden offload flush stats: drains %d calls %d jobs %.1f ms | "
                "hit-only flushes skipped: parked %d (%d keys) launched %d | "
                "lookups missing >=8k tokens: %s",
                c["drain_calls"],
                c["drain_jobs"],
                c["drain_us"] / 1e3,
                c["hitflush_skipped_parked"],
                c["hitflush_skipped_parked_keys"],
                c["hitflush_skipped_launched"],
                " ".join(
                    f"{k[len('miss8k_'):]}={c[k]}/{c['miss8k_tok_' + k[len('miss8k_'):]]}"
                    for k in sorted(c)
                    if k.startswith("miss8k_") and not k.startswith("miss8k_tok_")
                )
                or "-",
            )
            if mgr_stats:
                mgr_stats.clear()
        held = c["held_requests"]
        c.clear()
        c["held_requests"] = held
        self._wb_last_log = now

    def _write_back_output(self, scheduler_output: SchedulerOutput) -> SchedulerOutput:
        """The scheduler output the base sees in write-back mode: no running
        request stores (no write-through), newly finished requests are held
        (their finish deferred), and held requests near eviction finish now,
        so the base stores them through its normal finish path."""
        allocated = _allocated_block_ids(scheduler_output)
        self.wb_counters["steps"] += 1
        self.wb_counters["allocated_blocks"] += len(allocated)
        finished = set(scheduler_output.finished_req_ids or ())
        passed = {r for r in finished if not self._wb_register(r)}
        # After registering: a request that finished last step may already
        # have a block reused by this step's allocation.
        reused = allocated & self._wb_owners.keys()
        if reused and self._offload_stats:
            hits = _prefix_hit_block_ids(scheduler_output, self.config)
            self.wb_counters["drop_blocks_hit"] += len(reused & hits)
            self.wb_counters["drop_blocks_fresh"] += len(reused - hits)
        for req_id in {r for bid in reused for r in self._wb_owners.get(bid, ())}:
            self._wb_drop(req_id)
        flush = self._wb_near_eviction()
        for req_id in flush:
            self._wb_unregister(req_id)
        self.wb_counters["flushed_requests"] += len(flush)
        self.wb_counters["held_requests"] = len(self._wb_blocks)
        self._maybe_log_write_back()
        return dataclasses.replace(
            scheduler_output,
            finished_req_ids=passed | set(flush),
            num_scheduled_tokens={},
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        if self._write_back is not None:
            scheduler_output = self._write_back_output(scheduler_output)
        meta = super().build_connector_meta(scheduler_output)
        assert isinstance(meta, OffloadingConnectorMetadata)

        manager = self._raiden_manager

        # Park new store jobs: wait for all ranks to complete device fence.
        new_fence_ids: set[int] = set()
        for job_id, job in meta.store_jobs.items():
            src_spec, dst_spec = job.src_spec, job.dst_spec
            assert isinstance(src_spec, GPULoadStoreSpec)
            assert isinstance(dst_spec, RaidenLoadStoreSpec)
            device_block_ids = src_spec.block_ids.tolist()
            assert len(device_block_ids) == len(dst_spec.keys), (
                len(device_block_ids),
                len(dst_spec.keys),
            )
            self._fence_pending[job_id] = _ParkedStore(
                keys=dst_spec.keys, device_block_ids=device_block_ids
            )
            new_fence_ids.add(job_id)

        # Submit load jobs immediately: destination HBM blocks are freshly
        # allocated and safe to write into.
        for job_id, job in meta.load_jobs.items():
            src_spec, dst_spec = job.src_spec, job.dst_spec
            assert isinstance(src_spec, RaidenLoadStoreSpec)
            assert isinstance(dst_spec, GPULoadStoreSpec)
            device_block_ids = dst_spec.block_ids.tolist()
            assert len(device_block_ids) == len(src_spec.keys), (
                len(device_block_ids),
                len(src_spec.keys),
            )
            manager.submit_load_job(
                job_id,
                src_spec.keys,
                device_block_ids,
                job.req_id,
                pinned=src_spec.pinned,
                host_block_ids=src_spec.host_block_ids,
            )

        # Handle jobs_to_flush (source blocks are about to be recycled):
        # - Cancel fence-pending stores before they start.
        # - Synchronously drain launched stores so DMA finishes before blocks
        #   are overwritten.
        finished_store_ids: list[int] = []
        finished_loads: dict[int, str] = {}
        failed_load_req_ids: set[str] = set()
        flush_ids = meta.jobs_to_flush or set()
        launched_flush_ids: set[int] = set()
        for job_id in flush_ids:
            parked = self._fence_pending.pop(job_id, None)
            if parked is not None:
                manager.on_store_job_cancelled(parked.keys)
                finished_store_ids.append(job_id)
                # Avoid useless TPU sync if created and cancelled in the same step.
                new_fence_ids.discard(job_id)
                logger.debug(
                    "TPURaidenOffloadingConnector: cancelled fence-pending "
                    "store job %d (source blocks reused)",
                    job_id,
                )
            elif manager.is_job_launched(job_id):
                launched_flush_ids.add(job_id)
        drained: list = []
        if launched_flush_ids:
            drain_start = time.perf_counter()
            drained = manager.drain_jobs(launched_flush_ids)
            if self._offload_stats:
                self.wb_counters["drain_calls"] += 1
                self.wb_counters["drain_jobs"] += len(launched_flush_ids)
                self.wb_counters["drain_us"] += int(
                    (time.perf_counter() - drain_start) * 1e6
                )

        # Collect completed/drained jobs for the worker echo channel.
        for finished_job in list(drained) + manager.poll_finished_jobs():
            if finished_job.op == AdmissionOp.STORE:
                finished_store_ids.append(finished_job.job_id)
            else:
                assert finished_job.req_id is not None
                finished_loads[finished_job.job_id] = finished_job.req_id
                if not finished_job.success:
                    failed_load_req_ids.add(finished_job.req_id)

        # Return empty load_jobs/store_jobs: workers act as passive DMA endpoints.
        return RaidenOffloadingConnectorMetadata(
            load_jobs={},  # pyrefly: ignore[unexpected-keyword]
            store_jobs={},  # pyrefly: ignore[unexpected-keyword]
            # pyrefly: ignore[unexpected-keyword]
            jobs_to_flush=meta.jobs_to_flush,
            fence_job_ids=new_fence_ids,
            finished_store_job_ids=finished_store_ids,
            finished_load_jobs=finished_loads,
            failed_load_req_ids=failed_load_req_ids,
        )

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        if num_external_tokens:
            self._align_load_blocks_to_host_order(request, blocks)

    def _align_load_blocks_to_host_order(
        self, request: Request, blocks: KVCacheBlocks
    ) -> None:
        """Give the load's fresh blocks to its keys in host-block order.

        A load moves fastest into device blocks that ascend with the host
        blocks it reads (`raiden_store._dma_order`). The blocks the scheduler
        just allocated for the load hold nothing yet, so which key lands in
        which of them is free: sort them, and hand the lowest one to the key
        with the lowest host block. The request's block list is permuted in
        place, so its block table and later prefix caching follow: `blocks`
        must wrap the KV cache manager's own lists, as the scheduler's
        `get_blocks()` does on the async-load path this connector takes.
        """
        req_status = self._req_status.get(request.request_id)
        if req_status is None or len(req_status.transfer_jobs) != 1:
            return
        job = self._current_batch_load_jobs.get(next(iter(req_status.transfer_jobs)))
        if job is None or len(self.config.kv_group_configs) != 1:
            return
        src_spec, dst_spec = job.src_spec, job.dst_spec
        assert isinstance(src_spec, RaidenLoadStoreSpec)
        num_blocks = len(dst_spec.block_ids)
        host_ids = src_spec.host_block_ids
        if not src_spec.pinned or not host_ids or num_blocks < 2:
            return
        if len(src_spec.keys) != num_blocks or len(host_ids) % num_blocks:
            return
        if min(host_ids) < 0:
            return  # some bytes come from a peer or the storage tier
        # Every key expands to the same number of sub-hashes, which travel
        # together: order keys by their first one. Any assignment of the fresh
        # blocks is correct; the order only decides how well the copy coalesces.
        key_host_ids = host_ids[:: len(host_ids) // num_blocks]
        start = dst_spec.block_indices[0]
        group_blocks = blocks.blocks[self.config.kv_group_configs[0].group_idx]
        fresh = group_blocks[start : start + num_blocks]
        if [b.block_id for b in fresh] != dst_spec.block_ids.tolist() or any(
            b.block_hash is not None or b.ref_cnt != 1 for b in fresh
        ):
            return
        by_host = sorted(range(num_blocks), key=key_host_ids.__getitem__)
        block_at = dict(zip(by_host, sorted(fresh, key=lambda b: b.block_id)))
        aligned = [block_at[pos] for pos in range(num_blocks)]
        group_blocks[start : start + num_blocks] = aligned
        dst_spec.block_ids = np.array(
            [b.block_id for b in aligned], dtype=dst_spec.block_ids.dtype
        )

    def _prefix_offload_keys(self, request: Request, start_token: int):
        """(tokens_per_chunk, keys) of the first prefix group's chunks from the
        chunk holding `start_token` on; None without a prefix group."""
        block_hashes = request.block_hashes
        for group_config in self.config.kv_group_configs:
            if group_config.is_eagle_group:
                continue
            if group_config.sliding_window_size_in_chunks is not None:
                return None
            hashes_per_chunk = group_config.hashes_per_chunk
            start_chunk = start_token // group_config.tokens_per_chunk
            keys = [
                make_offload_key(block_hashes[idx], group_config.group_idx)
                for idx in range(
                    (start_chunk + 1) * hashes_per_chunk - 1,
                    len(block_hashes),
                    hashes_per_chunk,
                )
            ]
            return group_config.tokens_per_chunk, keys
        return None

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int, *args, **kwargs):
        result = super().get_num_new_matched_tokens(
            request, num_computed_tokens, *args, **kwargs
        )
        if self._offload_stats and result[0] is not None:
            self._count_stale_head(request, num_computed_tokens, result[0])
        if self._loss_trace and result[0] is not None:
            self._trace_miss(request, num_computed_tokens, result[0])
        return result

    _LOSS_TRACE_MIN_TOKENS = 8192

    def _trace_miss(
        self, request: Request, num_computed_tokens: int, num_hit_tokens: int
    ) -> None:
        """TPU_RAIDEN_LOSS_TRACE: once per request whose local + host hit stops
        >= 8k tokens short of its prompt, log why the first missing key is
        not on the host (never offered, store cancelled, evicted, ...)."""
        req_id = request.request_id
        if req_id in self._traced_req_ids:
            return
        self._traced_req_ids.add(req_id)
        stop = num_computed_tokens + num_hit_tokens
        missing = request.num_prompt_tokens - stop
        if missing < self._LOSS_TRACE_MIN_TOKENS:
            return
        found = self._prefix_offload_keys(request, stop)
        if found is None or not found[1]:
            return
        event, age, refreshed, stored = self._raiden_manager.explain_missing(
            found[1][0]
        )
        c = self.wb_counters
        c["miss8k_" + event] += 1
        c["miss8k_tok_" + event] += missing
        logger.info(
            "Raiden loss trace: req %s prompt %d local %d host %d missing %d: "
            "first missing key %s %.0fs ago (refreshed %.0fs, stored %.0fs ago)",
            req_id,
            request.num_prompt_tokens,
            num_computed_tokens,
            num_hit_tokens,
            missing,
            event,
            age,
            refreshed,
            stored,
        )

    def _count_stale_head(
        self, request: Request, num_computed_tokens: int, num_hit_tokens: int
    ) -> None:
        """Stats only: a host lookup that stopped short of the chain while its
        last key is resident is a chain stranded behind a missing head."""
        found = self._prefix_offload_keys(
            request, num_computed_tokens + num_hit_tokens
        )
        if found is None:
            return
        tokens_per_chunk, keys = found
        # The prompt's last token is never a hit; ignore a trailing one-chunk gap.
        if len(keys) < 2:
            return
        self.wb_counters["lookup_partial"] += 1
        if self._raiden_manager.count_resident_prefix(keys[-1:]):
            self.wb_counters["lookup_stale_head"] += 1
            self.wb_counters["lookup_stale_head_tokens"] += (
                len(keys) * tokens_per_chunk
            )

    def request_finished(self, request: Request, *args, **kwargs):
        traced = getattr(self, "_traced_req_ids", None)
        if traced:
            traced.discard(request.request_id)
        if self._lru_refresh_on_finish:
            req_status = self._req_status.get(request.request_id)
            if req_status is not None:
                for group_config, group_state in zip(
                    self.config.kv_group_configs, req_status.group_states
                ):
                    if group_config.is_eagle_group or not group_state.offload_keys:
                        continue
                    self._raiden_manager.refresh_resident(group_state.offload_keys)
        return super().request_finished(request, *args, **kwargs)

    def peek_external_prefix_tokens(
        self, request: Request, num_local_tokens: int
    ) -> int:
        """Prompt tokens past `num_local_tokens` resident in the host store.

        Side-effect free (see `ExternalPrefixProbe`). Prefix groups only: if a
        sliding-window group is offloaded the estimate is 0, since its hit is
        a suffix window that this probe does not model.
        """
        if request.skip_reading_prefix_cache:
            return 0
        block_hashes = request.block_hashes
        limit = max(request.num_tokens - 1, 0)
        hit_end: int | None = None
        for group_config in self.config.kv_group_configs:
            if group_config.is_eagle_group:
                continue
            if group_config.sliding_window_size_in_chunks is not None:
                return 0
            tokens_per_chunk = group_config.tokens_per_chunk
            hashes_per_chunk = group_config.hashes_per_chunk
            start_chunk = num_local_tokens // tokens_per_chunk
            keys = [
                make_offload_key(block_hashes[idx], group_config.group_idx)
                for idx in range(
                    (start_chunk + 1) * hashes_per_chunk - 1,
                    len(block_hashes),
                    hashes_per_chunk,
                )
            ]
            num_chunks = self._raiden_manager.count_resident_prefix(keys)
            group_end = (start_chunk + num_chunks) * tokens_per_chunk
            hit_end = group_end if hit_end is None else min(hit_end, group_end)
        if hit_end is None:
            return 0
        return max(0, min(hit_end, limit) - num_local_tokens)

    def update_connector_output(self, connector_output: KVConnectorOutput):
        meta = connector_output.kv_connector_worker_meta
        if isinstance(meta, RaidenOffloadingWorkerMetadata) and meta.fenced_jobs:
            for job_id, count in meta.fenced_jobs.items():
                parked = self._fence_pending.get(job_id)
                if parked is None:
                    # Cancelled while the acks were in flight.
                    continue
                parked.acks += count
                assert parked.acks <= self.config.num_workers
                if parked.acks == self.config.num_workers:
                    del self._fence_pending[job_id]
                    self._raiden_manager.submit_store_job(
                        job_id, parked.keys, parked.device_block_ids
                    )
        super().update_connector_output(connector_output)


class TPURaidenOffloadingConnector(OffloadingConnector):
    """Top-level facade connector for the store-backed (V2) raiden KV offload path."""

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        """No layout preference, unlike the `OffloadingConnector` this extends.

        Upstream hardcodes "HND" for NIXL on CUDA. On TPU "HND" names
        SEQ_ALONG_LANE, so inheriting it would select a layout nobody asked
        for; these pages are opaque bytes to this connector.
        """
        return None

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        # Opt in to uniform cross-layer KV cache allocation when block-major layout is
        # enabled. Satisfies `KVConnectorModelRunnerMixin.use_uniform_kv_cache()`
        # preconditions to enable bundled tensor allocation in the TPU model runner.
        return bool(tpu_envs.VLLM_TPU_BLOCK_MAJOR_KV)

    @property
    def supports_divergent_local_hybrid_hits(self) -> bool:
        """Complete a deeper full-attention local hit, behind the Stage-3 flag.

        Upstream offload lookup treats a Mamba group as a one-chunk window, so
        a hit from this store ends where it holds the recurrent state for its
        last chunk: a load restores the state the local hit lacks, and a miss
        lets the scheduler reconcile to the common boundary. Decode composes
        this connector with TPURaidenConnector in TPUMultiConnector, which
        needs every child to agree before Stage-3 suffix loads can run. Gated
        on the prefix-aware flag so standalone offload keeps today's lookup.
        """
        # Deferred: tpu_connector is heavy and only this property needs it.
        from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
            _prefix_aware_load_enabled,
        )

        return _prefix_aware_load_enabled()

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        # Skip OffloadingConnector.__init__: default factory cannot carry
        # vllm_config / kv_cache_config directly into the spec.
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)

        # The raiden singleton worker cannot coexist with this connector: the
        # disagg stack runs its own control plane in the same process and the
        # two collide on ports.
        if not tpu_envs.RAIDEN_DISABLE_SINGLETON_WORKER:
            raise ValueError(
                "TPURaidenOffloadingConnector requires the raiden singleton "
                "worker to be disabled (it collides with the in-process "
                "control plane's ports), but RAIDEN_DISABLE_SINGLETON_WORKER "
                "was explicitly set to off. Unset it or set it to '1'."
            )
        os.environ["RAIDEN_DISABLE_SINGLETON_WORKER"] = "1"

        # Call through module to preserve PCP-aware patch on build_offloading_config.
        offloading_config = offloading_config_module.build_offloading_config(
            vllm_config, kv_cache_config
        )
        spec = TPURaidenStoreOffloadingSpec(
            offloading_config, vllm_config, kv_cache_config
        )
        self._skip_draft_kv = spec.skip_draft_kv

        # pyrefly: ignore[bad-override-mutable-attribute]
        self.connector_scheduler: TPURaidenOffloadingScheduler | None = None
        self.connector_worker: OffloadingConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = TPURaidenOffloadingScheduler(
                spec, vllm_config, kv_cache_config
            )
        elif role == KVConnectorRole.WORKER:
            # Worker wrapper is only used for KV buffer registration; job dicts
            # remain empty.
            self.connector_worker = OffloadingConnectorWorker(
                spec, vllm_config, kv_cache_config
            )
            # Track per-step fence acknowledgments and completion echoes for
            # worker metadata.
            self._fenced_jobs: dict[int, int] = {}
            self._completed_jobs: dict[int, int] = {}
            # Registered KV pool byte views whose deferred writes are dispatched before
            # recording fence events.
            self._pool_sync_tensors: list[torch.Tensor] = []
            # In-flight fence events awaiting device retirement before acknowledging
            # store jobs, recorded as pairs of (fence_job_ids, event).
            self._pending_fences: list[tuple[set[int], torch.Event]] = []

    # -- scheduler-side hooks ----------------------------------------------
    def reset_cache(self) -> bool | None:
        # External cache reset is unsupported by the Raiden store. Report
        # failure through the endpoint's documented channel instead of raising an
        # exception here
        logger.warning(
            "TPURaidenOffloadingConnector: external cache reset is not "
            "supported by the Raiden offload store; reporting failure."
        )
        return False

    def take_events(self) -> Iterable[KVCacheEvent]:
        if self.connector_scheduler is not None:
            return self.connector_scheduler.take_events()
        return []

    def attach_block_pool(self, block_pool) -> None:
        """`EvictionAwareConnector`: hand the block pool to the scheduler side
        (write-back reads its free queue)."""
        if self.connector_scheduler is not None:
            self.connector_scheduler.attach_block_pool(block_pool)

    def peek_external_prefix_tokens(
        self, request: Request, num_local_tokens: int
    ) -> int:
        """`ExternalPrefixProbe`: host-resident prompt tokens past the local hit."""
        if self.connector_scheduler is None:
            return 0
        return self.connector_scheduler.peek_external_prefix_tokens(
            request, num_local_tokens
        )

    # -- worker-side hooks -------------------------------------------------
    def register_kv_caches(
        self, kv_caches: Mapping[str, torch.Tensor | list[torch.Tensor]]
    ):
        """Register the TPU UBP as zero-copy canonical offload tensors.

        vLLM's generic per-layer registration does not support the TPU
        unified block pool layout or Mamba tensor lists, so this
        implementation deduplicates the UBP backing storages and exposes
        each as an `int8` byte view.
        """
        assert self.connector_worker is not None
        kv_cache_config = self.connector_worker.kv_cache_config
        num_blocks = kv_cache_config.num_blocks

        # skip_draft_kv: the drafter's arrays stay out of the host store (the
        # spec already sized the host pool without them).
        skipped_layers = (
            draft_kv_layer_names(kv_cache_config)
            if getattr(self, "_skip_draft_kv", False)
            else set()
        )

        pool_storages: dict[int, tuple[torch.UntypedStorage, torch.device]] = {}
        skipped_storages: set[int] = set()
        for layer_name, layer_kv_cache in kv_caches.items():
            state_tensors = (
                [layer_kv_cache]
                if isinstance(layer_kv_cache, torch.Tensor)
                else layer_kv_cache
            )
            for tensor in state_tensors:
                storage = tensor.untyped_storage()
                if layer_name in skipped_layers:
                    skipped_storages.add(storage.data_ptr())
                    continue
                pool_storages.setdefault(storage.data_ptr(), (storage, tensor.device))
        shared = skipped_storages & pool_storages.keys()
        assert not shared, (
            "TPURaidenOffloadingConnector: skip_draft_kv needs the drafter's K/V "
            "in arrays of their own; it shares storage with offloaded layers"
        )
        assert not skipped_layers or skipped_storages, (
            "TPURaidenOffloadingConnector: skip_draft_kv found drafter layers "
            f"{sorted(skipped_layers)} in the KV config but none in kv_caches"
        )

        pool_tensors: list[CanonicalKVCacheTensor] = []
        for storage, device in pool_storages.values():
            storage_bytes = storage.nbytes()
            assert storage_bytes % num_blocks == 0, (
                "KV pool storage must hold whole scheduler blocks: "
                f"storage_bytes={storage_bytes}, num_blocks={num_blocks}"
            )
            page_size_bytes = storage_bytes // num_blocks
            int8_view = (
                torch.empty(0, dtype=torch.int8, device=device)
                .set_(storage)
                .view(num_blocks, page_size_bytes)
            )
            pool_tensors.append(
                CanonicalKVCacheTensor(
                    tensor=int8_view, page_size_bytes=page_size_bytes
                )
            )

        # Retain pool tensor references to dispatch their writes during save fences.
        self._pool_sync_tensors = [pool_tensor.tensor for pool_tensor in pool_tensors]

        # One ref list per offloaded group, as vLLM's own registration builds:
        # groups that are not prefix-cacheable are not offloaded.
        group_data_refs = [
            [
                CanonicalKVCacheRef(
                    tensor_idx=idx, page_size_bytes=pool_tensor.page_size_bytes
                )
                for idx, pool_tensor in enumerate(pool_tensors)
            ]
            for _ in offloading_config_module.get_offloading_group_ids(
                self.connector_worker.kv_cache_config
            )
        ]

        self.connector_worker._init_worker(
            CanonicalKVCaches(tensors=pool_tensors, group_data_refs=group_data_refs)
        )

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        # Load failures only travel in get_transfer_results().failed_recving.
        # Returning a failed request here would pass its load as a good one,
        # so refuse loudly; callers must use get_transfer_results().
        results = self.get_transfer_results(finished_req_ids)
        if results.failed_recving:
            raise RuntimeError(
                "TPURaidenOffloadingConnector.get_finished() cannot report the "
                f"failed loads of requests {sorted(results.failed_recving)}; "
                "call get_transfer_results() instead"
            )
        return results.finished_sending, results.finished_recving

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        assert self.connector_worker is not None
        meta = self._get_connector_metadata()
        assert isinstance(meta, RaidenOffloadingConnectorMetadata)

        if meta.fence_job_ids:
            # Establish an asynchronous device fence before acknowledging store jobs:
            # 1. Dispatch any deferred pool writes so they precede the fence event.
            # 2. Record a stream event to track completion without host-side blocking.
            assert self._pool_sync_tensors, (
                "KV pool tensors must be registered before the save fence"
            )
            _enqueue_pool_writes(self._pool_sync_tensors)
            self._pending_fences.append(
                (set(meta.fence_job_ids), _record_fence_event())
            )

        if self._pending_fences:
            # Non-blockingly drain retired events and acknowledge completed fences.
            still_pending: list[tuple[set[int], torch.Event]] = []
            for job_ids, event in self._pending_fences:
                if _event_is_retired(event):
                    for job_id in job_ids:
                        self._fenced_jobs[job_id] = 1
                else:
                    still_pending.append((job_ids, event))
            self._pending_fences = still_pending

        finished_recving: set[str] = set()
        for job_id in meta.finished_store_job_ids:
            self._completed_jobs[job_id] = 1
        for job_id, req_id in meta.finished_load_jobs.items():
            self._completed_jobs[job_id] = 1
            finished_recving.add(req_id)

        # A failed request also stays in finished_recving: that is what
        # releases it from WAITING_FOR_REMOTE_KVS.
        return KVConnectorTransferResults(
            finished_recving=finished_recving,
            failed_recving=set(meta.failed_load_req_ids),
        )

    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        if not (self._fenced_jobs or self._completed_jobs):
            return None
        # completed_jobs is inherited from vLLM base dataclass.
        meta = RaidenOffloadingWorkerMetadata(
            # pyrefly: ignore[unexpected-keyword]
            completed_jobs=self._completed_jobs,
            fenced_jobs=self._fenced_jobs,
        )
        self._completed_jobs = {}
        self._fenced_jobs = {}
        return meta
