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
  report their destination blocks for automatic recomputation.

- Memory reuse (jobs_to_flush): Handled entirely scheduler-side. Fence-pending
  stores are cancelled before launching, while active transfers are drained
  synchronously before shipping metadata, preventing torn reads upon block reuse.

- Lazy offload (`"lazy_offload": true` in kv_connector_extra_config): blocks
  are stored only on their way out of HBM instead of as they are computed.
  Each step the scheduler scans the first `lazy_window_blocks` free HBM blocks
  (the ones the allocator hands out next, evicting what they cache) and stores
  every whole cached block the tier does not hold yet. The tier then holds
  what HBM dropped rather than a copy of what HBM still has, so it adds its
  capacity to HBM's even when smaller. A scanned block stays where it is in
  the free queue (no pin): if the allocator takes it before its store
  finishes, the memory-reuse flush above cancels or drains the store. Blocks
  are stored as the HBM prefix cache holds them, so a tier hit serves what an
  HBM hit on the same block would. A lookup refreshes only the tier's copies
  past the request's HBM hit, so the tier's LRU order follows what HBM
  dropped.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
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
    RequestOffloadState,
    TransferJobStatus,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.core.kv_cache_utils import get_block_hash, get_group_id
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    OffloadKey,
    make_offload_key,
)
from vllm.v1.outputs import KVConnectorOutput

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.offload.raiden_store import (
    AdmissionOp,
    RaidenLoadStoreSpec,
    RaidenOffloadingManager,
    TPURaidenStoreOffloadingSpec,
)

if TYPE_CHECKING:
    from vllm.distributed.kv_events import KVCacheEvent
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        SchedulerOffloadConfig,
    )
    from vllm.v1.core.block_pool import BlockPool

logger = init_logger(__name__)

# Lazy offload scans this many free HBM blocks by default: 64 DCP16 blocks
# (4096 tokens each) cover a ~256K-token tier load, the largest allocation a
# single prefill step makes in practice. A block the allocator takes before
# its store completes is not offloaded, so the window must run ahead of a
# step's allocations.
_DEFAULT_LAZY_WINDOW_BLOCKS = 64
# Lazy store jobs belong to no request; this stands in for the owner.
_LAZY_JOB_OWNER = "raiden-lazy-offload"
_LAZY_STATS_INTERVAL_S = 30.0


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


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
    # Destination device blocks of failed loads, reported for scheduler recompute.
    failed_load_device_block_ids: list[int] = field(default_factory=list)


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


@dataclass
class _LazyOffload:
    """Lazy-offload state: HBM blocks are stored only as they near eviction."""

    window_blocks: int
    # Tokens per offload chunk (one HBM block) by KV cache group id.
    chunk_tokens: dict[int, int]
    gpu_block_pool: BlockPool | None = None
    # In-flight lazy store jobs; they belong to no request.
    jobs: set[int] = field(default_factory=set)
    # Keys whose store failed, not offered again while their block waits in
    # the window (a retry would most likely fail the same way).
    failed_keys: set[OffloadKey] = field(default_factory=set)
    # stored: jobs created; cancelled: flushed before launch (the block was
    # reused first); drained: flushed in flight (waited for); failed.
    stats: Counter = field(default_factory=Counter)
    logged_stats: Counter = field(default_factory=Counter)
    last_log: float = 0.0


# TODO(vllm-project/vllm#52735): Remove TPU_OFFLOAD_KEEP_EAGLE_TAIL and
# keep_unannotated_eagle_tail once the pinned vLLM includes
# vllm-project/vllm#52771.
def keep_unannotated_eagle_tail(
    config: SchedulerOffloadConfig,
    kv_cache_groups: Sequence[KVCacheGroupSpec],
) -> tuple[SchedulerOffloadConfig, tuple[int, ...]]:
    """Clears the drafter flag vLLM sets on unannotated offload groups.

    Returns ``(config, cleared_group_ids)``. The pinned vLLM
    (``SchedulerOffloadConfig.from_spec``) flags every KV cache group as an
    EAGLE/MTP drafter group when speculative decoding is on and no group is
    annotated as one. A tier lookup then pops the last chunk of each flagged
    group's hit as a "volatile draft tail", and drops a one-chunk hit
    entirely. GLM-5.2's MTP layer shares the target's single group, so every
    tier hit recomputes one whole chunk (4096 tokens under DCP16). vLLM main
    treats an unannotated group as a regular group (vllm-project/vllm#52771):
    drafter KV served one chunk stale can only lower speculative acceptance,
    because the target verifies every draft. On a P/D prefill leg the chunk is
    not even stale: lazy offload stores only whole blocks written by prefills,
    and a prefilled chunk's last draft input is the next prompt token.

    Annotated drafter groups keep their flag.
    """
    groups = config.kv_group_configs
    annotated = {
        idx for idx, group in enumerate(kv_cache_groups) if group.is_eagle_group
    }
    cleared = tuple(
        group.group_idx
        for group in groups
        if group.is_eagle_group and group.group_idx not in annotated
    )
    if not cleared:
        return config, ()
    new_groups = tuple(
        group._replace(is_eagle_group=False) if group.group_idx in cleared else group
        for group in groups
    )
    return config._replace(kv_group_configs=new_groups), cleared


class TPURaidenOffloadingScheduler(OffloadingConnectorScheduler):
    """Scheduler for the store-backed offload path.

    Strips per-rank transfer jobs and orchestrates the store engine directly
    via RaidenOffloadingManager.
    """

    # None unless kv_connector_extra_config sets lazy_offload.
    _lazy: _LazyOffload | None = None

    def __init__(
        self,
        spec: TPURaidenStoreOffloadingSpec,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(spec, vllm_config, kv_cache_config)
        if tpu_envs.TPU_OFFLOAD_KEEP_EAGLE_TAIL:
            # Before any request: each RequestOffloadState copies self.config.
            self.config, cleared = keep_unannotated_eagle_tail(
                self.config, kv_cache_config.kv_cache_groups
            )
            logger.info(
                "TPU_OFFLOAD_KEEP_EAGLE_TAIL: offload tier hits keep their last "
                "chunk for KV cache groups %s; drafter groups left flagged: %s",
                list(cleared),
                [
                    group.group_idx
                    for group in self.config.kv_group_configs
                    if group.is_eagle_group
                ],
            )
        assert isinstance(self.manager, RaidenOffloadingManager)
        self._raiden_manager: RaidenOffloadingManager = self.manager
        self._fence_pending: dict[int, _ParkedStore] = {}
        if _as_bool(spec.extra_config.get("lazy_offload", False)):
            self._lazy = self._init_lazy_offload(spec, vllm_config, kv_cache_config)

    def _init_lazy_offload(
        self,
        spec: TPURaidenStoreOffloadingSpec,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> _LazyOffload:
        # A lazy store copies one HBM block under the key its cached hash
        # names, so every group must offload whole blocks, one chunk each.
        parallel_config = vllm_config.parallel_config
        cp_size = int(parallel_config.decode_context_parallel_size or 1) * int(
            parallel_config.prefill_context_parallel_size or 1
        )
        unsupported = [
            group.group_idx
            for group in self.config.kv_group_configs
            if group.sliding_window_size_in_chunks is not None
            or group.requires_cow_source
            or group.tokens_per_chunk
            != kv_cache_config.kv_cache_groups[group.group_idx].kv_cache_spec.block_size
            * cp_size
        ]
        if self.config.blocks_per_chunk != 1 or unsupported:
            raise ValueError(
                "TPURaidenOffloadingConnector: lazy_offload stores whole HBM "
                "blocks, so it needs full-attention KV cache groups offloaded "
                "one scheduler block per chunk; got blocks_per_chunk="
                f"{self.config.blocks_per_chunk} and unsupported groups "
                f"{unsupported}"
            )
        window_blocks = int(
            spec.extra_config.get("lazy_window_blocks", _DEFAULT_LAZY_WINDOW_BLOCKS)
        )
        if window_blocks <= 0:
            raise ValueError(
                "TPURaidenOffloadingConnector: lazy_window_blocks must be "
                f"positive, got {window_blocks}"
            )
        self._raiden_manager.disable_request_stores()
        chunk_tokens = {
            group.group_idx: group.tokens_per_chunk
            for group in self.config.kv_group_configs
        }
        logger.info(
            "TPURaidenOffloadingConnector: lazy offload: storing HBM blocks only "
            "as they near eviction (window %d blocks, %s tokens per block)",
            window_blocks,
            sorted(set(chunk_tokens.values())),
        )
        return _LazyOffload(window_blocks=window_blocks, chunk_tokens=chunk_tokens)

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        if self._lazy is not None:
            self._lazy.gpu_block_pool = gpu_block_pool

    def _touch(self, req_status: RequestOffloadState) -> None:
        """Under lazy offload, refresh only the chunks past the HBM hit.

        A lookup touches every key of the request, the prefix HBM serves
        included. A lazy tier holds what HBM dropped, so refreshing its copies
        of blocks HBM still serves would keep them at the Most Recently Used
        end, and the tier would evict the blocks HBM actually dropped first.
        The tier holds no partial-tail entries in lazy mode (request stores
        are off), so there is no boundary key to refresh either.
        """
        if self._lazy is None:
            super()._touch(req_status)
            return
        for group_config, group_state in zip(
            self.config.kv_group_configs, req_status.group_states
        ):
            served = (
                req_status.num_locally_computed_tokens // group_config.tokens_per_chunk
            )
            self.manager.touch(
                group_state.offload_keys[served:], req_status.req_context
            )

    def _schedule_lazy_stores(self) -> set[int]:
        """Park store jobs for the cached HBM blocks the allocator takes next.

        Scans the first `window_blocks` blocks of the free queue in eviction
        order and parks one store job per whole cached block the tier does
        not hold yet. The block is left in place: a store job is watched
        through `_block_id_to_pending_jobs`, so when the allocator takes the
        block, `build_connector_meta` cancels the job if it has not launched
        and drains it if it has, before the block is written.
        """
        lazy = self._lazy
        assert lazy is not None
        pool = lazy.gpu_block_pool
        if pool is None:
            return set()
        keys: list[OffloadKey] = []
        block_ids: list[int] = []
        seen: set[OffloadKey] = set()
        for position, block in enumerate(pool.free_block_queue.iter_blocks_after(None)):
            if position >= lazy.window_blocks:
                break
            block_hash = block.block_hash
            if (
                block_hash is None
                or block.is_null
                or block.block_id in self._block_id_to_pending_jobs
            ):
                continue
            chunk_tokens = lazy.chunk_tokens.get(get_group_id(block_hash))
            num_tokens = block.block_hash_num_tokens
            # A partial entry (a fine prefix hit ending inside the block) is
            # no whole chunk; neither the lookup nor a load can use it.
            if chunk_tokens is None or not num_tokens or num_tokens % chunk_tokens:
                continue
            key = make_offload_key(get_block_hash(block_hash), get_group_id(block_hash))
            if key in lazy.failed_keys or key in seen:
                continue
            seen.add(key)
            keys.append(key)
            block_ids.append(block.block_id)
        if not keys:
            return set()
        store_output = self._raiden_manager.prepare_evicting_store(keys)
        if store_output is None or not store_output.keys_to_store:
            return set()
        accepted = set(store_output.keys_to_store)
        job_ids: set[int] = set()
        for key, block_id in zip(keys, block_ids):
            if key not in accepted:
                continue
            job_id = self._generate_job_id()
            self._jobs[job_id] = TransferJobStatus(
                req_id=_LAZY_JOB_OWNER,
                pending_count=self.config.num_workers,
                keys={key},
                is_store=True,
                fenced_block_ids=[block_id],
            )
            self._block_id_to_pending_jobs.setdefault(block_id, set()).add(job_id)
            self._fence_pending[job_id] = _ParkedStore(
                keys=[key], device_block_ids=[block_id]
            )
            lazy.jobs.add(job_id)
            job_ids.add(job_id)
        lazy.stats["stored"] += len(job_ids)
        return job_ids

    def _complete_lazy_jobs(
        self, connector_output: KVConnectorOutput, meta: OffloadingWorkerMetadata
    ) -> KVConnectorOutput:
        """Retire echoed lazy store jobs, and hand the rest to the base class,
        which looks every job's request up and would find none for these."""
        lazy = self._lazy
        assert lazy is not None
        others: dict[int, int] = {}
        for job_id, count in meta.completed_jobs.items():
            if job_id not in lazy.jobs:
                others[job_id] = count
                continue
            job = self._jobs.get(job_id)
            if job is None:
                lazy.jobs.discard(job_id)
                continue
            job.pending_count -= count
            assert job.pending_count >= 0, (job_id, job.pending_count)
            if job.pending_count:
                continue
            self._remove_pending_job(job_id, job.fenced_block_ids)
            del self._jobs[job_id]
            lazy.jobs.discard(job_id)
        if len(others) == len(meta.completed_jobs):
            return connector_output
        return replace(
            connector_output,
            kv_connector_worker_meta=replace(meta, completed_jobs=others),
        )

    def _maybe_log_lazy_stats(self) -> None:
        lazy = self._lazy
        assert lazy is not None
        now = time.monotonic()
        if (
            lazy.stats == lazy.logged_stats
            or now - lazy.last_log < _LAZY_STATS_INTERVAL_S
        ):
            return
        lazy.last_log = now
        lazy.logged_stats = Counter(lazy.stats)
        logger.info(
            "TPURaidenOffloadingConnector: lazy offload totals: stored=%d "
            "cancelled=%d drained=%d failed=%d in_flight=%d",
            lazy.stats["stored"],
            lazy.stats["cancelled"],
            lazy.stats["drained"],
            lazy.stats["failed"],
            len(lazy.jobs),
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
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
            )

        # Handle jobs_to_flush (source blocks are about to be recycled):
        # - Cancel fence-pending stores before they start.
        # - Synchronously drain launched stores so DMA finishes before blocks
        #   are overwritten.
        finished_store_ids: list[int] = []
        finished_loads: dict[int, str] = {}
        failed_load_block_ids: list[int] = []
        flush_ids = meta.jobs_to_flush or set()
        launched_flush_ids: set[int] = set()
        lazy = self._lazy
        for job_id in flush_ids:
            parked = self._fence_pending.pop(job_id, None)
            if parked is not None:
                manager.on_store_job_cancelled(parked.keys)
                finished_store_ids.append(job_id)
                # Avoid useless TPU sync if created and cancelled in the same step.
                new_fence_ids.discard(job_id)
                if lazy is not None and job_id in lazy.jobs:
                    lazy.stats["cancelled"] += 1
                logger.debug(
                    "TPURaidenOffloadingConnector: cancelled fence-pending "
                    "store job %d (source blocks reused)",
                    job_id,
                )
            elif manager.is_job_launched(job_id):
                launched_flush_ids.add(job_id)
                if lazy is not None and job_id in lazy.jobs:
                    lazy.stats["drained"] += 1
        drained = manager.drain_jobs(launched_flush_ids) if launched_flush_ids else []

        # Collect completed/drained jobs for the worker echo channel.
        for finished_job in list(drained) + manager.poll_finished_jobs():
            if finished_job.op == AdmissionOp.STORE:
                finished_store_ids.append(finished_job.job_id)
                if (
                    lazy is not None
                    and not finished_job.success
                    and finished_job.job_id in lazy.jobs
                ):
                    lazy.stats["failed"] += 1
                    job = self._jobs.get(finished_job.job_id)
                    if job is not None:
                        if len(lazy.failed_keys) > 4096:
                            lazy.failed_keys.clear()
                        lazy.failed_keys.update(job.keys)
            else:
                assert finished_job.req_id is not None
                finished_loads[finished_job.job_id] = finished_job.req_id
                failed_load_block_ids.extend(finished_job.failed_device_block_ids)

        if lazy is not None:
            # After the flush above: every block this step allocated has left
            # the free queue, so the scan sees only blocks still free.
            new_fence_ids.update(self._schedule_lazy_stores())
            self._maybe_log_lazy_stats()

        # Return empty load_jobs/store_jobs: workers act as passive DMA endpoints.
        return RaidenOffloadingConnectorMetadata(
            load_jobs={},  # pyrefly: ignore[unexpected-keyword]
            store_jobs={},  # pyrefly: ignore[unexpected-keyword]
            # pyrefly: ignore[unexpected-keyword]
            jobs_to_flush=meta.jobs_to_flush,
            fence_job_ids=new_fence_ids,
            finished_store_job_ids=finished_store_ids,
            finished_load_jobs=finished_loads,
            failed_load_device_block_ids=failed_load_block_ids,
        )

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
        if (
            self._lazy is not None
            and isinstance(meta, OffloadingWorkerMetadata)
            and meta.completed_jobs
        ):
            connector_output = self._complete_lazy_jobs(connector_output, meta)
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
            self._load_error_block_ids: set[int] = set()
            # Registered KV pool byte views whose deferred writes are dispatched before
            # recording fence events.
            self._pool_sync_tensors: list[torch.Tensor] = []
            # In-flight fence events awaiting device retirement before acknowledging
            # store jobs, recorded as pairs of (fence_job_ids, event).
            self._pending_fences: list[tuple[set[int], torch.Event]] = []

    # -- scheduler-side hooks ----------------------------------------------
    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        # Lazy offload scans the pool's free queue for blocks near eviction.
        if self.connector_scheduler is not None:
            self.connector_scheduler.bind_gpu_block_pool(gpu_block_pool)

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
        num_blocks = self.connector_worker.kv_cache_config.num_blocks

        pool_storages: dict[int, tuple[torch.UntypedStorage, torch.device]] = {}
        for layer_kv_cache in kv_caches.values():
            state_tensors = (
                [layer_kv_cache]
                if isinstance(layer_kv_cache, torch.Tensor)
                else layer_kv_cache
            )
            for tensor in state_tensors:
                storage = tensor.untyped_storage()
                pool_storages.setdefault(storage.data_ptr(), (storage, tensor.device))

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

        group_data_refs = [
            [
                CanonicalKVCacheRef(
                    tensor_idx=idx, page_size_bytes=pool_tensor.page_size_bytes
                )
                for idx, pool_tensor in enumerate(pool_tensors)
            ]
            for _ in self.connector_worker.kv_cache_config.kv_cache_groups
        ]

        self.connector_worker._init_worker(
            CanonicalKVCaches(tensors=pool_tensors, group_data_refs=group_data_refs)
        )

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
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
        self._load_error_block_ids.update(meta.failed_load_device_block_ids)

        return set(), finished_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        block_ids = self._load_error_block_ids
        self._load_error_block_ids = set()
        return block_ids

    def get_block_ids_with_load_errors_group_index(self) -> int | None:
        """Returns None: offload load errors are not scoped to one KV group.

        The runner reads this whenever ``get_block_ids_with_load_errors``
        returns a non-empty set. Failed device block ids come from the
        scheduler-side block table and resolve against every cache group,
        so there is no single group index to report.
        """
        return None

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
