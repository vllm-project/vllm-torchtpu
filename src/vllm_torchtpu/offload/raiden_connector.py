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

- Partial tails under DCP: `"dcp_partial_tail": true` in
  `kv_connector_extra_config` (default off) offloads and loads hybrid-model
  prefixes that end inside a DCP-scaled attention block. See
  `TPURaidenOffloadingScheduler`.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
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
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    _ConnectorMetricName,
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
from vllm.utils.math_utils import round_down
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    LookupResult,
    OffloadKey,
    OffloadPolicy,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.offload.raiden_store import (
    AdmissionOp,
    RaidenLoadStoreSpec,
    RaidenOffloadingManager,
    TPURaidenStoreOffloadingSpec,
    _parse_storage_flag,
)

if TYPE_CHECKING:
    from vllm.distributed.kv_events import KVCacheEvent

logger = init_logger(__name__)

# kv_connector_extra_config key enabling partial tails under DCP.
_DCP_PARTIAL_TAIL_KEY = "dcp_partial_tail"


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


class TPURaidenOffloadingScheduler(OffloadingConnectorScheduler):
    """Scheduler for the store-backed offload path.

    Strips per-rank transfer jobs and orchestrates the store engine directly
    via RaidenOffloadingManager.

    With `dcp_partial_tail` set, also stores and loads partial tails for
    hybrid models under DCP. vLLM scales attention blocks by DCP but not
    recurrent state blocks, so an attention block spans several hash
    boundaries, each with its own state snapshot. Upstream partial tails need
    one block size and DCP == 1. Here a tail is the partly filled attention
    block plus the state snapshots, stored under the boundary's keys.
    """

    def __init__(
        self,
        spec: TPURaidenStoreOffloadingSpec,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(spec, vllm_config, kv_cache_config)
        if self.config.sync_load_budget_bytes:
            raise ValueError(
                "TPURaidenOffloadingConnector does not support "
                "sync_load_budget_bytes: its loads always complete asynchronously."
            )
        assert isinstance(self.manager, RaidenOffloadingManager)
        self._raiden_manager: RaidenOffloadingManager = self.manager
        self._fence_pending: dict[int, _ParkedStore] = {}
        # Recurrent snapshot block per (group_idx, boundary), held until the
        # step that computes the rest of the prompt.
        self._deferred_tail_states: dict[str, dict[tuple[int, int], int]] = {}
        self._mixed_partial_tail = _parse_storage_flag(
            spec.extra_config.get(_DCP_PARTIAL_TAIL_KEY),
            default=False,
            what=_DCP_PARTIAL_TAIL_KEY,
        ) and self._enable_mixed_partial_tail(
            vllm_config.parallel_config.decode_context_parallel_size
        )

    def _enable_mixed_partial_tail(self, dcp: int) -> bool:
        """Turn on partial tails if the KV cache groups have the DCP shape."""
        groups = self.config.kv_group_configs
        attention_sizes = {
            group.tokens_per_block for group in groups if not group.requires_cow_source
        }
        recurrent_sizes = {
            group.tokens_per_block for group in groups if group.requires_cow_source
        }
        has_eagle_group = any(group.is_eagle_group for group in groups)
        if not (
            not self.config.supports_partial_tail
            and self.config.blocks_per_chunk == 1
            and len(attention_sizes) == 1
            and min(attention_sizes) > self.config.tokens_per_hash
            and bool(self._cow_source_groups)
            and all(
                group.tokens_per_block == self.config.tokens_per_hash
                if group.requires_cow_source
                else group.sliding_window_size_in_chunks is None
                for group in groups
            )
            and not has_eagle_group
        ):
            logger.info(
                "Raiden partial tail offloading stays off despite %s: it needs "
                "blocks_per_chunk=1, one full-attention block size above "
                "tokens_per_hash, align-mode recurrent groups at tokens_per_hash, "
                "no EAGLE group, and no upstream partial tails; got "
                "blocks_per_chunk=%d, attention blocks=%s, recurrent blocks=%s, "
                "tokens_per_hash=%d, EAGLE group=%s, upstream partial tails=%s, "
                "DCP=%d",
                _DCP_PARTIAL_TAIL_KEY,
                self.config.blocks_per_chunk,
                sorted(attention_sizes),
                sorted(recurrent_sizes),
                self.config.tokens_per_hash,
                has_eagle_group,
                self.config.supports_partial_tail,
                dcp,
            )
            return False
        # The base class reads the tail block size only where partial tails
        # are supported, so both are set together.
        self.config = self.config._replace(supports_partial_tail=True)
        self._partial_tail_block_size = min(attention_sizes)
        logger.info(
            "Raiden partial tail offloading enabled: attention block=%d tokens, "
            "recurrent state/hash=%d tokens, DCP=%d",
            self._partial_tail_block_size,
            self.config.tokens_per_hash,
            dcp,
        )
        return True

    def _lookup(
        self,
        req_status: RequestOffloadState,
        max_num_new_tokens: int | None = None,
    ) -> int | None:
        if not self._mixed_partial_tail:
            return super()._lookup(req_status, max_num_new_tokens)
        req_status.partial_tail_boundary = None
        req = req_status.req
        req_context = req_status.req_context
        block_size = self._partial_tail_block_size
        tokens_per_hash = self.config.tokens_per_hash
        local_tokens = req_status.num_locally_computed_tokens
        # A load fills attention blocks from the start of one.
        if local_tokens % block_size:
            return 0
        complete_hit = self._lookup_complete_chunks(req_status, max_num_new_tokens)
        complete_boundary = local_tokens + (complete_hit or 0)
        attention_boundary = round_down(complete_boundary, block_size)
        pending = complete_hit is None
        max_tokens = req.num_prompt_tokens - 1
        if max_num_new_tokens is not None:
            max_tokens = min(max_tokens, local_tokens + max_num_new_tokens)
        if req_status.max_load_tokens is not None:
            max_tokens = min(max_tokens, local_tokens + req_status.max_load_tokens)
        attention_groups = [
            group
            for group in self.config.kv_group_configs
            if not group.requires_cow_source
        ]
        # An earlier recurrent checkpoint may be evicted while the attention
        # prefix and a later tail checkpoint are still available. At a block
        # boundary, an attention group's boundary key is its chunk key.
        while attention_boundary + block_size <= max_tokens:
            boundary = attention_boundary + block_size
            results = [
                self.manager.lookup(
                    self._make_boundary_key(
                        req, group.group_idx, boundary, req_context
                    ),
                    req_context,
                )
                for group in attention_groups
            ]
            pending |= any(
                r in (LookupResult.HIT_PENDING, LookupResult.RETRY) for r in results
            )
            if not all(r is LookupResult.HIT for r in results):
                break
            attention_boundary = boundary

        max_boundary = round_down(
            min(max_tokens, attention_boundary + block_size - 1), tokens_per_hash
        )
        for boundary in range(max_boundary, complete_boundary, -tokens_per_hash):
            hit, boundary_pending = self._lookup_partial_tail(
                req_status, boundary, False
            )
            pending |= boundary_pending
            if not hit:
                continue
            if self._chunks_being_loaded:
                load_keys = {
                    self._make_boundary_key(req, group.group_idx, boundary, req_context)
                    for group in self.config.kv_group_configs
                }
                for group in attention_groups:
                    group_state = req_status.group_states[group.group_idx]
                    start = local_tokens // group.tokens_per_chunk
                    end = (boundary - 1) // group.tokens_per_chunk
                    load_keys.update(group_state.offload_keys[start:end])
                if not self._chunks_being_loaded.isdisjoint(load_keys):
                    logger.debug(
                        "Raiden partial tail delayed for in-flight load: "
                        "request=%s boundary=%d",
                        req.request_id,
                        boundary,
                    )
                    return None
            req_status.partial_tail_boundary = boundary
            logger.debug(
                "Raiden partial tail hit: request=%s local=%d boundary=%d",
                req.request_id,
                local_tokens,
                boundary,
            )
            return boundary - local_tokens
        return None if pending and not complete_hit else complete_hit

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        req_status = self._req_status.get(request.request_id)
        # The base method clears partial_tail_boundary, so read it first.
        loaded_tail = (
            self._mixed_partial_tail
            and req_status is not None
            and req_status.partial_tail_boundary is not None
        )
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        if not loaded_tail or not num_external_tokens:
            return
        assert req_status is not None
        if req_status.offloading_context.policy != OffloadPolicy.CHUNK_LEVEL:
            return
        cached_tokens = req_status.num_locally_computed_tokens + num_external_tokens
        for group, group_state in zip(
            self.config.kv_group_configs, req_status.group_states
        ):
            # The base method rounds up, which counts the loaded tail's
            # attention block as stored. Round down, so the block is still
            # stored under its full-block key once the request fills it.
            group_state.next_stored_chunk_idx = cached_tokens // group.tokens_per_chunk

    def _build_partial_tail_store_jobs(
        self, scheduler_output: SchedulerOutput
    ) -> dict[int, TransferJob]:
        if not self._mixed_partial_tail:
            return super()._build_partial_tail_store_jobs(scheduler_output)

        discarded = set(scheduler_output.preempted_req_ids or ()) | set(
            scheduler_output.finished_req_ids or ()
        )
        allocated = self._current_batch_allocated_block_ids
        for req_id, entries in list(self._deferred_tail_states.items()):
            if req_id in discarded or req_id not in self._req_status:
                del self._deferred_tail_states[req_id]
                continue
            self._deferred_tail_states[req_id] = {
                key: block_id
                for key, block_id in entries.items()
                if block_id not in allocated
            }

        block_state = scheduler_output.kv_connector_block_state
        handoffs = block_state.boundary_state_offloads if block_state else {}
        aligned: dict[str, list[tuple[int, int, int]]] = {}
        for req_id, handoff_entries in handoffs.items():
            if req_id in discarded or req_id not in self._req_status:
                continue
            for group_idx, block_id, boundary in handoff_entries:
                if boundary % self._partial_tail_block_size == 0:
                    aligned.setdefault(req_id, []).append(
                        (group_idx, block_id, boundary)
                    )
                elif group_idx in self._cow_source_groups and block_id:
                    self._deferred_tail_states.setdefault(req_id, {})[
                        group_idx, boundary
                    ] = block_id

        jobs = self._build_aligned_boundary_store_jobs(aligned)
        for req_id, scheduled in scheduler_output.num_scheduled_tokens.items():
            if req_id in discarded or req_id not in self._req_status:
                continue
            req_status = self._req_status[req_id]
            req = req_status.req
            boundary = req.num_prompt_tokens
            if (
                not scheduled
                or req.sampling_params is None
                or req.sampling_params.max_tokens != 1
                or req.num_computed_tokens + scheduled != boundary
                or boundary % self.config.tokens_per_hash
                or boundary % self._partial_tail_block_size == 0
            ):
                continue
            # Sparse retention may omit this final prompt state. With
            # max_tokens=1, no later decode step can overwrite it.
            deferred = self._deferred_tail_states.setdefault(req_id, {})
            for group_idx in self._cow_source_groups:
                group = self.config.kv_group_configs[group_idx]
                block_ids = req_status.group_states[group_idx].block_ids
                block_idx = boundary // group.tokens_per_block - 1
                if block_idx < len(block_ids) and block_ids[block_idx]:
                    deferred.setdefault((group_idx, boundary), block_ids[block_idx])

        for req_id, deferred in list(self._deferred_tail_states.items()):
            req_status = self._req_status[req_id]
            req = req_status.req
            scheduled = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            # Wait until the prompt is fully computed: the tail's attention
            # block may be written by a later step than its state hand-off,
            # and the save fence of the step that finishes the prompt covers
            # both writes.
            if not scheduled or req.num_computed_tokens + scheduled < req.num_tokens:
                continue
            del self._deferred_tail_states[req_id]
            max_boundary = self._calc_num_offloadable_tokens(req_status, req.num_tokens)
            for boundary in sorted({boundary for _, boundary in deferred}):
                if boundary <= 0 or boundary > max_boundary:
                    continue
                if boundary % self.config.tokens_per_hash:
                    continue
                sources = []
                for group in self.config.kv_group_configs:
                    if group.requires_cow_source:
                        block_id = deferred.get((group.group_idx, boundary), 0)
                    else:
                        block_ids = req_status.group_states[group.group_idx].block_ids
                        block_idx = boundary // group.tokens_per_block
                        block_id = (
                            block_ids[block_idx] if block_idx < len(block_ids) else 0
                        )
                    if not block_id:
                        break
                    sources.append(block_id)
                else:
                    jobs.update(
                        self._build_tail_store_job(req_status, boundary, sources)
                    )
        return jobs

    def _build_tail_store_job(
        self, req_status: RequestOffloadState, boundary: int, source_blocks: list[int]
    ) -> dict[int, TransferJob]:
        """Store each group's block at `boundary` under its boundary key."""
        req = req_status.req
        groups = self.config.kv_group_configs
        keys = [
            self._make_boundary_key(
                req, group.group_idx, boundary, req_status.req_context
            )
            for group in groups
        ]
        output = self.manager.prepare_store(keys, req_status.req_context)
        if output is None:
            self._connector_stats.increase_counter(
                _ConnectorMetricName.ALLOCATION_FAILURE
            )
            return {}
        if not output.keys_to_store:
            return {}

        sources = dict(zip(keys, zip(groups, source_blocks)))
        group_sizes = [0] * len(groups)
        block_indices = [0] * len(groups)
        accepted_blocks = []
        for key in output.keys_to_store:
            group, block_id = sources[key]
            group_sizes[group.group_idx] = 1
            block_indices[group.group_idx] = (boundary - 1) // group.tokens_per_block
            accepted_blocks.append(block_id)
            self._events_tracker.record_partial_store(req, group, boundary, key)

        job_id = self._generate_job_id()
        req_status.transfer_jobs.add(job_id)
        for block_id in accepted_blocks:
            self._block_id_to_pending_jobs.setdefault(block_id, set()).add(job_id)
        self._jobs[job_id] = TransferJobStatus(
            req_id=req.request_id,
            pending_count=self.config.num_workers,
            keys=set(output.keys_to_store),
            is_store=True,
            fenced_block_ids=accepted_blocks,
        )
        logger.debug(
            "Raiden partial tail store: request=%s boundary=%d "
            "groups=%s blocks=%s job=%d",
            req.request_id,
            boundary,
            group_sizes,
            accepted_blocks,
            job_id,
        )
        return {
            job_id: TransferJob(
                req_id=req.request_id,
                src_spec=GPULoadStoreSpec(
                    accepted_blocks,
                    group_sizes=group_sizes,
                    block_indices=block_indices,
                ),
                dst_spec=output.store_spec,
            )
        }

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
        drained = manager.drain_jobs(launched_flush_ids) if launched_flush_ids else []

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

        Upstream asks for "LBHNC" ("BLHNC" with HiSparse), a layout the TPU
        KV cache does not use; these pages are opaque bytes to this connector.
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
