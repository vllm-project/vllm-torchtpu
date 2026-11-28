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

import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.utils.math_utils import round_down
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    OffloadKey,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.distributed.kv_transfer.connector_options import (
    RaidenOffloadingOptions,
    read_options,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.offload.raiden_store import (
    AdmissionOp,
    RaidenLoadStoreSpec,
    RaidenOffloadingManager,
    TPURaidenStoreOffloadingSpec,
)

if TYPE_CHECKING:
    from vllm.distributed.kv_events import KVCacheEvent

logger = init_logger(__name__)

# How long poll_and_complete_transfers waits for a pending host-tier load when
# a request is blocked on one (at least 5 ms when nothing else is running).
_LOAD_WAIT_MS = 1.5
# How long estimate_external_hit_tokens reuses a request's answer while the
# request's block hashes and local hit stay the same. The prefix-aware waiting
# sort asks for every request in its window on every step.
_EXTERNAL_HIT_ESTIMATE_TTL_S = 0.25


def _local_hit_tokens(request: Request) -> int:
    """The local HBM hit that the pipeline chunk scheduler records on a
    request (``_tpu_local_hit_tokens``), with the sub-block tail it strips
    before asking the connector; 0 for a request it did not record."""
    return getattr(request, "_tpu_local_hit_tokens", 0)


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
    """

    # Option host_tier_recency (RaidenOffloadingOptions); off unless configured.
    _host_tier_recency: bool = False

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
        # Set by the first poll_and_complete_transfers: from then on the
        # scheduler polls at the start of every schedule(), and
        # build_connector_meta hands the transfers it finds to the next poll
        # (_unreported_jobs, _unreported_store_ids) instead of the workers.
        self._scheduler_polls = False
        self._unreported_jobs: list[Any] = []
        self._unreported_store_ids: list[int] = []
        # Failed loads a poll found, for the worker echo of the next
        # build_connector_meta: vLLM takes load failures only from a step's
        # output (failed_recving).
        self._pending_failed_jobs: list[Any] = []
        # request id -> (number of block hashes, chunk-aligned local hit, time,
        # host-tier tokens) of the last estimate_external_hit_tokens answer.
        self._external_hit_estimates: dict[str, tuple[int, int, float, int]] = {}
        options = read_options(RaidenOffloadingOptions, spec.extra_config)
        self._max_async_load_reqs = options.max_async_loads
        self._host_tier_recency = options.host_tier_recency

    def _collect_finished(
        self,
        finished_jobs: Iterable[Any],
        completed_jobs: dict[int, int],
        finished_recving: set[str],
    ) -> None:
        """Adds finished Raiden jobs to a poll's completed job counts and
        finished requests, as the worker echo would report them: every
        worker's count for a job, and a load's request once none of its
        other loads is pending. A failed job waits for the worker echo."""
        num_workers = self.config.num_workers
        load_req_ids: list[str] = []
        for finished_job in finished_jobs:
            if finished_job.failed_device_block_ids:
                self._pending_failed_jobs.append(finished_job)
                continue
            if finished_job.job_id in self._jobs:
                completed_jobs[finished_job.job_id] = num_workers
            if finished_job.op == AdmissionOp.LOAD and finished_job.req_id is not None:
                load_req_ids.append(finished_job.req_id)
        for req_id in load_req_ids:
            req_status = self._req_status.get(req_id)
            if req_status is None or not any(
                self._load_pending(jid, completed_jobs)
                for jid in req_status.transfer_jobs
            ):
                finished_recving.add(req_id)

    def _load_pending(self, job_id: int, completed_jobs: dict[int, int]) -> bool:
        job = self._jobs.get(job_id)
        return job is not None and not job.is_store and job_id not in completed_jobs

    def _load_awaited(self, completed_jobs: dict[int, int]) -> bool:
        """Whether a request waits for a host-tier load still pending."""
        for job_id, job in self._jobs.items():
            if not self._load_pending(job_id, completed_jobs):
                continue
            req_status = self._req_status.get(job.req_id)
            if (
                req_status is not None
                and req_status.req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
            ):
                return True
        return False

    def poll_and_complete_transfers(self, idle: bool) -> KVConnectorOutput | None:
        """Host-tier transfers that finished, for the scheduler to apply at
        the start of ``schedule()``, so a request whose load landed is
        scheduled in that step instead of after the worker echo.

        Returns them as the worker echo reports them: a ``KVConnectorOutput``
        whose ``finished_recving`` names each request whose loads have all
        finished and whose worker metadata counts the completed jobs, or None
        when nothing finished. The scheduler applies it with
        ``_update_from_kv_xfer_finished``, the method that applies a step's
        output; this connector neither keeps nor calls the scheduler. A
        failed load still goes through the worker echo, which reports it in
        ``failed_recving``.

        If no request's loads have finished and a request waits for one, the
        poll waits up to ``_LOAD_WAIT_MS`` for it (at least 5 ms when the
        scheduler is ``idle``, with no request running).

        The pipeline chunk scheduler's prefix-aware admission calls it at the
        start of each step."""
        self._scheduler_polls = True
        manager = self._raiden_manager
        completed_jobs = {
            job_id: self.config.num_workers
            for job_id in self._unreported_store_ids
            if job_id in self._jobs
        }
        finished_recving: set[str] = set()
        self._collect_finished(
            self._unreported_jobs + list(manager.poll_finished_jobs()),
            completed_jobs,
            finished_recving,
        )
        self._unreported_jobs = []
        self._unreported_store_ids = []
        if not finished_recving and self._load_awaited(completed_jobs):
            wait_ms = max(_LOAD_WAIT_MS, 5.0) if idle else _LOAD_WAIT_MS
            deadline_ns = time.perf_counter_ns() + int(wait_ms * 1_000_000.0)
            try:
                while time.perf_counter_ns() < deadline_ns:
                    time.sleep(0.0001)
                    polled = manager.poll_finished_jobs()
                    if polled:
                        self._collect_finished(polled, completed_jobs, finished_recving)
                        if finished_recving or not any(
                            self._load_pending(jid, completed_jobs)
                            for jid in self._jobs
                        ):
                            break
                    elif not manager.has_pending_work():
                        break
            except Exception:
                # The transfers collected so far are taken off the manager
                # already: return them rather than lose them with the wait.
                logger.exception(
                    "TPURaidenOffloadingConnector: waiting for a host-tier load "
                    "failed; returning the %d finished jobs collected before it",
                    len(completed_jobs),
                )
        if not completed_jobs and not finished_recving:
            return None
        return KVConnectorOutput(
            finished_recving=finished_recving,
            kv_connector_worker_meta=RaidenOffloadingWorkerMetadata(
                # pyrefly: ignore[unexpected-keyword]
                completed_jobs=completed_jobs,
                fenced_jobs={},
            ),
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
        polled = manager.poll_finished_jobs()

        if self._scheduler_polls:
            # The next poll returns these to the scheduler, at the start of
            # the next schedule() and before it decides anything; failed
            # jobs go through the worker echo.
            for finished_job in list(drained) + polled:
                if finished_job.failed_device_block_ids:
                    self._pending_failed_jobs.append(finished_job)
                else:
                    self._unreported_jobs.append(finished_job)
            self._unreported_store_ids.extend(finished_store_ids)
            finished_store_ids = []
            for failed_job in self._pending_failed_jobs:
                if failed_job.op == AdmissionOp.STORE:
                    finished_store_ids.append(failed_job.job_id)
                else:
                    assert failed_job.req_id is not None
                    finished_loads[failed_job.job_id] = failed_job.req_id
                    failed_load_req_ids.add(failed_job.req_id)
            self._pending_failed_jobs.clear()
        else:
            # Collect completed/drained jobs for the worker echo channel.
            for finished_job in list(drained) + polled:
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

    def _lookup_partial_tail(
        self, req_status: Any, boundary: int, bridge_pending: bool
    ) -> tuple[bool, bool]:
        """Upstream's partial-tail lookup, which skips under host_tier_recency
        a boundary at or below the local HBM hit inside the block that starts
        at the locally computed tokens: HBM already holds those tokens, so a
        hit there loads nothing it lacks. ``_local_hit_tokens`` keeps the
        sub-block tail the scheduler strips before asking the connector, so
        the hit ends inside that block."""
        if self._host_tier_recency:
            local_tokens = req_status.num_locally_computed_tokens
            in_local_block = boundary < local_tokens + self._partial_tail_block_size
            if in_local_block and boundary <= _local_hit_tokens(req_status.req):
                return False, bridge_pending
        return super()._lookup_partial_tail(req_status, boundary, bridge_pending)

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
        max_num_new_tokens: int | None = None,
    ) -> tuple[int | None, bool]:
        recency = self._host_tier_recency
        if (
            recency
            and num_computed_tokens > 0
            and self.config.kv_group_configs
            and num_computed_tokens % self.config.kv_group_configs[0].tokens_per_chunk
            != 0
        ):
            req_status = self._req_status.get(request.request_id)
            if req_status is None:
                self.on_new_request(request)
                req_status = self._req_status.get(request.request_id)
            if req_status is not None:
                req_status.num_locally_computed_tokens = num_computed_tokens
                req_status.update_offload_keys()
                req_status.update_num_hit_chunks(num_computed_tokens)
                # This hook can be the step's first to touch the manager, so
                # it opens schedule()'s step region, as upstream's hook does
                # before its lookup.
                self._acquire_step()
                self._touch(req_status)
            return 0, False
        max_loads = self._max_async_load_reqs
        if max_loads > 0:
            active_loads = sum(1 for js in self._jobs.values() if not js.is_store)
            if active_loads >= max_loads:
                req_status = self._req_status.get(request.request_id)
                if req_status is None:
                    self.on_new_request(request)
                    req_status = self._req_status.get(request.request_id)
                if req_status is not None and not req_status.transfer_jobs:
                    req_status.update_offload_keys()
                    orig_local = req_status.num_locally_computed_tokens
                    orig_partial = req_status.partial_tail_boundary
                    self._acquire_step()
                    try:
                        probe_hit = self._quiet_lookup(req_status, num_computed_tokens)
                    finally:
                        req_status.num_locally_computed_tokens = orig_local
                        req_status.partial_tail_boundary = orig_partial
                    if probe_hit is None or probe_hit > 0:
                        return None, False
        num_hit_tokens, load_async = super().get_num_new_matched_tokens(
            request, num_computed_tokens, max_num_new_tokens
        )
        if not recency:
            return num_hit_tokens, load_async
        req_status = self._req_status.get(request.request_id)
        if req_status is not None and not req_status.transfer_jobs:
            # Upstream leaves cache recency to the manager, and the Raiden
            # store learns it only through touch: refresh this request's keys
            # on every lookup that ran (one deferred for in-flight transfers
            # did not).
            self._touch(req_status)
        if num_hit_tokens == 0:
            local_hit = _local_hit_tokens(request)
            if local_hit > num_computed_tokens and req_status is not None:
                req_status.num_locally_computed_tokens = local_hit
                if (
                    self.config.supports_partial_tail
                    and local_hit % self._partial_tail_block_size != 0
                    and local_hit % self.config.tokens_per_hash == 0
                ):
                    self._touch_partial_tail(request, req_status.req_context, local_hit)
        return num_hit_tokens, load_async

    def update_state_after_alloc(
        self, request: Request, blocks: Any, num_external_tokens: int
    ) -> None:
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        if not self._host_tier_recency:
            return
        req_status = self._req_status.get(request.request_id)
        if req_status is not None:
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                if group_config.sliding_window_size_in_chunks is None:
                    # Keep next_stored_chunk_idx = 0 for FullAttention so
                    # _build_store_jobs lets prepare_store probe 0..num_chunks
                    # (skipping resident HOST/HOST_AND_HBM blocks in memory and
                    # re-storing any HBM-resident prefix chunks displaced from
                    # the host LRU).
                    group_state.next_stored_chunk_idx = 0

    def _build_store_jobs(
        self, scheduler_output: SchedulerOutput
    ) -> dict[int, TransferJob]:
        store_jobs = super()._build_store_jobs(scheduler_output)
        if not self._host_tier_recency:
            return store_jobs
        # Every request that offloads this step refreshes its keys, as each
        # lookup does in get_num_new_matched_tokens.
        for req_id in dict.fromkeys(job.req_id for job in store_jobs.values()):
            req_status = self._req_status.get(req_id)
            if req_status is not None:
                self._touch(req_status)
        return store_jobs

    def _touch(self, req_status: Any) -> None:
        """Refreshes the Raiden host LRU for one request's offloaded keys.

        Keys touched later end nearer the MRU end, so the order below decides
        what an active conversation loses first under host pressure.
        """
        local_tokens = req_status.num_locally_computed_tokens
        local_hit = _local_hit_tokens(req_status.req)
        if local_hit > local_tokens:
            local_tokens = local_hit
        shared_boundary = int(getattr(req_status.req, "shared_prefix_boundary", 0) or 0)
        # 1. Touch ALL FullAttention chunks in reverse order (tail -> chunk_0)
        #    so the entire active conversation stays ahead of inactive
        #    conversations AND chunk_0 (the prefix root) is the most recently
        #    used entry and can never be evicted before the tail.
        for group_config, group_state in zip(
            self.config.kv_group_configs, req_status.group_states
        ):
            if group_config.sliding_window_size_in_chunks is not None:
                continue
            if group_state.offload_keys:
                self.manager.touch(
                    list(reversed(group_state.offload_keys)),
                    req_status.req_context,
                )

        # 2. Touch Mamba / sliding-window boundary chunks AFTER FullAttention so
        #    active recurrent boundary states stay at the MRU head of the host cache.
        for group_config, group_state in zip(
            self.config.kv_group_configs, req_status.group_states
        ):
            if group_config.sliding_window_size_in_chunks is None:
                continue
            chunks_to_skip = max(
                0,
                group_state.num_hit_chunks - group_config.sliding_window_size_in_chunks,
            )
            keys_to_touch = list(group_state.offload_keys[chunks_to_skip:])
            if shared_boundary >= group_config.tokens_per_chunk:
                fork_idx = (shared_boundary // group_config.tokens_per_chunk) - 1
                if 0 <= fork_idx < chunks_to_skip and fork_idx < len(
                    group_state.offload_keys
                ):
                    keys_to_touch.append(group_state.offload_keys[fork_idx])
            if keys_to_touch:
                self.manager.touch(keys_to_touch, req_status.req_context)

        # 3. Touch partial-tail boundary keys LAST (both CPU-matched partial
        #    tails and HBM-matched partial tails).
        boundaries_to_touch: list[int] = []
        if req_status.partial_tail_boundary is not None:
            boundaries_to_touch.append(int(req_status.partial_tail_boundary))
        if (
            self.config.supports_partial_tail
            and local_tokens > 0
            and local_tokens % self._partial_tail_block_size != 0
            and local_tokens % self.config.tokens_per_hash == 0
            and local_tokens not in boundaries_to_touch
        ):
            boundaries_to_touch.append(local_tokens)
        for boundary in boundaries_to_touch:
            self._touch_partial_tail(req_status.req, req_status.req_context, boundary)

    def _touch_partial_tail(
        self, request: Request, req_context: Any, boundary: int
    ) -> None:
        """Refreshes the keys of the partial tail at ``boundary``: one per
        group, at the token position upstream keys that group's partial tail
        at in store, load and lookup (``_partial_tail_key_tokens``). Nothing
        when the request has no block hash at a key's position yet."""
        positions = [
            (group, self._partial_tail_key_tokens(group, boundary))
            for group in self.config.kv_group_configs
        ]
        if any(
            tokens // self.config.tokens_per_hash - 1 >= len(request.block_hashes)
            for _, tokens in positions
        ):
            return
        self.manager.touch(
            tuple(
                self._make_boundary_key(request, group.group_idx, tokens, req_context)
                for group, tokens in positions
            ),
            req_context,
        )

    def update_connector_output(self, connector_output: KVConnectorOutput):
        meta = connector_output.kv_connector_worker_meta
        if isinstance(meta, RaidenOffloadingWorkerMetadata) and meta.fenced_jobs:
            ready_jobs: list[tuple[int, _ParkedStore]] = []
            for job_id, count in meta.fenced_jobs.items():
                parked = self._fence_pending.get(job_id)
                if parked is None:
                    # Cancelled while the acks were in flight.
                    continue
                parked.acks += count
                assert parked.acks <= self.config.num_workers
                if parked.acks == self.config.num_workers:
                    del self._fence_pending[job_id]
                    ready_jobs.append((job_id, parked))
            if self._host_tier_recency:
                ready_jobs.sort(key=lambda item: (-len(item[1].keys), item[0]))
            for job_id, parked in ready_jobs:
                self._raiden_manager.submit_store_job(
                    job_id, parked.keys, parked.device_block_ids
                )
        super().update_connector_output(connector_output)

    def _quiet_lookup(self, req_status: Any, base_tokens: int) -> int | None:
        """Probe ``self._lookup(req_status)`` starting from ``base_tokens``
        without emitting duplicate lookup events."""
        req_status.num_locally_computed_tokens = base_tokens
        # vLLM's OffloadingConnectorScheduler.__init__ builds the tracker.
        tracker = self._events_tracker
        orig_record = tracker.record_lookup
        orig_partial = tracker.record_partial_lookup
        tracker.record_lookup = lambda *a, **kw: None
        tracker.record_partial_lookup = lambda *a, **kw: None
        try:
            return self._lookup(req_status)
        finally:
            tracker.record_lookup = orig_record
            tracker.record_partial_lookup = orig_partial

    def estimate_external_hit_tokens(
        self, request: Request, num_local_tokens: int
    ) -> int:
        """Host-tier tokens past ``num_local_tokens`` that a lookup would load.

        The pipeline chunk scheduler's ``_external_warmth`` asks it for its
        waiting sort, its warmest-first promotion and its pending-work
        estimate. Without an
        answer, a request whose prefix is in the host tier but no longer in HBM
        is ranked as a cold prompt and waits behind cheaper work while its
        host-tier entries age towards eviction.

        The probe changes nothing: lookups take no pin and refresh no recency,
        no lookup event is emitted, and the lookup state the probe moves is
        restored. It answers 0 for a request that skips the prefix cache or has
        less than a hash block past its local hit. An answer is reused for
        ``_EXTERNAL_HIT_ESTIMATE_TTL_S`` while the request's block hashes and
        chunk-aligned local hit are unchanged, so a request whose load started
        within that time keeps its last answer. Otherwise a request the
        connector does not track, or one with a transfer in flight, answers 0.
        """
        if request.skip_reading_prefix_cache:
            return 0
        if not self.config.kv_group_configs:
            return 0
        tokens_per_chunk = self.config.kv_group_configs[0].tokens_per_chunk
        if tokens_per_chunk <= 0:
            return 0
        aligned_local = round_down(max(0, int(num_local_tokens)), tokens_per_chunk)
        if request.num_tokens - aligned_local < self.config.tokens_per_hash:
            return 0
        estimates = self._external_hit_estimates
        req_id = request.request_id
        num_hashes = len(request.block_hashes)
        now = time.monotonic()
        cached = estimates.get(req_id)
        if (
            cached is not None
            and cached[0] == num_hashes
            and cached[1] == aligned_local
            and now - cached[2] < _EXTERNAL_HIT_ESTIMATE_TTL_S
        ):
            return max(0, aligned_local + cached[3] - num_local_tokens)
        req_status = self._req_status.get(req_id)
        if req_status is None or req_status.transfer_jobs:
            return 0
        req_status.update_offload_keys()
        orig_local = req_status.num_locally_computed_tokens
        orig_partial = req_status.partial_tail_boundary
        try:
            # Callers outside schedule()'s step region hold the manager for the
            # probe; inside it the region already holds it.
            if self._step_region is None:
                with self._manager_locked():
                    ext = self._quiet_lookup(req_status, aligned_local)
            else:
                ext = self._quiet_lookup(req_status, aligned_local)
        finally:
            req_status.num_locally_computed_tokens = orig_local
            req_status.partial_tail_boundary = orig_partial
        ext_tokens = ext if ext is not None and ext > 0 else 0
        if len(estimates) > 2 * len(self._req_status) + 64:
            # Drop the answers of requests the connector no longer tracks.
            for stale in [r for r in estimates if r not in self._req_status]:
                del estimates[stale]
        estimates[req_id] = (num_hashes, aligned_local, now, ext_tokens)
        return max(0, aligned_local + ext_tokens - num_local_tokens)

    def resolve_divergent_local_hit_tokens(
        self,
        request: Request,
        num_local: int,
        num_reconciled: int,
        block_size: int,
    ) -> int | None:
        """Return the block-aligned local FullAttention prefix length to retain
        in HBM when HBM has a divergent hybrid hit (FullAttention > Mamba), or
        None if Raiden CPU cannot advance past ``num_reconciled``.

        The pipeline chunk scheduler's local-hit lookup
        (``TpuPipelineChunkScheduler._get_local_prefix_cache_hit``) asks this
        when the connector reports divergent local hybrid hits."""
        if request.skip_reading_prefix_cache:
            return None
        req_status = self._req_status.get(request.request_id)
        if req_status is None:
            self.on_new_request(request)
            req_status = self._req_status.get(request.request_id)
        if req_status is None:
            return None
        if req_status.transfer_jobs:
            return (num_local // block_size) * block_size
        # The local-hit lookup can run before the step's first connector hook
        # opens schedule()'s region; outside it the probes hold the manager.
        if self._step_region is None:
            with self._manager_locked():
                return self._resolve_divergent_hit(
                    req_status, num_local, num_reconciled, block_size
                )
        return self._resolve_divergent_hit(
            req_status, num_local, num_reconciled, block_size
        )

    def _resolve_divergent_hit(
        self,
        req_status: Any,
        num_local: int,
        num_reconciled: int,
        block_size: int,
    ) -> int | None:
        req_status.update_offload_keys()
        fa_aligned = (num_local // block_size) * block_size
        reconciled_aligned = (num_reconciled // block_size) * block_size

        # 1. Check if Raiden CPU has tokens strictly above fa_aligned
        #    (e.g. a partial tail at fa_aligned + 128 or additional chunks).
        ext_above = self._quiet_lookup(req_status, fa_aligned)
        if ext_above is None:
            return fa_aligned
        if ext_above > 0 and fa_aligned + ext_above > num_reconciled:
            return fa_aligned

        # 2. Common case: FullAttention in HBM covers all full blocks of the
        #    previous turn (or its partial tail), and Raiden CPU holds the
        #    Mamba boundary in that final block [fa_aligned - block_size, fa_aligned].
        #    Back off by 1 block so get_num_new_matched_tokens loads only the
        #    final block (1 FA block + 1 Mamba block) while keeping the other
        #    (fa_aligned // block_size - 1) FA blocks in HBM.
        one_back = fa_aligned - block_size
        if one_back >= reconciled_aligned and one_back >= 0:
            ext_last = self._quiet_lookup(req_status, one_back)
            if ext_last is None:
                return one_back
            if ext_last > 0 and one_back + ext_last > num_reconciled:
                ext_end = one_back + ext_last
                return min(fa_aligned, ((ext_end - 1) // block_size) * block_size)

        # 3. General case: FullAttention in HBM extends past the latest Mamba
        #    boundary stored in Raiden CPU (e.g. shared prefix from a longer
        #    request). Scan downward from one_back - block_size to
        #    reconciled_aligned to find the highest chunk where Raiden CPU can
        #    supply both the Mamba boundary and its corresponding FA chunk.
        min_chunk = max(0, reconciled_aligned // block_size)
        max_chunk = (one_back // block_size) - 1
        for chunk_idx in range(max_chunk, min_chunk - 1, -1):
            base = chunk_idx * block_size
            ext_hit = self._quiet_lookup(req_status, base)
            if ext_hit is None:
                return base
            if ext_hit > 0 and base + ext_hit > num_reconciled:
                ext_end = base + ext_hit
                return min(fa_aligned, ((ext_end - 1) // block_size) * block_size)

        return None


class TPURaidenOffloadingConnector(OffloadingConnector):
    """Top-level facade connector for the store-backed (V2) raiden KV offload path."""

    def resolve_divergent_local_hit_tokens(
        self,
        request: Request,
        num_local: int,
        num_reconciled: int,
        block_size: int,
    ) -> int | None:
        if self.connector_scheduler is None:
            return None
        return self.connector_scheduler.resolve_divergent_local_hit_tokens(
            request, num_local, num_reconciled, block_size
        )

    def estimate_external_hit_tokens(
        self, request: Request, num_local_tokens: int
    ) -> int:
        """The scheduler side's host-tier hit estimate; 0 on a worker."""
        scheduler = self.connector_scheduler
        if not isinstance(scheduler, TPURaidenOffloadingScheduler):
            return 0
        return scheduler.estimate_external_hit_tokens(request, num_local_tokens)

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

    def poll_and_complete_transfers(self, idle: bool) -> KVConnectorOutput | None:
        """The scheduler side's finished transfers; None on a worker."""
        if self.connector_scheduler is None:
            return None
        return self.connector_scheduler.poll_and_complete_transfers(idle)

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
