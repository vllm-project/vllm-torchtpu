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
"""Connector wiring for the store-backed (V2) raiden KV offload path.

`TPURaidenOffloadingConnector` subclasses vLLM's `OffloadingConnector`;
select it with:

    --kv-transfer-config '{
        "kv_connector": "TPURaidenOffloadingConnector",
        "kv_connector_module_path": "vllm_torchtpu.offload.raiden_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"cpu_bytes_to_use": ...}
    }'

Control flow vs the stock connector: transfers originate at the scheduler
(the manager's `KVCacheStore`), so workers receive **no** load/store jobs.
Two things still ride the scheduler<->worker metadata channel:

- **Save fence.** A store job's `save()` may only read HBM after the forward
  that wrote the blocks. Newly created store jobs ship as `fence_job_ids`;
  each rank runs a post-forward `_tpu_sync(wait=True)` and acks via
  `fenced_jobs` on the worker metadata. Only after every rank acked does the
  scheduler admit (`insert_and_lock`) and launch (`save`) the job.

- **Completion echoes.** The stock completion machinery (per-job pending
  counts, block-free fencing, WAITING_FOR_REMOTE_KVS release) is count/ID
  driven from worker reports. Terminal jobs discovered by the manager's
  finalizer ship to the workers, which echo them back as `completed_jobs` /
  `finished_recving` — so that machinery runs unchanged, at the cost of one
  extra step of latency on completion signals. Failed loads additionally
  report their destination GPU blocks through
  `get_block_ids_with_load_errors`, which the scheduler core already handles
  by invalidating and recomputing those blocks.

`jobs_to_flush` (blocks about to be reused) is handled scheduler-side:
fence-pending stores are cancelled, launched jobs are drained by blocking on
the store polls before this step's metadata ships.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata, OffloadingWorkerMetadata)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import \
    build_offloading_config
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import \
    OffloadingConnectorScheduler
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import \
    OffloadingConnectorWorker
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import \
    OffloadingConnector
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import GPULoadStoreSpec, OffloadKey
from vllm.v1.outputs import KVConnectorOutput

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.offload.raiden_store import (AdmissionOp,
                                                RaidenLoadStoreSpec,
                                                RaidenOffloadingManager,
                                                RaidenStoreOffloadingWorker,
                                                TPURaidenStoreOffloadingSpec)

logger = init_logger(__name__)


@dataclass
class RaidenOffloadingConnectorMetadata(OffloadingConnectorMetadata):
    """Scheduler -> worker metadata. load_jobs/store_jobs are always empty
    (transfers are controller-driven); the extra fields carry the fence
    requests and the completion echoes."""
    # Store jobs newly parked this step: every rank must fence (post-forward
    # _tpu_sync) and ack before the scheduler launches the save.
    fence_job_ids: set[int] = field(default_factory=set)
    # Terminal store jobs to echo back as completed_jobs.
    finished_store_job_ids: list[int] = field(default_factory=list)
    # Terminal load jobs to echo back: job_id -> req_id (reported both as
    # completed_jobs and finished_recving, on success AND failure).
    finished_load_jobs: dict[int, str] = field(default_factory=dict)
    # Destination GPU block IDs of failed loads, reported through
    # get_block_ids_with_load_errors.
    failed_load_gpu_block_ids: list[int] = field(default_factory=list)


@dataclass
class RaidenOffloadingWorkerMetadata(OffloadingWorkerMetadata):
    """Worker -> scheduler metadata: stock completed-job echoes plus the
    per-rank fence acks ({job_id: 1}, summed across ranks)."""
    fenced_jobs: dict[int, int] = field(default_factory=dict)

    def aggregate(self, other):
        assert isinstance(other, RaidenOffloadingWorkerMetadata)
        merged_completed = dict(self.completed_jobs)
        for job_id, v in other.completed_jobs.items():
            merged_completed[job_id] = merged_completed.get(job_id, 0) + v
        merged_fenced = dict(self.fenced_jobs)
        for job_id, v in other.fenced_jobs.items():
            merged_fenced[job_id] = merged_fenced.get(job_id, 0) + v
        return RaidenOffloadingWorkerMetadata(
            completed_jobs=merged_completed,
            transfer_stats=self.transfer_stats.aggregate(other.transfer_stats),
            fenced_jobs=merged_fenced,
        )


@dataclass
class _ParkedStore:
    """A store job awaiting fence acks from every rank."""
    keys: list[OffloadKey]
    gpu_block_ids: list[int]
    acks: int = 0


class TPURaidenOffloadingScheduler(OffloadingConnectorScheduler):
    """Scheduler that strips jobs from the worker metadata and drives the
    store through the manager's submission bridge instead."""

    def __init__(self, spec: TPURaidenStoreOffloadingSpec,
                 vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(spec, vllm_config, kv_cache_config)
        assert isinstance(self.manager, RaidenOffloadingManager)
        self._raiden_manager: RaidenOffloadingManager = self.manager
        self._fence_pending: dict[int, _ParkedStore] = {}

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = super().build_connector_meta(scheduler_output)
        assert isinstance(meta, OffloadingConnectorMetadata)

        manager = self._raiden_manager

        # Park new store jobs until every rank fences. keys (dst_spec, from
        # prepare_store) and GPU block IDs (src_spec) align 1:1 with
        # blocks_per_chunk == 1.
        new_fence_ids: set[int] = set()
        for job_id, job in meta.store_jobs.items():
            src_spec, dst_spec = job.src_spec, job.dst_spec
            assert isinstance(src_spec, GPULoadStoreSpec)
            assert isinstance(dst_spec, RaidenLoadStoreSpec)
            gpu_block_ids = src_spec.block_ids.tolist()
            assert len(gpu_block_ids) == len(
                dst_spec.keys), (len(gpu_block_ids), len(dst_spec.keys))
            self._fence_pending[job_id] = _ParkedStore(
                keys=dst_spec.keys, gpu_block_ids=gpu_block_ids)
            new_fence_ids.add(job_id)

        # Loads are safe by construction (freshly allocated destination
        # blocks; request scheduled only after the poll reports done), so
        # submit them to the store immediately.
        for job_id, job in meta.load_jobs.items():
            src_spec, dst_spec = job.src_spec, job.dst_spec
            assert isinstance(src_spec, RaidenLoadStoreSpec)
            assert isinstance(dst_spec, GPULoadStoreSpec)
            gpu_block_ids = dst_spec.block_ids.tolist()
            assert len(gpu_block_ids) == len(
                src_spec.keys), (len(gpu_block_ids), len(src_spec.keys))
            manager.submit_load_job(job_id,
                                    src_spec.keys,
                                    gpu_block_ids,
                                    job.req_id,
                                    pinned=src_spec.pinned,
                                    displaced_keys=src_spec.displaced_keys)

        # jobs_to_flush: the jobs' source blocks are about to be reused.
        # Fence-pending stores are cancelled (nothing was admitted yet);
        # launched jobs are drained by blocking on the polls, so the DMA
        # finishes before the forward that overwrites the blocks (workers
        # only start it after receiving this step's metadata).
        finished_store_ids: list[int] = []
        finished_loads: dict[int, str] = {}
        failed_load_block_ids: list[int] = []
        flush_ids = meta.jobs_to_flush or set()
        launched_flush_ids: set[int] = set()
        for job_id in flush_ids:
            parked = self._fence_pending.pop(job_id, None)
            if parked is not None:
                manager.on_store_job_cancelled(parked.keys)
                finished_store_ids.append(job_id)
                logger.debug(
                    "[kv-offload-v2] cancelled fence-pending store job %d "
                    "(source blocks reused)", job_id)
            elif manager.is_job_launched(job_id):
                launched_flush_ids.add(job_id)
        drained = (manager.drain_jobs(launched_flush_ids)
                   if launched_flush_ids else [])

        # Collect newly terminal jobs (regular completions + the drain) for
        # the echo channel.
        for fj in list(drained) + manager.poll_finished_jobs():
            if fj.op == AdmissionOp.STORE:
                finished_store_ids.append(fj.job_id)
            else:
                assert fj.req_id is not None
                finished_loads[fj.job_id] = fj.req_id
                failed_load_block_ids.extend(fj.failed_gpu_block_ids)

        # Ship empty job dicts: no per-rank transfer submission exists.
        return RaidenOffloadingConnectorMetadata(
            load_jobs={},
            store_jobs={},
            jobs_to_flush=meta.jobs_to_flush,
            fence_job_ids=new_fence_ids,
            finished_store_job_ids=finished_store_ids,
            finished_load_jobs=finished_loads,
            failed_load_gpu_block_ids=failed_load_block_ids,
        )

    def update_connector_output(self, connector_output: KVConnectorOutput):
        meta = connector_output.kv_connector_worker_meta
        if isinstance(meta, RaidenOffloadingWorkerMetadata) and \
                meta.fenced_jobs:
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
                        job_id, parked.keys, parked.gpu_block_ids)
        super().update_connector_output(connector_output)


class TPURaidenOffloadingConnector(OffloadingConnector):
    """OffloadingConnector variant for the raiden KVCacheStore path."""

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole,
                 kv_cache_config: KVCacheConfig):
        # Deliberately skip OffloadingConnector.__init__: it resolves the
        # spec through the OffloadingSpecFactory, which cannot carry
        # vllm_config / kv_cache_config into the spec.
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)

        # Without this, ControllerServer / WorkerServiceServer are
        # process-wide singletons that silently reuse the first bound port —
        # fatal when the disagg stack runs its own raiden control plane in
        # the same process.
        if os.environ.get("RAIDEN_DISABLE_SINGLETON_WORKER") not in ("1",
                                                                     "true"):
            os.environ["RAIDEN_DISABLE_SINGLETON_WORKER"] = "1"

        offloading_config = build_offloading_config(vllm_config,
                                                    kv_cache_config)
        spec = TPURaidenStoreOffloadingSpec(offloading_config, vllm_config,
                                            kv_cache_config)

        self.connector_scheduler: TPURaidenOffloadingScheduler | None = None
        self.connector_worker: OffloadingConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = TPURaidenOffloadingScheduler(
                spec, vllm_config, kv_cache_config)
        elif role == KVConnectorRole.WORKER:
            # The stock worker wrapper is reused only for its KV cache
            # registration (canonicalized views -> spec.get_worker); the job
            # dicts it iterates are always empty.
            self.connector_worker = OffloadingConnectorWorker(
                spec, kv_cache_config)
            # Per-step fence acks and completion echoes for the next
            # build_connector_worker_meta.
            self._fenced_jobs: dict[int, int] = {}
            self._completed_jobs: dict[int, int] = {}
            self._load_error_block_ids: set[int] = set()

    # -- worker-side hooks -------------------------------------------------

    def _raiden_worker(self) -> RaidenStoreOffloadingWorker:
        assert self.connector_worker is not None
        worker = self.connector_worker.worker
        assert isinstance(worker, RaidenStoreOffloadingWorker)
        return worker

    def start_load_kv(self, forward_context, **kwargs) -> None:
        # Registration is deferred to here (the first per-step hook): the
        # scheduler-side controller is deterministically up by the first
        # step, while register_kv_caches runs strictly before it exists.
        self._raiden_worker().ensure_registered()

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata):
        # Flushes are handled scheduler-side (cancel or drain); there is no
        # per-rank transfer state to flush.
        return

    def get_finished(self,
                     finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        meta = self._get_connector_metadata()
        assert isinstance(meta, RaidenOffloadingConnectorMetadata)

        if meta.fence_job_ids:
            # Post-forward device fence: after this, every KV write of the
            # forward that filled the fenced jobs' blocks is complete, so
            # the controller's save may read them.
            from torch_tpu._internal.sync import synchronize as _tpu_sync
            _tpu_sync(wait=True)
            for job_id in meta.fence_job_ids:
                self._fenced_jobs[job_id] = 1

        finished_recving: set[str] = set()
        for job_id in meta.finished_store_job_ids:
            self._completed_jobs[job_id] = 1
        for job_id, req_id in meta.finished_load_jobs.items():
            self._completed_jobs[job_id] = 1
            finished_recving.add(req_id)
        self._load_error_block_ids.update(meta.failed_load_gpu_block_ids)

        return set(), finished_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        block_ids = self._load_error_block_ids
        self._load_error_block_ids = set()
        return block_ids

    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        if not (self._fenced_jobs or self._completed_jobs):
            return None
        meta = RaidenOffloadingWorkerMetadata(
            completed_jobs=self._completed_jobs,
            fenced_jobs=self._fenced_jobs,
        )
        self._completed_jobs = {}
        self._fenced_jobs = {}
        return meta
