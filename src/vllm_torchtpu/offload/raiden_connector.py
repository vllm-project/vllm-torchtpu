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
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import \
    config as offloading_config_module
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata, OffloadingWorkerMetadata)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import \
    OffloadingConnectorScheduler
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import \
    OffloadingConnectorWorker
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import \
    OffloadingConnector
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (CanonicalKVCacheRef, CanonicalKVCaches,
                                     CanonicalKVCacheTensor, GPULoadStoreSpec,
                                     OffloadKey)
from vllm.v1.outputs import KVConnectorOutput

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.offload.raiden_store import (AdmissionOp,
                                                RaidenLoadStoreSpec,
                                                RaidenOffloadingManager,
                                                TPURaidenStoreOffloadingSpec)

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
    # Destination device blocks of failed loads, reported for scheduler recompute.
    failed_load_device_block_ids: list[int] = field(default_factory=list)


@dataclass
class RaidenOffloadingWorkerMetadata(OffloadingWorkerMetadata):
    """Worker -> scheduler metadata: contains completed-job echoes and per-rank fence acks."""
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
    """A store job parked until every worker rank acknowledges the device fence."""
    keys: list[OffloadKey]
    device_block_ids: list[int]
    acks: int = 0


class TPURaidenOffloadingScheduler(OffloadingConnectorScheduler):
    """Scheduler for the store-backed offload path.

    Strips per-rank transfer jobs and orchestrates the store engine directly
    via RaidenOffloadingManager.
    """

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

        # Park new store jobs: wait for all ranks to complete device fence.
        new_fence_ids: set[int] = set()
        for job_id, job in meta.store_jobs.items():
            src_spec, dst_spec = job.src_spec, job.dst_spec
            assert isinstance(src_spec, GPULoadStoreSpec)
            assert isinstance(dst_spec, RaidenLoadStoreSpec)
            device_block_ids = src_spec.block_ids.tolist()
            assert len(device_block_ids) == len(
                dst_spec.keys), (len(device_block_ids), len(dst_spec.keys))
            self._fence_pending[job_id] = _ParkedStore(
                keys=dst_spec.keys, device_block_ids=device_block_ids)
            new_fence_ids.add(job_id)

        # Submit load jobs immediately: destination HBM blocks are freshly
        # allocated and safe to write into.
        for job_id, job in meta.load_jobs.items():
            src_spec, dst_spec = job.src_spec, job.dst_spec
            assert isinstance(src_spec, RaidenLoadStoreSpec)
            assert isinstance(dst_spec, GPULoadStoreSpec)
            device_block_ids = dst_spec.block_ids.tolist()
            assert len(device_block_ids) == len(
                src_spec.keys), (len(device_block_ids), len(src_spec.keys))
            manager.submit_load_job(job_id,
                                    src_spec.keys,
                                    device_block_ids,
                                    job.req_id,
                                    pinned=src_spec.pinned)

        # Handle jobs_to_flush (source blocks are about to be recycled):
        # - Cancel fence-pending stores before they start.
        # - Synchronously drain launched stores so DMA finishes before blocks are overwritten.
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
                # Avoid useless TPU sync if created and cancelled in the same step.
                new_fence_ids.discard(job_id)
                logger.debug(
                    "TPURaidenOffloadingConnector: cancelled fence-pending store job %d "
                    "(source blocks reused)", job_id)
            elif manager.is_job_launched(job_id):
                launched_flush_ids.add(job_id)
        drained = (manager.drain_jobs(launched_flush_ids)
                   if launched_flush_ids else [])

        # Collect completed/drained jobs for the worker echo channel.
        for finished_job in list(drained) + manager.poll_finished_jobs():
            if finished_job.op == AdmissionOp.STORE:
                finished_store_ids.append(finished_job.job_id)
            else:
                assert finished_job.req_id is not None
                finished_loads[finished_job.job_id] = finished_job.req_id
                failed_load_block_ids.extend(
                    finished_job.failed_device_block_ids)

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
                        job_id, parked.keys, parked.device_block_ids)
        super().update_connector_output(connector_output)


class TPURaidenOffloadingConnector(OffloadingConnector):
    """Top-level facade connector for the store-backed (V2) raiden KV offload path."""

    @classmethod
    def get_required_kvcache_layout(cls,
                                    vllm_config: VllmConfig) -> str | None:
        """No layout preference, unlike the `OffloadingConnector` this extends.

        Upstream hardcodes "HND" for NIXL on CUDA. On TPU "HND" names
        SEQ_ALONG_LANE, so inheriting it would select a layout nobody asked
        for; these pages are opaque bytes to this connector.
        """
        return None

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        # Opt in to uniform cross-layer KV cache allocation when block-major layout is enabled.
        # Satisfies `KVConnectorModelRunnerMixin.use_uniform_kv_cache()` preconditions to enable
        # bundled tensor allocation in the TPU model runner.
        from vllm_torchtpu import envs as tpu_envs
        return bool(tpu_envs.VLLM_TPU_BLOCK_MAJOR_KV)

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole,
                 kv_cache_config: KVCacheConfig):
        # Skip OffloadingConnector.__init__: default factory cannot carry
        # vllm_config / kv_cache_config directly into the spec.
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)

        # The raiden singleton worker cannot coexist with this connector: the
        # disagg stack runs its own control plane in the same process and the
        # two collide on ports.
        singleton_env = os.environ.get("RAIDEN_DISABLE_SINGLETON_WORKER")
        if singleton_env is not None and singleton_env not in ("1", "true"):
            raise ValueError(
                "TPURaidenOffloadingConnector requires the raiden singleton "
                "worker to be disabled (it collides with the in-process "
                "control plane's ports), but RAIDEN_DISABLE_SINGLETON_WORKER="
                f"{singleton_env!r} was set explicitly. Unset it or set it "
                "to '1'.")
        os.environ["RAIDEN_DISABLE_SINGLETON_WORKER"] = "1"

        # Call through module to preserve PCP-aware patch on build_offloading_config.
        offloading_config = offloading_config_module.build_offloading_config(
            vllm_config, kv_cache_config)
        spec = TPURaidenStoreOffloadingSpec(offloading_config, vllm_config,
                                            kv_cache_config)

        self.connector_scheduler: TPURaidenOffloadingScheduler | None = None
        self.connector_worker: OffloadingConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = TPURaidenOffloadingScheduler(
                spec, vllm_config, kv_cache_config)
        elif role == KVConnectorRole.WORKER:
            # Worker wrapper is only used for KV buffer registration; job dicts remain empty.
            self.connector_worker = OffloadingConnectorWorker(
                spec, vllm_config, kv_cache_config)
            # Track per-step fence acknowledgments and completion echoes for worker metadata.
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
    def reset_cache(self) -> bool | None:
        # External cache reset is unsupported by the Raiden store. Report
        # failure through the endpoint's documented channel instead of raising an exception here
        logger.warning(
            "TPURaidenOffloadingConnector: external cache reset is not "
            "supported by the Raiden offload store; reporting failure.")
        return False

    # -- worker-side hooks -------------------------------------------------
    def register_kv_caches(self,
                           kv_caches: dict[str,
                                           torch.Tensor | list[torch.Tensor]]):
        """Register the TPU UBP as zero-copy canonical offload tensors.

        vLLM's generic per-layer registration does not support the TPU
        unified block pool layout or Mamba tensor lists, so this
        implementation deduplicates the UBP backing storages and exposes
        each as an `int8` byte view.
        """
        assert self.connector_worker is not None
        num_blocks = self.connector_worker.kv_cache_config.num_blocks

        pool_storages: dict[int, tuple[torch.UntypedStorage,
                                       torch.device]] = {}
        for layer_kv_cache in kv_caches.values():
            state_tensors = ([layer_kv_cache] if isinstance(
                layer_kv_cache, torch.Tensor) else layer_kv_cache)
            for tensor in state_tensors:
                storage = tensor.untyped_storage()
                pool_storages.setdefault(storage.data_ptr(),
                                         (storage, tensor.device))

        pool_tensors: list[CanonicalKVCacheTensor] = []
        for storage, device in pool_storages.values():
            storage_bytes = storage.nbytes()
            assert storage_bytes % num_blocks == 0, (
                "KV pool storage must hold whole scheduler blocks: "
                f"storage_bytes={storage_bytes}, num_blocks={num_blocks}")
            page_size_bytes = storage_bytes // num_blocks
            int8_view = torch.empty(0, dtype=torch.int8,
                                    device=device).set_(storage).view(
                                        num_blocks, page_size_bytes)
            pool_tensors.append(
                CanonicalKVCacheTensor(tensor=int8_view,
                                       page_size_bytes=page_size_bytes))

        # Retain pool tensor references to dispatch their writes during save fences.
        self._pool_sync_tensors = [
            pool_tensor.tensor for pool_tensor in pool_tensors
        ]

        group_data_refs = [[
            CanonicalKVCacheRef(tensor_idx=idx,
                                page_size_bytes=pool_tensor.page_size_bytes)
            for idx, pool_tensor in enumerate(pool_tensors)
        ] for _ in self.connector_worker.kv_cache_config.kv_cache_groups]

        self.connector_worker._init_worker(
            CanonicalKVCaches(tensors=pool_tensors,
                              group_data_refs=group_data_refs))

    def get_finished(self,
                     finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        meta = self._get_connector_metadata()
        assert isinstance(meta, RaidenOffloadingConnectorMetadata)

        if meta.fence_job_ids:
            # Establish an asynchronous device fence before acknowledging store jobs:
            # 1. Dispatch any deferred pool writes so they precede the fence event.
            # 2. Record a stream event to track completion without host-side blocking.
            assert self._pool_sync_tensors, (
                "KV pool tensors must be registered before the save fence")
            _enqueue_pool_writes(self._pool_sync_tensors)
            self._pending_fences.append(
                (set(meta.fence_job_ids), _record_fence_event()))

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
