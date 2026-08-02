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
"""Controller-driven KV offloading via raiden's `KVCacheStore` (V2).

Replaces the worker-driven V1 offload stack for the raiden path: the
directory (hash -> host slot), LRU eviction, pinning, and the transfers
themselves all live in raiden's `KVCacheStore`, which the scheduler-side
`RaidenOffloadingManager` owns. Workers register their device KV buffers
with the store's in-process controller once (`KVCacheManager`) and then act
as passive DMA endpoints for the controller's `save`/`load` fanout — no
per-rank transfer submission or completion aggregation exists anymore.

Block granularity: one store entry per physical kernel block. A vLLM
offloaded block (= one scheduler GPU block, `block_size_factor == 1`) maps
to `gpu_block_size_factor` store entries with derived sub-hashes
`OffloadKey ‖ sub_idx`. Kernel blocks of one GPU block are
device-contiguous, so raiden's copy coalescing fuses them back into single
DMAs.

Supported models: dense full-attention (single KV group) and hybrid
attention+Mamba models on the TPU unified block pool, where every group's
blocks are fungible rows of the same buffers at scheduler-block granularity
— GPU block IDs are pool row IDs for every group, so one sub-hash mapping
covers both. Constraints (asserted): `blocks_per_chunk == 1` (one OffloadKey
= one scheduler GPU block), no context parallelism.

Phase-2 hardening: store jobs are batch-capped and retried once on transfer
failure; keys that fail terminally are quarantined (reported MISS, never
re-admitted). With `RAIDEN_SHM_KEY` set, the host pools and the store
directory live in shared memory and survive an engine-process crash; the
directory is recovered inside the plain constructors on restart, gated by a
config-derived `RAIDEN_SHM_MODEL_UID`.

See `raiden_connector.py` for the connector wiring (fence + echo channel).
"""
from __future__ import annotations

import enum
import hashlib
import os
import time
from dataclasses import dataclass, field

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.kv_offload.base import (CanonicalKVCaches, LoadStoreSpec,
                                     LookupResult, OffloadingManager,
                                     OffloadingSpec, OffloadingWorker,
                                     OffloadKey, PrepareStoreOutput,
                                     ReqContext, RequestOffloadingContext,
                                     TransferResult)
from vllm.v1.kv_offload.config import OffloadingConfig

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Upper bound (seconds) on a scheduler-side drain of in-flight store/load
# jobs whose source/destination blocks are about to be reused. Shares the V1
# knob so recipes tune one value.
_DRAIN_TIMEOUT_S = float(
    os.environ.get("VLLM_TPU_OFFLOAD_WAIT_TIMEOUT_S", "30"))

# Store-job batch cap (offload keys per job). One raiden save() batch is one
# joined failure unit, so modest batches keep one bad block from failing
# unrelated work; keys beyond the cap are simply re-offered by the scheduler
# on the next step.
_MAX_STORE_KEYS_PER_JOB = int(
    os.environ.get("VLLM_TPU_OFFLOAD_MAX_STORE_KEYS_PER_JOB", "64"))

# A save whose transfer failed is retried this many times (§5.4: a failed
# save leaves the entry HBM+pinned and a second save() works). Retries are
# suppressed for drained jobs: a drain means the source blocks are about to
# be reused, so re-reading them would ship garbage.
_SAVE_RETRIES = int(os.environ.get("VLLM_TPU_OFFLOAD_SAVE_RETRIES", "1"))

_DEFAULT_CONTROLLER_PORT = 51515


class RaidenLoadStoreSpec(LoadStoreSpec):
    """Offload-side spec: the ordered offload keys a job covers.

    Device block IDs are bound later, at job-submit time, from the matching
    `GPULoadStoreSpec` (they are not known to the manager interface).

    `pinned` is False when prepare_load lost the race against eviction (the
    manager interface has no failure return there); the load job then fails
    cleanly at submit time and vLLM recomputes the blocks.
    """

    def __init__(self, keys: list[OffloadKey], pinned: bool = True):
        self.keys = keys
        self.pinned = pinned

    def __repr__(self) -> str:
        return (f"RaidenLoadStoreSpec({len(self.keys)} keys"
                f"{'' if self.pinned else ', unpinned'})")


class AdmissionOp(enum.Enum):
    STORE = "store"
    LOAD = "load"


@dataclass
class Admission:
    """Exact-batch record of one launched store/load job.

    Raiden keys its candidate-restoration state by the sorted hash batch
    passed to `insert_and_lock`, so release / rollback must use exactly the
    admitted batch, exactly once. This record is the single source of truth
    for that batch and for which cleanup the job's terminal state owes.
    """
    job_id: int
    op: AdmissionOp
    keys: list[OffloadKey]
    sub_hashes: list[bytes]
    # Sub-hashes still awaiting a poll_{save,load}_status verdict.
    pending: set[bytes]
    failed: bool = False
    # Loads only: owning request and destination GPU block IDs, reported
    # through get_block_ids_with_load_errors on failure.
    req_id: str | None = None
    dst_gpu_block_ids: list[int] = field(default_factory=list)
    # Stores only: the sub-hashes actually saved (newly admitted HBM entries;
    # pre-existing HOST entries were pinned, not re-saved), the subset the
    # poll reported failed (retry targets), and the remaining retry budget.
    saved_sub_hashes: list[bytes] = field(default_factory=list)
    failed_sub_hashes: list[bytes] = field(default_factory=list)
    retries_left: int = 0


@dataclass
class FinishedJob:
    job_id: int
    op: AdmissionOp
    success: bool
    req_id: str | None = None
    failed_gpu_block_ids: list[int] = field(default_factory=list)


class RaidenOffloadingManager(OffloadingManager):
    """Scheduler-side OffloadingManager backed by raiden's KVCacheStore.

    The store owns the directory, LRU, pins, and transfers; this class maps
    vLLM's manager contract onto it and owns the failure-aware job finalizer
    (`poll_finished_jobs`). The stock `complete_store` / `complete_load`
    callbacks are deliberately no-ops: all raiden-side state transitions
    (release on success, release_and_delete on failure) happen in the
    finalizer, keyed by the exact admitted batch.
    """

    def __init__(
        self,
        *,
        capacity_kernel_blocks: int,
        num_offloaded_blocks: int,
        sub_blocks_per_key: int,
        world_size: int,
        raiden_controller_port: int,
        raiden_job_name: str,
        store: object | None = None,
        store_types: object | None = None,
    ):
        assert capacity_kernel_blocks == (num_offloaded_blocks *
                                          sub_blocks_per_key)
        self._capacity_keys = num_offloaded_blocks
        self._f = sub_blocks_per_key
        self._world_size = world_size

        if store is None:
            from tpu_raiden.api.torch import kv_cache_store as store_types
            store = store_types.KVCacheStore(
                capacity=capacity_kernel_blocks,
                raiden_id=store_types.RaidenId(raiden_job_name, "0",
                                               "kv_cache", 0),
                # Minimum accepted values; neither feeds the copy math.
                # Transfer fanout comes from the registered workers.
                num_shards=1,
                shard_size_bytes=1,
                raiden_controller_port=raiden_controller_port,
            )
            logger.info(
                "[kv-offload-v2] KVCacheStore up: capacity=%d kernel blocks "
                "(%d offloaded blocks x %d), controller at %s",
                capacity_kernel_blocks, num_offloaded_blocks,
                sub_blocks_per_key, store.raiden_controller_address)
        else:
            # Injected store (tests): the caller supplies the module holding
            # BlockStatus / RaidenBlockID so tpu_raiden isn't imported.
            assert store_types is not None
        self._store = store
        self._block_status = store_types.BlockStatus
        self._block_id_cls = store_types.RaidenBlockID
        self._raiden_id = self._store.raiden_id

        # Registration gate: stores/loads stay disabled until every rank's
        # KVCacheManager has registered with the store's controller. A
        # partial worker set would otherwise silently partial-save and
        # report success (data corruption on a later load).
        self._gate_open = False

        # Advisory admission state. The store directory is the source of
        # truth for lookups; these only filter prepare_store so the same key
        # is not re-admitted while stored or in flight.
        self._stored_keys: set[OffloadKey] = set()
        self._inflight_store_keys: set[OffloadKey] = set()

        # Poison-key quarantine: keys whose transfer failed terminally
        # (stores: after the retry budget; loads: first terminal failure).
        # Quarantined keys report MISS and are never re-admitted, so a
        # persistently failing block cannot fail-loop through fence/save or
        # load/recompute cycles. Any HOST leftovers they have in the store
        # age out through normal LRU pressure.
        self._quarantined_keys: set[OffloadKey] = set()

        self._admissions: dict[int, Admission] = {}
        self._hash_to_admission: dict[bytes, Admission] = {}
        self._finished: list[FinishedJob] = []

    # -- sub-hash / block-id expansion ----------------------------------

    def _sub_hashes(self, key: OffloadKey) -> list[bytes]:
        return [key + i.to_bytes(4, "big") for i in range(self._f)]

    def _expand_keys(self, keys: list[OffloadKey]) -> list[bytes]:
        out: list[bytes] = []
        for key in keys:
            out.extend(self._sub_hashes(key))
        return out

    def _expand_gpu_blocks(self, gpu_block_ids: list[int]) -> list[int]:
        f = self._f
        return [b * f + i for b in gpu_block_ids for i in range(f)]

    def _sub_hash_status(self, sub_hash: bytes):
        """Peek one sub-hash's directory status (None when absent)."""
        matches = self._store.lookup([sub_hash])
        return matches[0][1].status if matches else None

    # -- registration gate ----------------------------------------------

    def is_gate_open(self) -> bool:
        if self._gate_open:
            return True
        num = self._store.num_registered_workers()
        if num > self._world_size:
            raise RuntimeError(
                f"[kv-offload-v2] {num} workers registered with the offload "
                f"controller, expected at most world_size={self._world_size}."
                " A stale or foreign worker registration would corrupt the "
                "save fanout.")
        if num == self._world_size:
            self._gate_open = True
            logger.info(
                "[kv-offload-v2] registration gate open: %d/%d workers "
                "registered; stores and loads enabled.", num, self._world_size)
        return self._gate_open

    # -- OffloadingManager contract --------------------------------------

    def on_new_request(self,
                       req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        if not self.is_gate_open():
            return LookupResult.MISS
        if key in self._quarantined_keys:
            return LookupResult.MISS
        if key in self._inflight_store_keys:
            return LookupResult.HIT_PENDING
        sub_hashes = self._sub_hashes(key)
        # Raiden lookup halts at the first miss (prefix semantics), which is
        # exactly all-or-nothing over one key's sub-hashes. Peek only: no
        # LRU-recency update, eviction candidates report as misses.
        matches = self._store.lookup(sub_hashes)
        if len(matches) < len(sub_hashes):
            self._stored_keys.discard(key)
            return LookupResult.MISS
        for _, block in matches:
            status = block.status
            if status in (self._block_status.HOST,
                          self._block_status.HOST_AND_HBM):
                continue
            if status == self._block_status.HBM:
                # Save still in flight (should be covered by the in-flight
                # set above; kept as a consistency net).
                return LookupResult.HIT_PENDING
            # REMOTE entries become loadable in phase 4 (read_remote).
            return LookupResult.MISS
        # Learn hits the advisory state has not seen: entries recovered from
        # the shm-backed directory across a restart predate this process, and
        # this keeps touch/prepare_store treating them like our own stores.
        self._stored_keys.add(key)
        return LookupResult.HIT

    def touch(self, keys, req_context: ReqContext) -> None:
        if not self._gate_open:
            return
        # pin+release: Unpin splices the entry back to the MRU front, which
        # is exactly a touch. Plain lookup/insert never refresh recency.
        # The advisory filter keeps this from probing keys we never stored.
        for key in keys:
            if key not in self._stored_keys:
                continue
            sub_hashes = self._sub_hashes(key)
            if self._store.pin(sub_hashes):
                self._store.release(sub_hashes)

    def prepare_load(self, keys, req_context: ReqContext) -> LoadStoreSpec:
        keys = list(keys)
        sub_hashes = self._expand_keys(keys)
        if not self._store.pin(sub_hashes):
            # Lost the race against eviction between lookup and pin (raiden's
            # pin is all-or-nothing, so nothing is held). The manager
            # interface has no failure return here; hand the connector an
            # unpinned spec so submit_load_job fails the job cleanly and the
            # scheduler recomputes the blocks.
            logger.warning(
                "[kv-offload-v2] prepare_load lost %d sub-blocks for request "
                "%s to eviction between lookup and pin; the load job will "
                "fail cleanly and the blocks recompute", len(sub_hashes),
                req_context.req_id)
            return RaidenLoadStoreSpec(keys, pinned=False)
        return RaidenLoadStoreSpec(keys)

    def complete_load(self, keys, req_context: ReqContext) -> None:
        # Intentionally a no-op: the finalizer releases the pinned batch
        # when the poll reports the load terminal (host-observed DMA
        # completion). Releasing again here could drop another request's
        # fresh pin on the same entries.
        return

    def prepare_store(self, keys,
                      req_context: ReqContext) -> PrepareStoreOutput | None:
        if not self.is_gate_open():
            # Loud (stock scheduler logs "cannot store chunks") and safe:
            # nothing advances, the chunks are retried next step.
            return None
        keys_to_store = [
            k for k in keys if k not in self._stored_keys and k not in
            self._inflight_store_keys and k not in self._quarantined_keys
        ]
        # Batch cap: one store job is one raiden save() failure unit, so
        # keep it modest. Dropped keys are not accepted, so the scheduler
        # re-offers them next step (per-group chunk advancement only moves
        # past accepted keys).
        keys_to_store = keys_to_store[:_MAX_STORE_KEYS_PER_JOB]
        if len(keys_to_store) > self._capacity_keys:
            return None
        # Atomic admission happens at job-submit time via insert_and_lock;
        # in-flight tracking here only stops duplicate admission of the same
        # key from a concurrent request.
        self._inflight_store_keys.update(keys_to_store)
        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=RaidenLoadStoreSpec(keys_to_store),
            evicted_keys=[],
        )

    def complete_store(self,
                       keys,
                       req_context: ReqContext,
                       success: bool = True) -> None:
        # Intentionally a no-op; see complete_load. The stock completion
        # channel asserts success and cannot express raiden's outcomes — the
        # finalizer already drove the raiden-side transition per outcome.
        return

    def take_events(self):
        return ()

    def has_pending_work(self) -> bool:
        return bool(self._admissions)

    def reset_cache(self) -> None:
        raise NotImplementedError(
            "[kv-offload-v2] reset_cache is unsupported until raiden ships a "
            "public clear/evict_all: release_and_delete is a rollback "
            "primitive that preserves HOST entries, so a vLLM-side reset "
            "would silently not clear the cache.")

    def shutdown(self) -> None:
        if self._admissions:
            self.drain_jobs(set(self._admissions.keys()))

    # -- job submission bridge (called by the connector scheduler) --------

    def submit_store_job(self, job_id: int, keys: list[OffloadKey],
                         gpu_block_ids: list[int]) -> None:
        """Admit and launch a store job after every rank fenced.

        insert_and_lock pins existing entries without overwriting their
        slices, checks capacity for the whole batch, displaces LRU entries
        revertibly, and rolls back fully on failure. On success the save is
        issued for the admitted batch; completion is observed by the
        finalizer.
        """
        assert len(keys) == len(gpu_block_ids), (len(keys), len(gpu_block_ids))
        sub_hashes = self._expand_keys(keys)
        device_ids = self._expand_gpu_blocks(gpu_block_ids)
        slices = [
            self._block_id_cls(
                raiden_id=self._raiden_id,
                host_block_id=-1,
                device_block_id=d,
                status=self._block_status.HBM,
            ) for d in device_ids
        ]
        if not self._store.insert_and_lock(sub_hashes, slices, on_host=False):
            # Rolled back inside raiden; no save issued, no cleanup owed.
            logger.warning(
                "[kv-offload-v2] store job %d: insert_and_lock rejected %d "
                "sub-blocks (capacity pressure with pinned entries)", job_id,
                len(sub_hashes))
            self._inflight_store_keys.difference_update(keys)
            self._finished.append(
                FinishedJob(job_id=job_id, op=AdmissionOp.STORE,
                            success=False))
            return
        # Save only the newly admitted HBM entries. Pre-existing HOST /
        # HOST_AND_HBM sub-hashes (recovered across a restart, or committed
        # by a partially failed earlier save) were locked without touching
        # their slices; their bytes are valid (content-addressed) and
        # passing them to save() would fail the whole launch on its
        # HBM-status precondition. insert_and_lock returns a bare bool, so
        # classify with per-sub-hash peeks (in-process, and only on the
        # store path).
        to_save = [
            h for h in sub_hashes
            if self._sub_hash_status(h) == self._block_status.HBM
        ]
        if not to_save:
            # Every sub-hash was already host-resident: nothing to transfer.
            # Release the exact admitted batch and finish successfully.
            self._store.release(sub_hashes)
            self._inflight_store_keys.difference_update(keys)
            self._stored_keys.update(keys)
            self._finished.append(
                FinishedJob(job_id=job_id, op=AdmissionOp.STORE, success=True))
            return
        if not self._store.save(to_save):
            # Synchronous launch failure: revert the admission exactly once
            # with the exact batch; never entered into the poll path.
            self._store.release_and_delete(sub_hashes)
            logger.error(
                "[kv-offload-v2] store job %d: save() launch failed for %d "
                "sub-blocks", job_id, len(to_save))
            self._inflight_store_keys.difference_update(keys)
            self._finished.append(
                FinishedJob(job_id=job_id, op=AdmissionOp.STORE,
                            success=False))
            return
        admission = Admission(
            job_id=job_id,
            op=AdmissionOp.STORE,
            keys=keys,
            sub_hashes=sub_hashes,
            pending=set(to_save),
            saved_sub_hashes=to_save,
            retries_left=_SAVE_RETRIES,
        )
        self._admissions[job_id] = admission
        for h in to_save:
            self._hash_to_admission[h] = admission

    def submit_load_job(self,
                        job_id: int,
                        keys: list[OffloadKey],
                        gpu_block_ids: list[int],
                        req_id: str,
                        pinned: bool = True) -> None:
        """Launch a load job; entries were pinned in prepare_load."""
        assert len(keys) == len(gpu_block_ids), (len(keys), len(gpu_block_ids))
        if not pinned:
            # prepare_load lost the pin race; nothing is held, so fail the
            # job without touching the store and let vLLM recompute.
            self._finished.append(
                FinishedJob(job_id=job_id,
                            op=AdmissionOp.LOAD,
                            success=False,
                            req_id=req_id,
                            failed_gpu_block_ids=list(gpu_block_ids)))
            return
        sub_hashes = self._expand_keys(keys)
        device_ids = self._expand_gpu_blocks(gpu_block_ids)
        if not self._store.load(sub_hashes, device_ids):
            # Failed launch leaves HOST entries intact; destination blocks
            # are undefined and vLLM recomputes them.
            self._store.release(sub_hashes)
            logger.error(
                "[kv-offload-v2] load job %d (req %s): load() launch failed "
                "for %d sub-blocks", job_id, req_id, len(sub_hashes))
            self._finished.append(
                FinishedJob(job_id=job_id,
                            op=AdmissionOp.LOAD,
                            success=False,
                            req_id=req_id,
                            failed_gpu_block_ids=list(gpu_block_ids)))
            return
        admission = Admission(
            job_id=job_id,
            op=AdmissionOp.LOAD,
            keys=keys,
            sub_hashes=sub_hashes,
            pending=set(sub_hashes),
            req_id=req_id,
            dst_gpu_block_ids=list(gpu_block_ids),
        )
        self._admissions[job_id] = admission
        for h in sub_hashes:
            self._hash_to_admission[h] = admission

    def on_store_job_cancelled(self, keys: list[OffloadKey]) -> None:
        """A fence-pending (never admitted) store job was cancelled."""
        self._inflight_store_keys.difference_update(keys)

    def is_job_launched(self, job_id: int) -> bool:
        """Whether a job has been admitted+launched and is not yet terminal."""
        return job_id in self._admissions

    # -- finalizer --------------------------------------------------------

    def _quarantine(self, admission: Admission) -> None:
        self._quarantined_keys.update(admission.keys)
        logger.error(
            "[kv-offload-v2] quarantining %d keys after a terminal %s "
            "transfer failure (job %d); they will not be offloaded again "
            "(%d keys quarantined in total)", len(admission.keys),
            admission.op.value, admission.job_id, len(self._quarantined_keys))

    def _finalize(self, admission: Admission) -> None:
        del self._admissions[admission.job_id]
        for h in admission.sub_hashes:
            self._hash_to_admission.pop(h, None)
        success = not admission.failed
        if not success:
            self._quarantine(admission)
        if admission.op == AdmissionOp.STORE:
            self._inflight_store_keys.difference_update(admission.keys)
            if success:
                # Unpin the exact admitted batch; entries become evictable.
                self._store.release(admission.sub_hashes)
                self._stored_keys.update(admission.keys)
            else:
                # Unpin and delete the newly admitted HBM entries, restoring
                # displaced candidates.
                self._store.release_and_delete(admission.sub_hashes)
            self._finished.append(
                FinishedJob(job_id=admission.job_id,
                            op=AdmissionOp.STORE,
                            success=success))
        else:
            # Loads: release the pins taken in prepare_load. On failure the
            # HOST entries stay intact; only the destination device blocks
            # are undefined and get recomputed.
            self._store.release(admission.sub_hashes)
            self._finished.append(
                FinishedJob(
                    job_id=admission.job_id,
                    op=AdmissionOp.LOAD,
                    success=success,
                    req_id=admission.req_id,
                    failed_gpu_block_ids=([] if success else list(
                        admission.dst_gpu_block_ids)),
                ))

    def _poll_once(self) -> None:
        if not self._admissions:
            return
        has_stores = any(a.op == AdmissionOp.STORE
                         for a in self._admissions.values())
        has_loads = any(a.op == AdmissionOp.LOAD
                        for a in self._admissions.values())
        results: list[tuple[list[bytes], list[bytes]]] = []
        if has_stores:
            done, failed, _pending = self._store.poll_save_status()
            results.append((done, failed))
        if has_loads:
            done, failed, _pending = self._store.poll_load_status()
            results.append((done, failed))
        # Keyed by job_id: a mixed batch whose last hashes resolve in the
        # same poll (some done, some failed) must finalize exactly once.
        newly_terminal: dict[int, Admission] = {}
        for done, failed in results:
            for h in failed:
                admission = self._hash_to_admission.get(h)
                if admission is None:
                    continue
                admission.failed = True
                admission.failed_sub_hashes.append(h)
                admission.pending.discard(h)
                if not admission.pending:
                    newly_terminal[admission.job_id] = admission
            for h in done:
                admission = self._hash_to_admission.get(h)
                if admission is None:
                    continue
                admission.pending.discard(h)
                if not admission.pending:
                    newly_terminal[admission.job_id] = admission
        for admission in newly_terminal.values():
            if self._maybe_retry_save(admission):
                continue
            self._finalize(admission)

    def _maybe_retry_save(self, admission: Admission) -> bool:
        """Re-issue the failed subset of a store job's save, at most
        `retries_left` times.

        Safe by §5.4: a failed save leaves its entries HBM + pinned with the
        host slot freed, and the job stays non-terminal so the connector's
        pending-job gate keeps the source device blocks retained. Committed
        sub-hashes are not re-saved. drain_jobs zeroes retries_left first —
        after a flush the source blocks are about to be reused, so a retry
        would read garbage.
        """
        if (admission.op != AdmissionOp.STORE or not admission.failed
                or admission.retries_left <= 0):
            return False
        retry_hashes = admission.failed_sub_hashes
        admission.retries_left -= 1
        if not self._store.save(retry_hashes):
            logger.error(
                "[kv-offload-v2] store job %d: save() retry launch failed "
                "for %d sub-blocks; job fails", admission.job_id,
                len(retry_hashes))
            return False
        logger.warning(
            "[kv-offload-v2] store job %d: retrying save of %d failed "
            "sub-blocks (%d retries left)", admission.job_id,
            len(retry_hashes), admission.retries_left)
        admission.pending = set(retry_hashes)
        admission.failed_sub_hashes = []
        admission.failed = False
        return True

    def poll_finished_jobs(self) -> list[FinishedJob]:
        """Poll raiden completion and return newly terminal jobs."""
        self._poll_once()
        finished = self._finished
        self._finished = []
        return finished

    def drain_jobs(self, job_ids: set[int]) -> list[FinishedJob]:
        """Block until the named launched jobs are terminal (bounded).

        Used when a job's source device blocks are about to be reused: the
        scheduler drains before shipping this step's metadata, so the DMA
        completes before the forward that overwrites the blocks.
        """
        # No retries once a job is being drained: its source blocks are
        # being handed back, so re-reading them would save garbage.
        for jid in job_ids:
            admission = self._admissions.get(jid)
            if admission is not None:
                admission.retries_left = 0
        deadline = time.perf_counter() + _DRAIN_TIMEOUT_S
        drained: list[FinishedJob] = []
        while True:
            self._poll_once()
            drained.extend(f for f in self._finished if f.job_id in job_ids)
            self._finished = [
                f for f in self._finished if f.job_id not in job_ids
            ]
            if not any(jid in self._admissions for jid in job_ids):
                return drained
            if time.perf_counter() >= deadline:
                logger.error(
                    "[kv-offload-v2] drain timed out after %.1fs for jobs %s;"
                    " their transfers are still in flight and their blocks "
                    "are about to be reused", _DRAIN_TIMEOUT_S,
                    sorted(jid for jid in job_ids if jid in self._admissions))
                return drained
            time.sleep(0.0005)


def derive_shm_model_uid(vllm_config: VllmConfig, *, kernel_block_size: int,
                         per_block_shape: tuple[int, ...], kv_dtype: object,
                         gpu_block_size: int, world_size: int,
                         capacity_kernel_blocks: int) -> str:
    """Deterministic RAIDEN_SHM_MODEL_UID for shm-backed restart recovery.

    Covers everything that must match for recovered host bytes to be valid;
    an incompatible restart therefore fails raiden's segment validation and
    cold-starts instead of serving garbage.
    """
    material = repr(
        (vllm_config.model_config.model, vllm_config.model_config.revision,
         str(kv_dtype), kernel_block_size, per_block_shape, gpu_block_size,
         world_size, capacity_kernel_blocks, "kv-offload-v2-subhash"))
    return hashlib.sha256(material.encode()).hexdigest()[:32]


def resolve_kernel_geometry(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> tuple[int, tuple[int, ...], object, int]:
    """Resolve (kernel_block_size, per_block_shape, kv_dtype, gpu_block_size).

    Shared by the scheduler side (capacity sizing, sub-hash factor) and the
    worker side (device view rebuild) so their geometry can never diverge.

    Dense models: the single group's full-attention spec defines everything.

    Hybrid attention+Mamba models: requires the TPU unified block pool, in
    which every group's blocks are fungible rows of the same attention-shaped
    buffers at scheduler-block granularity with one padded page size. GPU
    block IDs are then pool row IDs for *every* group, so the dense sub-hash
    and kernel-row expansion applies unchanged; the attention geometry defines
    the row unit (the mamba state lives inside attention-shaped pages).
    """
    from vllm.v1.kv_cache_interface import AttentionSpec
    from vllm.v1.worker.utils import select_common_block_size

    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    groups = kv_cache_config.kv_cache_groups
    assert groups, "[kv-offload-v2] no KV cache groups"
    if len(groups) == 1:
        spec0 = groups[0].kv_cache_spec
        assert isinstance(spec0, FullAttentionSpec), (
            "[kv-offload-v2] single-group models must be full-attention; "
            f"got {type(spec0).__name__}")
    else:
        # Hybrid: group order follows model layer order, so group 0 may be
        # a Mamba group — filter for the attention groups instead.
        from vllm_torchtpu.platforms.tpu_block_size_utils import \
            unified_kv_layout_enabled
        assert unified_kv_layout_enabled(vllm_config), (
            "[kv-offload-v2] hybrid models require the unified block pool "
            "(TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1)")
        attn_specs = [
            g.kv_cache_spec for g in groups
            if isinstance(g.kv_cache_spec, AttentionSpec)
        ]
        assert attn_specs, (
            "[kv-offload-v2] hybrid model has no attention KV group")
        spec0 = attn_specs[0]
        assert all((s.block_size, s.num_kv_heads, s.head_size, s.dtype,
                    s.page_size_bytes) == (spec0.block_size,
                                           spec0.num_kv_heads, spec0.head_size,
                                           spec0.dtype, spec0.page_size_bytes)
                   for s in attn_specs), (
                       "[kv-offload-v2] attention groups disagree on geometry")
        # The normalizer pads every group's page to the shared pool page, so
        # pool rows sit at scheduler-block granularity for all groups.
        assert all(g.kv_cache_spec.page_size_bytes == spec0.page_size_bytes
                   for g in groups), (
                       "[kv-offload-v2] unified pool groups must share one "
                       "padded page size")

    backend_cls = TpuPlatform._find_non_ssm_backend(vllm_config)
    assert backend_cls is not None, (
        "[kv-offload-v2] requires a non-SSM attention backend")
    kernel_block_size = select_common_block_size(spec0.block_size,
                                                 [backend_cls])
    full_5d_shape = backend_cls.get_kv_cache_shape(
        num_blocks=1,
        block_size=kernel_block_size,
        num_kv_heads=spec0.num_kv_heads,
        head_size=spec0.head_size,
        cache_dtype_str=spec0.dtype,
    )
    per_block_shape = tuple(full_5d_shape[1:])
    return kernel_block_size, per_block_shape, spec0.dtype, spec0.block_size


class RaidenStoreOffloadingWorker(OffloadingWorker):
    """Per-rank worker: registers the device KV buffers as one shard.

    Nearly a stub — registration makes the rank a passive DMA endpoint for
    the controller's TransferBuffers fanout. Jobs are never routed here (the
    scheduler strips them), so the submit paths fail loudly if the routing
    regresses.

    Registration is deferred: `get_worker()` runs inside
    `register_kv_caches`, strictly before the scheduler-side controller
    exists, so an eager `KVCacheManager` would dial an unbound port on every
    startup. `ensure_registered()` is called from the per-step worker hook;
    by the first step the controller is deterministically up.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        *,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        host_blocks_to_allocate: int,
        controller_address: str,
        rank: int,
    ):
        import numpy as np
        import torch

        (kernel_block_size, per_block_shape, kv_dtype,
         gpu_block_size) = resolve_kernel_geometry(vllm_config,
                                                   kv_cache_config)
        assert gpu_block_size % kernel_block_size == 0
        gpu_block_size_factor = gpu_block_size // kernel_block_size

        bytes_per_kernel_block = (
            int(np.prod(per_block_shape)) *
            torch.empty(0, dtype=kv_dtype).element_size())

        # Rebuild the kv_dtype-typed kernel-block views from the same
        # storage: raiden DMAs against the live base buffer, so the view
        # must keep the physical kernel-block leading dimension.
        device_tensors: list[torch.Tensor] = []
        for kv_cache_tensor in kv_caches.tensors:
            # One canonical row must be exactly gpu_block_size_factor kernel
            # rows. On the hybrid unified pool this is the load-bearing
            # uniformity check: every pool buffer carries the same padded
            # page, so pool rows are fungible across groups and GPU block
            # IDs are pool row IDs for every group.
            assert kv_cache_tensor.page_size_bytes == (
                gpu_block_size_factor * bytes_per_kernel_block), (
                    "canonical page size does not match the kernel-row "
                    f"geometry: page={kv_cache_tensor.page_size_bytes}, "
                    f"factor={gpu_block_size_factor}, "
                    f"kernel_row_bytes={bytes_per_kernel_block}")
            int8_view = kv_cache_tensor.tensor
            storage_bytes = int8_view.numel() * int8_view.element_size()
            assert storage_bytes % bytes_per_kernel_block == 0, (
                "KV cache storage must contain whole physical kernel blocks:"
                f" storage_bytes={storage_bytes}, "
                f"bytes_per_kernel_block={bytes_per_kernel_block}")
            num_kernel_blocks = storage_bytes // bytes_per_kernel_block
            assert num_kernel_blocks == (
                int8_view.shape[0] * gpu_block_size_factor), (
                    "canonical KV cache geometry does not match physical "
                    f"kernel-block geometry: canonical={int8_view.shape[0]}, "
                    f"factor={gpu_block_size_factor}, "
                    f"kernel_blocks={num_kernel_blocks}")
            device_tensors.append(
                torch.empty(0, dtype=kv_dtype, device=int8_view.device).set_(
                    int8_view.untyped_storage()).view((num_kernel_blocks, ) +
                                                      per_block_shape))

        # raiden holds raw pointers to these buffers for its lifetime; keep
        # the tensors alive on this worker.
        self._device_tensors = [[t] for t in device_tensors]
        self._host_blocks_to_allocate = host_blocks_to_allocate
        self._controller_address = controller_address
        self._rank = rank
        self._mgr = None

    def ensure_registered(self) -> None:
        """Idempotently construct the KVCacheManager (registers the rank).

        Raises if registration fails (wheel #1's ctor-raise): a rank that
        silently fails to register would be skipped by every save fanout
        while the store still reports success.
        """
        if self._mgr is not None:
            return
        from tpu_raiden.api.torch import kv_cache_manager as _kcm
        self._mgr = _kcm.KVCacheManager(
            kv_caches=self._device_tensors,
            local_control_port=0,
            host_blocks_to_allocate=self._host_blocks_to_allocate,
            raiden_worker_port=0,
            raiden_controller_address=self._controller_address,
            worker_id=f"shard_{self._rank}",
            node_id=self._rank,
        )
        logger.info(
            "[kv-offload-v2] rank %d registered with offload controller at "
            "%s (%d host kernel blocks)", self._rank, self._controller_address,
            self._host_blocks_to_allocate)

    def submit_store(self, job_id, src_spec, dst_spec) -> bool:
        raise AssertionError(
            "[kv-offload-v2] submit_store reached a worker; transfers are "
            "controller-driven and jobs must be stripped scheduler-side")

    def submit_load(self, job_id, src_spec, dst_spec) -> bool:
        raise AssertionError(
            "[kv-offload-v2] submit_load reached a worker; transfers are "
            "controller-driven and jobs must be stripped scheduler-side")

    def get_finished(self) -> list[TransferResult]:
        return []

    def wait(self, job_ids: set[int]) -> None:
        # Flush is a controller-side drain (see
        # RaidenOffloadingManager.drain_jobs); nothing to wait on per rank.
        return

    def shutdown(self) -> None:
        self._mgr = None


class TPURaidenStoreOffloadingSpec(OffloadingSpec):
    """Spec wiring the store-backed manager and the thin per-rank worker.

    Constructed directly by TPURaidenOffloadingConnector (not through the
    OffloadingSpecFactory) so it can carry vllm_config / kv_cache_config,
    which the 0.26.0 OffloadingConfig does not.
    """

    def __init__(self, config: OffloadingConfig, vllm_config: VllmConfig,
                 kv_cache_config: KVCacheConfig):
        super().__init__(config)
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config

        assert config.cache.blocks_per_chunk == 1, (
            "[kv-offload-v2] requires blocks_per_chunk == 1 (one OffloadKey "
            "= one scheduler GPU block); got "
            f"{config.cache.blocks_per_chunk}")
        parallel = config.parallel
        assert parallel.pcp_size == 1 and parallel.dcp_size == 1, (
            "[kv-offload-v2] does not support context parallelism yet")
        self.world_size = parallel.world_size

        # Consistency net (mirrors the V1 spec): the model-level hybrid flag
        # and the KV group composition must agree, since the geometry
        # resolver branches on the group structure.
        from vllm.v1.kv_cache_interface import MambaSpec
        self.is_hybrid_model = vllm_config.model_config.is_hybrid
        has_mamba_group = any(
            isinstance(g.kv_cache_spec, MambaSpec)
            for g in kv_cache_config.kv_cache_groups)
        assert self.is_hybrid_model == has_mamba_group, (
            "[kv-offload-v2] model_config.is_hybrid="
            f"{self.is_hybrid_model} but the KV cache groups "
            f"{'contain' if has_mamba_group else 'lack'} a MambaSpec group")

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise ValueError(
                "cpu_bytes_to_use must be specified in "
                "kv_connector_extra_config (--kv-offloading-size)")

        (kernel_block_size, per_block_shape, kv_dtype,
         gpu_block_size) = resolve_kernel_geometry(vllm_config,
                                                   kv_cache_config)
        assert gpu_block_size % kernel_block_size == 0
        self.gpu_block_size_factor = gpu_block_size // kernel_block_size

        # Bytes of one scheduler block across all ranks decide how many
        # offloaded blocks the host budget holds; capacity and the workers'
        # host pools are then both denominated in kernel blocks, sized
        # consistently (raiden never validates the pairing).
        bytes_per_block_all_ranks = (config.worker_kv_bytes_per_block *
                                     self.world_size)
        assert bytes_per_block_all_ranks > 0
        self.num_offloaded_blocks = int(
            int(cpu_bytes_to_use) // bytes_per_block_all_ranks)
        assert self.num_offloaded_blocks > 0, (
            "[kv-offload-v2] cpu_bytes_to_use too small for even one "
            f"offloaded block ({bytes_per_block_all_ranks} bytes)")
        self.capacity_kernel_blocks = (self.num_offloaded_blocks *
                                       self.gpu_block_size_factor)

        # Restart recovery: RAIDEN_SHM_KEY (recipe-set) moves the workers'
        # host KV pools and the store's metadata table into shared memory;
        # re-attach and directory recovery then happen inside the plain
        # raiden constructors on the next start. The model UID gates the
        # re-attach, so derive it from everything that must match for the
        # surviving bytes to be valid — an incompatible restart (different
        # model, dtype, geometry, world size, or offload capacity) fails
        # validation and cold-starts instead of serving garbage. Both the
        # scheduler and worker processes construct this spec before creating
        # any raiden object, so setting it here covers both sides.
        if os.environ.get("RAIDEN_SHM_KEY"):
            os.environ.setdefault(
                "RAIDEN_SHM_MODEL_UID",
                derive_shm_model_uid(
                    vllm_config,
                    kernel_block_size=kernel_block_size,
                    per_block_shape=per_block_shape,
                    kv_dtype=kv_dtype,
                    gpu_block_size=gpu_block_size,
                    world_size=self.world_size,
                    capacity_kernel_blocks=self.capacity_kernel_blocks))
            logger.info(
                "[kv-offload-v2] shm recovery enabled: RAIDEN_SHM_KEY=%s "
                "RAIDEN_SHM_MODEL_UID=%s", os.environ["RAIDEN_SHM_KEY"],
                os.environ["RAIDEN_SHM_MODEL_UID"])

        # Each DP engine owns a full V2 stack (store + in-process
        # controller) and DP-local engines share one host, so offset the
        # configured port by the local DP rank — with a static port every
        # engine past the first would fail its controller bind. The workers
        # of an engine see the same parallel config, so they derive the same
        # address.
        dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        if dp_rank is None:
            dp_rank = vllm_config.parallel_config.data_parallel_rank
        if (vllm_config.parallel_config.data_parallel_size > 1
                and os.environ.get("RAIDEN_SHM_KEY")):
            raise ValueError(
                "[kv-offload-v2] shm restart recovery (RAIDEN_SHM_KEY) is "
                "not supported with data parallelism: DP-local engines "
                "would collide on the per-host shm segment names")
        self.raiden_controller_port = dp_rank + int(
            self.extra_config.get("raiden_controller_port",
                                  _DEFAULT_CONTROLLER_PORT))
        self.raiden_controller_host = str(
            self.extra_config.get("raiden_controller_host", "127.0.0.1"))
        self.raiden_job_name = str(
            self.extra_config.get("raiden_job_name",
                                  f"vllm-offload-{config.engine_id}"))
        if dp_rank:
            self.raiden_job_name = f"{self.raiden_job_name}-dp{dp_rank}"

        self._manager: RaidenOffloadingManager | None = None
        self._worker: RaidenStoreOffloadingWorker | None = None

    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            self._manager = RaidenOffloadingManager(
                capacity_kernel_blocks=self.capacity_kernel_blocks,
                num_offloaded_blocks=self.num_offloaded_blocks,
                sub_blocks_per_key=self.gpu_block_size_factor,
                world_size=self.world_size,
                raiden_controller_port=self.raiden_controller_port,
                raiden_job_name=self.raiden_job_name,
            )
        return self._manager

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if self._worker is None:
            from vllm.distributed import parallel_state
            rank = parallel_state.get_world_group().rank_in_group
            controller_address = (f"{self.raiden_controller_host}:"
                                  f"{self.raiden_controller_port}")
            self._worker = RaidenStoreOffloadingWorker(
                kv_caches,
                vllm_config=self.vllm_config,
                kv_cache_config=self.kv_cache_config,
                host_blocks_to_allocate=self.capacity_kernel_blocks,
                controller_address=controller_address,
                rank=rank,
            )
        return self._worker
