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
"""Controller-driven KV cache offloading via Raiden `KVCacheStore`.

This module provides the scheduler-side offloading manager and worker
integration for Raiden-backed KV cache offloading. All directory tracking
(hash -> host slot), LRU eviction, pinning, and DMA transfers are managed
by Raiden's C++ `KVCacheStore`.

Workers register device KV buffers once with the controller (`KVCacheManager`)
and act as passive DMA endpoints for `save`/`load` operations initiated by
the scheduler.

Key components and mechanisms:

- Block granularity and coalescing: One store entry per physical kernel block.
  A logical GPU block maps to `gpu_block_size_factor` store entries with derived
  sub-hashes `OffloadKey ‖ sub_idx`. Contiguous device kernel blocks are fused
  by Raiden into single hardware DMAs.

- Model and parallelism support: Supports dense full-attention (single KV group)
  and hybrid attention+Mamba models on the TPU unified block pool. Supports
  Prefill Context Parallelism (PCP) across ranks and Data Parallelism (DP)
  with independent per-engine controller ports and job names.

- Two-phase admission: Uses `insert_and_lock_detailed` to pin pre-existing host
  entries without re-saving, while newly inserted entries trigger DMA saves.
  Displaced entries are surfaced to vLLM as eviction events.

- Fault tolerance and recovery: Failed saves retry once. Timed-out drains
  poison in-flight jobs to prevent committing torn bytes from reused blocks.

- Cross-instance KV sharing: Saved blocks are published to the global registry
  for cluster-wide reuse. Lookups resolve `REMOTE` hits, reserve host landing
  capacity, and issue `read_remote` directly into HBM, promoting successful
  fetches to `HOST_AND_HBM`.

See `raiden_connector.py` for connector wiring, device fencing, and echo channels.
"""
from __future__ import annotations

import enum
import hashlib
import os
import time
from dataclasses import dataclass, field

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,
                                        KVCacheConfig)
from vllm.v1.kv_offload.base import (CanonicalKVCaches, LoadStoreSpec,
                                     LookupResult, OffloadingEvent,
                                     OffloadingManager, OffloadingSpec,
                                     OffloadingWorker, OffloadKey,
                                     PrepareStoreOutput, ReqContext,
                                     RequestOffloadingContext, TransferResult)
from vllm.v1.kv_offload.config import OffloadingConfig

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Upper bound (seconds) on a scheduler-side drain of in-flight store/load
# jobs whose source/destination blocks are about to be reused.
_DRAIN_TIMEOUT_S = float(
    os.environ.get("VLLM_TPU_OFFLOAD_WAIT_TIMEOUT_S", "30"))

# A save whose transfer failed is retried this many times (a failed save
# leaves the entry HBM+pinned, so a second save() works). Retries are
# suppressed for drained jobs: a drain means the source blocks are about to
# be reused, so re-reading them would ship garbage.
_SAVE_RETRIES = int(os.environ.get("VLLM_TPU_OFFLOAD_SAVE_RETRIES", "1"))

_DEFAULT_CONTROLLER_PORT = 51515


class RaidenLoadStoreSpec(LoadStoreSpec):
    """Offload-side specification describing the ordered keys for a job.

    Device block IDs are bound later at submit time from the matching
    `GPULoadStoreSpec` (as they are not yet known during prepare).

    `pinned` is set to False if `prepare_load` lost the race against eviction
    before acquiring pins. The load job then fails cleanly at submit time,
    allowing vLLM to recompute the blocks.

    `displaced_keys` (loads): Local cache keys displaced when `prepare_load`
    reserves host landing capacity for remote fetches. Forwarded to
    `submit_load_job` so displacements commit only if the load succeeds.
    """

    def __init__(self,
                 keys: list[OffloadKey],
                 pinned: bool = True,
                 displaced_keys: list[OffloadKey] | None = None):
        self.keys = keys
        self.pinned = pinned
        self.displaced_keys = displaced_keys or []

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
    # Set by a timed-out drain: the job's blocks were handed back for reuse
    # while its transfers were still in flight, so even a store-reported
    # success may cover torn bytes. The finalizer treats the admission as
    # failed regardless of the poll verdict.
    poisoned: bool = False
    # Loads only: owning request and destination GPU block IDs, reported
    # through get_block_ids_with_load_errors on failure.
    req_id: str | None = None
    dst_gpu_block_ids: list[int] = field(default_factory=list)
    # Loads only: the sub-hashes fetched from a peer via read_remote
    # rather than the local host pool. Drives the remote poll.
    remote_sub_hashes: set[bytes] = field(default_factory=set)
    # Stores only: the sub-hashes actually saved (newly admitted HBM entries;
    # pre-existing HOST entries were pinned, not re-saved) and the remaining
    # retry budget for the failed subset.
    saved_sub_hashes: list[bytes] = field(default_factory=list)
    # Sub-hashes that failed: the polls append transfer failures (stores
    # retry exactly this subset), and launch failures pre-fill the batch.
    failed_sub_hashes: list[bytes] = field(default_factory=list)
    retries_left: int = 0
    # Keys whose entries this admission's insert displaced into the
    # eviction-candidate list (stores: the HBM admission; loads: the
    # host-landing install for a remote fetch). Committed (as eviction events +
    # advisory forget) only if the job SUCCEEDS — a failed job's rollback
    # restores the candidates.
    displaced_keys: list[OffloadKey] = field(default_factory=list)


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
        kernel_physical_blocks_capacity: int,
        offload_logical_blocks_capacity: int,
        gpu_block_size_factor: int,
        world_size: int,
        raiden_controller_port: int,
        raiden_job_name: str,
        global_registry_address: str = "",
        raiden_orchestrator_address: str = "",
        store_server_ip: str = "",
        key_namespace: bytes = b"",
        # Test overrides: allow injecting mock store instance and type bindings.
        store: object | None = None,
        store_types: object | None = None,
    ):
        assert kernel_physical_blocks_capacity == (
            offload_logical_blocks_capacity * gpu_block_size_factor)
        self._capacity_keys = offload_logical_blocks_capacity
        self._gpu_block_size_factor = gpu_block_size_factor
        self._world_size = world_size
        self._key_namespace = key_namespace
        # Cross-instance sharing: publishing blocks to the global registry
        # requires a peer-reachable store_server_ip.
        self._enable_global = bool(global_registry_address)
        if self._enable_global and not store_server_ip:
            # KVCacheStore requires a dialable store_server_ip to advertise
            # reachable endpoints to the global registry. Falling back to
            # localhost (127.0.0.1) would publish an address peers cannot dial.
            # Require the recipe to configure a valid Pod IP explicitly.
            raise ValueError(
                "TPURaidenOffloadingConnector: global registry configured without "
                "store_server_ip: peers could never dial this replica. "
                "Set store_server_ip to the pod IP in "
                "kv_connector_extra_config.")
        if self._enable_global and not os.environ.get("PYTHONHASHSEED"):
            # vLLM seeds its block-hash chain from os.urandom when
            # PYTHONHASHSEED is unset, so unseeded replicas build disjoint
            # hash chains and every cross-instance lookup misses — a silent
            # zero-hit run with no error anywhere.
            raise ValueError(
                "TPURaidenOffloadingConnector: global_registry_address requires "
                "PYTHONHASHSEED to be set (to the same value on every "
                "replica): unseeded engines derive disjoint block hashes "
                "and cross-instance lookups can never hit.")

        if store is None:
            # Load shared C++ libraries before importing the native extension.
            from tpu_raiden.api.torch import torch_tpu_common_loader
            torch_tpu_common_loader.load_torch_tpu_common()
            from tpu_raiden.frameworks.torch import \
                _tpu_raiden_torch as store_types
            store = store_types.KVCacheStore(
                capacity=kernel_physical_blocks_capacity,
                raiden_id=store_types.RaidenId(raiden_job_name, "0",
                                               "kv_cache", 0),
                # Transfer fanout is dynamically driven by registered workers.
                num_shards=1,
                shard_size_bytes=1,
                raiden_controller_port=raiden_controller_port,
                global_registry_address=global_registry_address,
                raiden_orchestrator_address=raiden_orchestrator_address,
                # Local-only runs default to 127.0.0.1 to satisfy C++ non-empty check.
                store_server_ip=store_server_ip or "127.0.0.1",
            )
            logger.info(
                "TPURaidenOffloadingConnector: KVCacheStore up: capacity=%d kernel blocks "
                "(%d offloaded blocks x %d), namespace=%s, controller at "
                "%s%s", kernel_physical_blocks_capacity,
                offload_logical_blocks_capacity, gpu_block_size_factor,
                key_namespace.hex(), store.raiden_controller_address,
                (f", published at {store.store_server_address or '<nothing>'}"
                 f" in registry {global_registry_address}"
                 if self._enable_global else ""))
        else:
            # Injected store (tests): the caller supplies the module holding
            # BlockStatus / RaidenBlockID so tpu_raiden isn't imported.
            assert store_types is not None
        self._store = store
        self._block_status = store_types.BlockStatus
        self._block_id_cls = store_types.RaidenBlockID
        self._raiden_id = self._store.raiden_id

        # Ensure the installed `tpu_raiden` binary supports detailed admission
        # and eviction tracking (`insert_and_lock_detailed`). Fail fast at startup
        # to prevent silent eviction loss.
        if not hasattr(self._store, "insert_and_lock_detailed"):
            raise RuntimeError(
                "TPURaidenOffloadingConnector: the tpu_raiden wheel lacks "
                "insert_and_lock_detailed; upgrade the pinned wheel.")

        # Registration gate: offloading operations remain disabled until all
        # worker ranks have registered device buffers with the controller,
        # preventing partial or corrupted transfers.
        self._gate_open = False

        # Advisory admission state: filters `prepare_store` to prevent redundant
        # admissions for keys already stored or in-flight.
        self._stored_keys: set[OffloadKey] = set()
        self._inflight_store_keys: set[OffloadKey] = set()

        # Peer location slices cached from global lookups and consumed during
        # `prepare_load` to reserve local host landing slots.
        self._remote_slices: dict[bytes, object] = {}

        self._admissions: dict[int, Admission] = {}
        self._hash_to_admission: dict[bytes, Admission] = {}
        self._finished: list[FinishedJob] = []
        # Eviction events surfaced via `take_events()` when store admissions
        # displace existing cache entries.
        self._pending_events: list[OffloadingEvent] = []

    # -- sub-hash / block-id expansion ----------------------------------

    def _sub_hashes(self, key: OffloadKey) -> list[bytes]:
        # The namespace prefix makes registry keys byte-distinct
        # across incompatible engines; locally it is inert.
        return [
            self._key_namespace + key + i.to_bytes(4, "big")
            for i in range(self._gpu_block_size_factor)
        ]

    def _key_of(self, sub_hash: bytes) -> OffloadKey:
        return OffloadKey(sub_hash[len(self._key_namespace):-4])

    def _expand_keys(self, keys: list[OffloadKey]) -> list[bytes]:
        out: list[bytes] = []
        for key in keys:
            out.extend(self._sub_hashes(key))
        return out

    def _expand_gpu_blocks(self, gpu_block_ids: list[int]) -> list[int]:
        f = self._gpu_block_size_factor
        return [b * f + i for b in gpu_block_ids for i in range(f)]

    def _sub_hash_status(self, sub_hash: bytes):
        """Peek one sub-hash's directory status (None when absent)."""
        matches = self._store.lookup([sub_hash])
        return matches[0][1].status if matches else None

    def _displaced_keys_of(self, raw_displaced) -> list[OffloadKey]:
        """Unique OffloadKeys of a detailed insert result's displacements,
        in displacement order (sub-hashes collapse back to their key)."""
        displaced: list[OffloadKey] = []
        seen: set[OffloadKey] = set()
        for sub_hash, _ in raw_displaced:
            key = self._key_of(sub_hash)
            if key not in seen:
                seen.add(key)
                displaced.append(key)
        return displaced

    # -- registration gate ----------------------------------------------

    def is_gate_open(self) -> bool:
        if self._gate_open:
            return True
        num = self._store.num_registered_workers()
        if num > self._world_size:
            raise RuntimeError(
                f"TPURaidenOffloadingConnector: {num} workers registered with the offload "
                f"controller, expected at most world_size={self._world_size}."
                " A stale or foreign worker registration would corrupt the "
                "save fanout.")
        if num == self._world_size:
            self._gate_open = True
            logger.info(
                "TPURaidenOffloadingConnector: registration gate open: %d/%d workers "
                "registered; stores and loads enabled.", num, self._world_size)
        return self._gate_open

    # -- OffloadingManager contract --------------------------------------

    def on_new_request(self,
                       req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        if not self.is_gate_open():
            return LookupResult.MISS
        if key in self._inflight_store_keys:
            return LookupResult.HIT_PENDING
        sub_hashes = self._sub_hashes(key)
        # Non-mutating peek across all sub-hashes (no LRU recency update).
        # With the global registry enabled, local misses query remote peer slices.
        # If the registry is unreachable or times out, the error is swallowed and
        # lookup safely falls back to local-only caching without failing requests.
        allow_remote = self._enable_global
        matches = self._store.lookup(sub_hashes, enable_global=allow_remote)
        if len(matches) < len(sub_hashes):
            self._stored_keys.discard(key)
            return LookupResult.MISS
        all_local = True
        remote_slices: dict[bytes, object] = {}
        for sub_hash, block in matches:
            status = block.status
            if status in (self._block_status.HOST,
                          self._block_status.HOST_AND_HBM):
                continue
            if status == self._block_status.HBM:
                # Save still in flight (should be covered by the in-flight
                # set above; kept as a consistency net).
                return LookupResult.HIT_PENDING
            if status == self._block_status.REMOTE and allow_remote:
                if sub_hash in self._hash_to_admission:
                    # A read_remote for this sub-hash is already in flight;
                    # a second launch on the same hash must not race it.
                    return LookupResult.HIT_PENDING
                all_local = False
                remote_slices[sub_hash] = block
                continue
            return LookupResult.MISS
        if not all_local:
            self._remote_slices.update(remote_slices)
            return LookupResult.HIT
        # Record newly observed local hits in the advisory filter (e.g. entries
        # restored by a rollback), ensuring touch() and prepare_store() recognize
        # them as locally resident.
        self._stored_keys.add(key)
        return LookupResult.HIT

    def touch(self, keys, req_context: ReqContext) -> None:
        if not self._gate_open:
            return
        # Pin and immediate release moves entries to the Most Recently Used
        # (MRU) position, refreshing their LRU order. The advisory filter
        # prevents unnecessary probes for keys not stored locally.
        for key in keys:
            if key not in self._stored_keys:
                continue
            sub_hashes = self._sub_hashes(key)
            if self._store.pin(sub_hashes):
                self._store.release(sub_hashes)

    def prepare_load(self, keys, req_context: ReqContext) -> LoadStoreSpec:
        keys = list(keys)
        sub_hashes = self._expand_keys(keys)
        # Install and pin remote sub-hashes via insert_and_lock(on_host=True),
        # which records peer mappings and reserves host landing capacity.
        # Entries that became local in the interim are pinned in-place;
        # submit_load_job partitions by live status and loads them locally.
        remote_hashes: list[bytes] = []
        remote_slices: list[object] = []
        local_hashes: list[bytes] = []
        for sub_hash in sub_hashes:
            block = self._remote_slices.pop(sub_hash, None)
            if block is not None:
                remote_hashes.append(sub_hash)
                remote_slices.append(block)
            else:
                local_hashes.append(sub_hash)
        if local_hashes and not self._store.pin(local_hashes):
            # Lost race against eviction between lookup and pin (Raiden's pin is
            # all-or-nothing, so no entries are held). Return an unpinned spec
            # so submit_load_job fails the job cleanly and triggers recomputation.
            logger.warning(
                "TPURaidenOffloadingConnector: prepare_load lost %d sub-blocks for request "
                "%s to eviction between lookup and pin; the load job will "
                "fail cleanly and the blocks recompute", len(local_hashes),
                req_context.req_id)
            return RaidenLoadStoreSpec(keys, pinned=False)
        displaced_keys: list[OffloadKey] = []
        if remote_hashes:
            # Reserving host landing capacity may displace older LRU entries;
            # tracked displacements commit permanently only upon load success.
            installed, _existing, _inserted, raw_displaced = (
                self._store.insert_and_lock_detailed(remote_hashes,
                                                     remote_slices,
                                                     on_host=True))
            if installed:
                displaced_keys = self._displaced_keys_of(raw_displaced)
            if not installed:
                # Host landing capacity is exhausted under active pin pressure.
                # Unwind pre-existing local pins and return an unpinned spec to
                # fail the job cleanly, allowing the blocks to recompute.
                if local_hashes:
                    self._store.release(local_hashes)
                logger.warning(
                    "TPURaidenOffloadingConnector: prepare_load could not reserve landing "
                    "capacity for %d remote sub-blocks (request %s); the "
                    "load job will fail cleanly and the blocks recompute",
                    len(remote_hashes), req_context.req_id)
                return RaidenLoadStoreSpec(keys, pinned=False)
        return RaidenLoadStoreSpec(keys, displaced_keys=displaced_keys)

    def complete_load(self, keys, req_context: ReqContext) -> None:
        # Intentionally a no-op: _finalize() releases pinned entries upon
        # polled DMA completion. Releasing here would cause double-unpinning
        # and risk dropping concurrent locks on shared entries.
        return

    def prepare_store(self, keys,
                      req_context: ReqContext) -> PrepareStoreOutput | None:
        if not self.is_gate_open():
            return None
        keys_to_store = [
            key for key in keys if key not in self._stored_keys
            and key not in self._inflight_store_keys
        ]
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
        # Intentionally a no-op: _finalize() drives store state transitions
        # and unpinning upon polled hardware completion.
        return

    def take_events(self):
        events = self._pending_events
        self._pending_events = []
        return events

    def reset_cache(self) -> None:
        # Upstream's default hook is a silent no-op, which would falsely report
        # success to POST /reset_prefix_cache?reset_external=true while the
        # Raiden store continues serving cached entries. Explicitly raise
        # NotImplementedError since external cache reset is unsupported.
        raise NotImplementedError(
            "TPURaidenOffloadingConnector: external cache reset is not supported "
            "by the Raiden offload store.")

    def has_pending_work(self) -> bool:
        return bool(self._admissions)

    def shutdown(self) -> None:
        if self._admissions:
            self.drain_jobs(set(self._admissions.keys()))

    # -- job submission bridge (called by the connector scheduler) --------

    def submit_store_job(self, job_id: int, keys: list[OffloadKey],
                         gpu_block_ids: list[int]) -> None:
        """Admit and launch a store job after all worker ranks acknowledge the save fence.

        Atomically admits the batch via insert_and_lock_detailed: pins pre-existing
        entries in-place, reserves capacity, displaces older LRU entries revertibly,
        and rolls back on failure. Dispatches save() for newly inserted entries;
        completion is tracked asynchronously by the finalizer.
        """
        assert len(keys) == len(gpu_block_ids), (len(keys), len(gpu_block_ids))
        sub_hashes = self._expand_keys(keys)
        device_ids = self._expand_gpu_blocks(gpu_block_ids)
        slices = [
            self._block_id_cls(
                raiden_id=self._raiden_id,
                host_block_id=-1,
                device_block_id=dev_id,
                status=self._block_status.HBM,
            ) for dev_id in device_ids
        ]
        # Only save newly inserted HBM entries. Pre-existing HOST entries are
        # pinned in-place without re-saving; passing them to save() would fail
        # Raiden's HBM-status precondition.
        displaced_keys: list[OffloadKey] = []
        admitted, _existing, inserted, raw_displaced = (
            self._store.insert_and_lock_detailed(sub_hashes,
                                                 slices,
                                                 on_host=False))
        if admitted:
            to_save = list(inserted)
            displaced_keys = self._displaced_keys_of(raw_displaced)
        if not admitted:
            # Atomically rolled back inside Raiden; no DMA save was issued.
            logger.warning(
                "TPURaidenOffloadingConnector: store job %d: insert_and_lock rejected %d "
                "sub-blocks (capacity pressure with pinned entries)", job_id,
                len(sub_hashes))
            self._inflight_store_keys.difference_update(keys)
            self._finished.append(
                FinishedJob(job_id=job_id, op=AdmissionOp.STORE,
                            success=False))
            return
        if not to_save:
            # All sub-hashes are already host-resident: unpin and finish successfully
            # without issuing redundant DMA transfers.
            self._store.release(sub_hashes)
            self._inflight_store_keys.difference_update(keys)
            self._stored_keys.update(keys)
            self._commit_displacement(displaced_keys)
            self._finished.append(
                FinishedJob(job_id=job_id, op=AdmissionOp.STORE, success=True))
            return
        if not self._store.save(to_save):
            # Synchronous launch failure: finalize immediately to revert admitted
            # entries and restore displaced candidates.
            logger.error(
                "TPURaidenOffloadingConnector: store job %d: save() launch failed for %d "
                "sub-blocks", job_id, len(to_save))
            self._finalize(
                Admission(
                    job_id=job_id,
                    op=AdmissionOp.STORE,
                    keys=keys,
                    sub_hashes=sub_hashes,
                    pending=set(),
                    failed=True,
                    failed_sub_hashes=list(to_save),
                    displaced_keys=displaced_keys,
                ))
            return
        admission = Admission(
            job_id=job_id,
            op=AdmissionOp.STORE,
            keys=keys,
            sub_hashes=sub_hashes,
            pending=set(to_save),
            saved_sub_hashes=to_save,
            retries_left=_SAVE_RETRIES,
            displaced_keys=displaced_keys,
        )
        self._admissions[job_id] = admission
        for sub_hash in to_save:
            self._hash_to_admission[sub_hash] = admission

    def _commit_displacement(self, displaced_keys: list[OffloadKey]) -> None:
        """A successful admission's displacements are permanent: surface them
        as eviction events and forget them in the advisory state, so the
        scheduler stops treating the keys as stored and a later re-offer
        re-admits them. (A failed admission's rollback restores the
        candidates instead — never commit those.)"""
        if not displaced_keys:
            return
        self._stored_keys.difference_update(displaced_keys)
        self._pending_events.append(
            OffloadingEvent(keys=list(displaced_keys),
                            medium="raiden_host",
                            removed=True))

    def submit_load_job(
            self,
            job_id: int,
            keys: list[OffloadKey],
            gpu_block_ids: list[int],
            req_id: str,
            pinned: bool = True,
            displaced_keys: list[OffloadKey] | None = None) -> None:
        """Launch a load job; entries were pinned in prepare_load.

        `displaced_keys` carries prepare_load's host-landing displacements;
        they commit only if the job succeeds.
        """
        assert len(keys) == len(gpu_block_ids), (len(keys), len(gpu_block_ids))
        displaced_keys = list(displaced_keys or [])
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
        # Partition by live directory status: REMOTE sub-hashes (installed by
        # prepare_load's insert_and_lock) are fetched from their peer with
        # read_remote, everything else comes from the local host pool. The
        # live peek — not the lookup-time stash — decides, so a sub-hash that
        # became local in between is simply loaded.
        remote_hashes: list[bytes] = []
        remote_device_ids: list[int] = []
        local_hashes: list[bytes] = []
        local_device_ids: list[int] = []
        for sub_hash, dev_id in zip(sub_hashes, device_ids):
            if self._sub_hash_status(sub_hash) == self._block_status.REMOTE:
                remote_hashes.append(sub_hash)
                remote_device_ids.append(dev_id)
            else:
                local_hashes.append(sub_hash)
                local_device_ids.append(dev_id)

        def _fail_launch(what: str, failed_hashes: list[bytes],
                         in_flight: list[bytes]) -> None:
            # failed_hashes is the batch whose launch failed.
            logger.error(
                "TPURaidenOffloadingConnector: load job %d (req %s): %s launch failed "
                "(%d local, %d remote sub-blocks)", job_id, req_id, what,
                len(local_hashes), len(remote_hashes))
            admission = Admission(
                job_id=job_id,
                op=AdmissionOp.LOAD,
                keys=keys,
                sub_hashes=sub_hashes,
                pending=set(in_flight),
                failed=True,
                req_id=req_id,
                dst_gpu_block_ids=list(gpu_block_ids),
                remote_sub_hashes=set(remote_hashes),
                failed_sub_hashes=list(failed_hashes),
                displaced_keys=displaced_keys,
            )
            if in_flight:
                # The other launch already succeeded: the job must stay
                # non-terminal until its transfer resolves, then the
                # finalizer runs the failure cleanup exactly once.
                self._admissions[job_id] = admission
                for sub_hash in in_flight:
                    self._hash_to_admission[sub_hash] = admission
                return
            # Nothing in flight: immediately terminal. The finalizer keeps
            # HOST entries intact, deletes freshly installed REMOTE entries
            # (landing blocks freed, the registry keeps the peer mapping),
            # matching a polled failure.
            self._finalize(admission)

        if local_hashes and not self._store.load(local_hashes,
                                                 local_device_ids):
            _fail_launch("load()", local_hashes, [])
            return
        if remote_hashes and not self._store.read_remote(
                remote_hashes, remote_device_ids):
            _fail_launch("read_remote()", remote_hashes, local_hashes)
            return
        admission = Admission(
            job_id=job_id,
            op=AdmissionOp.LOAD,
            keys=keys,
            sub_hashes=sub_hashes,
            pending=set(sub_hashes),
            req_id=req_id,
            dst_gpu_block_ids=list(gpu_block_ids),
            remote_sub_hashes=set(remote_hashes),
            displaced_keys=displaced_keys,
        )
        self._admissions[job_id] = admission
        for sub_hash in sub_hashes:
            self._hash_to_admission[sub_hash] = admission

    def on_store_job_cancelled(self, keys: list[OffloadKey]) -> None:
        """A fence-pending (never admitted) store job was cancelled."""
        self._inflight_store_keys.difference_update(keys)

    def is_job_launched(self, job_id: int) -> bool:
        """Whether a job has been admitted+launched and is not yet terminal."""
        return job_id in self._admissions

    # -- finalizer --------------------------------------------------------

    def _finalize(self, admission: Admission) -> None:
        # Launch failures finalize an admission that was never registered,
        # so the pop tolerates absence; both paths share this one terminal
        # transition (cleanup, finished-job report).
        self._admissions.pop(admission.job_id, None)
        for sub_hash in admission.sub_hashes:
            self._hash_to_admission.pop(sub_hash, None)
        success = not (admission.failed or admission.poisoned)
        if admission.op == AdmissionOp.STORE:
            self._inflight_store_keys.difference_update(admission.keys)
            if success:
                # Unpin the exact admitted batch; entries become evictable.
                self._store.release(admission.sub_hashes)
                self._stored_keys.update(admission.keys)
                self._commit_displacement(admission.displaced_keys)
            else:
                # Unpin and delete the newly admitted HBM entries, restoring
                # displaced candidates.
                self._store.release_and_delete(admission.sub_hashes)
            self._finished.append(
                FinishedJob(job_id=admission.job_id,
                            op=AdmissionOp.STORE,
                            success=success))
        else:
            # Release pins acquired in prepare_load:
            # - On success: release() unpins entries (remote fetches are now
            #   HOST_AND_HBM) and permanently commits host-landing displacements.
            # - On failure: release_and_delete() frees unlanded REMOTE buffers
            #   while preserving valid HOST entries; destination blocks recompute.
            if success:
                self._store.release(admission.sub_hashes)
                self._commit_displacement(admission.displaced_keys)
            else:
                self._store.release_and_delete(admission.sub_hashes)
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
        has_stores = any(admission.op == AdmissionOp.STORE
                         for admission in self._admissions.values())
        has_loads = any(admission.op == AdmissionOp.LOAD and any(
            sub_hash not in admission.remote_sub_hashes
            for sub_hash in admission.pending)
                        for admission in self._admissions.values())
        has_remote = any(admission.op == AdmissionOp.LOAD and any(
            sub_hash in admission.remote_sub_hashes
            for sub_hash in admission.pending)
                         for admission in self._admissions.values())
        results: list[tuple[list[bytes], list[bytes]]] = []
        if has_stores:
            done, failed, _pending = self._store.poll_save_status()
            results.append((done, failed))
        if has_loads:
            done, failed, _pending = self._store.poll_load_status()
            results.append((done, failed))
        if has_remote:
            done, failed, _pending = self._store.poll_remote_read_status()
            results.append((done, failed))
        # Keyed by job_id: a mixed batch whose last hashes resolve in the
        # same poll (some done, some failed) must finalize exactly once.
        newly_terminal: dict[int, Admission] = {}
        for done, failed in results:
            for sub_hash in failed:
                admission = self._hash_to_admission.get(sub_hash)
                if admission is None:
                    continue
                admission.failed = True
                admission.failed_sub_hashes.append(sub_hash)
                admission.pending.discard(sub_hash)
                if not admission.pending:
                    newly_terminal[admission.job_id] = admission
            for sub_hash in done:
                admission = self._hash_to_admission.get(sub_hash)
                if admission is None:
                    continue
                admission.pending.discard(sub_hash)
                if not admission.pending:
                    newly_terminal[admission.job_id] = admission
        for admission in newly_terminal.values():
            if self._maybe_retry_save(admission):
                continue
            self._finalize(admission)

    def _maybe_retry_save(self, admission: Admission) -> bool:
        """Re-issue the failed subset of a store job's save, at most
        `retries_left` times.

        Safe because a failed save leaves its entries HBM + pinned with the
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
                "TPURaidenOffloadingConnector: store job %d: save() retry launch failed "
                "for %d sub-blocks; job fails", admission.job_id,
                len(retry_hashes))
            return False
        logger.warning(
            "TPURaidenOffloadingConnector: store job %d: retrying save of %d failed "
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
        """Drain in-flight DMA transfers to safely reclaim HBM blocks before reuse.

        Synchronously waits until the specified launched jobs complete (bounded
        by timeout) so active transfers finish before subsequent forward passes
        overwrite the source memory, preventing torn reads and cache corruption.

        On timeout, in-flight admissions are marked as poisoned rather than
        blocking indefinitely. Because Raiden lacks an abort API, poisoned jobs
        remain live until hardware completion, but the finalizer treats them as
        terminal failures to prevent committing corrupted/torn bytes from reused
        blocks. Poisoned jobs are considered drained on subsequent calls.
        """
        # Suppress retries for draining jobs: source blocks are being reclaimed,
        # so subsequent reads would transfer invalid/overwritten data.
        for job_id in job_ids:
            admission = self._admissions.get(job_id)
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
            live = [
                job_id for job_id in job_ids if job_id in self._admissions
                and not self._admissions[job_id].poisoned
            ]
            if not live:
                return drained
            if time.perf_counter() >= deadline:
                for job_id in live:
                    self._admissions[job_id].poisoned = True
                logger.error(
                    "TPURaidenOffloadingConnector: drain timed out after %.1fs; poisoned "
                    "jobs %s (%d keys): their transfers are still in flight "
                    "while their blocks are reused, so they will finalize "
                    "as failures and uncommitted entries revert",
                    _DRAIN_TIMEOUT_S, sorted(live),
                    sum(len(self._admissions[job_id].keys) for job_id in live))
                return drained
            time.sleep(0.0005)


def derive_offload_namespace(
    vllm_config: VllmConfig,
    *,
    kernel_block_size: int,
    per_block_shape: tuple[int, ...],
    kv_dtype: object,
    gpu_block_size: int,
    world_size: int,
    num_kv_cache_groups: int,
    num_kv_cache_tensors: int,
    cp_geometry: tuple[int, ...] = ()) -> bytes:
    """Derive a deterministic compatibility namespace prefixed to every store and registry key.

    Ensures cache keys never collide across incompatible engine deployments by
    hashing all configuration parameters that alter binary KV cache layout:
    - Model identity: model name, revision, quantization.
    - KV cache geometry: KV data type, kernel block size, per-block tensor shape,
      GPU block size, and cache group/tensor counts.
    - Distributed topology: world size (TP/PP) and context parallelism geometry
      (pcp_size, dcp_size, interleave_size) which alters rank-local token placement.
    - Hash scheme: prefix caching algorithm and namespace version salt.

    Returns:
        A 16-byte SHA-256 digest unique to this engine configuration.
    """
    model_config = vllm_config.model_config
    material = repr((model_config.model, model_config.revision,
                     model_config.quantization, str(kv_dtype),
                     kernel_block_size, per_block_shape, gpu_block_size,
                     world_size, num_kv_cache_groups, num_kv_cache_tensors,
                     vllm_config.cache_config.prefix_caching_hash_algo,
                     "tpu-raiden-offload-ns1") + cp_geometry)
    return hashlib.sha256(material.encode()).digest()[:16]


def resolve_kernel_geometry(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> tuple[int, tuple[int, ...], object, int]:
    """Resolve physical TPU kernel geometry: (kernel_block_size, per_block_shape, kv_dtype, gpu_block_size).

    Shared between the scheduler (capacity sizing, sub-hash factor) and worker ranks
    (device view reconstruction) to guarantee exact geometry parity.

    Supported architectures:
    - Dense models: Defined directly by the single group's full-attention spec.
    - Hybrid (Attention + Mamba) models: Requires TPU's unified block pool, where all
      groups share uniform, attention-sized page rows. Attention geometry defines the
      common row unit, allowing the dense sub-hash expansion to apply uniformly.
    """
    from vllm.v1.kv_cache_interface import AttentionSpec
    from vllm.v1.worker.utils import select_common_block_size

    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    groups = kv_cache_config.kv_cache_groups
    assert groups, "TPURaidenOffloadingConnector: no KV cache groups"
    if len(groups) == 1:
        spec0 = groups[0].kv_cache_spec
        assert isinstance(spec0, FullAttentionSpec), (
            "TPURaidenOffloadingConnector: single-group models must be full-attention; "
            f"got {type(spec0).__name__}")
    else:
        # In hybrid models, group 0 may be a Mamba layer; filter for attention groups.
        from vllm_torchtpu.platforms.tpu_block_size_utils import \
            unified_kv_layout_enabled
        assert unified_kv_layout_enabled(vllm_config), (
            "TPURaidenOffloadingConnector: hybrid models require the unified block pool "
            "(TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1)")
        attn_specs = [
            g.kv_cache_spec for g in groups
            if isinstance(g.kv_cache_spec, AttentionSpec)
        ]
        assert attn_specs, (
            "TPURaidenOffloadingConnector: hybrid model has no attention KV group"
        )
        spec0 = attn_specs[0]
        assert all(
            (s.block_size, s.num_kv_heads, s.head_size, s.dtype,
             s.page_size_bytes) == (spec0.block_size, spec0.num_kv_heads,
                                    spec0.head_size, spec0.dtype,
                                    spec0.page_size_bytes) for s in attn_specs
        ), ("TPURaidenOffloadingConnector: attention groups disagree on geometry"
            )
        # Ensure all groups share the unified pool's padded page size.
        assert all(
            g.kv_cache_spec.page_size_bytes == spec0.page_size_bytes
            for g in groups
        ), ("TPURaidenOffloadingConnector: unified pool groups must share one "
            "padded page size")

    attn_backend = TpuPlatform._find_non_ssm_backend(vllm_config)
    if attn_backend is None:
        # In the EngineCore process, the model is not instantiated, so
        # _find_non_ssm_backend returns None. Resolve the attention backend
        # directly via get_attn_backend using the model configuration.
        from vllm.config import set_current_vllm_config
        from vllm.v1.attention.selector import get_attn_backend
        model_config = vllm_config.model_config
        with set_current_vllm_config(vllm_config):
            attn_backend = get_attn_backend(
                head_size=model_config.get_head_size(),
                dtype=model_config.dtype,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                use_mla=model_config.use_mla,
            )
    assert attn_backend is not None, (
        "TPURaidenOffloadingConnector: requires a non-SSM attention backend")
    assert not attn_backend.is_ssm(), (
        "TPURaidenOffloadingConnector: resolved attention backend is SSM: "
        f"{attn_backend.get_name()}")
    kernel_block_size = select_common_block_size(spec0.block_size,
                                                 [attn_backend])
    full_5d_shape = attn_backend.get_kv_cache_shape(
        num_blocks=1,
        block_size=kernel_block_size,
        num_kv_heads=spec0.num_kv_heads,
        head_size=spec0.head_size,
        cache_dtype_str=spec0.dtype,
    )
    per_block_shape = tuple(full_5d_shape[1:])
    return kernel_block_size, per_block_shape, spec0.dtype, spec0.block_size


class RaidenStoreOffloadingWorker(OffloadingWorker):
    """Worker-side endpoint that registers local TPU HBM device buffers with the Controller.

    Acts as a passive DMA target for controller-driven transfers. Offload jobs are
    managed centrally by the controller and never dispatched to workers directly;
    the worker submit hooks assert fail loudly if invoked.

    Registration is deferred: worker initialization occurs before the scheduler
    creates the controller server. Worker ranks register lazily on the first
    inference step via ensure_registered() once the controller is running.
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

        # Rebuild typed tensor views shaped by physical kernel blocks from the
        # underlying storage, ensuring leading dimensions match DMA tile layouts.
        device_tensors: list[torch.Tensor] = []
        for kv_cache_tensor in kv_caches.tensors:
            # Verify that one canonical scheduler page equals exactly
            # gpu_block_size_factor physical kernel rows. For hybrid models
            # using the unified pool, this confirms that all group pages are
            # uniformly padded and fungible across pool rows.
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

        # Hold strong references to ensure PyTorch tensors remain allocated
        # while Raiden C++ retains raw pointers to their underlying storage.
        self._device_tensors = [[t] for t in device_tensors]
        self._host_blocks_to_allocate = host_blocks_to_allocate
        self._controller_address = controller_address
        self._rank = rank
        self._mgr = None

    def ensure_registered(self) -> None:
        """Idempotently register this worker rank's HBM buffers with the Raiden controller.

        Fails fast if registration fails to prevent partial transfer fanouts where
        unregistered ranks are skipped while stores falsely report success.
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
            "TPURaidenOffloadingConnector: rank %d registered with offload controller at "
            "%s (%d host kernel blocks)", self._rank, self._controller_address,
            self._host_blocks_to_allocate)

    def submit_store(self, job_id, src_spec, dst_spec) -> bool:
        raise AssertionError(
            "TPURaidenOffloadingConnector: submit_store reached a worker; transfers are "
            "controller-driven and jobs must be stripped scheduler-side")

    def submit_load(self, job_id, src_spec, dst_spec) -> bool:
        raise AssertionError(
            "TPURaidenOffloadingConnector: submit_load reached a worker; transfers are "
            "controller-driven and jobs must be stripped scheduler-side")

    def get_finished(self) -> list[TransferResult]:
        return []

    def wait(self, job_ids: set[int]) -> None:
        # No-op on workers: jobs are drained centrally on the controller.
        return

    def shutdown(self) -> None:
        self._mgr = None


class TPURaidenStoreOffloadingSpec(OffloadingSpec):
    """Configuration specification wiring the Raiden offloading manager and per-rank workers.

    Constructed directly by TPURaidenOffloadingConnector to encapsulate full engine
    and KV cache configurations, compute hardware block capacities, and configure
    distributed registry settings.
    """

    def __init__(self, config: OffloadingConfig, vllm_config: VllmConfig,
                 kv_cache_config: KVCacheConfig):
        super().__init__(config)
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config

        assert config.cache.blocks_per_chunk == 1, (
            "TPURaidenOffloadingConnector: requires blocks_per_chunk == 1 (one OffloadKey "
            "= one scheduler GPU block); got "
            f"{config.cache.blocks_per_chunk}")
        parallel = config.parallel
        assert parallel.dcp_size == 1, (
            "TPURaidenOffloadingConnector: does not support decode context parallelism"
        )
        # Count all ranks in the distributed execution group (TP * PP * PCP)
        # for registration barriers, save fanout, and total host capacity sizing.
        self.world_size = parallel.world_size

        # Ensure model-level hybrid flag and KV cache group specs agree before
        # resolving attention kernel geometry.
        from vllm.v1.kv_cache_interface import MambaSpec
        self.is_hybrid_model = vllm_config.model_config.is_hybrid
        has_mamba_group = any(
            isinstance(g.kv_cache_spec, MambaSpec)
            for g in kv_cache_config.kv_cache_groups)
        assert self.is_hybrid_model == has_mamba_group, (
            "TPURaidenOffloadingConnector: model_config.is_hybrid="
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

        # Verify PCP scaling invariant: scheduler chunks at the logical all-PCP-rank
        # block span, whereas KV cache specs are rank-local. Ensure tokens_per_block
        # scales by pcp_size so OffloadKey spans match scheduler blocks.
        for group_config, group in zip(config.groups,
                                       kv_cache_config.kv_cache_groups):
            if isinstance(group.kv_cache_spec, AttentionSpec):
                assert group_config.tokens_per_block == (
                    group.kv_cache_spec.block_size * parallel.pcp_size
                ), ("TPURaidenOffloadingConnector: offloading group tokens_per_block="
                    f"{group_config.tokens_per_block} != rank-local "
                    f"block_size={group.kv_cache_spec.block_size} x "
                    f"pcp={parallel.pcp_size}; the PCP-aware "
                    "OffloadingConfig build patch did not run")

        # Include context parallelism dimensions in key namespace only when CP
        # is active, preserving namespace compatibility for standard pcp=1 runs.
        cp_geometry: tuple[int, ...] = ()
        if parallel.pcp_size > 1:
            cp_geometry = (
                parallel.pcp_size, parallel.dcp_size,
                vllm_config.parallel_config.cp_kv_cache_interleave_size)

        # Calculate host block capacity from total bytes per logical block across
        # all worker ranks, sizing both manager and worker pools consistently.
        bytes_per_block_all_ranks = (config.worker_kv_bytes_per_block *
                                     self.world_size)
        assert bytes_per_block_all_ranks > 0
        self.offload_logical_blocks_capacity = int(
            int(cpu_bytes_to_use) // bytes_per_block_all_ranks)
        assert self.offload_logical_blocks_capacity > 0, (
            "TPURaidenOffloadingConnector: cpu_bytes_to_use too small for even one "
            f"offloaded block ({bytes_per_block_all_ranks} bytes)")
        self.kernel_physical_blocks_capacity = (
            self.offload_logical_blocks_capacity * self.gpu_block_size_factor)

        # Cross-instance KV sharing: configured via extra_config. Setting
        # global_registry_address enables remote lookups and RDMA fetches;
        # store_server_ip advertises the pod IP that peers dial for remote reads.
        self.global_registry_address = str(
            self.extra_config.get("global_registry_address", ""))
        self.raiden_orchestrator_address = str(
            self.extra_config.get("raiden_orchestrator_address", ""))
        self.store_server_ip = str(self.extra_config.get(
            "store_server_ip", ""))

        # Generate the compatibility namespace hash to isolate global registry
        # entries across incompatible engine configurations.
        self.key_namespace = derive_offload_namespace(
            vllm_config,
            kernel_block_size=kernel_block_size,
            per_block_shape=per_block_shape,
            kv_dtype=kv_dtype,
            gpu_block_size=gpu_block_size,
            world_size=self.world_size,
            num_kv_cache_groups=len(kv_cache_config.kv_cache_groups),
            num_kv_cache_tensors=len(kv_cache_config.kv_cache_tensors),
            cp_geometry=cp_geometry,
        )

        # Shared-memory host pools (RAIDEN_SHM_KEY) are currently unsupported.
        # Fail fast during initialization to prevent serving corrupted cache.
        if os.environ.get("RAIDEN_SHM_KEY"):
            raise ValueError(
                "TPURaidenOffloadingConnector: RAIDEN_SHM_KEY is not supported: "
                "shm-backed host pools are unsupported.")

        # Offset controller port by local data parallel (DP) rank so multiple
        # DP replicas colocated on the same host bind unique control ports.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        dp_rank_local = vllm_config.parallel_config.data_parallel_rank_local
        if dp_rank_local is None:
            dp_rank_local = dp_rank
        self.raiden_controller_port = dp_rank_local + int(
            self.extra_config.get("raiden_controller_port",
                                  _DEFAULT_CONTROLLER_PORT))
        self.raiden_job_name = str(
            self.extra_config.get("raiden_job_name",
                                  f"vllm-offload-{config.engine_id}"))
        # Suffix by global DP rank to ensure distinct RaidenId and registry
        # entries across multi-node data parallel replicas.
        if dp_rank:
            self.raiden_job_name = f"{self.raiden_job_name}-dp{dp_rank}"

        self._manager: RaidenOffloadingManager | None = None
        self._worker: RaidenStoreOffloadingWorker | None = None

    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            self._manager = RaidenOffloadingManager(
                kernel_physical_blocks_capacity=self.
                kernel_physical_blocks_capacity,
                offload_logical_blocks_capacity=self.
                offload_logical_blocks_capacity,
                gpu_block_size_factor=self.gpu_block_size_factor,
                world_size=self.world_size,
                raiden_controller_port=self.raiden_controller_port,
                raiden_job_name=self.raiden_job_name,
                global_registry_address=self.global_registry_address,
                raiden_orchestrator_address=self.raiden_orchestrator_address,
                store_server_ip=self.store_server_ip,
                key_namespace=self.key_namespace,
            )
        return self._manager

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if self._worker is None:
            from vllm.distributed import parallel_state
            rank = parallel_state.get_world_group().rank_in_group
            # Direct workers to dial the controller on store_server_ip when configured
            # (bind-and-advertise), falling back to loopback (127.0.0.1) for local runs.
            controller_address = (f"{self.store_server_ip or '127.0.0.1'}:"
                                  f"{self.raiden_controller_port}")
            self._worker = RaidenStoreOffloadingWorker(
                kv_caches,
                vllm_config=self.vllm_config,
                kv_cache_config=self.kv_cache_config,
                host_blocks_to_allocate=self.kernel_physical_blocks_capacity,
                controller_address=controller_address,
                rank=rank,
            )
        return self._worker
