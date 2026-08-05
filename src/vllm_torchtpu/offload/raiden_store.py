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
= one scheduler GPU block), no decode context parallelism.

Prefill context parallelism (PCP) is supported: the scheduler operates on
logical blocks spanning all PCP ranks while every rank's physical page —
and all geometry in this module — is rank-local, with scheduler block b
mapping to each rank's local page b. `world_size` (and with it the
registration gate, the save fanout, and the capacity math) already counts
the PCP ranks, so the only PCP-specific pieces are the consistency net in
the spec (the PCP-aware `build_offloading_config` patch must have scaled
`tokens_per_block` to the logical span) and the CP geometry folded into the
compatibility namespace. Data parallelism runs one independent V2 stack per
engine (per-DP-rank controller port and job name).

Phase-2 hardening: store jobs are batch-capped and retried once on transfer
failure; keys that fail terminally are quarantined (reported MISS, never
re-admitted). A drain that times out poisons the still-in-flight jobs, so
their eventual completions finalize as failures instead of committing bytes
possibly torn by the block reuse that motivated the drain.

Phase 3 (newer store primitives, detected at runtime): store-job batches are
classified by `insert_and_lock_detailed` (existing entries pinned in place,
only newly inserted ones saved) with displacements surfaced as eviction
events through `take_events`. On an older tpu_raiden, classification falls
back to per-sub-hash peeks and no events are emitted.

Phase 4 (cross-instance KV sharing, enabled by `global_registry_address`):
saved blocks are published to the global registry by raiden, so a prefix
computed by any replica is a `REMOTE` hit everywhere. The receiver flow is
`lookup(enable_global=True)` -> stash the peer `REMOTE` slices ->
`insert_and_lock(remote_slices, on_host=True)` in prepare_load (installs the
peer mapping locally, reserves host landing capacity, pins; on wheel #2 the
detailed variant also reports what the landing reservation displaced) ->
`read_remote(hashes, device_block_ids)` straight into HBM at job-submit ->
`poll_remote_read_status` in the finalizer. Success leaves the entries
`HOST_AND_HBM` (the host landing blocks make a later local load cheap);
failure `release_and_delete`s the batch, which frees the landing blocks and
removes the local install while the registry keeps the peer mapping.
Registry keys are namespaced by engine compatibility (§4 of the design), and
raiden itself degrades a dead registry to local-only lookups.

See `raiden_connector.py` for the connector wiring (fence + echo channel).
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

# Diagnostic only: log the namespaced sub-hash of every store batch and of
# every registry-eligible lookup miss, so two replicas' key spaces can be
# diffed from their logs (j-c92b643e: cross-replica lookups missed 100%
# despite both replicas provably publishing and querying the registry).
_DEBUG_KEYS = os.environ.get("VLLM_TPU_OFFLOAD_DEBUG_KEYS", "") == "1"

_DEFAULT_CONTROLLER_PORT = 51515


class RaidenLoadStoreSpec(LoadStoreSpec):
    """Offload-side spec: the ordered offload keys a job covers.

    Device block IDs are bound later, at job-submit time, from the matching
    `GPULoadStoreSpec` (they are not known to the manager interface).

    `pinned` is False when prepare_load lost the race against eviction (the
    manager interface has no failure return there); the load job then fails
    cleanly at submit time and vLLM recomputes the blocks.

    `displaced_keys` (loads, wheel #2): keys whose entries prepare_load's
    host-landing install displaced. Carried to submit_load_job so the
    displacement commits with the job's outcome, exactly like a store
    admission's.
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
    # Loads only (phase 4): the sub-hashes fetched from a peer via
    # read_remote rather than the local host pool. Drives the remote poll
    # and the weaker remote-only quarantine on failure.
    remote_sub_hashes: set[bytes] = field(default_factory=set)
    # Stores only: the sub-hashes actually saved (newly admitted HBM entries;
    # pre-existing HOST entries were pinned, not re-saved) and the remaining
    # retry budget for the failed subset.
    saved_sub_hashes: list[bytes] = field(default_factory=list)
    # Sub-hashes that failed: the polls append transfer failures (stores
    # retry exactly this subset), and launch failures pre-fill the batch
    # that never launched. _quarantine picks the remote-only bucket iff
    # every entry was a peer fetch.
    failed_sub_hashes: list[bytes] = field(default_factory=list)
    retries_left: int = 0
    # Keys whose entries this admission's insert displaced into the
    # eviction-candidate list (stores: the HBM admission; loads: the
    # phase-4 host-landing install). Committed (as eviction events +
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
        capacity_kernel_blocks: int,
        num_offloaded_blocks: int,
        sub_blocks_per_key: int,
        world_size: int,
        raiden_controller_port: int,
        raiden_job_name: str,
        global_registry_address: str = "",
        raiden_orchestrator_address: str = "",
        store_server_ip: str = "",
        key_namespace: bytes = b"",
        store: object | None = None,
        store_types: object | None = None,
    ):
        assert capacity_kernel_blocks == (num_offloaded_blocks *
                                          sub_blocks_per_key)
        self._capacity_keys = num_offloaded_blocks
        self._f = sub_blocks_per_key
        self._world_size = world_size
        self._ns = key_namespace
        # Cross-instance sharing: consuming REMOTE hits needs only the
        # registry; publishing this replica's own blocks additionally needs a
        # peer-reachable store_server_ip (raiden logs the asymmetry too).
        self._enable_global = bool(global_registry_address)
        if self._enable_global and not store_server_ip:
            # KVCacheStore::Create now requires a dialable store_server_ip,
            # so the old consume-only degradation (empty ip, no publishes)
            # cannot be expressed anymore — and falling back to 127.0.0.1
            # would publish an address peers cannot dial. Fail loudly; the
            # recipe must pass the pod IP.
            raise ValueError(
                "[kv-offload-v2] global registry configured without "
                "store_server_ip: peers could never dial this replica. "
                "Set store_server_ip to the pod IP in "
                "kv_connector_extra_config.")

        if store is None:
            # Bind the native extension rather than
            # tpu_raiden.api.torch.kv_cache_store: that Python wrapper is not
            # packaged in the torch wheel, while the extension it wraps is.
            # The extension exports the same names (KVCacheStore, RaidenId,
            # RaidenBlockID, BlockStatus) with the same keyword arguments;
            # the only difference that reaches this module is
            # insert_and_lock_detailed returning the raw
            # (success, existing, inserted, displaced) tuple instead of a
            # dataclass. The loader call is what the wrapper does first and
            # is required before importing the extension; it is idempotent.
            from tpu_raiden.api.torch import torch_tpu_common_loader
            torch_tpu_common_loader.load_torch_tpu_common()
            from tpu_raiden.frameworks.torch import \
                _tpu_raiden_torch as store_types
            store = store_types.KVCacheStore(
                capacity=capacity_kernel_blocks,
                raiden_id=store_types.RaidenId(raiden_job_name, "0",
                                               "kv_cache", 0),
                # Minimum accepted values; neither feeds the copy math.
                # Transfer fanout comes from the registered workers.
                num_shards=1,
                shard_size_bytes=1,
                raiden_controller_port=raiden_controller_port,
                global_registry_address=global_registry_address,
                raiden_orchestrator_address=raiden_orchestrator_address,
                # KVCacheStore::Create rejects an empty store_server_ip and
                # names 127.0.0.1 as the same-host value. Only registry runs
                # have a real peers-dial-this address (the pod IP from the
                # recipe); local-only stores never publish, so localhost is
                # exact there. Keep the raw value in self.store_server_ip
                # checks above: a registry run that FORGOT the pod IP must
                # keep warning, not silently publish 127.0.0.1 to peers.
                store_server_ip=store_server_ip or "127.0.0.1",
            )
            logger.info(
                "[kv-offload-v2] KVCacheStore up: capacity=%d kernel blocks "
                "(%d offloaded blocks x %d), namespace=%s, controller at "
                "%s%s", capacity_kernel_blocks, num_offloaded_blocks,
                sub_blocks_per_key, key_namespace.hex(),
                store.raiden_controller_address,
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

        # Newer store primitives. hasattr is deliberate: the tpu_raiden wheel
        # is pinned per recipe and older wheels genuinely lack this —
        # store-job classification then falls back to per-sub-hash peeks (no
        # eviction events).
        self._has_detailed = hasattr(self._store, "insert_and_lock_detailed")
        if not self._has_detailed:
            logger.warning(
                "[kv-offload-v2] tpu_raiden wheel lacks "
                "insert_and_lock_detailed: store-job classification uses "
                "per-sub-hash peeks and eviction events are unavailable.")

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

        # Remote-fetch quarantine (phase 4): keys whose read_remote failed
        # terminally. Weaker than the full quarantine — only the *global*
        # fallback is suppressed (a dead peer or stale registry entry must
        # not fail-loop), while local stores and loads of the same key stay
        # allowed: the key may well be computed and stored locally later.
        self._remote_quarantined_keys: set[OffloadKey] = set()

        # Peer slices stashed by the last global lookup, keyed by sub-hash.
        # Consumed by prepare_load's insert_and_lock(on_host=True); entries
        # are overwritten by later lookups, so staleness is bounded by the
        # lookup->prepare_load window of one scheduler step (a stale peer
        # mapping surfaces as a failed read, not corruption).
        self._remote_slices: dict[bytes, object] = {}

        self._admissions: dict[int, Admission] = {}
        self._hash_to_admission: dict[bytes, Admission] = {}
        self._finished: list[FinishedJob] = []
        # Eviction events for take_events(): populated when a successful
        # store job's admission displaced entries (wheel #2 detailed result).
        self._pending_events: list[OffloadingEvent] = []

    # -- sub-hash / block-id expansion ----------------------------------

    def _sub_hashes(self, key: OffloadKey) -> list[bytes]:
        # The namespace prefix (phase 4) makes registry keys byte-distinct
        # across incompatible engines; locally it is inert.
        return [self._ns + key + i.to_bytes(4, "big") for i in range(self._f)]

    def _key_of(self, sub_hash: bytes) -> OffloadKey:
        return OffloadKey(sub_hash[len(self._ns):-4])

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

    def _displaced_keys_of(self, raw_displaced) -> list[OffloadKey]:
        """Unique OffloadKeys of a detailed insert result's displacements,
        in displacement order (sub-hashes collapse back to their key)."""
        displaced: list[OffloadKey] = []
        seen: set[OffloadKey] = set()
        for h, _ in raw_displaced:
            key = self._key_of(h)
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
        # LRU-recency update, eviction candidates report as misses. With the
        # global registry enabled, local misses fall through to a registry
        # lookup and come back as REMOTE peer slices (raiden swallows
        # registry outages, so a dead registry degrades to local-only).
        allow_remote = (self._enable_global
                        and key not in self._remote_quarantined_keys)
        matches = self._store.lookup(sub_hashes, enable_global=allow_remote)
        if len(matches) < len(sub_hashes):
            if _DEBUG_KEYS and allow_remote:
                logger.info(
                    "[kv-offload-v2] DEBUG_KEYS global-miss sub_hash=%s",
                    sub_hashes[len(matches)].hex())
            self._stored_keys.discard(key)
            return LookupResult.MISS
        all_local = True
        remote_slices: dict[bytes, object] = {}
        for h, block in matches:
            status = block.status
            if status in (self._block_status.HOST,
                          self._block_status.HOST_AND_HBM):
                continue
            if status == self._block_status.HBM:
                # Save still in flight (should be covered by the in-flight
                # set above; kept as a consistency net).
                return LookupResult.HIT_PENDING
            if status == self._block_status.REMOTE and allow_remote:
                if h in self._hash_to_admission:
                    # A read_remote for this sub-hash is already in flight;
                    # a second launch on the same hash must not race it.
                    return LookupResult.HIT_PENDING
                all_local = False
                remote_slices[h] = block
                continue
            return LookupResult.MISS
        if not all_local:
            self._remote_slices.update(remote_slices)
            return LookupResult.HIT
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
        # Remote sub-hashes (stashed by the global lookup) are installed and
        # pinned via insert_and_lock(on_host=True): it inserts the peer
        # mapping and reserves host landing capacity for new entries, and
        # only pins (slices untouched) any that became local since the stash
        # — submit_load_job re-partitions by live status, so a stale stash
        # entry degenerates to a plain local load.
        remote_hashes: list[bytes] = []
        remote_slices: list[object] = []
        local_hashes: list[bytes] = []
        for h in sub_hashes:
            block = self._remote_slices.pop(h, None)
            if block is not None:
                remote_hashes.append(h)
                remote_slices.append(block)
            else:
                local_hashes.append(h)
        if local_hashes and not self._store.pin(local_hashes):
            # Lost the race against eviction between lookup and pin (raiden's
            # pin is all-or-nothing, so nothing is held). The manager
            # interface has no failure return here; hand the connector an
            # unpinned spec so submit_load_job fails the job cleanly and the
            # scheduler recomputes the blocks.
            logger.warning(
                "[kv-offload-v2] prepare_load lost %d sub-blocks for request "
                "%s to eviction between lookup and pin; the load job will "
                "fail cleanly and the blocks recompute", len(local_hashes),
                req_context.req_id)
            return RaidenLoadStoreSpec(keys, pinned=False)
        displaced_keys: list[OffloadKey] = []
        if remote_hashes:
            # The landing reservation can displace stored entries exactly
            # like a store admission; on wheel #2 the detailed result
            # surfaces them, and they commit with the load's outcome.
            if self._has_detailed:
                installed, _existing, _inserted, raw_displaced = (
                    self._store.insert_and_lock_detailed(remote_hashes,
                                                         remote_slices,
                                                         on_host=True))
                if installed:
                    displaced_keys = self._displaced_keys_of(raw_displaced)
            else:
                installed = self._store.insert_and_lock(remote_hashes,
                                                        remote_slices,
                                                        on_host=True)
            if not installed:
                # Landing capacity is exhausted (capacity pressure with
                # pinned entries) — rolled back inside raiden, nothing
                # remote is held. Never launch read_remote for an
                # uninstalled batch; fail the job cleanly and let the
                # blocks recompute.
                if local_hashes:
                    self._store.release(local_hashes)
                logger.warning(
                    "[kv-offload-v2] prepare_load could not reserve landing "
                    "capacity for %d remote sub-blocks (request %s); the "
                    "load job will fail cleanly and the blocks recompute",
                    len(remote_hashes), req_context.req_id)
                return RaidenLoadStoreSpec(keys, pinned=False)
        return RaidenLoadStoreSpec(keys, displaced_keys=displaced_keys)

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
        # keep it modest. Dropped keys are not re-offered — the 0.26.0
        # scheduler advances its per-group chunk index past every offered
        # chunk, so keys beyond the cap are never stored.
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
        events = self._pending_events
        self._pending_events = []
        return events

    def has_pending_work(self) -> bool:
        return bool(self._admissions)

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
        # Save only the newly admitted HBM entries. Pre-existing HOST /
        # HOST_AND_HBM sub-hashes (recovered across a restart, or committed
        # by a partially failed earlier save) were locked without touching
        # their slices; their bytes are valid (content-addressed) and
        # passing them to save() would fail the whole launch on its
        # HBM-status precondition. Wheel #2's detailed result classifies the
        # batch directly (and reports displacements); on wheel #1, classify
        # with per-sub-hash peeks after a bare insert_and_lock.
        displaced_keys: list[OffloadKey] = []
        if self._has_detailed:
            admitted, _existing, inserted, raw_displaced = (
                self._store.insert_and_lock_detailed(sub_hashes,
                                                     slices,
                                                     on_host=False))
            if admitted:
                to_save = list(inserted)
                displaced_keys = self._displaced_keys_of(raw_displaced)
        else:
            admitted = self._store.insert_and_lock(sub_hashes,
                                                   slices,
                                                   on_host=False)
            if admitted:
                to_save = [
                    h for h in sub_hashes
                    if self._sub_hash_status(h) == self._block_status.HBM
                ]
        if not admitted:
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
        if not to_save:
            # Every sub-hash was already host-resident: nothing to transfer.
            # Release the exact admitted batch and finish successfully.
            self._store.release(sub_hashes)
            self._inflight_store_keys.difference_update(keys)
            self._stored_keys.update(keys)
            self._commit_displacement(displaced_keys)
            self._finished.append(
                FinishedJob(job_id=job_id, op=AdmissionOp.STORE, success=True))
            return
        if not self._store.save(to_save):
            # Synchronous launch failure: never entered into the poll path;
            # finalize an already-failed admission immediately so the
            # cleanup (exact-batch revert, rollback of displaced candidates)
            # and the quarantine match a polled terminal failure.
            logger.error(
                "[kv-offload-v2] store job %d: save() launch failed for %d "
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
        if _DEBUG_KEYS and to_save:
            logger.info(
                "[kv-offload-v2] DEBUG_KEYS store-batch n=%d first=%s "
                "last=%s", len(to_save), to_save[0].hex(), to_save[-1].hex())
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
        for h in to_save:
            self._hash_to_admission[h] = admission

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
        for h, d in zip(sub_hashes, device_ids):
            if self._sub_hash_status(h) == self._block_status.REMOTE:
                remote_hashes.append(h)
                remote_device_ids.append(d)
            else:
                local_hashes.append(h)
                local_device_ids.append(d)

        def _fail_launch(what: str, failed_hashes: list[bytes],
                         in_flight: list[bytes]) -> None:
            # failed_hashes is the batch whose launch failed: it decides the
            # quarantine bucket (remote-only iff every hash was a peer
            # fetch) instead of leaving _quarantine an empty list.
            logger.error(
                "[kv-offload-v2] load job %d (req %s): %s launch failed "
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
                for h in in_flight:
                    self._hash_to_admission[h] = admission
                return
            # Nothing in flight: immediately terminal. The finalizer keeps
            # HOST entries intact, deletes freshly installed REMOTE entries
            # (landing blocks freed, the registry keeps the peer mapping),
            # and applies the same quarantine policy as a polled failure.
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
        if (admission.op == AdmissionOp.LOAD and admission.failed_sub_hashes
                and all(h in admission.remote_sub_hashes
                        for h in admission.failed_sub_hashes)):
            # Every failed sub-hash was a peer fetch: suppress only the
            # global fallback for these keys (a dead peer or stale registry
            # entry must not fail-loop), keep local stores/loads allowed.
            # (A poisoned admission has no failed sub-hashes and lands in
            # the full quarantine below.)
            self._remote_quarantined_keys.update(admission.keys)
            logger.error(
                "[kv-offload-v2] remote-quarantining %d keys after a "
                "terminal read_remote failure (job %d); they fall back to "
                "local-only (%d keys remote-quarantined in total)",
                len(admission.keys), admission.job_id,
                len(self._remote_quarantined_keys))
            return
        self._quarantined_keys.update(admission.keys)
        logger.error(
            "[kv-offload-v2] quarantining %d keys after a terminal %s "
            "transfer failure (job %d); they will not be offloaded again "
            "(%d keys quarantined in total)", len(admission.keys),
            admission.op.value, admission.job_id, len(self._quarantined_keys))

    def _finalize(self, admission: Admission) -> None:
        # Launch failures finalize an admission that was never registered,
        # so the pop tolerates absence; both paths share this one terminal
        # transition (cleanup, quarantine, finished-job report).
        self._admissions.pop(admission.job_id, None)
        for h in admission.sub_hashes:
            self._hash_to_admission.pop(h, None)
        success = not (admission.failed or admission.poisoned)
        if not success:
            self._quarantine(admission)
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
            # Loads: release the pins taken in prepare_load. On success a
            # plain release (remote fetches are HOST_AND_HBM by now), and
            # the host-landing install's displacements become permanent. On
            # failure release_and_delete: HOST entries stay intact, but
            # still-REMOTE installs from prepare_load are deleted — their
            # host landing blocks are freed while the registry keeps the
            # peer mapping. The destination device blocks are undefined
            # either way and get recomputed.
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
        has_stores = any(a.op == AdmissionOp.STORE
                         for a in self._admissions.values())
        has_loads = any(
            a.op == AdmissionOp.LOAD and any(h not in a.remote_sub_hashes
                                             for h in a.pending)
            for a in self._admissions.values())
        has_remote = any(
            a.op == AdmissionOp.LOAD and any(h in a.remote_sub_hashes
                                             for h in a.pending)
            for a in self._admissions.values())
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

        On timeout the still-in-flight admissions are poisoned rather than
        waited on forever or left to commit: the store has no abort API, so
        the jobs stay live until their polls resolve, but the finalizer
        then treats them as failed — the reuse may tear the bytes under the
        transfer, so even a reported success must not be committed. A
        poisoned job counts as drained for later calls (waiting on it again
        buys nothing).
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
            live = [
                jid for jid in job_ids if jid in self._admissions
                and not self._admissions[jid].poisoned
            ]
            if not live:
                return drained
            if time.perf_counter() >= deadline:
                for jid in live:
                    self._admissions[jid].poisoned = True
                logger.error(
                    "[kv-offload-v2] drain timed out after %.1fs; poisoned "
                    "jobs %s (%d keys): their transfers are still in flight "
                    "while their blocks are reused, so they will finalize "
                    "as failures and their keys quarantine", _DRAIN_TIMEOUT_S,
                    sorted(live),
                    sum(len(self._admissions[jid].keys) for jid in live))
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
    """Compatibility namespace prefixed to every store/registry key (§4).

    A registry key must never match across incompatible engines: covers
    model+revision and quantization (weight identity), KV dtype and the
    kernel-row layout, scheduler and raiden block geometry, the TP shard
    topology (read_remote pulls per-shard bytes), the KV group/tensor
    structure, and the block-hash algorithm plus a scheme version. vLLM's
    prefix-cache hashes themselves are deterministic for every supported
    algo in the pinned 0.26.0, and already fold in LoRA/MM extras.

    `cp_geometry` is `(pcp_size, dcp_size, interleave_size)` when context
    parallelism is active (it changes how tokens map onto each rank's
    pages) and MUST stay `()` otherwise, so existing pcp=1 namespaces are
    unaffected by its introduction.
    """
    model_config = vllm_config.model_config
    material = repr((model_config.model, model_config.revision,
                     model_config.quantization, str(kv_dtype),
                     kernel_block_size, per_block_shape, gpu_block_size,
                     world_size, num_kv_cache_groups, num_kv_cache_tensors,
                     vllm_config.cache_config.prefix_caching_hash_algo,
                     "kv-offload-v2-ns1") + cp_geometry)
    return hashlib.sha256(material.encode()).digest()[:16]


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
    if backend_cls is None:
        # _find_non_ssm_backend walks the instantiated model's layers, but the
        # scheduler-role connector lives in the EngineCore process where the
        # model is never built and that registry is empty. Resolve the backend
        # the same way the workers' attention layers will instead.
        from vllm.config import set_current_vllm_config
        from vllm.v1.attention.selector import get_attn_backend
        model_config = vllm_config.model_config
        with set_current_vllm_config(vllm_config):
            backend_cls = get_attn_backend(
                head_size=model_config.get_head_size(),
                dtype=model_config.dtype,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                use_mla=model_config.use_mla,
            )
    assert backend_cls is not None, (
        "[kv-offload-v2] requires a non-SSM attention backend")
    assert not backend_cls.is_ssm(), (
        "[kv-offload-v2] resolved attention backend is SSM: "
        f"{backend_cls.get_name()}")
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
        assert parallel.dcp_size == 1, (
            "[kv-offload-v2] does not support decode context parallelism")
        # PCP ranks are part of the worker world (world_size = TP*PP*PCP), so
        # the registration gate, the save fanout, and the all-ranks capacity
        # math below already count them.
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

        # Consistency net for PCP: the scheduler chunks (and hashes) at the
        # logical all-PCP-rank block span while kv_cache_config specs — and
        # every device/host mapping in this module — are rank-local. The
        # PCP-aware build_offloading_config patch is what scales each
        # group's tokens_per_block to that logical span; if it did not run,
        # OffloadKeys would cover 1/pcp of a scheduler block and every
        # key<->block mapping would be silently wrong.
        for group_config, group in zip(config.groups,
                                       kv_cache_config.kv_cache_groups):
            if isinstance(group.kv_cache_spec, AttentionSpec):
                assert group_config.tokens_per_block == (
                    group.kv_cache_spec.block_size * parallel.pcp_size), (
                        "[kv-offload-v2] offloading group tokens_per_block="
                        f"{group_config.tokens_per_block} != rank-local "
                        f"block_size={group.kv_cache_spec.block_size} x "
                        f"pcp={parallel.pcp_size}; the PCP-aware "
                        "OffloadingConfig build patch did not run")

        # Folded into the namespace / shm UID only when CP is active, so
        # pre-existing pcp=1 namespaces (registry entries, recoverable shm
        # segments) stay valid across this code change.
        cp_geometry: tuple[int, ...] = ()
        if parallel.pcp_size > 1:
            cp_geometry = (
                parallel.pcp_size, parallel.dcp_size,
                vllm_config.parallel_config.cp_kv_cache_interleave_size)

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

        # Cross-instance KV sharing (phase 4): configured entirely from
        # kv_connector_extra_config. The registry address turns on global
        # lookups + read_remote; store_server_ip is the pod IP peers dial to
        # pull this replica's blocks (bind-and-advertise, so it must be an
        # IP this process can bind — without it the replica consumes remote
        # KV but publishes nothing).
        self.global_registry_address = str(
            self.extra_config.get("global_registry_address", ""))
        self.raiden_orchestrator_address = str(
            self.extra_config.get("raiden_orchestrator_address", ""))
        self.store_server_ip = str(self.extra_config.get(
            "store_server_ip", ""))

        # Every store/registry key carries the compatibility namespace, so
        # incompatible engines can never share registry entries; locally it
        # is inert.
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

        # Shm-backed host pools are unsupported: the raiden shm allocator
        # aliases every host mirror onto one segment, so serving from a
        # RAIDEN_SHM_KEY store returns corrupt KV behind full-looking hits.
        # Fail construction rather than let it through (see
        # TODO/reset-and-shm-recovery.md).
        if os.environ.get("RAIDEN_SHM_KEY"):
            raise ValueError(
                "[kv-offload-v2] RAIDEN_SHM_KEY is not supported: shm-backed "
                "host pools serve corrupt KV on the current tpu_raiden "
                "wheel. See TODO/reset-and-shm-recovery.md.")

        # Each DP engine owns a full V2 stack (store + in-process
        # controller) and DP-local engines share one host, so offset the
        # configured port by the local DP rank — with a static port every
        # engine past the first would fail its controller bind. The workers
        # of an engine see the same parallel config, so they derive the same
        # address.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        dp_rank_local = vllm_config.parallel_config.data_parallel_rank_local
        if dp_rank_local is None:
            dp_rank_local = dp_rank
        self.raiden_controller_port = dp_rank_local + int(
            self.extra_config.get("raiden_controller_port",
                                  _DEFAULT_CONTROLLER_PORT))
        # With store_server_ip set, the store binds its controller on that
        # IP (bind-and-advertise), so the workers must dial it there —
        # 127.0.0.1 would no longer reach the controller.
        self.raiden_controller_host = str(
            self.extra_config.get("raiden_controller_host",
                                  self.store_server_ip or "127.0.0.1"))
        self.raiden_job_name = str(
            self.extra_config.get("raiden_job_name",
                                  f"vllm-offload-{config.engine_id}"))
        # Suffix by the GLOBAL DP rank: with multi-node DP every node's
        # first engine has local rank 0, so a local-rank suffix would give
        # replicas sharing a recipe-set job name identical RaidenIds — and
        # colliding global-registry entries (phase 4).
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
