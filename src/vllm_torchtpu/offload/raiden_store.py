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
  A logical device block maps to `device_block_size_factor` store entries with derived
  sub-hashes `OffloadKey ‖ sub_idx`. Contiguous device kernel blocks are fused
  by Raiden into single hardware DMAs.

- Model and parallelism support: Supports dense full-attention (single KV group)
  and hybrid attention+Mamba models on the TPU unified block pool. Supports
  Prefill Context Parallelism (PCP) across ranks and Data Parallelism (DP)
  with independent per-engine controller ports and job names.

- Two-phase admission: `insert` atomically pins the whole batch, then a
  post-admission `lookup` (stable under the job's own pins) picks out the
  entries this admission inserted, and issues a DMA save for exactly those.

- Pin lifecycle: the store hands out a pin with the answer — `lookup` pins what
  it found, `insert` pins what it admitted — and a successful `save` or local
  `load` spends that pin. Only the pins no transfer consumed are given back by
  hand, which is what a job's terminal cleanup does. Bytes read from a peer are
  never pinned here and never released.

- Directory as the single source of truth: `prepare_store` probes Raiden per
  key to decide what to offer, so a key displaced from the LRU is re-offered
  on its next appearance. The connector emits no eviction events, and
  configurations that expect KV cache events are refused at startup.

- Fault tolerance and recovery: Failed saves retry once. Timed-out drains
  poison in-flight jobs to prevent committing torn bytes from reused blocks.

- Cross-instance KV sharing: Saved blocks are published to the global registry
  for cluster-wide reuse. Lookups resolve `REMOTE` hits and issue `read_remote`
  straight into HBM. A peer read borrows nothing from the local store: it
  reserves no capacity, keeps no host copy, and leaves no directory entry, so
  the same key is fetched again on every miss until a local store admits it.

See `raiden_connector.py` for connector wiring, device fencing, and echo channels.
"""
from __future__ import annotations

import enum
import hashlib
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,
                                        KVCacheConfig)
from vllm.v1.kv_offload.base import (CanonicalKVCaches, LoadStoreSpec,
                                     LookupResult, OffloadingManager,
                                     OffloadingSpec, OffloadingWorker,
                                     OffloadKey, PrepareStoreOutput,
                                     ReqContext, RequestOffloadingContext,
                                     ScheduleEndContext, TransferResult)
from vllm.v1.kv_offload.config import OffloadingConfig

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.offload.block_major_layout import (
    BlockMajorContract, resolve_block_major_contract)

if TYPE_CHECKING:
    import torch

logger = init_logger(__name__)

# Upper bound (seconds) on a scheduler-side drain of in-flight store/load
# jobs whose source/destination blocks are about to be reused.
_DRAIN_TIMEOUT_S = envs.VLLM_TPU_OFFLOAD_WAIT_TIMEOUT_S

# A save whose transfer failed is retried this many times (a failed save
# leaves the entry HBM+pinned, so a second save() works). Retries are
# suppressed for drained jobs: a drain means the source blocks are about to
# be reused, so re-reading them would ship garbage.
_SAVE_RETRIES = envs.VLLM_TPU_OFFLOAD_SAVE_RETRIES

_DEFAULT_CONTROLLER_PORT = 51515


def iter_leaf_kv_cache_specs(kv_cache_spec):
    """Yield the per-layer specs behind a KV cache group spec."""
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        for leaf in kv_cache_spec.kv_cache_specs.values():
            yield from iter_leaf_kv_cache_specs(leaf)
    else:
        yield kv_cache_spec


def has_multi_shapes_kv_caches(vllm_config: VllmConfig) -> bool:
    """Whether this model's KV cache consists of multiple tensors
       of different shapes.
    """
    return (vllm_config.model_config.architecture
            in ("DeepseekV4ForCausalLM", "GlmMoeDsaForCausalLM"))


class RaidenLoadStoreSpec(LoadStoreSpec):
    """Offload-side specification describing the ordered keys for a job.

    Device block IDs are bound later at submit time from the matching
    `GPULoadStoreSpec` (as they are not yet known during prepare).

    `pinned` is set to False if `prepare_load` lost the race against eviction
    before acquiring pins. The load job then fails cleanly at submit time,
    allowing vLLM to recompute the blocks.
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
    passed to `insert`, so release / rollback must use exactly the admitted
    batch, exactly once. This record is the single source of truth for that
    batch, for which of its pins are still held, and for which cleanup the
    job's terminal state owes.
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
    # Loads only: owning request and destination device block IDs, reported
    # through get_block_ids_with_load_errors on failure.
    req_id: str | None = None
    dst_device_block_ids: list[int] = field(default_factory=list)
    # Loads only: the sub-hashes fetched from a peer via read_remote rather
    # than the local host pool. Drives the remote poll, and marks the
    # sub-hashes this job holds no pin for — a peer read neither takes one nor
    # spends one.
    remote_sub_hashes: set[bytes] = field(default_factory=set)
    # Sub-hashes whose pin a completed transfer already spent. Releasing one of
    # these again would drop a pin the job no longer holds.
    consumed_sub_hashes: set[bytes] = field(default_factory=set)
    # Stores only: the sub-hashes actually saved (newly admitted HBM entries;
    # pre-existing HOST entries were pinned, not re-saved) and the remaining
    # retry budget for the failed subset.
    saved_sub_hashes: list[bytes] = field(default_factory=list)
    # Sub-hashes that failed: the polls append transfer failures (stores
    # retry exactly this subset), and launch failures pre-fill the batch.
    failed_sub_hashes: list[bytes] = field(default_factory=list)
    retries_left: int = 0


@dataclass
class FinishedJob:
    job_id: int
    op: AdmissionOp
    success: bool
    req_id: str | None = None
    failed_device_block_ids: list[int] = field(default_factory=list)


class RaidenOffloadingManager(OffloadingManager):
    """Scheduler-side OffloadingManager backed by raiden's KVCacheStore.

    The store owns the directory, LRU, pins, and transfers; this class maps
    vLLM's manager contract onto it and owns the failure-aware job finalizer
    (`poll_finished_jobs`).
    """

    def __init__(
        self,
        *,
        kernel_physical_blocks_capacity: int,
        offload_logical_blocks_capacity: int,
        device_block_size_factor: int,
        world_size: int,
        raiden_controller_port: int,
        raiden_job_name: str,
        global_registry_address: str = "",
        store_server_ip: str = "",
        kv_pool_group: str = "",
        key_namespace: bytes = b"",
        # Test overrides: allow injecting mock store instance and type bindings.
        store: object | None = None,
        store_types: object | None = None,
    ):
        assert kernel_physical_blocks_capacity == (
            offload_logical_blocks_capacity * device_block_size_factor)
        self._capacity_keys = offload_logical_blocks_capacity
        self._device_block_size_factor = device_block_size_factor
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
            from tpu_sync.api.torch.kv_cache_manager import _torch_impl
            store_types = _torch_impl()
            logger.info(
                "TPURaidenOffloadingConnector: blocking KVCacheStore construction "
                "until %d worker(s) register", world_size)
            store = store_types.KVCacheStore(
                capacity=kernel_physical_blocks_capacity,
                raiden_id=store_types.RaidenId(raiden_job_name, "0",
                                               "kv_cache", 0),
                # Transfer fanout is dynamically driven by registered workers.
                num_shards=1,
                shard_size_bytes=1,
                raiden_controller_port=raiden_controller_port,
                global_registry_address=global_registry_address,
                # Local-only runs default to 127.0.0.1 to satisfy C++ non-empty check.
                store_server_ip=store_server_ip or "127.0.0.1",
                # Pool group this store's KV cache flows within (spec
                # publication and eviction placement).
                kv_pool_group=kv_pool_group,
                # Construction blocks in RaidenController::Init() until every
                # rank registers, and raises on timeout
                # (RAIDEN_EXPECTED_WORKERS_TIMEOUT_S, default 120s).
                expected_worker_count=world_size,
            )
            logger.info(
                "TPURaidenOffloadingConnector: KVCacheStore up: capacity=%d kernel blocks "
                "(%d offloaded blocks x %d), namespace=%s, controller at "
                "%s, all %d workers registered%s",
                kernel_physical_blocks_capacity,
                offload_logical_blocks_capacity, device_block_size_factor,
                key_namespace.hex(), store.raiden_controller_address,
                world_size,
                (f", published at {store.store_server_address or '<nothing>'}"
                 f" in registry {global_registry_address}"
                 if self._enable_global else ""))
        else:
            # Injected store (tests): the caller supplies the module holding
            # BlockStatus / RaidenBlockId so tpu_sync isn't imported.
            assert store_types is not None
        self._store = store
        self._block_status = store_types.BlockStatus
        self._block_id_cls = store_types.RaidenBlockId
        self._raiden_id = self._store.raiden_id

        # The connector's only cache-state bookkeeping, and it is
        # load-bearing: it bars a second admission of a key whose store job is
        # in flight (fence-parked or launched), covering the window between
        # `prepare_store` and `insert` where the Raiden directory
        # holds nothing for the key. Every other store/skip/touch decision
        # probes the directory itself.
        self._inflight_store_keys: set[OffloadKey] = set()

        # Peer location slices cached from global lookups and consumed during
        # `prepare_load`.
        self._remote_slices: dict[bytes, object] = {}

        # Peer location slices `prepare_load` accepted, held until
        # `submit_load_job` hands them to read_remote. A peer read leaves no
        # local directory entry, so the slice is the only record that a
        # sub-hash lives elsewhere, and it must outlive the scheduler step that
        # found it. An entry a job never claims is harmless: the next lookup
        # re-resolves the hash, and a slice that has gone stale simply fails
        # its read and recomputes.
        self._pending_remote_slices: dict[bytes, object] = {}

        self._admissions: dict[int, Admission] = {}
        self._hash_to_admission: dict[bytes, Admission] = {}
        self._finished: list[FinishedJob] = []

    # -- sub-hash / block-id expansion ----------------------------------

    def _sub_hashes(self, key: OffloadKey) -> list[bytes]:
        # The namespace prefix makes registry keys byte-distinct
        # across incompatible engines; locally it is inert.
        return [
            self._key_namespace + key + i.to_bytes(4, "big")
            for i in range(self._device_block_size_factor)
        ]

    def _expand_keys(self, keys: list[OffloadKey]) -> list[bytes]:
        out: list[bytes] = []
        for key in keys:
            out.extend(self._sub_hashes(key))
        return out

    def _expand_device_blocks(self, device_block_ids: list[int]) -> list[int]:
        f = self._device_block_size_factor
        return [b * f + i for b in device_block_ids for i in range(f)]

    # -- pin bookkeeping -------------------------------------------------

    def _probe(self,
               sub_hashes: list[bytes],
               *,
               enable_global: bool = False) -> list[tuple[bytes, object]]:
        """Read the directory without laying claim to what it reports.

        `lookup` pins its hits by default so the answer cannot be evicted
        before the caller uses it. A probe uses the answer to make a decision
        and transfers nothing, so it asks for no pin: taking one and handing it
        straight back would drop the entry at the Most Recently Used position
        and quietly reorder eviction, which is `touch`'s job to do on purpose.
        """
        return self._store.lookup(sub_hashes,
                                  enable_global=enable_global,
                                  pin_found=False)

    def _pins_still_held(self, admission: Admission) -> list[bytes]:
        """The job's sub-hashes whose pin nothing has spent yet.

        A successful save or local load consumes the pin it was handed, and a
        peer read never took one; releasing either would drop a pin the job
        does not hold. What is left is what a terminal job owes back.
        """
        return [
            sub_hash for sub_hash in admission.sub_hashes
            if sub_hash not in admission.consumed_sub_hashes
            and sub_hash not in admission.remote_sub_hashes
        ]

    def _newly_inserted_sub_hashes(
            self, sub_hashes: list[bytes],
            device_ids: list[int]) -> list[bytes] | None:
        """Return the sub-hashes this admission itself inserted — the ones a
        DMA save must copy to host.

        Called immediately after `insert` pinned the whole batch,
        `sub_hashes[i]` having been admitted with device block `device_ids[i]`.
        The pins hold the directory still, so a single lookup decides every
        sub-hash:
        - HBM bound to the device block this job supplied: this admission put
          it there, so its bytes exist only on device; save it.
        - HOST / HOST_AND_HBM: already host-resident and now pinned in place;
          handing it to save() would trip Raiden's HBM gate and get the whole
          batch rejected.
        - REMOTE: the bytes live on a peer; nothing to transfer locally.

        Returns None when a sub-hash is missing, or sits in a state none of
        the above covers: an HBM entry bound to a device block this job never
        supplied. That entry is the residue of an earlier save that failed and
        exhausted its retries — the store has no removal API, so a dead device
        binding outlives the job that made it and an insert pins such an entry
        in place rather than rebinding it. Its recorded device block has since
        been reused, so the bytes behind it are not this key's; the caller
        fails the job closed rather than ship them. The entry is unpinned and
        evictable, so the key stores normally again once the store reclaims
        the slot.
        """
        # The admission already holds one pin per sub-hash, and that is the
        # pin the save spends; this only needs to read their status.
        matched = self._probe(sub_hashes)
        if len(matched) < len(sub_hashes):
            logger.error(
                "TPURaidenOffloadingConnector: admitted sub-hash vanished under its own "
                "pin (%d of %d resident); failing the store job", len(matched),
                len(sub_hashes))
            return None
        to_save: list[bytes] = []
        for i, (sub_hash, block) in enumerate(matched):
            status = block.status
            if status in (self._block_status.HOST,
                          self._block_status.HOST_AND_HBM,
                          self._block_status.REMOTE):
                continue
            if (status == self._block_status.HBM
                    and block.device_block_id == device_ids[i]):
                to_save.append(sub_hash)
                continue
            logger.error(
                "TPURaidenOffloadingConnector: admitted sub-hash in unexpected state "
                "(status=%s, device_block_id=%d, expected device %d); failing "
                "the store job", status, block.device_block_id, device_ids[i])
            return None
        return to_save

    # -- OffloadingManager contract --------------------------------------

    def on_new_request(self,
                       req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        if key in self._inflight_store_keys:
            return LookupResult.HIT_PENDING
        sub_hashes = self._sub_hashes(key)
        # A probe across all sub-hashes, not a claim: the store pins every
        # local hit it reports, and those pins go straight back below.
        # `prepare_load` takes them again when the scheduler commits to a load.
        # With the global registry enabled, local misses query remote peer slices.
        # If the registry is unreachable or times out, the error is swallowed and
        # lookup safely falls back to local-only caching without failing requests.
        allow_remote = self._enable_global
        matches = self._probe(sub_hashes, enable_global=allow_remote)
        if len(matches) < len(sub_hashes):
            return LookupResult.MISS
        all_local = True
        remote_slices: dict[bytes, object] = {}
        for sub_hash, block in matches:
            status = block.status
            if status in (self._block_status.HOST,
                          self._block_status.HOST_AND_HBM):
                continue
            if status == self._block_status.HBM:
                if sub_hash in self._hash_to_admission:
                    # A save for this sub-hash is genuinely in flight; the
                    # scheduler should come back for it. (The common case is
                    # already caught by the per-key in-flight set above; this
                    # is the sub-hash-level net.)
                    return LookupResult.HIT_PENDING
                # HBM with no live job behind it is not a pending save. It is
                # the residue of a save that failed and exhausted its retries
                # -- `prepare_store` documents the same state from the store
                # side. The store has no removal API, so the entry outlives
                # the job that made it, and a host pool that can no longer
                # accept writes never reclaims it. Answering HIT_PENDING here
                # would park the request on a save that will never complete:
                # the scheduler re-looks-up every step, gets HIT_PENDING
                # again, and the request defers forever. The bytes are not on
                # the host, so a load could not serve it either. It is a miss
                # -- recompute, and let `prepare_store` re-offer the key.
                return LookupResult.MISS
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

    def touch(self, keys, req_context: ReqContext) -> None:
        # A lookup pins what it finds and releasing drops the entry back at
        # the Most Recently Used position, so probe-then-release is the touch:
        # it refreshes LRU order and leaves the entry evictable again, holding
        # nothing. A partially resident chain needs no special case — only the
        # entries that were actually found move, and prepare_store re-offers
        # the key on its next appearance.
        for key in keys:
            sub_hashes = self._sub_hashes(key)
            held = [
                sub_hash for sub_hash, block in self._store.lookup(sub_hashes)
                if block.status != self._block_status.REMOTE
            ]
            if held:
                self._store.release(held)

    def prepare_load(self, keys, req_context: ReqContext) -> LoadStoreSpec:
        keys = list(keys)
        sub_hashes = self._expand_keys(keys)
        # Split the batch by where the bytes are. A local sub-hash is pinned
        # here so it survives until the load runs; a peer sub-hash carries its
        # slice forward to submit_load_job instead, because a peer read borrows
        # nothing from this store — there is no landing capacity to reserve and
        # no local entry to pin.
        #
        # Residency decides, not the stash: a sub-hash the stash calls remote
        # but that is resident here — a concurrent fetch landed it, or another
        # request stored it — is loaded locally, which is both cheaper and the
        # only option once the peer stops advertising it. That is why the batch
        # is walked rather than partitioned in one pass: a lookup answers with
        # the resident PREFIX of what it was asked, stopping at the first hash
        # it cannot find, so each miss is resolved against the stash and the
        # walk resumes after it.
        remote_slices: dict[bytes, object] = {}
        local_hashes: list[bytes] = []
        idx = 0
        while idx < len(sub_hashes):
            matched = self._store.lookup(sub_hashes[idx:])
            local_hashes.extend(sub_hash for sub_hash, _ in matched)
            idx += len(matched)
            if idx == len(sub_hashes):
                break
            block = self._remote_slices.pop(sub_hashes[idx], None)
            if block is None:
                # Neither resident nor a peer's: the entry the lookup promised
                # is gone. Give back the pins taken so far and return an
                # unpinned spec so submit_load_job fails the job cleanly and
                # the blocks recompute.
                if local_hashes:
                    self._store.release(local_hashes)
                logger.warning(
                    "TPURaidenOffloadingConnector: prepare_load lost sub-block %d of %d "
                    "for request %s to eviction; the load job will fail "
                    "cleanly and the blocks recompute", idx + 1,
                    len(sub_hashes), req_context.req_id)
                return RaidenLoadStoreSpec(keys, pinned=False)
            remote_slices[sub_hashes[idx]] = block
            idx += 1
        self._pending_remote_slices.update(remote_slices)
        return RaidenLoadStoreSpec(keys)

    def prepare_store(self, keys,
                      req_context: ReqContext) -> PrepareStoreOutput | None:
        # The Raiden directory is the sole authority on what is stored: probe
        # it per key, so a key whose entries were displaced from the LRU is
        # re-offered here on its next appearance. A key is skipped only when
        # its whole sub-hash chain is host-resident (HOST/HOST_AND_HBM) or
        # cluster-resident (REMOTE — the data exists on a peer; no local
        # re-store).
        keys_to_store: list[OffloadKey] = []
        resident_statuses = (self._block_status.HOST,
                             self._block_status.HOST_AND_HBM,
                             self._block_status.REMOTE)
        for key in keys:
            if key in self._inflight_store_keys:
                continue
            sub_hashes = self._sub_hashes(key)
            matches = self._probe(sub_hashes)
            if len(matches) == len(sub_hashes) and all(
                    block.status in resident_statuses for _, block in matches):
                continue
            keys_to_store.append(key)
        if len(keys_to_store) > self._capacity_keys:
            return None
        # Atomic admission happens at job-submit time via insert; in-flight
        # tracking here only stops duplicate admission of the same key from a
        # concurrent request.
        self._inflight_store_keys.update(keys_to_store)
        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=RaidenLoadStoreSpec(keys_to_store),
            evicted_keys=[],
        )

    def take_events(self):
        # This manager produces no offloading events: Raiden's admission API
        # does not report which entries an insert displaced, and `prepare_store`
        # rediscovers them by probing the directory. Configurations that
        # consume the event stream are refused at startup (see
        # `TPURaidenStoreOffloadingSpec.__init__`).
        return []

    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        # Remote slices stashed by lookup() are consumed by prepare_load()
        # within the same scheduler step. A hit that never reaches
        # prepare_load (request aborted, or the scheduler declined it under
        # HBM pressure) would otherwise leave its entries behind forever —
        # and a stale slice popped by a later request could be handed to
        # read_remote. Drop the leftovers each step; a request retried
        # next step re-runs lookup() and repopulates the stash.
        self._remote_slices.clear()

    def reset_cache(self) -> None:
        # Backstop only: TPURaidenOffloadingConnector.reset_cache() reports
        # failure before the upstream scheduler reset can reach this hook.
        # Kept as a raise (not the inherited silent no-op) so any future path
        # that bypasses the connector override cannot falsely report success
        # to POST /reset_prefix_cache?reset_external=true while the Raiden
        # store continues serving cached entries.
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
                         device_block_ids: list[int]) -> None:
        """Admit and launch a store job after all worker ranks acknowledge the save fence.

        Atomically admits the batch via insert (pinning pre-existing entries
        in-place, reserving capacity, rolling back on failure), then a
        post-admission lookup picks out the entries this admission inserted
        and dispatches save() for exactly those; completion is tracked
        asynchronously by the finalizer. A directory state that lookup cannot
        account for fails the job closed (rollback + recompute).
        """
        assert len(keys) == len(device_block_ids), (len(keys),
                                                    len(device_block_ids))
        sub_hashes = self._expand_keys(keys)
        device_ids = self._expand_device_blocks(device_block_ids)
        slices = [
            self._block_id_cls(
                raiden_id=self._raiden_id,
                host_block_id=-1,
                device_block_id=dev_id,
                status=self._block_status.HBM,
            ) for dev_id in device_ids
        ]
        if not self._store.insert(sub_hashes, slices, on_host=False):
            # Atomically rolled back inside Raiden; no DMA save was issued.
            logger.warning(
                "TPURaidenOffloadingConnector: store job %d: insert rejected %d "
                "sub-blocks (capacity pressure with pinned entries)", job_id,
                len(sub_hashes))
            self._inflight_store_keys.difference_update(keys)
            self._finished.append(
                FinishedJob(job_id=job_id, op=AdmissionOp.STORE,
                            success=False))
            return
        to_save = self._newly_inserted_sub_hashes(sub_hashes, device_ids)
        if to_save is None:
            # Broken invariant: unwind the admission and fail the job; the
            # blocks recompute and the key is re-offered on its next miss.
            self._store.release(sub_hashes)
            self._inflight_store_keys.difference_update(keys)
            self._finished.append(
                FinishedJob(job_id=job_id, op=AdmissionOp.STORE,
                            success=False))
            return
        if not to_save:
            # All sub-hashes are already resident: unpin and finish successfully
            # without issuing redundant DMA transfers.
            self._store.release(sub_hashes)
            self._inflight_store_keys.difference_update(keys)
            self._finished.append(
                FinishedJob(job_id=job_id, op=AdmissionOp.STORE, success=True))
            return
        if not self._store.save(to_save):
            # Synchronous launch failure: finalize immediately to revert the
            # admitted entries.
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
        )
        self._admissions[job_id] = admission
        for sub_hash in to_save:
            self._hash_to_admission[sub_hash] = admission

    def submit_load_job(self,
                        job_id: int,
                        keys: list[OffloadKey],
                        device_block_ids: list[int],
                        req_id: str,
                        pinned: bool = True) -> None:
        """Launch a load job; entries were pinned in prepare_load."""
        assert len(keys) == len(device_block_ids), (len(keys),
                                                    len(device_block_ids))
        if not pinned:
            # prepare_load lost the pin race; nothing is held, so fail the
            # job without touching the store and let vLLM recompute.
            self._finished.append(
                FinishedJob(job_id=job_id,
                            op=AdmissionOp.LOAD,
                            success=False,
                            req_id=req_id,
                            failed_device_block_ids=list(device_block_ids)))
            return
        sub_hashes = self._expand_keys(keys)
        device_ids = self._expand_device_blocks(device_block_ids)
        # Partition by where prepare_load found each sub-hash: one that came
        # with a peer slice is read from that peer, everything else is pinned
        # in the local host pool. The slice is the only record of a peer block
        # — read_remote writes nothing to the local directory, so no live probe
        # could rediscover it. Only the BYTES are remote: every destination is
        # one of this instance's HBM blocks. The two destination lists are
        # disjoint subsequences of `device_ids`, each preserving its batch's
        # hash-to-block pairing, because load() and read_remote() both require
        # block_hashes[i] to line up with device_block_ids[i].
        remote_hashes: list[bytes] = []
        remote_slices: list[object] = []
        dst_device_ids_for_remote: list[int] = []
        local_hashes: list[bytes] = []
        dst_device_ids_for_local: list[int] = []
        for sub_hash, dev_id in zip(sub_hashes, device_ids):
            block = self._pending_remote_slices.pop(sub_hash, None)
            if block is not None:
                remote_hashes.append(sub_hash)
                remote_slices.append(block)
                dst_device_ids_for_remote.append(dev_id)
            else:
                local_hashes.append(sub_hash)
                dst_device_ids_for_local.append(dev_id)

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
                dst_device_block_ids=list(device_block_ids),
                remote_sub_hashes=set(remote_hashes),
                failed_sub_hashes=list(failed_hashes),
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
                                                 dst_device_ids_for_local):
            _fail_launch("load()", local_hashes, [])
            return
        if remote_hashes and not self._store.read_remote(
                remote_hashes, remote_slices, dst_device_ids_for_remote):
            _fail_launch("read_remote()", remote_hashes, local_hashes)
            return
        admission = Admission(
            job_id=job_id,
            op=AdmissionOp.LOAD,
            keys=keys,
            sub_hashes=sub_hashes,
            pending=set(sub_hashes),
            req_id=req_id,
            dst_device_block_ids=list(device_block_ids),
            remote_sub_hashes=set(remote_hashes),
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
        # Give back only what the job still holds. A transfer that landed
        # already spent its pin and a peer read never took one, so the release
        # covers the admitted entries no transfer accounted for: on a store,
        # the sub-hashes that were pinned in place rather than saved; on any
        # failure, the whole unspent remainder.
        held = self._pins_still_held(admission)
        if held:
            self._store.release(held)
        if admission.op == AdmissionOp.STORE:
            self._inflight_store_keys.difference_update(admission.keys)
            self._finished.append(
                FinishedJob(job_id=admission.job_id,
                            op=AdmissionOp.STORE,
                            success=success))
        else:
            self._finished.append(
                FinishedJob(
                    job_id=admission.job_id,
                    op=AdmissionOp.LOAD,
                    success=success,
                    req_id=admission.req_id,
                    failed_device_block_ids=([] if success else list(
                        admission.dst_device_block_ids)),
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
            # Save polls carry two extra lists that annotate why a REMOTE save
            # failed; a hash in either is already in `failed`, and both are
            # empty for the local saves this connector issues.
            done, failed, _pending, _existing, _unregistered = (
                self._store.poll_save_status())
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
                if sub_hash not in admission.remote_sub_hashes:
                    # A save or a local load that landed spends the pin the
                    # job was holding for that sub-hash.
                    admission.consumed_sub_hashes.add(sub_hash)
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
        per_block_shape: tuple[int, ...] | set[tuple[int, ...]],
        kv_dtype: object,
        device_block_size: int,
        world_size: int,
        num_kv_cache_groups: int,
        num_kv_cache_tensors: int,
        cp_geometry: tuple[int, ...] = (),
        block_major_contract: "BlockMajorContract | None" = None) -> bytes:
    """Derive a deterministic compatibility namespace prefixed to every store and registry key.

    Ensures cache keys never collide across incompatible engine deployments by
    hashing all configuration parameters that alter binary KV cache layout:
    - Model identity: model name, revision, quantization.
    - KV cache geometry: KV data type, kernel block size, per-block tensor shape,
      device block size, and cache group/tensor counts.
    - Distributed topology: world size (TP/PP) and context parallelism geometry
      (pcp_size, dcp_size, interleave_size) which alters rank-local token placement.
    - Hash scheme: prefix caching algorithm and namespace version salt.
    - Layout contract: block-major layout version, row bytes, and logical fingerprint.

    Returns:
        A 16-byte SHA-256 digest unique to this engine and layout configuration.
    """
    model_config = vllm_config.model_config
    # A multi-shape layout hands over sets, whose iteration order is not
    # stable across processes -- `torch.dtype` hashes by identity, so
    # `repr({torch.uint8, torch.bfloat16})` differs between runs. Canonicalize
    # to sorted tuples: the namespace must be a pure function of the config,
    # or every restart re-salts the registry keys.
    shape_field: object = (tuple(sorted(per_block_shape)) if isinstance(
        per_block_shape, (set, frozenset)) else per_block_shape)
    dtype_field: object = (tuple(sorted(
        str(d)
        for d in kv_dtype)) if isinstance(kv_dtype,
                                          (set, frozenset)) else str(kv_dtype))
    material_fields = (model_config.model, model_config.revision,
                       model_config.quantization, dtype_field,
                       kernel_block_size, shape_field, device_block_size,
                       world_size, num_kv_cache_groups, num_kv_cache_tensors,
                       vllm_config.cache_config.prefix_caching_hash_algo,
                       "tpu-raiden-offload-ns1") + cp_geometry
    if block_major_contract is not None:
        # Block-major layout bundles per-block bytes across layers; salt the namespace
        # with the layout fingerprint to isolate cache entries and prevent collisions
        # with layer-major or incompatible bundle geometries.
        material_fields += ("tpu-raiden-offload-ns2-block-major",
                            block_major_contract.fragment_count,
                            block_major_contract.bundle_row_bytes,
                            block_major_contract.logical_fingerprint)
    material = repr(material_fields)
    return hashlib.sha256(material.encode()).digest()[:16]


def is_multi_shapes_geometry(per_block_shape: object) -> bool:
    """True when `resolve_kernel_geometry` resolved a multi-shape KV layout.

    A multi-shape layout answers with the *set* of its per-block shapes
    rather than one shape.
    """
    return isinstance(per_block_shape, (set, frozenset))


def _resolve_multi_shapes_kv_geometry(
    kv_cache_config: KVCacheConfig
) -> tuple[int, set[tuple[int, ...]], set[object], int]:
    from vllm.v1.kv_cache_interface import (MLAAttentionSpec,
                                            SlidingWindowMLASpec)

    groups = kv_cache_config.kv_cache_groups
    for group in groups:
        for leaf in iter_leaf_kv_cache_specs(group.kv_cache_spec):
            assert isinstance(
                leaf, (MLAAttentionSpec, SlidingWindowMLASpec)), (
                    "TPURaidenOffloadingConnector: an aliased KV "
                    "layout has only MLA-family layers; got "
                    f"{type(leaf).__name__}")

    mla_leaves = [
        leaf for group in groups
        for leaf in iter_leaf_kv_cache_specs(group.kv_cache_spec)
        if isinstance(leaf, MLAAttentionSpec)
    ]
    assert mla_leaves
    anchor_block_sizes = sorted({leaf.block_size for leaf in mla_leaves})
    assert len(anchor_block_sizes) == 1
    block_size = anchor_block_sizes[0]

    # One entry per full-attention MLA cache.
    per_block_shapes: set[tuple[int, ...]] = set()
    kv_dtypes: set[object] = set()
    for leaf in mla_leaves:
        shape = (leaf.num_states, leaf.num_kv_heads, leaf.head_size)
        per_block_shapes.add(shape)
        kv_dtypes.add(leaf.dtype)
    return block_size, per_block_shapes, kv_dtypes, block_size


def resolve_kernel_geometry(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> tuple[int, tuple[int, ...] | set[tuple[int, ...]], object, int]:
    """Resolve physical TPU kernel geometry: (kernel_block_size, per_block_shape, kv_dtype, device_block_size).

    Shared between the scheduler (capacity sizing, sub-hash factor) and worker ranks
    (device view reconstruction) to guarantee exact geometry parity.

    Supported architectures:
    - Dense models: Defined directly by the single group's full-attention spec.
    - Hybrid (Attention + Mamba) models: Requires TPU's unified block pool, where all
      groups share uniform, attention-sized page rows. Attention geometry defines the
      common row unit, allowing the dense sub-hash expansion to apply uniformly.
    - Multi-kv-shape models: a model whose kv cache consists of multiple tensors with
      different shapes, it's handled by `_resolve_multi_shapes_kv_geometry`.
    """
    from vllm.v1.kv_cache_interface import AttentionSpec
    from vllm.v1.worker.utils import select_common_block_size

    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    groups = kv_cache_config.kv_cache_groups
    assert groups, "TPURaidenOffloadingConnector: no KV cache groups"

    if has_multi_shapes_kv_caches(vllm_config):
        return _resolve_multi_shapes_kv_geometry(kv_cache_config)

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

    from vllm.config import set_current_vllm_config

    attn_backend = TpuPlatform._find_non_ssm_backend(vllm_config)
    if attn_backend is None:
        # In the EngineCore process, the model is not instantiated, so
        # _find_non_ssm_backend returns None. Resolve the attention backend
        # directly via get_attn_backend using the model configuration.
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
    # Both calls resolve the KV cache layout, which falls through to the KV
    # connector and so needs a current config.
    with set_current_vllm_config(vllm_config):
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

    Registration is eager but asynchronous: worker initialization occurs
    before the scheduler creates the controller server, so a background
    thread probes the controller endpoint and registers as soon as it is
    reachable — normally while engine init is still loading/compiling, which
    keeps the host-pool allocation and registration RPCs off the first
    request's critical path. The scheduler side blocks KVCacheStore
    construction on all ranks registering, so a rank that cannot register
    fails engine startup when that barrier times out — and conversely, no
    inference step can ever run unregistered, so there is no per-step
    registration hook.
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
        block_major_contract: BlockMajorContract | None = None,
    ):
        import numpy as np
        import torch

        (kernel_block_size, per_block_shape, kv_dtype,
         device_block_size) = resolve_kernel_geometry(vllm_config,
                                                      kv_cache_config)
        assert device_block_size % kernel_block_size == 0
        device_block_size_factor = device_block_size // kernel_block_size

        if is_multi_shapes_geometry(per_block_shape):
            # Branch for models whose kv cache consists of multiple tensors
            # with different shapes. The resolved shapes describe those
            # arrays but do not tile a common row, so registration walks the
            # canonical tensors instead of computing bytes per kernel block.
            assert block_major_contract is None, (
                "block-major bundling not yet implemented for this code path")
            assert device_block_size_factor == 1
            device_tensors: list[torch.Tensor] = []
            for kv_cache_tensor in kv_caches.tensors:
                int8_view = kv_cache_tensor.tensor
                page_size_bytes = kv_cache_tensor.page_size_bytes
                storage_bytes = int8_view.numel() * int8_view.element_size()
                assert storage_bytes == int8_view.shape[0] * page_size_bytes, (
                    "canonical KV array is not whole pages: "
                    f"storage_bytes={storage_bytes}, "
                    f"rows={int8_view.shape[0]}, page={page_size_bytes}")
                device_tensors.append(int8_view)
            self._init_registration(device_tensors, host_blocks_to_allocate,
                                    controller_address, rank)
            return

        bytes_per_kernel_block = (
            int(np.prod(per_block_shape)) *
            torch.empty(0, dtype=kv_dtype).element_size())

        if block_major_contract is not None:
            # Block-major registration: Register the entire multi-layer bundle as a single
            # canonical storage ([kernel_blocks, F, *R]). Raiden views the bundle as 1 block array,
            # executing 1 hardware DMA per block transfer rather than F DMAs.
            #
            # Safety invariants:
            # - Consistent row geometry: Validates that fragment_row_bytes matches physical row bytes.
            # - Deployment protection: Layer-major peers registering fragment_row_bytes are rejected
            #   by Raiden's block_array_bytes consistency check against our bundle_row_bytes.
            assert (block_major_contract.fragment_row_bytes ==
                    bytes_per_kernel_block), (
                        "block-major contract kernel-row bytes "
                        f"{block_major_contract.fragment_row_bytes} != "
                        f"resolved geometry {bytes_per_kernel_block}")
            assert len(kv_caches.tensors) == 1, (
                "block-major requires the device bundle to be one canonical "
                f"storage; got {len(kv_caches.tensors)} — the runner did not "
                "materialize the [kernel_blocks, F, *R] bundle")
            per_block_shape = (
                block_major_contract.fragment_count, ) + per_block_shape
            bytes_per_kernel_block = block_major_contract.bundle_row_bytes

        # Rebuild typed tensor views shaped by physical kernel blocks from the
        # underlying storage, ensuring leading dimensions match DMA tile layouts.
        device_tensors: list[torch.Tensor] = []
        for kv_cache_tensor in kv_caches.tensors:
            # Verify that one canonical scheduler page equals exactly
            # device_block_size_factor physical kernel rows. For hybrid models
            # using the unified pool, this confirms that all group pages are
            # uniformly padded and fungible across pool rows.
            assert kv_cache_tensor.page_size_bytes == (
                device_block_size_factor * bytes_per_kernel_block), (
                    "canonical page size does not match the kernel-row "
                    f"geometry: page={kv_cache_tensor.page_size_bytes}, "
                    f"factor={device_block_size_factor}, "
                    f"kernel_row_bytes={bytes_per_kernel_block}")
            int8_view = kv_cache_tensor.tensor
            storage_bytes = int8_view.numel() * int8_view.element_size()
            assert storage_bytes % bytes_per_kernel_block == 0, (
                "KV cache storage must contain whole physical kernel blocks:"
                f" storage_bytes={storage_bytes}, "
                f"bytes_per_kernel_block={bytes_per_kernel_block}")
            num_kernel_blocks = storage_bytes // bytes_per_kernel_block
            assert num_kernel_blocks == (
                int8_view.shape[0] * device_block_size_factor), (
                    "canonical KV cache geometry does not match physical "
                    f"kernel-block geometry: canonical={int8_view.shape[0]}, "
                    f"factor={device_block_size_factor}, "
                    f"kernel_blocks={num_kernel_blocks}")
            device_tensors.append(
                torch.empty(0, dtype=kv_dtype, device=int8_view.device).set_(
                    int8_view.untyped_storage()).view((num_kernel_blocks, ) +
                                                      per_block_shape))

        self._init_registration(device_tensors, host_blocks_to_allocate,
                                controller_address, rank)

    def _init_registration(self, device_tensors: list[torch.Tensor],
                           host_blocks_to_allocate: int,
                           controller_address: str, rank: int) -> None:
        # Hold strong references to ensure PyTorch tensors remain allocated
        # while Raiden C++ retains raw pointers to their underlying storage.
        self._device_tensors = [[t] for t in device_tensors]
        self._host_blocks_to_allocate = host_blocks_to_allocate
        self._controller_address = controller_address
        self._rank = rank
        self._mgr = None
        self._stop_registration = threading.Event()
        self._register_thread = threading.Thread(
            target=self._register_loop,
            name=f"raiden-offload-register-rank{rank}",
            daemon=True)
        self._register_thread.start()

    def _controller_reachable(self) -> bool:
        """TCP-probe the controller endpoint (a listening server is a
        precondition for registering, not proof of registration)."""
        host, _, port = self._controller_address.rpartition(":")
        host = host.strip("[]")
        try:
            with socket.create_connection((host, int(port)), timeout=1.0):
                return True
        except OSError:
            return False

    def _register_loop(self) -> None:
        """Background registration: wait for the controller server to come up
        (it is created scheduler-side after this worker), then register.

        Retries with backoff until it succeeds or shutdown() stops it. The
        verdict lives scheduler-side: KVCacheStore construction blocks until
        every rank has registered and fails engine startup on timeout, so
        this loop only has to keep trying, never to give up or escalate.
        """
        delay = 0.5
        next_note = time.monotonic() + 30.0
        while not self._stop_registration.is_set():
            if self._controller_reachable():
                try:
                    from tpu_sync.api.torch import kv_cache_manager as _kcm
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
                        "TPURaidenOffloadingConnector: rank %d registered with offload "
                        "controller at %s (%d host kernel blocks)", self._rank,
                        self._controller_address,
                        self._host_blocks_to_allocate)
                    return
                except Exception:
                    # A persistent failure escalates via the
                    # scheduler's construction barrier timeout.
                    logger.warning(
                        "TPURaidenOffloadingConnector: rank %d: background "
                        "registration attempt failed; retrying in %.1fs",
                        self._rank,
                        delay,
                        exc_info=True)
            elif time.monotonic() >= next_note:
                # The unreachable case produces no exception to log, so this
                # throttled note is its only signal (and names the probed
                # address, the thing a misconfigured recipe gets wrong).
                logger.warning(
                    "TPURaidenOffloadingConnector: rank %d: offload controller at %s "
                    "not reachable yet; still probing (the scheduler's "
                    "registration barrier will fail engine startup if "
                    "registration does not succeed in time)", self._rank,
                    self._controller_address)
                next_note = time.monotonic() + 30.0
            self._stop_registration.wait(delay)
            delay = min(delay * 2, 5.0)

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
        self._stop_registration.set()
        self._register_thread.join(timeout=5.0)
        self._mgr = None


class TPURaidenStoreOffloadingSpec(OffloadingSpec):
    """Configuration specification wiring the Raiden offloading manager and per-rank workers.

    Constructed directly by TPURaidenOffloadingConnector to encapsulate full engine
    and KV cache configurations, compute hardware block capacities, and configure
    distributed registry settings.
    """

    def __init__(self, config: OffloadingConfig, vllm_config: VllmConfig,
                 kv_cache_config: KVCacheConfig):
        # This connector emits no OffloadingEvents — displaced keys are
        # rediscovered by `prepare_store`'s directory probe. Refuse
        # configurations that expect KV cache events rather than silently
        # dropping BlockRemoved notifications.
        kv_events = vllm_config.kv_events_config
        if kv_events is not None and kv_events.enable_kv_cache_events:
            raise ValueError(
                "TPURaidenOffloadingConnector: enable_kv_cache_events is not "
                "supported: the Raiden offload manager does not emit "
                "offloading events.")

        super().__init__(config)
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config

        assert config.cache.blocks_per_chunk == 1, (
            "TPURaidenOffloadingConnector: requires blocks_per_chunk == 1 (one OffloadKey "
            "= one scheduler device block); got "
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
         device_block_size) = resolve_kernel_geometry(vllm_config,
                                                      kv_cache_config)
        assert device_block_size % kernel_block_size == 0
        self.device_block_size_factor = device_block_size // kernel_block_size

        # Derive the block-major layout contract (returns None if disabled).
        # Ensures scheduler and worker ranks agree on namespace salt and bundle geometry ahead of tensor allocation.
        self.block_major_contract = resolve_block_major_contract(
            vllm_config, kv_cache_config)

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
            self.offload_logical_blocks_capacity *
            self.device_block_size_factor)

        # Cross-instance KV sharing: configured via extra_config. Setting
        # global_registry_address enables remote lookups and RDMA fetches;
        # store_server_ip advertises the pod IP that peers dial for remote reads.
        self.global_registry_address = str(
            self.extra_config.get("global_registry_address", ""))
        self.store_server_ip = str(self.extra_config.get(
            "store_server_ip", ""))
        # KV pool group this store's KVTransferSpec is published under; the
        # store nodes absorbing this pool's evictions name the same group.
        # Empty falls back to the raiden job name, which the DP suffix below
        # makes per-replica -- set it explicitly whenever another party must
        # resolve this pool's spec.
        self.kv_pool_group = str(self.extra_config.get("kv_pool_group", ""))

        # Generate the compatibility namespace hash to isolate global registry
        # entries across incompatible engine configurations.
        self.key_namespace = derive_offload_namespace(
            vllm_config,
            kernel_block_size=kernel_block_size,
            per_block_shape=per_block_shape,
            kv_dtype=kv_dtype,
            device_block_size=device_block_size,
            world_size=self.world_size,
            num_kv_cache_groups=len(kv_cache_config.kv_cache_groups),
            num_kv_cache_tensors=len(kv_cache_config.kv_cache_tensors),
            cp_geometry=cp_geometry,
            block_major_contract=self.block_major_contract,
        )

        # Shared-memory host pools (RAIDEN_SHM_KEY) are currently unsupported.
        # Fail fast during initialization to prevent serving corrupted cache.
        if envs.RAIDEN_SHM_KEY:
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
                device_block_size_factor=self.device_block_size_factor,
                world_size=self.world_size,
                raiden_controller_port=self.raiden_controller_port,
                raiden_job_name=self.raiden_job_name,
                global_registry_address=self.global_registry_address,
                store_server_ip=self.store_server_ip,
                kv_pool_group=self.kv_pool_group,
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
                block_major_contract=self.block_major_contract,
            )
        return self._worker
