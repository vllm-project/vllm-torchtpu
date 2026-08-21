# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Raiden KV offload subsystem.

Tests scheduler and store components using a FakeKVCacheStore without requiring
physical TPU hardware or native C++ wheels:

1. Registration & Lifecycle:
   - Registration gate: Blocks operations until all worker ranks register.
   - Sub-hash expansion: OffloadKey maps to device_block_size_factor physical sub-hashes.
   - Atomic admission & cleanup: insert on submit, a post-admission lookup
     classifying the pinned batch, and a terminal release of exactly the pins
     no transfer spent.
   - Pin lifecycle: lookup and insert hand out pins, a landed save or local
     load spends one, and a peer read neither takes nor spends one.
   - Load lifecycle: Pins acquired in prepare_load, spent on polled completion.
   - LRU touch: Refreshes MRU priority via lookup+release.

2. Resilience & Fault Tolerance:
   - Store retry: Re-issues failed saves (suppressed during memory reclamation drains).
   - Drain synchronization: Reclaims HBM blocks safely and poisons timed-out jobs.
   - Sub-hash classification: Dispatches DMA only for newly admitted HBM entries.
   - Pin race recovery: Recovers gracefully if an entry is evicted during prepare_load.
   - Candidate safety: lookup never resurrects a displaced entry out of the
     eviction-candidate list, so touch() cannot either.

3. Distributed Cross-Node Sharing:
   - Tiered lookup: Queries global registry on local misses; degrades safely on outages.
   - Hybrid loading: Partitions requests into local DMA and remote RDMA concurrently.
   - Peer reads: Borrow nothing locally — no landing slot, no pin, and no
     directory entry once the bytes land.
   - Compatibility namespace: Generates deterministic 16-byte hashes to isolate entries.
"""
import enum
import os
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.v1.kv_offload.base import GPULoadStoreSpec, LookupResult, ReqContext


def _key(n: int) -> bytes:
    return b"hash_%04d" % n


def _ctx(req_id: str = "req") -> ReqContext:
    return ReqContext(req_id=req_id)


class FakeBlockStatus(enum.Enum):
    INIT = 0
    REMOTE = 1
    HOST = 2
    HBM = 3
    HOST_AND_HBM = 4


class FakeRaidenBlockId:

    def __init__(self,
                 raiden_id=None,
                 host_block_id=-1,
                 status=FakeBlockStatus.INIT,
                 device_block_id=-1):
        self.raiden_id = raiden_id
        self.host_block_id = host_block_id
        self.device_block_id = device_block_id
        self.status = status


FAKE_STORE_TYPES = SimpleNamespace(BlockStatus=FakeBlockStatus,
                                   RaidenBlockId=FakeRaidenBlockId)


@dataclass
class _Entry:
    status: FakeBlockStatus
    device_block_id: int = -1
    pin_count: int = 0
    # Mirrors Raiden's eviction-candidate list: a displaced entry that still
    # physically exists (and holds its host block) but is invisible to
    # lookup. A same-hash insert reclaims it; nothing else brings it back.
    is_candidate: bool = False


@dataclass
class _Call:
    name: str
    args: tuple


class FakeKVCacheStore:
    """In-memory stand-in for tpu_sync's KVCacheStore python wrapper."""

    def __init__(self):
        self.entries: dict[bytes, _Entry] = {}
        self.calls: list[_Call] = []
        self.fail_insert = False
        self.fail_save = False
        self.fail_load = False
        self.fail_read_remote = False
        self.raiden_id = object()
        # Global registry entries: maps sub-hash -> peer REMOTE slice.
        # Queried on local miss when enable_global=True. registry_down simulates
        # graceful degradation during registry outages (falls back to local).
        self.global_entries: dict[bytes, FakeRaidenBlockId] = {}
        self.registry_down = False
        # Queued terminal outcomes drained asynchronously by poll methods.
        self._save_done: list[bytes] = []
        self._save_failed: list[bytes] = []
        self._load_done: list[bytes] = []
        self._load_failed: list[bytes] = []
        self._remote_done: list[bytes] = []
        self._remote_failed: list[bytes] = []
        self._saving: list[bytes] = []
        self._loading: list[bytes] = []
        self._remote_reading: list[bytes] = []
        self._remote_device_ids: dict[bytes, int] = {}

    # -- store surface used by the manager -----------------------------

    def lookup(self, block_hashes, enable_global=False, pin_found=True):
        # Mirrors Raiden: every locally cached hit is PINNED as it is found, so
        # the answer cannot be evicted before the caller uses it, unless the
        # caller opts out with pin_found=False. Registry-only hits name a
        # peer's block and are never pinned. Candidates are peeked, never
        # pinned, so a lookup cannot resurrect one.
        self.calls.append(
            _Call("lookup", (list(block_hashes), enable_global, pin_found)))
        out = []
        for h in block_hashes:
            entry = self.entries.get(h)
            if entry is not None and entry.is_candidate:
                # Candidates are invisible to lookup (Peek semantics).
                entry = None
            if entry is None:
                if enable_global and not self.registry_down:
                    remote = self.global_entries.get(h)
                    if remote is not None:
                        out.append((h, remote))
                        continue
                break
            if pin_found:
                entry.pin_count += 1
            out.append((
                h,
                SimpleNamespace(status=entry.status,
                                device_block_id=entry.device_block_id,
                                host_block_id=-1),
            ))
        return out

    def release(self, block_hashes) -> None:
        self.calls.append(_Call("release", (list(block_hashes), )))
        for h in block_hashes:
            entry = self.entries.get(h)
            if entry is not None and entry.pin_count > 0:
                entry.pin_count -= 1

    def insert(self, block_hashes, slices, on_host) -> bool:
        self.calls.append(
            _Call("insert", (list(block_hashes), list(slices), on_host)))
        if self.fail_insert:
            return False
        for h, s in zip(block_hashes, slices):
            entry = self.entries.get(h)
            if entry is not None and entry.is_candidate:
                # Mirrors Raiden's ReclaimStaleCandidate: a same-hash
                # re-admission erases the stale candidate (host block
                # returned, registry unregistered) and inserts fresh.
                del self.entries[h]
                entry = None
            if entry is None:
                self.entries[h] = _Entry(status=s.status,
                                         device_block_id=s.device_block_id,
                                         pin_count=1)
            else:
                entry.pin_count += 1
        return True

    def save(self, block_hashes, dst_raiden_id=None) -> bool:
        self.calls.append(_Call("save", (list(block_hashes), dst_raiden_id)))
        if self.fail_save:
            return False
        # Mirror Raiden C++ preconditions: hashes must exist in HBM status and be pinned.
        for h in block_hashes:
            entry = self.entries.get(h)
            if (entry is None or entry.status != FakeBlockStatus.HBM
                    or entry.pin_count <= 0 or h in self._saving):
                return False
        self._saving.extend(block_hashes)
        return True

    def load(self, block_hashes, device_block_ids, slices=None) -> bool:
        self.calls.append(
            _Call("load", (list(block_hashes), list(device_block_ids))))
        if self.fail_load:
            return False
        # Mirror Raiden C++ preconditions for a local source: the entry must be
        # resident and pinned, and the successful load spends that pin.
        for h in block_hashes:
            entry = self.entries.get(h)
            if entry is None or entry.pin_count <= 0 or h in self._loading:
                return False
        self._loading.extend(block_hashes)
        return True

    def read_remote(self, block_hashes, slices, device_block_ids):
        self.calls.append(
            _Call("read_remote",
                  (list(block_hashes), list(slices), list(device_block_ids))))
        if self.fail_read_remote:
            return False
        # Mirror Raiden: a peer read consults nothing locally and needs no pin.
        # It only needs a REMOTE slice naming the owner, and one destination
        # device block per hash.
        if len(slices) != len(block_hashes) or len(device_block_ids) != len(
                block_hashes):
            return False
        for h, sl in zip(block_hashes, slices):
            if sl is None or sl.status != FakeBlockStatus.REMOTE:
                return False
            if h in self._remote_reading:
                return False
        self._remote_reading.extend(block_hashes)
        for h, d in zip(block_hashes, device_block_ids):
            self._remote_device_ids[h] = d
        return True

    def poll_save_status(self):
        # Mirrors Raiden's five-element save poll: (done, failed, pending,
        # existing, unregistered). The last two annotate REMOTE save failures
        # and stay empty here, because this connector only saves locally.
        done, self._save_done = self._save_done, []
        failed, self._save_failed = self._save_failed, []
        return done, failed, list(self._saving), [], []

    def poll_load_status(self):
        done, self._load_done = self._load_done, []
        failed, self._load_failed = self._load_failed, []
        return done, failed, list(self._loading)

    def poll_remote_read_status(self):
        done, self._remote_done = self._remote_done, []
        failed, self._remote_failed = self._remote_failed, []
        return done, failed, list(self._remote_reading)

    # -- test controls --------------------------------------------------

    def complete_save(self, block_hashes, success=True) -> None:
        # A save that lands spends one pin per hash; a failed one spends none,
        # which is what makes the caller's retry legal.
        for h in block_hashes:
            self._saving.remove(h)
            if success:
                entry = self.entries[h]
                entry.status = FakeBlockStatus.HOST_AND_HBM
                entry.pin_count -= 1
                self._save_done.append(h)
            else:
                self._save_failed.append(h)

    def complete_load(self, block_hashes, success=True) -> None:
        # Same contract as a save: only a landed load spends the pin.
        for h in block_hashes:
            self._loading.remove(h)
            if success:
                self.entries[h].pin_count -= 1
                self._load_done.append(h)
            else:
                self._load_failed.append(h)

    def complete_remote_read(self, block_hashes, success=True) -> None:
        # The bytes land in the destination device blocks and nowhere else:
        # a peer read records nothing locally either way.
        for h in block_hashes:
            self._remote_reading.remove(h)
            (self._remote_done if success else self._remote_failed).append(h)

    def calls_named(self, name: str) -> list[_Call]:
        return [c for c in self.calls if c.name == name]

    def pinned_hashes(self) -> dict[bytes, int]:
        """Sub-hashes the store still holds a pin for, and how many."""
        return {h: e.pin_count for h, e in self.entries.items() if e.pin_count}


def make_manager(store,
                 *,
                 device_block_size_factor=4,
                 num_blocks=8,
                 world_size=2,
                 **kwargs):
    from vllm_torchtpu.offload.raiden_store import RaidenOffloadingManager
    return RaidenOffloadingManager(
        kernel_physical_blocks_capacity=num_blocks * device_block_size_factor,
        offload_logical_blocks_capacity=num_blocks,
        device_block_size_factor=device_block_size_factor,
        world_size=world_size,
        raiden_controller_port=0,
        raiden_job_name="test",
        store=store,
        store_types=FAKE_STORE_TYPES,
        **kwargs,
    )


def make_global_manager(store, **kwargs):
    """Manager with the global registry enabled."""
    # Set PYTHONHASHSEED to satisfy registry requirement for cross-replica hash parity.
    with patch.dict(os.environ, {"PYTHONHASHSEED": "0"}):
        return make_manager(store,
                            global_registry_address="registry:50051",
                            store_server_ip="10.0.0.1",
                            **kwargs)


def sub_hashes(key: bytes,
               device_block_size_factor: int = 4,
               key_namespace: bytes = b"",
               ns: bytes = b"") -> list[bytes]:
    prefix = key_namespace or ns
    return [
        prefix + key + i.to_bytes(4, "big")
        for i in range(device_block_size_factor)
    ]


def seed_host_entries(store,
                      key: bytes,
                      device_block_size_factor: int = 4) -> None:
    for sub_hash in sub_hashes(key, device_block_size_factor):
        store.entries[sub_hash] = _Entry(status=FakeBlockStatus.HOST)


def displace_entries(store,
                     key: bytes,
                     device_block_size_factor: int = 4) -> None:
    """Move `key`'s entries to the eviction-candidate list: invisible to
    lookup but still physically present (holding their host blocks), exactly
    as a store admission's displacement leaves them until the GC or a
    same-hash re-admission collects them."""
    for sub_hash in sub_hashes(key, device_block_size_factor):
        store.entries[sub_hash].is_candidate = True


def seed_global_entries(store,
                        key: bytes,
                        device_block_size_factor: int = 4,
                        key_namespace: bytes = b"",
                        ns: bytes = b"") -> None:
    """Publish a peer's REMOTE slices for `key` in the fake registry."""
    prefix = key_namespace or ns
    for i, sub_hash in enumerate(
            sub_hashes(key, device_block_size_factor, prefix)):
        store.global_entries[sub_hash] = FakeRaidenBlockId(
            raiden_id=object(),
            host_block_id=100 + i,
            status=FakeBlockStatus.REMOTE)


class TestConstructionGuards(unittest.TestCase):

    def test_spec_rejects_kv_cache_events(self):
        # The manager emits no offloading events (displaced keys are
        # rediscovered by directory probes), so a configuration that expects
        # KV cache events must be refused at startup, before any config
        # derivation, rather than silently dropping BlockRemoved
        # notifications.
        from vllm_torchtpu.offload.raiden_store import \
            TPURaidenStoreOffloadingSpec
        vllm_config = MagicMock()
        vllm_config.kv_events_config.enable_kv_cache_events = True
        with self.assertRaisesRegex(ValueError, "enable_kv_cache_events"):
            TPURaidenStoreOffloadingSpec(MagicMock(), vllm_config, MagicMock())

    def test_registry_requires_pythonhashseed(self):
        # Require PYTHONHASHSEED to prevent hash mismatch across distributed replicas.
        env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                make_manager(FakeKVCacheStore(),
                             global_registry_address="registry:50051",
                             store_server_ip="10.0.0.1")


class TestLookup(unittest.TestCase):

    def setUp(self):
        self.store = FakeKVCacheStore()
        self.manager = make_manager(self.store)

    def test_hit_requires_every_sub_hash(self):
        seed_host_entries(self.store, _key(0))
        self.assertEqual(self.manager.lookup(_key(0), _ctx()),
                         LookupResult.HIT)
        # Partial sub-block availability results in a lookup MISS.
        del self.store.entries[sub_hashes(_key(0))[2]]
        self.assertEqual(self.manager.lookup(_key(0), _ctx()),
                         LookupResult.MISS)

    def test_hbm_status_is_hit_pending(self):
        for h in sub_hashes(_key(0)):
            self.store.entries[h] = _Entry(status=FakeBlockStatus.HBM)
        self.assertEqual(self.manager.lookup(_key(0), _ctx()),
                         LookupResult.HIT_PENDING)

    def test_remote_status_is_miss_without_registry(self):
        for h in sub_hashes(_key(0)):
            self.store.entries[h] = _Entry(status=FakeBlockStatus.REMOTE)
        self.assertEqual(self.manager.lookup(_key(0), _ctx()),
                         LookupResult.MISS)


class TestStoreJobLifecycle(unittest.TestCase):

    def setUp(self):
        self.store = FakeKVCacheStore()
        self.manager = make_manager(self.store)

    def _admit(self, keys, device_blocks):
        out = self.manager.prepare_store(keys, _ctx())
        self.assertEqual(out.keys_to_store, keys)
        self.manager.submit_store_job(7, keys, device_blocks)

    def test_submit_expands_sub_hashes_and_device_blocks(self):
        self._admit([_key(0), _key(1)], [5, 9])

        (call, ) = self.store.calls_named("insert")
        hashes, slices, on_host = call.args
        self.assertEqual(hashes, sub_hashes(_key(0)) + sub_hashes(_key(1)))
        self.assertFalse(on_host)
        self.assertEqual([s.device_block_id for s in slices],
                         [20, 21, 22, 23, 36, 37, 38, 39])
        self.assertTrue(all(s.status == FakeBlockStatus.HBM for s in slices))
        (save_call, ) = self.store.calls_named("save")
        self.assertEqual(save_call.args[0], hashes)

    def test_success_spends_every_pin_and_marks_stored(self):
        self._admit([_key(0)], [5])
        self.assertTrue(self.manager.has_pending_work())
        self.assertEqual(self.manager.poll_finished_jobs(), [])

        releases_before = len(self.store.calls_named("release"))
        self.store.complete_save(sub_hashes(_key(0)))
        (finished_job, ) = self.manager.poll_finished_jobs()
        self.assertTrue(finished_job.success)
        self.assertEqual(finished_job.job_id, 7)
        # Every sub-hash was saved, and each save spent the pin its admission
        # took, so the job ends owing nothing back.
        self.assertEqual(len(self.store.calls_named("release")),
                         releases_before)
        self.assertEqual(self.store.pinned_hashes(), {})
        self.assertFalse(self.manager.has_pending_work())
        # Promoted to HIT and excluded from subsequent prepare_store calls.
        self.assertEqual(self.manager.lookup(_key(0), _ctx()),
                         LookupResult.HIT)
        out = self.manager.prepare_store([_key(0), _key(1)], _ctx())
        self.assertEqual(out.keys_to_store, [_key(1)])

    def test_partial_batch_failure_retries_then_cleans_exact_batch(self):
        self._admit([_key(0), _key(1)], [5, 9])
        batch = sub_hashes(_key(0)) + sub_hashes(_key(1))
        # Failed sub-blocks are retried once; committed entries are not re-saved.
        self.store.complete_save(batch[:1], success=False)
        self.assertEqual(self.manager.poll_finished_jobs(), [])
        self.store.complete_save(batch[1:], success=True)
        self.assertEqual(self.manager.poll_finished_jobs(), [])
        retry = self.store.calls_named("save")[-1]
        self.assertEqual(retry.args[0], batch[:1])
        # Terminal failure upon retry exhaustion triggers exact-batch rollback.
        self.store.complete_save(batch[:1], success=False)
        (finished_job, ) = self.manager.poll_finished_jobs()
        self.assertFalse(finished_job.success)
        # The saves that landed already spent their pins; the terminal cleanup
        # gives back exactly the one the sub-hash that never saved still holds.
        release_call = self.store.calls_named("release")[-1]
        self.assertEqual(release_call.args[0], batch[:1])
        self.assertEqual(self.store.pinned_hashes(), {})
        # The store offers no way to remove a directory entry, so the sub-hash
        # whose save failed stays behind as an unpinned HBM entry — evictable,
        # but visible until the store reclaims it. That leaves key 0 with a
        # broken chain (its first sub-hash reads as a save still in flight, so
        # the key is re-offered) while key 1 is fully host-resident (HIT,
        # skipped by the directory probe).
        self.assertEqual(self.manager.lookup(_key(0), _ctx()),
                         LookupResult.HIT_PENDING)
        self.assertEqual(self.manager.lookup(_key(1), _ctx()),
                         LookupResult.HIT)
        out = self.manager.prepare_store([_key(0), _key(1)], _ctx())
        self.assertEqual(out.keys_to_store, [_key(0)])

    def test_save_retry_succeeds_and_marks_stored(self):
        self._admit([_key(0)], [5])
        batch = sub_hashes(_key(0))
        self.store.complete_save(batch, success=False)
        self.assertEqual(self.manager.poll_finished_jobs(), [])
        retry = self.store.calls_named("save")[-1]
        self.assertEqual(retry.args[0], batch)
        self.store.complete_save(batch, success=True)
        (finished_job, ) = self.manager.poll_finished_jobs()
        self.assertTrue(finished_job.success)
        self.assertEqual(self.manager.lookup(_key(0), _ctx()),
                         LookupResult.HIT)

    def test_mixed_terminal_hashes_in_one_poll_finalize_once(self):
        self._admit([_key(0)], [5])
        batch = sub_hashes(_key(0))
        # Mixed batch outcomes in a single poll trigger a retry followed by cleanup.
        self.store.complete_save(batch[:1], success=False)
        self.store.complete_save(batch[1:], success=True)
        self.assertEqual(self.manager.poll_finished_jobs(), [])
        self.store.complete_save(batch[:1], success=False)
        finished = self.manager.poll_finished_jobs()
        self.assertEqual(len(finished), 1)
        self.assertFalse(finished[0].success)
        # Only the sub-hash whose save never landed still owes its pin back.
        self.assertEqual(
            self.store.calls_named("release")[-1].args[0], batch[:1])
        self.assertEqual(self.store.pinned_hashes(), {})
        self.assertFalse(self.manager.has_pending_work())

    def test_rejected_admission_is_immediate_terminal_failure(self):
        self.store.fail_insert = True
        self._admit([_key(0)], [5])
        self.assertEqual(self.store.calls_named("save"), [])
        # A rejected insert pins nothing, so there is nothing to give back.
        self.assertEqual(self.store.calls_named("release"), [])
        (finished_job, ) = self.manager.poll_finished_jobs()
        self.assertFalse(finished_job.success)
        self.assertFalse(self.manager.has_pending_work())

    def test_failed_save_launch_reverts_admission(self):
        self.store.fail_save = True
        self._admit([_key(0)], [5])
        # Nothing transferred, so the whole admitted batch is given back.
        release_call = self.store.calls_named("release")[-1]
        self.assertEqual(release_call.args[0], sub_hashes(_key(0)))
        self.assertEqual(self.store.pinned_hashes(), {})
        (finished_job, ) = self.manager.poll_finished_jobs()
        self.assertFalse(finished_job.success)

    def test_inflight_keys_report_hit_pending_and_skip_restore(self):
        self._admit([_key(0)], [5])
        self.assertEqual(self.manager.lookup(_key(0), _ctx()),
                         LookupResult.HIT_PENDING)
        out = self.manager.prepare_store([_key(0)], _ctx())
        self.assertEqual(out.keys_to_store, [])

    def test_cancel_clears_inflight_tracking(self):
        out = self.manager.prepare_store([_key(0)], _ctx())
        self.assertEqual(out.keys_to_store, [_key(0)])
        self.manager.on_store_job_cancelled([_key(0)])
        out = self.manager.prepare_store([_key(0)], _ctx())
        self.assertEqual(out.keys_to_store, [_key(0)])


class TestLoadJobLifecycle(unittest.TestCase):

    def setUp(self):
        self.store = FakeKVCacheStore()
        self.manager = make_manager(self.store)
        seed_host_entries(self.store, _key(0))
        seed_host_entries(self.store, _key(1))

    def test_load_success_releases_only_from_finalizer(self):
        self.manager.prepare_load([_key(0), _key(1)], _ctx("r1"))
        batch = sub_hashes(_key(0)) + sub_hashes(_key(1))
        for h in batch:
            self.assertEqual(self.store.entries[h].pin_count, 1)

        self.manager.submit_load_job(3, [_key(0), _key(1)], [10, 11], "r1")
        (load, ) = self.store.calls_named("load")
        self.assertEqual(load.args[0], batch)
        self.assertEqual(load.args[1], [40, 41, 42, 43, 44, 45, 46, 47])

        # complete_load is a no-op; unpinning is driven by polled DMA completion.
        self.manager.complete_load([_key(0), _key(1)], _ctx("r1"))
        self.assertEqual(self.store.entries[batch[0]].pin_count, 1)

        self.store.complete_load(batch)
        (finished_job, ) = self.manager.poll_finished_jobs()
        self.assertTrue(finished_job.success)
        self.assertEqual(finished_job.req_id, "r1")
        self.assertEqual(finished_job.failed_device_block_ids, [])
        for h in batch:
            self.assertEqual(self.store.entries[h].pin_count, 0)

    def test_load_failure_reports_dst_device_blocks_and_keeps_host(self):
        self.manager.prepare_load([_key(0)], _ctx("r1"))
        self.manager.submit_load_job(3, [_key(0)], [10], "r1")
        self.store.complete_load(sub_hashes(_key(0)), success=False)
        (finished_job, ) = self.manager.poll_finished_jobs()
        self.assertFalse(finished_job.success)
        self.assertEqual(finished_job.req_id, "r1")
        self.assertEqual(finished_job.failed_device_block_ids, [10])
        # Pre-existing HOST entries remain intact and unpinned upon failure.
        for h in sub_hashes(_key(0)):
            self.assertEqual(self.store.entries[h].status,
                             FakeBlockStatus.HOST)
            self.assertEqual(self.store.entries[h].pin_count, 0)

    def test_failed_load_launch_is_immediate_terminal(self):
        self.store.fail_load = True
        self.manager.prepare_load([_key(0)], _ctx("r1"))
        self.manager.submit_load_job(3, [_key(0)], [10], "r1")
        (finished_job, ) = self.manager.poll_finished_jobs()
        self.assertFalse(finished_job.success)
        self.assertEqual(finished_job.failed_device_block_ids, [10])
        for h in sub_hashes(_key(0)):
            self.assertEqual(self.store.entries[h].pin_count, 0)

    def test_prepare_load_pin_failure_fails_job_cleanly(self):
        # Losing the lookup-to-pin race against eviction fails the job cleanly without raising.
        spec = self.manager.prepare_load([_key(9)], _ctx("r1"))
        self.assertFalse(spec.pinned)
        self.manager.submit_load_job(3, [_key(9)], [10],
                                     "r1",
                                     pinned=spec.pinned)
        self.assertEqual(self.store.calls_named("load"), [])
        self.assertEqual(self.store.calls_named("release"), [])
        (finished_job, ) = self.manager.poll_finished_jobs()
        self.assertFalse(finished_job.success)
        self.assertEqual(finished_job.failed_device_block_ids, [10])


class TestTouchAndMisc(unittest.TestCase):

    def setUp(self):
        self.store = FakeKVCacheStore()
        self.manager = make_manager(self.store)

    def test_reset_cache_raises(self):
        # External cache reset is unsupported; verify that calling reset_cache raises NotImplementedError.
        with self.assertRaises(NotImplementedError):
            self.manager.reset_cache()

    def test_connector_reset_cache_reports_failure(self):
        # The connector-level override must return False (not raise) so the
        # scheduler's reset_connector_cache surfaces {"success": false}
        # through /reset_prefix_cache instead of an opaque 500.
        from vllm_torchtpu.offload.raiden_connector import \
            TPURaidenOffloadingConnector
        self.assertIs(TPURaidenOffloadingConnector.reset_cache(MagicMock()),
                      False)

    def test_touch_refreshes_resident_keys_only(self):
        seed_host_entries(self.store, _key(0))
        # The directory decides: the lookup pins whatever it finds and the
        # release hands it straight back, which is the touch. An absent key
        # matches nothing, so nothing is pinned or released for it.
        self.manager.touch([_key(0), _key(1)], _ctx())
        (release, ) = self.store.calls_named("release")
        self.assertEqual(release.args[0], sub_hashes(_key(0)))
        self.assertEqual(self.store.pinned_hashes(), {})
        self.store.calls.clear()

        # A key made resident by a full store job is touchable the same way.
        out = self.manager.prepare_store([_key(2)], _ctx())
        self.manager.submit_store_job(1, out.keys_to_store, [5])
        self.store.complete_save(sub_hashes(_key(2)))
        self.manager.poll_finished_jobs()
        self.store.calls.clear()

        self.manager.touch([_key(2), _key(1)], _ctx())
        (release, ) = self.store.calls_named("release")
        self.assertEqual(release.args[0], sub_hashes(_key(2)))
        self.assertEqual(self.store.pinned_hashes(), {})

    def test_probes_do_not_reorder_lru(self):
        # lookup/prepare_store only read the directory to decide what to do.
        # Pinning a hit and releasing it would leave the entry at the Most
        # Recently Used position, silently reordering eviction -- so they ask
        # for no pin at all. Moving an entry is touch's job, and only touch's.
        seed_host_entries(self.store, _key(0))
        self.store.calls.clear()

        self.manager.lookup(_key(0), _ctx())
        self.manager.prepare_store([_key(0)], _ctx())
        for call in self.store.calls_named("lookup"):
            pin_found = call.args[2]
            self.assertFalse(pin_found,
                             "a probe must not take a pin: %r" % (call.args, ))
        self.assertEqual(self.store.pinned_hashes(), {})

        # touch, by contrast, deliberately does take one and give it back.
        self.store.calls.clear()
        self.manager.touch([_key(0)], _ctx())
        (probe, ) = self.store.calls_named("lookup")
        self.assertTrue(probe.args[2])
        self.assertEqual(len(self.store.calls_named("release")), 1)
        self.assertEqual(self.store.pinned_hashes(), {})

    def test_touch_never_resurrects_displaced_entries(self):
        # A displaced entry sits in Raiden's eviction-candidate list: invisible
        # to lookup, and a lookup pins only what it can see, so a touch cannot
        # pull one back into the active set past capacity — on any call,
        # because nothing caches the key as "stored".
        seed_host_entries(self.store, _key(0))
        self.assertEqual(self.manager.lookup(_key(0), _ctx()),
                         LookupResult.HIT)
        displace_entries(self.store, _key(0))
        self.store.calls.clear()

        self.manager.touch([_key(0)], _ctx())
        # The candidates were left untouched (not resurrected, not pinned).
        for h in sub_hashes(_key(0)):
            self.assertTrue(self.store.entries[h].is_candidate)
        self.assertEqual(self.store.pinned_hashes(), {})
        # Still true on a repeat touch: probe again, still see nothing.
        self.store.calls.clear()
        self.manager.touch([_key(0)], _ctx())
        self.assertEqual(len(self.store.calls_named("lookup")), 1)
        self.assertEqual(self.store.pinned_hashes(), {})

    def test_drain_jobs_blocks_until_terminal(self):
        out = self.manager.prepare_store([_key(0)], _ctx())
        self.manager.submit_store_job(4, out.keys_to_store, [5])

        # Simulate concurrent DMA completion arriving before drain.
        self.store.complete_save(sub_hashes(_key(0)))
        drained = self.manager.drain_jobs({4})
        self.assertEqual([fj.job_id for fj in drained], [4])
        self.assertFalse(self.manager.is_job_launched(4))
        # Ensure completed jobs are not double-reported.
        self.assertEqual(self.manager.poll_finished_jobs(), [])


class TestSchedulerFenceFlow(unittest.TestCase):
    """TPURaidenOffloadingScheduler park/fence/cancel logic, with the stock
    base-class methods patched out."""

    def _make_scheduler(self, manager, num_workers=2):
        from vllm_torchtpu.offload.raiden_connector import \
            TPURaidenOffloadingScheduler
        sched = object.__new__(TPURaidenOffloadingScheduler)
        sched.config = SimpleNamespace(num_workers=num_workers)
        sched._raiden_manager = manager
        sched._fence_pending = {}
        return sched

    def _stock_meta(self, store_jobs=None, load_jobs=None, jobs_to_flush=None):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import \
            OffloadingConnectorMetadata
        return OffloadingConnectorMetadata(
            load_jobs=load_jobs or {},
            store_jobs=store_jobs or {},
            jobs_to_flush=jobs_to_flush or set(),
        )

    def _store_job(self, keys, device_block_ids, req_id="r1"):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import \
            TransferJob

        from vllm_torchtpu.offload.raiden_store import RaidenLoadStoreSpec
        return TransferJob(
            req_id=req_id,
            src_spec=GPULoadStoreSpec(device_block_ids,
                                      group_sizes=[len(device_block_ids)],
                                      block_indices=[0]),
            dst_spec=RaidenLoadStoreSpec(keys),
        )

    def _load_job(self, keys, device_block_ids, req_id="r1"):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import \
            TransferJob

        from vllm_torchtpu.offload.raiden_store import RaidenLoadStoreSpec
        return TransferJob(
            req_id=req_id,
            src_spec=RaidenLoadStoreSpec(keys),
            dst_spec=GPULoadStoreSpec(device_block_ids,
                                      group_sizes=[len(device_block_ids)],
                                      block_indices=[0]),
        )

    def _build_meta(self, sched, stock_meta):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import \
            OffloadingConnectorScheduler
        with patch.object(OffloadingConnectorScheduler,
                          "build_connector_meta",
                          return_value=stock_meta):
            return sched.build_connector_meta(MagicMock())

    def _ack(self, sched, job_ids, count=1):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import \
            OffloadingConnectorScheduler

        from vllm_torchtpu.offload.raiden_connector import \
            RaidenOffloadingWorkerMetadata
        output = SimpleNamespace(
            kv_connector_worker_meta=RaidenOffloadingWorkerMetadata(
                fenced_jobs={jid: count
                             for jid in job_ids}))
        with patch.object(OffloadingConnectorScheduler,
                          "update_connector_output"):
            sched.update_connector_output(output)

    def test_store_jobs_parked_until_all_ranks_fence(self):
        store = FakeKVCacheStore()
        manager = make_manager(store)
        manager.prepare_store([_key(0)], _ctx())
        sched = self._make_scheduler(manager)

        meta = self._build_meta(
            sched,
            self._stock_meta(store_jobs={7: self._store_job([_key(0)], [5])}))

        # Shipped metadata: parked jobs, fence requests, and no echoes yet.
        self.assertEqual(meta.store_jobs, {})
        self.assertEqual(meta.load_jobs, {})
        self.assertEqual(meta.fence_job_ids, {7})
        self.assertEqual(meta.finished_store_job_ids, [])
        self.assertEqual(store.calls_named("insert"), [])

        # First rank acknowledges fence: store job remains parked.
        self._ack(sched, [7])
        self.assertEqual(store.calls_named("insert"), [])
        # All ranks acknowledge fence: store job is admitted and launched.
        self._ack(sched, [7])
        self.assertEqual(len(store.calls_named("insert")), 1)
        self.assertEqual(len(store.calls_named("save")), 1)
        self.assertNotIn(7, sched._fence_pending)

    def test_load_jobs_submit_immediately(self):
        store = FakeKVCacheStore()
        manager = make_manager(store)
        seed_host_entries(store, _key(0))
        manager.prepare_load([_key(0)], _ctx("r1"))
        sched = self._make_scheduler(manager)

        meta = self._build_meta(
            sched,
            self._stock_meta(load_jobs={3: self._load_job([_key(0)], [10])}))
        self.assertEqual(meta.load_jobs, {})
        (load, ) = store.calls_named("load")
        self.assertEqual(load.args[1], [40, 41, 42, 43])

        # Job completion is echoed in the subsequent step's metadata.
        store.complete_load(sub_hashes(_key(0)))
        meta2 = self._build_meta(sched, self._stock_meta())
        self.assertEqual(meta2.finished_load_jobs, {3: "r1"})
        self.assertEqual(meta2.failed_load_device_block_ids, [])

    def test_flush_cancels_fence_pending_store(self):
        store = FakeKVCacheStore()
        manager = make_manager(store)
        manager.prepare_store([_key(0)], _ctx())
        sched = self._make_scheduler(manager)
        self._build_meta(
            sched,
            self._stock_meta(store_jobs={7: self._store_job([_key(0)], [5])}))

        # Cancel parked store before rank acknowledgements when source blocks are reclaimed.
        meta = self._build_meta(sched, self._stock_meta(jobs_to_flush={7}))
        self.assertIn(7, meta.finished_store_job_ids)
        self.assertNotIn(7, sched._fence_pending)
        self.assertEqual(store.calls_named("insert"), [])
        # Cancelled keys become admittable again.
        out = manager.prepare_store([_key(0)], _ctx())
        self.assertEqual(out.keys_to_store, [_key(0)])

    def test_same_step_flush_ships_no_fence(self):
        store = FakeKVCacheStore()
        manager = make_manager(store)
        manager.prepare_store([_key(0)], _ctx())
        sched = self._make_scheduler(manager)

        # Created and flushed in same step: cancelled immediately without shipping redundant fence requests.
        meta = self._build_meta(
            sched,
            self._stock_meta(store_jobs={7: self._store_job([_key(0)], [5])},
                             jobs_to_flush={7}))
        self.assertEqual(meta.fence_job_ids, set())
        self.assertIn(7, meta.finished_store_job_ids)
        self.assertNotIn(7, sched._fence_pending)

    def test_flush_drains_launched_store(self):
        store = FakeKVCacheStore()
        manager = make_manager(store)
        manager.prepare_store([_key(0)], _ctx())
        sched = self._make_scheduler(manager)
        self._build_meta(
            sched,
            self._stock_meta(store_jobs={7: self._store_job([_key(0)], [5])}))
        self._ack(sched, [7], count=2)
        self.assertTrue(manager.is_job_launched(7))

        # Queue DMA completion to be observed during drain.
        store.complete_save(sub_hashes(_key(0)))
        meta = self._build_meta(sched, self._stock_meta(jobs_to_flush={7}))
        self.assertIn(7, meta.finished_store_job_ids)
        self.assertFalse(manager.is_job_launched(7))


class _FakeFinishedRequest:
    """Request stand-in for the finished-request store path."""

    def __init__(self, req_id, num_tokens, num_computed_tokens, status):
        self.request_id = req_id
        self.num_tokens = num_tokens
        self.num_computed_tokens = num_computed_tokens
        self.num_prompt_tokens = num_tokens
        self.status = status
        self.kv_transfer_params = None

    def is_finished(self):
        return True


class TestFinishedRequestStoreBuild(unittest.TestCase):
    """_build_store_jobs on aborted/finished requests, against the real
    upstream state machine (config, RequestOffloadState, req_status).

    An aborted request's offload_keys cover every full token block (hashes
    exist as soon as tokens are known) while block_ids only cover what got
    allocated before the abort. The connector relies on upstream bounding
    the stored chunks to blocks that were allocated AND computed
    (FINISHED_ABORTED caps at num_computed_tokens, storable_chunks caps at
    allocated blocks); these tests pin that contract so a vLLM bump that
    regresses it fails here instead of crashing an eval pod mid-abort.
    """

    TPC = 4  # tokens_per_chunk (= tokens_per_block, blocks_per_chunk=1)

    def _make_sched(self, manager):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (  # noqa: E501
            GroupOffloadConfig, SchedulerOffloadConfig)

        from vllm_torchtpu.offload.raiden_connector import \
            TPURaidenOffloadingScheduler
        sched = object.__new__(TPURaidenOffloadingScheduler)
        sched.config = SchedulerOffloadConfig(
            kv_group_configs=(GroupOffloadConfig(
                group_idx=0,
                tokens_per_block=self.TPC,
                tokens_per_chunk=self.TPC,
                hashes_per_chunk=1,
                kv_event_group_spec=MagicMock(),
                sliding_window_size_in_chunks=None), ),
            blocks_per_chunk=1,
            num_workers=2,
            offload_prompt_only=False)
        sched.manager = manager
        sched._raiden_manager = manager
        sched._fence_pending = {}
        sched._req_status = {}
        sched._jobs = {}
        sched._job_counter = 0
        sched._block_id_to_pending_jobs = {}
        sched._current_batch_allocated_block_ids = set()
        sched._current_batch_jobs_to_flush = set()
        sched._connector_stats = MagicMock()
        sched._events_tracker = MagicMock()
        return sched

    def _track(self, sched, req, keys, block_ids):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import \
            RequestOffloadState  # noqa: E501
        from vllm.v1.kv_offload.base import RequestOffloadingContext
        req_status = RequestOffloadState(
            config=sched.config,
            req=req,
            req_context=_ctx(req.request_id),
            offloading_context=RequestOffloadingContext())
        req_status.group_states[0].offload_keys.extend(keys)
        req_status.group_states[0].block_ids.extend(block_ids)
        sched._req_status[req.request_id] = req_status
        return req_status

    def _add_pending_store_job(self, sched, req_status, job_id=99):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import \
            TransferJobStatus  # noqa: E501
        req_status.transfer_jobs.add(job_id)
        sched._jobs[job_id] = TransferJobStatus(
            req_id=req_status.req.request_id,
            pending_count=2,
            keys=set(),
            is_store=True)

    def _sched_output(self, req_id):
        return SimpleNamespace(num_scheduled_tokens={},
                               finished_req_ids={req_id})

    def _aborted_setup(self,
                       num_tokens=32,
                       num_computed_tokens=12,
                       num_blocks=3):
        from vllm.v1.request import RequestStatus
        store = FakeKVCacheStore()
        sched = self._make_sched(make_manager(store))
        req = _FakeFinishedRequest("r1", num_tokens, num_computed_tokens,
                                   RequestStatus.FINISHED_ABORTED)
        keys = [_key(i) for i in range(num_tokens // self.TPC)]
        req_status = self._track(sched, req, keys,
                                 list(range(1, num_blocks + 1)))
        self._add_pending_store_job(sched, req_status)
        return sched, keys

    def test_aborted_request_stores_only_allocated_computed_chunks(self):
        # 8 chunks of hashes, 3 blocks allocated, 12 tokens computed: store chunks 0-2.
        sched, keys = self._aborted_setup()
        jobs = sched._build_store_jobs(self._sched_output("r1"))
        (job, ) = jobs.values()
        self.assertEqual(job.dst_spec.keys, keys[:3])
        self.assertEqual(job.src_spec.block_ids.tolist(), [1, 2, 3])

    def test_aborted_request_caps_at_computed_tokens(self):
        # 3 blocks allocated but only 10 tokens computed: unwritten third block is omitted.
        sched, keys = self._aborted_setup(num_computed_tokens=10)
        jobs = sched._build_store_jobs(self._sched_output("r1"))
        (job, ) = jobs.values()
        self.assertEqual(job.dst_spec.keys, keys[:2])
        self.assertEqual(job.src_spec.block_ids.tolist(), [1, 2])

    def test_waiting_abort_with_no_blocks_builds_no_job(self):
        # Request aborted from waiting queue before block allocation: no store issued.
        from vllm.v1.request import RequestStatus
        store = FakeKVCacheStore()
        sched = self._make_sched(make_manager(store))
        req = _FakeFinishedRequest("r1", 32, 0, RequestStatus.FINISHED_ABORTED)
        self._track(sched, req, [_key(i) for i in range(8)], [])
        jobs = sched._build_store_jobs(self._sched_output("r1"))
        self.assertEqual(jobs, {})
        # _build_store_jobs no longer cleans up request state; that is
        # deferred to build_connector_meta, which drops finished requests
        # with no outstanding transfer jobs.
        self.assertIn("r1", sched._req_status)

    def test_normally_finished_request_keys_untouched(self):
        # Normal completion: computed = num_tokens - 1, both chunks backed by blocks.
        from vllm.v1.request import RequestStatus
        store = FakeKVCacheStore()
        sched = self._make_sched(make_manager(store))
        req = _FakeFinishedRequest("r1", 8, 7, RequestStatus.FINISHED_STOPPED)
        keys = [_key(0), _key(1)]
        self._track(sched, req, keys, [1, 2])
        jobs = sched._build_store_jobs(self._sched_output("r1"))
        (job, ) = jobs.values()
        self.assertEqual(job.dst_spec.keys, keys)
        self.assertEqual(job.src_spec.block_ids.tolist(), [1, 2])


class TestWorkerSideConnector(unittest.TestCase):

    def _make_worker_connector(self):
        from vllm_torchtpu.offload.raiden_connector import \
            TPURaidenOffloadingConnector
        conn = object.__new__(TPURaidenOffloadingConnector)
        conn.connector_worker = MagicMock()
        conn._fenced_jobs = {}
        conn._completed_jobs = {}
        conn._load_error_block_ids = set()
        conn._pool_sync_tensors = []
        return conn

    def _meta(self, **kwargs):
        from vllm_torchtpu.offload.raiden_connector import \
            RaidenOffloadingConnectorMetadata
        return RaidenOffloadingConnectorMetadata(load_jobs={},
                                                 store_jobs={},
                                                 **kwargs)

    def test_fence_syncs_device_and_acks(self):
        conn = self._make_worker_connector()
        kv_tensors = [MagicMock()]
        conn._pool_sync_tensors = kv_tensors
        conn._connector_metadata = self._meta(fence_job_ids={7, 8})

        with patch("vllm_torchtpu.offload.raiden_connector.synchronize_tensors"
                   ) as sync:
            sending, recving = conn.get_finished(set())
        sync.assert_called_once_with(kv_tensors)
        self.assertEqual(conn._fenced_jobs, {7: 1, 8: 1})
        self.assertEqual((sending, recving), (set(), set()))

    def test_fence_requires_registered_pool_tensors(self):
        conn = self._make_worker_connector()
        conn._connector_metadata = self._meta(fence_job_ids={7})

        with patch("vllm_torchtpu.offload.raiden_connector.synchronize_tensors"
                   ) as sync:
            with self.assertRaises(AssertionError):
                conn.get_finished(set())
        sync.assert_not_called()
        self.assertEqual(conn._fenced_jobs, {})

    def test_fence_scopes_sync_to_registered_pool_tensors(self):
        conn = self._make_worker_connector()
        kv_tensors = [MagicMock(), MagicMock()]
        conn._pool_sync_tensors = kv_tensors
        conn._connector_metadata = self._meta(fence_job_ids={7})

        with patch("vllm_torchtpu.offload.raiden_connector.synchronize_tensors"
                   ) as sync:
            conn.get_finished(set())
        sync.assert_called_once_with(kv_tensors)

    def test_echoes_completions_and_finished_recving(self):
        conn = self._make_worker_connector()
        conn._connector_metadata = self._meta(
            finished_store_job_ids=[1, 2],
            finished_load_jobs={3: "req-a"},
            failed_load_device_block_ids=[10, 11],
        )
        sending, recving = conn.get_finished(set())
        self.assertEqual(sending, set())
        self.assertEqual(recving, {"req-a"})
        self.assertEqual(conn._completed_jobs, {1: 1, 2: 1, 3: 1})
        self.assertEqual(conn.get_block_ids_with_load_errors(), {10, 11})
        self.assertEqual(conn.get_block_ids_with_load_errors(), set())

    def test_build_connector_worker_meta_drains_state(self):
        conn = self._make_worker_connector()
        self.assertIsNone(conn.build_connector_worker_meta())
        conn._fenced_jobs = {7: 1}
        conn._completed_jobs = {3: 1}
        meta = conn.build_connector_worker_meta()
        self.assertEqual(meta.fenced_jobs, {7: 1})
        self.assertEqual(meta.completed_jobs, {3: 1})
        self.assertIsNone(conn.build_connector_worker_meta())

    def test_worker_metadata_aggregation_sums_both_channels(self):
        from vllm_torchtpu.offload.raiden_connector import \
            RaidenOffloadingWorkerMetadata
        a = RaidenOffloadingWorkerMetadata(completed_jobs={3: 1},
                                           fenced_jobs={7: 1})
        b = RaidenOffloadingWorkerMetadata(completed_jobs={
            3: 1,
            4: 1
        },
                                           fenced_jobs={7: 1})
        merged = a.aggregate(b)
        self.assertEqual(merged.completed_jobs, {3: 2, 4: 1})
        self.assertEqual(merged.fenced_jobs, {7: 2})


class TestFailureHardening(unittest.TestCase):
    """Drain-suppressed retries and failure handling."""

    def setUp(self):
        self.store = FakeKVCacheStore()
        self.manager = make_manager(self.store)

    def test_drain_suppresses_save_retry(self):
        out = self.manager.prepare_store([_key(0)], _ctx())
        self.manager.submit_store_job(4, out.keys_to_store, [5])
        # Drain disables retries to prevent re-reading reclaimed source blocks.
        self.store.complete_save(sub_hashes(_key(0)), success=False)
        drained = self.manager.drain_jobs({4})
        self.assertEqual([(fj.job_id, fj.success) for fj in drained],
                         [(4, False)])
        self.assertEqual(len(self.store.calls_named("save")), 1)
        # The save never landed, so the job gives its whole admitted batch back.
        self.assertEqual(
            self.store.calls_named("release")[-1].args[0], sub_hashes(_key(0)))
        self.assertEqual(self.store.pinned_hashes(), {})

    def test_failed_load_reports_failure(self):
        seed_host_entries(self.store, _key(0))
        self.assertEqual(self.manager.lookup(_key(0), _ctx()),
                         LookupResult.HIT)
        self.manager.prepare_load([_key(0)], _ctx("r1"))
        self.manager.submit_load_job(3, [_key(0)], [10], "r1")
        self.store.complete_load(sub_hashes(_key(0)), success=False)
        (finished_job, ) = self.manager.poll_finished_jobs()
        self.assertFalse(finished_job.success)
        self.assertEqual(finished_job.failed_device_block_ids, [10])

    def test_failed_save_launch_reports_failure(self):
        self.store.fail_save = True
        out = self.manager.prepare_store([_key(0)], _ctx())
        self.manager.submit_store_job(4, out.keys_to_store, [5])
        (finished_job, ) = self.manager.poll_finished_jobs()
        self.assertFalse(finished_job.success)
        self.assertEqual(finished_job.job_id, 4)

    def test_failed_load_launch_reports_failure(self):
        seed_host_entries(self.store, _key(0))
        self.assertEqual(self.manager.lookup(_key(0), _ctx()),
                         LookupResult.HIT)
        self.store.fail_load = True
        self.manager.prepare_load([_key(0)], _ctx("r1"))
        self.manager.submit_load_job(3, [_key(0)], [10], "r1")
        (finished_job, ) = self.manager.poll_finished_jobs()
        self.assertFalse(finished_job.success)
        self.assertEqual(finished_job.failed_device_block_ids, [10])

    def test_drain_timeout_poisons_inflight_job(self):
        import vllm_torchtpu.offload.raiden_store as rs
        out = self.manager.prepare_store([_key(0)], _ctx())
        self.manager.submit_store_job(4, out.keys_to_store, [5])
        # Timed-out drain poisons in-flight jobs to prevent committing corrupted bytes.
        with patch.object(rs, "_DRAIN_TIMEOUT_S", 0.01):
            self.assertEqual(self.manager.drain_jobs({4}), [])
            # Subsequent drains on already-poisoned jobs return immediately.
            self.assertEqual(self.manager.drain_jobs({4}), [])
        self.assertTrue(self.manager.is_job_launched(4))

        # Poisoned jobs finalize as terminal failures even if hardware reports success.
        releases_before = len(self.store.calls_named("release"))
        self.store.complete_save(sub_hashes(_key(0)))
        (finished_job, ) = self.manager.poll_finished_jobs()
        self.assertFalse(finished_job.success)
        self.assertEqual(finished_job.job_id, 4)
        # The hardware save landed and spent every pin, so a poisoned job that
        # reports failure still has nothing to give back.
        self.assertEqual(len(self.store.calls_named("release")),
                         releases_before)
        self.assertEqual(self.store.pinned_hashes(), {})
        # Verify poisoned job finalizes exactly once.
        self.assertEqual(self.manager.poll_finished_jobs(), [])
        self.assertFalse(self.manager.has_pending_work())


class TestMixedBatchesAndRecovery(unittest.TestCase):
    """Sub-hash classification at submit time: pre-existing HOST entries
    (recovered across a restart, or committed by a partially failed earlier
    save) are locked but never re-saved — raiden's save() rejects the whole
    launch if any hash is not HBM status."""

    def setUp(self):
        self.store = FakeKVCacheStore()
        self.manager = make_manager(self.store)

    def test_mixed_batch_saves_only_new_sub_hashes(self):
        batch = sub_hashes(_key(0))
        # Simulate partial residency where subset of sub-blocks already committed.
        for h in batch[:2]:
            self.store.entries[h] = _Entry(status=FakeBlockStatus.HOST)
        out = self.manager.prepare_store([_key(0)], _ctx())
        self.assertEqual(out.keys_to_store, [_key(0)])
        self.manager.submit_store_job(7, [_key(0)], [5])
        (save, ) = self.store.calls_named("save")
        self.assertEqual(save.args[0], batch[2:])
        self.store.complete_save(batch[2:])
        (finished_job, ) = self.manager.poll_finished_jobs()
        self.assertTrue(finished_job.success)
        # The saved sub-hashes spent their pins; cleanup gives back exactly
        # the pre-existing HOST entries the admission pinned in place.
        self.assertEqual(
            self.store.calls_named("release")[-1].args[0], batch[:2])
        self.assertEqual(self.store.pinned_hashes(), {})
        self.assertEqual(self.manager.lookup(_key(0), _ctx()),
                         LookupResult.HIT)

    def test_all_host_batch_finishes_without_save(self):
        out = self.manager.prepare_store([_key(0)], _ctx())
        self.assertEqual(out.keys_to_store, [_key(0)])
        # The whole batch becomes host-resident between the offer and the
        # submit (recovered across a restart, or committed by a partially
        # failed earlier save); submit pins it in place and finishes without
        # issuing a DMA save.
        seed_host_entries(self.store, _key(0))
        self.manager.submit_store_job(7, out.keys_to_store, [5])
        self.assertEqual(self.store.calls_named("save"), [])
        self.assertEqual(
            self.store.calls_named("release")[-1].args[0], sub_hashes(_key(0)))
        (finished_job, ) = self.manager.poll_finished_jobs()
        self.assertTrue(finished_job.success)
        self.assertEqual(self.store.pinned_hashes(), {})
        self.assertFalse(self.manager.has_pending_work())
        # The directory probe sees them host-resident, so a second offer of
        # the same key is skipped.
        out = self.manager.prepare_store([_key(0)], _ctx())
        self.assertEqual(out.keys_to_store, [])


class TestPostAdmissionClassification(unittest.TestCase):
    """A post-admission lookup decides which sub-hashes of the pinned batch
    this admission inserted (and must save), and displaced keys are
    rediscovered by directory probes rather than eviction events."""

    def setUp(self):
        self.store = FakeKVCacheStore()
        self.manager = make_manager(self.store)

    def _store_key(self, key, device_block, job_id):
        out = self.manager.prepare_store([key], _ctx())
        self.manager.submit_store_job(job_id, out.keys_to_store,
                                      [device_block])

    def test_lookup_classification_saves_only_new_sub_hashes(self):
        batch = sub_hashes(_key(0))
        for h in batch[:2]:
            self.store.entries[h] = _Entry(status=FakeBlockStatus.HOST)
        self._store_key(_key(0), 5, 7)
        (call, ) = self.store.calls_named("insert")
        self.assertEqual(call.args[0], batch)
        (save, ) = self.store.calls_named("save")
        self.assertEqual(save.args[0], batch[2:])

        self.store.complete_save(batch[2:])
        (finished_job, ) = self.manager.poll_finished_jobs()
        self.assertTrue(finished_job.success)

    def test_remote_sub_hashes_are_resident_and_not_saved(self):
        batch = sub_hashes(_key(0))
        for h in batch[:2]:
            self.store.entries[h] = _Entry(status=FakeBlockStatus.REMOTE)
        self._store_key(_key(0), 5, 7)
        (save, ) = self.store.calls_named("save")
        self.assertEqual(save.args[0], batch[2:])

    def test_fully_remote_key_is_not_offered(self):
        for h in sub_hashes(_key(0)):
            self.store.entries[h] = _Entry(status=FakeBlockStatus.REMOTE)
        out = self.manager.prepare_store([_key(0)], _ctx())
        self.assertEqual(out.keys_to_store, [])

    def test_foreign_hbm_binding_fails_closed(self):
        batch = sub_hashes(_key(0))
        # An HBM entry bound to a device block this job did not supply breaks
        # the connector's invariants; the job must fail, not guess.
        self.store.entries[batch[0]] = _Entry(status=FakeBlockStatus.HBM,
                                              device_block_id=99)
        self._store_key(_key(0), 5, 7)
        self.assertEqual(self.store.calls_named("save"), [])
        self.assertEqual(self.store.calls_named("release")[-1].args[0], batch)
        self.assertEqual(self.store.pinned_hashes(), {})
        (finished_job, ) = self.manager.poll_finished_jobs()
        self.assertFalse(finished_job.success)
        self.assertFalse(self.manager.has_pending_work())

    def test_displaced_key_rediscovered_by_directory_probe(self):
        seed_host_entries(self.store, _key(1))
        self.assertEqual(
            self.manager.prepare_store([_key(1)], _ctx()).keys_to_store, [])
        # Displacement moves the entries to the candidate list, invisible to
        # lookup. No event is emitted; the next prepare_store probe simply
        # re-offers the key.
        displace_entries(self.store, _key(1))
        self.assertEqual(list(self.manager.take_events()), [])
        out = self.manager.prepare_store([_key(1)], _ctx())
        self.assertEqual(out.keys_to_store, [_key(1)])

    def test_candidate_readmission_stores_cleanly(self):
        # The dangerous window: key 1's old entries still sit in the
        # eviction-candidate list when the key is re-offered and re-admitted.
        # Raiden's reclaim erases the stale candidates so the admission
        # inserts fresh HBM entries, and the full batch is saved.
        batch = sub_hashes(_key(1))
        seed_host_entries(self.store, _key(1))
        displace_entries(self.store, _key(1))

        out = self.manager.prepare_store([_key(1)], _ctx())
        self.assertEqual(out.keys_to_store, [_key(1)])
        self.manager.submit_store_job(7, [_key(1)], [5])
        # Reclaimed and re-inserted: fresh pinned HBM entries, old
        # candidates gone.
        for i, h in enumerate(batch):
            entry = self.store.entries[h]
            self.assertFalse(entry.is_candidate)
            self.assertEqual(entry.status, FakeBlockStatus.HBM)
            self.assertEqual(entry.device_block_id, 20 + i)
        (save, ) = self.store.calls_named("save")
        self.assertEqual(save.args[0], batch)

        self.store.complete_save(batch)
        (finished_job, ) = self.manager.poll_finished_jobs()
        self.assertTrue(finished_job.success)
        self.assertEqual(self.manager.lookup(_key(1), _ctx()),
                         LookupResult.HIT)


class TestRemoteReadFlow(unittest.TestCase):
    """Cross-instance receiver flow against the fake registry."""

    def setUp(self):
        self.store = FakeKVCacheStore()
        self.manager = make_global_manager(self.store)

    def test_remote_hit_requires_global_enabled(self):
        seed_global_entries(self.store, _key(1))
        local_manager = make_manager(self.store)
        self.assertEqual(local_manager.lookup(_key(1), _ctx()),
                         LookupResult.MISS)
        self.assertEqual(self.manager.lookup(_key(1), _ctx()),
                         LookupResult.HIT)

    def test_on_schedule_end_drops_unconsumed_remote_slices(self):
        # A remote HIT whose request never reaches prepare_load (aborted, or
        # declined by the scheduler) must not leave its peer slices stashed
        # past the step: they would accumulate forever and a stale slice
        # could later be handed to read_remote.
        from vllm.v1.kv_offload.base import ScheduleEndContext
        key = _key(1)
        seed_global_entries(self.store, key)
        self.assertEqual(self.manager.lookup(key, _ctx()), LookupResult.HIT)
        self.assertTrue(self.manager._remote_slices)

        self.manager.on_schedule_end(
            ScheduleEndContext(new_req_ids=(), preempted_req_ids=()))
        self.assertEqual(self.manager._remote_slices, {})

        # A retried request re-runs lookup and repopulates the stash.
        self.assertEqual(self.manager.lookup(key, _ctx()), LookupResult.HIT)
        self.assertTrue(self.manager._remote_slices)

    def test_registry_down_degrades_to_local_only(self):
        seed_global_entries(self.store, _key(1))
        seed_host_entries(self.store, _key(2))
        self.store.registry_down = True
        self.assertEqual(self.manager.lookup(_key(1), _ctx()),
                         LookupResult.MISS)
        self.assertEqual(self.manager.lookup(_key(2), _ctx()),
                         LookupResult.HIT)

    def test_receiver_flow_end_to_end(self):
        key = _key(1)
        seed_global_entries(self.store, key)
        self.assertEqual(self.manager.lookup(key, _ctx()), LookupResult.HIT)

        spec = self.manager.prepare_load([key], _ctx())
        self.assertTrue(spec.pinned)
        # A peer read borrows nothing here: no admission, no local entry, no
        # pin. The peer slice is simply carried to the submit.
        self.assertEqual(self.store.calls_named("insert"), [])
        for h in sub_hashes(key):
            self.assertNotIn(h, self.store.entries)

        self.manager.submit_load_job(7, [key], [5], "req")
        # Pure remote prefix dispatches read_remote directly without local load.
        self.assertEqual(self.store.calls_named("load"), [])
        (rr, ) = self.store.calls_named("read_remote")
        self.assertEqual(rr.args[0], sub_hashes(key))
        self.assertEqual(rr.args[2], [20, 21, 22, 23])
        self.assertTrue(
            all(sl.status == FakeBlockStatus.REMOTE for sl in rr.args[1]))
        self.assertTrue(self.manager.has_pending_work())
        self.assertEqual(self.manager.poll_finished_jobs(), [])

        # Concurrent lookup for in-flight fetch returns HIT_PENDING without duplicate read.
        self.assertEqual(self.manager.lookup(key, _ctx()),
                         LookupResult.HIT_PENDING)

        self.store.complete_remote_read(sub_hashes(key))
        (finished_job, ) = self.manager.poll_finished_jobs()
        self.assertTrue(finished_job.success)
        self.assertEqual(finished_job.req_id, "req")
        # The bytes landed in the destination device blocks and nowhere else:
        # a peer read leaves no local entry and nothing pinned, so the key is
        # still found only through the registry.
        for h in sub_hashes(key):
            self.assertNotIn(h, self.store.entries)
        self.assertEqual(self.store.pinned_hashes(), {})
        self.assertEqual(self.manager.lookup(key, _ctx()), LookupResult.HIT)
        self.assertFalse(self.manager.has_pending_work())

    def test_mixed_local_and_remote_job(self):
        local_key, remote_key = _key(1), _key(2)
        seed_host_entries(self.store, local_key)
        seed_global_entries(self.store, remote_key)
        self.assertEqual(self.manager.lookup(local_key, _ctx()),
                         LookupResult.HIT)
        self.assertEqual(self.manager.lookup(remote_key, _ctx()),
                         LookupResult.HIT)
        spec = self.manager.prepare_load([local_key, remote_key], _ctx())
        self.assertTrue(spec.pinned)

        self.manager.submit_load_job(7, [local_key, remote_key], [5, 6], "req")
        (ld, ) = self.store.calls_named("load")
        self.assertEqual(ld.args, (sub_hashes(local_key), [20, 21, 22, 23]))
        (rr, ) = self.store.calls_named("read_remote")
        self.assertEqual(rr.args[0], sub_hashes(remote_key))
        self.assertEqual(rr.args[2], [24, 25, 26, 27])

        # Mixed local and remote fetches wait until all sub-hashes resolve.
        self.store.complete_load(sub_hashes(local_key))
        self.assertEqual(self.manager.poll_finished_jobs(), [])
        self.store.complete_remote_read(sub_hashes(remote_key))
        (finished_job, ) = self.manager.poll_finished_jobs()
        self.assertTrue(finished_job.success)

    def test_remote_failure_leaves_nothing_behind(self):
        key = _key(1)
        seed_global_entries(self.store, key)
        self.assertEqual(self.manager.lookup(key, _ctx()), LookupResult.HIT)
        self.manager.prepare_load([key], _ctx())
        self.manager.submit_load_job(7, [key], [5], "req")

        self.store.complete_remote_read(sub_hashes(key), success=False)
        (finished_job, ) = self.manager.poll_finished_jobs()
        self.assertFalse(finished_job.success)
        self.assertEqual(finished_job.failed_device_block_ids, [5])
        # A peer read records nothing locally either way, and the registry
        # still names the owner, so the next lookup finds it again.
        for h in sub_hashes(key):
            self.assertNotIn(h, self.store.entries)
            self.assertIn(h, self.store.global_entries)

    def test_vanished_sub_block_fails_job_cleanly(self):
        # A sub-hash that is neither resident nor a peer's — evicted since the
        # lookup that reported the hit — fails the whole job before any
        # transfer, and the pins taken for the rest are handed back.
        local_key, remote_key = _key(1), _key(2)
        seed_host_entries(self.store, local_key)
        seed_global_entries(self.store, remote_key)
        self.manager.lookup(local_key, _ctx())
        self.manager.lookup(remote_key, _ctx())
        for h in sub_hashes(remote_key):
            self.manager._remote_slices.pop(h)
        spec = self.manager.prepare_load([local_key, remote_key], _ctx())
        self.assertFalse(spec.pinned)
        self.assertEqual(self.store.pinned_hashes(), {})
        self.manager.submit_load_job(7, [local_key, remote_key], [5, 6],
                                     "req",
                                     pinned=spec.pinned)
        (finished_job, ) = self.manager.poll_finished_jobs()
        self.assertFalse(finished_job.success)
        self.assertEqual(finished_job.failed_device_block_ids, [5, 6])
        self.assertEqual(self.store.calls_named("read_remote"), [])

    def test_read_remote_launch_failure_after_local_load(self):
        local_key, remote_key = _key(1), _key(2)
        seed_host_entries(self.store, local_key)
        seed_global_entries(self.store, remote_key)
        self.manager.lookup(local_key, _ctx())
        self.manager.lookup(remote_key, _ctx())
        self.manager.prepare_load([local_key, remote_key], _ctx())
        self.store.fail_read_remote = True
        self.manager.submit_load_job(7, [local_key, remote_key], [5, 6], "req")
        # Local load in-flight delays failure cleanup until transfer resolves.
        self.assertTrue(self.manager.has_pending_work())
        self.assertEqual(self.manager.poll_finished_jobs(), [])
        self.store.complete_load(sub_hashes(local_key))
        (finished_job, ) = self.manager.poll_finished_jobs()
        self.assertFalse(finished_job.success)
        self.assertEqual(finished_job.failed_device_block_ids, [5, 6])
        # The peer read left nothing local to clean up, and the HOST entries
        # the local load spent its pins on are untouched.
        for h in sub_hashes(remote_key):
            self.assertNotIn(h, self.store.entries)
        for h in sub_hashes(local_key):
            self.assertEqual(self.store.entries[h].status,
                             FakeBlockStatus.HOST)
            self.assertEqual(self.store.entries[h].pin_count, 0)
        self.assertEqual(self.manager.lookup(local_key, _ctx()),
                         LookupResult.HIT)

    def test_read_remote_launch_failure_cleans_up(self):
        key = _key(1)
        seed_global_entries(self.store, key)
        self.assertEqual(self.manager.lookup(key, _ctx()), LookupResult.HIT)
        self.manager.prepare_load([key], _ctx())
        self.store.fail_read_remote = True
        self.manager.submit_load_job(7, [key], [5], "req")
        # Immediate terminal failure; a peer read left nothing local behind.
        (finished_job, ) = self.manager.poll_finished_jobs()
        self.assertFalse(finished_job.success)
        self.assertEqual(finished_job.failed_device_block_ids, [5])
        for h in sub_hashes(key):
            self.assertNotIn(h, self.store.entries)
            self.assertIn(h, self.store.global_entries)

    def test_stale_stash_degenerates_to_local_load(self):
        # Keys promoted to local before prepare_load are routed to load() instead of read_remote().
        key = _key(1)
        seed_global_entries(self.store, key)
        self.assertEqual(self.manager.lookup(key, _ctx()), LookupResult.HIT)
        # Concurrent fetch promoted key to HOST status.
        for h in sub_hashes(key):
            self.store.global_entries.pop(h)
            self.store.entries[h] = _Entry(status=FakeBlockStatus.HOST)
        spec = self.manager.prepare_load([key], _ctx())
        self.assertTrue(spec.pinned)
        self.manager.submit_load_job(7, [key], [5], "req")
        (ld, ) = self.store.calls_named("load")
        self.assertEqual(ld.args, (sub_hashes(key), [20, 21, 22, 23]))
        self.assertEqual(self.store.calls_named("read_remote"), [])


class TestRemoteLanding(unittest.TestCase):
    """A peer read reserves no local capacity and admits nothing, so it can
    neither displace a local entry nor emit an offloading event."""

    def setUp(self):
        self.store = FakeKVCacheStore()
        self.manager = make_global_manager(self.store)
        seed_global_entries(self.store, _key(2))
        self.assertEqual(self.manager.lookup(_key(2), _ctx()),
                         LookupResult.HIT)

    def test_peer_read_admits_nothing_locally(self):
        seed_host_entries(self.store, _key(3))
        resident_before = dict(self.store.entries)
        spec = self.manager.prepare_load([_key(2)], _ctx("r1"))
        self.assertTrue(spec.pinned)
        self.assertEqual(self.store.calls_named("insert"), [])
        self.manager.submit_load_job(7, [_key(2)], [5], "r1")
        self.assertEqual(list(self.manager.take_events()), [])
        self.store.complete_remote_read(sub_hashes(_key(2)))
        (finished_job, ) = self.manager.poll_finished_jobs()
        self.assertTrue(finished_job.success)
        # Nothing admitted, nothing displaced, nothing pinned.
        self.assertEqual(set(self.store.entries), set(resident_before))
        self.assertEqual(self.store.pinned_hashes(), {})
        self.assertEqual(list(self.manager.take_events()), [])


class TestNamespace(unittest.TestCase):
    """Compatibility-namespaced keys flow through every store surface."""

    NS = b"NS16bytes--------"[:16]

    def setUp(self):
        self.store = FakeKVCacheStore()
        self.manager = make_global_manager(self.store, key_namespace=self.NS)

    def test_namespace_prefixes_every_sub_hash(self):
        key = _key(1)
        self.manager.prepare_store([key], _ctx())
        self.manager.submit_store_job(1, [key], [3])
        (ial, ) = self.store.calls_named("insert")
        self.assertEqual(ial.args[0], sub_hashes(key, ns=self.NS))
        (sv, ) = self.store.calls_named("save")
        self.assertEqual(sv.args[0], sub_hashes(key, ns=self.NS))

    def test_namespaced_remote_flow(self):
        key = _key(1)
        seed_global_entries(self.store, key, ns=self.NS)
        self.assertEqual(self.manager.lookup(key, _ctx()), LookupResult.HIT)
        self.manager.prepare_load([key], _ctx())
        self.manager.submit_load_job(7, [key], [5], "req")
        (rr, ) = self.store.calls_named("read_remote")
        self.assertEqual(rr.args[0], sub_hashes(key, ns=self.NS))
        self.assertEqual(rr.args[2], [20, 21, 22, 23])

    def test_derive_offload_namespace_config_sensitivity(self):
        from vllm_torchtpu.offload.raiden_store import derive_offload_namespace

        def cfg(model="m", revision="r", quant=None, algo="sha256"):
            c = MagicMock()
            c.model_config.model = model
            c.model_config.revision = revision
            c.model_config.quantization = quant
            c.cache_config.prefix_caching_hash_algo = algo
            return c

        kwargs = dict(kernel_block_size=256,
                      per_block_shape=(256, 8, 2, 128),
                      kv_dtype="bfloat16",
                      device_block_size=1280,
                      world_size=4,
                      num_kv_cache_groups=1,
                      num_kv_cache_tensors=64)
        ns = derive_offload_namespace(cfg(), **kwargs)
        self.assertEqual(len(ns), 16)
        self.assertEqual(ns, derive_offload_namespace(cfg(), **kwargs))
        # Compatibility namespace prevents cache collisions across incompatible engine configurations.
        self.assertNotEqual(ns,
                            derive_offload_namespace(cfg(model="o"), **kwargs))
        self.assertNotEqual(
            ns, derive_offload_namespace(cfg(quant="fp8"), **kwargs))
        self.assertNotEqual(
            ns, derive_offload_namespace(cfg(algo="sha256_cbor"), **kwargs))
        self.assertNotEqual(
            ns, derive_offload_namespace(cfg(), **{
                **kwargs, "world_size": 8
            }))
        # Context parallelism geometry is included in namespace only when CP is active.
        self.assertEqual(
            ns, derive_offload_namespace(cfg(), **kwargs, cp_geometry=()))
        pcp_ns = derive_offload_namespace(cfg(),
                                          **kwargs,
                                          cp_geometry=(8, 1, 1))
        self.assertNotEqual(ns, pcp_ns)
        self.assertNotEqual(
            pcp_ns,
            derive_offload_namespace(cfg(), **kwargs, cp_geometry=(8, 1, 16)))


class TestHybridGeometry(unittest.TestCase):
    """resolve_kernel_geometry on hybrid unified-pool group structures."""

    def _resolve(self, groups, unified=True):
        from vllm_torchtpu.offload import raiden_store as rs
        kv_cache_config = SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=s) for s in groups])
        backend = MagicMock()
        backend.get_kv_cache_shape.return_value = (1, 256, 8, 2, 128)
        backend.is_ssm.return_value = False
        with patch("vllm_torchtpu.platforms.tpu_platform.TpuPlatform."
                   "_find_non_ssm_backend", return_value=backend), \
             patch("vllm.v1.worker.utils.select_common_block_size",
                   return_value=256), \
             patch("vllm_torchtpu.platforms.tpu_block_size_utils."
                   "unified_kv_layout_enabled", return_value=unified):
            return rs.resolve_kernel_geometry(MagicMock(), kv_cache_config)

    def _attn_spec(self, page=1310720):
        from vllm.v1.kv_cache_interface import FullAttentionSpec
        spec = MagicMock(spec=FullAttentionSpec)
        spec.block_size = 1280
        spec.num_kv_heads = 8
        spec.head_size = 128
        spec.dtype = "bfloat16"
        spec.page_size_bytes = page
        return spec

    def _mamba_spec(self, page=1310720):
        from vllm.v1.kv_cache_interface import MambaSpec
        spec = MagicMock(spec=MambaSpec)
        spec.page_size_bytes = page
        return spec

    def test_hybrid_uses_attention_geometry_with_mamba_group_first(self):
        # In hybrid models, group 0 may be Mamba; verify attention spec resolution.
        kernel, shape, dtype, device_block = self._resolve(
            [self._mamba_spec(), self._attn_spec()])
        self.assertEqual((kernel, device_block), (256, 1280))
        self.assertEqual(shape, (256, 8, 2, 128))
        self.assertEqual(dtype, "bfloat16")

    def test_hybrid_requires_unified_pool(self):
        with self.assertRaises(AssertionError):
            self._resolve(
                [self._mamba_spec(), self._attn_spec()], unified=False)

    def test_hybrid_rejects_mismatched_page_sizes(self):
        with self.assertRaises(AssertionError):
            self._resolve([self._mamba_spec(page=999), self._attn_spec()])

    def test_dense_rejects_non_full_attention(self):
        with self.assertRaises(AssertionError):
            self._resolve([self._mamba_spec()])

    def test_scheduler_process_falls_back_to_attn_selector(self):
        # In EngineCore process, resolve attention backend via selector fallback.
        from contextlib import nullcontext

        from vllm_torchtpu.offload import raiden_store as rs
        kv_cache_config = SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=self._attn_spec())])
        backend = MagicMock()
        backend.get_kv_cache_shape.return_value = (1, 256, 8, 2, 128)
        backend.is_ssm.return_value = False
        with patch("vllm_torchtpu.platforms.tpu_platform.TpuPlatform."
                   "_find_non_ssm_backend", return_value=None), \
             patch("vllm.v1.attention.selector.get_attn_backend",
                   return_value=backend) as selector, \
             patch("vllm.config.set_current_vllm_config",
                   return_value=nullcontext()), \
             patch("vllm.v1.worker.utils.select_common_block_size",
                   return_value=256):
            kernel, shape, dtype, device_block = rs.resolve_kernel_geometry(
                MagicMock(), kv_cache_config)
        self.assertEqual((kernel, device_block), (256, 1280))
        self.assertEqual(shape, (256, 8, 2, 128))
        selector.assert_called_once()


class TestRaidenStoreWorkerStub(unittest.TestCase):

    def test_submit_paths_fail_loudly(self):
        from vllm_torchtpu.offload.raiden_store import \
            RaidenStoreOffloadingWorker
        worker = object.__new__(RaidenStoreOffloadingWorker)
        with self.assertRaises(AssertionError):
            worker.submit_store(1, MagicMock(), MagicMock())
        with self.assertRaises(AssertionError):
            worker.submit_load(1, MagicMock(), MagicMock())
        self.assertEqual(worker.get_finished(), [])
        self.assertIsNone(worker.wait({1, 2}))


if __name__ == "__main__":
    unittest.main()
