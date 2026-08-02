# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the store-backed (V2) raiden KV offload stack.

Covers the scheduler-process pieces against a fake KVCacheStore, so no TPU,
tpu_raiden wheel, or torch_tpu runtime is required:

- Registration gate: stores/loads stay disabled until the accessor reports
  world_size registered workers; more than world_size raises.
- Sub-hash-per-kernel-block mapping: one OffloadKey expands to
  gpu_block_size_factor store entries `key ‖ sub_idx`, device kernel-block
  IDs expand contiguously.
- Atomic admission + exact-batch cleanup: insert_and_lock/save on submit,
  release on success, release_and_delete on failure, immediate terminal
  failure on rejected admission or failed launch (no cleanup owed).
- Load lifecycle: pin in prepare_load, release only from the finalizer
  (complete_load is a no-op), failed loads report their destination GPU
  blocks.
- touch via pin+release, lookup prefix semantics over sub-hashes,
  reset_cache failing loudly, drain_jobs blocking until terminal.
- Scheduler subclass: store jobs are parked until every rank's fence ack,
  jobs_to_flush cancels fence-pending jobs and drains launched ones, and the
  shipped metadata carries empty job dicts plus fence/echo fields.
- Worker-side connector: fence sync + ack, completion echoes to
  completed_jobs/finished_recving, failed-load block-ID reporting, and
  worker-metadata aggregation.

Stage-2 hardening and hybrid coverage:
- Store batch cap, save retry-once (suppressed during drains), and the
  poison-key quarantine on terminal transfer failures.
- Submit-time sub-hash classification: mixed batches (recovered entries or
  partial-failure leftovers) save only the newly admitted HBM sub-hashes —
  raiden's save() rejects any batch containing non-HBM hashes.
- prepare_load pin races fail the job cleanly instead of raising.
- Recovered-directory learning (lookup HIT teaches the advisory state),
  RAIDEN_SHM_MODEL_UID derivation, and hybrid unified-pool geometry
  resolution (attention-spec filtering, page uniformity).
"""
import enum
import sys
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


class FakeRaidenBlockID:

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
                                   RaidenBlockID=FakeRaidenBlockID)


@dataclass
class _Entry:
    status: FakeBlockStatus
    device_block_id: int = -1
    pin_count: int = 0


@dataclass
class _Call:
    name: str
    args: tuple


class FakeKVCacheStore:
    """In-memory stand-in for tpu_raiden's KVCacheStore python wrapper."""

    def __init__(self, registered_workers: int = 0):
        self.registered_workers = registered_workers
        self.entries: dict[bytes, _Entry] = {}
        self.calls: list[_Call] = []
        self.fail_insert_and_lock = False
        self.fail_save = False
        self.fail_load = False
        self.raiden_id = object()
        # Terminal outcomes queued by the test, drained by the polls.
        self._save_done: list[bytes] = []
        self._save_failed: list[bytes] = []
        self._load_done: list[bytes] = []
        self._load_failed: list[bytes] = []
        self._saving: list[bytes] = []
        self._loading: list[bytes] = []

    # -- store surface used by the manager -----------------------------

    def num_registered_workers(self) -> int:
        return self.registered_workers

    def lookup(self, block_hashes, enable_global=False):
        out = []
        for h in block_hashes:
            entry = self.entries.get(h)
            if entry is None:
                break
            out.append((h, SimpleNamespace(status=entry.status)))
        return out

    def pin(self, block_hashes) -> bool:
        self.calls.append(_Call("pin", (list(block_hashes), )))
        if any(h not in self.entries for h in block_hashes):
            return False
        for h in block_hashes:
            self.entries[h].pin_count += 1
        return True

    def release(self, block_hashes) -> None:
        self.calls.append(_Call("release", (list(block_hashes), )))
        for h in block_hashes:
            entry = self.entries.get(h)
            if entry is not None and entry.pin_count > 0:
                entry.pin_count -= 1

    def insert_and_lock(self, block_hashes, slices, on_host) -> bool:
        self.calls.append(
            _Call("insert_and_lock",
                  (list(block_hashes), list(slices), on_host)))
        if self.fail_insert_and_lock:
            return False
        for h, s in zip(block_hashes, slices):
            entry = self.entries.get(h)
            if entry is None:
                self.entries[h] = _Entry(status=s.status,
                                         device_block_id=s.device_block_id,
                                         pin_count=1)
            else:
                entry.pin_count += 1
        return True

    def release_and_delete(self, block_hashes) -> int:
        self.calls.append(_Call("release_and_delete", (list(block_hashes), )))
        deleted = 0
        for h in block_hashes:
            entry = self.entries.get(h)
            if entry is None:
                continue
            if entry.pin_count > 0:
                entry.pin_count -= 1
            if entry.pin_count == 0 and entry.status not in (
                    FakeBlockStatus.HOST, FakeBlockStatus.HOST_AND_HBM):
                del self.entries[h]
                deleted += 1
        return deleted

    def save(self, block_hashes) -> bool:
        self.calls.append(_Call("save", (list(block_hashes), )))
        if self.fail_save:
            return False
        # Mirror raiden's synchronous preconditions (kv_cache_store.cc
        # Save): every hash must exist, be HBM status, be pinned, and not
        # already be saving — else the WHOLE launch fails. This is what
        # forces the manager to classify mixed batches before saving.
        for h in block_hashes:
            entry = self.entries.get(h)
            if (entry is None or entry.status != FakeBlockStatus.HBM
                    or entry.pin_count <= 0 or h in self._saving):
                return False
        self._saving.extend(block_hashes)
        return True

    def load(self, block_hashes, device_block_ids) -> bool:
        self.calls.append(
            _Call("load", (list(block_hashes), list(device_block_ids))))
        if self.fail_load:
            return False
        self._loading.extend(block_hashes)
        return True

    def poll_save_status(self):
        done, self._save_done = self._save_done, []
        failed, self._save_failed = self._save_failed, []
        return done, failed, list(self._saving)

    def poll_load_status(self):
        done, self._load_done = self._load_done, []
        failed, self._load_failed = self._load_failed, []
        return done, failed, list(self._loading)

    # -- test controls --------------------------------------------------

    def complete_save(self, block_hashes, success=True) -> None:
        for h in block_hashes:
            self._saving.remove(h)
            if success:
                self.entries[h].status = FakeBlockStatus.HOST_AND_HBM
                self._save_done.append(h)
            else:
                self._save_failed.append(h)

    def complete_load(self, block_hashes, success=True) -> None:
        for h in block_hashes:
            self._loading.remove(h)
            (self._load_done if success else self._load_failed).append(h)

    def calls_named(self, name: str) -> list[_Call]:
        return [c for c in self.calls if c.name == name]


def make_manager(store, *, f=4, num_blocks=8, world_size=2):
    from vllm_torchtpu.offload.raiden_store import RaidenOffloadingManager
    return RaidenOffloadingManager(
        capacity_kernel_blocks=num_blocks * f,
        num_offloaded_blocks=num_blocks,
        sub_blocks_per_key=f,
        world_size=world_size,
        raiden_controller_port=0,
        raiden_job_name="test",
        store=store,
        store_types=FAKE_STORE_TYPES,
    )


def sub_hashes(key: bytes, f: int = 4) -> list[bytes]:
    return [key + i.to_bytes(4, "big") for i in range(f)]


def seed_host_entries(store, key: bytes, f: int = 4) -> None:
    for h in sub_hashes(key, f):
        store.entries[h] = _Entry(status=FakeBlockStatus.HOST)


class TestRegistrationGate(unittest.TestCase):

    def test_gate_disables_lookup_and_store_until_all_ranks(self):
        store = FakeKVCacheStore(registered_workers=1)
        manager = make_manager(store, world_size=2)
        seed_host_entries(store, _key(0))

        # k < n: everything disabled, loudly no-op.
        self.assertEqual(manager.lookup(_key(0), _ctx()), LookupResult.MISS)
        self.assertIsNone(manager.prepare_store([_key(1)], _ctx()))

        # gate opens at exactly world_size.
        store.registered_workers = 2
        self.assertEqual(manager.lookup(_key(0), _ctx()), LookupResult.HIT)
        out = manager.prepare_store([_key(1)], _ctx())
        self.assertIsNotNone(out)
        self.assertEqual(out.keys_to_store, [_key(1)])

        # once open, stays open without re-querying.
        store.registered_workers = 0
        self.assertEqual(manager.lookup(_key(0), _ctx()), LookupResult.HIT)

    def test_more_workers_than_world_size_raises(self):
        store = FakeKVCacheStore(registered_workers=3)
        manager = make_manager(store, world_size=2)
        with self.assertRaises(RuntimeError):
            manager.lookup(_key(0), _ctx())


class TestLookup(unittest.TestCase):

    def setUp(self):
        self.store = FakeKVCacheStore(registered_workers=2)
        self.manager = make_manager(self.store)

    def test_hit_requires_every_sub_hash(self):
        seed_host_entries(self.store, _key(0))
        self.assertEqual(self.manager.lookup(_key(0), _ctx()),
                         LookupResult.HIT)
        # Remove one sub-block: partial entries are a miss.
        del self.store.entries[sub_hashes(_key(0))[2]]
        self.assertEqual(self.manager.lookup(_key(0), _ctx()),
                         LookupResult.MISS)

    def test_hbm_status_is_hit_pending(self):
        for h in sub_hashes(_key(0)):
            self.store.entries[h] = _Entry(status=FakeBlockStatus.HBM)
        self.assertEqual(self.manager.lookup(_key(0), _ctx()),
                         LookupResult.HIT_PENDING)

    def test_remote_status_is_miss_in_phase1(self):
        for h in sub_hashes(_key(0)):
            self.store.entries[h] = _Entry(status=FakeBlockStatus.REMOTE)
        self.assertEqual(self.manager.lookup(_key(0), _ctx()),
                         LookupResult.MISS)


class TestStoreJobLifecycle(unittest.TestCase):

    def setUp(self):
        self.store = FakeKVCacheStore(registered_workers=2)
        self.manager = make_manager(self.store)

    def _admit(self, keys, gpu_blocks):
        out = self.manager.prepare_store(keys, _ctx())
        self.assertEqual(out.keys_to_store, keys)
        self.manager.submit_store_job(7, keys, gpu_blocks)

    def test_submit_expands_sub_hashes_and_device_blocks(self):
        self._admit([_key(0), _key(1)], [5, 9])

        (call, ) = self.store.calls_named("insert_and_lock")
        hashes, slices, on_host = call.args
        self.assertEqual(hashes, sub_hashes(_key(0)) + sub_hashes(_key(1)))
        self.assertFalse(on_host)
        self.assertEqual([s.device_block_id for s in slices],
                         [20, 21, 22, 23, 36, 37, 38, 39])
        self.assertTrue(all(s.status == FakeBlockStatus.HBM for s in slices))
        (save_call, ) = self.store.calls_named("save")
        self.assertEqual(save_call.args[0], hashes)

    def test_success_releases_exact_batch_and_marks_stored(self):
        self._admit([_key(0)], [5])
        self.assertTrue(self.manager.has_pending_work())
        self.assertEqual(self.manager.poll_finished_jobs(), [])

        self.store.complete_save(sub_hashes(_key(0)))
        (fj, ) = self.manager.poll_finished_jobs()
        self.assertTrue(fj.success)
        self.assertEqual(fj.job_id, 7)
        (release, ) = self.store.calls_named("release")
        self.assertEqual(release.args[0], sub_hashes(_key(0)))
        self.assertFalse(self.manager.has_pending_work())
        # Now a HIT, and not re-admitted by prepare_store.
        self.assertEqual(self.manager.lookup(_key(0), _ctx()),
                         LookupResult.HIT)
        out = self.manager.prepare_store([_key(0), _key(1)], _ctx())
        self.assertEqual(out.keys_to_store, [_key(1)])

    def test_partial_batch_failure_retries_then_cleans_exact_batch(self):
        self._admit([_key(0), _key(1)], [5, 9])
        batch = sub_hashes(_key(0)) + sub_hashes(_key(1))
        # One sub-block fails, the rest succeed: the failed subset is
        # retried once (committed sub-hashes are NOT re-saved — they are
        # HOST_AND_HBM now and would fail save()'s HBM precondition).
        self.store.complete_save(batch[:1], success=False)
        self.assertEqual(self.manager.poll_finished_jobs(), [])
        self.store.complete_save(batch[1:], success=True)
        self.assertEqual(self.manager.poll_finished_jobs(), [])
        retry = self.store.calls_named("save")[-1]
        self.assertEqual(retry.args[0], batch[:1])
        # The retry fails too: the job is terminal, cleanup uses the exact
        # admitted batch exactly once, and the keys are quarantined.
        self.store.complete_save(batch[:1], success=False)
        (fj, ) = self.manager.poll_finished_jobs()
        self.assertFalse(fj.success)
        (rad, ) = self.store.calls_named("release_and_delete")
        self.assertEqual(rad.args[0], batch)
        self.assertEqual(self.store.calls_named("release"), [])
        out = self.manager.prepare_store([_key(0), _key(1)], _ctx())
        self.assertEqual(out.keys_to_store, [])
        self.assertEqual(self.manager.lookup(_key(0), _ctx()),
                         LookupResult.MISS)

    def test_save_retry_succeeds_and_marks_stored(self):
        self._admit([_key(0)], [5])
        batch = sub_hashes(_key(0))
        self.store.complete_save(batch, success=False)
        self.assertEqual(self.manager.poll_finished_jobs(), [])
        retry = self.store.calls_named("save")[-1]
        self.assertEqual(retry.args[0], batch)
        self.store.complete_save(batch, success=True)
        (fj, ) = self.manager.poll_finished_jobs()
        self.assertTrue(fj.success)
        self.assertEqual(self.manager.lookup(_key(0), _ctx()),
                         LookupResult.HIT)

    def test_mixed_terminal_hashes_in_one_poll_finalize_once(self):
        self._admit([_key(0)], [5])
        batch = sub_hashes(_key(0))
        # Some sub-blocks fail and the rest succeed, all observed by the
        # same poll: one retry of the failed subset, then (failing again)
        # exactly one terminal job and one cleanup.
        self.store.complete_save(batch[:1], success=False)
        self.store.complete_save(batch[1:], success=True)
        self.assertEqual(self.manager.poll_finished_jobs(), [])
        self.store.complete_save(batch[:1], success=False)
        finished = self.manager.poll_finished_jobs()
        self.assertEqual(len(finished), 1)
        self.assertFalse(finished[0].success)
        self.assertEqual(len(self.store.calls_named("release_and_delete")), 1)
        self.assertFalse(self.manager.has_pending_work())

    def test_rejected_admission_is_immediate_terminal_failure(self):
        self.store.fail_insert_and_lock = True
        self._admit([_key(0)], [5])
        self.assertEqual(self.store.calls_named("save"), [])
        self.assertEqual(self.store.calls_named("release_and_delete"), [])
        (fj, ) = self.manager.poll_finished_jobs()
        self.assertFalse(fj.success)
        self.assertFalse(self.manager.has_pending_work())

    def test_failed_save_launch_reverts_admission(self):
        self.store.fail_save = True
        self._admit([_key(0)], [5])
        (rad, ) = self.store.calls_named("release_and_delete")
        self.assertEqual(rad.args[0], sub_hashes(_key(0)))
        (fj, ) = self.manager.poll_finished_jobs()
        self.assertFalse(fj.success)

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
        self.store = FakeKVCacheStore(registered_workers=2)
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

        # complete_load is a no-op: pins survive until the poll is terminal.
        self.manager.complete_load([_key(0), _key(1)], _ctx("r1"))
        self.assertEqual(self.store.entries[batch[0]].pin_count, 1)

        self.store.complete_load(batch)
        (fj, ) = self.manager.poll_finished_jobs()
        self.assertTrue(fj.success)
        self.assertEqual(fj.req_id, "r1")
        self.assertEqual(fj.failed_gpu_block_ids, [])
        for h in batch:
            self.assertEqual(self.store.entries[h].pin_count, 0)

    def test_load_failure_reports_dst_gpu_blocks_and_keeps_host(self):
        self.manager.prepare_load([_key(0)], _ctx("r1"))
        self.manager.submit_load_job(3, [_key(0)], [10], "r1")
        self.store.complete_load(sub_hashes(_key(0)), success=False)
        (fj, ) = self.manager.poll_finished_jobs()
        self.assertFalse(fj.success)
        self.assertEqual(fj.req_id, "r1")
        self.assertEqual(fj.failed_gpu_block_ids, [10])
        # HOST entries stay intact and unpinned.
        for h in sub_hashes(_key(0)):
            self.assertEqual(self.store.entries[h].status,
                             FakeBlockStatus.HOST)
            self.assertEqual(self.store.entries[h].pin_count, 0)

    def test_failed_load_launch_is_immediate_terminal(self):
        self.store.fail_load = True
        self.manager.prepare_load([_key(0)], _ctx("r1"))
        self.manager.submit_load_job(3, [_key(0)], [10], "r1")
        (fj, ) = self.manager.poll_finished_jobs()
        self.assertFalse(fj.success)
        self.assertEqual(fj.failed_gpu_block_ids, [10])
        for h in sub_hashes(_key(0)):
            self.assertEqual(self.store.entries[h].pin_count, 0)

    def test_prepare_load_pin_failure_fails_job_cleanly(self):
        # Losing the lookup->pin race against eviction must not crash the
        # engine: the spec comes back unpinned and the submitted job fails
        # immediately, reporting its destination blocks for recompute.
        spec = self.manager.prepare_load([_key(9)], _ctx("r1"))
        self.assertFalse(spec.pinned)
        self.manager.submit_load_job(3, [_key(9)], [10],
                                     "r1",
                                     pinned=spec.pinned)
        self.assertEqual(self.store.calls_named("load"), [])
        self.assertEqual(self.store.calls_named("release"), [])
        (fj, ) = self.manager.poll_finished_jobs()
        self.assertFalse(fj.success)
        self.assertEqual(fj.failed_gpu_block_ids, [10])


class TestTouchAndMisc(unittest.TestCase):

    def setUp(self):
        self.store = FakeKVCacheStore(registered_workers=2)
        self.manager = make_manager(self.store)

    def test_touch_pins_and_releases_stored_keys_only(self):
        seed_host_entries(self.store, _key(0))
        # Not tracked as stored: advisory filter skips the probe.
        self.manager.touch([_key(0), _key(1)], _ctx())
        self.assertEqual(self.store.calls_named("pin"), [])

        # Store key 2 through a full job so the advisory set learns it.
        out = self.manager.prepare_store([_key(2)], _ctx())
        self.manager.submit_store_job(1, out.keys_to_store, [5])
        self.store.complete_save(sub_hashes(_key(2)))
        self.manager.poll_finished_jobs()
        self.store.calls.clear()

        self.manager.touch([_key(2), _key(1)], _ctx())
        (pin, ) = self.store.calls_named("pin")
        self.assertEqual(pin.args[0], sub_hashes(_key(2)))
        (release, ) = self.store.calls_named("release")
        self.assertEqual(release.args[0], sub_hashes(_key(2)))

    def test_reset_cache_fails_loudly(self):
        with self.assertRaises(NotImplementedError):
            self.manager.reset_cache()

    def test_drain_jobs_blocks_until_terminal(self):
        out = self.manager.prepare_store([_key(0)], _ctx())
        self.manager.submit_store_job(4, out.keys_to_store, [5])

        # Completion arrives "concurrently": queue it before draining.
        self.store.complete_save(sub_hashes(_key(0)))
        drained = self.manager.drain_jobs({4})
        self.assertEqual([fj.job_id for fj in drained], [4])
        self.assertFalse(self.manager.is_job_launched(4))
        # Nothing double-reported afterwards.
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

    def _store_job(self, keys, gpu_block_ids, req_id="r1"):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import \
            TransferJob

        from vllm_torchtpu.offload.raiden_store import RaidenLoadStoreSpec
        return TransferJob(
            req_id=req_id,
            src_spec=GPULoadStoreSpec(gpu_block_ids,
                                      group_sizes=[len(gpu_block_ids)],
                                      block_indices=[0]),
            dst_spec=RaidenLoadStoreSpec(keys),
        )

    def _load_job(self, keys, gpu_block_ids, req_id="r1"):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import \
            TransferJob

        from vllm_torchtpu.offload.raiden_store import RaidenLoadStoreSpec
        return TransferJob(
            req_id=req_id,
            src_spec=RaidenLoadStoreSpec(keys),
            dst_spec=GPULoadStoreSpec(gpu_block_ids,
                                      group_sizes=[len(gpu_block_ids)],
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
        store = FakeKVCacheStore(registered_workers=2)
        manager = make_manager(store)
        manager.prepare_store([_key(0)], _ctx())
        sched = self._make_scheduler(manager)

        meta = self._build_meta(
            sched,
            self._stock_meta(store_jobs={7: self._store_job([_key(0)], [5])}))

        # Shipped metadata: empty jobs, one fence request, no echoes yet.
        self.assertEqual(meta.store_jobs, {})
        self.assertEqual(meta.load_jobs, {})
        self.assertEqual(meta.fence_job_ids, {7})
        self.assertEqual(meta.finished_store_job_ids, [])
        self.assertEqual(store.calls_named("insert_and_lock"), [])

        # First rank ack: still parked.
        self._ack(sched, [7])
        self.assertEqual(store.calls_named("insert_and_lock"), [])
        # Second rank ack: admitted and launched.
        self._ack(sched, [7])
        self.assertEqual(len(store.calls_named("insert_and_lock")), 1)
        self.assertEqual(len(store.calls_named("save")), 1)
        self.assertNotIn(7, sched._fence_pending)

    def test_load_jobs_submit_immediately(self):
        store = FakeKVCacheStore(registered_workers=2)
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

        # Completion is echoed on the next step's metadata.
        store.complete_load(sub_hashes(_key(0)))
        meta2 = self._build_meta(sched, self._stock_meta())
        self.assertEqual(meta2.finished_load_jobs, {3: "r1"})
        self.assertEqual(meta2.failed_load_gpu_block_ids, [])

    def test_flush_cancels_fence_pending_store(self):
        store = FakeKVCacheStore(registered_workers=2)
        manager = make_manager(store)
        manager.prepare_store([_key(0)], _ctx())
        sched = self._make_scheduler(manager)
        self._build_meta(
            sched,
            self._stock_meta(store_jobs={7: self._store_job([_key(0)], [5])}))

        # The job's source block is reused before any rank acked: the job is
        # cancelled, echoed as complete, and nothing ever reaches the store.
        meta = self._build_meta(sched, self._stock_meta(jobs_to_flush={7}))
        self.assertIn(7, meta.finished_store_job_ids)
        self.assertNotIn(7, sched._fence_pending)
        self.assertEqual(store.calls_named("insert_and_lock"), [])
        # The key is admittable again.
        out = manager.prepare_store([_key(0)], _ctx())
        self.assertEqual(out.keys_to_store, [_key(0)])

    def test_flush_drains_launched_store(self):
        store = FakeKVCacheStore(registered_workers=2)
        manager = make_manager(store)
        manager.prepare_store([_key(0)], _ctx())
        sched = self._make_scheduler(manager)
        self._build_meta(
            sched,
            self._stock_meta(store_jobs={7: self._store_job([_key(0)], [5])}))
        self._ack(sched, [7], count=2)
        self.assertTrue(manager.is_job_launched(7))

        # Queue the completion the drain will observe, then flush.
        store.complete_save(sub_hashes(_key(0)))
        meta = self._build_meta(sched, self._stock_meta(jobs_to_flush={7}))
        self.assertIn(7, meta.finished_store_job_ids)
        self.assertFalse(manager.is_job_launched(7))


class TestWorkerSideConnector(unittest.TestCase):

    def _make_worker_connector(self):
        from vllm_torchtpu.offload.raiden_connector import \
            TPURaidenOffloadingConnector
        conn = object.__new__(TPURaidenOffloadingConnector)
        conn.connector_worker = MagicMock()
        conn._fenced_jobs = {}
        conn._completed_jobs = {}
        conn._load_error_block_ids = set()
        return conn

    def _meta(self, **kwargs):
        from vllm_torchtpu.offload.raiden_connector import \
            RaidenOffloadingConnectorMetadata
        return RaidenOffloadingConnectorMetadata(load_jobs={},
                                                 store_jobs={},
                                                 **kwargs)

    def test_fence_syncs_device_and_acks(self):
        conn = self._make_worker_connector()
        conn._connector_metadata = self._meta(fence_job_ids={7, 8})

        sync_calls = []
        fake_sync_mod = SimpleNamespace(
            synchronize=lambda *a, **k: sync_calls.append((a, k)))
        with patch.dict(
                sys.modules, {
                    "torch_tpu": SimpleNamespace(_internal=None),
                    "torch_tpu._internal": SimpleNamespace(sync=fake_sync_mod),
                    "torch_tpu._internal.sync": fake_sync_mod,
                }):
            sending, recving = conn.get_finished(set())
        self.assertEqual(len(sync_calls), 1)
        self.assertEqual(sync_calls[0][1], {"wait": True})
        self.assertEqual(conn._fenced_jobs, {7: 1, 8: 1})
        self.assertEqual((sending, recving), (set(), set()))

    def test_echoes_completions_and_finished_recving(self):
        conn = self._make_worker_connector()
        conn._connector_metadata = self._meta(
            finished_store_job_ids=[1, 2],
            finished_load_jobs={3: "req-a"},
            failed_load_gpu_block_ids=[10, 11],
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
    """Stage-2 hardening: batch cap, drain-suppressed retries, and the
    poison-key quarantine."""

    def setUp(self):
        self.store = FakeKVCacheStore(registered_workers=2)
        self.manager = make_manager(self.store)

    def test_store_batch_cap_truncates_keys(self):
        import vllm_torchtpu.offload.raiden_store as rs
        keys = [_key(i) for i in range(5)]
        with patch.object(rs, "_MAX_STORE_KEYS_PER_JOB", 2):
            out = self.manager.prepare_store(keys, _ctx())
        # Dropped keys are not accepted, so the scheduler re-offers them
        # next step; only the accepted prefix is tracked in flight.
        self.assertEqual(out.keys_to_store, keys[:2])
        self.manager.on_store_job_cancelled(keys[:2])
        out = self.manager.prepare_store(keys, _ctx())
        self.assertEqual(out.keys_to_store, keys)

    def test_drain_suppresses_save_retry(self):
        out = self.manager.prepare_store([_key(0)], _ctx())
        self.manager.submit_store_job(4, out.keys_to_store, [5])
        # Queue a failure, then drain: the job must finalize as failed with
        # exactly one save() attempt — its source blocks are about to be
        # reused, so a retry would read garbage.
        self.store.complete_save(sub_hashes(_key(0)), success=False)
        drained = self.manager.drain_jobs({4})
        self.assertEqual([(fj.job_id, fj.success) for fj in drained],
                         [(4, False)])
        self.assertEqual(len(self.store.calls_named("save")), 1)
        self.assertEqual(len(self.store.calls_named("release_and_delete")), 1)

    def test_failed_load_quarantines_keys(self):
        seed_host_entries(self.store, _key(0))
        self.assertEqual(self.manager.lookup(_key(0), _ctx()),
                         LookupResult.HIT)
        self.manager.prepare_load([_key(0)], _ctx("r1"))
        self.manager.submit_load_job(3, [_key(0)], [10], "r1")
        self.store.complete_load(sub_hashes(_key(0)), success=False)
        (fj, ) = self.manager.poll_finished_jobs()
        self.assertFalse(fj.success)
        # The HOST entries survive, but the key is quarantined: no repeat
        # load/recompute loop, no re-admission.
        self.assertEqual(self.manager.lookup(_key(0), _ctx()),
                         LookupResult.MISS)
        out = self.manager.prepare_store([_key(0)], _ctx())
        self.assertEqual(out.keys_to_store, [])


class TestMixedBatchesAndRecovery(unittest.TestCase):
    """Sub-hash classification at submit time: pre-existing HOST entries
    (recovered across a restart, or committed by a partially failed earlier
    save) are locked but never re-saved — raiden's save() rejects the whole
    launch if any hash is not HBM status."""

    def setUp(self):
        self.store = FakeKVCacheStore(registered_workers=2)
        self.manager = make_manager(self.store)

    def test_mixed_batch_saves_only_new_sub_hashes(self):
        batch = sub_hashes(_key(0))
        # Two of four sub-blocks already committed (partial crash window).
        for h in batch[:2]:
            self.store.entries[h] = _Entry(status=FakeBlockStatus.HOST)
        out = self.manager.prepare_store([_key(0)], _ctx())
        self.assertEqual(out.keys_to_store, [_key(0)])
        self.manager.submit_store_job(7, [_key(0)], [5])
        (save, ) = self.store.calls_named("save")
        self.assertEqual(save.args[0], batch[2:])
        self.store.complete_save(batch[2:])
        (fj, ) = self.manager.poll_finished_jobs()
        self.assertTrue(fj.success)
        # Cleanup released the exact admitted batch: the pre-existing HOST
        # entries were pinned by insert_and_lock and must be unpinned too.
        (release, ) = self.store.calls_named("release")
        self.assertEqual(release.args[0], batch)
        for h in batch:
            self.assertEqual(self.store.entries[h].pin_count, 0)
        self.assertEqual(self.manager.lookup(_key(0), _ctx()),
                         LookupResult.HIT)

    def test_all_host_batch_finishes_without_save(self):
        seed_host_entries(self.store, _key(0))
        out = self.manager.prepare_store([_key(0)], _ctx())
        self.manager.submit_store_job(7, out.keys_to_store, [5])
        self.assertEqual(self.store.calls_named("save"), [])
        (release, ) = self.store.calls_named("release")
        self.assertEqual(release.args[0], sub_hashes(_key(0)))
        (fj, ) = self.manager.poll_finished_jobs()
        self.assertTrue(fj.success)
        self.assertFalse(self.manager.has_pending_work())
        # Learned as stored: not re-admitted.
        out = self.manager.prepare_store([_key(0)], _ctx())
        self.assertEqual(out.keys_to_store, [])

    def test_lookup_learns_recovered_keys_for_touch(self):
        # Entries recovered from shm predate this process's advisory state;
        # a HIT must teach the manager so touch reaches them.
        seed_host_entries(self.store, _key(0))
        self.manager.touch([_key(0)], _ctx())
        self.assertEqual(self.store.calls_named("pin"), [])
        self.assertEqual(self.manager.lookup(_key(0), _ctx()),
                         LookupResult.HIT)
        self.manager.touch([_key(0)], _ctx())
        (pin, ) = self.store.calls_named("pin")
        self.assertEqual(pin.args[0], sub_hashes(_key(0)))


class TestShmModelUid(unittest.TestCase):

    def _cfg(self, model="m", revision="r"):
        cfg = MagicMock()
        cfg.model_config.model = model
        cfg.model_config.revision = revision
        return cfg

    def test_uid_is_deterministic_and_config_sensitive(self):
        from vllm_torchtpu.offload.raiden_store import derive_shm_model_uid
        kwargs = dict(kernel_block_size=256,
                      per_block_shape=(256, 8, 2, 128),
                      kv_dtype="bfloat16",
                      gpu_block_size=1280,
                      world_size=4,
                      capacity_kernel_blocks=1000)
        uid = derive_shm_model_uid(self._cfg(), **kwargs)
        self.assertEqual(uid, derive_shm_model_uid(self._cfg(), **kwargs))
        self.assertNotEqual(
            uid, derive_shm_model_uid(self._cfg(model="other"), **kwargs))
        self.assertNotEqual(
            uid,
            derive_shm_model_uid(self._cfg(), **{
                **kwargs, "capacity_kernel_blocks": 999
            }))


class TestHybridGeometry(unittest.TestCase):
    """resolve_kernel_geometry on hybrid unified-pool group structures."""

    def _resolve(self, groups, unified=True):
        from vllm_torchtpu.offload import raiden_store as rs
        kv_cache_config = SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=s) for s in groups])
        backend = MagicMock()
        backend.get_kv_cache_shape.return_value = (1, 256, 8, 2, 128)
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
        # Group order follows model layer order; group 0 being Mamba must
        # not break the attention-spec selection.
        kernel, shape, dtype, gpu_block = self._resolve(
            [self._mamba_spec(), self._attn_spec()])
        self.assertEqual((kernel, gpu_block), (256, 1280))
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
