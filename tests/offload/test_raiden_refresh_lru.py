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
"""Raiden host LRU refresh (additional_config raiden_offload_refresh_lru) and
the manager's periodic host-pool stats line."""

import unittest
from collections import Counter, OrderedDict
from unittest.mock import patch

from vllm.v1.kv_offload.base import LookupResult

from tests.offload import test_raiden_store_offload as base
from tests.offload.test_raiden_store_offload import (
    FakeKVCacheStore,
    _ctx,
    _FakeFinishedRequest,
    _key,
    make_manager,
    sub_hashes,
)


class LRUStore(FakeKVCacheStore):
    """FakeKVCacheStore with Raiden's host LRU order: an insert enters at the
    Most Recently Used end, a release that drops the last pin moves the entry
    there, and an unpinned probe moves nothing."""

    def __init__(self):
        super().__init__()
        self.lru: OrderedDict[bytes, None] = OrderedDict()

    def insert(self, block_hashes, slices, on_host) -> bool:
        if not super().insert(block_hashes, slices, on_host):
            return False
        for h in block_hashes:
            self.lru[h] = None
            self.lru.move_to_end(h)
        return True

    def release(self, block_hashes) -> None:
        super().release(block_hashes)
        for h in block_hashes:
            entry = self.entries.get(h)
            if entry is not None and entry.pin_count == 0 and h in self.lru:
                self.lru.move_to_end(h)

    def evict(self, n: int) -> list[bytes]:
        """Evict the `n` least recently used unpinned entries."""
        evicted = []
        for h in list(self.lru):
            if len(evicted) == n:
                break
            if self.entries[h].pin_count:
                continue
            del self.lru[h]
            del self.entries[h]
            evicted.append(h)
        self.simulate_evict(evicted)
        return evicted


def _store(manager, store, keys, job_id):
    out = manager.prepare_store(keys, _ctx())
    assert out is not None and out.keys_to_store == keys
    manager.submit_store_job(job_id, keys, list(range(len(keys))))
    store.complete_save([h for key in keys for h in sub_hashes(key)])
    manager.poll_finished_jobs()


def _seeded(refresh_lru, n=4):
    """A, B, C, D stored in that order (A least recently used)."""
    store = LRUStore()
    manager = make_manager(store, refresh_lru=refresh_lru)
    for i in range(n):
        _store(manager, store, [_key(i)], job_id=i)
    store.calls.clear()
    return store, manager


class TestPrepareStoreRefresh(unittest.TestCase):
    def test_off_re_store_keeps_the_old_eviction_order(self):
        # Baseline: nothing re-offered.
        store0, _ = _seeded(refresh_lru=False)
        baseline = store0.evict(8)
        # Key off: re-offering a resident key is a probe; it moves nothing.
        store, manager = _seeded(refresh_lru=False)
        out = manager.prepare_store([_key(0)], _ctx())
        self.assertEqual(out.keys_to_store, [])
        self.assertEqual(store.calls_named("release"), [])
        for call in store.calls_named("lookup"):
            self.assertFalse(call.args[2])
        evicted = store.evict(8)
        self.assertEqual(evicted, baseline)
        self.assertEqual(evicted, sub_hashes(_key(0)) + sub_hashes(_key(1)))

    def test_on_re_store_moves_resident_key_to_mru(self):
        store, manager = _seeded(refresh_lru=True)
        out = manager.prepare_store([_key(0)], _ctx())
        # Same store decision as with the key off.
        self.assertEqual(out.keys_to_store, [])
        self.assertEqual(store.pinned_hashes(), {})
        evicted = store.evict(4)
        self.assertEqual(evicted, sub_hashes(_key(1)))
        for h in sub_hashes(_key(0)):
            self.assertIn(h, store.entries)

    def test_on_mixed_batch_stores_the_same_keys(self):
        store, manager = _seeded(refresh_lru=True, n=2)
        out = manager.prepare_store([_key(0), _key(7)], _ctx())
        self.assertEqual(out.keys_to_store, [_key(7)])
        self.assertEqual(manager.host_stats["skipped_resident"], 1)
        self.assertEqual(manager.host_stats["refreshed"], 1)


class TestTouchPrefix(unittest.TestCase):
    def test_touched_key_survives_the_pass_that_evicts_it_untouched(self):
        store, _ = _seeded(refresh_lru=False)
        self.assertEqual(store.evict(4), sub_hashes(_key(0)))

        store, manager = _seeded(refresh_lru=True)
        self.assertEqual(manager.touch_prefix([_key(0)]), 1)
        self.assertEqual(store.pinned_hashes(), {})
        self.assertEqual(store.evict(4), sub_hashes(_key(1)))

    def test_chain_head_ends_most_recent(self):
        store, manager = _seeded(refresh_lru=True)
        # The resident prefix is A, B; X is absent and ends the chain.
        self.assertEqual(manager.touch_prefix([_key(0), _key(1), _key(9)]), 2)
        # Untouched C, D first, then the chain from its tail back to its head.
        chain = sub_hashes(_key(0)) + sub_hashes(_key(1))
        self.assertEqual(
            store.evict(16),
            sub_hashes(_key(2)) + sub_hashes(_key(3)) + chain[::-1],
        )

    def test_absent_prefix_touches_nothing(self):
        store, manager = _seeded(refresh_lru=True)
        self.assertEqual(manager.touch_prefix([_key(9), _key(0)]), 0)
        self.assertEqual(store.calls_named("release"), [])


class TestWriteBackFinishTouches(unittest.TestCase):
    TPC = base.TestFinishedRequestStoreBuild.TPC

    def _finish(self, refresh_lru):
        from vllm.v1.request import RequestStatus

        store, manager = _seeded(refresh_lru=refresh_lru)
        helper = base.TestFinishedRequestStoreBuild()
        sched = helper._make_sched(manager)
        sched._wb_blocks = OrderedDict()
        sched._wb_owners = {}
        sched._wb_hash = {}
        sched.wb_counters = Counter()
        # The request reused A, B (resident) and computed X.
        req = _FakeFinishedRequest(
            "a", 3 * self.TPC, 3 * self.TPC - 1, RequestStatus.FINISHED_STOPPED
        )
        helper._track(sched, req, [_key(0), _key(1), _key(9)], [5, 6, 7])
        store.calls.clear()
        self.assertTrue(sched._wb_register("a"))
        return store

    def test_on_finish_refreshes_the_resident_prefix(self):
        store = self._finish(refresh_lru=True)
        self.assertEqual(store.evict(8), sub_hashes(_key(2)) + sub_hashes(_key(3)))

    def test_off_finish_touches_nothing(self):
        store = self._finish(refresh_lru=False)
        self.assertEqual(store.calls, [])
        self.assertEqual(store.evict(8), sub_hashes(_key(0)) + sub_hashes(_key(1)))


class TestHostStats(unittest.TestCase):
    def test_counters_and_log_line(self):
        store, manager = _seeded(refresh_lru=True, n=3)
        self.assertEqual(manager.lookup(_key(0), _ctx()), LookupResult.HIT)
        self.assertEqual(manager.lookup(_key(1), _ctx()), LookupResult.HIT)
        self.assertEqual(manager.lookup(_key(9), _ctx()), LookupResult.MISS)
        # A store job in flight: HIT_PENDING, and its four pins are held.
        manager.prepare_store([_key(5)], _ctx())
        self.assertEqual(manager.lookup(_key(5), _ctx()), LookupResult.HIT_PENDING)
        manager.submit_store_job(10, [_key(5)], [3])
        manager.prepare_store([_key(1)], _ctx())  # resident: skipped, refreshed
        store.evict(4)
        c = manager.host_stats
        self.assertEqual((c["HIT"], c["MISS"], c["HIT_PENDING"]), (2, 1, 1))
        self.assertEqual((c["skipped_resident"], c["refreshed"]), (1, 1))

        from vllm_torchtpu.offload import raiden_store

        with patch.object(raiden_store.logger, "info") as info:
            manager._maybe_log_host_stats(now=100.0)  # arms the clock
            manager._maybe_log_host_stats(now=110.0)  # under 30 s: silent
            info.assert_not_called()
            manager._maybe_log_host_stats(now=140.0)
        (call,) = info.call_args_list
        fmt, *args = call.args
        self.assertTrue(fmt.startswith("Raiden host offload: "))
        # 40 s; HIT 2, MISS 1, HIT_PENDING 1; 4 blocks / 40 s; 4 pinned;
        # 3 keys x 4 saved - 4 evicted = 8 of 32; 1 skipped; 1 refreshed.
        self.assertEqual(args, [40.0, 2, 1, 1, 0.1, 4, 8, 32, 1, 1, True])
        # Counters restart after the line.
        self.assertEqual(sum(manager.host_stats.values()), 0)
        self.assertEqual(manager._host_evicted_blocks, 0)
        # The landed save of the in-flight job counts toward used blocks.
        store.complete_save(sub_hashes(_key(5)))
        manager.poll_finished_jobs()
        self.assertEqual(manager._host_blocks_used, 12)

    def test_schedule_end_drives_the_line(self):
        _, manager = _seeded(refresh_lru=False, n=1)
        with patch.object(manager, "_maybe_log_host_stats") as log:
            manager.on_schedule_end(None)
        log.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
