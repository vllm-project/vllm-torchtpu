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
"""In-flight bounds of Raiden offload loads and saves (additional_config
offload_max_inflight_load_mib / offload_max_inflight_store_mib)."""

import unittest
from types import SimpleNamespace

from tests.offload.test_raiden_store_offload import (
    FakeKVCacheStore,
    _ctx,
    _key,
    make_manager,
    seed_host_entries,
    sub_hashes,
)


def _load_calls(store):
    return [c.args for c in store.calls_named("load")]


class TestUnbounded(unittest.TestCase):
    """Keys unset: one load()/insert()/save() per job, as before."""

    def test_default_manager_has_no_bounds(self):
        manager = make_manager(FakeKVCacheStore())
        self.assertEqual(manager._max_inflight_loads, 0)
        self.assertEqual(manager._max_inflight_stores, 0)

    def test_load_and_store_issue_one_call_each(self):
        store = FakeKVCacheStore()
        manager = make_manager(store)
        keys = [_key(i) for i in range(4)]
        for k in keys:
            seed_host_entries(store, k)
        manager.prepare_load(keys, _ctx("r"))
        manager.submit_load_job(1, keys, [0, 1, 2, 3], "r")
        self.assertEqual(len(_load_calls(store)), 1)
        self.assertFalse(manager._queued_loads)

        store_keys = [_key(10 + i) for i in range(4)]
        manager.prepare_store(store_keys, _ctx())
        manager.submit_store_job(2, store_keys, [7, 6, 5, 4])
        self.assertEqual(len(store.calls_named("insert")), 1)
        self.assertEqual(len(store.calls_named("save")), 1)
        self.assertFalse(manager._store_jobs)
        # Unbounded keeps the caller's key order (no device-block sort).
        (insert,) = store.calls_named("insert")
        self.assertEqual(insert.args[0][:4], sub_hashes(store_keys[0]))


class TestLoadBound(unittest.TestCase):
    def setUp(self):
        self.store = FakeKVCacheStore()
        # 6 sub-blocks in flight; keys have 4 sub-blocks each.
        self.manager = make_manager(self.store, max_inflight_load_sub_blocks=6)
        self.keys = [_key(i) for i in range(4)]
        for k in self.keys:
            seed_host_entries(self.store, k)
        self.batch = [h for k in self.keys for h in sub_hashes(k)]

    def test_load_split_at_bound_in_order(self):
        self.manager.prepare_load(self.keys, _ctx("r"))
        self.manager.submit_load_job(3, self.keys, [0, 1, 2, 3], "r")
        calls = _load_calls(self.store)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], self.batch[:6])
        self.assertEqual(calls[0][1], list(range(6)))
        self.assertTrue(self.manager.is_job_launched(3))
        # Nothing landed: no room, nothing more launches.
        self.assertEqual(self.manager.poll_finished_jobs(), [])
        self.assertEqual(len(_load_calls(self.store)), 1)

        launched = []
        while len(launched) < len(self.batch):
            last = _load_calls(self.store)[-1][0]
            self.assertLessEqual(len(last), 6)
            launched += last
            self.store.complete_load(last)
            finished = self.manager.poll_finished_jobs()
            if len(launched) < len(self.batch):
                self.assertEqual(finished, [])
        # Every sub-block once, in the original order and device pairing.
        self.assertEqual(launched, self.batch)
        dev = [d for c in _load_calls(self.store) for d in c[1]]
        self.assertEqual(dev, list(range(16)))
        (job,) = finished
        self.assertTrue(job.success)
        self.assertEqual(job.job_id, 3)
        self.assertFalse(self.manager.has_pending_work())
        self.assertEqual(self.store.pinned_hashes(), {})

    def test_second_load_queues_behind_first(self):
        self.manager.prepare_load(self.keys, _ctx("a"))
        self.manager.submit_load_job(1, self.keys[:1], [0], "a")
        self.manager.submit_load_job(2, self.keys[1:3], [1, 2], "a")
        calls = _load_calls(self.store)
        # Job 1 (4) + 2 of job 2 fit the bound of 6.
        self.assertEqual(calls[0][0], sub_hashes(self.keys[0]))
        self.assertEqual(calls[1][0], sub_hashes(self.keys[1])[:2])
        self.assertEqual(len(self.manager._inflight_loads), 6)

    def test_drain_cancels_unlaunched_part(self):
        self.manager.prepare_load(self.keys, _ctx("r"))
        self.manager.submit_load_job(3, self.keys, [0, 1, 2, 3], "r")
        first = _load_calls(self.store)[0][0]
        self.store.complete_load(first)
        drained = self.manager.drain_jobs({3})
        (job,) = drained
        self.assertFalse(job.success)
        self.assertFalse(self.manager._queued_loads)
        self.assertFalse(self.manager.has_pending_work())


class TestStoreBound(unittest.TestCase):
    def setUp(self):
        self.store = FakeKVCacheStore()
        # 8 sub-blocks = 2 keys per piece.
        self.manager = make_manager(self.store, max_inflight_store_sub_blocks=8)

    def test_store_runs_as_pieces_in_device_order(self):
        keys = [_key(i) for i in range(5)]
        devs = [9, 3, 7, 1, 5]
        self.manager.prepare_store(keys, _ctx())
        self.manager.submit_store_job(4, keys, devs)
        self.assertEqual(len(self.store.calls_named("save")), 1)
        self.assertTrue(self.manager.is_job_launched(4))
        self.assertTrue(self.manager.has_pending_work())
        order = [k for _, k in sorted(zip(devs, keys))]
        saved = []
        finished = []
        while len(saved) < len(keys) * 4:
            last = self.store.calls_named("save")[-1].args[0]
            self.assertLessEqual(len(last), 8)
            saved += last
            self.store.complete_save(last)
            finished = self.manager.poll_finished_jobs()
            if len(saved) < len(keys) * 4:
                self.assertEqual(finished, [])
        self.assertEqual(saved, [h for k in order for h in sub_hashes(k)])
        (job,) = finished
        self.assertEqual(job.job_id, 4)
        self.assertTrue(job.success)
        self.assertFalse(self.manager.has_pending_work())
        self.assertEqual(self.store.pinned_hashes(), {})

    def test_failed_piece_fails_job(self):
        keys = [_key(i) for i in range(4)]
        self.manager.prepare_store(keys, _ctx())
        self.manager.submit_store_job(5, keys, [0, 1, 2, 3])
        self.store.fail_insert = True
        first = self.store.calls_named("save")[-1].args[0]
        self.store.complete_save(first)
        (job,) = self.manager.poll_finished_jobs()
        self.assertEqual(job.job_id, 5)
        self.assertFalse(job.success)

    def test_drain_drops_queued_pieces(self):
        keys = [_key(i) for i in range(4)]
        self.manager.prepare_store(keys, _ctx())
        self.manager.submit_store_job(6, keys, [0, 1, 2, 3])
        first = self.store.calls_named("save")[-1].args[0]
        self.store.complete_save(first)
        drained = self.manager.drain_jobs({6})
        (job,) = drained
        self.assertEqual(job.job_id, 6)
        self.assertFalse(job.success)
        self.assertEqual(len(self.store.calls_named("save")), 1)
        self.assertFalse(self.manager.has_pending_work())


class TestSpecConversion(unittest.TestCase):
    def _spec(self, config):
        from vllm_torchtpu.offload.raiden_store import TPURaidenStoreOffloadingSpec

        spec = object.__new__(TPURaidenStoreOffloadingSpec)
        spec.vllm_config = SimpleNamespace(additional_config=config)
        spec._sub_block_bytes = 1 << 18  # 256 KiB per sub-block
        return spec

    def test_unset_is_zero(self):
        spec = self._spec({})
        self.assertEqual(spec._inflight_sub_blocks("offload_max_inflight_load_mib"), 0)

    def test_mib_to_sub_blocks(self):
        spec = self._spec(
            {"offload_max_inflight_load_mib": 128, "offload_max_inflight_store_mib": 64}
        )
        self.assertEqual(
            spec._inflight_sub_blocks("offload_max_inflight_load_mib"), 512
        )
        self.assertEqual(
            spec._inflight_sub_blocks("offload_max_inflight_store_mib"), 256
        )


if __name__ == "__main__":
    unittest.main()
