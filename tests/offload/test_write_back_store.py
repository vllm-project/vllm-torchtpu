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
"""Write-back store policy of the Raiden offloading scheduler: finished
requests are stored only when their blocks near eviction, never on compute."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from tests.offload import test_raiden_store_offload as base
from tests.offload.test_raiden_store_offload import (
    FakeKVCacheStore,
    _FakeFinishedRequest,
    _key,
    make_manager,
)
from vllm_torchtpu.offload.raiden_connector import WriteBackPolicy


class _Block(SimpleNamespace):
    pass


def _pool(free_order):
    """A block pool whose free queue hands out ``free_order`` first."""
    head, tail = _Block(block_id=-1), _Block(block_id=-1)
    prev = head
    for bid in free_order:
        block = _Block(block_id=bid)
        prev.next_free_block = block
        prev = block
    prev.next_free_block = tail
    tail.next_free_block = None
    queue = SimpleNamespace(fake_free_list_head=head, fake_free_list_tail=tail)
    return SimpleNamespace(free_block_queue=queue)


def _output(finished=(), new_blocks=None):
    from vllm.v1.core.sched.output import SchedulerOutput

    out = SchedulerOutput.make_empty()
    out.finished_req_ids = set(finished)
    out.num_scheduled_tokens = {"running": 4}
    if new_blocks:
        out.scheduled_cached_reqs.req_ids = ["running"]
        out.scheduled_cached_reqs.new_block_ids = [(list(new_blocks),)]
        out.scheduled_cached_reqs.resumed_req_ids = set()
    return out


class TestWriteBackPolicyConfig(unittest.TestCase):
    def test_parse(self):
        self.assertIsNone(WriteBackPolicy.from_extra_config({}))
        self.assertIsNone(
            WriteBackPolicy.from_extra_config({"store_policy": "write_through"})
        )
        self.assertEqual(
            WriteBackPolicy.from_extra_config({"store_policy": "write_back"}),
            WriteBackPolicy(),
        )
        policy = WriteBackPolicy.from_extra_config(
            {"store_policy": {"mode": "write_back", "window_blocks": 7}}
        )
        self.assertEqual(policy, WriteBackPolicy(window_blocks=7))
        for bad in ("lru", {"mode": "write_back", "w": 1}, {"mode": "x"}):
            with self.assertRaises(ValueError):
                WriteBackPolicy.from_extra_config({"store_policy": bad})


_POLICY = WriteBackPolicy(window_blocks=2, max_requests_per_step=1)


class TestWriteBackScheduling(unittest.TestCase):
    TPC = base.TestFinishedRequestStoreBuild.TPC

    def _sched(self, policy=_POLICY):
        helper = base.TestFinishedRequestStoreBuild()
        sched = helper._make_sched(make_manager(FakeKVCacheStore()))
        sched.manager = MagicMock(wraps=sched.manager)
        sched._write_back = policy
        sched._block_pool = None
        from collections import Counter, OrderedDict

        sched._wb_owners = {}
        sched._wb_blocks = OrderedDict()
        sched.wb_counters = Counter()
        self.helper = helper
        return sched

    def _finished(self, sched, req_id, block_ids):
        from vllm.v1.request import RequestStatus

        n = len(block_ids)
        req = _FakeFinishedRequest(
            req_id, n * self.TPC, n * self.TPC - 1, RequestStatus.FINISHED_STOPPED
        )
        keys = [_key(100 * len(req_id) + i) for i in range(n)]
        self.helper._track(sched, req, keys, list(block_ids))
        return keys

    def test_held_until_near_eviction_and_no_write_through(self):
        sched = self._sched()
        keys = self._finished(sched, "a", [5, 6])
        sched.attach_block_pool(_pool([1, 2, 3, 5, 6]))
        out = sched._write_back_output(_output(finished={"a"}))
        # Held: neither stored nor finished; running requests never store.
        self.assertEqual(out.finished_req_ids, set())
        self.assertEqual(out.num_scheduled_tokens, {})
        self.assertEqual(sched._build_store_jobs(out), {})
        self.assertIn("a", sched._wb_blocks)
        sched.manager.on_request_finished.assert_not_called()
        # Its blocks reach the head of the free queue: stored now.
        sched.attach_block_pool(_pool([5, 6, 1]))
        out = sched._write_back_output(_output())
        self.assertEqual(out.finished_req_ids, {"a"})
        (job,) = sched._build_store_jobs(out).values()
        self.assertEqual(job.dst_spec.keys, keys)
        self.assertEqual(job.src_spec.block_ids.tolist(), [5, 6])
        self.assertEqual(sched._wb_blocks, {})
        self.assertEqual(sched._wb_owners, {})

    def test_reused_block_drops_the_request_without_storing(self):
        sched = self._sched()
        self._finished(sched, "a", [5, 6])
        sched.attach_block_pool(_pool([1, 2, 5, 6]))
        sched._write_back_output(_output(finished={"a"}))
        out = sched._write_back_output(_output(new_blocks=[6]))
        self.assertEqual(out.finished_req_ids, set())
        self.assertNotIn("a", sched._req_status)
        self.assertEqual(sched._wb_owners, {})
        sched.manager.on_request_finished.assert_called_once()
        self.assertEqual(sched.wb_counters["dropped_requests"], 1)

    def test_without_a_block_pool_stores_at_finish(self):
        sched = self._sched()
        keys = self._finished(sched, "a", [5, 6])
        out = sched._write_back_output(_output(finished={"a"}))
        self.assertEqual(out.finished_req_ids, {"a"})
        (job,) = sched._build_store_jobs(out).values()
        self.assertEqual(job.dst_spec.keys, keys)

    def test_flush_is_bounded_per_step_and_shared_blocks_have_owners(self):
        sched = self._sched()
        self._finished(sched, "a", [5, 6])
        self._finished(sched, "bb", [5, 7])  # shares block 5
        sched.attach_block_pool(_pool([9, 9, 9]))
        sched._write_back_output(_output(finished={"a", "bb"}))
        self.assertEqual(sched._wb_owners[5], {"a", "bb"})
        sched.attach_block_pool(_pool([5, 7]))
        first = sched._write_back_output(_output()).finished_req_ids
        second = sched._write_back_output(_output()).finished_req_ids
        self.assertEqual(len(first), 1)
        self.assertEqual(first | second, {"a", "bb"})
        self.assertEqual(sched._wb_owners, {})


class TestEvictionAwareConnector(unittest.TestCase):
    def test_connector_forwards_the_block_pool_to_its_scheduler(self):
        from vllm_torchtpu.core.eviction_aware_connector import EvictionAwareConnector
        from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector

        connector = object.__new__(TPURaidenOffloadingConnector)
        connector.connector_scheduler = MagicMock()
        self.assertIsInstance(connector, EvictionAwareConnector)
        pool = object()
        connector.attach_block_pool(pool)
        connector.connector_scheduler.attach_block_pool.assert_called_once_with(pool)
        connector.connector_scheduler = None
        connector.attach_block_pool(pool)  # worker side: no-op
