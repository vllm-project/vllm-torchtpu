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
"""TPU_RAIDEN_FLUSH_REAL_REUSE / TPU_RAIDEN_LOSS_TRACE: stores are flushed
only for blocks the pool hands out fresh, and the store-side key trace."""

import collections
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.v1.core.block_pool import BlockPool

from tests.offload.test_raiden_lru_refresh import _bare_scheduler, _group
from tests.offload.test_raiden_store_offload import (
    FakeKVCacheStore,
    _ctx,
    _group_key,
    make_manager,
    sub_hashes,
)


def _scheduler_with_pool(flush_real_reuse=True, num_blocks=16):
    s = _bare_scheduler([_group(0)])
    s._write_back = None
    s._contiguous_alloc_scan_factor = 1  # leave the pool's order alone
    s._flush_real_reuse = flush_real_reuse
    s._offload_stats = True
    s._fence_pending = {}
    pool = BlockPool(num_gpu_blocks=num_blocks, enable_caching=True, hash_block_size=16)
    s.attach_block_pool(pool)
    return s, pool


def test_fresh_watch_records_only_popped_blocks():
    s, pool = _scheduler_with_pool()
    blocks = pool.get_new_blocks(3)
    assert s._fresh_block_ids == {b.block_id for b in blocks}
    # Off: the pool is left alone.
    s2, pool2 = _scheduler_with_pool(flush_real_reuse=False)
    pool2.get_new_blocks(2)
    assert s2._fresh_block_ids is None
    assert not getattr(pool2, "_tpu_raiden_fresh_watch", False)


def _run_update(s, listed: set[int], pending: dict[int, set[int]]):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    s._block_id_to_pending_jobs = pending

    def base_update(self, scheduler_output):
        # The base lists every block of every scheduled request (prefix hits
        # included) as allocated.
        self._current_batch_allocated_block_ids = set(listed)

    with patch.object(OffloadingConnectorScheduler, "_update_req_states", base_update):
        s._update_req_states(SimpleNamespace())


def test_prefix_hit_blocks_do_not_flush_parked_store():
    s, pool = _scheduler_with_pool()
    fresh = {b.block_id for b in pool.get_new_blocks(2)}
    hit = {b for b in range(1, 16) if b not in fresh}
    hit_block = min(hit)
    fresh_block = min(fresh)
    from vllm_torchtpu.offload.raiden_connector import _ParkedStore

    # Job 10 stores the chain the new request hits; job 11 a block reused fresh.
    s._fence_pending = {
        10: _ParkedStore(keys=[b"k0", b"k1"], device_block_ids=[hit_block, 0]),
        11: _ParkedStore(keys=[b"k2"], device_block_ids=[fresh_block]),
    }
    s._raiden_manager.is_job_launched.return_value = False
    _run_update(s, {hit_block} | fresh, {hit_block: {10}, fresh_block: {11}})
    assert s._current_batch_allocated_block_ids == fresh
    assert s.wb_counters["hitflush_skipped_parked"] == 1
    assert s.wb_counters["hitflush_skipped_parked_keys"] == 2
    # The fresh set starts over for the next step.
    assert s._fresh_block_ids == set()


def test_fresh_block_not_listed_by_scheduler_output_still_flushes():
    s, pool = _scheduler_with_pool()
    fresh = {b.block_id for b in pool.get_new_blocks(1)}
    _run_update(s, set(), {})
    assert s._current_batch_allocated_block_ids == fresh


def test_off_keeps_base_allocated_set():
    s, _ = _scheduler_with_pool(flush_real_reuse=False)
    _run_update(s, {1, 2, 3}, {})
    assert s._current_batch_allocated_block_ids == {1, 2, 3}


def test_flush_check_end_to_end_on_base_build_connector_meta():
    """The base's own re-allocation check sees only fresh blocks."""
    s, pool = _scheduler_with_pool()
    fresh = {b.block_id for b in pool.get_new_blocks(1)}
    (fresh_block,) = fresh
    s._current_batch_jobs_to_flush = set()
    _run_update(s, {1, 2, fresh_block}, {1: {10}, fresh_block: {11}})
    s._current_batch_jobs_to_flush.update(
        jid
        for bid in s._current_batch_allocated_block_ids
        if bid in s._block_id_to_pending_jobs
        for jid in s._block_id_to_pending_jobs[bid]
    )
    assert s._current_batch_jobs_to_flush == {11}


# -- loss trace ----------------------------------------------------------------


def test_manager_key_trace_events():
    store = FakeKVCacheStore()
    m = make_manager(store, device_block_size_factor=2)
    keys = [_group_key(i, 0) for i in range(3)]
    assert m.explain_missing(keys[0])[0] == "never"  # trace off: nothing known
    m.enable_loss_trace()
    out = m.prepare_store(keys, _ctx())
    assert out is not None and out.keys_to_store == keys
    assert m.explain_missing(keys[0])[0] == "offered"
    m.on_store_job_cancelled(keys[:1])
    event, age, refreshed, stored = m.explain_missing(keys[0])
    assert event == "cancelled" and age >= 0 and refreshed == -1 and stored == -1
    m._trace(keys[2:], "stored")
    m._on_eviction(sub_hashes(keys[2], 2))
    event, _, _, stored = m.explain_missing(keys[2])
    assert event == "evicted" and stored >= 0
    m._on_eviction(sub_hashes(keys[1], 2)[:1])
    assert m.explain_missing(keys[1])[0] == "evicted"
    assert m.explain_missing(_group_key(7, 0)) [0] == "never"


def test_trace_miss_logs_once_per_request_and_only_large_misses():
    s = _bare_scheduler([_group(0)])
    s._loss_trace = True
    s._traced_req_ids = set()
    s._raiden_manager.explain_missing.return_value = ("evicted", 120.0, -1.0, 900.0)
    req = SimpleNamespace(
        request_id="r1",
        num_prompt_tokens=128 * 100,
        block_hashes=[b"h%d" % i for i in range(100)],
    )
    s._trace_miss(req, 128 * 10, 128 * 5)
    s._trace_miss(req, 128 * 10, 128 * 5)  # same request again: no recount
    assert s.wb_counters["miss8k_evicted"] == 1
    assert s.wb_counters["miss8k_tok_evicted"] == 128 * 85
    ((key,), _) = s._raiden_manager.explain_missing.call_args
    from vllm.v1.kv_offload.base import make_offload_key

    assert key == make_offload_key(b"h15", 0)  # first chunk past the hit
    small = SimpleNamespace(
        request_id="r2", num_prompt_tokens=128 * 100, block_hashes=req.block_hashes
    )
    s._trace_miss(small, 128 * 40, 128 * 1)  # 59 chunks = 7552 tokens < 8k
    assert s.wb_counters["miss8k_evicted"] == 1
    # request_finished forgets the request.
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    s._lru_refresh_on_finish = False
    with patch.object(
        OffloadingConnectorScheduler, "request_finished", return_value=(False, None)
    ):
        s.request_finished(req)
    assert "r1" not in s._traced_req_ids


def test_bare_scheduler_request_finished_without_trace_state():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    s = _bare_scheduler([_group(0)])
    s._lru_refresh_on_finish = False
    with patch.object(
        OffloadingConnectorScheduler, "request_finished", return_value=(False, None)
    ):
        assert s.request_finished(SimpleNamespace(request_id="x")) == (False, None)
