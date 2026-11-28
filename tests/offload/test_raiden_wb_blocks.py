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
"""Raiden write-back with ``granularity="block"``: per-chunk stores of the
held blocks near the free-queue head."""

import collections
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.core.block_pool import BlockPool

from tests.offload.test_raiden_lru_refresh import _bare_scheduler, _group
from vllm_torchtpu.offload.raiden_connector import WriteBackPolicy


def test_policy_granularity_parsing():
    p = WriteBackPolicy.from_extra_config(
        {"store_policy": {"mode": "write_back", "granularity": "block"}}
    )
    assert p.per_block
    default = WriteBackPolicy.from_extra_config({"store_policy": "write_back"})
    assert not default.per_block
    with pytest.raises(ValueError):
        WriteBackPolicy.from_extra_config(
            {"store_policy": {"mode": "write_back", "granularity": "chunk"}}
        )


def _scheduler(window_blocks=2, num_blocks=16):
    s = _bare_scheduler([_group(0)])
    s.config.blocks_per_chunk = 1
    s.config.offload_prompt_only = True
    s._write_back = WriteBackPolicy(window_blocks=window_blocks, granularity="block")
    s._contiguous_alloc_scan_factor = 1  # leave the pool's order alone
    s._flush_real_reuse = False
    s._wb_drop_covered = False
    s._offload_stats = False
    s._fence_pending = {}
    s._wb_blocks = collections.OrderedDict()
    s._wb_owners = {}
    s._maybe_log_write_back = lambda: None
    s.manager = MagicMock()
    pool = BlockPool(num_gpu_blocks=num_blocks, enable_caching=True, hash_block_size=16)
    s.attach_block_pool(pool)
    return s, pool


def _held(s, req_id, block_ids, num_prompt_tokens=10_000):
    keys = [f"{req_id}-k{i}" for i in range(len(block_ids))]
    gs = SimpleNamespace(
        next_stored_chunk_idx=0, offload_keys=keys, block_ids=block_ids
    )
    s._req_status[req_id] = SimpleNamespace(
        req=SimpleNamespace(num_prompt_tokens=num_prompt_tokens),
        req_context=object(),
        group_states=[gs],
        transfer_jobs=set(),
    )
    assert s._wb_register(req_id)
    return keys, gs


def _step(s, allocated=(), finished=()):
    out = SimpleNamespace(
        finished_req_ids=set(finished), num_scheduled_tokens={}, scheduled_new_reqs=[]
    )
    with patch(
        "vllm_torchtpu.offload.raiden_connector._allocated_block_ids",
        return_value=set(allocated),
    ), patch(
        "vllm_torchtpu.offload.raiden_connector.dataclasses.replace",
        side_effect=lambda o, **kw: SimpleNamespace(**{**vars(o), **kw}),
    ):
        return s._write_back_output_blocks(out)


def test_window_stores_only_chunks_near_the_head():
    s, pool = _scheduler(window_blocks=2)
    # The free queue starts 1, 2, 3, ...: blocks 1 and 2 are in the window.
    keys, _ = _held(s, "r", [1, 2, 3, 4])
    out = _step(s)
    assert s._wb_allowed == {"r": {keys[0], keys[1]}}
    assert out.num_scheduled_tokens == {"r": 0}
    assert out.finished_req_ids == set()
    assert s._wb_blocks["r"] == {3, 4}  # still held for the rest
    assert set(s._wb_owners) == {3, 4}
    assert s._wb_to_release == set()


def test_reused_block_is_forgotten_without_dropping_the_holder():
    s, pool = _scheduler(window_blocks=1)
    _held(s, "r", [5, 6, 7])
    _step(s, allocated={6})
    assert s._wb_blocks["r"] == {5, 7}
    assert s.wb_counters["dropped_requests"] == 0
    # Every block gone: the holder is released, not finished through the base.
    _step(s, allocated={5, 7})
    assert s._wb_to_release == {"r"}


def test_build_partial_limits_prepare_store_and_restores_index():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    s, pool = _scheduler(window_blocks=2)
    keys, gs = _held(s, "r", [1, 2, 3])
    _held(s, "q", [9, 10])  # outside the window: untouched
    out = _step(s)
    seen = []
    s.manager.prepare_store = lambda ks, ctx: seen.append(list(ks)) or "ok"

    def fake_base(self, scheduler_output):
        # The base offers every chunk past next_stored_chunk_idx and advances it.
        for req_id in scheduler_output.num_scheduled_tokens:
            st = self._req_status[req_id]
            self.manager.prepare_store(st.group_states[0].offload_keys, st.req_context)
            st.group_states[0].next_stored_chunk_idx = 3
        return "meta"

    with patch.object(OffloadingConnectorScheduler, "build_connector_meta", fake_base):
        assert s._wb_build_partial(out) == "meta"
    assert seen == [[keys[0], keys[1]]]
    assert gs.next_stored_chunk_idx == 0
    assert s.manager.prepare_store.__name__ == "<lambda>"  # unwrapped again
    assert s._wb_allowed == {}


def test_release_happens_after_the_partial_store():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    s, pool = _scheduler(window_blocks=4)
    _held(s, "r", [1, 2])  # both in the window: stored, then nothing is left
    out = _step(s)
    assert s._wb_to_release == {"r"}
    s.manager.prepare_store = lambda ks, ctx: "ok"
    with patch.object(
        OffloadingConnectorScheduler, "build_connector_meta", lambda self, o: "meta"
    ):
        s._wb_build_partial(out)
    assert "r" not in s._req_status
    assert "r" not in s._wb_blocks
    assert s.wb_counters["released_requests"] == 1
    s.manager.on_request_finished.assert_called_once()


def test_request_granularity_unchanged():
    s, pool = _scheduler(window_blocks=2)
    s._write_back = WriteBackPolicy(window_blocks=2)
    _held(s, "r", [1, 2, 3, 4])
    assert s._wb_near_eviction() == ["r"]


def test_late_store_policy_needs_block_granularity():
    with pytest.raises(ValueError):
        WriteBackPolicy.from_extra_config(
            {"store_policy": {"mode": "write_back", "late_store": True}}
        )


def _late_scheduler(window_blocks=1):
    s, pool = _scheduler(window_blocks=window_blocks)
    s._write_back = WriteBackPolicy(
        window_blocks=window_blocks, granularity="block", late_store=True
    )
    # attach again: late_store installs the fresh-block watch
    s.attach_block_pool(pool)
    assert s._fresh_block_ids is not None
    return s, pool


def test_late_store_stores_reallocated_held_chunk_now():
    s, pool = _late_scheduler(window_blocks=1)
    keys, _ = _held(s, "r", [5, 6, 7])  # free queue head is block 1: not r's
    _step(s)
    assert s._wb_allowed == {}
    s._fresh_block_ids.update({6})
    _step(s, allocated={6})
    assert s._wb_allowed == {"r": {keys[1]}}
    assert s._wb_urgent == {"r"}
    assert s._wb_blocks["r"] == {5, 7}
    # A prefix hit (not fresh) is only forgotten.
    s._fresh_block_ids.clear()
    _step(s, allocated={5})
    assert s._wb_allowed == {} and s._wb_urgent == set()
    assert s._wb_blocks["r"] == {7}


def test_late_store_skips_holders_registered_this_step():
    s, pool = _late_scheduler(window_blocks=1)
    keys = [f"n-k{i}" for i in range(2)]
    gs = SimpleNamespace(
        next_stored_chunk_idx=0, offload_keys=keys, block_ids=[8, 9]
    )
    s._req_status["n"] = SimpleNamespace(
        req=SimpleNamespace(num_prompt_tokens=10_000),
        req_context=object(),
        group_states=[gs],
        transfer_jobs=set(),
    )
    s._fresh_block_ids.update({8})
    _step(s, allocated={8}, finished={"n"})
    assert s._wb_urgent == set()
    assert s._wb_blocks["n"] == {9}


def test_late_store_job_skips_the_fence():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingConnectorMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )
    from vllm.v1.kv_offload.base import GPULoadStoreSpec

    from vllm_torchtpu.offload.raiden_store import RaidenLoadStoreSpec

    s, pool = _late_scheduler(window_blocks=1)
    keys, _ = _held(s, "r", [5, 6])
    _step(s)
    s._fresh_block_ids.update({6})
    mgr = MagicMock()
    mgr.poll_finished_jobs.return_value = []
    mgr.drain_jobs.return_value = []
    mgr.is_job_launched.return_value = True
    s._raiden_manager = mgr
    s.manager = mgr
    job = SimpleNamespace(
        req_id="r",
        src_spec=GPULoadStoreSpec([6], group_sizes=[1], block_indices=[1]),
        dst_spec=RaidenLoadStoreSpec([keys[1]]),
    )
    meta = OffloadingConnectorMetadata(
        load_jobs={}, store_jobs={7: job}, jobs_to_flush={7}
    )
    out = SimpleNamespace(
        finished_req_ids=set(), num_scheduled_tokens={}, scheduled_new_reqs=[]
    )
    with patch(
        "vllm_torchtpu.offload.raiden_connector._allocated_block_ids",
        return_value={6},
    ), patch(
        "vllm_torchtpu.offload.raiden_connector.dataclasses.replace",
        side_effect=lambda o, **kw: SimpleNamespace(**{**vars(o), **kw}),
    ), patch.object(
        OffloadingConnectorScheduler, "build_connector_meta", lambda self, o: meta
    ):
        s.build_connector_meta(out)
    mgr.submit_store_job.assert_called_once_with(7, [keys[1]], [6])
    assert 7 not in s._fence_pending
    mgr.drain_jobs.assert_called_once_with({7})


def _staged_late_step(
    num_rows, taken=0, rescue=False, store_blocks=(6,), rescue_split=False
):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingConnectorMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )
    from vllm.v1.kv_offload.base import GPULoadStoreSpec

    from vllm_torchtpu.offload.raiden_staging import StagedTransferScheduler
    from vllm_torchtpu.offload.raiden_store import RaidenLoadStoreSpec

    s, pool = _late_scheduler(window_blocks=1)
    keys, _ = _held(s, "r", [5, 6, 7])
    _step(s)
    s._fresh_block_ids.update({6})
    s._staged_rescue = rescue
    mgr = MagicMock()
    mgr.poll_finished_jobs.return_value = []
    s._raiden_manager = mgr
    s.manager = mgr
    s._staged = StagedTransferScheduler(mgr, num_rows, 1, rescue_split=rescue_split)
    if taken:
        s._staged.rows.take(taken)  # rows busy with other transfers
    blocks = list(store_blocks)
    job = SimpleNamespace(
        req_id="r",
        src_spec=GPULoadStoreSpec(
            blocks, group_sizes=[len(blocks)], block_indices=[1]
        ),
        dst_spec=RaidenLoadStoreSpec([keys[b - 5] for b in blocks]),
    )
    meta = OffloadingConnectorMetadata(
        load_jobs={}, store_jobs={7: job}, jobs_to_flush={7}
    )
    out = SimpleNamespace(
        finished_req_ids=set(), num_scheduled_tokens={}, scheduled_new_reqs=[]
    )
    with patch(
        "vllm_torchtpu.offload.raiden_connector._allocated_block_ids",
        return_value={6},
    ), patch(
        "vllm_torchtpu.offload.raiden_connector.dataclasses.replace",
        side_effect=lambda o, **kw: SimpleNamespace(**{**vars(o), **kw}),
    ), patch.object(
        OffloadingConnectorScheduler, "build_connector_meta", lambda self, o: meta
    ):
        res = s.build_connector_meta(out)
    return s, mgr, keys, res


def test_late_store_under_staging_packs_before_forward():
    """Late store with staging: rows now, an urgent pack (worker
    start_load_kv, before the forward), fenced, and not cancelled by the flush
    of its reused block; the save launches from the rows after the fence."""
    s, mgr, keys, res = _staged_late_step(num_rows=8)
    assert res.staging_packs == {}
    assert len(res.staging_urgent_packs) == 1
    (part_id, (blocks, rows)), = res.staging_urgent_packs.items()
    assert blocks == [6] and len(rows) == 1
    assert part_id in res.fence_job_ids
    mgr.on_store_job_cancelled.assert_not_called()
    # Every rank retires the fence: the save reads the rows.
    s._staged.on_fenced({part_id: 1})
    mgr.submit_store_job.assert_called_once_with(part_id, [keys[1]], rows)


def test_late_store_under_staging_without_rows_is_cancelled():
    s, mgr, keys, res = _staged_late_step(num_rows=2, taken=2)
    assert res.staging_urgent_packs == {} and res.staging_packs == {}
    mgr.submit_store_job.assert_not_called()
    mgr.on_store_job_cancelled.assert_called_once_with([keys[1]])


def test_staged_rescue_packs_the_reallocated_part_of_a_late_store():
    """TPU_RAIDEN_STAGED_RESCUE: a late store that does not fit the free rows
    as a whole still packs its reallocated part before the forward; the part
    whose block is not reallocated stays queued instead of being cancelled."""
    s, mgr, keys, res = _staged_late_step(
        num_rows=2, taken=1, rescue=True, store_blocks=(6, 7)
    )
    ((part_id, (blocks, rows)),) = res.staging_urgent_packs.items()
    assert blocks == [6] and len(rows) == 1
    assert part_id in res.fence_job_ids
    mgr.on_store_job_cancelled.assert_not_called()
    assert s._staged.stats["store_urgent_no_rows"] == 1
    assert s._staged.stats["store_kept_queued_parts"] == 1
    # Without the flag the same step cancels the whole job.
    s, mgr, keys, res = _staged_late_step(num_rows=2, taken=1, store_blocks=(6, 7))
    assert res.staging_urgent_packs == {}
    cancelled = [c.args[0] for c in mgr.on_store_job_cancelled.call_args_list]
    assert sorted(cancelled) == [[keys[1]], [keys[2]]]


def test_staged_rescue_split_packs_only_the_reallocated_block():
    """TPU_RAIDEN_STAGED_RESCUE_SPLIT: one free row, a queued part [6, 7] of
    which only 6 is reallocated: 6 packs before the forward, 7 stays queued
    (without the split the part needs 2 rows and is cancelled)."""
    s, mgr, keys, res = _staged_late_step(
        num_rows=4, taken=3, rescue=True, store_blocks=(6, 7), rescue_split=True
    )
    ((part_id, (blocks, rows)),) = res.staging_urgent_packs.items()
    assert blocks == [6] and len(rows) == 1
    assert part_id in res.fence_job_ids
    mgr.on_store_job_cancelled.assert_not_called()
    assert s._staged.stats["store_split_queued_parts"] == 1
    s, mgr, keys, res = _staged_late_step(
        num_rows=4, taken=3, rescue=True, store_blocks=(6, 7)
    )
    assert res.staging_urgent_packs == {}
    mgr.on_store_job_cancelled.assert_called_once_with([keys[1], keys[2]])


def test_staged_rescue_installs_the_fresh_watch():
    s, pool = _scheduler(window_blocks=1)
    s._fresh_block_ids = None
    pool2 = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    s._staged_rescue = True
    s.attach_block_pool(pool2)
    assert s._fresh_block_ids is not None
    pool2.get_new_blocks(2)
    assert len(s._fresh_block_ids) == 2


def test_worker_start_load_kv_packs_urgent_rows_first():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnector,
    )

    from vllm_torchtpu.offload.raiden_connector import (
        RaidenOffloadingConnectorMetadata,
        TPURaidenOffloadingConnector,
    )

    c = TPURaidenOffloadingConnector.__new__(TPURaidenOffloadingConnector)
    c._staging = MagicMock()
    order = []
    c._staging.pack.side_effect = lambda b, r: order.append(("pack", b, r))
    c._connector_metadata = RaidenOffloadingConnectorMetadata(
        load_jobs={},
        store_jobs={},
        jobs_to_flush=set(),
        staging_urgent_packs={11: ([6, 9], [0, 1])},
    )
    with patch.object(
        OffloadingConnector,
        "start_load_kv",
        lambda self, fc, **kw: order.append(("base",)),
    ), patch(
        "vllm_torchtpu.offload.raiden_connector._enqueue_pool_writes",
        side_effect=lambda t: order.append(("enqueue",)),
    ):
        c.start_load_kv(None)
    assert order == [("base",), ("pack", [6, 9], [0, 1]), ("enqueue",)]
    # Nothing urgent: no pack.
    c._connector_metadata = RaidenOffloadingConnectorMetadata(
        load_jobs={}, store_jobs={}, jobs_to_flush=set()
    )
    with patch.object(
        OffloadingConnector, "start_load_kv", lambda self, fc, **kw: None
    ):
        c.start_load_kv(None)
    assert len(order) == 3


def test_hold_all_chunks_needs_block_granularity():
    with pytest.raises(ValueError):
        WriteBackPolicy.from_extra_config(
            {"store_policy": {"mode": "write_back", "hold_all_chunks": True}}
        )


def _host_load_holder(s, hold_all):
    """A finished turn that hit chunks 0-1 in HBM, loaded chunk 2 from the
    host and computed chunk 3: vLLM's CHUNK_LEVEL policy left
    next_stored_chunk_idx at 3 (all cached chunks)."""
    s._write_back = WriteBackPolicy(
        window_blocks=s._write_back.window_blocks,
        granularity="block",
        hold_all_chunks=hold_all,
    )
    keys, gs = _held_unregistered(s, "r", [11, 12, 13, 14], next_stored=3)
    assert s._wb_register("r")
    return keys, gs


def _held_unregistered(s, req_id, block_ids, next_stored):
    keys = [f"{req_id}-k{i}" for i in range(len(block_ids))]
    gs = SimpleNamespace(
        next_stored_chunk_idx=next_stored, offload_keys=keys, block_ids=block_ids
    )
    s._req_status[req_id] = SimpleNamespace(
        req=SimpleNamespace(num_prompt_tokens=10_000),
        req_context=object(),
        group_states=[gs],
        transfer_jobs=set(),
    )
    return keys, gs


def test_host_load_turn_holds_its_hbm_hit_chunks_with_hold_all():
    s, pool = _scheduler(window_blocks=2)
    _host_load_holder(s, hold_all=False)
    assert s._wb_blocks["r"] == {14}  # the HBM-hit chunks 0-1 have no holder
    s2, pool2 = _scheduler(window_blocks=2)
    _host_load_holder(s2, hold_all=True)
    assert s2._wb_blocks["r"] == {11, 12, 13, 14}


def test_hold_all_partial_store_offers_chunks_below_next_stored():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    s, pool = _scheduler(window_blocks=16)
    keys, gs = _host_load_holder(s, hold_all=True)
    # Block 11 (an HBM-hit chunk) reaches the window.
    s._wb_owners = {11: {"r"}}
    s._wb_blocks["r"] = {11}
    out = _step(s)
    assert s._wb_allowed == {"r": {keys[0]}}
    seen = []
    s.manager.prepare_store = lambda ks, ctx: seen.append(list(ks)) or "ok"

    def fake_base(self, scheduler_output):
        # Like the base: offer the chunks from next_stored_chunk_idx on.
        for req_id in scheduler_output.num_scheduled_tokens:
            st = self._req_status[req_id]
            g = st.group_states[0]
            self.manager.prepare_store(
                g.offload_keys[g.next_stored_chunk_idx :], st.req_context
            )
            g.next_stored_chunk_idx = 4
        return "meta"

    with patch.object(OffloadingConnectorScheduler, "build_connector_meta", fake_base):
        s._wb_build_partial(out)
    assert seen == [[keys[0]]]
    assert gs.next_stored_chunk_idx == 3  # restored
