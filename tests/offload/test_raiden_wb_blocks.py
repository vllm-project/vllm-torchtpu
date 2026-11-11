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


def test_late_store_under_staging_is_cancelled_not_packed():
    """Staged packs run after the forward: a late store must be cancelled by
    the flush instead of packing reallocated blocks."""
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
    keys, _ = _held(s, "r", [5, 6])
    _step(s)
    s._fresh_block_ids.update({6})
    mgr = MagicMock()
    mgr.poll_finished_jobs.return_value = []
    s._raiden_manager = mgr
    s.manager = mgr
    s._staged = StagedTransferScheduler(mgr, 8, 1)
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
        res = s.build_connector_meta(out)
    assert res.staging_packs == {}
    mgr.submit_store_job.assert_not_called()
    mgr.on_store_job_cancelled.assert_called_once_with([keys[1]])
