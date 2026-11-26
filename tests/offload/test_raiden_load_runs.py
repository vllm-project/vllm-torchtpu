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
"""VLLM_TPU_OFFLOAD_MAX_INFLIGHT_LOAD_RUNS: paced local loads also bounded by
the Raiden copies (host- and HBM-consecutive runs) they keep in flight."""

from unittest.mock import MagicMock

from tests.offload.test_raiden_store_offload import (
    FakeBlockStatus,
    FakeKVCacheStore,
    _Entry,
    make_manager,
)
from vllm_torchtpu.offload.raiden_store import (
    _dma_order,
    _prefix_within_runs,
    _QueuedLoad,
    _run_starts,
)


def test_run_starts_needs_both_sides_consecutive():
    assert _run_starts([4, 5, 6, 9, 10], [0, 1, 3, 4, 5]) == [
        True, False, True, True, False,
    ]
    # Unknown host blocks (-1): the device side alone decides.
    assert _run_starts([1, 2, 4], [-1, -1, -1]) == [True, False, True]


def test_prefix_within_runs():
    starts = [True, False, True, False, True, True]
    assert _prefix_within_runs(starts, 6, 1) == 2
    assert _prefix_within_runs(starts, 6, 2) == 4
    assert _prefix_within_runs(starts, 3, 2) == 3
    assert _prefix_within_runs(starts, 6, 10) == 6
    # At least one pair, whatever the room.
    assert _prefix_within_runs([True, True], 2, 1) == 1


def test_dma_order_returns_host_ids_sorted():
    hashes, dev, host = _dma_order([b"a", b"b", b"c"], [7, 8, 9], [5, 3, 4])
    assert hashes == [b"b", b"c", b"a"] and dev == [8, 9, 7] and host == [3, 4, 5]


def _queue(m, device_ids, host_ids):
    hashes = [bytes([i]) for i in range(len(device_ids))]
    for h in hashes:  # resident and pinned, as prepare_load leaves them
        m._store.entries[h] = _Entry(status=FakeBlockStatus.HOST, pin_count=1)
    starts = _run_starts(device_ids, host_ids) if m._max_inflight_load_runs else []
    m._queued_loads.append(_QueuedLoad(MagicMock(), hashes, device_ids, starts))
    return hashes


def _loads(store):
    return [c.args[0] for c in store.calls if c.name == "load"]


def test_runs_bound_cuts_pieces_at_copy_boundaries():
    store = FakeKVCacheStore()
    m = make_manager(store, max_inflight_load_sub_blocks=100, max_inflight_load_runs=2)
    # Runs: [0, 1] [2] [3, 4, 5] by the scattered host side.
    hashes = _queue(m, [10, 11, 12, 13, 14, 15], [0, 1, 7, 20, 21, 22])
    m._launch_queued_loads()
    assert _loads(store) == [hashes[:3]]
    assert m._inflight_run_starts == {hashes[0], hashes[2]}
    # The first copy lands: one more run fits.
    m._inflight_loads.difference_update(hashes[:2])
    m._inflight_run_starts.discard(hashes[0])
    m._launch_queued_loads()
    assert _loads(store)[-1] == hashes[3:]
    assert not m._queued_loads


def test_runs_bound_off_keeps_sub_block_pacing():
    store = FakeKVCacheStore()
    m = make_manager(store, max_inflight_load_sub_blocks=4)
    assert m._max_inflight_load_runs == 0
    hashes = _queue(m, [10, 11, 12, 13, 14, 15], [0, 1, 7, 20, 21, 22])
    m._launch_queued_loads()
    assert _loads(store) == [hashes[:4]]
    assert not m._inflight_run_starts


def test_flag_default_off(monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.delenv("VLLM_TPU_OFFLOAD_MAX_INFLIGHT_LOAD_RUNS", raising=False)
    assert envs.VLLM_TPU_OFFLOAD_MAX_INFLIGHT_LOAD_RUNS == 0
    monkeypatch.setenv("VLLM_TPU_OFFLOAD_MAX_INFLIGHT_LOAD_RUNS", "12")
    assert envs.VLLM_TPU_OFFLOAD_MAX_INFLIGHT_LOAD_RUNS == 12
