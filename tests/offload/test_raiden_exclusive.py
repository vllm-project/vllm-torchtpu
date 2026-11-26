# SPDX-License-Identifier: Apache-2.0
"""Exclusive host tier (store_policy exclusive): a load that lands erases its
host copy through Raiden's KVCacheStore.erase, so the host holds only blocks
HBM does not. Uses the fake store of test_raiden_store_offload."""

import pytest

from tests.offload.test_raiden_store_offload import (
    FakeBlockStatus,
    FakeKVCacheStore,
    _ctx,
    _key,
    make_manager,
    seed_host_entries,
    sub_hashes,
)
from tests.offload.test_raiden_wb_blocks import _host_load_holder, _scheduler
from vllm_torchtpu.offload.raiden_connector import WriteBackPolicy


class ErasingStore(FakeKVCacheStore):
    """Fake store with Raiden's erase: drops unpinned host-resident entries
    and fires the eviction callback for them, like KVCacheStore::Evict."""

    def __init__(self, raise_on_erase=False):
        super().__init__()
        self.raise_on_erase = raise_on_erase

    def erase(self, block_hashes):
        self.calls.append(("erase", list(block_hashes)))
        if self.raise_on_erase:
            raise RuntimeError("erase broke")
        host = (FakeBlockStatus.HOST, FakeBlockStatus.HOST_AND_HBM)
        erased = []
        for h in block_hashes:
            entry = self.entries.get(h)
            if entry is None or entry.pin_count > 0 or entry.status not in host:
                continue
            del self.entries[h]
            erased.append(h)
        if erased and self.eviction_callback is not None:
            self.eviction_callback(erased)
        return len(erased)

    def erase_calls(self):
        return [c for c in self.calls if isinstance(c, tuple) and c[0] == "erase"]


def _load(manager, store, keys, req_id="r1", success=True, finish_first=False):
    manager.on_new_request(_ctx(req_id))
    manager.prepare_load(keys, _ctx(req_id))
    manager.submit_load_job(3, keys, list(range(10, 10 + len(keys))), req_id)
    if finish_first:
        manager.on_request_finished(_ctx(req_id))
    batch = [h for k in keys for h in sub_hashes(k)]
    store.complete_load(batch, success=success)
    (job,) = manager.poll_finished_jobs()
    return job, batch


def _manager(store, exclusive=True):
    m = make_manager(store)
    if exclusive:
        assert m.enable_exclusive() is callable(getattr(store, "erase", None))
    return m


def test_policy_exclusive_parsing():
    p = WriteBackPolicy.from_extra_config(
        {"store_policy": {"mode": "write_back", "granularity": "block", "exclusive": True}}
    )
    assert p.exclusive and p.holds_all_chunks and not p.hold_all_chunks
    assert not WriteBackPolicy(granularity="block").holds_all_chunks
    with pytest.raises(ValueError):
        WriteBackPolicy.from_extra_config(
            {"store_policy": {"mode": "write_back", "exclusive": True}}
        )


def test_landed_load_erases_its_host_copy():
    store = ErasingStore()
    m = _manager(store)
    seed_host_entries(store, _key(0))
    seed_host_entries(store, _key(1))
    job, batch = _load(m, store, [_key(0), _key(1)])
    assert job.success
    assert store.erase_calls() == [("erase", batch)]
    assert not any(h in store.entries for h in batch)
    # The erase is reported like an eviction (the host copy is gone).
    events = m.take_events()
    assert len(events) == 1 and set(events[0].keys) == {_key(0), _key(1)}
    assert events[0].removed


def test_erase_skips_entries_pinned_by_another_request():
    store = ErasingStore()
    m = _manager(store)
    seed_host_entries(store, _key(0))
    # Another request's lookup holds a pin on the first sub-block.
    store.entries[sub_hashes(_key(0))[0]].pin_count = 1
    m.on_new_request(_ctx("r1"))
    m.prepare_load([_key(0)], _ctx("r1"))
    m.submit_load_job(3, [_key(0)], [10], "r1")
    store.complete_load(sub_hashes(_key(0)))
    (job,) = m.poll_finished_jobs()
    assert job.success
    first, *rest = sub_hashes(_key(0))
    assert first in store.entries
    assert not any(h in store.entries for h in rest)


def test_failed_load_keeps_host_copy():
    store = ErasingStore()
    m = _manager(store)
    seed_host_entries(store, _key(0))
    job, batch = _load(m, store, [_key(0)], success=False)
    assert not job.success
    assert store.erase_calls() == []
    assert all(store.entries[h].status == FakeBlockStatus.HOST for h in batch)


def test_load_of_finished_request_keeps_host_copy():
    # An aborted request's destination blocks may never be cached in HBM.
    store = ErasingStore()
    m = _manager(store)
    seed_host_entries(store, _key(0))
    job, batch = _load(m, store, [_key(0)], finish_first=True)
    assert job.success
    assert store.erase_calls() == []
    assert all(h in store.entries for h in batch)


def test_inclusive_by_default():
    store = ErasingStore()
    m = _manager(store, exclusive=False)
    seed_host_entries(store, _key(0))
    job, batch = _load(m, store, [_key(0)])
    assert job.success
    assert store.erase_calls() == []
    assert all(h in store.entries for h in batch)


def test_old_wheel_without_erase_stays_inclusive():
    store = FakeKVCacheStore()
    assert not hasattr(store, "erase")
    m = _manager(store)
    assert m._exclusive is False
    seed_host_entries(store, _key(0))
    job, batch = _load(m, store, [_key(0)])
    assert job.success
    assert all(h in store.entries for h in batch)


def test_broken_erase_turns_exclusive_off_and_load_still_succeeds():
    store = ErasingStore(raise_on_erase=True)
    m = _manager(store)
    seed_host_entries(store, _key(0))
    seed_host_entries(store, _key(1))
    job, _ = _load(m, store, [_key(0)])
    assert job.success
    assert m._exclusive is False
    job, _ = _load(m, store, [_key(1)], req_id="r2")
    assert job.success
    assert len(store.erase_calls()) == 1


def test_erase_stats_counted():
    store = ErasingStore()
    m = _manager(store)
    seed_host_entries(store, _key(0))
    _load(m, store, [_key(0)])
    assert m._stats["exclusive_erase_calls"] == 1
    assert m._stats["exclusive_erased_entries"] == 4


def test_exclusive_holds_loaded_and_hbm_hit_chunks():
    # The loaded chunk must be re-stored when HBM evicts it: exclusive holds
    # every referenced chunk, like hold_all_chunks.
    s, _pool = _scheduler(window_blocks=2)
    s._write_back = WriteBackPolicy(window_blocks=2, granularity="block", exclusive=True)
    from tests.offload.test_raiden_wb_blocks import _held_unregistered

    _held_unregistered(s, "r", [11, 12, 13, 14], next_stored=3)
    assert s._wb_register("r")
    assert s._wb_blocks["r"] == {11, 12, 13, 14}
    # Without exclusive / hold_all only the computed chunk is held.
    s2, _ = _scheduler(window_blocks=2)
    _host_load_holder(s2, hold_all=False)
    assert s2._wb_blocks["r"] == {14}
