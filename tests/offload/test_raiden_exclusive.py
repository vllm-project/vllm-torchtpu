# SPDX-License-Identifier: Apache-2.0
"""Exclusive host tier (store_policy exclusive): once a request whose load
landed is scheduled, its host copy is erased through Raiden's
KVCacheStore.erase, so the host holds only blocks HBM does not. Uses the fake
store of test_raiden_store_offload."""

import collections
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.v1.kv_offload.base import LookupResult, make_offload_key

from tests.offload.test_raiden_store_offload import (
    FakeBlockStatus,
    FakeKVCacheStore,
    _ctx,
    _group_key,
    _key,
    make_manager,
    seed_host_entries,
    sub_hashes,
)
from tests.offload.test_raiden_lru_refresh import _bare_scheduler, _group
from tests.offload.test_raiden_wb_blocks import (
    _held,
    _held_unregistered,
    _host_load_holder,
    _late_scheduler,
    _scheduler,
    _step,
)
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
    """A load of `keys` for `req_id` that lands (poll), not yet scheduled."""
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
    # Exclusive no longer implies hold_all_chunks: it holds erased chunks only.
    assert p.exclusive and not p.holds_all_chunks and not p.hold_all_chunks
    assert p.holds_erased_chunks
    both = WriteBackPolicy(granularity="block", exclusive=True, hold_all_chunks=True)
    assert both.holds_all_chunks and not both.holds_erased_chunks
    assert not WriteBackPolicy(granularity="block").holds_all_chunks
    assert not WriteBackPolicy(granularity="block").holds_erased_chunks
    with pytest.raises(ValueError):
        WriteBackPolicy.from_extra_config(
            {"store_policy": {"mode": "write_back", "exclusive": True}}
        )


# -- 1. deferred erase ------------------------------------------------------


def test_landed_load_erases_its_host_copy_only_once_scheduled():
    store = ErasingStore()
    m = _manager(store)
    seed_host_entries(store, _key(0))
    seed_host_entries(store, _key(1))
    job, batch = _load(m, store, [_key(0), _key(1)])
    assert job.success
    # Landed but not scheduled: the host copy stays (HBM cannot serve it yet).
    assert store.erase_calls() == []
    assert all(h in store.entries for h in batch)
    assert m.take_events() == []
    m.release_pending_erase(["r1"])
    assert store.erase_calls() == [("erase", batch)]
    assert not any(h in store.entries for h in batch)
    # The erase is reported like an eviction (the host copy is gone).
    events = m.take_events()
    assert len(events) == 1 and set(events[0].keys) == {_key(0), _key(1)}
    assert events[0].removed
    # Released once: a second schedule of the same request erases nothing.
    m.release_pending_erase(["r1"])
    assert len(store.erase_calls()) == 1


def test_sibling_retrying_in_the_load_window_still_hits():
    """The race of the j-69a31a45 loss: a sibling deferred on the loading
    chunks retries the step the load lands, before the loader is scheduled.
    It must still find the prefix on the host (and loads it, pinning it)."""
    store = ErasingStore()
    m = _manager(store)
    seed_host_entries(store, _key(0))
    seed_host_entries(store, _key(1))
    keys = [_key(0), _key(1)]
    _load(m, store, keys, req_id="lead")
    m.on_new_request(_ctx("sib"))
    assert [m.lookup(k, _ctx("sib")) for k in keys] == [LookupResult.HIT] * 2
    # The sibling commits to a load; the leader is scheduled meanwhile. Erase
    # skips the entries the sibling's load pins.
    m.prepare_load(keys, _ctx("sib"))
    m.submit_load_job(4, keys, [20, 21], "sib")
    m.release_pending_erase(["lead"])
    batch = [h for k in keys for h in sub_hashes(k)]
    assert all(h in store.entries for h in batch)
    store.complete_load(batch)
    (job,) = m.poll_finished_jobs()
    assert job.success
    # The sibling is scheduled in turn: now the host copy goes.
    m.release_pending_erase(["sib"])
    assert not any(h in store.entries for h in batch)


def test_erase_skips_entries_pinned_by_another_request():
    store = ErasingStore()
    m = _manager(store)
    seed_host_entries(store, _key(0))
    m.on_new_request(_ctx("r1"))
    m.prepare_load([_key(0)], _ctx("r1"))
    m.submit_load_job(3, [_key(0)], [10], "r1")
    store.complete_load(sub_hashes(_key(0)))
    (job,) = m.poll_finished_jobs()
    assert job.success
    # Another request's lookup holds a pin on the first sub-block.
    store.entries[sub_hashes(_key(0))[0]].pin_count = 1
    m.release_pending_erase(["r1"])
    first, *rest = sub_hashes(_key(0))
    assert first in store.entries
    assert not any(h in store.entries for h in rest)


def test_failed_load_keeps_host_copy():
    store = ErasingStore()
    m = _manager(store)
    seed_host_entries(store, _key(0))
    job, batch = _load(m, store, [_key(0)], success=False)
    assert not job.success
    m.release_pending_erase(["r1"])
    assert store.erase_calls() == []
    assert all(store.entries[h].status == FakeBlockStatus.HOST for h in batch)


def test_load_of_finished_request_keeps_host_copy():
    # An aborted request's destination blocks may never be cached in HBM.
    store = ErasingStore()
    m = _manager(store)
    seed_host_entries(store, _key(0))
    job, batch = _load(m, store, [_key(0)], finish_first=True)
    assert job.success
    m.release_pending_erase(["r1"])
    assert store.erase_calls() == []
    assert all(h in store.entries for h in batch)


def test_abort_after_landing_before_scheduling_keeps_host_copy():
    store = ErasingStore()
    m = _manager(store)
    seed_host_entries(store, _key(0))
    _, batch = _load(m, store, [_key(0)])
    m.on_request_finished(_ctx("r1"))
    m.release_pending_erase(["r1"])
    assert store.erase_calls() == []
    assert all(h in store.entries for h in batch)
    assert m._stats["exclusive_erase_dropped"] == 1


def _excl_scheduler(store):
    """A bare connector scheduler on a real manager with exclusive on."""
    s = _bare_scheduler([_group(0)])
    s._raiden_manager = _manager(store)
    s._write_back = WriteBackPolicy(granularity="block", exclusive=True)
    return s


def _sched_out(new=(), preempted=None, resumed=()):
    return SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id=r) for r in new],
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set(resumed)),
        preempted_req_ids=preempted,
    )


def test_connector_erases_when_the_request_is_scheduled():
    store = ErasingStore()
    s = _excl_scheduler(store)
    m = s._raiden_manager
    for n in range(3):
        seed_host_entries(store, _key(n))
    _load(m, store, [_key(0)], req_id="a")
    _load(m, store, [_key(1)], req_id="b")
    _load(m, store, [_key(2)], req_id="c")
    # A step that schedules other requests erases nothing.
    s._release_scheduled_erases(_sched_out(new=["x"]))
    assert store.erase_calls() == []
    # a is scheduled new, c resumed; b stays waiting.
    s._release_scheduled_erases(_sched_out(new=["a"], resumed=["c"]))
    erased = {h for _, hs in store.erase_calls() for h in hs}
    assert erased == set(sub_hashes(_key(0))) | set(sub_hashes(_key(2)))
    assert all(h in store.entries for h in sub_hashes(_key(1)))
    assert set(m._pending_erase) == {"b"}


def test_connector_preempted_request_keeps_host_copy():
    store = ErasingStore()
    s = _excl_scheduler(store)
    m = s._raiden_manager
    seed_host_entries(store, _key(0))
    seed_host_entries(store, _key(1))
    _load(m, store, [_key(0)], req_id="a")
    _load(m, store, [_key(1)], req_id="b")
    # a is preempted before its erase ran (even if listed as scheduled);
    # b is preempted while still waiting.
    s._release_scheduled_erases(_sched_out(new=["a"], preempted={"a", "b"}))
    assert store.erase_calls() == []
    assert not m._pending_erase
    # A later schedule no longer erases: the load is not trusted any more.
    s._release_scheduled_erases(_sched_out(new=["a", "b"]))
    assert store.erase_calls() == []


def test_connector_request_finished_drops_pending_erase():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    store = ErasingStore()
    s = _excl_scheduler(store)
    s._lru_refresh_on_finish = False
    m = s._raiden_manager
    seed_host_entries(store, _key(0))
    _load(m, store, [_key(0)], req_id="a")
    with patch.object(
        OffloadingConnectorScheduler, "request_finished", return_value=(False, None)
    ):
        s.request_finished(SimpleNamespace(request_id="a"))
    s._release_scheduled_erases(_sched_out(new=["a"]))
    assert store.erase_calls() == []
    assert all(h in store.entries for h in sub_hashes(_key(0)))


def test_inclusive_by_default():
    store = ErasingStore()
    m = _manager(store, exclusive=False)
    seed_host_entries(store, _key(0))
    job, batch = _load(m, store, [_key(0)])
    assert job.success
    m.release_pending_erase(["r1"])
    assert store.erase_calls() == []
    assert all(h in store.entries for h in batch)
    assert m._pending_erase is None and m._erased_keys is None
    assert m.take_erased_block_hashes() == set()
    assert not m.is_erased(_key(0)) and m.erased_age(_key(0)) == -1.0


def test_inclusive_connector_never_touches_the_store_on_schedule():
    store = ErasingStore()
    s = _excl_scheduler(store)
    s._write_back = WriteBackPolicy(granularity="block")
    s._raiden_manager = _manager(store, exclusive=False)
    seed_host_entries(store, _key(0))
    _load(s._raiden_manager, store, [_key(0)], req_id="a")
    with patch.object(
        type(s), "_wb_build_partial", side_effect=RuntimeError("stop")
    ), patch.object(
        type(s), "_write_back_output_blocks", side_effect=lambda o: o
    ):
        with pytest.raises(RuntimeError):
            s.build_connector_meta(_sched_out(new=["a"]))
    assert store.erase_calls() == []


def test_old_wheel_without_erase_stays_inclusive():
    store = FakeKVCacheStore()
    assert not hasattr(store, "erase")
    m = _manager(store)
    assert m._exclusive is False
    seed_host_entries(store, _key(0))
    job, batch = _load(m, store, [_key(0)])
    assert job.success
    m.release_pending_erase(["r1"])
    assert all(h in store.entries for h in batch)


def test_broken_erase_turns_exclusive_off_and_load_still_succeeds():
    store = ErasingStore(raise_on_erase=True)
    m = _manager(store)
    seed_host_entries(store, _key(0))
    seed_host_entries(store, _key(1))
    job, _ = _load(m, store, [_key(0)])
    assert job.success
    m.release_pending_erase(["r1"])
    assert m._exclusive is False
    job, _ = _load(m, store, [_key(1)], req_id="r2")
    assert job.success
    m.release_pending_erase(["r2"])
    assert len(store.erase_calls()) == 1


def test_erase_stats_counted():
    store = ErasingStore()
    m = _manager(store)
    seed_host_entries(store, _key(0))
    _load(m, store, [_key(0)])
    assert m._stats["exclusive_erase_deferred"] == 1
    assert m._stats["exclusive_erase_calls"] == 0
    m.release_pending_erase(["r1"])
    assert m._stats["exclusive_erase_calls"] == 1
    assert m._stats["exclusive_erased_entries"] == 4


# -- 2. erased keys: tracking, re-store, warmth invalidation ---------------


def test_erased_keys_tracked_until_stored_again():
    store = ErasingStore()
    m = _manager(store)
    key = _group_key(0, 0)
    seed_host_entries(store, key)
    _load(m, store, [key])
    assert not m.is_erased(key)
    m.release_pending_erase(["r1"])
    assert m.is_erased(key)
    assert 0.0 <= m.erased_age(key) < 2.0
    assert m.take_erased_block_hashes() == {b"hash_0000"}
    assert m.take_erased_block_hashes() == set()
    # HBM evicts it: the write-back store puts it back, and it is no longer
    # "erased".
    out = m.prepare_store([key], _ctx("w"))
    assert out.keys_to_store == [key]
    m.submit_store_job(9, [key], [5])
    store.complete_save(sub_hashes(key))
    m.poll_finished_jobs()
    assert not m.is_erased(key) and m.erased_age(key) == -1.0


def test_erased_block_hashes_overflow_drops_everything():
    store = ErasingStore()
    m = _manager(store)
    m._ERASED_BLOCK_HASHES_CAP = 1
    for n in range(2):
        seed_host_entries(store, _group_key(n, 0))
    _load(m, store, [_group_key(0, 0), _group_key(1, 0)])
    m.release_pending_erase(["r1"])
    assert m.take_erased_block_hashes() is None
    assert m.take_erased_block_hashes() == set()


def _paw(requests, cache, erased, hash_bs=16):
    from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler

    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    probes = []

    class Probe:
        def peek_external_prefix_tokens(self, request, local):
            probes.append(request.request_id)
            return 0

        def take_erased_block_hashes(self):
            if erased is None:
                return None
            out = set(erased)
            erased[:] = []
            return out

    s._paw_probe = Probe()
    s._paw_external_cache = dict(cache)
    s._paw_step = 1
    s.requests = {r.request_id: r for r in requests}
    s.block_size = hash_bs
    s.hash_block_size = hash_bs
    return s, probes


def test_paw_warmth_cache_drops_entries_covering_erased_blocks():
    hashes = [b"b%d" % i for i in range(8)]
    sib = SimpleNamespace(request_id="sib", block_hashes=hashes, num_tokens=128)
    other = SimpleNamespace(
        request_id="other", block_hashes=[b"o%d" % i for i in range(8)], num_tokens=128
    )
    cold = SimpleNamespace(request_id="cold", block_hashes=hashes, num_tokens=128)
    cache = {
        # local 32 tokens (2 blocks), host warmth 64 tokens: blocks 2..6.
        "sib": (32, 8, 64, 1),
        "other": (0, 8, 64, 1),
        "cold": (0, 8, 0, 1),  # no host warmth: nothing to drop
    }
    erased = [b"b3"]
    s, probes = _paw([sib, other, cold], cache, erased)
    s._drop_erased_external_warmth()
    assert set(s._paw_external_cache) == {"other", "cold"}
    # The sibling's warmth (sibling hold's last check and admission) is
    # probed again instead of trusted for up to 8 steps.
    assert s._external_warmth(sib, 32) == 0
    assert probes == ["sib"]
    # Erased hashes outside the cached range keep the entry.
    s2, _ = _paw([sib], {"sib": (32, 8, 16, 1)}, [b"b7"])
    s2._drop_erased_external_warmth()
    assert "sib" in s2._paw_external_cache


def test_paw_warmth_cache_overflow_and_inclusive():
    sib = SimpleNamespace(request_id="sib", block_hashes=[b"x"], num_tokens=16)
    s, _ = _paw([sib], {"sib": (0, 1, 16, 1)}, None)
    s._drop_erased_external_warmth()
    assert s._paw_external_cache == {}
    # Nothing erased (inclusive tier): the cache is untouched.
    s2, _ = _paw([sib], {"sib": (0, 1, 16, 1)}, [])
    s2._drop_erased_external_warmth()
    assert s2._paw_external_cache == {"sib": (0, 1, 16, 1)}
    # A probe without the hook (another connector): untouched.
    s2._paw_probe = SimpleNamespace(peek_external_prefix_tokens=lambda r, n: 0)
    s2._drop_erased_external_warmth()
    assert "sib" in s2._paw_external_cache


def test_facade_take_erased_block_hashes():
    from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector

    c = TPURaidenOffloadingConnector.__new__(TPURaidenOffloadingConnector)
    c.connector_scheduler = None
    assert c.take_erased_block_hashes() == set()
    store = ErasingStore()
    c.connector_scheduler = _excl_scheduler(store)
    m = c.connector_scheduler._raiden_manager
    seed_host_entries(store, _group_key(0, 0))
    _load(m, store, [_group_key(0, 0)])
    m.release_pending_erase(["r1"])
    assert c.take_erased_block_hashes() == {b"hash_0000"}


# -- 3. holders: exclusive holds erased chunks, independent of hold_all -----


class _ErasedKeys:
    """Manager stand-in: the exclusive tier's erased keys."""

    def __init__(self, *keys):
        self._erased_keys = collections.OrderedDict((k, 0.0) for k in keys)

    def is_erased(self, key):
        return key in self._erased_keys


def test_exclusive_holds_erased_chunks_only():
    # Chunks 0-1 hit in HBM, chunk 2 was loaded (its host copy erased), chunk
    # 3 computed. Exclusive holds the erased chunk and the computed one, not
    # the HBM hits whose host copy is still there.
    s, _pool = _scheduler(window_blocks=2)
    s._write_back = WriteBackPolicy(window_blocks=2, granularity="block", exclusive=True)
    keys, _ = _held_unregistered(s, "r", [11, 12, 13, 14], next_stored=3)
    s._raiden_manager = _ErasedKeys(keys[2])
    assert s._wb_register("r")
    assert s._wb_blocks["r"] == {13, 14}
    # An HBM hit whose host copy another request's load erased is held too.
    s1, _ = _scheduler(window_blocks=2)
    s1._write_back = s._write_back
    keys1, _ = _held_unregistered(s1, "r", [11, 12, 13, 14], next_stored=3)
    s1._raiden_manager = _ErasedKeys(keys1[0], keys1[2])
    assert s1._wb_register("r")
    assert s1._wb_blocks["r"] == {11, 13, 14}
    # Nothing erased: vLLM's next_stored_chunk_idx rule alone.
    s2, _ = _scheduler(window_blocks=2)
    s2._write_back = s._write_back
    _held_unregistered(s2, "r", [11, 12, 13, 14], next_stored=3)
    s2._raiden_manager = _ErasedKeys()
    assert s2._wb_register("r")
    assert s2._wb_blocks["r"] == {14}
    # Without exclusive only the computed chunk is held.
    s3, _ = _scheduler(window_blocks=2)
    _host_load_holder(s3, hold_all=False)
    assert s3._wb_blocks["r"] == {14}


def test_hold_all_stays_an_independent_flag():
    s, _ = _scheduler(window_blocks=2)
    s._write_back = WriteBackPolicy(
        window_blocks=2, granularity="block", exclusive=True, hold_all_chunks=True
    )
    keys, _ = _held_unregistered(s, "r", [11, 12, 13, 14], next_stored=3)
    s._raiden_manager = _ErasedKeys()
    assert s._wb_register("r")
    assert s._wb_blocks["r"] == {11, 12, 13, 14}


def test_exclusive_partial_store_offers_erased_chunk_below_next_stored():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    s, pool = _scheduler(window_blocks=16)
    s._write_back = WriteBackPolicy(window_blocks=16, granularity="block", exclusive=True)
    keys, gs = _held_unregistered(s, "r", [11, 12, 13, 14], next_stored=3)
    s._raiden_manager = _ErasedKeys(keys[2])
    assert s._wb_register("r")
    # Block 13 (the erased, loaded chunk) reaches the window.
    s._wb_owners = {13: {"r"}}
    s._wb_blocks["r"] = {13}
    out = _step(s)
    assert s._wb_allowed == {"r": {keys[2]}}
    seen = []
    s.manager.prepare_store = lambda ks, ctx: seen.append(list(ks)) or "ok"

    def fake_base(self, scheduler_output):
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
    assert seen == [[keys[2]]]
    assert gs.next_stored_chunk_idx == 3  # restored


# -- 4. late_store covers async-load destinations --------------------------


def _late_excl_scheduler(exclusive):
    s, pool = _late_scheduler(window_blocks=1)
    s._write_back = WriteBackPolicy(
        window_blocks=1, granularity="block", late_store=True, exclusive=exclusive
    )
    s._raiden_manager = _ErasedKeys()
    return s, pool


def test_late_store_covers_async_load_destination_blocks():
    # A held block popped as an async load's destination: fresh this step,
    # but its request is not in the scheduler output until the load lands.
    s, pool = _late_excl_scheduler(exclusive=True)
    keys, _ = _held(s, "r", [5, 6, 7])
    _step(s)
    s._fresh_block_ids.update({6})
    _step(s, allocated=())
    assert s._wb_allowed == {"r": {keys[1]}}
    assert s._wb_urgent == {"r"}
    assert s._wb_blocks["r"] == {5, 7}
    # When the loading request is scheduled later, block 6 is no longer held:
    # nothing is dropped or stored twice.
    s._fresh_block_ids.clear()
    _step(s, allocated={6})
    assert s._wb_allowed == {} and s._wb_blocks["r"] == {5, 7}


def test_late_store_async_load_gap_unchanged_when_inclusive():
    s, pool = _late_excl_scheduler(exclusive=False)
    keys, _ = _held(s, "r", [5, 6, 7])
    _step(s)
    s._fresh_block_ids.update({6})
    _step(s, allocated=())
    assert s._wb_allowed == {} and s._wb_urgent == set()
    assert s._wb_blocks["r"] == {5, 6, 7}


# -- diagnostics: a miss on a key erased < 2 s ago --------------------------


def _miss_request(n_chunks=80, prompt=80 * 128):
    return SimpleNamespace(
        request_id="q",
        num_prompt_tokens=prompt,
        block_hashes=[b"blk%04d" % i for i in range(n_chunks)],
    )


def test_stats_count_miss_on_recently_erased_key():
    store = ErasingStore()
    s = _excl_scheduler(store)
    m = s._raiden_manager
    req = _miss_request()
    # Local 2 chunks, host 0: the first missing key is chunk 2's.
    first_missing = make_offload_key(req.block_hashes[2], 0)
    m._erased_keys[first_missing] = time.monotonic() - 0.5
    s._count_recent_erase_miss(req, 256, 0)
    assert s.wb_counters["miss_erased_2s"] == 1
    assert s.wb_counters["miss_erased_2s_tokens"] == req.num_prompt_tokens - 256
    # Once per request.
    s._count_recent_erase_miss(req, 256, 0)
    assert s.wb_counters["miss_erased_2s"] == 1
    # Erased long ago, or never: not counted.
    s2 = _excl_scheduler(store)
    s2._raiden_manager._erased_keys[first_missing] = time.monotonic() - 5.0
    s2._count_recent_erase_miss(req, 256, 0)
    s2._erase_miss_req_ids = None
    s2._raiden_manager._erased_keys.clear()
    s2._raiden_manager._erased_keys[b"other"] = time.monotonic()
    s2._count_recent_erase_miss(req, 256, 0)
    assert s2.wb_counters["miss_erased_2s"] == 0


def test_get_num_new_matched_tokens_counts_recent_erase_miss_with_stats():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    store = ErasingStore()
    req = _miss_request()
    for exclusive, stats, expect in ((True, True, 1), (True, False, 0), (False, True, 0)):
        s = _excl_scheduler(store)
        s._write_back = WriteBackPolicy(granularity="block", exclusive=exclusive)
        s._offload_stats = stats
        s._loss_trace = False
        s._count_stale_head = lambda *a: None
        s._raiden_manager._erased_keys[make_offload_key(req.block_hashes[2], 0)] = (
            time.monotonic()
        )
        with patch.object(
            OffloadingConnectorScheduler,
            "get_num_new_matched_tokens",
            return_value=(0, True),
        ):
            s.get_num_new_matched_tokens(req, 256)
        assert s.wb_counters["miss_erased_2s"] == expect


def test_loss_trace_counts_miss_on_recently_erased_key():
    store = ErasingStore()
    s = _excl_scheduler(store)
    m = s._raiden_manager
    m.enable_loss_trace()
    s._traced_req_ids = set()
    req = _miss_request()
    m._trace([make_offload_key(req.block_hashes[2], 0)], "erased")
    s._trace_miss(req, 256, 0)
    assert s.wb_counters["miss8k_erased"] == 1
    assert s.wb_counters["miss8k_erased_lt2s"] == 1
    assert s.wb_counters["miss8k_tok_erased_lt2s"] == req.num_prompt_tokens - 256
