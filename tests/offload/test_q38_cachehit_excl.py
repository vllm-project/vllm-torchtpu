# SPDX-License-Identifier: Apache-2.0
"""q38 CACHEHIT-EXCL: host-tier exclusive restore (VLLM_TPU_OFFLOAD_TIER_EXCLUSIVE), on CPU.

The victim policy keeps a block's host copy when a load restores it ("keep on load"), so
while HBM holds the block the host tier holds a duplicate. Exclusive restore evicts that
host copy once the load lands, through KVCacheStore.evict (the W1e wheel; the fake store
below follows its proven host semantics: only unpinned HOST / HOST_AND_HBM entries go,
one eviction callback per call, before evict returns). HBM's later eviction stores the
block again through the victim path.

Rows (tag in each assertion message):
- EXCL-OFF   flag off: no evict call, the host copy stays, a later victim store skips it.
- EXCL-ON    a landed local load evicts exactly its local sub-hashes, counted as discards
             (not LRU evictions); the key is then re-offered to the victim store.
- EXCL-FAIL  a failed load evicts nothing.
- EXCL-PIN   another request's pin keeps the entry (evict asks, the store declines).
- EXCL-PEER  peer reads (read_remote) are never evicted; only the local part of a job is.
- EXCL-REF   a store without evict is refused; so is the flag without the victim policy.
- EXCL-CLS   tier classes: no class evicts before the whole parent load lands; a failed
             sibling keeps every class's host copy; telemetry shows discards per class.
- EXCL-CONN  the connector's install line (exact text) and its telemetry token; flag off
             prints neither.
- EXCL-CYCLE served groups, victim policy: after an exclusive restore the next victim
             batch stores the blocks again; flag off skips them as resident.
- EXCL-ENV   the key is registered (default off) and compile-ignored.
"""

from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.kv_offload.base import ReqContext, make_offload_key

import tests.offload.test_tier_classes as ttc
from tests.offload.test_raiden_store_offload import (
    FakeBlockStatus,
    FakeKVCacheStore,
    _Call,
    _Entry,
    make_global_manager,
    make_manager,
    seed_global_entries,
    seed_host_entries,
    sub_hashes,
)

EXCL_ENV = "VLLM_TPU_OFFLOAD_TIER_EXCLUSIVE"
INSTALL_LINE = (
    "TPURaidenOffloadingConnector: host-tier exclusive restore on (a landed load's host "
    "copy is evicted with KVCacheStore.evict; HBM eviction stores it again); %s"
)


def _ctx(req_id="req"):
    return ReqContext(req_id=req_id)


def _key(n):
    return b"hash_%04d" % n


class _EvictStore(FakeKVCacheStore):
    """FakeKVCacheStore plus KVCacheStore.evict as W1e behaves (proof_rec1 j-9d2662dd P2-E)."""

    def evict(self, block_hashes) -> int:
        self.calls.append(_Call("evict", (list(block_hashes),)))
        gone = []
        for h in block_hashes:
            entry = self.entries.get(h)
            if (
                entry is None
                or entry.is_candidate
                or entry.pin_count > 0
                or entry.status not in (FakeBlockStatus.HOST, FakeBlockStatus.HOST_AND_HBM)
            ):
                continue
            del self.entries[h]
            gone.append(h)
        if gone and self.eviction_callback is not None:
            self.eviction_callback(gone)
        return len(gone)


def _evicted(store):
    return [h for c in store.calls_named("evict") for h in c.args[0]]


def _load(manager, store, keys, job_id=7, req="r1", success=True, factor=4):
    spec = manager.prepare_load(keys, _ctx(req))
    assert spec.pinned
    manager.submit_load_job(job_id, spec.keys, list(range(len(keys))), req)
    store.complete_load([h for k in keys for h in sub_hashes(k, factor)], success=success)
    return manager.poll_finished_jobs()


def _stored_key(manager, store, key, job_id=1):
    """A key the tier stored itself (so the held-key telemetry tracks it)."""
    manager.submit_store_job(job_id, [key], [3])
    store.complete_save(sub_hashes(key))
    (done,) = manager.poll_finished_jobs()
    assert done.success
    for h in sub_hashes(key):
        store.entries[h].status = FakeBlockStatus.HOST


# -- single store ------------------------------------------------------------------


def test_flag_off_never_evicts_and_keeps_on_load():
    store = _EvictStore()
    manager = make_manager(store)
    seed_host_entries(store, _key(1))
    (done,) = _load(manager, store, [_key(1)])
    assert done.success
    assert store.calls_named("evict") == [], "EXCL-OFF evict called with the flag off"
    assert all(h in store.entries for h in sub_hashes(_key(1))), "EXCL-OFF host copy gone"
    assert manager.prepare_victim_store([_key(1)]) == [], "EXCL-OFF resident key re-offered"


def test_landed_load_evicts_its_local_host_copy_as_a_discard():
    store = _EvictStore()
    manager = make_manager(store, track_held=True)
    manager.set_exclusive_restore(True)
    _stored_key(manager, store, _key(1))
    seed_host_entries(store, _key(2))  # not tracked as held: a pre-existing entry
    held0, stored0, evicted0, _ = manager.held_stats()
    assert (held0, stored0, evicted0) == (1, 1, 0)
    (done,) = _load(manager, store, [_key(1), _key(2)], job_id=9)
    assert done.success
    want = sub_hashes(_key(1)) + sub_hashes(_key(2))
    assert store.calls_named("evict") == [_Call("evict", (want,))], "EXCL-ON evict args"
    assert not any(h in store.entries for h in want), "EXCL-ON host copy kept"
    stats = manager.excl_stats()
    assert stats == {
        "calls": 1,
        "asked": 8,
        "evicted": 8,
        "abandoned": 0,
        "discarded": 1,
    }, f"EXCL-ON counters {stats}"
    held, stored, evicted, _ = manager.held_stats()
    assert (held, stored, evicted) == (0, 1, 0), "EXCL-ON discard counted as an LRU eviction"
    # An LRU eviction of another held key is still an eviction, not a discard.
    _stored_key(manager, store, _key(3), job_id=10)
    store.simulate_evict(sub_hashes(_key(3)))
    assert manager.held_stats()[2] == 1 and manager.excl_stats()["discarded"] == 1, (
        "EXCL-ON LRU eviction counted as a discard"
    )
    # The restored key is no longer resident, so HBM's eviction stores it again.
    assert manager.prepare_victim_store([_key(1)]) == [_key(1)], "EXCL-ON not re-offered"
    # The callback queued removal events for both keys.
    (event,) = manager.take_events()
    assert set(event.keys) == {_key(1), _key(2), _key(3)} and event.removed


def test_failed_load_evicts_nothing():
    store = _EvictStore()
    manager = make_manager(store)
    manager.set_exclusive_restore(True)
    seed_host_entries(store, _key(1))
    (done,) = _load(manager, store, [_key(1)], success=False)
    assert not done.success
    assert store.calls_named("evict") == [], "EXCL-FAIL evict on a failed load"
    assert all(h in store.entries for h in sub_hashes(_key(1))), "EXCL-FAIL host copy gone"
    assert manager.excl_stats()["calls"] == 0, "EXCL-FAIL counted"


def test_another_requests_pin_keeps_the_entry():
    store = _EvictStore()
    manager = make_manager(store)
    manager.set_exclusive_restore(True)
    seed_host_entries(store, _key(1))
    spec = manager.prepare_load([_key(1)], _ctx("r1"))
    manager.submit_load_job(7, spec.keys, [0], "r1")
    second = manager.prepare_load([_key(1)], _ctx("r2"))  # r2 pins the same entries
    assert second.pinned
    store.complete_load(sub_hashes(_key(1)))
    (done,) = manager.poll_finished_jobs()
    assert done.success
    assert _evicted(store) == sub_hashes(_key(1)), "EXCL-PIN evict not asked"
    assert all(store.entries[h].pin_count == 1 for h in sub_hashes(_key(1))), (
        "EXCL-PIN r2's pin lost"
    )
    stats = manager.excl_stats()
    assert (stats["asked"], stats["evicted"]) == (4, 0), f"EXCL-PIN counters {stats}"


def test_peer_reads_are_never_evicted():
    store = _EvictStore()
    manager = make_global_manager(store)
    manager.set_exclusive_restore(True)
    local_key, remote_key = _key(1), _key(2)
    seed_host_entries(store, local_key)
    seed_global_entries(store, remote_key)
    assert manager.lookup(remote_key, _ctx()) is not None
    spec = manager.prepare_load([local_key, remote_key], _ctx())
    assert spec.pinned
    manager.submit_load_job(7, [local_key, remote_key], [5, 6], "req")
    store.complete_load(sub_hashes(local_key))
    store.complete_remote_read(sub_hashes(remote_key))
    (done,) = manager.poll_finished_jobs()
    assert done.success
    assert _evicted(store) == sub_hashes(local_key), "EXCL-PEER peer sub-hashes evicted"


def test_a_store_without_evict_is_refused():
    manager = make_manager(FakeKVCacheStore())
    with pytest.raises(ValueError, match=r"needs a Raiden build whose KVCacheStore has evict\(\)"):
        manager.set_exclusive_restore(True)
    manager.set_exclusive_restore(False)  # off needs nothing from the store


# -- tier classes --------------------------------------------------------------------


def _excl_tier(monkeypatch, on=True):
    monkeypatch.setattr(ttc, "FakeKVCacheStore", _EvictStore)
    tier, stores = ttc._tier()
    if on:
        tier.set_exclusive_restore(True)
    return tier, stores


def _three_class_load(tier, stores, job_id=6):
    keys = [ttc._key(1, ttc.MAIN), ttc._key(1, ttc.KEYS), ttc._key(1, ttc.GDN_C)]
    for k, n in zip(keys, ("main", "keys", "gdn-c")):
        ttc._resident(stores[n], k)
    spec = tier.prepare_load(keys, _ctx("r1"))
    tier.submit_load_job(job_id, spec.keys, [21, 22, 23], "r1", pinned=spec.pinned)
    return keys


def test_classes_evict_only_after_the_whole_parent_lands(monkeypatch):
    tier, stores = _excl_tier(monkeypatch)
    keys = _three_class_load(tier, stores)
    stores["main"].complete_load([ttc._sub(keys[0])])
    stores["gdn-c"].complete_load([ttc._sub(keys[2])])
    assert tier.poll_finished_jobs() == []
    assert not any(s.calls_named("evict") for s in stores.values()), (
        "EXCL-CLS a class evicted before its parent ended"
    )
    stores["keys"].complete_load([ttc._sub(keys[1])])
    (done,) = tier.poll_finished_jobs()
    assert done.success
    got = {n: _evicted(s) for n, s in stores.items() if s.calls_named("evict")}
    assert got == {
        "main": [ttc._sub(keys[0])],
        "keys": [ttc._sub(keys[1])],
        "gdn-c": [ttc._sub(keys[2])],
    }, f"EXCL-CLS evicts {got}"
    for k, n in zip(keys, ("main", "keys", "gdn-c")):
        assert ttc._sub(k) not in stores[n].entries, f"EXCL-CLS {n} host copy kept"
    line = tier.telemetry_line()
    assert " discarded=0" in line, "EXCL-CLS telemetry lacks discarded="


def test_classes_failed_sibling_keeps_every_host_copy(monkeypatch):
    tier, stores = _excl_tier(monkeypatch)
    keys = _three_class_load(tier, stores)
    stores["main"].complete_load([ttc._sub(keys[0])])
    stores["gdn-c"].complete_load([ttc._sub(keys[2])])
    assert tier.poll_finished_jobs() == []
    stores["keys"].complete_load([ttc._sub(keys[1])], success=False)
    (done,) = tier.poll_finished_jobs()
    assert not done.success
    assert not any(s.calls_named("evict") for s in stores.values()), (
        "EXCL-CLS evicted under a failed parent"
    )
    assert ttc._sub(keys[0]) in stores["main"].entries, "EXCL-CLS main copy gone"
    assert ttc._sub(keys[2]) in stores["gdn-c"].entries, "EXCL-CLS gdn-c copy gone"
    assert sum(s["abandoned"] for s in tier.excl_stats()) == 2, "EXCL-CLS abandoned count"
    assert not any(m._excl_deferred for m in tier.managers), "EXCL-CLS deferred left over"


def test_classes_flag_off_prints_no_discards(monkeypatch):
    tier, stores = _excl_tier(monkeypatch, on=False)
    keys = _three_class_load(tier, stores)
    for k, n in zip(keys, ("main", "keys", "gdn-c")):
        stores[n].complete_load([ttc._sub(k)])
    (done,) = tier.poll_finished_jobs()
    assert done.success and not any(s.calls_named("evict") for s in stores.values())
    assert "discarded=" not in tier.telemetry_line(), "EXCL-OFF telemetry changed"


# -- the connector on the served groups ------------------------------------------------


def _sched(monkeypatch, on, victim_policy=True, store_cls=_EvictStore, auto=True):
    from vllm_torchtpu.offload import raiden_connector

    if on:
        monkeypatch.setenv(EXCL_ENV, "1")
    else:
        monkeypatch.delenv(EXCL_ENV, raising=False)
    log = MagicMock()
    monkeypatch.setattr(raiden_connector, "logger", log)

    base = ttc._AutoSave if auto else FakeKVCacheStore
    bases = (base,) if store_cls is FakeKVCacheStore else (store_cls, base)
    _Store = type("_Store", bases, {})  # evict (when store_cls has it) + auto-landing saves

    stores = [_Store() for _ in range(5)]
    sched, kvc, _ = ttc._tier_scheduler(monkeypatch, stores=stores, victim_policy=victim_policy)
    return sched, kvc, stores, log


def _install_lines(log):
    return [c for c in log.info.call_args_list if c.args and c.args[0] == INSTALL_LINE]


def test_connector_refuses_the_flag_without_the_victim_policy(monkeypatch):
    from vllm_torchtpu.offload import victim

    try:
        with pytest.raises(ValueError, match="needs QWEN4_EXP_HOST_TIER_POLICY=victim"):
            _sched(monkeypatch, on=True, victim_policy=False)
    finally:
        victim.deactivate()


def test_connector_refuses_a_wheel_without_evict(monkeypatch):
    from vllm_torchtpu.offload import victim

    try:
        with pytest.raises(ValueError, match=r"KVCacheStore has evict\(\)"):
            _sched(monkeypatch, on=True, store_cls=FakeKVCacheStore)
    finally:
        victim.deactivate()


@pytest.mark.parametrize("on", [True, False])
def test_connector_install_line_and_telemetry(monkeypatch, on):
    from vllm_torchtpu.offload import victim

    try:
        sched, kvc, stores, log = _sched(monkeypatch, on=on)
        lines = _install_lines(log)
        line = sched.telemetry_line()
        if on:
            assert len(lines) == 1, f"EXCL-CONN install lines {len(lines)}"
            assert lines[0].args[1] == "classes main,keys,gdn-ab,gdn-c,ple"
            assert line.endswith(
                " excl_restore=calls:0,asked:0,evicted:0,abandoned:0,discarded:0"
            ), "EXCL-CONN telemetry token"
            assert line.count(" discarded=0") == 5
        else:
            assert lines == [], "EXCL-CONN install line with the flag off"
            assert "excl_restore=" not in line and "discarded=" not in line
    finally:
        victim.deactivate()


def _turn_one_keys(first, kvc):
    keys = []
    for gid in kvc.prefix_cacheable_group_ids:
        if isinstance(kvc.kv_cache_groups[gid].kv_cache_spec, MambaSpec):
            keys.append(make_offload_key(bytes(first.block_hashes[2]), gid))
        else:
            keys.extend(make_offload_key(bytes(first.block_hashes[c]), gid) for c in range(3))
    return keys


@pytest.mark.parametrize("on", [True, False])
def test_victim_cycle_stores_restored_blocks_again_only_when_exclusive(monkeypatch, on):
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_torchtpu.offload import victim

    try:
        sched, kvc, stores, _ = _sched(monkeypatch, on=on)
        first, table, _ = ttc._turn_one(sched, kvc)
        evicted = ttc._victims_of_turn_one(sched, first, table, kvc)
        sched.build_connector_meta(SchedulerOutput.make_empty())
        n = len(evicted)
        assert sched.telemetry["victim_store_keys"] == n
        # A second request restores every key of turn one from the host tier.
        manager = sched._raiden_manager
        keys = _turn_one_keys(first, kvc)
        assert len(keys) == n
        spec = manager.prepare_load(keys, _ctx("r2"))
        assert spec.pinned
        manager.submit_load_job(901, spec.keys, list(range(100, 100 + n)), "r2")
        for store in stores:
            store.complete_load([h for c in store.calls_named("load") for h in c.args[0]])
        (done,) = manager.poll_finished_jobs()
        assert done.success
        held = sum(m.held_stats()[0] for m in manager.managers)
        # HBM evicts the same blocks again, long after.
        ttc._victims_of_turn_one(sched, first, table, kvc)
        sched.build_connector_meta(SchedulerOutput.make_empty())
        tel = sched.telemetry
        if on:
            assert held == 0, f"EXCL-CYCLE host still holds {held} restored keys"
            assert tel["victim_store_keys"] == 2 * n, "EXCL-CYCLE not stored again"
            assert tel.get("victim_resident", 0) == 0
            stats = manager.excl_stats()
            assert sum(s["evicted"] for s in stats) == n
            assert sum(s["discarded"] for s in stats) == n
        else:
            assert held == n
            assert tel["victim_store_keys"] == n, "EXCL-CYCLE flag off stored again"
            assert tel["victim_resident"] == n
    finally:
        victim.deactivate()


# -- registration ---------------------------------------------------------------------


def test_env_registered_default_off_and_compile_ignored(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import tpu_compiler

    assert f'"{EXCL_ENV}"' in open(envs.__file__).read()
    monkeypatch.delenv(EXCL_ENV, raising=False)
    assert envs.environment_variables[EXCL_ENV]() is False, "EXCL-ENV default on"
    monkeypatch.setenv(EXCL_ENV, "1")
    assert envs.environment_variables[EXCL_ENV]() is True
    assert EXCL_ENV in tpu_compiler._TPU_COMPILE_ENV_IGNORED, "EXCL-ENV not compile-ignored"
    assert EXCL_ENV not in tpu_compiler._tpu_compile_env_factors(), "EXCL-ENV hashed"
