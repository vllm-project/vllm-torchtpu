# SPDX-License-Identifier: Apache-2.0
"""TIER-X2 read room (VLLM_TPU_OFFLOAD_TIER_X2_READ_ROOM), on CPU.

A peer read lands through the reading store's own host pool: Raiden's
LoadRemoteBlocks takes one free host block per sub-block (AllocateBlockIds, free
space only) and fails the whole read when it cannot. The fake below models exactly
that. The counters (always on with TIER-X2) say how often a read launched short of
room and split its failures by it; the flag makes room first by evicting the
store's oldest unpinned held entries, which are discards, never maybe-remote.
"""

import os
from unittest.mock import patch

from vllm.v1.kv_offload.base import LookupResult, ReqContext, make_offload_key

from tests.offload.test_q38_tier_x2 import REGISTRY, ByteStore, FakeNode
from tests.offload.test_raiden_store_offload import FAKE_STORE_TYPES
from tests.offload.test_tier_classes import MAIN


def _ctx(req_id="req"):
    return ReqContext(req_id=req_id)


def _key(n):
    return make_offload_key(b"roomkey_%04d" % n, MAIN)


class StagingByteStore(ByteStore):
    """ByteStore whose peer reads stage through its own host blocks: a read
    launched with fewer free blocks than sub-blocks fails whole when it settles
    (LoadRemoteBlocks: AllocateBlockIds fails -> MarkFailed every hash)."""

    def __init__(self, name, registry, node, host_blocks):
        super().__init__(name, registry, node)
        self.host_blocks = host_blocks
        self._staged: set[bytes] = set()
        self._starved: set[bytes] = set()

    def free_blocks(self):
        return self.host_blocks - len(self.entries) - len(self._staged)

    def read_remote(self, block_hashes, slices, device_block_ids):
        free = self.free_blocks()
        ok = super().read_remote(block_hashes, slices, device_block_ids)
        if ok:
            if free < len(block_hashes):
                self._starved.update(block_hashes)
            else:
                self._staged.update(block_hashes)
        return ok

    def finish(self):
        for h in list(self._remote_reading):
            if h in self._starved:
                self._starved.discard(h)
                self._remote_src.pop(h, None)
                self.complete_remote_read([h], success=False)
        super().finish()
        self._staged.clear()


def _manager(capacity=8, room=False, margin=0):
    from vllm_torchtpu.offload.raiden_store import RaidenOffloadingManager

    store = StagingByteStore("main", {}, FakeNode("node-main"), capacity)
    with patch.dict(os.environ, {"PYTHONHASHSEED": "0"}):
        m = RaidenOffloadingManager(
            kernel_physical_blocks_capacity=capacity,
            offload_logical_blocks_capacity=capacity,
            device_block_size_factor=1,
            world_size=1,
            raiden_controller_port=0,
            raiden_job_name="test-room",
            global_registry_address=REGISTRY["global_registry_address"],
            store_server_ip=REGISTRY["store_server_ip"],
            kv_pool_group="q38x2-main",
            key_namespace=b"R" * 16,
            track_held=True,
            tier_x2=True,
            x2_read_room=room,
            x2_read_room_margin=margin,
            store=store,
            store_types=FAKE_STORE_TYPES,
        )
    return m, store


def _finish(m, store):
    store.finish()
    return m.poll_finished_jobs()


def _save(m, store, keys, first_block, job_id):
    blocks = list(range(first_block, first_block + len(keys)))
    for k, b in zip(keys, blocks):
        store.hbm[b] = b"payload|" + bytes(k)
    out = m.prepare_store(list(keys), _ctx())
    assert list(out.keys_to_store) == list(keys)
    m.submit_store_job(job_id, list(keys), blocks)
    (done,) = _finish(m, store)
    assert done.success


def _demote(m, store, keys):
    store.demote([m._sub_hashes(k)[0] for k in keys])


def _full_pool_with_demoted(m, store, n_demoted=4, capacity=8):
    """Fill the pool, demote the first n_demoted keys to the node, then refill the
    pool: the demoted keys are only on the node and the pool has 0 free blocks."""
    first = [_key(i) for i in range(capacity)]
    _save(m, store, first, 100, job_id=1)
    _demote(m, store, first[:n_demoted])
    refill = [_key(100 + i) for i in range(n_demoted)]
    _save(m, store, refill, 200, job_id=2)
    assert store.free_blocks() == 0
    return first[:n_demoted], first[n_demoted:], refill


def _read(m, store, keys, first_dst, job_id):
    for k in keys:
        assert m.lookup(k, _ctx()) == LookupResult.HIT, k
    spec = m.prepare_load(keys, _ctx())
    assert spec.pinned
    dst = list(range(first_dst, first_dst + len(keys)))
    m.submit_load_job(job_id, list(keys), dst, "req")
    return dst


def test_flag_off_a_short_read_fails_whole_and_the_counters_say_why():
    m, store = _manager(room=False)
    demoted, _, _ = _full_pool_with_demoted(m, store)
    _read(m, store, demoted, 300, job_id=3)
    (done,) = _finish(m, store)
    assert not done.success
    s = m.x2_stats()
    assert s["read_room_short"] == 1
    assert s["read_room_evicted"] == 0 and s["read_room_still_short"] == 0
    assert s["remote_read_blocks_failed"] == 4
    assert s["remote_failed_short"] == 4 and s["remote_failed_roomy"] == 0


def test_flag_on_makes_room_from_the_oldest_held_and_the_read_lands():
    m, store = _manager(room=True)
    demoted, oldest, refill = _full_pool_with_demoted(m, store)
    dst = _read(m, store, demoted, 300, job_id=3)
    (done,) = _finish(m, store)
    assert done.success and done.failed_device_block_ids == []
    for k, d in zip(demoted, dst):
        assert store.hbm[d] == b"payload|" + bytes(k)
    s = m.x2_stats()
    assert s["read_room_short"] == 1 and s["read_room_evicted"] == 4
    assert s["read_room_still_short"] == 0
    assert s["remote_read_blocks_failed"] == 0
    # The oldest held went (store order), the refill stayed.
    gone = {m._sub_hashes(k)[0] for k in oldest}
    assert gone <= set(store.evicted)
    assert all(m._sub_hashes(k)[0] in store.entries for k in refill)
    # Discards, not demotions: never recorded as maybe on the node.
    assert not any(m._x2_maybe_remote(k) for k in oldest)
    assert all(m._x2_maybe_remote(k) for k in demoted)


def test_the_room_step_skips_pinned_entries():
    m, store = _manager(room=True)
    demoted, oldest, refill = _full_pool_with_demoted(m, store)
    pinned = m._sub_hashes(oldest[0])[0]
    store.entries[pinned].pin_count += 1
    _read(m, store, demoted, 300, job_id=3)
    (done,) = _finish(m, store)
    assert done.success
    assert pinned in store.entries and pinned not in store.evicted
    assert m.x2_stats()["read_room_evicted"] == 4


def test_the_margin_makes_more_room():
    m, store = _manager(room=True, margin=2)
    demoted, _, _ = _full_pool_with_demoted(m, store)
    _read(m, store, demoted, 300, job_id=3)
    (done,) = _finish(m, store)
    assert done.success
    assert m.x2_stats()["read_room_evicted"] == 6


def test_a_roomy_read_evicts_nothing_and_a_node_loss_counts_as_roomy():
    m, store = _manager(capacity=16, room=True)
    keys = [_key(i) for i in range(4)]
    _save(m, store, keys, 100, job_id=1)
    _demote(m, store, keys)
    for k in keys:
        assert m.lookup(k, _ctx()) == LookupResult.HIT
    m.prepare_load(keys, _ctx())
    store.node_evicts([m._sub_hashes(keys[3])[0]])
    m.submit_load_job(3, keys, [300, 301, 302, 303], "req")
    (done,) = _finish(m, store)
    assert not done.success
    s = m.x2_stats()
    assert s["read_room_short"] == 0 and s["read_room_evicted"] == 0
    assert s["remote_failed_roomy"] == 1 and s["remote_failed_short"] == 0


def test_the_estimate_counts_peer_reads_still_landing():
    m, store = _manager(room=False)
    first = [_key(i) for i in range(8)]
    _save(m, store, first, 100, job_id=1)
    _demote(m, store, first)  # pool empty: 8 free
    _read(m, store, first[:5], 300, job_id=3)  # stages 5, 3 left
    assert m._x2_staging_free_estimate() == 3
    _read(m, store, first[5:], 400, job_id=4)  # 3 fit
    assert m.x2_stats()["read_room_short"] == 0
    done = _finish(m, store)
    assert all(d.success for d in done)


def test_the_estimate_counts_saves_in_flight():
    m, store = _manager(room=False)
    first = [_key(i) for i in range(8)]
    _save(m, store, first, 100, job_id=1)
    _demote(m, store, first)  # pool empty: 8 free
    more = [_key(50 + i) for i in range(3)]
    for k, b in zip(more, (500, 501, 502)):
        store.hbm[b] = b"payload|" + bytes(k)
    out = m.prepare_store(more, _ctx())
    assert list(out.keys_to_store) == more
    m.submit_store_job(9, more, [500, 501, 502])  # not landed yet
    assert m._x2_staging_free_estimate() == 5
    _finish(m, store)
    assert m._x2_staging_free_estimate() == 5  # now held


def test_without_tier_x2_the_flag_is_inert():
    from vllm_torchtpu.offload.raiden_store import RaidenOffloadingManager
    from tests.offload.test_raiden_store_offload import FakeKVCacheStore

    with patch.dict(os.environ, {"PYTHONHASHSEED": "0"}):
        m = RaidenOffloadingManager(
            kernel_physical_blocks_capacity=8,
            offload_logical_blocks_capacity=8,
            device_block_size_factor=1,
            world_size=1,
            raiden_controller_port=0,
            raiden_job_name="test-room-off",
            track_held=True,
            x2_read_room=True,
            store=FakeKVCacheStore(),
            store_types=FAKE_STORE_TYPES,
        )
    assert m._x2_read_room is False


def test_the_env_flags_default_off(monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.delenv("VLLM_TPU_OFFLOAD_TIER_X2_READ_ROOM", raising=False)
    monkeypatch.delenv("VLLM_TPU_OFFLOAD_TIER_X2_READ_ROOM_MARGIN", raising=False)
    assert envs.VLLM_TPU_OFFLOAD_TIER_X2_READ_ROOM is False
    assert envs.VLLM_TPU_OFFLOAD_TIER_X2_READ_ROOM_MARGIN == 16
