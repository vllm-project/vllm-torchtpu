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
"""TPU_RAIDEN_LRU_REFRESH / TPU_RAIDEN_OFFLOAD_STATS: host-LRU refresh of
resident chains and the offload counters, on the fake Raiden store."""

import collections
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.offload.test_raiden_store_offload import (
    FakeBlockStatus,
    FakeKVCacheStore,
    _ctx,
    _Entry,
    _group_key,
    make_manager,
    sub_hashes,
)

F = 2  # device_block_size_factor


def _resident(store, key, status=FakeBlockStatus.HOST):
    for h in sub_hashes(key, F):
        store.entries[h] = _Entry(status=status)


def _releases(store):
    return [c.args[0] for c in store.calls if c.name == "release"]


def test_refresh_resident_releases_chain_head_last_and_stops_at_miss():
    store = FakeKVCacheStore()
    m = make_manager(store, device_block_size_factor=F)
    keys = [_group_key(i, 0) for i in range(4)]
    for k in keys[:2] + keys[3:]:
        _resident(store, k)
    store.calls.clear()
    assert m.refresh_resident(keys) == 2 * F
    # One pinned lookup, one release in reverse: k0's sub-hashes go back last.
    expected = sub_hashes(keys[0], F) + sub_hashes(keys[1], F)
    assert _releases(store) == [expected[::-1]]
    # Every pin handed back; k3 (past the miss) is not touched.
    assert all(e.pin_count == 0 for e in store.entries.values())
    assert m._stats["refreshed_entries"] == 2 * F


def test_refresh_resident_gives_back_pins_past_the_host_run():
    store = FakeKVCacheStore()
    m = make_manager(store, device_block_size_factor=F)
    keys = [_group_key(i, 0) for i in range(3)]
    _resident(store, keys[0])
    _resident(store, keys[1], FakeBlockStatus.HBM)  # a save in flight
    _resident(store, keys[2])
    assert m.refresh_resident(keys) == F
    assert all(e.pin_count == 0 for e in store.entries.values())
    # Only k0 is refreshed as a run; the HBM entry and what follows are
    # released unrefreshed (balanced pins).
    assert _releases(store)[-1] == sub_hashes(keys[0], F)[::-1]


def test_prepare_store_refreshes_skipped_resident_keys_only_when_on():
    for on in (False, True):
        store = FakeKVCacheStore()
        m = make_manager(store, device_block_size_factor=F)
        m._lru_refresh_on_store = on
        keys = [_group_key(i, 0) for i in range(4)]
        _resident(store, keys[0])
        _resident(store, keys[1])
        store.calls.clear()
        out = m.prepare_store(keys, _ctx())
        assert out.keys_to_store == keys[2:]
        rel = _releases(store)
        if on:
            assert rel == [(sub_hashes(keys[0], F) + sub_hashes(keys[1], F))[::-1]]
        else:
            assert rel == []
        assert all(e.pin_count == 0 for e in store.entries.values())
        assert m._stats["store_keys_offered"] == 4
        assert m._stats["store_keys_resident"] == 2
        assert m._stats["store_keys_new"] == 2


def test_prepare_store_does_not_refresh_remote_keys():
    store = FakeKVCacheStore()
    m = make_manager(store, device_block_size_factor=F)
    m._lru_refresh_on_store = True
    keys = [_group_key(0, 0)]
    _resident(store, keys[0], FakeBlockStatus.REMOTE)
    store.calls.clear()
    out = m.prepare_store(keys, _ctx())
    assert out.keys_to_store == []
    assert _releases(store) == []


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("TPU_RAIDEN_LRU_REFRESH", raising=False)
    monkeypatch.delenv("TPU_RAIDEN_OFFLOAD_STATS", raising=False)
    from vllm_torchtpu import envs

    assert envs.TPU_RAIDEN_LRU_REFRESH == 0
    assert envs.TPU_RAIDEN_OFFLOAD_STATS is False
    m = make_manager(FakeKVCacheStore(), device_block_size_factor=F)
    assert m._lru_refresh_on_store is False
    monkeypatch.setenv("TPU_RAIDEN_LRU_REFRESH", "1")
    assert make_manager(
        FakeKVCacheStore(), device_block_size_factor=F
    )._lru_refresh_on_store


# -- scheduler side -----------------------------------------------------------


def _bare_scheduler(group_configs):
    from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingScheduler

    s = TPURaidenOffloadingScheduler.__new__(TPURaidenOffloadingScheduler)
    s.config = SimpleNamespace(kv_group_configs=group_configs)
    s._raiden_manager = MagicMock()
    s._req_status = {}
    s.wb_counters = collections.Counter()
    return s


def _group(idx=0, eagle=False, tokens_per_block=128, tokens_per_chunk=128):
    return SimpleNamespace(
        group_idx=idx,
        is_eagle_group=eagle,
        tokens_per_block=tokens_per_block,
        tokens_per_chunk=tokens_per_chunk,
        hashes_per_chunk=tokens_per_chunk // 128,
        sliding_window_size_in_chunks=None,
    )


def test_request_finished_refreshes_prefix_groups_when_mode_2():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    for mode2 in (False, True):
        s = _bare_scheduler([_group(0), _group(1, eagle=True)])
        s._lru_refresh_on_finish = mode2
        keys = [_group_key(i, 0) for i in range(3)]
        s._req_status["r"] = SimpleNamespace(
            group_states=[
                SimpleNamespace(offload_keys=keys),
                SimpleNamespace(offload_keys=[_group_key(9, 1)]),
            ]
        )
        req = SimpleNamespace(request_id="r")
        with patch.object(
            OffloadingConnectorScheduler, "request_finished", return_value=(False, None)
        ) as base:
            assert s.request_finished(req) == (False, None)
        base.assert_called_once()
        if mode2:
            s._raiden_manager.refresh_resident.assert_called_once_with(keys)
        else:
            s._raiden_manager.refresh_resident.assert_not_called()


def test_prefix_hit_block_ids():
    from vllm_torchtpu.offload.raiden_connector import _prefix_hit_block_ids

    out = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(block_ids=([5, 6, 7, 8],), num_computed_tokens=256),
            SimpleNamespace(block_ids=([9, 10],), num_computed_tokens=0),
        ]
    )
    cfg = SimpleNamespace(kv_group_configs=[_group(0)])
    assert _prefix_hit_block_ids(out, cfg) == {5, 6}


def test_stale_head_counter():
    s = _bare_scheduler([_group(0)])
    s._offload_stats = True
    req = SimpleNamespace(block_hashes=[b"h%d" % i for i in range(6)])
    # Lookup stopped at token 128 (chunk 1); chunks 1..5 not loaded.
    s._raiden_manager.count_resident_prefix.return_value = 1
    s._count_stale_head(req, 0, 128)
    assert s.wb_counters["lookup_partial"] == 1
    assert s.wb_counters["lookup_stale_head"] == 1
    assert s.wb_counters["lookup_stale_head_tokens"] == 5 * 128
    # The last key is checked, nothing is pinned or loaded.
    (keys,), _ = s._raiden_manager.count_resident_prefix.call_args
    assert len(keys) == 1
    s._raiden_manager.count_resident_prefix.return_value = 0
    s._count_stale_head(req, 0, 128)
    assert s.wb_counters["lookup_stale_head"] == 1
    assert s.wb_counters["lookup_partial"] == 2
