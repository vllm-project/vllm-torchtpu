# SPDX-License-Identifier: Apache-2.0
"""Victim host-tier policy (QWEN4_EXP_HOST_TIER_POLICY=victim): the BlockPool hooks, the young-block guard, and
the manager's store admission. The scheduler path on the served PP8 groups is in
tests/runner/test_qwen4_exp_host_tier.py (test_victim_policy_*)."""

from types import SimpleNamespace as NS

import pytest

from vllm_torchtpu.offload import victim

BS = 16  # tokens per block in the BlockPool tests


@pytest.fixture(autouse=True)
def _no_tap_leaks():
    victim.deactivate()
    yield
    victim.deactivate()


def _request(req_id, n_tokens, seed=0):
    from vllm import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    init_none_hash(sha256)
    return Request(
        req_id, prompt_token_ids=[seed * 100_000 + i for i in range(n_tokens)],
        sampling_params=SamplingParams(max_tokens=1), pooling_params=None,
        block_hasher=get_request_block_hasher(BS, sha256),
    )


def _pool(n=8):
    from vllm.v1.core.block_pool import BlockPool

    victim.patch_block_pool_victims()
    return BlockPool(num_gpu_blocks=n, enable_caching=True, hash_block_size=BS)


def _cached(pool, req, n_blocks, group=0):
    """Allocate n_blocks, cache them under req's hashes (group), return them (ref 1)."""
    blocks = pool.get_new_blocks(n_blocks)
    pool.cache_full_blocks(req, blocks, 0, n_blocks, BS, group)
    return blocks


def _hash(req, i, group=0):
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

    return make_block_hash_with_group_id(req.block_hashes[i], group)


# -- BlockPool hooks -------------------------------------------------------------------------------------------


def test_eviction_for_reuse_records_each_victim_with_its_hash_and_free_step():
    pool, tap = _pool(), victim.VictimTap()
    victim.activate(tap)
    req = _request("r", 3 * BS)
    blocks = _cached(pool, req, 3)
    assert tap.pending == []  # caching and fresh allocation evict nothing
    tap.step = 4
    pool.free_blocks(blocks)
    assert {b.block_id: tap.freed_at[b.block_id] for b in blocks} == {b.block_id: 4 for b in blocks}
    tap.step = 9
    pool.get_new_blocks(pool.get_num_free_blocks())  # takes every free block: the 3 cached ones are evicted
    got = {(v.block_id, v.block_hash, v.freed_step) for v in tap.take()}
    assert got == {(b.block_id, _hash(req, i), 4) for i, b in enumerate(blocks)}
    assert tap.take() == []


def test_evictions_outside_an_allocation_are_not_victims():
    """The retention patch's superseded checkpoints and failed-load invalidation clear hashes outside
    get_new_blocks; such blocks must never be stored (their bytes are stale or deliberately dropped)."""
    from vllm_torchtpu.core.mamba_checkpoint_retention import _drop_superseded

    pool, tap = _pool(), victim.VictimTap()
    victim.activate(tap)
    req = _request("r", 3 * BS)
    superseded, failed, plain = _cached(pool, req, 3)
    # retention: a pinned, superseded decode checkpoint of request "r"
    manager = NS(block_pool=pool)
    manager.__dict__["_tpu_own_checkpoints"] = {"r": {superseded.block_id: (superseded, True)}}
    manager.__dict__["_tpu_pinned_checkpoints"] = {"r": [superseded]}
    _drop_superseded(manager, "r", keep=[])
    assert superseded.block_hash is None and superseded.ref_cnt == 0
    # failed load: the connector reports the block invalid
    pool.evict_blocks({failed.block_id})
    assert failed.block_hash is None
    assert tap.take() == []
    pool.free_blocks([failed, plain])
    tap.step = 50
    pool.get_new_blocks(pool.get_num_free_blocks())
    assert [v.block_id for v in tap.take()] == [plain.block_id]  # only the still-cached block


def test_hooks_are_inert_without_a_tap():
    pool = _pool()
    req = _request("r", 2 * BS)
    blocks = _cached(pool, req, 2)
    pool.free_blocks(blocks)
    tap = victim.VictimTap()  # not active
    pool.get_new_blocks(pool.get_num_free_blocks())
    assert tap.pending == [] and tap.freed_at == {}


def test_unpin_blocks_stamps_the_free_step():
    """The PD producer releases its held blocks with unpin_blocks; those reach ref 0 there."""
    from vllm.v1.core.block_pool import BlockPool

    if "unpin_blocks" not in BlockPool.__dict__:
        pytest.skip("this vLLM has no unpin_blocks")
    pool, tap = _pool(), victim.VictimTap()
    victim.activate(tap)
    req = _request("r", 2 * BS)
    blocks = _cached(pool, req, 2)
    tap.step = 7
    pool.unpin_blocks(blocks, on_reuse=lambda b: None)
    assert all(tap.freed_at[b.block_id] == 7 for b in blocks)


# -- young-block guard and selection ---------------------------------------------------------------------------


def test_young_unstamped_foreign_group_and_duplicate_victims_are_not_stored():
    from vllm.v1.kv_offload.base import make_offload_key

    req = _request("r", 6 * BS)
    step, age = 20, 10
    vs = [
        victim.Victim(1, _hash(req, 0), freed_step=step - age),       # exactly old enough: kept
        victim.Victim(2, _hash(req, 1), freed_step=step - age + 1),   # one step too young
        victim.Victim(3, _hash(req, 2), freed_step=None),             # unstamped: never stored blind
        victim.Victim(4, _hash(req, 3, group=5), freed_step=0),       # group 5 is not offloaded
        victim.Victim(5, _hash(req, 4), freed_step=0, extra_hashes=2),
        victim.Victim(6, _hash(req, 4), freed_step=0),                # same key again
    ]
    keep, counts = victim.select_victims(vs, step, age, offload_groups={0})
    assert keep == [(make_offload_key(req.block_hashes[0], 0), 1),
                    (make_offload_key(req.block_hashes[4], 0), 5)]
    assert counts == {"victim_seen": 6, "victim_young": 1, "victim_unstamped": 1,
                      "victim_not_offloaded": 1, "victim_duplicate": 1, "victim_extra_hashes": 2}


def test_policy_env(monkeypatch):
    monkeypatch.delenv(victim.POLICY_ENV, raising=False)
    assert victim.host_tier_policy() == "write_through"
    monkeypatch.setenv(victim.POLICY_ENV, "victim")
    assert victim.host_tier_policy() == "victim"
    monkeypatch.setenv(victim.POLICY_ENV, "exclusive")
    with pytest.raises(ValueError, match="expected one of"):
        victim.host_tier_policy()


# -- manager admission -----------------------------------------------------------------------------------------


def _manager():
    from tests.offload.test_raiden_store_offload import FakeKVCacheStore, make_manager

    store = FakeKVCacheStore()
    return make_manager(store, device_block_size_factor=1, num_blocks=8, world_size=1), store


def test_victim_mode_stores_nothing_on_compute():
    from vllm.v1.kv_offload.base import OffloadKey

    manager, _ = _manager()
    k = OffloadKey(b"k" * 36)
    out = manager.prepare_store([k], None)
    assert out.keys_to_store == [k]
    manager.on_store_job_cancelled([k])
    manager.set_victim_mode(True)
    out = manager.prepare_store([k], None)
    assert out.keys_to_store == [] and k not in manager._inflight_store_keys


def test_prepare_victim_store_skips_resident_and_in_flight_keys():
    from tests.offload.test_raiden_store_offload import seed_host_entries
    from vllm.v1.kv_offload.base import OffloadKey

    manager, store = _manager()
    manager.set_victim_mode(True)
    resident, fresh = OffloadKey(b"a" * 36), OffloadKey(b"b" * 36)
    seed_host_entries(store, resident, device_block_size_factor=1)
    assert manager.prepare_victim_store([resident, fresh]) == [fresh]
    assert manager.prepare_victim_store([fresh]) == []  # already in flight


_SERVED_ORDER = r'''
import json
from vllm_torchtpu.offload import victim
import vllm_torchtpu
victim.patch_block_pool_victims()            # engine_core stage (and the connector's _enable_victim_policy)
vllm_torchtpu._patch_vllm_block_pool_lifo_free()   # platform_activation, 4 s later in the same EngineCore process
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request
init_none_hash(sha256)
req = Request("r", prompt_token_ids=list(range(48)), sampling_params=SamplingParams(max_tokens=1), pooling_params=None,
              block_hasher=get_request_block_hasher(16, sha256))
tap = victim.VictimTap(); victim.activate(tap)
pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
blocks = pool.get_new_blocks(3); pool.cache_full_blocks(req, blocks, 0, 3, 16, 0)
tap.step = 4; pool.free_blocks(blocks)
tap.step = 9; pool.get_new_blocks(pool.get_num_free_blocks())
print("Q38_SERVED " + json.dumps(sorted((v.block_id, v.freed_step) for v in tap.take())))
'''


def test_victims_are_stamped_when_the_lifo_patch_lands_after_the_hook():
    """j-47fee8a1 (served, multi-process engine core): the victim hook was installed at the engine_core stage and the
    LIFO free patch replaced BlockPool.free_blocks 4 s later at platform_activation, so all 220,445 victims arrived
    unstamped and none was stored. Stamping must survive any later replacement of free_blocks (served order)."""
    import os
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "-c", _SERVED_ORDER], capture_output=True, text=True, env=dict(os.environ))
    line = next((l for l in r.stdout.splitlines() if l.startswith("Q38_SERVED ")), None)
    assert line is not None, r.stderr[-2000:]
    import json

    got = json.loads(line[len("Q38_SERVED "):])
    assert len(got) == 3 and all(step == 4 for _, step in got), got
