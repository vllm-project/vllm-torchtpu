# SPDX-License-Identifier: Apache-2.0
"""q38 F2-HOLD next to the victim host tier (QWEN4_EXP_HOST_TIER_POLICY=victim), on the REAL vLLM KVCacheManager over
the served F2 producer plan (the same harness as test_qwen4_exp_pd_draft_hold.py).

The two engine patches meet in the block pool: the victim hooks (offload/victim.py) wrap ``get_new_blocks`` and
``_maybe_evict_cached_block`` and stamp the free queue; the hold (core/producer_draft_hold.py) gives the draft's
main-KV and compressed-key groups null blocks below each prefix hit and never frees a written draft block before the
request ends. Both are installed in one engine on q38-int-6 + F2-HOLD (patch_registry: victim, then hold).

- With the victim tap active, allocating a new request evicts cached main-model blocks of a finished request and the
  tap records them; every recorded victim is a hashed block of a prefix-cacheable group (never a draft group, never the
  null block, never a block a live held request owns) and carries a free stamp.
- The held request's draft tables, the hit and the pool counts are the same with the victim tap active as without it.
- The pool's free count returns to its baseline once every request is freed.

The pool is sized from a first, unconstrained pass so that the evicting request needs about half of the cached
filler's blocks after the never-used ones run out (vLLM pops never-used blocks first, then cached ones, LRU).
"""

import pytest

from tests.distributed import test_qwen4_exp_decode_prefix_cache as dpc
from tests.distributed import test_qwen4_exp_pd_draft_hold as th
from tests.distributed.test_qwen4_exp_decode_prefix_cache import (  # noqa: F401
    draft_kv,
    engine_patches,
    served_decode_env,
)
from tests.distributed.test_qwen4_exp_pd_draft_hold import hold  # noqa: F401

P = th.P


@pytest.fixture
def victim_tap(monkeypatch):
    """The victim policy's block-pool hooks installed and a tap active for the test's duration (the engine-core
    patch plus what TPURaidenOffloadingScheduler does at construction)."""
    from vllm_torchtpu.offload import victim

    monkeypatch.setenv(victim.POLICY_ENV, "victim")
    assert victim.host_tier_policy() == "victim"
    victim.patch_block_pool_victims()
    tap = victim.VictimTap()
    victim.activate(tap)
    yield tap
    victim.deactivate(tap)


def _pool_counts(prod):
    pool = prod.kvm.block_pool
    null = pool.null_block
    blocks = [b for b in pool.blocks if b is not null]
    never_used = sum(1 for b in blocks if b.ref_cnt == 0 and b.block_hash is None)
    cached_free = sum(1 for b in blocks if b.ref_cnt == 0 and b.block_hash is not None)
    return len(blocks), never_used, cached_free


def _before_big(prod):
    """A cached seed (prefix X), a held request on X (a 3-page hit), then a cached, unused filler (prefix Y)."""
    _, seed_prompt = th._shared_prompt(3, 700, 1)
    seed = prod.request(seed_prompt)
    assert prod.prefill(seed) == 0
    prod.kvm.free(seed)
    _, held_prompt = th._shared_prompt(3, 2000, 1)
    held = prod.request(held_prompt)
    hit = prod.prefill(held)
    tables = prod.tables(held)
    _, filler_prompt = th._shared_prompt(4, 900, 2)
    filler = prod.request(filler_prompt)
    assert prod.prefill(filler) == 0
    prod.kvm.free(filler)
    return held, hit, tables


BIG_TOKENS = 6 * P + 300


def _big(prod):
    big = prod.request(dpc.tokens(BIG_TOKENS, 777))
    assert prod.prefill(big) == 0
    return big


def _sized_pool(monkeypatch, cfg):
    """Pass 1 on the default pool measures what the evicting request needs; the pool is then cut so that it must evict
    about half of the cached filler's blocks. Returns the expected number of evictions (lower bound)."""
    prod = th.Producer(cfg)
    held, _, _ = _before_big(prod)
    total, never_used, cached_free = _pool_counts(prod)
    free_before = prod.free()
    big = _big(prod)
    big_need = free_before - prod.free()
    assert big_need > 0 and cached_free > 1, (big_need, cached_free)
    evict = max(1, cached_free // 2)
    target = (total - never_used) + big_need - evict  # never-used blocks left before the big request: big_need - evict
    monkeypatch.setattr(th, "NUM_BLOCKS", max(1, round(th.NUM_BLOCKS * (target + 1) / (total + 1))))
    probe = th.Producer(cfg)
    got = _pool_counts(probe)[0]
    # The pool is linear in NUM_BLOCKS; nudge by one if rounding missed the target.
    while got < target:
        monkeypatch.setattr(th, "NUM_BLOCKS", th.NUM_BLOCKS + 1)
        got = _pool_counts(th.Producer(cfg))[0]
    while got > target:
        monkeypatch.setattr(th, "NUM_BLOCKS", th.NUM_BLOCKS - 1)
        got = _pool_counts(th.Producer(cfg))[0]
    return evict


@pytest.mark.usefixtures("draft_kv", "hold")
def test_victims_are_cacheable_main_blocks_and_the_hold_is_unchanged_by_the_victim_tap(monkeypatch, victim_tap):
    from vllm.v1.core.kv_cache_utils import get_group_id

    cfg = dpc.decode_config()
    victim_tap.take()
    evict = _sized_pool(monkeypatch, cfg)  # pass 1 runs with the tap active too: discard its victims below
    prod = th.Producer(cfg)
    baseline = prod.free()
    draft_groups = {prod.main, prod.comp, prod.ring}
    held, hit, t = _before_big(prod)

    # The hold engaged: null blocks below the 3-page hit, live blocks above; the ring keeps its block.
    assert hit == 3 * P
    for g in (prod.main, prod.comp):
        assert th._nulls(t[g]) == hit // P
        assert all(t[g][hit // P:]), "only the leading entries may be null"
    assert len(t[prod.ring]) == 1 and t[prod.ring][0]

    victim_tap.take()
    big = _big(prod)
    victims = victim_tap.take()

    # The victim path was exercised, and every victim is a hashed, prefix-cacheable main-model block.
    assert len(victims) >= evict, (len(victims), evict)
    held_draft_live = {b for g in draft_groups for b in t[g] if b}
    null_id = prod.kvm.block_pool.null_block.block_id
    for v in victims:
        assert v.block_hash is not None
        assert get_group_id(v.block_hash) not in draft_groups, ("a draft-group block was recorded as a victim", v)
        assert v.block_id != null_id
        assert v.block_id not in held_draft_live, ("a live held draft block was recorded as a victim", v)
        assert v.freed_step is not None, "an evicted cached block entered the free queue, so it carries a free stamp"
    # The held request still owns its draft blocks after the eviction (none of them was handed to the big request).
    assert prod.tables(held) == t
    print("Q38_F2HOLD_VICTIM", {"victims": len(victims), "expected_at_least": evict, "hit": hit,
                                "groups": sorted({get_group_id(v.block_hash) for v in victims})}, flush=True)

    for req in (held, big):
        prod.kvm.free(req)
    assert prod.free() == baseline, "the pool's free count returns to its baseline with both patches active"


@pytest.mark.usefixtures("draft_kv", "hold")
def test_the_victim_tap_changes_no_table_and_no_pool_count(monkeypatch, request):
    """Same requests, same sized pool, with and without the victim tap: identical held tables, hit and free counts."""
    cfg = dpc.decode_config()
    _sized_pool(monkeypatch, cfg)

    def run():
        prod = th.Producer(cfg)
        baseline = prod.free()
        held, hit, t = _before_big(prod)
        mid = prod.free()
        big = _big(prod)
        after_big = prod.free()
        t_after = prod.tables(held)
        for req in (held, big):
            prod.kvm.free(req)
        return hit, t, t_after, baseline - mid, baseline - after_big, prod.free() - baseline

    plain = run()
    tap = request.getfixturevalue("victim_tap")
    tapped = run()
    assert tap.take(), "the tapped run must record victims (the pool is sized for evictions)"
    assert plain == tapped, (plain, tapped)
