# SPDX-License-Identifier: Apache-2.0
"""core/prefix_cache_policy.py against the REAL vLLM scheduler classes.

Skipped without vLLM (CPU hosts); run in the serving image. Uses the same
KV cache manager construction as tests/platforms/test_mamba_checkpoint_retention.py
(state group first, align mode, block 16). Written after srv1 (j-4e8d3ff8):
both policies refused to install there because the image's KVCacheBlock has
no block_hash setter, and nothing in the CPU suite exercised the real class.
"""

import inspect

import pytest

vllm = pytest.importorskip("vllm")
torch = pytest.importorskip("torch")

from vllm import SamplingParams  # noqa: E402
from vllm.utils.hashing import sha256  # noqa: E402
from vllm.v1.core.block_pool import BlockPool  # noqa: E402
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator  # noqa: E402
from vllm.v1.core.kv_cache_manager import KVCacheManager  # noqa: E402
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash  # noqa: E402
from vllm.v1.core.single_type_kv_cache_manager import MambaManager  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request  # noqa: E402

from vllm_torchtpu.core import mamba_checkpoint_retention as mcr  # noqa: E402
from vllm_torchtpu.core import prefix_cache_policy as pcp  # noqa: E402

B = 16
PROMPT = list(range(40))  # replay boundary (40 - 1) // 16 * 16 = 32
_PATCHED = (
    (MambaManager, "cache_blocks"),
    (MambaManager, "_remove_blocks_in_range"),
    (MambaManager, "remove_skipped_blocks"),
    (MambaManager, "pop_blocks_for_free"),
    (KVCacheCoordinator, "free"),
    (KVCacheCoordinator, "pop_blocks_for_free"),
    (BlockPool, "get_cached_block"),
    (BlockPool, "_insert_block_hash"),
    (BlockPool, "free_blocks"),
    (KVCacheCoordinator, "__init__"),
)


@pytest.fixture
def retention(monkeypatch):
    for owner, name in _PATCHED:
        if hasattr(owner, name):
            monkeypatch.setattr(owner, name, inspect.unwrap(getattr(owner, name)))
    monkeypatch.setattr(mcr, "_release", inspect.unwrap(mcr._release))
    monkeypatch.setenv("VLLM_TPU_MAMBA_CHECKPOINT_RETENTION", "1")
    # Only the prompt checkpoint at 32, so the next turn hits exactly it.
    monkeypatch.setenv("VLLM_TPU_MAMBA_DECODE_CHECKPOINT", "0")
    mcr.patch_mamba_checkpoint_retention()


def _manager() -> KVCacheManager:
    init_none_hash(sha256)
    attention = FullAttentionSpec(block_size=B, num_kv_heads=1, head_size=64, dtype=torch.float32)
    state = MambaSpec(
        block_size=B,
        shapes=((1, 64),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        page_size_padded=attention.page_size_bytes,
    )
    config = KVCacheConfig(
        num_blocks=96,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["state"], state),
            KVCacheGroupSpec(["attention"], attention),
        ],
        prefix_cache_retention_interval=0,
    )
    return KVCacheManager(config, max_model_len=1024, scheduler_block_size=B, hash_block_size=B, enable_caching=True)


def _request(request_id, tokens):
    return Request(request_id, tokens, SamplingParams(max_tokens=200), None,
                   block_hasher=get_request_block_hasher(B, sha256))


def _prefill(manager, request):
    manager.new_step_starts()
    blocks, hit = manager.get_computed_blocks(request)[:2]
    prompt, start, first = request.num_tokens, hit, True
    while start < prompt:
        end = prompt if prompt // B * B <= start else prompt // B * B
        manager.new_step_starts()
        assert manager.allocate_slots(request, end - start, hit if first else 0, blocks if first else None) is not None
        first = False
        request.num_computed_tokens = start = end
    request.append_output_token_ids(10_000)
    return hit


def _queue(manager):
    return manager.block_pool.free_block_queue.get_all_free_blocks()


def _first_hashed_index(blocks):
    return next(i for i, b in enumerate(blocks) if b.block_hash is not None)


def test_contract_check_passes_on_this_vllm():
    assert pcp.queue_contract_ok() is True


@pytest.mark.parametrize("demote", [False, True])
def test_stale_checkpoint_placement(retention, monkeypatch, demote):
    if demote:
        monkeypatch.setenv("VLLM_TPU_PREFIX_DEMOTE_STALE_CHECKPOINT", "1")
        pcp.patch_demote_stale_checkpoints()
    manager = _manager()
    turn1 = _request("t1", PROMPT)
    _prefill(manager, turn1)
    ckpt = manager.block_pool.get_cached_block(turn1.block_hashes[32 // B - 1], [0])[0]
    manager.free(turn1)
    other = _request("o", list(range(2000, 2040)))           # another conversation, cached after turn 1
    _prefill(manager, other)
    manager.free(other)
    turn2 = _request("t2", PROMPT + list(range(500, 530)))  # extends turn 1
    assert _prefill(manager, turn2) == 32                   # hit on turn 1's checkpoint
    queue = _queue(manager)
    assert ckpt in queue and ckpt.block_hash is not None    # released, still cached
    hashed = [b for b in queue if b.block_hash is not None]
    assert len(hashed) >= 2                                   # the other conversation's blocks are cached too
    if demote:
        assert hashed[0] is ckpt                             # evicted before any other cached block
        assert queue.index(ckpt) == _first_hashed_index(queue)
        assert pcp.STATS["demoted"] >= 1
    else:
        assert hashed[-1] is ckpt                            # today: MRU end


@pytest.mark.parametrize("demote", [False, True])
def test_same_boundary_hit_is_live_not_demoted(retention, monkeypatch, demote):
    """A turn that crosses no new boundary hits exactly at its own replay boundary (32 here).

    The block it hit is the conversation's live checkpoint: the next turn needs it. Served A/B
    j-74b70fab demoted it on about half of all hit requests and lost 0.11-0.19 hit.
    """
    pcp.STATS.update(demoted=0, demote_calls=0, kept_live=0)
    if demote:
        monkeypatch.setenv("VLLM_TPU_PREFIX_DEMOTE_STALE_CHECKPOINT", "1")
        pcp.patch_demote_stale_checkpoints()
    manager = _manager()
    turn1 = _request("t1", PROMPT)
    _prefill(manager, turn1)
    ckpt = manager.block_pool.get_cached_block(turn1.block_hashes[32 // B - 1], [0])[0]
    assert ckpt.block_hash_num_tokens == 32
    manager.free(turn1)
    other = _request("o", list(range(2000, 2040)))
    _prefill(manager, other)
    manager.free(other)
    turn2 = _request("t2", PROMPT + list(range(500, 504)))  # 44 tokens: replay boundary (44 - 1) // 16 * 16 = 32
    assert _prefill(manager, turn2) == 32
    # The scheduler's post-step call (scheduler.py, remove_skipped_blocks on processed tokens) releases the hit block.
    manager.remove_skipped_blocks("t2", turn2.num_computed_tokens, turn2.num_prompt_tokens)
    queue = _queue(manager)
    assert ckpt in queue and ckpt.block_hash is not None
    hashed = [b for b in queue if b.block_hash is not None]
    assert len(hashed) >= 2
    assert hashed[-1] is ckpt                                 # MRU end in both arms
    assert pcp.STATS["demoted"] == 0
    if demote:
        assert pcp.STATS["kept_live"] >= 1
        state = manager.coordinator.single_type_managers[0]
        assert state.__dict__["_tpu_live_from"]["t2"] == 32
        manager.free(turn2)
        assert "t2" not in state.__dict__["_tpu_live_from"]   # forgotten with the request


# --------------------------------------------------------------------------
# Scheduler-level regression of the served REGRESS (j-74b70fab vs j-a0673a2d):
# the real KVCacheManager with 4 align-Mamba groups and 2 attention groups (the
# served Qwen3.8 prefill's structure), as LEV-CACHE's CPU localization
# (scratch/levcache/demote/mech/q38_demote_mech.py). Before the fix, the "same"
# cells kept 32 turn-3 hit blocks without demotion and 0-16 with it.
# --------------------------------------------------------------------------
SIM_BUDGET, SIM_SEQS, SIM_GEN = 40, 8, 4


def _sim_manager(num_blocks):
    init_none_hash(sha256)
    att = FullAttentionSpec(block_size=B, num_kv_heads=1, head_size=64, dtype=torch.float32)
    st = MambaSpec(block_size=B, shapes=((1, 64),), dtypes=(torch.float32,), mamba_cache_mode="align",
                   page_size_padded=att.page_size_bytes)
    groups = [KVCacheGroupSpec([f"s{i}"], st) for i in range(4)]
    groups += [KVCacheGroupSpec([f"a{i}"], att) for i in range(2)]
    cfg = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups,
                        prefix_cache_retention_interval=0)
    return KVCacheManager(cfg, max_model_len=65536, scheduler_block_size=B, hash_block_size=B, enable_caching=True)


def _chunk(start, prompt, budget):
    last = prompt // B * B
    if start >= last:
        return prompt - start if prompt - start <= budget else 0
    return min(last, start + budget // B * B) - start


def _run_batch(mgr, reqs, hits):
    waiting, running = list(reqs), []
    while waiting or running:
        mgr.new_step_starts()
        budget, sched = SIM_BUDGET, []
        for r in running:
            if r.num_computed_tokens < r.num_prompt_tokens:
                n = _chunk(r.num_computed_tokens, r.num_prompt_tokens, budget)
                if n <= 0:
                    continue
            else:
                n = 1
            assert mgr.allocate_slots(r, n) is not None
            budget -= n
            sched.append((r, n))
        while waiting and len(running) < SIM_SEQS and budget > 0:
            r = waiting[0]
            blocks, hit = mgr.get_computed_blocks(r)[:2]
            n = _chunk(hit, r.num_prompt_tokens, budget)
            if n <= 0:
                break
            assert mgr.allocate_slots(r, n, hit, blocks) is not None
            hits[r.request_id] = hit
            waiting.pop(0)
            running.append(r)
            budget -= n
            sched.append((r, n))
        for r, n in sched:
            r.num_computed_tokens += n
            if r.num_computed_tokens >= r.num_prompt_tokens:
                r.append_output_token_ids(10_000 + len(r.output_token_ids))
                if len(r.output_token_ids) >= SIM_GEN:
                    mgr.free(r)
                    running.remove(r)


def _three_turns(mode, dead, filler, conversations=8, n1=4):
    """Dead one-turn conversations, then turn 1, turn 2 (+4 tokens "same" / +B "cross"), filler, turn 3 (+B)."""
    base, batches = 1000, []
    deads = []
    for _ in range(dead):
        deads.append(list(range(base, base + n1 * B + 6)))
        base += n1 * B + 200
    convs = []
    for _ in range(conversations):
        convs.append(list(range(base, base + n1 * B + 10 + 2 * B)))
        base += n1 * B + 200
    fills = []
    for _ in range(filler):
        fills.append(list(range(base, base + B + 6)))
        base += B + 106
    l1 = n1 * B + 6
    l2 = l1 + 4 if mode == "same" else l1 + B
    l3 = l2 + B
    for i in range(0, dead, 8):
        batches.append([(f"d-{d}", deads[d]) for d in range(i, min(i + 8, dead))])
    batches.append([(f"t1-{c}", convs[c][:l1]) for c in range(conversations)])
    batches.append([(f"t2-{c}", convs[c][:l2]) for c in range(conversations)])
    for i in range(0, filler, 8):
        batches.append([(f"f-{f}", fills[f]) for f in range(i, min(i + 8, filler))])
    batches.append([(f"t3-{c}", convs[c][:l3]) for c in range(conversations)])
    return batches


def _simulate(monkeypatch, batches, num_blocks, demote):
    for owner, name in _PATCHED:
        if hasattr(owner, name):
            monkeypatch.setattr(owner, name, inspect.unwrap(getattr(owner, name)))
    monkeypatch.setattr(mcr, "_release", inspect.unwrap(mcr._release))
    monkeypatch.setenv("VLLM_TPU_MAMBA_CHECKPOINT_RETENTION", "1")
    monkeypatch.setenv("VLLM_TPU_MAMBA_DECODE_CHECKPOINT", "1")
    monkeypatch.setenv("VLLM_TPU_PREFIX_DEMOTE_STALE_CHECKPOINT", "1" if demote else "0")
    mcr.patch_mamba_checkpoint_retention()
    pcp.STATS.update(demoted=0, demote_calls=0, kept_live=0)
    pcp.patch_demote_stale_checkpoints()
    mgr = _sim_manager(num_blocks)
    hits = {}
    for batch in batches:
        reqs = [Request(rid, toks, SamplingParams(max_tokens=SIM_GEN), None,
                        block_hasher=get_request_block_hasher(B, sha256)) for rid, toks in batch]
        _run_batch(mgr, reqs, hits)
    t3 = sum(v for k, v in hits.items() if k.startswith("t3-")) // B
    return t3, dict(pcp.STATS)


_SAME_CELLS = [(pool, dead, filler) for pool in (180, 220) for dead, filler in ((8, 2), (8, 4), (16, 4), (16, 8))]


@pytest.mark.parametrize("pool,dead,filler", _SAME_CELLS)
def test_same_boundary_turns_keep_their_next_turn_hits(monkeypatch, pool, dead, filler):
    batches = _three_turns("same", dead, filler)
    t3_n, _ = _simulate(monkeypatch, batches, pool + 1, demote=False)
    t3_d, stats = _simulate(monkeypatch, batches, pool + 1, demote=True)
    assert t3_n > 0                       # the cell has hits to lose
    assert t3_d >= t3_n, (t3_d, t3_n)     # REGRESS before the fix: 0-16 vs 32
    assert stats["kept_live"] >= 8 * 4    # every turn-2 release (8 conversations x 4 state groups) kept live
    assert stats["demoted"] > 0           # turn 3 crosses a boundary: its hit-on checkpoint is still demoted


@pytest.mark.parametrize("pool,dead,filler", [(180, 8, 4), (220, 16, 8)])
def test_cross_boundary_turns_still_demote(monkeypatch, pool, dead, filler):
    batches = _three_turns("cross", dead, filler)
    t3_n, _ = _simulate(monkeypatch, batches, pool + 1, demote=False)
    t3_d, stats = _simulate(monkeypatch, batches, pool + 1, demote=True)
    assert t3_d >= t3_n
    assert stats["demoted"] >= 2 * 8 * 4  # turns 2 and 3 each supersede their hit-on checkpoint


def test_refresh_moves_waiting_prefix_to_mru(retention):
    manager = _manager()
    x = _request("x", PROMPT)
    _prefill(manager, x)
    manager.free(x)                                          # x's blocks cached, older
    y = _request("y", list(range(1000, 1040)))
    _prefill(manager, y)
    manager.free(y)                                          # y's blocks cached, newer
    z = _request("z", PROMPT + list(range(700, 720)))        # waits; prefix = x's prompt

    class _Sched:
        pass

    sched = _Sched()
    sched.kv_cache_manager = manager
    sched.waiting = [z]
    moved = pcp.refresh_waiting(sched, limit=8, interval=0.0, mamba_cls=MambaManager, now=1.0)
    assert moved > 0
    queue = [b for b in _queue(manager) if b.block_hash is not None]
    hits = manager.coordinator.find_longest_cache_hit(z.block_hashes, z.num_tokens - 1)
    assert hits[1] == 32
    x_blocks = {id(b) for group in hits[0] for b in group if not b.is_null}
    # Every block of z's hit now sits after every block that was not refreshed.
    tail = queue[-len(x_blocks):]
    assert {id(b) for b in tail} == x_blocks
    assert manager.block_pool.free_block_queue.num_free_blocks == len(_queue(manager))


# --------------------------------------------------------------------------
# Tails evict first, on the real KVCacheManager with a prefix-match unit below
# the block (hash unit 4, block 16; the served prefill: 128 / 3,200).
# --------------------------------------------------------------------------
TAIL_UNIT = 4


def _tail_manager():
    init_none_hash(sha256)
    att = FullAttentionSpec(block_size=B, num_kv_heads=1, head_size=64, dtype=torch.float32)
    st = MambaSpec(block_size=B, shapes=((1, 64),), dtypes=(torch.float32,), mamba_cache_mode="align",
                   page_size_padded=att.page_size_bytes)
    groups = [KVCacheGroupSpec([f"s{i}"], st) for i in range(4)]
    groups += [KVCacheGroupSpec([f"a{i}"], att) for i in range(2)]
    cfg = KVCacheConfig(num_blocks=97, kv_cache_tensors=[], kv_cache_groups=groups,
                        prefix_cache_retention_interval=0)
    return KVCacheManager(cfg, max_model_len=4096, scheduler_block_size=B, hash_block_size=TAIL_UNIT,
                          enable_caching=True)


def _tail_turn(manager, rid, tokens):
    r = Request(rid, tokens, SamplingParams(max_tokens=4), None,
                block_hasher=get_request_block_hasher(TAIL_UNIT, sha256))
    manager.new_step_starts()
    blocks, hit = manager.get_computed_blocks(r)[:2]
    start, first = hit, True
    while start < r.num_tokens:
        end = r.num_tokens if r.num_tokens // B * B <= start else min(r.num_tokens // B * B, start + 32)
        manager.new_step_starts()
        assert manager.allocate_slots(r, end - start, hit if first else 0, blocks if first else None) is not None
        first = False
        r.num_computed_tokens = start = end
    r.append_output_token_ids(10_000)
    manager.remove_skipped_blocks(rid, r.num_computed_tokens, r.num_prompt_tokens)
    manager.free(r)
    return hit


@pytest.mark.parametrize("tail_first", [False, True])
def test_tails_join_the_evict_first_segment_oldest_first(monkeypatch, tail_first):
    for owner, name in _PATCHED:
        if hasattr(owner, name):
            monkeypatch.setattr(owner, name, inspect.unwrap(getattr(owner, name)))
    monkeypatch.setattr(mcr, "_release", inspect.unwrap(mcr._release))
    monkeypatch.setenv("VLLM_TPU_MAMBA_CHECKPOINT_RETENTION", "1")
    monkeypatch.setenv("VLLM_TPU_MAMBA_DECODE_CHECKPOINT", "0")
    monkeypatch.setenv("VLLM_TPU_PREFIX_TAIL_EVICT_FIRST", "1" if tail_first else "0")
    mcr.patch_mamba_checkpoint_retention()
    pcp.STATS.update(tails_queued=0, tail_calls=0)
    pcp.patch_tail_evict_first()
    manager = _tail_manager()
    assert getattr(manager.block_pool, "_tpu_tail_unit", None) == (B if tail_first else None)
    _tail_turn(manager, "a", list(range(40)))          # tails at 40 (4 state + 2 KV pages); full keys at 16 / 32
    _tail_turn(manager, "b", list(range(1000, 1040)))
    cached = [b for b in manager.block_pool.free_block_queue.get_all_free_blocks() if b.block_hash is not None]
    covered = [b.block_hash_num_tokens for b in cached]
    assert covered.count(40) == 12                      # 6 tails per conversation
    first_full = next(i for i, c in enumerate(covered) if c % B == 0)
    if tail_first:
        assert covered[:12] == [40] * 12 and first_full == 12   # every tail before every full key
        assert pcp.STATS["tails_queued"] == 12 and pcp.STATS["tail_calls"] == 2
        # oldest first: conversation a's tails (allocated first, lower ids) before b's
        assert max(b.block_id for b in cached[:6]) < min(b.block_id for b in cached[6:12])
    else:
        assert first_full < 12 and pcp.STATS["tails_queued"] == 0   # today: conversation a's full keys first
    assert _tail_turn(manager, "a2", list(range(48))) == 40          # the tail still hits while the pool has room
