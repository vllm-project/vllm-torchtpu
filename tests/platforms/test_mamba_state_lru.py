# SPDX-License-Identifier: Apache-2.0
"""Separate LRU budget for cached Mamba state blocks.

See `vllm_torchtpu.core.mamba_state_lru`. These tests drive a real
scheduler-side `KVCacheManager` (one Mamba group, one attention group, the
served retention patch on) through prefill, decode and free, with and
without the state LRU, and check:

- the two budgets are independent: attention allocations take KV victims and
  leave parked checkpoints alone, budget evictions take state victims;
- a lost checkpoint degrades the next hit to the previous retained boundary
  (its KV and state kept), not to 0, and releases only the KV it alone needed;
- a lost KV block takes the checkpoints behind it out of the state LRU;
- with the budget at 0 nothing is installed and behaviour is unchanged.
"""

import inspect

import pytest
import torch
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

from vllm_torchtpu.core import mamba_checkpoint_retention, mamba_state_lru

B = 16
NUM_BLOCKS = 96
STATE, ATTN = 0, 1
_RETENTION_PATCHED = (
    (MambaManager, "cache_blocks"),
    (MambaManager, "_remove_blocks_in_range"),
    (MambaManager, "remove_skipped_blocks"),
    (MambaManager, "pop_blocks_for_free"),
    (KVCacheCoordinator, "free"),
    (KVCacheCoordinator, "pop_blocks_for_free"),
)


@pytest.fixture(autouse=True)
def served_retention(monkeypatch):
    """Upstream plus the served retention patch (MTP: no decode checkpoints)."""
    for owner, name in (
        *_RETENTION_PATCHED,
        (BlockPool, "get_cached_block"),
        (BlockPool, "_insert_block_hash"),
        (KVCacheManager, "__init__"),
    ):
        monkeypatch.setattr(owner, name, inspect.unwrap(getattr(owner, name)))
    monkeypatch.setenv("VLLM_TPU_MAMBA_CHECKPOINT_RETENTION", "1")
    monkeypatch.setenv("VLLM_TPU_MAMBA_DECODE_CHECKPOINT", "1")
    mamba_checkpoint_retention.patch_mamba_checkpoint_retention()


def _manager(budget: int | None = None) -> KVCacheManager:
    init_none_hash(sha256)
    attention = FullAttentionSpec(
        block_size=B, num_kv_heads=1, head_size=64, dtype=torch.float32
    )
    state = MambaSpec(
        block_size=B,
        shapes=((1, 64),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        page_size_padded=attention.page_size_bytes,
        # Served with MTP: speculative blocks, so no decode checkpoints.
        num_speculative_blocks=2,
    )
    config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["state"], state),
            KVCacheGroupSpec(["attention"], attention),
        ],
        prefix_cache_retention_interval=0,
    )
    manager = KVCacheManager(
        config,
        max_model_len=1024,
        scheduler_block_size=B,
        hash_block_size=B,
        enable_caching=True,
    )
    if budget is not None:
        assert mamba_state_lru.install_mamba_state_lru(manager.coordinator, budget)
    return manager


def _request(request_id: str, tokens: list[int]) -> Request:
    return Request(
        request_id,
        tokens,
        SamplingParams(max_tokens=200),
        None,
        block_hasher=get_request_block_hasher(B, sha256),
    )


def _run(manager: KVCacheManager, request: Request, num_decode: int = 3) -> int:
    """Admit, prefill in block-aligned chunks, decode, free. Returns the hit."""
    manager.new_step_starts()
    blocks, hit = manager.get_computed_blocks(request)[:2]
    prompt = request.num_tokens
    start, first = hit, True
    while start < prompt:
        end = prompt if prompt // B * B <= start else prompt // B * B
        manager.new_step_starts()
        assert (
            manager.allocate_slots(
                request,
                end - start,
                hit if first else 0,
                blocks if first else None,
            )
            is not None
        )
        first = False
        request.num_computed_tokens = start = end
    request.append_output_token_ids(10_000)
    for i in range(num_decode):
        manager.new_step_starts()
        assert manager.allocate_slots(request, 1) is not None
        request.num_computed_tokens += 1
        request.append_output_token_ids(10_001 + i)
    manager.free(request)
    return hit


def _lookup(manager: KVCacheManager, request: Request, tokens: int, group: int):
    key = make_block_hash_with_group_id(request.block_hashes[tokens // B - 1], group)
    return manager.block_pool.cached_block_hash_to_block.get_one_block(key)


def _queue_ids(manager: KVCacheManager) -> list[int]:
    return [
        b.block_id for b in manager.block_pool.free_block_queue.get_all_free_blocks()
    ]


def _lru(manager: KVCacheManager) -> mamba_state_lru.MambaStateLRU:
    return manager.block_pool._tpu_mamba_state_lru


def _check_accounting(manager: KVCacheManager) -> None:
    pool, lru = manager.block_pool, _lru(manager)
    queue = set(_queue_ids(manager))
    parked = {b.block_id for blocks in lru.lru.values() for b in blocks}
    assert not queue & parked
    assert len(parked) == lru.lru_blocks == len(lru.lru_key)
    assert pool.get_num_free_blocks() == len(queue) + len(parked)
    for blocks in lru.lru.values():
        for block in blocks:
            assert block.ref_cnt == 0 and block.block_hash is not None


def _conversation():
    """Three turns; turn N+1's prompt extends turn N's prompt and output."""
    turn1 = _request("t1", list(range(40)))
    return turn1


def _next_turn(prev: Request, request_id: str, extra: int) -> Request:
    tokens = list(prev.all_token_ids) + list(range(5000, 5000 + extra))
    return _request(request_id, tokens)


# --------------------------------------------------------------------------- #


def test_parked_checkpoint_counts_as_free_and_hits():
    manager = _manager(budget=8)
    turn1 = _conversation()
    _run(manager, turn1)
    # Replay boundary of a 40-token prompt: 32.
    checkpoint = _lookup(manager, turn1, 32, STATE)
    assert checkpoint is not None and checkpoint.ref_cnt == 0
    assert checkpoint.block_id not in _queue_ids(manager)
    assert _lru(manager).lru_blocks >= 1
    # Every block but the null block is free, as without the budget.
    assert manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1
    _check_accounting(manager)

    turn2 = _next_turn(turn1, "t2", 20)
    assert _run(manager, turn2) == 32
    assert _lru(manager).stats["hits"] >= 1
    _check_accounting(manager)


def test_free_capacity_matches_baseline():
    plain, budgeted = _manager(), _manager(budget=8)
    for manager in (plain, budgeted):
        prev = _conversation()
        _run(manager, prev)
        for i in range(3):
            prev = _next_turn(prev, f"t{i + 2}", 20)
            _run(manager, prev)
        _run(manager, _request("other", list(range(900, 950))))
    assert (
        plain.block_pool.get_num_free_blocks()
        == budgeted.block_pool.get_num_free_blocks()
    )
    assert plain.usage == budgeted.usage
    _check_accounting(budgeted)


def test_attention_allocations_do_not_evict_parked_checkpoints():
    """KV victims come from the shared queue only; parked states stay."""
    plain, budgeted = _manager(), _manager(budget=32)
    for manager in (plain, budgeted):
        x1 = _request("x1", list(range(40)))
        _run(manager, x1)
        _run(manager, _next_turn(x1, "x2", 30))
        _run(manager, _request("y", list(range(2000, 2040))))
    # Plain LRU: an allocation of all but a few free blocks takes cached
    # state blocks as victims along with KV.
    pool = plain.block_pool
    states_before = {
        b.block_id
        for b in pool.free_block_queue.get_all_free_blocks()
        if b.block_hash is not None and b.block_hash[-4:] == STATE.to_bytes(4, "big")
    }
    victims = pool.get_new_blocks(pool.get_num_free_blocks() - 2)
    assert states_before & {b.block_id for b in victims}
    # With the budget: the same number of blocks can be allocated; the
    # shared queue supplies them all and no parked state is a victim.
    pool, lru = budgeted.block_pool, _lru(budgeted)
    parked_before = set(lru.lru_key)
    assert len(parked_before) >= 3
    n = pool.free_block_queue.num_free_blocks
    victims = pool.get_new_blocks(n)
    assert not parked_before & {b.block_id for b in victims}
    assert lru.stats["spills"] == 0
    _check_accounting(budgeted)
    pool.free_blocks(victims)
    _check_accounting(budgeted)


def test_budget_evictions_take_only_state_victims():
    manager = _manager(budget=1)
    a = _request("a", list(range(40)))
    _run(manager, a)
    a_kv = [
        b.block_id
        for b in manager.block_pool.blocks
        if b.block_hash is not None and b.block_hash[-4:] == ATTN.to_bytes(4, "big")
    ]
    # A second session parks its checkpoint; the budget (one block) evicts
    # the oldest checkpoint, a's.
    b = _request("b", list(range(3000, 3040)))
    _run(manager, b)
    lru = _lru(manager)
    assert lru.stats["budget_evictions"] >= 1
    assert _lookup(manager, a, 32, STATE) is None
    assert _lookup(manager, b, 32, STATE) is not None
    assert lru.lru_blocks <= 1
    # a's KV was not chosen as a budget victim; with no checkpoint left on
    # its prefix it is released (hashless, reuse-first), not kept stranded.
    for block_id in a_kv:
        block = manager.block_pool.blocks[block_id]
        assert block.block_hash is None and block.ref_cnt == 0
    assert set(a_kv) <= set(_queue_ids(manager)[: len(a_kv) + 8])
    _check_accounting(manager)


def test_lost_checkpoint_degrades_to_previous_boundary():
    manager = _manager(budget=16)
    turn1 = _conversation()
    _run(manager, turn1)
    turn2 = _next_turn(turn1, "t2", 30)  # prompt 74 -> checkpoint at 64
    assert _run(manager, turn2) == 32
    ckpt64 = _lookup(manager, turn2, 64, STATE)
    assert ckpt64 is not None and _lookup(manager, turn2, 32, STATE) is not None

    # Lose the newest state (any eviction path).
    manager.block_pool.evict_blocks({ckpt64.block_id})
    _check_accounting(manager)
    # Its own KV (48, 64] is released; the shared prefix [0, 32] is kept.
    for tokens in (48, 64):
        assert _lookup(manager, turn2, tokens, ATTN) is None
    for tokens in (16, 32):
        assert _lookup(manager, turn2, tokens, ATTN) is not None

    turn3 = _next_turn(turn2, "t3", 20)
    assert _run(manager, turn3) == 32


def test_lost_branch_checkpoint_keeps_the_sibling_branch():
    """Two forks share (32, 48]; losing one fork's state keeps the other's."""
    manager = _manager(budget=16)
    turn1 = _conversation()
    _run(manager, turn1)
    shared = list(turn1.all_token_ids) + list(range(6000, 6012))  # to 56
    fork_a = _request("a", shared + list(range(7000, 7020)))  # ckpt 64
    fork_b = _request("b", shared + list(range(8000, 8020)))  # ckpt 64'
    assert _run(manager, fork_a) == 32
    assert _run(manager, fork_b) == 32
    assert fork_a.block_hashes[2] == fork_b.block_hashes[2]  # 48 shared
    assert fork_a.block_hashes[3] != fork_b.block_hashes[3]
    ckpt_a = _lookup(manager, fork_a, 64, STATE)
    manager.block_pool.evict_blocks({ckpt_a.block_id})
    # a's own KV at 64 is released; the shared 48 stays for b.
    assert _lookup(manager, fork_a, 64, ATTN) is None
    assert _lookup(manager, fork_b, 48, ATTN) is not None
    assert _run(manager, _next_turn(fork_b, "b2", 20)) == 64
    assert _run(manager, _next_turn(fork_a, "a2", 20)) == 32
    _check_accounting(manager)


def test_lost_checkpoint_degrades_to_zero_without_the_tree_baseline():
    """Control: a budget eviction of the only checkpoint frees everything."""
    manager = _manager(budget=16)
    turn1 = _conversation()
    _run(manager, turn1)
    ckpt = _lookup(manager, turn1, 32, STATE)
    manager.block_pool.evict_blocks({ckpt.block_id})
    for tokens in (16, 32):
        assert _lookup(manager, turn1, tokens, ATTN) is None
    assert _run(manager, _next_turn(turn1, "t2", 20)) == 0
    _check_accounting(manager)


def test_lost_kv_unparks_the_checkpoints_behind_it():
    manager = _manager(budget=16)
    turn1 = _conversation()
    _run(manager, turn1)
    turn2 = _next_turn(turn1, "t2", 30)
    _run(manager, turn2)
    lru = _lru(manager)
    assert lru.lru_blocks >= 2
    # Evict one KV block inside turn1's prefix: neither checkpoint can be
    # reached any more, so both leave the state LRU and their KV is freed.
    kv16 = _lookup(manager, turn1, 16, ATTN)
    manager.block_pool.evict_blocks({kv16.block_id})
    assert lru.stats["kv_kills"] >= 1
    assert _lookup(manager, turn1, 32, STATE) is None
    assert _lookup(manager, turn2, 64, STATE) is None
    assert lru.lru_blocks == 0
    assert _lookup(manager, turn2, 64, ATTN) is None
    assert not lru.nodes and not lru.hash_nodes
    _check_accounting(manager)


def test_kv_loss_past_a_checkpoint_keeps_it():
    manager = _manager(budget=16)
    turn1 = _conversation()
    _run(manager, turn1)
    turn2 = _next_turn(turn1, "t2", 30)
    _run(manager, turn2)
    # A KV block in (32, 64] only turn2's checkpoint needs.
    kv48 = _lookup(manager, turn2, 48, ATTN)
    manager.block_pool.evict_blocks({kv48.block_id})
    assert _lookup(manager, turn2, 64, STATE) is None
    assert _lookup(manager, turn1, 32, STATE) is not None
    assert _run(manager, _next_turn(turn2, "t3", 20)) == 32
    _check_accounting(manager)


def test_state_lru_spills_only_when_the_queue_is_exhausted():
    manager = _manager(budget=64)
    for i in range(4):
        _run(manager, _request(f"s{i}", list(range(i * 1000, i * 1000 + 40))))
    pool, lru = manager.block_pool, _lru(manager)
    parked = lru.lru_blocks
    assert parked >= 4
    everything = pool.get_new_blocks(pool.get_num_free_blocks())
    assert len(everything) == NUM_BLOCKS - 1
    assert lru.lru_blocks == 0 and lru.stats["spills"] >= 4
    pool.free_blocks(everything)
    _check_accounting(manager)


def test_reset_prefix_cache_returns_parked_blocks():
    manager = _manager(budget=8)
    _run(manager, _conversation())
    assert _lru(manager).lru_blocks
    assert manager.reset_prefix_cache()
    assert _lru(manager).lru_blocks == 0
    assert len(_queue_ids(manager)) == NUM_BLOCKS - 1
    assert all(b.block_hash is None for b in manager.block_pool.blocks)


def test_long_session_churn_keeps_invariants():
    manager = _manager(budget=6)
    sessions = [_request(f"c{i}", list(range(i * 7000, i * 7000 + 40))) for i in range(5)]
    for step in range(4):
        for i, prev in enumerate(sessions):
            nxt = _next_turn(prev, f"c{i}-{step}", 17 + 3 * i)
            _run(manager, nxt)
            sessions[i] = nxt
            _check_accounting(manager)
    lru = _lru(manager)
    assert lru.lru_blocks <= 6
    # Every tracked node is either alive or an ancestor of one.
    for key, node in lru.nodes.items():
        assert node.children or lru._state_alive(key)


def test_budget_zero_installs_nothing(monkeypatch):
    original = KVCacheManager.__init__
    monkeypatch.setenv("VLLM_TPU_MAMBA_LRU_BUDGET", "0")
    mamba_state_lru.patch_mamba_state_lru()
    assert KVCacheManager.__init__ is original
    plain, after = _manager(), _manager()
    assert not hasattr(after.block_pool, "_tpu_mamba_state_lru")
    assert "free_blocks" not in vars(after.block_pool)
    hits = []
    for manager in (plain, after):
        prev = _conversation()
        got = [_run(manager, prev)]
        for i in range(3):
            prev = _next_turn(prev, f"t{i + 2}", 25)
            got.append(_run(manager, prev))
        hits.append(got)
    assert hits[0] == hits[1]
    assert _queue_ids(plain) == _queue_ids(after)


def test_budget_env_installs_on_new_managers(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_MAMBA_LRU_BUDGET", "5")
    mamba_state_lru.patch_mamba_state_lru()
    manager = _manager()
    assert _lru(manager).budget == 5
    mamba_state_lru.patch_mamba_state_lru()  # idempotent
    assert getattr(KVCacheManager.__init__, mamba_state_lru._PATCH_FLAG)
