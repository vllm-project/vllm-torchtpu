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
"""Host-only tests for HybridStateRetention free-queue reordering and
cross-turn Mamba boundary + partial-tail LRU retention."""

from types import SimpleNamespace
from typing import NamedTuple

import pytest

from vllm_torchtpu.core.hybrid_retention import HybridStateRetention

BS = 256  # attention / mamba block size
HB = 128  # hash block size (fine-grained partial hits)


class FullAttentionSpec:
    pass


class MambaSpec:
    pass


class SpecGroup(NamedTuple):
    spec: object
    group_ids: list
    manager_cls: type
    use_eagle: bool


class Block:
    def __init__(self, bid, block_hash=None, is_null=False):
        self.block_id = bid
        self.block_hash = block_hash
        self.ref_cnt = 0
        self.is_null = is_null

    def __repr__(self):
        return f"B{self.block_id}"


class FreeQueue:
    def __init__(self):
        self.q = []

    def remove(self, b):
        self.q.remove(b)

    def append(self, b):
        self.q.append(b)

    def prepend_n(self, bs):
        self.q[:0] = bs

    def popleft_n(self, n):
        out = self.q[:n]
        del self.q[:n]
        return out


class Pool:
    def __init__(self):
        self.free_block_queue = FreeQueue()
        self.cached = {}  # (hash_idx, gid) -> block
        self.null = Block(-1, is_null=True)

    def get_num_free_blocks(self):
        return len(self.free_block_queue.q)

    def get_new_blocks(self, num_blocks):
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")
        ret = self.free_block_queue.popleft_n(num_blocks)
        for b in ret:
            assert b.ref_cnt == 0
            b.ref_cnt += 1
        return ret

    def touch(self, blocks):
        for b in blocks:
            if b.ref_cnt == 0 and not b.is_null:
                self.free_block_queue.remove(b)
            b.ref_cnt += 1

    def free_blocks(self, blocks):
        first, last = [], []
        for b in blocks:
            b.ref_cnt -= 1
            if b.ref_cnt == 0 and not b.is_null:
                (first if b.block_hash is None else last).append(b)
        self.free_block_queue.prepend_n(first)
        for b in last:
            self.free_block_queue.append(b)


POOL = None


class FakeMambaManager:
    @classmethod
    def find_longest_cache_hit(cls, block_hashes, max_length, kv_cache_group_ids,
                               block_pool, kv_cache_spec, drop_eagle_block,
                               alignment_tokens, dcp_world_size=1, pcp_world_size=1):
        for fine_idx in range(min(max_length // HB, len(block_hashes)) - 1, -1, -1):
            found = [block_pool.cached.get((block_hashes[fine_idx], g)) for g in kv_cache_group_ids]
            if all(found):
                n = (fine_idx * HB) // BS
                return tuple([block_pool.null] * n + [b] for b in found), (fine_idx + 1) * HB
        return tuple([] for _ in kv_cache_group_ids), 0


class FakeAttnManager:
    pass


def _setup(prompt=1000, osl=600, mamba=True, max_conversations=None, bs=BS):
    pool = Pool()
    attn_mgr = SimpleNamespace(block_size=bs, req_to_blocks={}, dcp_world_size=1)
    mamba_mgr = SimpleNamespace(block_size=bs, req_to_blocks={}, dcp_world_size=1)
    groups = [SpecGroup(FullAttentionSpec(), [0], FakeAttnManager, False)]
    managers = [attn_mgr]
    if mamba:
        groups.append(SpecGroup(MambaSpec(), [1], FakeMambaManager, False))
        managers.append(mamba_mgr)
    coord = SimpleNamespace(attention_groups=groups, single_type_managers=managers,
                            block_pool=pool, _cache_hit_alignment_tokens=HB)
    kvm = SimpleNamespace(coordinator=coord, enable_caching=True)
    total = prompt + osl
    req = SimpleNamespace(request_id="r", num_prompt_tokens=prompt, num_tokens=total,
                          num_computed_tokens=total, block_hashes=list(range(total // HB)))
    # Attention blocks: all full blocks hashed, owned by the request.
    attn = [Block(i, block_hash=("a", i)) for i in range(-(-total // bs))]
    for b in attn:
        b.ref_cnt = 1
    attn_mgr.req_to_blocks["r"] = attn
    # Boundary snapshot at floor128(prompt) (partial tail), already released.
    state = Block(100, block_hash=("s", prompt // HB - 1))
    pool.cached[(prompt // HB - 1, 1)] = state
    other = Block(200, block_hash=("x", 0))  # someone else's cached block
    pool.free_block_queue.q = [state, other]  # state is OLD (freed at prefill end)
    ret = HybridStateRetention(kvm, max_conversations=max_conversations)
    return ret, pool, req, attn, state, other


def _scheduler_free(pool, attn):
    # vLLM frees a finished request's blocks tail-first (reversed).
    pool.free_blocks(list(reversed(attn)))


def test_disabled_without_mamba():
    r, *_ = _setup(mamba=False)
    assert not r.enabled


def test_pin_then_release_puts_state_after_attention_and_demotes_output():
    r, pool, req, attn, state, other = _setup(prompt=1000, osl=600)
    r.pin_running([req])
    assert state.ref_cnt == 1 and state not in pool.free_block_queue.q
    snap = r.snapshot_attention_blocks(req)
    _scheduler_free(pool, attn)
    r.on_request_freed(req, snap)
    q = pool.free_block_queue.q
    # Hit = floor128(1000) = 896 -> attention blocks 0..3 needed (+1 eagle) ->
    # blocks >= 5 are unreachable output blocks and go to the head.
    hit_blocks = -(-896 // BS) + 1
    dead = attn[hit_blocks:]
    assert set(q[: len(dead)]) == set(dead) and len(dead) == 2
    assert q.index(other) < q.index(attn[0])
    # Cross-turn LRU keeps the boundary snapshot pinned until evicted.
    assert state.ref_cnt == 1 and state not in q
    r.evict_all_conversations()
    assert q[-1] is state  # snapshot is MRU: evicted after its attention blocks
    assert state.ref_cnt == 0


def test_unreleased_state_is_not_pinned_until_released():
    r, pool, req, attn, state, other = _setup()
    pool.free_block_queue.q.remove(state)
    state.ref_cnt = 1  # still owned (CoW retention / skipped-block not yet freed)
    r.pin_running([req])
    assert state.ref_cnt == 1 and "r" not in r._pinned
    pool.free_blocks([state])  # owner releases it
    r.pin_running([req])
    assert state.ref_cnt == 1 and "r" in r._pinned


def test_refresh_without_pin_moves_state_to_mru():
    r, pool, req, attn, state, other = _setup()
    snap = r.snapshot_attention_blocks(req)
    _scheduler_free(pool, attn)
    r.on_request_freed(req, snap)
    assert state.ref_cnt == 1 and state not in pool.free_block_queue.q
    r.evict_all_conversations()
    assert pool.free_block_queue.q[-1] is state


def test_prefilling_request_is_not_probed():
    r, pool, req, attn, state, other = _setup()
    req.num_computed_tokens = 500
    r.pin_running([req])
    assert state.ref_cnt == 0 and "r" not in r._missed


def test_junction_hit_does_not_demote():
    r, pool, req, attn, state, other = _setup(prompt=4000, osl=600)
    # Only an old shared-prefix junction state exists (at 512 tokens).
    pool.cached.clear()
    pool.free_block_queue.q.remove(state)
    junction = Block(300, block_hash=("j", 3))
    pool.cached[(3, 1)] = junction
    pool.free_block_queue.q.insert(0, junction)
    snap = r.snapshot_attention_blocks(req)
    _scheduler_free(pool, attn)
    r.on_request_freed(req, snap)
    q = pool.free_block_queue.q
    assert q[0] is junction  # untouched
    assert q[-1] is attn[0]  # attention order unchanged (tail-first free)


def test_missing_state_gives_up_after_bounded_probes():
    r, pool, req, attn, state, other = _setup()
    pool.cached.clear()
    for _ in range(10):
        r.pin_running([req])
    assert "r" in r._gave_up and r._missed["r"] == 3


def test_dense_retention_demotes_nothing():
    r, pool, req, attn, state, other = _setup(prompt=1000, osl=600)
    # A decode-time snapshot at the last full hash unit is cached too.
    last = len(req.block_hashes) - 1
    pool.cached[(last, 1)] = Block(400, block_hash=("s", last))
    pool.free_block_queue.q.append(pool.cached[(last, 1)])
    snap = r.snapshot_attention_blocks(req)
    _scheduler_free(pool, attn)
    r.on_request_freed(req, snap)
    q = pool.free_block_queue.q
    assert q.index(other) < min(q.index(b) for b in attn)


def test_pins_both_full_boundary_and_partial_tail_blocks():
    # BS=640, HB=128, prompt=800 -> full boundary at 640 (idx 4), partial tail at 768 (idx 5).
    r, pool, req, attn, partial_state, other = _setup(prompt=800, osl=200, bs=640)
    full_state = Block(99, block_hash=("s_full", 640 // HB - 1))
    pool.cached[(640 // HB - 1, 1)] = full_state
    pool.free_block_queue.q.insert(0, full_state)

    # Suppose full_state is released at prefill end, while partial_state is held 1 step by CoW.
    pool.free_block_queue.q.remove(partial_state)
    partial_state.ref_cnt = 1

    r.pin_running([req])
    # full_state is pinned immediately; req stays in _pending_partial waiting for partial_state.
    assert full_state.ref_cnt == 1 and full_state not in pool.free_block_queue.q
    assert "r" in r._pinned and "r" in r._pending_partial

    # Next step: CoW retention releases partial_state.
    pool.free_blocks([partial_state])
    r.pin_running([req])
    assert partial_state.ref_cnt == 1 and partial_state not in pool.free_block_queue.q
    assert r._pinned["r"][0] == [full_state, partial_state]
    assert "r" not in r._pending_partial

    # On request finish, both blocks transfer to _conv_lru (max 2 blocks).
    snap = r.snapshot_attention_blocks(req)
    _scheduler_free(pool, attn)
    r.on_request_freed(req, snap)
    assert full_state.ref_cnt == 1 and partial_state.ref_cnt == 1
    conv_entry = next(iter(r._conv_lru.values()))
    assert conv_entry.blocks == [full_state, partial_state]


def test_cross_turn_lru_replaces_old_turn_and_auto_evicts_on_get_new_blocks():
    r, pool, req1, attn1, state1, other = _setup(prompt=1000, osl=600, max_conversations=2)
    req1.request_id = "convA_turn1"
    r.coordinator.single_type_managers[0].req_to_blocks["convA_turn1"] = attn1

    r.pin_running([req1])
    snap1 = r.snapshot_attention_blocks(req1)
    _scheduler_free(pool, attn1)
    r.on_request_freed(req1, snap1)
    assert state1.ref_cnt == 1 and len(r._conv_lru) == 1

    # Turn 2 of convA finishes at a deeper boundary: state1 is unpinned, state2 is retained.
    prompt2, total2 = 2000, 2400
    req2 = SimpleNamespace(
        request_id="convA_turn2",
        num_prompt_tokens=prompt2,
        num_tokens=total2,
        num_computed_tokens=total2,
        block_hashes=list(range(total2 // HB)),
    )
    state2 = Block(102, block_hash=("s", prompt2 // HB - 1))
    pool.cached[(prompt2 // HB - 1, 1)] = state2
    pool.free_block_queue.q.append(state2)
    r.pin_running([req2])
    r.on_request_freed(req2, [])
    assert state1.ref_cnt == 0 and state1 in pool.free_block_queue.q
    assert state2.ref_cnt == 1 and len(r._conv_lru) == 1

    # Allocating all blocks in the pool (+1 for state2) automatically evicts state2 from _conv_lru.
    total_available_with_lru = pool.get_num_free_blocks() + 1
    allocated = pool.get_new_blocks(total_available_with_lru)
    assert len(allocated) == total_available_with_lru
    assert len(r._conv_lru) == 0
    assert state2 in allocated
