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


def test_safe_prompt_tail_offset_handles_qwen_think_4_token_overwrite():
    # prompt=3201 (3201 % 640 == 1 < 4): eff_prompt=3197 -> full=2560, fine=3072, pre_full=1920.
    r, pool, req, attn, _, _ = _setup(prompt=3201, osl=200, bs=640)
    req.shared_prefix_boundary = 0
    pre_state = Block(301, block_hash=("s_pre", 1920 // HB - 1))
    full_state = Block(302, block_hash=("s_full", 2560 // HB - 1))
    fine_state = Block(303, block_hash=("s_fine", 3072 // HB - 1))
    pool.cached[(1920 // HB - 1, 1)] = pre_state
    pool.cached[(2560 // HB - 1, 1)] = full_state
    pool.cached[(3072 // HB - 1, 1)] = fine_state
    pool.free_block_queue.q.extend([pre_state, full_state, fine_state])

    # 1. Prefill split stops at pre_full=1920, then full=2560, then fine=3072.
    r.prepare_prefill_split(req, 0, 3201)
    assert req.shared_prefix_boundary == 1920
    assert r.adjust_prefill_split(req, 0, 3200) == 1920

    r.prepare_prefill_split(req, 1920, 1281)
    assert req.shared_prefix_boundary == 2560
    assert r.adjust_prefill_split(req, 1920, 1280) == 640

    assert r.adjust_prefill_split(req, 2560, 640) == 512  # 2560 -> 3072

    # 2. Pin running and release retains untainted full_state (2560) + fine_state (3072)
    #    in _conv_lru and pre_state (1920) in _fork_lru.
    r.pin_running([req])
    assert r._pinned["r"][0] == [full_state, fine_state]
    assert r._fork_pinned["r"][0] == [pre_state]

    snap = r.snapshot_attention_blocks(req)
    _scheduler_free(pool, attn)
    r.on_request_freed(req, snap)
    assert full_state.ref_cnt == 1 and fine_state.ref_cnt == 1
    assert pre_state.ref_cnt == 1
    assert len(r._conv_lru) == 1 and len(r._fork_lru) == 1


def test_smart_fork_checkpoints_pre_final_block_and_promotes_root_anchor():
    r, pool, req1, attn1, _, _ = _setup(prompt=19900, osl=200, bs=640)
    req1.request_id = "req-0"
    r.coordinator.single_type_managers[0].req_to_blocks[req1.request_id] = attn1
    pre_19200 = Block(501, block_hash=("s", 19200 // HB - 1))
    full_19840 = Block(502, block_hash=("s", 19840 // HB - 1))
    pool.cached[(19200 // HB - 1, 1)] = pre_19200
    pool.cached[(19840 // HB - 1, 1)] = full_19840
    pool.free_block_queue.q.extend([pre_19200, full_19840])

    r.pin_running([req1])
    snap1 = r.snapshot_attention_blocks(req1)
    _scheduler_free(pool, attn1)
    r.on_request_freed(req1, snap1)
    assert pre_19200.ref_cnt == 1 and full_19840.ref_cnt == 1

    # When subagent_001 advances through turn1 and turn2, turn0's root full_19840
    # is promoted into _fork_lru rather than unpinned.
    for t_idx, prm in [(1, 21200), (2, 22500)]:
        tot = prm + 200
        req_t = SimpleNamespace(
            request_id=f"req-{t_idx}",
            num_prompt_tokens=prm,
            num_tokens=tot,
            num_computed_tokens=tot,
            block_hashes=list(range(tot // HB)),
        )
        f640 = (prm // 640) * 640
        st = Block(510 + t_idx, block_hash=("s", f640 // HB - 1))
        pool.cached[(f640 // HB - 1, 1)] = st
        pool.free_block_queue.q.append(st)
        r.pin_running([req_t])
        r.on_request_freed(req_t, [])

    assert full_19840.ref_cnt == 1
    assert any(full_19840 in e.blocks for e in r._fork_lru.values())


def test_compact_prior_turn_retains_full_block_and_frees_partial_tail():
    # Turn 1 has full_state (640) + partial_state (768).
    r, pool, req1, attn1, partial1, _ = _setup(prompt=800, osl=200, bs=640)
    req1.request_id = "convB_turn1"
    r.coordinator.single_type_managers[0].req_to_blocks["convB_turn1"] = attn1
    full1 = Block(601, block_hash=("s_full1", 640 // HB - 1))
    pool.cached[(640 // HB - 1, 1)] = full1
    pool.free_block_queue.q.insert(0, full1)

    r.pin_running([req1])
    snap1 = r.snapshot_attention_blocks(req1)
    _scheduler_free(pool, attn1)
    r.on_request_freed(req1, snap1)
    assert full1.ref_cnt == 1 and partial1.ref_cnt == 1
    assert r.num_pinned_blocks() == 2

    # Turn 2 of convB finishes at 1440 (full2=1280, partial2=1408):
    # Turn 1 is compacted in-place -> partial1 is freed (ref_cnt=0), full1 stays pinned (ref_cnt=1).
    prm2, tot2 = 1440, 1640
    req2 = SimpleNamespace(
        request_id="convB_turn2",
        num_prompt_tokens=prm2,
        num_tokens=tot2,
        num_computed_tokens=tot2,
        block_hashes=list(range(tot2 // HB)),
    )
    full2 = Block(602, block_hash=("s_full2", 1280 // HB - 1))
    partial2 = Block(603, block_hash=("s_part2", 1408 // HB - 1))
    pool.cached[(1280 // HB - 1, 1)] = full2
    pool.cached[(1408 // HB - 1, 1)] = partial2
    pool.free_block_queue.q.extend([full2, partial2])

    r.pin_running([req2])
    r.on_request_freed(req2, [])
    assert partial1.ref_cnt == 0 and partial1 in pool.free_block_queue.q
    assert full1.ref_cnt == 1
    assert full2.ref_cnt == 1 and partial2.ref_cnt == 1
    assert r.num_pinned_blocks() == 3

    # Enforcing max_pinned_blocks=2 evicts the compacted prior-turn entry (full1) first,
    # keeping the active latest turn (full2, partial2) intact.
    r.max_pinned_blocks = 2
    r._enforce_pinned_block_cap()
    assert full1.ref_cnt == 0
    assert full2.ref_cnt == 1 and partial2.ref_cnt == 1
    assert r.num_pinned_blocks() == 2


def test_mamba_manager_cache_partial_tail_hook_with_safe_tail_offset():
    pool = Pool()
    attn_mgr = SimpleNamespace(block_size=640, req_to_blocks={}, dcp_world_size=1)
    partial_blk = Block(701, block_hash=("p", 2688 // HB - 1))

    class HookableMambaMgr:
        def __init__(self):
            self.block_size = 640
            self.dcp_world_size = 1
            self.req_to_blocks = {"r_safe": [Block(700)] * 4 + [partial_blk]}
            self._partial_hit_reqs = {}

        def _cache_partial_tail_block(self, request, num_tokens):
            latest = (request.num_prompt_tokens // HB) * HB
            if num_tokens != latest:
                return None
            self._partial_hit_reqs[request.request_id] = (4, partial_blk)
            return ("hash", num_tokens)

    mamba_mgr = HookableMambaMgr()
    groups = [
        SpecGroup(FullAttentionSpec(), [0], FakeAttnManager, False),
        SpecGroup(MambaSpec(), [1], FakeMambaManager, False),
    ]
    coord = SimpleNamespace(
        attention_groups=groups,
        single_type_managers=[attn_mgr, mamba_mgr],
        block_pool=pool,
        _cache_hit_alignment_tokens=HB,
    )
    kvm = SimpleNamespace(coordinator=coord, enable_caching=True)
    r = HybridStateRetention(kvm)

    # prompt=2817 (2817 % 128 == 1 < 4) -> safe fine boundary is 2688, tainted boundary is 2816.
    req = SimpleNamespace(
        request_id="r_safe",
        num_prompt_tokens=2817,
        num_tokens=2817,
        num_computed_tokens=2688,
        block_hashes=list(range(2817 // HB)),
    )
    assert mamba_mgr._cache_partial_tail_block(req, 2688) == ("hash", 2688)
    assert req.num_prompt_tokens == 2817  # restored after hook
    assert r._tracked_blocks["r_safe"][(1, 2688)] == (partial_blk, 2688)
    # Tainted 2816 boundary is ignored and does not overwrite 2688.
    assert mamba_mgr._cache_partial_tail_block(req, 2816) is None




# ---------------------------------------------------------------- v20: fork
# candidate and fork key decisions use cache state only, never the request id.


def _fork_setup(prompt=6500, bs=640):
    r, pool, req, attn, _, _ = _setup(prompt=prompt, osl=200, bs=bs)
    req.shared_prefix_boundary = 0
    return r, pool, req, attn


def _req(rid, prompt, osl=200, hashes=None):
    tot = prompt + osl
    return SimpleNamespace(
        request_id=rid,
        num_prompt_tokens=prompt,
        num_tokens=tot,
        num_computed_tokens=0,
        shared_prefix_boundary=0,
        block_hashes=list(hashes if hashes is not None else range(tot // HB)),
    )


def _finish(r, pool, req, states):
    """Run a request's lifecycle after prefill: its Mamba snapshots
    (``states``: token offset -> block) are cached and released, it decodes
    and finishes."""
    for off, blk in states.items():
        pool.cached[(req.block_hashes[off // HB - 1], 1)] = blk
        pool.free_block_queue.q.append(blk)
    req.num_computed_tokens = req.num_tokens
    req.is_finished = lambda: True
    r.pin_running([req])
    r.on_request_freed(req, [])


def _decisions_and_keys(rids):
    """Replay the same cache-state sequence with the given request ids:
    a cold root (prompt 6500), its continuation (extends the root's tail),
    a second request extending the root's tail, and a request that shares
    only the root's first 3840 tokens."""
    r, pool, _, _ = _fork_setup()
    out = []
    bid = iter(range(1000, 2000))

    root = _req(rids[0], 6500)
    out.append(r._is_fork_candidate(root, 0))
    r.prepare_prefill_split(root, 0, 6500)
    out.append(root.shared_prefix_boundary)
    _finish(r, pool, root, {o: Block(next(bid), block_hash=("s", o))
                            for o in (5760, 6400)})

    cont = _req(rids[1], 8000)  # same prefix as root, more tokens
    out.append(r._is_fork_candidate(cont, 6400))
    sib = _req(rids[2], 9000)  # also extends root's tail (6400)
    out.append(r._is_fork_candidate(sib, 6400))
    hashes = list(range(30)) + [("other", i) for i in range(30, 60)]
    inner = _req(rids[3], 7300, hashes=hashes)
    out.append(r._is_fork_candidate(inner, 3840))
    for q, stops in ((cont, (7680, 7936)), (sib, (8960,))):
        out.append(r.prefill_checkpoint_stops(q, 6400))
        _finish(r, pool, q, {o: Block(next(bid), block_hash=("s", o))
                             for o in stops})
    return out, list(r._fork_lru.keys()), list(r._conv_lru.keys())


def test_fork_decisions_and_keys_independent_of_request_id_format():
    agentx_like = [
        "rootA_turn0",
        "rootA_turn1",
        "rootA::sa:subagent_001_103f4c24_turn0",
        "rootA::aux:001_turn0",
    ]
    generic = [
        "chatcmpl-1f0e6c5a",
        "chatcmpl-9b2d7e41",
        "chatcmpl-77aa0c13",
        "cmpl-0c4e8d2f-0",
    ]
    a = _decisions_and_keys(agentx_like)
    b = _decisions_and_keys(generic)
    assert a == b
    decisions, fork_keys, _ = a
    # cold root (start == 0): candidate, checkpoints pre_full = 6400 - 640.
    assert decisions[0] is True and decisions[1] == 5760
    # partial prefix hits (start > 0): not fork candidates.
    assert decisions[2:5] == [False, False, False]
    assert decisions[5] == [7680, 7936]
    assert decisions[6] == [8960]
    # Fork anchors are keyed by block hash: root's pre_full 5760 (idx 44)
    # and root's tail 6400 (idx 49), promoted from the conversation LRU once
    # it fell beyond keep_prior.
    assert fork_keys == [
        ("fork_hash", 44, 44), ("fork_hash", 49, 49)
    ]


def test_fork_decision_is_taken_at_first_chunk_and_dropped_at_finish():
    r, pool, _, _ = _fork_setup()
    req = _req("x", 6500)
    assert r._is_fork_candidate(req, 0) is True
    # Later chunks (start > 0) reuse the first-chunk decision.
    assert r._is_fork_candidate(req, 2560) is True
    assert r.prefill_checkpoint_stops(req, 2560)[0] == 5760
    # Preemption keeps it; finishing drops it.
    req.is_finished = lambda: False
    r.on_request_freed(req, [])
    assert r._fork_decision == {"x": True}
    req.is_finished = lambda: True
    r.on_request_freed(req, [])
    assert r._fork_decision == {}


def test_fork_candidate_off_without_smart_fork():
    r, pool, _, _ = _fork_setup()
    r.smart_fork = False
    assert r._is_fork_candidate(_req("y", 6500), 0) is False
    assert r._fork_decision == {}
