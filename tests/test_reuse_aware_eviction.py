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
"""Reuse-aware eviction: decode-output blocks first, 2-segment LRU."""

import collections
import random
from types import SimpleNamespace

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

from vllm_torchtpu.core import reuse_aware_eviction as rae
from vllm_torchtpu.offload import contiguous_alloc

BS = 16


class FakeManager:
    """The slice of KVCacheManager reuse-aware eviction uses."""

    def __init__(self, num_blocks):
        self.block_pool = BlockPool(
            num_gpu_blocks=num_blocks, enable_caching=True, hash_block_size=BS
        )
        self.stm = SimpleNamespace(req_to_blocks={}, block_size=BS)
        self.coordinator = SimpleNamespace(single_type_managers=[self.stm])
        self._n = 0

    def alloc(self, req_id, n, num_prompt_tokens):
        blocks = self.block_pool.get_new_blocks(n)
        for b in blocks:
            self._n += 1
            b.set_block_hash(
                make_block_hash_with_group_id(self._n.to_bytes(8, "little"), 0)
            )
        self.stm.req_to_blocks[req_id] = blocks
        return SimpleNamespace(request_id=req_id, num_prompt_tokens=num_prompt_tokens)

    def hit(self, req_id, blocks, num_prompt_tokens):
        self.block_pool.touch(blocks)
        self.stm.req_to_blocks[req_id] = list(blocks)
        return SimpleNamespace(request_id=req_id, num_prompt_tokens=num_prompt_tokens)

    def free(self, request):
        blocks = self.stm.req_to_blocks.pop(request.request_id)
        self.block_pool.free_blocks(reversed(blocks))


def _env(monkeypatch, decode_first=False, hits=0, frac=0.3):
    monkeypatch.setenv("VLLM_TPU_EVICT_DECODE_FIRST", "1" if decode_first else "0")
    monkeypatch.setenv("VLLM_TPU_EVICT_PROTECT_HITS", str(hits))
    monkeypatch.setenv("VLLM_TPU_EVICT_PROTECTED_FRAC", str(frac))


def _ids(queue):
    return [b.block_id for b in queue.iter_blocks_after(None)]


def _check(queue):
    """Invariants: count matches the list, links are consistent, and the
    protected ids are exactly the blocks from the boundary to the tail."""
    blocks = list(queue.iter_blocks_after(None))
    assert len(blocks) == queue.num_free_blocks
    prev = queue.fake_free_list_head
    for b in blocks:
        assert b.prev_free_block is prev
        prev = b
    assert queue.fake_free_list_tail.prev_free_block is prev
    tail_part = []
    b = queue.boundary
    while b is not queue.fake_free_list_tail:
        tail_part.append(b.block_id)
        b = b.next_free_block
    assert set(tail_part) == queue.protected_ids
    assert len(tail_part) == len(queue.protected_ids)
    assert len(queue.protected_ids) <= max(queue.protected_cap, 0) or not tail_part


def test_off_installs_nothing(monkeypatch):
    _env(monkeypatch)
    m = FakeManager(8)
    assert not rae.enabled()
    assert rae.install(m) is None
    assert type(m.block_pool.free_block_queue).__name__ == "FreeKVCacheBlockQueue"


def test_decode_blocks_go_to_the_head(monkeypatch):
    _env(monkeypatch, decode_first=True)
    m = FakeManager(12)
    q = rae.install(m)
    assert isinstance(q, rae.SegmentedFreeBlockQueue)
    a = m.alloc("a", 4, num_prompt_tokens=2 * BS + 5)  # blocks 2, 3 hold output
    b = m.alloc("b", 3, num_prompt_tokens=3 * BS)  # block 2 is the first decode block
    a_ids = [x.block_id for x in m.stm.req_to_blocks["a"]]
    b_ids = [x.block_id for x in m.stm.req_to_blocks["b"]]
    m.free(a)
    m.free(b)
    order = _ids(q)
    # b's prompt blocks are all full (3 * BS): none of its blocks is decode.
    assert order[:2] == [a_ids[3], a_ids[2]]
    assert order[-5:] == [a_ids[1], a_ids[0], b_ids[2], b_ids[1], b_ids[0]]
    assert q.decode_ids == set()
    _check(q)
    # The next allocation reuses a's output blocks first (after empty blocks
    # would be equivalent: output blocks never hit).
    got = m.block_pool.get_new_blocks(2)
    assert {x.block_id for x in got} == {a_ids[2], a_ids[3]}


def test_protected_segment_order_and_cap(monkeypatch):
    _env(monkeypatch, hits=1, frac=0.25)  # cap = 4 of 16 blocks
    m = FakeManager(16)
    q = rae.install(m)
    a = m.alloc("a", 3, num_prompt_tokens=3 * BS)
    shared = list(m.stm.req_to_blocks["a"])
    m.free(a)
    one_shot = m.alloc("x", 2, num_prompt_tokens=2 * BS)
    x_ids = [x.block_id for x in m.stm.req_to_blocks["x"]]
    # A fork hits a's blocks: they are protected once freed again.
    f = m.hit("f", shared, num_prompt_tokens=3 * BS)
    m.free(f)
    m.free(one_shot)
    _check(q)
    order = _ids(q)
    s_ids = [x.block_id for x in shared]
    assert order[-3:] == s_ids[::-1]
    assert q.protected_ids == set(s_ids)
    # x's one-shot blocks sit in front of the protected segment.
    assert order[-5:-3] == x_ids[::-1]
    # Allocation exhausts the probationary part before any protected block.
    n_unprotected = q.num_free_blocks - 3
    got = m.block_pool.get_new_blocks(n_unprotected)
    assert not ({x.block_id for x in got} & set(s_ids))
    _check(q)
    got = m.block_pool.get_new_blocks(1)
    assert got[0].block_id == s_ids[-1]
    assert q.hits[s_ids[-1]] == 0  # refilled: hit count restarts
    _check(q)


def test_overflow_demotes_oldest_protected(monkeypatch):
    _env(monkeypatch, hits=1, frac=0.25)  # cap 4
    m = FakeManager(16)
    q = rae.install(m)
    reqs = []
    for i in range(3):
        r = m.alloc(f"r{i}", 2, num_prompt_tokens=2 * BS)
        reqs.append(list(m.stm.req_to_blocks[f"r{i}"]))
        m.free(r)
    for i, blocks in enumerate(reqs):
        m.free(m.hit(f"h{i}", blocks, num_prompt_tokens=2 * BS))
    _check(q)
    assert q.num_protected == 4
    # r0's blocks (protected first) were demoted to the probationary tail.
    assert q.protected_ids == {b.block_id for b in reqs[1] + reqs[2]}
    order = _ids(q)
    assert order[-6:-4] == [b.block_id for b in reqs[0]][::-1]


def test_remove_of_boundary_and_contiguous_alloc(monkeypatch):
    _env(monkeypatch, decode_first=True, hits=1, frac=0.5)
    m = FakeManager(32)
    q = rae.install(m)
    contiguous_alloc.install(m.block_pool, scan_factor=3, max_scan=64)
    rng = random.Random(0)
    live = {}
    cached = []
    for step in range(400):
        op = rng.random()
        if op < 0.45 and m.block_pool.get_num_free_blocks() >= 6:
            n = rng.randint(1, 5)
            rid = f"r{step}"
            hit = []
            if cached and rng.random() < 0.5:
                # Touch still-cached blocks (the pool may have reused some).
                hit = list(
                    {
                        b.block_id: b
                        for b in rng.sample(cached, min(len(cached), 3))
                        if b.block_hash is not None
                    }.values()
                )
            if hit:
                m.block_pool.touch(hit)
            new = m.block_pool.get_new_blocks(n)
            for b in new:
                m._n += 1
                b.set_block_hash(
                    make_block_hash_with_group_id(m._n.to_bytes(8, "little"), 0)
                )
            blocks = hit + new
            m.stm.req_to_blocks[rid] = blocks
            live[rid] = SimpleNamespace(
                request_id=rid, num_prompt_tokens=rng.randint(0, len(blocks) * BS)
            )
        elif live:
            rid = rng.choice(list(live))
            cached.extend(m.stm.req_to_blocks[rid])
            m.free(live.pop(rid))
            cached = cached[-40:]
        _check(q)
    # Every block is either held or free exactly once.
    held = {b.block_id for v in m.stm.req_to_blocks.values() for b in v}
    assert all(m.block_pool.blocks[i].ref_cnt > 0 for i in held)
    free = _ids(q)
    assert len(free) == len(set(free))
    assert not (held & set(free))
    assert len(held) + len(free) == 31  # null block excluded


def test_adopt_keeps_order(monkeypatch):
    _env(monkeypatch, hits=2)
    m = FakeManager(8)
    before = _ids(m.block_pool.free_block_queue)
    q = rae.install(m)
    assert _ids(q) == before
    assert q.num_free_blocks == len(before)
    assert rae.install(m) is q  # idempotent


def test_write_back_holds_prompt_chunks_only_with_decode_first(monkeypatch):
    from tests.offload.test_raiden_lru_refresh import _bare_scheduler, _group

    for decode_first in (False, True):
        _env(monkeypatch, decode_first=decode_first)
        s = _bare_scheduler([_group(0, tokens_per_chunk=256)])
        s.config.blocks_per_chunk = 2
        s.config.offload_prompt_only = True
        s._wb_blocks = collections.OrderedDict()
        s._wb_owners = {}
        s._req_status["r"] = SimpleNamespace(
            req=SimpleNamespace(num_prompt_tokens=2 * 256 + 100),
            group_states=[
                SimpleNamespace(
                    next_stored_chunk_idx=0,
                    offload_keys=[object()] * 4,  # 2 prompt + 2 decode chunks
                    block_ids=list(range(1, 9)),
                )
            ],
        )
        assert s._wb_register("r")
        expected = set(range(1, 5)) if decode_first else set(range(1, 9))
        assert s._wb_blocks["r"] == expected


def test_prefix_aware_scheduler_installs_before_connector(monkeypatch):
    from unittest.mock import patch

    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    from vllm_torchtpu.core.eviction_aware_connector import EvictionAwareConnector
    from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler

    seen = []

    class Conn(EvictionAwareConnector):
        def attach_block_pool(self, block_pool):
            seen.append(type(block_pool.free_block_queue))

    for on in (False, True):
        _env(monkeypatch, decode_first=on)
        m = FakeManager(8)

        def base_init(self, *a, m=m, **kw):
            self.kv_cache_manager = m
            self.connector = Conn()
            self.vllm_config = None

        with patch.object(AsyncScheduler, "__init__", base_init):
            PrefixAwareScheduler()
        queue = m.block_pool.free_block_queue
        assert isinstance(queue, rae.SegmentedFreeBlockQueue) == on
    assert seen[-1] is rae.SegmentedFreeBlockQueue
