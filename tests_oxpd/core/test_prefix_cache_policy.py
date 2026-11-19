# SPDX-License-Identifier: Apache-2.0
"""Eviction-order helpers of core/prefix_cache_policy.py, without vLLM.

The module is loaded by path, and the free queue below reproduces the
linked-list contract of vLLM's FreeKVCacheBlockQueue (fake head and tail,
prev_free_block / next_free_block, num_free_blocks) plus the LIFO free path
TorchTPU installs (hashless blocks prepended, cached blocks appended).
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "vllm_torchtpu"
    / "core"
    / "prefix_cache_policy.py"
)


def _load():
    name = "_test_prefix_cache_policy"
    spec = importlib.util.spec_from_file_location(name, _PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P = _load()


class Block:
    def __init__(self, block_id, block_hash=None, ref_cnt=0, is_null=False, num_tokens=None):
        self.block_id = block_id
        self.block_hash = block_hash
        self.block_hash_num_tokens = num_tokens  # position the hash covers (image vLLM)
        self.ref_cnt = ref_cnt
        self.is_null = is_null
        self.prev_free_block = None
        self.next_free_block = None

    def __repr__(self):
        return f"B{self.block_id}{'h' if self.block_hash else ''}"


class Queue:
    """vLLM's FreeKVCacheBlockQueue contract, minimal."""

    def __init__(self, blocks):
        self.fake_free_list_head = Block(-1)
        self.fake_free_list_tail = Block(-2)
        self.fake_free_list_head.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = self.fake_free_list_head
        self.num_free_blocks = 0
        for b in blocks:
            self.append(b)

    def append(self, b):
        last = self.fake_free_list_tail.prev_free_block
        last.next_free_block = b
        b.prev_free_block = last
        b.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = b
        self.num_free_blocks += 1

    def prepend(self, b):
        first = self.fake_free_list_head.next_free_block
        self.fake_free_list_head.next_free_block = b
        b.prev_free_block = self.fake_free_list_head
        b.next_free_block = first
        first.prev_free_block = b
        self.num_free_blocks += 1

    def popleft(self):
        b = self.fake_free_list_head.next_free_block
        assert b is not self.fake_free_list_tail
        self.fake_free_list_head.next_free_block = b.next_free_block
        b.next_free_block.prev_free_block = self.fake_free_list_head
        b.prev_free_block = b.next_free_block = None
        self.num_free_blocks -= 1
        return b

    def order(self):
        out, node = [], self.fake_free_list_head.next_free_block
        while node is not self.fake_free_list_tail:
            out.append(node)
            node = node.next_free_block
        assert len(out) == self.num_free_blocks
        back, node = [], self.fake_free_list_tail.prev_free_block
        while node is not self.fake_free_list_head:
            back.append(node)
            node = node.prev_free_block
        assert back[::-1] == out, "prev links disagree with next links"
        return out


def lifo_free(queue, blocks):
    """TorchTPU's BlockPool.free_blocks (LIFO free patch)."""
    hashless, cached = [], []
    for b in blocks:
        b.ref_cnt -= 1
        if b.ref_cnt == 0 and not b.is_null:
            (hashless if b.block_hash is None else cached).append(b)
    for b in reversed(hashless):
        queue.prepend(b)
    for b in cached:
        queue.append(b)


# ---------------------------------------------------------------- demote
def test_stale_checkpoint_goes_after_hashless_head_under_pressure():
    h1 = Block(1)
    c = [Block(10 + i, block_hash=f"c{i}") for i in range(3)]
    q = Queue([h1, *c])
    stale = Block(99, block_hash="stale", ref_cnt=1)
    demoted = P.free_evict_first(q, [stale])
    assert demoted == [stale]
    assert q.order() == [h1, stale, *c]
    # Allocation reuses the hashless block first, then evicts the stale one.
    assert q.popleft() is h1
    assert q.popleft() is stale


def test_demote_keeps_given_order_and_skips_referenced_blocks():
    c = [Block(10 + i, block_hash=f"c{i}") for i in range(2)]
    q = Queue(c)
    x = Block(1, block_hash="x", ref_cnt=1)
    y = Block(2, block_hash="y", ref_cnt=1)
    held = Block(3, block_hash="held", ref_cnt=2)  # another request still uses it
    P.free_evict_first(q, [x, held, y])
    assert q.order() == [x, y, *c]
    assert held.ref_cnt == 1 and held.prev_free_block is None


def test_no_pressure_keeps_mru_placement():
    hashless = [Block(i) for i in range(6)]
    c = Block(50, block_hash="c")
    q = Queue([*hashless, c])
    stale = Block(99, block_hash="stale", ref_cnt=1)
    assert P.free_evict_first(q, [stale], max_walk=4) == []
    assert q.order() == [*hashless, c, stale]


def test_hashless_freed_block_goes_to_head():
    c = Block(10, block_hash="c")
    q = Queue([c])
    run = Block(5, ref_cnt=1)
    P.free_evict_first(q, [run])
    assert q.order() == [run, c]


def test_split_stale_classifies_superseded_hit_on_checkpoint_only():
    own_ckpt = Block(1, block_hash="own", num_tokens=64)
    stale = Block(2, block_hash="prev", num_tokens=32)  # below this request's boundary 64
    running = Block(3)  # hashless running state
    null = Block(0, block_hash=None, is_null=True)
    own = {1: (own_ckpt, False)}
    keep, out, live = P.split_stale(own, [own_ckpt, stale, running, null], live_from=64)
    assert out == [stale] and live == 0
    assert keep == [own_ckpt, running, null]
    # A recycled ID: same block id, different block object -> not own.
    impostor = Block(1, block_hash="other", num_tokens=32)
    keep, out, live = P.split_stale(own, [impostor], live_from=64)
    assert out == [impostor]
    # No own map at all.
    assert P.split_stale(None, [stale], live_from=64) == ([], [stale], 0)
    # A block owned by another BlockPool keeps the normal free path.
    this_pool, other_pool = object(), object()
    foreign = Block(7, block_hash="f", num_tokens=16)
    foreign.pool = other_pool
    mine = Block(8, block_hash="m", num_tokens=16)
    mine.pool = this_pool
    keep, out, live = P.split_stale(None, [foreign, mine, stale], pool=this_pool, live_from=64)
    assert keep == [foreign] and out == [mine, stale] and live == 0


def test_split_stale_keeps_the_live_checkpoint_at_the_own_boundary():
    """Served A/B j-74b70fab: a turn that crosses no new boundary hits exactly at its own replay boundary.

    That block is the conversation's live checkpoint (the next turn needs it), so it is not stale.
    """
    hit = Block(2, block_hash="prev", num_tokens=32)
    assert P.split_stale(None, [hit], live_from=32) == ([hit], [], 1)
    assert P.split_stale(None, [hit], live_from=16) == ([hit], [], 1)
    assert P.split_stale(None, [hit], live_from=48) == ([], [hit], 0)
    # Unknown boundary or position: today's (retention) path, never demoted.
    assert P.split_stale(None, [hit], live_from=None) == ([hit], [], 1)
    nopos = Block(4, block_hash="x")
    assert P.split_stale(None, [nopos], live_from=64) == ([nopos], [], 1)
    img = ImageBlock(5)
    img.set_block_hash(b"i", num_tokens=32)
    assert P.split_stale(None, [img], live_from=32) == ([img], [], 1)
    assert P.split_stale(None, [img], live_from=64) == ([], [img], 0)


def test_live_boundary_is_the_lowest_block_aligned_replay_boundary():
    assert P.live_boundary((39,), 16) == 32       # num_prompt - 1 = 39 -> checkpoint at 32
    assert P.live_boundary((32,), 16) == 32       # block-aligned resend
    assert P.live_boundary((47, 16), 16) == 16    # EAGLE's two boundaries: the lower one
    assert P.live_boundary((), 16) is None
    assert P.live_boundary((39,), 0) is None


def test_demote_frees_stale_before_older_conversation_under_lru():
    """Pool of 4 cached blocks; a new allocation must evict one.

    Conversation A is old; conversation B just finished a turn that hit its
    previous checkpoint `b_old`. Today `b_old` is appended MRU and A's block
    is evicted; with demotion `b_old` is evicted and A survives.
    """
    for demote in (False, True):
        a = Block(1, block_hash="A")
        b_kv = Block(2, block_hash="Bkv")
        q = Queue([a, b_kv])
        b_old = Block(3, block_hash="Bold", ref_cnt=1)
        if demote:
            P.free_evict_first(q, [b_old])
        else:
            lifo_free(q, [b_old])
        evicted = q.popleft()
        assert evicted is (b_old if demote else a)


# ---------------------------------------------------------------- refresh
def test_refresh_moves_cached_unreferenced_blocks_to_mru_in_order():
    hashless = Block(8)  # free, at the head: reused first, must stay there
    c = [Block(i, block_hash=f"c{i}") for i in range(4)]
    q = Queue([hashless, *c])
    held = Block(9, block_hash="held", ref_cnt=1)  # referenced: not in the queue
    moved = P.refresh_to_mru(q, [c[2], held, hashless, c[0]])
    assert moved == 2
    assert q.order() == [hashless, c[1], c[3], c[2], c[0]]
    # A block of another BlockPool is never moved into this queue.
    this_pool, other_pool = object(), object()
    c[1].pool = other_pool
    assert P.refresh_to_mru(q, [c[1]], pool=this_pool) == 0
    assert q.order() == [hashless, c[1], c[3], c[2], c[0]]


def test_eviction_ordered_is_tail_first_then_state():
    a = [Block(i, block_hash=f"a{i}") for i in range(3)]
    b = [Block(10 + i, block_hash=f"b{i}") for i in range(3)]
    null = Block(0, is_null=True)
    s = Block(20, block_hash="s")
    order = P.eviction_ordered([a, b, [null, null, s]], [False, False, True])
    assert order == [a[2], b[2], a[1], b[1], a[0], b[0], s]


class Req:
    def __init__(self, rid, hashes, num_tokens):
        self.request_id = rid
        self.block_hashes = hashes
        self.num_tokens = num_tokens


class Mamba:
    pass


class Attn:
    pass


class Coordinator:
    def __init__(self, hits):
        self.hits = hits  # rid -> (per-group blocks, hit_len)
        self.single_type_managers = [Attn(), Mamba()]
        self.calls = []

    def find_longest_cache_hit(self, hashes, max_len):
        rid = hashes[0]
        self.calls.append((rid, max_len))
        groups, n = self.hits[rid]
        return groups, n, 0  # fork's 3-tuple


class Scheduler:
    def __init__(self, waiting, coordinator, queue):
        self.waiting = waiting
        self.kv_holding_waiting = []
        self.kv_cache_manager = type(
            "M",
            (),
            {
                "enable_caching": True,
                "block_pool": type("BP", (), {"free_block_queue": queue})(),
                "coordinator": coordinator,
            },
        )()


def _scenario():
    a = [Block(i, block_hash=f"a{i}") for i in range(2)]
    a_ck = Block(5, block_hash="ack")
    other = [Block(10 + i, block_hash=f"o{i}") for i in range(3)]
    q = Queue([*a, a_ck, *other])
    coord = Coordinator({"r1": ([a, [Block(0, is_null=True), a_ck]], 2 * 3200)})
    sched = Scheduler([Req("r1", ["r1"], 7000)], coord, q)
    return q, coord, sched, a, a_ck, other


def test_refresh_waiting_moves_queued_prefix_behind_idle_cache():
    q, coord, sched, a, a_ck, other = _scenario()
    moved = P.refresh_waiting(sched, limit=8, interval=2.0, mamba_cls=Mamba, now=100.0)
    assert moved == 3
    assert q.order() == [*other, a[1], a[0], a_ck]
    assert coord.calls == [("r1", 6999)]


def test_refresh_waiting_counts_engagement():
    q, coord, sched, a, a_ck, other = _scenario()
    before = dict(P.STATS)
    P.refresh_waiting(sched, 8, 2.0, Mamba, now=100.0)
    assert P.STATS["refreshed"] - before["refreshed"] == 3
    assert P.STATS["refresh_passes"] - before["refresh_passes"] == 1


def test_refresh_waiting_respects_interval_limit_and_forgets_finished():
    q, coord, sched, a, a_ck, other = _scenario()
    P.refresh_waiting(sched, 8, 2.0, Mamba, now=100.0)
    P.refresh_waiting(sched, 8, 2.0, Mamba, now=101.0)  # within the interval
    assert len(coord.calls) == 1
    P.refresh_waiting(sched, 8, 2.0, Mamba, now=102.5)
    assert len(coord.calls) == 2
    assert P.refresh_waiting(sched, 0, 2.0, Mamba, now=200.0) == 0  # limit 0
    sched.waiting = []
    P.refresh_waiting(sched, 8, 2.0, Mamba, now=300.0)
    assert sched._tpu_prefix_refresh_stamp == {}


def test_refresh_waiting_off_without_caching():
    q, coord, sched, *_ = _scenario()
    sched.kv_cache_manager.enable_caching = False
    assert P.refresh_waiting(sched, 8, 2.0, Mamba, now=1.0) == 0
    assert coord.calls == []


# ---------------------------------------------------------------- contract
class ImageBlock(Block):
    """The image's vLLM KVCacheBlock: ``block_hash`` is a read-only property,
    set through ``set_block_hash`` (kv_cache_utils.py in the Qwen3.5 image
    copy; srv1 j-4e8d3ff8 hit 'property block_hash ... has no setter')."""

    def __init__(self, block_id, block_hash=None, ref_cnt=0, is_null=False):
        self._block_hash = block_hash
        self._block_hash_num_tokens = None
        self.block_id = block_id
        self.ref_cnt = ref_cnt
        self.is_null = is_null
        self.prev_free_block = None
        self.next_free_block = None

    @property
    def block_hash(self):
        return self._block_hash

    @property
    def block_hash_num_tokens(self):
        return self._block_hash_num_tokens

    def set_block_hash(self, block_hash, num_tokens=None):
        assert self._block_hash is None
        self._block_hash = block_hash
        self._block_hash_num_tokens = num_tokens


def _fake_vllm(monkeypatch, queue_cls, block_cls=Block):
    import types

    mods = {}
    for name in ("vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.kv_cache_utils"):
        mods[name] = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, mods[name])
    mods["vllm.v1.core.kv_cache_utils"].FreeKVCacheBlockQueue = queue_cls
    mods["vllm.v1.core.kv_cache_utils"].KVCacheBlock = block_cls
    return mods


def test_contract_check_passes_on_the_vllm_queue_shape(monkeypatch):
    _fake_vllm(monkeypatch, Queue)
    assert P.queue_contract_ok() is True


def test_contract_check_passes_on_the_image_block_api(monkeypatch):
    _fake_vllm(monkeypatch, Queue, ImageBlock)
    assert P.queue_contract_ok() is True


def test_helpers_work_on_image_blocks():
    c = ImageBlock(10)
    c.set_block_hash(b"c")
    q = Queue([ImageBlock(1), c])
    stale = ImageBlock(99, ref_cnt=1)
    stale.set_block_hash(b"stale")
    assert P.free_evict_first(q, [stale]) == [stale]
    assert [b.block_id for b in q.order()] == [1, 99, 10]


def test_logger_is_under_vllm_hierarchy(monkeypatch):
    mods = _fake_vllm(monkeypatch, Queue)
    import types

    seen = []
    log_mod = types.ModuleType("vllm.logger")
    log_mod.init_logger = lambda name: seen.append(name) or __import__("logging").getLogger(name)
    monkeypatch.setitem(sys.modules, "vllm.logger", log_mod)
    lazy = P._LazyLogger(P._LOGGER_NAME)
    lazy.info("probe")
    assert seen == ["vllm.vllm_torchtpu.core.prefix_cache_policy"]


def test_contract_check_rejects_a_different_queue(monkeypatch):
    class Renamed(Queue):
        def __init__(self, blocks):
            super().__init__(blocks)
            self.head = self.fake_free_list_head
            del self.fake_free_list_head

    _fake_vllm(monkeypatch, Renamed)
    assert P.queue_contract_ok() is False


def test_contract_check_rejects_a_queue_that_pops_from_the_other_end(monkeypatch):
    class TailPop(Queue):
        def popleft(self):
            b = self.fake_free_list_tail.prev_free_block
            assert b is not self.fake_free_list_head
            self.fake_free_list_tail.prev_free_block = b.prev_free_block
            b.prev_free_block.next_free_block = self.fake_free_list_tail
            b.prev_free_block = b.next_free_block = None
            self.num_free_blocks -= 1
            return b

    _fake_vllm(monkeypatch, TailPop)
    assert P.queue_contract_ok() is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------- tails evict first
U = 3200  # pool block (scheduler block) of the served prefill


def _tail(i, covered, ref_cnt=1):
    return Block(i, block_hash=f"t{i}", ref_cnt=ref_cnt, num_tokens=covered)


def test_is_tail_is_a_partial_primary_key():
    assert P.is_tail(_tail(1, U + 128), U)
    assert not P.is_tail(Block(2, block_hash="f", num_tokens=2 * U), U)   # full block
    assert not P.is_tail(Block(3), U)                                      # hashless
    assert not P.is_tail(Block(4, block_hash="x"), U)                      # position unknown
    assert not P.is_tail(Block(5, block_hash="n", is_null=True, num_tokens=7), U)
    assert not P.is_tail(_tail(6, U + 128), 0)                             # no unit: off


def test_tails_queue_oldest_first_after_the_hashless_head():
    h, c = Block(1), Block(2, block_hash="c", num_tokens=U)
    q = Queue([h, c])
    t1, t2, t3 = _tail(11, U + 128), _tail(12, U + 256), _tail(13, 2 * U + 128)
    assert P.free_tails_first(q, [t1]) == [t1]
    assert P.free_tails_first(q, [t2, t3]) == [t2, t3]
    assert q.order() == [h, t1, t2, t3, c]                 # oldest tail evicted first, all before c


def test_segment_order_hashless_then_stale_newest_first_then_tails_oldest_first():
    h, c = Block(1), Block(2, block_hash="c", num_tokens=U)
    q = Queue([h, c])
    s1, s2 = (Block(i, block_hash=f"s{i}", ref_cnt=1, num_tokens=U) for i in (21, 22))
    t1, t2 = _tail(11, U + 128), _tail(12, U + 256)
    P.free_evict_first(q, [s1])
    P.free_tails_first(q, [t1])
    P.free_evict_first(q, [s2])
    P.free_tails_first(q, [t2])
    assert q.order() == [h, s2, s1, t1, t2, c]


def test_segment_end_is_found_again_when_the_newest_tail_left():
    h, c = Block(1), Block(2, block_hash="c", num_tokens=U)
    q = Queue([h, c])
    t1, t2, t3 = _tail(11, U + 128), _tail(12, U + 256), _tail(13, U + 384)
    P.free_tails_first(q, [t1, t2])
    P._unlink(q, t2)                                       # t2 hit by a request (touch)
    assert P.free_tails_first(q, [t3]) == [t3]
    assert q.order() == [h, t1, t3, c]                     # t3 after t1, not before it
    # the newest tail freed to the MRU end by the plain path leaves the segment
    t4 = _tail(14, U + 512)
    P.free_tails_first(q, [t4])
    P._unlink(q, t4)
    P.clear_marks(q, t4)
    q.append(t4)
    t5 = _tail(15, U + 640)
    P.free_tails_first(q, [t5])
    assert q.order() == [h, t1, t3, t5, c, t4]


def test_tails_keep_mru_placement_without_pressure_and_shared_tails_stay_put():
    head = [Block(100 + i) for i in range(P._MAX_WALK + 1)]
    c = Block(2, block_hash="c", num_tokens=U)
    q = Queue(head + [c])
    t1 = _tail(11, U + 128)
    assert P.free_tails_first(q, [t1]) == []
    assert q.order()[-1] is t1
    shared = _tail(12, U + 256, ref_cnt=2)
    assert P.free_tails_first(q, [shared]) == [] and shared.ref_cnt == 1 and not P._in_queue(shared)


def test_refresh_moves_a_tail_out_of_the_segment():
    h, c = Block(1), Block(2, block_hash="c", num_tokens=U)
    q = Queue([h, c])
    t1, t2 = _tail(11, U + 128), _tail(12, U + 256)
    P.free_tails_first(q, [t1])
    assert P.refresh_to_mru(q, [t1]) == 1
    P.free_tails_first(q, [t2])
    assert q.order() == [h, t2, c, t1]                     # t1 is no longer a segment member


def test_split_tails_keeps_order_and_leaves_foreign_blocks():
    this_pool, other_pool = object(), object()
    t1, f, foreign = _tail(11, U + 128), Block(2, block_hash="f", num_tokens=U), _tail(12, U + 256)
    foreign.pool = other_pool
    tails, rest = P.split_tails([f, t1, foreign], U, this_pool)
    assert tails == [t1] and rest == [f, foreign]


def test_tail_contract_needs_the_covered_token_count(monkeypatch):
    _fake_vllm(monkeypatch, Queue, ImageBlock)
    assert P.tail_contract_ok() is True
    _fake_vllm(monkeypatch, Queue, Block)                  # upstream block: hash setter, no covered count
    assert P.tail_contract_ok() is False


def test_counters_include_tails():
    assert {"tails_queued", "tail_calls"} <= set(P.STATS)


class _Pool:
    def __init__(self, queue):
        self.free_block_queue = queue


def test_free_with_tails_routes_tails_and_clears_marks_of_plain_frees():
    h, c = Block(1), Block(2, block_hash="c", num_tokens=U)
    q = Queue([h, c])
    pool = _Pool(q)
    plain = lambda p, blocks: lifo_free(p.free_block_queue, blocks)  # noqa: E731
    t1 = _tail(11, U + 128)
    full = Block(3, block_hash="f", ref_cnt=1, num_tokens=2 * U)
    before = dict(P.STATS)
    assert P.free_with_tails(pool, [full, t1], U, plain) is True
    assert q.order() == [h, t1, c, full]
    assert P.STATS["tails_queued"] - before["tails_queued"] == 1 and P.STATS["tail_calls"] - before["tail_calls"] == 1
    # t1 is hit (leaves the queue), then comes back through the plain path as a full block: it leaves the segment
    P._unlink(q, t1)
    t1.block_hash_num_tokens = 2 * U
    t1.ref_cnt = 1
    assert P.free_with_tails(pool, [t1], U, plain) is False
    t2 = _tail(12, U + 256)
    P.free_with_tails(pool, [t2], U, plain)
    assert q.order() == [h, t2, c, full, t1]       # not queued after t1 at the MRU end
