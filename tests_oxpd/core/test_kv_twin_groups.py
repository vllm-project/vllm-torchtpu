# SPDX-License-Identifier: Apache-2.0
"""q38 D1k: VLLM_TPU_KV_TWIN_GROUPS leaves every block table identical.

A small hybrid plan shaped like the served decode's (align-mode Mamba groups with
speculative blocks, 1024-token full attention, 4-token circular rings, one
singleton class) runs the scheduler's decode-step allocation for several
requests, with requests finishing and new PD admissions (external tokens,
delayed caching) along the way. With the twin coordinator installed the
per-request block ids, cached counts, Mamba checkpoints and the pool's free
count must equal the original loop's exactly, while most twin calls are
skipped. A coordinator that skips twins even when their representative
changed something must produce different tables (the mutant).

A second scenario drives multi-turn PD admissions through the prefix cache, as
the decode's scheduler does with the Qwen PD connector: the local hit from
get_computed_blocks_for_connector (divergent hybrid hits), the block-aligned
trim, external tokens up to prompt - 1 with delayed caching, cache_blocks when
the load lands, decode steps, and the next turn's prompt = the conversation +
a new message. Some prompts share a system prefix, some end exactly on (or 1-3
tokens past) a hash-block boundary, and every step starts by clearing the
same-step prefix-hit set as Scheduler.schedule() does. With the eagle block
drop on (use_eagle=True), a full-attention group registers its prompt tail as a
partial prefix-cache entry (block_pool.cache_partial_block) on a call that
caches no full block; a coordinator that does not count prefix-cache key
changes skips the twins' entries (found by HOST-MM, q38-2c). Every admission's
local hit, every block-count answer, the hash map (keys and block ids), every
block's hash metadata and the block tables must be identical; it runs with the
eagle block drop off (the served decode: VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=1)
and on.
"""

import hashlib
from collections import Counter

import random

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu import patch_registry
from vllm_torchtpu.core import kv_twin_groups

HASH_BLOCK = 256


def _plan():
    specs = (
        [MambaSpec(block_size=256, shapes=((2, 2),), dtypes=(torch.bfloat16,), mamba_cache_mode="align",
                   num_speculative_blocks=3) for _ in range(4)]
        + [FullAttentionSpec(block_size=1024, num_kv_heads=1, head_size=128, dtype=torch.bfloat16) for _ in range(3)]
        + [CircularBufferSpec(block_size=4, num_kv_heads=1, head_size=132, head_size_v=0, dtype=torch.bfloat16,
                              page_size_padded=8192) for _ in range(2)]
        + [FullAttentionSpec(block_size=256, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)]
    )
    groups = [KVCacheGroupSpec([f"layer.{i}"], s) for i, s in enumerate(specs)]
    return KVCacheConfig(num_blocks=30000, kv_cache_tensors=[], kv_cache_groups=groups)


def _manager():
    return KVCacheManager(_plan(), max_model_len=65536, scheduler_block_size=1024, hash_block_size=HASH_BLOCK,
                          enable_caching=True, use_eagle=False, dcp_world_size=1, pcp_world_size=1)


def _admit(m, rnd, hasher, i):
    n = 5000 + rnd.randrange(0, 3000)
    r = Request(f"r{i}", [rnd.randrange(1, 50000) for _ in range(n)], SamplingParams(max_tokens=100000), None,
                block_hasher=hasher)
    r.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    assert m.allocate_slots(r, 0, num_external_computed_tokens=n - 1, delay_cache_blocks=True) is not None
    r.num_computed_tokens = n - 1
    m.cache_blocks(r, r.num_computed_tokens)
    r.status = RequestStatus.RUNNING
    r.append_output_token_ids([rnd.randrange(1, 50000)])
    return r


def _run(mode, steps=300, n=6, seed=1):
    patch_registry.apply("platform_activation")
    patch_registry.apply("engine_core")
    fn = get_hash_fn_by_name("sha256")
    init_none_hash(fn)
    hasher = get_request_block_hasher(HASH_BLOCK, fn)
    rnd = random.Random(seed)
    m = _manager()
    tc = None
    if mode != "base":
        tc = kv_twin_groups.install(m.coordinator)
        if mode == "mutant":
            tc._sig = lambda c, rid: 0  # every representative call reads as "nothing changed"
    # Every answer to "how many blocks would this request need" is recorded: the scheduler's admission and
    # free-block checks depend on it, so it must match the original loop exactly, not just the tables.
    nums = []
    count = m.coordinator.get_num_blocks_to_allocate

    def recording(*a, **k):
        v = count(*a, **k)
        nums.append(v)
        return v

    m.coordinator.get_num_blocks_to_allocate = recording
    reqs = [_admit(m, rnd, hasher, i) for i in range(n)]
    nid = n
    for s in range(steps):
        for r in reqs:
            r.spec_token_ids = [1, 2, 3]
            assert m.allocate_slots(r, 4, num_lookahead_tokens=3) is not None
        for r in reqs:
            acc = 4 if rnd.random() < 0.32 else 3
            r.num_computed_tokens += acc
            r.append_output_token_ids([rnd.randrange(1, 50000) for _ in range(acc)])
            r.spec_token_ids = []
        if (s + 1) % 37 == 0:
            m.free(reqs.pop(rnd.randrange(len(reqs))))
            reqs.append(_admit(m, rnd, hasher, nid))
            nid += 1
    state = {"free": m.block_pool.get_num_free_blocks(), "nums": nums}
    for gi, sm in enumerate(m.coordinator.single_type_managers):
        state[gi] = {
            r.request_id: ([b.block_id for b in sm.req_to_blocks.get(r.request_id) or []],
                           sm.num_cached_block.get(r.request_id),
                           repr(getattr(sm, "_checkpoints", {}).get(r.request_id)))
            for r in reqs
        }
    return state, tc


def test_twin_coordinator_keeps_block_tables_identical():
    base, _ = _run("base")
    twin, tc = _run("twins")
    assert [len(c.twins) for c in tc.classes] == [3, 2, 1, 0]
    assert tc.skipped > 10 * tc.ran > 0  # most twin calls skipped, some (block boundaries) ran
    assert any(v > 0 for v in base["nums"])  # boundary steps that need blocks are exercised
    assert twin == base


def test_skipping_changed_twins_breaks_the_tables():
    base, _ = _run("base")
    mutant, _ = _run("mutant")
    assert mutant != base


def _pd_turns(mode, use_eagle, users=8, turns=5, seed=7):
    """Multi-turn PD decode admissions through the prefix cache (see the module docstring)."""
    patch_registry.apply("platform_activation")
    patch_registry.apply("engine_core")
    fn = get_hash_fn_by_name("sha256")
    init_none_hash(fn)
    hasher = get_request_block_hasher(HASH_BLOCK, fn)
    rnd = random.Random(seed)
    m = KVCacheManager(_plan(), max_model_len=65536, scheduler_block_size=1024, hash_block_size=HASH_BLOCK,
                       enable_caching=True, use_eagle=use_eagle, dcp_world_size=1, pcp_world_size=1)
    pool = m.block_pool
    tc = kv_twin_groups.install(m.coordinator) if mode != "base" else None
    seen = Counter()
    if mode == "base":
        # How many prompt-tail entries were inserted, and how many in a manager call that cached no full block.
        hmap, flag = pool.cached_block_hash_to_block, {"partial": False, "full": False}
        ins, cpb, cfb = hmap.insert, pool.cache_partial_block, pool.cache_full_blocks

        def insert(*a, **k):
            if flag["partial"]:
                seen["partial"] += 1
                seen["tail_only"] += not flag["full"]
            return ins(*a, **k)

        def cache_partial_block(*a, **k):
            flag["partial"] = True
            try:
                return cpb(*a, **k)
            finally:
                flag["partial"] = False

        def cache_full_blocks(*a, **k):
            flag["full"] = flag["full"] or k.get("num_full_blocks", 0) > k.get("num_cached_blocks", 0)
            return cfb(*a, **k)

        hmap.insert, pool.cache_partial_block, pool.cache_full_blocks = insert, cache_partial_block, cache_full_blocks
        for sm in m.coordinator.single_type_managers:
            def cb(*a, _o=sm.cache_blocks, **k):
                flag["full"] = False
                return _o(*a, **k)

            sm.cache_blocks = cb
    nums = []
    count = m.coordinator.get_num_blocks_to_allocate

    def recording(*a, **k):
        v = count(*a, **k)
        nums.append(v)
        return v

    m.coordinator.get_num_blocks_to_allocate = recording
    hits, turn, live, pending = [], {}, {}, []

    def length(lo, hi):
        n, k = rnd.randint(lo, hi), rnd.random()
        if k < 0.3:
            n = n // HASH_BLOCK * HASH_BLOCK
        elif k < 0.5:
            n = n // HASH_BLOCK * HASH_BLOCK + rnd.randint(1, 3)
        return max(n, 2)

    def admit(u, prompt):
        n = len(prompt)
        r = Request(f"u{u}.t{turn.get(u, 0) + 1}", prompt, SamplingParams(max_tokens=100000), None, block_hasher=hasher)
        r.kv_transfer_params = {"num_tokens": n - 1}
        blocks, local, _, diverged = m.get_computed_blocks_for_connector(r)
        tail = local % 1024
        ext = (n - 1) - (local - tail)
        if tail and ext > tail:
            blocks, local = m.truncate_computed_blocks(blocks, local - tail), local - tail
        elif tail:
            ext = 0
        hits.append((r.request_id, n, local, ext, diverged))
        r.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        if m.allocate_slots(r, 0, num_new_computed_tokens=local, new_computed_blocks=blocks,
                            num_external_computed_tokens=ext, delay_cache_blocks=ext > 0) is None:
            seen["deferred"] += 1  # the scheduler leaves it waiting and retries next step
            return False
        turn[u] = turn.get(u, 0) + 1
        seen["hit"] += local > 0
        r.num_computed_tokens = local + ext
        m.cache_blocks(r, r.num_computed_tokens)  # _update_waiting_for_remote_kv
        r.status = RequestStatus.RUNNING
        r.append_output_token_ids([rnd.randrange(1, 50000)])
        live[r.request_id] = (u, r, rnd.randint(10, 200))
        return True

    system = [rnd.randrange(1, 50000) for _ in range(2304)]
    for u in range(users):
        body = [rnd.randrange(1, 50000) for _ in range(length(*rnd.choice([(250, 1000), (1000, 6000)])))]
        pending.append((u, (system if u % 2 == 0 else []) + body))
    steps = 0
    while (live or pending) and steps < 5000:
        steps += 1
        # Scheduler.schedule() first clears the same-step prefix-hit set (_patch_vllm_same_step_prefix_hits)
        pool.__dict__.get("_tpu_unwritten_block_ids", set()).clear()
        m.new_step_starts()  # and resets Mamba's per-step cached_blocks_this_step
        for u, r, _ in live.values():
            r.spec_token_ids = [1, 2, 3]
            assert m.allocate_slots(r, 4, num_lookahead_tokens=3) is not None
        for rid, (u, r, mx) in list(live.items()):
            acc = 1
            while acc < 4 and rnd.random() < 0.45:
                acc += 1
            r.num_computed_tokens += acc
            r.append_output_token_ids([rnd.randrange(1, 50000) for _ in range(acc)])
            r.spec_token_ids = []
            if r.num_output_tokens >= mx:
                del live[rid]
                m.free(r)
                if turn[u] < turns:
                    msg = [rnd.randrange(1, 50000) for _ in range(length(*rnd.choice([(30, 700), (700, 2500)])))]
                    pending.append((u, list(r.all_token_ids) + msg))
        pending[:] = [p for p in pending if not admit(*p)]
    hmap = pool.cached_block_hash_to_block._cache
    state = {
        "hits": hits,
        "nums": nums,
        "free": pool.get_num_free_blocks(),
        "hash_map": hashlib.sha256(repr(sorted(
            (k, sorted([v.block_id] if hasattr(v, "block_id") else v)) for k, v in hmap.items())).encode()).hexdigest(),
        "block_hashes": hashlib.sha256(repr([
            (b.block_id, b.block_hash, sorted(pool.cached_block_hashes_by_block.get(b.block_id, ())))
            for b in pool.blocks]).encode()).hexdigest(),
    }
    for gi, sm in enumerate(m.coordinator.single_type_managers):
        state[gi] = sorted((rid, [b.block_id for b in bl], sm.num_cached_block.get(rid))
                           for rid, bl in sm.req_to_blocks.items())
    return state, seen, tc


@pytest.mark.parametrize("use_eagle", [False, True], ids=["served-eagle-drop-off", "eagle-drop-on"])
def test_twin_coordinator_keeps_prefix_cache_identical(use_eagle):
    base, seen, _ = _pd_turns("base", use_eagle)
    twin, _, tc = _pd_turns("twins", use_eagle)
    # The traffic exercises prefix hits, prompt-tail entries and (eagle on) tail-only registrations.
    assert seen["hit"] >= 10 and seen["partial"] > 0, seen
    if use_eagle:
        assert seen["tail_only"] > 0, seen
    assert tc.skipped > 10 * tc.ran > 0
    assert twin["hits"] == base["hits"]
    assert twin["hash_map"] == base["hash_map"]
    assert twin["block_hashes"] == base["block_hashes"]
    assert twin == base


def test_reset_prefix_cache_keeps_counting_prefix_cache_keys():
    """reset_prefix_cache replaces the hash map; the new map's insert / pop must still count as pool changes."""
    patch_registry.apply("platform_activation")
    patch_registry.apply("engine_core")
    m = _manager()
    tc = kv_twin_groups.install(m.coordinator)
    pool = m.block_pool
    old = pool.cached_block_hash_to_block
    before = tc.mutations
    assert pool.reset_prefix_cache()
    assert pool.cached_block_hash_to_block is not old
    after_reset = tc.mutations
    assert after_reset > before
    block = pool.blocks[1]
    key = b"\x01" * 32 + (0).to_bytes(4, "big")
    pool.cached_block_hash_to_block.insert(key, block)
    assert tc.mutations == after_reset + 1
    pool.cached_block_hash_to_block.pop(key, block.block_id)
    assert tc.mutations == after_reset + 2


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_KV_TWIN_GROUPS", raising=False)
    assert not kv_twin_groups.patch_kv_cache_manager_for_twins()


def test_compile_neutral():
    from vllm_torchtpu.compilation import tpu_compiler

    assert "VLLM_TPU_KV_TWIN_GROUPS" in tpu_compiler._TPU_COMPILE_ENV_IGNORED
