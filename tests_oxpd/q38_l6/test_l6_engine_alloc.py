# SPDX-License-Identifier: Apache-2.0
"""q38 L6 lever engalloc (core/l6_engine_alloc.py): host-only tests; run in a CDK CPU pod.

The core test drives one KVCacheManager through a random stream shaped like the
PD decode's engine core under async scheduling with MTP k = 3, twice with the
same seed: once on the stock path, once with the fast path installed and the
lever on. Each step runs the scheduler's calls in its order: the running
requests' allocate_slots (preempting the last running request when a call
fails), PD admissions (a local prefix hit through get_computed_blocks plus
external tokens with delayed caching, then the landing's cache_blocks),
resumptions of preempted requests (chunked local prefill from their prefix
hit), then the previous step's outputs (rejections roll back
num_computed_tokens and the placeholders, accepted tokens append, and each
running request's cache_blocks), finishes, and occasional connector
evict_blocks on held blocks. Prompts share a system prefix or extend a
previous turn, so prefix hits, prompt-tail partial entries and block-boundary
crossings all occur. After every step the full state must be equal: every
manager's per-request block ids and bookkeeping (Mamba retire / last-state /
checkpoint / CoW tables, the retention patch's pinned checkpoints), the block
pool's free-queue order, every block's ref count and hash metadata, the
prefix-cache hash map, the same-step unwritten set, and every allocate_slots
result.
"""

import hashlib
import os
import random
from collections import deque

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
    SlidingWindowSpec,
)
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu import patch_registry
from vllm_torchtpu.core import kv_twin_groups
from vllm_torchtpu.core import l6_ab
from vllm_torchtpu.core import l6_engine_alloc as ea_mod

ENV = "VLLM_TPU_L6_ENGINE_ALLOC_FAST"
CHECK = "VLLM_TPU_L6_ENGINE_ALLOC_FAST_CHECK"


@pytest.fixture(autouse=True)
def lever(monkeypatch):
    """Lever on (uniform) unless a test sets it otherwise; module state restored after."""
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_") or k in ("VLLM_TPU_HOST_AB_PERIOD_MS", "VLLM_TPU_D1N_AB_PERIOD_MS"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(ENV, "1")
    l6_ab._resolve()
    ea_mod._INSTANCES.clear()
    yield monkeypatch
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_"):
            monkeypatch.delenv(k, raising=False)
    l6_ab._resolve()
    ea_mod._INSTANCES.clear()


def _specs(kind):
    bf = torch.bfloat16
    if kind == "coarse":  # the D1k twins test's plan: Mamba block == hash block (no fine-grained hits)
        hsz, sched = 256, 1024
        specs = ([MambaSpec(block_size=256, shapes=((2, 2),), dtypes=(bf,), mamba_cache_mode="align",
                            num_speculative_blocks=3) for _ in range(4)]
                 + [FullAttentionSpec(block_size=1024, num_kv_heads=1, head_size=128, dtype=bf) for _ in range(3)]
                 + [CircularBufferSpec(block_size=4, num_kv_heads=1, head_size=132, head_size_v=0, dtype=bf,
                                       page_size_padded=8192) for _ in range(2)]
                 + [FullAttentionSpec(block_size=256, num_kv_heads=1, head_size=128, dtype=bf)])
    elif kind == "skew":  # block boundaries that do not coincide: Mamba 384 vs full attention 512 (scheduler 1536)
        hsz, sched = 128, 1536
        specs = ([MambaSpec(block_size=384, shapes=((2, 2),), dtypes=(bf,), mamba_cache_mode="align",
                            num_speculative_blocks=3) for _ in range(4)]
                 + [FullAttentionSpec(block_size=512, num_kv_heads=1, head_size=128, dtype=bf) for _ in range(3)]
                 + [CircularBufferSpec(block_size=4, num_kv_heads=1, head_size=132, head_size_v=0, dtype=bf,
                                       page_size_padded=8192) for _ in range(2)])
    else:  # "fine": served-like ratios (Mamba 512 / FA 2048 / compressed 512 / hash 128) scaled down 4x
        hsz, sched = 32, 512
        specs = ([MambaSpec(block_size=128, shapes=((2, 2),), dtypes=(bf,), mamba_cache_mode="align",
                            num_speculative_blocks=3, num_prefill_checkpoint_blocks=1,
                            prefill_checkpoint_alignment=hsz) for _ in range(5)]
                 + [FullAttentionSpec(block_size=512, num_kv_heads=1, head_size=128, dtype=bf) for _ in range(3)]
                 + [FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=64, dtype=bf) for _ in range(2)]
                 + [CircularBufferSpec(block_size=4, num_kv_heads=1, head_size=132, head_size_v=0, dtype=bf,
                                       page_size_padded=8192) for _ in range(3)])
    return specs, hsz, sched


def _manager(kind, num_blocks, use_eagle=False):
    specs, hsz, sched = _specs(kind)
    groups = [KVCacheGroupSpec([f"layer.{i}"], s) for i, s in enumerate(specs)]
    cfg = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups)
    m = KVCacheManager(cfg, max_model_len=65536, scheduler_block_size=sched, hash_block_size=hsz,
                       enable_caching=True, use_eagle=use_eagle, dcp_world_size=1, pcp_world_size=1, watermark=0.01)
    return m, hsz


def _free_order(pool):
    q = pool.free_block_queue
    out, b = [], q.fake_free_list_head.next_free_block
    while b is not None and b is not q.fake_free_list_tail:
        out.append(b.block_id)
        b = b.next_free_block
    return out


def _state(m, reqs, trace):
    pool = m.block_pool
    st = {"free": _free_order(pool), "trace": trace,
          "blocks": [(b.block_id, b.ref_cnt, b.block_hash, b.block_hash_num_tokens) for b in pool.blocks],
          "by_block": sorted((k, sorted(v)) for k, v in pool.cached_block_hashes_by_block.items()),
          "map": sorted((k, sorted([v.block_id] if hasattr(v, "block_id") else list(v)))
                        for k, v in pool.cached_block_hash_to_block._cache.items()),
          "unwritten": sorted(pool.__dict__.get("_tpu_unwritten_block_ids", ()))}
    for gi, sm in enumerate(m.coordinator.single_type_managers):
        g = {}
        for rid in sorted(set(sm.req_to_blocks) | {r.request_id for r in reqs}):
            row = [[b.block_id for b in sm.req_to_blocks.get(rid, ())], sm.num_cached_block.get(rid),
                   rid in sm._partial_hit_reqs]
            for name in ("_checkpoints", "last_state_block_idx", "_num_retired_blocks"):
                if hasattr(sm, name):
                    row.append(repr(getattr(sm, name).get(rid)))
            if hasattr(sm, "_allocated_block_reqs"):
                row.append(rid in sm._allocated_block_reqs)
            own = sm.__dict__.get("_tpu_own_checkpoints", {}).get(rid)
            row.append(None if own is None else sorted((k, v[1]) for k, v in own.items()))
            pinned = sm.__dict__.get("_tpu_pinned_checkpoints", {}).get(rid)
            row.append(None if pinned is None else [b.block_id for b in pinned])
            g[rid] = row
        st[gi] = (g, sorted(getattr(sm, "cached_blocks_this_step", ())), list(sm.new_block_ids),
                  [(a.block_id, b.block_id) for a, b in sm._pending_cow_copies],
                  [(e[0], e[1], e[2].block_id, e[3]) for e in sm._pending_boundary_state_offloads])
    return st


def _digest(st):
    return hashlib.sha256(repr(st).encode()).hexdigest()


def _call_sig(m, request, kind, res):
    """Right after one allocate_slots / cache_blocks call: its result, the calling request's table and bookkeeping in
    every group, and the pool's free count, prefix-cache size and same-step unwritten set size. A skipped event that a
    later call in the same step would redo (caching up to n is idempotent) still shows here."""
    pool = m.block_pool
    rid = request.request_id
    out = [kind, rid, repr(None if res is None else res.get_block_ids(allow_none=True)), pool.get_num_free_blocks(),
           len(pool.cached_block_hash_to_block), len(pool.__dict__.get("_tpu_unwritten_block_ids", ()))]
    for sm in m.coordinator.single_type_managers:
        bl = sm.req_to_blocks.get(rid)
        row = (None if bl is None else tuple(b.block_id for b in bl), sm.num_cached_block.get(rid),
               rid in sm._partial_hit_reqs)
        if hasattr(sm, "_checkpoints"):
            row += (sm._checkpoints.get(rid), sm.last_state_block_idx.get(rid), sm._num_retired_blocks.get(rid),
                    rid in sm._allocated_block_reqs)
        out.append(row)
    return hash(repr(out))


class Stream:
    """A seeded engine-core call stream (see the module docstring)."""

    SPEC = 3
    LOOKAHEAD = 3
    CHUNK = 700

    def __init__(self, kind="fine", *, fast, twins, seed=0, check=0, use_eagle=False, num_blocks=None,
                 evict=True, cert_hook=None):
        if num_blocks is None:
            num_blocks = {"fine": 3000, "skew": 2000}.get(kind, 1500)
        patch_registry.apply("platform_activation")
        patch_registry.apply("engine_core")
        fn = get_hash_fn_by_name("sha256")
        init_none_hash(fn)
        self.m, self.hsz = _manager(kind, num_blocks, use_eagle)
        self.hasher = get_request_block_hasher(self.hsz, fn)
        self.tc = kv_twin_groups.install(self.m.coordinator) if twins else None
        self.ea = ea_mod.install(self.m, check=check) if fast else None
        assert (self.ea is not None) == fast
        if cert_hook and self.ea:
            cert_hook(self.ea)
        self.calls: list[int] = []  # one _call_sig per allocate_slots / cache_blocks call
        alloc, cache = self.m.allocate_slots, self.m.cache_blocks

        def allocate_slots(request, *a, **k):
            res = alloc(request, *a, **k)
            self.calls.append(_call_sig(self.m, request, "a", res))
            return res

        def cache_blocks(request, *a, **k):
            res = cache(request, *a, **k)
            self.calls.append(_call_sig(self.m, request, "c", res))
            return res

        self.m.allocate_slots, self.m.cache_blocks = allocate_slots, cache_blocks
        self.fast = fast
        self.rnd = random.Random(seed)
        self.evict = evict
        self.running: list[Request] = []
        self.landing: list[tuple[Request, int]] = []
        self.resume: deque[Request] = deque()
        self.prev: list[tuple[Request, int, int, bool]] = []
        self.live: dict[str, Request] = {}
        self.maxout: dict[str, int] = {}
        self.turns: dict[str, int] = {}
        self.trace: list = []
        self.uid = 0
        self.system = [self.rnd.randrange(1, 50000) for _ in range(5 * self.hsz + 7)]
        self.convs: list[list[int]] = []
        self.seen = {"preempt": 0, "resume": 0, "hit": 0, "finish": 0, "evict": 0, "alloc": 0, "popfree": 0,
                     "failed": 0}
        self.deferred: list = []

    # ---- arrivals -------------------------------------------------------------------------------
    def _length(self, lo, hi):
        n, k = self.rnd.randint(lo, hi), self.rnd.random()
        if k < 0.3:
            n = n // self.hsz * self.hsz
        elif k < 0.45:
            n = n // self.hsz * self.hsz + self.rnd.randint(1, 3)
        return max(n, 2)

    def _new_prompt(self):
        rnd = self.rnd
        if self.convs and rnd.random() < 0.6:
            base = self.convs.pop(rnd.randrange(len(self.convs)))
        elif rnd.random() < 0.5:
            base = list(self.system)
        else:
            base = []
        return base + [rnd.randrange(1, 50000) for _ in range(self._length(40, 1500))]

    def _admit(self):
        m, rnd = self.m, self.rnd
        prompt = self._new_prompt()
        self.uid += 1
        r = Request(f"q{self.uid}", prompt, SamplingParams(max_tokens=100000), None, block_hasher=self.hasher)
        n = len(prompt)
        blocks, hits, r.shared_prefix_boundary = m.get_computed_blocks(r)  # as Scheduler.schedule() stores it
        hits = min(hits, n - 1)
        ext = n - 1 - hits
        if hits == 0 and ext == 0:
            return
        r.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        res = m.allocate_slots(r, 0, num_new_computed_tokens=hits, new_computed_blocks=blocks if hits else None,
                               num_external_computed_tokens=ext, delay_cache_blocks=ext > 0)
        self.trace.append(("admit", r.request_id, None if res is None else res.get_block_ids(allow_none=True)))
        if res is None:
            return
        self.seen["hit"] += hits > 0
        r.num_computed_tokens = hits + ext
        self.landing.append((r, rnd.randint(0, 2)))
        self.live[r.request_id] = r
        self.maxout[r.request_id] = rnd.randint(20, 420)

    def _land(self):
        keep = []
        for r, wait in self.landing:
            if wait > 0:
                keep.append((r, wait - 1))
                continue
            self.m.cache_blocks(r, r.num_computed_tokens)  # _update_waiting_for_remote_kv
            r.status = RequestStatus.RUNNING
            r.append_output_token_ids([self.rnd.randrange(1, 50000)])
            self.running.append(r)
        self.landing = keep

    # ---- the scheduler's calls ------------------------------------------------------------------
    def _schedule_one(self, r):
        """allocate_slots for one running request as Scheduler.schedule() calls it; preempts on failure."""
        m = self.m
        num_new = r.num_tokens + len(r.spec_token_ids) + r.num_output_placeholders - r.num_computed_tokens
        if r.num_computed_tokens < r.num_tokens - 1:
            num_new = min(num_new, self.CHUNK)  # resumed request: chunked recompute
        if num_new <= 0:
            return None
        while True:
            res = m.allocate_slots(r, num_new, num_lookahead_tokens=self.LOOKAHEAD)
            self.seen["alloc"] += 1
            self.trace.append(("alloc", r.request_id, None if res is None else res.get_block_ids(allow_none=True)))
            if res is not None:
                break
            self.seen["failed"] += 1
            victim = self.running[-1]
            self._preempt(victim)
            if victim is r:
                return False
        ns = num_new + r.num_computed_tokens - r.num_tokens - r.num_output_placeholders
        ns = max(0, min(ns, len(r.spec_token_ids)))
        r.spec_token_ids = []
        return (r, num_new, ns)

    def _preempt(self, r):
        self.seen["preempt"] += 1
        self.running.remove(r)
        self.m.free(r)
        r.status = RequestStatus.PREEMPTED
        r.num_computed_tokens = 0
        r.spec_token_ids = []
        r.num_stale_output_tokens = r.num_in_flight_tokens
        r.num_output_placeholders = 0
        self.resume.appendleft(r)

    def _resume_one(self):
        m = self.m
        r = self.resume[0]
        blocks, hits, r.shared_prefix_boundary = m.get_computed_blocks(r)
        hits = min(hits, r.num_tokens - 1)
        num_new = min(r.num_tokens - hits, self.CHUNK)
        res = m.allocate_slots(r, num_new, num_new_computed_tokens=hits,
                               new_computed_blocks=blocks if hits else None,
                               num_lookahead_tokens=self.LOOKAHEAD, has_scheduled_reqs=bool(self.running))
        self.trace.append(("resume", r.request_id, None if res is None else res.get_block_ids(allow_none=True)))
        if res is None:
            return None
        self.resume.popleft()
        self.seen["resume"] += 1
        r.status = RequestStatus.RUNNING
        r.num_computed_tokens = hits
        self.running.append(r)
        return (r, num_new, 0)

    def step(self):
        m, rnd = self.m, self.rnd
        m.block_pool.__dict__.get("_tpu_unwritten_block_ids", set()).clear()  # Scheduler.schedule() (same-step patch)
        m.new_step_starts()
        if self.fast:
            ea_mod.flush()
            l6_ab.begin_step("engine_core")
        self.trace = []
        for blocks in self.deferred:  # deferred frees / CoW retentions of the previous step
            m.block_pool.free_blocks(blocks)
        self.deferred = []
        if self.running and rnd.random() < 0.02:
            self._preempt(self.running[-1])  # as if the step before had run out of blocks
        sched = []
        preempted_before = self.seen["preempt"]
        for r in list(self.running):
            if r not in self.running:
                continue
            out = self._schedule_one(r)
            if out is False:
                break  # Scheduler.schedule(): this request cannot be scheduled
            if out is not None:
                sched.append(out)
        if self.seen["preempt"] == preempted_before:
            if self.resume and rnd.random() < 0.7:
                out = self._resume_one()
                if out is not None:
                    sched.append(out)
            if rnd.random() < 0.35 and len(self.live) < 14:
                self._admit()
        _, retained = m.take_kv_cache_block_copies()
        if retained:
            self.deferred.append(retained)
        # _update_after_schedule (+ AsyncScheduler's placeholders)
        cur = []
        for r, k, ns in sched:
            r.num_computed_tokens += k
            r.num_in_flight_tokens += k
            decode = r.num_computed_tokens >= r.num_tokens + r.num_output_placeholders
            if decode:
                r.num_output_placeholders += 1 + ns
                r.spec_token_ids = [-1] * self.SPEC
            cur.append((r, k, ns, decode))
        # the previous step's outputs (async: one step late)
        for r, k, ns, decode in self.prev:
            r.num_in_flight_tokens -= k
            if r.request_id not in self.live:
                continue
            if r.num_stale_output_tokens > 0:
                r.num_stale_output_tokens -= k
                continue
            if not decode:
                continue
            acc = 0
            while acc < ns and rnd.random() < 0.6:
                acc += 1
            rej = ns - acc
            if r.num_computed_tokens > 0:
                r.num_computed_tokens -= rej
            if r.num_output_placeholders > 0:
                r.num_output_placeholders -= rej
            toks = [rnd.randrange(1, 50000) for _ in range(1 + acc)]
            for t in toks:
                r.append_output_token_ids(t)
            r.num_output_placeholders -= len(toks)
            if r.status == RequestStatus.RUNNING:
                m.cache_blocks(r, r.num_computed_tokens - r.num_output_placeholders)
            if r.num_output_tokens >= self.maxout[r.request_id] and r in self.running:
                self.seen["finish"] += 1
                self.running.remove(r)
                del self.live[r.request_id]
                if rnd.random() < 0.5:
                    m.free(r)
                else:  # an in-flight step may still write them: pop now, return to the pool next step
                    self.seen["popfree"] += 1
                    self.deferred.append(list(reversed(m.pop_blocks_for_free(r))))
                if rnd.random() < 0.6:
                    self.convs.append(list(r.all_token_ids))
        self.prev = cur
        self._land()
        if self.evict and self.running and rnd.random() < 0.05:
            # A connector reports blocks whose external KV failed (evict_blocks): held blocks lose their hashes.
            r = rnd.choice(self.running)
            sm = rnd.choice(m.coordinator.single_type_managers)
            ids = [b.block_id for b in sm.req_to_blocks.get(r.request_id, ()) if not b.is_null and b.block_hash]
            if ids:
                self.seen["evict"] += 1
                m.evict_blocks({rnd.choice(ids)})

    def run(self, steps):
        digests = []
        for _ in range(steps):
            self.step()
            digests.append(_digest(_state(self.m, list(self.live.values()), self.trace)))
        return digests

    def digest(self):
        return _digest(_state(self.m, list(self.live.values()), self.trace))


def _pair(steps=500, **kw):
    base = Stream(fast=False, **kw)
    db = base.run(steps)
    fast = Stream(fast=True, **kw)
    df = fast.run(steps)
    return base, db, fast, df


def _first_diff(db, df):
    return next((i for i, (a, b) in enumerate(zip(db, df)) if a != b), None)


@pytest.mark.parametrize(
    ("kind", "twins", "use_eagle", "seed"),
    [("fine", True, False, 1), ("fine", False, False, 2), ("coarse", True, False, 3), ("fine", True, True, 4),
     ("skew", True, False, 5)],
    ids=["served-fine-twins", "fine-no-twins", "coarse-twins", "fine-twins-eagle-drop-on", "skew-twins"],
)
def test_fast_path_matches_stock_over_random_streams(kind, twins, use_eagle, seed):
    base, db, fast, df = _pair(kind=kind, twins=twins, use_eagle=use_eagle, seed=seed)
    n = dict(fast.ea.n)
    print(f"STREAM {kind} twins={twins} eagle={use_eagle} seen={base.seen} engalloc={n}")
    assert _first_diff(base.calls, fast.calls) is None, f"state diverges at call {_first_diff(base.calls, fast.calls)}"
    assert len(base.calls) == len(fast.calls) > 10000
    assert _first_diff(db, df) is None, f"state diverges at step {_first_diff(db, df)}"
    assert base.seen == fast.seen
    s = base.seen
    # The stream exercises what the fast path must not skip.
    assert (s["preempt"] >= 2 and s["resume"] >= 2 and s["hit"] >= 1 and s["finish"] >= 10 and s["evict"] >= 2
            and s["popfree"] >= 2), s
    assert n["mismatch"] == 0 and not fast.ea.off
    assert n["fast"] > n["full"] > 0, n


def test_most_decode_calls_take_the_fast_path():
    st = Stream("fine", fast=True, twins=True, seed=11)
    st.run(500)
    counts = st.ea.n
    print(f"ENGAGEMENT {counts} seen={st.seen}")
    assert counts["mismatch"] == 0
    assert counts["fast"] > counts["full"] > 0, counts


def test_lever_off_never_takes_the_fast_path(lever):
    lever.setenv(ENV, "0")
    l6_ab._resolve()
    base, db, fast, df = _pair(steps=200, seed=5, twins=True)
    assert db == df
    assert fast.ea.n["fast"] == 0 and fast.ea.n["full"] == 0


def test_ab_windows_switch_the_lever(lever):
    """ab:1 with 2 classes: class-0 windows stay on the stock path, class-1 windows use the fast path."""
    clock = {"t": 0}
    lever.setattr(l6_ab, "_now_ns", lambda: clock["t"])
    lever.setenv("VLLM_TPU_L6_AB_PERIOD_MS", "1000")
    lever.setenv(ENV, "ab:1")
    l6_ab._resolve()
    lines = []
    lever.setattr(l6_ab.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    base = Stream(fast=False, twins=True, seed=6)
    fast = Stream(fast=True, twins=True, seed=6)
    per = {0: 0, 1: 0}
    for i in range(400):
        clock["t"] = (i // 40) * 1_000_000_000  # 40 steps per 1 s window
        base.step()
        before = fast.ea.n["fast"]
        fast.step()
        per[(i // 40) % 2] += fast.ea.n["fast"] - before
        assert base.digest() == fast.digest(), i
    assert per[0] == 0 and per[1] > 50, per
    assert any("role=engine_core" in ln and "engalloc.fast=" in ln for ln in lines), lines[-3:]


def test_check_mode_runs_stock_and_finds_no_mismatch():
    base = Stream(fast=False, twins=True, seed=7)
    db = base.run(400)
    st = Stream(fast=True, twins=True, seed=7, check=3)
    df = st.run(400)
    assert db == df
    assert st.ea.n["check"] > 20 and st.ea.n["fast"] > st.ea.n["check"] and st.ea.n["mismatch"] == 0, st.ea.n


def _loose_need(ea):
    """A broken certificate: one block of slack past every full-attention group's span."""
    orig = ea._cert

    def cert(request):
        c = orig(request)
        if c is not None:
            c.need += 512
        return c

    ea._cert = cert


def test_broken_certificate_diverges_without_check():
    """The equivalence test has teeth: a certificate that skips a needed allocation changes the tables
    (mismatch detection silenced, so the broken certificates stay in use)."""

    def hook(ea):
        _loose_need(ea)
        ea._mismatch = lambda *a, **k: None

    base = Stream(fast=False, twins=True, seed=8)
    db = base.run(400)
    st = Stream(fast=True, twins=True, seed=8, cert_hook=hook)
    df = st.run(400)
    assert _first_diff(db, df) is not None


def test_broken_certificate_is_caught_by_check(lever):
    errors = []
    lever.setattr(ea_mod.logger, "error", lambda fmt, *a: errors.append(fmt % a))
    base = Stream(fast=False, twins=True, seed=8)
    db = base.run(400)
    st = Stream(fast=True, twins=True, seed=8, check=1, cert_hook=_loose_need)
    df = st.run(400)
    assert db == df  # check=1: every predicted-quiet call ran stock
    assert st.ea.n["mismatch"] == 1 and st.ea.off
    assert any("MISMATCH" in e for e in errors)


def _active_request(st, pick, steps=800):
    """Step the stream until a running request has an active certificate c with pick(r, c)."""
    for _ in range(steps):
        st.step()
        for r in st.running:
            c = st.ea.certs.get(r.request_id)
            if c is not None and c.active and pick(r, c):
                return r, c
    raise AssertionError("no running request reached the wanted state")


def _full_block_bound(m, rid):
    """Largest cache length at which group m caches no new full block (fine-grained hits: lengths are not rounded)."""
    return (m.num_cached_block.get(rid, 0) + 1) * m.block_size - 1


def _cache_bound_is_tight(kind_of_bound):
    """Skew plan (Mamba 384 / full attention 512), a request whose next-full-block bound comes from one group alone:
    the certificate's cache-length bound equals that group's, cache_blocks at that length takes the fast path and
    changes nothing, and one token more is refused, runs stock and caches exactly that group's next full block."""
    st = Stream("skew", fast=True, twins=True, seed=15, evict=False)
    ea = st.ea
    fa, mam = ea.fa[0], ea.mamba[0]
    m, other = (fa, mam) if kind_of_bound == "fa" else (mam, fa)
    assert st.ea.P and m.block_size != other.block_size

    def pick(r, c):
        rid = r.request_id
        return c.nhi == _full_block_bound(m, rid) < _full_block_bound(other, rid) and c.nlo <= c.nhi

    r, c = _active_request(st, pick)
    rid = r.request_id
    while r.num_tokens < c.nhi + 2:  # tokens (and their hashes) past the bound, nothing scheduled
        r.append_output_token_ids(7)
    nc, oc = m.num_cached_block.get(rid, 0), other.num_cached_block.get(rid, 0)
    f, u, ops = ea.n["fast"], ea.n["full"], ea.ops
    st.m.cache_blocks(r, c.nhi)  # the last legal length: taken
    assert ea.n["fast"] == f + 1 and ea.ops == ops and m.num_cached_block.get(rid, 0) == nc
    st.m.cache_blocks(r, c.nhi + 1)  # one more: refused, and stock caches this group's next full block
    assert ea.n["full"] == u + 1 and ea.n["fast"] == f + 1
    assert m.num_cached_block.get(rid, 0) == nc + 1 and other.num_cached_block.get(rid, 0) == oc


def test_full_attention_full_block_bound_is_tight():
    _cache_bound_is_tight("fa")


def test_mamba_full_block_bound_is_tight():
    _cache_bound_is_tight("mamba")


def test_mamba_last_state_bound_is_tight():
    """The processed-token bound from a live Mamba last-state block: allocate_slots at that bound takes the fast
    path and leaves the block, one token later is refused and the stock path frees it. The retire bound is lifted
    for this request so the last-state term alone binds (in steady decode the two coincide, which is why the
    random streams cannot tell them apart); the certificate is then rebuilt by the fast path's own observation."""
    st = Stream(fast=True, twins=True, seed=14, evict=False)
    ea = st.ea

    def live_last(r):
        for m in ea.mamba:
            bl = m.req_to_blocks.get(r.request_id)
            last = m.last_state_block_idx.get(r.request_id)
            if bl and last is not None and last < len(bl) and not bl[last].is_null:
                return m, last
        return None

    def pick(r, c):
        main = r.num_computed_tokens + 1
        return (live_last(r) is not None and main <= c.main and main <= c.need
                and c.nlo <= min(main, r.num_tokens) <= c.nhi
                and c.nlo <= r.num_computed_tokens - r.num_output_placeholders <= c.nhi)

    r, _ = _active_request(st, pick)
    m, last = live_last(r)
    rid = r.request_id
    for sm in st.m.coordinator.single_type_managers:  # lift the retire bound (every Mamba group, twins too)
        if hasattr(sm, "_num_retired_blocks") and rid in sm.req_to_blocks:
            sm._num_retired_blocks[rid] = len(sm.req_to_blocks[rid])
    ea._drop(rid)
    n = r.num_computed_tokens - r.num_output_placeholders
    st.m.cache_blocks(r, n)  # stock: a new (inactive) certificate
    st.m.cache_blocks(r, n)  # observed quiet: active
    c = ea.certs[rid]
    assert c.active and c.p == (last + 1) * m.block_size, (c, last, m.block_size)
    blk = m.req_to_blocks[rid][last]

    def alloc_at(p):  # one new token, no lookahead, processed-token count p
        saved = r.num_in_flight_tokens
        r.num_in_flight_tokens = r.num_computed_tokens - p
        try:
            return st.m.allocate_slots(r, 1, num_lookahead_tokens=0)
        finally:
            r.num_in_flight_tokens = saved

    f, u = ea.n["fast"], ea.n["full"]
    assert alloc_at(c.p) is st.m.empty_kv_cache_blocks
    assert ea.n["fast"] == f + 1 and m.req_to_blocks[rid][last] is blk and not blk.is_null  # taken
    assert alloc_at(c.p + 1) is st.m.empty_kv_cache_blocks
    assert ea.n["full"] == u + 1 and m.req_to_blocks[rid][last].is_null  # refused; stock frees the block


def test_coordinator_calls_drop_the_certificate():
    """Any coordinator call for a request outside the fast path (here: free, a connector-style remove_skipped_blocks,
    a direct coordinator.cache_blocks) drops its certificate."""
    st = Stream(fast=True, twins=True, seed=9)
    live = []
    for _ in range(400):
        st.step()
        live = [r for r in st.running if r.request_id in st.ea.certs]
        if len(live) >= 3:
            break
    assert len(live) >= 3
    a, b, c = live[:3]
    st.m.free(a)
    assert a.request_id not in st.ea.certs
    st.m.remove_skipped_blocks(b.request_id, 0)
    assert b.request_id not in st.ea.certs
    st.m.coordinator.cache_blocks(c, c.num_computed_tokens - c.num_output_placeholders)
    assert c.request_id not in st.ea.certs


def test_hash_removal_on_a_replayed_block_drops_its_certificates():
    st = Stream(fast=True, twins=True, seed=10, evict=False)
    st.run(200)
    act = [(rid, c) for rid, c in st.ea.certs.items() if c.active and c.ins]
    assert act
    rid, c = act[0]
    blk = c.ins[0][1]
    st.m.evict_blocks({blk.block_id})
    assert rid not in st.ea.certs
    assert blk.block_id not in st.ea.deps


def test_reset_prefix_cache_drops_all_certificates():
    st = Stream(fast=True, twins=True, seed=12)
    st.ea.certs["x"] = object()
    st.ea.deps[5] = {"x"}
    old = st.m.block_pool.cached_block_hash_to_block
    assert st.m.block_pool.reset_prefix_cache()
    assert not st.ea.certs and not st.ea.deps
    new = st.m.block_pool.cached_block_hash_to_block
    assert new is not old and getattr(new, "_q38_engalloc", False)  # the new hash map is counted too
    ops = st.ea.ops
    new.insert(b"\x01" * 32 + (0).to_bytes(4, "big"), st.m.block_pool.blocks[1])
    assert st.ea.ops == ops + 1


def test_unsupported_manager_is_refused(lever):
    errors = []
    lever.setattr(ea_mod.logger, "error", lambda fmt, *a: errors.append(fmt % a))
    bf = torch.bfloat16
    specs = [FullAttentionSpec(block_size=64, num_kv_heads=1, head_size=64, dtype=bf),
             SlidingWindowSpec(block_size=64, num_kv_heads=1, head_size=64, dtype=bf, sliding_window=256)]
    cfg = KVCacheConfig(num_blocks=500, kv_cache_tensors=[],
                        kv_cache_groups=[KVCacheGroupSpec([f"l{i}"], s) for i, s in enumerate(specs)])
    m = KVCacheManager(cfg, max_model_len=4096, scheduler_block_size=64, hash_block_size=64,
                       enable_caching=True, use_eagle=False, dcp_world_size=1, pcp_world_size=1)
    assert ea_mod.install(m) is None
    assert "allocate_slots" not in vars(m)
    assert any("NOT installed" in e and "SlidingWindowManager" in e for e in errors), errors


def test_patches_install_only_when_configured(lever):
    from vllm.v1.core.kv_cache_manager import KVCacheManager as K
    from vllm.v1.engine.core import EngineCore

    lever.setattr(K, "__init__", K.__init__)
    lever.setattr(EngineCore, "__init__", EngineCore.__init__)
    for cls in (K, EngineCore):  # restored (removed) at teardown
        lever.setattr(cls, "_q38_engalloc_patch", False, raising=False)
    lever.setenv(ENV, "0")
    l6_ab._resolve()
    assert not ea_mod.patch_kv_cache_manager_for_engine_alloc()
    assert not ea_mod.patch_engine_core_for_engine_alloc()
    lever.setenv(ENV, "1")
    l6_ab._resolve()
    assert ea_mod.patch_kv_cache_manager_for_engine_alloc()
    assert ea_mod.patch_engine_core_for_engine_alloc()
    patch_registry.apply("platform_activation")
    patch_registry.apply("engine_core")
    m, _ = _manager("fine", 400)
    assert isinstance(getattr(m, "_q38_engalloc", None), ea_mod.EngineAllocFast)
    assert m.allocate_slots == m._q38_engalloc.allocate_slots


def test_engine_core_entry_installs_after_the_twins_patch():
    import inspect

    import vllm_torchtpu

    src = inspect.getsource(vllm_torchtpu._run_engine_core_with_tpu_patches)
    assert src.index("patch_kv_cache_manager_for_twins()") < src.index("patch_kv_cache_manager_for_engine_alloc()")
    assert "patch_engine_core_for_engine_alloc()" in src


def test_engine_core_hook_flushes_counts_and_begins_the_step(lever):
    lines = []
    lever.setattr(l6_ab.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    clock = {"t": 5 * 20000 * 1_000_000}
    lever.setattr(l6_ab, "_now_ns", lambda: clock["t"])
    l6_ab._resolve()
    st = Stream(fast=True, twins=True, seed=13)
    calls = []

    class Sched:
        def schedule(self):
            calls.append(l6_ab.on("engalloc"))
            st.ea.n["fast"] += 2
            return "out"

    class Core:
        scheduler = Sched()

    core = Core()
    assert ea_mod.install_engine_core(core)
    assert not ea_mod.install_engine_core(core)  # idempotent
    assert core.scheduler.schedule() == "out"
    assert calls == [True] and l6_ab._state["role"] == "engine_core"
    core.scheduler.schedule()  # flushes the first call's counts into the window
    clock["t"] += 21000 * 1_000_000
    core.scheduler.schedule()  # next 20 s uniform log window: the line carries the counts
    assert any("role=engine_core" in ln and "engalloc.fast=4" in ln for ln in lines), lines


def test_lever_is_registered_and_compile_ignored():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import tpu_compiler

    assert l6_ab.LEVERS["engalloc"] == ENV
    for name in (ENV, CHECK):
        assert name in envs.environment_variables
        assert name in tpu_compiler._TPU_COMPILE_ENV_IGNORED


def test_import_resolves_to_the_tree_under_test():
    """Guard against a staging that silently imports the pod's installed package."""
    import vllm_torchtpu

    want = os.environ.get("Q38_L6_EXPECT_TREE")
    if not want:
        pytest.skip("Q38_L6_EXPECT_TREE not set (the CPU harness sets it)")
    got = os.path.realpath(vllm_torchtpu.__file__)
    assert got.startswith(os.path.realpath(want) + os.sep), (got, want)


def test_check_steps_bound_the_check_turns(lever):
    """VLLM_TPU_L6_CHECK_STEPS=M: check turns run only in the engine core's first M schedule() calls
    (l6_ab.check_allowed()); after that every predicted-quiet call is replayed, and state stays stock's."""
    lever.setenv("VLLM_TPU_L6_CHECK_STEPS", "60")
    l6_ab._resolve()
    base = Stream(fast=False, twins=True, seed=7)
    st = Stream(fast=True, twins=True, seed=7, check=3)
    seen = []
    for i in range(400):
        base.step()
        st.step()
        assert base.digest() == st.digest(), i
        seen.append(dict(st.ea.n))
    at_m = seen[59]
    assert at_m["check"] > 5, at_m  # checks ran inside the window
    assert seen[-1]["check"] == seen[60]["check"], (seen[60], seen[-1])  # none after it
    assert seen[-1]["fast"] - seen[60]["fast"] > 100 and seen[-1]["mismatch"] == 0, seen[-1]


def test_ab_windows_resolve_without_served_ab(lever):
    """A tree without L5's SAB (megakernel_runtime/served_ab.py absent, e.g. q38-l2-stackmark): windowed
    engalloc resolves, SAB counts as unarmed even when its env var is set, and the lever is not refused."""
    import importlib.util

    has_sab = importlib.util.find_spec("vllm_torchtpu.megakernel_runtime.served_ab") is not None
    lever.setenv(ENV, "ab:1")
    lever.setenv("VLLM_TPU_L6_AB_STEPS", "200")
    lever.setenv("VLLM_TPU_SERVED_AB", "whole:4000")
    l6_ab._resolve()
    assert l6_ab.WINDOWED and not l6_ab.REFUSED
    if has_sab:
        assert l6_ab._sab_period() is not None
    else:
        assert l6_ab._sab_period() is None
