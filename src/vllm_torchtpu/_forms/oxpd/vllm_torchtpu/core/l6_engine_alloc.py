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
"""q38 L6 lever engalloc: quiet decode calls skip the engine core's per-group KV walk.

``VLLM_TPU_L6_ENGINE_ALLOC_FAST`` = 0 / 1 / ab:<classes> (read once per
``schedule()`` through ``core/l6_ab.py``; never at trace time, so it changes no
program and is compile-key ignored).

Why. Every running request calls ``KVCacheManager.allocate_slots`` once per
step (scheduler) and ``KVCacheManager.cache_blocks`` once per step (the async
scheduler's ``_update_request_with_output``). Each call walks every KV cache
group (65 on the PD decode: 37 Mamba, 13 full attention, 2 compressed-key, 13
rings) through four coordinator loops; D1k's twin classes cut that to one
representative per class, but a call still costs ~125 us (served T85m
ENGINEPHASES: alloc 418 us at 3.11 calls per step). In a decode step almost
every call changes nothing: no group needs a block, no full block becomes
cacheable, nothing is freed.

How. After a stock call for a decode request this module derives, from each
manager's per-request state, the input ranges for which the next stock call is
provably a no-op ("certificate"): ``num_tokens_need_slot`` below every
full-attention group's allocated span, ``num_tokens_main_model`` below every
Mamba group's span minus its speculative blocks, the processed-token count
below the next Mamba retire / last-state free, the cache length inside every
group's [partial-tail boundary, next full block) window, at most a hash block
of new tokens (so no Mamba prefill checkpoint), plus membership conditions
(no pending checkpoint, partial hit or CoW; the request already allocated,
already in the retention patch's checkpoint table). A certificate becomes
active only after one stock call inside its ranges was *observed* to be
quiet: same empty result, no block-pool operation (counted on the pool and its
hash map), the certificate recomputed after the call equal to the one before.
That observed call also records the prefix-cache inserts it attempted (the
full-attention prompt-tail entries, already registered, which re-mark their
block as written-this-step through the plugin's same-step-hit patch); an
active certificate replays exactly those calls. A call inside the ranges of an
active certificate returns ``empty_kv_cache_blocks`` (allocate_slots) or None
(cache_blocks) without touching the coordinator.

Invalidation. Any coordinator call for the request outside the fast path
(stock allocate / cache / remove_skipped_blocks / get_num_blocks_to_allocate /
free / pop_blocks_for_free, from anywhere) drops its certificate; a
prefix-cache hash removed from a block an active certificate replays an insert
on (hash-map pop, ``_remove_cached_block_hashes``, eviction) drops the
certificates that depend on it; ``reset_prefix_cache`` drops all.

Check mode. ``VLLM_TPU_L6_ENGINE_ALLOC_FAST_CHECK=N`` runs the stock path
instead of every N-th fast call and compares (quiet, same certificate, same
inserts). Any observed call whose certificate predicted quiet but was not is a
mismatch: one ERROR line, ``engalloc.mismatch`` counted, and the fast path is
off for the rest of the process (stock already ran, so state is stock's).
VLLM_TPU_L6_CHECK_STEPS=M (> 0) bounds the check turns to the engine core's first M
schedule() calls (l6_ab.check_allowed(); the engine core counts its own steps), so a served car
verifies during warm-up and the start of its first level and times the fast path afterwards.

With D1k twins (VLLM_TPU_KV_TWIN_GROUPS=1) a quiet stock call touches only each class's representative (its
twins are skipped when it changes nothing), so certificates read the representatives; without twins, every group.

Scope guards (refuse to install, one ERROR, lever inert): a coordinator other
than vLLM's HybridKVCacheCoordinator (twins' instance wrappers allowed), a
manager other than full attention / circular ring / align-mode Mamba, or any
modelled method that is neither vLLM's nor the Mamba checkpoint retention
patch (core/mamba_checkpoint_retention.py).

Engagement: counts ``engalloc.fast`` / ``engalloc.full`` / ``engalloc.check``
/ ``engalloc.mismatch`` in the engine core's ``q38 L6AB role=engine_core``
window line, plus one install line.
"""

from __future__ import annotations

import inspect
import os

from vllm_torchtpu.core import l6_ab as _ab
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

LEVER = "engalloc"
ENV = "VLLM_TPU_L6_ENGINE_ALLOC_FAST"
CHECK_ENV = "VLLM_TPU_L6_ENGINE_ALLOC_FAST_CHECK"
_BIG = 1 << 62

_ALLOC_PARAMS = (
    "self", "request", "num_new_tokens", "num_new_computed_tokens", "new_computed_blocks",
    "num_lookahead_tokens", "num_external_computed_tokens", "delay_cache_blocks", "num_encoder_tokens",
    "full_sequence_must_fit", "reserved_blocks", "has_scheduled_reqs",
)
# Source files of the modelled vLLM code (checked on the function's code object: functools.wraps copies
# __module__ / __qualname__, so only co_filename tells a wrapper from the original).
_VLLM_STM = "/vllm/v1/core/single_type_kv_cache_manager.py"
_VLLM_COORD = "/vllm/v1/core/kv_cache_coordinator.py"
_VLLM_KVM = "/vllm/v1/core/kv_cache_manager.py"
_VLLM_POOL = "/vllm/v1/core/block_pool.py"
_RETENTION_FILE = "/vllm_torchtpu/core/mamba_checkpoint_retention.py"
_RETENTION_FLAG = "_tpu_mamba_checkpoint_retention_patch"

_FA_METHODS = ("remove_skipped_blocks", "get_num_skipped_tokens", "get_num_blocks_to_allocate",
               "allocate_new_blocks", "cache_blocks", "_cache_partial_tail_block")
_RING_METHODS = ("remove_skipped_blocks", "get_num_blocks_to_allocate", "allocate_new_blocks",
                 "cache_blocks", "_claim_ring_block")
_MAMBA_METHODS = ("remove_skipped_blocks", "_remove_blocks_in_range", "get_num_skipped_tokens",
                  "get_num_blocks_to_allocate", "_needs_internal_checkpoint", "allocate_new_blocks",
                  "cache_blocks", "_cache_partial_tail_block")
_COORD_METHODS = ("remove_skipped_blocks", "get_num_blocks_to_allocate", "allocate_new_blocks",
                  "cache_blocks", "_align_cacheable")
# Coordinator calls that touch one request's state; any of them outside the fast path drops its certificate.
_COORD_REQ_METHODS = ("remove_skipped_blocks", "get_num_blocks_to_allocate", "allocate_new_computed_blocks",
                      "allocate_new_blocks", "cache_blocks", "free", "pop_blocks_for_free")
_POOL_OPS = ("get_new_blocks", "free_blocks", "touch", "cache_full_blocks", "move_block_hashes",
             "evict_blocks", "unpin_blocks", "_maybe_evict_cached_block")


def configured() -> bool:
    """Whether the lever's flag is set to anything but off (installs the hooks)."""
    return _ab._cfg.get(LEVER, (False, None)) != (False, None)


def _check_every() -> int:
    raw = os.environ.get(CHECK_ENV, "").strip()
    if not raw:
        return 0
    v = int(raw)
    if v < 0:
        raise ValueError(f"{CHECK_ENV} must be >= 0, got {v}")
    return v


def _code_file(fn) -> str:
    code = getattr(fn, "__code__", None)
    return code.co_filename.replace("\\", "/") if code is not None else ""


def _vllm_fn(fn, path: str) -> bool:
    """`fn` is the function defined in vLLM's `path` (not a wrapper of it)."""
    return _code_file(fn).endswith(path)


def _mamba_fn_ok(fn) -> bool:
    """vLLM's MambaManager method, or the checkpoint-retention patch's wrapper of exactly it."""
    if _vllm_fn(fn, _VLLM_STM):
        return True
    return (_code_file(fn).endswith(_RETENTION_FILE) and bool(getattr(fn, _RETENTION_FLAG, False))
            and _vllm_fn(getattr(fn, "__wrapped__", None), _VLLM_STM))


class _Cert:
    """Input ranges in which the next stock call for one request is a no-op, and the
    prefix-cache inserts such a call attempts (known once one was observed)."""

    __slots__ = ("need", "main", "p", "nlo", "nhi", "new", "sig", "active", "ins")

    def __init__(self, need, main, p, nlo, nhi, new, sig):
        self.need, self.main, self.p, self.nlo, self.nhi, self.new, self.sig = need, main, p, nlo, nhi, new, sig
        self.active = False
        self.ins = ()

    def same(self, o) -> bool:
        return (o is not None and self.sig == o.sig and self.need == o.need and self.main == o.main
                and self.p == o.p and self.nlo == o.nlo and self.nhi == o.nhi and self.new == o.new)

    def __repr__(self):
        return (f"cert(need<={self.need} main<={self.main} p<={self.p} n in [{self.nlo},{self.nhi}] "
                f"new<={self.new} active={self.active} ins={len(self.ins)})")


_INSTANCES: list["EngineAllocFast"] = []


class EngineAllocFast:
    """Instance-level fast path for one KVCacheManager (see the module docstring)."""

    def __init__(self, kvm, check: int | None = None):
        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
        from vllm.v1.request import RequestStatus

        self._mk = make_block_hash_with_group_id
        reason = _unsupported(kvm)
        if reason:
            raise ValueError(reason)
        self.kvm = kvm
        coord = kvm.coordinator
        pool = kvm.block_pool
        self.coord, self.pool = coord, pool
        self.running = RequestStatus.RUNNING
        self.empty = kvm.empty_kv_cache_blocks
        self.mml = kvm.max_model_len
        self.hash = pool.hash_block_size
        self.P = bool(coord.enable_partial_hash_hits)
        self.S = coord.scheduler_block_size
        self.r = coord.num_reprefillable_tokens
        self.check = _check_every() if check is None else check
        # The managers a quiet stock call touches: with D1k twins, one representative per class (a representative
        # that changes nothing skips its twins); without, every group.
        tc = getattr(coord, "_q38_twins", None)
        self.twins = tc is not None
        managers = list(tc.reps) if tc is not None else list(coord.single_type_managers)
        self.fa, self.ring, self.mamba = _classify(managers)
        self.caching = bool(kvm.enable_caching)
        self.new_max = self.hash if self.mamba else _BIG
        self.retention = bool(self.mamba) and bool(getattr(type(self.mamba[0]).cache_blocks, _RETENTION_FLAG, False))
        self.certs: dict[str, _Cert] = {}
        self.deps: dict[int, set[str]] = {}
        self.ops = 0
        self.rec: list | None = None
        self.k = 0
        self.off = False
        self.n = {"fast": 0, "full": 0, "check": 0, "mismatch": 0}  # cumulative
        self._flushed = dict(self.n)
        self._ins_inner = pool._insert_block_hash
        self._wrap_pool()
        self._wrap_coordinator()
        self._alloc = kvm.allocate_slots
        self._cache = kvm.cache_blocks
        kvm.allocate_slots = self.allocate_slots
        kvm.cache_blocks = self.cache_blocks
        kvm._q38_engalloc = self
        _INSTANCES.append(self)

    # ---- hooks on the pool and the coordinator -------------------------------------------------
    def _count(self, fn):
        def w(*a, **k):
            self.ops += 1
            return fn(*a, **k)

        w.__wrapped__ = fn
        return w

    def _wrap_pool(self):
        pool = self.pool
        for name in _POOL_OPS:
            fn = getattr(pool, name, None)
            if fn is not None:
                setattr(pool, name, self._count(fn))
        remove = pool._remove_cached_block_hashes

        def _remove_cached_block_hashes(block):
            self.ops += 1
            if block.block_id in self.deps:
                self._inval_block(block.block_id)
            return remove(block)

        pool._remove_cached_block_hashes = _remove_cached_block_hashes
        inner = self._ins_inner

        def _insert_block_hash(block_hash_with_group_id, block, num_tokens):
            rec = self.rec
            if rec is not None:
                rec.append((block_hash_with_group_id, block, num_tokens))
            return inner(block_hash_with_group_id, block, num_tokens)

        pool._insert_block_hash = _insert_block_hash
        reset = pool.reset_prefix_cache

        def reset_prefix_cache(*a, **k):
            self.ops += 1
            self.certs.clear()
            self.deps.clear()
            r = reset(*a, **k)
            self._wrap_hash_map()
            return r

        pool.reset_prefix_cache = reset_prefix_cache
        self._wrap_hash_map()

    def _wrap_hash_map(self):
        hmap = self.pool.cached_block_hash_to_block
        if getattr(hmap, "_q38_engalloc", False):
            return
        ins, pop = hmap.insert, hmap.pop

        def insert(*a, **k):
            self.ops += 1
            return ins(*a, **k)

        def pop_(key, block_id, *a, **k):
            self.ops += 1
            if block_id in self.deps:
                self._inval_block(block_id)
            return pop(key, block_id, *a, **k)

        hmap.insert, hmap.pop = insert, pop_
        hmap._q38_engalloc = True

    def _wrap_coordinator(self):
        coord = self.coord
        for name in _COORD_REQ_METHODS:
            fn = getattr(coord, name)
            if name == "cache_blocks":
                def w(request, *a, _fn=fn, **k):
                    self._drop(request.request_id)
                    return _fn(request, *a, **k)
            else:
                def w(*a, _fn=fn, **k):
                    self._drop(k["request_id"] if "request_id" in k else a[0])
                    return _fn(*a, **k)
            w.__wrapped__ = fn
            setattr(coord, name, w)

    # ---- certificates --------------------------------------------------------------------------
    def _drop(self, rid):
        c = self.certs.pop(rid, None)
        if c is not None and c.active:
            self._undep(rid, c)

    def _undep(self, rid, c):
        deps = self.deps
        for _, b, _ in c.ins:
            s = deps.get(b.block_id)
            if s is not None:
                s.discard(rid)
                if not s:
                    del deps[b.block_id]

    def _set(self, rid, c):
        self._drop(rid)
        self.certs[rid] = c
        if c.active:
            deps = self.deps
            for _, b, _ in c.ins:
                deps.setdefault(b.block_id, set()).add(rid)

    def _inval_block(self, block_id):
        for rid in list(self.deps.pop(block_id, ())):
            self._drop(rid)

    def _align(self, n):
        return n if self.P else n // self.S * self.S

    def _tok(self, eagle, bs, n):
        """The token count HybridKVCacheCoordinator.cache_blocks hands this manager for cache length n."""
        a = self._align(n)
        if eagle and a > 0:
            nf = max(0, n - self.r)
            return min(nf, self._align(nf) + bs)
        return a

    def _hi(self, eagle, bs, t):
        """Largest n with tok(n) <= t (tok is non-decreasing in n)."""
        if t < 0:
            return -1
        if self.P:
            return t + self.r if eagle else t
        if not eagle:
            return t // self.S * self.S + self.S - 1
        lo, hi = 0, self.mml + self.r + self.S + bs + 1
        if self._tok(eagle, bs, lo) > t:
            return -1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._tok(eagle, bs, mid) <= t:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _lo(self, eagle, bs, b):
        """Smallest n with tok(n) >= b (b >= 1)."""
        if self.P:
            return b + self.r if eagle else b
        if not eagle:
            return -(-b // self.S) * self.S
        lo, hi = 0, self.mml + self.r + self.S + bs + 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._tok(eagle, bs, mid) >= b:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def _cert(self, request):
        """The certificate for `request` from the managers' current state, or None when a membership
        condition fails (the next call is not provably quiet)."""
        rid = request.request_id
        prompt = request.num_prompt_tokens
        hsz = self.hash
        caching = self.caching
        need, main, p, nlo, nhi = self.mml, _BIG, _BIG, prompt + 1, _BIG
        sig = []
        app = sig.append
        hmap = self.pool.cached_block_hash_to_block
        hashes = None
        for m in self.fa:
            blocks = m.req_to_blocks.get(rid)
            if blocks is None or rid in m._partial_hit_reqs:
                return None
            nb = len(blocks)
            bs = m.block_size
            if nb * bs < need:
                need = nb * bs
            nc = -1
            if caching and m.enable_caching:
                eagle = m.use_eagle
                if m.kv_cache_spec.prefix_cacheable:
                    nc = m.num_cached_block.get(rid, 0)
                    fh = self._hi(eagle, bs, (nc + 1) * bs - 1)  # the next full block becomes cacheable past fh
                    if fh < nhi:
                        nhi = fh
                if bs != hsz:
                    boundary = prompt // hsz * hsz
                    if boundary and boundary % bs:
                        idx = boundary // bs
                        reg = 2
                        if idx < nb and not blocks[idx].is_null:
                            if hashes is None:
                                hashes = request.block_hashes
                            j = boundary // hsz
                            if j > len(hashes):
                                return None
                            key = self._mk(hashes[j - 1], m.kv_cache_group_id)
                            blk = blocks[idx]
                            reg = int(blk.block_hash == key or hmap.contain(key, blk.block_id))
                        lo = self._lo(eagle, bs, boundary)
                        if reg:
                            if lo > nlo:
                                nlo = lo  # always the cache_partial_block branch, on a registered entry
                        elif lo - 1 < nhi:
                            nhi = lo - 1  # quiet only below the branch (it would register the entry)
                        app(reg)
            app((nb, nc))
        for m in self.ring:
            if not m.req_to_blocks.get(rid):
                return None
        for m in self.mamba:
            blocks = m.req_to_blocks.get(rid)
            if (blocks is None or rid in m._partial_hit_reqs or rid in m._checkpoints
                    or rid not in m._allocated_block_reqs):
                return None
            nb = len(blocks)
            bs = m.block_size
            t = (nb - m.num_speculative_blocks) * bs
            if t < main:
                main = t
            r = m._num_retired_blocks.get(rid, 0)
            if nb > r and (r + 1) * bs < p:
                p = (r + 1) * bs
            last = m.last_state_block_idx.get(rid)
            live = last is not None and (last >= nb or blocks[last] != m._null_block)
            if live and (last + 1) * bs < p:
                p = (last + 1) * bs
            nc = -1
            if caching and m.enable_caching:
                if self.retention and nb and rid not in m.__dict__.get("_tpu_own_checkpoints", ()):
                    return None
                eagle = m.use_eagle
                if m.kv_cache_spec.prefix_cacheable:
                    nc = m.num_cached_block.get(rid, 0)
                    mh = self._hi(eagle, bs, (nc + 1) * bs - 1)
                    if mh < nhi:
                        nhi = mh
                if bs != hsz and (self.P or eagle):
                    # The align-mode partial tail is a no-op past the prompt. (With bs == hash it returns before
                    # registering anything; with scheduler-block rounding and no eagle drop its token count is a
                    # multiple of the scheduler block, hence of bs, and it returns too.)
                    lo = self._lo(eagle, bs, prompt + 1)
                    if lo > nlo:
                        nlo = lo
            app((nb, nc, r, last, live))
        return _Cert(need, main, p, nlo, nhi, self.new_max, tuple(sig))

    # ---- the two fast paths --------------------------------------------------------------------
    def _replay(self, c):
        inner = self._ins_inner
        for key, block, nt in c.ins:
            inner(key, block, nt)

    def _check_turn(self) -> bool:
        self.k += 1
        if self.k >= self.check:
            self.k = 0
            return True
        return False

    def allocate_slots(self, request, num_new_tokens, num_new_computed_tokens=0, new_computed_blocks=None,
                       num_lookahead_tokens=0, num_external_computed_tokens=0, delay_cache_blocks=False,
                       num_encoder_tokens=0, full_sequence_must_fit=False, reserved_blocks=0,
                       has_scheduled_reqs=True):
        args = (request, num_new_tokens, num_new_computed_tokens, new_computed_blocks, num_lookahead_tokens,
                num_external_computed_tokens, delay_cache_blocks, num_encoder_tokens, full_sequence_must_fit,
                reserved_blocks, has_scheduled_reqs)
        if self.off or not _ab.on(LEVER):
            return self._alloc(*args)
        n = -1
        c = None
        within = checking = False
        if (num_new_tokens > 0 and new_computed_blocks is None and not num_new_computed_tokens
                and not num_external_computed_tokens and not delay_cache_blocks and not num_encoder_tokens
                and not full_sequence_must_fit and not reserved_blocks and request.status is self.running):
            nct = request.num_computed_tokens
            main = nct + num_new_tokens
            nt = request.num_tokens
            n = main if main < nt else nt
            c = self.certs.get(request.request_id)
            if (c is not None and main + num_lookahead_tokens <= c.need and main <= c.main
                    and num_new_tokens <= c.new and nct - request.num_in_flight_tokens <= c.p
                    and c.nlo <= n <= c.nhi):
                within = True
                if c.active:
                    if not (self.check and self._check_turn() and _ab.check_allowed()):
                        self._replay(c)
                        self.n["fast"] += 1
                        return self.empty
                    checking = True
        ops0 = self.ops
        self.rec = rec = []
        try:
            res = self._alloc(*args)
        finally:
            self.rec = None
        self.n["full"] += 1
        if checking:
            self.n["check"] += 1
        if n > request.num_prompt_tokens and res is not None:
            self._after(request, c, within, res is self.empty and self.ops == ops0, rec, "allocate_slots")
        return res

    def cache_blocks(self, request, num_computed_tokens):
        if self.off or not _ab.on(LEVER):
            return self._cache(request, num_computed_tokens)
        c = self.certs.get(request.request_id)
        within = checking = False
        if c is not None and c.nlo <= num_computed_tokens <= c.nhi:
            within = True
            if c.active:
                if not (self.check and self._check_turn() and _ab.check_allowed()):
                    self._replay(c)
                    self.n["fast"] += 1
                    return None
                checking = True
        ops0 = self.ops
        self.rec = rec = []
        try:
            res = self._cache(request, num_computed_tokens)
        finally:
            self.rec = None
        self.n["full"] += 1
        if checking:
            self.n["check"] += 1
        if num_computed_tokens > request.num_prompt_tokens:
            self._after(request, c, within, self.ops == ops0, rec, "cache_blocks")
        return res

    def _after(self, request, pre, within, quiet, rec, kind):
        """After a stock call: verify a predicted-quiet call, activate an observed-quiet certificate,
        store the new (inactive) certificate otherwise."""
        rid = request.request_id
        new = self._cert(request)
        if within:
            same = new is not None and new.same(pre)
            ins_ok = (not pre.active) or [(k, id(b), t) for k, b, t in rec] == [
                (k, id(b), t) for k, b, t in pre.ins]
            if not (quiet and same and ins_ok):
                self._mismatch(rid, kind, pre, new, quiet, same, ins_ok, rec)
                return
            new.active = True
            new.ins = tuple(rec)
        if new is not None:
            self._set(rid, new)

    def _mismatch(self, rid, kind, pre, new, quiet, same, ins_ok, rec):
        self.n["mismatch"] += 1
        logger.error(
            "q38 L6 engalloc: MISMATCH on %s req=%s: predicted quiet by %r, observed quiet=%s "
            "cert_same=%s inserts_same=%s (inserts %d) -> new %r; the fast path is OFF for this engine core "
            "(stock already ran; state is stock's)",
            kind, rid, pre, quiet, same, ins_ok, len(rec), new,
        )
        self.off = True
        self.certs.clear()
        self.deps.clear()

    def flush(self):
        """Note the counts since the last flush in the current l6_ab window."""
        done = self._flushed
        for key, v in self.n.items():
            d = v - done[key]
            if d:
                _ab.note(LEVER, key, d)
                done[key] = v


def _classify(managers):
    from vllm.v1.core.single_type_kv_cache_manager import (
        CircularBufferManager,
        FullAttentionManager,
        MambaManager,
    )

    fa, ring, mamba = [], [], []
    for m in managers:
        t = type(m)
        if issubclass(t, CircularBufferManager):
            ring.append(m)
        elif issubclass(t, MambaManager):
            mamba.append(m)
        elif issubclass(t, FullAttentionManager):
            fa.append(m)
    return fa, ring, mamba


def _kind(m) -> str:
    fa, ring, mamba = _classify([m])
    return "fa" if fa else "ring" if ring else "mamba" if mamba else "?"


def _unsupported(kvm) -> str:
    """Why the fast path cannot model this KVCacheManager ('' when it can)."""
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.single_type_kv_cache_manager import (
        CircularBufferManager,
        FullAttentionManager,
        MambaManager,
    )

    if type(kvm) is not KVCacheManager:
        return f"KV cache manager is {type(kvm).__name__}, not vLLM's KVCacheManager"
    for name in ("allocate_slots", "cache_blocks", "create_kv_cache_blocks"):
        if name in vars(kvm):
            return f"KVCacheManager.{name} is already wrapped on the instance"
        if not _vllm_fn(getattr(KVCacheManager, name), _VLLM_KVM):
            return f"KVCacheManager.{name} is patched"
    params = tuple(inspect.signature(KVCacheManager.allocate_slots).parameters)
    if params != _ALLOC_PARAMS:
        return f"allocate_slots signature {params} is not the modelled one"
    if not kvm.enable_caching:
        return "prefix caching is off (nothing to skip beyond the no-cache coordinator)"
    coord = kvm.coordinator
    if not isinstance(coord, HybridKVCacheCoordinator):
        return f"coordinator is {type(coord).__name__}, not HybridKVCacheCoordinator"
    for name in _COORD_METHODS:
        if not _vllm_fn(getattr(type(coord), name), _VLLM_COORD):
            return f"{type(coord).__name__}.{name} is patched"
        if name in vars(coord) and getattr(coord, "_q38_twins", None) is None:
            return f"coordinator.{name} is wrapped on the instance by something other than D1k twins"
    pool = kvm.block_pool
    if any(m.block_pool is not pool for m in coord.single_type_managers):
        return "managers do not share one block pool"
    if not _vllm_fn(type(pool).cache_partial_block, _VLLM_POOL):
        return "BlockPool.cache_partial_block is patched"
    for m in coord.single_type_managers:
        t = type(m)
        if issubclass(t, CircularBufferManager):
            base, names, ok = CircularBufferManager, _RING_METHODS, None
        elif issubclass(t, MambaManager):
            if m.mamba_cache_mode != "align":
                return f"group {m.kv_cache_group_id}: Mamba cache mode {m.mamba_cache_mode!r} (only 'align' modelled)"
            base, names, ok = MambaManager, _MAMBA_METHODS, _mamba_fn_ok
        elif issubclass(t, FullAttentionManager):
            base, names, ok = FullAttentionManager, _FA_METHODS, None
        else:
            return f"group {m.kv_cache_group_id}: {t.__name__} is not modelled"
        for name in names:
            fn = getattr(t, name)
            if fn is not getattr(base, name):
                return f"group {m.kv_cache_group_id}: {t.__name__}.{name} overrides {base.__name__}'s"
            if not (ok(fn) if ok else _vllm_fn(fn, _VLLM_STM)):
                return f"{base.__name__}.{name} is patched ({_code_file(fn)})"
            if name in vars(m):
                return f"group {m.kv_cache_group_id}: {name} is wrapped on the instance"
    return ""


def install(kvm, check: int | None = None) -> EngineAllocFast | None:
    """Install the fast path on one KVCacheManager; None (one ERROR) when it cannot be modelled."""
    reason = _unsupported(kvm)
    if reason:
        logger.error("q38 L6 engalloc: NOT installed (%s); the lever stays inert", reason)
        return None
    ea = EngineAllocFast(kvm, check)
    logger.info(
        "q38 L6 engalloc: installed (%s=%s, check every %d): %d groups = %d full attention + %d ring + %d Mamba; "
        "hash block %d, scheduler block %d, partial hash hits %s, twins %s (certificates read %d managers)",
        ENV, os.environ.get(ENV, ""), ea.check, len(kvm.coordinator.single_type_managers),
        *(sum(1 for m in kvm.coordinator.single_type_managers if _kind(m) == k) for k in ("fa", "ring", "mamba")),
        ea.hash, ea.S, ea.P, "on" if ea.twins else "off", len(ea.fa) + len(ea.ring) + len(ea.mamba),
    )
    return ea


def flush() -> None:
    """Move this process's engagement counts into the current l6_ab window."""
    for ea in _INSTANCES:
        ea.flush()


def install_engine_core(core) -> bool:
    """Wrap one engine core's scheduler.schedule: counts flushed, then l6_ab.begin_step('engine_core')."""
    sched = getattr(core, "scheduler", None)
    if sched is None or getattr(sched.schedule, "_q38_engalloc", False):
        return False
    orig = sched.schedule

    def schedule(*args, **kwargs):
        flush()
        _ab.begin_step("engine_core")
        return orig(*args, **kwargs)

    schedule._q38_engalloc = True  # type: ignore[attr-defined]
    schedule.__wrapped__ = orig  # type: ignore[attr-defined]
    sched.schedule = schedule
    return True


def patch_kv_cache_manager_for_engine_alloc() -> bool:
    """Install on every KVCacheManager after construction (engine-core entry point, after D1k twins);
    idempotent; a no-op unless the lever's flag is set."""
    if not configured():
        return False
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    if getattr(KVCacheManager, "_q38_engalloc_patch", False):
        return True
    orig_init = KVCacheManager.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            install(self)
        except Exception:  # never stop the engine for an optimisation
            logger.exception("q38 L6 engalloc: install failed; continuing on the stock path")

    KVCacheManager.__init__ = __init__
    KVCacheManager._q38_engalloc_patch = True
    return True


def patch_engine_core_for_engine_alloc() -> bool:
    """Call l6_ab.begin_step('engine_core') once per schedule in every engine core; a no-op unless set."""
    if not configured():
        return False
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore, "_q38_engalloc_patch", False):
        return True
    orig_init = EngineCore.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            install_engine_core(self)
        except Exception:
            logger.exception("q38 L6 engalloc: engine-core hook failed; the lever stays off")

    EngineCore.__init__ = __init__
    EngineCore._q38_engalloc_patch = True
    return True
