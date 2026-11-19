# SPDX-License-Identifier: Apache-2.0
"""q38 D1k: skip the per-request work of identical ("twin") KV cache groups (VLLM_TPU_KV_TWIN_GROUPS=1).

The PD decode's unified pool plans 65 single-layer KV cache groups (37 Mamba,
13 circular-buffer rings, 12 + 1 full-attention, 2 MLA). Every scheduled
request calls ``KVCacheManager.allocate_slots`` once per step, and the
coordinator calls each group's single-type manager in four loops
(``remove_skipped_blocks``, ``get_num_blocks_to_allocate``,
``allocate_new_blocks``, ``cache_blocks``). Measured on the served decode
(D1 CP0 arm C, engine-core timers): allocate_slots costs ~0.28 ms per running
request per step, 6.35 ms per step at c64. Managers of one class (same manager
type, same allocation-relevant spec and settings) receive the same arguments
and hold the same per-request state, so they make the same decision; in a
decode step that decision is almost always "nothing to do".

With the flag on, those four coordinator loops first run over one
representative manager per class (the class's first group). A representative
whose call changed nothing for the request (no block-pool mutation, no
prefix-cache key inserted or removed, the same block-list length and tail, the
same per-request bookkeeping and pending lists) speaks for its whole class: its
twins are skipped, their ``allocate_new_blocks`` result is ``[]``, and
``get_num_blocks_to_allocate`` counts the representative's answer once per
member. Only the twins of a class whose representative changed something run,
through the same coordinator loop. ``get_num_blocks_to_allocate`` for a prefix
hit (non-empty ``new_computed_blocks``) runs the original loop unchanged;
frees, hit lookups and every other path are untouched, so twin state never
diverges. Block tables, prefix hits and the prefix-cache hash map are identical
with and without the flag (CPU identity tests: the served plan with request
churn, and multi-turn PD admissions with prefix hits and partial prompt-tail
entries, eagle block drop on and off).

The prefix-cache key counter (the hash map's ``insert`` / ``pop``) is from
HOST-MM (q38-2c, d0733368b): a representative call that only registered a
partial prompt-tail entry (``block_pool.cache_partial_block``) used to read as
"unchanged".
"""

from __future__ import annotations

from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_POOL_MUTATORS = ("get_new_blocks", "free_blocks", "cache_full_blocks", "touch", "evict_blocks")
# Per-group identity or memory layout: never part of an allocation decision.
_ID_FIELDS = {"kv_cache_group_id"}
_LAYOUT_FIELDS = {"shapes", "dtypes", "dtype", "head_size", "head_size_v", "num_kv_heads", "page_size_padded",
                  "state_content_bytes", "num_head_slots", "kv_quant_mode", "tp_replicated", "mamba_type"}


def enabled() -> bool:
    from vllm_torchtpu import envs

    return bool(envs.VLLM_TPU_KV_TWIN_GROUPS)


def _class_key(m: Any):
    spec = m.kv_cache_spec
    fields = getattr(spec, "__dataclass_fields__", {})
    spec_key = tuple((f, repr(getattr(spec, f))) for f in sorted(fields) if f not in _LAYOUT_FIELDS)
    attrs = tuple(
        (k, repr(v)) for k, v in sorted(vars(m).items())
        if (isinstance(v, (bool, int, float, str)) or v is None) and k not in _ID_FIELDS
    )
    return (type(m), type(spec), spec_key, attrs)


_DICT_MUTATORS = ("__setitem__", "__delitem__", "pop", "popitem", "setdefault", "update", "clear", "__missing__")
_SET_MUTATORS = ("add", "discard", "remove", "pop", "clear", "update", "difference_update",
                 "intersection_update", "symmetric_difference_update", "__ior__", "__iand__", "__isub__", "__ixor__")
_COUNTING: dict[type, type] = {}


def _counting_type(base: type) -> type:
    """A subclass of a dict/set type whose mutators bump the owning class's version (``self._q38_bump``)."""
    cls = _COUNTING.get(base)
    if cls is None:
        names = _DICT_MUTATORS if issubclass(base, dict) else _SET_MUTATORS
        ns = {"__slots__": ("_q38_bump",)} if not hasattr(base, "__dict__") else {}

        def make(name, f):
            # Bump only when the container really changes, so idempotent writes (a pop of an absent key, a
            # set of the same value, an add of a present member) do not force the twins to run.
            if name == "__setitem__":
                def w(self, key, value):
                    if key not in self or dict.__getitem__(self, key) is not value:
                        self._q38_bump()
                    return f(self, key, value)
            elif name == "pop" and issubclass(base, dict):
                def w(self, key, *default):
                    if key in self:
                        self._q38_bump()
                    return f(self, key, *default)
            elif name == "setdefault":
                def w(self, key, *default):
                    if key not in self:
                        self._q38_bump()
                    return f(self, key, *default)
            elif name == "add":
                def w(self, item):
                    if item not in self:
                        self._q38_bump()
                    return f(self, item)
            elif name == "discard":
                def w(self, item):
                    if item in self:
                        self._q38_bump()
                    return f(self, item)
            elif name in ("clear", "popitem"):
                def w(self, *a, **k):
                    if len(self):
                        self._q38_bump()
                    return f(self, *a, **k)
            else:
                def w(self, *a, **k):
                    self._q38_bump()
                    return f(self, *a, **k)
            return w

        for name in names:
            if hasattr(base, name):
                ns[name] = make(name, getattr(base, name))
        cls = type("Q38Counting" + base.__name__, (base,), ns)
        _COUNTING[base] = cls
    return cls


class _Class:
    """One class of twin managers. The representative's per-request dicts and sets are swapped for counting
    copies, so any write to them bumps ``version``; lists (pending copies, new block ids) are compared by
    length; the request's own block list by length and tail."""

    def __init__(self, members: list[Any]):
        from collections import defaultdict

        self.rep = members[0]
        self.twins = members[1:]
        self.version = 0
        self.lists = []
        if not self.twins:
            return
        for k, v in list(vars(self.rep).items()):
            if k == "req_to_blocks":
                continue
            if isinstance(v, (dict, set)) and not isinstance(v, frozenset):
                cls = _counting_type(type(v))
                new = cls(v.default_factory, v) if isinstance(v, defaultdict) else cls(v)
                new._q38_bump = self._bump
                setattr(self.rep, k, new)
            elif isinstance(v, list):
                self.lists.append(k)

    def _bump(self):
        self.version += 1


class TwinCoordinator:
    """Replaces four loops of one coordinator instance (attributes only)."""

    def __init__(self, coord: Any):
        self.coord = coord
        self.all = list(coord.single_type_managers)
        self.index = {id(m): i for i, m in enumerate(self.all)}
        pool = self.all[0].block_pool
        self.pool = pool
        self.mutations = 0
        for name in _POOL_MUTATORS:
            fn = getattr(pool, name, None)
            if fn is not None:
                setattr(pool, name, self._count(fn))
        # Every prefix-cache key change, whichever pool method makes it: full blocks, the partial tail entries that
        # FullAttentionManager.cache_blocks registers through cache_partial_block when its block is larger than the
        # hash block (a no-op once registered, so the public call cannot be counted), copy-on-write moves, evictions.
        # A representative whose call only registered a partial tail changed the pool, so its twins must run.
        self._count_hash_map()
        reset = getattr(pool, "reset_prefix_cache", None)
        if reset is not None:  # reset replaces the hash map; count it and wrap the new one

            def reset_and_rewrap(*a, **k):
                self.mutations += 1
                r = reset(*a, **k)
                self._count_hash_map()
                return r

            pool.reset_prefix_cache = reset_and_rewrap
        classes: dict[Any, list[Any]] = {}
        for m in self.all:
            classes.setdefault(_class_key(m), []).append(m)
        self.classes = [_Class(ms) for ms in classes.values()]
        for ms in classes.values():
            idx = [self.all.index(m) for m in ms]
            if idx != list(range(idx[0], idx[0] + len(idx))):
                raise ValueError("KV twin groups: a class is not a contiguous run of groups; refusing")
        self.classes.sort(key=lambda c: self.all.index(c.rep))
        self.reps = [c.rep for c in self.classes]
        self.orig = {}
        for name in ("remove_skipped_blocks", "get_num_blocks_to_allocate", "allocate_new_blocks", "cache_blocks"):
            self.orig[name] = getattr(coord, name)
            setattr(coord, name, getattr(self, name))
        self.skipped = 0
        self.ran = 0

    def _count(self, fn):
        def w(*a, **k):
            self.mutations += 1
            return fn(*a, **k)

        w.__wrapped__ = fn
        return w

    def _count_hash_map(self):
        hmap = getattr(self.pool, "cached_block_hash_to_block", None)
        if hmap is None or getattr(hmap, "_q38_counted", False):
            return
        for name in ("insert", "pop"):
            fn = getattr(hmap, name, None)
            if fn is not None:
                setattr(hmap, name, self._count(fn))
        hmap._q38_counted = True

    def _sig(self, c: _Class, rid: str):
        m = c.rep
        blocks = m.req_to_blocks.get(rid)
        if blocks:
            n = len(blocks)
            head = (n, id(blocks[-1]), id(blocks[-2]) if n > 1 else 0)
        else:
            head = -1 if blocks is None else 0
        return (head, c.version, self.mutations, [len(getattr(m, k)) for k in c.lists])

    def _call(self, name: str, managers: list, args: tuple, kwargs: dict):
        coord = self.coord
        coord.single_type_managers = managers
        try:
            return self.orig[name](*args, **kwargs)
        finally:
            coord.single_type_managers = self.all

    def _run(self, name: str, rid: str, args: tuple, kwargs: dict):
        """Class by class in group order (the classes are contiguous runs of groups, checked at install, so
        the block pool sees the same call order as the original loop): the representative alone, then, only
        if it changed something for this request, its twins. Returns per-manager results (group order) for
        the managers that ran, and None for skipped twins."""
        out = []
        for c in self.classes:
            if not c.twins:  # a class of one: nothing to skip, no signature needed
                r = self._call(name, [c.rep], args, kwargs)
                out.append((c, r[0] if name == "allocate_new_blocks" else r, False, None))
                continue
            before = self._sig(c, rid)
            r = self._call(name, [c.rep], args, kwargs)
            rep_res = r[0] if name == "allocate_new_blocks" else r
            changed = self._sig(c, rid) != before or (name == "allocate_new_blocks" and len(rep_res) > 0)
            if c.twins and changed:
                tr = self._call(name, c.twins, args, kwargs)
                self.ran += len(c.twins)
                out.append((c, rep_res, True, tr))
            else:
                self.skipped += len(c.twins)
                out.append((c, rep_res, False, None))
        return out

    # --- the four loops -------------------------------------------------------------------------
    def remove_skipped_blocks(self, request_id, *args, **kwargs):
        self._run("remove_skipped_blocks", request_id, (request_id,) + args, kwargs)

    def cache_blocks(self, request, num_computed_tokens, *args, **kwargs):
        self._run("cache_blocks", request.request_id, (request, num_computed_tokens) + args, kwargs)

    def get_num_blocks_to_allocate(self, request_id, num_tokens, new_computed_blocks, *args, **kwargs):
        if any(len(g) for g in new_computed_blocks):  # a prefix hit: the original loop, unchanged
            return self.orig["get_num_blocks_to_allocate"](request_id, num_tokens, new_computed_blocks, *args, **kwargs)
        total = 0
        for c, rep_n, ran, twin_n in self._run(
            "get_num_blocks_to_allocate", request_id, (request_id, num_tokens, new_computed_blocks) + args, kwargs
        ):
            # A representative that changed nothing answers for every member of its class.
            total += rep_n + (twin_n if ran else rep_n * len(c.twins))
        return total

    def allocate_new_blocks(self, request_id, *args, **kwargs):
        out = [[] for _ in self.all]
        for c, rep_res, ran, twin_res in self._run("allocate_new_blocks", request_id, (request_id,) + args, kwargs):
            out[self.index[id(c.rep)]] = rep_res
            if ran:
                for m, r in zip(c.twins, twin_res):
                    out[self.index[id(m)]] = r
        return tuple(out)


def install(coordinator: Any) -> TwinCoordinator:
    tc = TwinCoordinator(coordinator)
    logger.info(
        "KV twin groups: on (VLLM_TPU_KV_TWIN_GROUPS=1): %d groups in %d classes",
        len(tc.all), len(tc.classes),
    )
    coordinator._q38_twins = tc
    return tc


def patch_kv_cache_manager_for_twins() -> bool:
    """Wrap every KVCacheManager's coordinator after construction (engine-core
    entry point); idempotent; a no-op unless VLLM_TPU_KV_TWIN_GROUPS=1."""
    if not enabled():
        return False
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    if getattr(KVCacheManager, "_q38_twin_patch", False):
        return True
    orig_init = KVCacheManager.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            install(self.coordinator)
        except Exception:  # never stop the engine for an optimisation
            logger.exception("KV twin groups: install failed; continuing without it")

    KVCacheManager.__init__ = __init__
    KVCacheManager._q38_twin_patch = True
    return True
