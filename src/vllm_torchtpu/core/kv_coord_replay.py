# SPDX-License-Identifier: Apache-2.0
"""Cheaper hybrid KV cache coordination for many Mamba "align" groups.

Qwen3.8 has 38 KV cache groups: 37 Mamba "align" groups that share one spec,
plus the attention group. ``HybridKVCacheCoordinator`` calls every group's
manager for every request in every ``allocate_slots``: a running request
pays about 5.8 us per group per step in no-op Python, and an admission scans
each group's ``[null] * k + [state]`` hit list again. Both patches below give
exactly the allocations, block tables, free-block accounting and cache hits
of the unpatched coordinator.

``VLLM_TPU_KV_COORD_TAIL_SCAN=1`` (``_install_tail_scan``): a Mamba hit list
is nulls followed by at most one state block, so
``MambaManager._get_num_evictable_blocks`` counts from the tail and stops at
the first null, after a C-speed ``list.count`` check that everything before
that null is null too. Any other list falls back to the full sum.

``VLLM_TPU_KV_COORD_GROUP_REPLAY=1`` (``_install_group_replay``): for each set
of two or more Mamba "align" groups that ``HybridKVCacheCoordinator`` already
batches for lookup (same spec, manager class and block pool), the first group
is the leader. In ``remove_skipped_blocks``, ``get_num_blocks_to_allocate``,
``allocate_new_blocks`` and ``cache_blocks`` the coordinator still visits the
groups in their order, but after the leader's call it skips the followers
when all of these hold:

1. The leader's call changed none of the leader manager's state: every
   dict / set / list length and every scalar attribute is the same before
   and after; every per-request entry (``d.get(request_id)`` of each dict
   attribute, by identity, with container lengths; membership in each set
   attribute) is the same as recorded after the request's previous leader
   call; and the call did not touch the block pool
   (for the leader's call its ``block_pool`` is a stand-in that counts every
   access other than a scalar or the null block).
2. The request's per-request state is identical in every member (the
   "sync" check below). It is verified once the leader's calls for the
   request have been no-ops ``_VERIFY_AFTER`` times in a row since its last
   change, so a request that changes every step (prefill) never pays it.
3. ``allocate_new_blocks`` returned no blocks; ``cache_blocks`` has no
   pending prefill checkpoint for the request (that path reads the
   checkpoint block's hash, which other requests can change).
4. ``get_num_blocks_to_allocate`` is not given new prefix-hit blocks in any
   group (their eviction state is per group).

Exactness. A manager's no-op decisions for a request read only the request's
own state: the length and null pattern of its block table, its entries in
``num_cached_block``, ``last_state_block_idx``, ``_num_retired_blocks``,
``_checkpoints``, ``_partial_hit_reqs``, ``_allocated_block_reqs`` (and the
TPU patches' per-request dicts), the call's inputs, and per-manager constants
(block size, speculative blocks, checkpoint flags, caching flags, spec). Every
path that reads anything else (a block's hash or refcount, the pool's hash
map, free capacity) goes through the block pool, which condition 1 counts,
except the hash of a pending prefill checkpoint block, which condition 3
excludes. The sync check compares all of that request state, with blocks
reduced to their null flag, and the constants, between the leader and each
follower. So when the leader's call was a no-op, each follower's call would
take the same branches and also be a no-op returning the same value; skipping
it leaves every manager, the pool and the return value exactly as the
unpatched loop does. A request's table and entries change only through its
own manager calls, so the verified sync stays valid while the leader's
per-request state still matches the record it was verified against; it is
dropped when the followers run while the leader's state did not change.
Calls outside these four methods (admission, free, the TPU retention and
tail-fold hooks) loop over all groups, and each of them changes the leader's
per-request state, so the record no longer matches. A request that holds no
blocks yet (being admitted) runs the upstream loop.

The four coordinator loops are re-implemented here; the patch checks the
upstream source hash of each and stays off (with a warning) on any other
vLLM.
"""

import hashlib
import inspect
import textwrap
from collections import defaultdict
from operator import attrgetter, is_, itemgetter

from vllm.logger import init_logger

from vllm_torchtpu import envs

logger = init_logger(__name__)

_PATCH_FLAG = "_tpu_kv_coord_patch"
_PLAN_ATTR = "_tpu_kv_coord_plan"

# sha256[:16] of the upstream sources this module re-implements (vLLM in the
# Qwen3.8 images, /root/cloud-devkit/vllm).
_UPSTREAM_SOURCES = {
    "remove_skipped_blocks": "5282503854a6acc7",
    "get_num_blocks_to_allocate": "8dc113565bd525c3",
    "allocate_new_blocks": "89398e35cfeac2a9",
    "cache_blocks": "f57aea84f04db711",
}
_UPSTREAM_EVICTABLE = "1bbd29b19321d3a4"

_MISSING = object()
_SCALARS = (int, bool, float, str, type(None))
_CONTAINERS = (list, dict, set, defaultdict)
_IS_NULL = attrgetter("is_null")
_FILL_NONE = "none"
_FILL_SAME = "same"
_FILL_EMPTY = "empty"
# Consecutive leader no-ops for a request before its cross-group sync is
# verified: a prefilling request changes state every step and never pays it.
_VERIFY_AFTER = 4
# Manager attributes left out of the cross-group sync comparison: the group
# id differs by construction; the rest are not per-request state.
_SYNC_SKIP = frozenset(
    {"kv_cache_group_id", "req_to_blocks", "block_pool", "_null_block"}
)
_SYNC_CAP = 1 << 16


def _source_hash(fn) -> str:
    src = textwrap.dedent(inspect.getsource(fn))
    return hashlib.sha256(src.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# (a) tail scan


def _install_tail_scan() -> None:
    from vllm.v1.core.single_type_kv_cache_manager import (
        MambaManager,
        SingleTypeKVCacheManager,
    )

    current = MambaManager._get_num_evictable_blocks
    if getattr(current, _PATCH_FLAG, False):
        return
    base = SingleTypeKVCacheManager._get_num_evictable_blocks.__func__
    if current.__func__ is not base or _source_hash(base) != _UPSTREAM_EVICTABLE:
        logger.warning(
            "VLLM_TPU_KV_COORD_TAIL_SCAN: not applied, "
            "_get_num_evictable_blocks is not the known upstream one."
        )
        return

    def _get_num_evictable_blocks(cls, blocks):
        if type(blocks) is not list and type(blocks) is not tuple:
            return base(cls, blocks)
        count = 0
        j = len(blocks) - 1
        while j >= 0:
            blk = blocks[j]
            if blk.is_null:
                # Exact only if everything before is null too. ``count`` runs
                # at C speed (identity first, then ``==``, which for a slots
                # dataclass needs every field, ``is_null`` included, equal);
                # nothing after ``j`` is null, so ``j + 1`` matches mean the
                # whole prefix is null and adds 0 to the upstream sum.
                if j == 0 or blocks.count(blk) == j + 1:
                    return count
                return base(cls, blocks)
            if blk.ref_cnt == 0:
                count += 1
            j -= 1
        return count

    setattr(_get_num_evictable_blocks, _PATCH_FLAG, True)
    MambaManager._get_num_evictable_blocks = classmethod(_get_num_evictable_blocks)
    logger.info(
        "Applied TPU patch: Mamba prefix-hit evictable blocks counted from the "
        "tail (VLLM_TPU_KV_COORD_TAIL_SCAN)."
    )


# --------------------------------------------------------------------------
# (b) leader-group replay


class _Shape:
    """A manager's attributes by kind, for fast snapshots."""

    __slots__ = ("n", "get", "nd", "ns", "nl")

    def __init__(self, manager) -> None:
        d = manager.__dict__
        self.n = len(d)
        kinds = {"d": [], "s": [], "l": [], "c": []}
        for name, value in d.items():
            if isinstance(value, dict):
                kinds["d"].append(name)
            elif isinstance(value, set):
                kinds["s"].append(name)
            elif isinstance(value, list):
                kinds["l"].append(name)
            elif isinstance(value, _SCALARS):
                kinds["c"].append(name)
        names = kinds["d"] + kinds["s"] + kinds["l"] + kinds["c"]
        self.nd = len(kinds["d"])
        self.ns = self.nd + len(kinds["s"])
        self.nl = self.ns + len(kinds["l"])
        if len(names) >= 2:
            self.get = itemgetter(*names)
        else:
            self.get = lambda d: tuple(d[name] for name in names)


def _shape(shapes, manager) -> _Shape:
    shape = shapes.get(id(manager))
    if shape is None or shape.n != len(manager.__dict__):
        shape = shapes[id(manager)] = _Shape(manager)
    return shape


def _snapshot(shapes, manager, request_id):
    """(per-request entries, per-request rest, global rest) of a manager.

    Entries hold the objects themselves and are compared by identity, so an
    entry replaced by an equal new object counts as a change (that only
    costs a skip). Container entries also record their length.
    """
    shape = _shape(shapes, manager)
    try:
        values = shape.get(manager.__dict__)
        nd, ns, nl = shape.nd, shape.ns, shape.nl
        dicts = values[:nd]
        sets = values[nd:ns]
        entries = tuple([v.get(request_id, _MISSING) for v in dicts])
        rest = (
            tuple([request_id in v for v in sets]),
            tuple([len(e) if type(e) in _CONTAINERS else -1 for e in entries]),
        )
        glob = (tuple(map(len, values[:nl])), values[nl:])
    except (AttributeError, KeyError, TypeError):
        # An attribute changed kind: re-classify and take it again.
        shapes[id(manager)] = _Shape(manager)
        return _snapshot(shapes, manager, request_id)
    return entries, rest, glob


def _global(shapes, manager):
    """The global part of ``_snapshot``."""
    shape = _shape(shapes, manager)
    values = shape.get(manager.__dict__)
    return (tuple(map(len, values[: shape.nl])), values[shape.nl :])


def _same_entries(a, b) -> bool:
    return len(a) == len(b) and all(map(is_, a, b))


def _canon_value(value):
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    if value is _MISSING:
        return "<missing>"
    if isinstance(value, _SCALARS):
        return value
    if isinstance(value, KVCacheBlock):
        return ("block", value.is_null)
    if isinstance(value, tuple):
        return tuple(_canon_value(v) for v in value)
    if isinstance(value, _CONTAINERS):
        return ("container", len(value))
    return ("object", type(value).__name__)


def _canon(manager, request_id, names):
    """The request's state in one manager, block ids left out."""
    blocks = manager.req_to_blocks.get(request_id)
    out = [
        type(manager),
        None if blocks is None else list(map(_IS_NULL, blocks)),
    ]
    d = manager.__dict__
    for name in names:
        value = d.get(name, _MISSING)
        if value is _MISSING:
            out.append("<absent>")
        elif isinstance(value, dict):
            e = value.get(request_id, _MISSING)
            out.append(e if type(e) is int else _canon_value(e))
        elif isinstance(value, set):
            out.append(request_id in value)
        elif isinstance(value, _SCALARS) or name == "kv_cache_spec":
            out.append(value)
        else:
            out.append(None)
    return out


class _Plan:
    """Per-coordinator replay state."""

    def __init__(self, coordinator, groups) -> None:
        managers = coordinator.single_type_managers
        self.n = len(managers)
        self.groups = groups  # [(leader_idx, [follower_idx, ...])]
        role = [None] * self.n
        for g, (leader, followers) in enumerate(groups):
            role[leader] = (g, True)
            for f in followers:
                role[f] = (g, False)
        # Units in visiting order: (kind, g, idx, manager, followers)
        # kind 0 = plain manager, 1 = leader, 2 = lone follower. A leader
        # whose followers come right after it carries them as a block
        # (a tuple of (idx, manager) plus their index range).
        units = []
        i = 0
        while i < self.n:
            r = role[i]
            if r is None:
                units.append((0, -1, i, managers[i], None))
                i += 1
            elif r[1]:
                g = r[0]
                followers = groups[g][1]
                if followers == list(range(i + 1, i + 1 + len(followers))):
                    block = tuple((f, managers[f]) for f in followers)
                    units.append((1, g, i, managers[i], block))
                    i += 1 + len(followers)
                else:
                    units.append((1, g, i, managers[i], None))
                    i += 1
            else:
                units.append((2, r[0], i, managers[i], None))
                i += 1
        self.units = tuple(units)
        self.member_ids = tuple(idx for leader, fs in groups for idx in (leader, *fs))
        self.managers = managers
        self.first_leader = managers[groups[0][0]]
        # Per group: request id -> [entries, rest, status]; status is the
        # count of consecutive leader no-ops (int) until verified, then the
        # verification result (bool).
        self.sync = [dict() for _ in groups]
        self.shapes: dict[int, _Shape] = {}
        # Per group: the leader's real pool, a counter of the leader call's
        # pool accesses, and the counting stand-in.
        self.pools = [managers[leader].block_pool for leader, _ in groups]
        self.cells = [[0] for _ in groups]
        self.proxies = [
            _CountingPool(pool, cell) for pool, cell in zip(self.pools, self.cells)
        ]
        self.skips = 0
        self.runs = 0
        self.verifies = 0

    def _verify(self, g, request_id) -> bool:
        self.verifies += 1
        leader, followers = self.groups[g]
        lm = self.managers[leader]
        names = set(lm.__dict__)
        for f in followers:
            names |= set(self.managers[f].__dict__)
        names = sorted(names - _SYNC_SKIP)
        ref = _canon(lm, request_id, names)
        for f in followers:
            fm = self.managers[f]
            if fm.block_pool is not lm.block_pool:
                return False
            if _canon(fm, request_id, names) != ref:
                return False
        return True

    def decide(self, g, manager, request_id, glob_before, calls0, result_ok):
        """After the leader's call: may the followers be skipped?

        The call was a no-op if it did not touch the pool, left the global
        part (container lengths, scalars) as it was, and left the request's
        part equal to the record taken after the request's last leader call.
        A record that does not match (the state changed in this call or since
        the last one) restarts the count; a sync verified for it holds while
        it keeps matching.
        """
        sync = self.sync[g]
        if self.cells[g][0] != calls0:
            sync.pop(request_id, None)
            return False
        entries, rest, glob = _snapshot(self.shapes, manager, request_id)
        if glob != glob_before:
            sync.pop(request_id, None)
            return False
        record = sync.get(request_id)
        if record is None or record[1] != rest or not _same_entries(record[0], entries):
            if len(sync) >= _SYNC_CAP:
                sync.clear()
            sync[request_id] = [entries, rest, 1]
            return False
        if not result_ok:
            self.followers_run(g, request_id)
            return False
        status = record[2]
        if status is True or status is False:
            return status
        if status + 1 < _VERIFY_AFTER:
            record[2] = status + 1
            return False
        record[2] = ok = self._verify(g, request_id)
        return ok

    def followers_run(self, g, request_id) -> None:
        """The followers ran while the leader's state did not change: a
        verified sync no longer holds."""
        record = self.sync[g].get(request_id)
        if record is not None and record[2] is True:
            del self.sync[g][request_id]

    def drive(self, request_id, call, fill, check_result, extra_ok=None):
        """Visit the managers in order; returns the per-group results.

        ``fill``: what a skipped follower returns: _FILL_NONE (None, the
        initial value), _FILL_SAME (the leader's result) or _FILL_EMPTY (a
        new empty list).
        """
        res = [None] * self.n
        skip = None
        for kind, g, i, manager, block in self.units:
            if kind == 0:
                res[i] = call(i, manager)
            elif kind == 1:
                before = _global(self.shapes, manager)
                calls0 = self.cells[g][0]
                pool = manager.block_pool
                if pool is not self.pools[g]:
                    raise RuntimeError("Mamba leader block pool changed")
                manager.block_pool = self.proxies[g]
                try:
                    r = res[i] = call(i, manager)
                finally:
                    manager.block_pool = pool
                ok = (check_result is None or check_result(r)) and (
                    extra_ok is None or extra_ok(manager)
                )
                skipped = self.decide(g, manager, request_id, before, calls0, ok)
                if skipped:
                    self.skips += 1
                else:
                    self.runs += 1
                if block is not None:
                    if skipped:
                        if fill is _FILL_SAME:
                            res[i + 1 : i + 1 + len(block)] = [r] * len(block)
                        elif fill is _FILL_EMPTY:
                            res[i + 1 : i + 1 + len(block)] = [[] for _ in block]
                    else:
                        for j, fm in block:
                            res[j] = call(j, fm)
                        if ok:
                            self.followers_run(g, request_id)
                else:
                    if skip is None:
                        skip = {}
                    skip[g] = (skipped, r)
                    if not skipped and ok:
                        self.followers_run(g, request_id)
            else:
                state = skip.get(g) if skip is not None else None
                if state is not None and state[0]:
                    if fill is _FILL_SAME:
                        res[i] = state[1]
                    elif fill is _FILL_EMPTY:
                        res[i] = []
                else:
                    res[i] = call(i, manager)
        return res

    def tracks(self, request_id) -> bool:
        """Whether the request holds blocks (is running): a request being
        admitted runs the upstream loop, which skips nothing."""
        return request_id in self.first_leader.num_cached_block

    def forget(self, request_id) -> None:
        for sync in self.sync:
            sync.pop(request_id, None)


class _CountingPool:
    """Stands in for the leader manager's ``block_pool`` during its call and
    counts every access to it that is not a plain scalar or the null block
    (every method call, every mutable member)."""

    __slots__ = ("_tpu_pool", "_tpu_cell")

    def __init__(self, pool, cell) -> None:
        object.__setattr__(self, "_tpu_pool", pool)
        object.__setattr__(self, "_tpu_cell", cell)

    def __getattr__(self, name):
        pool = self._tpu_pool
        value = getattr(pool, name)
        if type(value) not in _SCALARS and value is not pool.null_block:
            self._tpu_cell[0] += 1
        return value

    def __setattr__(self, name, value) -> None:
        self._tpu_cell[0] += 1
        setattr(self._tpu_pool, name, value)


def _build_plan(coordinator):
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    groups = []
    managers = coordinator.single_type_managers
    for spec_group in getattr(coordinator, "attention_groups", ()):
        ids = list(spec_group.group_ids)
        if len(ids) < 2:
            continue
        members = [managers[i] for i in ids]
        if not all(
            isinstance(m, MambaManager)
            and type(m) is type(members[0])
            and getattr(m, "mamba_cache_mode", None) == "align"
            and m.block_pool is members[0].block_pool
            for m in members
        ):
            continue
        ids.sort()
        groups.append((ids[0], ids[1:]))
    if not groups:
        return None
    plan = _Plan(coordinator, groups)
    logger.info(
        "VLLM_TPU_KV_COORD_GROUP_REPLAY: %d Mamba align group set(s) of "
        "sizes %s replay their leader.",
        len(groups),
        [1 + len(f) for _, f in groups],
    )
    return plan


def _plan(coordinator):
    plan = coordinator.__dict__.get(_PLAN_ATTR, _MISSING)
    if plan is _MISSING:
        plan = _build_plan(coordinator)
        coordinator.__dict__[_PLAN_ATTR] = plan
    return plan


def _return_empty(result) -> bool:
    return type(result) is list and not result


def _install_group_replay() -> None:
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
    from vllm.v1.core.single_type_kv_cache_manager import CrossAttentionManager

    cls = HybridKVCacheCoordinator
    if getattr(cls.remove_skipped_blocks, _PATCH_FLAG, False):
        return
    for name, digest in _UPSTREAM_SOURCES.items():
        fn = getattr(cls, name)
        if _source_hash(fn) != digest:
            logger.warning(
                "VLLM_TPU_KV_COORD_GROUP_REPLAY: not applied, "
                "%s.%s is not the known upstream one.",
                cls.__name__,
                name,
            )
            return

    orig_remove = cls.remove_skipped_blocks
    orig_count = cls.get_num_blocks_to_allocate
    orig_allocate = cls.allocate_new_blocks
    orig_cache = cls.cache_blocks
    orig_free = cls.free
    orig_pop = cls.pop_blocks_for_free

    def remove_skipped_blocks(
        self, request_id, processed_computed_tokens, num_prompt_tokens=None
    ):
        plan = _plan(self)
        if plan is None or not plan.tracks(request_id):
            return orig_remove(
                self, request_id, processed_computed_tokens, num_prompt_tokens
            )

        def call(i, manager):
            manager.remove_skipped_blocks(
                request_id, processed_computed_tokens, num_prompt_tokens
            )

        plan.drive(request_id, call, _FILL_NONE, None)

    def get_num_blocks_to_allocate(
        self,
        request_id,
        num_tokens,
        new_computed_blocks,
        num_encoder_tokens,
        total_computed_tokens,
        num_local_computed_tokens,
        num_tokens_main_model,
        apply_admission_cap=False,
        prefill_end=0,
    ):
        plan = _plan(self)
        # A prefix hit in any group: upstream loop (hit eviction state is
        # per group).
        if plan is None or not plan.tracks(request_id) or any(new_computed_blocks):
            return orig_count(
                self,
                request_id,
                num_tokens,
                new_computed_blocks,
                num_encoder_tokens,
                total_computed_tokens,
                num_local_computed_tokens,
                num_tokens_main_model,
                apply_admission_cap,
                prefill_end,
            )
        if self.retention_interval != 0:
            prefill_end = 0

        def call(i, manager):
            if isinstance(manager, CrossAttentionManager):
                return manager.get_num_blocks_to_allocate(
                    request_id,
                    num_encoder_tokens,
                    [],
                    0,
                    0,
                    num_encoder_tokens,
                    apply_admission_cap=apply_admission_cap,
                )
            return manager.get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                new_computed_blocks[i],
                total_computed_tokens,
                num_local_computed_tokens,
                num_tokens_main_model,
                apply_admission_cap=apply_admission_cap,
                prefill_end=prefill_end,
            )

        return sum(plan.drive(request_id, call, _FILL_SAME, None))

    def allocate_new_blocks(
        self, request_id, num_tokens, num_tokens_main_model, num_encoder_tokens=0
    ):
        plan = _plan(self)
        if plan is None or not plan.tracks(request_id):
            return orig_allocate(
                self, request_id, num_tokens, num_tokens_main_model, num_encoder_tokens
            )

        def call(i, manager):
            return manager.allocate_new_blocks(
                request_id,
                num_encoder_tokens
                if isinstance(manager, CrossAttentionManager)
                else num_tokens,
                num_tokens_main_model,
            )

        return tuple(plan.drive(request_id, call, _FILL_EMPTY, _return_empty))

    def cache_blocks(self, request, num_computed_tokens):
        plan = _plan(self)
        request_id = request.request_id
        if plan is None or not plan.tracks(request_id):
            return orig_cache(self, request, num_computed_tokens)
        cached_num_computed_tokens = self._align_cacheable(num_computed_tokens)
        boundaries = self.get_replay_boundaries(request)

        def call(i, manager):
            if not manager.enable_caching:
                return None
            num_tokens_to_cache = cached_num_computed_tokens
            if manager.use_eagle and cached_num_computed_tokens > 0:
                num_finalized_computed_tokens = max(
                    0, num_computed_tokens - self.num_reprefillable_tokens
                )
                cached_num_finalized_computed_tokens = self._align_cacheable(
                    num_finalized_computed_tokens
                )
                num_tokens_to_cache = min(
                    num_finalized_computed_tokens,
                    cached_num_finalized_computed_tokens + manager.block_size,
                )
            manager.cache_blocks(
                request,
                num_tokens_to_cache,
                retention_interval=self.retention_interval,
                replay_boundaries=boundaries,
            )
            return None

        def no_checkpoint(manager):
            checkpoints = getattr(manager, "_checkpoints", None)
            return not checkpoints or request_id not in checkpoints

        plan.drive(request_id, call, _FILL_NONE, None, no_checkpoint)

    def free(self, request_id):
        plan = self.__dict__.get(_PLAN_ATTR)
        if plan:
            plan.forget(request_id)
        return orig_free(self, request_id)

    def pop_blocks_for_free(self, request_id):
        plan = self.__dict__.get(_PLAN_ATTR)
        if plan:
            plan.forget(request_id)
        return orig_pop(self, request_id)

    for fn in (
        remove_skipped_blocks,
        get_num_blocks_to_allocate,
        allocate_new_blocks,
        cache_blocks,
        free,
        pop_blocks_for_free,
    ):
        setattr(fn, _PATCH_FLAG, True)
    cls.remove_skipped_blocks = remove_skipped_blocks
    cls.get_num_blocks_to_allocate = get_num_blocks_to_allocate
    cls.allocate_new_blocks = allocate_new_blocks
    cls.cache_blocks = cache_blocks
    cls.free = free
    cls.pop_blocks_for_free = pop_blocks_for_free
    logger.info(
        "Applied TPU patch: hybrid KV coordinator replays one Mamba align "
        "group per same-spec set (VLLM_TPU_KV_COORD_GROUP_REPLAY)."
    )


def patch_kv_coord() -> None:
    """Engine core: install the flag-gated coordinator speedups."""
    if envs.VLLM_TPU_KV_COORD_TAIL_SCAN:
        _install_tail_scan()
    if envs.VLLM_TPU_KV_COORD_GROUP_REPLAY:
        _install_group_replay()
