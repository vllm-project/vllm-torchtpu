# SPDX-License-Identifier: Apache-2.0
"""Victim host-tier policy: store a cached block to the host tier when HBM evicts it for reuse.

``QWEN4_EXP_HOST_TIER_POLICY`` selects the policy of the store-backed offload connector:

* ``write_through`` (default): every computed block is stored as it is computed (today's behaviour). The host
  tier then holds a copy of the most recent blocks, mostly the same set HBM holds (inclusive), so its capacity
  does not add to HBM's (CP3, ``reports/cache_ht_cp3_readout.md``).
* ``victim``: nothing is stored on compute. A cached block is stored when ``BlockPool.get_new_blocks`` evicts it
  to hand its ID to a new owner, so the host tier holds what HBM no longer holds (exclusive on the store side;
  a host hit keeps its host copy, "keep-on-load").

Mechanism (engine-core process only; the scheduler and the block pool live there):

* ``patch_block_pool_victims`` wraps two ``BlockPool`` methods and three ``FreeKVCacheBlockQueue`` methods at
  class level. They are inert unless a ``VictimTap`` is active.
  - ``get_new_blocks`` marks the allocation window, and ``_maybe_evict_cached_block`` records a hashed block
    evicted *inside* that window. Every other eviction is excluded by construction: the retention patch's
    superseded checkpoints (``_drop_superseded``), failed-load invalidation (``evict_blocks``) and the
    transient Mamba checkpoint all call ``_maybe_evict_cached_block`` outside an allocation, and a block they
    clear reaches the free-queue head hashless.
  - ``FreeKVCacheBlockQueue.append`` / ``append_n`` / ``prepend_n`` stamp the scheduler step at which a block
    entered the free queue, i.e. its reference count reached 0, whichever free path put it there
    (``_patch_free_queue_stamps``).
* The scheduler connector (``TPURaidenOffloadingScheduler``) takes the recorded victims in its
  ``build_connector_meta``, keeps those whose stamp is at least ``min_age`` steps old (young-block guard: a
  block freed recently may still be written by a batch in flight), stores them, and drains the store
  synchronously before the step's metadata ships, i.e. before the new owner's forward pass can write the
  block (``select_victims``).
"""

from __future__ import annotations

import os
from collections import Counter, OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

POLICY_ENV = "QWEN4_EXP_HOST_TIER_POLICY"
POLICIES = ("write_through", "victim")


def host_tier_policy() -> str:
    """The configured host-tier policy; refuses unknown values."""
    value = os.environ.get(POLICY_ENV, "").strip() or "write_through"
    if value not in POLICIES:
        raise ValueError(f"{POLICY_ENV}={value!r}: expected one of {POLICIES}")
    return value


@dataclass(frozen=True)
class Victim:
    """A hashed block evicted for reuse: its ID (the store's source), its primary hash, and when it was freed."""

    block_id: int
    block_hash: Any  # vllm BlockHashWithGroupId
    freed_step: int | None
    extra_hashes: int = 0


@dataclass
class VictimTap:
    """Collects victims and free stamps between two ``build_connector_meta`` calls."""

    step: int = 0
    freed_at: dict[int, int] = field(default_factory=dict)
    pending: list[Victim] = field(default_factory=list)

    def on_freed(self, block_ids) -> None:
        for block_id in block_ids:
            self.freed_at[block_id] = self.step

    def on_evict(self, pool, block) -> None:
        extra = len(getattr(pool, "cached_block_hashes_by_block", {}).get(block.block_id, ()))
        self.pending.append(
            Victim(block.block_id, block.block_hash, self.freed_at.get(block.block_id), extra)
        )

    def take(self) -> list[Victim]:
        out, self.pending = self.pending, []
        return out

    def advance(self) -> None:
        self.step += 1


_TAP: VictimTap | None = None
_IN_ALLOC = 0


def activate(tap: VictimTap) -> None:
    global _TAP
    if _TAP is not None and _TAP is not tap:
        raise RuntimeError("a victim tap is already active in this process")
    _TAP = tap


def deactivate(tap: VictimTap | None = None) -> None:
    global _TAP
    if tap is None or _TAP is tap:
        _TAP = None


def active_tap() -> VictimTap | None:
    return _TAP


def patch_block_pool_victims() -> None:
    """Install the (inert unless a tap is active) BlockPool hooks. Idempotent. Free steps are stamped on the free
    queue (_patch_free_queue_stamps), the one place every free path ends, so a later replacement of free_blocks (the
    served engine's LIFO patch) cannot drop them."""
    from vllm.v1.core.block_pool import BlockPool

    if BlockPool.__dict__.get("_q38_victim_patch", False):
        return
    orig_get = BlockPool.get_new_blocks
    orig_evict = BlockPool._maybe_evict_cached_block

    def get_new_blocks(self, num_blocks):
        global _IN_ALLOC
        if _TAP is None:
            return orig_get(self, num_blocks)
        _IN_ALLOC += 1
        try:
            return orig_get(self, num_blocks)
        finally:
            _IN_ALLOC -= 1

    def _maybe_evict_cached_block(self, block):
        if _TAP is not None and _IN_ALLOC and block.block_hash is not None:
            _TAP.on_evict(self, block)
        return orig_evict(self, block)

    BlockPool.get_new_blocks = get_new_blocks
    BlockPool._maybe_evict_cached_block = _maybe_evict_cached_block
    BlockPool._q38_victim_patch = True
    _patch_free_queue_stamps()


def _patch_free_queue_stamps() -> None:
    """Stamp a block's free step where every free path ends: entering the free queue.

    Stamping in BlockPool.free_blocks / unpin_blocks (the earlier design) fails on the served engine: there the
    victim hook is installed at the engine_core stage, and the LIFO free patch
    (vllm_torchtpu._patch_vllm_block_pool_lifo_free, platform_activation) replaces BlockPool.free_blocks seconds
    later in the same EngineCore process, so the wrapper is dropped and every victim arrives unstamped (j-47fee8a1:
    220,445 of 220,445). Any free path, the LIFO one included, puts the block on FreeKVCacheBlockQueue, so stamping
    here survives later replacements. Idempotent; inert unless a tap is active."""
    from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue

    if FreeKVCacheBlockQueue.__dict__.get("_q38_victim_stamp", False):
        return
    for name in ("append", "append_n", "prepend_n"):
        orig = FreeKVCacheBlockQueue.__dict__.get(name)
        if orig is None:
            continue

        def wrapped(self, arg, _orig=orig, _many=name != "append"):
            out = _orig(self, arg)
            if _TAP is not None:
                _TAP.on_freed(b.block_id for b in (arg if _many else (arg,)) if not b.is_null)
            return out

        setattr(FreeKVCacheBlockQueue, name, wrapped)
    FreeKVCacheBlockQueue._q38_victim_stamp = True


def select_victims(
    victims: list[Victim], step: int, min_age: int, offload_groups: set[int]
) -> tuple[list[tuple[Any, int]], Counter]:
    """(offload key, device block id) for each victim worth storing, and counts of the rest.

    Kept: a primary hash in an offloaded KV cache group, freed at least ``min_age`` steps before ``step``. A
    block with no free stamp is treated as young (its free path bypassed the hook; never store blind). Extra
    hashes of a multi-hash block are counted, not stored (one source block per key).
    """
    from vllm.v1.core.kv_cache_utils import get_block_hash, get_group_id
    from vllm.v1.kv_offload.base import make_offload_key

    keep: list[tuple[Any, int]] = []
    seen: set[Any] = set()
    counts: Counter = Counter()
    for v in victims:
        counts["victim_seen"] += 1
        counts["victim_extra_hashes"] += v.extra_hashes
        group = get_group_id(v.block_hash)
        if group not in offload_groups:
            counts["victim_not_offloaded"] += 1
            continue
        if v.freed_step is None:
            counts["victim_unstamped"] += 1
            continue
        if step - v.freed_step < min_age:
            counts["victim_young"] += 1
            continue
        key = make_offload_key(get_block_hash(v.block_hash), group)
        if key in seen:
            counts["victim_duplicate"] += 1
            continue
        seen.add(key)
        keep.append((key, v.block_id))
    return keep, counts


def _trim_ordered(mapping: OrderedDict, limit: int) -> None:
    while len(mapping) > limit:
        mapping.popitem(last=False)


class SupersededCheckpointTracker:
    """Tracks single-line turn-``t`` recurrent-state checkpoints superseded by turn ``t+1``.

    When a request resumes from checkpoint boundary ``resumed_tokens`` and
    completes prefill with a strictly deeper prompt checkpoint boundary
    ``new_tokens > resumed_tokens``, the previous turn's state-group checkpoint
    keys are no longer needed for that linear continuation. In victim mode,
    skipping those keys when HBM evicts them keeps host DRAM checkpoint capacity
    free for active leaves without dropping any shared branch checkpoints:

    * Only state-group checkpoints (caller passes ``MambaSpec`` keys) are tracked;
      attention KV blocks are never skipped.
    * Any checkpoint resumed by ``>= 2`` distinct request IDs (a branch/fork
      junction) is promoted to ``_shared`` and removed from ``_superseded``, so
      it is **never** skipped.
    * Aborted/incomplete requests (``completed_prompt=False``) or turns ending
      within the same checkpoint block (``new_boundary_tokens <= resumed_tokens``)
      do **not** supersede their resumed checkpoint.
    * All internal structures are strictly bounded to ``max_entries`` for O(1)
      memory over long serving runs.
    """

    def __init__(self, max_entries: int = 65_536) -> None:
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}")
        self.max_entries = int(max_entries)
        # req_id -> (resumed_tokens, tuple of state OffloadKeys at that boundary)
        self._pending: OrderedDict[str, tuple[int, tuple[Any, ...]]] = OrderedDict()
        # key -> set-size approximation (1 until a 2nd distinct req marks it shared)
        self._first_owner: OrderedDict[Any, str] = OrderedDict()
        # Shared branch/fork junction checkpoints (never superseded).
        self._shared: OrderedDict[Any, None] = OrderedDict()
        # Keys ready to skip when HBM evicts them as victims.
        self._superseded: OrderedDict[Any, None] = OrderedDict()
        self.marked = 0
        self.skipped = 0
        self.shared_protected = 0

    def record_resume(
        self,
        req_id: str,
        resumed_tokens: int,
        state_keys: Iterable[Any],
    ) -> None:
        """Record the checkpoint keys that ``req_id`` resumed from on first lookup."""
        keys = tuple(dict.fromkeys(state_keys))
        prev = self._pending.get(req_id)
        if prev is not None:
            if prev[0] == resumed_tokens and prev[1] == keys:
                return
            # Divergent-hit fallback can re-lookup the same request at a smaller
            # convergent boundary: undo tentative sole-owner entries for keys the
            # request no longer resumes from.
            for old_k in prev[1]:
                if old_k not in keys and self._first_owner.get(old_k) == req_id:
                    self._first_owner.pop(old_k, None)
            del self._pending[req_id]
        if resumed_tokens <= 0 or not keys:
            return
        for k in keys:
            if k in self._shared:
                self._shared.move_to_end(k)
                continue
            owner = self._first_owner.get(k)
            if owner is None:
                self._first_owner[k] = req_id
                _trim_ordered(self._first_owner, self.max_entries)
            elif owner != req_id:
                # Distinct second request resumed from this checkpoint: fork junction!
                self._first_owner.pop(k, None)
                self._shared[k] = None
                _trim_ordered(self._shared, self.max_entries)
                if k in self._superseded:
                    del self._superseded[k]
                    self.shared_protected += 1
        self._pending[req_id] = (resumed_tokens, keys)
        _trim_ordered(self._pending, self.max_entries)

    def on_request_finished(
        self,
        req_id: str,
        new_boundary_tokens: int,
        *,
        completed_prompt: bool,
    ) -> list[Any]:
        """Mark single-owner resumed checkpoint keys superseded if ``req_id`` completed
        prefill past its resumed checkpoint boundary. Returns newly superseded keys."""
        entry = self._pending.pop(req_id, None)
        if entry is None:
            return []
        resumed_tokens, keys = entry
        if not completed_prompt or new_boundary_tokens <= resumed_tokens:
            return []
        newly: list[Any] = []
        for k in keys:
            if k in self._shared:
                continue
            if k not in self._superseded:
                self._superseded[k] = None
                self.marked += 1
                newly.append(k)
            else:
                self._superseded.move_to_end(k)
        _trim_ordered(self._superseded, self.max_entries)
        return newly

    def is_superseded(self, key: Any) -> bool:
        return key in self._superseded

    def filter_victims(
        self, keep: list[tuple[Any, int]]
    ) -> tuple[list[tuple[Any, int]], int]:
        """Drop superseded checkpoint victims before host-tier admission."""
        if not self._superseded or not keep:
            return keep, 0
        out: list[tuple[Any, int]] = []
        skipped = 0
        for key, block_id in keep:
            if key in self._superseded:
                del self._superseded[key]
                self._first_owner.pop(key, None)
                skipped += 1
            else:
                out.append((key, block_id))
        self.skipped += skipped
        return out, skipped
