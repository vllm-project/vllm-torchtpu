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

* ``patch_block_pool_victims`` wraps four ``BlockPool`` methods at class level. They are inert unless a
  ``VictimTap`` is active.
  - ``get_new_blocks`` marks the allocation window, and ``_maybe_evict_cached_block`` records a hashed block
    evicted *inside* that window. Every other eviction is excluded by construction: the retention patch's
    superseded checkpoints (``_drop_superseded``), failed-load invalidation (``evict_blocks``) and the
    transient Mamba checkpoint all call ``_maybe_evict_cached_block`` outside an allocation, and a block they
    clear reaches the free-queue head hashless.
  - ``free_blocks`` and ``unpin_blocks`` stamp the scheduler step at which a block's reference count reached 0.
* The scheduler connector (``TPURaidenOffloadingScheduler``) takes the recorded victims in its
  ``build_connector_meta``, keeps those whose stamp is at least ``min_age`` steps old (young-block guard: a
  block freed recently may still be written by a batch in flight), stores them, and drains the store
  synchronously before the step's metadata ships, i.e. before the new owner's forward pass can write the
  block (``select_victims``).
"""

from __future__ import annotations

import os
from collections import Counter
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
    """Install the (inert unless a tap is active) BlockPool hooks. Idempotent."""
    from vllm.v1.core.block_pool import BlockPool

    if BlockPool.__dict__.get("_q38_victim_patch", False):
        return
    orig_get = BlockPool.get_new_blocks
    orig_evict = BlockPool._maybe_evict_cached_block
    orig_free = BlockPool.free_blocks
    orig_unpin = BlockPool.__dict__.get("unpin_blocks")

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

    def free_blocks(self, ordered_blocks):
        if _TAP is None:
            return orig_free(self, ordered_blocks)
        blocks = list(ordered_blocks)
        out = orig_free(self, blocks)
        _TAP.on_freed(b.block_id for b in blocks if b.ref_cnt == 0 and not b.is_null)
        return out

    BlockPool.get_new_blocks = get_new_blocks
    BlockPool._maybe_evict_cached_block = _maybe_evict_cached_block
    BlockPool.free_blocks = free_blocks
    if orig_unpin is not None:

        def unpin_blocks(self, blocks, on_reuse):
            if _TAP is None:
                return orig_unpin(self, blocks, on_reuse)
            blocks = list(blocks)
            out = orig_unpin(self, blocks, on_reuse)
            _TAP.on_freed(b.block_id for b in blocks if b.ref_cnt == 0 and not b.is_null)
            return out

        BlockPool.unpin_blocks = unpin_blocks
    BlockPool._q38_victim_patch = True
    _patch_free_queue_stamps()


def _patch_free_queue_stamps() -> None:
    """Stamp a block's free step where every free path ends: entering the free queue.

    The free_blocks/unpin_blocks wrappers above are not enough on the served engine: there the victim hook is
    installed at the engine_core stage, and the LIFO free patch (vllm_torchtpu._patch_vllm_block_pool_lifo_free,
    platform_activation) replaces BlockPool.free_blocks seconds later in the same EngineCore process, so the
    wrapper is dropped and every victim arrives unstamped (j-47fee8a1: 220,445 of 220,445). Any free path, the LIFO
    one included, puts the block on FreeKVCacheBlockQueue, so stamping here survives later replacements. Idempotent;
    inert unless a tap is active."""
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
