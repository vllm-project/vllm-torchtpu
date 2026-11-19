# SPDX-License-Identifier: Apache-2.0
"""Prefix-cache eviction policy for the PD prefill (engine core, host-only).

Two opt-in changes to which cached blocks the LRU free queue gives up first.
Neither changes what a hit contains or how it is computed, so outputs are
bit-identical; only the hit rate under memory pressure moves.

1. Demote stale checkpoints (``VLLM_TPU_PREFIX_DEMOTE_STALE_CHECKPOINT=1``).
   A turn that hits a GDN checkpoint published by its previous turn releases
   that checkpoint as soon as its state column moves on
   (``core/mamba_checkpoint_retention.py``). The checkpoint keeps its hash and
   is appended to the MRU end of the free queue, so it outlives most of the
   cache although the turn that hit it has just published a newer one. With
   this switch it is inserted at the evict-first end of the *cached* part of
   the queue instead: right after the hashless blocks at the head, which are
   always reused before any cached block. It still hits until it is evicted,
   so a later turn that diverges before the newest checkpoint can still fall
   back to it while the pool has room. When more than ``_MAX_WALK`` hashless
   blocks sit at the head the pool is not under pressure, and the block is
   appended as before.

   Only a *superseded* hit-on checkpoint is demoted: one strictly below the
   lowest replay boundary this request publishes itself (recorded from
   ``cache_blocks``). A turn that adds fewer tokens than it takes to reach
   a new boundary hits exactly at its own replay boundary. It publishes
   nothing newer, so the block it hit is still the conversation's live
   checkpoint, and the next turn needs exactly that block. That block keeps
   the retention patch's path. Served A/B j-74b70fab vs j-a0673a2d demoted
   it too, about half of all hit requests, and lost 0.11-0.19 hit. A block
   whose position or boundary is unknown also keeps today's path.

3. Tails evict first (``VLLM_TPU_PREFIX_TAIL_EVICT_FIRST=1``).
   With ``--prefix-match-unit`` below the block size, a request also leaves
   a *partial* prefix-cache entry: a state checkpoint, or a KV page, keyed at
   a boundary inside its block (``block_hash_num_tokens`` not a multiple of
   the block). These tails make the next turn's hit end at the previous
   prompt's unit-aligned end instead of its block-aligned one, but each costs
   a whole pool block per group. Kept at the MRU end, they push full
   checkpoints out of the pool under pressure: replay of FR1b's timeline puts
   the tail as kept today at +10% to +26% computed tokens at c64-c96. With
   this switch a freed tail joins an evict-first segment of the cached blocks.
   The segment comes after the hashless head and after any stale checkpoints
   demoted by (1), and it is **oldest tail first** (FIFO), so the most recent
   turn's tail survives longest. Under pressure the tails go before any full
   checkpoint, and the next turn falls back to the block-aligned checkpoint
   (at most one block recomputed). The no-pressure fallback is (1)'s.

2. Refresh waiting prefixes (``VLLM_TPU_PREFIX_REFRESH_WAITING=N``).
   A prefix hit is computed when a request is scheduled, not when it
   arrives. Under queueing (mean waits of 195-411 s at c128-c256 on i6) the
   prefix of a queued request ages in the LRU and is evicted before the
   request reaches the head. Before each scheduling step this moves the
   cached prefix blocks of the first ``N`` waiting requests to the MRU end of
   the free queue, in the request's own eviction order (attention tail first,
   state blocks last). Blocks are not pinned (no reference is taken), so the
   pool can still evict them when nothing else is left: no allocation can
   fail because of this. ``VLLM_TPU_PREFIX_REFRESH_INTERVAL_S`` (default 2)
   bounds how often one request is refreshed.

All switches are off by default. The pure helpers below touch only the
linked-list fields of vLLM's ``FreeKVCacheBlockQueue`` and ``KVCacheBlock``
(``prev_free_block``, ``next_free_block``, ``fake_free_list_head`` / ``_tail``,
``num_free_blocks``, ``ref_cnt``, ``block_hash``, ``is_null``), so they are
unit-tested without vLLM.
"""

import logging
import time
from collections.abc import Iterable, Sequence
from functools import wraps

# No vLLM / vllm_torchtpu import at module level: the pure helpers are
# unit-tested on a host without vLLM (tests load this file by path).
_LOGGER_NAME = "vllm.vllm_torchtpu.core.prefix_cache_policy"


class _LazyLogger:
    """vLLM's logger, resolved on first use, under the ``vllm.`` hierarchy.

    Only loggers under ``vllm.`` get vLLM's handler (see
    ``vllm_torchtpu/logger.py``); a plain ``logging.getLogger`` outside it
    prints nothing below WARNING, so the INFO install and engagement lines
    would be invisible. Without vLLM (CPU tests) it falls back to stdlib.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._logger = None

    def _resolve(self):
        if self._logger is None:
            try:
                from vllm.logger import init_logger

                self._logger = init_logger(self._name)
            except ImportError:
                self._logger = logging.getLogger(self._name)
        return self._logger

    def __getattr__(self, attr):
        return getattr(self._resolve(), attr)


logger = _LazyLogger(_LOGGER_NAME)

_DEMOTE_FLAG = "_tpu_prefix_demote_stale_checkpoint"
_REFRESH_FLAG = "_tpu_prefix_refresh_waiting"
_TAIL_FLAG = "_tpu_prefix_tail_evict_first"
# Longest hashless run at the queue head we walk past before deciding the pool
# is not under pressure (then the block keeps today's MRU placement).
_MAX_WALK = 512
# Segment bookkeeping, kept on the free queue (vLLM's KVCacheBlock has slots):
# the ids of demoted stale checkpoints and of queued tails while they sit in
# the evict-first segment, and the newest queued tail.
_DEMOTED_MARK = "_tpu_demoted_ids"
_TAIL_MARK = "_tpu_tail_ids"
_TAIL_LAST = "_tpu_tail_last"
# Longest walk over marked blocks when the newest-tail pointer is stale.
_MAX_SEGMENT_WALK = 4096
# Engagement counters, logged at most every _LOG_EVERY_S seconds (a policy that
# reports nothing moved has not engaged).
_LOG_EVERY_S = 60.0
STATS = {
    "demoted": 0,
    "demote_calls": 0,
    "kept_live": 0,
    "tails_queued": 0,
    "tail_calls": 0,
    "refreshed": 0,
    "refresh_passes": 0,
    "logged_at": 0.0,
}


def _maybe_log(now: float) -> None:
    if now - STATS["logged_at"] >= _LOG_EVERY_S:
        STATS["logged_at"] = now
        logger.info(
            "prefix cache policy: demoted=%d (calls %d) refreshed=%d (passes %d) "
            "kept_live=%d tails=%d (calls %d)",
            STATS["demoted"],
            STATS["demote_calls"],
            STATS["refreshed"],
            STATS["refresh_passes"],
            STATS["kept_live"],
            STATS["tails_queued"],
            STATS["tail_calls"],
        )


# --------------------------------------------------------------------------
# Pure free-queue helpers (no vLLM import)
# --------------------------------------------------------------------------
def _in_queue(block) -> bool:
    return block.prev_free_block is not None and block.next_free_block is not None


def _unlink(queue, block) -> None:
    block.prev_free_block.next_free_block = block.next_free_block
    block.next_free_block.prev_free_block = block.prev_free_block
    block.prev_free_block = block.next_free_block = None
    queue.num_free_blocks -= 1


def _link_after(queue, anchor, block) -> None:
    nxt = anchor.next_free_block
    anchor.next_free_block = block
    block.prev_free_block = anchor
    block.next_free_block = nxt
    nxt.prev_free_block = block
    queue.num_free_blocks += 1


def evict_first_anchor(queue, max_walk: int = _MAX_WALK):
    """The node after which a cached block is the next one evicted.

    Returns the last hashless block of the run at the head of the queue (or
    the fake head when the queue starts with a cached block), or None when the
    hashless run is longer than ``max_walk`` (no memory pressure).
    """
    anchor = queue.fake_free_list_head
    node = anchor.next_free_block
    tail = queue.fake_free_list_tail
    walked = 0
    while node is not None and node is not tail and node.block_hash is None:
        anchor = node
        node = node.next_free_block
        walked += 1
        if walked > max_walk:
            return None
    return anchor


def free_evict_first(queue, blocks: Iterable, max_walk: int = _MAX_WALK) -> list:
    """Drop one reference from each block; queue the freed ones evict-first.

    Blocks whose count reaches 0 and that carry a hash go right after the
    hashless head run, in the given order (the first given is evicted first).
    Hashless ones, and every block when the pool is not under pressure, go
    where vLLM's free path would put them: hashless at the head, cached at
    the tail. Returns the blocks that were queued evict-first.
    """
    freed = []
    for block in blocks:
        block.ref_cnt -= 1
        if block.ref_cnt == 0 and not block.is_null:
            freed.append(block)
    demoted = []
    anchor = None
    for block in freed:
        _set_marks(queue, block, demoted=False, tail=False)
        if block.block_hash is None:
            _link_after(queue, queue.fake_free_list_head, block)
            continue
        if anchor is None:
            anchor = evict_first_anchor(queue, max_walk)
            if anchor is None:
                anchor = False  # no pressure: plain MRU placement
        if anchor is False:
            _link_after(queue, queue.fake_free_list_tail.prev_free_block, block)
        else:
            _link_after(queue, anchor, block)
            _set_marks(queue, block, demoted=True, tail=False)
            anchor = block  # keep the given order
            demoted.append(block)
    return demoted


def _mark_sets(queue) -> tuple[set, set]:
    demoted = getattr(queue, _DEMOTED_MARK, None)
    if demoted is None:
        demoted = set()
        setattr(queue, _DEMOTED_MARK, demoted)
    tails = getattr(queue, _TAIL_MARK, None)
    if tails is None:
        tails = set()
        setattr(queue, _TAIL_MARK, tails)
    return demoted, tails


def _set_marks(queue, block, demoted: bool, tail: bool) -> None:
    demoted_ids, tail_ids = _mark_sets(queue)
    (demoted_ids.add if demoted else demoted_ids.discard)(block.block_id)
    (tail_ids.add if tail else tail_ids.discard)(block.block_id)


def _marked(queue, block, which: str) -> bool:
    ids = getattr(queue, which, None)
    return bool(ids) and block.block_id in ids


def clear_marks(queue, block) -> None:
    """The block leaves the evict-first segment (freed to the MRU end, moved)."""
    if _marked(queue, block, _DEMOTED_MARK) or _marked(queue, block, _TAIL_MARK):
        _set_marks(queue, block, demoted=False, tail=False)


def is_tail(block, unit: int) -> bool:
    """A cached block whose primary key is a partial prefix entry.

    vLLM records the prefix length a key covers (``block_hash_num_tokens``);
    a full block's key covers a multiple of its block size, a partial entry
    (``BlockPool.cache_partial_block``) ends inside the block.
    """
    if block.is_null or block.block_hash is None or unit <= 0:
        return False
    covered = getattr(block, "block_hash_num_tokens", None)
    return covered is not None and covered % unit != 0


def tail_segment_end(
    queue, max_walk: int = _MAX_WALK, max_segment_walk: int = _MAX_SEGMENT_WALK
):
    """The node after which the next tail is queued, or None (no pressure).

    The evict-first segment is: hashless head, demoted stale checkpoints
    (newest first), then tails (oldest first). A new tail goes after the
    newest queued tail, kept on the queue; if that block has left the
    segment, the end is found by walking the marked blocks after the head.
    """
    last = getattr(queue, _TAIL_LAST, None)
    if (
        last is not None
        and _in_queue(last)
        and _marked(queue, last, _TAIL_MARK)
        and last.block_hash is not None
    ):
        return last
    anchor = evict_first_anchor(queue, max_walk)
    if anchor is None:
        return None
    node = anchor.next_free_block
    end = queue.fake_free_list_tail
    walked = 0
    while (
        node is not None
        and node is not end
        and node.block_hash is not None
        and (_marked(queue, node, _DEMOTED_MARK) or _marked(queue, node, _TAIL_MARK))
        and walked < max_segment_walk
    ):
        anchor = node
        node = node.next_free_block
        walked += 1
    return anchor


def free_tails_first(
    queue,
    blocks: Iterable,
    max_walk: int = _MAX_WALK,
    max_segment_walk: int = _MAX_SEGMENT_WALK,
) -> list:
    """Drop one reference from each tail; queue the freed ones at the segment end.

    Oldest tail first: each newly freed tail goes after every tail already in
    the segment, in the given order. Without pressure (long hashless head) it
    goes to the MRU end as vLLM's free path would. Returns the tails queued in
    the segment.
    """
    freed = []
    for block in blocks:
        block.ref_cnt -= 1
        if block.ref_cnt == 0 and not block.is_null:
            freed.append(block)
    queued = []
    anchor = None
    for block in freed:
        _set_marks(queue, block, demoted=False, tail=False)
        if block.block_hash is None:
            _link_after(queue, queue.fake_free_list_head, block)
            continue
        if anchor is None:
            anchor = tail_segment_end(queue, max_walk, max_segment_walk)
            if anchor is None:
                anchor = False
        if anchor is False:
            _link_after(queue, queue.fake_free_list_tail.prev_free_block, block)
        else:
            _link_after(queue, anchor, block)
            _set_marks(queue, block, demoted=False, tail=True)
            anchor = block
            queued.append(block)
    if queued:
        setattr(queue, _TAIL_LAST, queued[-1])
    return queued


def free_with_tails(pool, blocks: Sequence, unit: int, plain_free) -> bool:
    """BlockPool.free_blocks with tails routed to the segment.

    Non-tail blocks take ``plain_free`` (vLLM's path) and leave the segment
    bookkeeping; tails take ``free_tails_first``. Returns whether any tail was
    given (counted in STATS).
    """
    tails, rest = split_tails(blocks, unit, pool)
    queue = pool.free_block_queue
    for block in rest:
        clear_marks(queue, block)
    if rest:
        plain_free(pool, rest)
    if not tails:
        return False
    STATS["tail_calls"] += 1
    STATS["tails_queued"] += len(free_tails_first(queue, tails))
    return True


def split_tails(blocks: Sequence, unit: int, pool=None) -> tuple[list, list]:
    """(tails, rest) in the given order; blocks of another pool are rest."""
    tails, rest = [], []
    for block in blocks:
        foreign = pool is not None and getattr(block, "pool", None) not in (None, pool)
        (tails if not foreign and is_tail(block, unit) else rest).append(block)
    return tails, rest


def refresh_to_mru(queue, blocks: Sequence, pool=None) -> int:
    """Move cached, unreferenced blocks to the MRU end, in the given order.

    The last block given ends up last evicted. Referenced blocks (not in the
    queue) and hashless ones are left alone. Returns the number moved.
    """
    moved = 0
    for block in blocks:
        # A referenced block is never in the free queue (vLLM invariant), so
        # queue membership is the test; hashless blocks are not cache.
        if block.block_hash is None or block.is_null or not _in_queue(block):
            continue
        if pool is not None and getattr(block, "pool", None) not in (None, pool):
            continue  # sits in another BlockPool's queue
        clear_marks(queue, block)
        _unlink(queue, block)
        _link_after(queue, queue.fake_free_list_tail.prev_free_block, block)
        moved += 1
    return moved


def live_boundary(replay_boundaries: Sequence[int], block_size: int) -> int | None:
    """Lowest position a request publishes a state checkpoint at.

    Align mode keeps the state block ending on each replay boundary rounded
    down to the block (``MambaManager.reachable_block_mask`` with the
    alignment equal to the block size, as on the served prefill).
    """
    if not replay_boundaries or block_size <= 0:
        return None
    return min(b // block_size * block_size for b in replay_boundaries)


def split_stale(
    own: dict | None, blocks: Sequence, pool=None, live_from: int | None = None
) -> tuple[list, list, int]:
    """Split state blocks leaving a request's table into (keep, stale, live).

    ``own`` is the retention patch's map of checkpoints this request
    published (block id -> (block, decode)). A hashed block the request did
    not publish came from a prefix hit. It is stale only if this request
    publishes a newer checkpoint: its position (``block_hash_num_tokens``)
    is below ``live_from``, the request's lowest replay boundary. A hit-on
    block at or past ``live_from`` is the conversation's live checkpoint (the
    turn crossed no new boundary) and is kept; ``live`` counts those. Own
    checkpoints, hashless running state, and blocks whose position or
    boundary is unknown keep the retention patch's path.
    """
    keep, stale, live = [], [], 0
    for block in blocks:
        entry = own.get(block.block_id) if own else None
        is_own = entry is not None and entry[0] is block
        # A block of another BlockPool (the image's vLLM tags deferred frees
        # with ``block.pool``) goes through that pool's own free path.
        foreign = pool is not None and getattr(block, "pool", None) not in (None, pool)
        if not is_own and not foreign and block.block_hash is not None and not block.is_null:
            position = getattr(block, "block_hash_num_tokens", None)
            if live_from is not None and position is not None and position < live_from:
                stale.append(block)
                continue
            live += 1
        keep.append(block)
    return keep, stale, live


def _live_from(manager) -> dict:
    """Per request: the lowest replay boundary it publishes (``live_boundary``)."""
    return manager.__dict__.setdefault("_tpu_live_from", {})


def eviction_ordered(hit_blocks: Sequence[Sequence], is_state: Sequence[bool]) -> list:
    """Flatten per-group hit blocks so the refresh keeps the free order.

    Matches ``mamba_checkpoint_retention``: attention blocks position-major,
    tail first; then state blocks (the checkpoint) last, so the checkpoint is
    the most protected block of the prefix.
    """
    attention = [g for g, s in zip(hit_blocks, is_state) if not s]
    state = [g for g, s in zip(hit_blocks, is_state) if s]
    order = []
    depth = max((len(g) for g in attention), default=0)
    for pos in range(depth - 1, -1, -1):
        order.extend(g[pos] for g in attention if pos < len(g))
    for g in state:
        order.extend(b for b in g if not b.is_null)
    return order


# --------------------------------------------------------------------------
# Patches
# --------------------------------------------------------------------------
def set_hash(block, block_hash, num_tokens=None) -> None:
    """Give a probe block a hash with whichever API this vLLM build has.

    The image's vLLM keeps ``block_hash`` a read-only property set through
    ``set_block_hash`` (it also records the covered token count); upstream
    vLLM has a ``block_hash`` setter. j-4e8d3ff8 (srv1) failed the contract
    check on exactly this and installed neither policy.
    """
    setter = getattr(block, "set_block_hash", None)
    if callable(setter):
        if num_tokens is None:
            setter(block_hash)
        else:
            setter(block_hash, num_tokens=num_tokens)
    else:
        block.block_hash = block_hash


def queue_contract_ok() -> bool:
    """Run the helpers on a real vLLM free queue before installing a patch.

    The helpers rely on FreeKVCacheBlockQueue's linked-list fields; if this
    vLLM build differs, skip the patch (logged) rather than corrupt the queue.
    """
    try:
        from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock

        free, cached, stale = (KVCacheBlock(block_id=i) for i in (1, 2, 3))
        set_hash(cached, b"cached")
        set_hash(stale, b"stale")
        stale.ref_cnt = 1
        queue = FreeKVCacheBlockQueue([free, cached])
        if free_evict_first(queue, [stale]) != [stale]:
            return False
        if refresh_to_mru(queue, [stale]) != 1:
            return False
        order = [queue.popleft() for _ in range(queue.num_free_blocks)]
        return order == [free, cached, stale]
    except Exception:
        logger.exception("prefix cache policy: free-queue contract check failed")
        return False


def patch_demote_stale_checkpoints() -> None:
    """Route hit-on checkpoints released by the retention patch evict-first."""
    from vllm_torchtpu import envs

    if not envs.VLLM_TPU_PREFIX_DEMOTE_STALE_CHECKPOINT:
        return
    if not envs.VLLM_TPU_MAMBA_CHECKPOINT_RETENTION:
        logger.warning(
            "VLLM_TPU_PREFIX_DEMOTE_STALE_CHECKPOINT needs the checkpoint "
            "retention patch (VLLM_TPU_MAMBA_CHECKPOINT_RETENTION=1); ignored."
        )
        return
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    from vllm_torchtpu.core import mamba_checkpoint_retention as mcr

    if getattr(mcr._release, _DEMOTE_FLAG, False):
        return
    if not queue_contract_ok():
        logger.warning("VLLM_TPU_PREFIX_DEMOTE_STALE_CHECKPOINT ignored (contract).")
        return
    original_release = mcr._release
    original_cache_blocks = MambaManager.cache_blocks
    original_pop = MambaManager.pop_blocks_for_free

    @wraps(original_cache_blocks)
    def cache_blocks(
        self,
        request,
        num_tokens: int,
        retention_interval: int | None = None,
        *,
        replay_boundaries: Sequence[int],
    ) -> None:
        if getattr(self, "mamba_cache_mode", None) == "align":
            _live_from(self)[request.request_id] = live_boundary(
                replay_boundaries, self.block_size
            )
        return original_cache_blocks(
            self,
            request,
            num_tokens,
            retention_interval,
            replay_boundaries=replay_boundaries,
        )

    @wraps(original_pop)
    def pop_blocks_for_free(self, request_id: str):
        _live_from(self).pop(request_id, None)
        return original_pop(self, request_id)

    @wraps(original_release)
    def _release(manager, request_id: str, blocks: list) -> None:
        keep, stale, live = split_stale(
            mcr._own(manager).get(request_id),
            blocks,
            manager.block_pool,
            _live_from(manager).get(request_id),
        )
        STATS["kept_live"] += live
        if keep:
            original_release(manager, request_id, keep)
        if stale:
            STATS["demote_calls"] += 1
            STATS["demoted"] += len(
                free_evict_first(manager.block_pool.free_block_queue, stale)
            )
            _maybe_log(time.monotonic())

    for fn in (_release, cache_blocks, pop_blocks_for_free):
        setattr(fn, _DEMOTE_FLAG, True)
    mcr._release = _release
    MambaManager.cache_blocks = cache_blocks
    MambaManager.pop_blocks_for_free = pop_blocks_for_free
    logger.info("Applied TPU patch: stale (hit-on) Mamba checkpoints freed evict-first.")
    logger.info(
        "prefix cache policy: a hit-on checkpoint at the request's own replay "
        "boundary is live and keeps the retention path."
    )


def tail_contract_ok(unit: int = 3200) -> bool:
    """Run the tail helpers on a real vLLM free queue and blocks first.

    Needs the image's ``block_hash_num_tokens`` (set through
    ``set_block_hash(..., num_tokens=)``) and a free queue that takes the
    segment bookkeeping; otherwise skip the patch (logged).
    """
    try:
        from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock

        free, cached, stale, t1, t2 = (KVCacheBlock(block_id=i) for i in (1, 2, 3, 4, 5))
        set_hash(cached, b"cached", num_tokens=unit)
        set_hash(stale, b"stale", num_tokens=unit)
        set_hash(t1, b"t1", num_tokens=unit + 128)
        set_hash(t2, b"t2", num_tokens=2 * unit + 128)
        if not (is_tail(t1, unit) and is_tail(t2, unit) and not is_tail(cached, unit)):
            return False
        for b in (stale, t1, t2):
            b.ref_cnt = 1
        queue = FreeKVCacheBlockQueue([free, cached])
        if free_evict_first(queue, [stale]) != [stale]:
            return False
        if free_tails_first(queue, [t1]) != [t1] or free_tails_first(queue, [t2]) != [t2]:
            return False
        order = [queue.popleft() for _ in range(queue.num_free_blocks)]
        return order == [free, stale, t1, t2, cached]
    except Exception:
        logger.exception("prefix cache policy: tail contract check failed")
        return False


def patch_tail_evict_first() -> None:
    """Free partial-entry (tail) blocks into the evict-first segment, oldest first."""
    from vllm_torchtpu import envs

    if not envs.VLLM_TPU_PREFIX_TAIL_EVICT_FIRST:
        return
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator

    if getattr(BlockPool.free_blocks, _TAIL_FLAG, False):
        return
    if not (queue_contract_ok() and tail_contract_ok()):
        logger.warning("VLLM_TPU_PREFIX_TAIL_EVICT_FIRST ignored (contract).")
        return
    original_init = KVCacheCoordinator.__init__
    original_free = BlockPool.free_blocks

    @wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        sizes = [g.kv_cache_spec.block_size for g in self.kv_cache_config.kv_cache_groups]
        # A partial entry ends inside its group's block. The smallest group
        # block is the conservative unit: a key that is a multiple of it is
        # never treated as a tail.
        self.block_pool._tpu_tail_unit = min(sizes) if sizes else 0

    @wraps(original_free)
    def free_blocks(self, ordered_blocks) -> None:
        unit = getattr(self, "_tpu_tail_unit", 0)
        if not unit or not self.enable_caching:
            return original_free(self, ordered_blocks)
        if free_with_tails(self, list(ordered_blocks), unit, original_free):
            _maybe_log(time.monotonic())

    for fn in (__init__, free_blocks):
        setattr(fn, _TAIL_FLAG, True)
    KVCacheCoordinator.__init__ = __init__
    BlockPool.free_blocks = free_blocks
    logger.info(
        "Applied TPU patch: partial-entry (tail) blocks freed evict-first, oldest first."
    )


def patch_refresh_waiting_prefixes() -> None:
    """Refresh the cached prefixes of queued requests before each step."""
    from vllm_torchtpu import envs

    limit = int(envs.VLLM_TPU_PREFIX_REFRESH_WAITING)
    if limit <= 0:
        return
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    if getattr(Scheduler.schedule, _REFRESH_FLAG, False):
        return
    if not queue_contract_ok():
        logger.warning("VLLM_TPU_PREFIX_REFRESH_WAITING ignored (contract).")
        return
    interval = float(envs.VLLM_TPU_PREFIX_REFRESH_INTERVAL_S)
    original_schedule = Scheduler.schedule

    @wraps(original_schedule)
    def schedule(self, *args, **kwargs):
        try:
            refresh_waiting(self, limit, interval, MambaManager)
        except Exception:  # never let a cache-policy hint break scheduling
            logger.exception("prefix refresh of waiting requests failed; disabled")
            self._tpu_prefix_refresh_off = True
        return original_schedule(self, *args, **kwargs)

    setattr(schedule, _REFRESH_FLAG, True)
    Scheduler.schedule = schedule
    logger.info(
        "Applied TPU patch: cached prefixes of the first %d waiting requests "
        "refreshed to MRU every %.1f s.",
        limit,
        interval,
    )


def refresh_waiting(scheduler, limit: int, interval: float, mamba_cls, now=None) -> int:
    """One refresh pass over the waiting queue; returns blocks moved."""
    if getattr(scheduler, "_tpu_prefix_refresh_off", False):
        return 0
    manager = scheduler.kv_cache_manager
    if not getattr(manager, "enable_caching", False):
        return 0
    queue = manager.block_pool.free_block_queue
    coordinator = manager.coordinator
    stamps = scheduler.__dict__.setdefault("_tpu_prefix_refresh_stamp", {})
    now = time.monotonic() if now is None else now
    is_state = [isinstance(m, mamba_cls) for m in coordinator.single_type_managers]
    moved = 0
    seen = set()
    queues = [getattr(scheduler, "kv_holding_waiting", None), scheduler.waiting]
    count = 0
    for waiting in queues:
        if not waiting:
            continue
        for request in waiting:
            if count >= limit:
                break
            count += 1
            rid = request.request_id
            seen.add(rid)
            if not request.block_hashes or now - stamps.get(rid, -1e18) < interval:
                continue
            stamps[rid] = now
            res = coordinator.find_longest_cache_hit(
                request.block_hashes, request.num_tokens - 1
            )
            hit_blocks, hit_len = res[0], res[1]
            if hit_len <= 0:
                continue
            moved += refresh_to_mru(
                queue, eviction_ordered(hit_blocks, is_state), manager.block_pool
            )
    for rid in [r for r in stamps if r not in seen]:
        del stamps[rid]
    STATS["refreshed"] += moved
    STATS["refresh_passes"] += 1
    _maybe_log(now)
    return moved
