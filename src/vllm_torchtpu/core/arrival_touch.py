# SPDX-License-Identifier: Apache-2.0
"""Refresh cached prefix LRU positions on request arrival (VLLM_TPU_ARRIVAL_TOUCH).

Under queue congestion, a returning multi-turn request can sit in the prefill
``waiting`` queue for tens of seconds before ``Scheduler.schedule()`` looks up
its prefix cache. Plain LRU evicts the request's own resident prefix blocks from
HBM (and from the host offload tier) while it waits, forcing expensive
recomputation.

When ``VLLM_TPU_ARRIVAL_TOUCH=1`` (default OFF), this engine-core patch wraps
``Scheduler.add_request`` to perform a non-pinning LRU refresh of the arriving
request's already-cached prefix:

1. **HBM (``BlockPool``):** locates cached attention blocks (contiguous from
   block 0 per group) and the rightmost cached recurrent-state checkpoint inside
   the shared attention run. Any such block currently unreferenced in
   ``free_block_queue`` (``ref_cnt == 0``) is rotated to the tail (MRU, evicted
   last) **without modifying ``ref_cnt``**, keeping tail-first attention order
   and checkpoint blocks last (matching ``mamba_checkpoint_retention.py``).
   Existing ``victim.active_tap().freed_at`` step stamps are saved and restored
   so the young-block guard retains the true step each block was freed.
2. **Host tier (optional ``TPURaidenOffloadingConnector``):** starting at the
   local hit boundary, finds the resident host-tier span via
   ``host_resident_tokens`` and calls ``manager.touch(keys, req_context)``
   (pin-balanced ``lookup`` + ``release``: ``Δpin == 0``), moving those host
   entries to MRU (and promoting attention entries under SLRU).

100% lossless, zero TTFT shields, zero persistent pins, zero DMA transfers.
Inert unless ``VLLM_TPU_ARRIVAL_TOUCH=1``.
"""

from __future__ import annotations

import time
from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_PATCH_FLAG = "_tpu_arrival_touch_patch"
_STATE_ATTR = "_tpu_arrival_touch_stats"
_LOG_INTERVAL_S = 60.0


def enabled() -> bool:
    return bool(envs.VLLM_TPU_ARRIVAL_TOUCH)


def _group_kinds(kv_cache_config: Any, block_pool: Any, managers: Any) -> tuple[list[int], list[int]] | None:
    """Return ``(attn_gids, state_gids)`` for prefix-cacheable groups matching ``hash_block_size``."""
    from vllm.v1.kv_cache_interface import MambaSpec

    attn_gids: list[int] = []
    state_gids: list[int] = []
    hash_bs = block_pool.hash_block_size
    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        if not getattr(spec, "prefix_cacheable", False):
            continue
        if managers[gid].block_size != hash_bs:
            return None
        if isinstance(spec, MambaSpec):
            state_gids.append(gid)
        else:
            attn_gids.append(gid)
    return attn_gids, state_gids


def _is_linked_free_block(block: Any) -> bool:
    """True iff ``block`` is unreferenced, hashed, non-null, and still in the free queue."""
    if block.ref_cnt != 0 or block.is_null or block.block_hash is None:
        return False
    # vLLM KVCacheBlock links free entries between head/tail dummy sentinels;
    # an unlinked block (not present in FreeKVCacheBlockQueue) has both None.
    if (
        hasattr(block, "prev_free_block")
        and block.prev_free_block is None
        and getattr(block, "next_free_block", None) is None
    ):
        return False
    return True


def _requeue_tail_preserving_stamps(free_queue: Any, blocks: list[Any]) -> int:
    """Move ``blocks`` to the MRU tail of ``free_queue`` without changing ``ref_cnt``.

    Preserves ``offload.victim.active_tap().freed_at`` timestamps so rotating a
    cached block to MRU never resets its write-retirement age for the victim
    policy's young-block guard.
    """
    if not blocks:
        return 0
    from vllm_torchtpu.offload import victim as offload_victim

    tap = offload_victim.active_tap()
    saved_stamps: dict[int, int] = {}
    unstamped: list[int] = []
    if tap is not None:
        for block in blocks:
            bid = block.block_id
            if bid in tap.freed_at:
                saved_stamps[bid] = tap.freed_at[bid]
            else:
                unstamped.append(bid)

    removed: list[Any] = []
    try:
        for block in blocks:
            free_queue.remove(block)
            removed.append(block)
    finally:
        if removed:
            append_n = getattr(free_queue, "append_n", None)
            if callable(append_n):
                append_n(removed)
            else:
                for block in removed:
                    free_queue.append(block)
        if tap is not None:
            for bid, step in saved_stamps.items():
                tap.freed_at[bid] = step
            for bid in unstamped:
                tap.freed_at.pop(bid, None)
    return len(removed)


def _touch_hbm_blocks(
    block_pool: Any,
    block_hashes: Any,
    num_tokens: int,
    attn_gids: list[int],
    state_gids: list[int],
) -> tuple[int, int, int]:
    """Rotate unreferenced cached HBM prefix blocks to the MRU end of ``free_block_queue``.

    Returns ``(moved_count, convergent_hit_tokens, attn_hit_tokens)``.
    """
    hash_bs = block_pool.hash_block_size
    if hash_bs <= 0 or not attn_gids or not block_hashes:
        return 0, 0, 0
    cap = min(max(num_tokens - 1, 0) // hash_bs, len(block_hashes))
    if cap <= 0:
        return 0, 0, 0

    attn_by_pos: dict[int, list[Any]] = {}
    min_attn_hit = cap
    max_attn_hit = 0
    for gid in attn_gids:
        run = 0
        for k in range(cap):
            found = block_pool.get_cached_block(block_hashes[k], [gid])
            if not found:
                break
            attn_by_pos.setdefault(k, []).append(found[0])
            run += 1
        if run < min_attn_hit:
            min_attn_hit = run
        if run > max_attn_hit:
            max_attn_hit = run

    if max_attn_hit == 0:
        return 0, 0, 0

    ckpt_blocks: list[Any] = []
    convergent_blocks = min_attn_hit
    if state_gids:
        convergent_blocks = 0
        for k in range(min_attn_hit - 1, -1, -1):
            found = block_pool.get_cached_block(block_hashes[k], state_gids)
            if found:
                convergent_blocks = k + 1
                ckpt_blocks = list(found)
                break

    # Evict-last order: tail-first across attention positions (block 0 near tail,
    # keeping shared roots warmest), followed by the active checkpoint blocks
    # (evicted last, after the KV blocks that depend on them).
    seen_ids: set[int] = set()
    to_move: list[Any] = []
    for k in range(max_attn_hit - 1, -1, -1):
        for block in attn_by_pos.get(k, ()):
            if block.block_id in seen_ids:
                continue
            if _is_linked_free_block(block):
                seen_ids.add(block.block_id)
                to_move.append(block)
    for block in ckpt_blocks:
        if block.block_id in seen_ids:
            continue
        if _is_linked_free_block(block):
            seen_ids.add(block.block_id)
            to_move.append(block)

    moved = _requeue_tail_preserving_stamps(block_pool.free_block_queue, to_move)
    return moved, convergent_blocks * hash_bs, min_attn_hit * hash_bs


def touch_hbm_prefix(kv_cache_manager: Any, request: Any) -> tuple[int, int, int]:
    """Touch cached HBM prefix blocks of ``request`` using its ``kv_cache_manager``.

    Returns ``(moved_blocks, convergent_hit_tokens, attn_hit_tokens)``.
    """
    if kv_cache_manager is None or not getattr(kv_cache_manager, "enable_caching", False):
        return 0, 0, 0
    block_pool = getattr(kv_cache_manager, "block_pool", None)
    kv_cache_config = getattr(kv_cache_manager, "kv_cache_config", None)
    coordinator = getattr(kv_cache_manager, "coordinator", None)
    if block_pool is None or kv_cache_config is None or coordinator is None:
        return 0, 0, 0
    kinds = _group_kinds(kv_cache_config, block_pool, coordinator.single_type_managers)
    if kinds is None:
        return 0, 0, 0
    attn_gids, state_gids = kinds
    return _touch_hbm_blocks(
        block_pool,
        getattr(request, "block_hashes", None),
        int(getattr(request, "num_tokens", 0) or 0),
        attn_gids,
        state_gids,
    )


def _find_offload_scheduler(connector: Any) -> Any | None:
    """Return the host-tier ``TPURaidenOffloadingScheduler`` inside ``connector`` if any."""
    if connector is None:
        return None
    members = getattr(connector, "sub_connectors", None)
    if members is None:
        members = getattr(connector, "_connectors", None)
    candidates = [connector] if members is None else list(members)
    for member in candidates:
        if not callable(getattr(member, "host_tier_hit_tokens", None)):
            continue
        sched = getattr(member, "connector_scheduler", None)
        if sched is not None and callable(getattr(sched, "host_resident_tokens", None)):
            return sched
    return None


def _host_tier_keys_to_touch(
    offload_sched: Any, request: Any, start_tokens: int, tier_tokens: int
) -> list[Any]:
    """Collect the resident host-tier ``OffloadKey``s covering ``[start_tokens, end)``."""
    from vllm.v1.kv_offload.base import make_offload_key

    hashes = getattr(request, "block_hashes", None)
    if not hashes or tier_tokens <= 0:
        return []
    end = start_tokens + tier_tokens
    groups = offload_sched.config.kv_group_configs
    keys: list[Any] = []
    seen: set[Any] = set()
    for group in groups:
        tpc = group.tokens_per_chunk
        step = group.hashes_per_chunk
        if tpc <= 0 or step <= 0:
            continue
        max_chunks = len(hashes) // step
        stop = min(end // tpc, max_chunks)
        if group.sliding_window_size_in_chunks is None:
            first = start_tokens // tpc
        else:
            first = max(stop - group.sliding_window_size_in_chunks, 0)
        for chunk in range(first, stop):
            key = make_offload_key(hashes[(chunk + 1) * step - 1], group.group_idx)
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def touch_host_tier_prefix(
    connector: Any,
    request: Any,
    convergent_tokens: int,
    attn_tokens: int | None = None,
) -> int:
    """Refresh MRU order (and SLRU promotion) of resident host-tier blocks for ``request``."""
    offload_sched = _find_offload_scheduler(connector)
    if offload_sched is None:
        return 0
    manager = getattr(offload_sched, "_raiden_manager", None)
    if manager is None or not callable(getattr(manager, "touch", None)):
        return 0

    start = convergent_tokens
    tier_tokens = 0
    if (
        attn_tokens is not None
        and attn_tokens > convergent_tokens
        and envs.VLLM_TPU_OFFLOAD_DIVERGENT_HITS
    ):
        tier_tokens = int(offload_sched.host_resident_tokens(request, attn_tokens))
        if tier_tokens > 0:
            start = attn_tokens
    if tier_tokens <= 0:
        tier_tokens = int(offload_sched.host_resident_tokens(request, convergent_tokens))
        start = convergent_tokens
    if tier_tokens <= 0:
        return 0

    keys = _host_tier_keys_to_touch(offload_sched, request, start, tier_tokens)
    if not keys:
        return 0
    from vllm.v1.kv_offload.base import ReqContext

    manager.touch(keys, ReqContext(req_id=request.request_id))
    return len(keys)


def touch_on_arrival(scheduler: Any, request: Any) -> tuple[int, int]:
    """Refresh HBM and host-tier cached prefixes for an arriving request.

    Returns ``(hbm_blocks_moved, host_tier_keys_touched)``.
    """
    if getattr(request, "num_computed_tokens", 0) or getattr(
        request, "skip_reading_prefix_cache", False
    ):
        return 0, 0
    kvm = getattr(scheduler, "kv_cache_manager", None)
    moved, conv_toks, attn_toks = touch_hbm_prefix(kvm, request)
    tier_keys = touch_host_tier_prefix(
        getattr(scheduler, "connector", None),
        request,
        conv_toks,
        attn_toks,
    )
    return moved, tier_keys


def patch_arrival_touch() -> None:
    """Wrap ``Scheduler.add_request`` to touch cached prefixes on arrival.

    Inert unless ``VLLM_TPU_ARRIVAL_TOUCH=1``.
    """
    if not enabled():
        return
    from vllm.v1.core.sched.scheduler import Scheduler

    original = Scheduler.add_request
    if getattr(original, _PATCH_FLAG, False):
        return

    def add_request(self, request):
        is_new = request.request_id not in self.requests
        result = original(self, request)
        if is_new:
            stats = self.__dict__.get(_STATE_ATTR)
            if stats is None:
                stats = self.__dict__[_STATE_ATTR] = {
                    "reqs": 0,
                    "hbm_blocks": 0,
                    "tier_keys": 0,
                    "broken": False,
                    "last_log": time.monotonic(),
                }
            if not stats["broken"]:
                try:
                    moved, tier_keys = touch_on_arrival(self, request)
                    stats["reqs"] += 1
                    stats["hbm_blocks"] += moved
                    stats["tier_keys"] += tier_keys
                    now = time.monotonic()
                    if now - stats["last_log"] >= _LOG_INTERVAL_S:
                        stats["last_log"] = now
                        logger.info(
                            "arrival-touch: reqs=%d hbm_blocks=%d tier_keys=%d "
                            "waiting=%d",
                            stats["reqs"],
                            stats["hbm_blocks"],
                            stats["tier_keys"],
                            len(self.waiting),
                        )
                except Exception:
                    stats["broken"] = True
                    logger.exception(
                        "arrival-touch: failed for %s; disabling for this scheduler.",
                        getattr(request, "request_id", "?"),
                    )
        return result

    setattr(add_request, _PATCH_FLAG, True)
    Scheduler.add_request = add_request
    logger.info(
        "Applied TPU patch: arrival prefix-cache touch (VLLM_TPU_ARRIVAL_TOUCH)."
    )
