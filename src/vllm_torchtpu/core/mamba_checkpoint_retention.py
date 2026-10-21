# SPDX-License-Identifier: Apache-2.0
"""Lifetime of Mamba "align" prefix-cache checkpoints (engine core).

A hybrid prefix hit needs both the attention KV of the prefix and a Mamba
(GDN, short-conv) state block saved exactly at the hit boundary. The pinned
vLLM release manages the two very differently, and three things follow:

1. A request publishes its prompt checkpoint (the replay boundary) while it
   prefills, and ``MambaManager.remove_skipped_blocks`` frees that block as
   soon as the state column moves past it, i.e. at decode start. The
   checkpoint enters the LRU free queue long before the request's KV does
   (on finish), so under pressure it is evicted first, and all of that
   session's still-resident KV becomes unusable for the next turn.
2. ``KVCacheCoordinator.free`` frees group by group, so whatever group comes
   first (the GDN groups for Qwen4Exp) lands at the evict-first end of the
   queue, and one group's whole range is evicted before any block of the
   next. A hit needs every group, so evicting the first few blocks strands
   the rest.
3. With ``prefix_cache_retention_interval=0``, no state reached during
   decode is ever hashed. The next turn of a conversation, whose prompt
   starts with this request's prompt and output, can only resume at this
   request's prompt end and recomputes the whole output.

This patch:

- Pins every checkpoint a request publishes until the request is freed,
  instead of releasing it when the state column moves on.
- Frees a request's blocks position by position across groups, tail first,
  with the Mamba state blocks after all attention blocks and the pinned
  checkpoints last. A session is then evicted from its useless tail
  (output KV past the last checkpoint) toward its head, and a checkpoint
  never leaves before the KV it needs.
- Without speculative decoding, also publishes the latest block boundary
  reached during decode (the newest one only: publishing it drops the
  previous decode checkpoint of the same request, hash and block). A
  decode step processes exactly one token, so the state block of column
  ``i`` holds the state after exactly ``(i + 1) * block_size`` tokens once
  the request has moved to column ``i + 1``: the step that processed token
  ``(i + 1) * block_size - 1`` was the last to write it. With MTP the last
  write into a column is the first position of a verify window, not the
  column's last token, so no exact decode-boundary state exists and decode
  checkpoints stay off.

``VLLM_TPU_MAMBA_CHECKPOINT_RETENTION=0`` disables the whole patch;
``VLLM_TPU_MAMBA_DECODE_CHECKPOINT=0`` disables only the decode checkpoints.
"""

from collections.abc import Sequence
from functools import wraps

from vllm.logger import init_logger

from vllm_torchtpu import envs

logger = init_logger(__name__)

_PATCH_FLAG = "_tpu_mamba_checkpoint_retention_patch"


def _own(manager) -> dict[str, dict[int, tuple[object, bool]]]:
    """Checkpoints a request published itself: block id -> (block, decode)."""
    return manager.__dict__.setdefault("_tpu_own_checkpoints", {})


def _pinned(manager) -> dict[str, list]:
    """Published checkpoints already dropped from the block table."""
    return manager.__dict__.setdefault("_tpu_pinned_checkpoints", {})


def _release(manager, request_id: str, blocks: list) -> None:
    """Free blocks leaving a request's table, keeping its own checkpoints."""
    own = _own(manager).get(request_id)
    freed = []
    for block in blocks:
        entry = own.get(block.block_id) if own else None
        if entry is not None and entry[0] is block and block.block_hash is not None:
            _pinned(manager).setdefault(request_id, []).append(block)
        else:
            freed.append(block)
    if freed:
        manager.block_pool.free_blocks(freed)


def _take_pinned(manager, request_id: str) -> list:
    _own(manager).pop(request_id, None)
    return _pinned(manager).pop(request_id, [])


def _drop_superseded(manager, request_id: str, keep: list) -> None:
    """Retire this request's older decode checkpoints: hash and block."""
    own = _own(manager).get(request_id)
    if not own:
        return
    pinned = _pinned(manager).get(request_id, [])
    for block_id, (block, decode) in list(own.items()):
        if not decode or any(block is k for k in keep):
            continue
        del own[block_id]
        manager.block_pool._maybe_evict_cached_block(block)
        if any(block is p for p in pinned):
            pinned[:] = [p for p in pinned if p is not block]
            # Hashless now, so it goes to the reuse-first end of the queue.
            manager.block_pool.free_blocks([block])
        # A block still in the table is freed with it, hashless.


def _tail_first(lists: Sequence[list]) -> list:
    """Position-major across groups, highest position first."""
    order = []
    for pos in range(max((len(b) for b in lists), default=0) - 1, -1, -1):
        order.extend(b[pos] for b in lists if pos < len(b))
    return order


def _head_first(lists: Sequence[list]) -> list:
    order = []
    for pos in range(max((len(b) for b in lists), default=0)):
        order.extend(b[pos] for b in lists if pos < len(b))
    return order


def _eviction_order(coordinator, request_id: str) -> list:
    """Pop the request's blocks in eviction order (first = evicted first)."""
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    attention, state, pinned = [], [], []
    for manager in coordinator.single_type_managers:
        if isinstance(manager, MambaManager):
            pinned.append(_take_pinned(manager, request_id))
            state.append(manager.pop_blocks_for_free(request_id))
        else:
            attention.append(manager.pop_blocks_for_free(request_id))
    # Pinned checkpoints are in publication order: the newest goes last.
    return _tail_first(attention) + _tail_first(state) + _head_first(pinned)


def _has_align_mamba(coordinator) -> bool:
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    return any(
        isinstance(m, MambaManager) and m.mamba_cache_mode == "align"
        for m in coordinator.single_type_managers
    )


def patch_mamba_checkpoint_retention() -> None:
    if not envs.VLLM_TPU_MAMBA_CHECKPOINT_RETENTION:
        return
    from vllm.utils.math_utils import cdiv
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
    from vllm.v1.core.single_type_kv_cache_manager import (
        MambaManager,
        SingleTypeKVCacheManager,
    )

    if getattr(MambaManager.cache_blocks, _PATCH_FLAG, False):
        return

    original_cache_blocks = MambaManager.cache_blocks
    original_remove_range = MambaManager._remove_blocks_in_range
    original_remove_skipped = MambaManager.remove_skipped_blocks
    original_pop = MambaManager.pop_blocks_for_free
    original_free = KVCacheCoordinator.free
    original_coordinator_pop = KVCacheCoordinator.pop_blocks_for_free
    decode_checkpoints = envs.VLLM_TPU_MAMBA_DECODE_CHECKPOINT

    @wraps(original_cache_blocks)
    def cache_blocks(
        self,
        request,
        num_tokens: int,
        retention_interval: int | None = None,
        *,
        replay_boundaries: Sequence[int],
    ) -> None:
        if self.mamba_cache_mode != "align":
            return original_cache_blocks(
                self,
                request,
                num_tokens,
                retention_interval,
                replay_boundaries=replay_boundaries,
            )
        request_id = request.request_id
        decode_boundary = None
        if (
            decode_checkpoints
            and self.num_speculative_blocks == 0
            # Dense retention (None) already caches every boundary.
            and retention_interval is not None
            # Decode only: one token per step, so the column the state just
            # left was last written by the step ending on its last token.
            and request.num_computed_tokens >= request.num_prompt_tokens
        ):
            boundary = num_tokens // self.block_size * self.block_size
            if boundary >= request.num_prompt_tokens and boundary > 0:
                decode_boundary = boundary
                replay_boundaries = (*replay_boundaries, boundary)
        before = self.num_cached_block.get(request_id, 0)
        original_cache_blocks(
            self,
            request,
            num_tokens,
            retention_interval,
            replay_boundaries=replay_boundaries,
        )
        after = self.num_cached_block.get(request_id, 0)
        blocks = self.req_to_blocks.get(request_id)
        if not blocks:
            return
        own = _own(self).setdefault(request_id, {})
        published_decode = []
        for idx in range(before, min(after, len(blocks))):
            block = blocks[idx]
            if block.is_null or block.block_hash is None:
                continue
            decode = (
                decode_boundary is not None
                and (idx + 1) * self.block_size == decode_boundary
            )
            own[block.block_id] = (block, decode)
            if decode:
                published_decode.append(block)
        checkpoint = self._checkpoints.get(request_id)
        if checkpoint is not None and 0 <= checkpoint[1] < len(blocks):
            block = blocks[checkpoint[1]]
            if not block.is_null and block.block_hash is not None:
                own.setdefault(block.block_id, (block, False))
        if published_decode:
            _drop_superseded(self, request_id, published_decode)

    @wraps(original_remove_range)
    def _remove_blocks_in_range(
        self, request_id: str, first_block: int, last_block: int
    ) -> None:
        if self.mamba_cache_mode != "align":
            return original_remove_range(self, request_id, first_block, last_block)
        blocks = self.req_to_blocks.get(request_id, [])
        first_block = max(first_block, self._num_retired_blocks.get(request_id, 0))
        last_block = min(last_block, len(blocks))
        if first_block >= last_block:
            return
        removed = []
        for i in range(last_block - 1, first_block - 1, -1):
            if blocks[i].is_null:
                continue
            removed.append(blocks[i])
            blocks[i] = self._null_block
        if removed:
            _release(self, request_id, removed)
        self._num_retired_blocks[request_id] = last_block

    @wraps(original_remove_skipped)
    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        if self.mamba_cache_mode != "align":
            return original_remove_skipped(
                self, request_id, processed_computed_tokens, num_prompt_tokens
            )
        SingleTypeKVCacheManager.remove_skipped_blocks(
            self, request_id, processed_computed_tokens, num_prompt_tokens
        )
        # Upstream: the block written two steps ago seeded the previous one
        # and is no longer the running state.
        last = self.last_state_block_idx.get(request_id)
        if (
            last is not None
            and last < cdiv(processed_computed_tokens, self.block_size) - 1
        ):
            blocks = self.req_to_blocks[request_id]
            if blocks[last] != self._null_block:
                block = blocks[last]
                blocks[last] = self._null_block
                _release(self, request_id, [block])

    @wraps(original_pop)
    def pop_blocks_for_free(self, request_id: str):
        pinned = _take_pinned(self, request_id)
        return original_pop(self, request_id) + pinned

    @wraps(original_free)
    def free(self, request_id: str) -> None:
        if not _has_align_mamba(self):
            return original_free(self, request_id)
        self.block_pool.free_blocks(_eviction_order(self, request_id))

    @wraps(original_coordinator_pop)
    def coordinator_pop_blocks_for_free(self, request_id: str):
        if not _has_align_mamba(self):
            return original_coordinator_pop(self, request_id)
        # Callers free the returned list reversed.
        return _eviction_order(self, request_id)[::-1]

    for fn in (
        cache_blocks,
        _remove_blocks_in_range,
        remove_skipped_blocks,
        pop_blocks_for_free,
        free,
        coordinator_pop_blocks_for_free,
    ):
        setattr(fn, _PATCH_FLAG, True)
    MambaManager.cache_blocks = cache_blocks
    MambaManager._remove_blocks_in_range = _remove_blocks_in_range
    MambaManager.remove_skipped_blocks = remove_skipped_blocks
    MambaManager.pop_blocks_for_free = pop_blocks_for_free
    KVCacheCoordinator.free = free
    KVCacheCoordinator.pop_blocks_for_free = coordinator_pop_blocks_for_free
    logger.info(
        "Applied TPU patch: Mamba align checkpoints pinned until request free, "
        "freed after their KV (decode checkpoints: %s).",
        "on (no spec decode)" if decode_checkpoints else "off",
    )
