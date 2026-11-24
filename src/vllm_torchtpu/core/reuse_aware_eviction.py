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
"""Reuse-aware eviction order for the prefix-cache free block queue.

vLLM evicts free cached blocks in plain LRU order (time of last free). Two
kinds of blocks are known to be worth less or more than their age says:

* Decode-output blocks (``VLLM_TPU_EVICT_DECODE_FIRST``). A request's blocks
  past its last full prompt block hold generated tokens. When the client does
  not send our output back verbatim (agentic replays such as AgentX send the
  trace's assistant text; reasoning clients strip thinking), these blocks can
  never be a prefix hit, yet LRU keeps them as long as the request's prompt.
  With the flag they join the head of the free queue when the request frees
  them, so they are reused before any prompt block.
* Reused blocks (``VLLM_TPU_EVICT_PROTECT_HITS`` = T > 0). A block that has
  been a prefix hit at least T times since it was filled (a shared system /
  tool prefix of forks and subagents, a conversation's history) joins a
  protected segment at the tail of the queue; blocks below T join a
  probationary segment in front of it. Both segments are LRU. The protected
  segment holds at most ``VLLM_TPU_EVICT_PROTECTED_FRAC`` of the pool's blocks;
  past that its oldest blocks are demoted to the probationary tail (2-segment
  LRU, as in SLRU caches).

The queue stays one doubly linked list in eviction order
(``[decode-output | probationary | protected]``), with a pointer to the first
protected block, so everything that walks it from the head (the Raiden
write-back window, ``contiguous_alloc``) sees the true eviction order, and
every operation stays O(1) per block.

Installed by ``PrefixAwareScheduler`` (and its subclasses) when either flag is
set; DCP and hybrid pools are handled the same way since only the scheduler's
block pool is touched. Default off.
"""

from __future__ import annotations

from typing import Any

from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


class SegmentedFreeBlockQueue(FreeKVCacheBlockQueue):
    """`FreeKVCacheBlockQueue` with decode-first and 2-segment LRU order.

    ``append``/``append_n`` place a block by its class: decode-output blocks
    (ids in `decode_ids`) at the head, blocks with at least `protect_hits`
    prefix hits at the tail, all others just before the first protected
    block. Hashless blocks keep the pool's own order (it prepends them).
    """

    def __init__(
        self,
        blocks: list[KVCacheBlock],
        num_pool_blocks: int | None = None,
        protect_hits: int = 0,
        protected_frac: float = 0.3,
    ) -> None:
        super().__init__(blocks)
        n = len(blocks) if num_pool_blocks is None else num_pool_blocks
        self._init_segments(n, protect_hits, protected_frac)

    def _init_segments(
        self, num_pool_blocks: int, protect_hits: int, protected_frac: float
    ) -> None:
        self.protect_hits = protect_hits
        self.protected_cap = int(protected_frac * num_pool_blocks)
        # Prefix hits of each block since it was last filled (by block id).
        self.hits = [0] * num_pool_blocks
        # First block of the protected segment (the fake tail when empty).
        self.boundary: KVCacheBlock = self.fake_free_list_tail
        self.protected_ids: set[int] = set()
        # Blocks the current free() call should put at the head.
        self.decode_ids: set[int] = set()

    @classmethod
    def adopt(
        cls,
        queue: FreeKVCacheBlockQueue,
        num_pool_blocks: int,
        protect_hits: int,
        protected_frac: float,
    ) -> SegmentedFreeBlockQueue:
        """Take over `queue`'s linked list in place (its order is kept and
        every block in it starts probationary)."""
        new = cls.__new__(cls)
        new.num_free_blocks = queue.num_free_blocks
        new.fake_free_list_head = queue.fake_free_list_head
        new.fake_free_list_tail = queue.fake_free_list_tail
        new._init_segments(num_pool_blocks, protect_hits, protected_frac)
        return new

    @property
    def num_protected(self) -> int:
        return len(self.protected_ids)

    # -- bookkeeping for blocks leaving the list ---------------------------
    def _leave(self, block: KVCacheBlock) -> None:
        if block is self.boundary:
            nxt = block.next_free_block
            self.boundary = nxt if nxt is not None else self.fake_free_list_tail
        self.protected_ids.discard(block.block_id)

    def popleft(self) -> KVCacheBlock:
        first = self.fake_free_list_head.next_free_block
        if first is not None and first is not self.fake_free_list_tail:
            self._leave(first)
        return super().popleft()

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        if n <= 0:
            return []
        if not self.protected_ids:
            return super().popleft_n(n)
        return [self.popleft() for _ in range(n)]

    def remove(self, block: KVCacheBlock) -> None:
        self._leave(block)
        super().remove(block)

    # -- insertion -----------------------------------------------------------
    def _is_protected(self, block: KVCacheBlock) -> bool:
        t = self.protect_hits
        return t > 0 and block.block_hash is not None and self.hits[block.block_id] >= t

    def _insert_before(self, nxt: KVCacheBlock, blocks: list[KVCacheBlock]) -> None:
        prev = nxt.prev_free_block
        assert prev is not None
        for block in blocks:
            block.prev_free_block = prev
            prev.next_free_block = block
            prev = block
        prev.next_free_block = nxt
        nxt.prev_free_block = prev
        self.num_free_blocks += len(blocks)

    def append(self, block: KVCacheBlock) -> None:
        self.append_n([block])

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        if not blocks:
            return
        decode = self.decode_ids
        head: list[KVCacheBlock] = []
        probation: list[KVCacheBlock] = []
        protected: list[KVCacheBlock] = []
        for block in blocks:
            if decode and block.block_id in decode:
                head.append(block)
            elif self._is_protected(block):
                protected.append(block)
            else:
                probation.append(block)
        if head:
            super().prepend_n(head)
        if probation:
            self._insert_before(self.boundary, probation)
        if protected:
            was_empty = self.boundary is self.fake_free_list_tail
            super().append_n(protected)
            self.protected_ids.update(b.block_id for b in protected)
            if was_empty:
                self.boundary = protected[0]
            self._demote_overflow()

    def _demote_overflow(self) -> None:
        # The boundary block becomes the probationary tail: just move the pointer.
        while len(self.protected_ids) > self.protected_cap:
            block = self.boundary
            if block is self.fake_free_list_tail:
                break
            self.protected_ids.discard(block.block_id)
            nxt = block.next_free_block
            self.boundary = nxt if nxt is not None else self.fake_free_list_tail

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        return list(self.iter_blocks_after(None))


def _config() -> tuple[bool, int, float]:
    return (
        bool(envs.VLLM_TPU_EVICT_DECODE_FIRST),
        int(envs.VLLM_TPU_EVICT_PROTECT_HITS),
        float(envs.VLLM_TPU_EVICT_PROTECTED_FRAC),
    )


def enabled() -> bool:
    decode_first, protect_hits, _ = _config()
    return decode_first or protect_hits > 0


def install(kv_cache_manager: Any) -> SegmentedFreeBlockQueue | None:
    """Swap the manager's block pool queue for a `SegmentedFreeBlockQueue`
    per the flags. Must run before anything captures the queue object
    (`contiguous_alloc.install`). Returns the queue, or None when off."""
    decode_first, protect_hits, frac = _config()
    if not (decode_first or protect_hits > 0):
        return None
    pool = kv_cache_manager.block_pool
    if not getattr(pool, "enable_caching", False):
        logger.warning("reuse-aware eviction: prefix caching is off; not installed")
        return None
    if isinstance(pool.free_block_queue, SegmentedFreeBlockQueue):
        return pool.free_block_queue
    queue = SegmentedFreeBlockQueue.adopt(
        pool.free_block_queue, len(pool.blocks), protect_hits, frac
    )
    pool.free_block_queue = queue

    if protect_hits > 0:
        hits = queue.hits
        pool_touch = pool.touch

        def touch(blocks) -> None:
            for block in blocks:
                if not block.is_null:
                    hits[block.block_id] += 1
            pool_touch(blocks)

        pool_evict = pool._maybe_evict_cached_block

        def maybe_evict_cached_block(block) -> bool:
            # Called on every block the pool hands out: it is refilled.
            hits[block.block_id] = 0
            return pool_evict(block)

        pool.touch = touch
        pool._maybe_evict_cached_block = maybe_evict_cached_block

    if decode_first:
        _wrap_free(kv_cache_manager, queue)

    logger.info(
        "reuse-aware eviction: decode-output blocks first=%s, protect blocks "
        "with >= %d prefix hits (protected segment <= %.2f of %d blocks)",
        decode_first,
        protect_hits,
        frac,
        len(pool.blocks),
    )
    return queue


def decode_block_ids(kv_cache_manager: Any, request: Any) -> set[int]:
    """Ids of `request`'s blocks past its last full prompt block (the block
    holding the prompt end and every later one: they hold generated tokens)."""
    out: set[int] = set()
    coordinator = getattr(kv_cache_manager, "coordinator", None)
    managers = getattr(coordinator, "single_type_managers", ()) or ()
    num_prompt = int(getattr(request, "num_prompt_tokens", 0) or 0)
    for manager in managers:
        blocks = manager.req_to_blocks.get(request.request_id)
        if not blocks:
            continue
        block_size = int(manager.block_size)
        first = num_prompt // block_size if block_size > 0 else len(blocks)
        out.update(b.block_id for b in blocks[first:] if not b.is_null)
    return out


def _wrap_free(kv_cache_manager: Any, queue: SegmentedFreeBlockQueue) -> None:
    manager_free = kv_cache_manager.free

    def free(request) -> None:
        try:
            ids = decode_block_ids(kv_cache_manager, request)
        except Exception:  # never let ordering break a free
            logger.exception("reuse-aware eviction: decode block lookup failed")
            ids = set()
        queue.decode_ids = ids
        try:
            manager_free(request)
        finally:
            queue.decode_ids = set()

    kv_cache_manager.free = free
