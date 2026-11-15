# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import deque
from functools import wraps
from weakref import WeakKeyDictionary

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_PENDING_ALIGN_INDICES: WeakKeyDictionary[MambaManager, dict[str, deque[int]]] = (
    WeakKeyDictionary()
)
_INSTALLED = False


def _pending_for_manager(manager: MambaManager) -> dict[str, deque[int]]:
    pending = _PENDING_ALIGN_INDICES.get(manager)
    if pending is None:
        pending = {}
        _PENDING_ALIGN_INDICES[manager] = pending
    return pending


def _enqueue_monotonic(queue: deque[int], block_idx: int) -> None:
    if not queue or queue[-1] < block_idx:
        queue.append(block_idx)


def patch_mamba_async_align_free() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    orig_allocate_new_blocks = MambaManager.allocate_new_blocks
    orig_remove_skipped_blocks = MambaManager.remove_skipped_blocks
    orig_pop_blocks_for_free = MambaManager.pop_blocks_for_free

    @wraps(orig_allocate_new_blocks)
    def allocate_new_blocks_with_async_queue(
        self: MambaManager,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
    ) -> list[KVCacheBlock]:
        if self.mamba_cache_mode != "align":
            return orig_allocate_new_blocks(
                self, request_id, num_tokens, num_tokens_main_model
            )

        prev_block_len = len(self.req_to_blocks[request_id])
        returned = orig_allocate_new_blocks(
            self, request_id, num_tokens, num_tokens_main_model
        )
        req_blocks = self.req_to_blocks[request_id]
        if len(req_blocks) <= prev_block_len:
            return returned

        active_block_idx = len(req_blocks) - 1 - self.num_speculative_blocks
        pending = _pending_for_manager(self)

        prev_state_idx = self.last_state_block_idx.get(request_id)
        if (
            prev_state_idx is not None
            and 0 <= prev_state_idx < active_block_idx
            and not req_blocks[prev_state_idx].is_null
        ):
            _enqueue_monotonic(pending.setdefault(request_id, deque()), prev_state_idx)

        assert isinstance(self.kv_cache_spec, MambaSpec)
        if self.kv_cache_spec.num_prefill_checkpoint_blocks > 0:
            checkpoint_idx = active_block_idx - 1
            if (
                prev_block_len <= checkpoint_idx < active_block_idx
                and not req_blocks[checkpoint_idx].is_null
            ):
                _enqueue_monotonic(
                    pending.setdefault(request_id, deque()), checkpoint_idx
                )

        return returned

    @wraps(orig_remove_skipped_blocks)
    def remove_skipped_blocks_with_async_queue(
        self: MambaManager,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        orig_remove_skipped_blocks(
            self, request_id, processed_computed_tokens, num_prompt_tokens
        )
        if self.mamba_cache_mode != "align" or processed_computed_tokens <= 0:
            return

        pending = _PENDING_ALIGN_INDICES.get(self)
        if pending is None:
            return
        queue = pending.get(request_id)
        if not queue:
            return

        blocks = self.req_to_blocks.get(request_id)
        if not blocks:
            pending.pop(request_id, None)
            return

        max_freeable_idx = min(
            cdiv(processed_computed_tokens, self.block_size) - 1,
            len(blocks) - 1 - self.num_speculative_blocks,
        )

        null_block = self.block_pool.null_block
        freed: list[KVCacheBlock] = []
        while queue and queue[0] < max_freeable_idx:
            block_idx = queue.popleft()
            block = blocks[block_idx]
            if not block.is_null:
                freed.append(block)
                blocks[block_idx] = null_block

        if freed:
            self.block_pool.free_blocks(freed)
        if not queue:
            pending.pop(request_id, None)

    @wraps(orig_pop_blocks_for_free)
    def pop_blocks_for_free_with_async_queue(
        self: MambaManager, request_id: str
    ) -> list[KVCacheBlock]:
        pending = _PENDING_ALIGN_INDICES.get(self)
        if pending is not None:
            pending.pop(request_id, None)
        return orig_pop_blocks_for_free(self, request_id)

    MambaManager.allocate_new_blocks = allocate_new_blocks_with_async_queue
    MambaManager.remove_skipped_blocks = remove_skipped_blocks_with_async_queue
    MambaManager.pop_blocks_for_free = pop_blocks_for_free_with_async_queue
    _INSTALLED = True
    logger.debug("Patched MambaManager align-mode async skipped-block freeing")
