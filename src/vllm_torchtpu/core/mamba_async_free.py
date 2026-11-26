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

from __future__ import annotations

from collections import deque
from functools import wraps
from typing import Any
from weakref import WeakKeyDictionary, WeakSet

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_PENDING_ALIGN_INDICES: WeakKeyDictionary[MambaManager, dict[str, deque[int]]] = (
    WeakKeyDictionary()
)
_ENABLED_MANAGERS: WeakSet[MambaManager] = WeakSet()
_INSTALLED = False
_SCHEDULER_PATCHED = False


def _pending_for_manager(manager: MambaManager) -> dict[str, deque[int]]:
    pending = _PENDING_ALIGN_INDICES.get(manager)
    if pending is None:
        pending = {}
        _PENDING_ALIGN_INDICES[manager] = pending
    return pending


def _enqueue_monotonic(queue: deque[int], block_idx: int) -> None:
    if not queue or queue[-1] < block_idx:
        queue.append(block_idx)


def enable_mamba_async_align_free(kv_cache_manager: Any) -> bool:
    """Opt a KVCacheManager's MambaManagers into K3 async align-mode freeing."""
    coordinator = getattr(kv_cache_manager, "coordinator", None)
    managers = getattr(coordinator, "single_type_managers", ()) or ()
    enabled = False
    for manager in managers:
        if isinstance(manager, MambaManager):
            _ENABLED_MANAGERS.add(manager)
            enabled = True
    if enabled:
        _install_mamba_manager_wrappers()
    return enabled


def _install_mamba_manager_wrappers() -> None:
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
        if self not in _ENABLED_MANAGERS or self.mamba_cache_mode != "align":
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
        if (
            self not in _ENABLED_MANAGERS
            or self.mamba_cache_mode != "align"
            or processed_computed_tokens <= 0
        ):
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


def patch_mamba_async_align_free() -> None:
    """Enable async align-mode skipped-block freeing only for K3 megakernel engines."""
    global _SCHEDULER_PATCHED
    if _SCHEDULER_PATCHED:
        return

    from vllm.v1.core.sched.scheduler import Scheduler

    original_init = Scheduler.__init__

    @wraps(original_init)
    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        from vllm_torchtpu.models.kimi_k3.megakernel import backend

        if backend.role(self.vllm_config) is None:
            return
        enable_mamba_async_align_free(self.kv_cache_manager)

    Scheduler.__init__ = init  # type: ignore[method-assign]
    _SCHEDULER_PATCHED = True
