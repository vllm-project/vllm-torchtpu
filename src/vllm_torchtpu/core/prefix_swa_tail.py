# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Keep the sliding-window KV before each prompt's last block boundary, so the
next turn's prefix hit there skips the SWA bounded replay
(VLLM_TPU_PREFIX_KEEP_SWA_TAIL).

With SWA bounded replay (DeepSeek-V4.1 with prefix caching), the sliding-window
groups are left out of prefix caching, and the scheduler rewinds every hit by
the window (``Scheduler._mark_prefix_replay``): the model computes the hit's
last ``prefix_replay_tokens`` tokens again to rebuild their sliding-window KV.
A turn that extends its conversation hits at the last full block of the
previous turn's prompt: the previous turn's output is rendered into the next
prompt with other tokens, so its decoded blocks are not hit.

This patch caches, at the last block boundary ``X`` of each prompt, the
sliding-window pages of the ``prefix_replay_tokens`` tokens before ``X``, under
keys derived from the hash of the prefix ending at ``X``.
They stay cached after the window passes them, like any freed cached block. A
hit at ``X`` adopts them when every replaying group still holds them and the
prompt extends at least one window past ``X``; it then computes from ``X`` with
no replay. Any other hit replays as before.

What the model computes: the adopted rows are the ones the earlier request
wrote for the same token prefix. Positions below ``X`` are read only by the
encoder layers' sliding window, which the writer computed for every prompt
position: the CED decoder masks keys below ``prompt - window >= X``, so its
rows there, which the writer never computed, are not read. To the worker the
request looks like the second chunk of a chunked prefill whose block table
holds the first chunk's pages. The replay instead rebuilds those rows with
their windows cut at the replay start.

Installed in the engine-core process (patch_registry's ``engine_core`` stage).
"""

import collections
import functools
from typing import TYPE_CHECKING

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)

_STATE = "_tpu_swa_tail"
# The hit length a request's latest lookup adopted tails for; 0 when none.
_HIT = "_tpu_swa_tail_hit"
_TAG = b"\x00swa-tail"
_LOG_EVERY = 500


class SwaTails:
    """The replaying groups of one KV cache manager and their tail geometry."""

    def __init__(self, group_ids, window, page_tokens, boundary) -> None:
        self.group_ids = tuple(group_ids)
        self.window = window
        self.page_tokens = page_tokens
        # Hits land on scheduler blocks; tails are kept at those boundaries.
        self.boundary = boundary
        self.lookups = 0
        self.adopted = 0
        self.stored = 0
        # Why hits replayed, by reason, for the log.
        self.replayed = collections.Counter()

    def pages(self, boundary: int) -> range:
        """Page indices of the window before ``boundary``."""
        return range(
            (boundary - self.window) // self.page_tokens,
            boundary // self.page_tokens,
        )

    def keys(self, request: "Request", block_pool, boundary: int, group_id: int):
        """Cache keys of the window pages before ``boundary``, or None when the
        request has no hash for the prefix ending there."""
        from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id

        index = boundary // block_pool.hash_block_size - 1
        if index < 0 or index >= len(request.block_hashes):
            return None
        parent = request.block_hashes[index]
        return [
            make_block_hash_with_group_id(
                BlockHash(parent + _TAG + page.to_bytes(4, "big")), group_id
            )
            for page in range(len(self.pages(boundary)))
        ]

    def counted(self, replayed: str | None) -> None:
        self.lookups += 1
        if replayed is None:
            self.adopted += 1
        else:
            self.replayed[replayed] += 1
        if self.lookups % _LOG_EVERY == 0:
            logger.info(
                "SWA tails: %d of %d prefix hits adopted the window (no replay); "
                "%d tails stored; replayed: %s.",
                self.adopted,
                self.lookups,
                self.stored,
                dict(self.replayed),
            )


def _state(manager) -> SwaTails | None:
    return getattr(manager, _STATE, None)


def store_tails(manager, request: "Request", num_tokens: int) -> None:
    """Cache the window pages before the prompt's last block boundary once the
    request has scheduled past it, if this request holds them."""
    state: SwaTails | None = getattr(manager, "_tpu_swa_tail_owner", None)
    if state is None:
        return
    done = manager._tpu_swa_tail_done
    request_id = request.request_id
    boundary = request.num_prompt_tokens // state.boundary * state.boundary
    if request_id in done or boundary < state.window or num_tokens < boundary:
        return
    done.add(request_id)
    # The window manager nulls pages only below the scheduled chunk's own
    # window, so the chunk that reaches the boundary still holds these pages;
    # a hit past the boundary left them null, and nothing is stored.
    blocks = manager.req_to_blocks.get(request_id, ())
    pages = [blocks[i] for i in state.pages(boundary) if i < len(blocks)]
    keys = state.keys(request, manager.block_pool, boundary, manager.kv_cache_group_id)
    if keys is None or len(pages) != len(keys) or any(p.is_null for p in pages):
        return
    for key, page in zip(keys, pages):
        manager.block_pool._insert_block_hash(key, page, num_tokens=None)
    state.stored += 1


def adopt_tails(kv_cache_manager, request: "Request", blocks, num_hit_tokens: int):
    """Return ``(blocks with each replaying group's window pages before the
    hit, None)``, or ``(None, why the hit must replay)``."""
    state = _state(kv_cache_manager)
    assert state is not None, "adopt_tails runs only with SWA tails on"
    if num_hit_tokens % state.boundary:
        return None, "unaligned"
    if request.num_prompt_tokens - num_hit_tokens < state.window:
        return None, "short suffix"
    coordinator = kv_cache_manager.coordinator
    block_pool = coordinator.block_pool
    lists = [list(group) for group in blocks.blocks]
    for group_id in state.group_ids:
        keys = state.keys(request, block_pool, num_hit_tokens, group_id)
        if keys is None:
            return None, "no hash"
        pages = [block_pool.cached_block_hash_to_block.get_one_block(k) for k in keys]
        if not all(pages):
            return None, f"no tail in group {group_id}"
        assert not lists[group_id], "replaying groups take no prefix-cache hits"
        null_pages = num_hit_tokens // state.page_tokens - len(pages)
        lists[group_id] = [block_pool.null_block] * null_pages + pages
    return kv_cache_manager.create_kv_cache_blocks(tuple(lists)), None


def _patch_window_manager(manager_cls) -> None:
    cache_blocks = manager_cls.cache_blocks
    free = manager_cls.free

    @functools.wraps(cache_blocks)
    def cache_blocks_and_tails(self, request, num_tokens, *args, **kwargs):
        cache_blocks(self, request, num_tokens, *args, **kwargs)
        store_tails(self, request, num_tokens)

    @functools.wraps(free)
    def free_and_forget(self, request_id):
        done = getattr(self, "_tpu_swa_tail_done", None)
        if done is not None:
            done.discard(request_id)
        free(self, request_id)

    manager_cls.cache_blocks = cache_blocks_and_tails
    manager_cls.free = free_and_forget


def _patch_kv_cache_manager(manager_cls) -> None:
    init = manager_cls.__init__
    get_computed_blocks = manager_cls.get_computed_blocks

    @functools.wraps(init)
    def init_with_tails(self, *args, **kwargs):
        init(self, *args, **kwargs)
        coordinator = self.coordinator
        groups = coordinator.kv_cache_config.kv_cache_groups
        replaying = [
            i
            for i, group in enumerate(groups)
            if getattr(group.kv_cache_spec, "prefix_replay_tokens", 0) > 0
        ]
        if not (self.enable_caching and replaying):
            return
        block_pool = coordinator.block_pool
        managers = [coordinator.single_type_managers[i] for i in replaying]
        windows = {groups[i].kv_cache_spec.prefix_replay_tokens for i in replaying}
        page_tokens = {manager.block_size for manager in managers}
        boundary = coordinator.scheduler_block_size
        window, page = max(windows), max(page_tokens)
        reason = None
        if len(windows) != 1 or len(page_tokens) != 1:
            reason = f"windows {sorted(windows)}, page sizes {sorted(page_tokens)}"
        elif block_pool.enable_kv_cache_events:
            reason = "KV cache events are on"
        elif window % page or boundary % page or boundary % block_pool.hash_block_size:
            reason = f"window {window}, page {page}, boundary {boundary}"
        if reason is not None:
            logger.warning("SWA tails stay off: %s.", reason)
            return
        state = SwaTails(replaying, window, page, boundary)
        setattr(self, _STATE, state)
        for manager in managers:
            manager._tpu_swa_tail_owner = state
            manager._tpu_swa_tail_done = set()
        logger.info(
            "SWA tails on: groups %s keep the %d-token window before each "
            "prompt's last %d-token boundary (%d-token pages).",
            replaying,
            window,
            boundary,
            page,
        )

    @functools.wraps(get_computed_blocks)
    def get_computed_blocks_with_tails(self, request):
        blocks, num_hit_tokens, boundary = get_computed_blocks(self, request)
        state = _state(self)
        if state is None:
            return blocks, num_hit_tokens, boundary
        if num_hit_tokens <= 0:
            setattr(request, _HIT, 0)
            return blocks, num_hit_tokens, boundary
        adopted, replayed = adopt_tails(self, request, blocks, num_hit_tokens)
        setattr(request, _HIT, num_hit_tokens if adopted is not None else 0)
        state.counted(replayed)
        return (adopted or blocks), num_hit_tokens, boundary

    manager_cls.__init__ = init_with_tails
    manager_cls.get_computed_blocks = get_computed_blocks_with_tails


def _patch_scheduler(scheduler_cls) -> None:
    mark_prefix_replay = scheduler_cls._mark_prefix_replay

    @functools.wraps(mark_prefix_replay)
    def mark_prefix_replay_unless_adopted(self, request, num_hit_tokens):
        if num_hit_tokens > 0 and getattr(request, _HIT, 0) == num_hit_tokens:
            # The hit adopted its window: nothing to rebuild.
            request.replay_start = 0
            return 0
        return mark_prefix_replay(self, request, num_hit_tokens)

    scheduler_cls._mark_prefix_replay = mark_prefix_replay_unless_adopted


def install() -> None:
    """Patch the engine core's KV cache manager and scheduler when the switch
    is on."""
    from vllm_torchtpu import envs

    if not envs.VLLM_TPU_PREFIX_KEEP_SWA_TAIL:
        return
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager

    if getattr(KVCacheManager, "_tpu_prefix_swa_tail_patch", False):
        return
    _patch_window_manager(SlidingWindowManager)
    _patch_kv_cache_manager(KVCacheManager)
    _patch_scheduler(Scheduler)
    KVCacheManager._tpu_prefix_swa_tail_patch = True
    logger.info("Applied TPU patch: prefix hits adopt the kept SWA window.")
