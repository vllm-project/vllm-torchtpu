# SPDX-License-Identifier: Apache-2.0
"""Prefix-cache hits finer than the scheduler block on full-attention +
sliding-window hybrids (``TPU_FINE_PREFIX_HIT_TOKENS``).

DeepSeek-V4 on TPU pairs 1024-token full-attention (compressed MLA) blocks with
128-token sliding-window blocks and 16/32-token compressor-state blocks, and
vLLM hashes prompts every 16 tokens. Without a Mamba group vLLM still aligns
every hit to the 1024-token scheduler block, so each turn of a conversation
recomputes on average half a block of the previous turn's prompt.

With ``TPU_FINE_PREFIX_HIT_TOKENS=U`` hits land on multiples of ``U``:

- ``U`` is a whole number of every windowed group's blocks, so those groups
  keep their own lookup and checkpoint mask, just evaluated at ``U``.
- The full-attention group probes, past its run of cached full blocks, the
  ``U`` boundaries inside the next block. A hit there is copied on write by
  vLLM's partial-hit path, so the hitting request writes into its own copy.
- Each request registers its full-attention prompt-tail block under the last
  ``U`` boundary before its final prompt token, where the next turn of the
  conversation resumes, and that entry survives the block filling up during
  decode (KV rows below the boundary never change). Windowed groups keep a
  checkpoint at that boundary and at the block-aligned one before it.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from itertools import islice
from typing import TYPE_CHECKING

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.request import Request

logger = init_logger(__name__)


def fine_prefix_hit_tokens() -> int:
    """The configured hit unit in tokens, 0 when the feature is off."""
    return envs.TPU_FINE_PREFIX_HIT_TOKENS


def fine_boundary(num_prompt_tokens: int, unit: int) -> int:
    """Last ``unit`` boundary a request with an identical prefix can hit at.

    Hits are capped at ``num_prompt_tokens - 1`` (the last prompt token is
    always recomputed for its logits).
    """
    return (num_prompt_tokens - 1) // unit * unit


def _validate(coordinator, unit: int) -> None:
    from vllm.v1.core.kv_cache_utils import iter_layer_specs
    from vllm.v1.core.single_type_kv_cache_manager import (
        FullAttentionManager,
        SlidingWindowManager,
    )
    from vllm.v1.kv_cache_interface import MambaSpec

    def fail(why: str) -> None:
        raise ValueError(f"TPU_FINE_PREFIX_HIT_TOKENS={unit}: {why}")

    hash_bs = coordinator.hash_block_size
    sched_bs = coordinator.scheduler_block_size
    # vLLM 0.31 keeps enable_caching on the block pool and the managers, not
    # on the coordinator.
    if not coordinator.block_pool.enable_caching:
        fail("needs prefix caching")
    if coordinator.enable_partial_hash_hits:
        fail("vLLM's own partial hash hits are already active")
    if unit % hash_bs or sched_bs % unit or unit >= sched_bs:
        fail(
            f"must be a multiple of the hash block ({hash_bs}) and a proper "
            f"divisor of the scheduler block ({sched_bs})"
        )
    full = 0
    for group, manager in zip(
        coordinator.kv_cache_config.kv_cache_groups,
        coordinator.single_type_managers,
    ):
        spec = group.kv_cache_spec
        if not spec.prefix_cacheable:
            continue
        for leaf in iter_layer_specs(spec):
            if isinstance(leaf, MambaSpec):
                fail("Mamba groups are not supported")
            per_state = leaf.tokens_per_state
            if isinstance(per_state, int) and per_state > 1 and unit % per_state:
                fail(f"must be a multiple of the compression ratio {per_state}")
        if type(manager) is FullAttentionManager:
            full += 1
            if manager.block_size % unit:
                fail(f"must divide the full-attention block {manager.block_size}")
        elif type(manager) is SlidingWindowManager:
            if unit % manager.block_size:
                fail(f"must be a multiple of the windowed block {manager.block_size}")
        else:
            fail(f"unsupported KV cache manager {type(manager).__name__}")
    if full == 0:
        fail("needs a full-attention group")


def _keep_block_aligned_checkpoints(manager, scheduler_block_size: int) -> None:
    """Windowed groups also keep the block-aligned replay checkpoint.

    The fine checkpoint serves the next turn; a request that diverges between
    the two boundaries (a truncated history, a sibling stream) still needs the
    block-aligned one, which block-aligned hits would have kept.
    """
    block_mask = manager.reachable_block_mask

    def reachable_block_mask(*args, reachable_boundaries=(), **kwargs):
        boundaries = list(reachable_boundaries)
        boundaries += [
            b // scheduler_block_size * scheduler_block_size for b in boundaries
        ]
        return block_mask(*args, reachable_boundaries=boundaries, **kwargs)

    manager.reachable_block_mask = reachable_block_mask
    manager._tpu_coarse_alignment = scheduler_block_size


def _enable(coordinator, unit: int) -> None:
    from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager

    _validate(coordinator, unit)
    coordinator._tpu_fine_prefix_unit = unit
    for manager in coordinator.single_type_managers:
        manager.cache_hit_alignment_tokens = unit
        if isinstance(manager, SlidingWindowManager):
            _keep_block_aligned_checkpoints(manager, coordinator.scheduler_block_size)
    coordinator.block_pool._tpu_keep_partial_entries = True
    logger.info(
        "TPU_FINE_PREFIX_HIT_TOKENS: prefix-cache hits land on %d-token "
        "boundaries (scheduler block %d, hash block %d)",
        unit,
        coordinator.scheduler_block_size,
        coordinator.hash_block_size,
    )


def find_full_attention_fine_hit(
    block_hashes,
    max_length: int,
    kv_cache_group_ids: list[int],
    block_pool: BlockPool,
    block_size: int,
    unit: int,
) -> tuple[tuple[list[KVCacheBlock], ...], int]:
    """Longest full-attention hit at ``unit`` granularity.

    The run of cached full blocks first, then the ``unit`` boundaries inside
    the next block, longest first. ``block_hashes`` are at the hash block
    granularity; the chained hash at a boundary identifies that prefix.
    """
    from vllm.v1.core.kv_cache_utils import BlockHashListWithBlockSize

    hash_bs = block_pool.hash_block_size
    computed: tuple[list[KVCacheBlock], ...] = tuple([] for _ in kv_cache_group_ids)
    full_hashes = BlockHashListWithBlockSize(block_hashes, hash_bs, block_size)
    for block_hash in islice(full_hashes, max_length // block_size):
        cached = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
        if not cached:
            break
        for blocks, block in zip(computed, cached):
            blocks.append(block)
    hit = len(computed[0]) * block_size
    top = min(hit + block_size - unit, max_length // unit * unit)
    for boundary in range(top, hit, -unit):
        hash_idx = boundary // hash_bs - 1
        if hash_idx >= len(block_hashes):
            continue
        cached = block_pool.get_cached_block(block_hashes[hash_idx], kv_cache_group_ids)
        if cached:
            for blocks, block in zip(computed, cached):
                blocks.append(block)
            hit = boundary
            break
    return computed, hit


def _cache_fine_prompt_tail(manager, request: Request, num_tokens: int) -> None:
    """Register the prompt-tail block under the request's fine boundary."""
    boundary = fine_boundary(
        request.num_prompt_tokens, manager.cache_hit_alignment_tokens
    )
    if boundary == 0 or boundary > num_tokens or boundary % manager.block_size == 0:
        return
    blocks = manager.req_to_blocks[request.request_id]
    block_idx = boundary // manager.block_size
    if block_idx >= len(blocks):
        return
    manager.block_pool.cache_partial_block(
        request=request,
        block=blocks[block_idx],
        num_tokens=boundary,
        kv_cache_group_id=manager.kv_cache_group_id,
        block_size=manager.block_size,
    )


def _is_fine(manager) -> bool:
    alignment = manager.cache_hit_alignment_tokens
    return (
        alignment < manager.block_size
        and alignment != manager.block_pool.hash_block_size
    )


def patch_fine_prefix_hits() -> None:
    unit = fine_prefix_hit_tokens()
    if unit == 0:
        return
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator

    if getattr(HybridKVCacheCoordinator, "_tpu_fine_prefix_patch", False):
        return
    install(unit, setattr)
    HybridKVCacheCoordinator._tpu_fine_prefix_patch = True
    logger.info("Applied TPU patch: fine prefix-cache hits (%d tokens).", unit)


def install(unit: int, set_attr: Callable[[object, str, object], None]) -> None:
    """Install the wrappers with ``set_attr`` (``setattr``, or a test's
    ``monkeypatch.setattr``)."""
    from vllm.utils.math_utils import round_down
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

    original_init = HybridKVCacheCoordinator.__init__

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _enable(self, unit)

    original_alignment = HybridKVCacheCoordinator._cache_hit_alignment_tokens

    def _cache_hit_alignment_tokens(self) -> int:
        fine = getattr(self, "_tpu_fine_prefix_unit", None)
        return fine if fine is not None else original_alignment.fget(self)

    original_align_cacheable = HybridKVCacheCoordinator._align_cacheable

    @functools.wraps(original_align_cacheable)
    def _align_cacheable(self, num_tokens: int) -> int:
        fine = getattr(self, "_tpu_fine_prefix_unit", None)
        if fine is None:
            return original_align_cacheable(self, num_tokens)
        return round_down(num_tokens, fine)

    original_find = FullAttentionManager.find_longest_cache_hit.__func__

    @functools.wraps(original_find)
    def find_longest_cache_hit(
        cls,
        block_hashes,
        max_length,
        kv_cache_group_ids,
        block_pool,
        kv_cache_spec,
        drop_eagle_block,
        alignment_tokens,
        dcp_world_size=1,
        pcp_world_size=1,
    ):
        block_size = kv_cache_spec.block_size
        if (
            alignment_tokens >= block_size
            or alignment_tokens == block_pool.hash_block_size
        ):
            return original_find(
                cls,
                block_hashes,
                max_length,
                kv_cache_group_ids,
                block_pool,
                kv_cache_spec,
                drop_eagle_block,
                alignment_tokens,
                dcp_world_size,
                pcp_world_size,
            )
        if drop_eagle_block:
            raise ValueError(
                "TPU_FINE_PREFIX_HIT_TOKENS needs the EAGLE/MTP block drop off "
                "(VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=1)"
            )
        if dcp_world_size != 1 or pcp_world_size != 1:
            raise ValueError("TPU_FINE_PREFIX_HIT_TOKENS does not support DCP/PCP")
        return find_full_attention_fine_hit(
            block_hashes,
            max_length,
            kv_cache_group_ids,
            block_pool,
            block_size,
            alignment_tokens,
        )

    original_tail = FullAttentionManager._cache_partial_tail_block

    @functools.wraps(original_tail)
    def _cache_partial_tail_block(self, request, num_tokens):
        if _is_fine(self):
            _cache_fine_prompt_tail(self, request, num_tokens)
        else:
            original_tail(self, request, num_tokens)

    original_cache_full = BlockPool.cache_full_blocks

    @functools.wraps(original_cache_full)
    def cache_full_blocks(
        self, request, blocks, num_cached_blocks, num_full_blocks, *args, **kwargs
    ):
        if not getattr(self, "_tpu_keep_partial_entries", False):
            return original_cache_full(
                self,
                request,
                blocks,
                num_cached_blocks,
                num_full_blocks,
                *args,
                **kwargs,
            )
        # A partial entry names a prefix that ends inside the block; promoting
        # the block to a full entry leaves those rows untouched, so keep it.
        partial = [
            (
                blk,
                [
                    blk.block_hash,
                    *self.cached_block_hashes_by_block.get(blk.block_id, ()),
                ],
            )
            for blk in blocks[num_cached_blocks:num_full_blocks]
            if not blk.is_null and blk.block_hash is not None
        ]
        original_cache_full(
            self, request, blocks, num_cached_blocks, num_full_blocks, *args, **kwargs
        )
        for blk, hashes in partial:
            if blk.block_hash is None:
                continue
            for block_hash in hashes:
                self._insert_block_hash(block_hash, blk, num_tokens=None)

    set_attr(HybridKVCacheCoordinator, "__init__", __init__)
    set_attr(
        HybridKVCacheCoordinator,
        "_cache_hit_alignment_tokens",
        property(_cache_hit_alignment_tokens),
    )
    set_attr(HybridKVCacheCoordinator, "_align_cacheable", _align_cacheable)
    set_attr(
        FullAttentionManager,
        "find_longest_cache_hit",
        classmethod(find_longest_cache_hit),
    )
    set_attr(
        FullAttentionManager, "_cache_partial_tail_block", _cache_partial_tail_block
    )
    set_attr(BlockPool, "cache_full_blocks", cache_full_blocks)
