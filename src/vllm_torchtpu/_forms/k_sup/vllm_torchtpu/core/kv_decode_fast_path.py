# SPDX-License-Identifier: Apache-2.0
"""Skips the per-group KV bookkeeping of steady decode steps (engine core).

Every scheduled request costs ``KVCacheManager.allocate_slots`` once per step
and, under async scheduling, ``KVCacheManager.cache_blocks`` once more when
its output arrives. Both walk every KV cache group in Python: four calls per
group in ``allocate_slots`` (``remove_skipped_blocks``,
``get_num_blocks_to_allocate``, ``allocate_new_blocks``, ``cache_blocks``),
one in ``cache_blocks``. Qwen3.8-Flash's decode has 65 groups, about 0.4 ms
per running request per step on the engine core's main thread.

In a steady decode step none of those calls changes anything: a request
needs a new block only when its tokens cross a block boundary (512 tokens
for the state groups, 2048 for the main KV under main-KV DCP4), caches new
blocks only when the cacheable length crosses a scheduler block, and frees
a state block only right after a state boundary. This patch keeps, per
request, the token thresholds below which every group's call is a no-op,
derived from the managers' own state after each call that runs the
original code. A call whose inputs stay below them returns what the
original returns without walking the groups. Any other call (an
admission, a prefix hit, a connector load, a new block, a block to free or
cache, a request in the waiting path) runs the original code and
recomputes the thresholds from the resulting state.

The one per-step effect of a steady step is the prompt-tail registration
of full-attention groups whose block is larger than the hash block
(``FullAttentionManager._cache_partial_tail_block``): it re-inserts an
already-cached partial-block hash, which is a no-op, but the same-step
prefix-hit patch (``vllm_torchtpu._patch_vllm_same_step_prefix_hits``)
marks the block unwritten for the current schedule pass on every insert.
The fast path checks that each such hash is still registered (otherwise it
runs the original code) and marks the same blocks.

Per-request state changes outside these two methods go through
``KVCacheManager.free``, ``pop_blocks_for_free`` and
``remove_skipped_blocks``, which drop the request's thresholds.

The thresholds encode the behaviour of specific vLLM and patch functions.
The patch installs only when those functions are unwrapped and their
sources match the ones the thresholds were derived from
(``_FINGERPRINTS``), and a manager uses it only when every group's manager
is a full-attention, Mamba "align" or circular-buffer manager of the
analysed kind. ``VLLM_TPU_KV_DECODE_FAST_PATH=1`` enables it.
"""

import hashlib
import inspect
import textwrap
from functools import wraps

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_PATCH_FLAG = "_tpu_kv_decode_fast_path_patch"
# Per-KVCacheManager state, in the instance __dict__; None turns the fast
# path off for that manager.
_STATE = "_tpu_kv_decode_fast_path"

# sha1 prefix of the dedented source of each function whose behaviour the
# thresholds encode (vLLM v0.31.0 and this package's patches). A mismatch
# means the analysis no longer applies and the patch stays off.
_FINGERPRINTS = {
    "KVCacheManager.allocate_slots": "d36a13bd1cf5471e",
    "KVCacheManager.cache_blocks": "ebb8c1092085013f",
    "KVCacheManager.create_kv_cache_blocks": "633061f5cae4c65e",
    "KVCacheCoordinator.remove_skipped_blocks": "d78e8a3969c1cf77",
    "KVCacheCoordinator.get_num_blocks_to_allocate": "b7490e90f02dc880",
    "KVCacheCoordinator.allocate_new_blocks": "364f87f493d232db",
    "KVCacheCoordinator.get_replay_boundaries": "4e1fdce4df44225f",
    "HybridKVCacheCoordinator.cache_blocks": "3563f1f8ecbd7644",
    "HybridKVCacheCoordinator._align_cacheable": "54157ea5a62cfecc",
    "SingleTypeKVCacheManager.remove_skipped_blocks": "25ff5ec76473032b",
    "SingleTypeKVCacheManager.get_num_skipped_tokens": "6b1d814fded86f33",
    "SingleTypeKVCacheManager.get_num_blocks_to_allocate": "9c39c1f31a8817fa",
    "SingleTypeKVCacheManager.allocate_new_blocks": "fcb7e541e28fb58a",
    "SingleTypeKVCacheManager.cache_blocks": "ed4aaada8adb4e78",
    "SingleTypeKVCacheManager._has_partial_local_hit": "a45c5ab8bf71a01d",
    "FullAttentionManager.cache_blocks": "3e8348f4ff3e097e",
    "FullAttentionManager._cache_partial_tail_block": "107e88157e024d6b",
    "MambaManager.get_num_blocks_to_allocate": "583f823d232aae7a",
    "MambaManager.allocate_new_blocks": "947dd8dac0ac5426",
    "MambaManager._needs_internal_checkpoint": "c31248eb2b23eabd",
    "MambaManager.get_num_skipped_tokens": "7cff17654837cbdb",
    "MambaManager.cache_blocks": "64b3585709cc461b",
    "MambaManager._cache_partial_tail_block": "b79d0680aacba861",
    "MambaManager.remove_skipped_blocks": "a0c2ae7d6c0e4281",
    "MambaManager._remove_blocks_in_range": "bbafa977676fcfd0",
    "CircularBufferManager.get_num_blocks_to_allocate": "74e496df3c9d52e3",
    "CircularBufferManager.allocate_new_blocks": "198e4ee19f2f4df9",
    "CircularBufferManager._claim_ring_block": "37f87ec153de3f2a",
    "CircularBufferManager.cache_blocks": "8918735a49dde702",
    "CircularBufferManager.remove_skipped_blocks": "af1ff4ac7a454c72",
    "BlockPool.cache_partial_block": "8cea600dafafb8f9",
    "BlockPool._insert_block_hash": "c53c2bfe9a8032ea",
    "KVCacheBlock.block_hash": "fc06403d66c7b6bc",
    "retention.cache_blocks": "91c2574fe76115a6",
    "retention.remove_skipped_blocks": "ad815d82a0a5e146",
    "retention._remove_blocks_in_range": "617327e7f0da7c33",
    "same_step._insert_block_hash": "4b6163fb4882879b",
}

_INF = float("inf")


def _source_hash(fn) -> str:
    if isinstance(fn, property):
        fn = fn.fget
    # The code object: inspect.getsource(fn) would follow __wrapped__.
    source = textwrap.dedent(inspect.getsource(fn.__code__))
    return hashlib.sha1(source.encode()).hexdigest()[:16]


def _analysed_functions() -> dict:
    """Resolves the functions named in ``_FINGERPRINTS`` as installed now.

    Patched methods are listed under the patch's name; their ``__wrapped__``
    vLLM originals are listed under the class name. A missing one is None.
    """
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_coordinator import (
        HybridKVCacheCoordinator,
        KVCacheCoordinator,
    )
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.core.single_type_kv_cache_manager import (
        CircularBufferManager,
        FullAttentionManager,
        MambaManager,
        SingleTypeKVCacheManager,
    )

    owners = {
        "KVCacheManager": KVCacheManager,
        "KVCacheCoordinator": KVCacheCoordinator,
        "HybridKVCacheCoordinator": HybridKVCacheCoordinator,
        "SingleTypeKVCacheManager": SingleTypeKVCacheManager,
        "FullAttentionManager": FullAttentionManager,
        "MambaManager": MambaManager,
        "CircularBufferManager": CircularBufferManager,
        "BlockPool": BlockPool,
        "KVCacheBlock": KVCacheBlock,
    }
    retention_flag = "_tpu_mamba_checkpoint_retention_patch"
    found = {}
    for name in _FINGERPRINTS:
        owner, _, attr = name.partition(".")
        if owner in ("retention", "same_step"):
            continue
        fn = owners[owner].__dict__.get(attr)
        if fn is None:
            found[name] = None
            continue
        if owner == "MambaManager" and getattr(fn, retention_flag, False):
            found["retention." + attr] = fn
            fn = fn.__wrapped__
        if owner == "BlockPool" and getattr(
            fn, "_tpu_same_step_prefix_hit_patch", False
        ):
            found["same_step." + attr] = fn
            fn = fn.__wrapped__
        found[name] = fn
    return found


def check_fingerprints() -> list[str]:
    """Names of analysed functions whose installed source differs."""
    return [
        name
        for name, fn in _analysed_functions().items()
        if fn is None or _source_hash(fn) != _FINGERPRINTS[name]
    ]


class _Groups:
    """The coordinator's managers split by kind, and fixed geometry."""

    __slots__ = (
        "attention",
        "state",
        "rings",
        "hash_block_size",
        "scheduler_block_size",
        "partial_hits",
        "pool",
        "tail_mode",
        "retention",
        "plans",
    )


def _build_state(kv_cache_manager):
    """Returns the fast-path state for a manager, or None if unsupported."""
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
    from vllm.v1.core.single_type_kv_cache_manager import (
        CircularBufferManager,
        FullAttentionManager,
        MambaManager,
    )

    coordinator = kv_cache_manager.coordinator
    reason = None
    if type(coordinator) is not HybridKVCacheCoordinator:
        reason = f"coordinator {type(coordinator).__name__}"
    elif not kv_cache_manager.enable_caching:
        reason = "prefix caching off"
    groups = _Groups()
    groups.attention, groups.state, groups.rings = [], [], []
    for manager in coordinator.single_type_managers if reason is None else ():
        kind = type(manager)
        if manager.use_eagle:
            reason = "EAGLE block drop"
        elif kind is FullAttentionManager:
            groups.attention.append(manager)
        elif kind is MambaManager:
            if manager.mamba_cache_mode != "align":
                reason = f"mamba_cache_mode {manager.mamba_cache_mode}"
            elif manager.has_prefill_checkpoint_blocks:
                reason = "Mamba prefill checkpoint blocks"
            elif manager.block_size != coordinator.hash_block_size:
                reason = "Mamba block larger than the hash block"
            groups.state.append(manager)
        elif kind is CircularBufferManager:
            groups.rings.append(manager)
        else:
            reason = f"manager {kind.__name__}"
        if reason is not None:
            break
    if reason is not None:
        logger.info_once("KV decode fast path off for this engine: %s.", reason)
        return None
    pool = coordinator.block_pool
    groups.hash_block_size = coordinator.hash_block_size
    groups.scheduler_block_size = coordinator.scheduler_block_size
    groups.partial_hits = coordinator.enable_partial_hash_hits
    groups.pool = pool
    insert = type(pool)._insert_block_hash
    if getattr(insert, "_tpu_same_step_prefix_hit_patch", False):
        groups.tail_mode = "unwritten"
    elif getattr(insert, "__wrapped__", None) is None:
        groups.tail_mode = "none"
    else:
        groups.tail_mode = "call"
    groups.retention = getattr(
        MambaManager.cache_blocks, "_tpu_mamba_checkpoint_retention_patch", False
    )
    groups.plans = {}
    return groups


def _state(kv_cache_manager):
    state = kv_cache_manager.__dict__.get(_STATE, False)
    if state is False:
        state = _build_state(kv_cache_manager)
        kv_cache_manager.__dict__[_STATE] = state
    return state


def _cache_limit(groups: _Groups, num_cached_blocks: int, block_size: int) -> int:
    """Smallest cacheable length that caches a block past ``num_cached_blocks``."""
    limit = (num_cached_blocks + 1) * block_size
    if not groups.partial_hits:
        # HybridKVCacheCoordinator._align_cacheable rounds down to a
        # scheduler block, so only multiples of it are reached.
        step = groups.scheduler_block_size
        limit = -(-limit // step) * step
    return limit


def _plan(groups: _Groups, request):
    """Token thresholds below which every group's step call is a no-op.

    Returns ``(request, num_prompt_tokens, slot_limit, main_limit,
    skip_limit, cache_limit, tail_start, tails)``, or None when some group's
    next call may change state at any length. Allocation is a no-op while
    the slot count (target tokens plus lookahead) is at most ``slot_limit``
    and the target-model token count at most ``main_limit``; freeing while
    the processed-token count is at most ``skip_limit``; caching while the
    cacheable length is below ``cache_limit``. ``tails`` are the prompt-tail
    entries ``(hash key, block, block id)`` re-registered once the aligned
    cacheable length reaches ``tail_start`` (0: no tail); None when a tail's
    hash is not computed yet, which sends those calls to the original code.
    """
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

    request_id = request.request_id
    hash_block_size = groups.hash_block_size
    slot_limit = main_limit = skip_limit = cache_limit = _INF
    boundary = request.num_prompt_tokens // hash_block_size * hash_block_size
    tails = []
    tail_start = 0
    tails_known = True
    for manager in groups.attention:
        blocks = manager.req_to_blocks.get(request_id)
        if (
            not blocks
            or request_id in manager._partial_hit_reqs
            or request_id not in manager.num_cached_block
        ):
            return None
        block_size = manager.block_size
        slot_limit = min(slot_limit, len(blocks) * block_size)
        if manager.kv_cache_spec.prefix_cacheable:
            cache_limit = min(
                cache_limit,
                _cache_limit(groups, manager.num_cached_block[request_id], block_size),
            )
        if (
            block_size != hash_block_size
            and boundary > 0
            and boundary % block_size != 0
            and boundary // block_size < len(blocks)
        ):
            block = blocks[boundary // block_size]
            if block.is_null:
                continue
            tail_start = boundary
            num_hash_blocks = boundary // hash_block_size
            if num_hash_blocks > len(request.block_hashes):
                tails_known = False
                continue
            key = make_block_hash_with_group_id(
                request.block_hashes[num_hash_blocks - 1], manager.kv_cache_group_id
            )
            tails.append((key, block, block.block_id))
    for manager in groups.state:
        blocks = manager.req_to_blocks.get(request_id)
        if (
            not blocks
            or request_id in manager._partial_hit_reqs
            or request_id in manager._checkpoints
            or request_id not in manager._allocated_block_reqs
        ):
            return None
        if groups.retention and request_id not in manager.__dict__.get(
            "_tpu_own_checkpoints", ()
        ):
            return None
        block_size = manager.block_size
        main_limit = min(
            main_limit, (len(blocks) - manager.num_speculative_blocks) * block_size
        )
        retired = manager._num_retired_blocks.get(request_id, 0)
        if len(blocks) > retired:
            skip_limit = min(skip_limit, (retired + 1) * block_size)
        last = manager.last_state_block_idx.get(request_id)
        if last is not None:
            if last >= len(blocks):
                return None
            if not blocks[last].is_null:
                skip_limit = min(skip_limit, (last + 1) * block_size)
        cache_limit = min(
            cache_limit,
            _cache_limit(
                groups, manager.num_cached_block.get(request_id, 0), block_size
            ),
        )
    for manager in groups.rings:
        if not manager.req_to_blocks.get(request_id):
            return None
    return (
        request,
        request.num_prompt_tokens,
        slot_limit,
        main_limit,
        skip_limit,
        cache_limit,
        tail_start,
        tuple(tails) if tails_known else None,
    )


def _register_tails(groups: _Groups, tail_start: int, tails, num_tokens: int) -> bool:
    """Replays the steady prompt-tail registration; False if it is not steady."""
    if not groups.partial_hits:
        step = groups.scheduler_block_size
        num_tokens = num_tokens // step * step
    if tail_start == 0 or num_tokens < tail_start:
        return True
    if tails is None:
        return False
    contain = groups.pool.cached_block_hash_to_block.contain
    for key, block, block_id in tails:
        if block.block_hash != key and not contain(key, block_id):
            return False
    mode = groups.tail_mode
    if mode == "unwritten":
        groups.pool.__dict__.setdefault("_tpu_unwritten_block_ids", set()).update(
            block_id for _, _, block_id in tails
        )
    elif mode == "call":
        insert = groups.pool._insert_block_hash
        for key, block, _ in tails:
            insert(key, block, num_tokens=tail_start)
    return True


def _refresh(groups: _Groups, request) -> None:
    plan = _plan(groups, request)
    if plan is None:
        groups.plans.pop(request.request_id, None)
    else:
        groups.plans[request.request_id] = plan


def patch_kv_decode_fast_path() -> None:
    if not envs.VLLM_TPU_KV_DECODE_FAST_PATH:
        return
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import RequestStatus

    if getattr(KVCacheManager.allocate_slots, _PATCH_FLAG, False):
        return
    for name in ("allocate_slots", "cache_blocks", "free", "pop_blocks_for_free"):
        if hasattr(KVCacheManager.__dict__[name], "__wrapped__"):
            logger.warning(
                "KV decode fast path not installed: KVCacheManager.%s is wrapped.",
                name,
            )
            return
    mismatched = check_fingerprints()
    if mismatched:
        logger.warning(
            "KV decode fast path not installed: these functions differ from "
            "the analysed sources: %s.",
            ", ".join(mismatched),
        )
        return

    original_allocate = KVCacheManager.allocate_slots
    original_cache = KVCacheManager.cache_blocks
    original_free = KVCacheManager.free
    original_pop = KVCacheManager.pop_blocks_for_free
    original_remove = KVCacheManager.remove_skipped_blocks
    running = RequestStatus.RUNNING

    @wraps(original_allocate)
    def allocate_slots(
        self,
        request,
        num_new_tokens,
        num_new_computed_tokens=0,
        new_computed_blocks=None,
        num_lookahead_tokens=0,
        num_external_computed_tokens=0,
        delay_cache_blocks=False,
        num_encoder_tokens=0,
        full_sequence_must_fit=False,
        reserved_blocks=0,
        has_scheduled_reqs=True,
    ):
        groups = _state(self)
        if groups is None:
            return original_allocate(
                self,
                request,
                num_new_tokens,
                num_new_computed_tokens,
                new_computed_blocks,
                num_lookahead_tokens,
                num_external_computed_tokens,
                delay_cache_blocks,
                num_encoder_tokens,
                full_sequence_must_fit,
                reserved_blocks,
                has_scheduled_reqs,
            )
        plan = groups.plans.pop(request.request_id, None)
        if (
            plan is not None
            and new_computed_blocks is None
            and num_new_tokens > 0
            and not (
                num_new_computed_tokens
                or num_external_computed_tokens
                or delay_cache_blocks
                or num_encoder_tokens
                or full_sequence_must_fit
                or reserved_blocks
            )
            and request.status == running
            and plan[0] is request
            and plan[1] == request.num_prompt_tokens
        ):
            total = min(request.num_computed_tokens, self.max_model_len)
            main = total + num_new_tokens
            if (
                min(main + num_lookahead_tokens, self.max_model_len) <= plan[2]
                and main <= plan[3]
                and total - request.num_in_flight_tokens <= plan[4]
            ):
                num_tokens = min(main, request.num_tokens)
                if num_tokens < plan[5] and _register_tails(
                    groups, plan[6], plan[7], num_tokens
                ):
                    groups.plans[request.request_id] = plan
                    return self.empty_kv_cache_blocks
        blocks = original_allocate(
            self,
            request,
            num_new_tokens,
            num_new_computed_tokens,
            new_computed_blocks,
            num_lookahead_tokens,
            num_external_computed_tokens,
            delay_cache_blocks,
            num_encoder_tokens,
            full_sequence_must_fit,
            reserved_blocks,
            has_scheduled_reqs,
        )
        if blocks is not None:
            _refresh(groups, request)
        return blocks

    @wraps(original_cache)
    def cache_blocks(self, request, num_computed_tokens):
        groups = _state(self)
        if groups is None:
            return original_cache(self, request, num_computed_tokens)
        plan = groups.plans.pop(request.request_id, None)
        if (
            plan is not None
            and plan[0] is request
            and plan[1] == request.num_prompt_tokens
            and num_computed_tokens < plan[5]
            and _register_tails(groups, plan[6], plan[7], num_computed_tokens)
        ):
            groups.plans[request.request_id] = plan
            return None
        original_cache(self, request, num_computed_tokens)
        _refresh(groups, request)
        return None

    @wraps(original_free)
    def free(self, request):
        groups = self.__dict__.get(_STATE)
        if groups:
            groups.plans.pop(request.request_id, None)
        return original_free(self, request)

    @wraps(original_pop)
    def pop_blocks_for_free(self, request):
        groups = self.__dict__.get(_STATE)
        if groups:
            groups.plans.pop(request.request_id, None)
        return original_pop(self, request)

    @wraps(original_remove)
    def remove_skipped_blocks(
        self, request_id, processed_computed_tokens, num_prompt_tokens=None
    ):
        groups = self.__dict__.get(_STATE)
        if groups:
            groups.plans.pop(request_id, None)
        return original_remove(
            self, request_id, processed_computed_tokens, num_prompt_tokens
        )

    for fn in (allocate_slots, cache_blocks, free, pop_blocks_for_free):
        setattr(fn, _PATCH_FLAG, True)
    setattr(remove_skipped_blocks, _PATCH_FLAG, True)
    KVCacheManager.allocate_slots = allocate_slots
    KVCacheManager.cache_blocks = cache_blocks
    KVCacheManager.free = free
    KVCacheManager.pop_blocks_for_free = pop_blocks_for_free
    KVCacheManager.remove_skipped_blocks = remove_skipped_blocks
    logger.info(
        "Applied TPU patch: steady decode steps skip the per-group KV cache "
        "bookkeeping (VLLM_TPU_KV_DECODE_FAST_PATH)."
    )
