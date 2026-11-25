# SPDX-License-Identifier: Apache-2.0
"""q38 F2-HOLD: an F2 producer holds MTP draft blocks only above each prefix hit.

Under q38 F2 (``QWEN4_EXP_PD_DRAFT_KV=1``) a TP1 pipeline producer runs the
MTP draft over its prefill chunks and ships the draft layer's KV. Its draft
specs are not prefix-cacheable (option (a), ``config.non_prefix_cacheable``),
so on a prefix hit of ``Hp`` tokens the draft writes rows only from ``Hp`` on,
and a staged decode zero-fills the draft rows below ``Hp`` it does not hold
itself (``pd_staging._DraftGap``). vLLM still gave the draft fresh blocks for
the whole prompt: ``2 * cdiv(P, 3200) + 1`` pool IDs per running request
(main KV, compressed keys, a ring block), held until the decode's pull ends,
of which the ``2 * floor(Hp / 3200)`` below the hit are never written
(CACHE-2, cache_hits.md section 5.18).

With ``QWEN4_EXP_PD_DRAFT_HOLD=1`` the draft's main-KV and compressed-key
groups use ``ProducerDraftHoldManager``: the blocks wholly inside the
request's computed prefix at its first allocation (local plus external hit)
are vLLM's null block, through the skipped-block path sliding-window
attention already uses (``get_num_skipped_tokens``), and the admission gate
counts only the blocks above it. The ring keeps its one block. Nothing is
freed later: a written draft block stays until the request is freed.

The send side (``pd_connector``) registers only the live pages of such a
group and reads the null positions' rows from a live page; the decode
zero-fills every draft row below the published hit, so its landed bytes are
the same as without the hold.
"""

# The plugin's logger (vllm_torchtpu.logger -> 'vllm.<module>'): vllm.logger.init_logger(__name__) names a logger
# outside vLLM's 'vllm' tree, whose handler never emits it (LOG-AUDIT 2026-11-10). F2-HOLD gate H / HD
# (j-d7417a2d / j-6621454c) printed 0 registration lines for that reason.
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_PATCH_FLAG = "_q38_producer_draft_hold_registered"


def _manager_class():
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

    class ProducerDraftHoldManager(FullAttentionManager):
        """``FullAttentionManager`` whose request tables skip the computed prefix
        at first allocation (null blocks) and never free a block afterwards.

        ``get_num_skipped_tokens`` is read only where a request first gets
        blocks (``get_num_blocks_to_allocate`` before the request is tracked,
        ``add_local_computed_blocks``, ``allocate_external_computed_blocks``);
        ``remove_skipped_blocks``, the one per-step reader, is a no-op here, so
        a block the draft wrote above the hit is held until the request ends.
        The group is not prefix-cacheable, so no hit ever adopts its blocks."""

        def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
            return max(0, int(num_computed_tokens))

        def remove_skipped_blocks(
            self,
            request_id: str,
            processed_computed_tokens: int,
            num_prompt_tokens: int | None = None,
        ) -> None:
            return None

        def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
            # Per-request scratch: never shared (and the null block's ref count
            # means nothing).
            return 0

    ProducerDraftHoldManager.__module__ = __name__
    ProducerDraftHoldManager.__qualname__ = "ProducerDraftHoldManager"
    return ProducerDraftHoldManager


_MANAGER = None


def producer_draft_hold_manager():
    """The manager class (created once, after vLLM is importable)."""
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = _manager_class()
    return _MANAGER


def held_spec_types() -> list[type]:
    """The producer draft's non-prefix-cacheable token-page spec types: the
    subclasses ``config.non_prefix_cacheable`` makes of ``FullAttentionSpec``
    (main KV) and ``MLAAttentionSpec`` (compressed keys). Not the ring's
    (``CircularBufferSpec`` keeps its one block)."""
    from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec

    from vllm_torchtpu.models.vllm.qwen4_exp import config as qconfig

    return [
        sub
        for sub in qconfig._NON_PREFIX_CACHEABLE.values()
        if issubclass(sub, FullAttentionSpec) and not issubclass(sub, CircularBufferSpec)
    ]


def register_producer_draft_hold(force: bool = False) -> bool:
    """Register ``ProducerDraftHoldManager`` for the held spec types when
    ``QWEN4_EXP_PD_DRAFT_HOLD`` and ``QWEN4_EXP_PD_DRAFT_KV`` are both on (or
    ``force``). Idempotent. Returns whether the manager is registered.

    vLLM fills its spec registry lazily, on the first lookup, only while the
    registry is empty, so the built-in specs are registered first here."""
    from vllm_torchtpu import envs

    if not force and not (envs.QWEN4_EXP_PD_DRAFT_HOLD and envs.QWEN4_EXP_PD_DRAFT_KV):
        return False
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    KVCacheSpecRegistry._ensure_registered()
    manager = producer_draft_hold_manager()
    types = held_spec_types()
    if not types:
        # config.non_prefix_cacheable took the field path (dataclasses.replace):
        # the draft's specs keep their base types, so no manager can be chosen
        # by type and the hold would silently do nothing.
        raise ValueError(
            "QWEN4_EXP_PD_DRAFT_HOLD=1 needs the producer draft's non-prefix-cacheable "
            "spec subclasses (config._NON_PREFIX_CACHEABLE is empty in this vLLM)"
        )
    for spec_cls in types:
        KVCacheSpecRegistry.register(
            spec_cls, manager, uniform_type_base_spec=FullAttentionSpec
        )
    if not getattr(KVCacheSpecRegistry, _PATCH_FLAG, False):
        setattr(KVCacheSpecRegistry, _PATCH_FLAG, True)
        logger.info(
            "q38 F2-HOLD: the producer's MTP draft holds blocks only above each "
            "prefix hit (%s -> %s)",
            [t.__name__ for t in types],
            manager.__name__,
        )
    return True


def unregister_producer_draft_hold() -> None:
    """Tests: drop the held spec types from vLLM's registry again (they fall
    back to their bases' managers through the registry's MRO walk)."""
    from vllm.v1.kv_cache_spec_registry import (
        _REGISTRY_KVCACHESPEC_LIST,
        KVCacheSpecRegistry,
    )

    for spec_cls in held_spec_types():
        _REGISTRY_KVCACHESPEC_LIST.pop(spec_cls, None)
    if getattr(KVCacheSpecRegistry, _PATCH_FLAG, False):
        delattr(KVCacheSpecRegistry, _PATCH_FLAG)


def patch_producer_draft_hold() -> None:
    """Engine-core / worker patch (patch_registry): see
    ``register_producer_draft_hold``. A no-op with the flags off."""
    register_producer_draft_hold()
