# SPDX-License-Identifier: Apache-2.0
"""Connector-backed local prefix hits for hybrids with several token groups.

When a KV connector restores recurrent state with every load
(``supports_divergent_local_hybrid_hits``), the pinned vLLM release reports
the local hit of one *reference* full-attention group
(``KVCacheManager.get_computed_blocks_for_connector``) and lets the
connector supply the rest. That is right for a model with one attention
group. Qwen4Exp has many: one main-KV group per QSA layer (2048-token
entries under main-KV DCP4), packed compressed-key groups and the MTP
draft's main KV. Eviction frees them position by position, so they can stop
at different depths; a lagging group would then get fresh, never-written
blocks for the gap between its own hit and the reference's, because the
connector's suffix starts at the reference hit. Wrong KV, not a crash.

This patch, for hybrids with more than one prefix-cacheable token group:

- takes the local hit as the shortest per-group hit over every
  prefix-cacheable non-Mamba group (each is downward-closed and aligned to
  the scheduler block, so the minimum is a boundary every one of them holds);
- adopts no recurrent-state blocks unless every state group holds a state
  exactly at that boundary (then the hit is convergent and reported as
  such); otherwise the connector's load supplies the state;
- keeps the hit strictly below the connector's published extent
  (``kv_transfer_params["num_tokens"]``): a divergent hit must leave the
  connector at least one token to load, since the scheduler falls back to a
  state-reconciled lookup when it reports none.

Models with a single token group keep the upstream behaviour.
"""

from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCH_FLAG = "_tpu_divergent_hybrid_hits_patch"


def _token_groups(kv_cache_config) -> list[int]:
    from vllm.v1.kv_cache_interface import MambaSpec

    return [
        gid
        for gid, group in enumerate(kv_cache_config.kv_cache_groups)
        if group.kv_cache_spec.prefix_cacheable
        and not isinstance(group.kv_cache_spec, MambaSpec)
    ]


def _transfer_extent(request) -> int | None:
    params = getattr(request, "kv_transfer_params", None)
    if not isinstance(params, dict):
        return None
    extent = params.get("num_tokens")
    if type(extent) is not int or extent <= 0:
        return None
    return extent


def divergent_local_hit(manager, request):
    """``get_computed_blocks_for_connector`` over every token group.

    Returns ``(blocks, num_local_tokens, shared_prefix_boundary, diverged)``.
    """
    from vllm.v1.kv_cache_interface import MambaSpec
    from vllm.utils.math_utils import cdiv

    coordinator = manager.coordinator
    config = manager.kv_cache_config
    if not manager.prefix_cache_lookup_enabled(request):
        return manager.empty_kv_cache_blocks, 0, 0, False
    max_length = request.num_tokens - 1
    extent = _transfer_extent(request)
    if extent is not None:
        max_length = min(max_length, extent - 1)
    if max_length <= 0:
        return manager.empty_kv_cache_blocks, 0, 0, False
    computed, per_group = coordinator.find_longest_cache_hit_per_group(
        request.block_hashes, max_length
    )
    token_groups = _token_groups(config)
    hit = min(per_group[gid] for gid in token_groups)
    if hit <= 0:
        return manager.empty_kv_cache_blocks, 0, 0, False
    blocks = []
    diverged = False
    managers = coordinator.single_type_managers
    for gid, group in enumerate(config.kv_cache_groups):
        group_blocks = list(computed[gid])
        spec = group.kv_cache_spec
        if not spec.prefix_cacheable:
            group_blocks = []
        elif isinstance(spec, MambaSpec):
            if per_group[gid] != hit:
                # No state saved at the boundary (or one past it): adopt none
                # and let the load supply the committed state.
                diverged = True
                group_blocks = []
        else:
            del group_blocks[cdiv(hit, managers[gid].block_size) :]
        blocks.append(group_blocks)
    return manager.create_kv_cache_blocks(tuple(blocks)), hit, 0, diverged


def patch_divergent_hybrid_hits() -> None:
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    original = KVCacheManager.get_computed_blocks_for_connector
    if getattr(original, _PATCH_FLAG, False):
        return

    def get_computed_blocks_for_connector(self, request):
        if not (
            self.kv_cache_config.has_mamba_layers
            and isinstance(self.coordinator, HybridKVCacheCoordinator)
            and len(_token_groups(self.kv_cache_config)) > 1
        ):
            return original(self, request)
        return divergent_local_hit(self, request)

    setattr(get_computed_blocks_for_connector, _PATCH_FLAG, True)
    KVCacheManager.get_computed_blocks_for_connector = get_computed_blocks_for_connector
    logger.info(
        "Applied TPU patch: connector-backed hybrid prefix hits take the "
        "shortest hit over every token group."
    )
