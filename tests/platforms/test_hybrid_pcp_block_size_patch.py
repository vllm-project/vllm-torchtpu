from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, MambaSpec,
                                        SlidingWindowSpec)

from vllm_torchtpu import _patch_vllm_hybrid_pcp_block_sizes


def _config(*,
            pcp: int,
            enable_prefix_caching: bool = False,
            connector_enabled: bool = False):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=512,
            enable_prefix_caching=enable_prefix_caching,
            prefix_match_unit=None,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=pcp,
        ),
        kv_transfer_config=object() if connector_enabled else None,
    )


def _full_spec(block_size: int = 512):
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.bfloat16,
    )


def _mamba_spec(block_size: int = 4096):
    return MambaSpec(
        block_size=block_size,
        shapes=((3, 8192), ),
        dtypes=(torch.bfloat16, ),
        mamba_cache_mode="align",
    )


def _resolve_outcome(resolver, kv_cache_config, vllm_config):
    try:
        return "return", resolver(kv_cache_config, vllm_config)
    except ValueError as error:
        return "value_error", str(error)


def test_hybrid_full_attention_mamba_pcp_resolves_effective_block_size():
    _patch_vllm_hybrid_pcp_block_sizes()
    from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
    from vllm.v1.engine import core as engine_core

    assert engine_core.resolve_kv_cache_block_sizes is resolve_kv_cache_block_sizes

    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], _full_spec()),
            KVCacheGroupSpec(["gdn"], _mamba_spec()),
        ],
    )

    assert resolve_kv_cache_block_sizes(kv_cache_config,
                                        _config(pcp=8)) == (32768, 32768)
    assert resolve_kv_cache_block_sizes(
        kv_cache_config, _config(pcp=8,
                                 enable_prefix_caching=True)) == (32768, 4096)


def test_hybrid_full_attention_mamba_pcp_builds_prefix_cache_coordinator():
    _patch_vllm_hybrid_pcp_block_sizes()
    from vllm.v1.core.kv_cache_coordinator import (HybridKVCacheCoordinator,
                                                   get_kv_cache_coordinator)
    from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], _full_spec(block_size=4096)),
            KVCacheGroupSpec(["gdn"], _mamba_spec()),
        ],
    )
    vllm_config = _config(pcp=8,
                          enable_prefix_caching=True,
                          connector_enabled=True)
    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config)

    assert (scheduler_block_size, hash_block_size) == (32768, 32768)
    coordinator = get_kv_cache_coordinator(
        kv_cache_config=kv_cache_config,
        max_model_len=65536,
        max_in_flight_tokens=4096,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=8,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
    )

    assert isinstance(coordinator, HybridKVCacheCoordinator)
    assert [
        group.kv_cache_spec.block_size
        for group in coordinator.kv_cache_config.kv_cache_groups
    ] == [32768, 32768]
    assert [
        manager.block_size for manager in coordinator.single_type_managers
    ] == [32768, 32768]
    # PCP is folded into the group specs, so the coordinator runs unaware of
    # it (it asserts pcp_world_size == 1) and never multiplies the block
    # sizes a second time. The caller's config is left untouched, i.e. still
    # holds the physical, rank-local page sizes.
    assert [
        group.kv_cache_spec.block_size
        for group in kv_cache_config.kv_cache_groups
    ] == [4096, 4096]


def test_hybrid_pcp_reuses_externally_materialized_prefix():
    """A synchronous connector fill must become a later local cache hit."""
    _patch_vllm_hybrid_pcp_block_sizes()
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import BlockHash

    kv_cache_config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], _full_spec(block_size=4096)),
            KVCacheGroupSpec(["gdn"], _mamba_spec()),
        ],
    )
    cache_manager = KVCacheManager(
        kv_cache_config=kv_cache_config,
        max_model_len=65536,
        max_in_flight_tokens=4096,
        scheduler_block_size=32768,
        hash_block_size=32768,
        enable_caching=True,
        use_eagle=False,
        dcp_world_size=1,
        pcp_world_size=8,
    )
    coordinator = cache_manager.coordinator
    request = SimpleNamespace(
        request_id="first",
        block_hashes=[BlockHash(b"x" * 32)],
        num_prompt_tokens=36864,
        shared_prefix_boundary=0,
    )
    empty_blocks = tuple([] for _ in kv_cache_config.kv_cache_groups)

    coordinator.allocate_new_computed_blocks(
        request.request_id,
        empty_blocks,
        num_local_computed_tokens=0,
        num_external_computed_tokens=36863,
    )
    coordinator.allocate_new_blocks(request.request_id, 36864, 36864)
    coordinator.cache_blocks(request, 36864)
    coordinator.free(request.request_id)

    blocks, hits = coordinator.find_longest_cache_hit_per_group(
        request.block_hashes, 36863)

    # Align-mode Mamba retains only the latest state checkpoint, while full
    # attention retains the reusable page. A consumer connector may resume
    # from that divergent attention hit because it supplies final recurrent
    # state for the synthetic suffix.
    assert hits == (32768, 0)
    assert [len(group) for group in blocks] == [1, 0]

    cached, num_local, boundary, diverged = (
        cache_manager.get_computed_blocks_for_connector(
            SimpleNamespace(
                block_hashes=request.block_hashes,
                num_tokens=36864,
                skip_reading_prefix_cache=False,
            )))
    assert num_local == 32768
    assert boundary == 0
    assert diverged
    assert [len(group) for group in cached.blocks] == [1, 0]


def test_hybrid_pcp_coordinator_recovers_pcp_folded_by_vllm_scheduler():
    """vLLM 0.26 passes PCP=1 after folding PCP into scheduler blocks."""
    _patch_vllm_hybrid_pcp_block_sizes()
    from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator

    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], _full_spec(block_size=2304)),
            KVCacheGroupSpec(["gdn"], _mamba_spec(block_size=2304)),
        ],
    )

    coordinator = get_kv_cache_coordinator(
        kv_cache_config=kv_cache_config,
        max_model_len=4096,
        max_in_flight_tokens=4096,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        # This is intentionally 1: vLLM 0.26's Scheduler hard-codes it.
        pcp_world_size=1,
        scheduler_block_size=9216,
        hash_block_size=9216,
    )

    assert [
        manager.block_size for manager in coordinator.single_type_managers
    ] == [9216, 9216]


def test_hybrid_pcp_coordinator_does_not_misclassify_dcp():
    _patch_vllm_hybrid_pcp_block_sizes()
    from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator

    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], _full_spec(block_size=16)),
            KVCacheGroupSpec(["gdn"], _mamba_spec(block_size=16)),
        ],
    )

    coordinator = get_kv_cache_coordinator(
        kv_cache_config=kv_cache_config,
        max_model_len=32768,
        max_in_flight_tokens=512,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=8,
        pcp_world_size=1,
        scheduler_block_size=128,
        hash_block_size=16,
    )

    assert [
        manager.block_size for manager in coordinator.single_type_managers
    ] == [128, 16]


def test_hybrid_dcp_reuses_externally_materialized_attention_prefix():
    """Per-group connector lookup must use DCP's logical page geometry."""
    _patch_vllm_hybrid_pcp_block_sizes()
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import BlockHash

    kv_cache_config = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], _full_spec(block_size=16)),
            KVCacheGroupSpec(["kda"], _mamba_spec(block_size=16)),
        ],
    )
    cache_manager = KVCacheManager(
        kv_cache_config=kv_cache_config,
        max_model_len=384,
        max_in_flight_tokens=128,
        scheduler_block_size=128,
        hash_block_size=16,
        enable_caching=True,
        use_eagle=False,
        dcp_world_size=8,
        pcp_world_size=1,
    )
    coordinator = cache_manager.coordinator
    first_hashes = [BlockHash(i.to_bytes(32, "little")) for i in range(24)]
    second_hashes = first_hashes[:16] + [
        BlockHash((100 + i).to_bytes(32, "little")) for i in range(8)
    ]
    request = SimpleNamespace(
        request_id="first",
        block_hashes=first_hashes,
        num_prompt_tokens=384,
        shared_prefix_boundary=0,
    )
    empty_blocks = tuple([] for _ in kv_cache_config.kv_cache_groups)

    coordinator.allocate_new_computed_blocks(
        request.request_id,
        empty_blocks,
        num_local_computed_tokens=0,
        num_external_computed_tokens=383,
    )
    coordinator.allocate_new_blocks(request.request_id, 384, 384)
    coordinator.cache_blocks(request, 384)
    coordinator.free(request.request_id)

    blocks, hits = coordinator.find_longest_cache_hit_per_group(
        second_hashes, 383)

    assert hits == (256, 0)
    assert [len(group) for group in blocks] == [2, 0]

    cached, num_local, boundary, diverged = (
        cache_manager.get_computed_blocks_for_connector(
            SimpleNamespace(
                block_hashes=second_hashes,
                num_tokens=384,
                skip_reading_prefix_cache=False,
            )))
    assert num_local == 256
    assert boundary == 0
    assert diverged
    assert [len(group) for group in cached.blocks] == [2, 0]


def test_hybrid_pcp_patch_keeps_non_mamba_scope():
    _patch_vllm_hybrid_pcp_block_sizes()
    from vllm.v1.core import kv_cache_utils

    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], _full_spec()),
            KVCacheGroupSpec(
                ["sliding"],
                SlidingWindowSpec(
                    block_size=512,
                    num_kv_heads=4,
                    head_size=128,
                    sliding_window=1024,
                    dtype=torch.bfloat16,
                ),
            ),
        ],
    )

    patched_resolver = kv_cache_utils.resolve_kv_cache_block_sizes
    original_resolver = (
        kv_cache_utils._tpu_original_resolve_kv_cache_block_sizes)
    vllm_config = _config(pcp=8)

    assert _resolve_outcome(
        patched_resolver,
        kv_cache_config,
        vllm_config,
    ) == _resolve_outcome(
        original_resolver,
        kv_cache_config,
        vllm_config,
    )


def test_hybrid_pcp_coordinator_patch_keeps_non_mamba_scope():
    _patch_vllm_hybrid_pcp_block_sizes()
    from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator

    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], _full_spec()),
            KVCacheGroupSpec(
                ["sliding"],
                SlidingWindowSpec(
                    block_size=512,
                    num_kv_heads=4,
                    head_size=128,
                    sliding_window=1024,
                    dtype=torch.bfloat16,
                ),
            ),
        ],
    )

    with pytest.raises(AssertionError, match="PCP not support hybrid"):
        get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=4096,
            max_in_flight_tokens=4096,
            use_eagle=False,
            enable_caching=True,
            enable_kv_cache_events=False,
            dcp_world_size=1,
            pcp_world_size=8,
            scheduler_block_size=512,
            hash_block_size=512,
        )
