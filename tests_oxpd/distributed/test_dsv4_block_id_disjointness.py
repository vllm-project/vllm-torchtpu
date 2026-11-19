# SPDX-License-Identifier: Apache-2.0
"""The block-ID invariant DeepSeek-V4's Raiden admission rests on.

DSv4 overlays several logical caches onto one physical array (see
``TPUModelRunner._allocate_ds_v4_kv_caches``), and the manifest declares those
pools at ``base_offset_bytes=0`` with overlapping block ranges. Only block
ownership keeps them apart, so a violation would be silent: Raiden would copy
real bytes to real addresses, just the wrong ones.
"""

from __future__ import annotations

import pytest
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.kv_cache_interface import KVCacheConfig

from .raiden_test_utils import (
    DSV4_CSA_BLOCK_TOKENS,
    DSV4_SWA_BLOCK_TOKENS,
    dsv4_kv_cache_groups,
)

# The SWA group takes 8 blocks to the CSA group's 1, so four 2-page requests
# need 4 * (2 + 2 + 16).
_NUM_BLOCKS = 256


def _coordinator():
    """A DSv4-shaped coordinator: CSA, indexer and SWA in separate groups.

    Prefix caching is off so the probe measures plain allocation rather than
    cache-hit block sharing, which is a deliberate form of reuse.
    """
    config = KVCacheConfig(
        num_blocks=_NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=dsv4_kv_cache_groups(),
    )
    return get_kv_cache_coordinator(
        config,
        max_model_len=DSV4_CSA_BLOCK_TOKENS * 8,
        max_in_flight_tokens=DSV4_CSA_BLOCK_TOKENS,
        use_eagle=False,
        enable_caching=False,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        scheduler_block_size=DSV4_CSA_BLOCK_TOKENS,
        hash_block_size=DSV4_CSA_BLOCK_TOKENS,
    )


def _group_block_ids(coordinator, request_id: str) -> list[set[int]]:
    return [
        {block.block_id for block in blocks if block.block_id != 0}
        for blocks in coordinator.get_blocks(request_id)
    ]


@pytest.mark.parametrize(
    "sizes",
    [
        (1,),
        (DSV4_CSA_BLOCK_TOKENS, 3 * DSV4_CSA_BLOCK_TOKENS),
        (2 * DSV4_CSA_BLOCK_TOKENS,) * 4,
    ],
)
def test_no_two_cache_groups_ever_hold_the_same_block_id(sizes):
    """No live block ID is claimed twice, across groups and across requests.

    Every single-type manager is handed the same BlockPool, which is the
    structural reason IDs are globally unique: one free-block queue, so a block
    popped for one group cannot also be popped for another.
    """
    coordinator = _coordinator()
    assert {id(m.block_pool) for m in coordinator.single_type_managers} == {
        id(coordinator.block_pool)
    }

    owner: dict[int, tuple[str, int]] = {}
    for index, num_tokens in enumerate(sizes):
        request_id = f"req-{index}"
        coordinator.allocate_new_blocks(
            request_id, num_tokens=num_tokens, num_tokens_main_model=num_tokens
        )
        per_group = _group_block_ids(coordinator, request_id)
        assert all(per_group), "every group should have been allocated blocks"
        for group_index, block_ids in enumerate(per_group):
            for block_id in block_ids:
                held = owner.get(block_id)
                assert held is None, (
                    f"block {block_id} is held by both {held} and "
                    f"({request_id}, group {group_index}); the DSv4 overlay "
                    "would corrupt that page"
                )
                owner[block_id] = (request_id, group_index)


def test_swa_group_pages_more_blocks_than_csa_for_the_same_tokens():
    """SWA's smaller logical block is a count difference, not a byte one.

    _swa_block_size divides the CSA block by 8, so the SWA group takes eight
    times the blocks for the same tokens while each block still covers the same
    bytes. That is why the manifest gives both pools the same
    block_stride_bytes and num_blocks.
    """
    coordinator = _coordinator()
    coordinator.allocate_new_blocks(
        "req-0",
        num_tokens=DSV4_CSA_BLOCK_TOKENS,
        num_tokens_main_model=DSV4_CSA_BLOCK_TOKENS,
    )
    csa_blocks, _, swa_blocks = _group_block_ids(coordinator, "req-0")

    assert DSV4_CSA_BLOCK_TOKENS // DSV4_SWA_BLOCK_TOKENS == 8
    assert len(swa_blocks) == 8 * len(csa_blocks)
