# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Unit tests for standalone KV cache logic in llm_tpu_core."""

import torch

from vllm_torchtpu.llm_tpu_core.kv_cache.block_major_pool import (
    BlockMajorPoolLayout,
    flat_kernel_ids,
    flat_manager_ids,
    is_block_major_layout_applicable,
    kernel_view,
)
from vllm_torchtpu.llm_tpu_core.kv_cache.specs import (
    AttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
)


def test_block_major_pool_layout_geometry():
    layout = BlockMajorPoolLayout(
        num_blocks=128,
        num_pools=4,
        split=2,
        pool_index_by_layer={"l0": 0, "l1": 1, "l2": 2, "l3": 3},
    )
    assert layout.rows_per_block == 8
    assert is_block_major_layout_applicable(
        block_major_flag=True, is_unified_kv_layout=True, is_hybrid=True
    )
    assert not is_block_major_layout_applicable(
        block_major_flag=False, is_unified_kv_layout=True, is_hybrid=True
    )


def test_flat_id_mappings():
    mgr_ids = torch.tensor([0, 1, 2, 5], dtype=torch.int32)
    # 4 pools: region 2 maps mgr_id to mgr_id * 4 + 2
    flat_mgr = flat_manager_ids(mgr_ids, pool_index=2, num_pools=4)
    expected_mgr = torch.tensor([2, 6, 10, 22], dtype=torch.int32)
    assert torch.equal(flat_mgr, expected_mgr)

    # Kernel IDs with split=2
    k_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    flat_k = flat_kernel_ids(k_ids, pool_index=1, num_pools=4, split=2)
    # k_id=0 (mgr 0, intra 0) -> (0*4+1)*2 + 0 = 2
    # k_id=1 (mgr 0, intra 1) -> (0*4+1)*2 + 1 = 3
    # k_id=2 (mgr 1, intra 0) -> (1*4+1)*2 + 0 = 10
    # k_id=3 (mgr 1, intra 1) -> (1*4+1)*2 + 1 = 11
    assert torch.equal(flat_k, torch.tensor([2, 3, 10, 11], dtype=torch.int32))


def test_kernel_view_folding():
    # 3D tensor: (num_blocks, rows_per_block, page_size)
    pool = torch.randn(8, 4, 128)
    folded = kernel_view(pool)
    assert folded.shape == (32, 128)


def test_standalone_kv_cache_specs():
    attn_spec = AttentionSpec(block_size=16, num_kv_heads=8, head_size=128)
    assert attn_spec.page_shape == (16, 8, 128)

    hnd_spec = AttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128, cache_layout="HND"
    )
    assert hnd_spec.page_shape == (8, 16, 128)

    mla_spec = MLAAttentionSpec(block_size=16, kv_lora_rank=512, qk_rope_head_dim=64)
    assert mla_spec.head_size == 576
    assert mla_spec.page_shape == (16, 576)

    mamba_spec = MambaSpec(block_size=16, d_state=64, d_conv=4)
    assert mamba_spec.d_state == 64
