# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Unit tests for standalone kernel helpers in llm_tpu_core."""

import torch

from vllm_torchtpu.llm_tpu_core.kernels.block_sparse import (
    compute_sparse_block_offsets,
)
from vllm_torchtpu.llm_tpu_core.kernels.paged_attention import (
    compute_paged_attention_shape,
)
from vllm_torchtpu.llm_tpu_core.kernels.rpa import (
    AttentionScope,
    BlockSizes,
    get_rpa_kernel_version,
)


def test_paged_attention_shapes():
    nhd_shape = compute_paged_attention_shape(
        num_blocks=32,
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        cache_layout="NHD",
    )
    assert nhd_shape == (32, 16, 8, 128)

    hnd_shape = compute_paged_attention_shape(
        num_blocks=32,
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        cache_layout="HND",
    )
    assert hnd_shape == (32, 8, 16, 128)


def test_sparse_block_offsets():
    indices = torch.tensor([0, 2, 5, 8], dtype=torch.int32)
    block_size = 64
    offsets = compute_sparse_block_offsets(indices, block_size)
    assert torch.equal(offsets, torch.tensor([0, 128, 320, 512], dtype=torch.int32))


def test_rpa_metadata():
    assert get_rpa_kernel_version() == "v3"
    assert AttentionScope.FULL == "full"
    assert AttentionScope.CACHE_ONLY == "cache_only"
    assert AttentionScope.NEW_TOKENS_ONLY == "new_tokens_only"

    sizes = BlockSizes(bq_sz=64, bq_c_sz=64, bkv_sz=128, batch_size=4, n_buffer=2)
    assert sizes.bq_sz == 64
    assert sizes.bkv_sz == 128
