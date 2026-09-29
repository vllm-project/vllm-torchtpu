# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Pallas Paged Attention kernel interfaces for TPU.

Strictly zero dependencies on vLLM.
"""

from __future__ import annotations


def compute_paged_attention_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    cache_layout: str = "NHD",
) -> tuple[int, ...]:
    """Compute physical shape of paged KV cache tensor."""
    if cache_layout == "HND":
        return (num_blocks, num_kv_heads, block_size, head_size)
    return (num_blocks, block_size, num_kv_heads, head_size)
