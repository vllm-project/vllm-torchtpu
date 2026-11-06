# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Thin ``pallas.jax_op`` custom-op bridges for DeepSeek-V4.1 indexer kernels."""

from typing import Any

import jax
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.experimental.deepseek_v41.indexer import (
    paged_indexer_topk,
)

_REPLICATED_SPEC: Any = PartitionSpec()
PAGED_INDEXER_TOPK_PARTITION_SPECS: Any = (
    *(_REPLICATED_SPEC,) * 6,
    None,
    _REPLICATED_SPEC,
    _REPLICATED_SPEC,
    None,
)


def _paged_indexer_topk_jax(
    q: jax.Array,
    weights: jax.Array,
    k_pool_lo: jax.Array,
    block_tables: jax.Array,
    seq_lens: jax.Array,
    cu_q_lens: jax.Array,
    topk: int,
    k_pool_hi: jax.Array | None = None,
    candidate_indices: jax.Array | None = None,
    compress_ratio: int = 1,
) -> jax.Array:
    """Forward paged indexer arguments into ``paged_indexer_topk``."""
    return paged_indexer_topk(
        q,
        weights,
        k_pool_lo,
        block_tables,
        seq_lens,
        cu_q_lens,
        topk=topk,
        k_pool_hi=k_pool_hi,
        candidate_indices=candidate_indices,
        compress_ratio=compress_ratio,
    )


_paged_indexer_topk_op = pallas.jax_op(
    "pallas::dsv41_paged_indexer_topk",
    _paged_indexer_topk_jax,
    input_partition_specs=PAGED_INDEXER_TOPK_PARTITION_SPECS,
    output_partition_specs=_REPLICATED_SPEC,
)


@_paged_indexer_topk_op.register_fake
def _fake_paged_indexer_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    k_pool_lo: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_q_lens: torch.Tensor,
    topk: int,
    k_pool_hi: torch.Tensor | None = None,
    candidate_indices: torch.Tensor | None = None,
    compress_ratio: int = 1,
) -> torch.Tensor:
    """Return an abstract ``int32[num_tokens, topk]`` tensor for FakeTensor tracing."""
    return q.new_empty((q.shape[0], topk), dtype=torch.int32)


def dsv41_paged_indexer_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    k_pool_lo: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_q_lens: torch.Tensor,
    topk: int,
    k_pool_hi: torch.Tensor | None = None,
    candidate_indices: torch.Tensor | None = None,
    compress_ratio: int = 1,
) -> torch.Tensor:
    """Gather 1/2-bank paged index rows, score split-BF16 keys, and select top-k.

    Args:
        q: Split-element query tensor of shape ``(num_tokens, num_heads, 128)``
            and dtype ``torch.bfloat16``.
        weights: Per-head query gates of shape ``(num_tokens, num_heads, 1)``
            and dtype ``torch.float32``.
        k_pool_lo: Low HBM indexer key bank of shape
            ``(num_pages_lo, page_rows, 128)`` and dtype ``torch.int32``.
        block_tables: Padded per-sequence page table of shape
            ``(num_seqs, max_blocks_per_seq)`` and dtype ``torch.int32``.
        seq_lens: Total context lengths per sequence of shape ``(num_seqs,)``
            and dtype ``torch.int32``.
        cu_q_lens: Cumulative query token offsets of shape ``(num_seqs + 1,)``
            and dtype ``torch.int32``.
        topk: Number of top-scoring key indices selected per query token.
        k_pool_hi: Optional high HBM indexer key bank of shape
            ``(num_pages_hi, page_rows, 128)`` and dtype ``torch.int32``.
        candidate_indices: Optional expanded per-key candidate whitelist of shape
            ``(num_tokens, num_candidates)`` and dtype ``torch.int32``.
        compress_ratio: Token compression ratio per key (``1`` for full-resolution
            or ``2`` for 2x compressed layers).

    Returns:
        Selected key indices of shape ``(num_tokens, topk)`` and dtype
        ``torch.int32`` (with ``-1`` in invalid/masked slots).

    Data lifecycle:
        - Used: ``q`` (split-element ``bfloat16[tokens, heads, 128]``), ``weights``,
          ``k_pool_lo``, ``block_tables``, ``seq_lens``, ``cu_q_lens``, optional
          ``k_pool_hi``, optional expanded per-key ``candidate_indices``
          (``int32[tokens, num_candidates]``), and static ``topk`` /
          ``compress_ratio`` (each distinct static value compiles separately).
        - Changed: None.
        - Created: ``int32[tokens, topk]`` key indices (``-1`` in invalid slots).
    """
    return _paged_indexer_topk_op(
        q,
        weights,
        k_pool_lo,
        block_tables,
        seq_lens,
        cu_q_lens,
        int(topk),
        k_pool_hi,
        candidate_indices,
        int(compress_ratio),
    )


__all__ = ["dsv41_paged_indexer_topk"]
