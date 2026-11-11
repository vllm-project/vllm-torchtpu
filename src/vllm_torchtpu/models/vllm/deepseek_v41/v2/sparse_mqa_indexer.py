# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MQA indexer over the packed FP8 Index-K cache."""

from typing import TYPE_CHECKING, Any

import torch

from . import indexer_fp8
from .indexer_cache import PackedFP8IndexCacheRows

if TYPE_CHECKING:
    from vllm.models.deepseek_v41.attention import DeepseekV4IndexerCache


def quantize_query(
    q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate Q, quantize each head and fold its scale into FP32 weights."""
    operation = indexer_fp8._query_op
    assert operation is not None
    return operation(q, positions, cos_sin, weights)


class SparseMQAIndexer(torch.nn.Module):
    """Score full or candidate context after the model writes the K cache."""

    def __init__(
        self,
        k_cache: "DeepseekV4IndexerCache",
        topk_tokens: int,
        candidate_blocks: torch.Tensor | None,
        candidate_block_size: int,
        *,
        candidate_source_layer: int | None,
        candidate_write: bool = False,
    ) -> None:
        super().__init__()
        self.k_cache = k_cache
        self.topk_tokens = topk_tokens
        self.candidate_blocks = candidate_blocks
        self.candidate_block_size = candidate_block_size
        self.candidate_source_layer = candidate_source_layer
        self.candidate_write = candidate_write

    def forward(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        cache: PackedFP8IndexCacheRows,
        records: torch.Tensor,
        lengths: torch.Tensor,
        requests: torch.Tensor,
        *,
        candidates: torch.Tensor,
        batch_max: torch.Tensor,
        unordered: bool,
        score_width: int,
        stage_cache: dict[Any, Any] | None,
        query_start_loc: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return indexer_fp8.paged_fp8_indexer(
            q,
            cache,
            records,
            weights,
            lengths,
            candidates,
            requests,
            topk=self.topk_tokens,
            block_size=max(1, self.candidate_block_size),
            candidate_write=self.candidate_write,
            sparse_logits=self.candidate_blocks is not None
            and not self.candidate_write,
            unordered=unordered,
            batch_max=batch_max,
            score_width=score_width,
            stage_cache=stage_cache,
            source_prefix=self.k_cache.prefix,
            candidate_source=self.candidate_source_layer,
            query_start_loc=query_start_loc,
        )
