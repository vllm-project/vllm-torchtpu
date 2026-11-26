# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MQA indexer over the packed FP8 Index-K cache."""

from typing import TYPE_CHECKING, Any

import torch

from . import indexer_fp8
from .indexer_cache import PackedFP8IndexCacheRows

if TYPE_CHECKING:
    from vllm.models.deepseek_v41.attention import DeepseekV4IndexerCache


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

    def prefetch_candidates(
        self,
        cache: PackedFP8IndexCacheRows,
        records: torch.Tensor,
        requests: torch.Tensor,
        *,
        candidates: torch.Tensor,
        score_width: int,
        stage_cache: dict[Any, Any] | None,
    ) -> None:
        """Gather the candidate K that `forward` will score, ahead of it.

        Issue it before the layer's in-place KV writes: the gather then reads
        the same Index-K under the previous layer's work instead of stalling
        this layer's scoring.
        """
        block_size = max(1, self.candidate_block_size)
        if (
            self.candidate_blocks is None
            or self.candidate_write
            or not indexer_fp8.uses_prepared_candidates(
                requests.shape[0], candidates, block_size, score_width, stage_cache
            )
        ):
            return
        assert self.candidate_source_layer is not None and stage_cache is not None
        indexer_fp8.prepared_candidates(
            cache,
            records,
            candidates,
            requests,
            block_size=block_size,
            score_width=score_width,
            stage_cache=stage_cache,
            source_prefix=self.k_cache.prefix,
            candidate_source=self.candidate_source_layer,
        )
