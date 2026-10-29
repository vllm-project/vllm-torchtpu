# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MQA indexer routing and stage-local candidate key reuse."""

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from vllm_torchtpu.kernels.deepseek_v41.index_selection.candidate import cache_capacity

from . import indexer_impl
from .cache import IndexCacheRows
from .candidate_cache import prepare_candidate_cache, select_candidate_cache
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
    operation = indexer_impl._query_op
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
        cache: IndexCacheRows | PackedFP8IndexCacheRows,
        records: torch.Tensor,
        lengths: torch.Tensor,
        requests: torch.Tensor,
        *,
        candidates: torch.Tensor,
        batch_max: torch.Tensor,
        page_records: int,
        unordered: bool,
        score_width: int,
        stage_cache: dict[Any, Any] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidate_filter = (
            self.candidate_blocks is not None and not self.candidate_write
        )
        reuse = candidate_filter and stage_cache is not None
        packed_fp8 = isinstance(cache, PackedFP8IndexCacheRows)
        capacity = (
            cache_capacity(
                tuple(q.shape),
                tuple(records.shape),
                candidates.shape[1],
                self.candidate_block_size,
                page_records,
            )
            if not packed_fp8
            and reuse
            and unordered
            and q.dtype == torch.bfloat16
            and self.topk_tokens == 512
            else None
        )
        if capacity is None:
            select: Callable[..., tuple[torch.Tensor, torch.Tensor]] = (
                functools.partial(
                    indexer_impl.paged_fp8_indexer,
                    score_width=score_width,
                    stage_cache=stage_cache,
                    source_prefix=self.k_cache.prefix,
                    candidate_source=self.candidate_source_layer,
                )
                if packed_fp8
                else functools.partial(
                    indexer_impl.paged_indexer, page_records=page_records
                )
            )
            return select(
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
                candidate_filter=candidate_filter,
                unordered=unordered,
                batch_max=batch_max,
            )

        assert stage_cache is not None
        assert isinstance(cache, IndexCacheRows)
        cache_key = (
            "candidate_index_keys",
            self.k_cache.prefix,
            stage_cache.get(("candidate_index_version", self.k_cache.prefix), 0),
            self.candidate_source_layer,
            stage_cache.get(("candidate_publication", self.candidate_source_layer), 0),
            capacity,
        )
        prepared = stage_cache.get(cache_key)
        if prepared is None:
            prepared = prepare_candidate_cache(
                cache,
                records,
                candidates,
                requests,
                lengths,
                block_size=self.candidate_block_size,
                capacity=capacity,
            )
            stage_cache[cache_key] = prepared

        selected = select_candidate_cache(
            q,
            cache,
            records,
            weights,
            lengths,
            candidates,
            requests,
            batch_max,
            prepared,
            block_size=self.candidate_block_size,
            page_records=page_records,
            request_tile=None,
        )
        return selected, candidates
