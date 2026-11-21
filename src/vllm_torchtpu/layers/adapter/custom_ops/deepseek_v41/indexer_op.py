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
"""Thin ``pallas.jax_op`` custom-op bridge for DeepSeek-V4.1 paged indexer."""

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.experimental.deepseek_v41.indexer import (
    paged_indexer_topk,
)


def _paged_indexer_topk_jax(
    q: jax.Array,
    weights: jax.Array,
    k_pool: jax.Array,
    block_tables: jax.Array,
    seq_lens: jax.Array,
    cu_q_lens: jax.Array,
    topk: int,
    candidate_indices: jax.Array | None = None,
    compress_ratio: int = 1,
) -> jax.Array:
    return paged_indexer_topk(
        q,
        weights,
        k_pool,
        block_tables,
        seq_lens,
        cu_q_lens,
        topk=topk,
        candidate_indices=candidate_indices,
        compress_ratio=compress_ratio,
    )


_paged_indexer_topk_op = pallas.jax_op(
    "pallas::dsv41_paged_indexer_topk",
    _paged_indexer_topk_jax,
)


@_paged_indexer_topk_op.register_fake
def _fake_paged_indexer_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    k_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_q_lens: torch.Tensor,
    topk: int,
    candidate_indices: torch.Tensor | None = None,
    compress_ratio: int = 1,
) -> torch.Tensor:
    del weights, k_pool, block_tables, seq_lens, cu_q_lens
    del candidate_indices, compress_ratio
    return q.new_empty((q.shape[0], topk), dtype=torch.int32)


def dsv41_paged_indexer_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    k_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_q_lens: torch.Tensor,
    topk: int,
    candidate_indices: torch.Tensor | None = None,
    compress_ratio: int = 1,
) -> torch.Tensor:
    """Gather paged single-bank index rows and return ``int32[T, topk]`` top-k ids.

    Scores ``q: bf16[T, H, 128]`` and ``weights: f32[T, H, 1]`` against split-BF16
    ``k_pool: int32[P, page_rows, 128]`` (with ``page_rows % 8 == 0``).
    """
    return _paged_indexer_topk_op(
        q,
        weights,
        k_pool,
        block_tables,
        seq_lens,
        cu_q_lens,
        int(topk),
        candidate_indices,
        int(compress_ratio),
    )


__all__ = ["dsv41_paged_indexer_topk"]
