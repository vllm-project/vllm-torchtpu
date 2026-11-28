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
"""Thin ``pallas.jax_op`` custom-op bridges for DeepSeek-V4.1 sparse MLA + SWA."""

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.experimental.deepseek_v41.sparse_mla import (
    gather_sparse_fp8_kv,
    sparse_mla_swa_attention,
)
from vllm_torchtpu.kernels.experimental.deepseek_v41.sparse_mla.kernel import (
    ROPE_TOKENS_PER_ROW,
    ROW_WORDS,
)


def _gather_sparse_fp8_kv_jax(
    kv_pool: jax.Array,
    rope_pool: jax.Array,
    topk_indices: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    page_size: int,
) -> tuple[jax.Array, jax.Array]:
    return gather_sparse_fp8_kv(
        kv_pool,
        rope_pool,
        topk_indices,
        block_tables,
        cu_q_lens,
        page_size=page_size,
    )


def _sparse_mla_swa_attention_jax(
    q: jax.Array,
    gathered_nope: jax.Array,
    gathered_rope: jax.Array,
    topk_indices: jax.Array,
    attention_sinks: jax.Array,
    seq_lens: jax.Array,
    cu_q_lens: jax.Array,
    sm_scale: float,
    swa_window: int,
    swa_kv: jax.Array | None = None,
    kv_compress_ratio: int = 1,
    swa_min_pos: jax.Array | None = None,
) -> jax.Array:
    return sparse_mla_swa_attention(
        q,
        gathered_nope,
        gathered_rope,
        topk_indices,
        attention_sinks,
        seq_lens,
        cu_q_lens,
        sm_scale=sm_scale,
        swa_window=swa_window,
        swa_kv=swa_kv,
        kv_compress_ratio=kv_compress_ratio,
        swa_min_pos=swa_min_pos,
    )


_gather_sparse_fp8_kv_op = pallas.jax_op(
    "pallas::dsv41_gather_sparse_fp8_kv",
    _gather_sparse_fp8_kv_jax,
)


@_gather_sparse_fp8_kv_op.register_fake
def _fake_gather_sparse_fp8_kv(
    kv_pool: torch.Tensor,
    rope_pool: torch.Tensor,
    topk_indices: torch.Tensor,
    block_tables: torch.Tensor,
    cu_q_lens: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del block_tables, cu_q_lens, page_size
    tokens, topk = topk_indices.shape
    return (
        kv_pool.new_empty((tokens, topk, ROW_WORDS), dtype=torch.int32),
        rope_pool.new_empty(
            (tokens, topk // ROPE_TOKENS_PER_ROW, ROW_WORDS), dtype=torch.int32
        ),
    )


_sparse_mla_swa_attention_op = pallas.jax_op(
    "pallas::dsv41_sparse_mla_swa_attention",
    _sparse_mla_swa_attention_jax,
)


@_sparse_mla_swa_attention_op.register_fake
def _fake_sparse_mla_swa_attention(
    q: torch.Tensor,
    gathered_nope: torch.Tensor,
    gathered_rope: torch.Tensor,
    topk_indices: torch.Tensor,
    attention_sinks: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_q_lens: torch.Tensor,
    sm_scale: float,
    swa_window: int,
    swa_kv: torch.Tensor | None = None,
    kv_compress_ratio: int = 1,
    swa_min_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    del gathered_nope, gathered_rope, topk_indices, attention_sinks, seq_lens
    del cu_q_lens, sm_scale, swa_window, swa_kv, kv_compress_ratio, swa_min_pos
    return q.new_empty(q.shape, dtype=torch.bfloat16)


def dsv41_gather_sparse_fp8_kv(
    kv_pool: torch.Tensor,
    rope_pool: torch.Tensor,
    topk_indices: torch.Tensor,
    block_tables: torch.Tensor,
    cu_q_lens: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather single-bank paged FP8 NoPE rows and packed BF16 RoPE quarters.

    Reads ``kv_pool: int32[P, page_size, 128]`` and
    ``rope_pool: int32[P, page_size // 4, 128]`` via ``topk_indices: int32[T, K]``.
    """
    return _gather_sparse_fp8_kv_op(
        kv_pool,
        rope_pool,
        topk_indices,
        block_tables,
        cu_q_lens,
        int(page_size),
    )


def dsv41_sparse_mla_swa_attention(
    q: torch.Tensor,
    gathered_nope: torch.Tensor,
    gathered_rope: torch.Tensor,
    topk_indices: torch.Tensor,
    attention_sinks: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_q_lens: torch.Tensor,
    sm_scale: float,
    swa_window: int,
    swa_kv: torch.Tensor | None = None,
    kv_compress_ratio: int = 1,
    swa_min_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute fused Sparse MLA + SWA + sink online softmax attention.

    Consumes ``q: bf16[T, H, 512]``, ``gathered_nope: int32[T, K, 128]``,
    ``gathered_rope: int32[T, K // 4, 128]``, and optional ``swa_kv``
    (``(T, W, 512)`` or ``(N, 512)``).
    """
    return _sparse_mla_swa_attention_op(
        q,
        gathered_nope,
        gathered_rope,
        topk_indices,
        attention_sinks,
        seq_lens,
        cu_q_lens,
        float(sm_scale),
        int(swa_window),
        swa_kv,
        int(kv_compress_ratio),
        swa_min_pos,
    )


__all__ = ["dsv41_gather_sparse_fp8_kv", "dsv41_sparse_mla_swa_attention"]
