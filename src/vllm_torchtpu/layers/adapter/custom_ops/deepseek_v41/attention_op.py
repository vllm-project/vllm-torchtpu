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

from typing import Any

import jax
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.experimental.deepseek_v41.sparse_mla import (
    gather_sparse_fp8_kv,
    sparse_mla_swa_attention,
)
from vllm_torchtpu.kernels.experimental.deepseek_v41.sparse_mla.kernel import (
    ROPE_TOKENS_PER_ROW,
    ROW_WORDS,
)

_REPLICATED_SPEC: Any = PartitionSpec()
_REPLICATED_PAIR_SPECS: Any = (_REPLICATED_SPEC, _REPLICATED_SPEC)
GATHER_SPARSE_FP8_KV_PARTITION_SPECS: Any = (
    *(_REPLICATED_SPEC,) * 5,
    None,
    _REPLICATED_SPEC,
    _REPLICATED_SPEC,
)
SPARSE_MLA_SWA_PARTITION_SPECS: Any = (
    *(_REPLICATED_SPEC,) * 7,
    None,
    None,
    _REPLICATED_SPEC,
    None,
    _REPLICATED_SPEC,
)


def _gather_sparse_fp8_kv_jax(
    kv_pool_lo: jax.Array,
    rope_pool_lo: jax.Array,
    topk_indices: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    page_size: int,
    kv_pool_hi: jax.Array | None = None,
    rope_pool_hi: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Forward paged FP8 NoPE + BF16 RoPE gather into ``gather_sparse_fp8_kv``."""
    return gather_sparse_fp8_kv(
        kv_pool_lo,
        rope_pool_lo,
        topk_indices,
        block_tables,
        cu_q_lens,
        page_size=page_size,
        kv_pool_hi=kv_pool_hi,
        rope_pool_hi=rope_pool_hi,
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
    """Forward Sparse MLA + SWA arguments into ``sparse_mla_swa_attention``."""
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
    input_partition_specs=GATHER_SPARSE_FP8_KV_PARTITION_SPECS,
    output_partition_specs=_REPLICATED_PAIR_SPECS,
)


@_gather_sparse_fp8_kv_op.register_fake
def _fake_gather_sparse_fp8_kv(
    kv_pool_lo: torch.Tensor,
    rope_pool_lo: torch.Tensor,
    topk_indices: torch.Tensor,
    block_tables: torch.Tensor,
    cu_q_lens: torch.Tensor,
    page_size: int,
    kv_pool_hi: torch.Tensor | None = None,
    rope_pool_hi: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return abstract ``(gathered_nope, gathered_rope)`` tensors for FakeTensor."""
    tokens, topk = topk_indices.shape
    return (
        kv_pool_lo.new_empty((tokens, topk, ROW_WORDS), dtype=torch.int32),
        rope_pool_lo.new_empty(
            (tokens, topk // ROPE_TOKENS_PER_ROW, ROW_WORDS), dtype=torch.int32
        ),
    )


_sparse_mla_swa_attention_op = pallas.jax_op(
    "pallas::dsv41_sparse_mla_swa_attention",
    _sparse_mla_swa_attention_jax,
    input_partition_specs=SPARSE_MLA_SWA_PARTITION_SPECS,
    output_partition_specs=_REPLICATED_SPEC,
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
    """Return an abstract tensor matching ``q`` for ``torch.compile`` tracing."""
    return torch.empty_like(q)


def dsv41_gather_sparse_fp8_kv(
    kv_pool_lo: torch.Tensor,
    rope_pool_lo: torch.Tensor,
    topk_indices: torch.Tensor,
    block_tables: torch.Tensor,
    cu_q_lens: torch.Tensor,
    page_size: int,
    kv_pool_hi: torch.Tensor | None = None,
    rope_pool_hi: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather 1-bank or 2-bank paged FP8 NoPE rows and packed BF16 RoPE quarters.

    Args:
        kv_pool_lo: Low HBM FP8 NoPE word-row bank of shape
            ``(num_pages_lo, page_size, 128)`` and dtype ``torch.int32``.
        rope_pool_lo: Low HBM packed BF16 RoPE word-row bank of shape
            ``(num_pages_lo, page_size // 4, 128)`` and dtype ``torch.int32``.
        topk_indices: Selected token/key positions of shape ``(num_tokens, topk)``
            and dtype ``torch.int32``.
        block_tables: Padded per-sequence page table of shape
            ``(num_seqs, max_blocks_per_seq)`` and dtype ``torch.int32``.
        cu_q_lens: Cumulative query token offsets of shape ``(num_seqs + 1,)``
            and dtype ``torch.int32``.
        page_size: Number of logical records stored per physical page.
        kv_pool_hi: Optional high HBM FP8 NoPE word-row bank of shape
            ``(num_pages_hi, page_size, 128)`` and dtype ``torch.int32``.
        rope_pool_hi: Optional high HBM packed BF16 RoPE word-row bank of shape
            ``(num_pages_hi, page_size // 4, 128)`` and dtype ``torch.int32``.

    Returns:
        Tuple ``(gathered_nope, gathered_rope)`` of ``torch.int32`` tensors with
        shapes ``(num_tokens, topk, 128)`` and ``(num_tokens, topk // 4, 128)``.

    Data lifecycle:
        - Used: ``kv_pool_lo``, ``rope_pool_lo``, ``topk_indices``, ``block_tables``,
          ``cu_q_lens``, static ``page_size`` (compiled per distinct value), and
          optional ``kv_pool_hi`` / ``rope_pool_hi``.
        - Changed: None.
        - Created: ``(gathered_nope, gathered_rope)`` of shapes
          ``(tokens, topk, 128)`` and ``(tokens, topk // 4, 128)``.
    """
    return _gather_sparse_fp8_kv_op(
        kv_pool_lo,
        rope_pool_lo,
        topk_indices,
        block_tables,
        cu_q_lens,
        int(page_size),
        kv_pool_hi,
        rope_pool_hi,
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

    Args:
        q: Absorbed query tensor of shape ``(num_tokens, num_heads, 512)`` and
            dtype ``torch.bfloat16`` (first ``448`` dims NoPE, last ``64`` dims RoPE).
        gathered_nope: Gathered FP8 NoPE words of shape ``(num_tokens, topk, 128)``
            and dtype ``torch.int32``.
        gathered_rope: Gathered packed BF16 RoPE words of shape
            ``(num_tokens, topk // 4, 128)`` and dtype ``torch.int32``.
        topk_indices: Selected key positions of shape ``(num_tokens, topk)`` and
            dtype ``torch.int32`` (``-1`` in invalid slots).
        attention_sinks: Per-head sink logits of shape ``(num_heads,)`` and dtype
            ``torch.float32``.
        seq_lens: Total context lengths per sequence of shape ``(num_seqs,)`` and
            dtype ``torch.int32``.
        cu_q_lens: Cumulative query token offsets of shape ``(num_seqs + 1,)`` and
            dtype ``torch.int32``.
        sm_scale: Softmax scaling factor (``1.0 / sqrt(d_qk)``).
        swa_window: Sliding-window attention token count (``128`` in DeepSeek-V4.1).
        swa_kv: Optional BF16 SWA KV tensor in 3-D ``(num_tokens, swa_window, 512)``
            or compact 2-D ``(num_seqs * swa_window + num_tokens, 512)`` layout.
        kv_compress_ratio: Key compression stride (``1`` for 1x, ``2`` for 2x;
            when ``>= 1``, global MLA keys inside the active SWA tail are masked).
        swa_min_pos: Optional per-sequence SWA lower-bound token positions of
            shape ``(num_seqs,)`` and dtype ``torch.int32``.

    Returns:
        Latent attention output tensor of shape ``(num_tokens, num_heads, 512)``
        and dtype ``torch.bfloat16``.

    Data lifecycle:
        - Used: ``q`` (``bfloat16[tokens, heads, 512]``), ``gathered_nope``,
          ``gathered_rope``, ``topk_indices``, ``attention_sinks``, ``seq_lens``,
          ``cu_q_lens``, static ``sm_scale`` / ``swa_window`` /
          ``kv_compress_ratio``, optional ``swa_kv`` (3-D ``[tokens, W, 512]`` or
          compact 2-D ``[num_seqs * W + tokens, 512]``), and optional
          ``swa_min_pos`` (``int32[num_seqs]`` per-sequence SWA lower bound).
        - Changed: None.
        - Created: ``bfloat16[tokens, heads, 512]`` latent output.
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
