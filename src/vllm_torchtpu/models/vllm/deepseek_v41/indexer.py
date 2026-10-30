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
"""DeepSeek-V4.1 sparse-attention indexer and candidate-block publisher for TPU."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, ClassVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.deepseek_v41 import attention as upstream
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import indexer_op
from vllm_torchtpu.models.vllm.deepseek_v4 import attention as v4_attn

from .cache import ROW_BYTES, ROW_WORDS, NativeCacheView, PagedCacheRows, write_cache
from .compressor import AttentionBackend
from .metadata import (
    active_cache,
    block_table,
    cache_slots,
    layer_meta,
    not_replayed,
    rotate_rope,
)

INDEX_HEAD_DIM: int = 128
_SENTINEL_BLOCK_ID: int = 0x7FFFFFFF
_KEYS_PER_WORD_ROW: int = 2
_DEFAULT_QUANT_BLOCK_SIZE: int = 128
_DEFAULT_TOPK_TOKENS: int = 2048


def publish_candidate_blocks(
    topk_indices: Tensor,
    positions: Tensor,
    query_end: Tensor,
    compress_ratio: int,
    candidate_block_size: int,
    candidate_topk_blocks: int,
) -> Tensor:
    """Select up to ``candidate_topk_blocks`` deduplicated block IDs per token.

    When total valid compressed blocks ``num_blocks <= candidate_topk_blocks``,
    publishes ``0..num_blocks-1`` directly with ``-1`` padding to prevent sparse
    candidate starvation. Otherwise pins the causal tail block
    ``(compressed_len - 1) // candidate_block_size`` first and fills the
    remaining ``candidate_topk_blocks - 1`` slots with sorted unique blocks from
    ``topk_indices``.

    Args:
        topk_indices: ``int32[num_tokens, topk]`` selected key indices (Used).
        positions: ``int32[num_tokens]`` token positions (Used).
        query_end: Scalar ``int32`` active token limit ``query_start_loc[-1]`` (Used).
        compress_ratio: KV compression ratio (``1`` or ``2``).
        candidate_block_size: Number of compressed keys per candidate block.
        candidate_topk_blocks: Output width of deduplicated candidate block IDs.

    Returns:
        ``int32[num_tokens, candidate_topk_blocks]`` candidate block IDs (Created).
    """
    num_tokens = topk_indices.shape[0]
    device = topk_indices.device
    token_rows = torch.arange(num_tokens, device=device)
    active = (token_rows < query_end.to(device)) & (positions >= 0)
    compressed_len = torch.where(active, (positions.int() + 1) // compress_ratio, 0)
    num_blocks = (compressed_len + candidate_block_size - 1) // candidate_block_size
    iota = torch.arange(candidate_topk_blocks, dtype=torch.int32, device=device)
    iota_2d = iota.unsqueeze(0)
    all_blocks = torch.where(iota_2d < num_blocks.unsqueeze(1), iota_2d, -1)

    tail_block = torch.where(
        compressed_len > 0, (compressed_len - 1) // candidate_block_size, -1
    ).unsqueeze(1)
    raw_blocks = torch.where(
        topk_indices >= 0, topk_indices // candidate_block_size, -1
    )
    non_tail = torch.where(
        (raw_blocks >= 0) & (raw_blocks != tail_block),
        raw_blocks,
        _SENTINEL_BLOCK_ID,
    )
    sorted_blocks, _ = non_tail.sort(dim=1)
    is_duplicate = F.pad(
        sorted_blocks[:, 1:] == sorted_blocks[:, :-1], (1, 0), value=False
    )
    unique_sorted, _ = torch.where(
        is_duplicate, _SENTINEL_BLOCK_ID, sorted_blocks
    ).sort(dim=1)
    unique_blocks = torch.where(unique_sorted < _SENTINEL_BLOCK_ID, unique_sorted, -1)

    pad_cols = max(0, candidate_topk_blocks - 1 - unique_blocks.shape[1])
    if pad_cols > 0:
        padding = unique_blocks.new_full((num_tokens, pad_cols), -1)
        merged = torch.cat((tail_block, unique_blocks, padding), dim=1)
    else:
        merged = torch.cat((tail_block, unique_blocks), dim=1)
    dedup_blocks = merged[:, :candidate_topk_blocks]
    use_all = (num_blocks <= candidate_topk_blocks).unsqueeze(1)
    return torch.where(use_all, all_blocks, dedup_blocks)


# pyrefly: ignore[inconsistent-inheritance]
class IndexerCache(NativeCacheView, upstream.DeepseekV4IndexerCache):
    """Unpacked ``bfloat16`` indexer key cache backed by ``NativeCacheView``."""

    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ) -> None:
        if head_dim != INDEX_HEAD_DIM or dtype != torch.bfloat16:
            raise ValueError(
                f"IndexerCache requires head_dim={INDEX_HEAD_DIM} and "
                f"dtype=torch.bfloat16, got head_dim={head_dim}, dtype={dtype}"
            )
        upstream.DeepseekV4IndexerCache.__init__(
            self,
            head_dim=head_dim,
            dtype=dtype,
            prefix=prefix,
            cache_config=cache_config,
            compress_ratio=compress_ratio,
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        spec = cast(MLAAttentionSpec, super().get_kv_cache_spec(vllm_config))
        return replace(spec, page_size_padded=None)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AttentionBackend


class TpuSparseAttnIndexer(nn.Module):
    """Callable TPU sparse-indexer operator bound as ``DeepseekV41Indexer.attn_cls``.

    State coupling: ``DeepseekV41Indexer.__init__`` binds ``self.select_topk``
    onto ``_select_topk`` because upstream ``DeepseekV4Attention`` dispatches
    top-K selection through ``self.indexer.indexer_op(...)``. Calling an unbound
    ``TpuSparseAttnIndexer`` fails fast with ``RuntimeError``.
    """

    def __init__(
        self,
        k_cache: IndexerCache,
        quant_block_size: int = _DEFAULT_QUANT_BLOCK_SIZE,
        scale_fmt: str | None = "ue8m0",
        topk_tokens: int = _DEFAULT_TOPK_TOKENS,
        head_dim: int = INDEX_HEAD_DIM,
        max_model_len: int = 0,
        max_total_seq_len: int = 0,
        topk_indices_buffer: Tensor | None = None,
        prefix: str = "",
        *,
        skip_k_cache_insert: bool = True,
        use_fp4_cache: bool = False,
        compress_ratio: int = 1,
        candidate_blocks: Tensor | None = None,
        candidate_block_size: int = 0,
        candidate_write: bool = False,
    ) -> None:
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.prefix = prefix
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        self.compress_ratio = compress_ratio
        self.candidate_blocks = candidate_blocks
        self.candidate_block_size = candidate_block_size
        self.candidate_write = candidate_write
        self._select_topk: Callable[[Tensor, Tensor, Tensor | None], Tensor] | None = (
            None
        )

    def forward(
        self,
        hidden_states: Tensor,
        q_quant: Tensor | tuple[Tensor, Tensor | None],
        positions: Tensor | None,
        weights: Tensor,
    ) -> None:
        del hidden_states
        if self._select_topk is None:
            raise RuntimeError("TpuSparseAttnIndexer._select_topk callback is unbound")
        q_tensor = q_quant[0] if isinstance(q_quant, tuple) else q_quant
        self._select_topk(q_tensor, weights, positions)


class DeepseekV41Indexer(upstream.DeepseekV4Indexer):
    """TPU sparse-attention indexer using ``dsv41_paged_indexer_topk``.

    State lifecycle & descriptor sharing:
        - ``__init__`` attaches ``self.select_topk`` to ``self.indexer_op._select_topk``
          for upstream ``indexer_op(...)`` dispatch.
        - ``forward()`` records ``_last_positions`` for the active step so callers
          invoking ``select_topk(query, weights)`` without ``positions`` resolve the
          current step's token positions; calling ``select_topk`` before ``forward()``
          without ``positions`` raises ``ValueError``.
        - ``kv_rope_op`` reuses ``VllmDeepseekV4MLAAttention.kv_rope_op`` (which
          stores ``_kv_rope_op_instance`` in ``self.__dict__``) so ``deepseek_v41``
          model modules stay free of direct ``jax`` imports.
    """

    attn_cls: ClassVar[type[Any]] = TpuSparseAttnIndexer
    # pyrefly: ignore[bad-override-mutable-attribute]
    kv_rope_op = cast(
        Callable[[Tensor, Tensor, Tensor], Tensor],
        v4_attn.VllmDeepseekV4MLAAttention.kv_rope_op,
    )
    # pyrefly: ignore[bad-override-mutable-attribute]
    indexer_op: TpuSparseAttnIndexer

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config | Any,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: Tensor | None,
        compress_ratio: int,
        prefix: str,
        *,
        owns_k: bool,
        k_cache: IndexerCache,
        main_head_dim: int,
        candidate_block_buffer: Tensor | None = None,
        candidate_block_size: int = 0,
        candidate_write: bool = False,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            config=config,
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            quant_config=quant_config,
            cache_config=cache_config,
            topk_indices_buffer=topk_indices_buffer,
            compress_ratio=compress_ratio,
            prefix=prefix,
            owns_k=owns_k,
            k_cache=k_cache,
            main_head_dim=main_head_dim,
            candidate_block_buffer=candidate_block_buffer,
            candidate_block_size=candidate_block_size,
            candidate_write=candidate_write,
        )
        self.window_size: int = vllm_config.model_config.hf_config.sliding_window
        self.indexer_op._select_topk = self.select_topk
        self._last_positions: Tensor | None = None

    def _produce_k(
        self,
        latent: Tensor | None,
        positions: Tensor,
        rotary_emb: nn.Module,
    ) -> None:
        """Project, normalize, rotate, and store compressor latents as index keys."""
        if latent is None or not self.owns_k:
            return
        k_cache = active_cache(self.k_cache)
        meta = layer_meta(self.k_cache.prefix)
        if meta is None or not k_cache.numel():
            return
        ratio = self.compress_ratio
        k_pre, _ = self.wk(latent)
        k_rotated = rotate_rope(
            self.k_norm(k_pre),
            positions // ratio,
            rotary_emb,
            self.kv_rope_op,
        )
        slots = cache_slots(k_cache, meta, positions, compress_ratio=ratio)
        keep = ((positions + 1) % ratio == 0) & not_replayed(
            meta, positions, self.window_size
        )
        write_cache(k_cache, slots, k_rotated, keep)

    # pyrefly: ignore[bad-override]
    def forward_q(
        self,
        q_lora: Tensor,
        q_lora_scale: Tensor | None,
        indexer_weights: Tensor,
        positions: Tensor,
        rotary_emb: nn.Module,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        """Project ``q_lora`` to split-lane BF16 queries and scaled FP32 weights.

        Args:
            q_lora: ``[num_tokens, q_lora_rank]`` query LoRA activations (Used).
            q_lora_scale: Optional quantization scale for ``q_lora`` (Used).
            indexer_weights: ``[num_tokens, n_head]`` per-head weights (Used).
            positions: ``int32[num_tokens]`` token positions (Used).
            rotary_emb: Rotary embedding module for query rotation (Used).

        Returns:
            Tuple ``(q_split, None, scaled_weights)`` (Created).
        """
        query_heads = self._wq_b_proj(q_lora, q_lora_scale).view(
            -1, self.n_head, self.head_dim
        )
        q_rotated = rotate_rope(query_heads, positions, rotary_emb, self.kv_rope_op)
        q_split = torch.cat((q_rotated[..., 0::2], q_rotated[..., 1::2]), dim=-1)
        scale = self.softmax_scale * (self.n_head**-0.5)
        scaled_weights = (indexer_weights.float() * scale).unsqueeze(-1)
        return q_split, None, scaled_weights

    # pyrefly: ignore[bad-override]
    def forward(
        self,
        q_lora: Tensor,
        latent: Tensor | None,
        indexer_weights: Tensor,
        positions: Tensor,
        rotary_emb: nn.Module,
        q_lora_scale: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Publish index keys when owning ``k_cache`` and return queries/weights.

        Args:
            q_lora: ``[num_tokens, q_lora_rank]`` query LoRA activations (Used).
            latent: Optional ``[num_tokens, main_head_dim]`` compressor latent (Used).
            indexer_weights: ``[num_tokens, n_head]`` per-head weights (Used).
            positions: ``int32[num_tokens]`` token positions (Used).
            rotary_emb: Rotary embedding module (Used).
            q_lora_scale: Optional quantization scale for ``q_lora`` (Used).

        Returns:
            Tuple ``(q_split, None, scaled_weights)`` (Created).
        """
        if self.owns_k and latent is not None:
            self._produce_k(latent, positions, rotary_emb)
        self._last_positions = positions
        return self.forward_q(
            q_lora, q_lora_scale, indexer_weights, positions, rotary_emb
        )

    def select_topk(
        self,
        query: Tensor,
        weights: Tensor,
        positions: Tensor | None = None,
    ) -> Tensor:
        """Score paged index keys via ``dsv41_paged_indexer_topk`` and write top-K.

        Args:
            query: ``bfloat16[num_tokens, n_head, head_dim]`` split-lane queries (Used).
            weights: ``float32[num_tokens, n_head, 1]`` scaled head weights (Used).
            positions: Optional ``int32[num_tokens]`` positions; falls back to
                ``_last_positions`` recorded in :meth:`forward`.

        Returns:
            ``int32[num_tokens, topk_tokens]`` selected key indices (Created).

        Raises:
            ValueError: If ``positions`` is ``None`` and :meth:`forward` has not run.
        """
        resolved_pos = self._last_positions if positions is None else positions
        if resolved_pos is None:
            raise ValueError(
                "DeepseekV41Indexer.select_topk requires positions or a prior forward()"
            )
        num_tokens = query.shape[0]
        k_cache = active_cache(self.k_cache)
        meta = layer_meta(self.k_cache.prefix)
        if meta is None or not k_cache.numel():
            topk = query.new_full((num_tokens, self.topk_tokens), -1, dtype=torch.int32)
        else:
            candidate_indices: Tensor | None = None
            indexer_op_module = self.indexer_op
            table = block_table(meta)
            candidate_blocks = indexer_op_module.candidate_blocks
            candidate_block_size = indexer_op_module.candidate_block_size
            if isinstance(k_cache, PagedCacheRows):
                records_per_page = k_cache.shape[1]
                page_rows = max(1, (records_per_page * k_cache.row_bytes) // ROW_BYTES)
                k_pool_lo = k_cache.raw.view(-1, page_rows, ROW_WORDS)
                raw_hi = k_cache.raw_hi
                k_pool_hi = (
                    None if raw_hi is None else raw_hi.view(-1, page_rows, ROW_WORDS)
                )
                page_stride = k_cache.page_stride // records_per_page
                page_offset = k_cache.row_offset // records_per_page
                min_block = 1 if k_cache.null_block else 0
                phys_block_tables = torch.where(
                    table < min_block,
                    -1,
                    table.int() * page_stride + page_offset,
                )
            else:
                k_pool_lo = k_cache
                k_pool_hi = None
                phys_block_tables = table.int()

            max_keys = _KEYS_PER_WORD_ROW * table.shape[1] * k_pool_lo.shape[1]
            if (
                candidate_blocks is not None
                and not indexer_op_module.candidate_write
                and candidate_block_size > 0
                and max_keys > candidate_blocks.shape[1] * candidate_block_size
            ):
                block_ids = candidate_blocks[:num_tokens].unsqueeze(-1)
                offsets = torch.arange(
                    candidate_block_size, dtype=torch.int32, device=block_ids.device
                )
                expanded = torch.where(
                    block_ids >= 0,
                    block_ids * candidate_block_size + offsets,
                    -1,
                )
                candidate_indices = expanded.view(num_tokens, -1)

            seq_lens = cast(Tensor, meta.seq_lens).int()
            cu_q_lens = cast(Tensor, meta.query_start_loc).int()
            topk = indexer_op.dsv41_paged_indexer_topk(
                query,
                weights,
                k_pool_lo,
                phys_block_tables,
                seq_lens,
                cu_q_lens,
                self.topk_tokens,
                k_pool_hi,
                candidate_indices,
                self.compress_ratio,
            )
            if (
                candidate_blocks is not None
                and indexer_op_module.candidate_write
                and candidate_block_size > 0
            ):
                candidate_topk_blocks = candidate_blocks.shape[1]
                published = publish_candidate_blocks(
                    topk,
                    resolved_pos,
                    cu_q_lens[-1],
                    self.compress_ratio,
                    candidate_block_size,
                    candidate_topk_blocks,
                )
                candidate_blocks[:num_tokens].copy_(published)

        if self.topk_indices_buffer is not None:
            self.topk_indices_buffer[:num_tokens].copy_(topk)
        return topk
