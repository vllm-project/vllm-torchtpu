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
"""QSA (query-selected sparse attention) core module for Qwen4Exp.

Owns the checkpoint indexer weights (``index_qk_proj``, ``q_layernorm``,
``k_layernorm``) and ``index_cache`` pool registration; all RoPE and sparse
compute live in ``ops/qsa.py``. Attached as ``self.indexer`` on
``Qwen4ExpAttention`` so checkpoint paths ``self_attn.indexer.*`` match HF.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from transformers import Qwen4ExpTextConfig
from vllm.config import CacheConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_torchtpu.models.qwen4_exp.layout import (
    QKVLayout,
    TensorLayout,
    unfused_o,
    unfused_qkv,
)
from vllm_torchtpu.models.qwen4_exp.ops.qsa import qsa_attention


def dense_max_model_len(config: Qwen4ExpTextConfig) -> int:
    """Longest sequence where every token fits in topk_blocks + open tail."""
    ratio = config.indexer_compress_ratio
    budget = config.indexer_budget
    assert ratio is not None and budget is not None
    topk_blocks = budget // ratio
    return (topk_blocks + 1) * ratio - 1


def qsa_input_layouts(
    dtype: torch.dtype, kv_layout: KVCacheLayout
) -> tuple[QKVLayout, ...]:
    """(q, kv) input layouts accepted by ``Qwen4ExpQSA.forward``."""
    if kv_layout is KVCacheLayout.LBHNC:
        return ()
    return (unfused_qkv(dtype),)


def qsa_output_layouts(
    dtype: torch.dtype, kv_layout: KVCacheLayout
) -> tuple[TensorLayout, ...]:
    """Attention output layouts produced by ``Qwen4ExpQSA.forward``."""
    if kv_layout is KVCacheLayout.LBHNC:
        return ()
    return (unfused_o(dtype),)


class Qwen4ExpQSA(nn.Module):
    """Sparse QSA core: indexer projections, index KV pool, and op dispatch."""

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scaling: float,
        qkv_layout: QKVLayout,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        if config.indexer_kv_heads != 1:
            raise NotImplementedError(
                f"QSA supports indexer_kv_heads=1, got {config.indexer_kv_heads}."
            )
        ratio = config.indexer_compress_ratio
        budget = config.indexer_budget
        index_head_dim = config.indexer_head_dim
        assert (
            config.indexer_n_heads is not None
            and index_head_dim is not None
            and ratio is not None
            and budget is not None
        )
        if index_head_dim > head_dim:
            raise NotImplementedError(
                f"QSA stores index keys in a head_dim={head_dim} cache; "
                f"indexer_head_dim={index_head_dim} does not fit."
            )
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = scaling
        self.qkv_layout = qkv_layout
        self.index_num_heads: int = config.indexer_n_heads
        self.index_head_dim: int = index_head_dim
        self.compress_ratio: int = ratio
        self.topk_blocks: int = budget // ratio
        self.norm_eps: float = config.rms_norm_eps

        self.index_qk_proj = ReplicatedLinear(
            config.hidden_size,
            (self.index_num_heads + 1) * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer.index_qk_proj",
        )
        self.q_layernorm = GemmaRMSNorm(self.index_head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = GemmaRMSNorm(self.index_head_dim, eps=config.rms_norm_eps)
        # Same spec as main_attn so index keys share its cache group and block
        # table. Its forward is never called: ops/qsa.py writes it directly.
        self.index_cache = Attention(
            num_heads,
            head_dim,
            scaling,
            num_kv_heads=num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.index_cache",
        )

    def forward(
        self,
        *,
        q: torch.Tensor,
        kv: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        main_pool: torch.Tensor,
        metadata: Any,
        kv_scales: torch.Tensor | None,
        cos_sin_cache: torch.Tensor,
        dense_attn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Runs QSA core attention (KV update + indexer + sparse attend).

        Accepts ``(q, kv)`` in ``self.qkv_layout`` and returns ``o`` in
        ``qsa_output_layouts`` so switching QSA to packed layouts or Pallas KV
        writes stays confined to this module.
        """
        if not isinstance(kv, tuple):
            raise NotImplementedError(
                f"Qwen4ExpQSA does not yet accept packed KV layout {self.qkv_layout}."
            )
        k, v = kv
        q2d = q.reshape(-1, self.num_heads * self.head_dim)
        k2d = k.reshape(-1, self.num_kv_heads * self.head_dim)
        v2d = v.reshape(-1, self.num_kv_heads * self.head_dim)
        dense_out = dense_attn(q2d, k2d, v2d).view(-1, self.num_heads, self.head_dim)
        if main_pool.numel() == 0:
            return dense_out
        assert metadata is not None and kv_scales is not None
        qk, _ = self.index_qk_proj(hidden_states)
        index_q, raw_keys = qk.split(
            [self.index_num_heads * self.index_head_dim, self.index_head_dim], -1
        )
        index_q = self.q_layernorm(
            index_q.reshape(-1, self.index_num_heads, self.index_head_dim)
        )
        return qsa_attention(
            q=q.reshape(-1, self.num_heads, self.head_dim),
            index_q=index_q,
            raw_keys=raw_keys,
            dense_out=dense_out,
            positions=positions,
            query_start_loc=metadata.query_start_loc,
            seq_lens=metadata.seq_lens,
            block_tables=metadata.block_tables,
            main_pool=main_pool,
            index_pool=self.index_cache.kv_cache,
            cos_sin_cache=cos_sin_cache,
            k_norm_weight=self.k_layernorm.weight,
            kv_scales=kv_scales,
            num_kv_heads=self.num_kv_heads,
            compress_ratio=self.compress_ratio,
            topk_blocks=self.topk_blocks,
            scale=self.scaling,
            norm_eps=self.norm_eps,
        )
