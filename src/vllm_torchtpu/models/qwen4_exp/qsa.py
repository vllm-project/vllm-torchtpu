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
"""QSA (query-selected sparse attention) PyTorch modules for Qwen4Exp.

Follows the HF reference (``Qwen4ExpTextQSAIndexer``, ``Qwen4ExpTextAttention``).
The sparse kernel op lives in ``ops/qsa.py`` so optimized Pallas/Tokamax
backends can be swapped in without changing model or weight wiring.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn
from transformers import Qwen3NextConfig, Qwen4ExpTextConfig
from vllm.config import CacheConfig, ModelConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_torchtpu.models.qwen4_exp.ops.qsa import qsa_attention


class Qwen4ExpQSAAttention(Qwen3NextAttention):
    """Qwen3-Next gated attention with the QSA indexer and sparse path."""

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        *,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        # The Qwen3-Next layer reads only fields the Qwen4Exp text config also
        # has; the transformers class just does not subclass Qwen3NextConfig.
        super().__init__(
            cast(Qwen3NextConfig, config),
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        ratio = config.indexer_compress_ratio
        budget = config.indexer_budget
        index_head_dim = config.indexer_head_dim
        assert ratio is not None and budget is not None and index_head_dim is not None
        topk_blocks = budget // ratio
        # Below this length every visible token fits in topk_blocks + open tail,
        # so dense paged attention is already exact.
        self.dense_max_model_len = (topk_blocks + 1) * ratio - 1
        self.sparse = model_config.max_model_len > self.dense_max_model_len
        if not self.sparse:
            return
        if (
            cache_config is not None
            and cache_config.get_resolved_kv_cache_layout() is KVCacheLayout.LBHNC
        ):
            raise NotImplementedError(
                "Qwen4Exp sparse QSA requires NHD (LBNHC) KV cache layout, "
                "got HND (LBHNC)."
            )
        if index_head_dim > self.head_dim:
            raise NotImplementedError(
                f"QSA stores index keys in a head_dim={self.head_dim} cache; "
                f"indexer_head_dim={index_head_dim} does not fit."
            )
        self.indexer = Qwen4ExpQSAIndexer(
            config, quant_config, prefix=f"{prefix}.indexer"
        )
        # Same spec as self.attn, so the index keys share its cache group and
        # block table. Its forward is never called: qsa_attention writes it.
        # TODO(QSA): this doubles KV memory for QSA layers; replace with a
        # compact/packed index cache spec so 256k workloads do not pay 2x KV.
        self.index_cache = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.index_cache",
        )
        self.compress_ratio = ratio
        self.topk_blocks = topk_blocks
        self.norm_eps = config.rms_norm_eps

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        # Bring-up trade-off: self.attn populates main_pool and supplies dense_out
        # for positions <= dense_max_model_len; _sparse_attention overwrites tokens
        # past that threshold. TODO(QSA): replace with a KV-write-only + Pallas QSA
        # kernel to avoid redundant dense attention on sparse steps.
        attn_output = self.attn(q, k, v)
        main_pool = self.attn.kv_cache
        if self.sparse and main_pool.numel() > 0:
            attn_output = self._sparse_attention(
                positions, hidden_states, q, attn_output, main_pool
            )
        assert gate is not None
        attn_output = attn_output * torch.sigmoid(gate)
        output, _ = self.o_proj(attn_output)
        return output

    def _sparse_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        q: torch.Tensor,
        dense_out: torch.Tensor,
        main_pool: torch.Tensor,
    ) -> torch.Tensor:
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        index_q, raw_keys = self.indexer(hidden_states, positions, cos_sin_cache)
        attn_metadata = get_forward_context().attn_metadata
        assert isinstance(attn_metadata, dict)
        metadata: Any = attn_metadata[self.attn.layer_name]
        kv_scales = torch.stack(
            [
                cast(torch.Tensor, self.attn._k_scale),
                cast(torch.Tensor, self.attn._v_scale),
            ]
        ).float()
        out = qsa_attention(
            q=q.view(-1, self.num_heads, self.head_dim),
            index_q=index_q,
            raw_keys=raw_keys,
            dense_out=dense_out.view(-1, self.num_heads, self.head_dim),
            positions=positions,
            query_start_loc=metadata.query_start_loc,
            seq_lens=metadata.seq_lens,
            block_tables=metadata.block_tables,
            main_pool=main_pool,
            index_pool=self.index_cache.kv_cache,
            cos_sin_cache=cos_sin_cache,
            k_norm_weight=self.indexer.k_layernorm.weight,
            kv_scales=kv_scales,
            num_kv_heads=self.num_kv_heads,
            compress_ratio=self.compress_ratio,
            topk_blocks=self.topk_blocks,
            scale=self.scaling,
            norm_eps=self.norm_eps,
        )
        return out.reshape(dense_out.shape)


class Qwen4ExpQSAIndexer(nn.Module):
    """Index queries and raw index keys; HF's ``Qwen4ExpTextQSAIndexer``."""

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        if config.indexer_kv_heads != 1:
            raise NotImplementedError(
                f"QSA supports indexer_kv_heads=1, got {config.indexer_kv_heads}."
            )
        assert (
            config.indexer_n_heads is not None and config.indexer_head_dim is not None
        )
        self.num_heads: int = config.indexer_n_heads
        self.head_dim: int = config.indexer_head_dim
        self.index_qk_proj = ReplicatedLinear(
            config.hidden_size,
            (self.num_heads + 1) * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.index_qk_proj",
        )
        self.q_layernorm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(normed, roped index queries [T, heads, dim], raw keys [T, dim])."""
        qk, _ = self.index_qk_proj(hidden_states)
        q, raw_keys = qk.split([self.num_heads * self.head_dim, self.head_dim], -1)
        q = self.q_layernorm(q.reshape(-1, self.num_heads, self.head_dim))
        return _apply_partial_rope(q, positions, cos_sin_cache), raw_keys


def _apply_partial_rope(
    x: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor
) -> torch.Tensor:
    """NeoX RoPE on the first ``cos_sin_cache.shape[-1]`` dims of ``x``.

    ``x`` is [num_tokens, heads, dim]; ``cos_sin_cache`` [max_pos, rotary_dim]
    holds cos then sin, as vLLM's RotaryEmbedding builds it.
    """
    rotary_dim = cos_sin_cache.shape[-1]
    cos, sin = cos_sin_cache[positions].float().unsqueeze(1).chunk(2, dim=-1)
    rot, rest = x[..., :rotary_dim].float(), x[..., rotary_dim:]
    x1, x2 = rot.chunk(2, dim=-1)
    rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return torch.cat([rot.to(x.dtype), rest], dim=-1)
