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
"""MiniMax Sparse Attention layer on TPU."""

import torch
from torch import nn
from transformers import PreTrainedConfig
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MinimaxM3QKVParallelLinearWithIndexer,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackend
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.core.owned_kv_cache import TpuOwnedKVCache
from vllm_torchtpu.models.minimax_m3.ops.msa import get_msa_op


class MiniMaxM3IndexCache(nn.Module, TpuOwnedKVCache):
    """Key-only cache holding one index key per token."""

    def __init__(self, head_dim: int, dtype: torch.dtype, prefix: str) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.dtype = dtype
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        static_forward_context = get_current_vllm_config().compilation_config
        static_forward_context = static_forward_context.static_forward_context
        if prefix in static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # MLAAttentionSpec budgets one vector per token, not K plus V.
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return PallasMLAttentionBackend


def _per_head_norm(x: torch.Tensor, norm: GemmaRMSNorm, head_dim: int) -> torch.Tensor:
    return norm(x.reshape(-1, head_dim)).reshape(x.shape)


class MiniMaxM3SparseAttention(nn.Module):
    """GQA attention over the KV blocks the lightning indexer selects.

    One fused GEMM emits ``[q | k | v | index_q | index_k]``. All four
    projections get a per-head Gemma RMSNorm and partial NeoX RoPE. The main
    K/V cache is owned by a plain `Attention` so it is allocated and
    described like any GQA cache; the index keys live in `index_cache`.
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        sparse_cfg = config.sparse_attention_config
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        self.head_dim = config.head_dim
        self.index_head_dim = sparse_cfg["sparse_index_dim"]
        assert sparse_cfg["sparse_num_index_heads"] == config.num_key_value_heads
        assert self.index_head_dim == self.head_dim
        assert sparse_cfg["sparse_score_type"] == "max", sparse_cfg
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        scaling = self.head_dim**-0.5

        self.qkv_proj = MinimaxM3QKVParallelLinearWithIndexer(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            config.num_key_value_heads,
            sparse_cfg["sparse_num_index_heads"],
            self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        eps = config.rms_norm_eps
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=eps)
        self.index_q_norm = GemmaRMSNorm(self.index_head_dim, eps=eps)
        self.index_k_norm = GemmaRMSNorm(self.index_head_dim, eps=eps)
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters={
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": config.partial_rotary_factor,
            },
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.index_cache = MiniMaxM3IndexCache(
            self.index_head_dim,
            torch.get_default_dtype(),
            prefix=f"{prefix}.attn.index_cache",
        )
        self.msa_op = get_msa_op(
            scaling,
            sparse_cfg["sparse_topk_blocks"],
            sparse_cfg.get("sparse_init_block", 0),
            sparse_cfg.get("sparse_local_block", 0),
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, index_q, index_k = qkv.split(
            [
                self.q_size,
                self.kv_size,
                self.kv_size,
                self.kv_size,
                self.index_head_dim,
            ],
            dim=-1,
        )
        q = _per_head_norm(q, self.q_norm, self.head_dim)
        k = _per_head_norm(k, self.k_norm, self.head_dim)
        index_q = _per_head_norm(index_q, self.index_q_norm, self.index_head_dim)
        index_k = self.index_k_norm(index_k)
        q, k = self.rotary_emb(positions, q, k)
        index_q, index_k = self.rotary_emb(positions, index_q, index_k)

        num_tokens = hidden_states.shape[0]
        metadata = get_forward_context().attn_metadata
        kv_cache = self.attn.kv_cache
        if not isinstance(metadata, dict) or kv_cache.numel() == 0:
            # Memory profiling runs before the caches are bound.
            attn_output = q
        else:
            layer_metadata = metadata[self.attn.layer_name]
            assert isinstance(layer_metadata, AttentionMetadata)
            attn_output = self.msa_op(
                q.view(num_tokens, self.num_heads, self.head_dim),
                k.view(num_tokens, self.num_kv_heads, self.head_dim),
                v.view(num_tokens, self.num_kv_heads, self.head_dim),
                index_q.view(num_tokens, self.num_kv_heads, self.index_head_dim),
                index_k,
                kv_cache,
                self.index_cache.kv_cache,
                layer_metadata,
            ).reshape(num_tokens, self.q_size)
        output, _ = self.o_proj(attn_output)
        return output
