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
from vllm import ir
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.forward_context import get_forward_context
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
from vllm_torchtpu.models.minimax_m3.ops.msa import (
    get_msa_dcp_op,
    get_msa_op,
    get_msa_sharded_op,
)


class MiniMaxM3Cache(nn.Module, TpuOwnedKVCache):
    """One row per token in the packed MLA page layout.

    Holds ``[K | V]`` rows for the attention cache and index keys for the
    indexer cache; only the MSA kernels read it, so no attention backend's
    layout is involved. With ``token_shards > 1`` a page holds only every
    ``token_shards``-th token of its block (the index-cache shard of one
    tensor-parallel rank), packed as ``[pages, S / shards / p, p, W]``.
    Under DCP a page holds whole blocks: vLLM's logical block ``j`` of
    ``dcp * S`` tokens is block ``j * dcp + r`` on DCP rank ``r``
    (`kernels.msa_dcp`); the spec is per rank and DCP-sharded, so the
    scheduler counts ``dcp`` times the tokens per page. A DCP group wider
    than a KV head's replicas stores all of its ``G`` KV heads per row,
    ``[K_0 | V_0 | ... | K_G-1 | V_G-1]``.
    """

    def __init__(
        self, row_width: int, dtype: torch.dtype, prefix: str, token_shards: int = 1
    ) -> None:
        super().__init__()
        self.row_width = row_width
        self.dtype = dtype
        self.prefix = prefix
        self.token_shards = token_shards
        self.kv_cache = torch.tensor([])
        vllm_config = get_current_vllm_config()
        self.block_size = vllm_config.cache_config.block_size
        if self.block_size % token_shards:
            raise ValueError(
                f"block size {self.block_size} is not a multiple of the "
                f"{token_shards} index-cache shards"
            )
        static_forward_context = vllm_config.compilation_config
        static_forward_context = static_forward_context.static_forward_context
        if prefix in static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # MLAAttentionSpec budgets one row of row_width per token; a shard
        # stores 1 / token_shards of the rows, budgeted as a narrower row.
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.row_width // self.token_shards,
            dtype=self.dtype,
        )

    def kv_cache_shape(self, num_blocks: int) -> tuple[int, ...] | None:
        if self.token_shards == 1:
            return None
        packing = 32 // (8 * self.dtype.itemsize)
        tokens = self.block_size // self.token_shards
        return (num_blocks, tokens // packing, packing, self.row_width)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return PallasMLAttentionBackend


# --additional-config key: shard the MSA index-K cache over the TP group
# (`kernels.msa` index-cache sharding). Part of the compile-cache key.
SHARD_INDEX_CACHE_KEY = "minimax_shard_index_cache"


def shard_index_cache_enabled(vllm_config: VllmConfig) -> bool:
    config = vllm_config.additional_config
    if not isinstance(config, dict):
        return False
    enabled = config.get(SHARD_INDEX_CACHE_KEY, False)
    if not isinstance(enabled, bool):
        raise ValueError(f"additional_config.{SHARD_INDEX_CACHE_KEY} must be a bool")
    return enabled


def cache_dtype(cache_config: CacheConfig | None) -> torch.dtype:
    """Storage dtype of both MSA caches for ``--kv-cache-dtype``."""
    name = cache_config.cache_dtype if cache_config is not None else "auto"
    if name in ("auto", "bfloat16"):
        return torch.bfloat16
    if name in ("fp8", "fp8_e4m3"):
        return torch.float8_e4m3fn
    raise NotImplementedError(
        f"MiniMax M3 on TPU supports --kv-cache-dtype auto, bfloat16 or fp8, "
        f"got {name!r}."
    )


def norm_rope_heads(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    norms: tuple[GemmaRMSNorm, GemmaRMSNorm, GemmaRMSNorm, GemmaRMSNorm],
    rotary_emb: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split ``[q | k | v | index_q | index_k]`` and norm + rotate all but v.

    The four Gemma RMSNorms differ only in their weights and RoPE acts on each
    head alone, so one norm over the stacked q, k, index_q and index_k heads
    with a per-head ``1 + w`` table, then one rotation, gives the same values
    as four norms and two rotations in fewer, larger ops.

    Returns ``(q, k, v, index_q, index_k)`` with shapes ``[T, num_heads * D]``,
    ``[T, num_kv_heads * D]`` (x3) and ``[T, D]``.
    """
    num_tokens = qkv.shape[0]
    heads = qkv.view(num_tokens, -1, head_dim)
    nq, nkv = num_heads, num_kv_heads
    v = heads[:, nq + nkv : nq + 2 * nkv]
    x = torch.cat([heads[:, : nq + nkv], heads[:, nq + 2 * nkv :]], dim=1)
    q_norm, k_norm, index_q_norm, index_k_norm = norms
    weight = (
        torch.cat(
            [
                q_norm.weight.expand(nq, head_dim),
                k_norm.weight.expand(nkv, head_dim),
                index_q_norm.weight.expand(nkv, head_dim),
                index_k_norm.weight.expand(1, head_dim),
            ]
        ).float()
        + 1.0
    )
    eps = q_norm.variance_epsilon
    assert all(n.variance_epsilon == eps for n in norms)
    x = ir.ops.rms_norm(x, weight, eps)
    x, _ = rotary_emb(positions, x.reshape(num_tokens, -1), None)
    x = x.view(num_tokens, -1, head_dim)
    q, k, index_q, index_k = x.split([nq, nkv, nkv, 1], dim=1)
    return (
        q.reshape(num_tokens, -1),
        k.reshape(num_tokens, -1),
        v.reshape(num_tokens, -1),
        index_q.reshape(num_tokens, -1),
        index_k.reshape(num_tokens, -1),
    )


class MiniMaxM3SparseAttention(nn.Module):
    """GQA attention over the KV blocks the lightning indexer selects.

    One fused GEMM emits ``[q | k | v | index_q | index_k]``. All four
    projections get a per-head Gemma RMSNorm and partial NeoX RoPE. K/V rows
    live in `kv_cache`, index keys in `index_cache`.
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
        # With one index head per rank, each rank can keep a 1 / tp share of
        # the index keys (`kernels.msa` index-cache sharding).
        shard = shard_index_cache_enabled(get_current_vllm_config())
        self.index_shards = tp_size if shard else 1
        if self.index_shards > 1 and tp_size != config.num_key_value_heads:
            raise NotImplementedError(
                f"additional_config.{SHARD_INDEX_CACHE_KEY} needs one index head "
                f"per rank (tensor parallel size {config.num_key_value_heads}), "
                f"got {tp_size}."
            )
        # Decode context parallelism: the ranks sharing this KV head split its
        # context by whole sparse blocks (`kernels.msa_dcp`).
        vllm_config = get_current_vllm_config()
        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        if self.dcp_size > 1 and self.index_shards > 1:
            raise NotImplementedError(
                f"additional_config.{SHARD_INDEX_CACHE_KEY} does not combine with "
                "--decode-context-parallel-size."
            )
        # TP ranks per KV head; a DCP group wider than that spans several KV
        # heads, and each rank stores all of their rows for its blocks.
        self.kv_head_replicas = max(1, tp_size // config.num_key_value_heads)
        self.dcp_kv_heads = max(1, self.dcp_size // self.kv_head_replicas)
        dtype = cache_dtype(cache_config)
        self.kv_cache = MiniMaxM3Cache(
            2 * self.head_dim * self.dcp_kv_heads, dtype, prefix=f"{prefix}.attn"
        )
        self.index_cache = MiniMaxM3Cache(
            self.index_head_dim,
            dtype,
            prefix=f"{prefix}.attn.index_cache",
            token_shards=self.index_shards,
        )
        msa_config = (
            scaling,
            sparse_cfg["sparse_topk_blocks"],
            sparse_cfg.get("sparse_init_block", 0),
            sparse_cfg.get("sparse_local_block", 0),
        )
        if self.dcp_size > 1:
            from vllm_torchtpu.distributed.dcp import get_dcp_rank, get_dcp_world_size

            if get_dcp_world_size() != self.dcp_size:
                raise RuntimeError(
                    f"DCP group of {get_dcp_world_size()} ranks, expected "
                    f"{self.dcp_size}."
                )
            self.msa_op = get_msa_dcp_op(
                *msa_config, self.dcp_size, self.kv_head_replicas
            )
            # A buffer, not a Python int, so the rank is a graph input and
            # every DCP rank compiles the same program (see get_msa_dcp_op).
            self.register_buffer(
                "dcp_rank",
                torch.tensor([get_dcp_rank()], dtype=torch.int32),
                persistent=False,
            )
        elif self.index_shards > 1:
            self.msa_op = get_msa_sharded_op(
                *msa_config, get_tensor_model_parallel_rank(), self.index_shards
            )
        else:
            self.msa_op = get_msa_op(*msa_config)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, index_q, index_k = norm_rope_heads(
            qkv,
            positions,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            (self.q_norm, self.k_norm, self.index_q_norm, self.index_k_norm),
            self.rotary_emb,
        )

        num_tokens = hidden_states.shape[0]
        metadata = get_forward_context().attn_metadata
        kv_cache = self.kv_cache.kv_cache
        if not isinstance(metadata, dict) or kv_cache.numel() == 0:
            # Memory profiling runs before the caches are bound.
            attn_output = q
        else:
            layer_metadata = metadata[self.kv_cache.prefix]
            assert isinstance(layer_metadata, AttentionMetadata)
            index_q = index_q.view(num_tokens, self.num_kv_heads, self.index_head_dim)
            if self.index_shards > 1:
                # Every rank scores all index heads on its share of the keys.
                index_q = get_tp_group().all_gather(index_q.contiguous(), dim=1)
            extra = {"dcp_rank": self.dcp_rank} if self.dcp_size > 1 else {}
            attn_output = self.msa_op(
                q.view(num_tokens, self.num_heads, self.head_dim),
                k.view(num_tokens, self.num_kv_heads, self.head_dim),
                v.view(num_tokens, self.num_kv_heads, self.head_dim),
                index_q,
                index_k,
                kv_cache,
                self.index_cache.kv_cache,
                layer_metadata,
                **extra,
            ).reshape(num_tokens, self.q_size)
        output, _ = self.o_proj(attn_output)
        return output
