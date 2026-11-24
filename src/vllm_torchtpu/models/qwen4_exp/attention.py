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
"""Qwen4Exp attention sublayer: (positions, hidden_states) -> [T, hidden_size]."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn
from transformers import Qwen4ExpTextConfig
from vllm.config import CacheConfig, ModelConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope

from vllm_torchtpu.models.qwen4_exp.attention_plan import (
    AttentionDesc,
    AttentionPlan,
    CoreAttn,
    OutProj,
    QKVProj,
)
from vllm_torchtpu.models.qwen4_exp.qsa import Qwen4ExpQSA, dense_max_model_len


class Qwen4ExpAttention(nn.Module):
    """Gated QK-norm attention with optional QSA sparse core."""

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        *,
        model_config: ModelConfig,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.num_heads, self.num_kv_heads = _tp_head_counts(config)
        self.head_dim = config.head_dim or (
            config.hidden_size // config.num_attention_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # Weights (match HF / Qwen3Next parameter names).
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads * 2,  # Q + output gate
            config.num_key_value_heads,
            bias=getattr(config, "qkv_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=cast(dict[str, Any] | None, config.rope_parameters),
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        # Resolve the 3-step plan once; forward() only matches on self.plan.
        self.dense_max_model_len = dense_max_model_len(config)
        self.desc = AttentionDesc.from_configs(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            sparse=model_config.max_model_len > self.dense_max_model_len,
            model_config=model_config,
            cache_config=cache_config,
        )
        self.plan = AttentionPlan.resolve(self.desc)

        # Attribute name "indexer" matches HF checkpoint paths self_attn.indexer.*
        self.indexer: Qwen4ExpQSA | None = (
            Qwen4ExpQSA(
                config,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                scaling=self.scaling,
                qkv_layout=self.plan.qkv_layout,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )
            if self.plan.core_attn is CoreAttn.QSA
            else None
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        q, kv, gate = self._qkv_proj(positions, hidden_states)
        attn_out = self._core_attn(positions, hidden_states, q, kv)
        return self._out_proj(attn_out, gate)

    def _qkv_proj(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        match self.plan.qkv_proj:
            case QKVProj.UNFUSED:
                qkv, _ = self.qkv_proj(hidden_states)
                q_gate, k, v = qkv.split(
                    [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
                )
                q, gate = q_gate.view(-1, self.num_heads, 2 * self.head_dim).chunk(
                    2, dim=-1
                )
                q = self.q_norm(q).reshape(-1, self.q_size)
                k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).reshape(
                    -1, self.kv_size
                )
                q, k = self.rotary_emb(positions, q, k)
                return (
                    q.view(-1, self.num_heads, self.head_dim),
                    (
                        k.view(-1, self.num_kv_heads, self.head_dim),
                        v.view(-1, self.num_kv_heads, self.head_dim),
                    ),
                    gate.reshape(-1, self.q_size),
                )
            case QKVProj.FUSED:
                raise NotImplementedError("QKVProj.FUSED is wired in a follow-up PR.")

    def _core_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        q: torch.Tensor,
        kv: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        match self.plan.core_attn:
            case CoreAttn.RPA_V3:
                assert isinstance(kv, tuple)
                k, v = kv
                out = self.attn(
                    q.reshape(-1, self.q_size),
                    k.reshape(-1, self.kv_size),
                    v.reshape(-1, self.kv_size),
                )
                return out.view(-1, self.num_heads, self.head_dim)
            case CoreAttn.BATCHED_RPA:
                raise NotImplementedError(
                    "CoreAttn.BATCHED_RPA is wired in a follow-up PR."
                )
            case CoreAttn.QSA:
                assert self.indexer is not None
                main_pool = self.attn.kv_cache
                if main_pool.numel() > 0:
                    attn_metadata = get_forward_context().attn_metadata
                    assert isinstance(attn_metadata, dict)
                    metadata = attn_metadata[self.attn.layer_name]
                    kv_scales = torch.stack(
                        [
                            cast(torch.Tensor, self.attn._k_scale),
                            cast(torch.Tensor, self.attn._v_scale),
                        ]
                    ).float()
                else:
                    metadata = None
                    kv_scales = None
                return self.indexer(
                    q=q,
                    kv=kv,
                    positions=positions,
                    hidden_states=hidden_states,
                    main_pool=main_pool,
                    metadata=metadata,
                    kv_scales=kv_scales,
                    cos_sin_cache=self.rotary_emb.cos_sin_cache,
                    dense_attn=self.attn,
                )

    def _out_proj(self, attn_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        match self.plan.out_proj:
            case OutProj.UNFUSED:
                out, _ = self.o_proj(
                    attn_out.reshape(-1, self.q_size) * torch.sigmoid(gate)
                )
                return out
            case OutProj.FUSED:
                raise NotImplementedError("OutProj.FUSED is wired in a follow-up PR.")


def _tp_head_counts(config: Qwen4ExpTextConfig) -> tuple[int, int]:
    tp = get_tensor_model_parallel_world_size()
    assert config.num_attention_heads % tp == 0
    kv = config.num_key_value_heads
    assert (kv % tp == 0) if kv >= tp else (tp % kv == 0)
    return config.num_attention_heads // tp, max(1, kv // tp)
