# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""ML4 query preparation for the shared TPU MLA implementation.

Uses native SSMax and passes separate NoPE/RoPE queries to TPU MLA.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)

from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
    VllmTPUMLAAttention,
)
from vllm_torchtpu.models.vllm.mistral_large4.checkpoint_config import MistralLayerSpec
from vllm_torchtpu.models.vllm.mistral_large4.indexer import MistralIndexer
from vllm_torchtpu.models.vllm.mistral_large4.rope import MistralRotaryEmbedding
from vllm_torchtpu.models.vllm.mistral_large4.ssmax import SSMaxScaling


class MistralMLAAttention(nn.Module):
    """Dense or sparse ML4 MLA with per-layer RoPE and index sharing."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: PretrainedConfig,
        spec: MistralLayerSpec,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        zero_kv_inputs: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % tp_size == 0
        assert config.q_lora_rank is None
        self.shares_kv = spec.shares_kv
        self.zero_kv_inputs = zero_kv_inputs
        assert not spec.shares_kv or zero_kv_inputs is not None
        self.use_rope = spec.use_rope
        self.use_sparse = spec.use_sparse
        self.indexer_use_rope = spec.indexer_use_rope
        self.topk_indices_buffer = topk_indices_buffer
        if self.use_sparse:
            assert topk_indices_buffer is not None
        self.num_local_heads = config.num_attention_heads // tp_size
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.qk_nope_head_dim = (
            config.qk_nope_head_dim if spec.use_rope else self.qk_head_dim
        )
        quant_config = vllm_config.quant_config
        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        if not spec.shares_kv:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                config.hidden_size,
                self.kv_lora_rank + (self.qk_rope_head_dim if spec.use_rope else 0),
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
            self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            config.num_attention_heads * (self.qk_nope_head_dim + config.v_head_dim),
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * config.v_head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.ssmax = (
            SSMaxScaling(self.num_local_heads) if config.ssmax_scaling else None
        )
        self.indexer = (
            MistralIndexer(
                vllm_config, config, topk_indices_buffer, prefix=f"{prefix}.indexer"
            )
            if spec.has_indexer and topk_indices_buffer is not None
            else None
        )
        kv_target = None
        if spec.shares_kv:
            layers_prefix = prefix.rsplit(".", 2)[0]
            kv_target = f"{layers_prefix}.{spec.kv_source_layer}.self_attn.attn"
        self.attn = VllmTPUMLAAttention(
            num_heads=self.num_local_heads,
            scale=self.qk_head_dim**-0.5,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=None,
            kv_lora_rank=self.kv_lora_rank,
            kv_b_proj=self.kv_b_proj,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_sharing_target_layer_name=kv_target,
            use_sparse=self.use_sparse,
            indexer=self.indexer,
            topk_indices_buffer=topk_indices_buffer,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        log_positions: torch.Tensor,
        rotary_emb: MistralRotaryEmbedding,
    ) -> torch.Tensor:
        q = self.q_proj(hidden_states)[0].view(
            -1, self.num_local_heads, self.qk_head_dim
        )
        if self.shares_kv:
            # Signature placeholders shared by all readers; MLA ignores them.
            assert self.zero_kv_inputs is not None
            kv_c_normed = self.zero_kv_inputs[0][: hidden_states.shape[0]]
            k_pe = self.zero_kv_inputs[1][: hidden_states.shape[0]]
        else:
            kv = self.kv_a_proj_with_mqa(hidden_states)[0]
            kv_c_normed = self.kv_a_layernorm(kv[:, : self.kv_lora_rank])
            k_pe = (
                kv[:, None, self.kv_lora_rank :]
                if self.use_rope
                else kv.new_zeros(kv.shape[0], 1, self.qk_rope_head_dim)
            )
        if self.use_rope:
            q_nope, q_pe = q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
            q_pe, k_pe = rotary_emb(positions, q_pe, k_pe)
            assert k_pe is not None
            # Scale the concatenated query, as in the reference, so both
            # components use identical activation-dtype rounding.
            q = torch.cat((q_nope, q_pe), dim=-1)
        if self.ssmax is not None:
            q = self.ssmax(q, log_positions)
        if self.use_rope:
            q_nope, q_pe = q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
        else:
            q_nope = q
            q_pe = q.new_zeros(q.shape[0], self.num_local_heads, self.qk_rope_head_dim)
        topk_indices = None
        if self.use_sparse:
            topk = self.topk_indices_buffer
            if self.indexer is not None:
                topk = self.indexer(
                    hidden_states,
                    positions,
                    log_positions,
                    rotary_emb if self.indexer_use_rope else None,
                )
            assert topk is not None
            topk_indices = topk[: hidden_states.shape[0]]
        attn_out = self.attn(
            (q_nope, q_pe), kv_c_normed, k_pe, topk_indices=topk_indices
        )
        return self.o_proj(attn_out)[0]
