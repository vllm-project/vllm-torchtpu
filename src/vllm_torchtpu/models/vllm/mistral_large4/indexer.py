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
"""ML4 preprocessing for the existing TPU DSA cache and StreamIndex kernels.

Cache insertion, scoring, top-k selection and buffer updates use TPU adapters.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import LayerNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
    VllmTPUIndexerCache,
    VllmTPUSparseAttnIndexer,
)
from vllm_torchtpu.models.vllm.mistral_large4.rope import MistralRotaryEmbedding
from vllm_torchtpu.models.vllm.mistral_large4.ssmax import SSMaxScaling


class MistralIndexer(nn.Module):
    """Replicated ML4 index heads writing the model's shared logical top-k table."""

    quant_block_size = 128

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: PretrainedConfig,
        topk_indices_buffer: torch.Tensor,
        prefix: str,
    ) -> None:
        super().__init__()
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.topk_tokens = config.index_topk
        if self.head_dim != self.quant_block_size:
            raise NotImplementedError(
                "TPU StreamIndex requires one quantization block per index head: "
                f"index_head_dim={self.quant_block_size}, got {self.head_dim}."
            )
        self.q_input_norm = LayerNorm(config.hidden_size, eps=config.indexer_norm_eps)
        self.wq_b = ReplicatedLinear(
            config.hidden_size,
            self.n_head * self.head_dim,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.wk_weights_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=config.indexer_norm_eps).to(
            dtype=torch.float32
        )
        self.ssmax = (
            SSMaxScaling(self.n_head, tp_sharded=False)
            if config.indexer_ssmax_scaling
            else None
        )
        self.topk_indices_buffer = topk_indices_buffer
        # The shared TPU adapter packs one UE8M0 exponent byte per block,
        # rather than the CUDA cache's four-byte scale slot.
        self.k_cache = VllmTPUIndexerCache(
            head_dim=self.head_dim + self.head_dim // self.quant_block_size,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=vllm_config.cache_config,
        )
        self.indexer_op = VllmTPUSparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            "ue8m0",
            self.topk_tokens,
            self.head_dim,
            vllm_config.model_config.max_model_len,
            get_max_prefill_buffer_size(vllm_config),
            topk_indices_buffer,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        log_positions: torch.Tensor,
        rotary_emb: MistralRotaryEmbedding | None,
    ) -> torch.Tensor:
        q = self.wq_b(self.q_input_norm(hidden_states))[0].view(
            -1, self.n_head, self.head_dim
        )
        kw = self.wk_weights_proj(hidden_states)[0]
        k = self.k_norm(kw[:, : self.head_dim])
        weights = kw[:, self.head_dim :]
        if self.ssmax is not None:
            q = self.ssmax(q, log_positions)
        if rotary_emb is not None:
            # Indexer layout is [rope | nope], unlike the MLA query layout.
            # CUDA rotates in FP32 and rounds once on output. Passing FP32
            # slices keeps the native rotary path from rounding each product.
            q_pe, k_pe = rotary_emb(
                positions,
                q[..., : self.rope_dim].float(),
                k[:, None, : self.rope_dim].float(),
            )
            assert k_pe is not None
            q = torch.cat((q_pe.to(q.dtype), q[..., self.rope_dim :]), dim=-1)
            k = torch.cat((k_pe.squeeze(1).to(k.dtype), k[:, self.rope_dim :]), dim=-1)

        # Match the reference's two scale floors and its FP32 weight folding.
        # The generic TPU query helper uses unconstrained FP32 scales, whereas
        # ML4's CUDA indexer uses a power-of-two scale in both RoPE/NoPE paths.
        q_float = q.float()
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        amax = q_float.abs().amax(dim=-1, keepdim=True)
        raw_scale = (
            amax.clamp_min(1e-10) * (1.0 / fp8_max)
            if rotary_emb is not None
            else (amax / fp8_max).clamp_min(1e-10)
        )
        q_scale = torch.exp2(torch.ceil(torch.log2(raw_scale)))
        q_fp8 = (q_float / q_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
        weights = weights.float() * q_scale.squeeze(-1)
        if rotary_emb is None:
            weights = weights * (self.head_dim**-0.5 * self.n_head**-0.5)
        else:
            weights = weights * self.head_dim**-0.5 * self.n_head**-0.5
        return self.indexer_op(hidden_states, q_fp8, k, weights)
