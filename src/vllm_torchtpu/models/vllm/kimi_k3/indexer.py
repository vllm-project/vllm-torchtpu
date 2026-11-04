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
"""Kimi K3.1 DSA indexer on the TPU StreamIndex cache and top-k kernels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import LayerNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
    VllmTPUIndexerCache,
    VllmTPUSparseAttnIndexer,
)


@dataclass(frozen=True)
class KimiDSAConfig:
    index_topk: int
    index_n_heads: int
    index_head_dim: int


def kimi_dsa_config(config: KimiLinearConfig) -> KimiDSAConfig | None:
    """Parse `text_config.dsa_config`; None for models without DSA (K3)."""
    dsa = getattr(config, "dsa_config", None)
    if dsa is None:
        return None
    required = {
        "index_rope_dim": (dsa.get("index_rope_dim", 0), 0),
        "index_k_norm_type": (dsa.get("index_k_norm_type", "rmsnorm"), "layernorm"),
        "index_topk_freq": (dsa.get("index_topk_freq", 1), 1),
        "index_head_dim": (dsa.get("index_head_dim", 64), 128),
    }
    for key, (value, expected) in required.items():
        if value != expected:
            raise NotImplementedError(
                f"Kimi DSA on TPU supports {key}={expected!r}, got {value!r}"
            )
    # The KV layouts size every MLA layer alike, and both lists are 1-based.
    kda_layers = (config.linear_attn_config or {}).get("kda_layers", ())
    mla_layers = set(range(1, config.num_hidden_layers + 1)) - set(kda_layers)
    if {int(layer) for layer in dsa["dsa_layers"]} != mla_layers:
        raise NotImplementedError(
            "Kimi DSA on TPU requires dsa_layers to list every MLA layer, "
            f"got {dsa['dsa_layers']} for MLA layers {sorted(mla_layers)}"
        )
    return KimiDSAConfig(
        index_topk=int(dsa["index_topk"]),
        index_n_heads=int(dsa["index_num_heads"]),
        index_head_dim=int(dsa["index_head_dim"]),
    )


class KimiDSAIndexer(nn.Module):
    """Replicated K3.1 index heads writing the model's shared top-k table.

    K3.1 indexes without RoPE and reads the MLA query LoRA output `q_c`, like
    DeepSeek V3.2.
    """

    quant_block_size = 128

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: KimiLinearConfig,
        dsa: KimiDSAConfig,
        topk_indices_buffer: torch.Tensor,
        prefix: str,
    ) -> None:
        super().__init__()
        assert config.q_lora_rank is not None
        self.n_head = dsa.index_n_heads
        self.head_dim = dsa.index_head_dim
        self.topk_tokens = dsa.index_topk
        self.wq_b = ReplicatedLinear(
            config.q_lora_rank,
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
        self.k_norm = LayerNorm(self.head_dim, eps=config.rms_norm_eps).to(
            dtype=torch.float32
        )
        self.topk_indices_buffer = topk_indices_buffer
        # One UE8M0 exponent byte per block, not the CUDA cache's fp32 scale.
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
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: object = None,
    ) -> torch.Tensor:
        del positions, rotary_emb
        q = self.wq_b(qr)[0].view(-1, self.n_head, self.head_dim)
        kw = self.wk_weights_proj(hidden_states)[0]
        k = self.k_norm(kw[:, : self.head_dim])
        weights = kw[:, self.head_dim :]

        # Power-of-two per-(token, head) query scale, folded into the weights
        # in FP32 together with both softmax scales, as the CUDA indexer does.
        q_float = q.float()
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        amax = q_float.abs().amax(dim=-1, keepdim=True)
        q_scale = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-10) / fp8_max)))
        q_fp8 = (q_float / q_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
        weights = (
            weights.float()
            * q_scale.squeeze(-1)
            * (self.head_dim**-0.5 * self.n_head**-0.5)
        )
        return self.indexer_op(hidden_states, q_fp8, k, weights)
