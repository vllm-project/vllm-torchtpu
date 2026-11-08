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
"""ERNIE 5.5 (``Ernie5_5_MoeForCausalLM``, model_type ``ernie5_5_moe``) config.

Ported from github.com/baidu/vLLM-ERNIE @ 34cceda1
``src/vllm_ernie/models/ernie5_5/config.py`` (Apache-2.0, Copyright contributors to the
vLLM project): the same fields and defaults, plus ``rope_parameters`` carrying
``rope_type`` (the checkpoint's ``rope_scaling`` says ``type``), which DeepSeek-V4's
rope builder reads.
"""

from typing import Any

from transformers import PretrainedConfig

# compress_ratios entry -> transformers' DeepseekV4Config layer type. A ratio <= 0 marks
# ERNIE's MLA-DSA layers (is_dsa_layer), which run their own attention (dsa.py), never
# the DeepSeek-V4 one.
_LAYER_TYPE = {4: "compressed_sparse_attention", 128: "heavily_compressed_attention"}


class Ernie5_5Config(PretrainedConfig):
    """Config for ERNIE 5.5 hybrid HCA / MLA-DSA with mHC and a latent MoE."""

    model_type = "ernie5_5_moe"

    def __init__(
        self,
        vocab_size: int = 201216,
        hidden_size: int = 4096,
        num_hidden_layers: int = 43,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 1,
        head_dim: int = 512,
        q_lora_rank: int = 1024,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 448,
        qk_rope_head_dim: int = 64,
        o_groups: int = 8,
        o_lora_rank: int = 1024,
        sliding_window: int = 128,
        compress_ratios: list[int] | None = None,
        hybrid_mla_num_attention_heads: int = 64,
        hybrid_mla_q_lora_rank: int = 1024,
        hybrid_mla_kv_lora_rank: int = 512,
        hybrid_mla_qk_nope_head_dim: int = 192,
        hybrid_mla_qk_rope_head_dim: int = 64,
        hybrid_mla_v_head_dim: int = 256,
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 2048,
        max_position_embeddings: int = 200192,
        rope_theta: float = 2500000.0,
        compress_rope_theta: float = 160000.0,
        rope_scaling: dict[str, Any] | None = None,
        rope_parameters: dict[str, Any] | None = None,
        rms_norm_eps: float = 1e-6,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        intermediate_size: int = 14336,
        moe_intermediate_size: int = 2048,
        moe_latent_size: int = 2048,
        n_routed_experts: int = 512,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 10,
        first_k_dense_replace: int = 1,
        routed_scaling_factor: float = 3.0,
        norm_topk_prob: bool = True,
        hidden_act: str = "situ",
        activation_situ_beta: float = 4.0,
        activation_situ_linear_beta: float = 25.0,
        multimax: str | None = "lm_head",
        num_nextn_predict_layers: int = 1,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.sliding_window = sliding_window
        self.compress_ratios = (
            compress_ratios
            if compress_ratios is not None
            else [128] * (num_hidden_layers + num_nextn_predict_layers)
        )
        # vLLM 0.33's DeepSeek-V4 attention reads config.layer_types[layer_id] and
        # config.compress_rates[type] (transformers' DeepseekV4Config spelling); derived
        # from compress_ratios, the MTP layer excluded
        self.layer_types = kwargs.pop("layer_types", None) or [
            _LAYER_TYPE[r] if r > 0 else "indexed_attention"
            for r in self.compress_ratios[:num_hidden_layers]
        ]
        self.compress_rates = kwargs.pop("compress_rates", None) or {
            _LAYER_TYPE[r]: r for r in self.compress_ratios[:num_hidden_layers] if r > 0
        }
        self.hybrid_mla_num_attention_heads = hybrid_mla_num_attention_heads
        self.hybrid_mla_q_lora_rank = hybrid_mla_q_lora_rank
        self.hybrid_mla_kv_lora_rank = hybrid_mla_kv_lora_rank
        self.hybrid_mla_qk_nope_head_dim = hybrid_mla_qk_nope_head_dim
        self.hybrid_mla_qk_rope_head_dim = hybrid_mla_qk_rope_head_dim
        self.hybrid_mla_v_head_dim = hybrid_mla_v_head_dim
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.compress_rope_theta = compress_rope_theta
        # Consumed by the HCA layers only (YaRN at compress_rope_theta); the DSA layers
        # use mla_rope_parameters.
        rope = dict(
            rope_scaling
            or rope_parameters
            or {"rope_type": "default", "rope_theta": rope_theta}
        )
        rope.setdefault("rope_type", rope.pop("type", "default"))
        self.rope_parameters = rope
        self.rms_norm_eps = rms_norm_eps
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_latent_size = moe_latent_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.hidden_act = hidden_act
        self.activation_situ_beta = activation_situ_beta
        self.activation_situ_linear_beta = activation_situ_linear_beta
        self.multimax = multimax
        self.num_nextn_predict_layers = num_nextn_predict_layers
        # The trainer computes logits (and MultiMax) in fp32.
        kwargs.setdefault("head_dtype", "float32")
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def is_dsa_layer(self, layer_id: int) -> bool:
        return (
            layer_id < len(self.compress_ratios) and self.compress_ratios[layer_id] <= 0
        )

    @property
    def mla_rope_parameters(self) -> dict[str, Any]:
        # YaRN is HCA-only (at compress_rope_theta); DSA and its indexer use plain RoPE
        # at rope_theta.
        return {"rope_type": "default", "rope_theta": self.rope_theta}
