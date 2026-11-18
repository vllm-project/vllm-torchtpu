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
"""HF config for Ai2's OLMo 3.5 hybrid KDA / attention / latent-MoE models.

transformers has no class for this architecture yet. The fields mirror the
MaxText `decoder_block: olmoe3` configs; the defaults are OLMo 3.5 Small
(72.6B total, 4.1B active).
"""

from __future__ import annotations

import contextlib

from transformers import PretrainedConfig


class Olmo35Config(PretrainedConfig):
    model_type = "olmo35"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 100352,
        hidden_size: int = 1536,
        num_hidden_layers: int = 40,
        # Full attention (every `full_attention_interval`-th layer): GQA,
        # NoPE, per-head QK RMSNorm, scalable softmax, sigmoid output gate.
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        head_dim: int = 256,
        full_attention_interval: int = 8,
        use_scalable_softmax: bool = True,
        # Kimi Delta Attention on every other layer.
        linear_num_heads: int = 8,
        linear_key_head_dim: int = 256,
        linear_value_head_dim: int = 512,
        linear_conv_kernel_dim: int = 4,
        # beta = linear_beta_scale * sigmoid(b); 2.0 is OLMo-core's
        # allow_neg_eigval.
        linear_beta_scale: float = 2.0,
        linear_norm_eps: float = 1e-5,
        # FFN: the first `first_k_dense_replace` layers are dense SwiGLU of
        # width `intermediate_size`; the rest run a shared SwiGLU expert plus
        # routed experts in a `moe_latent_size` latent space.
        intermediate_size: int = 12288,
        first_k_dense_replace: int = 1,
        shared_expert_intermediate_size: int = 3072,
        moe_intermediate_size: int = 3072,
        moe_latent_size: int = 768,
        num_experts: int = 256,
        num_experts_per_tok: int = 8,
        # Top-k softmax weights are renormalized and then multiplied by this
        # (OLMo-core rescales them to sum to top_k).
        routed_scaling_factor: float | None = None,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 65536,
        tie_word_embeddings: bool = False,
        layer_types: list[str] | None = None,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.full_attention_interval = full_attention_interval
        self.use_scalable_softmax = use_scalable_softmax
        self.linear_num_heads = linear_num_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_beta_scale = linear_beta_scale
        self.linear_norm_eps = linear_norm_eps
        self.intermediate_size = intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_latent_size = moe_latent_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = (
            float(num_experts_per_tok)
            if routed_scaling_factor is None
            else routed_scaling_factor
        )
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        if layer_types is None:
            # MaxText: full attention where (layer + 1) % interval == 0.
            layer_types = [
                "full_attention"
                if (i + 1) % full_attention_interval == 0
                else "linear_attention"
                for i in range(num_hidden_layers)
            ]
        if len(layer_types) != num_hidden_layers:
            raise ValueError("layer_types needs one entry per layer")
        self.layer_types = layer_types
        super().__init__(**kwargs)
        self.tie_word_embeddings = tie_word_embeddings


def register_olmo35_config() -> None:
    """Make `AutoConfig` (and so vLLM) resolve `model_type: olmo35`."""
    from transformers import AutoConfig

    # ValueError: already registered in this process.
    with contextlib.suppress(ValueError):
        AutoConfig.register(Olmo35Config.model_type, Olmo35Config)
