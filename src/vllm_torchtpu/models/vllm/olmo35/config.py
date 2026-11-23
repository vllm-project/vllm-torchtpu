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

from typing import TYPE_CHECKING

from transformers import PretrainedConfig
from vllm.model_executor.models.config import VerifyAndUpdateConfig

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


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
        # Kimi Delta Attention on every layer that is not full attention.
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
        unknown = sorted(set(layer_types) - {"full_attention", "linear_attention"})
        if unknown:
            raise ValueError(
                f"Unknown layer_types {unknown}; expected 'full_attention' or "
                "'linear_attention'"
            )
        self.layer_types = layer_types
        if tie_word_embeddings:
            # The model has its own lm_head and the loader has no tied path.
            raise ValueError("OLMo 3.5 does not support tie_word_embeddings=True")
        super().__init__(**kwargs)
        self.tie_word_embeddings = tie_word_embeddings


def register_olmo35_config() -> bool:
    """Make `AutoConfig` (and so vLLM) resolve `model_type: olmo35`.

    Runs for every model at plugin load, so a clash (e.g. a transformers
    release that ships its own `olmo35` config) logs and skips instead of
    failing every model. The OLMo 3.5 model itself then refuses a config that
    is not `Olmo35Config` (`model._config`). Returns whether ours is in place.
    """
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    try:
        registered = CONFIG_MAPPING[Olmo35Config.model_type]
    except KeyError:
        AutoConfig.register(Olmo35Config.model_type, Olmo35Config)
        return True
    if registered is Olmo35Config:
        # `register_models()` can run more than once per process.
        return True
    logger.warning(
        "model_type %r is already registered to %s.%s; not registering the "
        "TPU Olmo35Config.",
        Olmo35Config.model_type,
        registered.__module__,
        registered.__qualname__,
    )
    return False


class TPUOlmo35Config(VerifyAndUpdateConfig):
    """Reject or turn off what the TPU OLMo 3.5 model does not support yet."""

    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        from vllm_torchtpu.gdn_pool_layout import (
            unified_kv_layout_enabled_for_architecture,
        )

        # The KDA state lives in per-layer slot arrays, not the unified pool,
        # so it cannot be prefix-cached. The platform would otherwise demand
        # the unified pool for a hybrid model with prefix caching on.
        cache_config = vllm_config.cache_config
        if cache_config.enable_prefix_caching:
            logger.warning(
                "OLMo 3.5 on TPU does not support prefix caching; turning it "
                "off. Pass --no-enable-prefix-caching to silence this."
            )
            cache_config.enable_prefix_caching = False
        if unified_kv_layout_enabled_for_architecture(
            vllm_config.model_config.architecture
        ):
            raise NotImplementedError(
                "OLMo 3.5 on TPU has no unified-pool KDA path; set "
                "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0."
            )
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "OLMo 3.5 on TPU has no speculative decoding; remove "
                "--speculative-config."
            )
        kv_transfer = vllm_config.kv_transfer_config
        if kv_transfer is not None and kv_transfer.kv_connector != (
            "DecodeBenchConnector"
        ):
            raise NotImplementedError(
                "OLMo 3.5 on TPU supports no KV connector except "
                f"DecodeBenchConnector (got {kv_transfer.kv_connector}); remove "
                "--kv-transfer-config."
            )
        parallel = vllm_config.parallel_config
        if (
            parallel.prefill_context_parallel_size > 1
            or parallel.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "OLMo 3.5 on TPU has no context parallelism; set "
                "--prefill-context-parallel-size 1 and "
                "--decode-context-parallel-size 1."
            )
