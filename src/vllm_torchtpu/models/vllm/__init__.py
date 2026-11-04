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
import torch
from vllm.model_executor.models.config import VerifyAndUpdateConfig
from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config

from vllm_torchtpu import envs
from vllm_torchtpu.gqa_head_padding import update_gqa_head_padding_config

_KIMI_MODEL = "vllm_torchtpu.models.vllm.kimi_k3:KimiLinearForCausalLM"
_KIMI_K3_MODEL = "vllm_torchtpu.models.vllm.kimi_k3:KimiK3ForConditionalGeneration"
_DSV4_MODEL = "vllm_torchtpu.models.vllm.deepseek_v4:DeepseekV4ForCausalLM"
_K3_DSPARK_MODEL = "vllm_torchtpu.models.vllm.kimi_k3:K3DSparkForCausalLM"
_DSV4_DSPARK_MODEL = (
    "vllm_torchtpu.models.vllm.deepseek_v4.dspark:DSparkDeepseekV4ForCausalLM"
)
_QWEN35_MTP_MODEL = "vllm_torchtpu.models.vllm.qwen3_5_mtp:Qwen3_5MTP"
_QWEN35_MOE_MTP_MODEL = "vllm_torchtpu.models.vllm.qwen3_5_mtp:Qwen3_5MoeMTP"
_QWEN4_EXP_MODEL = "vllm_torchtpu.models.vllm.qwen4_exp.model:Qwen4ExpForCausalLM"


class _TPUKimiK3Config(VerifyAndUpdateConfig):
    """Keep K3 MXFP4 checkpoints on TPU's packed-weight loader."""


class _KimiK31Config(KimiK3Config):
    model_type = "kimi_k31"
    # Transformers generates an `__init__` for config subclasses without one.
    __init__ = KimiK3Config.__init__


class _TPUKimiK31Config(_TPUKimiK3Config):
    """Serve K3.1 as K3 plus DSA, rejecting what its sparse MLA cannot run."""

    @staticmethod
    def verify_and_update_model_config(model_config) -> None:
        # vLLM keys K3's chat rendering and tool-call ids on K3's own names.
        # TODO: drop once vLLM keys them on K3.1 too.
        if model_config.tokenizer_mode == "auto":
            model_config.tokenizer_mode = "kimi_k3"
        model_config.hf_config.model_type = "kimi_k3"

    @staticmethod
    def verify_and_update_config(vllm_config) -> None:
        from vllm_torchtpu.kernels.mla import kv_cache_utils

        model_config = vllm_config.model_config
        config = model_config.hf_text_config
        speculative_config = vllm_config.speculative_config
        if speculative_config is not None and speculative_config.method != "dspark":
            raise NotImplementedError(
                "Kimi K3.1 speculative decoding on TPU supports only DSpark."
            )
        if (
            speculative_config is not None
            and vllm_config.parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "The Kimi K3.1 DSpark draft runs dense MLA, which has no decode "
                "context parallel path. Unset --decode-context-parallel-size."
            )
        if envs.TPU_K3_SP_PREFILL:
            raise NotImplementedError(
                "Kimi K3.1 DSA layers do not support sequence parallel prefill. "
                "Unset TPU_K3_SP_PREFILL."
            )
        latent_dim = kv_cache_utils.WORD_BYTES * kv_cache_utils.TILE_LANE_BYTES
        rope_dim = kv_cache_utils.TILE_LANE_BYTES // 2
        if (config.kv_lora_rank, config.qk_rope_head_dim) != (latent_dim, rope_dim):
            raise NotImplementedError(
                "The TPU sparse MLA gather requires "
                f"kv_lora_rank={latent_dim} and qk_rope_head_dim={rope_dim}."
            )
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Kimi K3.1 sparse MLA requires --dtype bfloat16.")
        if vllm_config.cache_config.cache_dtype not in ("fp8", "fp8_e4m3"):
            raise NotImplementedError(
                "Kimi K3.1 sparse MLA requires an FP8 latent KV cache. "
                "Pass --kv-cache-dtype fp8_e4m3."
            )


class _TPUQwen2Config(VerifyAndUpdateConfig):
    """Pad Qwen2 GQA query heads per KV group when heads % tp_size != 0."""

    @staticmethod
    def verify_and_update_config(vllm_config) -> None:
        update_gqa_head_padding_config(
            vllm_config.model_config,
            vllm_config.parallel_config,
        )


def register_models() -> None:
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    from vllm_torchtpu.models.vllm.qwen4_exp.config import TPUQwen4ExpConfig

    ModelRegistry.register_model("KimiLinearForCausalLM", _KIMI_MODEL)
    ModelRegistry.register_model("KimiK3ForConditionalGeneration", _KIMI_K3_MODEL)
    # Kimi K3.1 is K3 plus DSA on the MLA layers, keyed on its dsa_config.
    ModelRegistry.register_model("KimiK31ForConditionalGeneration", _KIMI_K3_MODEL)
    # TODO: drop once vLLM lists K3.1. Only this registry overrides the
    # checkpoint's auto_map under --trust-remote-code; AutoConfig.register cannot.
    _CONFIG_REGISTRY.setdefault(_KimiK31Config.model_type, _KimiK31Config)
    ModelRegistry.register_model("DeepseekV4ForCausalLM", _DSV4_MODEL)
    ModelRegistry.register_model("K3DSparkModel", _K3_DSPARK_MODEL)
    ModelRegistry.register_model("DSparkDraftModel", _DSV4_DSPARK_MODEL)
    # The Qwen3.5 MTP drafts, so a pipeline-parallel producer can load them.
    ModelRegistry.register_model("Qwen3_5MTP", _QWEN35_MTP_MODEL)
    ModelRegistry.register_model("Qwen3_5MoeMTP", _QWEN35_MOE_MTP_MODEL)
    ModelRegistry.register_model("Qwen4ExpForCausalLM", _QWEN4_EXP_MODEL)
    # The released checkpoint declares the vision-language arch; serve it with
    # the text-only model.
    ModelRegistry.register_model("Qwen4ExpForConditionalGeneration", _QWEN4_EXP_MODEL)
    MODELS_CONFIG_MAP["KimiK3ForConditionalGeneration"] = _TPUKimiK3Config
    MODELS_CONFIG_MAP["KimiK31ForConditionalGeneration"] = _TPUKimiK31Config
    MODELS_CONFIG_MAP["Qwen4ExpForCausalLM"] = TPUQwen4ExpConfig
    MODELS_CONFIG_MAP["Qwen4ExpForConditionalGeneration"] = TPUQwen4ExpConfig
    MODELS_CONFIG_MAP["Qwen2ForCausalLM"] = _TPUQwen2Config
