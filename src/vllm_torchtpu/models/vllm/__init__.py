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
from typing import Any

from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.models.config import (
    MODELS_CONFIG_MAP,
    DeepseekV4ForCausalLMConfig,
    VerifyAndUpdateConfig,
)

from vllm_torchtpu import patch_registry
from vllm_torchtpu.gqa_head_padding import update_gqa_head_padding_config

_KIMI_MODEL = "vllm_torchtpu.models.vllm.kimi_k3:KimiLinearForCausalLM"
_KIMI_K3_MODEL = "vllm_torchtpu.models.vllm.kimi_k3:KimiK3ForConditionalGeneration"
_DSV4_MODEL = "vllm_torchtpu.models.vllm.deepseek_v4:DeepseekV4ForCausalLM"
_K3_DSPARK_MODEL = "vllm_torchtpu.models.vllm.kimi_k3:K3DSparkForCausalLM"
_DSV4_DSPARK_MODEL = (
    "vllm_torchtpu.models.vllm.deepseek_v4.dspark:DSparkDeepseekV4ForCausalLM"
)
_DSV41_MODEL = "vllm_torchtpu.models.vllm.deepseek_v41.model:DeepseekV41ForCausalLM"
_DSV41_DSPARK_MODEL = (
    "vllm_torchtpu.models.vllm.deepseek_v41.dspark:DSparkDeepseekV41ForCausalLM"
)
_QWEN35_MTP_MODEL = "vllm_torchtpu.models.vllm.qwen3_5_mtp:Qwen3_5MTP"
_QWEN35_MOE_MTP_MODEL = "vllm_torchtpu.models.vllm.qwen3_5_mtp:Qwen3_5MoeMTP"
_QWEN4_EXP_MODEL = "vllm_torchtpu.models.vllm.qwen4_exp.model:Qwen4ExpForCausalLM"


class _TPUDeepseekV41Config(DeepseekV4ForCausalLMConfig):
    """Refresh ``model_arch_config`` and validate TPU PCP settings."""

    uses_contiguous_pcp_layout: bool = True
    shares_engram_host_table_across_pcp: bool = True

    @staticmethod
    def verify_and_update_model_config(model_config: ModelConfig) -> None:
        patch_registry.apply("platform_activation", model_config=model_config)
        DeepseekV4ForCausalLMConfig.verify_and_update_model_config(model_config)
        model_config.model_arch_config = model_config.get_model_arch_config()

    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        DeepseekV4ForCausalLMConfig.verify_and_update_config(vllm_config)
        parallel_config = vllm_config.parallel_config
        pcp = parallel_config.prefill_context_parallel_size
        if pcp <= 1:
            return
        if parallel_config.tensor_parallel_size != 1:
            raise NotImplementedError(
                "DeepSeek-V4.1 PCP on TPU requires --tensor-parallel-size=1, "
                f"got tensor_parallel_size={parallel_config.tensor_parallel_size} "
                f"with prefill_context_parallel_size={pcp}."
            )
        if parallel_config.decode_context_parallel_size != 1:
            raise NotImplementedError(
                "DeepSeek-V4.1 PCP on TPU does not support DCP yet, got "
                "decode_context_parallel_size="
                f"{parallel_config.decode_context_parallel_size} "
                f"with prefill_context_parallel_size={pcp}."
            )
        if parallel_config.data_parallel_size != 1:
            raise NotImplementedError(
                "DeepSeek-V4.1 PCP on TPU requires --data-parallel-size=1, "
                f"got data_parallel_size={parallel_config.data_parallel_size} "
                f"with prefill_context_parallel_size={pcp}."
            )
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "DeepSeek-V4.1 PCP on TPU does not support speculative decoding yet."
            )
        if (
            vllm_config.model_config.hf_config.n_routed_experts > 0
            and not parallel_config.enable_expert_parallel
        ):
            raise NotImplementedError(
                "DeepSeek-V4.1 PCP on TPU requires --enable-expert-parallel."
            )


class _TPUKimiK3Config(VerifyAndUpdateConfig):
    """Keep K3 MXFP4 checkpoints on TPU's packed-weight loader."""


class _TPUQwen2Config(VerifyAndUpdateConfig):
    """Pad Qwen2 GQA query heads per KV group when heads % tp_size != 0."""

    @staticmethod
    def verify_and_update_config(vllm_config) -> None:
        update_gqa_head_padding_config(
            vllm_config.model_config,
            vllm_config.parallel_config,
        )


def install_pcp_model_capabilities() -> None:
    """Install TPU PCP capability properties on ``ModelConfig``."""
    for attr_name in (
        "uses_contiguous_pcp_layout",
        "shares_engram_host_table_across_pcp",
    ):
        if not hasattr(VerifyAndUpdateConfig, attr_name):
            setattr(VerifyAndUpdateConfig, attr_name, False)

    MODELS_CONFIG_MAP["DeepseekV41ForCausalLM"] = _TPUDeepseekV41Config

    def _uses_contiguous_pcp_layout(self: ModelConfig) -> bool:
        cfg_cls: Any = MODELS_CONFIG_MAP.get(self.architecture)
        return bool(cfg_cls is not None and cfg_cls.uses_contiguous_pcp_layout)

    def _shares_engram_host_table_across_pcp(self: ModelConfig) -> bool:
        cfg_cls: Any = MODELS_CONFIG_MAP.get(self.architecture)
        return bool(cfg_cls is not None and cfg_cls.shares_engram_host_table_across_pcp)

    for attr_name, getter in (
        ("uses_contiguous_pcp_layout", _uses_contiguous_pcp_layout),
        (
            "shares_engram_host_table_across_pcp",
            _shares_engram_host_table_across_pcp,
        ),
    ):
        if attr_name not in ModelConfig.__dict__:
            setattr(ModelConfig, attr_name, property(getter))


def register_models() -> None:
    from vllm.model_executor.models import ModelRegistry

    from vllm_torchtpu.models.vllm.qwen4_exp.config import TPUQwen4ExpConfig

    install_pcp_model_capabilities()
    ModelRegistry.register_model("KimiLinearForCausalLM", _KIMI_MODEL)
    ModelRegistry.register_model("KimiK3ForConditionalGeneration", _KIMI_K3_MODEL)
    ModelRegistry.register_model("DeepseekV4ForCausalLM", _DSV4_MODEL)
    ModelRegistry.register_model("DeepseekV41ForCausalLM", _DSV41_MODEL)
    ModelRegistry.register_model("K3DSparkModel", _K3_DSPARK_MODEL)
    ModelRegistry.register_model("DSparkDraftModel", _DSV4_DSPARK_MODEL)
    ModelRegistry.register_model("DSparkV41DraftModel", _DSV41_DSPARK_MODEL)
    # The Qwen3.5 MTP drafts, so a pipeline-parallel producer can load them.
    ModelRegistry.register_model("Qwen3_5MTP", _QWEN35_MTP_MODEL)
    ModelRegistry.register_model("Qwen3_5MoeMTP", _QWEN35_MOE_MTP_MODEL)
    ModelRegistry.register_model("Qwen4ExpForCausalLM", _QWEN4_EXP_MODEL)
    # The released checkpoint declares the vision-language arch; serve it with
    # the text-only model.
    ModelRegistry.register_model("Qwen4ExpForConditionalGeneration", _QWEN4_EXP_MODEL)
    MODELS_CONFIG_MAP["DeepseekV41ForCausalLM"] = _TPUDeepseekV41Config
    MODELS_CONFIG_MAP["KimiK3ForConditionalGeneration"] = _TPUKimiK3Config
    MODELS_CONFIG_MAP["Qwen4ExpForCausalLM"] = TPUQwen4ExpConfig
    MODELS_CONFIG_MAP["Qwen4ExpForConditionalGeneration"] = TPUQwen4ExpConfig
    MODELS_CONFIG_MAP["Qwen2ForCausalLM"] = _TPUQwen2Config


install_pcp_model_capabilities()
