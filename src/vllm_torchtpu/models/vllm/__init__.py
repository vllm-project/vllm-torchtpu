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
from vllm.model_executor.models.config import VerifyAndUpdateConfig

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
_ERNIE5_5_MODEL = "vllm_torchtpu.models.vllm.ernie5_5:Ernie5_5ForCausalLM"


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


def register_models() -> None:
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    from vllm_torchtpu.models.vllm.qwen4_exp.config import TPUQwen4ExpConfig

    ModelRegistry.register_model("KimiLinearForCausalLM", _KIMI_MODEL)
    ModelRegistry.register_model("KimiK3ForConditionalGeneration", _KIMI_K3_MODEL)
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
    MODELS_CONFIG_MAP["Qwen4ExpForCausalLM"] = TPUQwen4ExpConfig
    MODELS_CONFIG_MAP["Qwen4ExpForConditionalGeneration"] = TPUQwen4ExpConfig
    MODELS_CONFIG_MAP["Qwen2ForCausalLM"] = _TPUQwen2Config
    _register_ernie5_5()


def _register_ernie5_5() -> None:
    """ERNIE 5.5: no transformers config class and no remote code in the checkpoint, so
    register the config (vLLM-ERNIE's own registration, patches.py) and mark it
    DeepSeek-MLA for vLLM's arch-config conversion."""
    from vllm.model_executor.models import ModelRegistry
    from vllm.transformers_utils.config import _CONFIG_REGISTRY
    from vllm.transformers_utils.model_arch_config_convertor import (
        MODEL_ARCH_CONFIG_CONVERTORS,
        ModelArchConfigConvertorBase,
    )

    from vllm_torchtpu.models.vllm.ernie5_5.config import Ernie5_5Config

    class _Ernie5_5ArchConfigConvertor(ModelArchConfigConvertorBase):
        def is_deepseek_mla(self) -> bool:
            return True

    _CONFIG_REGISTRY.setdefault(Ernie5_5Config.model_type, Ernie5_5Config)
    MODEL_ARCH_CONFIG_CONVERTORS.setdefault(
        Ernie5_5Config.model_type, _Ernie5_5ArchConfigConvertor
    )
    ModelRegistry.register_model("Ernie5_5_MoeForCausalLM", _ERNIE5_5_MODEL)
