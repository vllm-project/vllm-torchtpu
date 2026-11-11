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
    register_model_directories()


def register_model_directories() -> None:
    """Call `register()` of every model directory under `vllm_torchtpu.models`.

    A model directory (`models/<model>/`, see its README) registers its own
    architectures, config hooks, quant methods and KV cache dtypes, so adding
    a model does not edit this file. Its `__init__` must stay import-free:
    this runs at plugin load in every process, before vLLM selects a model.
    """
    import importlib
    import pkgutil

    import vllm_torchtpu.models as models

    for module in sorted(pkgutil.iter_modules(models.__path__), key=lambda m: m.name):
        if not module.ispkg or module.name == "vllm":
            continue
        package = importlib.import_module(f"{models.__name__}.{module.name}")
        register = getattr(package, "register", None)
        if callable(register):
            register()
