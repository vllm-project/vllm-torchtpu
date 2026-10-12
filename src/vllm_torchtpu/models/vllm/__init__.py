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

_KIMI_MODEL = "vllm_torchtpu.models.vllm.kimi_k3:KimiLinearForCausalLM"
_KIMI_K3_MODEL = "vllm_torchtpu.models.vllm.kimi_k3:KimiK3ForConditionalGeneration"
_DSV4_MODEL = "vllm_torchtpu.models.vllm.deepseek_v4:DeepseekV4ForCausalLM"
_K3_DSPARK_MODEL = "vllm_torchtpu.models.vllm.kimi_k3:K3DSparkForCausalLM"
_DSV4_DSPARK_MODEL = (
    "vllm_torchtpu.models.vllm.deepseek_v4.dspark:DSparkDeepseekV4ForCausalLM"
)
_QWEN35_MTP_MODEL = "vllm_torchtpu.models.vllm.qwen3_5_mtp:Qwen3_5MTP"
_QWEN35_MOE_MTP_MODEL = "vllm_torchtpu.models.vllm.qwen3_5_mtp:Qwen3_5MoeMTP"


class _TPUKimiK3Config(VerifyAndUpdateConfig):
    """Keep K3 MXFP4 checkpoints on TPU's packed-weight loader."""


def register_models() -> None:
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    ModelRegistry.register_model("KimiLinearForCausalLM", _KIMI_MODEL)
    ModelRegistry.register_model("KimiK3ForConditionalGeneration", _KIMI_K3_MODEL)
    ModelRegistry.register_model("DeepseekV4ForCausalLM", _DSV4_MODEL)
    ModelRegistry.register_model("K3DSparkModel", _K3_DSPARK_MODEL)
    ModelRegistry.register_model("DSparkDraftModel", _DSV4_DSPARK_MODEL)
    # The Qwen3.5 MTP drafts, so a pipeline-parallel producer can load them.
    ModelRegistry.register_model("Qwen3_5MTP", _QWEN35_MTP_MODEL)
    ModelRegistry.register_model("Qwen3_5MoeMTP", _QWEN35_MOE_MTP_MODEL)
    MODELS_CONFIG_MAP["KimiK3ForConditionalGeneration"] = _TPUKimiK3Config
