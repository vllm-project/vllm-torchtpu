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
"""Qwen4Exp (Qwen3.8 Flash Next) on TPU.

Kept import-free: plugin registration imports ``config`` at startup, and the
registry imports ``model`` lazily by its module path.
"""

_MODEL = "vllm_torchtpu.models.qwen4_exp.model:Qwen4ExpForCausalLM"


def register() -> None:
    """Register the Qwen4Exp architectures and their config hook, then hand
    off to the experimental tier, which overrides them only when its
    migration selector is set."""
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    from vllm_torchtpu.models.qwen4_exp.config import TPUQwen4ExpConfig

    ModelRegistry.register_model("Qwen4ExpForCausalLM", _MODEL)
    # The released checkpoint declares the vision-language arch; serve it with
    # the text-only model.
    ModelRegistry.register_model("Qwen4ExpForConditionalGeneration", _MODEL)
    MODELS_CONFIG_MAP["Qwen4ExpForCausalLM"] = TPUQwen4ExpConfig
    MODELS_CONFIG_MAP["Qwen4ExpForConditionalGeneration"] = TPUQwen4ExpConfig

    from vllm_torchtpu.models.qwen4_exp.experimental import hooks

    hooks.register()
