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
"""Model-owned runner factories and communication requirements."""

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_torchtpu.runner.model_adapters import ModelAdapter
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _qwen4_adapter(runner: "TPUModelRunner") -> "ModelAdapter":
    from vllm_torchtpu.models.vllm.qwen4_exp.inputs import Qwen4ExpAdapter

    return Qwen4ExpAdapter(runner)


def _qwen4_dp_collectives(config: "VllmConfig") -> bool:
    options = config.additional_config
    if not isinstance(options, Mapping):
        return False
    qwen = options.get("qwen4_exp", {})
    engram = qwen.get("engram_config", {})
    return bool(engram.get("embedding_across_dp", False))


_QWEN4_ARCHITECTURES = ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration")
MODEL_ADAPTERS: dict[str, Callable[["TPUModelRunner"], "ModelAdapter"]] = dict.fromkeys(
    _QWEN4_ARCHITECTURES, _qwen4_adapter
)
MODEL_DP_COLLECTIVES: dict[str, Callable[["VllmConfig"], bool]] = dict.fromkeys(
    _QWEN4_ARCHITECTURES, _qwen4_dp_collectives
)
