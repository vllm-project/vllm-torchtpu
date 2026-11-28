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
"""Shared-core interface implementations for experimental Qwen3.8-Flash."""

from vllm_torchtpu.models.qwen38_flash.experimental import settings

QWEN4_EXP_ARCHITECTURES = (
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
)

_PKG = "vllm_torchtpu.models.qwen38_flash.experimental"
_MODEL = f"{_PKG}.model:Qwen4ExpForCausalLM"


def enabled() -> bool:
    """Whether this directory serves the Qwen4Exp architectures."""
    return bool(settings.VLLM_TPU_QWEN38_FLASH_EXPERIMENTAL)


def register() -> None:
    """Register the experimental model for the Qwen4Exp architectures.

    Called through `models/qwen38_flash.register()` by the shared model
    registration, after the default registrations. A no-op unless the
    migration selector is set.
    """
    if not enabled():
        return
    from vllm.model_executor.models import ModelRegistry

    for arch in QWEN4_EXP_ARCHITECTURES:
        ModelRegistry.register_model(arch, _MODEL)
