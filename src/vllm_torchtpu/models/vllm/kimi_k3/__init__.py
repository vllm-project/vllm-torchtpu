# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Kimi-K3 models for the TorchTPU vLLM plugin."""

from .attention import KimiDeltaAttention, MultiHeadLatentAttention
from .dspark import K3DSparkForCausalLM, K3DSparkModel
from .model import KimiK3ForConditionalGeneration, KimiLinearForCausalLM, KimiModel
from .moe import KimiMoE

__all__ = [
    "KimiDeltaAttention",
    "K3DSparkForCausalLM",
    "K3DSparkModel",
    "KimiK3ForConditionalGeneration",
    "KimiLinearForCausalLM",
    "KimiModel",
    "KimiMoE",
    "MultiHeadLatentAttention",
]
