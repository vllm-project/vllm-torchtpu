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
"""TPU model patches for Gemma 4 CausalLM and ConditionalGeneration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.model_executor.models import gemma4 as gemma4_mod

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.gemma4.cp_attention import (
    apply_gemma4_context_parallel_patch,
)
from vllm_torchtpu.models.vllm.gemma4.layers import apply_gemma4_layer_patches
from vllm_torchtpu.models.vllm.gemma4.routing import apply_gemma4_routing_patch
from vllm_torchtpu.models.vllm.gemma4.sequence_parallel import (
    apply_gemma4_sequence_parallel_patch,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)


def maybe_patch_gemma4(model_config: ModelConfig | None = None) -> bool | None:
    """Apply Gemma 4 TPU model patches idempotently when ``model_config`` matches."""
    hf_text_config = getattr(model_config, "hf_text_config", None)
    if getattr(hf_text_config, "model_type", None) != "gemma4_text":
        return False

    apply_gemma4_routing_patch(gemma4_mod)
    apply_gemma4_layer_patches()
    apply_gemma4_sequence_parallel_patch()
    apply_gemma4_context_parallel_patch(model_config)
    return None
