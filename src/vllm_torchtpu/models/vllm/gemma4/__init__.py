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
"""Flag-free TPU model adaptations for Gemma 4 dense and MoE checkpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import ModelConfig


def is_gemma4_model_config(model_config: Any) -> bool:
    """Return True strictly when ``model_config`` targets a Gemma 4 text backbone."""
    if model_config is None:
        return False
    hf_text_config = getattr(model_config, "hf_text_config", None)
    return getattr(hf_text_config, "model_type", None) == "gemma4_text"


def maybe_patch_gemma4(model_config: ModelConfig | None = None) -> bool | None:
    """Apply Gemma 4 TPU patches when ``model_config`` is Gemma 4; return False else."""
    if not is_gemma4_model_config(model_config):
        return False

    from vllm.model_executor.models import gemma4 as gemma4_mod

    from vllm_torchtpu import tpu_info
    from vllm_torchtpu.models.vllm.gemma4.attention import apply_gemma4_attention_patch
    from vllm_torchtpu.models.vllm.gemma4.cp_prefill import (
        apply_gemma4_context_parallel_patch,
    )
    from vllm_torchtpu.models.vllm.gemma4.layers import apply_gemma4_mlp_patch
    from vllm_torchtpu.models.vllm.gemma4.routing import apply_gemma4_routing_patch
    from vllm_torchtpu.models.vllm.gemma4.sequence_parallel import (
        apply_gemma4_sequence_parallel_patch,
    )

    tpu_info.set_v7x_model_fastpaths(True)
    apply_gemma4_routing_patch(gemma4_mod)
    apply_gemma4_mlp_patch(gemma4_mod)
    apply_gemma4_attention_patch(gemma4_mod)
    apply_gemma4_sequence_parallel_patch(gemma4_mod)
    apply_gemma4_context_parallel_patch(gemma4_mod)
    return None
