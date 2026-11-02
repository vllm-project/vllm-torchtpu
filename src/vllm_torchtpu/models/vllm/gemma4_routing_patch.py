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
"""Gemma4 MoE routing for TPU without gathers.

Upstream's torch routing (the non-CUDA path) takes a softmax over all
experts, masks it to the top-k with a one-hot indicator, renormalizes,
gathers the top-k weights and gathers ``per_expert_scale`` at the top-k ids.
XLA lowers the gathers from the [num_experts] table as slow scalar loops at
prefill sizes. Renormalizing a full softmax over the top-k is the softmax of
the top-k logits, and the scale lookup is a compare-select-sum over the
experts axis, so this routing has no gather and the same top-k ids.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)


def gemma4_topk_softmax_routing(
    gating_output: torch.Tensor,
    topk: int,
    per_expert_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in for upstream ``gemma4_routing_function_torch``."""
    topk_logits, topk_ids = torch.topk(gating_output, k=topk, dim=-1)
    topk_weights = torch.softmax(topk_logits.to(torch.float32), dim=-1)
    experts = torch.arange(
        gating_output.shape[-1], device=topk_ids.device, dtype=topk_ids.dtype
    )
    selected = topk_ids.unsqueeze(-1) == experts
    expert_scales = torch.where(selected, per_expert_scale.to(torch.float32), 0.0).sum(
        dim=-1
    )
    return topk_weights * expert_scales, topk_ids.to(torch.int32)


def maybe_patch_gemma4_routing(
    model_config: ModelConfig | None = None,
) -> bool | None:
    """Returns False until a Gemma4 model config is available."""
    if not envs.TPU_GEMMA4_TOPK_SOFTMAX_ROUTING:
        return None
    if model_config is None or model_config.hf_text_config.model_type != "gemma4_text":
        return False
    import vllm.model_executor.models.gemma4 as gemma4_mod

    # Gemma4Router.routing_function resolves this module global per call, and
    # no routing extension point reaches it. Fail loudly if upstream renames it.
    if not callable(gemma4_mod.gemma4_routing_function_torch):
        raise TypeError("vLLM gemma4_routing_function_torch is not a function")
    gemma4_mod.gemma4_routing_function_torch = gemma4_topk_softmax_routing
    logger.info("Applied TPU patch: Gemma4 MoE top-k softmax routing without gathers.")
    return None
