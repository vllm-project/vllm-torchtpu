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
"""Gather-free top-k softmax MoE routing for Gemma 4 on TPU.

Upstream ``gemma4_routing_function_torch`` computes a full softmax over all
experts, masks to top-k, renormalizes, and gathers ``per_expert_scale`` at the
selected expert indices. Renormalizing a full softmax restricted to the top-k
entries equals ``softmax`` over the top-k logits directly, and replacing the
1-D index gather with a broadcast compare-select-sum avoids scalar gather loops
during XLA lowering.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def gemma4_topk_softmax_routing(
    gating_output: torch.Tensor,
    topk: int,
    per_expert_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in TPU replacement for ``gemma4_routing_function_torch``."""
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


gemma4_topk_softmax_routing._TPU_PATCHED = True  # type: ignore[attr-defined]


def apply_gemma4_routing_patch(gemma4_mod: Any) -> bool:
    """Install gather-free top-k softmax routing on ``gemma4_mod`` idempotently."""
    current = getattr(gemma4_mod, "gemma4_routing_function_torch", None)
    if not callable(current):
        raise TypeError("vLLM gemma4_routing_function_torch is not callable")
    if getattr(current, "_TPU_PATCHED", False):
        return False
    gemma4_mod.gemma4_routing_function_torch = gemma4_topk_softmax_routing
    logger.info("Applied TPU patch: Gemma 4 gather-free top-k softmax MoE routing.")
    return True
