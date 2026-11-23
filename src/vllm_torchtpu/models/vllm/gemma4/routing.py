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
"""Gather-free MoE top-k softmax routing for Gemma 4 on TPU."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

ROUTER_TOPK_KERNEL_EXPERTS = 128
ROUTER_TOPK_KERNEL_K = 8

_select_scaled_fn: Callable[..., tuple[torch.Tensor, ...]] | None = None


def _get_select_scaled() -> Callable[..., tuple[torch.Tensor, ...]]:
    global _select_scaled_fn
    if _select_scaled_fn is None:
        from vllm_torchtpu.layers.adapter.router_topk import rowmax_select_scaled

        _select_scaled_fn = rowmax_select_scaled
    fn = _select_scaled_fn
    assert fn is not None
    return fn


def uses_router_topk_kernel(gating_output: torch.Tensor, topk: int) -> bool:
    """Return True when the sort-free TPU router kernel supports this call."""
    return (
        gating_output.device.type == "tpu"
        and gating_output.dim() == 2
        and gating_output.shape[-1] == ROUTER_TOPK_KERNEL_EXPERTS
        and topk == ROUTER_TOPK_KERNEL_K
    )


def gemma4_topk_softmax_routing(
    gating_output: torch.Tensor, topk: int, per_expert_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in TPU replacement for ``gemma4_routing_function_torch``."""
    if uses_router_topk_kernel(gating_output, topk):
        topk_logits, topk_ids, expert_scales = _get_select_scaled()(
            gating_output, topk, per_expert_scale
        )
    else:
        topk_logits, topk_ids = torch.topk(gating_output, k=topk, dim=-1)
        experts = torch.arange(
            gating_output.shape[-1], device=topk_ids.device, dtype=topk_ids.dtype
        )
        selected = topk_ids.unsqueeze(-1) == experts
        expert_scales = torch.where(
            selected, per_expert_scale.to(torch.float32), 0.0
        ).sum(dim=-1)
    topk_weights = torch.softmax(topk_logits.to(torch.float32), dim=-1)
    return topk_weights * expert_scales, topk_ids.to(torch.int32)


gemma4_topk_softmax_routing._TPU_GEMMA4_PATCHED = True  # type: ignore[attr-defined]


def apply_gemma4_routing_patch(gemma4_mod: Any) -> bool:
    """Install gather-free top-k softmax routing on ``gemma4_mod`` idempotently."""
    current = getattr(gemma4_mod, "gemma4_routing_function_torch", None)
    if not callable(current):
        raise TypeError("vLLM gemma4_routing_function_torch is not callable")
    if getattr(current, "_TPU_GEMMA4_PATCHED", False):
        return False
    gemma4_mod.gemma4_routing_function_torch = gemma4_topk_softmax_routing
    logger.info("Applied TPU patch: Gemma 4 gather-free top-k softmax MoE routing.")
    return True
