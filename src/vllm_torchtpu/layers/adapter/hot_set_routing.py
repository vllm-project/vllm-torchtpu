# SPDX-License-Identifier: Apache-2.0
"""Skewed MoE routing for profiling: a mix of a fixed hot set and uniform choice.

Uniform random routing (``VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random``) gives
every expert the same load, so an expert-parallel profile never shows the straggler
rank that real routing produces. This strategy routes a fraction ``rate`` of the tokens
entirely into ``hot_size`` fixed experts and the rest uniformly over all experts, which
reproduces a chosen per-expert load skew with dummy weights. Profiling only: outputs
are meaningless.

Expected load per expert (mean = T * k / E): hot experts carry ``rate * E / hot_size +
(1 - rate)`` times the mean, the others ``1 - rate``. The coefficient of variation over
experts that follows is printed at start-up so a run can be labelled by its skew.

Hot experts are the first ``hot_size`` ids when ``spread`` is false (contiguous: they
sit on the first expert-parallel ranks, the worst case for a straggler) or spaced
``E / hot_size`` apart when true (one per rank where the counts divide, the balanced
control).
"""

from __future__ import annotations

import math

import torch


def expected_load_cv(num_experts: int, hot_size: int, rate: float) -> float:
    """Coefficient of variation of the expected per-expert load."""
    hot = rate * num_experts / hot_size + (1.0 - rate)
    cold = 1.0 - rate
    frac = hot_size / num_experts
    var = frac * (hot - 1.0) ** 2 + (1.0 - frac) * (cold - 1.0) ** 2
    return math.sqrt(var)


def hot_set_ids(num_experts: int, hot_size: int, spread: bool) -> torch.Tensor:
    if spread:
        stride = num_experts // hot_size
        return torch.arange(hot_size, dtype=torch.int64) * stride
    return torch.arange(hot_size, dtype=torch.int64)


class HotSetRouting:
    """Routing strategy with the ``route_tokens`` interface of the RoutingSimulator."""

    def __init__(self, hot_size: int, rate: float, spread: bool = False):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"hot-set rate must be in [0, 1], got {rate}")
        if hot_size < 1:
            raise ValueError(f"hot-set size must be >= 1, got {hot_size}")
        self.hot_size = hot_size
        self.rate = rate
        self.spread = spread
        self._hot_mask: torch.Tensor | None = None

    def _mask(self, num_experts: int, device: torch.device) -> torch.Tensor:
        if self._hot_mask is None or self._hot_mask.device != device:
            if self.hot_size > num_experts:
                raise ValueError(
                    f"hot-set size {self.hot_size} exceeds {num_experts} experts"
                )
            mask = torch.zeros(num_experts, dtype=torch.bool)
            mask[hot_set_ids(num_experts, self.hot_size, self.spread)] = True
            self._hot_mask = mask.to(device)
        return self._hot_mask

    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        indices_type: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens, num_experts = router_logits.shape
        device = router_logits.device
        hot = self._mask(num_experts, device)
        # Gumbel top-k = weighted sampling without replacement. A hot token's weights
        # make every hot expert outrank every cold one, so it picks k of the hot set
        # (padding from the cold set only when the hot set is smaller than k).
        gumbel = -torch.log(
            -torch.log(torch.rand(num_tokens, num_experts, device=device))
        )
        is_hot_token = torch.rand(num_tokens, 1, device=device) < self.rate
        boost = torch.where(is_hot_token & hot[None, :], 64.0, 0.0)
        # Anchor to the activations so XLA keeps this layer's routing in place
        # (uniform shift, same argsort as without it).
        scores = gumbel + boost + router_logits.float().mean()
        _, topk_ids = torch.topk(scores, k=top_k, dim=-1)
        topk_weights = torch.full(
            (num_tokens, top_k), 1.0 / top_k, dtype=torch.float32, device=device
        )
        ids_dtype = indices_type or torch.int32
        return topk_weights.to(hidden_states.dtype), topk_ids.to(ids_dtype)
