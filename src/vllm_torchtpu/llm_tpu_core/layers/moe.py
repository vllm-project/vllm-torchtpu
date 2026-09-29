# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Hardware-adapted Mixture of Experts (MoE) operations for TPU.

Strictly zero dependencies on vLLM.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Clamp floor. Below any real routing score, and strictly above the dtype
# minimum used to mask a selected column -- see ``kernels/router_topk.py``.
NEG = -3.0e38


def rowmax_topk(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``torch.topk(scores, k, dim=-1)`` without the sort.

    Returns ``(values, indices)``, values descending and indices int32. Ties
    resolve to the lowest expert id; the sort path leaves them undefined,
    since torch_tpu emits ``is_stable=false``.

    Scores at or below ``NEG`` are clamped up to it, so every returned index
    is a real expert. A row made entirely of such scores keeps NaN weights and
    still names k distinct experts, as the sort path does.
    """
    n = scores.shape[-1]
    if k > n:
        raise ValueError(f"topk k={k} exceeds the expert count {n}")
    iota = torch.arange(n, device=scores.device, dtype=torch.int32)
    masked = torch.finfo(scores.dtype).min

    cur = torch.where(scores > NEG, scores, NEG)
    values, indices = [], []
    for _ in range(k):
        m = torch.amax(cur, dim=-1, keepdim=True)
        # Lowest column attaining the max; the rest fill with the last expert
        # so the min never selects a phantom.
        idx = torch.amin(torch.where(cur == m, iota, n - 1), dim=-1, keepdim=True)
        values.append(m)
        indices.append(idx)
        cur = torch.where(iota == idx, masked, cur)

    weights = torch.cat(values, dim=-1)
    ids = torch.cat(indices, dim=-1)
    # No score above the sentinel means no real maximum: keep the row
    # poisonous so the caller's renormalization propagates it.
    weights = torch.where(weights[..., :1] <= NEG, torch.nan, weights)
    return weights, ids


class TopKResult(NamedTuple):
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor


class MoERouter(nn.Module):
    """Standard top-k router for TPU Mixture-of-Experts layers."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        renormalize: bool = True,
        use_rowmax: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.renormalize = renormalize
        self.use_rowmax = use_rowmax
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> TopKResult:
        logits = self.gate(hidden_states)
        if self.use_rowmax:
            raw_weights, ids = rowmax_topk(logits, self.top_k)
        else:
            raw_weights, ids = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(raw_weights.to(torch.float32), dim=-1).to(
            hidden_states.dtype
        )
        if self.renormalize:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return TopKResult(topk_weights=weights, topk_ids=ids)


__all__ = [
    "NEG",
    "MoERouter",
    "TopKResult",
    "rowmax_topk",
]
