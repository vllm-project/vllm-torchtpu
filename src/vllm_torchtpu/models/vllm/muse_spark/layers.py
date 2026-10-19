# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unfused Muse math. Norms and residual arithmetic deliberately stay in FP32."""

import torch
from torch import nn
from torch.nn import functional as F

from vllm_torchtpu.layers.adapter.router_topk import rowmax_select


class MuseRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        *,
        has_weight: bool = True,
        zero_centered: bool = False,
        center: float = 1.0,
        weight_first: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.center = center if zero_centered else 0.0
        self.weight_first = weight_first
        self.weight = (
            nn.Parameter(torch.empty(hidden_size), requires_grad=False)
            if has_weight
            else None
        )

    def forward(
        self, x: torch.Tensor, out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        dtype = out_dtype or x.dtype
        y = x.float()
        weight = None if self.weight is None else self.weight.float() + self.center
        if self.weight_first and weight is not None:
            y = y * weight
        y = y * torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + self.eps)
        if not self.weight_first and weight is not None:
            y = y * weight
        return y.to(dtype)


class MuseResidual(nn.Module):
    def __init__(self, hidden_size: int, temperature: float):
        super().__init__()
        self.gate = nn.Parameter(torch.empty(hidden_size), requires_grad=False)
        self.temperature = temperature

    def forward(self, stream: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        branch = branch.to(stream.dtype)
        w = self.gate.float() / self.temperature
        beta = torch.sigmoid(w)
        alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1 + beta), min=1e-3))
        return (alpha * stream + beta * branch).to(stream.dtype)


def route_experts(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    top_k: int,
    multiplier: float,
    scoring: str,
    eps: float | None,
    route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = logits.float() * multiplier
    scores = logits.sigmoid() if scoring == "sigmoid" else logits.softmax(dim=-1)
    _, ids = rowmax_select(scores + bias.float(), top_k)
    weights = scores.gather(-1, ids.long())
    weights = weights / (weights.sum(dim=-1, keepdim=True) + (eps or 0.0))
    return weights * route_scale, ids


def selected_expert_outputs(
    x: torch.Tensor, ids: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    """Reference BF16 expert outputs [tokens, top_k, hidden], before normalization."""
    tokens, top_k = ids.shape
    flat_ids = ids.reshape(-1)
    inputs = x[:, None, :].expand(-1, top_k, -1).reshape(-1, 1, x.shape[-1])
    gate, up = torch.bmm(inputs, w13.index_select(0, flat_ids)).chunk(2, dim=-1)
    activated = (F.silu(gate.float()) * up.float()).to(x.dtype)
    rows = torch.bmm(activated, w2.index_select(0, flat_ids))
    return rows.reshape(tokens, top_k, x.shape[-1])


def attention_layout(config, layer_idx: int) -> tuple[bool, int | None]:
    use_rope = config.layer_rope_theta[layer_idx] != 0
    window = (
        config.sliding_window
        if config.layer_types[layer_idx] == "sliding_attention"
        else None
    )
    return use_rope, window


class MuseRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_length: int, theta: float):
        super().__init__()
        # Build only the serving-context RoPE table on CPU.
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, device="cpu").float() / head_dim)
        )
        angles = torch.outer(torch.arange(max_length, device="cpu").float(), inv_freq)
        device = torch.empty(0).device
        self.register_buffer("cos", angles.cos().to(device), persistent=False)
        self.register_buffer("sin", angles.sin().to(device), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[positions].unsqueeze(1)
        sin = self.sin[positions].unsqueeze(1)
        real, imag = x.float().chunk(2, dim=-1)
        return torch.cat((real * cos - imag * sin, real * sin + imag * cos), dim=-1).to(
            x.dtype
        )
