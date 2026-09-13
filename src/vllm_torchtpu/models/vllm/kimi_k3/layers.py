# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small Kimi layers that are not attention or MoE."""

from __future__ import annotations

import torch
from torch import nn
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig


class SituAndMul(nn.Module):
    """Kimi's SITU gated activation.

    Keep this implementation local because released vLLM versions do not
    expose ``SituAndMul`` from their generic activation module.
    """

    def __init__(self, beta: float, linear_beta: float | None) -> None:
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        gate = (self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate))
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return gate * up


class KimiMLP(nn.Module):
    """The standard vLLM merged-projection gated MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        reduce_results: bool = True,
        situ_beta: float | None = None,
        situ_linear_beta: float | None = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act == "silu":
            self.act_fn = SiluAndMul()
        elif hidden_act == "situ":
            self.act_fn = SituAndMul(
                beta=situ_beta or 1.0,
                linear_beta=situ_linear_beta,
            )
        else:
            raise ValueError(f"Unsupported Kimi activation {hidden_act!r}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class AttentionResidual(nn.Module):
    """K3's attention-residual weighted sum in ordinary PyTorch."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        prefix: str,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.norm = RMSNorm(hidden_size, eps)
        self.proj = ReplicatedLinear(
            hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=prefix,
        )

    def process_weights_after_loading(self) -> None:
        """Fold the norm and projection weights into one [hidden] vector."""
        folded = self.norm.weight.float() * self.proj.weight[0].float()
        out = torch.empty_like(folded)
        out.copy_(folded)
        self.folded_weight = out

    def forward(
        self,
        prefix_sum: torch.Tensor,
        block_residuals: torch.Tensor,
    ) -> torch.Tensor:
        w = self.folded_weight
        if prefix_sum.shape[0] == 1:
            # With one token the per-slot statistics below are scalars that
            # XLA leaves unfused; reducing over the concatenated slots keeps
            # them one vector. A shape branch, so bucket 1 gets its own trace
            # (see compilation.shape_variants).
            values = torch.cat((block_residuals, prefix_sum.unsqueeze(-2)),
                               dim=-2).float()  # [1, K, hidden]
            inv = torch.rsqrt(values.pow(2).mean(-1) + self.eps)
            probabilities = ((values * w).sum(-1) * inv).softmax(dim=-1)
            out = (probabilities.unsqueeze(-1) * values).sum(dim=-2)
            return out.to(block_residuals.dtype)
        v_blocks = block_residuals.float()
        v_prefix = prefix_sum.float()
        inv_blocks = torch.rsqrt(v_blocks.pow(2).mean(-1) + self.eps)
        inv_prefix = torch.rsqrt(v_prefix.pow(2).mean(-1) + self.eps)
        scores = torch.cat(
            ((v_blocks * w).sum(-1) * inv_blocks,
             ((v_prefix * w).sum(-1) * inv_prefix).unsqueeze(-1)),
            dim=-1)  # [T, K]
        probabilities = scores.softmax(dim=-1)
        out = (probabilities[..., :-1].unsqueeze(-1) * v_blocks).sum(dim=-2)
        out = out + probabilities[..., -1].unsqueeze(-1) * v_prefix
        return out.to(block_residuals.dtype)
