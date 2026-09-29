# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Hardware-adapted Root Mean Square Normalization (RMSNorm) for TPU.

Computes normalization in float32 for numerical stability on TPU matrix cores
and casts back to the input activation dtype.

Strictly zero dependencies on vLLM.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization adapted for TPU execution."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.var_hidden_size = var_hidden_size or hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply RMS normalization to input tensor x, optionally adding residual."""
        if residual is not None:
            x = x + residual
            residual_out = x
        else:
            residual_out = None

        orig_dtype = x.dtype
        # TPU numerical stability: compute variance in float32
        x_f32 = x.to(torch.float32)
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        normed = x_f32 * torch.rsqrt(variance + self.variance_epsilon)
        out = (normed * self.weight.to(torch.float32)).to(orig_dtype)

        if residual_out is not None:
            return out, residual_out
        return out

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.variance_epsilon}"
