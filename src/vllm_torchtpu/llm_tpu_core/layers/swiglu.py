# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Hardware-adapted SwiGLU activation for TPU.

Splits input along the hidden dimension and applies SiLU gate:
SwiGLU(x) = silu(gate) * x

Strictly zero dependencies on vLLM.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU activation function adapted for TPU fused matrix operations."""

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Splits tensor into gate and value, returning silu(gate) * value."""
        gate, x_val = x.chunk(2, dim=self.dim)
        return F.silu(gate) * x_val
