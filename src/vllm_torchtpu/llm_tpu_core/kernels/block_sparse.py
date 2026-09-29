# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Block-sparse kernels and helper routines for TPU.

Strictly zero dependencies on vLLM.
"""

from __future__ import annotations

import torch


def compute_sparse_block_offsets(
    block_indices: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Compute contiguous element offsets for block-sparse gathers."""
    return block_indices * block_size
