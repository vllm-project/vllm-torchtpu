# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Primitive distributed collective operations for TPU.

Strictly zero dependencies on vLLM.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def all_reduce_sum(
    tensor: torch.Tensor,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """All-reduce sum across the specified or default group."""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor


def all_gather_tensor(
    tensor: torch.Tensor,
    dim: int = 0,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """All-gather tensor along a specific dimension."""
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return tensor
    output_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(output_tensors, tensor, group=group)
    return torch.cat(output_tensors, dim=dim)
