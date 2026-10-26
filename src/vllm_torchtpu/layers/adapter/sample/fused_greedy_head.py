# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch bridge for the qualified greedy head."""

from dataclasses import dataclass

import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.fused_greedy_head import (
    fused_greedy_head,
    is_supported_shape,
)

_head_op = pallas.jax_op("pallas::fused_greedy_head", fused_greedy_head)


@dataclass
class GreedyTokenIds:
    token_ids: torch.Tensor


class FusedGreedyHead:
    """Greedy head sharing the original weight with other logits consumers."""

    @staticmethod
    def is_supported(weight: torch.Tensor, rows: int) -> bool:
        return weight.dtype == torch.bfloat16 and is_supported_shape(
            rows, tuple(weight.shape)
        )

    def __init__(self, weight: torch.Tensor, rows: int):
        assert self.is_supported(weight, rows), (
            f"Unsupported greedy head: rows={rows}, weight={tuple(weight.shape)}, "
            f"dtype={weight.dtype}; use TPU_GREEDY_HEAD_BACKEND=logits"
        )
        self.weight = weight
        self.rows = rows

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        return _head_op(hidden, self.weight)
