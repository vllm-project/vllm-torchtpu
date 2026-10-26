# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch bridge for the qualified packed greedy head."""

from dataclasses import dataclass

import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.fused_greedy_head import TUNED_TILES, fused_greedy_head
from vllm_torchtpu.utils import get_device_name, synchronize_tensors

_head_op = pallas.jax_op("pallas::fused_greedy_head", fused_greedy_head)


@dataclass
class GreedyTokenIds:
    token_ids: torch.Tensor


class PackedGreedyHead:
    """Immutable inference weight snapshot; rebuild after any weight update."""

    def __init__(self, weight: torch.Tensor, rows: int):
        vocab, width = weight.shape
        key = (rows, width, vocab)
        assert get_device_name() == "TPU v7"
        assert weight.dtype == torch.bfloat16 and key in TUNED_TILES
        tile = TUNED_TILES[key]
        self.weight = (
            weight.detach()
            .cpu()
            .reshape(vocab // tile, tile, width)
            .transpose(1, 2)
            .contiguous()
            .to(weight.device)
        )
        synchronize_tensors(self.weight)
        self.rows = rows

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        return _head_op(hidden, self.weight)
