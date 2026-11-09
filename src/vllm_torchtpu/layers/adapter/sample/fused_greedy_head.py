# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
