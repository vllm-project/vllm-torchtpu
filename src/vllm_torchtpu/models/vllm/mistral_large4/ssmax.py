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
"""Native ML4 scalable-softmax query scaling."""

from __future__ import annotations

import torch
from torch import nn
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import set_weight_attrs


class SSMaxScaling(nn.Module):
    """Activation-dtype scaling with sharded attention or replicated index heads."""

    def __init__(self, num_heads: int, *, tp_sharded: bool = True) -> None:
        super().__init__()
        self.s = nn.Parameter(torch.empty(num_heads), requires_grad=False)
        self.b = nn.Parameter(torch.empty(num_heads), requires_grad=False)
        if tp_sharded:
            for param in (self.s, self.b):
                set_weight_attrs(param, {"weight_loader": sharded_weight_loader(0)})

    def forward(self, q: torch.Tensor, log_positions: torch.Tensor) -> torch.Tensor:
        scale = (
            log_positions.to(q.dtype)[:, None, None] * self.s.to(q.dtype)[:, None]
            + self.b.to(q.dtype)[:, None]
        )
        return q * scale
