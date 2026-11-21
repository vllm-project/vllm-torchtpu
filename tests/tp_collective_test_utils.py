# Copyright 2026 Google LLC
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
"""Shared in-process tensor-parallel collective simulator for multi-rank CPU tests."""

from __future__ import annotations

import threading

import torch


class TPCollectiveMock:
    """Barrier-synchronized in-process collective simulator for TP ranks."""

    def __init__(self, tp_size: int) -> None:
        self.tp_size = tp_size
        self.rank = threading.local()
        self.barrier = threading.Barrier(tp_size)
        self.storage: list[torch.Tensor] = [torch.empty(0)] * tp_size
        self.calls = {"all_reduce": 0, "all_gather": 0, "reduce_scatter": 0}

    def exchange(
        self, tensor: torch.Tensor, name: str | None = None
    ) -> list[torch.Tensor]:
        if name is not None and self.rank.value == 0:
            self.calls[name] += 1
        self.storage[self.rank.value] = tensor
        self.barrier.wait(timeout=10.0)
        parts = list(self.storage)
        self.barrier.wait(timeout=10.0)
        return parts

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return tensor
        return torch.stack(self.exchange(tensor, "all_reduce")).sum(0)

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if self.tp_size == 1:
            return tensor
        return torch.cat(self.exchange(tensor, "all_gather"), dim=dim)

    def reduce_scatter(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if self.tp_size == 1:
            return tensor
        total = torch.stack(self.exchange(tensor, "reduce_scatter")).sum(0)
        return total.chunk(self.tp_size, dim=dim)[self.rank.value]
