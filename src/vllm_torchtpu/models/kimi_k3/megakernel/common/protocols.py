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
"""Shared structural typing protocols for Kimi-K3 megakernel modules."""

from __future__ import annotations

from typing import Protocol

import torch


class InPlaceSetTensor(Protocol):
    """Protocol for tensors supporting in-place storage binding via ``set_``."""

    def set_(self, source: torch.Tensor) -> torch.Tensor:
        """Rebind storage in place to ``source``."""
        ...


class DsparkKernelProtocol(Protocol):
    """Structural protocol for layout constants exposed by ``kernels.dspark``."""

    ROWS: int
    OUT_COUNT: int
    OUT_BONUS: int
    OUT_DRAFTS: int
