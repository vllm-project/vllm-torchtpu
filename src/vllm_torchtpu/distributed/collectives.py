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
"""Collectives over a vLLM group that its GroupCoordinator does not provide."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator


def all_to_all(x: torch.Tensor, group: "GroupCoordinator") -> torch.Tensor:
    """Block `d` of `x`'s leading axis goes to rank `d`; block `r` of the result
    came from rank `r`."""
    from torch.distributed import _functional_collectives as funcol

    out = funcol.all_to_all_single(x.contiguous(), None, None, group.device_group)
    if isinstance(out, funcol.AsyncCollectiveTensor):
        return funcol.wait_tensor(out)
    return out
