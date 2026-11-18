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

# Source provenance: inferact/k3-megakernel@30c9ba966b
# src/vllm_torchtpu/megakernel/layout.py
"""A program's declaration of the KV cache pools it reads and writes.

A megakernel program works on vLLM's allocated KV cache in place, so the
layout of every pool block -- where an attention layer's token entries sit,
where a recurrent layer's state regions sit -- is part of its contract. A
:class:`KVLayout` states it once; the program's modules read their geometry
from it and :meth:`KVLayout.check` verifies vLLM's allocation against it
when the pools are bound.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch


@dataclasses.dataclass(frozen=True)
class StateRegion:
    """Contiguous row slice ``[first_row, first_row + rows)`` in a state block.

    Represents one component of a recurrent layer's state stored as ``dtype``.
    """

    name: str
    first_row: int
    rows: int
    dtype: str


@dataclasses.dataclass(frozen=True)
class KVLayout:
    """Unified KV and recurrent state pool geometry contract.

    Allocates one tensor per attention layer of shape ``[num_blocks, *block_shape]``
    and ``dtype``, shared with recurrent layers (pool ``p`` holds the ``p``-th
    equal run of ``state_layers``). An attention block holds ``tokens_per_block``
    token entries (row ``t // tokens_per_row``, slot ``t % tokens_per_row``); a
    recurrent layer's state block holds its ``state_regions``.
    """

    block_shape: tuple[int, ...]
    dtype: str
    tokens_per_block: int
    tokens_per_row: int
    attention_layers: tuple[int, ...]
    state_layers: tuple[int, ...]
    state_regions: tuple[StateRegion, ...]

    @property
    def pools(self) -> int:
        """Return the number of unified KV cache pools (one per attention layer)."""
        return len(self.attention_layers)

    def region(self, name: str) -> StateRegion:
        """Look up a recurrent state region by ``name``.

        Args:
            name: Region identifier (for example ``"recurrent"``, ``"conv"``,
                or ``"draft"``).

        Returns:
            Matching :class:`StateRegion` descriptor.
        """
        return next(reg for reg in self.state_regions if reg.name == name)

    def check(self, tensors: Sequence[torch.Tensor]) -> None:
        """Verify ``tensors`` match this layout's pool count, block shape, and dtype.

        Args:
            tensors: Sequence of allocated KV cache pool tensors from vLLM.

        Raises:
            NotImplementedError: If pool count, block shapes, dtypes, or block
                counts do not match this layout contract.
        """
        shapes = {tuple(tensor.shape[1:]) for tensor in tensors}
        dtypes = {str(tensor.dtype).removeprefix("torch.") for tensor in tensors}
        if (
            len(tensors) != self.pools
            or shapes != {self.block_shape}
            or dtypes != {self.dtype}
        ):
            msg = (
                f"the program expects {self.pools} unified pools "
                f"[blocks, {', '.join(map(str, self.block_shape))}] "
                f"{self.dtype}; vLLM allocated {len(tensors)} with block shapes "
                f"{sorted(shapes)} and {sorted(dtypes)}"
            )
            raise NotImplementedError(msg)
        if any(int(tensor.shape[0]) == 0 for tensor in tensors):
            msg = "vLLM allocated empty KV cache pools"
            raise NotImplementedError(msg)
