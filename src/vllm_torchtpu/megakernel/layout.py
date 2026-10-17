# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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


@dataclasses.dataclass(frozen=True)
class StateRegion:
    """Rows ``[first_row, first_row + rows)`` of a state block: one part of
    a recurrent layer's state, stored as ``dtype``."""

    name: str
    first_row: int
    rows: int
    dtype: str


@dataclasses.dataclass(frozen=True)
class KVLayout:
    """Unified pools: one tensor per attention layer, ``[num_blocks,
    *block_shape]`` of ``dtype``, pool ``p`` shared with recurrent layers
    ``[p * states_per_pool, (p + 1) * states_per_pool)``. An attention
    block holds ``tokens_per_block`` token entries of ``entry_lanes`` used
    lanes (row ``t // tokens_per_row``, slot ``t % tokens_per_row``); a
    recurrent layer's state block holds its ``state_regions``."""

    block_shape: tuple[int, ...]
    dtype: str
    tokens_per_block: int
    tokens_per_row: int
    entry_lanes: int
    attention_layers: tuple[int, ...]
    state_layers: tuple[int, ...]
    state_regions: tuple[StateRegion, ...]

    @property
    def pools(self) -> int:
        return len(self.attention_layers)

    @property
    def states_per_pool(self) -> int:
        """The most recurrent layers one pool holds (pool ``p`` holds state
        layers ``[p * states_per_pool, (p + 1) * states_per_pool)``)."""
        return -(-len(self.state_layers) // self.pools)

    def region(self, name: str) -> StateRegion:
        return next(r for r in self.state_regions if r.name == name)

    def check(self, tensors) -> None:
        """Raise unless ``tensors`` are this layout's pools as vLLM allocated them."""
        shapes = {tuple(t.shape[1:]) for t in tensors}
        dtypes = {str(t.dtype).removeprefix("torch.") for t in tensors}
        if (
            len(tensors) != self.pools
            or shapes != {self.block_shape}
            or dtypes != {self.dtype}
        ):
            raise NotImplementedError(
                f"the program expects {self.pools} unified pools "
                f"[blocks, {', '.join(map(str, self.block_shape))}] "
                f"{self.dtype}; vLLM allocated {len(tensors)} with block shapes "
                f"{sorted(shapes)} and {sorted(dtypes)}"
            )
        if any(int(t.shape[0]) == 0 for t in tensors):
            raise NotImplementedError("vLLM allocated empty KV cache pools")
