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
"""Disjoint token ownership within groups of replicated MoE inputs."""

from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
from jax import lax


def replica_wire_rows(input_rows, width, row_alignment):
    """An upper bound on `TokenReplicaLayout.local_rows`, in arithmetic alone.

    The share padded to a whole block even below one, where the layout pads
    to a power of two instead; equal to it from one block up. Sizing an op's
    output with this keeps a symbolic `input_rows` free of guards when Dynamo
    traces the op's fake -- which matters because vLLM traces the MTP drafter
    once for every token bucket and refuses a graph that assumes any bound on
    the token count.
    """
    if width == 1:
        return input_rows
    share = (input_rows + width - 1) // width
    return (share + row_alignment - 1) // row_alignment * row_alignment


@dataclass(frozen=True)
class TokenReplicaLayout:
    """Groups are ordered mesh coordinates, independent of expert ownership.

    Every member supplies identical input rows. Each computes one contiguous
    slice, then all members recover the original row order. Singleton groups
    leave the input and output unchanged, including unaligned row counts.
    """

    groups: tuple[tuple[int, ...], ...]
    slots: tuple[int, ...]
    input_rows: int
    local_rows: int

    @classmethod
    def create(
        cls,
        *,
        ep: int,
        input_rows: int,
        groups: tuple[tuple[int, ...], ...] | None = None,
        row_alignment: int = 1,
    ) -> Self:
        if ep < 1 or input_rows < 1:
            raise ValueError("EP width and input row count must be positive")
        if row_alignment < 1 or row_alignment & (row_alignment - 1):
            raise ValueError("row alignment must be a positive power of two")
        groups = (
            tuple((i,) for i in range(ep))
            if groups is None
            else tuple(tuple(int(i) for i in group) for group in groups)
        )
        if (
            not groups
            or not groups[0]
            or any(len(g) != len(groups[0]) for g in groups)
            or sorted(i for g in groups for i in g) != list(range(ep))
        ):
            raise ValueError(
                "token replica groups must partition the mesh "
                "into equally sized, nonempty ordered groups"
            )
        slots = [0] * ep
        for group in groups:
            for slot, rank in enumerate(group):
                slots[rank] = slot
        width = len(groups[0])
        local_rows = (input_rows + width - 1) // width
        # Pallas routing tiles need aligned row blocks. Below one block,
        # keep a power-of-two whole-array tile rather than widening to eight.
        alignment = (
            min(row_alignment, 1 << (local_rows - 1).bit_length()) if width > 1 else 1
        )
        local_rows = (local_rows + alignment - 1) // alignment * alignment
        return cls(groups, tuple(slots), input_rows, local_rows)

    @property
    def width(self) -> int:
        return len(self.groups[0])

    @property
    def group_of(self) -> tuple[int, ...]:
        """The index of the replica group each mesh rank belongs to."""
        owner = [0] * sum(len(g) for g in self.groups)
        for g, group in enumerate(self.groups):
            for rank in group:
                owner[rank] = g
        return tuple(owner)

    @property
    def slot_groups(self) -> tuple[tuple[int, ...], ...]:
        """For each slot, the ranks holding that slot, in group order."""
        return tuple(
            tuple(group[s] for group in self.groups) for s in range(self.width)
        )

    def split_slot(
        self, rows: jax.Array, slot: jax.Array, *, fill_value: float = 0
    ) -> jax.Array:
        """Slot `slot`'s contiguous share of the replicated rows."""
        padding = self.local_rows * self.width - self.input_rows
        rows = jnp.pad(rows, ((0, padding), (0, 0)), constant_values=fill_value)
        return lax.dynamic_slice_in_dim(
            rows, slot * self.local_rows, self.local_rows, axis=0
        )

    def split(
        self, rows: jax.Array, rank: jax.Array, *, fill_value: float = 0
    ) -> jax.Array:
        if len(self.groups[0]) == 1:
            return rows
        slot = jnp.asarray(self.slots, dtype=jnp.int32)[rank]
        return self.split_slot(rows, slot, fill_value=fill_value)

    def restore(self, rows: jax.Array, axis: str) -> jax.Array:
        if len(self.groups[0]) == 1:
            return rows
        gathered = lax.all_gather(
            rows, axis, axis=0, tiled=True, axis_index_groups=self.groups
        )
        return gathered[: self.input_rows]

    def restore_with_partial(
        self, rows: jax.Array, partial: jax.Array, axis: str
    ) -> jax.Array:
        """`restore(rows)` plus the group-wide sum of `partial`, in one gather.

        `partial` is [input_rows, ...] and holds one replica's partial sum of
        a term every replica owes the output -- a tensor-parallel shared
        expert's down projection, say. Summing it with an all-reduce and then
        gathering `rows` pays the gather twice: an all-reduce is a
        reduce-scatter plus an all-gather. Here the reduce-scatter lands each
        slot's share of the sum beside the rows that slot owns, and the one
        gather `restore` already pays carries both.
        """
        if self.width == 1:
            return rows + partial.astype(rows.dtype)
        padding = self.local_rows * self.width - self.input_rows
        partial = jnp.pad(partial, ((0, padding),) + ((0, 0),) * (partial.ndim - 1))
        share = lax.psum_scatter(
            partial,
            axis,
            scatter_dimension=0,
            axis_index_groups=self.groups,
            tiled=True,
        )
        return self.restore(rows + share.astype(rows.dtype), axis)

    def local_dispatch_rows(
        self, own_rows: jax.Array, mine: jax.Array, rank: jax.Array, axis: str
    ) -> tuple[jax.Array, jax.Array]:
        """Every shard's wire rows without re-gathering the ones held already.

        Every member of a replica group holds the group's whole input, so it
        can produce every slot's wire rows itself (`own_rows`, [width *
        local_rows, ...] in slot order); a flat all-gather of `mine` (this
        shard's [local_rows, ...]) moves width - 1 of those shares again.
        Only the other group's rows have to travel: one share crosses from
        the same-slot shard of that group, then the group gathers the
        crossed shares among itself.

        Returns (rows, positions): rows [2 * width * local_rows, ...] with
        this group's shares first, and positions [ep] i32, the block of rows
        holding mesh rank m's share. Two groups only.
        """
        if len(self.groups) != 2:
            raise ValueError("local dispatch needs exactly two replica groups")
        group = jnp.asarray(self.group_of, dtype=jnp.int32)[rank]
        crossed = lax.all_gather(
            mine, axis, axis=0, tiled=False, axis_index_groups=self.slot_groups
        )
        foreign = lax.dynamic_index_in_dim(crossed, 1 - group, 0, keepdims=False)
        foreign = lax.all_gather(
            foreign, axis, axis=0, tiled=True, axis_index_groups=self.groups
        )
        width = self.width
        table = [
            [
                self.slots[m] if self.group_of[m] == g else width + self.slots[m]
                for m in range(len(self.slots))
            ]
            for g in range(2)
        ]
        positions = jnp.asarray(table, dtype=jnp.int32)[group]
        return jnp.concatenate([own_rows, foreign], axis=0), positions
