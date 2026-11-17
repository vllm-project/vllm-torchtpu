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
"""Two-level intra-group and inter-group XOR peer coordinate arithmetic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax

    RankCoord = int | jax.Array


def _is_host_int(x: object) -> bool:
    """Return True when `x` is a concrete host int (not bool or JAX tracer)."""
    t = type(x)
    if t is bool:
        return False
    if t is int:
        return True
    return t.__module__.startswith("numpy") and t.__name__ in (
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "intp",
        "uintp",
    )


def _reject_bool_coord(name: str, value: object) -> None:
    """Raise TypeError when a coordinate argument is a Python `bool`."""
    if isinstance(value, bool):
        msg = f"{name} cannot be bool, got {value!r}"
        raise TypeError(msg)


@dataclass(frozen=True, kw_only=True, slots=True)
class PeerTopology:
    """Coordinate helper for intra-group rings and inter-group XOR trees."""

    rank: RankCoord
    world_size: int
    group_size: int = 8

    def __post_init__(self) -> None:
        """Validate group divisibility, power-of-two groups, and host rank bounds."""
        _reject_bool_coord("rank", self.rank)
        if self.world_size < 1:
            msg = f"world_size must be >= 1, got {self.world_size}"
            raise ValueError(msg)
        if self.group_size < 1:
            msg = f"group_size must be >= 1, got {self.group_size}"
            raise ValueError(msg)
        if self.world_size % self.group_size != 0:
            msg = (
                f"world_size={self.world_size} must be divisible by "
                f"group_size={self.group_size}"
            )
            raise ValueError(msg)
        groups = self.world_size // self.group_size
        if groups & (groups - 1) != 0:
            msg = f"num_groups={groups} must be a power of two for XOR trees"
            raise ValueError(msg)
        if _is_host_int(self.rank):
            r = int(self.rank)
            if r < 0 or r >= self.world_size:
                msg = f"rank={self.rank} out of bounds for world_size={self.world_size}"
                raise ValueError(msg)

    @property
    def num_groups(self) -> int:
        """Number of equal-sized rank groups across `world_size`."""
        return self.world_size // self.group_size

    @property
    def intra_rank(self) -> RankCoord:
        """Rank offset `[0, group_size)` within the local group."""
        return self.rank % self.group_size

    @property
    def group_idx(self) -> RankCoord:
        """Group coordinate `[0, num_groups)` owning `rank`."""
        return self.rank // self.group_size

    def intra_peer(self, slot: RankCoord) -> RankCoord:
        """Global rank at intra-group offset `slot` inside the local group."""
        _reject_bool_coord("slot", slot)
        if _is_host_int(slot):
            s = int(slot)
            if s < 0 or s >= self.group_size:
                msg = f"slot={slot} out of bounds for group_size={self.group_size}"
                raise ValueError(msg)
        return self.group_idx * self.group_size + slot

    def ring_next_peer(self, step: int = 1) -> RankCoord:
        """Global rank `step` hops forward along the local intra-group ring."""
        _reject_bool_coord("step", step)
        return self.intra_peer((self.intra_rank + step) % self.group_size)

    def ring_prev_peer(self, step: int = 1) -> RankCoord:
        """Global rank `step` hops backward along the local intra-group ring."""
        _reject_bool_coord("step", step)
        return self.intra_peer(
            (self.intra_rank - step + self.group_size) % self.group_size
        )

    def inter_xor_peer(self, group_xor: RankCoord) -> RankCoord:
        """Global rank at the same `intra_rank` in group `group_idx ^ group_xor`."""
        _reject_bool_coord("group_xor", group_xor)
        if _is_host_int(group_xor):
            gx = int(group_xor)
            if gx < 0 or gx >= self.num_groups:
                msg = (
                    f"group_xor={group_xor} out of bounds for "
                    f"num_groups={self.num_groups}"
                )
                raise ValueError(msg)
        return ((self.group_idx ^ group_xor) * self.group_size) + self.intra_rank


__all__ = ["PeerTopology"]
