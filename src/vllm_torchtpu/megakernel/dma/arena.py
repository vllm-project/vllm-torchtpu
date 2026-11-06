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
"""Trace-time 4 KiB-aligned interval allocator over `dma.pool_alias`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..errors import MegakernelConfigError
from . import pool_alias

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jax.typing import DTypeLike

    ArenaRef = Any
    ScratchSpec = pl.MemoryRef | dict[str, pl.MemoryRef]

ALIGNMENT_BYTES: int = 4096
_MIN_SLICE_RANK: int = 2
_MAX_DTYPE_ITEMSIZE: int = 4
ArenaSignatureEntry = tuple[str, int, int, tuple[int, ...], str]
ArenaSignature = tuple[ArenaSignatureEntry, ...]


class _ArenaConfigError(MegakernelConfigError, ValueError):
    """Configuration or value error raised by `VmemArena`."""


def _resolve_byte_aligned_dtype(name: str, dtype: DTypeLike) -> jnp.dtype:
    """Resolve `dtype` to a byte-aligned `jnp.dtype` or raise `_ArenaConfigError`."""
    dt = jnp.dtype(dtype)
    try:
        bits = jnp.finfo(dt).bits
    except ValueError:
        try:
            bits = jnp.iinfo(dt).bits
        except ValueError as exc:
            msg = f"Unsupported dtype {dt} for slice {name!r}"
            raise _ArenaConfigError(msg) from exc
    if dt.itemsize * 8 != bits:
        msg = f"Sub-byte dtype {dt} ({bits}-bit) is not supported by VmemArena"
        raise _ArenaConfigError(msg)
    if dt.itemsize > _MAX_DTYPE_ITEMSIZE:
        msg = f"64-bit dtype {dt} is not supported by VmemArena for slice {name!r}"
        raise _ArenaConfigError(msg)
    return dt


def _tile_slice_shape(
    name: str,
    shape: Sequence[int],
    dt: jnp.dtype,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Validate `shape` and compute `(logical_shape, tiled_shape, size_bytes)`."""
    shp = tuple(int(d) for d in shape)
    if len(shp) < _MIN_SLICE_RANK or any(d <= 0 for d in shp):
        msg = f"slice {name!r} shape {shp} must be at least 2D with positive extents"
        raise _ArenaConfigError(msg)
    row_tile = 32 // dt.itemsize
    padded_r = ((shp[-2] + row_tile - 1) // row_tile) * row_tile
    padded_c = ((shp[-1] + 127) // 128) * 128
    tiled_shp = (*shp[:-2], padded_r, padded_c)
    size_bytes = pool_alias.storage_size(tiled_shp, dt)
    return shp, tiled_shp, size_bytes


def _first_fit_offset(
    live_slices: Sequence[ArenaSlice],
    size_bytes: int,
    capacity_bytes: int,
    name: str,
) -> int:
    """Return the lowest 4 KiB-aligned byte offset fitting `size_bytes`."""
    cand = 0
    for live in live_slices:
        if cand + size_bytes <= live.offset_bytes:
            break
        cand = max(cand, live.end_bytes)
    if cand + size_bytes > capacity_bytes:
        msg = (
            f"slice {name!r} ({size_bytes} B at offset {cand} B) "
            f"exceeds capacity_bytes={capacity_bytes} B"
        )
        raise _ArenaConfigError(msg)
    return cand


@dataclass(frozen=True, kw_only=True, slots=True)
class ArenaSlice:
    """Descriptor for a 4 KiB-aligned interval inside `VmemArena`."""

    name: str
    offset_bytes: int
    size_bytes: int
    shape: tuple[int, ...]
    tiled_shape: tuple[int, ...]
    dtype: jnp.dtype

    @property
    def end_bytes(self) -> int:
        """Exclusive upper byte offset of this slice."""
        return self.offset_bytes + self.size_bytes

    def interp_key(self) -> str:
        """Return the unique interpret-mode dictionary key for this slice."""
        return (
            f"{self.offset_bytes}:{self.size_bytes}:"
            f"{self.tiled_shape}:{self.dtype.name}"
        )


class VmemArena:
    """Trace-time interval allocator packing stage buffers into one backing pool."""

    def __init__(self, capacity_bytes: int) -> None:
        """Initialize an empty arena with 4 KiB-aligned `capacity_bytes`."""
        if isinstance(capacity_bytes, bool) or not isinstance(capacity_bytes, int):
            msg = f"capacity_bytes must be an int, got {capacity_bytes!r}"
            raise _ArenaConfigError(msg)
        if capacity_bytes <= 0 or capacity_bytes % ALIGNMENT_BYTES != 0:
            msg = (
                f"capacity_bytes={capacity_bytes} must be a positive multiple "
                f"of {ALIGNMENT_BYTES}"
            )
            raise _ArenaConfigError(msg)
        self.capacity_bytes: int = int(capacity_bytes)
        self.peak_bytes: int = 0
        self._live: dict[str, ArenaSlice] = {}
        self._all_slices: dict[str, ArenaSlice] = {}
        self._interp_specs: dict[str, pl.MemoryRef] = {}
        self._frozen: bool = False

    def live_slices(self) -> tuple[ArenaSlice, ...]:
        """Return currently live slices ordered by ascending `offset_bytes`."""
        return tuple(sorted(self._live.values(), key=lambda s: s.offset_bytes))

    def release(self, *names: str) -> None:
        """Mark `names` as dead so subsequent allocations can reuse their bytes."""
        if self._frozen:
            msg = "Cannot release from VmemArena after scratch_spec() has been called"
            raise _ArenaConfigError(msg)
        unique_names = tuple(dict.fromkeys(names))
        for name in unique_names:
            if name not in self._live:
                msg = f"slice {name!r} is not currently live in VmemArena"
                raise _ArenaConfigError(msg)
        for name in unique_names:
            del self._live[name]

    def allocate(
        self,
        name: str,
        shape: Sequence[int],
        dtype: DTypeLike,
        *,
        replaces: Sequence[str] = (),
    ) -> ArenaSlice:
        """Allocate a 4 KiB-aligned slice, optionally releasing `replaces` first."""
        if self._frozen:
            msg = "Cannot allocate from VmemArena after scratch_spec() has been called"
            raise _ArenaConfigError(msg)
        if name in self._all_slices:
            state = "already live" if name in self._live else "already allocated"
            msg = f"Duplicate slice name {name!r} ({state} in VmemArena)"
            raise _ArenaConfigError(msg)
        replaces_tuple = tuple(dict.fromkeys(replaces))
        for rep_name in replaces_tuple:
            if rep_name not in self._live:
                msg = f"slice {rep_name!r} is not currently live in VmemArena"
                raise _ArenaConfigError(msg)
        dt = _resolve_byte_aligned_dtype(name, dtype)
        shp, tiled_shp, size_bytes = _tile_slice_shape(name, shape, dt)

        candidate_live = sorted(
            (s for s in self._live.values() if s.name not in set(replaces_tuple)),
            key=lambda s: s.offset_bytes,
        )
        cand = _first_fit_offset(candidate_live, size_bytes, self.capacity_bytes, name)

        for rep_name in replaces_tuple:
            del self._live[rep_name]

        slc = ArenaSlice(
            name=name,
            offset_bytes=cand,
            size_bytes=size_bytes,
            shape=shp,
            tiled_shape=tiled_shp,
            dtype=dt,
        )
        self._live[name] = slc
        self._all_slices[name] = slc
        self._interp_specs[slc.interp_key()] = pltpu.VMEM(tiled_shp, dt)
        self.peak_bytes = max(self.peak_bytes, cand + size_bytes)
        return slc

    def scratch_spec(self, *, interpret: bool = False) -> ScratchSpec:
        """Return the Pallas scratch specification and freeze this arena.

        When ``interpret=True``, returns a per-slice dictionary of typed VMEM
        specs to bypass JAX CPU interpret ``state_discharge`` limitations on
        chained ``BitcastTransform`` and ``ReshapeTransform`` refs. When
        ``interpret=False``, aliases all slices into one backing
        ``(peak_bytes // 128, 128)`` ``uint8`` VMEM pool.
        """
        if self.peak_bytes == 0:
            msg = "Cannot build scratch_spec() for empty VmemArena (peak_bytes == 0)"
            raise _ArenaConfigError(msg)
        self._frozen = True
        if interpret:
            return dict(self._interp_specs)
        return pltpu.VMEM((self.peak_bytes // 128, 128), jnp.uint8)

    def view(self, arena_ref: ArenaRef, slc: ArenaSlice | str) -> ArenaRef:
        """Return a typed Pallas ref view over `arena_ref` for `slc`."""
        if isinstance(slc, str):
            if slc not in self._all_slices:
                msg = f"unknown arena slice {slc!r}"
                raise _ArenaConfigError(msg)
            target = self._all_slices[slc]
        else:
            target = slc

        if isinstance(arena_ref, dict):
            v = arena_ref[target.interp_key()]
        else:
            row_start = target.offset_bytes // 128
            num_rows = target.size_bytes // 128
            v = pool_alias.view(
                arena_ref.at[pl.ds(row_start, num_rows)],
                target.tiled_shape,
                target.dtype,
            )
        if target.shape != target.tiled_shape:
            return v.at[..., : target.shape[-2], : target.shape[-1]]
        return v

    def signature(self) -> ArenaSignature:
        """Return a deterministic snapshot of live slices for loop checks."""
        return tuple(
            (s.name, s.offset_bytes, s.size_bytes, s.shape, s.dtype.name)
            for s in self.live_slices()
        )

    def assert_matching_signature(self, expected: ArenaSignature) -> None:
        """Raise `MegakernelConfigError` if the live-slice signature drifted."""
        cur = self.signature()
        if cur != expected:
            msg = f"VmemArena loop signature drift: expected {expected}, got {cur}"
            raise _ArenaConfigError(msg)


__all__ = ["ALIGNMENT_BYTES", "ArenaSignature", "ArenaSlice", "VmemArena"]
