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
"""Shared structural typing protocols for Kimi-K3 megakernel modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    import jax
    import jax.numpy as jnp
    import torch

    from .pool_alias import BitcastableRef


class InPlaceSetTensor(Protocol):
    """Protocol for tensors supporting in-place storage binding via ``set_``."""

    def set_(self, source: torch.Tensor) -> torch.Tensor:
        """Rebind storage in place to ``source``."""
        ...


def set_tensor_in_place(target: torch.Tensor, source: torch.Tensor) -> None:
    """Replace ``target`` storage with ``source`` in place."""
    cast("InPlaceSetTensor", target).set_(source)


class DsparkKernelProtocol(Protocol):
    """Structural protocol for layout constants from K3 ``kernels.dspark``."""

    ROWS: int
    OUT_COUNT: int
    OUT_BONUS: int
    OUT_DRAFTS: int


class ControlLike(Protocol):
    """Structural protocol for decoder control layouts used by draft ops."""

    SLOT: int
    RESET: int
    FIELDS: int
    slots: int
    pages: int
    plan_start: int
    size: int

    def key(self) -> tuple[int, int, int]:
        """Return the ``(slots, rows, pages)`` layout key."""
        ...

    def fields(
        self,
        control: jax.Array,
        batch: int,
    ) -> tuple[jax.Array, jax.Array]:
        """Return ``(plan, tables)`` slices for ``batch`` rows."""
        ...


class PallasIndexerProtocol(Protocol):
    """Protocol for ``ref.at[...]`` returning a sliced :class:`PallasRefProtocol`."""

    def __getitem__(self, idx: object) -> PallasRefProtocol:
        """Return sliced Pallas reference view for ``idx``."""
        ...


class PallasRefProtocol(Protocol):
    """Structural protocol for Pallas memory references and array views."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return tensor shape tuple."""
        ...

    @property
    def ndim(self) -> int:
        """Return number of tensor dimensions."""
        ...

    @property
    def dtype(self) -> jnp.dtype:
        """Return element dtype."""
        ...

    @property
    def at(self) -> PallasIndexerProtocol:
        """Return indexable slice helper for the reference."""
        ...

    def __getitem__(self, idx: object) -> jax.Array:
        """Read slice from the reference."""
        ...

    def __setitem__(self, idx: object, val: object) -> None:
        """Write slice into the reference."""
        ...

    def bitcast(self, dtype: jax.typing.DTypeLike) -> PallasRefProtocol:
        """Reinterpret reference elements as ``dtype``."""
        ...

    def reshape(self, *shape: int) -> PallasRefProtocol:
        """Reshape reference view."""
        ...

    def astype(self, dtype: jax.typing.DTypeLike) -> jax.Array:
        """Cast array view to ``dtype``."""
        ...


class AsyncCopyProtocol(Protocol):
    """Protocol for Pallas async DMA and remote copy descriptors."""

    def start(self) -> None:
        """Start the async transfer."""
        ...

    def wait(self) -> None:
        """Wait for transfer completion."""
        ...

    def wait_recv(self) -> None:
        """Wait for remote receive completion."""
        ...

    def wait_send(self) -> None:
        """Wait for remote send completion."""
        ...


class ArenaRefProtocol(Protocol):
    """Protocol for a 2D arena uint8 reference supporting ``.at[...]`` slicing."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the ``(rows, row_bytes)`` arena shape."""
        ...

    @property
    def at(self) -> ArenaRefProtocol:
        """Return indexable view over the arena reference."""
        ...

    def __getitem__(self, indices: tuple[object, object]) -> BitcastableRef:
        """Return a sliced reference view."""
        ...


class IndexableRowsProtocol(Protocol):
    """Protocol for 2D row-sliceable views."""

    def __getitem__(self, rows: slice, /) -> jax.Array:
        """Return sliced rows as an array."""
        ...


class ReshapableProtocol(Protocol):
    """Protocol for array or packed-row views supporting 2D reshape."""

    def reshape(self, rows: int, width: int, /) -> IndexableRowsProtocol | jax.Array:
        """Reshape into ``(rows, width)`` array or row-sliceable view."""
        ...


__all__ = [
    "ArenaRefProtocol",
    "AsyncCopyProtocol",
    "ControlLike",
    "DsparkKernelProtocol",
    "InPlaceSetTensor",
    "IndexableRowsProtocol",
    "PallasIndexerProtocol",
    "PallasRefProtocol",
    "ReshapableProtocol",
    "set_tensor_in_place",
]
