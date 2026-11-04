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
"""Split-phase remote DMA descriptor batching over caller-owned scratch refs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypedDict, Unpack

from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

if TYPE_CHECKING:
    from collections.abc import Sequence

    import jax


class RemoteCopyDescriptor(Protocol):
    """Protocol for Pallas TPU async remote DMA copy descriptors."""

    def start(self) -> None:
        """Start the asynchronous remote DMA transfer."""

    def wait_recv(self) -> None:
        """Wait for incoming remote DMA write completion."""

    def wait_send(self) -> None:
        """Wait for outgoing remote DMA read completion."""


class _RemoteCopyOptions(TypedDict, total=False):
    """Optional keyword arguments for `make_remote_copy_batch`."""

    axis_name: str | None


@dataclass(frozen=True, slots=True)
class RemoteCopyBatch:
    """Batch of remote DMA descriptors with split start/wait phases.

    On physical TPU hardware, callers must execute a cross-rank entry barrier
    (e.g., via ``pltpu.get_barrier_semaphore()`` or a mesh barrier semaphore)
    before issuing the first remote DMA write so peer scratchpad buffers are
    allocated and live across all participating devices.
    """

    copies: tuple[RemoteCopyDescriptor, ...]

    @property
    def ops(self) -> tuple[RemoteCopyDescriptor, ...]:
        """Alias for ``copies`` descriptor tuple."""
        return self.copies

    def start(self) -> None:
        """Issue all remote DMA transfers in this batch."""
        for cp in self.copies:
            cp.start()

    def wait_recv(self) -> None:
        """Wait for incoming remote DMA writes to complete on local buffers."""
        for cp in self.copies:
            cp.wait_recv()

    def wait_send(self) -> None:
        """Wait for outgoing remote DMA reads from local buffers to complete."""
        for cp in self.copies:
            cp.wait_send()

    def wait(self) -> None:
        """Wait for both receive and send completion across the batch."""
        self.wait_recv()
        self.wait_send()


def make_remote_copy_batch(
    src_refs: Sequence[object],
    dst_refs: Sequence[object],
    peer_ids: Sequence[int | jax.Array],
    send_sems: Sequence[object],
    recv_sems: Sequence[object],
    **options: Unpack[_RemoteCopyOptions],
) -> RemoteCopyBatch:
    """Build a `RemoteCopyBatch` borrowing caller-provided refs and semaphores."""
    axis_name = options.get("axis_name")
    n = len(src_refs)
    if (
        len(dst_refs) != n
        or len(peer_ids) != n
        or len(send_sems) != n
        or len(recv_sems) != n
    ):
        msg = (
            "make_remote_copy_batch requires sequences of equal length: "
            f"got {(n, len(dst_refs), len(peer_ids), len(send_sems), len(recv_sems))}"
        )
        raise ValueError(msg)
    if n == 0:
        return RemoteCopyBatch(copies=())
    descs: list[RemoteCopyDescriptor] = []
    for src, dst, peer, ssem, rsem in zip(
        src_refs, dst_refs, peer_ids, send_sems, recv_sems, strict=True
    ):
        dev_id: (
            dict[str | tuple[str, ...], int | jax.Array] | tuple[int | jax.Array, ...]
        ) = {axis_name: peer} if axis_name is not None else (peer,)
        descs.append(
            pltpu.make_async_remote_copy(
                src_ref=src,
                dst_ref=dst,
                send_sem=ssem,
                recv_sem=rsem,
                device_id=dev_id,
                device_id_type=pl.DeviceIdType.MESH,
            )
        )
    return RemoteCopyBatch(copies=tuple(descs))


__all__ = ["RemoteCopyBatch", "RemoteCopyDescriptor", "make_remote_copy_batch"]
