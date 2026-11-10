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
# src/vllm_torchtpu/megakernel/common/vmem_pool.py
"""Trace-time manager for the kernel's shared VMEM weight arena.

Kernel code asks for weights by name and never places them itself:

    pool.request("router", nbytes, start, index=i)  # queue a load (FIFO)
    offset = pool.acquire("router", wait)   # wait for it; its arena offset
    pool.release("router")                  # its bytes are free again

A request is issued (its DMAs started) as soon as the arena has a free range
for it and every earlier request has been issued; otherwise it waits in the
queue and is issued by the release that makes room. Where and when each
weight loads therefore follows from the arena size and the order of
requests and releases, not from hand-placed offsets.

``allocate`` claims a range for a region its owner fills itself (the routed
expert slots); ``replaces`` lets it overlap loads the owner releases before
its first write (the router, consumed before the first expert copy).

Everything here runs while the kernel is traced: offsets, issue points and
semaphores are Python constants, and the emitted code is the same DMA starts
and waits as hand-placed calls. The layer loop goes through ``loop``, which
checks that its body ends in the arena state it started in, so a static
offset is valid on every iteration.

``start(offset, index)`` gets the loop index the load was requested for, which a
closure would misread for a load issued after the loop back-edge.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Protocol

import jax.numpy as jnp
from jax.experimental import pallas as pl

from vllm_torchtpu.megakernel.core import pool_alias

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import jax

ALIGN = 4096
_MIB = float(2**20)


class _ArenaRef(Protocol):
    """Protocol for a 2D VMEM uint8 reference supporting ``.at[...]`` slicing."""

    @property
    def at(self) -> _ArenaRef:
        """Return indexable view over the VMEM reference."""
        ...

    def __getitem__(self, indices: tuple[object, object]) -> object:
        """Return a sliced reference view."""
        ...


@dataclasses.dataclass
class _Entry:
    """State of one live or queued VMEM pool allocation."""

    name: str
    nbytes: int
    start: Callable[[int, object], None] | None = None
    index: object = None
    predicate: jax.Array | bool | None = None
    offset: int | None = None
    state: str = "queued"
    replaces: tuple[str, ...] = ()

    def signature(self) -> tuple[str, int, int | None, str, tuple[str, ...]]:
        """Return the structural state tuple checked across loop iterations.

        Returns:
            Tuple of name, byte size, byte offset, lifecycle state, and replaced keys.
        """
        return (self.name, self.nbytes, self.offset, self.state, self.replaces)


class PoolError(ValueError):
    """Error raised when a VMEM pool operation violates arena invariants."""


class VmemPool:
    """Trace-time FIFO allocator and DMA scheduler for a shared VMEM weight arena."""

    def __init__(
        self,
        arena: _ArenaRef | None,
        capacity: int,
        *,
        report: object = None,
    ) -> None:
        """Initialize a VMEM pool over ``arena`` with byte ``capacity``.

        Args:
            arena: ``[capacity / 128, 128]`` uint8 VMEM ref, or ``None`` for a dry run.
            capacity: Total arena capacity in bytes (must be a multiple of 4 KiB).
            report: Whether to emit timeline reports after tracing.

        Raises:
            PoolError: If ``capacity`` is not a multiple of 4 KiB.
        """
        if capacity % ALIGN:
            msg = "The arena capacity must be a multiple of 4 KiB"
            raise PoolError(msg)
        self.arena = arena
        self.capacity = capacity
        self.entries: dict[str, _Entry] = {}
        self.queue: list[str] = []
        self.events: list[tuple[str, str, str, int | None, int, int]] = []
        self.report = bool(report)
        self.phase = ""
        self.finished = False

    # -- placement -----------------------------------------------------------

    def _occupied(self, ignore: Sequence[str] = ()) -> list[tuple[int, int]]:
        """Return sorted ``(start, end)`` byte intervals of live placed entries.

        Args:
            ignore: Entry names to exclude when computing occupied ranges.

        Returns:
            Sorted list of ``(start_byte, end_byte)`` intervals.
        """
        return sorted(
            (entry.offset, entry.offset + _aligned(entry.nbytes))
            for entry in self.entries.values()
            if entry.offset is not None and entry.name not in ignore
        )

    def _fit(self, nbytes: int, ignore: Sequence[str] = ()) -> int | None:
        """Find the lowest free 4 KiB-aligned byte offset that fits ``nbytes``.

        Args:
            nbytes: Requested allocation size in bytes.
            ignore: Entry names whose ranges may be overlapped.

        Returns:
            Lowest free byte offset, or ``None`` when no contiguous gap fits.
        """
        size = _aligned(nbytes)
        cursor = 0
        for start, end in self._occupied(ignore):
            if start - cursor >= size:
                return cursor
            cursor = max(cursor, end)
        return cursor if self.capacity - cursor >= size else None

    def _issue_queued(self) -> None:
        """Issue queued requests in FIFO order while the head fits."""
        while self.queue:
            entry = self.entries[self.queue[0]]
            offset = self._fit(entry.nbytes)
            if offset is None or entry.start is None:
                return
            self.queue.pop(0)
            entry.offset, entry.state = offset, "in_flight"
            self._log("issue", entry)
            start_fn = entry.start
            index = entry.index
            if entry.predicate is None:
                start_fn(offset, index)
            else:

                def _issue_guarded(
                    *,
                    fn: Callable[[int, object], None] = start_fn,
                    target_offset: int = offset,
                    target_index: object = index,
                ) -> None:
                    """Invoke queued DMA start callback under entry predicate."""
                    fn(target_offset, target_index)

                pl.when(entry.predicate)(_issue_guarded)

    # -- API -----------------------------------------------------------------

    def request(
        self,
        name: str,
        nbytes: int,
        start: Callable[[int, object], None],
        *,
        index: object = None,
        predicate: jax.Array | bool | None = None,
    ) -> None:
        """Queue the load ``start(offset, index)`` of ``nbytes`` and issue if it fits.

        Args:
            name: Unique identifier for the requested weight buffer.
            nbytes: Unpadded byte size of the requested weight buffer.
            start: Callback ``start(offset, index)`` that starts the DMA copies.
            index: Loop iteration index for which the load was requested.
            predicate: Optional traced boolean guarding the DMA start.

        Raises:
            PoolError: If the pool is closed, ``name`` is live, or ``nbytes`` exceeds
                the total arena capacity.
        """
        self._check_open(name)
        if _aligned(nbytes) > self.capacity:
            msg = f"{name}: {nbytes} bytes exceed the {self.capacity}-byte arena"
            raise PoolError(msg)
        entry = _Entry(name, nbytes, start, index, predicate)
        self.entries[name] = entry
        self.queue.append(name)
        self._log("request", entry)
        self._issue_queued()

    def allocate(
        self,
        name: str,
        nbytes: int,
        *,
        replaces: Sequence[str] = (),
    ) -> int:
        """Claim an arena range immediately for a caller-filled scratch region.

        The region may overlap live entries listed in ``replaces``, which must be
        released before ``acquire(name)`` is called.

        Args:
            name: Unique identifier for the allocated scratch region.
            nbytes: Byte size to claim in the arena.
            replaces: Names of live entries that will be released before first write.

        Returns:
            Assigned byte offset in the VMEM arena.

        Raises:
            PoolError: If any entry in ``replaces`` is not live or no gap fits.
        """
        self._check_open(name)
        for other in replaces:
            if other not in self.entries:
                msg = f"{name}: replaces {other}, which is not live"
                raise PoolError(msg)
        offset = self._fit(nbytes, ignore=replaces)
        if offset is None:
            msg = f"allocate({name}): {self._describe(nbytes)}"
            raise PoolError(msg)
        entry = _Entry(
            name,
            nbytes,
            offset=offset,
            state="allocated",
            replaces=tuple(replaces),
        )
        self.entries[name] = entry
        self._log("allocate", entry)
        return offset

    def acquire(
        self,
        name: str,
        wait: Callable[[int], None] | None = None,
    ) -> int:
        """Wait for a requested load and return its arena byte offset.

        ``wait(offset)`` emits the DMA waits; pass ``None`` when the caller waits
        directly.

        Args:
            name: Name of the requested or allocated entry to acquire.
            wait: Optional callback ``wait(offset)`` that waits on the DMA semaphore.

        Returns:
            Assigned byte offset in the VMEM arena.

        Raises:
            PoolError: If replaced entries are still live or the load is still queued.
        """
        entry = self._live(name)
        if entry.state == "allocated":
            pending = [other for other in entry.replaces if other in self.entries]
            if pending:
                msg = f"acquire({name}): {pending} not released yet"
                raise PoolError(msg)
            if entry.offset is None:
                msg = f"acquire({name}): allocated entry has no offset"
                raise PoolError(msg)
            return entry.offset
        if entry.state == "queued" or entry.offset is None:
            msg = (
                f"acquire({name}) before its load could be issued: "
                f"{self._describe(entry.nbytes)}"
            )
            raise PoolError(msg)
        if entry.state == "in_flight":
            entry.state = "acquired"
            self._log("acquire", entry)
            if wait is not None:
                wait(entry.offset)
        return entry.offset

    def offset(self, name: str) -> int:
        """Return the arena byte offset of a live, issued entry.

        Args:
            name: Name of the live entry.

        Returns:
            Assigned byte offset in the VMEM arena.

        Raises:
            PoolError: If ``name`` is not live or has not been assigned an offset yet.
        """
        entry = self._live(name)
        if entry.offset is None:
            msg = f"{name} has no offset yet: {self._describe(entry.nbytes)}"
            raise PoolError(msg)
        return entry.offset

    def release(self, name: str) -> None:
        """Release a live entry and issue any queued loads that now fit.

        Args:
            name: Name of the acquired or allocated entry to release.

        Raises:
            PoolError: If the entry was never issued or never waited on.
        """
        entry = self._live(name)
        if entry.state == "queued":
            msg = f"release({name}) of a load that was never issued"
            raise PoolError(msg)
        if entry.state == "in_flight":
            msg = f"release({name}) of a load that was never waited on"
            raise PoolError(msg)
        del self.entries[name]
        self._log("release", entry)
        self._issue_queued()

    def view(
        self,
        offset: int,
        shape: tuple[int, ...],
        dtype: jnp.dtype | type | str,
    ) -> object:
        """Build a typed ``[shape]`` view of the VMEM arena starting at byte ``offset``.

        Args:
            offset: Starting byte offset in the arena (multiple of 128 bytes).
            shape: Target logical tensor shape.
            dtype: Target element dtype.

        Returns:
            Aliased Pallas reference view over the arena slice.

        Raises:
            PoolError: If called on a dry-run pool with ``arena=None``.
        """
        if self.arena is None:
            msg = "cannot build a view on a dry-run VmemPool without an arena"
            raise PoolError(msg)
        resolved_dtype = jnp.dtype(dtype)
        nbytes = pool_alias.storage_size(shape, resolved_dtype)
        arena_slice = self.arena.at[pl.ds(offset // 128, nbytes // 128), :]
        return pool_alias.view(arena_slice, shape, resolved_dtype)

    # -- control flow --------------------------------------------------------

    def signature(
        self,
    ) -> tuple[
        tuple[tuple[str, int, int | None, str, tuple[str, ...]], ...],
        tuple[str, ...],
    ]:
        """Return a snapshot of all live entries and the FIFO request queue.

        Returns:
            Pair of ``(entries_signature, queue_names)``.
        """
        return (
            tuple(entry.signature() for entry in self.entries.values()),
            tuple(self.queue),
        )

    def loop(
        self,
        lower: int | jax.Array,
        upper: int | jax.Array,
    ) -> Callable[[Callable[[int | jax.Array], None]], None]:
        """Wrap ``pl.loop`` and verify the body preserves the arena state invariant.

        Loads carried in from the previous iteration run unconditionally because the
        iteration exists, so their predicates are cleared before tracing ``body``.

        Args:
            lower: Loop lower bound passed to ``pl.loop``.
            upper: Loop upper bound passed to ``pl.loop``.

        Returns:
            Decorator that traces the loop body and marks the pool finished on exit.
        """

        def decorator(body: Callable[[int | jax.Array], None]) -> None:
            """Trace ``body`` across ``pl.loop(lower, upper)`` and check state parity.

            Args:
                body: Single-argument loop body taking the loop iteration index.
            """
            entry_signature = self.signature()

            def traced(index: int | jax.Array) -> None:
                """Execute one traced loop iteration and verify exit signature.

                Args:
                    index: Current loop iteration index from ``pl.loop``.

                Raises:
                    PoolError: If the arena state at loop exit differs from loop entry.
                """
                for entry in self.entries.values():
                    entry.predicate = None
                    if entry.state == "queued":
                        entry.index = index
                self.phase_push("loop")
                body(index)
                self.phase_pop()
                if self.signature() != entry_signature:
                    msg = (
                        "The loop body does not end in the arena state it began in:\n"
                        f"  entry: {entry_signature}\n  exit:  {self.signature()}"
                    )
                    raise PoolError(msg)
                for entry in self.entries.values():
                    entry.predicate = None

            pl.loop(lower, upper)(traced)
            self.finished = True

        return decorator

    # -- reporting -----------------------------------------------------------

    def phase_push(self, label: str) -> None:
        """Push a nested phase label onto the timeline scope stack.

        Args:
            label: Phase name to append to the active scope path.
        """
        self.phase = f"{self.phase}/{label}" if self.phase else label

    def phase_pop(self) -> None:
        """Pop the innermost phase label from the timeline scope stack."""
        self.phase = self.phase.rpartition("/")[0]

    def _log(self, op: str, entry: _Entry) -> None:
        """Record an allocation or DMA lifecycle event for timeline reporting.

        Args:
            op: Lifecycle operation name (``request``, ``issue``, ``acquire``, etc.).
            entry: Pool entry affected by the operation.
        """
        used = sum(
            _aligned(other.nbytes)
            for other in self.entries.values()
            if other.offset is not None
        )
        event = (self.phase, op, entry.name, entry.offset, entry.nbytes, used)
        self.events.append(event)

    def timeline(self) -> str:
        """Format recorded pool allocation and DMA events as a human-readable table.

        Returns:
            Multi-line string summarizing arena capacity and chronological events.
        """
        lines = [f"VMEM pool: {self.capacity / _MIB:.2f} MiB arena"]
        for phase, op, name, offset, nbytes, used in self.events:
            place = "" if offset is None else f"@{offset / _MIB:6.2f}"
            lines.append(
                f"  {phase:28s} {op:9s} {name:12s} {nbytes / _MIB:6.2f} MiB "
                f"{place:8s} in use {used / _MIB:6.2f}"
            )
        return "\n".join(lines)

    # -- helpers -------------------------------------------------------------

    def _check_open(self, name: str) -> None:
        """Verify the pool is open and ``name`` is not already live.

        Args:
            name: Candidate entry name to validate.

        Raises:
            PoolError: If the pool is closed or ``name`` is already in ``self.entries``.
        """
        if self.finished:
            msg = f"{name}: the pool is closed"
            raise PoolError(msg)
        if name in self.entries:
            msg = f"{name} is already live"
            raise PoolError(msg)

    def _live(self, name: str) -> _Entry:
        """Look up a live entry by ``name``.

        Args:
            name: Entry name to look up.

        Returns:
            Live ``_Entry`` record.

        Raises:
            PoolError: If the pool is closed or ``name`` is not currently live.
        """
        if self.finished and name in self.entries:
            msg = f"{name}: requested in the last loop iteration, never issued"
            raise PoolError(msg)
        if name not in self.entries:
            msg = f"{name} is not live"
            raise PoolError(msg)
        return self.entries[name]

    def _describe(self, nbytes: int) -> str:
        """Format current arena occupancy and queue state for error messages.

        Args:
            nbytes: Requested byte count that failed to fit or issue.

        Returns:
            Diagnostic summary of requested bytes, held ranges, and queued names.
        """
        held = ", ".join(
            f"{entry.name} [{entry.offset / _MIB:.2f}, "
            f"+{entry.nbytes / _MIB:.2f}) {entry.state}"
            for entry in self.entries.values()
            if entry.offset is not None
        )
        queued = ", ".join(self.queue)
        return (
            f"needs {nbytes / _MIB:.2f} MiB of a {self.capacity / _MIB:.2f} MiB "
            f"arena; held: {held or 'nothing'}; queued: {queued or 'nothing'}"
        )


def _aligned(nbytes: int) -> int:
    """Round ``nbytes`` up to the next 4 KiB arena alignment boundary.

    Args:
        nbytes: Unaligned size in bytes.

    Returns:
        Size rounded up to a multiple of ``ALIGN`` (4096 bytes).
    """
    return -(-nbytes // ALIGN) * ALIGN
