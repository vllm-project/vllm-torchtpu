# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
and waits as hand-placed calls. Traced control flow goes through ``cond``,
``when`` and ``loop``, which check that every path leaves the arena in the
same state (and a loop body in the state it started in), so a static offset
is valid on every path.

``start(offset, index)`` gets the loop index the load was requested for, which a
closure would misread for a load issued after the loop back-edge.
"""

import dataclasses

import jax
from jax.experimental import pallas as pl

from vllm_torchtpu.megakernel.common import pool_alias

ALIGN = 4096


@dataclasses.dataclass
class _Entry:
    name: str
    nbytes: int
    start: object = None  # start(offset, index) emits the DMA starts; None for allocate()
    index: object = None  # the loop index the load was requested for
    predicate: object = None  # traced bool guarding the starts, or None
    offset: int | None = None
    state: str = "queued"  # queued | in_flight | acquired | allocated
    replaces: tuple = ()

    def signature(self):
        return (self.name, self.nbytes, self.offset, self.state, self.replaces)


class PoolError(ValueError):
    pass


class VmemPool:
    def __init__(self, arena, capacity, *, report=None):
        """``arena``: [capacity / 128, 128] uint8 VMEM ref (None: dry run)."""
        if capacity % ALIGN:
            raise PoolError("The arena capacity must be a multiple of 4 KiB")
        self.arena, self.capacity = arena, capacity
        self.entries = {}  # live, in request order
        self.queue = []  # names of queued requests, FIFO
        self.events = []
        self.report = bool(report)
        self.phase = ""
        self.finished = False

    # -- placement -----------------------------------------------------------

    def _occupied(self, ignore=()):
        return sorted(
            (entry.offset, entry.offset + _aligned(entry.nbytes))
            for entry in self.entries.values()
            if entry.offset is not None and entry.name not in ignore
        )

    def _fit(self, nbytes, ignore=()):
        """Lowest free offset for ``nbytes`` (first fit), or None."""
        size = _aligned(nbytes)
        cursor = 0
        for start, end in self._occupied(ignore):
            if start - cursor >= size:
                return cursor
            cursor = max(cursor, end)
        return cursor if self.capacity - cursor >= size else None

    def _issue_queued(self):
        """Issue queued requests in FIFO order while the head fits."""
        while self.queue:
            entry = self.entries[self.queue[0]]
            offset = self._fit(entry.nbytes)
            if offset is None:
                return
            self.queue.pop(0)
            entry.offset, entry.state = offset, "in_flight"
            self._log("issue", entry)
            index = entry.index
            if entry.predicate is None:
                entry.start(offset, index)
            else:
                pl.when(entry.predicate)(lambda: entry.start(offset, index))

    # -- API -----------------------------------------------------------------

    def request(self, name, nbytes, start, *, index=None, predicate=None):
        """Queue the load ``start(offset, index)`` of ``nbytes``; issue it if it fits."""
        self._check_open(name)
        if _aligned(nbytes) > self.capacity:
            raise PoolError(f"{name}: {nbytes} bytes exceed the {self.capacity}-byte arena")
        entry = _Entry(name, nbytes, start, index, predicate)
        self.entries[name] = entry
        self.queue.append(name)
        self._log("request", entry)
        self._issue_queued()

    def allocate(self, name, nbytes, *, replaces=()):
        """Claim a range now for a region its owner fills; it may overlap the
        live entries in ``replaces``, which must be released before use."""
        self._check_open(name)
        for other in replaces:
            if other not in self.entries:
                raise PoolError(f"{name}: replaces {other}, which is not live")
        offset = self._fit(nbytes, ignore=replaces)
        if offset is None:
            raise PoolError(f"allocate({name}): {self._describe(nbytes)}")
        entry = _Entry(name, nbytes, offset=offset, state="allocated", replaces=tuple(replaces))
        self.entries[name] = entry
        self._log("allocate", entry)
        return offset

    def acquire(self, name, wait=None):
        """Wait for a requested load (``wait(offset)`` emits the waits; None
        when the consumer waits itself). Returns its offset."""
        entry = self._live(name)
        if entry.state == "allocated":
            pending = [other for other in entry.replaces if other in self.entries]
            if pending:
                raise PoolError(f"acquire({name}): {pending} not released yet")
            return entry.offset
        if entry.state == "queued":
            raise PoolError(
                f"acquire({name}) before its load could be issued: "
                f"{self._describe(entry.nbytes)}"
            )
        if entry.state == "in_flight":
            entry.state = "acquired"
            self._log("acquire", entry)
            if wait is not None:
                wait(entry.offset)
        return entry.offset

    def offset(self, name):
        """The arena offset of a live, issued entry (to build views)."""
        entry = self._live(name)
        if entry.offset is None:
            raise PoolError(f"{name} has no offset yet: {self._describe(entry.nbytes)}")
        return entry.offset

    def release(self, name):
        entry = self._live(name)
        if entry.state == "queued":
            raise PoolError(f"release({name}) of a load that was never issued")
        if entry.state == "in_flight":
            raise PoolError(f"release({name}) of a load that was never waited on")
        del self.entries[name]
        self._log("release", entry)
        self._issue_queued()

    def release_prefix(self, name, nbytes):
        """Release an acquired load's consumed prefix; retain its live suffix.

        ``nbytes`` must be a positive multiple of the arena alignment and
        smaller than the remaining load. The caller must finish every read
        and DMA touching the released bytes before this call. Existing views
        into the retained suffix remain valid; ``offset(name)`` now points
        to the suffix. Queued loads can immediately reuse the freed prefix.
        Use ``release`` once the suffix has also been consumed.
        """
        entry = self._live(name)
        if entry.state != "acquired":
            raise PoolError(f"release_prefix({name}) requires an acquired load")
        if nbytes <= 0 or nbytes % ALIGN or nbytes >= entry.nbytes:
            raise PoolError(
                f"release_prefix({name}) needs an aligned, nonempty proper prefix"
            )
        entry.offset += nbytes
        entry.nbytes -= nbytes
        self._log("release_prefix", entry)
        self._issue_queued()

    def view(self, offset, shape, dtype):
        """A [shape] dtype view of the arena from byte ``offset``."""
        nbytes = pool_alias.storage_size(shape, dtype)
        return pool_alias.view(
            self.arena.at[pl.ds(offset // 128, nbytes // 128), :], shape, dtype
        )

    # -- control flow --------------------------------------------------------

    def signature(self):
        return (
            tuple(entry.signature() for entry in self.entries.values()),
            tuple(self.queue),
        )

    def _snapshot(self):
        return (
            {name: dataclasses.replace(entry) for name, entry in self.entries.items()},
            list(self.queue),
        )

    def _restore(self, snapshot):
        entries, queue = snapshot
        self.entries = {name: dataclasses.replace(entry) for name, entry in entries.items()}
        self.queue = list(queue)

    def cond(self, predicate, true_function, false_function):
        """``jax.lax.cond`` whose branches must leave the arena in one state."""
        before = self._snapshot()
        ends = []

        def branch(function, label):
            def traced():
                self._restore(before)
                self.phase_push(label)
                function()
                self.phase_pop()
                ends.append((label, self.signature(), self._snapshot()))

            return traced

        jax.lax.cond(predicate, branch(true_function, "true"), branch(false_function, "false"))
        (_, first, snapshot), (_, second, _) = ends
        if first != second:
            raise PoolError(
                "The branches of a cond leave the arena in different states:\n"
                f"  true:  {first}\n  false: {second}"
            )
        self._restore(snapshot)

    def when(self, predicate):
        """``pl.when`` decorator; the guarded code must not change the arena."""
        def decorator(function):
            before = self.signature()

            def traced():
                function()
                if self.signature() != before:
                    raise PoolError(
                        f"pl.when body changed the arena: {before} -> {self.signature()}"
                    )

            pl.when(predicate)(traced)

        return decorator

    def loop(self, lower, upper):
        """``pl.loop`` decorator; the body must end in the state it began in.
        Loads carried in from the previous iteration run unconditionally
        (the iteration exists), so their predicates are dropped."""
        def decorator(body):
            entry_signature = self.signature()

            def traced(index):
                for entry in self.entries.values():
                    entry.predicate = None
                    if entry.state == "queued":
                        entry.index = index  # requested for this iteration
                self.phase_push("loop")
                body(index)
                self.phase_pop()
                if self.signature() != entry_signature:
                    raise PoolError(
                        "The loop body does not end in the arena state it began in:\n"
                        f"  entry: {entry_signature}\n  exit:  {self.signature()}"
                    )
                for entry in self.entries.values():
                    entry.predicate = None

            pl.loop(lower, upper)(traced)
            # The last iteration's requests were for a layer that does not
            # exist; nothing may wait for them.
            self.finished = True

        return decorator

    # -- reporting -----------------------------------------------------------

    def phase_push(self, label):
        self.phase = f"{self.phase}/{label}" if self.phase else label

    def phase_pop(self):
        self.phase = self.phase.rpartition("/")[0]

    def _log(self, op, entry):
        used = sum(_aligned(other.nbytes) for other in self.entries.values() if other.offset is not None)
        event = (self.phase, op, entry.name, entry.offset, entry.nbytes, used)
        self.events.append(event)

    def timeline(self):
        lines = [f"VMEM pool: {self.capacity / 2**20:.2f} MiB arena"]
        for phase, op, name, offset, nbytes, used in self.events:
            place = "" if offset is None else f"@{offset / 2**20:6.2f}"
            lines.append(
                f"  {phase:28s} {op:9s} {name:12s} {nbytes / 2**20:6.2f} MiB {place:8s}"
                f" in use {used / 2**20:6.2f}"
            )
        return "\n".join(lines)

    # -- helpers -------------------------------------------------------------

    def _check_open(self, name):
        if self.finished:
            raise PoolError(f"{name}: the pool is closed")
        if name in self.entries:
            raise PoolError(f"{name} is already live")

    def _live(self, name):
        if self.finished and name in self.entries:
            raise PoolError(f"{name}: requested in the last loop iteration, never issued")
        if name not in self.entries:
            raise PoolError(f"{name} is not live")
        return self.entries[name]

    def _describe(self, nbytes):
        held = ", ".join(
            f"{entry.name} [{entry.offset / 2**20:.2f}, +{entry.nbytes / 2**20:.2f}) {entry.state}"
            for entry in self.entries.values()
            if entry.offset is not None
        )
        queued = ", ".join(self.queue)
        return (
            f"needs {nbytes / 2**20:.2f} MiB of a {self.capacity / 2**20:.2f} MiB arena; "
            f"held: {held or 'nothing'}; queued: {queued or 'nothing'}"
        )


def _aligned(nbytes):
    return -(-nbytes // ALIGN) * ALIGN
