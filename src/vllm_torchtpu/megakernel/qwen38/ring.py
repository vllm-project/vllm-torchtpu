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
"""Reduce-scatters over the stage's cores as two levels of remote DMAs.

The host's chips form a 2x2 cycle (:func:`ring_order`). A core first adds
the partial its sibling (the other core of its chip) pushes for the rows the
core forwards: the two cores of a chip share a link to each neighbor chip,
and the chip-local DMA is several times faster. Then the cores of the same
parity on the four chips form a ring, which runs in both directions at once:
stream 0 forward with the first half of every owner's rows, stream 1
backward with the second half, so each link between neighbor chips carries
each parity's stream once each way. A stream moves chunks of ``chunk``
rows. At each hop a core adds the chip's partial of the chunk's rows (its
own FP32 partial and its sibling's), rounds the sum to BF16 and pushes it
into the next core's VMEM inbox; after three hops the chunk reaches its
owner, which adds its chip's partial and writes the rows out. Every byte
crosses each link once per parity: three quarters of a chip's partial over
its two outgoing links, against seven eighths for one ring over all eight
cores.

Chunks take the ring's steps in groups of ``group``: a chunk pushed at one
step is consumed ``group`` iterations later, while the group's other chunks
are on the wire, and ``group + 1`` inbox slots with credits back to the
sender keep every wait on a core that is behind. (All chunks of one step
before the next would make a sender wait on its receiver running ahead, all
around the ring.) The sibling's partial for an iteration's chunk is pushed
in the same iteration, before either core waits for the other's.

:class:`Plan` is the schedule and :class:`Streams` its pushes inside a
kernel, which supplies each iteration's two partials (``moe.combine_scatter``
combines them from the expert products).
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Sequence

import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.qwen38.reference import BF16, F32

RING_ID = 53  # the ring kernels' barrier semaphore
CHUNK = 64  # rows per push
GROUP = 2  # chunks on the wire per stream
SIDE = 2  # sibling inbox slots (and push buffers) per stream
CHIP_CYCLE = ((0, 0), (1, 0), (1, 1), (0, 1))  # a host's chips, neighbor to neighbor
FORWARD = (1, -1)  # each stream's direction along the ring


def ring_order(devices: Sequence) -> tuple[int, ...]:
    """The ring over ``devices`` (records with chip ``coords`` and
    ``core_on_chip``, one host's 2x2 chips with two cores each; an axis in
    mesh order) as their mesh positions: the chips around their cycle, both
    cores of each in turn."""
    devices = list(devices)
    xs = sorted({d.coords[0] for d in devices})
    ys = sorted({d.coords[1] for d in devices})
    where = {
        (xs.index(d.coords[0]), ys.index(d.coords[1]), d.core_on_chip): i
        for i, d in enumerate(devices)
    }
    cycle = [(x, y, c) for x, y in CHIP_CYCLE for c in (0, 1)]
    if len(devices) != len(cycle) or set(where) != set(cycle):
        raise ValueError("the ring needs one host's 2x2 chips with two cores each")
    return tuple(where[core] for core in cycle)


@dataclasses.dataclass(frozen=True)
class Plan:
    """A reduce-scatter of ``rows`` rows per core of ``tail``-shaped rows
    over the cores of ``ring`` (:func:`ring_order`: two per chip)."""

    ring: tuple[int, ...]
    rows: int
    tail: tuple[int, ...]
    chunk: int = CHUNK
    group: int = GROUP

    def __post_init__(self):
        if self.half % self.chunk or self.chunks % self.group:
            raise ValueError(
                f"{self.rows} rows per core are not whole {self.group} x "
                f"{self.chunk}-row groups per stream"
            )

    @property
    def chips(self) -> int:
        return len(self.ring) // 2

    @property
    def half(self) -> int:
        """Each owner's rows per stream."""
        return self.rows // 2

    @property
    def chunks(self) -> int:
        return self.half // self.chunk

    @property
    def steps(self) -> int:
        """A chunk's hops between chips, then its owner's own add."""
        return self.chips

    @property
    def slots(self) -> int:
        return self.group + 1

    @property
    def buffers(self) -> int:
        return 2 * self.group

    @property
    def iterations(self) -> int:
        return self.chunks * self.steps

    def prefetch(self):
        """The scalar-prefetch operands :class:`Streams` reads: the ring and
        each mesh position's place on it."""
        return (
            np.asarray(self.ring, np.int32),
            np.argsort(self.ring).astype(np.int32),
        )

    def scratch_shapes(self):
        rows = (self.chunk, *self.tail)
        return [
            pltpu.VMEM((2, self.slots, *rows), BF16),  # inbox
            pltpu.VMEM((2, self.buffers, *rows), BF16),  # results on their way
            pltpu.VMEM((2, SIDE, *rows), BF16),  # the sibling's partials
            pltpu.VMEM((2, SIDE, *rows), BF16),  # partials on their way to it
            pltpu.SemaphoreType.DMA((2, self.buffers)),  # pushes sent
            pltpu.SemaphoreType.DMA((2, self.slots)),  # pushes received
            pltpu.SemaphoreType.DMA((2, self.buffers)),  # outputs stored
            pltpu.SemaphoreType.DMA((2, SIDE)),  # sibling pushes sent
            pltpu.SemaphoreType.DMA((2, SIDE)),  # sibling pushes received
            pltpu.SemaphoreType.REGULAR((2,)),  # inbox credits
            pltpu.SemaphoreType.REGULAR((2,)),  # sibling inbox credits
        ]

    def vmem_bytes(self) -> int:
        rows = self.chunk * int(np.prod(self.tail))
        return 2 * (self.slots + self.buffers + 2 * SIDE) * rows * 2


class Streams:
    """A :class:`Plan`'s two streams inside its kernel along mesh axis
    ``axis``: ``ring_ref`` / ``position_ref`` from :meth:`Plan.prefetch`,
    ``out_hbm`` this core's rows ``[rows, *tail]`` BF16, ``scratch`` from
    :meth:`Plan.scratch_shapes`.

    A stream's iteration ``i`` takes the sibling's chunk at
    :meth:`sibling_row` (:meth:`give`) and then its own at
    :meth:`partial_row` (:meth:`finish`), both FP32 partials the kernel
    supplies."""

    def __init__(
        self, plan: Plan, axis: str, ring_ref, position_ref, out_hbm, *scratch
    ):
        self.plan, self.ring_ref, self.out_hbm = plan, ring_ref, out_hbm
        (
            self.inbox,
            self.result,
            self.side_inbox,
            self.side_result,
            self.send_sems,
            self.recv_sems,
            self.out_sems,
            self.side_send_sems,
            self.side_recv_sems,
            self.credits,
            self.side_credits,
        ) = scratch
        here = position_ref[lax.axis_index(axis)]
        self.chip, self.parity = here // 2, lax.rem(here, 2)
        self.sibling = ring_ref[2 * self.chip + 1 - self.parity]
        self.succ = [self._at(d, self.parity) for d in FORWARD]
        self.pred = [self._at(-d, self.parity) for d in FORWARD]

    def _at(self, offset, parity):
        """The core of ``parity`` on the chip ``offset`` chips along."""
        chips = self.plan.chips
        return self.ring_ref[2 * lax.rem(self.chip + offset + chips, chips) + parity]

    def enter(self) -> None:
        """The ring neighbors and the sibling are in the kernel before the
        first push: one round on the barrier semaphore (``collective_id``),
        then one on a scoped one."""
        peers = (self.succ[0], self.pred[0], self.sibling)
        barrier = pltpu.get_barrier_semaphore()
        for peer in peers:
            pl.semaphore_signal(
                barrier, 1, device_id=(peer,), device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(barrier, len(peers))

        @functools.partial(pl.run_scoped, ready=pltpu.SemaphoreType.REGULAR)
        def _(ready):
            for peer in peers:
                pl.semaphore_signal(
                    ready, 1, device_id=(peer,), device_id_type=pl.DeviceIdType.MESH
                )
            pl.semaphore_wait(ready, len(peers))

    def decompose(self, i):
        """Iteration ``i``: its step, chunk, and the indices of the push it
        makes and of the push it consumes."""
        p = self.plan
        g, r = i // (p.steps * p.group), i % (p.steps * p.group)
        s, j = r // p.group, g * p.group + r % p.group
        push = (g * (p.steps - 1) + s) * p.group + r % p.group
        return s, j, push, push - p.group

    def _row(self, d, i, parity):
        p = self.plan
        s, j, _, _ = self.decompose(i)
        owner = self._at(-FORWARD[d] * (s + 1), parity)
        return owner * p.rows + d * p.half + j * p.chunk

    def partial_row(self, d, i):
        """The first row, in the reduce-scatter's input (``[cores * rows,
        ...]``, owner-major), of the chunk stream ``d`` adds at iteration
        ``i``."""
        return self._row(d, i, self.parity)

    def sibling_row(self, d, i):
        """The first row of the chunk the sibling's stream ``d`` adds at
        iteration ``i``."""
        return self._row(d, i, 1 - self.parity)

    def _push(self, d, i, k):
        p = self.plan
        return pltpu.make_async_remote_copy(
            self.result.at[d, i % p.buffers],
            self.inbox.at[d, k % p.slots],
            self.send_sems.at[d, i % p.buffers],
            self.recv_sems.at[d, k % p.slots],
            device_id=(self.succ[d],),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def _side_push(self, d, i):
        return pltpu.make_async_remote_copy(
            self.side_result.at[d, i % SIDE],
            self.side_inbox.at[d, i % SIDE],
            self.side_send_sems.at[d, i % SIDE],
            self.side_recv_sems.at[d, i % SIDE],
            device_id=(self.sibling,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def _store(self, d, i):
        p = self.plan
        _, j, _, _ = self.decompose(i)
        return pltpu.make_async_copy(
            self.result.at[d, i % p.buffers],
            self.out_hbm.at[pl.ds(d * p.half + j * p.chunk, p.chunk)],
            self.out_sems.at[d, i % p.buffers],
        )

    def _release(self, d, i):
        """Iteration ``i``'s result buffer has left (pushed or stored)."""
        steps = self.plan.steps
        s, _, k, _ = self.decompose(i)
        pl.when(s < steps - 1)(lambda: self._push(d, i, k).wait_send())
        pl.when(s == steps - 1)(lambda: self._store(d, i).wait())

    def give(self, d, i, partial) -> None:
        """Push this core's partial of the sibling's chunk at iteration ``i``
        (an FP32 ref ``[chunk, *tail]``), rounded to BF16."""
        pl.when(i >= SIDE)(lambda: self._side_push(d, i - SIDE).wait_send())
        self.side_result[d, i % SIDE] = partial[...].astype(BF16)
        pl.when(i >= SIDE)(lambda: pl.semaphore_wait(self.side_credits.at[d], 1))
        self._side_push(d, i).start()

    def finish(self, d, i, partial) -> None:
        """Stream ``d``'s iteration ``i`` with this core's partial of its
        chunk (an FP32 ref ``[chunk, *tail]``): add the sibling's and the
        chunk received, then push the sum on or store it."""
        p = self.plan
        s, _, k, k_in = self.decompose(i)
        pl.when(i >= p.buffers)(lambda: self._release(d, i - p.buffers))
        self._side_push(d, i).wait_recv()
        chip = partial[...] + self.side_inbox[d, i % SIDE].astype(F32)
        pl.semaphore_signal(
            self.side_credits.at[d],
            1,
            device_id=(self.sibling,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        result = self.result.at[d, i % p.buffers]

        @pl.when(s == 0)
        def _():
            result[...] = chip.astype(BF16)

        @pl.when(s > 0)
        def _():
            self._push(d, i, k_in).wait_recv()
            received = self.inbox[d, k_in % p.slots].astype(F32)
            result[...] = (chip + received).astype(BF16)
            pl.semaphore_signal(
                self.credits.at[d],
                1,
                device_id=(self.pred[d],),
                device_id_type=pl.DeviceIdType.MESH,
            )

        @pl.when(s < p.steps - 1)
        def _():
            pl.when(k >= p.slots)(lambda: pl.semaphore_wait(self.credits.at[d], 1))
            self._push(d, i, k).start()

        pl.when(s == p.steps - 1)(lambda: self._store(d, i).start())

    def drain(self) -> None:
        """The last results and sibling partials have left and the last
        pushes' credits are back."""
        p = self.plan
        for d in range(2):
            for back in range(p.buffers, 0, -1):
                self._release(d, p.iterations - back)
            for back in range(SIDE, 0, -1):
                self._side_push(d, p.iterations - back).wait_send()
            pl.semaphore_wait(self.credits.at[d], p.slots)
            pl.semaphore_wait(self.side_credits.at[d], SIDE)
