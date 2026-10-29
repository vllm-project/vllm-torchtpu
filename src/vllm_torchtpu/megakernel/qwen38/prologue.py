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
"""The MoE layer's prologue as one kernel per core: the routed experts'
all-gather of every core's rows, with the shared expert computed while it is
on the wire.

The all-gather runs over two levels of remote HBM-to-HBM DMAs, in the ring
order of :mod:`ring`: a core copies its rows to its sibling (the other core
of its chip) and, in both directions, half of them around the ring of its
parity's cores on the four chips; it forwards each ring chunk it receives to
the next core and to its sibling. Every byte crosses a link between chips
once per parity, and the sibling copies ride the chip-local DMA. The routes'
metadata (each row's scale, experts and weights) goes straight to every
core.

The first hop's pushes start at once, and the shared expert's GEMM steps
fill the time between the ring chunks' arrivals, each of which is forwarded
as soon as the steps before it are done.
"""

from __future__ import annotations

import functools

import jax
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.qwen38.reference import BF16, F32
from vllm_torchtpu.megakernel.qwen38.ring import FORWARD

PROLOGUE_ID = 54  # the kernels' barrier semaphores
GATHER_ID = 55
CHUNK = 256  # rows per ring push
ROW_TILE = 256  # x rows per shared-expert GEMM step
GATE_TILE = 256  # gate (and up) columns per weight tile
DOWN_TILE = 1024  # shared-expert output columns per weight tile


def gather_and_share(rows, metadata, x, gate_up, down, ring, axis, *, interpret=False):
    """Every core's ``rows`` ``[n, ...]`` and ``metadata`` (flat arrays of
    ``n * k`` values each) gathered core-major along ``axis`` (``ring``:
    :func:`ring.ring_order`), and the shared expert of this core's normed
    rows ``x`` ``[n, H]`` BF16: ``down(silu(gate) * up)`` with ``gate_up``
    ``[2 I, H]`` and ``down`` ``[H, I]`` (``[out, in]``, BF16), FP32
    accumulation and BF16 between and out. Returns the gathered rows, the
    gathered metadata and the shared expert's ``[n, H]``. ``interpret``
    runs the kernel on the host."""
    gathered, metadata, shared = _gather(
        rows, metadata, (x, gate_up, down), ring, axis, interpret
    )
    assert shared is not None
    return gathered, metadata, shared


def all_gather(rows, ring, axis, *, interpret=False):
    """Every core's ``rows`` ``[n, ...]`` gathered core-major along ``axis``,
    as :func:`gather_and_share` moves them."""
    return _gather(rows, (), None, ring, axis, interpret)[0]


def _gather(rows, metadata, shared_args, ring, axis, interpret):
    n = rows.shape[0]
    share = shared_args is not None
    hidden = inter = row_tile = 0
    if shared_args is not None:
        x, gate_up, down = shared_args
        hidden, inter = x.shape[1], down.shape[1]
        row_tile = min(ROW_TILE, n)
        assert n % row_tile == 0
        assert inter % GATE_TILE == 0 and hidden % DOWN_TILE == 0
    cores, chips = len(ring), len(ring) // 2
    half = n // 2
    chunk = min(CHUNK, half)
    chunks, hops = half // chunk, chips - 1
    assert half % chunk == 0
    metadata = tuple(metadata)
    position = np.argsort(ring).astype(np.int32)

    def kernel(ring_ref, position_ref, rows_hbm, *refs):
        meta_hbm, refs = refs[: len(metadata)], refs[len(metadata) :]
        # The shared expert's inputs, output and activations, when it runs.
        inputs = 3 if share else 0
        shared_in, refs = refs[:inputs], refs[inputs:]
        gathered_hbm, *refs = refs
        meta_out, refs = refs[: len(metadata)], refs[len(metadata) :]
        outputs = 2 if share else 0
        shared_out, refs = refs[:outputs], refs[outputs:]
        own_sem, ring_send, ring_recv, side_send, side_recv, meta_sems = refs[:6]
        shared_scratch = refs[6:]
        here = position_ref[lax.axis_index(axis)]
        chip, parity = here // 2, lax.rem(here, 2)

        def at(offset, par):
            return ring_ref[2 * lax.rem(chip + offset + chips, chips) + par]

        me, sibling = at(0, parity), at(0, 1 - parity)
        succ = [at(d, parity) for d in FORWARD]
        _enter(me, cores)

        def block(owner, d, c):
            return pl.ds(owner * n + d * half + c * chunk, chunk)

        def ring_copy(d, s, c):
            """Hop ``s``'s chunk ``c`` on stream ``d``: the rows of the core
            ``s`` chips behind, to the next core."""
            owner = at(-FORWARD[d] * s, parity)
            local = rows_hbm.at[pl.ds(d * half + c * chunk, chunk)]
            src = local if s == 0 else gathered_hbm.at[block(owner, d, c)]
            return pltpu.make_async_remote_copy(
                src,
                gathered_hbm.at[block(owner, d, c)],
                ring_send.at[d],
                ring_recv.at[d, s, c],
                device_id=(succ[d],),
                device_id_type=pl.DeviceIdType.MESH,
            )

        def side_copy(owner, d, c, own):
            local = rows_hbm.at[pl.ds(d * half + c * chunk, chunk)]
            src = local if own else gathered_hbm.at[block(owner, d, c)]
            return pltpu.make_async_remote_copy(
                src,
                gathered_hbm.at[block(owner, d, c)],
                side_send,
                side_recv,
                device_id=(sibling,),
                device_id_type=pl.DeviceIdType.MESH,
            )

        def meta_copy(f, peer):
            size = metadata[f].shape[0]
            return pltpu.make_async_remote_copy(
                meta_hbm[f],
                meta_out[f].at[pl.ds(me * size, size)],
                meta_sems.at[0],
                meta_sems.at[1],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )

        own = pltpu.make_async_copy(
            rows_hbm, gathered_hbm.at[pl.ds(me * n, n)], own_sem
        )
        own.start()
        for d in range(2):
            for c in range(chunks):
                ring_copy(d, 0, c).start()
                side_copy(me, d, c, True).start()
        local_meta = [
            pltpu.make_async_copy(
                meta_hbm[f], meta_out[f].at[pl.ds(me * m.shape[0], m.shape[0])], own_sem
            )
            for f, m in enumerate(metadata)
        ]
        for copy in local_meta:
            copy.start()
        for hop in range(1, cores):
            for f in range(len(metadata)):
                meta_copy(f, lax.rem(me + hop, cores)).start()

        arrivals = [(s, c) for s in range(hops) for c in range(chunks)]
        shared = (
            SharedExpert(
                shared_in[0],
                shared_in[1],
                shared_in[2],
                shared_out[0],
                shared_out[1],
                shared_scratch,
                row_tile,
            )
            if share
            else None
        )
        bounds = []
        if shared is not None:
            shared.start()
            # The GEMM steps fill the gaps between the ring chunks' arrivals.
            bounds = [
                k * shared.steps // (len(arrivals) + 1)
                for k in range(len(arrivals) + 1)
            ]
            bounds.append(shared.steps)
            pl.loop(bounds[0], bounds[1])(shared.step)
        for k, (s, c) in enumerate(arrivals):
            for d in range(2):
                owner = at(-FORWARD[d] * (s + 1), parity)
                # Hop s of the previous core's stream d has landed.
                pltpu.make_async_remote_copy(
                    gathered_hbm.at[block(owner, d, c)],
                    gathered_hbm.at[block(owner, d, c)],
                    ring_send.at[d],
                    ring_recv.at[d, s, c],
                    device_id=(succ[d],),
                    device_id_type=pl.DeviceIdType.MESH,
                ).wait_recv()
                if s + 1 < hops:
                    ring_copy(d, s + 1, c).start()
                side_copy(owner, d, c, False).start()
            if shared is not None:
                pl.loop(bounds[k + 1], bounds[k + 2])(shared.step)
        if shared is not None:
            shared.finish()

        own.wait()
        for copy in local_meta:
            copy.wait()
        for s in range(hops):
            for d in range(2):
                for c in range(chunks):
                    ring_copy(d, s, c).wait_send()
        # Our rows and the three ring cores' went to the sibling, and as many
        # came from it.
        for _ in range(chips):
            for d in range(2):
                for c in range(chunks):
                    side_copy(me, d, c, True).wait_send()
                    side_copy(me, d, c, True).wait_recv()
        for _ in range(1, cores):
            for f in range(len(metadata)):
                meta_copy(f, me).wait_send()
                meta_copy(f, me).wait_recv()

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    outputs = (
        jax.ShapeDtypeStruct((cores * n, *rows.shape[1:]), rows.dtype),
        *(jax.ShapeDtypeStruct((cores * m.shape[0],), m.dtype) for m in metadata),
    )
    semaphores = [
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2, hops, chunks)),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA((2,)),
    ]
    if share:
        outputs += (jax.ShapeDtypeStruct((n, hidden), BF16),)
        scratch = [
            pltpu.VMEM((n, inter), BF16),
            *semaphores,
            *SharedExpert.scratch_shapes(row_tile, hidden, inter),
        ]
        vmem = SharedExpert.vmem_bytes(n, row_tile, hidden, inter) + 2**23
    else:
        scratch, vmem = semaphores, 2**23
    out = pl.pallas_call(
        kernel,
        out_shape=outputs,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1,),
            in_specs=[hbm] * (1 + len(metadata) + (3 if share else 0)),
            out_specs=(hbm,) * len(outputs),
            scratch_shapes=scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=PROLOGUE_ID if share else GATHER_ID, vmem_limit_bytes=vmem
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
        name="qwen38_moe_prologue" if share else "qwen38_all_gather",
    )(
        np.asarray(ring, np.int32),
        position,
        rows,
        *metadata,
        *(shared_args if share else ()),
    )
    if share:
        return out[0], tuple(out[1:-1]), out[-1]
    return out[0], tuple(out[1:]), None


class SharedExpert:
    """``down(silu(gate) * up)`` of ``x_hbm`` ``[n, H]`` into ``out_hbm`` as
    ``steps`` GEMM steps the kernel interleaves with its pushes: gate/up
    weight-stationary over column tiles (x streamed per tile, the
    activations kept in ``act`` ``[n, I]``), then down over output column
    tiles from ``act``. Weights and x arrive one step ahead; ``scratch`` is
    :meth:`scratch_shapes`'."""

    def __init__(self, x_hbm, gate_up_hbm, down_hbm, out_hbm, act, scratch, row_tile):
        self.x_hbm, self.gate_up_hbm, self.down_hbm, self.out_hbm = (
            x_hbm,
            gate_up_hbm,
            down_hbm,
            out_hbm,
        )
        self.act, self.row_tile = act, row_tile
        (
            self.x_vmem,
            self.w_vmem,
            self.w2_vmem,
            self.out_vmem,
            self.x_sems,
            self.w_sems,
            self.w2_sems,
            self.out_sems,
        ) = scratch
        n, hidden = x_hbm.shape
        self.inter = act.shape[1]
        self.row_tiles = n // row_tile
        self.gate_steps = self.inter // GATE_TILE * self.row_tiles
        self.down_steps = hidden // DOWN_TILE * self.row_tiles
        self.steps = self.gate_steps + self.down_steps

    @staticmethod
    def scratch_shapes(row_tile, hidden, inter):
        return [
            pltpu.VMEM((2, row_tile, hidden), BF16),  # x tiles
            pltpu.VMEM((2, 2, GATE_TILE, hidden), BF16),  # gate and up tiles
            pltpu.VMEM((2, DOWN_TILE, inter), BF16),  # down tiles
            pltpu.VMEM((2, row_tile, DOWN_TILE), BF16),  # output tiles
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2, 2)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
        ]

    @staticmethod
    def vmem_bytes(n, row_tile, hidden, inter):
        tiles = 2 * row_tile * hidden + 4 * GATE_TILE * hidden + 2 * DOWN_TILE * inter
        return 2 * (n * inter + tiles + 2 * row_tile * DOWN_TILE)

    def _gate_weights(self, j):
        return [
            pltpu.make_async_copy(
                self.gate_up_hbm.at[
                    pl.ds(part * self.inter + j * GATE_TILE, GATE_TILE)
                ],
                self.w_vmem.at[j % 2, part],
                self.w_sems.at[j % 2, part],
            )
            for part in range(2)
        ]

    def _x_tile(self, t):
        i = t % self.row_tiles
        return pltpu.make_async_copy(
            self.x_hbm.at[pl.ds(i * self.row_tile, self.row_tile)],
            self.x_vmem.at[t % 2],
            self.x_sems.at[t % 2],
        )

    def _down_weights(self, k):
        return pltpu.make_async_copy(
            self.down_hbm.at[pl.ds(k * DOWN_TILE, DOWN_TILE)],
            self.w2_vmem.at[k % 2],
            self.w2_sems.at[k % 2],
        )

    def _store(self, u):
        k, i = u // self.row_tiles, u % self.row_tiles
        rows, cols = (
            pl.ds(i * self.row_tile, self.row_tile),
            pl.ds(k * DOWN_TILE, DOWN_TILE),
        )
        return pltpu.make_async_copy(
            self.out_vmem.at[u % 2],
            self.out_hbm.at[rows, cols],
            self.out_sems.at[u % 2],
        )

    def start(self) -> None:
        """The first weight tiles and x tile on their way."""
        for copy in self._gate_weights(0):
            copy.start()
        self._x_tile(0).start()
        self._down_weights(0).start()

    def step(self, t) -> None:
        pl.when(t < self.gate_steps)(lambda: self._gate_up(t))
        pl.when(t >= self.gate_steps)(lambda: self._down(t - self.gate_steps))

    def _gate_up(self, t):
        j, i = t // self.row_tiles, t % self.row_tiles
        gate_tiles = self.gate_steps // self.row_tiles

        @pl.when(i == 0)
        def _():
            for copy in self._gate_weights(j):
                copy.wait()

            @pl.when(j + 1 < gate_tiles)
            def _():
                for copy in self._gate_weights(j + 1):
                    copy.start()

        self._x_tile(t).wait()
        pl.when(t + 1 < self.gate_steps)(lambda: self._x_tile(t + 1).start())
        x = self.x_vmem[t % 2]
        gate, up = (
            lax.dot_general(
                x,
                self.w_vmem[j % 2, part],
                (((1,), (1,)), ((), ())),
                preferred_element_type=F32,
            )
            for part in range(2)
        )
        cols = pl.ds(pl.multiple_of(j * GATE_TILE, GATE_TILE), GATE_TILE)
        rows = pl.ds(pl.multiple_of(i * self.row_tile, self.row_tile), self.row_tile)
        self.act[rows, cols] = (jax.nn.silu(gate) * up).astype(BF16)

    def _down(self, u):
        k, i = u // self.row_tiles, u % self.row_tiles
        down_tiles = self.down_steps // self.row_tiles

        @pl.when(i == 0)
        def _():
            self._down_weights(k).wait()
            pl.when(k + 1 < down_tiles)(lambda: self._down_weights(k + 1).start())

        pl.when(u >= 2)(lambda: self._store(u - 2).wait())
        rows = pl.ds(pl.multiple_of(i * self.row_tile, self.row_tile), self.row_tile)
        self.out_vmem[u % 2] = lax.dot_general(
            self.act[rows, :],
            self.w2_vmem[k % 2],
            (((1,), (1,)), ((), ())),
            preferred_element_type=F32,
        ).astype(BF16)
        self._store(u).start()

    def finish(self) -> None:
        """The last output tiles are stored."""
        for back in range(min(2, self.down_steps), 0, -1):
            self._store(self.down_steps - back).wait()


def _enter(me, cores: int) -> None:
    """Every core is in the kernel before the first push: one round on the
    barrier semaphore (``collective_id``), then one on a scoped one."""
    barrier = pltpu.get_barrier_semaphore()
    for hop in range(1, cores):
        pl.semaphore_signal(
            barrier,
            1,
            device_id=(lax.rem(me + hop, cores),),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier, cores - 1)

    @functools.partial(pl.run_scoped, ready=pltpu.SemaphoreType.REGULAR)
    def _(ready):
        for hop in range(1, cores):
            pl.semaphore_signal(
                ready,
                1,
                device_id=(lax.rem(me + hop, cores),),
                device_id_type=pl.DeviceIdType.MESH,
            )
        pl.semaphore_wait(ready, cores - 1)
