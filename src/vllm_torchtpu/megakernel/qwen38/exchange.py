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
"""The GDN mixer's exchanges between the stage's cores, as remote DMAs.

The residual stream's rows are ``chunk``-row blocks dealt round-robin over
the cores: core ``s``'s block ``j`` is flat block ``j * cores + s``. The GDN
conv and recurrence run on each core's value-head shard over all rows in
flat order. :func:`project` computes the input projections of the core's
rows and sends each destination core its columns, :func:`send_rows_home`
sends the mixer's output back to the rows' owners (the gate and the output
norm stay with the owner, which projects its rows' gate itself). In both,
block ``(j, d)`` of the source goes straight to block ``(j, s)`` of core
``d``'s output, ``s`` the sending core: one DMA per block and peer, all in
flight at once, pushed from an HBM staging copy so the next blocks' compute
never waits for the links.

The phases are functions over refs, so a kernel can run them around other
work; :func:`project_to_heads` and :func:`to_rows` run each alone.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.qwen38.reference import BF16, F32

TO_HEADS_ID = 50  # the kernels' barrier semaphores
TO_ROWS_ID = 51
TILE_COLUMNS = 1024  # a projection weight tile's columns at most ([1024, H] FP8)
TILE_BLOCKS = 4  # row blocks per projection row tile at most: one weight pass each


def _tile_columns(columns: int) -> int:
    """The widest lane-aligned tile of at most ``TILE_COLUMNS`` columns that
    divides ``columns``."""
    top = min(columns, TILE_COLUMNS)
    return max(c for c in range(128, top + 1, 128) if columns % c == 0)


@dataclasses.dataclass(frozen=True)
class Projection:
    """The GDN input projections of one core's ``rows`` (:func:`project`):
    ``width`` columns per destination core and ``gate`` columns kept, from
    FP8 ``[rows, hidden]``. A weight tile is ``[columns, hidden]``, taken in
    one dot over the whole contraction (the MXU accumulates it; no partial
    sums), for row tiles of up to ``TILE_BLOCKS`` row blocks."""

    cores: int
    chunk: int
    rows: int
    hidden: int
    width: int
    gate: int

    def __post_init__(self):
        assert self.rows % self.chunk == 0
        assert self.width % 128 == 0 and self.gate % 128 == 0

    @property
    def blocks(self) -> int:
        return self.rows // self.chunk

    @property
    def m_chunks(self) -> int:
        """Row blocks per row tile."""
        return max(c for c in range(1, TILE_BLOCKS + 1) if self.blocks % c == 0)

    @property
    def m_rows(self) -> int:
        return self.m_chunks * self.chunk

    @property
    def n_tile(self) -> int:
        """A destination's columns per weight tile."""
        return _tile_columns(self.width)

    @property
    def z_tile(self) -> int:
        """The gate's columns per weight tile."""
        return _tile_columns(self.gate)

    @property
    def subs(self) -> int:
        """Weight tiles per destination."""
        return self.width // self.n_tile

    @property
    def jobs(self) -> int:
        """Output tiles per row tile: the heads' (one per core), then the
        gate's (one per weight tile)."""
        return self.cores + self.gate // self.z_tile

    @property
    def steps(self) -> int:
        """Weight tiles per row tile."""
        return self.cores * self.subs + self.gate // self.z_tile

    def out_shapes(self) -> tuple[jax.ShapeDtypeStruct, ...]:
        """The received rows (flat order), the gate, the staging copy."""
        return (
            jax.ShapeDtypeStruct((self.cores * self.rows, self.width), BF16),
            jax.ShapeDtypeStruct((self.rows, self.gate), BF16),
            jax.ShapeDtypeStruct((self.rows, self.cores * self.width), BF16),
        )

    def vmem_shapes(self, x_dtype, w_dtype) -> list:
        """The row tile, both weight tiles and both output tiles: live only
        while :func:`project` runs."""
        return [
            pltpu.VMEM((self.m_rows, self.hidden), x_dtype),
            pltpu.VMEM((2, max(self.n_tile, self.z_tile), self.hidden), w_dtype),
            pltpu.VMEM((2, self.m_rows, max(self.width, self.z_tile)), BF16),
        ]

    def semaphore_shapes(self) -> list:
        """The row tile's, the weight tiles', the stores' and the sends' DMA
        semaphores; the receive semaphores are :func:`project`'s
        ``recv_sem``."""
        return [
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA,
        ]

    def vmem_bytes(self) -> int:
        """:meth:`vmem_shapes` plus the dot's F32 result."""
        tile = max(self.n_tile, self.z_tile)
        return (
            self.m_rows * self.hidden
            + 2 * tile * self.hidden
            + 4 * self.m_rows * max(self.width, self.z_tile)
            + 4 * self.m_rows * tile
        )


def project(
    p: Projection,
    me,
    inputs,
    outputs,
    vmem,
    sems,
    recv_sem: Callable[[jax.Array], jax.Array],
) -> None:
    """``x @ weight.T * scale`` to the heads' cores and the gate ``x @
    z_weight.T * z_scale`` kept: ``inputs`` = (``x`` ``[rows, hidden]`` FP8,
    ``weight`` ``[cores * width, hidden]`` FP8 with row block ``d`` for core
    ``d``, ``scale`` ``[1, cores * width]``, ``z_weight`` ``[gate, hidden]``,
    ``z_scale`` ``[1, gate]``), ``outputs`` = :meth:`Projection.out_shapes`,
    ``vmem`` and ``sems`` as :meth:`Projection.vmem_shapes` and
    :meth:`Projection.semaphore_shapes`: row block ``j`` of core ``s``'s
    product with block ``me`` lands at flat block ``j * cores + s``,
    signalling ``recv_sem(j)`` on the receiver. Returns with the pushes in
    flight (:func:`wait_projected`).

    Per row tile the core computes its ``[m_rows, width]`` tiles, the peers'
    first and its own last, then the gate's ``[m_rows, z_tile]`` tiles. A
    peer's tile goes from VMEM to the HBM staging copy and its blocks are
    pushed from there, so the pushes of all finished tiles are in flight
    while the next tiles compute."""
    x_hbm, w_hbm, scale_ref, z_hbm, z_scale_ref = inputs
    out_ref, z_ref, staging = outputs
    x_vmem, w_vmem, tile_vmem = vmem
    x_sem, w_sems, store_sems, send_sem = sems
    cores, chunk, width, jobs = p.cores, p.chunk, p.width, p.jobs
    head_steps = cores * p.subs
    tiles = p.rows // p.m_rows
    steps = tiles * p.steps

    def dest(job):
        return (me + 1 + job) % cores

    def place(step):
        """Weight tile ``step``'s row tile, job, tile within the job and
        whether it is a head tile."""
        m, in_tile = step // p.steps, step % p.steps
        is_head = in_tile < head_steps
        job = jnp.where(is_head, in_tile // p.subs, in_tile - head_steps + cores)
        return m, job, jnp.where(is_head, in_tile % p.subs, 0), is_head

    def head_cols(job, sub):
        return pl.ds(pl.multiple_of(dest(job) * width + sub * p.n_tile, 128), p.n_tile)

    def gate_cols(job):
        return _block(jnp.maximum(job - cores, 0), p.z_tile)

    def load_x(m):
        return pltpu.make_async_copy(x_hbm.at[_block(m, p.m_rows)], x_vmem, x_sem)

    def weight_copies(step):
        """(is a head tile, its weight copy, a gate tile's weight copy)."""
        _, job, sub, is_head = place(step)
        slot = step % 2
        head = pltpu.make_async_copy(
            w_hbm.at[head_cols(job, sub)],
            w_vmem.at[slot, pl.ds(0, p.n_tile)],
            w_sems.at[slot],
        )
        gate_copy = pltpu.make_async_copy(
            z_hbm.at[gate_cols(job)],
            w_vmem.at[slot, pl.ds(0, p.z_tile)],
            w_sems.at[slot],
        )
        return is_head, head, gate_copy

    def start_w(step):
        is_head, head, gate_copy = weight_copies(step)
        pl.when(is_head)(head.start)
        pl.when(jnp.logical_not(is_head))(gate_copy.start)

    def wait_w(step):
        is_head, head, gate_copy = weight_copies(step)
        pl.when(is_head)(head.wait)
        pl.when(jnp.logical_not(is_head))(gate_copy.wait)

    def stores(tile):
        """The tile's stores: to staging (a peer's), to the output's flat
        blocks (the own), to the gate (a gate tile)."""
        m, job, slot = tile // jobs, tile % jobs, tile % 2
        to_staging = pltpu.make_async_copy(
            tile_vmem.at[slot, :, pl.ds(0, width)],
            staging.at[_block(m, p.m_rows), _block(dest(job), width)],
            store_sems.at[slot],
        )
        to_out = [
            pltpu.make_async_copy(
                tile_vmem.at[slot, pl.ds(c * chunk, chunk), pl.ds(0, width)],
                out_ref.at[_block((m * p.m_chunks + c) * cores + me, chunk)],
                store_sems.at[slot],
            )
            for c in range(p.m_chunks)
        ]
        to_gate = pltpu.make_async_copy(
            tile_vmem.at[slot, :, pl.ds(0, p.z_tile)],
            z_ref.at[_block(m, p.m_rows), gate_cols(job)],
            store_sems.at[slot],
        )
        return to_staging, to_out, to_gate

    def wait_store(tile):
        to_staging, _, to_gate = stores(tile)
        is_head = tile % jobs < cores
        pl.when(is_head)(to_staging.wait)  # the own tile's stores add up to it
        pl.when(jnp.logical_not(is_head))(to_gate.wait)

    load_x(0).start()
    start_w(0)

    @pl.loop(0, steps)
    def _(step):
        m, job, sub, is_head = place(step)
        tile, slot = m * jobs + job, step % 2
        out_slot = tile % 2
        pl.when(step % p.steps == 0)(load_x(m).wait)
        wait_w(step)

        @pl.when(step + 1 < steps)
        def _():
            start_w(step + 1)

        def product(columns):
            return lax.dot_general(
                x_vmem[...],
                w_vmem[slot, pl.ds(0, columns)],
                (((1,), (1,)), ((), ())),
                preferred_element_type=F32,
            )

        @pl.when(is_head)
        def _():
            cols = pl.ds(pl.multiple_of(sub * p.n_tile, 128), p.n_tile)
            tile_vmem[out_slot, :, cols] = (
                product(p.n_tile) * scale_ref[:, head_cols(job, sub)]
            ).astype(BF16)

        @pl.when(jnp.logical_not(is_head))
        def _():
            tile_vmem[out_slot, :, pl.ds(0, p.z_tile)] = (
                product(p.z_tile) * z_scale_ref[:, gate_cols(job)]
            ).astype(BF16)

        # The row tile's last weight tile is multiplied: x's buffer is free.
        @pl.when((step % p.steps == p.steps - 1) & (m + 1 < tiles))
        def _():
            load_x(m + 1).start()

        @pl.when(jnp.logical_not(is_head) | (sub == p.subs - 1))
        def _():
            to_staging, to_out, to_gate = stores(tile)

            @pl.when(is_head)
            def _():
                @pl.when(job == cores - 1)
                def _():
                    for copy in to_out:
                        copy.start()

                pl.when(job < cores - 1)(to_staging.start)

            pl.when(jnp.logical_not(is_head))(to_gate.start)

            # The previous tile is stored (its slot free): push it if a peer's.
            @pl.when(tile > 0)
            def _():
                wait_store(tile - 1)
                previous = (tile - 1) % jobs

                @pl.when(previous < cores - 1)
                def _():
                    for c in range(p.m_chunks):
                        block = (tile - 1) // jobs * p.m_chunks + c
                        _push_projected(
                            p, me, outputs, send_sem, recv_sem, block, previous
                        ).start()

    wait_store(tiles * jobs - 1)


def wait_projected(p: Projection, me, outputs, sems, recv_sem, block) -> None:
    """Row block ``block``: this core's pushes sent and the peers' arrived."""
    out_ref = outputs[0]
    send_sem = sems[-1]
    for job in range(p.cores - 1):
        _push_projected(p, me, outputs, send_sem, recv_sem, block, job).wait_send()
        source = (me + 1 + job) % p.cores
        arrived = out_ref.at[_block(block * p.cores + source, p.chunk)]
        pltpu.make_async_copy(arrived, arrived, recv_sem(block)).wait()


def _push_projected(p: Projection, me, outputs, send_sem, recv_sem, block, job):
    out_ref, _, staging = outputs
    dest = (me + 1 + job) % p.cores
    return pltpu.make_async_remote_copy(
        staging.at[_block(block, p.chunk), _block(dest, p.width)],
        out_ref.at[_block(block * p.cores + me, p.chunk)],
        send_sem,
        recv_sem(block),
        device_id=(dest,),
        device_id_type=pl.DeviceIdType.MESH,
    )


def project_to_heads(
    x: jax.Array,
    weight: jax.Array,
    scale: jax.Array,
    z_weight: jax.Array,
    z_scale: jax.Array,
    axis: str,
    cores: int,
    chunk: int,
    *,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """:func:`project` alone: the received rows ``[cores * rows, width]``
    BF16 in flat order and the gate ``[rows, gate]``. ``interpret`` runs the
    kernel on the host, for tests without a TPU."""
    p = Projection(
        cores,
        chunk,
        x.shape[0],
        x.shape[1],
        weight.shape[0] // cores,
        z_weight.shape[0],
    )

    def kernel(*refs):
        inputs, outputs, scratch = refs[:5], refs[5:8], refs[8:]
        vmem, sems, recv = scratch[:3], scratch[3:7], scratch[7]
        me = lax.axis_index(axis)
        enter(me, cores)
        project(p, me, inputs, outputs, vmem, sems, lambda block: recv)

        @pl.loop(0, p.blocks)
        def _(block):
            wait_projected(p, me, outputs, sems, lambda _: recv, block)

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    flat, z, _ = pl.pallas_call(
        kernel,
        out_shape=p.out_shapes(),
        in_specs=[hbm, hbm, vmem_spec, hbm, vmem_spec],
        out_specs=(hbm, hbm, hbm),
        scratch_shapes=[
            *p.vmem_shapes(x.dtype, weight.dtype),
            *p.semaphore_shapes(),
            pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(
            collective_id=TO_HEADS_ID, vmem_limit_bytes=p.vmem_bytes() + 2**22
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
        name="qwen38_gdn_project_to_heads",
    )(x, weight, scale, z_weight, z_scale)
    return flat, z


def rows_scratch_shapes(chunk: int, heads: int, head_dim: int) -> list:
    """:func:`send_rows_home`'s VMEM (both input chunks, both output tiles)
    and DMA semaphores, except the receive semaphore."""
    return [
        pltpu.VMEM((2, chunk, heads, head_dim), BF16),
        pltpu.VMEM((2, chunk, heads * head_dim), BF16),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA,
    ]


def send_rows_home(
    me, cores, chunk, core_hbm, out_ref, staging, scratch, recv_sem, block
):
    """Start sending row block ``block`` of the GDN output home: ``core_hbm``
    ``[cores * rows, heads, head_dim]`` in flat order -> the owners' ``[rows,
    cores * heads * head_dim]`` at column block ``me`` (the heads in order).
    The chunks go owner by owner, peers first, through VMEM (to ``[chunk,
    heads * head_dim]``) to the HBM ``staging`` copy ``[cores * rows, heads *
    head_dim]`` and are pushed from there; the own chunk goes straight to the
    output. Returns with the pushes in flight (:func:`wait_rows`)."""
    x_vmem, tile_vmem, load_sems, store_sems, send_sem = scratch
    width = tile_vmem.shape[-1]

    def flat(hop):
        return block * cores + (me + 1 + hop) % cores

    def load(hop):
        return pltpu.make_async_copy(
            core_hbm.at[_block(flat(hop), chunk)],
            x_vmem.at[hop % 2],
            load_sems.at[hop % 2],
        )

    def store(hop):
        if hop == cores - 1:
            target = out_ref.at[_block(block, chunk), _block(me, width)]
        else:
            target = staging.at[_block(flat(hop), chunk)]
        return pltpu.make_async_copy(
            tile_vmem.at[hop % 2], target, store_sems.at[hop % 2]
        )

    load(0).start()
    for hop in range(cores):
        load(hop).wait()
        if hop + 1 < cores:
            load(hop + 1).start()
        if hop >= 2:
            store(hop - 2).wait()  # its slot is reused now
        tile_vmem[hop % 2] = x_vmem[hop % 2].reshape(tile_vmem.shape[1:])
        store(hop).start()
    store(cores - 2).wait()
    store(cores - 1).wait()
    for hop in range(cores - 1):
        _push_rows(
            me, cores, chunk, out_ref, staging, send_sem, recv_sem, block, hop
        ).start()


def wait_rows(me, cores, chunk, out_ref, staging, scratch, recv_sem, block) -> None:
    """Row block ``block``: this core's pushes sent and the owners' rows
    arrived, when ``recv_sem`` counts this block's pushes alone (a shared
    one only says that as many bytes arrived)."""
    send_sem = scratch[-1]
    width = out_ref.shape[-1] // cores
    for hop in range(cores - 1):
        _push_rows(
            me, cores, chunk, out_ref, staging, send_sem, recv_sem, block, hop
        ).wait_send()
        source = (me + 1 + hop) % cores
        arrived = out_ref.at[_block(block, chunk), _block(source, width)]
        pltpu.make_async_copy(arrived, arrived, recv_sem).wait()


def _push_rows(me, cores, chunk, out_ref, staging, send_sem, recv_sem, block, hop):
    owner = (me + 1 + hop) % cores
    width = out_ref.shape[-1] // cores
    return pltpu.make_async_remote_copy(
        staging.at[_block(block * cores + owner, chunk)],
        out_ref.at[_block(block, chunk), _block(me, width)],
        send_sem,
        recv_sem,
        device_id=(owner,),
        device_id_type=pl.DeviceIdType.MESH,
    )


def to_rows(
    core: jax.Array, axis: str, cores: int, chunk: int, *, interpret: bool = False
) -> jax.Array:
    """:func:`send_rows_home` alone: ``core`` ``[cores * rows, heads,
    head_dim]`` in flat order -> ``[rows, cores * heads * head_dim]``,
    column block ``s`` from core ``s``. ``interpret`` runs the kernel on the
    host, for tests without a TPU."""
    total, heads, head_dim = core.shape
    rows, width = total // cores, heads * head_dim
    blocks = rows // chunk
    assert rows % chunk == 0 and width % 128 == 0

    def kernel(core_hbm, out_ref, staging, *refs):
        scratch, recv = refs[:-1], refs[-1]
        me = lax.axis_index(axis)
        enter(me, cores)
        args = (me, cores, chunk)

        @pl.loop(0, blocks)
        def _(block):
            send_rows_home(*args, core_hbm, out_ref, staging, scratch, recv, block)

        @pl.loop(0, blocks)
        def _(block):
            wait_rows(*args, out_ref, staging, scratch, recv, block)

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    out, _ = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((rows, cores * width), BF16),
            jax.ShapeDtypeStruct((total, width), BF16),
        ),
        in_specs=[hbm],
        out_specs=(hbm, hbm),
        scratch_shapes=[
            *rows_scratch_shapes(chunk, heads, head_dim),
            pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(collective_id=TO_ROWS_ID),
        interpret=pltpu.InterpretParams() if interpret else False,
        name="qwen38_gdn_to_rows",
    )(core)
    return out


def _block(index, size: int) -> pl.Slice:
    return pl.ds(pl.multiple_of(index * size, size), size)


def enter(me, cores: int) -> None:
    """Every peer is in the kernel (its outputs allocated) before any push;
    the kernel's ``collective_id`` names the barrier semaphore."""
    barrier = pltpu.get_barrier_semaphore()
    for hop in range(1, cores):
        pl.semaphore_signal(
            barrier,
            1,
            device_id=((me + hop) % cores,),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier, cores - 1)
