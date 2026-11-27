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
"""Reduce-scatter of a bf16 [8 * rows, hidden] tensor over the 8 TensorCores
of a tpu7x 2x2x1 (four chips of two cores), chip pairs first.

Mesh index i owns output rows block i. Each core:

  1. sends its partner (the other core of its chip) the blocks of the
     partner's four same-core devices, die to die;
  2. adds the partner's partial to its own for its four same-core blocks (in
     f32), and sends each other chip's block to that chip's same-core device;
  3. sums its own block's chip sum with the three it received (f32, one
     rounding).

A core sends 1/2 of the tensor die to die and 3/8 across chips, against 3/4
across chips for a direct reduce-scatter. Both phases run per row chunk,
every transfer started as soon as its input is ready. The input and the
received partials stay in HBM; VMEM holds a chunk at a time.

`wire` is how the chip sums cross chips: "bf16" (rounded once), or "int8"
with one f32 scale per row and 128 columns (amax / 127), which halves the
cross-chip bytes. Either way the bits differ from an f32 sum of the 8
partials.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BF16, F32, I8 = jnp.bfloat16, jnp.float32, jnp.int8
COLLECTIVE_ID = 0
N_DEVICES = 8
WIRES = ("bf16", "int8")
_MESH = pl.DeviceIdType.MESH
_LANES = 128
_INT8_MAX = 127.0
# Rows per pipelined chunk; a chunk of 6144 bf16 columns is 768 KiB.
CHUNK_ROWS = 64
# int8 sublane tiling (bf16 needs 16 rows).
_TILE_ROWS = 32
# VMEM a call may plan (its buffers and the own block's f32 chip sum), below
# tpu7x's 64 MiB per TensorCore with room for Mosaic's own.
_VMEM_BUDGET = 48 << 20
# Columns of `topology_table`: the die-to-die partner, then the four devices
# of this core's index, by relation (self, x neighbour chip, y neighbour chip,
# diagonal chip). The relations are symmetric, so a device's slot for a peer
# is the peer's slot for it.
PARTNER, GROUP = 0, 1
_RELATIONS = 4


def topology_table(device_ids) -> np.ndarray:
    """int32 [8, 5] per mesh index (`device_ids` in mesh order). A tpu7x 2x2x1
    numbers its TensorCores 2 * chip + core (pair_all_reduce.topology_table).
    Raises ValueError unless the ids are 0..7."""
    ids = [int(d) for d in device_ids]
    if sorted(ids) != list(range(N_DEVICES)):
        raise ValueError(f"not the 8 TensorCores of a 2x2x1: device ids {ids}")
    index = {d: i for i, d in enumerate(ids)}
    table = np.zeros((N_DEVICES, 1 + _RELATIONS), np.int32)
    for i, d in enumerate(ids):
        table[i] = (index[d ^ 1], i, index[d ^ 2], index[d ^ 4], index[d ^ 6])
    return table


def supported(rows: int, hidden: int, n: int, chunk: int = CHUNK_ROWS) -> bool:
    """Whether bf16 [rows, hidden] over n ranks fits the kernel: 8 ranks,
    every rank's block whole chunks of whole 32-row tiles, lane-aligned
    columns, and its VMEM within the budget."""
    block = rows // N_DEVICES
    return (
        n == N_DEVICES
        and rows % N_DEVICES == 0
        and chunk % _TILE_ROWS == 0
        and block >= chunk
        and block % chunk == 0
        and hidden % _LANES == 0
        and _vmem_bytes(block, hidden, chunk) <= _VMEM_BUDGET
    )


def chunk_rows(rows: int, hidden: int, n: int) -> int | None:
    """The largest pipelined chunk the kernel takes for this shape, else None."""
    for chunk in (CHUNK_ROWS, _TILE_ROWS):
        if supported(rows, hidden, n, chunk):
            return chunk
    return None


def _vmem_bytes(block: int, hidden: int, chunk: int) -> int:
    tile = chunk * hidden * 2
    scales = chunk * _LANES * 4
    # Loads (x and the partner's partial, 2 slots x 4 relations), sends and
    # received chunks (2 slots x 3 peers, with scales), the output staging,
    # and the own block's f32 chip sum.
    return (
        2 * 4 * tile * 2 + 2 * 2 * 3 * (tile + scales) + 2 * tile + block * hidden * 4
    )


@functools.cache
def _call(rows: int, hidden: int, chunk: int, wire: str, interpret: bool):
    if wire not in WIRES:
        raise ValueError(f"wire {wire!r} not in {WIRES}")
    block = rows // N_DEVICES
    nch = block // chunk
    ncb = hidden // _LANES
    wdt = I8 if wire == "int8" else BF16
    peers_n = _RELATIONS - 1

    def kernel(rank_ref, tbl, x_hbm, out_hbm, recv1, recv2, *refs):
        if wire == "int8":
            recv2s, *refs = refs
        xb, pb, sb, ss, rb, rs, ob, acc, *sems = refs
        ld_sem, st_sem, d2d_send, d2d_recv, s2_send, s2_recv = sems
        me = rank_ref[0]
        partner = tbl[me, PARTNER]
        group = [tbl[me, GROUP + j] for j in range(_RELATIONS)]
        partner_group = [tbl[partner, GROUP + j] for j in range(_RELATIONS)]
        peers = (partner, *group[1:])

        barrier = pltpu.get_barrier_semaphore()
        for p in peers:
            pl.semaphore_signal(barrier, inc=1, device_id=(p,), device_id_type=_MESH)
        pl.semaphore_wait(barrier, len(peers))

        # A second round: a peer that already left this call and entered the
        # next cannot have its next-call signal counted in this call's wait.
        @functools.partial(pl.run_scoped, ready=pltpu.SemaphoreType.REGULAR)
        def _(ready):
            for p in peers:
                pl.semaphore_signal(ready, inc=1, device_id=(p,), device_id_type=_MESH)
            pl.semaphore_wait(ready, len(peers))

        def at(k):
            return pl.ds(pl.multiple_of(k * chunk, chunk), chunk)

        def rows_of(owner, k):
            return pl.ds(pl.multiple_of(owner * block + k * chunk, chunk), chunk)

        def cols(b):
            return pl.ds(pl.multiple_of(b * _LANES, _LANES), _LANES)

        def d2d(j, k):
            return pltpu.make_async_remote_copy(
                x_hbm.at[rows_of(partner_group[j], k)],
                recv1.at[j, at(k)],
                d2d_send.at[j, k],
                d2d_recv.at[j, k],
                device_id=(partner,),
                device_id_type=_MESH,
            )

        def cross(j, k, slot):
            """The data (and, int8, scale) copies of chunk k to relation j."""
            copies = [
                pltpu.make_async_remote_copy(
                    sb.at[slot, j - 1],
                    recv2.at[j - 1, at(k)],
                    s2_send.at[0, j - 1, slot],
                    s2_recv.at[0, j, k],
                    device_id=(group[j],),
                    device_id_type=_MESH,
                )
            ]
            if wire == "int8":
                copies.append(
                    pltpu.make_async_remote_copy(
                        ss.at[slot, j - 1],
                        recv2s.at[j - 1, at(k)],
                        s2_send.at[1, j - 1, slot],
                        s2_recv.at[1, j, k],
                        device_id=(group[j],),
                        device_id_type=_MESH,
                    )
                )
            return copies

        def loads(k, slot):
            copies = []
            for j in range(_RELATIONS):
                copies.append(
                    pltpu.make_async_copy(
                        x_hbm.at[rows_of(group[j], k)],
                        xb.at[slot, j],
                        ld_sem.at[0, slot, j],
                    )
                )
                copies.append(
                    pltpu.make_async_copy(
                        recv1.at[j, at(k)], pb.at[slot, j], ld_sem.at[1, slot, j]
                    )
                )
            return copies

        def start_loads(k, slot):
            for j in range(_RELATIONS):
                d2d(j, k).wait_recv()
            for c in loads(k, slot):
                c.start()

        lane = lax.broadcasted_iota(jnp.int32, (chunk, _LANES), 1)

        def chip_sum(slot, j, b):
            return xb[slot, j, :, cols(b)].astype(F32) + pb[slot, j, :, cols(b)].astype(
                F32
            )

        def encode(slot, j, k):
            """Chip sums of relation j's chunk k into its send slot."""
            if wire == "bf16":

                def body(b, carry):
                    sb[slot, j - 1, :, cols(b)] = chip_sum(slot, j, b).astype(BF16)
                    return carry

                lax.fori_loop(0, ncb, body, 0)
                return

            def body(b, scales):
                s = chip_sum(slot, j, b)
                amax = jnp.max(jnp.abs(s), axis=1, keepdims=True)
                inv = jnp.where(amax > 0, _INT8_MAX / amax, 0.0)
                sb[slot, j - 1, :, cols(b)] = jnp.round(s * inv).astype(I8)
                return jnp.where(lane == b, amax * (1.0 / _INT8_MAX), scales)

            ss[slot, j - 1] = lax.fori_loop(
                0, ncb, body, jnp.zeros((chunk, _LANES), F32)
            )

        # Phase 1: every die-to-die transfer, in chunk order.
        for k in range(nch):
            for j in range(_RELATIONS):
                d2d(j, k).start()
        start_loads(0, 0)

        def chip_sums(k, carry):
            slot = k % 2

            @pl.when(k + 1 < nch)
            def _():
                start_loads(k + 1, 1 - slot)

            for c in loads(k, slot):
                c.wait()

            def own(b, carry):
                acc[at(k), cols(b)] = chip_sum(slot, 0, b)
                return carry

            lax.fori_loop(0, ncb, own, 0)
            for j in range(1, _RELATIONS):

                @pl.when(k >= 2)
                def _():
                    for c in cross(j, k - 2, slot):
                        c.wait_send()

                encode(slot, j, k)
                for c in cross(j, k, slot):
                    c.start()
            return carry

        lax.fori_loop(0, nch, chip_sums, 0)

        # Phase 2: own block = own chip sum + the three received.
        def recv_loads(k, slot):
            copies = []
            for j in range(1, _RELATIONS):
                copies.append(
                    pltpu.make_async_copy(
                        recv2.at[j - 1, at(k)],
                        rb.at[slot, j - 1],
                        ld_sem.at[1, slot, j],
                    )
                )
                if wire == "int8":
                    copies.append(
                        pltpu.make_async_copy(
                            recv2s.at[j - 1, at(k)],
                            rs.at[slot, j - 1],
                            ld_sem.at[0, slot, j],
                        )
                    )
            return copies

        def start_recv_loads(k, slot):
            for j in range(1, _RELATIONS):
                for c in cross(j, k, slot):
                    c.wait_recv()
            for c in recv_loads(k, slot):
                c.start()

        def store(k, slot):
            return pltpu.make_async_copy(
                ob.at[slot], out_hbm.at[at(k)], st_sem.at[slot]
            )

        start_recv_loads(0, 0)

        def finals(k, carry):
            slot = k % 2

            @pl.when(k + 1 < nch)
            def _():
                start_recv_loads(k + 1, 1 - slot)

            for c in recv_loads(k, slot):
                c.wait()

            @pl.when(k >= 2)
            def _():
                store(k - 2, slot).wait()

            def body(b, carry):
                total = acc[at(k), cols(b)]
                for j in range(peers_n):
                    part = rb[slot, j, :, cols(b)].astype(F32)
                    if wire == "int8":
                        scales = jnp.where(lane == b, rs[slot, j], 0.0)
                        part = part * jnp.max(scales, axis=1, keepdims=True)
                    total = total + part
                ob[slot, :, cols(b)] = total.astype(BF16)
                return carry

            lax.fori_loop(0, ncb, body, 0)
            store(k, slot).start()
            return carry

        lax.fori_loop(0, nch, finals, 0)

        for k in range(max(nch - 2, 0), nch):
            store(k, k % 2).wait()
            for j in range(1, _RELATIONS):
                for c in cross(j, k, k % 2):
                    c.wait_send()
        for k in range(nch):
            for j in range(_RELATIONS):
                d2d(j, k).wait_send()

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem = _vmem_bytes(block, hidden, chunk)
    # The output, then the received partials: the partner's, the peers', and
    # with int8 their scales.
    out_shape = (
        jax.ShapeDtypeStruct((block, hidden), BF16),
        jax.ShapeDtypeStruct((_RELATIONS, block, hidden), BF16),
        jax.ShapeDtypeStruct((peers_n, block, hidden), wdt),
    )
    if wire == "int8":
        out_shape += (jax.ShapeDtypeStruct((peers_n, block, _LANES), F32),)
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[hbm],
            out_specs=(hbm,) * len(out_shape),
            scratch_shapes=[
                pltpu.VMEM((2, _RELATIONS, chunk, hidden), BF16),  # x chunks
                pltpu.VMEM((2, _RELATIONS, chunk, hidden), BF16),  # partner's
                pltpu.VMEM((2, peers_n, chunk, hidden), wdt),  # sends
                pltpu.VMEM((2, peers_n, chunk, _LANES), F32),  # send scales
                pltpu.VMEM((2, peers_n, chunk, hidden), wdt),  # received
                pltpu.VMEM((2, peers_n, chunk, _LANES), F32),  # received scales
                pltpu.VMEM((2, chunk, hidden), BF16),  # output staging
                pltpu.VMEM((block, hidden), F32),  # own block's chip sum
                pltpu.SemaphoreType.DMA((2, 2, _RELATIONS)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((_RELATIONS, nch)),
                pltpu.SemaphoreType.DMA((_RELATIONS, nch)),
                pltpu.SemaphoreType.DMA((2, peers_n, 2)),
                pltpu.SemaphoreType.DMA((2, _RELATIONS, nch)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID, vmem_limit_bytes=vmem + (8 << 20)
        ),
        name=f"pair_reduce_scatter_{wire}-m_{rows}-h_{hidden}",
        interpret=interpret,
    )


def reduce_scatter(
    x: jax.Array,
    rank: jax.Array,
    table,
    chunk: int = CHUNK_ROWS,
    wire: str = "bf16",
    interpret=False,
) -> jax.Array:
    """Reduce-scatter of bf16 x [rows, hidden] (`supported`) over the 8 ranks
    of the 1-D mesh this runs under (shard_map): mesh index i gets the sum of
    rows block i, [rows / 8, hidden]. rank: this device's mesh index, int32;
    table: `topology_table` of the mesh's devices; wire: see the module."""
    rows, hidden = x.shape
    out = _call(rows, hidden, chunk, wire, interpret)(
        jnp.reshape(rank, (1,)).astype(jnp.int32), jnp.asarray(table, jnp.int32), x
    )
    return out[0]
