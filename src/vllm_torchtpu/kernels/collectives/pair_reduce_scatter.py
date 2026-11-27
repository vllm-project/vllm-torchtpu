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

Mesh index i owns output rows block i. Each core, per row chunk:

  1. sends its partner (the other core of its chip) the blocks of the
     partner's four same-core devices, die to die, and adds the partner's
     partial to its own for its four same-core blocks (the chip sums);
  2. sends column half A of the x-neighbour chip's and of the diagonal
     chip's chip sums to its x neighbour, half B of the y-neighbour's and the
     diagonal's to its y neighbour;
  3. adds what it received for its own block, and for the far chip's half
     adds its kept chip sum and sends the two-chip sum on (half A to the y
     neighbour, half B to the x neighbour);
  4. sums its own block from the three.

Every cross-chip transfer is one hop, the x and y links carry the same
bytes, and a core sends 3/8 of the tensor across chips (a direct
reduce-scatter: 3/4, a third of it over two hops). Chunks are pipelined:
chunk t's pair stage, chunk t-1's second hop and chunk t-2's sum per step.
The input and the partner's partials stay in HBM; the hops land in VMEM.

`wire` is how the sums cross chips: "bf16", or "int8" with one f32 scale
per row and 128 columns (amax / 127), which halves the cross-chip bytes.
Either way the bits differ from an f32 sum of the 8 partials.
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
# int8 sublane tiling (bf16 needs 16 rows).
_TILE_ROWS = 32
# Rows per pipelined chunk by default (64 where 32 leaves VMEM short).
CHUNK_ROWS = _TILE_ROWS
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


def _barrier(peers) -> None:
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


def _cols(b: int):
    return pl.ds(b * _LANES, _LANES)


def _encode(wire: str, value_of, blocks, q_ref, s_ref) -> None:
    """value_of(b) (f32 [chunk, 128]) for column blocks `blocks` -> the wire
    dtype in q_ref at block i, int8 with block i's row scales in lane i of
    s_ref [chunk, 128]."""
    for i, b in enumerate(blocks):
        v = value_of(b)
        if wire == "bf16":
            q_ref[:, _cols(i)] = v.astype(BF16)
            continue
        amax = jnp.max(jnp.abs(v), axis=1, keepdims=True)
        inv = jnp.where(amax > 0, _INT8_MAX / amax, 0.0)
        q_ref[:, _cols(i)] = jnp.round(v * inv).astype(I8)
        s_ref[:, pl.ds(i, 1)] = amax * (1.0 / _INT8_MAX)


def _decode(wire: str, q_ref, s_ref, i: int):
    v = q_ref[:, _cols(i)].astype(F32)
    if wire == "int8":
        v = v * s_ref[:, pl.ds(i, 1)]
    return v


def _vmem_bytes(block: int, hidden: int, chunk: int, wire: str) -> int:
    tile = chunk * hidden * 2
    item = 1 if wire == "int8" else 2
    scales = _LANES * 4
    # Pair-stage loads (2 slots x 4 relations, x and the partner's), the own
    # block's sum and the kept halves (bf16), the hop sends (2 slots, with
    # scales), every received hop chunk (with scales), the output staging.
    return (
        2 * 4 * tile * 2
        + 2 * block * hidden * 2
        + 2 * 6 * chunk * (hidden // 2 * item + scales)
        + 6 * block * (hidden // 2 * item + scales)
        + 2 * tile
    )


def supported(
    rows: int, hidden: int, n: int, chunk: int = CHUNK_ROWS, wire: str = "int8"
) -> bool:
    """Whether bf16 [rows, hidden] over n ranks fits the kernel: 8 ranks,
    every rank's block whole chunks of whole 32-row tiles, column halves of
    whole lane tiles (at most 128 per half: their int8 scales share a lane
    tile), and its VMEM within the budget."""
    block = rows // N_DEVICES
    return (
        n == N_DEVICES
        and rows % N_DEVICES == 0
        and chunk % _TILE_ROWS == 0
        and block >= chunk
        and block % chunk == 0
        and hidden % (2 * _LANES) == 0
        and hidden // 2 // _LANES <= _LANES
        and _vmem_bytes(block, hidden, chunk, wire) <= _VMEM_BUDGET
    )


def chunk_rows(rows: int, hidden: int, n: int, wire: str = "int8") -> int | None:
    """The pipelined chunk the kernel takes for this shape, else None."""
    for chunk in (CHUNK_ROWS, 2 * CHUNK_ROWS):
        if supported(rows, hidden, n, chunk, wire):
            return chunk
    return None


@functools.cache
def _call(rows: int, hidden: int, chunk: int, wire: str, interpret: bool):
    if wire not in WIRES:
        raise ValueError(f"wire {wire!r} not in {WIRES}")
    block = rows // N_DEVICES
    nch = block // chunk
    nhb = hidden // 2 // _LANES  # column blocks per half
    wdt = I8 if wire == "int8" else BF16
    X, Y, D = 1, 2, 3  # relations
    halves = ((X, Y, 0), (Y, X, nhb))  # (hop-1 peer, hop-2 peer, first block)

    def kernel(rank_ref, tbl, x_hbm, out_hbm, recv1, *refs):
        xb, pb, acc, keep, s1, s1s, s2, s2s, r1, r1s, r2, r2s, ob, *sems = refs
        ld_sem, st_sem, d2d_send, d2d_recv, h1_send, h1_recv, h2_send, h2_recv = sems
        me = rank_ref[0]
        partner = tbl[me, PARTNER]
        group = [tbl[me, GROUP + j] for j in range(_RELATIONS)]
        partner_group = [tbl[partner, GROUP + j] for j in range(_RELATIONS)]
        _barrier((partner, *group[1:]))

        def at(k):
            return pl.ds(pl.multiple_of(k * chunk, chunk), chunk)

        def rows_of(owner, k):
            return pl.ds(pl.multiple_of(owner * block + k * chunk, chunk), chunk)

        def d2d(j, k):
            return pltpu.make_async_remote_copy(
                x_hbm.at[rows_of(partner_group[j], k)],
                recv1.at[j, at(k)],
                d2d_send.at[j, k],
                d2d_recv.at[j, k],
                device_id=(partner,),
                device_id_type=_MESH,
            )

        def hop(n, h, k, slot):
            """Hop n (1, 2) of half h's chunk k, from send slot `slot` into
            the peer's VMEM; hop 1 carries [its own block, its far
            neighbour's], hop 2 the two-chip sum of its own block."""
            dev = group[halves[h][n - 1]]
            if n == 1:
                pairs = [(s1, r1), (s1s, r1s)]
                dst = lambda ref: ref.at[h, :, at(k)]  # noqa: E731
                send, recv = h1_send, h1_recv
            else:
                pairs = [(s2, r2), (s2s, r2s)]
                dst = lambda ref: ref.at[h, at(k)]  # noqa: E731
                send, recv = h2_send, h2_recv
            if wire != "int8":
                pairs = pairs[:1]
            return [
                pltpu.make_async_remote_copy(
                    src.at[slot, h],
                    dst(d),
                    send.at[i, h, slot],
                    recv.at[i, h, k],
                    device_id=(dev,),
                    device_id_type=_MESH,
                )
                for i, (src, d) in enumerate(pairs)
            ]

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

        def chip_sum(slot, j, b):
            return xb[slot, j, :, _cols(b)].astype(F32) + pb[
                slot, j, :, _cols(b)
            ].astype(F32)

        def stage1(k):
            slot = k % 2

            @pl.when(k + 1 < nch)
            def _():
                start_loads(k + 1, 1 - slot)

            for c in loads(k, slot):
                c.wait()
            for b in range(2 * nhb):
                acc[at(k), _cols(b)] = chip_sum(slot, 0, b).astype(BF16)
            for h, (_, second, base) in enumerate(halves):
                for i in range(nhb):
                    keep[h, at(k), _cols(i)] = chip_sum(slot, second, base + i).astype(
                        BF16
                    )

                @pl.when(k >= 2)
                def _():
                    for c in hop(1, h, k - 2, slot):
                        c.wait_send()

                for f, j in enumerate((halves[h][0], D)):
                    _encode(
                        wire,
                        lambda b, j=j: chip_sum(slot, j, b),
                        range(base, base + nhb),
                        s1.at[slot, h, f],
                        s1s.at[slot, h, f],
                    )
                for c in hop(1, h, k, slot):
                    c.start()

        def stage2(k):
            slot = k % 2
            for h, (_, _, base) in enumerate(halves):
                for c in hop(1, h, k, slot):
                    c.wait_recv()

                @pl.when(k >= 2)
                def _():
                    for c in hop(2, h, k - 2, slot):
                        c.wait_send()

                near = (r1.at[h, 0, at(k)], r1s.at[h, 0, at(k)])
                far = (r1.at[h, 1, at(k)], r1s.at[h, 1, at(k)])
                for i in range(nhb):
                    cb = _cols(base + i)
                    acc[at(k), cb] = (
                        acc[at(k), cb].astype(F32) + _decode(wire, *near, i)
                    ).astype(BF16)
                _encode(
                    wire,
                    lambda i, h=h, far=far: keep[h, at(k), _cols(i)].astype(F32)
                    + _decode(wire, *far, i),
                    range(nhb),
                    s2.at[slot, h],
                    s2s.at[slot, h],
                )
                for c in hop(2, h, k, slot):
                    c.start()

        def store(k, slot):
            return pltpu.make_async_copy(
                ob.at[slot], out_hbm.at[at(k)], st_sem.at[slot]
            )

        def stage3(k):
            slot = k % 2
            for h in range(2):
                for c in hop(2, h, k, slot):
                    c.wait_recv()

            @pl.when(k >= 2)
            def _():
                store(k - 2, slot).wait()

            for h, (_, _, base) in enumerate(halves):
                got = (r2.at[h, at(k)], r2s.at[h, at(k)])
                for i in range(nhb):
                    cb = _cols(base + i)
                    ob[slot, :, cb] = (
                        acc[at(k), cb].astype(F32) + _decode(wire, *got, i)
                    ).astype(BF16)
            store(k, slot).start()

        for k in range(nch):
            for j in range(_RELATIONS):
                d2d(j, k).start()
        start_loads(0, 0)

        # Chunk t's pair stage and hop 1, chunk t-1's hop 2, chunk t-2's sum.
        def step(t, carry):
            @pl.when(t < nch)
            def _():
                stage1(t)

            @pl.when((t >= 1) & (t <= nch))
            def _():
                stage2(t - 1)

            @pl.when(t >= 2)
            def _():
                stage3(t - 2)

            return carry

        lax.fori_loop(0, nch + 2, step, 0)

        for k in range(max(nch - 2, 0), nch):
            store(k, k % 2).wait()
            for h in range(2):
                for c in hop(1, h, k, k % 2) + hop(2, h, k, k % 2):
                    c.wait_send()
        for k in range(nch):
            for j in range(_RELATIONS):
                d2d(j, k).wait_send()

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    hh = hidden // 2
    out_shape = (
        jax.ShapeDtypeStruct((block, hidden), BF16),
        jax.ShapeDtypeStruct((_RELATIONS, block, hidden), BF16),  # partner's
    )
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[hbm],
            out_specs=(hbm, hbm),
            scratch_shapes=[
                pltpu.VMEM((2, _RELATIONS, chunk, hidden), BF16),  # x chunks
                pltpu.VMEM((2, _RELATIONS, chunk, hidden), BF16),  # partner's
                pltpu.VMEM((block, hidden), BF16),  # own block's sum
                pltpu.VMEM((2, block, hh), BF16),  # kept halves (A: y, B: x)
                pltpu.VMEM((2, 2, 2, chunk, hh), wdt),  # hop-1 sends
                pltpu.VMEM((2, 2, 2, chunk, _LANES), F32),
                pltpu.VMEM((2, 2, chunk, hh), wdt),  # hop-2 sends
                pltpu.VMEM((2, 2, chunk, _LANES), F32),
                pltpu.VMEM((2, 2, block, hh), wdt),  # hop-1 received
                pltpu.VMEM((2, 2, block, _LANES), F32),
                pltpu.VMEM((2, block, hh), wdt),  # hop-2 received
                pltpu.VMEM((2, block, _LANES), F32),
                pltpu.VMEM((2, chunk, hidden), BF16),  # output staging
                pltpu.SemaphoreType.DMA((2, 2, _RELATIONS)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((_RELATIONS, nch)),
                pltpu.SemaphoreType.DMA((_RELATIONS, nch)),
                pltpu.SemaphoreType.DMA((2, 2, 2)),
                pltpu.SemaphoreType.DMA((2, 2, nch)),
                pltpu.SemaphoreType.DMA((2, 2, 2)),
                pltpu.SemaphoreType.DMA((2, 2, nch)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID,
            vmem_limit_bytes=_vmem_bytes(block, hidden, chunk, wire) + (8 << 20),
        ),
        name=f"pair_reduce_scatter_{wire}-m_{rows}-h_{hidden}",
        interpret=interpret,
    )


def reduce_scatter(
    x: jax.Array,
    rank: jax.Array,
    table,
    chunk: int = CHUNK_ROWS,
    wire: str = "int8",
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
