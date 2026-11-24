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
"""Bandwidth-bound all-reduce of a bf16 [rows, hidden] tensor over the 8
TensorCores of a tpu7x 2x2x1 (four chips of two cores), chip pairs first.

Per row block (the rows split into up to 4 blocks, every transfer started as
soon as its input is ready, so a block's cross-chip traffic overlaps the next
block's die-to-die traffic):

  1. die-to-die reduce-scatter of column halves: core c keeps half c of its
     chip's partial sum (rounded to bf16);
  2. among the four same-core devices (one per chip), a direct reduce-scatter
     of that half's quarters, summed in f32 in chip order and rounded once;
  3. the same four exchange the reduced quarters (all-gather), then
  4. the die-to-die all-gather of the halves.

Each core sends 3/8 of the tensor across chips per phase (the ring or direct
all-reduce: 3/4), which is what bounds a decode all-reduce of 64-256 rows: a
core's cross-chip DMAs run at ~70-90 GB/s against ~400 GB/s die to die (link
probes j-158a8ecb, j-6348ed08). Chained, it all-reduces bf16[192, 6144] in
33.2 us against 41.0 us for XLA's TensorCore all-reduce, [96, 6144] in 19.2
against 25.9 (j-6348ed08); at 384 rows XLA's SparseCore all-reduce is faster.

The chip partials are rounded to bf16 before the cross-chip sum, so the bits
differ from an f32 sum of the 8 partials; every rank gets the same bits.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BF16, F32 = jnp.bfloat16, jnp.float32
COLLECTIVE_ID = 0
N_DEVICES = 8
_MESH = pl.DeviceIdType.MESH
_TILE_ROWS = 16  # bf16 sublane tiling
_LANES = 128
_MAX_BLOCKS = 4
# x, the output and the scratch (~1.5 x the tensor) live in VMEM; compiled
# and timed up to 256 rows of 6144.
MAX_BYTES = 3 << 20
# Columns of `topology_table`.
PARTNER, X_PEER, Y_PEER, DIAG_PEER, CORE, X_BIT, Y_BIT = range(7)


def topology_table(device_ids) -> np.ndarray:
    """int32 [8, 7] per mesh index (`device_ids` in mesh order): the die-to-die
    partner, the same-core devices of the x, y and diagonal neighbour chips,
    the core index and the chip's x and y bits. A tpu7x 2x2x1 numbers its
    TensorCores 2 * chip + core with chip = x + 2 y (device coords and
    core_on_chip read back on a 2x2x1, j-03c8968b). Raises ValueError unless
    the ids are 0..7. Any id layout gives the right sums (the relations are
    symmetric); the layout only decides which hops cross chips."""
    ids = [int(d) for d in device_ids]
    if sorted(ids) != list(range(N_DEVICES)):
        raise ValueError(f"not the 8 TensorCores of a 2x2x1: device ids {ids}")
    index = {d: i for i, d in enumerate(ids)}
    table = np.zeros((N_DEVICES, 7), np.int32)
    for i, d in enumerate(ids):
        chip = d // 2
        table[i] = (
            index[d ^ 1],
            index[d ^ 2],
            index[d ^ 4],
            index[d ^ 6],
            d % 2,
            chip % 2,
            chip // 2,
        )
    return table


def _blocks(rows: int) -> int:
    if rows <= 0:
        return 0
    for blocks in (_MAX_BLOCKS, 2, 1):
        if rows % (blocks * _TILE_ROWS) == 0:
            return blocks
    return 0


def supported(rows: int, hidden: int, n: int) -> bool:
    """Whether bf16 [rows, hidden] over n ranks fits the kernel: 8 ranks,
    whole 16-row tiles, lane-aligned column eighths, at most MAX_BYTES."""
    return (
        n == N_DEVICES
        and _blocks(rows) > 0
        and hidden % (N_DEVICES * _LANES) == 0
        and rows * hidden * jnp.dtype(BF16).itemsize <= MAX_BYTES
    )


@functools.cache
def _call(rows: int, hidden: int, interpret: bool):
    blocks = _blocks(rows)
    br, h2, h8 = rows // blocks, hidden // 2, hidden // 8

    def kernel(rank_ref, tbl, x_ref, out_ref, r1, b1, rq, *sems):
        d2d_s, d2d_r, rs_s, rs_r, ag_s, ag_r, back_s, back_r = sems
        me = rank_ref[0]
        partner = tbl[me, PARTNER]
        core = tbl[me, CORE]
        bx, by = tbl[me, X_BIT], tbl[me, Y_BIT]
        chip = bx * 2 + by
        # (device, its chip index): the same-core devices of the other chips.
        peers = [
            (tbl[me, X_PEER], (1 - bx) * 2 + by),
            (tbl[me, Y_PEER], bx * 2 + (1 - by)),
            (tbl[me, DIAG_PEER], (1 - bx) * 2 + (1 - by)),
        ]
        barrier = pltpu.get_barrier_semaphore()
        for p in (partner, *(p for p, _ in peers)):
            pl.semaphore_signal(barrier, inc=1, device_id=(p,), device_id_type=_MESH)
        pl.semaphore_wait(barrier, 4)

        def cols(ref, b, start, size):
            return ref.at[pl.ds(b * br, br), pl.ds(pl.multiple_of(start, _LANES), size)]

        def remote(src, dst, send, recv, peer):
            copy = pltpu.make_async_remote_copy(
                src, dst, send, recv, device_id=(peer,), device_id_type=_MESH
            )
            copy.start()
            return copy

        d2d = [
            remote(
                cols(x_ref, b, (1 - core) * h2, h2),
                r1.at[b],
                d2d_s.at[b],
                d2d_r.at[b],
                partner,
            )
            for b in range(blocks)
        ]
        sends = []
        for b in range(blocks):
            d2d[b].wait_recv()
            b1[b] = (
                cols(x_ref, b, core * h2, h2)[...].astype(F32) + r1[b].astype(F32)
            ).astype(BF16)
            for k, (p, cp) in enumerate(peers):
                sends.append(
                    remote(
                        b1.at[b, :, pl.ds(pl.multiple_of(cp * h8, _LANES), h8)],
                        rq.at[chip, b],
                        rs_s.at[k, b],
                        rs_r.at[chip, b],
                        p,
                    )
                )
            rq[chip, b] = b1[b, :, pl.ds(pl.multiple_of(chip * h8, _LANES), h8)]
        own = core * h2 + chip * h8
        for b in range(blocks):
            for _, cp in peers:
                slot = rq.at[cp, b]
                pltpu.make_async_copy(slot, slot, rs_r.at[cp, b]).wait_recv()
            acc = rq[0, b].astype(F32)
            for q in range(1, 4):
                acc = acc + rq[q, b].astype(F32)
            mine = cols(out_ref, b, own, h8)
            mine[...] = acc.astype(BF16)
            for k, (p, _) in enumerate(peers):
                sends.append(remote(mine, mine, ag_s.at[k, b], ag_r.at[chip, b], p))
        back = []
        for b in range(blocks):
            for _, cp in peers:
                theirs = cols(out_ref, b, core * h2 + cp * h8, h8)
                pltpu.make_async_copy(theirs, theirs, ag_r.at[cp, b]).wait_recv()
            half = cols(out_ref, b, core * h2, h2)
            back.append(remote(half, half, back_s.at[b], back_r.at[b], partner))
        for copy in back:
            copy.wait_recv()
        for copy in d2d + sends + back:
            copy.wait_send()

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    view = rows * hidden * jnp.dtype(BF16).itemsize
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, hidden), BF16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[vmem],
            out_specs=vmem,
            scratch_shapes=[
                pltpu.VMEM((blocks, br, h2), BF16),  # the partner's half
                pltpu.VMEM((blocks, br, h2), BF16),  # the chip's sum of my half
                pltpu.VMEM((4, blocks, br, h8), BF16),  # chips' quarters, by chip
                pltpu.SemaphoreType.DMA((blocks,)),
                pltpu.SemaphoreType.DMA((blocks,)),
                pltpu.SemaphoreType.DMA((3, blocks)),
                pltpu.SemaphoreType.DMA((4, blocks)),
                pltpu.SemaphoreType.DMA((3, blocks)),
                pltpu.SemaphoreType.DMA((4, blocks)),
                pltpu.SemaphoreType.DMA((blocks,)),
                pltpu.SemaphoreType.DMA((blocks,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID, vmem_limit_bytes=4 * view + (2 << 20)
        ),
        name=f"pair_all_reduce-m_{rows}-h_{hidden}",
        interpret=interpret,
    )


def all_reduce(x: jax.Array, rank: jax.Array, table, interpret=False) -> jax.Array:
    """All-reduce of bf16 x [rows, hidden] (`supported`) over the 8 ranks of
    the 1-D mesh this runs under (shard_map); rank: this device's mesh index,
    int32; table: `topology_table` of the mesh's devices."""
    rows, hidden = x.shape
    return _call(rows, hidden, interpret)(
        jnp.reshape(rank, (1,)).astype(jnp.int32), jnp.asarray(table, jnp.int32), x
    )
