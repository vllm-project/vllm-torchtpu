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
"""All-gather of a [r, ...] array of any dtype over the 16 cores of a 2x2x2
slice of 2-core chips, as rotated halving-doubling by remote DMAs.

The schedule is rotated_collectives.all_gather's: three chip phases, each
sending every block held so far to the axis neighbour, each block forwarded to
the chip-mate as it lands, and three parts at once, each taking the chip axes
in another order. Here the parts split each core's block along its leading row
dimension instead of its columns, so the trailing dimensions may have any shape
and tiling: the sequence-parallel prefill gathers the fp8 MoE input rows
x_q [r, h/128, 128] with it (TPU_MOE_PREFILL_ROTATED_COLLECTIVES).

Row block m of the [16 r, ...] output is the rows of the core at mesh index m,
bitwise ``lax.all_gather(x, axis, axis=0, tiled=True)``. The peers come from
``rotated_collectives.peer_table(places, "all_gather")``.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental import rotated_collectives as rc

# Barrier semaphore group; the other collective kernels use 0, 5, 7, 9, 11, 13
# and 15.
COLLECTIVE_ID = 17
PARTS = len(rc.ROTATIONS)


def row_align(shape, dtype):
    """Rows each part holds a multiple of: 1 when the rows are an untiled
    leading dimension (3 or more dims), else one sublane tile of `dtype`."""
    if len(shape) >= 3:
        return 1
    return 8 * max(1, 4 // jnp.dtype(dtype).itemsize)


def part_rows(rows, align=1, parts=PARTS):
    """Row counts of the parts of a `rows`-row block: multiples of `align`, as
    even as they go (e.g. 86, 85, 85 of 256)."""
    units = rows // align
    return [align * (units // parts + (j < units % parts)) for j in range(parts)]


def supports(shape, dtype):
    """Whether all_gather takes a per-core block of `shape` and `dtype`: at
    least two dims and every part a non-empty whole number of row tiles."""
    if len(shape) < 2:
        return False
    align = row_align(shape, dtype)
    return shape[0] % align == 0 and shape[0] >= PARTS * align


def all_gather(x, rank, table, *, axis, dict_ids=True, interpret=None, name=None):
    """x [r, ...] (this core's rows) -> [16 r, ...] of x's dtype, row block m
    the rows of mesh index m. rank int32 [1] (this core's mesh index), table
    int32 [CORES * TABLE_WIDTH] (rotated_collectives.peer_table "all_gather")."""
    r = x.shape[0]
    if not supports(x.shape, x.dtype):
        raise ValueError(f"row all_gather does not take {x.dtype} {x.shape}")
    counts = part_rows(r, row_align(x.shape, x.dtype))
    starts = [sum(counts[:j]) for j in range(PARTS)]

    def blk(ref, block, j):
        return ref.at[pl.ds(block * r + starts[j], counts[j])]

    def kernel(rank_ref, t_ref, x_ref, o_ref, stage, lsem, send, recv, msend, mrecv):
        me = rank_ref[0]
        base = me * rc.TABLE_WIDTH

        def at(column):
            return t_ref[base + column]

        mate = at(rc._MATE)
        peers = [at(rc._PARTNER + j * 3 + k) for j in range(PARTS) for k in range(3)]
        rc._barrier([mate] + peers[:3], axis, dict_ids)

        local_in = pltpu.make_async_copy(x_ref, stage, lsem)
        local_in.start()

        def remote(src, dst, s, rv, p):
            pltpu.make_async_remote_copy(
                src,
                dst,
                s,
                rv,
                device_id=rc._peer(p, axis, dict_ids),
                device_id_type=rc.MESH,
            ).start()

        def own(j):
            return x_ref.at[pl.ds(starts[j], counts[j])]

        for j in range(PARTS):
            remote(own(j), blk(o_ref, me, j), msend, mrecv, mate)
        for k in range(4):
            for j in range(PARTS):
                if k > 0:
                    for i in range(2 ** (k - 1)):
                        n = j * 7 + 2 ** (k - 1) - 1 + i
                        block = at(rc._AG_RECV + n)
                        land = blk(o_ref, block, j)
                        rc._wait(land, recv.at[n])
                        remote(land, land, msend, mrecv, mate)
                if k < 3:
                    partner = peers[j * 3 + k]
                    for i in range(2**k):
                        n = j * 7 + 2**k - 1 + i
                        block = at(rc._AG_SEND + n)
                        src = own(j) if i == 0 else blk(o_ref, block, j)
                        remote(src, blk(o_ref, block, j), send, recv.at[n], partner)
        local_in.wait()
        local_out = pltpu.make_async_copy(stage, o_ref.at[pl.ds(me * r, r)], lsem)
        local_out.start()
        rc._wait(o_ref.at[pl.ds(0, 7 * r)], send)
        rc._wait(o_ref.at[pl.ds(0, 8 * r)], msend)
        rc._wait(o_ref.at[pl.ds(0, 8 * r)], mrecv)
        local_out.wait()

    kw = {"interpret": interpret} if interpret is not None else {}
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rc.CORES * r, *x.shape[1:]), x.dtype),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        scratch_shapes=[
            pltpu.VMEM(x.shape, x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((7 * PARTS,)),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID, vmem_limit_bytes=32 << 20
        ),
        name=name or f"rotated_row_all_gather_r{r}",
        **kw,
    )(rank.reshape(1).astype(jnp.int32), table, x)
