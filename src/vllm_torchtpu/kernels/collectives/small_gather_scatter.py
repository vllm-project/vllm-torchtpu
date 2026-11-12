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
"""Small Pallas reduce-scatter for the data-parallel MoE combine.

Every core pushes the partial sums for each rank's rows to that rank's
mesh core. Each destination sums its arrivals in float32 and rounds once
to the input dtype. The caller bounds the row count to fit VMEM.

Input per core: [cores * rows, hidden], with blocks in rank order.
Output per core: [rows, hidden], the sum for that core's rank.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.collectives.small_all_reduce import (
    _barrier,
    _push,
    _split,
    _wait_bytes_of,
)

LANE = 128  # the lane tile: a block's width must sit on it
# Distinct barrier groups from the all-reduce kernel's (5).
SCATTER_COLLECTIVE_ID = 7


def _check(hidden):
    if hidden % LANE:
        raise ValueError(f"hidden {hidden} is not a multiple of {LANE} lanes")


def _call(kernel, x, out_rows, hidden, rank, scratch, collective_id, name):
    rank_specs = [pl.BlockSpec(memory_space=pltpu.SMEM)] if rank is not None else []
    args = (() if rank is None else (rank.astype(jnp.int32),)) + (x,)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((out_rows, hidden), x.dtype),
        in_specs=rank_specs + [pl.BlockSpec()],
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=name,
    )(*args)


def small_reduce_scatter(
    x,
    *,
    num_cores,
    axis="d",
    rank=None,
    my_rank=None,
    core_of_rank=None,
    dict_ids=None,
    collective_id=SCATTER_COLLECTIVE_ID,
):
    """[num_cores * rows, hidden] on each core (block r = its partial sums
    for RANK r's rows, the order the gather placed them in) -> [rows,
    hidden]: the float32 sum over the cores of this core's own rank's block,
    rounded once to x's type; inside a shard_map.

    The kernel addresses cores by mesh index while the blocks are in rank
    order. The mesh core order may differ from the rank order. `core_of_rank`
    [num_cores] int32 (SMEM) is the mesh core that serves each rank and
    `my_rank` [1] int32 (SMEM) this core's rank; both None means the
    identity (mesh order is rank order). Block r goes to core
    core_of_rank[r]; every core sums the blocks it receives for its rank."""
    total, hidden = x.shape
    _check(hidden)
    n = num_cores
    if total % n:
        raise ValueError(f"{total} rows do not split over {n} cores")
    rows = total // n
    if dict_ids is None:
        dict_ids = rank is not None
    if (my_rank is None) != (core_of_rank is None):
        raise ValueError("my_rank and core_of_rank come together")
    permuted = core_of_rank is not None

    if rows % 8:
        # DMA slices need whole row tiles. Pad each rank's block separately
        # so the padding does not move rows between destination ranks.
        padded_rows = pl.cdiv(rows, 8) * 8
        x = jnp.pad(
            x.reshape(n, rows, hidden), ((0, 0), (0, padded_rows - rows), (0, 0))
        )
        return small_reduce_scatter(
            x.reshape(n * padded_rows, hidden),
            num_cores=n,
            axis=axis,
            rank=rank,
            my_rank=my_rank,
            core_of_rank=core_of_rank,
            dict_ids=dict_ids,
            collective_id=collective_id,
        )[:rows]

    def kernel(*refs):
        me, refs = _split(refs, rank is not None, axis)
        if permuted:
            my_rank_ref, core_ref, *refs = refs
            mine = my_rank_ref[0]
        else:
            mine = me
        x_ref, o_ref, slices, send_sem, recv_sem = refs
        _barrier(n, axis, dict_ids)
        # block r (rank r's rows) into slot `me` of the slices of the core
        # that serves rank r
        for r in range(n):

            @pl.when(jnp.int32(r) != mine)
            def _(r=r):
                dest = core_ref[r] if permuted else jnp.int32(r)
                _push(
                    x_ref.at[pl.ds(r * rows, rows)],
                    slices.at[me],
                    send_sem,
                    recv_sem,
                    dest,
                    axis,
                    dict_ids,
                )

        # my own rank's block: a static row slice picked by a scalar compare
        for r in range(n):

            @pl.when(jnp.int32(r) == mine)
            def _(r=r):
                slices[pl.ds(me, 1)] = x_ref[pl.ds(r * rows, rows), :][None]

        _wait_bytes_of(slices.at[pl.ds(0, n - 1)], recv_sem)
        acc = slices[0].astype(jnp.float32)
        for j in range(1, n):
            acc = acc + slices[j].astype(jnp.float32)
        o_ref[...] = acc.astype(o_ref.dtype)
        _wait_bytes_of(slices.at[pl.ds(0, n - 1)], send_sem)

    scratch = [
        pltpu.VMEM((n, rows, hidden), x.dtype),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
    ]
    if not permuted:
        return _call(
            kernel,
            x,
            rows,
            hidden,
            rank,
            scratch,
            collective_id,
            f"small_reduce_scatter_r{rows}",
        )
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    rank_specs = [smem] if rank is not None else []
    args = (() if rank is None else (rank.astype(jnp.int32),)) + (
        my_rank.astype(jnp.int32),
        core_of_rank.astype(jnp.int32),
        x,
    )
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, hidden), x.dtype),
        in_specs=rank_specs + [smem, smem, pl.BlockSpec()],
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=f"small_reduce_scatter_r{rows}_permuted",
    )(*args)


def reference_scatter(blocks, core):
    """[cores, cores * rows, hidden] -> core `core`'s output: the float32 sum
    over the cores of block `core`, rounded once."""
    rows = blocks.shape[1] // blocks.shape[0]
    mine = blocks[:, core * rows : (core + 1) * rows, :]
    return jnp.sum(mine.astype(jnp.float32), axis=0).astype(blocks.dtype)
