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
"""Reduce-scatter of a small bf16 block as one Pallas kernel, for the
data-parallel MoE combine.

Input per core: [cores * rows, hidden], block r holding its partial sums for
core r's rows. Output per core: [rows, hidden], the sum of its own block over
the cores, accumulated in float32 and rounded once. The whole input and one
[cores, rows, hidden] scratch sit in VMEM, so the caller bounds the rows.
"""

import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.small_collectives.common import (
    LANES,
    barrier,
    call,
    push,
    split_rank,
    sum_slots,
    wait_bytes_of,
)

# A barrier group distinct from the all-reduce kernel's.
COLLECTIVE_ID = 7
# Each block's rows are padded to this multiple: Mosaic refuses a row slice
# of the input that is not aligned to the sublane tile.
SUBLANES = 8


def reference(blocks, core):
    """[cores, cores * rows, hidden] -> core ``core``'s output."""
    rows = blocks.shape[1] // blocks.shape[0]
    mine = blocks[:, core * rows : (core + 1) * rows, :]
    return jnp.sum(mine.astype(jnp.float32), axis=0).astype(blocks.dtype)


def small_reduce_scatter(x, *, num_cores, axis, rank=None, collective_id=COLLECTIVE_ID):
    """Block r of ``x`` summed onto core r of mesh axis ``axis``, inside a
    shard_map over that axis; ``rank`` as in ``small_all_reduce``."""
    total, hidden = x.shape
    n = num_cores
    if hidden % LANES:
        raise ValueError(f"hidden {hidden} is not a multiple of {LANES} lanes")
    if total % n:
        raise ValueError(f"{total} rows do not split over {n} cores")
    rows = total // n
    padded = -(-rows // SUBLANES) * SUBLANES
    if padded == rows:
        return _reduce_scatter(x, n, axis, rank, collective_id)
    x = jnp.pad(x.reshape(n, rows, hidden), ((0, 0), (0, padded - rows), (0, 0)))
    out = _reduce_scatter(x.reshape(n * padded, hidden), n, axis, rank, collective_id)
    return out[:rows]


def _reduce_scatter(x, n, axis, rank, collective_id):
    total, hidden = x.shape
    rows = total // n
    dict_ids = rank is not None

    def kernel(*refs):
        me, (x_ref, o_ref, slices, send_sem, recv_sem) = split_rank(
            refs, dict_ids, axis
        )
        barrier(n, axis, dict_ids)
        # block r into slot `me` of core r's slices
        for r in range(n):

            @pl.when(jnp.int32(r) != me)
            def _(r=r):
                push(
                    x_ref.at[pl.ds(r * rows, rows)],
                    slices.at[me],
                    send_sem,
                    recv_sem,
                    r,
                    axis,
                    dict_ids,
                )

        # my own block: a static row slice picked by a scalar compare
        for r in range(n):

            @pl.when(jnp.int32(r) == me)
            def _(r=r):
                slices[pl.ds(me, 1)] = x_ref[pl.ds(r * rows, rows), :][None]

        wait_bytes_of(slices.at[pl.ds(0, n - 1)], recv_sem)
        o_ref[...] = sum_slots(slices, n).astype(o_ref.dtype)
        wait_bytes_of(slices.at[pl.ds(0, n - 1)], send_sem)

    scratch = [
        pltpu.VMEM((n, rows, hidden), x.dtype),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
    ]
    return call(
        kernel, x, rows, rank, scratch, collective_id, f"small_reduce_scatter_r{rows}"
    )
