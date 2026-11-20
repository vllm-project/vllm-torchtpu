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
"""All-reduce of a small bf16 [rows, hidden] block as one Pallas kernel.

    one-shot   Every core pushes its whole block to every other core and sums
               the arrivals locally.
    two-shot   Core j owns columns [j * own, (j + 1) * own). Every core pushes
               each owner its columns; the owner sums them and pushes the sum
               back to everyone.

Both forms start with a barrier so remote scratch is ready, sum in float32 and
round once to the input dtype.
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

# One-shot up to this many bytes per core, two-shot above: one-shot sends more
# bytes in one round, two-shot fewer in two; they cross at ~5.9 rows of H=4096
# bf16 on 8 cores (j-de6349e8).
ONE_SHOT_MAX_BYTES = 32 * 1024
# Barrier semaphore group. Every kernel compiled with the same id in one
# program shares it; the fused EP MoE kernels use 0.
COLLECTIVE_ID = 5


def form_for(rows, hidden, itemsize):
    """The form for a block of rows x hidden elements of itemsize bytes."""
    return "oneshot" if rows * hidden * itemsize <= ONE_SHOT_MAX_BYTES else "twoshot"


def reference(blocks):
    """The float32 sum over the cores' blocks [cores, rows, hidden], rounded
    once to the blocks' dtype."""
    return jnp.sum(blocks.astype(jnp.float32), axis=0).astype(blocks.dtype)


def _oneshot(x, n, axis, rank, collective_id):
    rows, hidden = x.shape
    dict_ids = rank is not None

    def kernel(*refs):
        me, (x_ref, o_ref, arrivals, send_sem, recv_sem) = split_rank(
            refs, dict_ids, axis
        )
        barrier(n, axis, dict_ids)
        for j in range(n):  # my block into slot `me` of every other core

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                push(x_ref, arrivals.at[me], send_sem, recv_sem, j, axis, dict_ids)

        arrivals[pl.ds(me, 1)] = x_ref[...][None]
        wait_bytes_of(arrivals.at[pl.ds(0, n - 1)], recv_sem)
        o_ref[...] = sum_slots(arrivals, n).astype(o_ref.dtype)
        wait_bytes_of(arrivals.at[pl.ds(0, n - 1)], send_sem)

    scratch = [
        pltpu.VMEM((n, rows, hidden), x.dtype),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
    ]
    return call(
        kernel,
        x,
        rows,
        rank,
        scratch,
        collective_id,
        f"small_all_reduce_oneshot_r{rows}",
    )


def _twoshot(x, n, axis, rank, collective_id):
    rows, hidden = x.shape
    own = hidden // n
    dict_ids = rank is not None

    def kernel(*refs):
        me, refs = split_rank(refs, dict_ids, axis)
        x_ref, o_ref, mine, slices, gathered, send1, recv1, send2, recv2 = refs
        barrier(n, axis, dict_ids)
        for j in range(n):  # slot j: the columns core j owns
            mine[j] = x_ref[:, pl.ds(j * own, own)]
        # round 1: owner j's columns into slot `me` of its slices
        for j in range(n):

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                push(mine.at[j], slices.at[me], send1, recv1, j, axis, dict_ids)

        slices[pl.ds(me, 1)] = mine[pl.ds(me, 1)]
        wait_bytes_of(slices.at[pl.ds(0, n - 1)], recv1)
        gathered[pl.ds(me, 1)] = sum_slots(slices, n).astype(gathered.dtype)[None]
        # round 2: my summed columns into slot `me` of every other core
        for j in range(n):

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                push(gathered.at[me], gathered.at[me], send2, recv2, j, axis, dict_ids)

        wait_bytes_of(gathered.at[pl.ds(0, n - 1)], recv2)
        for j in range(n):
            o_ref[:, pl.ds(j * own, own)] = gathered[j]
        wait_bytes_of(slices.at[pl.ds(0, n - 1)], send1)
        wait_bytes_of(gathered.at[pl.ds(0, n - 1)], send2)

    scratch = [
        pltpu.VMEM((n, rows, own), x.dtype),
        pltpu.VMEM((n, rows, own), x.dtype),
        pltpu.VMEM((n, rows, own), x.dtype),
        *[pltpu.SemaphoreType.DMA] * 4,
    ]
    return call(
        kernel,
        x,
        rows,
        rank,
        scratch,
        collective_id,
        f"small_all_reduce_twoshot_r{rows}",
    )


def small_all_reduce(
    x, *, num_cores, axis, rank=None, form=None, collective_id=COLLECTIVE_ID
):
    """The sum of ``x`` over the ``num_cores`` cores of mesh axis ``axis``,
    inside a shard_map over that axis.

    Args:
      x: [rows, hidden], this core's block; hidden a multiple of
        num_cores * LANES.
      rank: int32 [1], this core's index on the axis as data (the torch
        bridge), or None to read it from the axis.
      form: "oneshot" or "twoshot"; by default ``form_for`` the block.
    """
    rows, hidden = x.shape
    if hidden % (num_cores * LANES):
        raise ValueError(
            f"hidden {hidden} is not a multiple of {num_cores} cores x {LANES} lanes"
        )
    form = form or form_for(rows, hidden, jnp.dtype(x.dtype).itemsize)
    build = {"oneshot": _oneshot, "twoshot": _twoshot}[form]
    return build(x, num_cores, axis, rank, collective_id)
