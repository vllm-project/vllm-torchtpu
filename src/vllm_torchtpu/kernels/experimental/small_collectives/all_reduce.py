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

Valid rows (``n_valid``, data): a padded decode block (a batch-1 MTP step pads
4 rows to 16) need not move its padding. With ``n_valid`` the two-shot form
gets a fold path: when n_valid <= fold_rows the valid rows are folded into one
row tile, q[j * fold_rows + r, c] = x[r, j * hidden / k + c] with k the lane
groups of an owner's column chunk, q is all-reduced (1/k of the bytes, every
DMA whole row tiles) and unfolded; rows >= n_valid come out +0. Above
fold_rows it is the two-shot above, unchanged. Either way an element's sum is
the same float32 sum over the cores in core order, so valid rows are bitwise
the plain two-shot's.
"""

import jax
import jax.numpy as jnp
from jax import lax
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


def fold_geometry(rows, hidden, num_cores, tile_rows):
    """(k, fold_rows) of the fold path, or None when it does not exist: k lane
    groups per owner chunk (>= 2), fold_rows = tile_rows // k (>= 1)."""
    own = hidden // num_cores
    if own % LANES or rows < tile_rows:
        return None
    k = own // LANES
    fold_rows = tile_rows // k
    return (k, fold_rows) if k >= 2 and fold_rows >= 1 else None


def _twoshot_valid_rows(x, n, axis, collective_id, tile_rows):
    """Two-shot with the fold path; operands (n_valid int32 [1], rank int32
    [1], both in SMEM, x). Both paths run _twoshot's protocol on per-owner chunks
    staged by static slices (a column window at a dynamic offset is slow to
    load), the fold path on [tile_rows, LANES] chunks of the folded tile."""
    rows, hidden = x.shape
    own = hidden // n
    geometry = fold_geometry(rows, hidden, n, tile_rows)
    assert geometry is not None
    k, fr = geometry
    hk = hidden // k  # folded width; a core's chunk of it is LANES wide

    def exchange(rows_x, width, me, mine, slices, gathered, sems, write_out):
        """_twoshot's rounds on mine[j] = owner j's [rows_x, width] chunk."""
        send1, recv1, send2, recv2 = sems

        def slot(buf, j):
            return buf.at[j, pl.ds(0, rows_x), pl.ds(0, width)]

        def slots(buf):
            return buf.at[pl.ds(0, n - 1), pl.ds(0, rows_x), pl.ds(0, width)]

        for j in range(n):

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                push(slot(mine, j), slot(slices, me), send1, recv1, j, axis, True)

        slices[pl.ds(me, 1), pl.ds(0, rows_x), pl.ds(0, width)] = mine[
            pl.ds(me, 1), pl.ds(0, rows_x), pl.ds(0, width)
        ]
        wait_bytes_of(slots(slices), recv1)
        acc = slices[0, pl.ds(0, rows_x), pl.ds(0, width)].astype(jnp.float32)
        for j in range(1, n):
            acc = acc + slices[j, pl.ds(0, rows_x), pl.ds(0, width)].astype(jnp.float32)
        gathered[pl.ds(me, 1), pl.ds(0, rows_x), pl.ds(0, width)] = acc.astype(
            gathered.dtype
        )[None]
        for j in range(n):

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                push(
                    slot(gathered, me), slot(gathered, me), send2, recv2, j, axis, True
                )

        wait_bytes_of(slots(gathered), recv2)
        write_out()
        wait_bytes_of(slots(slices), send1)
        wait_bytes_of(slots(gathered), send2)

    def kernel(n_valid_ref, rank_ref, x_ref, o_ref, mine, slices, gathered, *sems):
        n_valid = n_valid_ref[0]
        me = rank_ref[0]
        barrier(n, axis, True)

        @pl.when(n_valid <= fr)
        def _fold():
            row = lax.broadcasted_iota(jnp.int32, (tile_rows, hk), 0)
            p = x_ref[pl.ds(0, tile_rows), :].astype(jnp.float32)
            p = jnp.where(lax.broadcasted_iota(jnp.int32, p.shape, 0) < n_valid, p, 0.0)
            folded = jnp.zeros((tile_rows, hk), jnp.float32)
            for j in range(k):
                blk = p[:, j * hk : (j + 1) * hk]
                if j:
                    blk = pltpu.roll(blk, j * fr, 0)
                folded = jnp.where((row >= j * fr) & (row < (j + 1) * fr), blk, folded)
            folded = folded.astype(x.dtype)
            for j in range(n):
                mine[j, pl.ds(0, tile_rows), pl.ds(0, LANES)] = folded[
                    :, j * LANES : (j + 1) * LANES
                ]

            def write_out():
                summed = jnp.concatenate(
                    [
                        gathered[j, pl.ds(0, tile_rows), pl.ds(0, LANES)]
                        for j in range(n)
                    ],
                    axis=1,
                ).astype(jnp.float32)
                for j in range(k):
                    blk = (
                        summed if j == 0 else pltpu.roll(summed, tile_rows - j * fr, 0)
                    )
                    o_ref[pl.ds(0, tile_rows), pl.ds(j * hk, hk)] = jnp.where(
                        row < n_valid, blk, 0.0
                    ).astype(o_ref.dtype)
                if rows > tile_rows:
                    o_ref[pl.ds(tile_rows, rows - tile_rows), :] = jnp.zeros(
                        (rows - tile_rows, hidden), o_ref.dtype
                    )

            exchange(tile_rows, LANES, me, mine, slices, gathered, sems, write_out)

        @pl.when(n_valid > fr)
        def _full():
            for j in range(n):
                mine[j] = x_ref[:, pl.ds(j * own, own)]

            def write_out():
                for j in range(n):
                    o_ref[:, pl.ds(j * own, own)] = gathered[j]

            exchange(rows, own, me, mine, slices, gathered, sems, write_out)

    scratch = [
        pltpu.VMEM((n, rows, own), x.dtype),
        pltpu.VMEM((n, rows, own), x.dtype),
        pltpu.VMEM((n, rows, own), x.dtype),
        *[pltpu.SemaphoreType.DMA] * 4,
    ]
    return scratch, kernel, f"small_all_reduce_twoshot_valid_r{rows}"


def small_all_reduce_valid_rows(
    x, n_valid, rank, *, num_cores, axis, tile_rows, collective_id=COLLECTIVE_ID
):
    """The sum of ``x`` over the ``num_cores`` cores of mesh axis ``axis``,
    moving only its first n_valid rows; rows >= n_valid come out +0 when the
    fold path runs (n_valid <= fold_rows) and as the plain two-shot otherwise.

    Args:
      x: [rows, hidden], this core's block; hidden a multiple of
        num_cores * LANES, rows >= tile_rows, and ``fold_geometry`` not None.
      n_valid: int32 [1], the rows [0, n_valid) that are real, as data; the
        same on every core.
      rank: int32 [1], this core's index on the axis, as data. (Two [1]
        operands rather than one [2]: a 1-element SMEM operand is produced
        in place, a stacked pair costs a copy per call.)
      tile_rows: the bf16 sublane tiling (16 on TPU v7).
    """
    rows, hidden = x.shape
    if (
        hidden % (num_cores * LANES)
        or fold_geometry(rows, hidden, num_cores, tile_rows) is None
    ):
        raise ValueError(
            f"[{rows}, {hidden}] over {num_cores} cores has no fold path "
            f"(tile_rows={tile_rows})"
        )
    scratch, kernel, name = _twoshot_valid_rows(
        x, num_cores, axis, collective_id, tile_rows
    )
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[smem, smem, pl.BlockSpec()],
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=name,
    )(n_valid.astype(jnp.int32), rank.astype(jnp.int32), x)


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
