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
"""All-reduce of a small [rows, hidden] block over one mesh axis, as one Pallas
kernel.

For a block of a few to about a hundred rows, XLA's all-reduce spends most of
its time on fixed launch and synchronization cost, not on moving bytes. This
kernel moves the bytes with direct remote copies between the cores, in one of
two forms:

    one-shot   Every core pushes its whole block to every other core and sums
               the arrivals locally. One round, n - 1 blocks sent per core.
    two-shot   Core j owns columns [j * own, (j + 1) * own), own = hidden / n.
               Every core pushes each owner its columns; the owner sums them
               and pushes the sum back to everyone. Two rounds, about
               2 (n - 1) / n blocks sent per core.

Both forms start with a barrier so remote scratch is ready, sum in float32 and
round once to the input dtype. Every core gets the same result.

The kernel runs inside a ``jax.shard_map`` over the reduced axis, on any number
of cores of that axis. When the caller does not pass the core's index as data,
the mesh must have that one axis only.
"""

import math

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

# One-shot up to this many bytes per core, two-shot above. One-shot sends n - 1
# blocks in one round. Two-shot sends about 2 blocks in two rounds. The value
# comes from an earlier serving configuration. A later measurement on 8 TPU v7x
# cores at H=4096 bf16 found that the two forms cross at about 6 rows (48 KiB).
# The reason the value stays below that crossover is not recorded. Other core
# counts and widths are not measured.
ONE_SHOT_MAX_BYTES = 32 * 1024
# Largest share of the core's VMEM capacity that one call may use. The same
# fraction as the all-reduce in kernels/experimental/moe_tp_decode.
VMEM_FRACTION = 0.75
# Barrier semaphore group. Every kernel compiled with the same id in one
# program shares it; the fused MoE kernels use 0 and the DeepSeek-V4
# o-projection fetch uses 3.
COLLECTIVE_ID = 5


def form_for(rows, hidden, itemsize):
    """The form for a block of rows x hidden elements of itemsize bytes."""
    return "oneshot" if rows * hidden * itemsize <= ONE_SHOT_MAX_BYTES else "twoshot"


def _tiled_bytes(shape, itemsize, info):
    """VMEM bytes of a buffer padded to Mosaic tiles: the last dimension to the
    lane count, the second-last to the sublane tiling of the dtype."""
    *lead, rows, cols = shape
    sublanes = info.num_sublanes * max(1, 4 // itemsize)
    rows = pl.cdiv(rows, sublanes) * sublanes
    cols = pl.cdiv(cols, info.num_lanes) * info.num_lanes
    return math.prod(lead) * rows * cols * itemsize


def vmem_bytes(form, rows, hidden, num_cores, itemsize, info):
    """VMEM bytes one call of ``form`` holds: the input block, the output block
    and the scratch buffers.

    One-shot holds n + 2 blocks: the input, the output and one arrival slot
    per core. Two-shot holds 5 blocks: the input, the output and three scratch
    buffers of n column chunks of hidden / n columns each. Example: bf16
    [16, 4096] on 8 cores takes 10 x 128 KiB = 1.25 MiB in one-shot and
    5 x 128 KiB = 640 KiB in two-shot.

    Args:
      form: "oneshot" or "twoshot".
      rows: The rows of the block.
      hidden: The columns of the block.
      num_cores: The number of cores on the reduced axis.
      itemsize: The bytes of one element.
      info: The chip description from ``pltpu.get_tpu_info()``.

    Returns:
      The VMEM bytes, with every buffer padded to Mosaic tiles.
    """
    block = 2 * _tiled_bytes((rows, hidden), itemsize, info)
    if form == "oneshot":
        return block + _tiled_bytes((num_cores, rows, hidden), itemsize, info)
    own = hidden // num_cores
    return block + 3 * _tiled_bytes((num_cores, rows, own), itemsize, info)


def _oneshot(x, n, axis, rank, collective_id, vmem_limit):
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
        vmem_limit,
    )


def _twoshot(x, n, axis, rank, collective_id, vmem_limit):
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
        vmem_limit,
    )


def small_all_reduce(
    x, *, num_cores, axis, rank=None, form=None, collective_id=COLLECTIVE_ID
):
    """The sum of ``x`` over the ``num_cores`` cores of mesh axis ``axis``.

    Call it inside a ``jax.shard_map`` over ``axis``. Example: on 8 cores,
    each holding a bf16 [16, 4096] block, every core gets the bf16 [16, 4096]
    elementwise sum of the 8 blocks, summed in float32 and rounded once.

    Args:
      x: bf16 [rows, hidden], this core's block. ``hidden`` must be a multiple
        of ``LANES`` (128 lanes). Two-shot also needs ``hidden`` to be a
        multiple of ``num_cores * LANES``, so that each core owns whole lane
        tiles.
      num_cores: The number of cores on ``axis``.
      axis: The name of the mesh axis to reduce over.
      rank: int32 [1], this core's index on the axis, passed as data. Use it
        where ``lax.axis_index`` cannot lower (the torch bridge's export
        path). None reads the index from the axis, which lowers only when the
        mesh has the one axis ``axis``.
      form: "oneshot" or "twoshot". None picks by size with ``form_for``. When
        ``form_for`` picks two-shot and ``hidden`` is not a multiple of
        ``num_cores * LANES``, None takes one-shot instead, which moves whole
        blocks and has no such need.
      collective_id: The barrier semaphore group. Kernels compiled into one
        program with the same id share their barrier semaphore.

    Returns:
      [rows, hidden] in the dtype of ``x``: the sum over the cores.

    Raises:
      ValueError: ``x`` is not bf16, the only dtype the tests cover.
      ValueError: ``hidden`` is not a multiple of ``LANES``.
      ValueError: ``form`` is "twoshot" and ``hidden`` is not a multiple of
        ``num_cores * LANES``.
      ValueError: the form needs more VMEM than ``VMEM_FRACTION`` of the
        core's capacity (see ``vmem_bytes``).
    """
    rows, hidden = x.shape
    if x.dtype != jnp.bfloat16:
        raise ValueError(f"dtype {x.dtype} is not supported; only bfloat16 is tested")
    if hidden % LANES:
        raise ValueError(f"hidden {hidden} is not a multiple of {LANES} lanes")
    itemsize = jnp.dtype(x.dtype).itemsize
    splits = hidden % (num_cores * LANES) == 0
    if form is None:
        form = form_for(rows, hidden, itemsize) if splits else "oneshot"
    if form == "twoshot" and not splits:
        raise ValueError(
            f"two-shot needs hidden {hidden} to be a multiple of {num_cores} "
            f"cores x {LANES} lanes"
        )
    build = {"oneshot": _oneshot, "twoshot": _twoshot}[form]
    info = pltpu.get_tpu_info()
    limit = int(info.vmem_capacity_bytes * VMEM_FRACTION)
    need = vmem_bytes(form, rows, hidden, num_cores, itemsize, info)
    if need > limit:
        raise ValueError(
            f"{form} all-reduce of rows {rows} x hidden {hidden} on {num_cores} "
            f"cores needs {need} B of VMEM, more than the limit of {limit} B "
            f"({VMEM_FRACTION} of the core's VMEM)"
        )
    return build(x, num_cores, axis, rank, collective_id, limit)
