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
"""The decode step's all-reduce as one Pallas kernel with remote DMAs.

A decode step sums a bf16 [rows, hidden] block over the cores of the
tensor-parallel group for the output projection and MoE combine.

    one-shot   Every core pushes its whole block to the others and sums
               the arrivals locally.
    two-shot   Core j owns columns [j * own, (j + 1) * own). Every core
               pushes each owner its columns; the owner sums its slices
               and pushes the sum back to everyone.

Both forms sum in float32 and round once to the input dtype. Both start
with a barrier over the cores so remote scratch is ready for incoming DMA.
A caller that fuses the sum into a longer kernel can overlap this barrier
with independent work.

Under the torch bridge the core's own index arrives as data (`rank`, an
int32 [1] in SMEM): `lax.axis_index` lowers to a partition-id the export
path's SPMD partitioner refuses. Peer addresses then take the dict form keyed
by the axis name, which names the mesh axis to Pallas without an axis lookup.
The waits go on local descriptors of the same bytes, since the bridge refuses
a wait on a descriptor that carries a peer address. These addressing and
wait conventions also apply to the fused EP transport.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Payloads up to this per-core byte limit use one-shot exchange; larger
# payloads use two-shot exchange. One-shot sends more bytes in one round,
# while two-shot reduces traffic at the cost of a second round.
ONE_SHOT_MAX_BYTES = 32 * 1024
# The barrier semaphore is shared by every kernel built with the same id in a
# program. The fused MoE kernels and the all-gather matmul use 0, the
# DeepSeek V4 output projection 3.
COLLECTIVE_ID = 5
LANES = 128
MESH = pl.DeviceIdType.MESH


def form_for(rows, hidden, itemsize):
    """The form for a block of rows x hidden elements of itemsize bytes."""
    return "oneshot" if rows * hidden * itemsize <= ONE_SHOT_MAX_BYTES else "twoshot"


def reference(blocks):
    """The sum the kernel computes: float32 over the cores' blocks
    [cores, rows, hidden], rounded once to the blocks' type."""
    return jnp.sum(blocks.astype(jnp.float32), axis=0).astype(blocks.dtype)


def _peer(j, axis, dict_ids):
    return {axis: jnp.int32(j)} if dict_ids else (jnp.int32(j),)


def _barrier(num_cores, axis, dict_ids):
    """Every core signals every core once, itself included, and waits for
    all of them: after this every core is inside the kernel, so its scratch
    exists and may be written into.

    One barrier is enough here, although a barrier reused across calls can
    let a fast core's signal for call k + 1 stand in for a slow core's signal
    for call k (kernels/collectives/util.local_barrier doubles its barrier
    for that reason). A core finishes call k only after every other core's
    data of call k has landed, and a core sends that data only after passing
    its own barrier of call k, so no core can signal call k + 1 while another
    still waits at the barrier of call k."""
    barrier = pltpu.get_barrier_semaphore()
    for j in range(num_cores):
        pl.semaphore_signal(
            barrier, 1, device_id=_peer(j, axis, dict_ids), device_id_type=MESH
        )
    pl.semaphore_wait(barrier, num_cores)


def _wait_bytes_of(ref_slice, sem):
    """Block until `sem` has counted the bytes of `ref_slice`: a wait on a
    local descriptor of the same size."""
    pltpu.make_async_copy(ref_slice, ref_slice, sem).wait()


def _push(src, dst, send_sem, recv_sem, j, axis, dict_ids):
    pltpu.make_async_remote_copy(
        src_ref=src,
        dst_ref=dst,
        send_sem=send_sem,
        recv_sem=recv_sem,
        device_id=_peer(j, axis, dict_ids),
        device_id_type=MESH,
    ).start()


def _split(refs, rank_operand, axis):
    """The core's index and the remaining refs."""
    if rank_operand:
        rank_ref, *refs = refs
        return rank_ref[0], refs
    return lax.axis_index(axis), refs


def _oneshot(x, rows, hidden, n, axis, rank, dict_ids, collective_id):
    def kernel(*refs):
        me, (x_ref, o_ref, arrivals, send_sem, recv_sem) = _split(
            refs, rank is not None, axis
        )
        _barrier(n, axis, dict_ids)
        for j in range(n):  # my block into slot `me` of every other core

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                _push(x_ref, arrivals.at[me], send_sem, recv_sem, j, axis, dict_ids)

        arrivals[pl.ds(me, 1)] = x_ref[...][None]
        _wait_bytes_of(arrivals.at[pl.ds(0, n - 1)], recv_sem)
        acc = arrivals[0].astype(jnp.float32)
        for j in range(1, n):
            acc = acc + arrivals[j].astype(jnp.float32)
        o_ref[...] = acc.astype(o_ref.dtype)
        _wait_bytes_of(arrivals.at[pl.ds(0, n - 1)], send_sem)

    return _call(
        kernel,
        x,
        rows,
        hidden,
        rank,
        [
            pltpu.VMEM((n, rows, hidden), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ],
        collective_id,
        f"small_all_reduce_oneshot_r{rows}",
    )


def _twoshot(x, rows, hidden, n, axis, rank, dict_ids, collective_id):
    own = hidden // n

    def kernel(*refs):
        (
            me,
            (
                x_ref,
                o_ref,
                mine,
                slices,
                gathered,
                send_sem,
                recv_sem,
                send2_sem,
                recv2_sem,
            ),
        ) = _split(refs, rank is not None, axis)
        _barrier(n, axis, dict_ids)
        # my block by owner: slot j holds the columns core j owns
        for j in range(n):
            mine[j] = x_ref[:, pl.ds(j * own, own)]
        # round 1: my columns for owner j into slot `me` of owner j's slices
        for j in range(n):

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                _push(mine.at[j], slices.at[me], send_sem, recv_sem, j, axis, dict_ids)

        slices[pl.ds(me, 1)] = mine[pl.ds(me, 1)]
        _wait_bytes_of(slices.at[pl.ds(0, n - 1)], recv_sem)
        acc = slices[0].astype(jnp.float32)
        for j in range(1, n):
            acc = acc + slices[j].astype(jnp.float32)
        gathered[pl.ds(me, 1)] = acc.astype(gathered.dtype)[None]
        # round 2: my summed columns into slot `me` of every other core
        for j in range(n):

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                _push(
                    gathered.at[me],
                    gathered.at[me],
                    send2_sem,
                    recv2_sem,
                    j,
                    axis,
                    dict_ids,
                )

        _wait_bytes_of(gathered.at[pl.ds(0, n - 1)], recv2_sem)
        for j in range(n):
            o_ref[:, pl.ds(j * own, own)] = gathered[j]
        _wait_bytes_of(slices.at[pl.ds(0, n - 1)], send_sem)
        _wait_bytes_of(gathered.at[pl.ds(0, n - 1)], send2_sem)

    return _call(
        kernel,
        x,
        rows,
        hidden,
        rank,
        [
            pltpu.VMEM((n, rows, own), x.dtype),
            pltpu.VMEM((n, rows, own), x.dtype),
            pltpu.VMEM((n, rows, own), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ],
        collective_id,
        f"small_all_reduce_twoshot_r{rows}",
    )


def _call(kernel, x, rows, hidden, rank, scratch, collective_id, name):
    rank_specs = [pl.BlockSpec(memory_space=pltpu.SMEM)] if rank is not None else []
    args = (() if rank is None else (rank.astype(jnp.int32),)) + (x,)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, hidden), x.dtype),
        in_specs=rank_specs + [pl.BlockSpec()],
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=name,
    )(*args)


def small_all_reduce(
    x,
    *,
    num_cores,
    axis="d",
    rank=None,
    dict_ids=None,
    form=None,
    collective_id=COLLECTIVE_ID,
):
    """The sum of `x` over the `num_cores` cores of mesh axis `axis`, inside a
    shard_map over that axis.

    Args:
      x: [rows, hidden] this core's block; hidden a multiple of
        num_cores * LANES.
      rank: int32 [1], this core's index on the axis as data (the torch
        bridge), or None to read it from the axis.
      dict_ids: peer addresses keyed by the axis name (the bridge's form);
        defaults to `rank is not None`.
      form: "oneshot" or "twoshot"; by default `form_for` of the block.
    """
    rows, hidden = x.shape
    if hidden % (num_cores * LANES):
        raise ValueError(
            f"hidden {hidden} is not a multiple of {num_cores} cores x {LANES} lanes"
        )
    if dict_ids is None:
        dict_ids = rank is not None
    form = form or form_for(rows, hidden, jnp.dtype(x.dtype).itemsize)
    build = {
        "oneshot": _oneshot,
        "twoshot": _twoshot,
    }[form]
    return build(x, rows, hidden, num_cores, axis, rank, dict_ids, collective_id)
