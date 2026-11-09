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
"""One-shot all-gather of a small array: one Pallas kernel of remote DMAs.

XLA all-gathers a few kilobytes per member over a ring, one step per member
(or per two, both directions), every step waiting on the previous one, so a
small all-gather over many members costs many network latencies. Here each
member signals the others once, then writes its block into every member's
output with one remote DMA and waits until the other members' blocks have
landed: two network latencies at any member count.
"""

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Barrier semaphore of this kernel: the TP all-reduce kernel uses 0 and the
# expert kernels 1 and 2.
_COLLECTIVE_ID = 6


def dma_all_gather(
    x: jax.Array,
    *,
    axis: str,
    num_members: int,
    collective_id: int = _COLLECTIVE_ID,
    interpret: bool = False,
) -> jax.Array:
    """``lax.all_gather(x, axis)`` (``[num_members, *x.shape]``, member order)
    as one Pallas kernel; call inside shard_map. ``num_members`` is the size
    of mesh axis ``axis``. Meant for blocks of a few kilobytes, where the
    gather is bound by network latency rather than bytes. ``interpret`` runs
    the kernel in the Pallas TPU interpreter (CPU tests)."""
    if num_members == 1:
        return x[None]

    def kernel(x_hbm, o_hbm, send_sem, recv_sem, local_sem):
        me = lax.axis_index(axis)
        peers = [lax.rem(me + i, num_members) for i in range(1, num_members)]
        # No block may land before its member runs this kernel: the output is
        # only this kernel's buffer from then on. A member leaves only after
        # every other member's block has landed, i.e. after every member
        # passed this barrier, so one round cannot meet the previous call's
        # signals.
        barrier = pltpu.get_barrier_semaphore()
        for peer in peers:
            pl.semaphore_signal(
                barrier, 1, device_id={axis: peer}, device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(barrier, num_members - 1)
        local = pltpu.make_async_copy(x_hbm, o_hbm.at[me], local_sem.at[0])
        local.start()
        copies = []
        for peer in peers:
            copy = pltpu.make_async_remote_copy(
                x_hbm,
                o_hbm.at[me],
                send_sem.at[0],
                recv_sem.at[0],
                device_id={axis: peer},
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait_send()
        # Every block has one size, so each wait counts one block's bytes.
        for copy in copies:
            copy.wait_recv()
        local.wait()

    shape = (num_members,) + x.shape
    if interpret:
        space, out_shape = pl.ANY, jax.ShapeDtypeStruct(shape, x.dtype)
        extra = {"interpret": pltpu.InterpretParams()}
    else:
        # The DMAs read the block from HBM and peers write into the output:
        # XLA would place both small buffers in VMEM otherwise.
        x = pltpu.with_memory_space_constraint(x, pltpu.HBM)
        space, out_shape, extra = pltpu.HBM, pltpu.HBM(shape, x.dtype), {}
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        in_specs=[pl.BlockSpec(memory_space=space)],
        out_specs=pl.BlockSpec(memory_space=space),
        scratch_shapes=[pltpu.SemaphoreType.DMA((1,))] * 3,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name="dma_all_gather",
        **extra,
    )(x)
