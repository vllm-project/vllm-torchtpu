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
"""All-gather of a small [rows, hidden] block over a group of cores, as one
Pallas kernel: a barrier so remote scratch is ready, then every core pushes its
block into its own slot of every other core's scratch. Pure data movement, so
the result is bit-identical to XLA's all-gather over the same group order.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.small_collectives.common import (
    LANES,
    barrier,
    push,
    scalar_operands,
    split_refs,
    wait_bytes_of,
)

# Barrier semaphore group, distinct from the all-reduce's and the fused EP MoE
# kernel's.
COLLECTIVE_ID = 6


def small_all_gather(x, *, num_cores, axis, rank, peers, collective_id=COLLECTIVE_ID):
    """[num_cores * rows, hidden]: every group member's ``x`` in group order,
    inside a shard_map over mesh axis ``axis``.

    rank (int32 [1]) is this core's position in its group and peers (int32
    [num_cores]) the mesh index of each member, both as data (see ``common``).
    """
    rows, hidden = x.shape
    n = num_cores
    if hidden % LANES:
        raise ValueError(f"hidden {hidden} is not a multiple of {LANES} lanes")

    def kernel(*refs):
        me, group, (x_ref, o_ref, slots, send_sem, recv_sem) = split_refs(
            refs, n, axis, True, True
        )
        barrier(group)
        for j in range(n):

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                push(x_ref, slots.at[me], send_sem, recv_sem, j, group)

        slots[pl.ds(me, 1)] = x_ref[...][None]
        wait_bytes_of(slots.at[pl.ds(0, n - 1)], recv_sem)
        for j in range(n):
            o_ref[pl.ds(j * rows, rows)] = slots[j]
        wait_bytes_of(slots.at[pl.ds(0, n - 1)], send_sem)

    args, specs = scalar_operands(rank, peers)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((n * rows, hidden), x.dtype),
        in_specs=[*specs, pl.BlockSpec()],
        scratch_shapes=[
            pltpu.VMEM((n, rows, hidden), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=f"small_all_gather_n{n}_r{rows}",
    )(*args, x)
