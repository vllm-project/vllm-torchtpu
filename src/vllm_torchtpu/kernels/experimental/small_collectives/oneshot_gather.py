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
"""One-shot all-gather of a small block over a whole 1-D mesh axis, HBM to HBM.

One all-pairs barrier, then every core DMAs its block straight into its own
slot of every peer's output -- no VMEM staging, no multi-step schedule. Pure
data movement, so the result is bit-identical to ``lax.all_gather(x, axis,
tiled=True)`` in mesh order. The idea is hwanginho/dsv4-agentx-c16 98b134ebc
(an int32 routing row); this form takes any [rows, ...] block of any dtype, so
it can carry the fused EP MoE's merged activation + routing payload.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.small_collectives.common import (
    Group,
    barrier,
    compiler_params,
    push,
    wait_bytes_of,
)

# Barrier semaphore group of its own: the fused EP MoE kernel uses 0, the
# small all-reduce 5 and the small all-gather 6.
COLLECTIVE_ID = 7


def oneshot_all_gather(
    x, *, num_cores, axis, rank, collective_id=COLLECTIVE_ID, skip_device_barrier=False
):
    """[num_cores * rows, ...]: every core's ``x`` in mesh order, inside a
    shard_map over mesh axis ``axis`` of ``num_cores`` cores.

    rank (int32 [1] or scalar) is this core's mesh index, as data.
    """
    rows = x.shape[0]
    n = num_cores

    def kernel(rank_ref, x_ref, o_ref, send_sem, recv_sem, local_sem):
        me = rank_ref[0]
        group = Group(n, axis, True)
        barrier(group)
        dst = o_ref.at[pl.ds(me * rows, rows)]
        for j in range(n):

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                push(x_ref, dst, send_sem, recv_sem, j, group)

        local = pltpu.make_async_copy(x_ref, dst, local_sem)
        local.start()
        # Each wait retires the bytes of n - 1 blocks: the arrivals from every
        # peer, then this core's own sends (the source must outlive them).
        wait_bytes_of(o_ref.at[pl.ds(0, (n - 1) * rows)], recv_sem)
        wait_bytes_of(o_ref.at[pl.ds(0, (n - 1) * rows)], send_sem)
        local.wait()

    hbm = pltpu.MemorySpace.HBM
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((n * rows, *x.shape[1:]), x.dtype),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=hbm),
        ],
        out_specs=pl.BlockSpec(memory_space=hbm),
        scratch_shapes=[pltpu.SemaphoreType.DMA] * 3,
        compiler_params=compiler_params(collective_id, skip_device_barrier),
        name=f"oneshot_all_gather_n{n}_r{rows}",
    )(jnp.reshape(rank, (1,)).astype(jnp.int32), x)
