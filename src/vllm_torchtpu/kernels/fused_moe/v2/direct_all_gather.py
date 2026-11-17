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
"""Direct all-gather of a small payload on the SparseCore.

Every shard stages its payload in the SparseCore's shared memory and writes
it into slot `rank` of every peer's shared memory in one round of remote
copies, then drains the gathered slots to HBM. XLA's offloaded all-gather
passes the same rows around a ring in several dependent steps; this one has
a single round of copies. It runs asynchronously on the SparseCore, like
XLA's, so the TensorCore keeps working until it reads the result.
"""

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.compute_on import compute_on
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


def supports_direct_all_gather(ep: int, local_bytes: int) -> bool:
    """Whether direct_all_gather serves ep shards of local_bytes each.

    It needs a TPU7x SparseCore, at least one peer, and every shard's payload
    in the 32 MiB scoped SPMEM allocation at once.
    """
    return (
        jax.default_backend() == "tpu"
        and pltpu.get_tpu_info().generation >= 7
        and ep >= 2
        and 0 < ep * local_bytes <= 32 * 1024 * 1024
    )


def direct_all_gather(x: jax.Array, axis_name: str, ep: int) -> jax.Array:
    """lax.all_gather(x, axis_name, axis=0, tiled=True) as one round of copies."""
    data = x.reshape(-1)
    n = data.size

    def body(src, out, send, recv, local_sem, barrier, shared):
        rank = lax.axis_index(axis_name)
        # Every peer is inside this call, its shared slots free, before any
        # copy lands in them.
        for peer in range(ep):
            pl.semaphore_signal(barrier, 1, device_id={axis_name: peer, "sc": 0})
        pl.semaphore_wait(barrier, ep)
        mine = shared.at[pl.ds(rank * n, n)]
        pltpu.async_copy(src, mine, local_sem.at[0]).wait()
        pushes = [
            pltpu.async_remote_copy(
                mine,
                mine,
                send.at[step - 1],
                recv.at[step - 1],
                device_id={axis_name: lax.rem(rank + step, ep), "sc": 0},
            )
            for step in range(1, ep)
        ]
        for push in pushes:
            push.wait()
        pltpu.async_copy(shared, out, local_sem.at[1]).wait()

    kernel = pl.kernel(
        body,
        mesh=plsc.ScalarSubcoreMesh(axis_name="sc", num_cores=1),
        out_type=jax.ShapeDtypeStruct((ep * n,), x.dtype),
        scratch_types=[
            pltpu.SemaphoreType.DMA((ep - 1,)),
            pltpu.SemaphoreType.DMA((ep - 1,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.REGULAR,
            pltpu.VMEM_SHARED((ep * n,), x.dtype),
        ],
        name="moe_direct_route_ag",
    )
    with jax.named_scope("moe_direct_route_ag"):
        result = compute_on(
            kernel,
            compute_type="tpu_sparsecore",
            out_memory_spaces=jax.memory.Space.Device,
            compiler_options={"sparse_core_config": {"core_ids": [1]}},
        )(data)
    return result.reshape((ep * x.shape[0], *x.shape[1:]))
