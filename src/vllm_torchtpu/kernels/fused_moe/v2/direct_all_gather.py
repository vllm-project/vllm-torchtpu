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

# The scoped SPMEM allocation on TPU7x, the bound
# preprocessing.supports_activation_gather also uses. Every shard's payload
# must fit it at once.
SHARED_MEMORY_BYTES = 32 * 1024 * 1024
# The measured exchanges: 8 shards on one 2x2 TPU7x host, routing rows of
# DeepSeek-V4.1-Flash from decode (640 int32) up to a 2,048-token step
# (14,720 int32, top-6 of 384 experts).
MAX_SHARDS = 8
MAX_LOCAL_BYTES = 64 * 1024
# Whole 128-lane int32 rows, as layer._exchange_layout pads every routing
# row; a multiple of the SparseCore's 64-byte DMA granule.
LOCAL_BYTES_ALIGNMENT = 512


def supports_direct_all_gather(ep: int, local_bytes: int, single_host: bool) -> bool:
    """Whether direct_all_gather serves ep shards of local_bytes each.

    It admits what was measured: a TPU7x SparseCore, 2 to 8 shards on one
    host, and payloads of whole 128-lane int32 rows up to the largest
    measured routing row. Everything else stays on lax.all_gather.
    """
    return (
        jax.default_backend() == "tpu"
        and pltpu.get_tpu_info().generation == 7
        and 2 <= ep <= MAX_SHARDS
        and single_host
        and 0 < local_bytes <= MAX_LOCAL_BYTES
        and local_bytes % LOCAL_BYTES_ALIGNMENT == 0
        and ep * local_bytes <= SHARED_MEMORY_BYTES
    )


def direct_all_gather(
    x: jax.Array, axis_name: str, ep: int, core_id: int = 1
) -> jax.Array:
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
            compiler_options={"sparse_core_config": {"core_ids": [core_id]}},
        )(data)
    return result.reshape((ep * x.shape[0], *x.shape[1:]))


def supports_direct_all_gather_to_hbm(
    ep: int, local_bytes: int, single_host: bool
) -> bool:
    """Whether direct_all_gather_to_hbm serves ep shards of local_bytes each."""
    return (
        jax.default_backend() == "tpu"
        and pltpu.get_tpu_info().generation == 7
        and 2 <= ep <= MAX_SHARDS
        and single_host
        and local_bytes > 0
        and local_bytes % 64 == 0
        and ep * local_bytes <= SHARED_MEMORY_BYTES
    )


def direct_all_gather_to_hbm(
    x: jax.Array, axis_name: str, ep: int, core_id: int, name: str
) -> jax.Array:
    """lax.all_gather(x, axis_name, axis=0, tiled=True) as one round of copies.

    Each shard copies its rows from HBM straight into slot `rank` of every
    peer's output, in x's own dtype and layout: no staging in shared memory,
    no drain, and no int32 repack of sub-word rows on the TensorCore.
    """
    t = x.shape[0]

    def body(src, out, send, recv, local_sem, barrier):
        rank = lax.axis_index(axis_name)
        # Every peer is inside this call, its output allocated and no longer
        # read by earlier work, before any copy lands in it.
        for peer in range(ep):
            pl.semaphore_signal(barrier, 1, device_id={axis_name: peer, "sc": 0})
        pl.semaphore_wait(barrier, ep)
        mine = out.at[pl.ds(rank * t, t)]
        local = pltpu.async_copy(src, mine, local_sem.at[0])
        pushes = [
            pltpu.async_remote_copy(
                src,
                mine,
                send.at[step - 1],
                recv.at[step - 1],
                device_id={axis_name: lax.rem(rank + step, ep), "sc": 0},
            )
            for step in range(1, ep)
        ]
        local.wait()
        # Each wait covers this shard's send and the same-sized copy that
        # peer rank - step writes into this shard's slot for it.
        for push in pushes:
            push.wait()

    kernel = pl.kernel(
        body,
        mesh=plsc.ScalarSubcoreMesh(axis_name="sc", num_cores=1),
        out_type=jax.ShapeDtypeStruct((ep * t, *x.shape[1:]), x.dtype),
        scratch_types=[
            pltpu.SemaphoreType.DMA((ep - 1,)),
            pltpu.SemaphoreType.DMA((ep - 1,)),
            pltpu.SemaphoreType.DMA((1,)),
            pltpu.SemaphoreType.REGULAR,
        ],
        name=name,
    )
    with jax.named_scope(name):
        return compute_on(
            kernel,
            compute_type="tpu_sparsecore",
            out_memory_spaces=jax.memory.Space.Device,
            compiler_options={"sparse_core_config": {"core_ids": [core_id]}},
        )(x)
