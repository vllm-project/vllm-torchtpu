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
"""Routing-exchange all-gather as a TensorCore Pallas remote-DMA kernel.

MOE_FUSED_EP_PUSH_TC_EXCHANGE=1 replaces the layer's routing-exchange
`lax.all_gather` (the i32 [ep, W] gather of the packed routing), which XLA
offloads to the SparseCore and serializes behind other SparseCore work, with
one small TensorCore kernel: every shard writes its row into row `me` of
every peer's output (one remote DMA each, its own row locally), then waits
for the ep - 1 incoming rows. Bit-exact with
`lax.all_gather(packed, axis, tiled=True)`. Details:

* the shard's rank arrives as data (scalar prefetch), never `lax.axis_index`;
* whole-tile slabs only: a row is [R, 128] i32 with R a multiple of 8 on an
  untiled leading axis, so each DMA moves complete (8, 128) tiles (a
  partial-tile remote DMA has crashed a slice);
* a two-phase entry barrier (barrier semaphore, then a scoped REGULAR
  semaphore), so every peer is inside this call -- output allocated, the
  previous call's exchange consumed -- before any row lands;
* its own collective id: the TensorCore MoE kernel uses 0 and the SparseCore
  kernels 1.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANES = 128
SUBLANES = 8
COLLECTIVE_ID = 2

_CACHE = {}


def _double_barrier(ep, me):
    barrier_sem = pltpu.get_barrier_semaphore()

    def sig(sem):
        def one(i, c):
            pl.semaphore_signal(
                sem,
                inc=1,
                device_id=(lax.rem(me + i, ep),),
                device_id_type=pl.DeviceIdType.MESH,
            )
            return c

        lax.fori_loop(0, ep, one, jnp.int32(0))
        pl.semaphore_wait(sem, ep)

    sig(barrier_sem)

    @functools.partial(pl.run_scoped, second=pltpu.SemaphoreType.REGULAR)
    def _(second):
        sig(second)


def build_tc_exchange_all_gather(*, ep, width):
    """fn(packed i32[width], rank i32 scalar / [1] / [1, 1]) -> i32[ep * width]."""
    tile = SUBLANES * LANES
    rows = -(-width // tile) * SUBLANES  # whole (8, 128) tiles
    key = (ep, width)
    if key in _CACHE:
        return _CACHE[key]

    def kernel(rank_sm, x_hbm, out_hbm, send_sem, recv_sem, lsem):
        me = rank_sm[0]
        _double_barrier(ep, me)
        mine = out_hbm.at[pl.ds(me, 1)]
        local = pltpu.make_async_copy(x_hbm, mine, lsem)
        local.start()

        def push(i, c):
            dest = lax.rem(me + i, ep)
            pltpu.make_async_remote_copy(
                src_ref=x_hbm,
                dst_ref=mine,
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=(dest,),
                device_id_type=pl.DeviceIdType.MESH,
            ).start()
            return c

        lax.fori_loop(1, ep, push, jnp.int32(0))
        local.wait()
        if ep > 1:
            # DMA semaphores count bytes: one wait the size of ep - 1 rows
            # retires every incoming (resp. outgoing) row.
            blk = out_hbm.at[pl.ds(0, ep - 1)]
            pltpu.make_async_copy(blk, blk, recv_sem).wait()
            pltpu.make_async_copy(blk, blk, send_sem).wait()

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[hbm],
            out_specs=hbm,
            scratch_shapes=(
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
            ),
            grid=(),
        ),
        out_shape=jax.ShapeDtypeStruct((ep, rows, LANES), jnp.int32),
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID,
            disable_bounds_checks=True,
        ),
        name=f"moe_tc_routing_exchange_ep{ep}_w{width}",
    )

    def fn(packed, rank):
        flat = packed.astype(jnp.int32).reshape(-1)
        if flat.shape[0] != width:
            raise ValueError(f"expected a [{width}] payload, got {flat.shape}")
        slab = jnp.pad(flat, (0, rows * LANES - width)).reshape(1, rows, LANES)
        r = jnp.asarray(rank).astype(jnp.int32).reshape(-1)[:1]
        out = call(r, slab)
        return out.reshape(ep, rows * LANES)[:, :width].reshape(-1)

    _CACHE[key] = fn
    return fn
