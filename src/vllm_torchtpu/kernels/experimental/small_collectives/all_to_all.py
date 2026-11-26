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
"""Single-round remote-DMA all-to-all of small `[n * block, V]` row blocks."""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.small_collectives.common import (
    MESH,
    wait_bytes_of,
)

COLLECTIVE_ID = 11


def reference(blocks, row0=0):
    out = blocks.swapaxes(0, 1).copy()
    out[:, :, :row0] = 0
    n = out.shape[0]
    return out.reshape(n, -1, out.shape[-1])


def small_all_to_all(
    x,
    *,
    num_cores,
    axis,
    row0=0,
    rank=None,
    lane_axis=None,
    lane=None,
    collective_id=COLLECTIVE_ID,
    hbm_operand=False,
):
    if x.ndim != 2:
        raise ValueError(f"small_all_to_all takes [n * block, V], got {x.shape}")
    n = num_cores
    total, _ = x.shape
    if total % n:
        raise ValueError(f"{total} rows do not split into {n} blocks")
    block = total // n
    if block % 8 or row0 % 8 or not 0 <= row0 < block:
        raise ValueError(
            f"block {block} / row0 {row0}: need multiples of 8, row0 < block"
        )
    rows = block - row0
    data_rank = rank is not None
    data_lane = lane_axis is not None and lane is not None

    def kernel(*refs):
        refs = list(refs)
        me = refs.pop(0)[0] if data_rank else lax.axis_index(axis)
        if lane_axis is None:
            my_lane = None
        elif data_lane:
            my_lane = refs.pop(0)[0]
        else:
            my_lane = lax.axis_index(lane_axis)
        x_ref, o_ref, send_sem, recv_sem, local_sem = refs

        def peer(j):
            ids = {axis: jnp.int32(j)}
            if lane_axis is not None:
                ids[lane_axis] = my_lane
            return ids

        sem = pltpu.get_barrier_semaphore()
        for j in range(n):
            pl.semaphore_signal(sem, 1, device_id=peer(j), device_id_type=MESH)
        pl.semaphore_wait(sem, n)

        mine = o_ref.at[pl.ds(pl.multiple_of(me * block + row0, 8), rows)]
        for j in range(n):

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                pltpu.make_async_remote_copy(
                    src_ref=x_ref.at[pl.ds(j * block + row0, rows)],
                    dst_ref=mine,
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id=peer(j),
                    device_id_type=MESH,
                ).start()

        own = x_ref.at[pl.ds(pl.multiple_of(me * block + row0, 8), rows)]
        local = pltpu.make_async_copy(own, mine, local_sem)
        local.start()
        others = o_ref.at[pl.ds(0, (n - 1) * rows)]
        wait_bytes_of(others, recv_sem)
        local.wait()
        wait_bytes_of(others, send_sem)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    any_ = pl.BlockSpec(memory_space=pl.ANY)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    scalars = []
    if data_rank:
        scalars.append(jnp.reshape(rank, (-1,))[:1].astype(jnp.int32))
    if data_lane:
        scalars.append(jnp.reshape(lane, (-1,))[:1].astype(jnp.int32))
    if hbm_operand:
        x = pltpu.with_memory_space_constraint(x, pltpu.HBM)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[smem] * len(scalars) + [hbm if hbm_operand else any_],
        out_specs=hbm,
        scratch_shapes=[pltpu.SemaphoreType.DMA] * 3,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=f"small_all_to_all_n{n}_b{block}_r{row0}",
    )(*scalars, x)
