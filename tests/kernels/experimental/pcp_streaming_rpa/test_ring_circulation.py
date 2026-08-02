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

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.collectives import util

pytestmark = pytest.mark.multichip

P = jax.sharding.PartitionSpec
AXIS = "pcp"
PCP_SIZE = 4
PAGE_SIZE = 2
WIDTH = 128


def _ring_circulation_kernel(
    local_page_ref,
    o_ref,
    local_dma_sem,
    remote_send_sems,
    remote_recv_sems,
    kv_vmem_ref,
    *,
    pcp_size,
):
    my_id = lax.axis_index(AXIS)
    next_rank = lax.rem(my_id + 1, pcp_size)
    prev_rank = lax.rem(my_id + pcp_size - 1, pcp_size)

    load_op = pltpu.make_async_copy(
        src_ref=local_page_ref.at[0, :, :],
        dst_ref=kv_vmem_ref.at[0],
        sem=local_dma_sem,
    )
    load_op.start()
    load_op.wait()

    util.local_barrier(prev_rank, next_rank)

    store_op = pltpu.make_async_copy(
        src_ref=kv_vmem_ref.at[0],
        dst_ref=o_ref.at[0, 0],
        sem=local_dma_sem,
    )
    store_op.start()
    store_op.wait()

    for round_idx in range(1, pcp_size):
        src_slot = (round_idx - 1) % 2
        dst_slot = round_idx % 2
        remote_op = pltpu.make_async_remote_copy(
            src_ref=kv_vmem_ref.at[src_slot],
            dst_ref=kv_vmem_ref.at[dst_slot],
            send_sem=remote_send_sems.at[round_idx - 1],
            recv_sem=remote_recv_sems.at[round_idx - 1],
            device_id=(next_rank, ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        remote_op.start()
        remote_op.wait()

        store_op = pltpu.make_async_copy(
            src_ref=kv_vmem_ref.at[dst_slot],
            dst_ref=o_ref.at[0, round_idx],
            sem=local_dma_sem,
        )
        store_op.start()
        store_op.wait()


def _ring_circulation_call(local_page, *, pcp_size):
    return pl.pallas_call(
        functools.partial(_ring_circulation_kernel, pcp_size=pcp_size),
        out_shape=jax.ShapeDtypeStruct(
            (1, pcp_size, PAGE_SIZE, WIDTH),
            local_page.dtype,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=(
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((pcp_size - 1, )),
                pltpu.SemaphoreType.DMA((pcp_size - 1, )),
                pltpu.VMEM((2, PAGE_SIZE, WIDTH), local_page.dtype),
            ),
            grid=(1, ),
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=11,
            vmem_limit_bytes=8 * 1024 * 1024,
        ),
        name="pcp_streaming_ring_circulation_smoke",
    )(local_page)


def _run_ring_circulation(local_pages):
    mesh = jax.sharding.Mesh(jax.local_devices()[:PCP_SIZE], (AXIS, ))
    fn = jax.jit(
        jax.shard_map(
            functools.partial(_ring_circulation_call, pcp_size=PCP_SIZE),
            mesh=mesh,
            in_specs=P(AXIS, None, None),
            out_specs=P(AXIS, None, None, None),
            check_vma=False,
        ))
    return fn(local_pages)


def _require_tpu_devices(min_count, reason):
    devices = jax.local_devices()
    if len(devices) < min_count or devices[0].platform != "tpu":
        pytest.skip(reason)
    return devices


def test_vmem_pages_can_circulate_through_ring(release_jax_backend):
    _require_tpu_devices(
        PCP_SIZE, "PCP ring circulation smoke test requires four TPU devices.")
    local_pages = jnp.arange(
        PCP_SIZE * PAGE_SIZE * WIDTH,
        dtype=jnp.int32,
    ).reshape(PCP_SIZE, PAGE_SIZE, WIDTH)

    out = _run_ring_circulation(local_pages)
    out.block_until_ready()

    out_np = np.asarray(jax.device_get(out))
    local_np = np.asarray(jax.device_get(local_pages))
    for rank in range(PCP_SIZE):
        for round_idx in range(PCP_SIZE):
            src_rank = (rank - round_idx) % PCP_SIZE
            np.testing.assert_array_equal(out_np[rank, round_idx],
                                          local_np[src_rank])
