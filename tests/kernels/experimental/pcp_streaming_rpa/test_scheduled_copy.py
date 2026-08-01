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

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import (
    ScheduleField, pack_pcp_streaming_schedule_fields)

P = jax.sharding.PartitionSpec
AXIS = "pcp"


def _scheduled_copy_kernel(
    kv_cache_ref,
    packed_schedule_ref,
    o_ref,
    sched_dma_sem,
    remote_send_sems,
    remote_recv_sems,
    local_dma_sem,
    sched_vmem_ref,
    kv_vmem_ref,
    *,
    pcp_size,
    num_lanes,
):
    my_id = lax.axis_index(AXIS)

    sched_load = pltpu.make_async_copy(
        src_ref=packed_schedule_ref.at[0],
        dst_ref=sched_vmem_ref.at[:, :, :],
        sem=sched_dma_sem,
    )
    sched_load.start()
    sched_load.wait()

    for consumer_rank in range(pcp_size):
        for lane in range(num_lanes):
            req_id = sched_vmem_ref[consumer_rank, lane, ScheduleField.REQ_ID]
            src_rank = sched_vmem_ref[consumer_rank, lane,
                                      ScheduleField.KV_PAGE_RANK]
            page = sched_vmem_ref[consumer_rank, lane,
                                  ScheduleField.KV_PAGE_IDX]
            is_source = jnp.logical_and(req_id != -1, src_rank == my_id)

            @pl.when(is_source)
            def _push_to_consumer(consumer_rank=consumer_rank, lane=lane):
                remote_op = pltpu.make_async_remote_copy(
                    src_ref=kv_cache_ref.at[0, page],
                    dst_ref=kv_vmem_ref.at[lane],
                    send_sem=remote_send_sems.at[consumer_rank, lane],
                    recv_sem=remote_recv_sems.at[lane],
                    device_id=(consumer_rank, ),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                remote_op.start()
                remote_op.wait()

    my_req_id = sched_vmem_ref[my_id, 0, ScheduleField.REQ_ID]

    @pl.when(my_req_id != -1)
    def _store_received_page():
        local_op = pltpu.make_async_copy(
            src_ref=kv_vmem_ref.at[0],
            dst_ref=o_ref.at[0, 0],
            sem=local_dma_sem,
        )
        local_op.start()
        local_op.wait()


def _scheduled_copy_call(kv_cache, packed_schedule, *, pcp_size, num_lanes):
    _, _, page_size, kv_heads, kv_pair, head_dim = kv_cache.shape
    out_shape = jax.ShapeDtypeStruct(
        (1, num_lanes, page_size, kv_heads, kv_pair, head_dim),
        kv_cache.dtype,
    )
    return pl.pallas_call(
        functools.partial(
            _scheduled_copy_kernel,
            pcp_size=pcp_size,
            num_lanes=num_lanes,
        ),
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=(
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((pcp_size, num_lanes)),
                pltpu.SemaphoreType.DMA((num_lanes, )),
                pltpu.SemaphoreType.DMA,
                pltpu.VMEM(
                    (pcp_size, num_lanes, ScheduleField.PACKED_NUM_FIELDS),
                    packed_schedule.dtype),
                pltpu.VMEM((num_lanes, page_size, kv_heads, kv_pair, head_dim),
                           kv_cache.dtype),
            ),
            grid=(1, ),
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=8 * 1024 *
                                             1024, ),
        name="pcp_streaming_scheduled_copy_smoke",
    )(kv_cache, packed_schedule)


def _run_scheduled_copy(kv_cache, packed_schedule, *, pcp_size, num_lanes):
    mesh = jax.sharding.Mesh(jax.local_devices()[:pcp_size], (AXIS, ))
    fn = jax.jit(
        jax.shard_map(
            functools.partial(
                _scheduled_copy_call,
                pcp_size=pcp_size,
                num_lanes=num_lanes,
            ),
            mesh=mesh,
            in_specs=(
                P(AXIS, None, None, None, None, None),
                P(None, None, None, None),
            ),
            out_specs=P(AXIS, None, None, None, None, None),
            check_vma=False,
        ))
    return fn(kv_cache, packed_schedule)


def _make_ring_schedule(pcp_size, num_lanes):
    shape = (pcp_size, 1, num_lanes)
    req_id = np.zeros(shape, dtype=np.int32)
    kv_page_rank = np.zeros(shape, dtype=np.int32)
    kv_page_idx = np.zeros(shape, dtype=np.int32)
    zeros = np.zeros(shape, dtype=np.int32)
    for consumer_rank in range(pcp_size):
        kv_page_rank[consumer_rank, 0, 0] = (consumer_rank - 1) % pcp_size
    return pack_pcp_streaming_schedule_fields(
        req_id=req_id,
        kv_page_rank=kv_page_rank,
        kv_page_idx=kv_page_idx,
        is_first_kv=zeros,
        is_last_kv=zeros,
        load_q=zeros,
        q_global_start=zeros,
        kv_global_start=zeros,
        kv_valid_len=zeros,
        q_hbm_offset=zeros,
        q_tile_size=zeros,
        o_hbm_offset=zeros,
    )


def _require_tpu_devices(min_count, reason):
    devices = jax.local_devices()
    if len(devices) < min_count or devices[0].platform != "tpu":
        pytest.skip(reason)
    return devices


def test_staged_schedule_can_drive_remote_kv_page_push(release_jax_backend):
    pcp_size = 4
    _require_tpu_devices(
        pcp_size,
        "Scheduled PCP remote-copy test requires at least four TPU devices.",
    )
    num_lanes = 1
    num_pages = 1
    page_size = 2
    kv_heads = 1
    head_dim = 1
    kv_cache = jnp.arange(
        pcp_size * num_pages * page_size * kv_heads * 2 * head_dim,
        dtype=jnp.int32,
    ).reshape(pcp_size, num_pages, page_size, kv_heads, 2, head_dim)
    packed_schedule = jnp.asarray(_make_ring_schedule(pcp_size, num_lanes))

    out = _run_scheduled_copy(kv_cache,
                              packed_schedule,
                              pcp_size=pcp_size,
                              num_lanes=num_lanes)
    out.block_until_ready()
    out_np = np.asarray(jax.device_get(out))
    kv_np = np.asarray(jax.device_get(kv_cache))

    for consumer_rank in range(pcp_size):
        src_rank = (consumer_rank - 1) % pcp_size
        np.testing.assert_array_equal(out_np[consumer_rank, 0], kv_np[src_rank,
                                                                      0])
