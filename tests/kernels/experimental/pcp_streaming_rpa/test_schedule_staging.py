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

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

sys.path.insert(0, str(Path(__file__).parent))

from schedule_reference import \
    generate_pcp_streaming_schedule_reference  # noqa: E402


def _stage_schedule_step_kernel(packed_ref, o_ref, dma_sem, sched_vmem_ref):
    step = pl.program_id(0)
    load_op = pltpu.make_async_copy(
        src_ref=packed_ref.at[step],
        dst_ref=sched_vmem_ref.at[:, :, :],
        sem=dma_sem,
    )
    load_op.start()
    load_op.wait()

    store_op = pltpu.make_async_copy(
        src_ref=sched_vmem_ref.at[:, :, :],
        dst_ref=o_ref.at[step],
        sem=dma_sem,
    )
    store_op.start()
    store_op.wait()


def _stage_schedule_steps(packed_schedule):
    max_steps, pcp_size, num_lanes, num_fields = packed_schedule.shape
    return pl.pallas_call(
        _stage_schedule_step_kernel,
        out_shape=jax.ShapeDtypeStruct(packed_schedule.shape,
                                       packed_schedule.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=(
                pltpu.SemaphoreType.DMA,
                pltpu.VMEM((pcp_size, num_lanes, num_fields),
                           packed_schedule.dtype),
            ),
            grid=(max_steps, ),
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=8 * 1024 *
                                             1024, ),
        name="pcp_streaming_schedule_staging_smoke",
    )(packed_schedule)


def _require_tpu_devices(min_count, reason):
    devices = jax.local_devices()
    if len(devices) < min_count or devices[0].platform != "tpu":
        pytest.skip(reason)
    return devices


def test_packed_schedule_step_can_be_staged_from_hbm_to_vmem(
        release_jax_backend):
    _require_tpu_devices(1, "Schedule staging smoke test requires TPU.")
    schedule = generate_pcp_streaming_schedule_reference(
        kv_lens=[12],
        cu_q_lens=[0, 7],
        q_start_offsets=[5],
        block_tables=np.array([[100, 101]], dtype=np.int32),
        page_size=2,
        pcp_size=4,
        interleave_size=2,
        num_lanes=1,
        bq_sz=2,
    )
    packed = jnp.asarray(schedule.packed_schedule)

    staged = _stage_schedule_steps(packed)
    staged.block_until_ready()

    np.testing.assert_array_equal(np.asarray(jax.device_get(staged)),
                                  schedule.packed_schedule)
