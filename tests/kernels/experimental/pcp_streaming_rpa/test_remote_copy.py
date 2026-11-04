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

import jax
import jax.numpy as jnp
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

pytestmark = pytest.mark.multichip

P = jax.sharding.PartitionSpec
AXIS = "pcp"
BLOCK = 16
LOCAL_ROWS = 1


def _copy_remote_hbm_to_vmem_kernel(
    x_ref, o_ref, send_sem, recv_sem, local_sem, vmem_ref
):
    my_id = lax.axis_index(AXIS)
    num_devices = lax.psum(1, AXIS)
    dst_id = lax.rem(my_id + 1, num_devices)
    remote_op = pltpu.make_async_remote_copy(
        src_ref=x_ref.at[:, :],
        dst_ref=vmem_ref.at[:, :],
        send_sem=send_sem,
        recv_sem=recv_sem,
        device_id=(dst_id,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    remote_op.start()
    remote_op.wait()

    local_op = pltpu.make_async_copy(
        src_ref=vmem_ref.at[:, :],
        dst_ref=o_ref.at[:, :],
        sem=local_sem,
    )
    local_op.start()
    local_op.wait()


def _pallas_remote_copy_call(x):
    return pl.pallas_call(
        _copy_remote_hbm_to_vmem_kernel,
        out_shape=jax.ShapeDtypeStruct((LOCAL_ROWS, BLOCK), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=(
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.VMEM((LOCAL_ROWS, BLOCK), x.dtype),
            ),
            grid=(1,),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=8 * 1024 * 1024,
        ),
        name="pcp_streaming_remote_hbm_to_vmem_smoke",
    )(x)


def _run_shard_map(x):
    mesh = jax.sharding.Mesh(jax.local_devices(), (AXIS,))
    fn = jax.jit(
        jax.shard_map(
            _pallas_remote_copy_call,
            mesh=mesh,
            in_specs=P(AXIS, None),
            out_specs=P(AXIS, None),
            check_vma=False,
        )
    )
    return fn(x)


def _require_tpu_devices(min_count, reason):
    devices = jax.local_devices()
    if len(devices) < min_count or devices[0].platform != "tpu":
        pytest.skip(reason)
    return devices


def test_make_async_remote_copy_can_push_hbm_to_destination_vmem(release_jax_backend):
    devices = _require_tpu_devices(
        2, "PCP remote-copy smoke test requires at least two TPU devices."
    )
    num_devices = len(devices)
    x = jnp.arange(num_devices * BLOCK, dtype=jnp.int32).reshape(num_devices, BLOCK)

    out = _run_shard_map(x)
    out.block_until_ready()

    expected = jnp.roll(x, shift=1, axis=0)
    assert bool(jnp.array_equal(out, expected))
