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
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.collectives import util
from vllm_torchtpu.kernels.experimental.batched_rpa.utils import broadcast_minor

pytestmark = pytest.mark.multichip

P = jax.sharding.PartitionSpec
AXIS = "pcp"
PCP_SIZE = 4
Q_TILE = 8
PAGE_SIZE = 128
HEAD_DIM = 128


def _consume_kv_page(q_vmem_ref, kv_vmem_ref, slot, m, l_state, acc, *, sm_scale):
    q = q_vmem_ref[...].astype(jnp.float32)
    k = kv_vmem_ref.at[slot, :, 0, :][...].astype(jnp.float32)
    v = kv_vmem_ref.at[slot, :, 1, :][...].astype(jnp.float32)

    scores = jnp.matmul(q, k.T, preferred_element_type=jnp.float32) * sm_scale
    m_curr = jnp.max(scores, axis=1, keepdims=True)
    m_next = jnp.maximum(m, m_curr)
    p = jnp.exp(scores - broadcast_minor(m_next, scores.shape))
    alpha = jnp.exp(m - m_next)
    l_next = alpha * l_state + jnp.sum(p, axis=1, keepdims=True)
    pv = jnp.matmul(p, v, preferred_element_type=jnp.float32)
    acc_next = broadcast_minor(alpha, acc.shape) * acc + pv
    return m_next, l_next, acc_next


def _ring_streaming_attention_kernel(
    q_ref,
    kv_ref,
    o_ref,
    local_dma_sem,
    remote_send_sems,
    remote_recv_sems,
    q_vmem_ref,
    kv_vmem_ref,
    o_vmem_ref,
    *,
    pcp_size,
    sm_scale,
):
    my_id = lax.axis_index(AXIS)
    next_rank = lax.rem(my_id + 1, pcp_size)
    prev_rank = lax.rem(my_id + pcp_size - 1, pcp_size)

    q_load = pltpu.make_async_copy(
        src_ref=q_ref.at[0, :, :],
        dst_ref=q_vmem_ref.at[:, :],
        sem=local_dma_sem,
    )
    q_load.start()
    q_load.wait()

    kv_load = pltpu.make_async_copy(
        src_ref=kv_ref.at[0, :, :, :],
        dst_ref=kv_vmem_ref.at[0],
        sem=local_dma_sem,
    )
    kv_load.start()
    kv_load.wait()

    util.local_barrier(prev_rank, next_rank)

    m = jnp.full((Q_TILE, 128), -jnp.inf, dtype=jnp.float32)
    l_state = jnp.zeros((Q_TILE, 128), dtype=jnp.float32)
    acc = jnp.zeros((Q_TILE, HEAD_DIM), dtype=jnp.float32)

    for round_idx in range(pcp_size):
        curr_slot = round_idx % 2
        next_slot = 1 - curr_slot
        if round_idx < pcp_size - 1:
            remote_op = pltpu.make_async_remote_copy(
                src_ref=kv_vmem_ref.at[curr_slot],
                dst_ref=kv_vmem_ref.at[next_slot],
                send_sem=remote_send_sems.at[round_idx],
                recv_sem=remote_recv_sems.at[round_idx],
                device_id=(next_rank,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            remote_op.start()

        m, l_state, acc = _consume_kv_page(
            q_vmem_ref,
            kv_vmem_ref,
            curr_slot,
            m,
            l_state,
            acc,
            sm_scale=sm_scale,
        )

        if round_idx < pcp_size - 1:
            remote_op.wait()

    o_vmem_ref[...] = (acc / broadcast_minor(l_state, acc.shape)).astype(
        o_vmem_ref.dtype
    )
    o_store = pltpu.make_async_copy(
        src_ref=o_vmem_ref.at[:, :],
        dst_ref=o_ref.at[0, :, :],
        sem=local_dma_sem,
    )
    o_store.start()
    o_store.wait()


def _ring_streaming_attention_call(q, kv, *, pcp_size, sm_scale):
    return pl.pallas_call(
        functools.partial(
            _ring_streaming_attention_kernel,
            pcp_size=pcp_size,
            sm_scale=sm_scale,
        ),
        out_shape=jax.ShapeDtypeStruct((1, Q_TILE, HEAD_DIM), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=(
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((pcp_size - 1,)),
                pltpu.SemaphoreType.DMA((pcp_size - 1,)),
                pltpu.VMEM((Q_TILE, HEAD_DIM), q.dtype),
                pltpu.VMEM((2, PAGE_SIZE, 2, HEAD_DIM), kv.dtype),
                pltpu.VMEM((Q_TILE, HEAD_DIM), jnp.float32),
            ),
            grid=(1,),
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=12,
            vmem_limit_bytes=8 * 1024 * 1024,
        ),
        name="pcp_streaming_ring_attention_compute_smoke",
    )(q, kv)


def _run_ring_streaming_attention(q, kv, *, sm_scale):
    mesh = jax.sharding.Mesh(jax.local_devices()[:PCP_SIZE], (AXIS,))
    fn = jax.jit(
        jax.shard_map(
            functools.partial(
                _ring_streaming_attention_call,
                pcp_size=PCP_SIZE,
                sm_scale=sm_scale,
            ),
            mesh=mesh,
            in_specs=(P(AXIS, None, None), P(AXIS, None, None, None)),
            out_specs=P(AXIS, None, None),
            check_vma=False,
        )
    )
    return fn(q, kv)


def _naive_attention(q, kv, *, sm_scale):
    q_np = np.asarray(q, dtype=np.float32)
    kv_np = np.asarray(kv, dtype=np.float32)
    k = kv_np[:, :, 0, :].reshape(PCP_SIZE * PAGE_SIZE, HEAD_DIM)
    v = kv_np[:, :, 1, :].reshape(PCP_SIZE * PAGE_SIZE, HEAD_DIM)
    out = np.zeros((PCP_SIZE, Q_TILE, HEAD_DIM), dtype=np.float32)
    for rank in range(PCP_SIZE):
        scores = q_np[rank] @ k.T * sm_scale
        scores = scores - np.max(scores, axis=1, keepdims=True)
        weights = np.exp(scores)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        out[rank] = weights @ v
    return out


def _require_tpu_devices(min_count, reason):
    devices = jax.local_devices()
    if len(devices) < min_count or devices[0].platform != "tpu":
        pytest.skip(reason)
    return devices


def test_ring_circulated_kv_can_be_consumed_with_online_softmax(release_jax_backend):
    _require_tpu_devices(
        PCP_SIZE,
        "PCP ring streaming compute smoke test requires four TPU devices.",
    )
    rng = np.random.default_rng(123)
    q = jnp.asarray(
        rng.normal(size=(PCP_SIZE, Q_TILE, HEAD_DIM)).astype(np.float32) * 0.1
    )
    kv = jnp.asarray(
        rng.normal(size=(PCP_SIZE, PAGE_SIZE, 2, HEAD_DIM)).astype(np.float32) * 0.1
    )
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = _run_ring_streaming_attention(q, kv, sm_scale=sm_scale)
    out.block_until_ready()

    expected = _naive_attention(
        jax.device_get(q), jax.device_get(kv), sm_scale=sm_scale
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(out)), expected, rtol=5e-4, atol=5e-5
    )
