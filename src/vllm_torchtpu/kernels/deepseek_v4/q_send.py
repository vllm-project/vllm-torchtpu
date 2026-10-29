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
"""Send the current chiplet's q heads for the neighboring chiplet's tokens
to the neighboring chiplet, as a start and a wait.

``start_q_send`` takes a chiplet's ``[2, T, H, D]`` block (both chiplets' tokens,
the current chiplet's heads), barriers with the neighboring chiplet, and
starts one remote DMA of block ``1 - chiplet_index`` straight into the
neighboring chiplet's receive buffer. ``wait_q_send``
waits for the local read and the incoming write to finish.

Both buffers stay in HBM, and the start returns its source as an output aliased
to it, which the wait takes: this forces the source buffer to be kept alive until
wait.

The start and wait are marked as an async collective pair (``async_type``), so
XLA's latency-hiding scheduler puts independent work between them.
"""

import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.xla_metadata import set_xla_metadata

_HBM = pl.BlockSpec(memory_space=pltpu.HBM)
_SEM = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
# The start and wait only issue and wait on DMAs.
_VMEM_LIMIT_BYTES = 1 << 20
# The transfer time the scheduler should hide, in ns.
_LATENCY_NS = "20000"


def _barrier(neighbor) -> None:
    """Wait until both chiplets have entered the kernel so both buffers exist."""
    barrier = pltpu.get_barrier_semaphore()
    pl.semaphore_signal(
        barrier, 1, device_id=(neighbor,), device_id_type=pl.DeviceIdType.MESH
    )
    pl.semaphore_wait(barrier, 1)


def _wait_bytes_of(ref_slice, sem) -> None:
    """Block until ``sem`` has counted the bytes of ``ref_slice`` using a local
    descriptor (avoids carrying a remote ``device_id`` on ``dma_wait``).
    """
    pltpu.make_async_copy(ref_slice, ref_slice, sem).wait()


def _start_kernel(
    neighbor_mesh_index_ref,  # SMEM int32 [1]: the neighboring chiplet's mesh index
    chiplet_index_ref,  # SMEM int32 [1]: the current chiplet's index (0 or 1)
    src_ref,
    out_ref,
    send_sem,
    recv_sem,
    src_out_ref,
):
    """``src`` [2, T, H, D]: start one remote DMA of block ``1 - chiplet_index``.

    Token-major per block, so the neighboring chiplet's block is one contiguous
    [T, H, D] copy that lands in the layout the consumer reads.
    """
    del src_out_ref
    neighbor = neighbor_mesh_index_ref[0]
    chiplet_index = chiplet_index_ref[0]

    _barrier(neighbor)
    for h in (0, 1):  # static per branch: the DMA source is a static slice

        @pl.when(chiplet_index == h)
        def _(h=h):
            pltpu.make_async_remote_copy(
                src_ref=src_ref.at[1 - h],
                dst_ref=out_ref,
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=(neighbor,),
                device_id_type=pl.DeviceIdType.MESH,
            ).start()


def _wait_kernel(src_ref, out_ref, send_sem, recv_sem, aliased_out_ref):
    """Wait for the block DMA's local read and remote write to complete."""
    del src_ref, aliased_out_ref  # src: kept alive until here; out: aliased
    _wait_bytes_of(out_ref, send_sem)
    _wait_bytes_of(out_ref, recv_sem)


def start_q_send(
    qh, neighbor_mesh_index, chiplet_index, *, collective_id, name="q_send_start"
):
    """Inside ``shard_map``: start the q send, [2, T, H, D] -> [T, H, D].

    ``neighbor_mesh_index`` and ``chiplet_index`` are ``[1]`` int32. Returns the
    handle ``(qh, out, send_sem, recv_sem)``: ``qh`` is the start's aliased
    output, to be read instead of the input, and ``out`` the receive buffer.
    """
    shape = qh.shape[1:]
    with set_xla_metadata(async_type="collective-start", latency_metadata=_LATENCY_NS):
        out, send_sem, recv_sem, qh = pl.pallas_call(
            _start_kernel,
            out_shape=(
                jax.ShapeDtypeStruct(shape, qh.dtype),
                pltpu.SemaphoreType.DMA(()),
                pltpu.SemaphoreType.DMA(()),
                jax.ShapeDtypeStruct(qh.shape, qh.dtype),
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2,
                grid=(1,),
                in_specs=(_HBM,),
                out_specs=(_HBM, _SEM, _SEM, _HBM),
            ),
            # Operand 2 (after the two scalar-prefetch operands) is the source.
            input_output_aliases={2: 3},
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id, vmem_limit_bytes=_VMEM_LIMIT_BYTES
            ),
            cost_estimate=pl.CostEstimate(
                flops=0,
                transcendentals=0,
                bytes_accessed=2 * math.prod(shape) * qh.dtype.itemsize,
            ),
            name=name,
        )(
            jnp.asarray(neighbor_mesh_index, jnp.int32),
            jnp.asarray(chiplet_index, jnp.int32),
            qh,
        )
    return qh, out, send_sem, recv_sem


def wait_q_send(handle, *, collective_id, name="q_send_wait"):
    """Inside ``shard_map``: finish the q send started by ``start_q_send``.

    Returns the received block: the neighboring chiplet's heads for the current
    chiplet's tokens, [T, H, D].
    """
    qh, out, send_sem, recv_sem = handle
    with set_xla_metadata(async_type="collective-done", latency_metadata=_LATENCY_NS):
        return pl.pallas_call(
            _wait_kernel,
            out_shape=jax.ShapeDtypeStruct(out.shape, out.dtype),
            in_specs=(_HBM, _HBM, _SEM, _SEM),
            out_specs=_HBM,
            input_output_aliases={1: 0},
            # An async done must carry its start's collective_id; the wait has no
            # barrier of its own.
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=_VMEM_LIMIT_BYTES,
                collective_id=collective_id,
                allow_collective_id_without_custom_barrier=True,
            ),
            cost_estimate=pl.CostEstimate(flops=0, transcendentals=0, bytes_accessed=0),
            name=name,
        )(qh, out, send_sem, recv_sem)
