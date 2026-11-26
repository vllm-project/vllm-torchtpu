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
"""Shared helpers for small Pallas collective kernels."""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANES = 128
MESH = pl.DeviceIdType.MESH


def peer(j, axis, dict_ids):
    return {axis: jnp.int32(j)} if dict_ids else (jnp.int32(j),)


def barrier(num_cores, axis, dict_ids):
    sem = pltpu.get_barrier_semaphore()
    for j in range(num_cores):
        pl.semaphore_signal(
            sem, 1, device_id=peer(j, axis, dict_ids), device_id_type=MESH
        )
    pl.semaphore_wait(sem, num_cores)


def wait_bytes_of(ref_slice, sem):
    pltpu.make_async_copy(ref_slice, ref_slice, sem).wait()


def push(src, dst, send_sem, recv_sem, j, axis, dict_ids):
    pltpu.make_async_remote_copy(
        src_ref=src,
        dst_ref=dst,
        send_sem=send_sem,
        recv_sem=recv_sem,
        device_id=peer(j, axis, dict_ids),
        device_id_type=MESH,
    ).start()


def split_rank(refs, rank_operand, axis):
    """Return `(core_rank, remaining_refs)`."""
    if rank_operand:
        rank_ref, *refs = refs
        return rank_ref[0], refs
    return lax.axis_index(axis), refs


def sum_slots(slots, n):
    """Float32 sum over the leading axis of a VMEM ref."""
    acc = slots[0].astype(jnp.float32)
    for j in range(1, n):
        acc = acc + slots[j].astype(jnp.float32)
    return acc


def call(kernel, x, out_rows, rank, scratch, collective_id, name):
    rank_specs = [pl.BlockSpec(memory_space=pltpu.SMEM)] if rank is not None else []
    args = (() if rank is None else (rank.astype(jnp.int32),)) + (x,)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((out_rows, x.shape[1]), x.dtype),
        in_specs=[*rank_specs, pl.BlockSpec()],
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=name,
    )(*args)
