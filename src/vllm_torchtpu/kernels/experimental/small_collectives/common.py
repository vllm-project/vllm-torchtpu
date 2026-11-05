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
"""Shared ring topology, barrier semaphore, and VMEM scratch utilities."""

import contextlib

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANES = 128
MESH = pl.DeviceIdType.MESH

# Axis names of the mesh the kernels are traced under, in mesh order, when it has more
# axes than the collective's own: a mesh device id names every axis, so the peer id
# carries this core's index on each other axis. () is a one-axis mesh.
_MESH_AXES: tuple = ()


@contextlib.contextmanager
def mesh_axes(axes):
    """Trace the enclosed collective calls for a mesh with these axes (mesh order)."""
    global _MESH_AXES
    prev = _MESH_AXES
    _MESH_AXES = tuple(axes)
    try:
        yield
    finally:
        _MESH_AXES = prev


def peer(j, axis, dict_ids):
    others = [a for a in _MESH_AXES if a != axis]
    if dict_ids:
        ids = {axis: jnp.int32(j)}
        for a in others:
            ids[a] = lax.axis_index(a)
        return ids
    if not others:
        return (jnp.int32(j),)
    return tuple(jnp.int32(j) if a == axis else lax.axis_index(a) for a in _MESH_AXES)


def barrier(num_cores, axis, dict_ids):
    """Every core signals every core once, itself included, and waits for all.

    One barrier per call is enough. A core finishes call k only after every
    peer's data of call k has landed, and a peer sends that data only after
    passing its own barrier of call k, so no core can signal call k + 1 while
    another still waits at the barrier of call k.
    """
    sem = pltpu.get_barrier_semaphore()
    for j in range(num_cores):
        pl.semaphore_signal(
            sem, 1, device_id=peer(j, axis, dict_ids), device_id_type=MESH
        )
    pl.semaphore_wait(sem, num_cores)


def barrier_peers(me, xor_masks, axis, dict_ids):
    """Every core signals its direct XOR communication peers ``me ^ m`` once
    and waits for all of them. Sufficient whenever the set of direct DMA peers
    is symmetric (`b == a ^ m <=> a == b ^ m`).
    """
    sem = pltpu.get_barrier_semaphore()
    for m in xor_masks:
        pl.semaphore_signal(
            sem,
            1,
            device_id=peer(me ^ jnp.int32(m), axis, dict_ids),
            device_id_type=MESH,
        )
    pl.semaphore_wait(sem, len(xor_masks))


def wait_bytes_of(ref_slice, sem):
    """Block until ``sem`` has counted the bytes of ``ref_slice``."""
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
    """This core's index and the remaining refs."""
    if rank_operand:
        rank_ref, *refs = refs
        return rank_ref[0], refs
    return lax.axis_index(axis), refs


def sum_slots(slots, n):
    """Float32 sum over the leading axis of a VMEM ref, rounded once."""
    acc = slots[0].astype(jnp.float32)
    for j in range(1, n):
        acc = acc + slots[j].astype(jnp.float32)
    return acc


def normalize_rank_of_core(rank_of_core, n: int) -> tuple[int, ...]:
    """Validate ``rank_of_core`` as a permutation of ``0..n - 1``."""
    if rank_of_core is None:
        return tuple(range(n))
    order = tuple(int(r) for r in rank_of_core)
    if len(order) != n or sorted(order) != list(range(n)):
        raise ValueError(
            f"rank_of_core {order} must be a permutation of 0..{n - 1}"
        )
    return order


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
