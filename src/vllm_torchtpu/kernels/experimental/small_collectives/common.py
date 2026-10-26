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
"""Pieces shared by the small collective kernels.

Under the torch bridge a core's own index arrives as data (an int32 [1] in
SMEM): ``lax.axis_index`` lowers to a partition-id the export path's SPMD
partitioner refuses. Peer addresses then take the dict form keyed by the axis
name, and waits go on local descriptors of the same bytes, since the bridge
refuses a wait on a descriptor that carries a peer address.

A kernel may also run over a *subgroup* of its mesh axis (a TP group inside the
expert-parallel mesh): ``peers`` (int32 [n] in SMEM, as data) then holds the
mesh index of each group member, and ``rank`` is the caller's position in the
group. torch_tpu needs every Pallas kernel of one compiled module on the same
device assignment, so a TP-group collective cannot get a mesh of its own next
to the fused MoE kernel's.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANES = 128
MESH = pl.DeviceIdType.MESH


class Group:
    """Device ids of the members of the group a kernel runs over.

    Member j is mesh index ``peers_ref[j]`` with a peer table, else mesh index
    j (the group is the axis)."""

    def __init__(self, n, axis, dict_ids, peers_ref=None):
        self.n = n
        self.axis = axis
        self.dict_ids = dict_ids or peers_ref is not None
        self.peers_ref = peers_ref

    def device_id(self, j):
        index = jnp.int32(j) if self.peers_ref is None else self.peers_ref[j]
        return {self.axis: index} if self.dict_ids else (index,)


def barrier(group):
    """Every member signals every member once, itself included, and waits for
    all.

    One barrier per call is enough. A core finishes call k only after every
    peer's data of call k has landed, and a peer sends that data only after
    passing its own barrier of call k, so no core can signal call k + 1 while
    another still waits at the barrier of call k. Members of other groups
    never signal this core, so disjoint groups barrier independently.
    """
    sem = pltpu.get_barrier_semaphore()
    for j in range(group.n):
        pl.semaphore_signal(sem, 1, device_id=group.device_id(j), device_id_type=MESH)
    pl.semaphore_wait(sem, group.n)


def wait_bytes_of(ref_slice, sem):
    """Block until ``sem`` has counted the bytes of ``ref_slice``."""
    pltpu.make_async_copy(ref_slice, ref_slice, sem).wait()


def push(src, dst, send_sem, recv_sem, j, group):
    pltpu.make_async_remote_copy(
        src_ref=src,
        dst_ref=dst,
        send_sem=send_sem,
        recv_sem=recv_sem,
        device_id=group.device_id(j),
        device_id_type=MESH,
    ).start()


def split_refs(refs, n, axis, rank_operand, peers_operand):
    """This core's position in the group, the group, and the remaining refs."""
    refs = list(refs)
    me = refs.pop(0)[0] if rank_operand else lax.axis_index(axis)
    peers_ref = refs.pop(0) if peers_operand else None
    return me, Group(n, axis, rank_operand, peers_ref), refs


def sum_slots(slots, n):
    """Float32 sum over the leading axis of a VMEM ref, rounded once."""
    acc = slots[0].astype(jnp.float32)
    for j in range(1, n):
        acc = acc + slots[j].astype(jnp.float32)
    return acc


def scalar_operands(rank, peers):
    """The SMEM operands (and their specs) for ``rank`` and ``peers``."""
    if peers is not None and rank is None:
        raise ValueError("a peer table needs the core's group position as data")
    args = [a.reshape(-1).astype(jnp.int32) for a in (rank, peers) if a is not None]
    return args, [pl.BlockSpec(memory_space=pltpu.SMEM)] * len(args)


def compiler_params(collective_id, skip_device_barrier=False):
    """CompilerParams for a communicating kernel. ``skip_device_barrier``
    drops the device barrier Mosaic otherwise runs at the entry of every
    kernel with remote DMAs: safe for these kernels, whose own group barrier
    (``barrier``) already proves every member entered the call and allocated
    its scratch before any byte is pushed into it."""
    if skip_device_barrier:
        return pltpu.CompilerParams(
            collective_id=collective_id, skip_device_barrier=True
        )
    return pltpu.CompilerParams(collective_id=collective_id)


def call(
    kernel,
    x,
    out_rows,
    rank,
    peers,
    scratch,
    collective_id,
    name,
    skip_device_barrier=False,
):
    args, specs = scalar_operands(rank, peers)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((out_rows, x.shape[1]), x.dtype),
        in_specs=[*specs, pl.BlockSpec()],
        scratch_shapes=scratch,
        compiler_params=compiler_params(collective_id, skip_device_barrier),
        name=name,
    )(*args, x)
