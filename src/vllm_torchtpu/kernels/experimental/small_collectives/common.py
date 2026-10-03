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
"""Pieces shared by the small all-reduce, reduce-scatter, all-gather, and
all-to-all Pallas kernels.

Under the torch bridge a core's own index arrives as data (an int32 [1] in
SMEM): ``lax.axis_index`` lowers to a partition-id the export path's SPMD
partitioner refuses. Peer addresses then take the dict form keyed by the axis
name, and waits go on local descriptors of the same bytes, since the bridge
refuses a wait on a descriptor that carries a peer address.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANES = 128
SUBLANES = 8
MESH = pl.DeviceIdType.MESH


def round_up(n: int, multiple: int) -> int:
    """Round a positive integer ``n`` up to the next multiple of ``multiple``."""
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    if n <= 0:
        return 0
    return -(-n // multiple) * multiple


def pad_rows(x: jax.Array, multiple: int = SUBLANES) -> jax.Array:
    """Pad axis -2 (rows) of a 2D [rows, cols] or 3D [blocks, rows, cols]
    array to a multiple of ``multiple`` (default ``SUBLANES = 8``)."""
    rows = x.shape[-2]
    padded = round_up(rows, multiple)
    if padded == rows:
        return x
    pad_spec = [(0, 0)] * x.ndim
    pad_spec[-2] = (0, padded - rows)
    return jnp.pad(x, pad_spec)


def pad_cols(x: jax.Array, multiple: int = LANES) -> jax.Array:
    """Pad axis -1 (columns) of ``x`` to a multiple of ``multiple``
    (default ``LANES = 128``)."""
    cols = x.shape[-1]
    padded = round_up(cols, multiple)
    if padded == cols:
        return x
    pad_spec = [(0, 0)] * x.ndim
    pad_spec[-1] = (0, padded - cols)
    return jnp.pad(x, pad_spec)


def resolve_axis_and_cores(
    axis: str | None, axis_name: str | None, num_cores: int | None
) -> tuple[str, int]:
    """Resolve ``axis`` / ``axis_name`` and ``num_cores`` inside a shard_map."""
    resolved_axis = axis if axis is not None else axis_name
    if not resolved_axis:
        raise ValueError("Must specify a non-empty mesh axis via `axis` or `axis_name`")
    if num_cores is None:
        num_cores = int(lax.axis_size(resolved_axis))
    if num_cores <= 0:
        raise ValueError(f"num_cores must be positive, got {num_cores}")
    return resolved_axis, int(num_cores)


def peer(j, axis, dict_ids):
    return {axis: jnp.int32(j)} if dict_ids else (jnp.int32(j),)


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


def call(kernel, x, out_rows, rank, scratch, collective_id, name, out_cols=None):
    if out_cols is None:
        out_cols = x.shape[1]
    rank_specs = [pl.BlockSpec(memory_space=pltpu.SMEM)] if rank is not None else []
    args = (() if rank is None else (rank.astype(jnp.int32),)) + (x,)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((out_rows, out_cols), x.dtype),
        in_specs=[*rank_specs, pl.BlockSpec()],
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=name,
    )(*args)
