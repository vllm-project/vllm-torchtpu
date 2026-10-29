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
"""Standalone in-kernel ICI `all_gather` and `reduce_scatter` primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..errors import MegakernelConfigError
from ..ops import BF16, F32


@dataclass(frozen=True)
class CollectiveHandle:
    """In-flight DMA descriptors and target buffer for group collectives."""

    descs: tuple[Any, ...]
    recv_ref: Any
    group_size: int
    mode: str = "stack"


def _validate_group(group_size: int, stride: int) -> None:
    if (
        group_size < 1
        or (group_size & (group_size - 1)) != 0
        or stride < 1
        or (stride & (stride - 1)) != 0
    ):
        raise MegakernelConfigError(
            f"group_size={group_size} and stride={stride} must be positive powers of 2"
        )


def all_gather_scratch(
    group_size: int, shape: tuple[int, int], dtype: Any = BF16, *, prefix: str = "ag"
) -> dict[str, Any]:
    """Allocate VMEM buffers and DMA semaphores for standalone `all_gather`."""
    _validate_group(group_size, 1)
    peers = max(group_size - 1, 1)
    return {
        f"{prefix}_send": pltpu.VMEM(shape, dtype),
        f"{prefix}_recv": pltpu.VMEM((group_size, *shape), dtype),
        f"{prefix}_ssem": pltpu.SemaphoreType.DMA((peers,)),
        f"{prefix}_rsem": pltpu.SemaphoreType.DMA((peers,)),
    }


def start_all_gather(
    x: jax.Array,
    send_ref: Any,
    recv_ref: Any,
    ssem: Any,
    rsem: Any,
    me: Any,
    group_size: int,
    *,
    stride: int = 1,
    mode: str = "stack",
) -> CollectiveHandle:
    """Start non-blocking group `all_gather` across `group_size` peers."""
    _validate_group(group_size, stride)
    if mode not in ("stack", "concat_rows", "concat_cols"):
        raise MegakernelConfigError(f"unsupported all_gather mode={mode!r}")
    gi = lax.rem(me // stride, group_size)
    send_ref[...] = x.astype(send_ref.dtype)
    recv_ref[gi] = send_ref[...]
    descs = tuple(
        pltpu.make_async_remote_copy(
            src_ref=send_ref,
            dst_ref=recv_ref.at[gi],
            send_sem=ssem.at[off - 1],
            recv_sem=rsem.at[off - 1],
            device_id=(me ^ (off * stride),),
            device_id_type=pl.DeviceIdType.MESH,
        )
        for off in range(1, group_size)
    )
    for d in descs:
        d.start()
    return CollectiveHandle(
        descs=descs, recv_ref=recv_ref, group_size=group_size, mode=mode
    )


def wait_all_gather(handle: CollectiveHandle) -> jax.Array:
    """Complete `start_all_gather` and format gathered slices per `handle.mode`."""
    for d in handle.descs:
        d.wait_recv()
    for d in handle.descs:
        d.wait_send()
    if handle.mode == "stack":
        return handle.recv_ref[...]
    slices = [handle.recv_ref[g] for g in range(handle.group_size)]
    return jnp.concatenate(slices, axis=0 if handle.mode == "concat_rows" else -1)


def all_gather(
    x: jax.Array,
    send_ref: Any,
    recv_ref: Any,
    ssem: Any,
    rsem: Any,
    me: Any,
    group_size: int,
    *,
    stride: int = 1,
    mode: str = "stack",
) -> jax.Array:
    """Synchronous in-kernel `all_gather` over a power-of-2 XOR peer group."""
    return wait_all_gather(
        start_all_gather(
            x, send_ref, recv_ref, ssem, rsem, me, group_size, stride=stride, mode=mode
        )
    )


def reduce_scatter_scratch(
    group_size: int,
    shape: tuple[int, int],
    dtype: Any = BF16,
    *,
    scatter_axis: int = -1,
    prefix: str = "rs",
) -> dict[str, Any]:
    """Allocate VMEM buffers and DMA semaphores for standalone `reduce_scatter`."""
    _validate_group(group_size, 1)
    rows, cols = shape
    if scatter_axis == -1:
        if cols % group_size != 0 or (cols // group_size) % 128 != 0:
            raise MegakernelConfigError(
                f"reduce_scatter cols={cols}/{group_size} must align to 128"
            )
        shard_shape = (rows, cols // group_size)
    elif scatter_axis == 0:
        if rows % group_size != 0 or (rows // group_size) % 8 != 0 or cols % 128 != 0:
            raise MegakernelConfigError(
                f"reduce_scatter rows={rows}/{group_size} must align to (8, 128)"
            )
        shard_shape = (rows // group_size, cols)
    else:
        raise MegakernelConfigError(f"unsupported scatter_axis={scatter_axis}")
    peers = max(group_size - 1, 1)
    return {
        f"{prefix}_send": pltpu.VMEM((group_size, *shard_shape), dtype),
        f"{prefix}_recv": pltpu.VMEM((group_size, *shard_shape), dtype),
        f"{prefix}_ssem": pltpu.SemaphoreType.DMA((peers,)),
        f"{prefix}_rsem": pltpu.SemaphoreType.DMA((peers,)),
    }


def start_reduce_scatter(
    x: jax.Array,
    send_ref: Any,
    recv_ref: Any,
    ssem: Any,
    rsem: Any,
    me: Any,
    group_size: int,
    *,
    stride: int = 1,
    scatter_axis: int = -1,
) -> CollectiveHandle:
    """Start non-blocking group `reduce_scatter` across `group_size` peers."""
    _validate_group(group_size, stride)
    rows, cols = x.shape
    gi = lax.rem(me // stride, group_size)
    if scatter_axis == -1:
        cw = cols // group_size
        for j in range(group_size):
            send_ref[j] = x[:, j * cw : (j + 1) * cw].astype(send_ref.dtype)
    elif scatter_axis == 0:
        rw = rows // group_size
        for j in range(group_size):
            send_ref[j] = x[j * rw : (j + 1) * rw, :].astype(send_ref.dtype)
    else:
        raise MegakernelConfigError(f"unsupported scatter_axis={scatter_axis}")
    recv_ref[gi] = send_ref[gi]
    descs = tuple(
        pltpu.make_async_remote_copy(
            src_ref=send_ref.at[gi ^ off],
            dst_ref=recv_ref.at[gi],
            send_sem=ssem.at[off - 1],
            recv_sem=rsem.at[off - 1],
            device_id=(me ^ (off * stride),),
            device_id_type=pl.DeviceIdType.MESH,
        )
        for off in range(1, group_size)
    )
    for d in descs:
        d.start()
    return CollectiveHandle(
        descs=descs, recv_ref=recv_ref, group_size=group_size, mode="sum"
    )


def wait_reduce_scatter(handle: CollectiveHandle) -> jax.Array:
    """Complete `start_reduce_scatter` and return the local F32 reduced shard."""
    for d in handle.descs:
        d.wait_recv()
    for d in handle.descs:
        d.wait_send()
    return jnp.sum(handle.recv_ref[...].astype(F32), axis=0)


def reduce_scatter(
    x: jax.Array,
    send_ref: Any,
    recv_ref: Any,
    ssem: Any,
    rsem: Any,
    me: Any,
    group_size: int,
    *,
    stride: int = 1,
    scatter_axis: int = -1,
) -> jax.Array:
    """Synchronous in-kernel `reduce_scatter` returning the F32 reduced shard."""
    return wait_reduce_scatter(
        start_reduce_scatter(
            x,
            send_ref,
            recv_ref,
            ssem,
            rsem,
            me,
            group_size,
            stride=stride,
            scatter_axis=scatter_axis,
        )
    )
