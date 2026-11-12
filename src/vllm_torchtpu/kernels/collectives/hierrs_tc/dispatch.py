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
"""Picks the hierrs_tc kernel for a reduce-scatter in EP-rank order."""

import jax
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.collectives.hierrs_tc.kernel_stream import (
    reduce_scatter_stream_local,
    supports_stream,
)
from vllm_torchtpu.kernels.collectives.hierrs_tc.wrapper import (
    hierarchical_reduce_scatter_local,
)

# Input bytes per device from which the row-streaming kernel serves a call.
# Wall-clock per call on tpu7x 2x2x1 at hidden 2048 (us, micro-batched vs
# row-streaming): 104.5 / 103.9 at 8 MiB, 198.6 / 198.4 at 16 MiB, 245.3 /
# 240.9 at 24 MiB, 298.0 / 292.7 at 32 MiB. They tie at 16 MiB, and from there
# the row-streaming kernel is also ahead on memory: it streams through VMEM,
# where the micro-batched one needs 3 bytes per element of work buffers.
STREAM_MIN_BYTES = 16 << 20


def uses_stream(rows: int, hidden: int, itemsize: int) -> bool:
    """Whether `rows` x `hidden` per device goes to the row-streaming kernel."""
    return rows * hidden * itemsize >= STREAM_MIN_BYTES and supports_stream(
        rows, hidden, itemsize
    )


def make_reduce_scatter_fn(mesh, order: tuple[int, ...], *, axis_name: str):
    """Reduce-scatter in which mesh index i gets the block of rank `order[i]`.

    Both kernels read the blocks in that order themselves, so the input is
    never copied to reorder it.
    """
    num_devices = len(order)

    def local(part: jax.Array) -> jax.Array:
        rows, hidden = part.shape
        if uses_stream(rows, hidden, part.dtype.itemsize):
            return reduce_scatter_stream_local(part, axis_name=axis_name, order=order)
        return hierarchical_reduce_scatter_local(
            part, num_devices=num_devices, axis_name=axis_name, order=order
        )

    def reduce_scatter(x: jax.Array) -> jax.Array:
        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=P(axis_name, None),
            out_specs=P(axis_name, None),
            check_vma=False,
        )(x)

    return reduce_scatter
