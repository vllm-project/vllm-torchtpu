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
"""Shared 1-row (n_valid == 1) 1-round remote-copy Pallas all-reduce primitives.

When a decode step has a single valid row (n_valid == 1), row 0 of [m, cols] is
folded into one bf16 row tile [tile_rows, cols // tile_rows], pushed directly to
slot `me` on every peer rank in a single round, summed in float32 in rank order,
rounded once to the output dtype, and unfolded back into row 0 (with rows >= 1
zeroed to +0.0). Because each element accumulates the same float32 sum in rank
order as the two-shot reduce-scatter + all-gather path, row 0 is bit-identical.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BF16 = jnp.bfloat16
F32 = jnp.float32
_MESH = pl.DeviceIdType.MESH


def bf16_tiling(info=None) -> tuple[int, int]:
    """Return (sublane_tile_rows, num_lanes) for bf16 from TPU hardware info."""
    if info is None:
        info = pltpu.get_tpu_info()
    return int(info.get_sublane_tiling(BF16)), int(info.num_lanes)


def one_row_supported(cols: int, info=None) -> bool:
    """Whether one row of `cols` columns folds into whole bf16 lane groups."""
    tile_rows, lanes = bf16_tiling(info)
    return cols > 0 and cols % (tile_rows * lanes) == 0


def row_tile_width(cols: int, tile_rows: int) -> int:
    """Column width after folding one row of `cols` into `tile_rows` rows."""
    return cols // tile_rows


def scratch_shape(num_ranks: int, cols: int, tile_rows: int, dtype=BF16):
    """Per-sender VMEM arrival slots `[num_ranks, tile_rows, cols // tile_rows]`."""
    return pltpu.VMEM((num_ranks, tile_rows, row_tile_width(cols, tile_rows)), dtype)


def fold_row(row0: jax.Array, tile_rows: int) -> jax.Array:
    """Fold bf16-rounded row 0 (`[1, cols]`) into `[tile_rows, cols // tile_rows]`."""
    w = row_tile_width(row0.shape[1], tile_rows)
    p0 = row0.astype(BF16).astype(F32)
    rows = lax.broadcasted_iota(jnp.int32, (tile_rows, w), 0)
    q1 = jnp.zeros((tile_rows, w), F32)
    for j in range(tile_rows):
        part = jnp.broadcast_to(p0[:, j * w : (j + 1) * w], (tile_rows, w))
        q1 = jnp.where(rows == j, part, q1)
    return q1.astype(BF16)


def sum_slots_f32(slots, num_ranks: int, out_dtype=BF16) -> jax.Array:
    """Sum `num_ranks` arrival slots in float32 in rank order and round once."""
    acc = slots[0].astype(F32)
    for r in range(1, num_ranks):
        acc = acc + slots[r].astype(F32)
    return acc.astype(out_dtype).astype(F32)


def unfold_row_into(total_f32: jax.Array, out_ref, tile_rows: int) -> None:
    """Unfold `[tile_rows, cols // tile_rows]` into row 0 of `out_ref`; rows >= 1 +0."""
    m, cols = out_ref.shape
    w = row_tile_width(cols, tile_rows)
    first = lax.broadcasted_iota(jnp.int32, (m, w), 0) == 0
    for j in range(tile_rows):
        part = jnp.broadcast_to(total_f32[j : j + 1, :], (m, w))
        out_ref[:, pl.ds(j * w, w)] = jnp.where(first, part, 0.0).astype(out_ref.dtype)


def one_row_exchange(
    row0: jax.Array,
    out_ref,
    slots,
    me,
    send_sem,
    recv_sem,
    *,
    num_ranks: int,
    tile_rows: int,
    axis: str | None = None,
    barrier=None,
) -> None:
    """Fold `row0`, push to slot `me` on every peer, sum in rank order, and unfold.

    Supports both Pallas mesh/semaphore conventions used by decode kernels:
      - `axis is None`: 1-D tuple `device_id=(t,)` with per-peer `(num_ranks,)`
        DMA semaphores (`moe_tp_decode.all_reduce`).
      - `axis is not None`: named-axis dict `device_id={axis: j}` with scalar
        byte-counted DMA semaphores (`fp8_oproj_all_reduce`).
    """
    slots.at[me][...] = fold_row(row0, tile_rows)
    if barrier is not None:
        pl.semaphore_wait(barrier, num_ranks)

    if axis is None:
        sends = []
        for step in range(1, num_ranks):
            t = lax.rem(me + step, num_ranks)
            copy = pltpu.make_async_remote_copy(
                src_ref=slots.at[me],
                dst_ref=slots.at[me],
                send_sem=send_sem.at[step],
                recv_sem=recv_sem.at[me],
                device_id=(t,),
                device_id_type=_MESH,
            )
            copy.start()
            sends.append(copy)
        for step in range(1, num_ranks):
            t = lax.rem(me + (num_ranks - step), num_ranks)
            pltpu.make_async_copy(slots.at[t], slots.at[t], recv_sem.at[t]).wait_recv()
        total = sum_slots_f32(slots, num_ranks, out_ref.dtype)
        unfold_row_into(total, out_ref, tile_rows)
        for copy in sends:
            copy.wait_send()
        return

    for j in range(num_ranks):

        @pl.when(jnp.int32(j) != me)
        def _(j=j):
            pltpu.make_async_remote_copy(
                src_ref=slots.at[me],
                dst_ref=slots.at[me],
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id={axis: jnp.int32(j)},
                device_id_type=_MESH,
            ).start()

    peers = slots.at[pl.ds(0, num_ranks - 1)]
    pltpu.make_async_copy(peers, peers, recv_sem).wait()
    total = sum_slots_f32(slots, num_ranks, out_ref.dtype)
    unfold_row_into(total, out_ref, tile_rows)
    pltpu.make_async_copy(peers, peers, send_sem).wait()
