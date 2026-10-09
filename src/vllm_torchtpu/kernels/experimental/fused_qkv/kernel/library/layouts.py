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

"""Mosaic layouts: how in-kernel values land in tiled refs.

Lane-broadcast scales: one f32 per row, stored as (rows, lanes) so that it
loads as plain vregs.
Strided row stores: view a 32-bit ref as rows of 128 lanes and write every
k-th row, a sublane-to-major relayout with no shuffles.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def align(x: int, a: int) -> int:
    """x rounded up to a multiple of a."""
    return -(-x // a) * a


# Lane-broadcast scales.


def broadcast_lanes(
    col: jax.Array,
) -> jax.Array:
    """The stored form of a per-row scale."""
    return jnp.broadcast_to(col, (col.shape[0], pltpu.get_tpu_info().num_lanes))


def widen(scale: jax.Array, width: int) -> jax.Array:
    """A lane-broadcast scale, repeated to `width` columns."""
    return jnp.concatenate([scale] * (width // pltpu.get_tpu_info().num_lanes), axis=1)


# Strided row stores.


def strided_store(
    ref: jax.Ref,  # (..., cols) 32-bit
    start: int,
    stride: int,
    val: jax.Array,
    col: int = 0,
) -> None:
    """ref[start::stride][:n, col:col + width] = val.

    Mosaic tiles the ref's rows as (1, 128): each row's cols / 128 lane chunks
    are consecutive 128-lane rows of the flat ref. So each lane chunk of val is
    one strided row store. col and width are multiples of 128.
    """
    num_lanes = pltpu.get_tpu_info().num_lanes
    folds = ref.shape[-1] // num_lanes
    flat = ref.reshape(-1, num_lanes)
    for c in range(val.shape[1] // num_lanes):
        rows = pl.ds(
            start * folds + col // num_lanes + c,
            val.shape[0],
            stride=stride * folds,
        )
        flat[rows] = val[:, c * num_lanes : (c + 1) * num_lanes]


def pack_heads(heads: list[jax.Array]) -> jax.Array:
    """Packs 2 bf16 (or 4 fp8) heads into u32s, heads[0] in the low bits."""
    bits = jax.dtypes.itemsize_bits(heads[0].dtype)
    packed = None
    for i, y in enumerate(heads):
        if bits == 16:
            # bf16 is the high half of its f32.
            y = jax.lax.bitcast_convert_type(y.astype(jnp.float32), jnp.uint32) >> 16
        else:
            y = jax.lax.bitcast_convert_type(y, jnp.dtype(f"uint{bits}"))
            y = y.astype(jnp.uint32)
        y = y << (i * bits)
        packed = y if packed is None else packed | y
    return packed


def strided_store_heads(
    ref: jax.Ref,  # (tile, num_groups, 32 // bits, align(D, 128)) bf16/fp8
    group: int,
    packed: jax.Array,
) -> None:
    """Stores one group of packed heads into ref.

    A group is the 32 // bits heads packed into each u32: 2 bf16 or 4 fp8. Per
    token the groups are back to back, so group `group` is every num_groups-th
    row. D is zero-padded to the ref's lanes.
    """
    tile_size, num_groups, _, padded_d = ref.shape
    if packed.shape[-1] < padded_d:
        pad = jnp.zeros((tile_size, padded_d - packed.shape[-1]), jnp.uint32)
        packed = jnp.concatenate([packed, pad], -1)
    u32_ref = ref.bitcast(jnp.uint32)  # (tile, num_groups, 1, align(D, 128))
    strided_store(u32_ref, start=group, stride=num_groups, val=packed)


def store_head_pair(
    ref: jax.Ref,  # (tile, H, D) bf16, H <= 8
    rows: pl.Slice,
    pair: int,
    val: jax.Array,
) -> None:
    """ref[rows, 2 * pair : 2 * pair + 2] = val, heads side by side in val.

    Mosaic tiles (tile, H, D) bf16 with H <= 8 as (H, 128)(2, 1), one tile per
    token: heads 2g and 2g + 1 share u32 row g. So the pair packs into one u32
    array that goes to every (H / 2)-th u32 row.
    """
    _, num_heads, d = ref.shape
    heads = [val[:, :d].astype(ref.dtype), val[:, d:].astype(ref.dtype)]
    flat = ref.bitcast(jnp.uint32).reshape(-1, d)  # (tile * H / 2, D)
    groups = num_heads // 2
    flat[pl.ds(rows.start * groups + pair, rows.size, stride=groups)] = pack_heads(
        heads
    )
