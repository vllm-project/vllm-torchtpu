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
"""Typed views of raw byte blocks inside Pallas TPU kernels.

A block ref holds bytes in the layout of its owning buffer (e.g. one
kernel block of the unified KV pool). Consumers that pack a different
element type into those bytes access them through a typed view: Mosaic
``ref.bitcast`` (which rescales the second-minor dim by the element-size
ratio) plus an optional narrowing reshape, both applied to the *ref*
before the load. Reshaping the ref keeps every access one whole vector
register; splitting lanes after the load costs a lane-crossing relayout
instead (measured on v7: 366 bundles / 448 vrot for the post-load split
against 87 / 0 for the ref-level reshape, on a 256 KiB pool block).
The pool gather/scatter kernels and the GDN V3 state seam share these
helpers so the bytes they exchange are identical.
"""

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu


def _typed_view(
    block_ref,
    view_dtype,
    lane_split: int,
    row0: int = 0,
    rows: int | None = None,
):
    """Raw block ref as a ``(rows, lanes // lane_split)`` view_dtype ref."""
    if jnp.dtype(block_ref.dtype) != jnp.dtype(view_dtype):
        block_ref = block_ref.bitcast(jnp.dtype(view_dtype))
    ref = block_ref.reshape(-1, block_ref.shape[-1] // lane_split)
    return ref if rows is None else ref.at[row0 : row0 + rows]


def load_typed(
    block_ref,
    *,
    view_dtype,
    lane_split: int = 1,
    row0: int = 0,
    rows: int | None = None,
) -> jax.Array:
    """Reads a raw block ref as ``(rows, lanes // lane_split)`` view_dtype."""
    return _typed_view(block_ref, view_dtype, lane_split, row0, rows)[...]


def store_typed(
    block_ref,
    values: jax.Array,
    *,
    lane_split: int = 1,
    row0: int = 0,
    rows: int | None = None,
) -> None:
    """Inverse of ``load_typed``: stores ``values`` through a typed view of
    the raw block ref."""
    ref = _typed_view(block_ref, values.dtype, lane_split, row0, rows)
    ref[...] = values.reshape(ref.shape)


def pack_bf16_pairs(values: jax.Array) -> jax.Array:
    """``(2 * rows, lanes)`` bf16 -> ``(rows, lanes)`` uint32 of BF16 pairs."""
    return pltpu.bitcast(values.astype(jnp.bfloat16), jnp.uint32)


def unpack_bf16_pairs(words: jax.Array) -> jax.Array:
    """Inverse of ``pack_bf16_pairs``: ``(rows, lanes)`` uint32 -> ``(2 * rows, lanes)`` bf16."""
    return pltpu.bitcast(words, jnp.bfloat16)


def pack_fp8_quads(values: jax.Array) -> jax.Array:
    """``(4 * rows, lanes)`` float8_e4m3fn -> ``(rows, lanes)`` uint32 of FP8 quads."""
    return pltpu.bitcast(values.astype(jnp.float8_e4m3fn), jnp.uint32)


def unpack_fp8_quads(words: jax.Array) -> jax.Array:
    """Inverse of ``pack_fp8_quads``: ``(rows, lanes)`` uint32 -> ``(4 * rows, lanes)`` float8_e4m3fn."""
    return pltpu.bitcast(words, jnp.float8_e4m3fn)


def pack_fp4_octets(values: jax.Array) -> jax.Array:
    """``(8 * rows, lanes)`` float4_e2m1fn -> ``(rows, lanes)`` uint32 of FP4 octets."""
    return pltpu.bitcast(values.astype(jnp.float4_e2m1fn), jnp.uint32)


def unpack_fp4_octets(words: jax.Array) -> jax.Array:
    """Inverse of ``pack_fp4_octets``: ``(rows, lanes)`` uint32 -> ``(8 * rows, lanes)`` float4_e2m1fn."""
    return pltpu.bitcast(words, jnp.float4_e2m1fn)
