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
ratio) plus an optional 128-lane split. The split maps each 128-wide
lane slice to a contiguous range of rows — lane slice ``i`` carries rows
``[i * n, (i + 1) * n)`` — so a narrower-lane state view is a pair of
128-aligned lane slices concatenated along rows, with no lane-crossing
relayout. The pool gather/scatter kernels and the GDN V3 state seam
share these helpers so the bytes they exchange are identical.
"""
import jax
import jax.numpy as jnp


def load_typed(block_ref, *, view_dtype, lane_split: int = 1) -> jax.Array:
    """Reads a raw block ref as ``(rows, lanes // lane_split)`` view_dtype."""
    if jnp.dtype(block_ref.dtype) != jnp.dtype(view_dtype):
        block_ref = block_ref.bitcast(jnp.dtype(view_dtype))
    lanes = block_ref.shape[-1]
    arr = block_ref[...].reshape(-1, lanes)
    if lane_split > 1:
        out_lanes = lanes // lane_split
        arr = jnp.concatenate([
            arr[:, i * out_lanes:(i + 1) * out_lanes]
            for i in range(lane_split)
        ],
                              axis=0)
    return arr


def store_typed(block_ref, values: jax.Array, *, lane_split: int = 1) -> None:
    """Inverse of ``load_typed``: lane-merges ``values`` and stores them
    through a bitcast view of the raw block ref, which is fully written."""
    if lane_split > 1:
        rows = values.shape[0] // lane_split
        values = jnp.concatenate(
            [values[i * rows:(i + 1) * rows] for i in range(lane_split)],
            axis=-1)
    if jnp.dtype(block_ref.dtype) != jnp.dtype(values.dtype):
        block_ref = block_ref.bitcast(jnp.dtype(values.dtype))
    block_ref[...] = values.reshape(block_ref.shape)
