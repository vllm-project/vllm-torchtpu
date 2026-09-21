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


def _row_shape(ref, lane_split: int):
    """``(rows, lanes // lane_split)`` for ``ref`` as it stands now."""
    return (-1, ref.shape[-1] // lane_split)


def _packed_view_on_byte_block(block_ref, view_dtype, lane_split: int) -> bool:
    """A 16-bit view with a lane split on an 8-bit block.

    The pool block is ``[rows, 1, 4, 256]`` bytes: four bytes in the second-minor dim, one 32-bit word.
    Its 32-bit view is ``[rows, 1, 1, 256]`` with a (1, 128) tiling, which Mosaic reshapes freely to
    ``[2 rows, 128]`` (the fp32 state path). Its 16-bit view is ``[rows, 1, 2, 256]`` with a packed
    (2, 128) tiling, and Mosaic refuses to reshape that to 128 lanes ("expected the minormost dimension
    to be unchanged": the fault that stopped the bf16 GDN recurrent state). So the 16-bit view goes through
    the 32-bit ref and a register-level bitcast of the loaded value (``pltpu.bitcast``, each 32-bit word
    reinterpreted as two 16-bit rows; no data movement). The 16-bit rows land pair-interleaved relative to
    a naive byte order, and the store below takes the same route, so the state round-trips; nothing else
    reads these bytes typed."""
    return (jnp.dtype(block_ref.dtype).itemsize == 1
            and jnp.dtype(view_dtype).itemsize == 2 and lane_split > 1)


def load_typed(block_ref, *, view_dtype, lane_split: int = 1) -> jax.Array:
    """Reads a raw block ref as ``(rows, lanes // lane_split)`` view_dtype.

    When the view dtype differs from the block's, the bitcast legalizes the
    retype and the narrowing reshape rides on the (bitcast) ref -- the fast
    path the module docstring measures. When the view dtype matches the
    block's there is no bitcast, so a ``lane_split`` reshape would change
    the ref's minormost dimension, which Mosaic rejects; the reshape moves
    onto the loaded value instead. Same-dtype with lane_split==1 keeps the
    ref path since its reshape only merges leading dims.

    The narrowing shape is always computed from the ref the reshape is
    applied to -- after the bitcast, whose element-size change moves the
    minormost dimension.
    """
    if _packed_view_on_byte_block(block_ref, view_dtype, lane_split):
        ref = block_ref.bitcast(jnp.float32)
        ref = ref.reshape(*_row_shape(ref, lane_split))
        return pltpu.bitcast(ref[...], jnp.dtype(view_dtype))
    if jnp.dtype(block_ref.dtype) != jnp.dtype(view_dtype):
        ref = block_ref.bitcast(jnp.dtype(view_dtype))
        return ref.reshape(*_row_shape(ref, lane_split))[...]
    if lane_split == 1:
        return block_ref.reshape(*_row_shape(block_ref, lane_split))[...]
    return block_ref[...].reshape(*_row_shape(block_ref, lane_split))


def store_typed(block_ref, values: jax.Array, *, lane_split: int = 1) -> None:
    """Inverse of ``load_typed``: stores ``values`` through a typed view of
    the raw block ref, which is fully written."""
    if _packed_view_on_byte_block(block_ref, values.dtype, lane_split):
        ref = block_ref.bitcast(jnp.float32)
        ref = ref.reshape(*_row_shape(ref, lane_split))
        ref[...] = pltpu.bitcast(values.reshape(-1, ref.shape[-1]),
                                 jnp.float32)
        return
    if jnp.dtype(block_ref.dtype) != jnp.dtype(values.dtype):
        ref = block_ref.bitcast(jnp.dtype(values.dtype))
        ref = ref.reshape(*_row_shape(ref, lane_split))
        ref[...] = values.reshape(ref.shape)
        return
    if lane_split == 1:
        ref = block_ref.reshape(*_row_shape(block_ref, lane_split))
        ref[...] = values.reshape(ref.shape)
        return
    # No bitcast to legalize a minor-dim change, so write the value in the
    # ref's own shape rather than reshaping the ref.
    block_ref[...] = values.reshape(block_ref.shape)
