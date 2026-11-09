# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""BF16 linear head with a fused first-index argmax."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.utils import get_device_name

# Qualified on v7 with full-K reuse and compact partials; other shapes fall back.
TUNED_TILES = {(288, 5120, 129280): 1280}

# Packed-key layout: the high 16 bits hold an order-preserving key of the BF16
# value, the low 16 bits hold _COL_MASK - column, so one int32 max returns the
# largest value and, among equal values, the smallest column.
_COL_MASK = 0xFFFF
_NAN_KEY = 0x7FFF0000  # above the key of +inf (0x7F800000): NaN wins


def is_supported_shape(rows: int, weight_shape: tuple[int, ...]) -> bool:
    return (
        len(weight_shape) == 2
        and (rows, weight_shape[1], weight_shape[0]) in TUNED_TILES
        and get_device_name() == "TPU v7"
    )


def packed_tile_argmax(accum: jax.Array) -> jax.Array:
    """FP32 [rows, cols] -> int32 [rows]: BF16-rounded max packed with its column.

    A BF16-rounded value as FP32 has 16 zero low bits. Sign-magnitude to two's
    complement makes the bits order like the floats and maps -0 and +0 to the
    same key, so they tie as with ``==``. Every NaN gets one key above +inf, so
    a row with NaN picks its earliest NaN. Needs cols <= _COL_MASK + 1.
    """
    rounded = accum.astype(jnp.bfloat16).astype(jnp.float32)
    bits = jax.lax.bitcast_convert_type(rounded, jnp.int32)
    magnitude = bits & 0x7FFFFFFF
    sign = bits >> 31
    key = (magnitude ^ sign) - sign
    key = jnp.where(magnitude > 0x7F800000, jnp.int32(_NAN_KEY), key)
    column = jax.lax.broadcasted_iota(jnp.int32, accum.shape, 1)
    return jnp.max(key | (_COL_MASK - column), axis=1)


def merge_tile(
    best_key: jax.Array,
    best_id: jax.Array,
    packed: jax.Array,
    base: jax.Array | int,
    first: jax.Array | bool,
) -> tuple[jax.Array, jax.Array]:
    """Fold one tile's packed maxima into the running (key, token id) per row.

    Tiles arrive in vocab order. A tile replaces the running best only on a
    strictly larger key, so an equal key keeps the earlier tile (first index);
    within a tile the packed column is already the first one. ``first`` forces
    the take for the first tile, ignoring whatever ``best_*`` held before.
    """
    key = packed >> 16
    token = base + (_COL_MASK - (packed & _COL_MASK))
    take = (key > best_key) | first
    return jnp.where(take, key, best_key), jnp.where(take, token, best_id)


def fused_greedy_head(hidden: jax.Array, weight: jax.Array) -> jax.Array:
    """Return first argmax IDs after BF16 rounding, for the original [vocab,K] W.

    Grid step n reduces tile n-1 from the accumulator scratch, then overwrites
    it with tile n's dot. Both sit in one basic block with no data dependency,
    so the scheduler may overlap the VPU/XLU reduce with the MXU dot. The
    cross-tile reduce also runs in the kernel: the running best key sits in a
    VMEM scratch and the [1, rows] id output keeps one block index, so it stays
    resident and is written back once. Step 0 reduces uninitialized scratch,
    which step 1 discards by forcing its take. Step ``tiles`` reuses the last
    weight block (no DMA) and its dot is discarded.
    """
    rows, width = hidden.shape
    vocab, weight_width = weight.shape
    assert hidden.dtype == weight.dtype == jnp.bfloat16
    assert width == weight_width
    tile = TUNED_TILES[(rows, width, vocab)]
    assert tile <= _COL_MASK + 1
    tiles = vocab // tile

    def kernel(hidden_ref, weight_ref, ids_ref, accum_ref, key_ref):
        n = pl.program_id(0)
        key, ids = merge_tile(
            key_ref[0, :],
            ids_ref[0, :],
            packed_tile_argmax(accum_ref[...]),
            (n - 1) * tile,
            n == 1,
        )
        key_ref[0, :] = key
        ids_ref[0, :] = ids
        accum_ref[...] = jnp.dot(
            hidden_ref[...], weight_ref[...].T, preferred_element_type=jnp.float32
        )

    ids = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1, rows), jnp.int32),
        grid=(tiles + 1,),
        in_specs=(
            pl.BlockSpec(hidden.shape, lambda n: (0, 0)),
            pl.BlockSpec((tile, width), lambda n: (jnp.minimum(n, tiles - 1), 0)),
        ),
        out_specs=pl.BlockSpec((1, rows), lambda n: (0, 0)),
        scratch_shapes=[
            pltpu.VMEM((rows, tile), jnp.float32),
            pltpu.VMEM((1, rows), jnp.int32),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        name=f"fused_greedy_head_m{rows}_n{tile}",
    )(hidden, weight)
    return ids[0]
