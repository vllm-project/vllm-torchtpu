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
"""Convert activation tiles directly into the FP32 Conv1D input layout."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.gdn.v3 import config


def _cast_input(qkv_ref: jax.Ref, output_ref: jax.Ref) -> None:
    output_ref[...] = qkv_ref[...].astype(jnp.float32)[:, None, :]


def cast_conv_input(
    qkv: jax.Array, padded_tokens: int, *, qkv_width: int | None = None
) -> jax.Array:
    """Cast the QKV columns and zero-pad rows for Conv1D.

    When qkv_width is supplied, the input also contains trailing Z columns.
    """
    tokens, input_width = qkv.shape
    width = input_width if qkv_width is None else qkv_width
    if not 0 <= width <= input_width:
        raise ValueError("qkv_width must fit in the input")
    if padded_tokens < tokens:
        raise ValueError("padded_tokens must cover every input token")
    info = pltpu.get_tpu_info()
    tile_rows = info.num_lanes
    padded_width = pl.cdiv(width, info.num_lanes) * info.num_lanes
    window_bytes = 2 * tile_rows * padded_width * (qkv.dtype.itemsize + 4)
    if (
        tokens == 0
        or width == 0
        or qkv.dtype not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32))
        or (
            width != input_width
            and (width != padded_width or input_width % info.num_lanes)
        )
        or window_bytes > config.GDNConfig.get_vmem_limit_bytes()
    ):
        return jnp.pad(
            qkv[:, :width].astype(jnp.float32), ((0, padded_tokens - tokens), (0, 0))
        ).reshape(padded_tokens, 1, width)
    # Small BF16 batches need only a sublane-aligned DMA row tile. FP32
    # retains its lane-sized tile; the resource check above is conservative.
    if qkv.dtype == jnp.bfloat16:
        tile_rows = min(
            tile_rows,
            max(
                info.num_sublanes,
                pl.cdiv(padded_tokens, info.num_sublanes) * info.num_sublanes,
            ),
        )
    padded_rows = pl.cdiv(padded_tokens, tile_rows) * tile_rows
    # Zero physical padding makes every DMA window complete, including tails.
    qkv = jnp.pad(
        qkv, ((0, padded_rows - tokens), (0, max(0, padded_width - input_width)))
    )
    # The block covers only QKV, while DMA uses the input's full row stride.
    output = pl.pallas_call(
        _cast_input,
        grid=(padded_rows // tile_rows,),
        in_specs=(pl.BlockSpec((tile_rows, padded_width), lambda tile: (tile, 0)),),
        out_specs=pl.BlockSpec((tile_rows, 1, padded_width), lambda tile: (tile, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((padded_rows, 1, padded_width), jnp.float32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name="gdn_cast_conv_input",
    )(qkv)
    return output[:padded_tokens, :, :width]


def cast_packed_conv_input(
    qkvz: jax.Array,
    padded_tokens: int,
    qkv_width: int,
    distribution: jax.Array,
) -> jax.Array:
    """Use the packed projection directly for batches containing only prefill."""
    pure_prefill = (
        (distribution[0] == 0) & (distribution[1] == 0) & (distribution[2] > 0)
    )
    return jax.lax.cond(
        pure_prefill,
        lambda packed: cast_conv_input(packed, padded_tokens, qkv_width=qkv_width),
        lambda packed: cast_conv_input(packed[:, :qkv_width], padded_tokens),
        qkvz,
    )
