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

import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Conv1DInput:
    """Prepared FP32 [tokens, 1, features] input and projection output dtype.

    The dtype is static pytree metadata, so preparing a BF16 projection does
    not change the output dtype of the shared convolution/recurrence kernel.
    """

    values: jax.Array
    output_dtype: jnp.dtype = dataclasses.field(metadata={"static": True})


def _compact_qkv_kernel(qkv_ref: jax.Array, out_ref: jax.Array) -> None:
    out_ref[...] = qkv_ref[...].astype(jnp.float32)[:, None, :]


def supports_compact_qkv(shape: tuple[int, int], dtype: jnp.dtype) -> bool:
    """Whether the tiled converter fits its aligned, double-buffered envelope."""
    tokens, width = shape
    tile_tokens = 128
    window_bytes = 2 * tile_tokens * width * (dtype.itemsize + 4)
    return (
        dtype == jnp.bfloat16
        and tokens >= tile_tokens
        and tokens % tile_tokens == 0
        and width % 128 == 0
        and window_bytes <= 24 * 1024**2
    )


def compact_qkv_input(qkv: jax.Array) -> jax.Array:
    """Cast directly into FP32 Conv1D input tiles."""
    tokens, width = qkv.shape
    tile_tokens = 128
    if not supports_compact_qkv((tokens, width), qkv.dtype):
        return qkv.astype(jnp.float32).reshape(tokens, 1, width)

    # Conv1D consumes T(1,128) FP32 rows. Convert BF16 tiles directly into
    # that layout; this does not assume an extra HBM buffer in the default path.
    return pl.pallas_call(
        _compact_qkv_kernel,
        grid=(tokens // tile_tokens,),
        in_specs=(pl.BlockSpec((tile_tokens, width), lambda tile: (tile, 0)),),
        out_specs=pl.BlockSpec((tile_tokens, 1, width), lambda tile: (tile, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((tokens, 1, width), jnp.float32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name="gdn_compact_qkv_input",
    )(qkv)
