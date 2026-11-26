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
"""Log-sum-exp rescale of a DCP attention partial written head-major (Pallas)."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

MIN_TOKENS = 8
MAX_BLOCK_TOKENS = 128
TILE_ALIGN = 128


def _kernel(lse_ref, merged_ref, x_ref, o_ref):
    lse = lse_ref[...].astype(jnp.float32)
    merged = merged_ref[...].astype(jnp.float32)
    weight = jnp.exp(jnp.where(jnp.isfinite(merged), lse - merged, -jnp.inf))
    x = pltpu.einshape("thd->htd", x_ref[...])
    for head in range(x.shape[0]):
        w = weight[:, head : head + 1]
        safe_x = jnp.where(w > 0, x[head].astype(jnp.float32), 0.0)
        o_ref[head] = jnp.where(w > 0, safe_x * w, 0.0).astype(o_ref.dtype)


def _lse_rescale_fallback(
    output: jax.Array,
    lse: jax.Array,
    merged_lse: jax.Array,
    dtype: jax.typing.DTypeLike,
) -> jax.Array:
    lse_f, merged_f = lse.astype(jnp.float32), merged_lse.astype(jnp.float32)
    diff = jnp.where(jnp.isfinite(merged_f), lse_f - merged_f, -jnp.inf)
    weight = jnp.exp(diff)[..., None]
    safe_x = jnp.where(weight > 0, output.astype(jnp.float32), 0.0)
    weighted = jnp.where(weight > 0, safe_x * weight, 0.0).astype(dtype)
    return jnp.swapaxes(weighted, 0, 1)


def _lse_rescale_head_major_impl(
    output: jax.Array,
    lse: jax.Array,
    merged_lse: jax.Array,
    out_dtype: jax.typing.DTypeLike,
) -> jax.Array:
    tokens, heads, dim = output.shape
    dtype = jnp.dtype(out_dtype)
    if (
        tokens < MIN_TOKENS
        or dim % TILE_ALIGN != 0
        or (tokens > MAX_BLOCK_TOKENS and tokens % MAX_BLOCK_TOKENS != 0)
        or jax.default_backend() != "tpu"
    ):
        return _lse_rescale_fallback(output, lse, merged_lse, dtype)
    info = pltpu.get_tpu_info()
    row_bytes = heads * dim * (output.dtype.itemsize + dtype.itemsize)
    tile = max(
        MIN_TOKENS, info.num_sublanes * 4 // min(output.dtype.itemsize, dtype.itemsize)
    )
    vmem_block = max(tile, info.vmem_capacity_bytes // 4 // row_bytes // tile * tile)
    block = min(tokens, MAX_BLOCK_TOKENS, vmem_block)
    if block < MIN_TOKENS or tokens % block != 0:
        return _lse_rescale_fallback(output, lse, merged_lse, dtype)
    return pl.pallas_call(
        _kernel,
        grid=(tokens // block,),
        in_specs=[
            pl.BlockSpec((block, heads), lambda i: (i, 0)),
            pl.BlockSpec((block, heads), lambda i: (i, 0)),
            pl.BlockSpec((block, heads, dim), lambda i: (i, 0, 0)),
        ],
        out_specs=pl.BlockSpec((heads, block, dim), lambda i: (0, i, 0)),
        out_shape=jax.ShapeDtypeStruct((heads, tokens, dim), dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=info.vmem_capacity_bytes * 3 // 4,
        ),
    )(lse, merged_lse, output)


def lse_rescale_head_major_f32(
    output: jax.Array, lse: jax.Array, merged_lse: jax.Array
) -> jax.Array:
    """Return ``out[h, t] = output[t, h] * exp(lse[t, h] - merged_lse[t, h])`` (f32)."""
    return _lse_rescale_head_major_impl(output, lse, merged_lse, jnp.float32)


def lse_rescale_head_major_bf16(
    output: jax.Array, lse: jax.Array, merged_lse: jax.Array
) -> jax.Array:
    """Return ``out[h, t] = output[t, h] * exp(lse[t, h] - merged_lse[t, h])``."""
    return _lse_rescale_head_major_impl(output, lse, merged_lse, jnp.bfloat16)
