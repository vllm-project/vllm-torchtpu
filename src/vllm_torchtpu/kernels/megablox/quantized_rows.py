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
"""gmm_v2's per-row lhs quantization, done ahead of the kernel.

gmm_v2 quantizes each lhs row per block of columns (``quantize_block``)
before its matmul. Rows that are moved before the kernel reads them, such as
an expert-parallel dispatch, can be quantized first and moved in half the
bytes: ``quantize_rows`` runs the same ``quantize_block`` on the same rows, so
the kernel computes the same bits from them.

A quantized row has ``size_k + SCALE_COLS`` columns of the 8-bit quantized
dtype: the quantized values in the first ``size_k``, then the bf16 scale of
each block as two bytes, the low one at column ``size_k + 2 * b`` and the
high one next to it, then zeros. The scales travel inside their rows, so
anything that moves or gathers rows moves them too.

Rows are handed around as the int32 words holding them,
``[rows // ROWS_PER_WORD, size_k + SCALE_COLS]``, row ``4 * w + j`` in byte
``j`` of word row ``w``. That is how a TensorCore Pallas kernel lays out 8-bit
rows in HBM, and a SparseCore kernel reads 32-bit words in the same layout,
where it would want an 8-bit array retiled.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# One lane row of columns after the quantized values holds the block scales.
SCALE_COLS = 128
# The dtype quantized rows come from: tile_matmul's scale has the lhs dtype.
SOURCE_DTYPE = jnp.bfloat16
# Quantized rows per int32 word.
ROWS_PER_WORD = 4
# Rows quantize_rows takes in multiples of: whole (8, 128) tiles of words, the
# 32-row sublane blocks the fused MLP reads 8-bit rows in.
QUANTIZE_ROWS_ALIGN = 8 * ROWS_PER_WORD
# Most rows per quantize_rows grid step.
QUANTIZE_ROWS_TILE = 256


def quantize_block(
    block_lhs: jax.Array, lhs_q_dtype: jnp.dtype
) -> tuple[jax.Array, jax.Array]:
    """Quantizes each row of block_lhs to lhs_q_dtype by its absolute max.

    Returns the quantized block and the per-row scale ``[rows, 1]`` (in
    block_lhs's dtype) that the block's products are multiplied by.
    """
    if jnp.issubdtype(lhs_q_dtype, jnp.floating):
        dtype_max = float(jnp.finfo(lhs_q_dtype).max)
    else:
        dtype_max = float(jnp.iinfo(lhs_q_dtype).max)
    block_abs_max = jnp.max(jnp.abs(block_lhs), axis=1, keepdims=True)
    block_scale = block_abs_max / dtype_max

    # If block_scale=0, it will cause division by zero and return either
    # NaN or Inf. Since this can cause numeric issue when downcasting to
    # quantized value, we convert them into 0.
    block_scale_inv = jnp.where(block_scale == 0, 0, 1 / block_scale)
    # Convert lhs into quantized dtype.
    block_lhs_q = (block_lhs * block_scale_inv).astype(lhs_q_dtype)
    return block_lhs_q, block_scale


def quantized_row_width(size_k: int) -> int:
    return size_k + SCALE_COLS


def encode_scales(scales: list[jax.Array], dtype: jnp.dtype) -> jax.Array:
    """The ``SCALE_COLS`` columns holding each ``[rows, 1]`` bf16 scale."""
    rows = scales[0].shape[0]
    lane = lax.broadcasted_iota(jnp.int32, (rows, SCALE_COLS), 1)
    cols = jnp.zeros((rows, SCALE_COLS), jnp.int32)
    for b, scale in enumerate(scales):
        bits = lax.bitcast_convert_type(scale, jnp.uint16).astype(jnp.int32)
        cols = jnp.where(lane == 2 * b, bits & 0xFF, cols)
        cols = jnp.where(lane == 2 * b + 1, bits >> 8, cols)
    return lax.bitcast_convert_type(cols.astype(jnp.uint8), dtype)


def decode_scales(cols: jax.Array, num_blocks: int) -> list[jax.Array]:
    """Inverse of ``encode_scales``: the ``[rows, 1]`` bf16 scale per block."""
    words = lax.bitcast_convert_type(cols, jnp.uint8).astype(jnp.int32)
    scales = []
    for b in range(num_blocks):
        bits = words[:, 2 * b : 2 * b + 1] | (words[:, 2 * b + 1 : 2 * b + 2] << 8)
        scales.append(lax.bitcast_convert_type(bits.astype(jnp.uint16), SOURCE_DTYPE))
    return scales


def _quantize_rows_kernel(x_ref, words_ref, *, block_size: int, q_dtype: jnp.dtype):
    out_ref = words_ref.bitcast(q_dtype)
    size_k = x_ref.shape[1]
    scales = []
    for start in range(0, size_k, block_size):
        block_q, scale = quantize_block(x_ref[:, start : start + block_size], q_dtype)
        out_ref[:, start : start + block_size] = block_q
        scales.append(scale)
    out_ref[:, size_k:] = encode_scales(scales, q_dtype)


@functools.partial(jax.jit, static_argnames=("block_size", "q_dtype"))
def quantize_rows(x: jax.Array, *, block_size: int, q_dtype: jnp.dtype) -> jax.Array:
    """x's rows quantized per ``block_size`` columns as gmm_v2 quantizes them,
    as the int32 words holding them."""
    rows, size_k = x.shape
    tile = math.gcd(rows, QUANTIZE_ROWS_TILE)
    if (
        x.dtype != SOURCE_DTYPE
        or jnp.dtype(q_dtype).itemsize * ROWS_PER_WORD != 4
        or rows % QUANTIZE_ROWS_ALIGN
        or size_k % block_size
        or 2 * (size_k // block_size) > SCALE_COLS
    ):
        raise ValueError(
            f"cannot quantize {rows} {x.dtype} rows of {size_k} columns to "
            f"{q_dtype} per {block_size}"
        )
    width = quantized_row_width(size_k)
    return pl.pallas_call(
        functools.partial(
            _quantize_rows_kernel, block_size=block_size, q_dtype=q_dtype
        ),
        out_shape=jax.ShapeDtypeStruct((rows // ROWS_PER_WORD, width), jnp.int32),
        grid=(rows // tile,),
        in_specs=[pl.BlockSpec((tile, size_k), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((tile // ROWS_PER_WORD, width), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name="quantize_rows",
    )(x)
