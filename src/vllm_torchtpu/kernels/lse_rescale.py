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
"""Log-sum-exp rescale of an attention partial, written head-major (Pallas).

The context-parallel merge weights each rank's partial `[T, H, D]` by
`exp(lse - merged_lse)` and reduce-scatters it over heads. A collective
scatters its leading axis, so XLA writes the weighted partial token-major and
then copies it head-major. Ordering the torch ops head-major does not avoid
that: XLA lowers them to the same fusion and copy. This kernel weights every
head and stores it head-major in the same pass.

`lse_rescale_quantize` serves TPU_DCP_MLA_OPROJ_ALL_REDUCE, where o_proj's
all-reduce sums the weighted partials instead of a reduce-scatter. It reads
the partial head-major, as the W_UV batched matmul writes it, and stores it
token-major in FP8, quantized per token for o_proj's matmul over (head, dim):
each head's tile is a column block of the token's row, so no relayout. Left
to XLA, the weighting fuses into that batched matmul, which then stores
token-minor at twice the cost, and the quantization is another pass.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(lse_ref, merged_ref, x_ref, o_ref):
    weight = jnp.exp(lse_ref[...] - merged_ref[...])
    # One relayout of the whole block: Mosaic has no strided load of a
    # packed row, and one per head runs at half the einshape's speed.
    x = pltpu.einshape("thd->htd", x_ref[...])
    for head in range(x.shape[0]):
        w = weight[:, head : head + 1]
        y = x[head].astype(jnp.float32) * w
        # where, not *: a zero-weight partial may be NaN (NaN * 0 = NaN).
        o_ref[head] = jnp.where(w > 0, y, 0.0).astype(o_ref.dtype)


def lse_rescale_head_major(
    output: jax.Array,
    lse: jax.Array,
    merged_lse: jax.Array,
    out_dtype: str,
) -> jax.Array:
    """`out[h, t] = output[t, h] * exp(lse[t, h] - merged_lse[t, h])` in
    `out_dtype`, and 0 where that weight is 0.

    Args:
      output: `[T, H, D]`, a rank's partial attention output.
      lse: `[T, H]` f32, its log-sum-exp.
      merged_lse: `[T, H]` f32, the log-sum-exp over every rank.
      out_dtype: dtype name of the result.

    Returns:
      `[H, T, D]` in `out_dtype`.
    """
    tokens, heads, dim = output.shape
    dtype = jnp.dtype(out_dtype)
    info = pltpu.get_tpu_info()
    # Two buffers each of the input and output blocks fill half of VMEM.
    row_bytes = heads * dim * (output.dtype.itemsize + dtype.itemsize)
    tile = info.num_sublanes * 4 // min(output.dtype.itemsize, dtype.itemsize)
    block = max(tile, info.vmem_capacity_bytes // 4 // row_bytes // tile * tile)
    block = min(block, tokens)
    return pl.pallas_call(
        _kernel,
        grid=(pl.cdiv(tokens, block),),
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


def _quantize_kernel(lse_ref, merged_ref, x_ref, q_ref, s_ref):
    weight = jnp.exp(lse_ref[...] - merged_ref[...])
    heads, _, dim = x_ref.shape
    # Each token's max over every head and lane, from each head's lane max:
    # the weight is >= 0, so it scales the max. where, not *: a zero-weight
    # partial may be NaN (NaN * 0 = NaN).
    head_max = jnp.concatenate(
        [
            jnp.max(jnp.abs(x_ref[head].astype(jnp.float32)), axis=1, keepdims=True)
            for head in range(heads)
        ],
        axis=1,
    )
    peak = jnp.max(jnp.where(weight > 0, head_max * weight, 0.0), axis=1, keepdims=True)
    q_max = float(jnp.finfo(q_ref.dtype).max)
    # As `quantize_block`, the quantization XLA's quantized matmul applies.
    scale = jnp.where(peak > 0, peak / q_max, 1.0)
    factor = weight / scale
    for head in range(heads):
        f = factor[:, head : head + 1]
        x = jnp.where(f > 0, x_ref[head].astype(jnp.float32) * f, 0.0)
        q_ref[:, head * dim : (head + 1) * dim] = jnp.clip(x, -q_max, q_max).astype(
            q_ref.dtype
        )
    s_ref[...] = scale


def lse_rescale_quantize(
    output: jax.Array,
    lse: jax.Array,
    merged_lse: jax.Array,
    start: int,
    out_dtype: str,
) -> tuple[jax.Array, jax.Array]:
    """The weighted tokens `start : start + R` of `lse_rescale_head_major`,
    token-major and quantized per token, for a matmul over (head, dim). The
    result is its own buffer: XLA's matmul runs slower on a slice of one.

    Args:
      output: `[H, T, D]`, a rank's partial attention output, head-major.
      lse: `[R, H]` f32, the log-sum-exp of those `R` tokens.
      merged_lse: `[R, H]` f32, their log-sum-exp over every rank.
      start: the first token; a multiple of 128.
      out_dtype: dtype name of the quantized result (a float8 type).

    Returns:
      `q` `[R, H * D]` in `out_dtype` and `scale` `[R, 1]` f32, with
      `q[t, h * D + d] * scale[t] = output[h, start + t, d] * exp(lse -
      merged_lse)[t, h]` up to rounding, and 0 where that weight is 0.
    """
    heads, _, dim = output.shape
    rows = lse.shape[0]
    block = 128
    if start % block:
        raise ValueError(f"{start=} is not a multiple of {block}")
    offset = start // block
    dtype = jnp.dtype(out_dtype)
    # The double-buffered blocks and as much again for the kernel's
    # temporaries, no more: the o_proj matmul that follows prefetches its
    # weight into the VMEM left free while this runs.
    block_bytes = block * heads * dim * (output.dtype.itemsize + dtype.itemsize)
    return pl.pallas_call(
        _quantize_kernel,
        grid=(pl.cdiv(rows, block),),
        in_specs=[
            pl.BlockSpec((block, heads), lambda i: (i, 0)),
            pl.BlockSpec((block, heads), lambda i: (i, 0)),
            pl.BlockSpec((heads, block, dim), lambda i: (0, offset + i, 0)),
        ],
        out_specs=[
            pl.BlockSpec((block, heads * dim), lambda i: (i, 0)),
            pl.BlockSpec((block, 1), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((rows, heads * dim), dtype),
            jax.ShapeDtypeStruct((rows, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=4 * block_bytes + (4 << 20),
        ),
    )(lse, merged_lse, output)
