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
