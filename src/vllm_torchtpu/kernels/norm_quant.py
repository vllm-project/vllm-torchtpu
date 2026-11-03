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
"""Residual add around an RMSNorm, then the next RMSNorm and its per-token FP8
quantization, in one pass over whole rows.

Between two projections a Gemma4 layer runs

    h = rms_norm(x, w_post) + residual    [* layer_scalar]
    q, s = quantize_tensor(rms_norm(h, w_next), fp8)

which XLA runs as four passes over `[tokens, hidden]`: the variance of `x`,
the normalization and the add (writing `h`), the next norm with its row
maximum, and the next norm again with the cast. This kernel holds a block of
whole rows in VMEM and writes `h`, `q` and `s` once.

Arithmetic follows the TPU `rms_norm` provider and `quantize_tensor`: the
norms run in f32 and round to the activation dtype, the add and the scalar
product round to it, and the f32 scale is the row maximum / dtype max (1 where
it is 0). The sums of squares are added in another order than XLA's, so values
can move by one rounding step.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.row_blocks import fit_block_rows

# Rows per block: on tpu7x a 4096-row Gemma4-31B TP2 shard takes 0.142 / 0.121 /
# 0.115 ms with 64 / 128 / 256 rows (device micro, same outputs).
_BLOCK_ROWS = 256
_VMEM_LIMIT_BYTES = 48 * 1024 * 1024


def _rms_norm(x: jax.Array, w: jax.Array, eps: float, dtype: jnp.dtype) -> jax.Array:
    var = jnp.mean(x * x, axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(var + eps) * w).astype(dtype).astype(jnp.float32)


def _kernel(*refs, eps, has_scalar, quant_dtype, bytes_out):
    if has_scalar:
        x_ref, r_ref, w_post_ref, w_next_ref, scalar_ref, h_ref, q_ref, s_ref = refs
    else:
        x_ref, r_ref, w_post_ref, w_next_ref, h_ref, q_ref, s_ref = refs
    dtype = h_ref.dtype
    x = x_ref[...].astype(jnp.float32)
    w_post = w_post_ref[...].astype(jnp.float32)
    n = _rms_norm(x, w_post, eps, dtype)
    h = (n + r_ref[...].astype(jnp.float32)).astype(dtype).astype(jnp.float32)
    if has_scalar:
        h = (h * scalar_ref[...].astype(jnp.float32)).astype(dtype).astype(jnp.float32)
    h_ref[...] = h.astype(dtype)
    m = _rms_norm(h, w_next_ref[...].astype(jnp.float32), eps, dtype)
    info = jnp.finfo(quant_dtype)
    scale = jnp.max(jnp.abs(m), axis=-1, keepdims=True) / float(info.max)
    scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)
    q = jnp.clip(m / scale, float(info.min), float(info.max)).astype(quant_dtype)
    q_ref[...] = jax.lax.bitcast_convert_type(q, jnp.int8) if bytes_out else q
    s_ref[...] = scale


@functools.partial(
    jax.jit,
    static_argnames=("eps", "quant_dtype", "bytes_out", "max_block_rows", "interpret"),
)
def add_rms_norm_quantize(
    x: jax.Array,
    residual: jax.Array,
    w_post: jax.Array,
    w_next: jax.Array,
    scalar: jax.Array | None,
    *,
    eps: float,
    quant_dtype: jnp.dtype,
    bytes_out: bool = False,
    max_block_rows: int = _BLOCK_ROWS,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """`h = rms_norm(x, w_post) + residual` (times `scalar` when given), and
    `quantize_tensor(rms_norm(h, w_next), quant_dtype)`.

    Returns `(h, q, scale)`: `h` in `x`'s dtype, `q` `[tokens, hidden]` in
    `quant_dtype` (with `bytes_out`, its bytes as int8, which a collective moves
    without a dtype view), `scale` f32 `[tokens, 1]` (module docstring for
    rounding).
    """
    tokens, hidden = x.shape
    # x, residual and h in the activation dtype, q in 1 byte, double-buffered.
    row_bytes = 2 * (3 * x.dtype.itemsize + 1) * hidden
    rows = fit_block_rows(tokens, row_bytes, max_block_rows, _VMEM_LIMIT_BYTES)
    row_block = pl.BlockSpec((rows, hidden), lambda i: (i, 0))
    weight = pl.BlockSpec((1, hidden), lambda i: (0, 0))
    operands = [x, residual, w_post.reshape(1, hidden), w_next.reshape(1, hidden)]
    in_specs = [row_block, row_block, weight, weight]
    if scalar is not None:
        operands.append(scalar.reshape(1, 1))
        in_specs.append(pl.BlockSpec((1, 1), lambda i: (0, 0)))
    return pl.pallas_call(
        functools.partial(
            _kernel,
            eps=eps,
            has_scalar=scalar is not None,
            quant_dtype=quant_dtype,
            bytes_out=bytes_out,
        ),
        grid=(tokens // rows,),
        in_specs=in_specs,
        out_specs=[row_block, row_block, pl.BlockSpec((rows, 1), lambda i: (i, 0))],
        out_shape=[
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            jax.ShapeDtypeStruct(x.shape, jnp.int8 if bytes_out else quant_dtype),
            jax.ShapeDtypeStruct((tokens, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=_VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="add_rms_norm_quantize",
    )(*operands)
