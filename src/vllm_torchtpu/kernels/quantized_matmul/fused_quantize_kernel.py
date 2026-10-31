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
"""FP8 matmul that quantizes its per-token activation in VMEM.

Computes what `util.xla_quantized_matmul` computes for an fp8 weight in the
(k, n) layout, without the separate abs-max and quantize passes over the
activation in HBM. The grid walks weight column tiles outermost; the first
column pass quantizes each row tile in the same basic block as its matmul, so
the quantize overlaps the MXU work, and keeps it in a scratch that every later
pass reuses.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Rows, and weight columns, per grid step.
TOKEN_BLOCK = 512
OUT_BLOCK = 1024
# Columns per quantize step; rows per step are one fp8 vreg's 32 sublanes.
QUANT_COLS = 512
QUANT_ROWS = 32
# Taken only while n / k is below this: the in-VMEM quantize saves work that
# grows with m * k, and the kernel's epilogue overlaps the MXU less than XLA's
# output fusion does, which costs work that grows with m * n.
MAX_OUT_PER_IN = 1.0
VMEM_LIMIT_BYTES = 48 * 1024 * 1024


def _out_block(n: int) -> int | None:
    for block in (OUT_BLOCK, 512, 256):
        if n % block == 0:
            return block
    return None


def _vmem_bytes(m: int, k: int, tn: int) -> int:
    x = 2 * TOKEN_BLOCK * k * 2  # bf16, double-buffered
    x_q = m * k + m * 128 * 4  # fp8 rows and their lane-padded f32 scales
    w = 2 * k * tn
    out = 2 * TOKEN_BLOCK * tn * 2
    acc = 2 * TOKEN_BLOCK * tn * 4  # f32 dot result and its scaled copy
    return x + x_q + w + out + acc


def supports(x: jax.Array, w_q: jax.Array, w_scale: jax.Array) -> bool:
    """Whether `fused_quantized_matmul` should take these operands."""
    m, k = x.shape
    n = w_q.shape[1]
    tn = _out_block(n)
    return (
        x.dtype == jnp.bfloat16
        and w_q.dtype == jnp.float8_e4m3fn
        and w_scale.shape == (n,)
        and m % TOKEN_BLOCK == 0
        and k % QUANT_COLS == 0
        and tn is not None
        and n < MAX_OUT_PER_IN * k
        and _vmem_bytes(m, k, tn) <= VMEM_LIMIT_BYTES
    )


def _quantize_rows(x_ref, x_q_ref, x_scale_ref, i, r, f8_max):
    """Quantizes rows [r, r + QUANT_ROWS) of the x tile into slot i.

    The same arithmetic as `util.quantize_block` as XLA compiles it.
    """
    rows = pl.ds(r, QUANT_ROWS)
    num_cols = x_ref.shape[1] // QUANT_COLS

    def cols(c):
        return pl.ds(c * QUANT_COLS, QUANT_COLS)

    # Two streaming passes over the tile in VMEM, so a row slab is never held
    # in registers whole.
    abs_max = jnp.abs(x_ref[rows, cols(0)])
    for c in range(1, num_cols):
        abs_max = jnp.maximum(abs_max, jnp.abs(x_ref[rows, cols(c)]))
    abs_max = jnp.max(abs_max, axis=1, keepdims=True)
    # XLA folds `abs_max / f8_max` into a multiply by the f32 reciprocal;
    # spelling it out keeps the scale bit-identical.
    scale = abs_max.astype(jnp.float32) * jnp.float32(1.0 / f8_max)
    scale = jnp.where(scale == 0, 1.0, scale)
    x_scale_ref[i, rows, :] = scale
    for c in range(num_cols):
        x = x_ref[rows, cols(c)].astype(jnp.float32)
        x_q = jnp.clip(x / scale, -f8_max, f8_max)
        x_q_ref[i, rows, cols(c)] = x_q.astype(x_q_ref.dtype)


def _kernel(x_ref, w_ref, w_scale_ref, out_ref, x_q_ref, x_scale_ref, *, f8_max):
    j, i = pl.program_id(0), pl.program_id(1)

    def matmul():
        acc = jnp.dot(x_q_ref[i], w_ref[...], preferred_element_type=jnp.float32)
        acc = acc * x_scale_ref[i]
        acc = acc * w_scale_ref[...]
        out_ref[...] = acc.astype(out_ref.dtype)

    # Unrolled, and in the same branch as the matmul, so the scheduler can
    # issue the quantize in the VALU slots of the MXU-bound matmul.
    @pl.when(j == 0)
    def _first_pass():
        for r in range(0, x_ref.shape[0], QUANT_ROWS):
            _quantize_rows(x_ref, x_q_ref, x_scale_ref, i, r, f8_max)
        matmul()

    @pl.when(j != 0)
    def _later_pass():
        matmul()


@jax.jit
def fused_quantized_matmul(
    x: jax.Array,  # [m, k] bfloat16
    w_q: jax.Array,  # [k, n] float8_e4m3fn
    w_scale: jax.Array,  # [n]
) -> jax.Array:
    """`xla_quantized_matmul(x, w_q, w_scale)` with the quantize kept in VMEM."""
    m, k = x.shape
    n = w_q.shape[1]
    tm, tn = TOKEN_BLOCK, _out_block(n)
    num_i, num_j = m // tm, n // tn
    f8_max = float(jnp.finfo(w_q.dtype).max)

    def x_index(j, i):
        # Only the first column pass reads x; later passes keep the last tile.
        return jnp.where(j == 0, i, num_i - 1), 0

    return pl.pallas_call(
        functools.partial(_kernel, f8_max=f8_max),
        grid=(num_j, num_i),
        in_specs=[
            pl.BlockSpec((tm, k), x_index),
            pl.BlockSpec((k, tn), lambda j, i: (0, j)),
            pl.BlockSpec((1, tn), lambda j, i: (0, j)),
        ],
        out_specs=pl.BlockSpec((tm, tn), lambda j, i: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        scratch_shapes=[
            pltpu.VMEM((num_i, tm, k), w_q.dtype),
            pltpu.VMEM((num_i, tm, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
        ),
        name="fused_quantized_matmul",
    )(x, w_q, w_scale.astype(jnp.float32).reshape(1, n))
