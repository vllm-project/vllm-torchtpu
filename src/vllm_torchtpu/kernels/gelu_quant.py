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
"""GeGLU (tanh GELU) and per-token FP8 quantization of its output in one pass.

A W8A8 down projection after `GeluAndMul(approximate="tanh")` costs XLA two
passes over the activation: one fusion computes gelu(gate) * up, writes it in
bf16 and reduces each row's absolute maximum; a second reads it back and
quantizes it. This kernel holds a block of whole rows in VMEM and does both,
reading `[gate | up]` once and writing only the FP8 rows and their scales.

Arithmetic follows `GeluAndMul` then `quantized_matmul.util.quantize_tensor`:
the product is rounded to the activation dtype, the f32 scale is that row's
maximum / dtype max (1 where it is 0), and the rows are divided by it in f32,
clipped and cast. GELU runs in f32 and rounds once, which XLA's fusion of the
same expression need not do, so values can move by one rounding step.

With `bf16_math`, GELU (as x * sigmoid(2z), which has no cancellation), the
product and the quantization's multiply run in bf16 (the row maximum and the
scale stay f32): about half the VALU work of the f32 path, which is
VALU-bound. The bf16 roundings, amplified by exp in the exponent's argument,
move small values by a few percent of themselves and every value by at most
one FP8 step of its row's scale.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.row_blocks import fit_block_rows

_SQRT_2_OVER_PI = math.sqrt(2 / math.pi)

# Rows per block. A block holds 2 x rows x intermediate bf16 inputs (double
# buffered) plus temporaries. Larger blocks spend less per row on pipeline
# overhead; the block size does not change the outputs.
_BLOCK_ROWS = 256
_VMEM_LIMIT_BYTES = 48 * 1024 * 1024


def _gelu_tanh(x: jax.Array) -> jax.Array:
    return 0.5 * x * (1.0 + jnp.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)))


def _kernel_bf16(gate_ref, up_ref, q_ref, scale_ref, *, quant_dtype):
    info = jnp.finfo(quant_dtype)
    gate, up = gate_ref[...], up_ref[...]
    # 0.5 * x * (1 + tanh(z)) as x * sigmoid(2z): no cancellation where tanh
    # is near -1, which bf16 would round to garbage.
    neg_2z = gate * (
        jnp.bfloat16(-2 * _SQRT_2_OVER_PI)
        + jnp.bfloat16(-2 * _SQRT_2_OVER_PI * 0.044715) * (gate * gate)
    )
    y = (gate * up) * (1.0 / (1.0 + jnp.exp(neg_2z)))
    scale = jnp.max(jnp.abs(y), axis=-1, keepdims=True).astype(jnp.float32)
    scale = scale / float(info.max)
    scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)
    q = y * (1.0 / scale).astype(jnp.bfloat16)
    q = jnp.clip(q, float(info.min), float(info.max))
    q_ref[...] = q.astype(quant_dtype)
    scale_ref[...] = scale


def _kernel(gate_ref, up_ref, q_ref, scale_ref, *, act_dtype, quant_dtype):
    info = jnp.finfo(quant_dtype)
    gate = gate_ref[...].astype(jnp.float32)
    up = up_ref[...].astype(jnp.float32)
    y = (_gelu_tanh(gate) * up).astype(act_dtype).astype(jnp.float32)
    scale = jnp.max(jnp.abs(y), axis=-1, keepdims=True) / float(info.max)
    scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)
    q = jnp.clip(y / scale, float(info.min), float(info.max))
    q_ref[...] = q.astype(quant_dtype)
    scale_ref[...] = scale


# The `[rows, d]` blocks of the up half start at column d, so d must be a
# whole number of lanes.
COLUMN_ALIGNMENT = 128


def block_rows(num_tokens: int, d: int, max_rows: int = _BLOCK_ROWS) -> int:
    """Rows per block (row_blocks.fit_block_rows): the `[gate | up]` inputs are
    2 x d bf16 per row and the FP8 output d bytes, each double-buffered."""
    return fit_block_rows(num_tokens, 2 * (4 * d + d), max_rows, _VMEM_LIMIT_BYTES)


@functools.partial(
    jax.jit, static_argnames=("quant_dtype", "bf16_math", "max_block_rows", "interpret")
)
def gelu_tanh_and_mul_quantize(
    x: jax.Array,
    quant_dtype: jnp.dtype,
    bf16_math: bool = False,
    max_block_rows: int = _BLOCK_ROWS,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """`[tokens, 2 * d]` -> (`[tokens, d]` in `quant_dtype`, f32 `[tokens, 1]` scales).

    The result is `quantize_tensor(GeluAndMul("tanh")(x), quant_dtype)` up to
    one rounding step (module docstring).
    """
    num_tokens, width = x.shape
    d = width // 2
    if width % 2 or d % COLUMN_ALIGNMENT:
        raise ValueError(
            f"gelu_tanh_and_mul_quantize: width {width} is not twice a multiple of "
            f"{COLUMN_ALIGNMENT}"
        )
    if bf16_math and x.dtype != jnp.bfloat16:
        raise ValueError(f"bf16_math needs bf16 gate/up rows, got {x.dtype}")
    rows = block_rows(num_tokens, d, max_block_rows)
    half = pl.BlockSpec((rows, d), lambda i: (i, 0))
    kernel = (
        functools.partial(_kernel_bf16, quant_dtype=quant_dtype)
        if bf16_math
        else functools.partial(_kernel, act_dtype=x.dtype, quant_dtype=quant_dtype)
    )
    return pl.pallas_call(
        kernel,
        grid=(num_tokens // rows,),
        in_specs=[half, pl.BlockSpec((rows, d), lambda i: (i, 1))],
        out_specs=[half, pl.BlockSpec((rows, 1), lambda i: (i, 0))],
        out_shape=[
            jax.ShapeDtypeStruct((num_tokens, d), quant_dtype),
            jax.ShapeDtypeStruct((num_tokens, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=_VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="gelu_tanh_and_mul_quantize",
    )(x, x)
