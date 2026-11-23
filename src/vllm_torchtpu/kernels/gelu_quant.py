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
"""GeGLU (tanh GELU) and per-token FP8 quantization in one Pallas pass."""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.row_blocks import fit_block_rows, vmem_limit_bytes

_SQRT_2_OVER_PI = math.sqrt(2 / math.pi)
_BLOCK_ROWS = 256
COLUMN_ALIGNMENT = 128


def _gelu_tanh(x: jax.Array) -> jax.Array:
    return 0.5 * x * (1.0 + jnp.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)))


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


def block_rows(num_tokens: int, d: int, max_rows: int = _BLOCK_ROWS) -> int:
    return fit_block_rows(num_tokens, 2 * (4 * d + d), max_rows, vmem_limit_bytes())


@functools.partial(
    jax.jit, static_argnames=("quant_dtype", "max_block_rows", "interpret")
)
def gelu_tanh_and_mul_quantize(
    x: jax.Array,
    quant_dtype: jnp.dtype,
    max_block_rows: int = _BLOCK_ROWS,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    num_tokens, width = x.shape
    d = width // 2
    if width % 2 or d % COLUMN_ALIGNMENT:
        raise ValueError(
            f"gelu_tanh_and_mul_quantize: width {width} is not twice a multiple "
            f"of {COLUMN_ALIGNMENT}"
        )
    vmem_limit = vmem_limit_bytes()
    rows = fit_block_rows(num_tokens, 2 * (4 * d + d), max_block_rows, vmem_limit)
    half = pl.BlockSpec((rows, d), lambda i: (i, 0))
    return pl.pallas_call(
        functools.partial(_kernel, act_dtype=x.dtype, quant_dtype=quant_dtype),
        grid=(num_tokens // rows,),
        in_specs=[half, pl.BlockSpec((rows, d), lambda i: (i, 1))],
        out_specs=[half, pl.BlockSpec((rows, 1), lambda i: (i, 0))],
        out_shape=[
            jax.ShapeDtypeStruct((num_tokens, d), quant_dtype),
            jax.ShapeDtypeStruct((num_tokens, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=vmem_limit
        ),
        interpret=interpret,
        name="gelu_tanh_and_mul_quantize",
    )(x, x)
