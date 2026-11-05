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
"""Fused tanh GeGLU activation and per-token FP8 quantization kernel."""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.row_blocks import fit_block_rows, get_vmem_capacity_bytes

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_GELU_COEFF = 0.044715
_BLOCK_ROWS = 256
# The second `[rows, d]` BlockSpec starts at column `d`, requiring lane alignment.
COLUMN_ALIGNMENT = 128


def _gelu_and_mul_quantize_kernel(gate_ref, up_ref, q_ref, scale_ref, *, quant_dtype):
    info = jnp.finfo(quant_dtype)
    bf16_max = float(jnp.finfo(jnp.bfloat16).max)
    gate = gate_ref[...]
    up = up_ref[...]
    # Rewrite 0.5 * x * (1 + tanh(z)) as x * sigmoid(2 * z) to avoid
    # cancellation near tanh(z) = -1 in bf16 arithmetic.
    neg_2z = gate * (
        jnp.bfloat16(-2.0 * _SQRT_2_OVER_PI)
        + jnp.bfloat16(-2.0 * _SQRT_2_OVER_PI * _GELU_COEFF) * (gate * gate)
    )
    y = (gate * up) * (1.0 / (1.0 + jnp.exp(neg_2z)))
    scale = jnp.max(jnp.abs(y), axis=-1, keepdims=True).astype(jnp.float32)
    scale = scale / float(info.max)
    scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)
    inv_scale = jnp.minimum(1.0 / scale, bf16_max).astype(jnp.bfloat16)
    q = jnp.clip(y * inv_scale, float(info.min), float(info.max))
    q_ref[...] = q.astype(quant_dtype)
    scale_ref[...] = scale


def block_rows(
    num_tokens: int,
    d: int,
    max_rows: int = _BLOCK_ROWS,
    vmem_limit_bytes: int | None = None,
) -> int:
    """Compute row tile size for ``[num_tokens, 2 * d]`` bf16 gate/up inputs.

    Each row buffers ``2 * d`` bf16 input elements (4 * d bytes) and ``d`` 1-byte
    FP8 output elements, both double-buffered across pipeline stages.
    """
    return fit_block_rows(
        num_tokens,
        2 * (4 * d + d),
        max_rows,
        vmem_limit_bytes=vmem_limit_bytes,
    )


@functools.partial(
    jax.jit,
    static_argnames=("quant_dtype", "max_block_rows", "interpret"),
)
def gelu_and_mul_quantize(
    gate_up: jax.Array,
    quant_dtype: jnp.dtype = jnp.float8_e4m3fn,
    *,
    max_block_rows: int = _BLOCK_ROWS,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Compute tanh GeGLU on ``gate_up`` and quantize per row to ``quant_dtype``.

    Args:
        gate_up: ``[..., 2 * d]`` bfloat16 tensor holding concatenated gate and up
            projections along the trailing axis.
        quant_dtype: Target 8-bit floating-point dtype.
        max_block_rows: Upper bound on the Pallas row tile size.
        interpret: Run the Pallas kernel in CPU interpret mode.

    Returns:
        Tuple ``(x_q, x_scale)`` with shapes ``(..., d)`` in ``quant_dtype`` and
        ``(..., 1)`` in ``float32``.
    """
    if gate_up.ndim < 2:
        raise ValueError(
            f"gelu_and_mul_quantize expects rank >= 2, got {gate_up.shape}"
        )
    if gate_up.dtype != jnp.bfloat16:
        raise ValueError(
            f"gelu_and_mul_quantize requires bfloat16 input, got {gate_up.dtype}"
        )
    quant_dtype = jnp.dtype(quant_dtype)
    if not jnp.issubdtype(quant_dtype, jnp.floating) or quant_dtype.itemsize != 1:
        raise ValueError(
            f"gelu_and_mul_quantize requires an 8-bit float quant_dtype, "
            f"got {quant_dtype}"
        )
    orig_prefix = gate_up.shape[:-1]
    width = gate_up.shape[-1]
    d = width // 2
    if width <= 0 or width % 2 != 0 or d % COLUMN_ALIGNMENT != 0:
        raise ValueError(
            f"gelu_and_mul_quantize width {width} must be a positive multiple of "
            f"{2 * COLUMN_ALIGNMENT}"
        )
    x_2d = gate_up.reshape(-1, width)
    num_tokens = x_2d.shape[0]
    vmem_limit = get_vmem_capacity_bytes()
    rows = block_rows(num_tokens, d, max_block_rows, vmem_limit_bytes=vmem_limit)
    half_spec = pl.BlockSpec((rows, d), lambda i: (i, 0))
    q_2d, scale_2d = pl.pallas_call(
        functools.partial(_gelu_and_mul_quantize_kernel, quant_dtype=quant_dtype),
        grid=(num_tokens // rows,),
        in_specs=[half_spec, pl.BlockSpec((rows, d), lambda i: (i, 1))],
        out_specs=[half_spec, pl.BlockSpec((rows, 1), lambda i: (i, 0))],
        out_shape=[
            jax.ShapeDtypeStruct((num_tokens, d), quant_dtype),
            jax.ShapeDtypeStruct((num_tokens, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit,
        ),
        interpret=interpret,
        name="gelu_and_mul_quantize",
    )(x_2d, x_2d)
    return q_2d.reshape(*orig_prefix, d), scale_2d.reshape(*orig_prefix, 1)
