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

"""fp8 quantization: vLLM's references and the kernels' sub-channel quant.

The kernels quantize x one act_quant_block of columns at a time, with one
scale per row and block (sub-channel). quantize_block makes those scales and
scaled_dot undoes them: each block's partial dot is scaled before the partial
dots are summed. Scales are stored broadcast across lanes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.experimental.fused_qkv.kernel.library import layouts

F8 = jnp.float8_e4m3fn
F8_MAX = float(jnp.finfo(F8).max)


def quantize_fp8_per_token(
    x: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Dynamic per-row fp8 quant: (x_q, scale), x ≈ x_q * scale.

    vllm_torchtpu quantized_matmul/util.py: quantize_block(x, axis=-1).
    """
    scale = jnp.max(jnp.abs(x), axis=-1, keepdims=True).astype(jnp.float32)
    scale = scale / F8_MAX
    scale = jnp.where(scale == 0, 1.0, scale)
    x_q = (x.astype(jnp.float32) / scale).clip(-F8_MAX, F8_MAX).astype(F8)
    return x_q, scale


def quantize_fp8_per_tensor(x: jax.Array, scale: float) -> jax.Array:
    """fp8 cast with a fixed, calibrated per-tensor scale (the KV cache).

    vllm_torchtpu quantization.py: static_per_tensor_quantize_tensor.
    """
    return (x.astype(jnp.float32) / scale).clip(-F8_MAX, F8_MAX).astype(F8)


# In-kernel.


def quantize_block(
    x: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Sub-channel fp8 quant of one block: (x * bf16(1 / scale), scale).

    The multiply is bf16, two values per lane; vLLM divides in f32.
    """
    amax = jnp.max(jnp.abs(x), axis=1, keepdims=True).astype(jnp.float32)
    scale = amax / F8_MAX
    scale = jnp.where(scale == 0, 1.0, scale)
    x_q = x * (1.0 / scale).astype(x.dtype)
    return x_q.clip(-F8_MAX, F8_MAX).astype(F8), scale


def scaled_dot(
    x_q_ref: jax.Ref,  # (tile, K) fp8
    x_scale_ref: jax.Ref,  # (K / block, tile, lanes) f32, lane-broadcast
    w_ref: jax.Ref,  # (K, N) fp8
    rows: pl.Slice,
    cols: pl.Slice,
) -> jax.Array:
    """x[rows] @ w[:, cols] in f32, rescaled per block by x's scale only."""
    num_blocks = x_scale_ref.shape[0]
    block = x_q_ref.shape[1] // num_blocks
    acc = jnp.zeros((rows.size, cols.size), jnp.float32)
    for kb in range(num_blocks):
        ks = pl.ds(kb * block, block)
        acc += layouts.widen(x_scale_ref[kb, rows], cols.size) * jnp.dot(
            x_q_ref[rows, ks], w_ref[ks, cols], preferred_element_type=jnp.float32
        )
    return acc
