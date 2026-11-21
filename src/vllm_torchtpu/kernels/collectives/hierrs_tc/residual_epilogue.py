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
"""VMEM residual-stream epilogue fused into hierarchical reduce-scatter kernels."""

import jax
import jax.numpy as jnp
from jax import lax

from vllm_torchtpu.kernels.quantized_matmul.util import quantize_block


def _rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True)
    return (x_f32 * lax.rsqrt(variance + eps) * weight).astype(jnp.bfloat16)


def apply_residual_epilogue(
    reduced: jax.Array,
    residual: jax.Array,
    *,
    post_norm_weight: jax.Array | None,
    scale: jax.Array | None,
    next_norm_weight: jax.Array | None,
    eps: float,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
    """Computes updated residual rows and optional FP8-quantized next RMSNorm.

    `reduced` (`bf16` or `f32`) and `residual` (`bf16`) have shape
    `[rows, hidden]`; weight operands are `f32[1, hidden]`. Computes:

        residual_out = (residual + rms_norm(reduced, post_norm_weight)) [* scale]

    (skipping `post_norm_weight` when `None`). When `next_norm_weight` is
    provided, also computes `rms_norm(residual_out, next_norm_weight)` quantized
    per row to `float8_e4m3fn` plus `f32[rows, 1]` row scales.
    """
    if post_norm_weight is not None:
        reduced = _rms_norm(reduced, post_norm_weight, eps)
    residual_out = (reduced.astype(jnp.float32) + residual.astype(jnp.float32)).astype(
        jnp.bfloat16
    )
    if scale is not None:
        residual_out = residual_out * scale.astype(jnp.bfloat16)
    if next_norm_weight is None:
        return residual_out, None, None
    x_q, x_scale = quantize_block(
        _rms_norm(residual_out, next_norm_weight, eps),
        axis=-1,
        target_dtype=jnp.float8_e4m3fn,
    )
    return residual_out, x_q, x_scale.astype(jnp.float32)
