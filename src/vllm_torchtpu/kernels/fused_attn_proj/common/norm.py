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

"""RMS norms: vLLM's references and the kernels' deferred input norm.

The kernels split the input norm in two (FlashNorm,
https://arxiv.org/abs/2407.09577). While x streams through VMEM they only
multiply by the norm weight and sum the squares. 1/rms is one scalar per row,
so it commutes with the projection, (r * u) @ W = r * (u @ W), and is applied
to the f32 matmul output instead. With fp8 the quant never needs it: the
block scales absorb any per-row factor.
"""

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.fused_attn_proj.common import layout_util, model_config


def norm(
    kind: model_config.Norm,
    x: jax.Array,  # (..., K)
    w: jax.Array | None,  # (1, K)
    eps: float,
    dtype: jax.typing.DTypeLike | None = None,
) -> jax.Array:  # (..., K)
    """vLLM RMSNorm / GemmaRMSNorm forward_native, in dtype (default x's)."""
    dtype = dtype or x.dtype
    if kind == model_config.Norm.NONE:
        return x.astype(dtype)
    x_f32 = x.astype(jnp.float32)
    inv_rms = jax.lax.rsqrt(jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
    return scale_norm(kind, x_f32, inv_rms, w, dtype)


def scale_norm(
    kind: model_config.Norm,
    x_f32: jax.Array,  # (..., K) f32
    inv_rms: jax.Array | None,  # (..., 1) f32
    w: jax.Array | None,  # (1, K)
    dtype: jax.typing.DTypeLike,
) -> jax.Array:  # (..., K)
    """norm() of x_f32, given its 1 / rms (inv_rms; unused for Norm.NONE)."""
    if kind == model_config.Norm.NONE:
        return x_f32.astype(dtype)
    assert inv_rms is not None
    normed = x_f32 * inv_rms
    match kind:
        case model_config.Norm.RMS:
            assert w is not None
            return (normed.astype(w.dtype) * w).astype(dtype)
        case model_config.Norm.RMS_ONE_PLUS_W:
            assert w is not None
            return (normed * (w.astype(jnp.float32) + 1.0)).astype(dtype)
        case model_config.Norm.RMS_NOSCALE:
            return normed.astype(dtype)
        case _:
            raise ValueError(kind)


def add_norm(
    kind: model_config.Norm,
    x: jax.Array,  # (T, K)
    residual: jax.Array | None,  # (T, K)
    w: jax.Array | None,  # (1, K)
    eps: float,
) -> tuple[jax.Array, jax.Array | None]:  # (T, K), (T, K)
    """Decoder input norm: (normed x, x + residual or None); the add is f32."""
    if residual is None:
        return norm(kind, x, w, eps), None
    h_f32 = x.astype(jnp.float32) + residual.astype(jnp.float32)
    return norm(kind, h_f32, w, eps, x.dtype), h_f32.astype(x.dtype)


# Weight layouts, prepared once at weight load time.


def _multiplier(
    kind: model_config.Norm,
    w: jax.Array,  # (1, K)
) -> jax.Array:  # (1, K) f32
    w = w.astype(jnp.float32)
    if kind == model_config.Norm.RMS_ONE_PLUS_W:
        w = w + 1.0
    return w


def packed_bf16_weight(
    kind: model_config.Norm,
    w: jax.Array,
    block: int,  # (1, K)
) -> jax.Array:  # (K/block, 1, block) u32
    """Norm weight pre-chopped into quant blocks, as bf16 in both u32 halves.

    Both halves, so one stride-0 load (layout_util.broadcast_row) broadcasts it
    over a whole bf16 vreg.

    Args:
      kind: the norm; RMS_ONE_PLUS_W adds its 1 here.
      w: (1, K) as loaded.
      block: act_quant_block.

    Returns:
      (K / block, 1, block) u32.
    """
    bits = jax.lax.bitcast_convert_type(
        _multiplier(kind, w).astype(jnp.bfloat16), jnp.uint16
    )
    bits = bits.astype(jnp.uint32)
    return ((bits << 16) | bits).reshape(-1, 1, block)


def f32_weight(
    kind: model_config.Norm,
    w: jax.Array,
    block: int,  # (1, K)
) -> jax.Array:  # (K/block, 1, block) f32
    """Norm weight pre-chopped into quant blocks, in f32 (for broadcast_row)."""
    return _multiplier(kind, w).reshape(-1, 1, block)


# In-kernel.


def sum_sq(h: jax.Array) -> jax.Array:  # (rows, cols) f32, (rows, 1) f32
    """Per-row sum of squares."""
    return jnp.sum(h * h, axis=1, keepdims=True)


def inv_rms_from_sum_sq(
    sum_squares: jax.Array,
    num_cols: int,
    eps: float,  # (rows, 1) f32
) -> jax.Array:  # (rows, 1) f32
    """1 / rms from the sum of squares over num_cols columns."""
    return jax.lax.rsqrt(sum_squares / num_cols + eps)


def apply_row_scale(
    acc: jax.Array,
    row_scale: jax.Array,  # (rows, N) f32, (rows, lanes) f32
) -> jax.Array:  # (rows, N) f32
    """acc times the lane-broadcast per-row 1/rms."""
    return acc * layout_util.repeat_lanes(row_scale, acc.shape[1])
