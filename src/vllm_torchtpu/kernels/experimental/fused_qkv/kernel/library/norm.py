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

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.fused_qkv.kernel import configs
from vllm_torchtpu.kernels.experimental.fused_qkv.kernel.library import layouts

Norm = configs.Norm


def norm(
    kind: Norm,
    x: jax.Array,
    w: jax.Array | None,
    eps: float,
    dtype: jnp.dtype | None = None,
) -> jax.Array:
    """vLLM RMSNorm / GemmaRMSNorm forward_native, in dtype (default x's)."""
    dtype = dtype or x.dtype
    if kind == Norm.NONE:
        return x.astype(dtype)
    xf = x.astype(jnp.float32)
    rrms = jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return scale_norm(kind, xf, rrms, w, dtype)


def scale_norm(
    kind: Norm,
    xf: jax.Array,
    rrms: jax.Array | None,
    w: jax.Array | None,
    dtype: jnp.dtype,
) -> jax.Array:
    """norm() of f32 xf, given its 1 / rms (unused for Norm.NONE)."""
    if kind == Norm.NONE:
        return xf.astype(dtype)
    xf = xf * rrms
    match kind:
        case Norm.RMS:
            return (xf.astype(w.dtype) * w).astype(dtype)
        case Norm.RMS_ONE_PLUS_W:
            return (xf * (w.astype(jnp.float32) + 1.0)).astype(dtype)
        case Norm.RMS_NOSCALE:
            return xf.astype(dtype)
        case _:
            raise ValueError(kind)


def add_norm(
    kind: Norm,
    x: jax.Array,
    residual: jax.Array | None,
    w: jax.Array | None,
    eps: float,
) -> tuple[jax.Array, jax.Array | None]:
    """Decoder input norm: (normed x, x + residual or None); the add is f32."""
    if residual is None:
        return norm(kind, x, w, eps), None
    xf = x.astype(jnp.float32) + residual.astype(jnp.float32)
    return norm(kind, xf, w, eps, x.dtype), xf.astype(x.dtype)


# Weight layouts, prepared once at weight load time.


def _multiplier(kind: Norm, w: jax.Array) -> jax.Array:
    w = w.astype(jnp.float32)
    if kind == Norm.RMS_ONE_PLUS_W:
        w = w + 1.0
    return w


def packed_bf16_weight(kind: Norm, w: jax.Array, block: int) -> jax.Array:
    """Norm weight pre-chopped into quant blocks, as bf16 in both u32 halves.

    Both halves, so one stride-0 load broadcasts it over a whole bf16 vreg.
    """
    bits = jax.lax.bitcast_convert_type(
        _multiplier(kind, w).astype(jnp.bfloat16), jnp.uint16
    )
    bits = bits.astype(jnp.uint32)
    return ((bits << 16) | bits).reshape(-1, 1, block)


def f32_weight(kind: Norm, w: jax.Array, block: int) -> jax.Array:
    """Norm weight pre-chopped into quant blocks, in f32."""
    return _multiplier(kind, w).reshape(-1, 1, block)


# In-kernel.


def sum_sq(h: jax.Array) -> jax.Array:
    return jnp.sum(h * h, axis=1, keepdims=True)


def inv_rms(sum_squares: jax.Array, n: int, eps: float) -> jax.Array:
    """1 / rms from the sum of squares over n columns."""
    return jax.lax.rsqrt(sum_squares / n + eps)


def load_weight(
    w_ref: jax.Ref,  # (K / block, 1, block) u32 packed bf16, or f32
    kb: int,
    num_rows: int,
    dtype: jnp.dtype,
) -> jax.Array:
    """Block kb of the norm weight, broadcast to num_rows rows.

    The stride-0 load replicates the stored row on all sublanes, one vld per
    vreg. A packed u32 row bitcasts to twice as many bf16 rows.
    """
    num_sublanes = pltpu.get_tpu_info().num_sublanes
    w = w_ref[kb, pl.ds(0, num_sublanes, stride=0), :]
    if w.dtype != dtype:
        w = pltpu.bitcast(w, dtype)  # (2 * num_sublanes, block)
    return jnp.concatenate([w] * (num_rows // w.shape[0]), axis=0)


def apply_row_scale(acc: jax.Array, row_scale: jax.Array) -> jax.Array:
    """acc times the lane-broadcast per-row 1/rms."""
    return acc * layouts.widen(row_scale, acc.shape[1])
