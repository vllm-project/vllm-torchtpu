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
"""Split-e4m3 activations x e4m3 weights for MiniMax-M3 decode linears
(TPU_MINIMAX_M3_FP8_SPLIT_MAX_ROWS)."""

import jax
import jax.numpy as jnp

_E4M3_MAX = 448.0
# The low term is quantized at 1/16 of the row scale: the high term's rounding
# error is at most half an e4m3 ulp (<= 16 * s at the top binade), so the
# scaled residual stays inside e4m3's range.
_SPLIT_LO = 16.0


def split_e4m3_matmul(x: jax.Array, w: jax.Array, w_scale: jax.Array) -> jax.Array:
    """``x @ w * w_scale`` for bf16 ``x[..., K]`` and e4m3 ``w[K, N]`` with the
    activations split into two e4m3 terms and one fp8 x fp8 matmul.

    Per row, ``s`` is a power of two that puts the row's abs-max at or below
    448; ``hi = e4m3(x / s)``, ``lo = e4m3((x / s - hi) * 16)``. Both terms
    share the weight push: the matmul runs over ``[hi; lo]`` (2 x rows) and
    ``out = (y_hi + y_lo / 16) * s * w_scale``, accumulated in f32."""
    lead, k = x.shape[:-1], x.shape[-1]
    x2 = x.reshape(-1, k).astype(jnp.float32)
    rows = x2.shape[0]
    amax = jnp.max(jnp.abs(x2), axis=-1, keepdims=True)
    amax = jnp.where(amax > 0, amax, 1.0)
    # Power-of-two scale: dividing by it is exact.
    s = jnp.exp2(jnp.ceil(jnp.log2(amax / _E4M3_MAX)))
    xs = x2 / s
    hi = xs.astype(jnp.float8_e4m3fn)
    lo = ((xs - hi.astype(jnp.float32)) * _SPLIT_LO).astype(jnp.float8_e4m3fn)
    y = jnp.dot(
        jnp.concatenate([hi, lo], axis=0), w, preferred_element_type=jnp.float32
    )
    out = (y[:rows] + y[rows:] * (1.0 / _SPLIT_LO)) * s
    out = out * w_scale.astype(jnp.float32)
    return out.astype(x.dtype).reshape(*lead, w.shape[-1])
