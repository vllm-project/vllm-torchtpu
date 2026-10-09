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

"""NeoX RoPE: vLLM's reference, cos/sin builders and the kernel's lane trick.

rope() works both on XLA arrays and in kernels. The cos_sin_* builders give
the per-token rows of vLLM's cos_sin_cache, for tests and benchmarks.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.fused_qkv.kernel import configs

CosSin = tuple[jax.Array, jax.Array]


def rope(
    spec: configs.Rope | None,
    x: jax.Array,
    cos: jax.Array | None,
    sin: jax.Array | None,
) -> jax.Array:
    """ApplyRotaryEmb.forward_static (NeoX) on the pairs described by `spec`."""
    if spec is None:
        return x
    n, o, d = spec.n_pairs, spec.offset, x.shape[-1]
    x1, x2 = x[..., :n], x[..., o : o + n]
    parts = [x1 * cos - x2 * sin]
    if o > n:
        parts.append(x[..., n:o])
    parts.append(x2 * cos + x1 * sin)
    if o + n < d:
        parts.append(x[..., o + n :])
    return jnp.concatenate(parts, -1)


def _cos_sin(
    positions: jax.Array,
    inv_freq: jax.Array,
    mscale: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    freqs = positions.astype(jnp.float32)[..., None] * inv_freq
    return (
        (jnp.cos(freqs) * mscale).astype(jnp.bfloat16),
        (jnp.sin(freqs) * mscale).astype(jnp.bfloat16),
    )


def _inv_freq(theta: float, n_pairs: int, denom: int) -> jax.Array:
    return 1.0 / theta ** (jnp.arange(0, 2 * n_pairs, 2, dtype=jnp.float32) / denom)


def cos_sin_default(positions: jax.Array, rotary_dim: int, theta: float) -> CosSin:
    """vLLM RotaryEmbedding."""
    return _cos_sin(positions, _inv_freq(theta, rotary_dim // 2, rotary_dim))


def cos_sin_mrope_interleaved(
    positions: jax.Array,
    rotary_dim: int,
    theta: float,
    section: tuple[int, int, int],
) -> CosSin:
    """vLLM MRotaryEmbedding (interleaved): t/h/w positions per token."""
    cos, sin = _cos_sin(positions, _inv_freq(theta, rotary_dim // 2, rotary_dim))
    c = jnp.arange(rotary_dim // 2)
    is_h = (c % 3 == 1) & (c < section[1] * 3)
    is_w = (c % 3 == 2) & (c < section[2] * 3)

    def pick(t):
        return jnp.where(is_w, t[2], jnp.where(is_h, t[1], t[0]))

    return pick(cos), pick(sin)


def cos_sin_proportional(
    positions: jax.Array,
    head_dim: int,
    partial_rotary_factor: float,
    theta: float,
) -> CosSin:
    """vLLM Gemma4RotaryEmbedding, only the non-identity angles.

    The rest have cos 1 and sin 0; Rope(n_pairs, head_dim // 2) leaves those
    lanes unchanged.
    """
    n_pairs = int(head_dim * partial_rotary_factor) // 2
    return _cos_sin(positions, _inv_freq(theta, n_pairs, head_dim))


def cos_sin_yarn(
    positions: jax.Array,
    rotary_dim: int,
    theta: float,
    factor: float,
    original_max_position: int,
    beta_fast: int = 32,
    beta_slow: int = 1,
    truncate: bool = True,
) -> CosSin:
    """vLLM YaRNScalingRotaryEmbedding, with mscale folded into cos/sin."""

    def correction_dim(rot):
        return (
            rotary_dim
            * math.log(original_max_position / (rot * 2 * math.pi))
            / (2 * math.log(theta))
        )

    low, high = correction_dim(beta_fast), correction_dim(beta_slow)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, rotary_dim - 1)
    if low == high:
        high += 0.001
    pos_freqs = theta ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim)
    ramp = jnp.clip(
        (jnp.arange(rotary_dim // 2, dtype=jnp.float32) - low) / (high - low),
        0,
        1,
    )
    mask = 1 - ramp
    inv_freq = (1.0 / (factor * pos_freqs)) * (1 - mask) + (1.0 / pos_freqs) * mask
    mscale = 0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0
    return _cos_sin(positions, inv_freq, mscale)


# In-kernel.


def gather_lanes(x: jax.Array, idx: jax.Array) -> jax.Array:
    """x[r, idx[r, l]]."""
    return jax.lax.gather(
        x,
        idx[..., None],
        jax.lax.GatherDimensionNumbers(
            offset_dims=(),
            collapsed_slice_dims=(1,),
            start_index_map=(1,),
            operand_batching_dims=(0,),
            start_indices_batching_dims=(0,),
        ),
        slice_sizes=(1, 1),
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )


def lane_rope_eligible(spec: configs.Rope | None, head_dim: int) -> bool:
    """Whether NeoX RoPE can be one lane shuffle plus a multiply-add."""
    if spec is None:
        return False
    n = spec.n_pairs
    num_lanes = pltpu.get_tpu_info().num_lanes
    return (
        spec.offset == n  # Pair i is (i, i + n).
        and (n & (n - 1)) == 0  # Power of two, so i + n == i XOR n.
        and 2 * n <= min(head_dim, num_lanes)  # Both halves in the first vreg.
    )


class LaneRope:
    """NeoX RoPE as one lane gather.

    With n = n_pairs, NeoX rotates lane l with lane l ^ n:
      rope(x)[l] = x[l] * cos[l] + x[l ^ n] * signed_sin[l],
    with signed_sin = -sin for l < n and +sin for n <= l < 2n. The per-lane cos
    and signed sin are expanded once per tile, so each head costs one gather
    instead of rope()'s slices and concats. Lanes >= 2n get cos 1 and sin 0.
    With two D = 64 heads per vreg the pattern repeats every D lanes, so both
    heads rotate at once.
    """

    def __init__(
        self,
        spec: configs.Rope,
        head_dim: int,
        cos: jax.Array,
        sin: jax.Array,
        dtype: jnp.dtype,
    ):
        """Expands cos/sin to per-lane factors for rotating `dtype` heads."""
        n = spec.n_pairs
        tile_size = cos.shape[0]
        num_lanes = pltpu.get_tpu_info().num_lanes

        def index(v, source_dtype):  # 16-bit sources gather with int16 indices.
            itemsize = jnp.dtype(source_dtype).itemsize
            return v.astype(jnp.int16 if itemsize == 2 else jnp.int32)

        def pad(v, fill):  # Angle n, the first pad lane, is the identity.
            fill = jnp.full((tile_size, num_lanes - n), fill, v.dtype)
            return jnp.concatenate([v, fill], -1)

        lane = jax.lax.broadcasted_iota(jnp.int32, (tile_size, num_lanes), 1)
        lane_in_head = lane & (head_dim - 1) if head_dim < num_lanes else lane
        angle = index(
            jnp.where(lane_in_head < 2 * n, lane_in_head & (n - 1), n), cos.dtype
        )
        # cos, sin and partner are all (tile, lanes).
        self.cos = gather_lanes(pad(cos, 1), angle)
        sin = gather_lanes(pad(sin, 0), angle)
        self.sin = sin * jnp.where(lane_in_head < n, -1.0, 1.0).astype(sin.dtype)
        self.partner = index(lane ^ n, dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        """RoPE of x; only its first vreg of lanes rotates."""
        num_lanes = self.cos.shape[-1]
        x0 = x[:, :num_lanes]
        y0 = x0 * self.cos + gather_lanes(x0, self.partner) * self.sin
        if x.shape[-1] == num_lanes:
            return y0
        return jnp.concatenate([y0, x[:, num_lanes:]], -1)
