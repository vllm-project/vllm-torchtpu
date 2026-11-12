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

"""NeoX RoPE: vLLM's reference, and the kernels' one-gather version (LaneRope).

NeoX style (GPT-NeoX's convention, vLLM's is_neox_style=True) splits a head's
rotated dims into two halves and rotates dim i with dim i + n_pairs, rather
than with its neighbor i + 1 (GPT-J style). rope() works on XLA arrays and in
kernels alike.
"""

import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_attn_proj.common import model_config


def rope(
    spec: model_config.Rope | None,
    x: jax.Array,  # (..., D)
    cos: jax.Array | None,  # (..., n_pairs)
    sin: jax.Array | None,  # (..., n_pairs)
) -> jax.Array:  # (..., D)
    """ApplyRotaryEmb.forward_static (NeoX) on the pairs described by `spec`.

    A head's D dims, with p = spec.n_pairs and o = spec.offset:

      x:    [  x1 = x[:p]   | x[p:o] |  x2 = x[o:o+p]  | x[o+p:] ]
      out:  [ x1*cos-x2*sin | x[p:o] | x2*cos+x1*sin   | x[o+p:] ]

    x[p:o] and x[o+p:] pass through unrotated. Both are empty for full NeoX
    RoPE (o == p, 2p == D); partial RoPE has o > p (Gemma 4) or o + p < D.

    Args:
      spec: which dims rotate; None for no RoPE.
      x: (..., D) heads.
      cos: (..., n_pairs), one angle per pair; None iff spec is None.
      sin: as cos.

    Returns:
      x with RoPE applied, same shape and dtype.
    """
    if spec is None:
        return x
    assert cos is not None and sin is not None
    n_pairs, offset, head_dim = spec.n_pairs, spec.offset, x.shape[-1]
    x1, x2 = x[..., :n_pairs], x[..., offset : offset + n_pairs]
    parts = [x1 * cos - x2 * sin]
    if offset > n_pairs:
        parts.append(x[..., n_pairs:offset])
    parts.append(x2 * cos + x1 * sin)
    if offset + n_pairs < head_dim:
        parts.append(x[..., offset + n_pairs :])
    return jnp.concatenate(parts, -1)


# In-kernel.


def gather_lanes(
    x: jax.Array,
    idx: jax.Array,  # (rows, lanes), (rows, lanes) int
) -> jax.Array:  # (rows, lanes)
    """x[r, idx[r, l]]: within each row, lane l reads lane idx[r, l].

    Mosaic lowers this take_along_axis form to one in-register lane permute
    (tpu.dynamic_gather). wrap_negative_indices=False keeps idx's dtype: the
    default casts the int16 indices used for 16-bit x to int32, which doesn't
    lower.

    Args:
      x: (rows, lanes).
      idx: (rows, lanes); int16 for 16-bit x, else int32.

    Returns:
      (rows, lanes), x's dtype.
    """
    return jnp.take_along_axis(
        x, idx, axis=1, mode="promise_in_bounds", wrap_negative_indices=False
    )


def lane_rope_eligible(spec: model_config.Rope | None, head_dim: int) -> bool:
    """Whether NeoX RoPE can be one lane shuffle plus a multiply-add."""
    if spec is None:
        return False
    n_pairs = spec.n_pairs
    num_lanes = pltpu.get_tpu_info().num_lanes
    return (
        spec.offset == n_pairs  # Pair i is (i, i + n_pairs).
        # Power of two, so i + n_pairs == i XOR n_pairs.
        and (n_pairs & (n_pairs - 1)) == 0
        and 2 * n_pairs <= min(head_dim, num_lanes)  # Both halves in one vreg.
    )


def _gather_index(
    values: jax.Array,
    source_dtype: jax.typing.DTypeLike,  # (rows, lanes) int
) -> jax.Array:  # (rows, lanes) int
    """Lane indices in the dtype gather_lanes needs: int16 for 16-bit sources."""
    itemsize = jnp.dtype(source_dtype).itemsize
    return values.astype(jnp.int16 if itemsize == 2 else jnp.int32)


def _pad_lanes(
    values: jax.Array,
    fill: float,  # (tile, n_pairs)
) -> jax.Array:  # (tile, lanes)
    """values padded with `fill` to a full vreg of lanes.

    Args:
      values: (tile, n_pairs) cos or sin.
      fill: 1 for cos, 0 for sin: the identity rotation. Unrotated lanes read
        angle n_pairs, the first pad lane.

    Returns:
      (tile, lanes).
    """
    tile_size, n_pairs = values.shape
    num_lanes = pltpu.get_tpu_info().num_lanes
    pad = jnp.full((tile_size, num_lanes - n_pairs), fill, values.dtype)
    return jnp.concatenate([values, pad], -1)


@dataclasses.dataclass(frozen=True)
class LaneRope:
    """NeoX RoPE as one lane gather. Build it with LaneRope.expand.

    With n = n_pairs, NeoX rotates lane l with lane l ^ n:
      rope(x)[l] = x[l] * cos[l] + x[l ^ n] * signed_sin[l],
    with signed_sin = -sin for l < n and +sin for n <= l < 2n. The per-lane cos
    and signed sin are expanded once per tile, so each head costs one gather
    instead of rope()'s slices and concats. Lanes >= 2n get cos 1 and sin 0.
    With two D = 64 heads per vreg the pattern repeats every D lanes, so both
    heads rotate at once.
    """

    cos: jax.Array  # cos per lane; 1 on unrotated lanes.; (tile, lanes)
    sin: jax.Array  # signed sin per lane; 0 on unrotated lanes.; (tile, lanes)
    partner: jax.Array  # Each lane's NeoX partner, l ^ n_pairs.; (tile, lanes) int

    @classmethod
    def expand(
        cls,
        spec: model_config.Rope,
        head_dim: int,
        cos: jax.Array,  # (tile, n_pairs) bf16
        sin: jax.Array,  # (tile, n_pairs) bf16
        dtype: jax.typing.DTypeLike,
    ) -> "LaneRope":
        """Expands (tile, n_pairs) cos/sin to per-lane factors for `dtype` heads."""
        n_pairs = spec.n_pairs
        tile_size = cos.shape[0]
        num_lanes = pltpu.get_tpu_info().num_lanes
        lane = jax.lax.broadcasted_iota(jnp.int32, (tile_size, num_lanes), 1)
        lane_in_head = lane & (head_dim - 1) if head_dim < num_lanes else lane
        angle = _gather_index(
            jnp.where(
                lane_in_head < 2 * n_pairs, lane_in_head & (n_pairs - 1), n_pairs
            ),
            cos.dtype,
        )
        lane_cos = gather_lanes(_pad_lanes(cos, 1), angle)
        lane_sin = gather_lanes(_pad_lanes(sin, 0), angle)
        sign = jnp.where(lane_in_head < n_pairs, -1.0, 1.0).astype(lane_sin.dtype)
        return cls(
            cos=lane_cos,
            sin=lane_sin * sign,
            partner=_gather_index(lane ^ n_pairs, dtype),
        )

    def __call__(self, x: jax.Array) -> jax.Array:  # (tile, width), (tile, width)
        """RoPE of x; only its first vreg of lanes rotates."""
        num_lanes = self.cos.shape[-1]
        first_vreg = x[:, :num_lanes]
        rotated = (
            first_vreg * self.cos + gather_lanes(first_vreg, self.partner) * self.sin
        )
        if x.shape[-1] == num_lanes:
            return rotated
        return jnp.concatenate([rotated, x[:, num_lanes:]], -1)
