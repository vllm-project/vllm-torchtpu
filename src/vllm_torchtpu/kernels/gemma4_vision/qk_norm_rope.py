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
"""Fused per-head RMSNorm + multidimensional RoPE for the Gemma-4 vision tower.

HF ``Gemma4VisionAttention`` runs, for both Q and K::

    x = q_norm(x)  # per-head RMSNorm
    x = apply_multidimensional_rope(x, cos, sin, pos)  # 2D axial RoPE

The RoPE splits each 72-channel head into ``ndim = 2`` chunks of 36 and applies
``rotate_half`` (slice + negate + concat) per chunk, which slices 18-channel
pieces. Those pad poorly on 128-lane TPU vregs (b/581011672), and the norm and
the rotation each make their own HBM round trip.

This kernel does both in one pass over a lane-aligned layout. Tokens are rows
and the ``num_heads * head_dim`` channels (16 * 72 = 1152 = 9 * 128) are the
lanes, so no head is sliced out:

  - Per-head sums of squares and their broadcast back are small matmuls with
    constant 0/1 head-membership matrices, each done as two plain bf16 MXU
    passes (no lane slicing, no 6-pass float32 matmul).
  - ``rotate_half`` becomes an in-register lane rotation: every lane reads the
    lane ``half`` channels ahead (first half of a chunk, negated) or behind
    (second half). Partners never leave their chunk, so the wrap-around of the
    row-wide roll is never selected.
  - cos/sin come in tiled across the heads (``tile_cos_sin``). Every layer's
    Q and K share the same tiled tensors, so they are built once per forward.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANE = 128
DEFAULT_BLOCK_TOKENS = 256
DEFAULT_VMEM_LIMIT_BYTES = 64 * 1024 * 1024


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def _rope_layout(head_dim: int, ndim: int) -> tuple[int, int]:
    """Chunk width and half width, exactly as HF ``apply_multidimensional_rope``."""
    chunk = 2 * (head_dim // (2 * ndim))
    if chunk <= 0 or chunk * ndim != head_dim:
        raise ValueError(
            f"head_dim={head_dim} does not split into {ndim} even RoPE chunks"
        )
    return chunk, chunk // 2


def _constants(num_heads: int, head_dim: int, ndim: int):
    """Constant matrices and masks; folded by XLA, fetched once by the kernel."""
    width = num_heads * head_dim
    padded_heads = _round_up(num_heads, LANE)
    chunk, half = _rope_layout(head_dim, ndim)

    lane = jnp.arange(width)
    head_of_lane = lane // head_dim
    channel = lane % head_dim

    # [width, padded_heads]: lane -> its head. 0/1 is exact in bf16, so the
    # matmuls below run as plain bf16 MXU passes.
    head_sum = (head_of_lane[:, None] == jnp.arange(padded_heads)[None, :]).astype(
        jnp.bfloat16
    )
    # [padded_heads, width]: per-head value back onto that head's lanes.
    head_spread = head_sum.T
    # 1 where a lane is in the first half of its RoPE chunk (reads +half,
    # negated), 0 in the second half (reads -half).
    first_half = ((channel % chunk) < half).astype(jnp.float32)[None, :]
    return head_sum, head_spread, first_half, half


def _dot_01(a: jax.Array, m01: jax.Array) -> jax.Array:
    """``a @ m01`` for float32 ``a`` and a bf16 0/1 matrix, in two bf16 passes.

    ``a`` is split into a bf16 high part and a bf16 low part. Every product
    with a 0/1 entry is exact and accumulation is float32, so the result is
    accurate to ~2^-16 relative -- far below the bf16 output rounding -- at a
    third of the cost of a ``Precision.HIGHEST`` float32 matmul.
    """
    hi = a.astype(jnp.bfloat16)
    lo = (a - hi.astype(jnp.float32)).astype(jnp.bfloat16)
    return jnp.dot(hi, m01, preferred_element_type=jnp.float32) + jnp.dot(
        lo, m01, preferred_element_type=jnp.float32
    )


def _kernel(
    x_ref,
    cos_ref,
    sin_ref,
    scale_ref,
    head_sum_ref,
    head_spread_ref,
    first_half_ref,
    out_ref,
    *,
    head_dim: int,
    half: int,
    eps: float,
):
    x = x_ref[...].astype(jnp.float32)  # [tokens, width]
    width = x.shape[-1]

    # Per-head RMSNorm, in float32 like HF's Gemma RMSNorm.
    sum_sq = _dot_01(x * x, head_sum_ref[...])  # [tokens, padded_heads]
    inv_rms = jax.lax.rsqrt(sum_sq * (1.0 / head_dim) + eps)
    inv_rms = _dot_01(inv_rms, head_spread_ref[...])  # [tokens, width]
    y = x * inv_rms * scale_ref[...]

    # rotate_half within every chunk as an in-register lane rotation.
    ahead = pltpu.roll(y, shift=width - half, axis=1)  # lane i reads i + half
    behind = pltpu.roll(y, shift=half, axis=1)  # lane i reads i - half
    rotated = jnp.where(first_half_ref[...] > 0.5, -ahead, behind)

    cos = cos_ref[...].astype(jnp.float32)
    sin = sin_ref[...].astype(jnp.float32)
    out_ref[...] = (y * cos + rotated * sin).astype(out_ref.dtype)


def tile_cos_sin(cos_or_sin: jax.Array, num_heads: int) -> jax.Array:
    """``[tokens, head_dim]`` -> ``[tokens, num_heads * head_dim]``.

    The tiled tensor is shared by every layer's Q and K, so callers build it
    once per forward rather than per kernel call.
    """
    return jnp.tile(cos_or_sin, (1, num_heads))


@functools.partial(
    jax.jit,
    static_argnames=("num_heads", "head_dim", "ndim", "eps", "block_tokens"),
)
def qk_norm_rope(
    x: jax.Array,  # [tokens, num_heads * head_dim]
    cos: jax.Array,  # [tokens, num_heads * head_dim], see tile_cos_sin
    sin: jax.Array,  # [tokens, num_heads * head_dim]
    scale: jax.Array,  # [head_dim], the RMSNorm's effective per-channel multiplier
    *,
    num_heads: int,
    head_dim: int,
    ndim: int = 2,
    eps: float = 1e-6,
    block_tokens: int = DEFAULT_BLOCK_TOKENS,
) -> jax.Array:
    """Per-head RMSNorm followed by HF multidimensional RoPE, in one pass.

    Args:
      x: ``[tokens, num_heads * head_dim]`` projection output (heads contiguous).
      cos, sin: ``[tokens, num_heads * head_dim]``: the per-token
        ``Gemma4VisionRotaryEmbedding`` output (``[H, H, W, W]`` frequency
        layout) tiled across heads with ``tile_cos_sin``.
      scale: ``[head_dim]`` multiplier applied after normalization, i.e. the
        RMSNorm weight in whatever form the module applies it.
      num_heads, head_dim: head layout of ``x``.
      ndim: number of RoPE spatial dimensions (2 for the vision tower).
      eps: RMSNorm epsilon.
      block_tokens: tokens per grid step.

    Returns:
      ``[tokens, num_heads * head_dim]`` in ``x.dtype``.
    """
    tokens, width = x.shape
    if width != num_heads * head_dim:
        raise ValueError(f"x width {width} != {num_heads} * {head_dim}")
    if width % LANE != 0:
        raise ValueError(f"num_heads * head_dim = {width} must be a multiple of 128")
    if cos.shape != x.shape or sin.shape != x.shape:
        raise ValueError("cos/sin must be tiled to x's shape (see tile_cos_sin)")

    head_sum, head_spread, first_half, half = _constants(num_heads, head_dim, ndim)
    scale_row = jnp.tile(scale.astype(jnp.float32), num_heads)[None, :]

    block = min(block_tokens, _round_up(tokens, 8))
    grid = (pl.cdiv(tokens, block),)

    def rows(i):
        return (i, 0)

    def whole(i):
        return (0, 0)

    row_spec = pl.BlockSpec((block, width), rows)
    in_specs = [
        row_spec,  # x
        row_spec,  # cos
        row_spec,  # sin
        pl.BlockSpec(scale_row.shape, whole),
        pl.BlockSpec(head_sum.shape, whole),
        pl.BlockSpec(head_spread.shape, whole),
        pl.BlockSpec(first_half.shape, whole),
    ]
    return pl.pallas_call(
        functools.partial(_kernel, head_dim=head_dim, half=half, eps=eps),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=in_specs,
        out_specs=row_spec,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
        ),
        name="gemma4_vision_qk_norm_rope",
    )(x, cos, sin, scale_row, head_sum, head_spread, first_half)


def reference(
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    scale: jax.Array,
    *,
    num_heads: int,
    head_dim: int,
    ndim: int = 2,
    eps: float = 1e-6,
) -> jax.Array:
    """Unfused XLA reference with HF's structure: norm, then split/rotate/concat."""
    chunk, half = _rope_layout(head_dim, ndim)
    tokens = x.shape[0]
    xh = x.astype(jnp.float32).reshape(tokens, num_heads, head_dim)
    inv_rms = jax.lax.rsqrt(jnp.mean(xh * xh, axis=-1, keepdims=True) + eps)
    y = xh * inv_rms * scale.astype(jnp.float32)
    c = cos.astype(jnp.float32)[:, None, :]
    s = sin.astype(jnp.float32)[:, None, :]
    parts = []
    for k in range(ndim):
        lo = k * chunk
        yk = y[..., lo : lo + chunk]
        rot = jnp.concatenate([-yk[..., half:], yk[..., :half]], axis=-1)
        parts.append(yk * c[..., lo : lo + chunk] + rot * s[..., lo : lo + chunk])
    return jnp.concatenate(parts, axis=-1).reshape(tokens, -1).astype(x.dtype)
