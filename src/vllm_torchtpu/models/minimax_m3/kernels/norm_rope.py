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
"""Per-head RMSNorm apply + partial NeoX RoPE of MiniMax M3's fused qkv (Pallas).

`attention.norm_rope_heads` normalizes the q, k, index_q and index_k heads of
``[q | k | v | index_q | index_k]`` and rotates their first ``rot`` lanes. In
XLA that is a cat, the norm apply, a split into rotated / passed halves (f32),
the two NeoX products and a cat back: five passes over ``[T, 11, 128]`` with
f32 intermediates, ~130 us per layer at 8K tokens (c48 twin) against ~15 us
for one read of the qkv and one write of the result.

This kernel does the elementwise part in one pass, with the operations and
rounding points of the XLA graph:

    y = bf16(f32(x) * r * w)                  # rms_norm (ir_ops.rms_norm)
    o1 = y1 * cos - y2 * sin                  # ApplyRotaryEmb, f32
    o2 = y2 * cos + y1 * sin
    out = bf16(cat(o1, o2, y_pass))

``r = rsqrt(mean(x^2) + eps)`` stays in XLA (the reduction order is XLA's),
so the result matches the XLA path bit for bit. ``o1`` is computed as
``y1 * cos + y2 * (-sin)``, which is the same IEEE value.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Rows per grid step (a [512, 12 * 128] bf16 block is 1.5 MiB).
MAX_BLOCK_ROWS = 512


def _kernel(qkv_ref, r_ref, w_ref, cs_ref, o_ref, *, sources, head_dim, rot):
    half = rot // 2
    shape = (qkv_ref.shape[0], head_dim)
    lane = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
    # cs: [cos (half) | sin (half) | 0 ...] per row.
    cs = cs_ref[...].astype(jnp.float32)
    ahead = pltpu.roll(cs, head_dim - half, 1)  # lane l <- l + half
    behind = pltpu.roll(cs, half, 1)  # lane l <- l - half
    cos = jnp.where(lane < half, cs, behind)  # cos on both halves of rot
    sin = jnp.where(lane < half, -ahead, cs)  # -sin, +sin
    rotated = lane < rot
    r = r_ref[...]
    w = w_ref[...]
    for j, src in enumerate(sources):
        x = qkv_ref[:, src * head_dim : (src + 1) * head_dim].astype(jnp.float32)
        y = (x * r[:, j : j + 1] * w[j : j + 1, :]).astype(jnp.bfloat16)
        y = y.astype(jnp.float32)
        swapped = jnp.where(
            lane < half,
            pltpu.roll(y, head_dim - half, 1),
            pltpu.roll(y, half, 1),
        )
        out = jnp.where(rotated, y * cos + swapped * sin, y)
        o_ref[:, j * head_dim : (j + 1) * head_dim] = out.astype(o_ref.dtype)


def norm_rope(
    qkv: jax.Array,
    r: jax.Array,
    w: jax.Array,
    cos_sin: jax.Array,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rot: int,
    interpret: bool = False,
) -> jax.Array:
    """Normed and rotated ``[T, (num_heads + 2 num_kv_heads + 1) * head_dim]``
    heads ``[q | k | index_q | index_k]`` of ``qkv [T, (nq + 3 nkv + 1) *
    head_dim]`` (v skipped).

    ``r [T, H]`` f32 is each head's ``rsqrt(mean(x^2) + eps)``, ``w [H,
    head_dim]`` f32 its norm weight, ``cos_sin [T, head_dim]`` the rope cache
    rows ``[cos | sin | 0 ...]`` (the first ``rot`` lanes rotated, the rest
    passed). ``interpret`` runs on the host, for tests without a TPU.
    """
    tokens = qkv.shape[0]
    nq, nkv = num_heads, num_kv_heads
    sources = tuple(range(nq + nkv)) + tuple(range(nq + 2 * nkv, nq + 3 * nkv + 1))
    heads = len(sources)
    assert r.shape == (tokens, heads), (r.shape, tokens, heads)
    assert w.shape == (heads, head_dim), w.shape
    assert cos_sin.shape[0] == tokens and cos_sin.shape[1] == head_dim
    block = min(MAX_BLOCK_ROWS, tokens)
    return pl.pallas_call(
        functools.partial(_kernel, sources=sources, head_dim=head_dim, rot=rot),
        grid=(pl.cdiv(tokens, block),),
        in_specs=[
            pl.BlockSpec((block, qkv.shape[1]), lambda i: (i, 0)),
            pl.BlockSpec((block, heads), lambda i: (i, 0)),
            pl.BlockSpec((heads, head_dim), lambda i: (0, 0)),
            pl.BlockSpec((block, head_dim), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((block, heads * head_dim), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((tokens, heads * head_dim), qkv.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="minimax_m3_norm_rope",
    )(qkv, r, w, cos_sin)


def minimax_m3_norm_rope(
    qkv: jax.Array,
    r: jax.Array,
    w: jax.Array,
    cos_sin: jax.Array,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rot: int,
) -> jax.Array:
    """`norm_rope` with a jax_op-compatible signature (arrays, then statics).
    ``cos_sin [T, rot]`` (``[cos | sin]``) is zero-padded to ``head_dim``."""
    pad = jnp.pad(cos_sin, ((0, 0), (0, head_dim - rot)))
    return norm_rope(qkv, r, w, pad, num_heads, num_kv_heads, head_dim, rot)
