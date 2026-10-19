# SPDX-License-Identifier: Apache-2.0
"""Fused q/k RMSNorm + interleaved RoPE for Wan self-attention on TPU.

Wan's self-attention normalizes q and k with an RMSNorm over all
``heads * head_dim`` channels, then rotates interleaved (even, odd) channel
pairs of every head. This kernel reads the q and k column blocks of the
fused QKV projection output once and writes the normalized, rotated q and k
once, instead of separate XLA passes for the norm statistics, the norm apply
and the rotation.

Per row of q (and k), with ``C = heads * head_dim``::

    y = bf16(x * rsqrt(mean(x ^ 2) + eps) * w)  # RMSNorm, f32 math
    out = y * cos + (y @ P) * sin  # per head, f32 math

``cos``/``sin`` hold each half-dim frequency repeated for its pair, and ``P``
is the signed pair-swap permutation (``(y @ P)[2i] = -y[2i+1]``,
``(y @ P)[2i+1] = y[2i]``). The permutation runs on the otherwise idle MXU
and is exact (one +-1 term per output).

Two static options cover MiniMax-H3's DiT self-attention (defaults keep the
Wan kernel above unchanged):

* ``per_head_norm``: RMSNorm statistics per ``head_dim``-lane head with one
  shared ``(head_dim,)`` weight (``F.rms_norm(q, (head_dim,), w)`` on
  ``[T, heads, head_dim]``), instead of over all ``C`` channels.
* ``half_split_rotary_dim=R``: NeoX ``rotate_half`` pairs ``(d, d + R/2)``
  over the first ``R`` lanes of each head, the rest passed through. ``P`` is
  then ``(y @ P)[d] = -y[d + R/2]``, ``(y @ P)[d + R/2] = y[d]`` for
  ``d < R/2`` and zero beyond ``R``; the caller's ``cos`` table holds
  ``[c, c, 1...]`` and ``sin`` ``[s, s, 0...]`` per row, so pass-through lanes
  come out as ``y``.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_BLOCK_ROWS = 256


def _pair_swap(head_dim: int) -> np.ndarray:
    p = np.zeros((head_dim, head_dim), np.float32)
    even = np.arange(0, head_dim, 2)
    p[even + 1, even] = -1.0
    p[even, even + 1] = 1.0
    return p


def _half_split_rotate(head_dim: int, rotary_dim: int) -> np.ndarray:
    """Signed NeoX ``rotate_half`` over the first ``rotary_dim`` lanes."""
    p = np.zeros((head_dim, head_dim), np.float32)
    half = rotary_dim // 2
    first = np.arange(half)
    p[first + half, first] = -1.0
    p[first, first + half] = 1.0
    return p


def _per_head_kernel(
    q_ref: jax.Ref,
    k_ref: jax.Ref,
    wq_ref: jax.Ref,
    wk_ref: jax.Ref,
    cos_ref: jax.Ref,
    sin_ref: jax.Ref,
    p_ref: jax.Ref,
    q_out_ref: jax.Ref,
    k_out_ref: jax.Ref,
    *,
    eps: float,
    head_dim: int,
    num_heads: int,
) -> None:
    cos = cos_ref[...].astype(jnp.float32)
    sin = sin_ref[...].astype(jnp.float32)
    inv_dim = 1.0 / head_dim
    heads = [slice(h * head_dim, (h + 1) * head_dim) for h in range(num_heads)]
    for x_ref, w_ref, o_ref in ((q_ref, wq_ref, q_out_ref), (k_ref, wk_ref, k_out_ref)):
        w = w_ref[...].astype(jnp.float32)
        for sl in heads:
            x = x_ref[:, sl].astype(jnp.float32)
            sum_sq = jnp.sum(jnp.square(x), -1, keepdims=True)
            inv_rms = jax.lax.rsqrt(sum_sq * inv_dim + eps)
            y = (x * inv_rms * w).astype(jnp.bfloat16)
            rotated = jnp.dot(y, p_ref[...], preferred_element_type=jnp.float32)
            o_ref[:, sl] = (y.astype(jnp.float32) * cos + rotated * sin).astype(
                o_ref.dtype
            )


def _kernel(
    q_ref: jax.Ref,
    k_ref: jax.Ref,
    wq_ref: jax.Ref,
    wk_ref: jax.Ref,
    cos_ref: jax.Ref,
    sin_ref: jax.Ref,
    p_ref: jax.Ref,
    q_out_ref: jax.Ref,
    k_out_ref: jax.Ref,
    *,
    eps: float,
    head_dim: int,
    num_heads: int,
) -> None:
    cos = cos_ref[...].astype(jnp.float32)
    sin = sin_ref[...].astype(jnp.float32)
    inv_dim = 1.0 / (head_dim * num_heads)
    heads = [slice(h * head_dim, (h + 1) * head_dim) for h in range(num_heads)]
    for x_ref, w_ref, o_ref in ((q_ref, wq_ref, q_out_ref), (k_ref, wk_ref, k_out_ref)):
        # Row statistics over all heads first; the head loop keeps the f32
        # temporaries at (block_rows, head_dim).
        sum_sq = sum(
            jnp.sum(jnp.square(x_ref[:, sl].astype(jnp.float32)), -1, keepdims=True)
            for sl in heads
        )
        inv_rms = jax.lax.rsqrt(sum_sq * inv_dim + eps)
        for sl in heads:
            x = x_ref[:, sl].astype(jnp.float32)
            y = (x * inv_rms * w_ref[:, sl].astype(jnp.float32)).astype(jnp.bfloat16)
            swapped = jnp.dot(y, p_ref[...], preferred_element_type=jnp.float32)
            o_ref[:, sl] = (y.astype(jnp.float32) * cos + swapped * sin).astype(
                o_ref.dtype
            )


@functools.partial(
    jax.jit,
    static_argnames=(
        "eps",
        "head_dim",
        "block_rows",
        "per_head_norm",
        "half_split_rotary_dim",
    ),
)
def qk_norm_rope(
    qkv: jax.Array,
    wq: jax.Array,
    wk: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *,
    eps: float,
    head_dim: int,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    per_head_norm: bool = False,
    half_split_rotary_dim: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Normalized, rotated ``(q, k)`` from a fused QKV projection output.

    Args:
      qkv: ``(T, 3C)`` bf16 with ``[q | k | v]`` column blocks.
      wq, wk: ``(C,)`` RMSNorm weights; ``(head_dim,)`` with
        ``per_head_norm``.
      cos, sin: ``(T, head_dim)`` per-row tables, each half-dim frequency
        repeated for its (even, odd) pair; with ``half_split_rotary_dim=R``
        ``[cos(theta) | cos(theta) | 1...]`` and
        ``[sin(theta) | sin(theta) | 0...]`` with ``theta`` of width ``R/2``.
      eps: RMSNorm epsilon.
      head_dim: channels per head (a multiple of 128).
      per_head_norm: RMSNorm statistics per head instead of per row.
      half_split_rotary_dim: rotate NeoX half-split pairs over this many
        leading lanes of each head instead of interleaved pairs over all.

    Returns:
      ``q`` and ``k``, each ``(T, C)`` in ``qkv.dtype``.
    """
    rows, three_dim = qkv.shape
    dim = three_dim // 3
    out = jax.ShapeDtypeStruct((rows, dim), qkv.dtype)
    row_block = pl.BlockSpec((block_rows, dim), lambda i: (i, 0))
    table_block = pl.BlockSpec((block_rows, head_dim), lambda i: (i, 0))
    weight_dim = head_dim if per_head_norm else dim
    weight_block = pl.BlockSpec((1, weight_dim), lambda i: (0, 0))
    kernel = _per_head_kernel if per_head_norm else _kernel
    if half_split_rotary_dim is None:
        rotate = _pair_swap(head_dim)
    else:
        rotate = _half_split_rotate(head_dim, half_split_rotary_dim)
    return pl.pallas_call(
        functools.partial(
            kernel, eps=eps, head_dim=head_dim, num_heads=dim // head_dim
        ),
        grid=(pl.cdiv(rows, block_rows),),
        in_specs=[
            row_block,  # q columns of qkv
            pl.BlockSpec((block_rows, dim), lambda i: (i, 1)),  # k columns
            weight_block,
            weight_block,
            table_block,
            table_block,
            pl.BlockSpec((head_dim, head_dim), lambda i: (0, 0)),
        ],
        out_specs=[row_block, row_block],
        out_shape=[out, out],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(
        qkv,
        qkv,
        wq.reshape(1, weight_dim),
        wk.reshape(1, weight_dim),
        cos,
        sin,
        jnp.asarray(rotate, jnp.bfloat16),
    )
