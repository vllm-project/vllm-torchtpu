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


@functools.partial(jax.jit, static_argnames=("eps", "head_dim", "block_rows"))
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
) -> tuple[jax.Array, jax.Array]:
    """Normalized, rotated ``(q, k)`` from a fused QKV projection output.

    Args:
      qkv: ``(T, 3C)`` bf16 with ``[q | k | v]`` column blocks.
      wq, wk: ``(C,)`` RMSNorm weights.
      cos, sin: ``(T, head_dim)`` per-row tables, each half-dim frequency
        repeated for its (even, odd) pair.
      eps: RMSNorm epsilon.
      head_dim: channels per head (a multiple of 128).

    Returns:
      ``q`` and ``k``, each ``(T, C)`` in ``qkv.dtype``.
    """
    rows, three_dim = qkv.shape
    dim = three_dim // 3
    out = jax.ShapeDtypeStruct((rows, dim), qkv.dtype)
    row_block = pl.BlockSpec((block_rows, dim), lambda i: (i, 0))
    table_block = pl.BlockSpec((block_rows, head_dim), lambda i: (i, 0))
    weight_block = pl.BlockSpec((1, dim), lambda i: (0, 0))
    return pl.pallas_call(
        functools.partial(
            _kernel, eps=eps, head_dim=head_dim, num_heads=dim // head_dim
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
        wq.reshape(1, dim),
        wk.reshape(1, dim),
        cos,
        sin,
        jnp.asarray(_pair_swap(head_dim), jnp.bfloat16),
    )
