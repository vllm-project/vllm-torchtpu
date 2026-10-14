# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused attention over gathered windows in a shared word cache."""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(q_ref, words_ref, mask_ref, max_ref, den_ref, num_ref, *, scale):
    queries, heads, dim = q_ref.shape
    chunks, _, keys, _ = words_ref.shape
    key_block = words_ref.shape[-1]
    q = q_ref[...].reshape(queries * heads, dim)

    def attend_block(block, state):
        old_max, old_den, old_num = state
        key_slice = pl.ds(block * key_block, key_block)
        # Unpack one key tile in VMEM; transposing the complete window spills
        # registers and can exceed VMEM even for a single query tile.
        words = jnp.concatenate(
            [words_ref[c, 0, key_slice] for c in range(chunks)], axis=1
        ).reshape(key_block, 1, dim // 2)
        packed = pltpu.bitcast(words, q_ref.dtype)
        kv = packed.reshape(key_block, dim)
        valid = mask_ref[0, :, key_slice]
        used = valid.astype(jnp.int32).sum(axis=0, keepdims=True).T > 0
        kv = jnp.where(used, kv, 0)
        scores = (
            jax.lax.dot_general(
                q, kv, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
            ).reshape(queries, heads, key_block)
            * scale
        )
        scores = jnp.where(valid[:, None, :], scores, -jnp.inf)
        maximum = jnp.maximum(old_max, scores.max(axis=-1))
        correction = jnp.exp(old_max - maximum)
        probabilities = jnp.exp(scores - maximum[..., None])
        denominator = old_den * correction + probabilities.sum(axis=-1)
        numerator = jax.lax.dot_general(
            probabilities.reshape(queries * heads, key_block).astype(q_ref.dtype),
            kv,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).reshape(queries, heads, dim)
        numerator += old_num * correction[..., None]
        return maximum, denominator, numerator

    maximum, denominator, numerator = jax.lax.fori_loop(
        0,
        keys // key_block,
        attend_block,
        (
            jnp.full((queries, heads), -1e30, jnp.float32),
            jnp.zeros((queries, heads), jnp.float32),
            jnp.zeros((queries, heads, dim), jnp.float32),
        ),
    )
    max_ref[0, 0, 0, 0] = maximum.reshape(-1)
    den_ref[0, 0, 0, 0] = denominator.reshape(-1)
    num_ref[...] = numerator


@functools.partial(jax.jit, static_argnames=("scale", "interpret"))
def attend_windows(
    q: jax.Array,
    words: jax.Array,
    mask: jax.Array,
    *,
    scale: float,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return FP32 max, denominator and numerator for independent windows.

    ``words[dim // 256, groups, keys, 128]`` pairs corresponding elements
    of the two contiguous feature halves. Each group owns queries from
    ``q[tokens, heads, dim]``; ``mask[groups, queries, keys]`` defines their
    exact windows, including request boundaries and padded tokens.
    """
    tokens, heads, dim = q.shape
    groups, queries, keys = mask.shape
    if (
        q.dtype != jnp.bfloat16
        or words.dtype != jnp.int32
        or dim % 256
        or words.shape != (dim // 256, groups, keys, 128)
        or tokens != groups * queries
        or heads % min(heads, 8)
        or keys % 128
    ):
        raise ValueError("Expected BF16 queries and aligned, packed window words")
    head_block = min(heads, 8)
    head_tiles = heads // head_block
    # Bound each score tile to at most one lane-width of flattened queries.
    query_block = math.gcd(queries, words.shape[-1] // head_block)
    query_tiles = queries // query_block
    stats_shape = (groups, query_tiles, head_tiles, 1, query_block * head_block)
    maximum, denominator, numerator = pl.pallas_call(
        functools.partial(_kernel, scale=scale),
        grid=(groups, query_tiles, head_tiles),
        in_specs=[
            pl.BlockSpec(
                (query_block, head_block, dim),
                lambda g, t, h: (g * query_tiles + t, h, 0),
            ),
            pl.BlockSpec((dim // 256, 1, keys, 128), lambda g, t, h: (0, g, 0, 0)),
            pl.BlockSpec((1, query_block, keys), lambda g, t, h: (g, t, 0)),
        ],
        out_specs=[
            pl.BlockSpec(
                (1, 1, 1, 1, query_block * head_block), lambda g, t, h: (g, t, h, 0, 0)
            ),
            pl.BlockSpec(
                (1, 1, 1, 1, query_block * head_block), lambda g, t, h: (g, t, h, 0, 0)
            ),
            pl.BlockSpec(
                (query_block, head_block, dim),
                lambda g, t, h: (g * query_tiles + t, h, 0),
            ),
        ],
        out_shape=[
            jax.ShapeDtypeStruct(stats_shape, jnp.float32),
            jax.ShapeDtypeStruct(stats_shape, jnp.float32),
            jax.ShapeDtypeStruct(q.shape, jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel")
        ),
        interpret=interpret,
        name="swa_window_attention",
    )(q, words, mask)

    def stats(x):
        return (
            x.reshape(groups, query_tiles, head_tiles, query_block, head_block)
            .transpose(0, 1, 3, 2, 4)
            .reshape(tokens, heads)
        )

    return stats(maximum), stats(denominator), numerator
