# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 linear head with a fused first-index argmax."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Qualified on v7 with full-K reuse and compact partials; other shapes fall back.
TUNED_TILES = {(288, 5120, 129280): 1280}


def ordered_bf16(value: jax.Array) -> jax.Array:
    value = value.astype(jnp.float32)
    bits = jax.lax.bitcast_convert_type(value, jnp.int32) >> 16
    magnitude_mask = jnp.iinfo(jnp.int16).max
    order = bits ^ ((bits >> 15) & magnitude_mask)
    order = jnp.where((bits & magnitude_mask) == 0, 0, order)
    return jnp.where(jnp.isnan(value), magnitude_mask, order)


def fused_greedy_head(hidden: jax.Array, weight: jax.Array) -> jax.Array:
    """Return first argmax IDs after BF16 rounding, for packed [tiles,K,N] W."""
    rows, width = hidden.shape
    tiles, packed_width, tile = weight.shape
    assert hidden.dtype == weight.dtype == jnp.bfloat16
    assert width == packed_width
    assert TUNED_TILES[(rows, width, tiles * tile)] == tile
    shift = (tile - 1).bit_length()
    mask = (1 << shift) - 1

    def kernel(hidden_ref, weight_ref, values_ref, indices_ref):
        accum = jnp.dot(
            hidden_ref[...], weight_ref[...], preferred_element_type=jnp.float32
        )
        rounded = accum.astype(jnp.bfloat16).astype(jnp.float32)
        keys = (ordered_bf16(rounded) << shift) | (
            mask - jnp.arange(tile, dtype=jnp.int32)[None, :]
        )
        best = jnp.max(keys, axis=1)
        values_ref[0, :] = best >> shift
        indices_ref[0, :] = pl.program_id(0) * tile + mask - (best & mask)

    values, indices = pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((tiles, 1, rows), jnp.int32),) * 2,
        grid=(tiles,),
        in_specs=(
            pl.BlockSpec(hidden.shape, lambda n: (0, 0)),
            pl.BlockSpec((None, width, tile), lambda n: (n, 0, 0)),
        ),
        out_specs=(pl.BlockSpec((None, 1, rows), lambda n: (n, 0, 0)),) * 2,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        name=f"fused_greedy_head_m{rows}_n{tile}",
    )(hidden, weight)
    values, indices = values[:, 0, :], indices[:, 0, :]
    maximum = jnp.max(values, axis=0)
    return jnp.min(
        jnp.where(values == maximum[None, :], indices, jnp.iinfo(jnp.int32).max),
        axis=0,
    )
