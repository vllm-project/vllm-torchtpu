# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 linear head with a fused first-index argmax."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.utils import get_device_name

# Qualified on v7 with full-K reuse and compact partials; other shapes fall back.
TUNED_TILES = {(288, 5120, 129280): 1280}


def is_supported_shape(rows: int, weight_shape: tuple[int, ...]) -> bool:
    return (
        len(weight_shape) == 2
        and (rows, weight_shape[1], weight_shape[0]) in TUNED_TILES
        and get_device_name() == "TPU v7"
    )


def fused_greedy_head(hidden: jax.Array, weight: jax.Array) -> jax.Array:
    """Return first argmax IDs after BF16 rounding, for the original [vocab,K] W."""
    rows, width = hidden.shape
    vocab, weight_width = weight.shape
    assert hidden.dtype == weight.dtype == jnp.bfloat16
    assert width == weight_width
    tile = TUNED_TILES[(rows, width, vocab)]
    tiles = vocab // tile

    def kernel(hidden_ref, weight_ref, values_ref, indices_ref):
        accum = jnp.dot(
            hidden_ref[...], weight_ref[...].T, preferred_element_type=jnp.float32
        )
        rounded = accum.astype(jnp.bfloat16).astype(jnp.float32)
        nan = jnp.isnan(rounded)
        has_nan = jnp.any(nan, axis=1)
        maximum = jnp.max(rounded, axis=1)
        chosen = nan | ((~has_nan[:, None]) & (rounded == maximum[:, None]))
        offsets = pl.program_id(0) * tile + jnp.arange(tile, dtype=jnp.int32)
        values_ref[0, :] = jnp.where(has_nan, jnp.nan, maximum)
        indices_ref[0, :] = jnp.min(
            jnp.where(chosen, offsets[None, :], jnp.iinfo(jnp.int32).max), axis=1
        )

    values, indices = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((tiles, 1, rows), jnp.float32),
            jax.ShapeDtypeStruct((tiles, 1, rows), jnp.int32),
        ),
        grid=(tiles,),
        in_specs=(
            pl.BlockSpec(hidden.shape, lambda n: (0, 0)),
            pl.BlockSpec((tile, width), lambda n: (n, 0)),
        ),
        out_specs=(pl.BlockSpec((None, 1, rows), lambda n: (n, 0, 0)),) * 2,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        name=f"fused_greedy_head_m{rows}_n{tile}",
    )(hidden, weight)
    values, indices = values[:, 0, :], indices[:, 0, :]
    nan = jnp.isnan(values)
    has_nan = jnp.any(nan, axis=0)
    maximum = jnp.max(values, axis=0)
    chosen = jnp.where(has_nan[None, :], nan, values == maximum[None, :])
    return jnp.min(
        jnp.where(chosen, indices, jnp.iinfo(jnp.int32).max),
        axis=0,
    )
