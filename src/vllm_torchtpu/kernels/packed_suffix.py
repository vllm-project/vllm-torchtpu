# SPDX-License-Identifier: Apache-2.0
"""Isolate a packed suffix from layout changes in its consumers."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def extract_packed_suffix(packed: jax.Array, num_words: int) -> jax.Array:
    assert packed.dtype == jnp.int32
    assert 0 < num_words <= packed.size
    lanes = pltpu.get_tpu_info().num_lanes
    padded_words = pl.cdiv(num_words, lanes) * lanes
    suffix = packed.reshape(-1)[-num_words:]
    suffix = jnp.pad(suffix, (0, padded_words - num_words)).reshape(1, -1)

    def copy_suffix(inp, out):
        out[...] = inp[...]

    # Keep downstream unpack/reshape fusion from relaying out the whole input.
    out = pl.pallas_call(
        copy_suffix,
        out_shape=jax.ShapeDtypeStruct((1, padded_words), packed.dtype),
        compiler_params=pltpu.CompilerParams(allow_input_fusion=(False,)),
    )(suffix)
    return out[:, :num_words]
