# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feature-major cache records with 128-lane, 32-bit storage words."""

import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def packed_record_shape(elements: int, itemsize: int) -> tuple[int, int]:
    """Return ``(packing, lanes)`` for a naturally ordered, tail-padded record.

    Follows sglang-jax's mem_cache/memory_pool.py `_cache_shape` feature
    packing, extended to FP32 states and the uint8 NoPE/scale payload.
    The word owner preserves the TPU tile layout across mixed cache types.
    """
    if elements <= 0 or itemsize not in (1, 2, 4):
        raise ValueError("Cache records require positive dimensions and 8/16/32 bits")
    packing = 4 // itemsize
    alignment = packing * 128
    return packing, ((elements + alignment - 1) // alignment) * 128


def pack_cache_values(values: jax.Array) -> jax.Array:
    """``[..., D]`` -> int32 ``[..., padded_D / packing]``, without a permutation."""
    packing, width = packed_record_shape(values.shape[-1], values.dtype.itemsize)
    if values.dtype not in (jnp.bfloat16, jnp.float32, jnp.uint8):
        raise ValueError("Cache values must be BF16, FP32 or uint8 payloads")
    if jax.default_backend() == "tpu" and packing > 1:
        return _native_bitcast(values, width, jnp.int32, packing=packing)
    unsigned = {1: jnp.uint32, 2: jnp.uint16, 4: jnp.uint8}[packing]
    bits = jax.lax.bitcast_convert_type(values, unsigned)
    padding = packing * width - values.shape[-1]
    if padding:
        # Floating-point padding can canonicalize BF16 NaN payloads.
        bits = jnp.pad(bits, [(0, 0)] * (bits.ndim - 1) + [(0, padding)])
    if packing == 1:
        return jax.lax.bitcast_convert_type(bits, jnp.int32)
    bits = bits.astype(jnp.uint32)
    bits = bits.reshape(*bits.shape[:-1], packing, width)
    words = bits[..., 0, :]
    for part in range(1, packing):
        words = words | (bits[..., part, :] << (32 // packing * part))
    return jax.lax.bitcast_convert_type(words, jnp.int32)


def unpack_cache_values(words: jax.Array, elements: int, dtype: jnp.dtype) -> jax.Array:
    """Inverse of `pack_cache_values`, discarding only the padded tail."""
    dtype = jnp.dtype(dtype)
    packing, width = packed_record_shape(elements, dtype.itemsize)
    if words.shape[-1] != width:
        raise ValueError("Word width does not match the padded cache record")
    if packing == 1:
        return jax.lax.bitcast_convert_type(words, dtype)[..., :elements]
    if jax.default_backend() == "tpu":
        return _native_bitcast(words, elements, dtype, packing=packing)
    unsigned = jax.lax.bitcast_convert_type(words, jnp.uint32)
    bits = jnp.stack(
        [unsigned >> (32 // packing * part) for part in range(packing)], axis=-2
    )
    bits = bits.astype(jnp.uint16 if packing == 2 else jnp.uint8)
    values = jax.lax.bitcast_convert_type(bits, dtype)
    return values.reshape(*words.shape[:-1], packing * width)[..., :elements]


def _native_bitcast(values, output_dim, dtype, *, packing):
    """Pack the sublane dimension directly, preserving BF16 payload bits."""
    input_dim = values.shape[-1]
    rows = math.prod(values.shape[:-1])
    unpack = values.dtype == jnp.int32
    width = input_dim if unpack else output_dim
    # Byte/BF16 batches need complete native sublane tiles, including when
    # a batch spans multiple tiles. The shared word owner is never relaid out.
    block_rows = 8 * packing
    unsigned = jnp.uint16 if packing == 2 else jnp.uint8

    def kernel(x_ref, y_ref):
        if unpack:
            x = x_ref[...]
            y = pltpu.bitcast(x.reshape(block_rows, 1, width), unsigned)
            y = y.reshape(block_rows, packing * width)[:, :output_dim]
            y_ref.bitcast(unsigned)[...] = y
        else:
            bits = x_ref.bitcast(unsigned)[...]
            bits = jnp.pad(bits, ((0, 0), (0, packing * width - input_dim)))
            y = pltpu.bitcast(bits.reshape(block_rows, packing, width), dtype)
            y_ref[...] = y.reshape(block_rows, width)

    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, output_dim), dtype),
        grid=(pl.cdiv(rows, block_rows),),
        in_specs=[pl.BlockSpec((block_rows, input_dim), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((block_rows, output_dim), lambda i: (i, 0)),
    )(values.reshape(rows, input_dim))
    return result.reshape(*values.shape[:-1], output_dim)
