# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sorted D128 FP8 preparation for verified bounded decode token counts."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def supports_sorted_fp8(
    banks: tuple[jax.Array, ...],
    records: jax.Array,
    keys: jax.Array,
    page_size: int,
) -> bool:
    return (
        jax.default_backend() == "tpu"
        and keys.ndim == 2
        and keys.shape[0] in (128, 256, 288, 384, 512)
        and keys.shape[1] == 128
        and keys.dtype == jnp.bfloat16
        and records.shape == (keys.shape[0],)
        and records.dtype == jnp.int32
        and page_size == 256
        and len(banks) == 2
        and all(
            bank.ndim == 2 and bank.shape[1] == 128 and bank.dtype == jnp.int32
            for bank in banks
        )
    )


def prepare_sorted_fp8(
    keys: jax.Array, columns: jax.Array, order: jax.Array, scales: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Quantize and reorder, yielding the original writer's exact layout."""
    count = keys.shape[0]
    padded = pl.cdiv(count, 128) * 128

    def kernel(key_ref, column_ref, order_ref, scale_ref, column_out, value_out):
        order = order_ref[:]
        # Mosaic dynamic_gather requires 32-bit data for int32 indices.
        # Encode first: reorder 33 packed fields instead of 128 widened keys.
        values = key_ref[:, :].astype(jnp.float32)
        # Preserve the original XLA ceil(log2) producer's rounding at powers
        # of two. Recomputing it inside Mosaic can change the encoded scale.
        scale = scale_ref[:]
        quantized = (values / scale[:, None]).astype(jnp.float8_e4m3fn)
        # Native bitcast packs the second-minor dimension. Feature-major
        # FP8 therefore becomes the writer's 32 packed int32 feature rows.
        words = pltpu.bitcast(quantized.T, jnp.int32)
        packed = jnp.concatenate(
            (
                words,
                jax.lax.bitcast_convert_type(scale, jnp.int32)[None, :],
            ),
            axis=0,
        )
        active = jnp.arange(padded) < count
        # Include columns in the same gather; both use the identical permutation.
        sorted_values = _tiled_gather(
            jnp.concatenate((packed, column_ref[:][None, :]), axis=0), order
        )
        value_out[:, :] = jnp.where(active[None, :], sorted_values[:33], 0)
        column_out[:] = jnp.where(active, sorted_values[33], 0)

    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((padded,), jnp.int32),
            jax.ShapeDtypeStruct((33, padded), jnp.int32),
        ),
        in_specs=(
            pl.BlockSpec((padded, 128), lambda: (0, 0)),
            pl.BlockSpec((padded,), lambda: (0,)),
            pl.BlockSpec((padded,), lambda: (0,)),
            pl.BlockSpec((padded,), lambda: (0,)),
        ),
        out_specs=(pl.BlockSpec(), pl.BlockSpec()),
        name="indexer_sorted_fp8_prepare",
    )(
        jnp.pad(keys, ((0, padded - count), (0, 0))),
        jnp.pad(columns, (0, padded - count)),
        jnp.pad(order, (0, padded - count)),
        jnp.pad(scales, (0, padded - count), constant_values=1),
    )


def _tiled_gather(values, order):
    """Mosaic permits one 128-lane source register per dynamic gather."""
    outputs = []
    for output in range(order.shape[0] // 128):
        indices = order[output * 128 : (output + 1) * 128]
        local = jnp.broadcast_to(indices[None, :] % 128, (values.shape[0], 128))
        result = jnp.zeros(local.shape, values.dtype)
        for source in range(values.shape[1] // 128):
            part = jnp.take_along_axis(
                values[:, source * 128 : (source + 1) * 128], local, axis=1
            )
            result = jnp.where(indices[None, :] // 128 == source, part, result)
        outputs.append(result)
    return jnp.concatenate(outputs, axis=1)
