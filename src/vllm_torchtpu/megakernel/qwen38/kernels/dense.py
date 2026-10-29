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
"""Full-K dense projections with bounded, double-buffered output tiles."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

DENSE_TILE_CHOICES = (128, 256, 512, 1024, 2048, 4096, 8192)
DENSE_BUFFER_BYTES = 16 * 1024**2


def output_tile_width(contraction, columns, itemsize, requested):
    """Keep both HBM staging tiles within16MiB, including long-K projections."""
    if requested not in DENSE_TILE_CHOICES:
        raise ValueError(f"Unsupported dense output tile {requested}")
    if contraction <= 0 or columns <= 0 or columns % 128:
        raise ValueError(f"Unaligned dense projection {(contraction, columns)}")
    capacity = DENSE_BUFFER_BYTES // (2 * contraction * itemsize) // 128 * 128
    if capacity < 128:
        raise ValueError("A minimum dense output tile exceeds the staging budget")
    return min(requested, columns, capacity)


def dense_matmul(
    source,
    matrix,
    destination,
    *,
    output_tile=128,
    rounded=False,
    input_scale=None,
    weight_scale=None,
    mask=None,
):
    """Project a VMEM value through HBM weights, preserving full-K MXU order.

    A final incomplete output tile is a separate aligned scope. The optional
    mask preserves the separately calibrated GDN B/A columns sharing one tile.
    FP8 input and output casts match the original128-column implementation.
    FP8 weights enter the MXU as stored, beside the FP8-valued BF16 activation.
    Rounded outputs also keep128-column MXUs to preserve BF16 midpoint choices.
    """
    contraction, columns = matrix.shape
    fp8 = input_scale is not None
    if fp8 != (weight_scale is not None):
        raise ValueError("Input and weight scales must be supplied together")
    if mask is not None and (columns != 128 or mask.shape != (128,)):
        raise ValueError("Partial calibrated projections must occupy one128-column tile")
    width = output_tile_width(contraction, columns, matrix.dtype.itemsize, output_tile)
    activation = source[:, :contraction].astype(jnp.bfloat16)
    if fp8:
        activation = jnp.clip(source[:, :contraction].astype(jnp.float32) / input_scale, -448, 448)
        activation = activation.astype(jnp.float8_e4m3fn).astype(jnp.bfloat16)

    def project(start, extent, tile_width):
        def compute(tile, tile_sem):
            def transfer(n, slot):
                return tpu.make_async_copy(
                    matrix.at[:, pl.ds(start + n * tile_width, tile_width)],
                    tile.at[slot],
                    tile_sem.at[slot],
                )

            transfer(0, 0).start()

            @pl.loop(0, extent // tile_width)
            def nloop(n):
                slot = n % 2
                transfer(n, slot).wait()

                @pl.when(n + 1 < extent // tile_width)
                def prefetch():
                    transfer(n + 1, 1 - slot).start()

                # Rounded projections retain the original128-column MXU
                # contraction. Wider N changes TPU FP32 association enough to
                # cross a BF16 midpoint (observed for sharedGU8192x256).
                # DMA still prefetches the wider contiguous weight tile.
                mxu_width = 128 if rounded else tile_width
                for offset in range(0, tile_width, mxu_width):
                    weights = tile[slot, :, pl.ds(offset, mxu_width)]
                    if not fp8:
                        weights = weights.astype(jnp.bfloat16)
                    value = jnp.dot(activation, weights, preferred_element_type=jnp.float32)
                    if fp8:
                        value *= input_scale * weight_scale
                    value = (
                        value.astype(jnp.bfloat16).astype(destination.dtype)
                        if rounded
                        else value.astype(destination.dtype)
                    )
                    output_slice = pl.ds(start + n * tile_width + offset, mxu_width)
                    if mask is not None:
                        value = jnp.where(mask[None, :], value, destination[:, output_slice])
                    destination[:, output_slice] = value

        pl.run_scoped(
            compute,
            tile=tpu.VMEM((2, contraction, tile_width), matrix.dtype),
            tile_sem=tpu.SemaphoreType.DMA((2,)),
        )

    with jax.named_scope("dense_fp8_projection" if fp8 else "dense_bf16_projection"):
        prefix = columns // width * width
        project(0, prefix, width)
        if prefix < columns:
            project(prefix, columns - prefix, columns - prefix)
