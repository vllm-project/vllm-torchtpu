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
"""B/A projections with independent calibration packed into one MXU contraction."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu


def fused_ba_projection(
    source,
    matrix,
    destination,
    *,
    batch_size,
    channels,
    b_input_scale,
    b_weight_scale,
    a_input_scale,
    a_weight_scale,
    resident=False,
):
    """Write the valid B/A columns; preserve all other destination elements.

    Each request retains the selected full-K/N128 contraction, independently
    calibrated FP8 activation and BF16 output cast. Only the physical M row
    changes for A; eight rows pack sixteen operands into one M16 contraction. A
    resident matrix is already in VMEM and is read in place.
    """
    contraction, columns = matrix.shape
    rows = source.shape[0]
    if batch_size not in (1, 2, 4, 8, 16, 32, 64) or columns != 128 or not 0 < channels <= 64:
        raise ValueError("Fused B/A requires 1/2/4/8/16/32/64 rows and one shared N128 tile")
    if (rows not in (8, 16, 32, 64) or batch_size > rows or source.shape != (rows, contraction)
            or destination.shape != (rows, 128)):
        raise ValueError("Fused B/A operands must retain the selected M8, M16, M32 or M64 row tile")

    # Keep each selected quantization expression intact. Quantizing after
    # packing with a vector of scales would introduce an unnecessary change
    # in lowering around division and the FP8 cast.
    b = jnp.clip(source[...].astype(jnp.float32) / b_input_scale, -448, 448)
    b = b.astype(jnp.float8_e4m3fn).astype(jnp.bfloat16)
    a = jnp.clip(source[...].astype(jnp.float32) / a_input_scale, -448, 448)
    a = a.astype(jnp.float8_e4m3fn).astype(jnp.bfloat16)
    pieces = (b[:batch_size], a[:batch_size])
    if batch_size < 4:
        pieces += (jnp.zeros((rows - 2 * batch_size, contraction), b.dtype),)
    packed = jnp.concatenate(pieces, axis=0)

    def project(weight):
        values = jnp.dot(packed, weight[...], preferred_element_type=jnp.float32)
        # Postscales are still separate scalar products, applied after the
        # MXU and before the original BF16 result rounding.
        b_value = (values[:batch_size] * (b_input_scale * b_weight_scale)).astype(jnp.bfloat16)
        a_value = (values[batch_size : 2 * batch_size] * (a_input_scale * a_weight_scale)).astype(
            jnp.bfloat16
        )
        b_value = jnp.pad(b_value, ((0, rows - batch_size), (0, 0))).astype(destination.dtype)
        a_value = jnp.pad(a_value, ((0, rows - batch_size), (0, 0))).astype(destination.dtype)
        valid = jnp.arange(rows)[:, None] < batch_size
        column = jnp.arange(128)[None, :]
        destination[...] = jnp.where(
            valid & (column < channels),
            b_value,
            jnp.where(
                valid & (column >= channels) & (column < 2 * channels), a_value, destination[...]
            ),
        )

    def compute(weight, semaphore):
        transfer = tpu.make_async_copy(matrix, weight, semaphore)
        transfer.start()
        transfer.wait()
        project(weight)

    with jax.named_scope("gdn_fused_ba_projection"):
        if resident:
            project(matrix)
        else:
            pl.run_scoped(
                compute,
                tpu.VMEM(matrix.shape, matrix.dtype),
                tpu.SemaphoreType.DMA,
            )
