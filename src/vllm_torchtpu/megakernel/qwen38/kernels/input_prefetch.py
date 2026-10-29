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
"""Resident FP8 first-projection consumer; source bytes arrive via private DMA."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def resident_fp8_projection(source, matrix, destination, input_scale, weight_scale, *, rounded=True):
    """Match dense_matmul's FP8 path without another HBM copy.

    Rounded projections keep its full-K/N128 contraction. Unrounded ones match
    a single full-width output tile, as with dense_output_tile=8192. FP8
    weights enter the MXU as stored.
    """
    contraction, columns = matrix.shape
    if contraction % 128 or columns % 128:
        raise ValueError("Resident input projection requires whole128 tiles")
    activation = jnp.clip(source[:, :contraction].astype(jnp.float32) / input_scale, -448, 448)
    activation = activation.astype(jnp.float8_e4m3fn).astype(jnp.bfloat16)
    mxu_width = 128 if rounded else columns
    with jax.named_scope("next_input_resident_mxu" if rounded else "output_resident_mxu"):
        for offset in range(0, columns, mxu_width):
            value = jnp.dot(
                activation,
                matrix[:, pl.ds(offset, mxu_width)],
                preferred_element_type=jnp.float32,
            )
            value *= input_scale * weight_scale
            destination[:, pl.ds(offset, mxu_width)] = (
                value.astype(jnp.bfloat16).astype(destination.dtype)
                if rounded
                else value.astype(destination.dtype)
            )
