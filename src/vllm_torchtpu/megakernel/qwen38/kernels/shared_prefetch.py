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
"""Ordinary resident BF16 shared-expert weights; no view or custom lowering."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def resident_shared_projection(source, matrix, destination, *, rounded):
    """Keep selected full-K/N128 GU or full-N8192 DOWN arithmetic exactly."""
    contraction, columns = matrix.shape
    expected = (8192, 128) if rounded else (64, 8192)
    if (contraction, columns) != expected or matrix.dtype != jnp.bfloat16:
        raise ValueError("Resident shared projection requires the exact production BF16 shape")
    activation = source[:, :contraction].astype(jnp.bfloat16)
    mxu_width = 128 if rounded else columns
    with jax.named_scope("shared_resident_projection"):
        for offset in range(0, columns, mxu_width):
            value = jnp.dot(
                activation,
                matrix[:, pl.ds(offset, mxu_width)].astype(jnp.bfloat16),
                preferred_element_type=jnp.float32,
            )
            value = (
                value.astype(jnp.bfloat16).astype(destination.dtype)
                if rounded
                else value.astype(destination.dtype)
            )
            destination[:, pl.ds(offset, mxu_width)] = value
