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
"""Public-API SPMD axis-coordinate helpers without HLO partition-id emission."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def kernel_axis_index(axis_name: str) -> jax.Array:
    """Return the int32 coordinate along mesh axis `axis_name` inside a kernel."""
    return lax.axis_index(axis_name).astype(jnp.int32)


def mesh_axis_index(axis_name: str, *, interpret: bool = False) -> jax.Array:
    """Return the int32 coordinate along `axis_name` via a scalar-scratchpad call."""

    def _write_axis_index(out_ref: jax.Array) -> None:
        """Write the int32 axis coordinate into scalar SMEM `out_ref`."""
        out_ref[0] = lax.axis_index(axis_name).astype(jnp.int32)

    out = pl.pallas_call(
        _write_axis_index,
        out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
        out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
        interpret=interpret,
    )()
    return out[0]


__all__ = ["kernel_axis_index", "mesh_axis_index"]
