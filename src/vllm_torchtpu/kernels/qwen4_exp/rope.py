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
"""Aligned trigonometric row selection for scalar or axial positions."""

import jax
import jax.numpy as jnp


def select_rope_rows(positions, cache, rotary_dim, rope_axes=None):
    rows = jax.lax.optimization_barrier(cache[positions])
    if positions.ndim == 1:
        return rows
    if positions.ndim != 2 or positions.shape[0] != 3 or rope_axes is None:
        raise ValueError(
            "Axial positions require three axes and a frequency assignment"
        )
    half = rotary_dim // 2
    if len(rope_axes) != half:
        raise ValueError("MRoPE frequency assignment differs from rotary dimension")
    axes = tuple(
        rope_axes[column % half] if column < rotary_dim else 0
        for column in range(cache.shape[-1])
    )
    return rows[
        jnp.asarray(axes)[None, :],
        jnp.arange(positions.shape[1])[:, None],
        jnp.arange(cache.shape[-1])[None, :],
    ]
