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
"""Explicit experimental producers; existing reader/scoring ABIs are unchanged."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .qsa import gemma_norm
from .rounding import bf16


def norm_with_multiplier(
    x: jax.Array, multiplier: jax.Array, norm_eps: float
) -> jax.Array:
    """Apply norm with the exact FP32 value of one plus the BF16 weight."""
    assert multiplier.dtype == jnp.float32
    value = x.astype(jnp.float32)
    return bf16(
        value
        * jax.lax.rsqrt(jnp.mean(value * value, -1, keepdims=True) + norm_eps)
        * multiplier
    )


def prepare_index_query(
    index_qk: jax.Array,
    norm: jax.Array,
    angles: jax.Array,
    norm_eps: float,
    rotary_dim: int,
) -> jax.Array:
    """Preserve index RoPE rounding while producing its complete reader input."""
    count, total = index_qk.shape
    width = norm.shape[0]
    heads = total // width - 1
    if (
        total % width
        or heads <= 0
        or rotary_dim <= 0
        or rotary_dim % 2
        or rotary_dim > min(width, angles.shape[-1])
    ):
        raise ValueError(
            "Index query projection and rotary dimensions are incompatible"
        )
    half = rotary_dim // 2

    def kernel(source, weight, trig, output):
        x = source[:, : heads * width].reshape(count, heads, width)
        x = gemma_norm(x, weight[...], norm_eps)
        t = trig[...].astype(jnp.float32)
        a, b = (
            x[..., :half].astype(jnp.float32),
            x[..., half:rotary_dim].astype(jnp.float32),
        )
        c, s = t[:, None, :half], t[:, None, half:rotary_dim]
        parts = (bf16(a * c - bf16(b * s)), bf16(a * s + bf16(b * c)))
        if rotary_dim < width:
            parts += (x[..., rotary_dim:],)
        output[...] = jnp.concatenate(parts, axis=-1)

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(1,),
            in_specs=[
                pl.BlockSpec(index_qk.shape, lambda *_: (0, 0)),
                pl.BlockSpec(norm.shape, lambda *_: (0,)),
                pl.BlockSpec(angles.shape, lambda *_: (0, 0)),
            ],
            out_specs=pl.BlockSpec((count, heads, width), lambda *_: (0, 0, 0)),
        ),
        out_shape=jax.ShapeDtypeStruct((count, heads, width), index_qk.dtype),
        name="qsa_index_query_prepare",
    )(index_qk, norm, angles)
