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
"""Test-only pure-JAX grouped matmul reference matching gmm_wrapper semantics."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def unpack_fp4_u8(rhs: jax.Array) -> jax.Array:
    """Inverse of ``quantize_to_native_fp4_kmajor(..., pack=True)``.

    ``quantize_to_native_fp4_kmajor`` reshapes ``[..., K/2, 2, N]``, swaps the
    last two axes to ``[..., K/2, N, 2]``, and bitcasts the trailing pair axis
    to ``uint8`` ``[..., K/2, N]``. Unpacking therefore bitcasts back to
    ``float4_e2m1fn`` ``[..., K/2, N, 2]``, swaps ``(-1, -2)`` to restore
    ``[..., K/2, 2, N]``, and collapses ``K``.
    """
    pairs = jax.lax.bitcast_convert_type(rhs, jnp.float4_e2m1fn)
    return jnp.swapaxes(pairs, -1, -2).reshape(
        *rhs.shape[:-2], rhs.shape[-2] * 2, rhs.shape[-1]
    )


def ref_gmm(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array | None = None,
    zero_initialize: bool = False,
    fuse_act: str | None = None,
    preferred_element_type: Any = None,
    rhs_quant_dtype: Any = None,
) -> jax.Array:
    """Pure-JAX grouped matmul reference for CPU unit tests."""
    del group_offset, zero_initialize, rhs_quant_dtype
    rhs_val = unpack_fp4_u8(rhs) if rhs.dtype == jnp.uint8 else rhs
    rhs_f32 = rhs_val.astype(jnp.float32)
    if rhs_scale is not None:
        s = rhs_scale.astype(jnp.float32).reshape(
            rhs_f32.shape[0], -1, 1, rhs_f32.shape[-1]
        )
        block_size = rhs_f32.shape[1] // s.shape[1]
        rhs_f32 = (
            rhs_f32.reshape(rhs_f32.shape[0], s.shape[1], block_size, rhs_f32.shape[2])
            * s
        ).reshape(rhs_f32.shape)

    ends = jnp.cumsum(group_sizes, dtype=jnp.int32)
    starts = ends - group_sizes
    row_idx = jnp.arange(lhs.shape[0], dtype=jnp.int32)[:, None]
    in_group = (row_idx >= starts[None, :]) & (row_idx < ends[None, :])
    group_id = jnp.argmax(in_group, axis=1)
    valid_row = jnp.any(in_group, axis=1)

    w_row = rhs_f32[group_id]
    acc = jnp.einsum(
        "mk,mkn->mn",
        lhs.astype(jnp.float32),
        w_row,
        precision=jax.lax.Precision.HIGHEST,
    )
    if rhs_bias is not None:
        b_row = rhs_bias.astype(jnp.float32).reshape(rhs_f32.shape[0], -1)[group_id]
        acc = acc + b_row
    acc = jnp.where(valid_row[:, None], acc, 0.0)

    out_dtype = (
        preferred_element_type if preferred_element_type is not None else lhs.dtype
    )
    if fuse_act is not None:
        gate, up = jnp.split(acc, 2, axis=-1)
        if fuse_act == "silu":
            acc = jax.nn.silu(gate) * up
        elif fuse_act == "gelu":
            acc = jax.nn.gelu(gate) * up
        elif fuse_act == "gelu_tanh":
            acc = jax.nn.gelu(gate, approximate=True) * up
        else:
            raise NotImplementedError(f"Unsupported activation: {fuse_act}")
    return acc.astype(out_dtype)
