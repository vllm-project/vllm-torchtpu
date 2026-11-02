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
) -> jax.Array:
    """Preserve index RoPE rounding while producing its complete reader input."""
    count, total = index_qk.shape
    width = norm.shape[0]
    heads = total // width - 1
    assert (count, heads, width, angles.shape) == (16, 4, 128, (16, 128))

    def kernel(source, weight, trig, output):
        x = source[:, : heads * width].reshape(count, heads, width)
        x = gemma_norm(x, weight[...], norm_eps)
        t = trig[...].astype(jnp.float32)
        a, b = x[..., :32].astype(jnp.float32), x[..., 32:64].astype(jnp.float32)
        c, s = t[:, None, :32], t[:, None, 32:64]
        output[...] = jnp.concatenate(
            (bf16(a * c - bf16(b * s)), bf16(a * s + bf16(b * c)), x[..., 64:]),
            axis=-1,
        )

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


def virtual_pages_unpadded(
    blocks: jax.Array,
    lengths: jax.Array,
    table: jax.Array,
    *,
    ratio: int = 4,
    page_size: int = 1664,
) -> tuple[jax.Array, jax.Array]:
    """Prepare the existing reader ABI without intermediate virtual-page owners."""
    batch, budget = blocks.shape
    assert (batch, budget, table.shape) == (16, 512, (16, 41))
    per_page = page_size // ratio

    def kernel(selected, length, physical_table, pages):
        n = length[...]
        complete = n // ratio
        count = jnp.minimum(complete, budget)
        columns = jnp.arange(budget + 1)[None, :]
        groups = jnp.concatenate(
            (selected[...], jnp.zeros((batch, 1), jnp.int32)), axis=1
        )
        groups = jnp.where(columns == count[:, None], complete[:, None], groups)
        live = (columns < count[:, None]) | (
            (columns == count[:, None]) & ((n % ratio)[:, None] > 0)
        )
        safe = jnp.where(live, groups, 0)
        physical = jnp.take_along_axis(
            jnp.pad(physical_table[...], ((0, 0), (0, 128 - table.shape[1]))),
            jnp.clip(safe // per_page, 0, table.shape[1] - 1),
            axis=1,
        )
        pages[...] = jnp.where(live, physical * per_page + safe % per_page, 0)

    pages = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(1,),
            in_specs=[
                pl.BlockSpec((batch, budget), lambda t: (0, 0)),
                pl.BlockSpec((batch,), lambda t: (0,)),
                pl.BlockSpec(table.shape, lambda t: (0, 0)),
            ],
            out_specs=pl.BlockSpec((batch, budget + 1), lambda t: (0, 0)),
        ),
        out_shape=jax.ShapeDtypeStruct((batch, budget + 1), jnp.int32),
        name="qsa_virtual_page_unpadded",
    )(blocks, lengths, table)
    # Keep schedule lengths independent of per-layer selected blocks. Returning
    # them from the opaque page kernel defeats cross-layer CSE and duplicates
    # the shared decode schedule twelve times in the production chain.
    kv_lens = jnp.minimum(lengths // ratio, budget) * ratio + lengths % ratio
    return pages, kv_lens
