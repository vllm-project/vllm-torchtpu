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
"""Exact BF16 query norm/RoPE preparation before the existing reader layout."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp.qsa import gemma_norm
from vllm_torchtpu.kernels.qwen4_exp.rounding import bf16

from .rope import select_rope_rows


def prepare_query_reader_layout(
    qgate: jax.Array,
    norm: jax.Array,
    positions: jax.Array,
    trig: jax.Array,
    norm_eps: float,
    rotary_dim: int,
    rope_axes: tuple[int, ...] | None = None,
) -> jax.Array:
    count, heads, doubled = qgate.shape
    width = doubled // 2
    selected = select_rope_rows(positions, trig, rotary_dim, rope_axes)
    shape = (2, count, heads // 2, 1, width)
    kv_heads = 2
    assert (count, heads, width) == (16, 24, 256)
    group = heads // kv_heads
    half = rotary_dim // 2

    def kernel(source, weight, angles, output):
        x = gemma_norm(source[..., :width], weight[...], norm_eps)
        t = angles[...].astype(jnp.float32)
        c, s = t[:, None, :half], t[:, None, half:rotary_dim]
        a, b = (
            x[..., :half].astype(jnp.float32),
            x[..., half:rotary_dim].astype(jnp.float32),
        )
        value = jnp.concatenate(
            (bf16(a * c - b * s), bf16(b * c + a * s), x[..., rotary_dim:]), axis=-1
        )
        output[...] = (
            jnp.concatenate(
                [value[:, h * group : (h + 1) * group] for h in range(kv_heads)],
                axis=0,
            )
            .reshape(shape)
            .astype(jnp.float32)
        )

    packed = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(1,),
            in_specs=[
                pl.BlockSpec(qgate.shape, lambda *_: (0, 0, 0)),
                pl.BlockSpec(norm.shape, lambda *_: (0,)),
                pl.BlockSpec(selected.shape, lambda *_: (0, 0)),
            ],
            out_specs=pl.BlockSpec(shape, lambda *_: (0, 0, 0, 0, 0)),
        ),
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        name="qsa_query_reader_direct",
    )(qgate, norm, selected)

    # Expose the standard logical query shape; the unchanged reader adapter
    # reverses these views, retaining the kernel-produced physical layout.
    return (
        packed.reshape(kv_heads, count, group, width)
        .swapaxes(0, 1)
        .reshape(count, heads, width)
    )
