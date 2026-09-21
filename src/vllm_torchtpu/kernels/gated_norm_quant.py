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
"""The GDN output's gated RMS norm and its fp8 quantisation for the out-projection, in one pass.

The gated-delta-net kernel writes bf16 [tokens, heads, head_dim]; vLLM's
RMSNormGated then normalises each head in f32, multiplies by the head weight,
gates with silu(z) and rounds to bf16, and the fp8 out-projection takes a
per-token absolute maximum over the columns and converts: several passes over
the tensor. This kernel does the same arithmetic in one pass per token tile,
and it rounds the normed output to bf16 before the absolute maximum exactly
as the two-step path does, so the fp8 rows it produces are the rows the
out-projection would have quantised itself.

Semantics (vLLM RMSNormGated, norm_before_gate=True, activation swish): per head h of a token,
``y = x_h * rsqrt(mean(x_h^2) + eps) * w * silu(z_h)``; then ``scale = max|y| / 448`` per token (1 where the row
is all zero) and ``q = clip(y / scale, -448, 448)`` in float8_e4m3fn."""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

FP8 = jnp.float8_e4m3fn
FP8_MAX = 448.0


def _kernel(o_ref, z_ref, w_ref, q_ref, s_ref, *, eps: float):
    o = o_ref[...].astype(jnp.float32)  # [tile, heads, dim]
    z = z_ref[...].astype(jnp.float32)
    w = w_ref[...].astype(jnp.float32)  # [dim]
    var = jnp.mean(o * o, axis=-1, keepdims=True)
    y = o * jax.lax.rsqrt(var + eps) * w[None, None, :]
    y = y * (z * jax.nn.sigmoid(z))
    y = y.astype(jnp.bfloat16).astype(
        jnp.float32)  # the norm's bf16 output, as the two-step path rounds it
    amax = jnp.max(jnp.abs(y), axis=(1, 2), keepdims=True)  # [tile, 1, 1]
    scale = jnp.where(amax == 0.0, 1.0, amax / FP8_MAX)
    q_ref[...] = jnp.clip(y / scale, -FP8_MAX, FP8_MAX).astype(FP8)
    s_ref[...] = scale.reshape(scale.shape[0], 1)


def legal_tile(rows: int, cap: int) -> int:
    """The largest row tile up to `cap` that is a legal Pallas block: a multiple of 8 dividing the row count, or the
    whole row count in one block when no such divisor exists (36, 44, 52, ... have none; a block equal to the full
    dimension is always legal)."""
    for t in range(min(cap, rows), 7, -1):
        if rows % t == 0 and t % 8 == 0:
            return t
    return rows


def gated_norm_quant(o: jax.Array,
                     z: jax.Array,
                     w: jax.Array,
                     eps: float,
                     *,
                     tile: int = 32,
                     interpret: bool = False) -> tuple[jax.Array, jax.Array]:
    """o, z: [tokens, heads, dim] (bf16); w: [dim]. Returns (q fp8 [tokens, heads, dim], scale f32 [tokens, 1])."""
    tokens, heads, dim = o.shape
    assert z.shape == o.shape and w.shape == (dim, ), (o.shape, z.shape,
                                                       w.shape)
    # The token buckets are 4, 8, 12, 16, 32, 48, 64, 80, ...: the tile is the largest divisor of the row count at
    # or below the requested one that Pallas accepts as a block: a multiple of 8, or the whole row count (48 rows run
    # in tiles of 16, 80 in tiles of 16, 12 in one tile of 12; 20 for 80 rows is a divisor but not a legal block).
    tile = legal_tile(tokens, tile)
    grid = (tokens // tile, )
    spec = pl.BlockSpec((tile, heads, dim), lambda i: (i, 0, 0))
    return pl.pallas_call(
        functools.partial(_kernel, eps=float(eps)),
        grid=grid,
        in_specs=[spec, spec,
                  pl.BlockSpec((dim, ), lambda i: (0, ))],
        out_specs=[spec, pl.BlockSpec((tile, 1), lambda i: (i, 0))],
        out_shape=[
            jax.ShapeDtypeStruct((tokens, heads, dim), FP8),
            jax.ShapeDtypeStruct((tokens, 1), jnp.float32)
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", )),
        interpret=interpret,
    )(o, z, w)


def reference(o: jax.Array, z: jax.Array, w: jax.Array,
              eps: float) -> tuple[jax.Array, jax.Array]:
    """The two-step path in plain jnp: RMSNormGated then the out-projection's per-token quantisation."""
    x = o.astype(jnp.float32)
    zz = z.astype(jnp.float32)
    var = jnp.mean(x * x, axis=-1, keepdims=True)
    y = (x * jax.lax.rsqrt(var + eps) *
         w.astype(jnp.float32)[None, None, :]) * (zz * jax.nn.sigmoid(zz))
    y = y.astype(jnp.bfloat16)
    flat = y.reshape(y.shape[0], -1)
    amax = jnp.max(jnp.abs(flat), axis=-1, keepdims=True).astype(jnp.float32)
    scale = jnp.where(amax == 0, 1.0, amax / FP8_MAX)
    q = jnp.clip(flat.astype(jnp.float32) / scale, -FP8_MAX,
                 FP8_MAX).astype(FP8)
    return q.reshape(o.shape), scale


def prequant_matmul_in_out(x_q: jax.Array,
                           x_scale: jax.Array,
                           w_q: jax.Array,
                           w_s: jax.Array,
                           out_dtype=jnp.bfloat16) -> jax.Array:
    """The per-channel fp8 matmul with the weight stored [n_in, n_out], on a pre-quantised activation:
    the same arithmetic as util.xla_quantized_matmul after its own quantize_tensor."""
    out = jax.lax.dot_general(x_q,
                              w_q,
                              dimension_numbers=(((1, ), (0, )), ((), ())),
                              preferred_element_type=jnp.float32)
    out = out * x_scale * jnp.expand_dims(w_s, 0)
    return out.astype(out_dtype)
