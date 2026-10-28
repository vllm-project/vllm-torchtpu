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
"""Tile-local PLE normalized BF16 product and FP32 reduction."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Eight-row tiles bound live operands; neutral padding handles partial tiles.
_TOKEN_TILE = 8
_VECTOR_WIDTH = 128
# Accumulate four 128-lane vectors, reduce the lanes, then add the blocks
# in FP32 before converting the result to BF16.
_REDUCTION_BLOCK = 4 * _VECTOR_WIDTH


def _gate_dot_kernel(
    key, query, key_inv, query_inv, key_weight, query_weight, out, *, streams, hidden
):
    # Stream slices use a two-dimensional VMEM layout.
    result = jnp.zeros((_TOKEN_TILE, streams), jnp.bfloat16)
    column = jnp.arange(streams)[None, :]
    for stream in range(streams):
        start = stream * hidden
        key_f32 = key[:, pl.ds(start, hidden)].astype(jnp.float32)
        query_f32 = query[:, pl.ds(start, hidden)].astype(jnp.float32)
        key_scale = key_inv[:, pl.ds(stream, 1)].astype(jnp.float32)
        query_scale = query_inv[:, pl.ds(stream, 1)].astype(jnp.float32)
        kw = 1.0 + key_weight[pl.ds(start, hidden)].astype(jnp.float32)[None, :]
        qw = 1.0 + query_weight[pl.ds(start, hidden)].astype(jnp.float32)[None, :]
        key_bf16 = (key_f32 * key_scale * kw).astype(jnp.bfloat16)
        query_bf16 = (query_f32 * query_scale * qw).astype(jnp.bfloat16)
        product_bf16 = (
            key_bf16.astype(jnp.float32) * query_bf16.astype(jnp.float32)
        ).astype(jnp.bfloat16)
        product_f32 = product_bf16.astype(jnp.float32)

        dot_f32 = jnp.zeros((_TOKEN_TILE, 1), jnp.float32)
        for block in range(hidden // _REDUCTION_BLOCK):
            start = block * _REDUCTION_BLOCK
            values = product_f32[:, start : start + _REDUCTION_BLOCK]
            lanes = values[:, :_VECTOR_WIDTH]
            for vector in range(1, _REDUCTION_BLOCK // _VECTOR_WIDTH):
                start = vector * _VECTOR_WIDTH
                lanes = lanes + values[:, start : start + _VECTOR_WIDTH]
            partial = jnp.sum(lanes, axis=-1, keepdims=True)
            dot_f32 = dot_f32 + partial
        dot = dot_f32.astype(jnp.bfloat16)
        result = jnp.where(column == stream, dot, result)
    out[...] = result


def normalized_gate_dot(
    key: jax.Array,
    query: jax.Array,
    key_inv: jax.Array,
    query_inv: jax.Array,
    key_weight: jax.Array,
    query_weight: jax.Array,
    *,
    streams: int,
    interpret: bool = False,
) -> jax.Array:
    """Return BF16 [T, S] normalized gate dots.

    Key and query are BF16 [T, S*H], inverse RMS statistics are FP32 [T, S],
    and weights are BF16 [S*H].
    """
    tokens, channels = key.shape
    if streams <= 0 or channels % streams:
        raise ValueError(f"PLE gate channels={channels} must divide streams={streams}")
    hidden = channels // streams
    if (
        hidden <= 0
        or query.shape != key.shape
        or key_inv.shape != (tokens, streams)
        or query_inv.shape != (tokens, streams)
    ):
        raise ValueError(
            "PLE gate requires matching raw operands and per-stream RMS stats"
        )
    if key.dtype != jnp.bfloat16 or query.dtype != jnp.bfloat16:
        raise ValueError("PLE gate raw operands must be BF16")
    if key_weight.shape != (channels,) or query_weight.shape != (channels,):
        raise ValueError("PLE gate norm weights must cover all stream channels")
    if tokens == 0:
        return jnp.empty((0, streams), jnp.bfloat16)
    padded_tokens = pl.cdiv(tokens, _TOKEN_TILE) * _TOKEN_TILE
    padded_hidden = pl.cdiv(hidden, _REDUCTION_BLOCK) * _REDUCTION_BLOCK

    def raw(x):
        return jnp.pad(
            x.reshape(tokens, streams, hidden),
            ((0, padded_tokens - tokens), (0, 0), (0, padded_hidden - hidden)),
        ).reshape(padded_tokens, streams * padded_hidden)

    def weight(w):
        return jnp.pad(
            w.reshape(streams, hidden), ((0, 0), (0, padded_hidden - hidden))
        ).reshape(streams * padded_hidden)

    key_inv = jnp.pad(key_inv, ((0, padded_tokens - tokens), (0, 0)))
    query_inv = jnp.pad(query_inv, ((0, padded_tokens - tokens), (0, 0)))
    wide = pl.BlockSpec((_TOKEN_TILE, streams * padded_hidden), lambda row: (row, 0))
    narrow = pl.BlockSpec((_TOKEN_TILE, streams), lambda row: (row, 0))
    weights = pl.BlockSpec((streams * padded_hidden,), lambda row: (0,))
    dot = pl.pallas_call(
        functools.partial(_gate_dot_kernel, streams=streams, hidden=padded_hidden),
        out_shape=jax.ShapeDtypeStruct((padded_tokens, streams), jnp.bfloat16),
        grid=(padded_tokens // _TOKEN_TILE,),
        in_specs=(wide, wide, narrow, narrow, weights, weights),
        out_specs=narrow,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="ple_normalized_gate_dot",
    )(
        raw(key),
        raw(query),
        key_inv,
        query_inv,
        weight(key_weight),
        weight(query_weight),
    )
    return dot[:tokens]
