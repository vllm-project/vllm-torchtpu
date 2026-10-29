# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Fused FP32 Block Attention Residual for token-local K3 activations."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _mix_kernel(prefix, history, weight, output, *, eps):
    values = jnp.concatenate((history[...], prefix[...][:, None, :]), axis=1).astype(
        jnp.float32
    )
    inv_norm = jax.lax.rsqrt(jnp.mean(values * values, axis=-1, keepdims=True) + eps)
    scores = (
        jnp.sum(values * weight[...].astype(jnp.float32), axis=-1, keepdims=True)
        * inv_norm
    )
    probabilities = jax.nn.softmax(scores, axis=1)
    output[...] = jnp.sum(values * probabilities, axis=1).astype(output.dtype)


def attention_residual(prefix, history, folded_weight, *, eps=1e-5, interpret=False):
    """Mix [tokens,hidden] prefix and [tokens,blocks,hidden] saved blocks.

    Norm and projection weights are folded to one FP32 vector at load time.
    All stored blocks are live; the production model carries their actual
    count in the history shape. No state or cache is modified.
    """
    tokens, hidden = prefix.shape
    if history.shape[0] != tokens or history.shape[-1] != hidden:
        raise ValueError("Attention Residual prefix/history shapes disagree")
    if folded_weight.shape != (hidden,):
        raise ValueError(
            "Attention Residual needs one folded weight per hidden feature"
        )
    if history.shape[1] == 0 or tokens == 0:
        return prefix
    if tokens % 16:
        raise ValueError("Fused Attention Residual requires 16-token tiles")
    return pl.pallas_call(
        functools.partial(_mix_kernel, eps=eps),
        out_shape=jax.ShapeDtypeStruct(prefix.shape, prefix.dtype),
        grid=(tokens // 16,),
        in_specs=(
            pl.BlockSpec((16, hidden), lambda i: (i, 0)),
            pl.BlockSpec((16, history.shape[1], hidden), lambda i: (i, 0, 0)),
            pl.BlockSpec((hidden,), lambda i: (0,)),
        ),
        out_specs=pl.BlockSpec((16, hidden), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=63 * 1024**2),
        interpret=interpret,
        name="kimi_attnres_compact_m16",
    )(prefix, history, folded_weight)


def _streamed_mix_kernel(prefix, history, weight, output, *, eps, blocks, chunk):
    """Two passes over 512-lane chunks: per-slot statistics, then the mix.

    `history` is `[tile, blocks * hidden]`, so slot k of a chunk is a plain
    column window. Nothing wider than one `[tile, chunk]` f32 slab is live.
    """
    tile, hidden = prefix.shape
    steps = hidden // chunk

    def score(load):
        def body(i, carry):
            sq, dot = carry
            c = pl.multiple_of(i * chunk, chunk)
            v = load(c).astype(jnp.float32)
            w = weight[pl.ds(c, chunk)].astype(jnp.float32)
            return sq + v * v, dot + v * w

        zero = jnp.zeros((tile, chunk), jnp.float32)
        sq, dot = jax.lax.fori_loop(0, steps, body, (zero, zero))
        sq = jnp.sum(sq, axis=-1, keepdims=True)
        dot = jnp.sum(dot, axis=-1, keepdims=True)
        return dot * jax.lax.rsqrt(sq * (1.0 / hidden) + eps)

    scores = [
        score(lambda c, k=k: history[:, pl.ds(k * hidden + c, chunk)])
        for k in range(blocks)
    ]
    scores.append(score(lambda c: prefix[:, pl.ds(c, chunk)]))
    probabilities = jax.nn.softmax(jnp.concatenate(scores, axis=1), axis=1)

    def mix(i, carry):
        c = pl.multiple_of(i * chunk, chunk)
        acc = probabilities[:, blocks : blocks + 1] * prefix[:, pl.ds(c, chunk)].astype(
            jnp.float32
        )
        for k in range(blocks):
            slot = history[:, pl.ds(k * hidden + c, chunk)].astype(jnp.float32)
            acc = acc + probabilities[:, k : k + 1] * slot
        output[:, pl.ds(c, chunk)] = acc.astype(output.dtype)
        return carry

    jax.lax.fori_loop(0, steps, mix, 0)


def attention_residual_streamed(
    prefix, history, folded_weight, *, eps=1e-5, tile=16, chunk=512, interpret=False
):
    """`attention_residual` with chunked two-pass statistics (same contract)."""
    tokens, hidden = prefix.shape
    if history.shape[0] != tokens or history.shape[-1] != hidden:
        raise ValueError("Attention Residual prefix/history shapes disagree")
    if folded_weight.shape != (hidden,):
        raise ValueError(
            "Attention Residual needs one folded weight per hidden feature"
        )
    blocks = history.shape[1]
    if blocks == 0 or tokens == 0:
        return prefix
    if tokens % tile or hidden % chunk:
        raise ValueError("Streamed Attention Residual needs whole token/lane tiles")
    flat = history.reshape(tokens, blocks * hidden)
    return pl.pallas_call(
        functools.partial(_streamed_mix_kernel, eps=eps, blocks=blocks, chunk=chunk),
        out_shape=jax.ShapeDtypeStruct(prefix.shape, prefix.dtype),
        grid=(tokens // tile,),
        in_specs=(
            pl.BlockSpec((tile, hidden), lambda i: (i, 0)),
            pl.BlockSpec((tile, blocks * hidden), lambda i: (i, 0)),
            pl.BlockSpec((hidden,), lambda i: (0,)),
        ),
        out_specs=pl.BlockSpec((tile, hidden), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=63 * 1024**2),
        interpret=interpret,
        name=f"kimi_attnres_streamed_m{tile}",
    )(prefix, flat, folded_weight)
