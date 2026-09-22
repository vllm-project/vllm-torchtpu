# SPDX-License-Identifier: Apache-2.0
"""Exact PLE hash, compact FP8 resident lookup, and packed causal convolution."""

import functools
import math

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas


def compute_ngram_ids(
    input_ids,
    query_start_loc,
    ngram_context,
    multipliers,
    vocab_sizes,
    offsets,
    *,
    eos_token_id: int,
):
    """Signed int64 multiply/xor/remainder, including wraparound on overflow.

    Context is oldest->newest. A preceding EOS terminates the predecessor
    chain; a current EOS is hashed normally. Starts delimit real tokens;
    trailing token padding returns zero IDs and never indexes a request OOB.
    """
    tokens = input_ids.reshape(-1).to(torch.int64)
    requests = ngram_context.shape[0]
    if tokens.shape[0] == 0 or requests == 0:
        return tokens.new_zeros((tokens.shape[0], 16))
    t = torch.arange(tokens.shape[0], device=tokens.device, dtype=torch.int64)
    req = torch.sum(t[:, None] >= query_start_loc[None, 1:], dim=1)
    req = req.clamp(max=requests - 1)
    rel = t - query_start_loc[req]
    prev1 = torch.where(rel >= 1, tokens[(t - 1).clamp(min=0)], ngram_context[req, 1])
    prev2 = torch.where(
        rel >= 2,
        tokens[(t - 2).clamp(min=0)],
        torch.where(rel == 1, ngram_context[req, 1], ngram_context[req, 0]),
    )
    prev2 = torch.where(
        (prev1 == eos_token_id) | (prev2 == eos_token_id), eos_token_id, prev2
    )
    h2 = (tokens * multipliers[0]) ^ (prev1 * multipliers[1])
    h3 = h2 ^ (prev2 * multipliers[2])
    words = torch.cat((h2[:, None].expand(-1, 8), h3[:, None].expand(-1, 8)), 1)
    ids = torch.remainder(words, vocab_sizes[None]) + offsets[None]
    return torch.where((t < query_start_loc[-1])[:, None], ids, 0)


def compact_rows(table, ids):
    """Four 160B rows occupy five 128B segments; no flattened byte indices.

    IDs are bounded local int32 row indices. Complete groups of four rows
    are required. The largest supported carrier has <2**31 segments.
    """
    ids = ids.astype(jnp.int32)
    segment = (ids // 4) * 5 + ids % 4
    pair = table[segment[..., None] + jnp.arange(2)].reshape(ids.shape + (256,))
    lane = (ids % 4)[..., None]
    return jnp.where(
        lane == 0,
        pair[..., :160],
        jnp.where(
            lane == 1,
            pair[..., 32:192],
            jnp.where(lane == 2, pair[..., 64:224], pair[..., 96:256]),
        ),
    )


def _bf16(x):
    # Keep source BF16 rounding boundaries even when XLA can fuse neighbors.
    return jax.lax.optimization_barrier(x.astype(jnp.bfloat16))


def resident_lookup(table, ids, scale, *, vocab_start: int, local_rows: int):
    """Owner-masked BF16 [T,2560]; caller performs the TP BF16 all-reduce.

    Fixed eight-query chunks are required for full-size SparseCore gather
    lowering. Pad/crop only the queries, never the persistent 6.4GB table.
    """
    token_count = ids.shape[0]
    if token_count == 0:
        return jnp.zeros((0, 2560), jnp.bfloat16)
    padded = jnp.pad(ids.astype(jnp.int32), ((0, (-token_count) % 8), (0, 0)))

    def gather(chunk):
        owned = (chunk >= vocab_start) & (chunk < vocab_start + local_rows)
        local = jnp.where(owned, chunk - vocab_start, 0)
        bits = compact_rows(table, local)
        values = jax.lax.bitcast_convert_type(bits, jnp.float8_e4m3fn)
        values = _bf16(_bf16(values) * scale.astype(jnp.bfloat16))
        return jnp.where(owned[..., None], values, jnp.bfloat16(0))

    result = jax.lax.map(gather, padded.reshape(-1, 8, 16))
    return result.reshape(-1, 2560)[:token_count]


def _norm(x, weight):
    x = x.astype(jnp.float32)
    variance = jnp.mean(x * x, axis=-1, keepdims=True)
    return _bf16(
        x
        * jax.lax.rsqrt(variance + 1e-6)
        * (1 + weight.reshape(4, 2560).astype(jnp.float32))
    )


def ple_values(
    embedding, query, key_weight, value_weight, norm_key, norm_query, norm_conv
):
    """Source-execution BF16 projections, grouped Gemma norms, signed sqrt gate."""

    def linear(weight):
        return _bf16(
            jnp.dot(
                embedding,
                weight.T,
                precision=jax.lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )
        )

    key, value = linear(key_weight).reshape(-1, 4, 2560), linear(value_weight)
    key = _norm(key, norm_key)
    query = _norm(query.reshape(-1, 4, 2560), norm_query)
    product = _bf16(key * query)
    score = _bf16(jnp.sum(product.astype(jnp.float32), axis=-1, keepdims=True))
    score = _bf16(score.astype(jnp.float32) / math.sqrt(2560))
    clamped = _bf16(jnp.maximum(jnp.abs(score), jnp.asarray(1e-6, jnp.bfloat16)))
    magnitude = _bf16(jnp.sqrt(clamped.astype(jnp.float32)))
    signed = _bf16(jnp.sign(score) * magnitude)
    gate = _bf16(jax.nn.sigmoid(signed.astype(jnp.float32)))
    gated = _bf16(gate * value[:, None, :])
    return gated.reshape(-1, 10240), _norm(gated, norm_conv).reshape(-1, 10240)


def short_conv(x, weight, history, query_start_loc, state_indices, has_initial_states):
    """Kernel4/dilation3, packed mixed requests, BF16 history [slots,9,C].

    Starts and slot indices are int32. The step metadata builder owns
    monotone starts in [0,T] and unique in-range slots; -1 denotes padded
    requests. Empty requests leave their slots untouched. Scratch is
    O(T*C + R*9*C), never R times the largest request length. History stores
    normalized gated values, not convolution results.
    """
    token_count, channels = x.shape
    requests = state_indices.shape[0]
    if token_count == 0 or requests == 0:
        return jnp.zeros_like(x), history
    starts = query_start_loc.astype(jnp.int32)
    slots = state_indices.astype(jnp.int32)
    t = jnp.arange(token_count, dtype=jnp.int32)
    req = jnp.minimum(jnp.sum(t[:, None] >= starts[None, 1:], axis=1), requests - 1)
    relative = t - starts[req]
    valid = (t < starts[-1]) & (slots[req] >= 0)
    safe_slots = jnp.clip(slots, 0, history.shape[0] - 1)
    accum = jnp.zeros((token_count, channels), jnp.float32)
    for tap in range(4):
        distance = 9 - tap * 3
        position = relative - distance
        past = history[safe_slots[req], jnp.clip(position + 9, 0, 8)]
        past = jnp.where(has_initial_states[req, None], past, 0)
        current = x[jnp.clip(t - distance, 0, token_count - 1)]
        value = jnp.where((position >= 0)[:, None], current, past)
        accum = accum + value.astype(jnp.float32) * weight[:, tap].astype(jnp.float32)
    pre = _bf16(accum)
    output = _bf16(jax.nn.silu(pre.astype(jnp.float32)))
    output = jnp.where(valid[:, None], output, 0)

    lengths = starts[1:] - starts[:-1]
    positions = lengths[:, None] - 9 + jnp.arange(9)[None, :]
    past = history[safe_slots[:, None], jnp.clip(positions + 9, 0, 8)]
    past = jnp.where(has_initial_states[:, None, None], past, 0)
    current = x[jnp.clip(starts[:-1, None] + positions, 0, token_count - 1)]
    updated = jnp.where((positions >= 0)[..., None], current, past)
    # Scatter padding out of bounds with drop, so it cannot alias an active
    # slot (including when starts has repeated trailing entries).
    destinations = jnp.where((lengths > 0) & (slots >= 0), slots, history.shape[0])
    history = history.at[destinations].set(updated, mode="drop")
    return output, history


@functools.cache
def build_lookup_op(vocab_start: int, local_rows: int):
    def fn(table: jax.Array, ids: jax.Array, scale: jax.Array) -> jax.Array:
        return resident_lookup(
            table, ids, scale, vocab_start=vocab_start, local_rows=local_rows
        )

    op = pallas.jax_op(f"qwen4_exp::ple_lookup_{vocab_start}_{local_rows}", fn)

    @op.register_fake
    def fake(table, ids, scale):
        return table.new_empty((ids.shape[0], 2560), dtype=torch.bfloat16)

    def run(table, ids, scale):
        # Hash stays signed int64 through remainder. The resident embedding
        # constructor bounds the completed global row IDs below 2**31. Cast
        # before the JAX bridge, whose x64-disabled tracing otherwise refines
        # an int64 Torch argument to an incompatible int32 function input.
        return op(table, ids.to(torch.int32), scale)

    return run


@functools.cache
def build_short_conv_op():
    def fn(
        x: jax.Array,
        weight: jax.Array,
        history: jax.Array,
        starts: jax.Array,
        slots: jax.Array,
        initial: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        output, updated = short_conv(x, weight, history, starts, slots, initial)
        return updated, output

    op = pallas.jax_op("qwen4_exp::ple_short_conv", fn, donate_argnums=(2,))

    @op.register_fake
    def fake(x, weight, history, starts, slots, initial):
        return torch.empty_like(history), torch.empty_like(x)

    def run(x, weight, history, starts, slots, initial):
        updated, output = op(x, weight, history, starts, slots, initial)
        history.copy_(updated)
        return output

    return run


@functools.cache
def build_values_op():
    def fn(
        embedding: jax.Array,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        nk: jax.Array,
        nq: jax.Array,
        nc: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return ple_values(embedding, query, key, value, nk, nq, nc)

    op = pallas.jax_op("qwen4_exp::ple_values_bf16", fn)

    @op.register_fake
    def fake(embedding, query, *args):
        return torch.empty_like(query), torch.empty_like(query)

    return op
