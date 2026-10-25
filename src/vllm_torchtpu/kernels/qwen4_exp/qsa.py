# SPDX-License-Identifier: Apache-2.0
"""QSA producers, RoPE and request-owned BF16 indexer state updates."""

import jax
import jax.numpy as jnp

from .rounding import bf16


def gemma_norm(x, weight, norm_eps):
    x = x.astype(jnp.float32)
    return bf16(
        x
        * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + norm_eps)
        * (1 + weight.astype(jnp.float32))
    )


def _fma(a, b, c):
    product = a.astype(jnp.float32) * b.astype(jnp.float32)
    c = c.astype(jnp.float32)
    high = jax.lax.optimization_barrier(product + c)
    z = jax.lax.optimization_barrier(high - product)
    residual = (product - (high - z)) + (c - z)
    bits = jax.lax.bitcast_convert_type(high, jnp.uint32)
    tie = (bits & jnp.uint32(0xFFFF)) == jnp.uint32(0x8000)
    adjusted = jnp.where(
        tie & (residual > 0),
        jnp.nextafter(high, jnp.float32(jnp.inf)),
        jnp.where(
            tie & (residual < 0), jnp.nextafter(high, jnp.float32(-jnp.inf)), high
        ),
    )
    return bf16(adjusted)


def _rope_parts(x, positions, cache, rotary_dim):
    # Gather complete aligned rows before slicing the logical cos/sin values.
    # Folding the slice into the gather lowers to serial partial-row reads;
    # a [positions, 2, half] owner instead incurs a whole-table layout copy.
    rows = jax.lax.optimization_barrier(cache[positions])
    half = rotary_dim // 2
    cs = rows[:, :rotary_dim].reshape(len(x), 2, half).astype(jnp.float32)
    shape = (len(x),) + (1,) * (x.ndim - 2) + (half,)
    return (
        x[..., :half].astype(jnp.float32),
        x[..., half : 2 * half].astype(jnp.float32),
        cs[:, 0].reshape(shape),
        cs[:, 1].reshape(shape),
    )


def main_rope(x, positions, cache, rotary_dim):
    """Main rotary uses promoted BF16 trig and FP32 two-product arithmetic."""
    a, b, c, s = _rope_parts(x, positions, cache, rotary_dim)
    return jnp.concatenate(
        [bf16(a * c - b * s), bf16(b * c + a * s), x[..., rotary_dim:]], -1
    )


def index_rope(x, positions, cache, rotary_dim):
    """Frozen Triton3.8 contract B(a*c-B(b*s)), B(a*s+B(b*c))."""
    a, b, c, s = _rope_parts(x, positions, cache, rotary_dim)
    return jnp.concatenate(
        [
            _fma(a, c, -bf16(b * s)),
            _fma(a, s, bf16(b * c)),
            x[..., rotary_dim:],
        ],
        -1,
    )


def produce_kv(key, value, nk, positions, trig, norm_eps, rotary_dim):
    """Cached keys and values ``[T, kv_heads, D]`` from the k / v projections."""
    head_dim = nk.shape[0]
    key = key.reshape(len(key), key.shape[1] // head_dim, head_dim)
    value = value.reshape(len(value), value.shape[1] // head_dim, head_dim)
    key = main_rope(gemma_norm(key, nk, norm_eps), positions, trig, rotary_dim)
    return key, value


def produce(
    qgate, key, value, index_qk, nq, nk, niq, positions, trig, norm_eps, rotary_dim
):
    head_dim, index_dim = nq.shape[0], niq.shape[0]
    qgate = qgate.reshape(len(qgate), qgate.shape[1] // (2 * head_dim), 2 * head_dim)
    q = main_rope(
        gemma_norm(qgate[..., :head_dim], nq, norm_eps), positions, trig, rotary_dim
    )
    key, value = produce_kv(key, value, nk, positions, trig, norm_eps, rotary_dim)
    index_heads = index_qk.shape[1] // index_dim - 1
    iq = index_rope(
        gemma_norm(
            index_qk[:, :-index_dim].reshape(len(index_qk), index_heads, index_dim),
            niq,
            norm_eps,
        ),
        positions,
        trig,
        rotary_dim,
    )
    return q, key, value, qgate[..., head_dim:], iq, index_qk[:, -index_dim:]


def _token_metadata(count, starts, context, slots):
    t = jnp.arange(count, dtype=jnp.int32)
    req = jnp.minimum(jnp.sum(t[:, None] >= starts[None, 1:], axis=1), len(slots) - 1)
    logical = context[req] + t - starts[req]
    valid = (t < starts[-1]) & (slots[req] >= 0)
    return req, logical, valid


def quantize_kv(value, scale):
    """CUDA reshape_and_cache: scalar FP32 division and saturated E4M3 RNE."""
    encoded = value.astype(jnp.float32) / jnp.float32(scale)
    return jnp.clip(encoded, -448, 448).astype(jnp.float8_e4m3fn)


def _indexer_updates(
    raw,
    positions,
    ring,
    tags,
    norm_k,
    trig,
    starts,
    context,
    slots,
    compress_ratio,
    norm_eps,
    rotary_dim,
    rows_are_slots=False,
):
    """``rows_are_slots``: request row ``r`` owns ring row ``r`` (``ring`` and
    ``tags`` hold one row per request and ``slots`` is ``r`` or -1, as in the
    pooled layer), so the whole-row ring gathers and the final scatter
    become selects.
    """
    count = len(raw)
    req, logical, valid = _token_metadata(count, starts, context, slots)
    safe_slots = (
        jnp.arange(len(slots), dtype=jnp.int32)
        if rows_are_slots
        else jnp.maximum(slots, 0)
    )
    ratio = compress_ratio
    # Every allocated ring row holds history: position p lives in lane
    # p % rows. Keeping the newest `rows` (>= ratio) positions rather than
    # only `ratio` lets speculative verification reject tokens without
    # evicting the accepted members of the partial compression group that
    # the next step re-reads (see config.max_qsa_speculative_tokens).
    rows = ring.shape[1]
    group_start = logical // ratio * ratio
    members = group_start[:, None] + jnp.arange(ratio)[None]
    chunk_indices = starts[req, None] + members - context[req, None]
    current = raw[jnp.clip(chunk_indices, 0, count - 1)]
    previous = ring[safe_slots[req, None], members % rows]
    previous = jnp.where((context[req] > 0)[:, None, None], previous, 0)
    joined = jnp.where(
        (chunk_indices >= starts[req, None])[..., None], current, previous
    )
    pooled = bf16(jnp.mean(joined.astype(jnp.float32), axis=1))
    first_index = starts[req] + group_start - context[req]
    old_position = tags[safe_slots[req], group_start % rows, 0]
    first_position = jnp.where(
        first_index >= starts[req],
        positions[jnp.clip(first_index, 0, count - 1)],
        old_position,
    )
    completed = index_rope(
        gemma_norm(pooled, norm_k, norm_eps),
        jnp.maximum(first_position, 0),
        trig,
        rotary_dim,
    )
    complete = valid & (logical % ratio == ratio - 1)

    # Exactly one scatter per request/lane: no duplicate-index last-writer race.
    lengths = starts[1:] - starts[:-1]
    end = context + lengths
    lanes = jnp.arange(rows)[None]
    last = end[:, None] - 1 - ((end[:, None] - 1 - lanes) % rows)
    local = starts[:-1, None] + last - context[:, None]
    old_ring = ring if rows_are_slots else ring[safe_slots]
    old_tags = tags if rows_are_slots else tags[safe_slots]
    old_ring = jnp.where((context > 0)[:, None, None], old_ring, 0)
    old_tags = jnp.where((context > 0)[:, None, None], old_tags, -1)
    new_lane = (last >= context[:, None]) & (last < end[:, None])
    rr = jnp.where(new_lane[..., None], raw[jnp.clip(local, 0, count - 1)], old_ring)
    pp = positions[jnp.clip(local, 0, count - 1)]
    tag_pairs = jnp.stack([pp, jnp.zeros_like(pp)], -1)
    tt = jnp.where(new_lane[..., None], tag_pairs, old_tags)
    live = (lengths > 0) & (slots >= 0)
    if rows_are_slots:
        ring = jnp.where(live[:, None, None], rr, ring)
        tags = jnp.where(live[:, None, None], tt, tags)
    else:
        dest = jnp.where(live, slots, ring.shape[0])
        ring = ring.at[dest].set(rr, mode="drop")
        tags = tags.at[dest].set(tt, mode="drop")
    return completed, complete, req, logical, ring, tags


def _update_indexer(
    raw,
    positions,
    compressed,
    ring,
    tags,
    norm_k,
    trig,
    starts,
    context,
    slots,
    compressed_table,
    compress_ratio,
    norm_eps,
    rotary_dim,
):
    if len(raw) == 0 or len(slots) == 0:
        return compressed, ring, tags
    completed, complete, req, logical, ring, tags = _indexer_updates(
        raw,
        positions,
        ring,
        tags,
        norm_k,
        trig,
        starts,
        context,
        slots,
        compress_ratio,
        norm_eps,
        rotary_dim,
    )
    groups_per_page = compressed.shape[1]
    cp = jnp.where(
        complete,
        compressed_table[req, logical // (groups_per_page * compress_ratio)],
        compressed.shape[0],
    )
    completed = jnp.pad(
        completed, ((0, 0), (0, compressed.shape[-1] - completed.shape[-1]))
    )
    compressed = compressed.at[
        cp, (logical // compress_ratio) % groups_per_page, 0
    ].set(completed, mode="drop")
    return compressed, ring, tags
