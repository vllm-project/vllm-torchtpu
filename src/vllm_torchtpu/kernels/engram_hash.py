# SPDX-License-Identifier: Apache-2.0
"""Engram n-gram hashing as one fused XLA program.

The reference (``NgramHashState._hash``) multiplies mapped token ids by
63-bit multipliers in int64. With JAX's 64-bit types off, this computes the
same integers on uint32 pairs: products from 15-bit multiplier limbs, the
rolling XOR on both halves, and the remainder by feeding the 64-bit value to
the prime one byte at a time. Every intermediate stays below 2^32 provided
token ids are below ``MAX_SOURCE`` and primes below ``MAX_PRIME`` (checked by
``hash_tables``).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

LIMB_BITS = 15
LIMBS = 4
MAX_SOURCE = (2**32 - 1) // (2**LIMB_BITS - 1)
MAX_PRIME = 2**24


def hash_tables(
    token_map: np.ndarray,
    multipliers: np.ndarray,
    primes: np.ndarray,
    offsets: np.ndarray,
    pad_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """INT32 multiplier limbs [L, N, LIMBS], primes [L, N-1, H], offsets [L, C]."""
    multipliers = np.asarray(multipliers, dtype=np.int64)
    primes = np.asarray(primes, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    if (
        multipliers.min() < 0
        or multipliers.max() >= 2 ** (LIMB_BITS * LIMBS)
        or primes.min() < 2
        or primes.max() >= MAX_PRIME
        or offsets.min() < 0
        or int(offsets.max()) + int(primes.max()) >= 2**31
        or int(np.asarray(token_map).max()) >= MAX_SOURCE
        or not 0 <= pad_id < MAX_SOURCE
    ):
        raise ValueError("Engram tables exceed the fused hash's exact range")
    mask = (1 << LIMB_BITS) - 1
    limbs = np.stack(
        [(multipliers >> (LIMB_BITS * i)) & mask for i in range(LIMBS)], axis=-1
    )
    return limbs.astype(np.int32), primes.astype(np.int32), offsets.astype(np.int32)


def _product(value, limbs):
    """64-bit ``value * multiplier`` as (high, low) uint32."""
    high = jnp.zeros_like(value)
    low = jnp.zeros_like(value)
    for i in range(LIMBS):
        part = value * limbs[..., i]
        shift = LIMB_BITS * i
        if shift == 0:
            part_high, part_low = jnp.zeros_like(part), part
        elif shift < 32:
            part_high, part_low = part >> (32 - shift), part << shift
        else:
            part_high, part_low = part << (shift - 32), jnp.zeros_like(part)
        total = low + part_low
        high = high + part_high + (total < low).astype(jnp.uint32)
        low = total
    return high, low


def _remainder(high, low, prime):
    """(high * 2^32 + low) mod prime, prime < 2^24."""
    rest = jnp.zeros_like(prime)
    for word in (high, low):
        for shift in (24, 16, 8, 0):
            rest = (rest * 256 + ((word >> shift) & 0xFF)) % prime
    return rest


# Above this a one-hot lookup costs more than the gather it replaces.
_ONE_HOT_ENTRIES = 256


def _lookup(table, index):
    """``table[clip(index)]`` for a 1-D table; one-hot when the table is short.

    XLA lowers a gather from a few dozen entries to a chain of selects with one
    scalar operand per entry; comparing against the whole table is one fusion.
    """
    if table.shape[0] > _ONE_HOT_ENTRIES:
        return jnp.take(table, index, mode="clip")
    index = jnp.clip(index, 0, table.shape[0] - 1)
    hit = index[..., None] == jnp.arange(table.shape[0], dtype=index.dtype)
    return jnp.sum(jnp.where(hit, table, 0), axis=-1, dtype=table.dtype)


def _shifted(values, ngram):
    """``values[max(i - s, 0)]`` for s in range(ngram), as [count, ngram]."""
    count = values.shape[0]
    cols = []
    for s in range(ngram):
        keep = max(count - s, 0)
        head = jnp.broadcast_to(values[:1], (count - keep,))
        cols.append(jnp.concatenate([head, values[:keep]]))
    return jnp.stack(cols, axis=1)


@functools.partial(
    jax.jit,
    static_argnames=(
        "lookback_depth",
        "block_size",
        "pad_id",
        "dead_id",
        "use_cache",
        "sentinel",
    ),
)
def engram_hash(
    token_map: jax.Array,
    limbs: jax.Array,
    primes: jax.Array,
    offsets: jax.Array,
    input_ids: jax.Array,
    positions: jax.Array,
    query_start_loc: jax.Array,
    dead_mask: jax.Array,
    lookback_token_ids: jax.Array,
    lookback_dead_mask: jax.Array,
    cache: jax.Array,
    block_table: jax.Array,
    *,
    lookback_depth: int,
    block_size: int,
    pad_id: int,
    dead_id: int,
    use_cache: bool,
    sentinel: int | None = None,
) -> jax.Array:
    """INT32 [T, L, (N-1)*H] hash ids, equal to ``NgramHashState._hash``.

    With ``sentinel`` the dead masks are ``ids == sentinel``, computed here
    instead of read from ``dead_mask`` and ``lookback_dead_mask``.
    """
    if sentinel is not None:
        dead_mask = input_ids == sentinel
        lookback_dead_mask = lookback_token_ids == sentinel
    count = input_ids.shape[0]
    layers, ngram = limbs.shape[:2]
    clip = functools.partial(jnp.take, mode="clip")
    positions = positions.astype(jnp.int32)
    query_start_loc = query_start_loc.astype(jnp.int32)
    rows = jnp.arange(count, dtype=jnp.int32)
    requests = jnp.sum(query_start_loc[None, 1:] <= rows[:, None], axis=1)
    requests = jnp.minimum(requests, lookback_token_ids.shape[0] - 1)
    starts = jnp.minimum(_lookup(query_start_loc, requests), count - 1)
    chunk_positions = clip(positions, starts)
    shifts = jnp.arange(ngram, dtype=jnp.int32)
    past = positions[:, None] - shifts
    in_chunk = past >= chunk_positions[:, None]
    mapped_batch = jnp.where(dead_mask, dead_id, clip(token_map, input_ids))
    batch = _shifted(mapped_batch, ngram)
    cols = chunk_positions[:, None] - 1 - past
    in_window = ~in_chunk & (cols >= 0) & (cols < lookback_depth)
    cols = jnp.clip(cols, 0, lookback_depth - 1)
    requests_col = requests[:, None]
    window_index = requests_col * lookback_token_ids.shape[1] + cols
    window_ids = _lookup(lookback_token_ids.reshape(-1), window_index)
    known = in_window & (window_ids >= 0)
    mapped_lookback = jnp.where(
        lookback_dead_mask,
        dead_id,
        clip(token_map, jnp.maximum(lookback_token_ids, 0)),
    )
    window = _lookup(mapped_lookback.reshape(-1), window_index)
    if use_cache:
        logical = jnp.clip(past, 0, block_table.shape[1] * block_size - 1)
        blocks = block_table[requests_col, logical // block_size]
        slots = jnp.clip(
            blocks * block_size + logical % block_size, 0, cache.shape[0] - 1
        )
        fallback = clip(cache, slots)
    else:
        fallback = jnp.full_like(window, pad_id)
    source = jnp.where(in_chunk, batch, jnp.where(known, window, fallback))
    blocked = jnp.cumsum((past < 0) | (source == dead_id), axis=1) > 0
    source = jnp.where(blocked, pad_id, source).astype(jnp.uint32)
    # Every (layer, shift) at once, tokens on lanes: [L, N, T] products.
    terms = _product(source.T[None], limbs.astype(jnp.uint32)[:, :, None])
    high, low = terms[0][:, 0], terms[1][:, 0]
    highs, lows = [], []
    for shift in range(1, ngram):
        high, low = high ^ terms[0][:, shift], low ^ terms[1][:, shift]
        highs.append(high)
        lows.append(low)
    # [L, N-1, 1, T] against primes [L, N-1, H, 1] -> [L, N-1, H, T].
    rest = _remainder(
        jnp.stack(highs, axis=1)[:, :, None],
        jnp.stack(lows, axis=1)[:, :, None],
        primes.astype(jnp.uint32)[..., None],
    )
    hashed = rest.reshape(-1, count).T.reshape(count, layers, -1)
    return hashed.astype(jnp.int32) + offsets[None].astype(jnp.int32)
