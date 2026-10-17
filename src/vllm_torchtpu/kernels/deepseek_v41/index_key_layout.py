# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-copy D128 BF16 index keys in native TPU bitcast halfplanes.

Each int32[128] owner row contains two records: low16 is the even record,
high16 is the odd record. Each record is already in split channel order
[ordinary even64, ordinary odd64]. Owner geometry and record IDs are unchanged.
"""

import jax
import jax.numpy as jnp

DIM = 128


def encode_records(keys: jax.Array) -> jax.Array:
    """Return uint32[n,128] split-channel bits with legacy NaN normalization."""
    if keys.ndim != 2 or keys.shape[1] != DIM or keys.dtype != jnp.bfloat16:
        raise ValueError("Expected ordinary-channel BF16 [records,128] index keys")
    bits = jax.lax.bitcast_convert_type(keys, jnp.uint16).astype(jnp.uint32)
    normalized = jax.lax.bitcast_convert_type(bits << 16, jnp.float32).astype(
        jnp.bfloat16
    )
    split = jnp.concatenate((normalized[:, ::2], normalized[:, 1::2]), axis=1)
    return jax.lax.bitcast_convert_type(split, jnp.uint16).astype(jnp.uint32)


def to_ordinary(keys: jax.Array) -> jax.Array:
    """Undo split channel order for ordinary model/eager BF16 reads."""
    if keys.shape[-1] != DIM:
        raise ValueError("Expected split-order [...,128] index keys")
    return jnp.stack((keys[..., : DIM // 2], keys[..., DIM // 2 :]), axis=-1).reshape(
        keys.shape
    )


def unpack_rows(words: jax.Array) -> jax.Array:
    """Decode [...,rows,128] to split-order BF16 [...,2*rows,128] outside Pallas.

    Pallas kernels use native pltpu.bitcast instead. The FP32 conversion here
    retains the legacy decoder's NaN semantics without a TPU-only primitive.
    """
    if words.dtype != jnp.int32 or words.shape[-1] != DIM or words.ndim < 2:
        raise ValueError("Expected int32 [...,rows,128] index words")
    low = jax.lax.bitcast_convert_type(words << 16, jnp.float32).astype(jnp.bfloat16)
    high = jax.lax.bitcast_convert_type(words & jnp.int32(-65536), jnp.float32).astype(
        jnp.bfloat16
    )
    return jnp.stack((low, high), axis=-2).reshape(
        (*words.shape[:-2], words.shape[-2] * 2, DIM)
    )


def gather_records(
    pool: jax.Array | tuple[jax.Array, jax.Array], records: jax.Array
) -> jax.Array:
    """Gather SPLIT-order BF16 [*records.shape,128]; invalid IDs return zeros.

    Tuple banks retain global record IDs. Only requested rows are decoded;
    this helper stores no persistent secondary K representation.
    """
    banks = pool if isinstance(pool, tuple) else (pool,)
    if not banks or any(
        b.ndim != 2 or b.shape[1] != DIM or b.dtype != jnp.int32 for b in banks
    ):
        raise ValueError("Expected int32 [rows,128] index banks")
    words = jnp.zeros((*records.shape, DIM), jnp.int32)
    offset = 0
    valid = jnp.zeros(records.shape, jnp.bool_)
    for bank in banks:
        local = records - offset
        selected = (local >= 0) & (local < bank.shape[0] * 2)
        fetched = bank[jnp.clip(local // 2, 0, bank.shape[0] - 1)]
        words = jnp.where(selected[..., None], fetched, words)
        valid |= selected
        offset += bank.shape[0] * 2
    bits = jax.lax.shift_right_logical(words, ((records % 2) * 16)[..., None])
    values = jax.lax.bitcast_convert_type(bits << 16, jnp.float32).astype(jnp.bfloat16)
    return jnp.where(valid[..., None], values, jnp.bfloat16(0))


def merge_record_rows(
    cache: jax.Array, records: jax.Array, values: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Build full-row writes, preserving sibling halfplanes and unused bits.

    Valid record IDs must be unique (as in scatter_cache_values). Records in
    the same owner row produce identical merged rows, so DMA order is harmless.
    Invalid IDs return row=-1 for the production full-row scatter's drop path.
    """
    if cache.ndim != 2 or cache.shape[1] != DIM or cache.dtype != jnp.int32:
        raise ValueError("Expected int32 [rows,128] index bank")
    if (
        records.ndim != 1
        or records.shape[0] != values.shape[0]
        or records.dtype != jnp.int32
    ):
        raise ValueError("Expected int32 record IDs matching incoming key count")
    encoded = encode_records(values)
    if values.shape[0] == 0:
        return records, jnp.zeros((0, DIM), jnp.int32)
    valid = (records >= 0) & (records < cache.shape[0] * 2)
    rows = jnp.where(valid, records // 2, -1)
    old = cache[jnp.clip(rows, 0, cache.shape[0] - 1)].astype(jnp.uint32)
    same = (rows[:, None] == rows[None, :]) & valid[:, None] & valid[None, :]
    parts = []
    for parity in (0, 1):
        writers = same & ((records[None, :] % 2) == parity)
        source = encoded[jnp.argmax(writers, axis=1)]
        original = (old >> (16 * parity)) & jnp.uint32(65535)
        parts.append(jnp.where(writers.any(axis=1)[:, None], source, original))
    return rows, (parts[0] | (parts[1] << 16)).astype(jnp.int32)
