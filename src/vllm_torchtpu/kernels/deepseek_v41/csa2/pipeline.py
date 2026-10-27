# SPDX-License-Identifier: Apache-2.0
"""Overlap independent SparseCore gathers and TensorCore CSA2 query chunks."""

import functools

import jax
import jax.numpy as jnp

from .cache import gather_csa2_kv_sc
from .decode import csa2_decode
from .prefill import csa2_prefill


def csa2_gather_chunk_size(num_tokens: int) -> int:
    # On v7x TP1 with 64 heads, 128-row chunks overlap SC gather with TC
    # attention while retaining full gathered KV for subsequent layers.
    # Up to 512 queries, launch and concatenation costs erase the stage-level
    # benefit; use one gather and one attention without extra staging HBM.
    return num_tokens if num_tokens <= 512 else 128


def csa2_attention(
    q: jax.Array,
    words: jax.Array,
    scales: jax.Array,
    swa: jax.Array | tuple[jax.Array, jax.Array],
    starts: jax.Array,
    positions: jax.Array,
    sink: jax.Array,
    rope: jax.Array,
    *,
    swa_metadata: jax.Array,
    swa_bounds: jax.Array | None,
    main_lengths: jax.Array,
    scale: float,
    compress_ratio: int,
    bq: int,
    decoder: bool = False,
) -> jax.Array:
    """Dispatch decode and decoder-layer prefill to the common decode pipeline.

    Request boundaries are device data. Check them before using the regular
    grid; tensor shapes alone cannot distinguish decode from a mixed batch.
    Empty trailing request slots are allowed, but token padding and partial
    requests retain the ragged path. Both gather and cache reuse call here.
    """

    def ragged():
        return csa2_prefill(
            q,
            words,
            scales,
            swa,
            starts,
            positions,
            sink,
            rope,
            swa_metadata=swa_metadata,
            swa_bounds=swa_bounds,
            main_lengths=main_lengths,
            scale=scale,
            compress_ratio=compress_ratio,
            bq=bq,
        )

    num_tokens = 6
    if decoder:
        request_slots = starts.shape[0] - 1
        if request_slots:
            width, remainder = divmod(q.shape[0], request_slots)
            if not remainder and width > 8 and width % 8 == 0:
                num_tokens = width
    batch_size, tail = divmod(q.shape[0], num_tokens)
    if (
        not batch_size
        or tail
        or starts.shape[0] < batch_size + 1
        or isinstance(swa, tuple)
        or swa_bounds is not None
    ):
        return ragged()

    # Extra request slots must be empty at the end, not interleaved with Qs.
    expected_starts = jnp.minimum(
        jnp.arange(starts.shape[0], dtype=jnp.int32) * num_tokens, q.shape[0]
    )
    first_positions = positions[::num_tokens]
    consecutive = first_positions[:, None] + jnp.arange(num_tokens)
    regular = jnp.all(starts == expected_starts) & jnp.all(
        positions.reshape(batch_size, num_tokens) == consecutive
    )
    decode_kwargs = dict(
        swa_page=swa_metadata[:batch_size],
        main_lengths=main_lengths,
        num_tokens=num_tokens,
        scale=scale,
    )
    return jax.lax.cond(
        regular,
        lambda: csa2_decode(
            q,
            words,
            scales,
            swa,
            first_positions,
            sink,
            rope,
            **decode_kwargs,
        ),
        ragged,
    )


@functools.partial(
    jax.jit, static_argnames=("chunk_size", "bq", "scale", "compress_ratio", "decoder")
)
def csa2_pipeline(
    q: jax.Array,
    words: jax.Array | tuple[jax.Array, jax.Array],
    scales: jax.Array | tuple[jax.Array, jax.Array],
    records: jax.Array,
    swa: jax.Array | tuple[jax.Array, jax.Array],
    starts: jax.Array,
    positions: jax.Array,
    sink: jax.Array,
    rope: jax.Array,
    metadata: jax.Array,
    bounds: jax.Array | None,
    counts: jax.Array,
    *,
    chunk_size: int,
    bq: int,
    scale: float,
    compress_ratio: int,
    decoder: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return attention and contiguous gathered KV for subsequent layer reuse.

    Gather i+1 and attention i have independent inputs. A dependency through
    attention i-1 bounds the prefetch distance without placing either operation
    behind the other. Gather into independent arrays; a shared destination
    would serialize its writer with the preceding chunk's attention reader.
    Join the gathered outputs once, after the pipeline, so later layers reuse
    them with one persistent attention launch rather than one launch per chunk.
    Only query boundaries become chunk-relative; SWA metadata and noncausal
    bounds retain their absolute positions, including future draft tokens.
    """
    if chunk_size < 1 or not q.shape[0]:
        raise ValueError("CSA2 pipeline requires nonempty queries and positive chunks")
    if records.shape != (q.shape[0], 512) or counts.shape != q.shape[:1]:
        raise ValueError("CSA2 pipeline requires [T,512] records and [T] counts")

    word_chunks, scale_chunks = [], []
    output = jnp.empty_like(q)
    previous = None
    current = gather_csa2_kv_sc(words, scales, records[:chunk_size])
    for offset in range(0, q.shape[0], chunk_size):
        end = min(offset + chunk_size, q.shape[0])
        next_ids = records[end : end + chunk_size]
        if previous is not None:
            # Consume every barrier result. Gating only the small index array
            # avoids carrying the multi-GiB source pools through the barrier.
            previous, current, next_ids = jax.lax.optimization_barrier(
                (previous, current, next_ids)
            )
            output = jax.lax.dynamic_update_slice_in_dim(
                output, previous, offset - chunk_size, axis=0
            )
        elif end < q.shape[0]:
            # Otherwise the scheduler can gather chunk 1 before chunk 0 and
            # wait for both before starting the first attention invocation.
            current, next_ids = jax.lax.optimization_barrier((current, next_ids))
        following = (
            gather_csa2_kv_sc(words, scales, next_ids) if end < q.shape[0] else None
        )
        chunk_words, chunk_scales = current
        word_chunks.append(chunk_words)
        scale_chunks.append(chunk_scales)
        previous = csa2_attention(
            q[offset:end],
            chunk_words,
            chunk_scales,
            swa,
            jnp.clip(starts - offset, 0, end - offset),
            positions[offset:end],
            sink,
            rope[offset:end],
            swa_metadata=metadata,
            swa_bounds=bounds,
            main_lengths=counts[offset:end],
            scale=scale,
            compress_ratio=compress_ratio,
            bq=bq,
            decoder=decoder,
        )
        if following is not None:
            current = following
    assert previous is not None
    last_offset = (q.shape[0] - 1) // chunk_size * chunk_size
    output = jax.lax.dynamic_update_slice_in_dim(output, previous, last_offset, axis=0)
    return output, jnp.concatenate(word_chunks), jnp.concatenate(scale_chunks)
