# SPDX-License-Identifier: Apache-2.0
"""Overlap independent SparseCore gathers and TensorCore CSA2 query chunks."""

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from .cache import gather_csa2_kv_sc
from .csa2 import csa2

# Measured v7x TP1, 64 heads, page 64, bq 6: the DSpark K5 decode bucket of
# 48 x 6 queries takes 257 us in 96-row chunks against 285 us in one chunk,
# with bitwise-equal results. 144-, 72- and 48-row chunks were all slower
# than one chunk, so other small buckets keep a single chunk until measured.
_SMALL_CHUNKS = {288: 96}


def csa2_gather_chunk_size(num_tokens: int) -> int:
    # On v7x TP1 with 64 heads, 128-row chunks overlap SC gather with TC
    # attention while retaining full gathered KV for subsequent layers.
    # Up to 512 queries, one gather and one attention is the default; see
    # `_SMALL_CHUNKS` for the measured exceptions.
    if num_tokens > 512:
        return 128
    return _SMALL_CHUNKS.get(num_tokens, num_tokens)


@functools.partial(
    jax.jit,
    static_argnames=(
        "chunk_size",
        "bq",
        "scale",
        "compress_ratio",
        "deep_gather",
    ),
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
    deep_gather: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return attention and contiguous gathered KV for subsequent layer reuse.

    ``records`` are physical [T,512] IDs with ``counts`` valid per row; see
    `csa2_pipeline_chunks` for the schedule.
    """
    if records.shape != (q.shape[0], 512) or counts.shape != q.shape[:1]:
        raise ValueError("CSA2 pipeline requires [T,512] records and [T] counts")
    output, gathered_words, gathered_scales, _ = csa2_pipeline_chunks(
        q,
        words,
        scales,
        lambda start, end: (records[start:end], counts[start:end]),
        swa,
        starts,
        positions,
        sink,
        rope,
        metadata,
        bounds,
        chunk_size=chunk_size,
        bq=bq,
        scale=scale,
        compress_ratio=compress_ratio,
        deep_gather=deep_gather,
    )
    return output, gathered_words, gathered_scales


def csa2_attend_rows(
    q: jax.Array,
    gathered: tuple[jax.Array, jax.Array],
    counts: jax.Array,
    owners: tuple[jax.Array, jax.Array] | None,
    swa: jax.Array | tuple[jax.Array, jax.Array],
    starts: jax.Array,
    positions: jax.Array,
    sink: jax.Array,
    rope: jax.Array,
    *,
    offset: int,
    kv_owner: bool = False,
    **kwargs: Any,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Attention of query rows from ``offset``, written into whole-T owners.

    ``gathered`` holds the rows' KV words and scales. Without ``owners`` the
    call allocates them; only these rows are written. Returns the owners
    (attention output, gathered KV words) for the next row range. With
    ``kv_owner``, ``gathered`` is already the whole-T KV owner pair holding
    these rows, and ``owners`` and the result are the attention output alone.
    """
    end = offset + counts.shape[0]
    if owners is None:
        destinations = dict(new_owners=True)
    elif kv_owner:
        destinations = dict(out=owners)
    else:
        destinations = dict(out=owners[0], kv_out=owners[1])
    return csa2(
        q,
        *gathered,
        swa,
        jnp.clip(starts - offset, 0, end - offset),
        positions[offset:end],
        sink,
        rope[offset:end],
        main_lengths=counts,
        query_offset=offset,
        kv_owner=kv_owner,
        **destinations,
        **kwargs,
    )


def csa2_pipeline_chunks(
    q: jax.Array,
    words: jax.Array | tuple[jax.Array, jax.Array],
    scales: jax.Array | tuple[jax.Array, jax.Array],
    chunk_records: Callable[[int, int], tuple[jax.Array, jax.Array]],
    swa: jax.Array | tuple[jax.Array, jax.Array],
    starts: jax.Array,
    positions: jax.Array,
    sink: jax.Array,
    rope: jax.Array,
    metadata: jax.Array,
    bounds: jax.Array | None,
    *,
    chunk_size: int,
    bq: int,
    scale: float,
    compress_ratio: int,
    deep_gather: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Attention, gathered KV and counts, resolving each chunk's records lazily.

    ``chunk_records(start, end)`` returns physical [end - start, 512] IDs and
    their valid counts. Records for chunk i+1 are resolved on the TensorCore
    while the SparseCore gathers chunk i, so a page lookup costs no time in
    this SparseCore-bound loop.

    Gather i+1 and attention i have independent inputs. A dependency through
    attention i-1 bounds the prefetch distance without placing either operation
    behind the other. Gather into independent arrays; a shared destination
    would serialize its writer with the preceding chunk's attention reader.
    Each attention chunk writes its output rows and copies its staged KV rows
    in place into two whole-T owners, so later layers reuse contiguous KV with
    one persistent attention launch, without a per-chunk output update or a
    KV-word concatenation after the pipeline. Only the small FP32 scale
    chunks are still joined.
    Only query boundaries become chunk-relative; SWA metadata and noncausal
    bounds retain their absolute positions, including future draft tokens.

    ``deep_gather`` selects the three-slot SparseCore gather
    (``gather_csa2_kv_sc(deep=True)``).
    """
    if chunk_size < 1 or not q.shape[0]:
        raise ValueError("CSA2 pipeline requires nonempty queries and positive chunks")
    kwargs = dict(
        swa_metadata=metadata,
        swa_bounds=bounds,
        scale=scale,
        compress_ratio=compress_ratio,
        bq=bq,
    )
    if q.shape[0] <= chunk_size:
        records, counts = chunk_records(0, q.shape[0])
        gathered_words, gathered_scales = gather_csa2_kv_sc(
            words, scales, records, deep=deep_gather
        )
        output = csa2(
            q,
            gathered_words,
            gathered_scales,
            swa,
            starts,
            positions,
            sink,
            rope,
            main_lengths=counts,
            **kwargs,
        )
        return output, gathered_words, gathered_scales, counts

    scale_chunks, count_chunks = [], []
    # Every row of both owners is written by exactly one chunk. The first
    # chunk allocates them as its own outputs: XLA merges identical
    # `lax.empty` buffers across layers, then copies one layer's owners so
    # the two in-place writers stay disjoint.
    owners = None
    previous = None
    records, counts = chunk_records(0, chunk_size)
    current = gather_csa2_kv_sc(words, scales, records, deep=deep_gather)
    for offset in range(0, q.shape[0], chunk_size):
        end = min(offset + chunk_size, q.shape[0])
        following_end = min(end + chunk_size, q.shape[0])
        next_ids = chunk_records(end, following_end) if end < q.shape[0] else None
        if previous is not None:
            # Consume every barrier result. Gating only the small index array
            # avoids carrying the multi-GiB source pools through the barrier.
            owners, current, next_ids = jax.lax.optimization_barrier(
                (previous, current, next_ids)
            )
        elif end < q.shape[0]:
            # Otherwise the scheduler can gather chunk 1 before chunk 0 and
            # wait for both before starting the first attention invocation.
            # Gating on q too keeps gather 1 beside attention 0: issued any
            # earlier, it competes with the Q projection for HBM bandwidth.
            current, next_ids, q = jax.lax.optimization_barrier((current, next_ids, q))
        following = (
            gather_csa2_kv_sc(words, scales, next_ids[0], deep=deep_gather)
            if next_ids is not None
            else None
        )
        scale_chunks.append(current[1])
        count_chunks.append(counts)
        previous = csa2_attend_rows(
            q,
            current,
            counts,
            owners,
            swa,
            starts,
            positions,
            sink,
            rope,
            offset=offset,
            **kwargs,
        )
        if following is not None:
            current = following
            counts = next_ids[1]
    assert previous is not None
    output, gathered_words = previous
    return (
        output,
        gathered_words,
        jnp.concatenate(scale_chunks),
        jnp.concatenate(count_chunks),
    )
