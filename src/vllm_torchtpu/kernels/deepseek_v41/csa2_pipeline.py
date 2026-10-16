# SPDX-License-Identifier: Apache-2.0
"""Overlap independent SparseCore gathers and TensorCore CSA2 query chunks."""

import functools

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.deepseek_v41.csa2 import csa2
from vllm_torchtpu.kernels.deepseek_v41.csa2_cache import gather_csa2_kv_sc


@functools.partial(
    jax.jit, static_argnames=("chunk_size", "bq", "scale", "compress_ratio")
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
        previous = csa2(
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
        )
        if following is not None:
            current = following
    assert previous is not None
    last_offset = (q.shape[0] - 1) // chunk_size * chunk_size
    output = jax.lax.dynamic_update_slice_in_dim(output, previous, last_offset, axis=0)
    return output, jnp.concatenate(word_chunks), jnp.concatenate(scale_chunks)
