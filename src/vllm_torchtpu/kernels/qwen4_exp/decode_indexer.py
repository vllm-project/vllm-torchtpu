# SPDX-License-Identifier: Apache-2.0
"""QSA geometry adapter for the shared StreamIndex scoring and top-k pipeline."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk
from vllm_torchtpu.kernels.qwen4_exp.indexer_components import qsa_indexing


def decode_query_chunk_size(pages_per_query: int) -> int:
    """Bound scalar-prefetched page tables with room for scoring metadata."""
    if pages_per_query < 1:
        raise ValueError("QSA page tables must have a positive row width")
    budget = (pltpu.get_tpu_info().smem_capacity_bytes - 32 * 1024) // 2
    queries = budget // (4 * (pages_per_query + 2))
    if queries < 1:
        raise ValueError(
            "One QSA page-table row exceeds the scalar-prefetch budget; "
            "reduce max_model_len."
        )
    return min(512, 1 << (queries.bit_length() - 1))


@functools.partial(jax.jit, static_argnames=("compress_ratio", "block_topk"))
def select_decode_blocks(
    indexq, compressed, lengths, table, request_ids, *, compress_ratio=4, block_topk=512
):
    """Return int32[B, block_topk] completed-group IDs with a -1 suffix.

    Inputs are BF16 query [B, H, D], BF16 cache [pages, groups_per_page,
    1, align_to(D, 128)], int32 lengths [B] including the current token, an int32
    fixed-stride page table [requests, pages_per_sequence], and request IDs [B].
    Valid page IDs and lengths bounded
    by the table capacity are the caller's contract. Only complete groups enter
    top-k; the attention caller adds the current partial group's tokens.

    Winner order is unspecified. Exact boundary-score ties choose lower group
    IDs, as in StreamIndex's forward SparseCore compaction.
    """
    if indexq.ndim != 3 or indexq.shape[1] < 1 or indexq.shape[2] < 1:
        raise ValueError("decode index query must have shape [B, H, D] with H,D >= 1")
    if indexq.dtype != jnp.bfloat16 or compressed.dtype != jnp.bfloat16:
        raise ValueError("decode index query and compressed cache must remain BF16")
    if compressed.ndim != 4 or compressed.shape[2:] != (
        1,
        (indexq.shape[2] + 127) // 128 * 128,
    ):
        raise ValueError(
            "compressed cache must have shape "
            "[pages, groups_per_page, 1, aligned_index_dim]"
        )
    page_size = compressed.shape[1]
    if page_size % 2 or 128 % page_size:
        raise ValueError("compressed groups_per_page must be even and divide 128")
    batch = indexq.shape[0]
    if (
        batch < 1
        or lengths.shape != (batch,)
        or table.ndim != 2
        or table.shape[0] < 1
        or request_ids.shape != (batch,)
        or table.shape[1] < 1
    ):
        raise ValueError(
            "lengths and fixed-stride table must match the nonempty query batch"
        )
    if any(a.dtype != jnp.int32 for a in (lengths, table, request_ids)):
        raise ValueError("lengths, table and request_ids must be int32")
    if compress_ratio < 1 or compress_ratio & (compress_ratio - 1):
        raise ValueError("compress_ratio must be a positive power of two")
    if block_topk < 16 or block_topk > 4096 or block_topk % 16:
        raise ValueError("block_topk must be a multiple of 16 in [16, 4096]")

    cache = compressed.reshape(compressed.shape[0], page_size // 2, 2, -1)
    chunk_size = min(batch, decode_query_chunk_size(table.shape[1]))

    def select_chunk(q, lens, req):
        size = len(q)
        return streamindex_topk(
            q,
            jnp.full(q.shape[:2], q.shape[-1] ** -0.5, jnp.float32),
            cache,
            lens,
            table[req].reshape(-1),
            jnp.arange(size + 1, dtype=jnp.int32),
            jnp.full((3,), size, jnp.int32),
            k=block_topk,
            compression_ratio=compress_ratio,
            num_kv_pages_per_block=max(1, 256 // page_size),
            num_queries_per_block=1,
            decode_req_batch_size=8 if size >= 8 else 4,
            enable_early_exit=True,
            components=qsa_indexing(),
        )

    if batch <= chunk_size:
        return select_chunk(indexq, lengths, request_ids)
    padding = (-batch) % chunk_size
    query = jnp.pad(indexq, ((0, padding), (0, 0), (0, 0)))
    lens = jnp.pad(lengths, (0, padding))
    req = jnp.pad(request_ids, (0, padding))
    chunks = (
        query.reshape(-1, chunk_size, *indexq.shape[1:]),
        lens.reshape(-1, chunk_size),
        req.reshape(-1, chunk_size),
    )
    _, selected = jax.lax.scan(
        lambda carry, inputs: (carry, select_chunk(*inputs)), None, chunks
    )
    return selected.reshape(-1, block_topk)[:batch]
