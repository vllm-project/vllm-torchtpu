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
"""QSA geometry adapter for the shared StreamIndex scoring and top-k pipeline."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.topk_backend import (
    SparseCoreTopK,
    TopKBackend,
)
from vllm_torchtpu.kernels.qwen4_exp.indexer_components import qsa_indexing


def _request_bounds(starts, distribution, tokens):
    """Runner boundaries are [decode_end, verify_end, active_request_end]."""
    if distribution.shape != (3,) or distribution.dtype != jnp.int32:
        raise ValueError("QSA request_distribution must be int32[3]")
    return jnp.minimum(starts[distribution[1]], tokens), jnp.minimum(
        starts[distribution[2]], tokens
    )


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


@functools.partial(
    jax.jit,
    static_argnames=(
        "compress_ratio",
        "block_topk",
        "topk_backend",
        "scheduling_group_namespace",
    ),
)
def select_decode_blocks(
    indexq,
    compressed,
    lengths,
    table,
    request_ids,
    *,
    compress_ratio=4,
    block_topk=512,
    starts=None,
    context=None,
    request_distribution: jax.Array | None = None,
    topk_backend: TopKBackend | None = None,
    scheduling_group_namespace: int | None = None,
):
    """Return int32[B, block_topk] completed-group IDs with a -1 suffix.

    Inputs are BF16 query [B, H, D], BF16 cache [pages, groups_per_page,
    1, align_to(D, 128)], int32 lengths [B] including the current token, an int32
    fixed-stride page table [requests, pages_per_sequence], and request IDs [B].
    Valid page IDs and lengths bounded
    by the table capacity are the caller's contract. Only complete groups enter
    top-k; the attention caller adds the current partial group's tokens.

    Ragged scheduling requires starts, context and Runner request_distribution.
    Without starts/context, every supplied query uses its own causal length.

    Winner order is unspecified. The default SparseCore backend chooses lower
    group IDs at exact boundary-score ties. Alternatives follow TopKBackend's
    tie contract.
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
    return select_index_blocks(
        indexq,
        cache,
        lengths,
        table,
        request_ids,
        starts=starts,
        context=context,
        request_distribution=request_distribution,
        ratio=compress_ratio,
        block_topk=block_topk,
        topk_backend=topk_backend,
        scheduling_group_namespace=scheduling_group_namespace,
    )


def select_index_blocks(
    query,
    cache,
    lengths,
    table,
    request_ids,
    *,
    starts,
    context,
    ratio,
    block_topk,
    request_distribution: jax.Array | None = None,
    topk_backend: TopKBackend | None = None,
    scheduling_group_namespace: int | None = None,
    vmem_limit_bytes: int | None = None,
    prefix_capacity: int | None = None,
):
    """Share ragged query-block reuse and bounded per-query scoring schedules.

    ``cache`` is the reader's BF16 or byte-carrier view; ``table`` addresses
    those physical pages. Runner order is decode, verify, then prefill/mixed.
    Decode and verify rows use per-query causal lengths; the suffix shares KV.
    The default SC/QSA prefill composition opts into query-sized score capacity.
    Injected backends stay full unless prefix_capacity explicitly certifies their
    reduced-pitch contract together with this extent-preserving QSA schedule.
    """
    resource_options = (
        {} if vmem_limit_bytes is None else {"vmem_limit_bytes": vmem_limit_bytes}
    )
    default_backend = topk_backend is None
    if default_backend:
        topk_backend = SparseCoreTopK()
    page_size = cache.shape[1] * 2
    query_width = query.shape[-1]
    query = jnp.pad(query, ((0, 0), (0, 0), (0, cache.shape[-1] - query_width)))
    scale = query_width**-0.5
    if (starts is None) != (context is None):
        raise ValueError("starts and context must be supplied together")

    batch = len(query)
    chunk_size = min(batch, decode_query_chunk_size(table.shape[1]))

    def select_chunk(q, lens, req):
        size = len(q)
        return streamindex_topk(
            q,
            jnp.full(q.shape[:2], scale, jnp.float32),
            cache,
            lens,
            table[req].reshape(-1),
            jnp.arange(size + 1, dtype=jnp.int32),
            jnp.full((3,), size, jnp.int32),
            k=block_topk,
            compression_ratio=ratio,
            num_kv_pages_per_block=max(1, 256 // page_size),
            num_queries_per_block=1,
            decode_req_batch_size=8 if size >= 8 else 4,
            enable_early_exit=True,
            components=qsa_indexing(),
            topk_backend=topk_backend,
            scheduling_group_namespace=scheduling_group_namespace,
            **resource_options,
        )

    def per_query():
        if batch <= chunk_size:
            return select_chunk(query, lengths, request_ids)
        padding = (-batch) % chunk_size
        padded_query = jnp.pad(query, ((0, padding), (0, 0), (0, 0)))
        lens = jnp.pad(lengths, (0, padding))
        req = jnp.pad(request_ids, (0, padding))
        chunks = (
            padded_query.reshape(-1, chunk_size, *query.shape[1:]),
            lens.reshape(-1, chunk_size),
            req.reshape(-1, chunk_size),
        )
        _, selected = jax.lax.scan(
            lambda carry, inputs: (carry, select_chunk(*inputs)), None, chunks
        )
        return selected.reshape(-1, block_topk)[:batch]

    if starts is None or context is None:
        return per_query()
    if request_distribution is None:
        raise ValueError("Ragged QSA selection requires Runner request_distribution")
    prefix_tokens, active_tokens = _request_bounds(starts, request_distribution, batch)
    if prefix_capacity is None and default_backend:
        prefix_capacity = max(block_topk, (batch + ratio - 1) // ratio)

    def prefill_query(q, cu, history):
        counts = cu[1:] - cu[:-1]
        num_seqs = jnp.max(
            jnp.where(counts > 0, jnp.arange(len(counts), dtype=jnp.int32) + 1, 0)
        )
        return streamindex_topk(
            q,
            jnp.full(q.shape[:2], scale, jnp.float32),
            cache,
            jnp.where(counts > 0, history + counts, 0),
            table.reshape(-1),
            cu,
            jnp.array([0, 0, num_seqs], jnp.int32),
            k=block_topk,
            compression_ratio=ratio,
            num_kv_pages_per_block=(max(1, 256 // page_size),) * 3,
            num_queries_per_block=(1, 128, 128),
            decode_req_batch_size=8,
            enable_early_exit=True,
            chunk_tokens=512 if len(q) % 512 == 0 else None,
            components=qsa_indexing(),
            prefix_capacity=prefix_capacity,
            topk_backend=topk_backend,
            scheduling_group_namespace=scheduling_group_namespace,
            **resource_options,
        )

    def mixed():
        from vllm_torchtpu.kernels.sparse_gqa_prefill.tuned_block_sizes import (
            PREFILL_DISPATCH_CHUNK_QUERIES,
        )

        window = min(batch, PREFILL_DISPATCH_CHUNK_QUERIES)
        output = jnp.full((batch + window, block_topk), -1, jnp.int32)

        def prefix(index, output):
            first = index * chunk_size
            rows = jnp.minimum(first + jnp.arange(chunk_size), batch - 1)
            valid = first + jnp.arange(chunk_size) < prefix_tokens
            result = select_chunk(
                query[rows], jnp.where(valid, lengths[rows], 0), request_ids[rows]
            )
            return jax.lax.dynamic_update_slice(output, result, (first, 0))

        output = jax.lax.fori_loop(
            0, (prefix_tokens + chunk_size - 1) // chunk_size, prefix, output
        )

        def prefill(index, output):
            first = prefix_tokens + index * window
            positions = first + jnp.arange(window, dtype=jnp.int32)
            rows = jnp.minimum(positions, batch - 1)
            valid = positions < active_tokens
            local_starts = jnp.clip(starts - first, 0, window)
            progress = jnp.clip(first - starts[:-1], 0, starts[1:] - starts[:-1])
            result = prefill_query(
                jnp.where(valid[:, None, None], query[rows], 0),
                local_starts,
                context + progress,
            )
            result = jnp.where((valid & (lengths[rows] > 0))[:, None], result, -1)
            return jax.lax.dynamic_update_slice(output, result, (first, 0))

        output = jax.lax.fori_loop(
            0, (active_tokens - prefix_tokens + window - 1) // window, prefill, output
        )
        return output[:batch]

    return jax.lax.cond(
        active_tokens == 0,
        lambda _: jnp.full((batch, block_topk), -1, jnp.int32),
        lambda _: jax.lax.cond(
            prefix_tokens == active_tokens,
            lambda _: per_query(),
            lambda _: jax.lax.cond(
                prefix_tokens == 0,
                lambda _: prefill_query(query, starts, context),
                lambda _: mixed(),
                None,
            ),
            None,
        ),
        None,
    )
