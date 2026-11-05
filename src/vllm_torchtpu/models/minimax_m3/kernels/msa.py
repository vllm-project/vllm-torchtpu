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
"""MiniMax Sparse Attention (MSA) in JAX.

Per query token at position ``p`` and KV head ``h`` the lightning indexer
scores every causal-visible KV block ``b`` as ``max_{j in b} iq[p, h] . ik[j]``,
forces the local block ``p // block_size`` (and ``init_blocks`` leading
blocks) into the selection, and keeps the ``topk`` best blocks. Attention then
runs over the selected blocks only (`msa_attention.msa_sparse_attention`).
"""

import functools

import jax
import jax.numpy as jnp

from vllm_torchtpu.models.minimax_m3.kernels import (
    msa_attention,
    msa_indexer,
    msa_select,
    msa_write,
)

_DROPPED_PAGE = 2**30


def token_positions(
    num_tokens: int, seq_lens: jax.Array, cu_q_lens: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-token ``(sequence id, absolute position, is-real-token)``."""
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    seq_id = jnp.searchsorted(cu_q_lens[1:], tok, side="right").astype(jnp.int32)
    seq_id = jnp.minimum(seq_id, seq_lens.shape[0] - 1)
    q_len = cu_q_lens[seq_id + 1] - cu_q_lens[seq_id]
    pos = seq_lens[seq_id] - q_len + (tok - cu_q_lens[seq_id])
    valid = tok < cu_q_lens[-1]
    return seq_id, jnp.where(valid, pos, 0), valid


def _token_pages(
    block_tables_2d: jax.Array,
    seq_id: jax.Array,
    pos: jax.Array,
    valid: jax.Array,
    block_size: int,
) -> jax.Array:
    page = block_tables_2d[seq_id, pos // block_size]
    return jnp.where(valid, page, _DROPPED_PAGE)


def page_size(cache: jax.Array) -> int:
    """Tokens per page of a packed ``[pages, S / p, p, W]`` cache."""
    return cache.shape[1] * cache.shape[2]


def _cache_rows(cache: jax.Array, rows: jax.Array) -> jax.Array:
    if jnp.issubdtype(cache.dtype, jnp.floating) and jnp.finfo(cache.dtype).bits == 8:
        # e4m3fn has no inf: an out-of-range value would be stored as NaN and
        # poison every token attending its page (0 * NaN in p @ v).
        limit = float(jnp.finfo(cache.dtype).max)
        rows = jnp.clip(rows, -limit, limit)
    return rows.astype(cache.dtype)


def write_rows_xla(
    cache: jax.Array, rows: jax.Array, pages: jax.Array, pos: jax.Array
) -> jax.Array:
    """XLA scatter of one row per token; relayouts the whole cache on TPU.

    Kept as the reference for `write_rows`.
    """
    packing = cache.shape[2]
    slot = pos % page_size(cache)
    return cache.at[pages, slot // packing, slot % packing].set(
        _cache_rows(cache, rows), mode="drop"
    )


def write_rows(
    cache: jax.Array,
    rows: jax.Array,
    pages: jax.Array,
    pos: jax.Array,
    *,
    max_num_seqs: int | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Write one row per token into a packed ``[pages, S / p, p, W]`` cache.

    Only the pages that receive rows are rewritten (`msa_write`). A request's
    tokens must be consecutive with consecutive positions. `max_num_seqs`
    bounds the requests in the batch (default: one per token).
    """
    num_tokens = rows.shape[0]
    max_runs = msa_write.max_write_runs(
        num_tokens,
        num_tokens if max_num_seqs is None else max_num_seqs,
        page_size(cache),
    )
    return msa_write.write_rows(
        cache,
        _cache_rows(cache, rows),
        pages,
        pos % page_size(cache),
        max_runs=max_runs,
        interpret=interpret,
    )


def sparse_attention(
    q: jax.Array,
    kv_cache: jax.Array,
    selected: jax.Array,
    block_tables_2d: jax.Array,
    seq_id: jax.Array,
    pos: jax.Array,
    token_pages: jax.Array,
    valid: jax.Array,
    *,
    sm_scale: float,
    interpret: bool = False,
) -> jax.Array:
    """GQA attention of every token over its selected blocks."""
    if selected.shape[1] != 1:
        raise NotImplementedError(
            "MSA attention selects pages for one KV head per rank; got "
            f"{selected.shape[1]}. Use a tensor parallel size of at least "
            "num_key_value_heads."
        )
    block_size = page_size(kv_cache)
    selected = selected[:, 0]
    # Page keys (seq * pages_per_seq + block): the kernel translates them to
    # physical pages in its page fetch, which saves a [T, topk] gather of the
    # block table (0.87 ms per layer at T=8192 on the v14 twin, j-30e45a15).
    pages_per_seq = block_tables_2d.shape[1]
    keys = jnp.where(
        (selected >= 0) & valid[:, None], seq_id[:, None] * pages_per_seq + selected, -1
    )
    local_key = jnp.where(valid, seq_id * pages_per_seq + pos // block_size, -1)
    return msa_attention.msa_sparse_attention(
        q,
        kv_cache,
        keys,
        local_key,
        pos % block_size + 1,
        sm_scale=sm_scale,
        block_q=msa_attention.DEFAULT_BLOCK_Q,
        pages_per_step=msa_attention.DEFAULT_PAGES_PER_STEP,
        page_table=block_tables_2d.reshape(-1),
        interpret=interpret,
    )


def msa_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    index_q: jax.Array,
    index_k: jax.Array,
    kv_cache: jax.Array,
    index_cache: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Write this step's K/V and index keys, select blocks, attend.

    Shapes: ``q [T, Hq, D]``, ``k``/``v [T, 1, D]``, ``index_q [T, 1, D]``,
    ``index_k [T, D]``; caches in the packed layout ``[pages, S / p, p, W]``
    with rows ``[K | V]`` (``W = 2D``) and index keys (``W = D``). Returns
    ``(kv_cache, index_cache, out [T, Hq, D])``.
    """
    num_tokens = q.shape[0]
    block_size = page_size(kv_cache)
    max_num_seqs = seq_lens.shape[0]
    block_tables_2d = block_tables.reshape(max_num_seqs, -1)
    seq_id, pos, valid = token_positions(num_tokens, seq_lens, cu_q_lens)
    pages = _token_pages(block_tables_2d, seq_id, pos, valid, block_size)

    write = functools.partial(
        write_rows, pages=pages, pos=pos, max_num_seqs=max_num_seqs, interpret=interpret
    )
    kv_cache = write(kv_cache, jnp.concatenate([k, v], axis=-1)[:, 0])
    index_cache = write(index_cache, index_k)

    scores = msa_indexer.block_scores(
        index_q,
        index_cache,
        seq_lens,
        block_tables,
        cu_q_lens,
        distribution[2],
        block_q=msa_indexer.DEFAULT_BLOCK_Q,
        large_block_q=msa_indexer.DEFAULT_LARGE_BLOCK_Q,
        pages_per_step=msa_indexer.DEFAULT_PAGES_PER_STEP,
        interpret=interpret,
    )
    selected = msa_select.select_blocks(
        scores,
        pos,
        block_size=block_size,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        interpret=interpret,
    )
    out = sparse_attention(
        q,
        kv_cache,
        selected,
        block_tables_2d,
        seq_id,
        pos,
        pages,
        valid,
        sm_scale=sm_scale,
        interpret=interpret,
    )
    return kv_cache, index_cache, out
