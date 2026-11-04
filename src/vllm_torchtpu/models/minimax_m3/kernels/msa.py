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
runs over the selected blocks only. The block attention reuses the shared
ragged paged attention kernel: every token becomes a one-token sequence whose
page table lists its selected blocks in ascending order, so the local block is
last and the kernel's causal mask truncates it at the token itself.
"""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import merge_kv

# Rows a single virtual-table attention call may carry. The kernel keeps the
# page table in SMEM: rows * topk * 4 bytes must stay within half of it.
MAX_ROWS_PER_ATTENTION_CALL = 2048
# Query windows the block scorer tries, smallest first; the step's token
# bucket is always the last tier.
SCORE_WINDOW_TIERS = (16, 256)
# Pages one scoring iteration covers. Decode-sized windows take bigger chunks:
# fewer serial iterations over a long context, at the cost of reading up to
# one chunk past the request's last page.
_PAGES_PER_CHUNK = 16
_SMALL_WINDOW_PAGES_PER_CHUNK = 64
_NEG_INF = float("-inf")
_FORCED_LOCAL_SCORE = 1e29
_FORCED_INIT_SCORE = 1e30
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


def write_kv_cache(
    kv_cache: jax.Array,
    k: jax.Array,
    v: jax.Array,
    pages: jax.Array,
    pos: jax.Array,
) -> jax.Array:
    """Scatter K/V rows into a ``[pages, page_size, Kx2/pack, pack, D]`` cache."""
    rows = merge_kv(k.astype(kv_cache.dtype), v.astype(kv_cache.dtype))
    slot = pos % kv_cache.shape[1]
    return kv_cache.at[pages, slot].set(rows, mode="drop")


def write_index_cache(
    index_cache: jax.Array, index_k: jax.Array, pages: jax.Array, pos: jax.Array
) -> jax.Array:
    """Scatter index keys into a ``[pages, page_size/pack, pack, D]`` cache."""
    packing = index_cache.shape[2]
    slot = pos % (index_cache.shape[1] * packing)
    return index_cache.at[pages, slot // packing, slot % packing].set(
        index_k.astype(index_cache.dtype), mode="drop"
    )


def _score_request(
    window: int,
    index_q_padded: jax.Array,
    index_cache: jax.Array,
    page_row: jax.Array,
    q_start: jax.Array,
    num_pages: jax.Array,
    pages_per_chunk: int,
) -> jax.Array:
    """Block-max scores ``[window, H, P]`` of one request's query window."""
    num_heads = index_q_padded.shape[1]
    num_padded_pages = page_row.shape[0]
    _, rows_per_page, packing, head_dim = index_cache.shape
    block_size = rows_per_page * packing
    iq = jax.lax.dynamic_slice_in_dim(index_q_padded, q_start, window, axis=0)

    def body(chunk, scores):
        pages = jax.lax.dynamic_slice_in_dim(
            page_row, chunk * pages_per_chunk, pages_per_chunk
        )
        keys = index_cache[pages].reshape(pages_per_chunk * block_size, head_dim)
        s = jnp.einsum("whd,kd->whk", iq, keys, preferred_element_type=jnp.float32)
        s = s.reshape(window, num_heads, pages_per_chunk, block_size).max(axis=-1)
        return jax.lax.dynamic_update_slice_in_dim(
            scores, s, chunk * pages_per_chunk, axis=2
        )

    scores = jnp.full((window, num_heads, num_padded_pages), _NEG_INF, jnp.float32)
    num_chunks = (num_pages + pages_per_chunk - 1) // pages_per_chunk
    return jax.lax.fori_loop(0, num_chunks, body, scores)


def pages_per_chunk(window: int) -> int:
    """Pages one scoring iteration covers for a query window."""
    if window <= SCORE_WINDOW_TIERS[0]:
        return _SMALL_WINDOW_PAGES_PER_CHUNK
    return _PAGES_PER_CHUNK


def block_scores(
    index_q: jax.Array,
    index_cache: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    num_seqs: jax.Array,
) -> jax.Array:
    """Block-max index scores ``[T, H, P]`` (``P`` padded to the chunk).

    Requests are scored one at a time against their own pages, so each index
    key is read once per step regardless of how many query tokens use it.
    Blocks past a token's own block hold scores of keys it must not see;
    ``select_blocks`` masks them.
    """
    num_tokens, num_heads, _ = index_q.shape
    max_num_seqs = seq_lens.shape[0]
    pages_per_seq = block_tables.shape[0] // max_num_seqs
    block_size = index_cache.shape[1] * index_cache.shape[2]
    windows = sorted({w for w in SCORE_WINDOW_TIERS if w < num_tokens} | {num_tokens})
    chunks = {w: pages_per_chunk(w) for w in windows}
    # The chunk sizes divide each other, so this is a multiple of all of them.
    largest_chunk = max(chunks.values())
    num_padded_pages = -(-pages_per_seq // largest_chunk) * largest_chunk
    page_rows = jnp.pad(
        block_tables.reshape(max_num_seqs, pages_per_seq),
        ((0, 0), (0, num_padded_pages - pages_per_seq)),
    )
    index_q_padded = jnp.pad(index_q, ((0, num_tokens), (0, 0), (0, 0)))
    row_ids = jnp.arange(num_tokens, dtype=jnp.int32)

    def body(seq, scores):
        q_start = cu_q_lens[seq]
        q_len = cu_q_lens[seq + 1] - q_start
        num_pages = (seq_lens[seq] + block_size - 1) // block_size
        branches = [
            functools.partial(
                _merge_window,
                functools.partial(
                    _score_request,
                    window,
                    index_q_padded,
                    index_cache,
                    page_rows[seq],
                    q_start,
                    num_pages,
                    chunks[window],
                ),
                q_start,
                q_len,
                row_ids[:window],
            )
            for window in windows
        ]
        tier = jnp.searchsorted(jnp.asarray(windows), q_len, side="left")
        return jax.lax.switch(tier, branches, scores)

    scores = jnp.full(
        (2 * num_tokens, num_heads, num_padded_pages), _NEG_INF, jnp.float32
    )
    return jax.lax.fori_loop(0, num_seqs, body, scores)[:num_tokens]


def _merge_window(
    score_fn: Callable[[], jax.Array],
    q_start: jax.Array,
    q_len: jax.Array,
    rows: jax.Array,
    scores: jax.Array,
) -> jax.Array:
    window_scores = score_fn()
    old = jax.lax.dynamic_slice_in_dim(scores, q_start, rows.shape[0], axis=0)
    merged = jnp.where((rows < q_len)[:, None, None], window_scores, old)
    return jax.lax.dynamic_update_slice_in_dim(scores, merged, q_start, axis=0)


def select_blocks(
    scores: jax.Array,
    pos: jax.Array,
    *,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> jax.Array:
    """Selected logical blocks ``[T, H, topk]``, ascending, ``-1`` padded."""
    blocks = jnp.arange(scores.shape[-1], dtype=jnp.int32)
    num_visible = (pos // block_size + 1)[:, None, None]
    visible = blocks < num_visible
    s = jnp.where(visible, scores, _NEG_INF)
    s = jnp.where(
        visible & (blocks >= num_visible - local_blocks), _FORCED_LOCAL_SCORE, s
    )
    s = jnp.where(visible & (blocks < init_blocks), _FORCED_INIT_SCORE, s)
    top_scores, top_blocks = jax.lax.top_k(s, topk)
    top_blocks = jnp.where(top_scores > _NEG_INF, top_blocks, jnp.iinfo(jnp.int32).max)
    top_blocks = jnp.sort(top_blocks, axis=-1)
    return jnp.where(top_blocks == jnp.iinfo(jnp.int32).max, -1, top_blocks)


def sparse_attention(
    q: jax.Array,
    kv_cache: jax.Array,
    selected: jax.Array,
    block_tables_2d: jax.Array,
    seq_id: jax.Array,
    pos: jax.Array,
    num_real_tokens: jax.Array,
    *,
    sm_scale: float,
    rpa_func: Callable,
    max_rows: int = MAX_ROWS_PER_ATTENTION_CALL,
) -> jax.Array:
    """GQA attention of every token over its selected blocks."""
    num_tokens, _, head_dim = q.shape
    if selected.shape[1] != 1:
        raise NotImplementedError(
            "MSA attention shares one page table across a rank's KV heads, so it "
            f"needs one KV head per rank; got {selected.shape[1]}. Use a tensor "
            "parallel size of at least num_key_value_heads."
        )
    block_size = kv_cache.shape[1]
    selected = selected[:, 0]
    is_selected = selected >= 0
    pages = block_tables_2d[seq_id[:, None], jnp.maximum(selected, 0)]
    pages = jnp.where(is_selected, pages, 0)
    num_selected = is_selected.sum(axis=-1, dtype=jnp.int32)
    kv_lens = (num_selected - 1) * block_size + pos % block_size + 1
    dummy_kv = jnp.zeros((num_tokens, 1, head_dim), q.dtype)

    outputs = []
    for start in range(0, num_tokens, max_rows):
        rows = min(max_rows, num_tokens - start)
        real = jnp.clip(num_real_tokens - start, 0, rows).astype(jnp.int32)
        cu_q_lens = jnp.minimum(jnp.arange(rows + 1, dtype=jnp.int32), real)
        out, _ = rpa_func(
            q[start : start + rows],
            dummy_kv[start : start + rows],
            dummy_kv[start : start + rows],
            kv_cache,
            jnp.where(jnp.arange(rows) < real, kv_lens[start : start + rows], 0),
            pages[start : start + rows].reshape(-1),
            cu_q_lens,
            jnp.stack([real, real, real]),
            sm_scale=sm_scale,
            skip_kv_update=True,
        )
        outputs.append(out)
    out = jnp.concatenate(outputs, axis=0)
    real_rows = jnp.arange(num_tokens) < num_real_tokens
    return jnp.where(real_rows[:, None, None], out, 0).astype(q.dtype)


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
    rpa_func: Callable,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Write this step's K/V and index keys, select blocks, attend.

    Shapes: ``q [T, Hq, D]``, ``k``/``v [T, Hkv, D]``, ``index_q [T, Hkv, D]``,
    ``index_k [T, D]``. Returns ``(kv_cache, index_cache, out [T, Hq, D])``.
    """
    num_tokens = q.shape[0]
    block_size = kv_cache.shape[1]
    max_num_seqs = seq_lens.shape[0]
    block_tables_2d = block_tables.reshape(max_num_seqs, -1)
    seq_id, pos, valid = token_positions(num_tokens, seq_lens, cu_q_lens)
    pages = _token_pages(block_tables_2d, seq_id, pos, valid, block_size)

    kv_cache = write_kv_cache(kv_cache, k, v, pages, pos)
    index_cache = write_index_cache(index_cache, index_k, pages, pos)

    scores = block_scores(
        index_q, index_cache, seq_lens, block_tables, cu_q_lens, distribution[2]
    )
    selected = select_blocks(
        scores,
        pos,
        block_size=block_size,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )
    out = sparse_attention(
        q,
        kv_cache,
        selected,
        block_tables_2d,
        seq_id,
        pos,
        cu_q_lens[-1],
        sm_scale=sm_scale,
        rpa_func=rpa_func,
    )
    return kv_cache, index_cache, out
