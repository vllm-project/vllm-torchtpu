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
    msa_fp4,
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
    seq_id = jnp.searchsorted(
        cu_q_lens[1:], tok, side="right", method="compare_all"
    ).astype(jnp.int32)
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
    if msa_fp4.is_fp4(cache):
        return msa_fp4.page_tokens(cache)
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
    scales: jax.Array | None = None,
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
    fp4 = msa_fp4.is_fp4(cache)
    return msa_write.write_rows(
        cache,
        rows if fp4 else _cache_rows(cache, rows),
        pages,
        pos % page_size(cache),
        max_runs=max_runs,
        bits=4 if fp4 else None,
        interpret=interpret,
        **({} if scales is None else {"scales": scales}),
    )


def write_fp4(
    kv_cache: jax.Array,
    index_cache: jax.Array,
    scale_cache: jax.Array,
    k: jax.Array,
    v: jax.Array,
    index_k: jax.Array,
    pages: jax.Array,
    pos: jax.Array,
    *,
    max_num_seqs: int | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Rotate, quantize and write one step's K/V (``[T, G, D]``) and index keys
    (``[T, D]``) into 4-bit caches and their scale page (`msa_fp4`), all three
    in one kernel (`msa_write.write_fp4_caches`) at the step's `pages` and
    `pos` (as `write_rows`).

    Scale-page rows ``2h`` / ``2h + 1`` hold the K / V scales of head ``h``;
    the index-key scales are folded into the index pages' last word row
    (`msa_fp4.scales_in_page`).
    """
    num_tokens, num_heads, _ = k.shape
    k_nib, k_scale = msa_fp4.quantize(msa_fp4.rotate_keys(k))
    v_nib, v_scale = msa_fp4.quantize(msa_fp4.rotate_keys(v))
    i_nib, i_scale = msa_fp4.quantize(msa_fp4.rotate_keys(index_k))
    scales = jnp.stack([k_scale, v_scale], axis=-1).reshape(num_tokens, -1)  # [T, 2G]
    block_size = page_size(kv_cache)
    return msa_write.write_fp4_caches(
        kv_cache,
        index_cache,
        scale_cache,
        jnp.concatenate([k_nib, v_nib], axis=-1).reshape(num_tokens, -1),
        i_nib,
        i_scale,
        scales,
        pages,
        pos % block_size,
        max_runs=_max_runs(num_tokens, max_num_seqs, block_size),
        interpret=interpret,
    )


def write_kv_and_index(
    kv_cache: jax.Array,
    index_cache: jax.Array,
    kv_rows: jax.Array,
    index_rows: jax.Array,
    pages: jax.Array,
    pos: jax.Array,
    *,
    max_num_seqs: int | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """`write_rows` of the K/V and the index rows in one kernel
    (`msa_write.write_rows_many`)."""
    block_size = page_size(kv_cache)
    fp4 = msa_fp4.is_fp4(kv_cache)
    if fp4 != msa_fp4.is_fp4(index_cache):  # one bit width per kernel
        write = functools.partial(
            write_rows, pages=pages, pos=pos, max_num_seqs=max_num_seqs, interpret=interpret
        )
        return write(kv_cache, kv_rows), write(index_cache, index_rows)
    return msa_write.write_rows_many(
        (kv_cache, index_cache),
        (kv_rows, index_rows)
        if fp4
        else (_cache_rows(kv_cache, kv_rows), _cache_rows(index_cache, index_rows)),
        pages,
        pos % block_size,
        max_runs=_max_runs(kv_rows.shape[0], max_num_seqs, block_size),
        bits=4 if fp4 else None,
        interpret=interpret,
    )


def _max_runs(num_tokens: int, max_num_seqs: int | None, block_size: int) -> int:
    return msa_write.max_write_runs(
        num_tokens, num_tokens if max_num_seqs is None else max_num_seqs, block_size
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
    scale_cache: jax.Array | None = None,
    interpret: bool = False,
) -> jax.Array:
    """GQA attention of every token over its selected blocks.

    With a 4-bit `kv_cache`, `q` must be rotated (`msa_fp4.rotate_queries`)
    and the output is in the rotated basis (`msa_fp4.unrotate_output`)."""
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
        block_q=(
            msa_attention.SMALL_BATCH_BLOCK_Q
            if q.shape[0] <= msa_attention.SMALL_BATCH_TOKENS
            else msa_attention.DEFAULT_BLOCK_Q
        ),
        pages_per_step=msa_attention.DEFAULT_PAGES_PER_STEP,
        page_table=block_tables_2d.reshape(-1),
        interpret=interpret,
        **_fp4_attention_kwargs(scale_cache),
    )


def _fp4_attention_kwargs(scale_cache: jax.Array | None) -> dict:
    """Scale-page inputs of the attention kernels for a 4-bit KV cache: one KV
    head per call, its K / V scales in scale-page rows 0 / 1."""
    if scale_cache is None:
        return {}
    return dict(scale_cache=scale_cache, k_scale_row=0, v_scale_row=1)


def _select_and_attend(
    q: jax.Array,
    kv_cache: jax.Array,
    scores: jax.Array,
    block_tables_2d: jax.Array,
    seq_id: jax.Array,
    pos: jax.Array,
    pages: jax.Array,
    valid: jax.Array,
    *,
    sm_scale: float,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    interpret: bool,
    scale_cache: jax.Array | None = None,
) -> jax.Array:
    selected = msa_select.select_blocks(
        scores,
        pos,
        block_size=page_size(kv_cache),
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        interpret=interpret,
    )
    return sparse_attention(
        q,
        kv_cache,
        selected,
        block_tables_2d,
        seq_id,
        pos,
        pages,
        valid,
        sm_scale=sm_scale,
        scale_cache=scale_cache,
        interpret=interpret,
    )


# Query rows of a decode request per index head: 1 + 7 speculative tokens.
_DECODE_ROWS = 8


def block_score_options(index_heads: int = 1) -> dict[str, int]:
    """The tile sizes the MSA forward scores blocks with, for query rows that
    fold `index_heads` heads per token. A folded 4-bit decode tile holds one
    request's rows: its transposed scoring costs per tile row, and 32-row
    tiles for one head's 8 rows ran the c48 / c64 DCP 2 decode indexer 7%
    slower, bit-identical results (bench j-74f8b992)."""
    return dict(
        block_q=msa_indexer.DEFAULT_BLOCK_Q,
        large_block_q=msa_indexer.DEFAULT_LARGE_BLOCK_Q,
        pages_per_step=msa_indexer.DEFAULT_PAGES_PER_STEP,
        small_pages_per_step=msa_indexer.DEFAULT_SMALL_PAGES_PER_STEP,
        folded_tile_q=min(_DECODE_ROWS * index_heads, msa_indexer.DEFAULT_BLOCK_Q),
    )


def _block_scores(
    index_q,
    index_cache,
    seq_lens,
    block_tables,
    cu_q_lens,
    num_seqs,
    interpret,
    scale_cache=None,
    scale_row=None,
    index_heads=1,
):
    """`index_q` must already be rotated for a 4-bit cache; its rows fold
    `index_heads` heads per token (`block_score_options`)."""
    return msa_indexer.block_scores(
        index_q,
        index_cache,
        seq_lens,
        block_tables,
        cu_q_lens,
        num_seqs,
        **block_score_options(index_heads),
        scale_cache=scale_cache,
        scale_row=scale_row,
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
    scale_cache: jax.Array | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, ...]:
    """Write this step's K/V and index keys, select blocks, attend.

    Shapes: ``q [T, Hq, D]``, ``k``/``v [T, 1, D]``, ``index_q [T, 1, D]``,
    ``index_k [T, D]``; caches in the packed layout ``[pages, S / p, p, W]``
    with rows ``[K | V]`` (``W = 2D``) and index keys (``W = D``). Returns
    ``(kv_cache, index_cache, out [T, Hq, D])``. With 4-bit caches
    (`msa_fp4`) pass their `scale_cache`; it is returned updated after
    `index_cache`.
    """
    num_tokens = q.shape[0]
    block_size = page_size(kv_cache)
    max_num_seqs = seq_lens.shape[0]
    block_tables_2d = block_tables.reshape(max_num_seqs, -1)
    seq_id, pos, valid = token_positions(num_tokens, seq_lens, cu_q_lens)
    pages = _token_pages(block_tables_2d, seq_id, pos, valid, block_size)

    fp4 = scale_cache is not None
    scale_row = None
    if fp4:
        kv_cache, index_cache, scale_cache = write_fp4(
            kv_cache,
            index_cache,
            scale_cache,
            k,
            v,
            index_k,
            pages,
            pos,
            max_num_seqs=max_num_seqs,
            interpret=interpret,
        )
        scale_row = msa_fp4.index_scale_row(k.shape[1])
        q = msa_fp4.rotate_queries(q)
        index_q = msa_fp4.rotate_queries(index_q)
    else:
        kv_cache, index_cache = write_kv_and_index(
            kv_cache,
            index_cache,
            jnp.concatenate([k, v], axis=-1)[:, 0],
            index_k,
            pages,
            pos,
            max_num_seqs=max_num_seqs,
            interpret=interpret,
        )

    scores = _block_scores(
        index_q,
        index_cache,
        seq_lens,
        block_tables,
        cu_q_lens,
        distribution[2],
        interpret,
        scale_cache,
        scale_row,
    )
    out = _select_and_attend(
        q,
        kv_cache,
        scores,
        block_tables_2d,
        seq_id,
        pos,
        pages,
        valid,
        sm_scale=sm_scale,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        interpret=interpret,
        scale_cache=scale_cache,
    )
    if fp4:
        return kv_cache, index_cache, scale_cache, msa_fp4.unrotate_output(out)
    return kv_cache, index_cache, out


# Index-cache sharding over the tensor-parallel group. Rank r of S keeps the
# index keys of the tokens at positions p with p % S == r, so its index cache
# holds block_size / S tokens per page. A block's score is a max over its
# tokens, so the max over ranks of each rank's per-page max is the exact
# block score. Every rank scores all S index heads on its keys; one
# all-to-all (outside JAX) brings each rank the S partials of its own head.


def write_index_shard(
    index_cache: jax.Array,
    index_k: jax.Array,
    pages: jax.Array,
    pos: jax.Array,
    valid: jax.Array,
    *,
    rank: int,
    shards: int,
    max_num_seqs: int,
    interpret: bool = False,
) -> jax.Array:
    """Write the index keys of this rank's tokens into its shard cache.

    The token at position ``p`` (``p % shards == rank``) lands at slot
    ``(p % block_size) // shards`` of its page. This rank's tokens are first
    compacted, so a request's rows stay consecutive with consecutive slots,
    as `msa_write.write_rows` needs.
    """
    num_tokens = index_k.shape[0]
    shard_page = page_size(index_cache)
    block_size = shard_page * shards
    # A request of q tokens owns at most ceil(q / shards) of them.
    size = min(num_tokens, num_tokens // shards + max_num_seqs)
    (idx,) = jnp.nonzero(
        valid & (pos % shards == rank), size=size, fill_value=num_tokens
    )
    take = jnp.minimum(idx, num_tokens - 1)
    own_pages = jnp.where(idx < num_tokens, pages[take], _DROPPED_PAGE)
    return msa_write.write_rows(
        index_cache,
        _cache_rows(index_cache, index_k[take]),
        own_pages,
        (pos[take] % block_size) // shards,
        max_runs=msa_write.max_write_runs(size, max_num_seqs, shard_page),
        interpret=interpret,
    )


def fold_index_heads(
    index_q: jax.Array, seq_lens: jax.Array, cu_q_lens: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """``index_q [T, H, D]`` folded into query rows ``[H * T, D]`` (request
    ``s`` owns rows ``[H * cu[s], H * cu[s + 1])`` in (head, token) order),
    and each head's and token's row ``[H, T]``."""
    num_tokens, num_heads, head_dim = index_q.shape
    seq_id, _, valid = token_positions(num_tokens, seq_lens, cu_q_lens)
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    q_len = cu_q_lens[seq_id + 1] - cu_q_lens[seq_id]
    head = jnp.arange(num_heads, dtype=jnp.int32)[:, None]
    num_real = cu_q_lens[-1]
    rows = jnp.where(
        valid[None],
        num_heads * cu_q_lens[seq_id][None]
        + head * q_len[None]
        + (tok - cu_q_lens[seq_id])[None],
        # Padding tokens after the last request, one unique row each.
        num_heads * num_real + head * (num_tokens - num_real) + (tok - num_real)[None],
    )
    folded = (
        jnp.zeros((num_heads * num_tokens, head_dim), index_q.dtype)
        .at[rows.reshape(-1)]
        .set(index_q.transpose(1, 0, 2).reshape(-1, head_dim))
    )
    return folded, rows


def partial_block_scores(
    index_q: jax.Array,
    index_cache: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    num_seqs: jax.Array,
    *,
    shards: int,
    interpret: bool = False,
    scale_cache: jax.Array | None = None,
    scale_row: int | None = None,
) -> jax.Array:
    """``[H, T, P]`` per-page maxima of all ``H`` index heads over this
    rank's keys (``index_q [T, H, D]``; the cache holds ``1 / shards`` of
    each page's tokens). A 4-bit cache (unsharded) takes its scale cache and
    rotated `index_q`.

    The heads are folded into each request's query rows (request ``s`` owns
    rows ``[H * cu[s], H * cu[s + 1])`` in (head, token) order), so one
    `block_scores` call reads the shard's pages once per query tile.
    """
    num_heads = index_q.shape[1]
    folded, rows = fold_index_heads(index_q, seq_lens, cu_q_lens)
    # block_scores counts a request's pages from its length and the cache's
    # tokens per page; a shard page holds block_size / shards tokens, so give
    # it ceil(len / shards) tokens: ceil(ceil(L / S) / (B / S)) = ceil(L / B).
    scores = _block_scores(
        folded[:, None],
        index_cache,
        -(-seq_lens // shards),
        block_tables,
        num_heads * cu_q_lens,
        num_seqs,
        interpret,
        scale_cache,
        scale_row,
        index_heads=num_heads,
    )
    return scores[rows, 0]


def msa_write_and_score_sharded(
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
    rank: int,
    shards: int,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """First half of the sharded forward: write K/V and this rank's index
    keys, score all heads on them. ``index_q [T, shards, D]`` holds every
    head (all-gathered). Returns ``(kv_cache, index_cache, partial
    [shards, T, P])``: rank ``r``'s block scores are the max over ranks of
    their ``partial[r]``."""
    num_tokens = k.shape[0]
    block_size = page_size(kv_cache)
    max_num_seqs = seq_lens.shape[0]
    block_tables_2d = block_tables.reshape(max_num_seqs, -1)
    seq_id, pos, valid = token_positions(num_tokens, seq_lens, cu_q_lens)
    pages = _token_pages(block_tables_2d, seq_id, pos, valid, block_size)
    kv_cache = write_rows(
        kv_cache,
        jnp.concatenate([k, v], axis=-1)[:, 0],
        pages,
        pos,
        max_num_seqs=max_num_seqs,
        interpret=interpret,
    )
    index_cache = write_index_shard(
        index_cache,
        index_k,
        pages,
        pos,
        valid,
        rank=rank,
        shards=shards,
        max_num_seqs=max_num_seqs,
        interpret=interpret,
    )
    partial = partial_block_scores(
        index_q,
        index_cache,
        seq_lens,
        block_tables,
        cu_q_lens,
        distribution[2],
        shards=shards,
        interpret=interpret,
    )
    return kv_cache, index_cache, partial


def msa_attend_sharded(
    q: jax.Array,
    kv_cache: jax.Array,
    scores: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    *,
    sm_scale: float,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    interpret: bool = False,
) -> jax.Array:
    """Second half of the sharded forward: select over the combined
    ``scores [T, 1, P]`` of this rank's head and attend."""
    num_tokens = q.shape[0]
    block_size = page_size(kv_cache)
    max_num_seqs = seq_lens.shape[0]
    block_tables_2d = block_tables.reshape(max_num_seqs, -1)
    seq_id, pos, valid = token_positions(num_tokens, seq_lens, cu_q_lens)
    pages = _token_pages(block_tables_2d, seq_id, pos, valid, block_size)
    return _select_and_attend(
        q,
        kv_cache,
        scores,
        block_tables_2d,
        seq_id,
        pos,
        pages,
        valid,
        sm_scale=sm_scale,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        interpret=interpret,
    )
