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

from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp

from vllm_torchtpu.models.minimax_m3.kernels import (
    msa_attention,
    msa_fp4,
    msa_indexer,
    msa_quantize,
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
    start = msa_indexer.lookup(cu_q_lens, seq_id)
    q_len = msa_indexer.lookup(cu_q_lens, seq_id + 1) - start
    pos = msa_indexer.lookup(seq_lens, seq_id) - q_len + (tok - start)
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


def write_caches(
    targets: Sequence[msa_write.Rows | msa_write.SlotValues],
    pages: jax.Array,
    pos: jax.Array,
    *,
    max_num_seqs: int | None = None,
    interpret: bool = False,
) -> list[jax.Array]:
    """Write one row (or slot values) per token into each packed cache of
    `targets` in one `msa_write.write_runs` call; only the pages that
    receive rows are rewritten. A request's tokens must be consecutive with
    consecutive positions. `max_num_seqs` bounds the requests in the batch
    (default: one per token)."""
    num_tokens = pages.shape[0]
    block_size = page_size(targets[0].cache)
    return msa_write.write_runs(
        cache_targets(targets),
        pages,
        pos % block_size,
        max_runs=write_runs_bound(num_tokens, max_num_seqs, block_size),
        interpret=interpret,
    )


def write_runs_bound(num_tokens: int, max_num_seqs: int | None, block_size: int) -> int:
    """`msa_write.max_write_runs` of `write_caches` (`max_num_seqs` default:
    one request per token)."""
    return msa_write.max_write_runs(
        num_tokens, num_tokens if max_num_seqs is None else max_num_seqs, block_size
    )


def cache_targets(
    targets: Sequence[msa_write.Rows | msa_write.SlotValues],
) -> list[msa_write.Rows | msa_write.SlotValues]:
    """`write_caches`' targets with 8-bit rows saturated to their cache dtype."""
    return [
        t
        if isinstance(t, msa_write.SlotValues) or msa_fp4.is_fp4(t.cache)
        else t._replace(rows=_cache_rows(t.cache, t.rows))
        for t in targets
    ]


def write_rows(
    cache: jax.Array,
    rows: jax.Array,
    pages: jax.Array,
    pos: jax.Array,
    *,
    max_num_seqs: int | None = None,
    scales: jax.Array | None = None,
    dim_major: bool = False,
    interpret: bool = False,
) -> jax.Array:
    """`write_caches` of one cache (`msa_write.Rows`: `scales` only for a
    4-bit cache with folded scales, `dim_major` for the index cache)."""
    bits = 4 if msa_fp4.is_fp4(cache) else None
    return write_caches(
        [msa_write.Rows(cache, rows, bits, scales, dim_major)],
        pages,
        pos,
        max_num_seqs=max_num_seqs,
        interpret=interpret,
    )[0]


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
    (``[T, D]``) into 4-bit caches and their scale page (`msa_fp4`), in one
    `write_caches` call.

    Scale-page rows ``2h`` / ``2h + 1`` hold the K / V scales of head ``h``;
    the index-key scales are folded into the index pages' last word row
    (`msa_fp4.scales_in_page`).
    """
    kv_cache, index_cache, scale_cache = write_caches(
        fp4_targets(
            kv_cache, index_cache, scale_cache, k, v, index_k, interpret=interpret
        ),
        pages,
        pos,
        max_num_seqs=max_num_seqs,
        interpret=interpret,
    )
    return kv_cache, index_cache, scale_cache


class QuantizedKV(NamedTuple):
    """A step's K/V already rotated and quantized for a 4-bit cache, as
    `msa_quantize.quantize_kv_rows` returns them."""

    rows: jax.Array  # uint8 nibbles [T, 2 * G * D]: [K_0 | V_0 | ... ]
    scales: jax.Array  # f32 [T, 2 * G]: [k_0, v_0, ...]


def fp4_targets(
    kv_cache: jax.Array,
    index_cache: jax.Array,
    scale_cache: jax.Array,
    k: jax.Array | QuantizedKV,
    v: jax.Array | None,
    index_k: jax.Array,
    *,
    interpret: bool = False,
) -> list[msa_write.Rows | msa_write.SlotValues]:
    """`write_fp4`'s rotated, quantized write targets: K/V nibbles, index-key
    nibbles with their folded scales, then the K/V scales. `k` may come
    quantized (`QuantizedKV`, `v` None)."""
    lanes = None
    kernel = (
        not isinstance(k, QuantizedKV)
        and k.shape[0] >= msa_quantize.MIN_TOKENS
        and k.dtype == jnp.bfloat16
    )
    if isinstance(k, QuantizedKV):
        kv_rows, scales = k
    elif kernel and index_k.dtype == jnp.bfloat16:
        # One pass for the index keys too, which also lays out the write's
        # lanes (the index scales are the only folded ones: the last lane).
        kv_rows, scales, i_nib, i_scale, lanes = msa_quantize.quantize_kv_rows(
            k, v, index_k, interpret=interpret
        )
    elif kernel:
        kv_rows, scales = msa_quantize.quantize_kv_rows(k, v, interpret=interpret)
    else:
        k_nib, k_scale = msa_fp4.quantize(msa_fp4.rotate_keys(k))
        v_nib, v_scale = msa_fp4.quantize(msa_fp4.rotate_keys(v))
        kv_rows = jnp.concatenate([k_nib, v_nib], axis=-1).reshape(k.shape[0], -1)
        scales = jnp.stack([k_scale, v_scale], axis=-1).reshape(k.shape[0], -1)
    if lanes is None:
        i_nib, i_scale = msa_fp4.quantize(msa_fp4.rotate_keys(index_k))
    return [
        msa_write.Rows(kv_cache, kv_rows, 4),
        msa_write.Rows(index_cache, i_nib, 4, i_scale, dim_major=True),
        msa_write.SlotValues(scale_cache, scales, lanes),
    ]


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
# Pages per large-tile step in `_block_scores` (the token-order path).
_CONDITIONAL_PAGES_PER_STEP = 32


def block_score_options(index_heads: int = 1, wide: int = 1) -> dict[str, int]:
    """The tile sizes the MSA forward scores blocks with, for query rows that
    fold `index_heads` heads per token. A transposed (folded 4-bit or fp8)
    decode tile holds one request's rows: its scoring costs per tile row, and 32-row
    tiles for one head's 8 rows ran the c48 / c64 DCP 2 decode indexer 7%
    slower, bit-identical results (bench j-74f8b992).

    `wide` (`_wide_decode_scores`) scores that many times 128 pages per
    small-class step: fewer grid steps, ~0.3-0.4 us each."""
    return dict(
        block_q=msa_indexer.DEFAULT_BLOCK_Q,
        large_block_q=msa_indexer.DEFAULT_LARGE_BLOCK_Q,
        pages_per_step=msa_indexer.DEFAULT_PAGES_PER_STEP,
        small_pages_per_step=wide * msa_indexer.DEFAULT_SMALL_PAGES_PER_STEP,
        transposed_tile_q=min(_DECODE_ROWS * index_heads, msa_indexer.DEFAULT_BLOCK_Q),
    )


def _wide_decode_scores(index_q, index_cache, seq_lens, block_tables, index_heads):
    """How many 128-page groups a step's small-class block score steps take:
    2 (256 pages) for a decode step
    (`msa_indexer.is_prefill_step`) of more query rows per index head than
    one request's `_DECODE_ROWS` (e.g. a verify step of 2+ requests), whose
    requests may hold more than 256 pages (a shard of up to 256 pages fits
    one group either way: c64 DCP 8 decode block scores 62.5 -> 69.4 us per
    layer at 256, twins j-545009b2, j-1ca3be70), over 8-bit or 4-bit index
    pages (a 256-page ring of bf16 pages is twice the VMEM). Standalone
    decode calls (j-83e37332): int4 folded 23 x 450 pages 94.7 -> 77.9 us,
    36 x 366 pages 110.6 -> 105.7, fp8 5 x 450 pages 24.0 -> 21.5; one
    request of 313 pages 8.7 -> 9.1 (so a single request keeps 128).

    4 (512 pages) over 4-bit pages when a request may hold 1024 or more:
    c48's 22-request decode (~430 pages each) took 83.3 -> 77.6 us per
    fused decode-candidates call (anvil, standalone); fp8 c16 steps of 3 / 6
    requests gained nothing (25.7 -> 25.9, 34.2 -> 33.7 us), and c64's
    DCP 4 requests hold at most 512 pages (~192 served), where 512-page
    groups idled most of a group (+22 % on its decode indexer)."""
    rows = index_q.shape[0]
    if rows <= _DECODE_ROWS * index_heads or msa_indexer.is_prefill_step(
        rows, seq_lens.shape[0]
    ):
        return 1
    return wide_groups(index_cache, block_tables.shape[0] // seq_lens.shape[0])


def wide_groups(index_cache: jax.Array, pages_per_seq: int) -> int:
    """`_wide_decode_scores`' group count for decode-sized requests of up to
    `pages_per_seq` pages, whatever the step."""
    if (
        pages_per_seq <= 2 * msa_indexer.DEFAULT_SMALL_PAGES_PER_STEP
        or jnp.dtype(index_cache.dtype).itemsize != 1
    ):
        return 1
    if (
        msa_fp4.is_fp4(index_cache)
        and pages_per_seq >= 8 * msa_indexer.DEFAULT_SMALL_PAGES_PER_STEP
    ):
        return 4
    return 2


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
    row_index=None,
):
    """`index_q` must already be rotated for a 4-bit cache; its rows fold
    `index_heads` heads per token (`block_score_options`)."""
    # 512-page groups (`_wide_decode_scores` 4) were measured in the fused
    # decode kernel (`msa_dcp.decode_candidates`) only; this path keeps 256.
    wide = _wide_decode_scores(
        index_q, index_cache, seq_lens, block_tables, index_heads
    )
    options = block_score_options(index_heads, min(wide, 2))
    # Decode steps take the large class in a conditional, whose cost grows
    # with the code it holds (c48 b22 conditional 1.3 -> 7.1 us per 5-layer
    # twin step with 64-page steps, 34 with 128): its tiles keep 32 pages.
    options["pages_per_step"] = _CONDITIONAL_PAGES_PER_STEP
    return msa_indexer.block_scores(
        index_q,
        index_cache,
        seq_lens,
        block_tables,
        cu_q_lens,
        num_seqs,
        **options,
        scale_cache=scale_cache,
        scale_row=scale_row,
        row_index=row_index,
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
        kv_cache, index_cache = write_caches(
            [
                msa_write.Rows(kv_cache, jnp.concatenate([k, v], axis=-1)[:, 0]),
                msa_write.Rows(index_cache, index_k, dim_major=True),
            ],
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
    """The index keys of this rank's tokens in its index-cache shard: not
    supported, as the indexer reads dim-major pages of whole blocks
    (`msa_indexer.check_square_pages`) and a shard page holds 1 / shards of
    one."""
    raise NotImplementedError(
        "index-cache shards: the indexer reads dim-major pages of whole blocks"
    )


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
    rows ``[H * cu[s], H * cu[s + 1])``), so one `block_scores` call reads
    the shard's pages once per query tile. Row ``t * H + h`` is head ``h`` of
    token ``t``, ``index_q`` as laid out: `block_scores` is blind to the row
    order within a request (it masks no key by position), and a (head,
    token) order took a scatter (~5 us per layer in the c64 decode twin,
    j-9a9fccc2).
    """
    num_tokens, num_heads, head_dim = index_q.shape
    folded = index_q.reshape(num_tokens * num_heads, head_dim)
    rows = jnp.arange(num_tokens * num_heads, dtype=jnp.int32)
    rows = rows.reshape(num_tokens, num_heads).T
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
        row_index=rows.reshape(-1),
    )
    return scores[:, 0].reshape(*rows.shape, -1)


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
