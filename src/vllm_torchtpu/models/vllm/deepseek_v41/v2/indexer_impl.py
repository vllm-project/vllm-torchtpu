# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged MQA scoring, index selection and torch/JAX operation registration."""

import functools
import math
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import torch
from jax.experimental import xla_metadata

from vllm_torchtpu.kernels.deepseek_v41 import (
    index_key_layout,
    indexer,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection import (
    decode as decode_index_selection,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection import (
    packed_scores,
    prefill_scores,
    publication,
    streaming_topk,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection import (
    tuned_block_sizes as prefill_tuning,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.cache import (
    PAGE_RECORDS,
    PAGE_ROWS,
    PackedFP8Layout,
    read_fp8_records,
    write_fp8_records,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.candidates import (
    PreparedCandidates,
    prepare_candidates,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.dense_logits import (
    fp8_paged_mqa_logits,
    supports_candidate_publication,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.sparse_logits import (
    candidate_windows,
    fp8_gathered_mqa_logits,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.tuned import (
    PACKED_FP8_MQA_CONFIG,
    indexer_query_rows,
    paged_mqa_key_tile,
    topk_reconstruction_threshold,
)

from .cache import (
    ROW_WORDS,
    _has_hi_bank,
    gather_cache_page_words,
    scatter_index_keys,
    split_halves,
)

if TYPE_CHECKING:
    from .indexer_cache import PackedFP8IndexCacheRows


# Decode-shaped batches (few queries per request row) skip query tiling in
# `paged_indexer_jax` while their full [queries, context] FP32 score buffer
# stays within this many elements (512 MiB), keeping grouped scoring intact.
_INDEX_UNTILED_SCORE_ELEMENTS = 1 << 27


def _compact_selected(records: jax.Array) -> jax.Array:
    """Sort valid (>= 0) indices ascending and move -1 to the tail."""
    limit = jnp.iinfo(jnp.int32).max
    records = jnp.sort(jnp.where(records >= 0, records, limit), axis=-1)
    return jnp.where(records == limit, -1, records)


def _prefix_count(mask: jax.Array, block: int = 128) -> jax.Array:
    """Inclusive ``cumsum(mask, -1)`` as exact blockwise 0/1 matmuls."""
    rows, width = mask.shape
    blocks = -(-width // block)
    x = jnp.pad(mask, ((0, 0), (0, blocks * block - width))).astype(jnp.bfloat16)
    x = x.reshape(rows, blocks, block)
    upper = jnp.triu(jnp.ones((block, block), jnp.bfloat16))
    inner = jnp.einsum("nbi,ij->nbj", x, upper, preferred_element_type=jnp.float32)
    totals = inner[..., -1].astype(jnp.bfloat16)  # at most `block`: exact
    strict = jnp.triu(jnp.ones((blocks, blocks), jnp.bfloat16), 1)
    before = jnp.einsum("nb,bc->nc", totals, strict, preferred_element_type=jnp.float32)
    return (inner + before[..., None]).reshape(rows, -1)[:, :width].astype(jnp.int32)


def _threshold_top_k(scores: jax.Array, k: int) -> jax.Array:
    """Ascending indices of ``jax.lax.top_k(scores, k)``, no sort.

    Finds the k-th largest value by bisecting an order-preserving integer key,
    breaks ties at it by lowest index (as top_k does), then compacts.
    """
    rows = scores.shape[0]
    # No -0.0 -> +0.0 folding: top_k ranks +0.0 above -0.0 (TPU and CPU), as
    # does the order-preserving key below.
    bits = jax.lax.bitcast_convert_type(scores, jnp.int32)
    key = jnp.where(bits < 0, bits ^ jnp.int32(0x7FFFFFFF), bits) ^ jnp.int32(-(2**31))
    key = jax.lax.bitcast_convert_type(key, jnp.uint32)

    def bisect(bit, threshold):
        candidate = threshold | (jnp.uint32(1) << (31 - bit).astype(jnp.uint32))
        enough = (key >= candidate[:, None]).sum(-1) >= k
        return jnp.where(enough, candidate, threshold)

    threshold = jax.lax.fori_loop(0, 32, bisect, jnp.zeros((rows,), jnp.uint32))
    above = key > threshold[:, None]
    tied = key == threshold[:, None]
    remaining = k - above.sum(-1)
    chosen = above | (tied & (_prefix_count(tied) <= remaining[:, None]))
    ranks = _prefix_count(chosen)
    targets = jnp.broadcast_to(jnp.arange(1, k + 1, dtype=jnp.int32), (rows, k))
    return jax.vmap(functools.partial(jnp.searchsorted, side="left"))(
        ranks, targets
    ).astype(jnp.int32)


def _sparsecore_top_k(scores: jax.Array, k: int, lengths: jax.Array) -> jax.Array:
    """V4's SparseCore top-k (`kernels/deepseek_v4/sparsecore_topk.py`).

    Same set as ``jax.lax.top_k`` over the finite scores (ties at the k-th
    value by lowest index), unordered, ``-1`` suffix-padded; ``-inf`` and
    columns ``>= lengths`` are never selected.
    """
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (
        sparsecore_topk,
        supports_topk_width,
    )
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    if not supports_topk_width(k):
        masked = jnp.where(
            jnp.arange(scores.shape[1]) < lengths[:, None], scores, -jnp.inf
        )
        values, ids = jax.lax.top_k(masked, k)
        return jnp.where(values > -jnp.inf, ids, -1)
    select = functools.partial(sparsecore_topk, k=k)
    try:
        mesh = get_vllm_model_wrapper_context().mesh
    except Exception:  # Standalone (tests): one device, no mesh.
        return select(scores, row_lengths=lengths)
    # Mosaic calls need manual partitioning; every device selects locally.
    return jax.shard_map(
        lambda s, n: select(s, row_lengths=n),
        mesh=mesh,
        in_specs=(P(), P()),
        out_specs=P(),
        check_vma=False,
    )(scores, lengths)


def _visible_top_k(
    scores: jax.Array,
    k: int,
    bound: jax.Array,
    lengths: jax.Array | None = None,
    unordered: bool = False,
) -> jax.Array:
    """Ascending ``jax.lax.top_k(scores, k)`` indices when columns ``>= bound``
    are ``-inf``.

    Selects only within a static power-of-two bucket of columns covering
    ``bound`` (the full width otherwise). Exact: buckets hold at least ``k``
    columns, so ``-inf`` ties still resolve to the lowest indices. Large
    selections use `_threshold_top_k`; small ones (decode) sort.

    ``unordered`` callers (which re-sort and drop ``-1``) accept the finite
    top-k set in any order, ``-1`` padded instead of ``-inf`` columns; large
    selections then run on SparseCore over the full width, bounded by
    ``lengths`` instead of buckets (no SparseCore op inside `lax.switch`).
    Small decode query batches stay unpadded to avoid extra subcore waves.
    """
    width = scores.shape[1]
    if (
        unordered
        and lengths is not None
        and width >= 2048
        and width % 16 == 0
        and k % 16 == 0
    ):
        return _sparsecore_top_k(scores, k, jnp.minimum(lengths, width))

    def select(columns, scores):
        scores = scores[:, :columns]
        if scores.shape[0] >= 16 and columns >= 2048:
            return _threshold_top_k(scores, k)
        return jnp.sort(jax.lax.top_k(scores, k)[1], axis=-1)

    buckets = []
    bucket = max(k, 1024)
    while bucket < width:
        buckets.append(bucket)
        bucket *= 2
    buckets.append(width)
    if len(buckets) == 1:
        return select(width, scores)
    branch = jnp.searchsorted(jnp.asarray(buckets[:-1]), bound, side="left")
    return jax.lax.switch(
        branch, [functools.partial(select, b) for b in buckets], scores
    )


def _publish_candidate_blocks(
    scores, lengths, count, block_size, unordered, *, block_scores=None
):
    """Match vLLM candidate_blocks.py: per-query block maxima and newest block."""
    rows, width = scores.shape
    lengths = jnp.clip(lengths, 0, width)
    if block_scores is None:
        block_scores = publication.candidate_block_scores(
            scores, lengths, block_size=block_size
        )
    blocks = block_scores.shape[1]
    cand_k = min(count, blocks)
    if unordered and blocks >= 2048 and blocks % 16 == 0 and cand_k % 16 == 0:
        block_lengths = (lengths + block_size - 1) // block_size
        published = _sparsecore_top_k(
            block_scores, cand_k, jnp.minimum(block_lengths, blocks)
        ).astype(jnp.int32)
    else:
        values, ids = jax.lax.top_k(block_scores, cand_k)
        published = jnp.where(values > -jnp.inf, ids, -1).astype(jnp.int32)
    return jnp.pad(published, ((0, 0), (0, count - cand_k)), constant_values=-1)


def select_indices_jax(
    scores: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    topk: int,
    block_size: int,
    candidate_write: bool,
    candidate_filter: bool,
    unordered: bool = False,
    batch_max: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Source publishes candidates without filtering its own selection.

    ``unordered``: see `_visible_top_k` (the selection may come back in any
    order with ``-1`` anywhere; FP8 attention re-sorts via `compact_records`).
    ``batch_max``: the whole batch's longest length when ``scores`` holds only
    part of the batch's queries (defaults to ``lengths.max()``).
    """
    count, width = scores.shape
    if topk < 1 or width < 1 or block_size < 1:
        raise ValueError("Positive top-k, score width and block size required")
    if candidate_write and candidate_filter:
        raise ValueError("Candidate sources must not filter their own scores")
    if candidates.ndim != 2 or candidates.shape[0] != count:
        raise ValueError("Candidate rows must match query rows")
    lengths = jnp.clip(lengths.reshape(count), 0, width)
    batch_max = (
        lengths.max() if batch_max is None else jnp.clip(batch_max.max(), 0, width)
    )
    logical = jnp.arange(width)[None, :]
    visible = logical < lengths[:, None]
    scores = jnp.where(visible, scores, -jnp.inf)
    published = candidates
    # When the candidate buffer covers every block, publication/filtering are
    # identities. Keep the caller's buffer untouched in this static case.
    if (candidate_write or candidate_filter) and candidates.shape[
        1
    ] * block_size < width:
        blocks = (width + block_size - 1) // block_size
        if candidate_write:
            published = _publish_candidate_blocks(
                scores, lengths, candidates.shape[1], block_size, unordered
            )
        else:
            # One extra column absorbs invalid candidate IDs; no 3-D membership
            # tensor and no dependence on the order of candidate IDs.
            padded_blocks = ((blocks + 128) // 128) * 128
            valid = (candidates >= 0) & (candidates < blocks)
            ids = jnp.where(valid, candidates, blocks)
            membership = (
                jnp.zeros((count, padded_blocks), jnp.int32)
                .at[jnp.arange(count)[:, None], ids]
                .add(1)
            )
            allowed = jnp.broadcast_to(
                membership[:, :blocks, None] > 0, (count, blocks, block_size)
            ).reshape(count, blocks * block_size)[:, :width]
            # Match the upstream batch-wide short-context bypass.
            scores = jnp.where(allowed | (batch_max <= topk), scores, -jnp.inf)
    # Ascending: preserves V4.1's logical, hence attention accumulation, order.
    k = min(topk, width)
    indices = _visible_top_k(scores, k, lengths.max(), lengths, unordered)
    if unordered and width >= 2048 and width % 16 == 0 and k % 16 == 0:
        return _compact_selected(indices.astype(jnp.int32)), published
    values = jnp.take_along_axis(scores, jnp.maximum(indices, 0), axis=1)
    valid = (indices >= 0) & (values > -jnp.inf)
    selected = jnp.where(valid, indices, -1).astype(jnp.int32)
    if unordered:
        selected = _compact_selected(selected)
    return selected, published


# Index-score tile: about `_INDEX_KEYS_PER_ITER` keys (all query rows) per
# loop iteration, at most `_INDEX_MAX_TILE` columns (see paged_index_scores_jax).
_INDEX_KEYS_PER_ITER = 262144
_INDEX_MAX_TILE = 16384
# The packed scorer unrolls its query loop. Keep shared decode groups small;
# large prefill groups use the existing shared matmul.
_INDEX_MAX_SHARED_QUERIES = 8
# Word-row scoring uses a larger key budget to amortize paged gathers and
# kernel launches; grouped rows size the tile by request rows.
_INDEX_WORD_KEYS_PER_ITER = 4608 * 256


def paged_index_scores_jax(
    q: jax.Array,
    pool: jax.Array,
    records: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    requests: jax.Array | None = None,
    page_records: int | None = None,
) -> jax.Array:
    """Index scores ``[tokens, width]``.

    ``records`` is ``[tokens, width]``, or ``[requests, width]`` when
    ``requests`` maps each token to its row: every token of a request then
    shares one paged gather of the key tile instead of reading it again.
    ``page_records``: records per cache page when logical rows ``k * page ..``
    of every records row are one page's contiguous, pool-word-aligned records
    (`_request_row_slots` layout); score tiles, shared or per-token, then
    gather whole pages before decoding their halfplane words.
    """
    tokens, dim = q.shape[0], q.shape[-1]
    width = records.shape[1]
    # Bound key work per iteration while amortizing small-batch gather latency.
    eff_tokens = (
        tokens // 6
        if (
            tokens <= records.shape[0]
            and tokens <= 64
            and tokens >= 6
            and tokens % 6 == 0
        )
        else tokens
    )
    word_rows = (
        page_records is not None
        and dim == index_key_layout.DIM
        and q.dtype == jnp.bfloat16
    )
    keys_per_iter = _INDEX_WORD_KEYS_PER_ITER if word_rows else _INDEX_KEYS_PER_ITER
    budget = min(
        max(512, min(_INDEX_MAX_TILE, keys_per_iter // max(eff_tokens, 1))),
        -(-width // 128) * 128,
    )
    tile = 1 << (budget.bit_length() - 1)
    if (
        word_rows
        and requests is not None
        and records.shape[0] > 1
        and records.shape[0] < tokens <= _INDEX_MAX_SHARED_QUERIES * records.shape[0]
    ):
        # Grouped decode scoring reads `requests * tile` keys per iteration,
        # not `tokens * tile`: size the tile by request rows.
        grouped = min(
            _INDEX_MAX_TILE,
            _INDEX_KEYS_PER_ITER // records.shape[0],
            -(-width // 128) * 128,
        )
        if grouped > tile:
            tile = 1 << (grouped.bit_length() - 1)
    padded_width = width + (-width % tile)
    scores = jnp.full((tokens, padded_width), -jnp.inf, jnp.float32)
    q = split_halves(q)  # keys are read in the split element order
    weights = weights.astype(jnp.float32)[:, :, None]

    def head_sum(logits):
        # Upstream quantized MQA logits and the weighted head reduction are
        # FP32. BF16 cache storage is not a request to round the dot product.
        return (jnp.maximum(logits, 0) * weights).sum(1)

    pages = (
        page_records is not None
        and tile % page_records == 0
        and (page_records * dim // 2) % ROW_WORDS == 0
    )

    if requests is not None and tokens > records.shape[0]:
        # One request row at a time, each bounded by its own longest token:
        # all tokens score the row's shared key tile, and only that row's
        # tokens keep the result. Rows without tokens (padding) run no tiles,
        # so work scales with active requests, not the padded row count.
        records = jnp.pad(records, ((0, 0), (0, -width % tile)), constant_values=-1)
        row_lengths = (
            jnp.zeros((records.shape[0],), jnp.int32)
            .at[requests]
            .max(jnp.maximum(lengths, 0))
        )

        def row_step(row, result):
            mine = (requests == row)[:, None]

            def step(index, result):
                addresses = jax.lax.dynamic_slice(
                    records, (row, index * tile), (1, tile)
                ).reshape(-1)
                if pages:
                    keys = index_key_layout.unpack_rows(
                        gather_cache_page_words(
                            pool, addresses[::page_records], page_records, dim
                        )
                    )
                else:
                    keys = index_key_layout.gather_records(pool, addresses)
                values = head_sum(
                    jnp.einsum(
                        "thd,cd->thc",
                        q,
                        keys.reshape(tile, dim),
                        preferred_element_type=jnp.float32,
                    )
                )
                current = jax.lax.dynamic_slice_in_dim(
                    result, index * tile, tile, axis=1
                )
                return jax.lax.dynamic_update_slice_in_dim(
                    result, jnp.where(mine, values, current), index * tile, axis=1
                )

            trips = jnp.minimum(
                (row_lengths[row] + tile - 1) // tile, padded_width // tile
            )
            return jax.lax.fori_loop(0, trips, step, result)

        def shared_rows(_):
            return jax.lax.fori_loop(0, records.shape[0], row_step, scores)[:, :width]

        if (
            pages
            and dim == index_key_layout.DIM
            and q.dtype == jnp.bfloat16
            and records.shape[0] > 1
            and tokens <= _INDEX_MAX_SHARED_QUERIES * records.shape[0]
        ):
            # Balanced batches can score only each request's own queries. The
            # old row loop contracts every query against every request, then
            # discards the cross-request scores. Ignore zero-length padding
            # when checking capacity; uneven batches retain the general path.
            assert page_records is not None  # implied by `pages`
            groups = records.shape[0]
            group_tokens = -(-tokens // groups)
            belongs = (jnp.arange(groups)[:, None] == requests) & (lengths > 0)
            fits = jnp.all(belongs.sum(axis=1) <= group_tokens)

            def grouped_rows(_):
                token_ids = jax.vmap(
                    lambda mask: jnp.nonzero(
                        mask, size=group_tokens, fill_value=tokens
                    )[0]
                )(belongs).reshape(-1)
                grouped_q = jnp.pad(q, ((0, 1), (0, 0), (0, 0)))[token_ids]
                grouped_weights = jnp.pad(weights, ((0, 1), (0, 0), (0, 0)))[token_ids]

                def grouped_step(index, result):
                    addresses = jax.lax.dynamic_slice_in_dim(
                        records, index * tile, tile, axis=1
                    )
                    words = gather_cache_page_words(
                        pool, addresses[:, ::page_records], page_records, dim
                    )
                    values = _score_word_rows(
                        grouped_q,
                        grouped_weights,
                        words.reshape(
                            groups, tile // index_key_layout.SLOTS, ROW_WORDS
                        ),
                    )
                    # Sentinel token IDs are dropped, including duplicate
                    # padding slots. Every active token is written once.
                    ordered = jnp.full((tokens, tile), -jnp.inf, jnp.float32)
                    ordered = ordered.at[token_ids].set(values, mode="drop")
                    return jax.lax.dynamic_update_slice_in_dim(
                        result, ordered, index * tile, axis=1
                    )

                trips = jnp.minimum(
                    (row_lengths.max() + tile - 1) // tile, padded_width // tile
                )
                return jax.lax.fori_loop(0, trips, grouped_step, scores)[:, :width]

            return jax.lax.cond(fits, grouped_rows, shared_rows, None)
        return shared_rows(None)

    trips = jnp.minimum(
        (jnp.maximum(lengths.max(), 0) + tile - 1) // tile, padded_width // tile
    )
    if pages and dim == index_key_layout.DIM and q.dtype == jnp.bfloat16:
        # Score the gathered word rows in place: splitting them into
        # BF16 keys and relaying out [rows, 2, 64] -> [keys, 128] cost three
        # HBM passes per tile (8.7 of 17.5 us per 8K-key tile at 2 rows).
        # The kernel emits record order: a packed [tokens, slots, rows]
        # result cost 68 us per call to interleave in XLA.
        assert page_records is not None  # implied by `pages`
        page = page_records
        rows = tile // index_key_layout.SLOTS

        def _run_word_loop(rec):
            rec = jnp.pad(rec, ((0, 0), (0, -width % tile)), constant_values=-1)
            n_rows = rec.shape[0]

            def word_step(index, result):
                addresses = jax.lax.dynamic_slice_in_dim(
                    rec, index * tile, tile, axis=1
                )
                words = gather_cache_page_words(pool, addresses[:, ::page], page, dim)
                values = _score_word_rows(
                    q,
                    weights,
                    words.reshape(n_rows, rows, ROW_WORDS),
                )
                return jax.lax.dynamic_update_slice_in_dim(
                    result, values, index * tile, axis=1
                )

            return jax.lax.fori_loop(0, trips, word_step, scores)[:, :width]

        if requests is not None and tokens >= 6 and tokens % 6 == 0:
            group_tokens = 6
            groups = tokens // group_tokens
            can_group = jnp.all(
                requests.reshape(groups, group_tokens) == requests[::group_tokens, None]
            )
            return jax.lax.cond(
                can_group,
                lambda _: _run_word_loop(
                    jnp.take(records, requests[::group_tokens], axis=0)
                ),
                lambda _: _run_word_loop(jnp.take(records, requests, axis=0)),
                None,
            )
        if requests is not None:
            records = jnp.take(records, requests, axis=0)
        return _run_word_loop(records)

    if requests is not None:
        records = jnp.take(records, requests, axis=0)
    records = jnp.pad(records, ((0, 0), (0, -width % tile)), constant_values=-1)

    def step(index, result):
        # Read only the tile consumed by this iteration. Gathering before
        # the loop materializes [tokens, max_context / ratio, dim] even when
        # lengths restrict execution to a single tile (or no tiles).
        addresses = jax.lax.dynamic_slice_in_dim(records, index * tile, tile, axis=1)
        if pages:
            assert page_records is not None
            # Tiles start on page boundaries: one gather per page.
            keys = index_key_layout.unpack_rows(
                gather_cache_page_words(
                    pool, addresses[:, ::page_records], page_records, dim
                )
            )
        else:
            keys = index_key_layout.gather_records(pool, addresses.reshape(-1))
        keys = keys.reshape(records.shape[0], tile, dim)
        logits = jnp.einsum("thd,tcd->thc", q, keys, preferred_element_type=jnp.float32)
        return jax.lax.dynamic_update_slice_in_dim(
            result, head_sum(logits), index * tile, axis=1
        )

    return jax.lax.fori_loop(0, trips, step, scores)[:, :width]


_indexer_op = None
_indexer_2bank_op = None


def _decode_select(
    q,
    weights,
    pool,
    pages,
    lengths,
    *,
    page_records,
    candidate_count=0,
    candidate_block_size=1,
    k=512,
):
    """Run the complete TC/SC decode pipeline inside the manual wrapper mesh."""
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    select = functools.partial(
        decode_index_selection.decode_indexer,
        page_records=page_records,
        candidate_count=candidate_count,
        candidate_block_size=candidate_block_size,
        k=k,
    )
    try:
        mesh = get_vllm_model_wrapper_context().mesh
    except AssertionError:
        return select(q, weights, pool, pages, lengths)
    return jax.shard_map(
        select, mesh=mesh, in_specs=(P(),) * 5, out_specs=P(), check_vma=False
    )(q, weights, pool, pages, lengths)


def _prefill_score(
    q,
    weights,
    pool,
    records,
    limit,
    *,
    page_records,
    scheduling_group_id: int | None = None,
):
    """Resident-query scorer, manually replicated across the wrapper mesh."""
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    def score(q, weights, pool, records, limit):
        return prefill_scores.prefill_index_scores(
            q,
            weights,
            pool,
            records[:, ::page_records],
            limit,
            page_records=page_records,
            query_tile=prefill_tuning.QUERY_TILE,
            key_tile=prefill_tuning.KEY_TILE,
            records=records,
            scheduling_group_id=scheduling_group_id,
        )

    try:
        mesh = get_vllm_model_wrapper_context().mesh
    except AssertionError:
        return score(q, weights, pool, records, limit)
    return jax.shard_map(
        score, mesh=mesh, in_specs=(P(),) * 5, out_specs=P(), check_vma=False
    )(q, weights, pool, records, limit)


def _prefill_top_k(scores, lengths, *, k: int, scheduling_group_id: int | None = None):
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    def select(s, n):
        return streaming_topk.streaming_topk(
            s,
            valid_lengths=n,
            k=k,
            exclude_neg_inf=True,
            scheduling_group_id=scheduling_group_id,
        )

    try:
        mesh = get_vllm_model_wrapper_context().mesh
    except AssertionError:
        return select(scores, lengths)
    return jax.shard_map(
        select, mesh=mesh, in_specs=(P(), P()), out_specs=P(), check_vma=False
    )(scores, lengths)


def _score_word_rows(q: jax.Array, weights: jax.Array, words: jax.Array) -> jax.Array:
    """`packed_scores.score_word_rows` on every device (replicated)."""
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    score = functools.partial(
        packed_scores.score_word_rows,
        interpret=jax.default_backend() != "tpu",
    )
    try:
        mesh = get_vllm_model_wrapper_context().mesh
    except Exception:  # Standalone (tests): one device, no mesh.
        return score(q, weights, words)
    # Mosaic calls need manual partitioning; every device scores locally.
    return jax.shard_map(
        score, mesh=mesh, in_specs=(P(), P(), P()), out_specs=P(), check_vma=False
    )(q, weights, words)


def paged_indexer_jax(
    q: jax.Array,
    pool: jax.Array,
    records: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    requests: jax.Array,
    topk: int,
    block_size: int,
    candidate_write: bool,
    candidate_filter: bool,
    unordered: bool = False,
    batch_max: jax.Array | None = None,
    page_records: int | None = None,
    prefill_mode: str = "streaming",
) -> tuple[jax.Array, jax.Array]:
    """``records`` is ``[requests, width]``; ``requests`` maps tokens to rows.

    ``batch_max``: see `select_indices_jax`. ``page_records``: see
    `paged_index_scores_jax`; candidate scoring gathers runs of
    ``gcd(block_size, page_records)`` records (a block lies in one page).
    """

    def score_context(records, query_tile, mode="baseline"):
        # Finish selection before reusing a query tile's score buffer. Key
        # tiling alone still materializes [all queries, max context]: at
        # 2048 x 1M a single FP32 score matrix is 8 GiB, before top-k copies.
        # Reuse the native indexer's query tile and score/selection pipeline.
        rows = q.shape[0]
        longest = lengths.max() if batch_max is None else batch_max
        group_tokens = -(-rows // records.shape[0])
        use_prefill = mode != "baseline"

        def run(grouped):
            tile = min(rows, query_tile)
            inputs = (q, weights, lengths, candidates, requests)
            if grouped:
                # Keep a request's queries together and slice its records too:
                # flat query tiles otherwise gather the same keys per query.
                group_rows = min(records.shape[0], tile // group_tokens)
                tile = group_rows * group_tokens
                token_ids = jax.vmap(
                    lambda mask: jnp.nonzero(mask, size=group_tokens, fill_value=rows)[
                        0
                    ]
                )(belongs).reshape(-1)
                token_ids = jnp.pad(
                    token_ids, (0, -token_ids.size % tile), constant_values=rows
                )
                inputs = jax.tree.map(
                    lambda x: jnp.pad(x, ((0, 1),) + ((0, 0),) * (x.ndim - 1))[
                        token_ids
                    ],
                    inputs,
                )
                local_requests = jnp.repeat(jnp.arange(group_rows), group_tokens)
                record_tiles = jnp.pad(
                    records,
                    ((0, -records.shape[0] % group_rows), (0, 0)),
                    constant_values=-1,
                )
                count = token_ids.size
            else:
                count = rows + (-rows % tile)
                inputs = jax.tree.map(
                    lambda x: jnp.pad(
                        x, ((0, count - rows),) + ((0, 0),) * (x.ndim - 1)
                    ),
                    inputs,
                )

            def prepare(i):
                values = jax.tree.map(
                    lambda x: jax.lax.dynamic_slice_in_dim(x, i * tile, tile), inputs
                )
                if grouped:
                    row_records = jax.lax.dynamic_slice_in_dim(
                        record_tiles, i * group_rows, group_rows
                    )
                    return (*values[:-1], local_requests, row_records)
                return (*values, records)

            def score(args, scheduling_group_id: int | None = None):
                queries, row_weights, row_lengths, _, row_requests, row_records = args
                if use_prefill:
                    # The outer route proves every live query shares a request.
                    # Entries are first-record addresses, not page numbers.
                    request = requests[jnp.argmax(lengths)]
                    shared_records = jax.lax.dynamic_slice_in_dim(
                        row_records, request, 1
                    )
                    return _prefill_score(
                        split_halves(queries),
                        row_weights.astype(jnp.float32),
                        pool,
                        shared_records,
                        jnp.max(row_lengths, keepdims=True),
                        page_records=page_records,
                        scheduling_group_id=scheduling_group_id,
                    )
                else:
                    scores = paged_index_scores_jax(
                        queries,
                        pool,
                        row_records,
                        row_weights,
                        row_lengths,
                        row_requests,
                        page_records,
                    )
                return jnp.where(
                    jnp.take(row_records >= 0, row_requests, axis=0), scores, -jnp.inf
                )

            def select(i, scores, scheduling_group_id: int | None = None):
                _, _, row_lengths, row_candidates, _, _ = prepare(i)
                if use_prefill:
                    selected = _prefill_top_k(
                        scores,
                        row_lengths,
                        k=topk,
                        scheduling_group_id=scheduling_group_id,
                    )
                    return selected, row_candidates
                return select_indices_jax(
                    scores,
                    row_lengths,
                    row_candidates,
                    topk,
                    block_size,
                    candidate_write,
                    candidate_filter,
                    unordered,
                    longest,
                )

            if not grouped and rows <= tile:
                return select(jnp.int32(0), score(prepare(jnp.int32(0))))

            def write(i, selected, output):
                if grouped:
                    ids = jax.lax.dynamic_slice_in_dim(token_ids, i * tile, tile)
                    return jax.tree.map(
                        lambda dst, src: dst.at[ids].set(src, mode="drop"),
                        output,
                        selected,
                    )
                return jax.tree.map(
                    lambda dst, src: jax.lax.dynamic_update_slice_in_dim(
                        dst, src, i * tile, axis=0
                    ),
                    output,
                    selected,
                )

            if grouped:
                # Zero-length queries were omitted from the request groups.
                published = (
                    jnp.full_like(candidates, -1)
                    if candidate_write
                    and candidates.shape[1] * block_size < records.shape[1]
                    else candidates
                )
            else:
                published = jnp.zeros((count, candidates.shape[1]), candidates.dtype)
            output = (
                jnp.full(
                    (rows if grouped else count, min(topk, records.shape[1])),
                    -1,
                    jnp.int32,
                ),
                published,
            )
            pipeline = (
                indexer.pipeline_score_topk_bounded
                if use_prefill
                else indexer.pipeline_score_topk
            )
            selected, published = pipeline(
                count // tile, prepare, score, select, write, output
            )
            return selected[:rows], published[:rows]

        if (
            not use_prefill
            and page_records is not None
            and q.dtype == jnp.bfloat16
            and q.shape[-1] == index_key_layout.DIM
            and records.shape[0] > 1
            and 1 < group_tokens <= min(query_tile, _INDEX_MAX_SHARED_QUERIES)
        ):
            belongs = (jnp.arange(records.shape[0])[:, None] == requests) & (
                lengths > 0
            )
            fits = jnp.all(belongs.sum(axis=1) <= group_tokens)
            return jax.lax.cond(fits, lambda _: run(True), lambda _: run(False), None)
        return run(False)

    def prefill_context():
        widths = prefill_scores.context_buckets(records.shape[1])

        def run(columns):
            bounded_records = records[:, :columns]
            bounded_records = jnp.pad(
                bounded_records,
                ((0, 0), (0, columns - bounded_records.shape[1])),
                constant_values=-1,
            )
            return score_context(
                bounded_records, prefill_tuning.QUERY_TILE, prefill_mode
            )

        def all_indices(_):
            # No ranking is necessary when every visible key fits in TopK.
            ids = jnp.broadcast_to(
                jnp.arange(min(topk, records.shape[1]), dtype=jnp.int32),
                (q.shape[0], min(topk, records.shape[1])),
            )
            valid = (ids < lengths[:, None]) & (
                jnp.take(records[:, : ids.shape[1]], requests, axis=0) >= 0
            )
            return _compact_selected(jnp.where(valid, ids, -1)), candidates

        def ranked(_):
            branch = jnp.searchsorted(
                jnp.asarray(widths[:-1]), lengths.max(), side="left"
            )
            return jax.lax.switch(branch, [functools.partial(run, b) for b in widths])

        return jax.lax.cond(lengths.max() <= topk, all_indices, ranked, None)

    def decode_context():
        count = records.shape[0]
        queries = q.shape[0] // count
        # Pad the compact page table, not the full per-token record matrix.
        width = max(4096, 1 << (records.shape[1] - 1).bit_length())
        pages = jnp.pad(
            records[:, ::page_records],
            ((0, 0), (0, (width - records.shape[1]) // page_records)),
            constant_values=-1,
        )
        publish = (
            candidate_write and candidates.shape[1] * block_size < records.shape[1]
        )
        result = _decode_select(
            split_halves(q).reshape(count, queries, *q.shape[1:]),
            weights.astype(jnp.float32).reshape(count, queries, q.shape[1]),
            pool,
            pages,
            lengths.reshape(count, queries),
            page_records=page_records,
            candidate_count=candidates.shape[1] if publish else 0,
            candidate_block_size=block_size,
            k=topk,
        )
        return result if publish else (result, candidates)

    def full_context(_):
        if (
            jax.default_backend() == "tpu"
            and unordered
            and not candidate_filter
            and (
                not candidate_write
                or (
                    block_size > 0
                    and 4096 % block_size == 0
                    and candidates.shape[1] > 0
                    and candidates.shape[1] % 16 == 0
                )
            )
            and streaming_topk.is_supported(records.shape[1], topk)
            and decode_index_selection.supports_decode(q, pool, records, page_records)
        ):
            expected = jnp.repeat(
                jnp.arange(records.shape[0], dtype=jnp.int32),
                q.shape[0] // records.shape[0],
            )
            # Draft/mixed batches can share the padded shape but have different
            # request boundaries. Never reinterpret those rows as uniform groups.
            regular = jnp.all((requests == expected) | (lengths == 0))
            return jax.lax.cond(regular, decode_context, existing_context)
        return existing_context()

    def existing_context():
        rows = q.shape[0]
        if (
            records.shape[0] > 1
            and records.shape[0] < rows <= _INDEX_MAX_SHARED_QUERIES * records.shape[0]
            and rows * records.shape[1] <= _INDEX_UNTILED_SCORE_ELEMENTS
        ):
            # Decode-shaped batch (e.g. DSpark verify: 48 requests x 6 rows).
            # Query tiles of 32/128 rows hold fewer rows than request rows, so
            # `paged_index_scores_jax` could not share each request's key tile
            # across its queries and would gather every row's context again.
            # The untiled score buffer is small here; keep one query tile.
            return score_context(records, rows)
        columns, query_tile = prefill_tuning.PAGED_INDEXER_PREFILL_BUCKET
        if (
            not candidate_write
            and not candidate_filter
            and topk <= columns
            and q.shape[0] >= query_tile
            and records.shape[1] > columns
        ):
            # Select a smaller working shape without limiting request length.
            # Publication/filtering retain their full-width candidate contract.
            def bounded(_):
                return score_context(records[:, :columns], query_tile)

            return jax.lax.cond(
                lengths.max() <= columns,
                bounded,
                lambda _: score_context(records, indexer.QUERY_TILE),
                None,
            )
        return score_context(records, indexer.QUERY_TILE)

    if (
        prefill_mode != "baseline"
        and jax.default_backend() == "tpu"
        and unordered
        and not candidate_write
        and not candidate_filter
        and streaming_topk.is_supported(records.shape[1], topk)
        and prefill_scores.supports_chunked_prefill(q, pool, page_records)
    ):
        live = lengths > 0
        request = requests[jnp.argmax(lengths)]
        # Infer eligibility from live rows, not a prefill/decode phase flag.
        # Short contexts need no ranking; padding does not select a request.
        prefill = (live.sum() >= prefill_tuning.QUERY_TILE) | (lengths.max() <= topk)
        prefill &= jnp.all((requests == request) | ~live)
        return jax.lax.cond(prefill, prefill_context, lambda: full_context(None))

    width = records.shape[1]
    if not candidate_filter or candidates.shape[1] * block_size >= width:
        return full_context(None)

    # Bind shared-query prefill to one scoring pipeline when its static key
    # volume is no larger than per-token candidate scoring. Both paths apply
    # the same candidate mask and batch-wide short-context bypass. Keeping
    # this choice static avoids compiling both pipelines into every layer.
    if (
        q.shape[0] > records.shape[0]
        and records.shape[0] * width <= q.shape[0] * candidates.shape[1] * block_size
    ):
        return full_context(None)

    def candidate_context(_):
        # Candidates differ per token, so this path reads per-token records.
        # Sort before expanding so equal-score ties retain the full-context
        # top_k's logical ordering. Duplicate and invalid blocks cannot select
        # the same key twice or read outside this request's causal history.
        blocks = (width + block_size - 1) // block_size
        ids = jnp.sort(
            jnp.where((candidates >= 0) & (candidates < blocks), candidates, blocks),
            axis=-1,
        )
        unique = jnp.concatenate(
            (jnp.ones_like(ids[:, :1], dtype=jnp.bool_), ids[:, 1:] != ids[:, :-1]),
            axis=-1,
        )
        logical = (ids[..., None] * block_size + jnp.arange(block_size)).reshape(
            q.shape[0], -1
        )
        valid = logical < jnp.minimum(lengths[:, None], width)
        valid &= jnp.repeat(unique & (ids < blocks), block_size, axis=-1)
        # Gather only the candidate columns: per-token copies of whole
        # records rows ([tokens, width]) cost more than the candidates.
        compact = records[requests[:, None], jnp.minimum(logical, width - 1)]
        compact = jnp.where(valid, compact, -1)
        # This bounds both key reads and QK work by the candidate pool size;
        # zero-length padding rows do not trigger key reads.
        compact_lengths = jnp.where(lengths > 0, compact.shape[1], 0)
        # Candidate blocks are aligned runs of a page's contiguous records.
        segment = math.gcd(block_size, page_records) if page_records else None
        scores = paged_index_scores_jax(
            q,
            pool,
            compact,
            weights,
            compact_lengths,
            page_records=segment,
        )
        scores = jnp.where(compact >= 0, scores, -jnp.inf)
        output_width = min(topk, width)
        if (
            unordered
            and scores.shape[1] >= 2048
            and scores.shape[1] % 16 == 0
            and output_width % 16 == 0
        ):
            # Same finite top-k set as `jax.lax.top_k` below (SparseCore never
            # selects -inf); `_compact_selected` sorts, so order is irrelevant.
            picked = _sparsecore_top_k(scores, output_width, compact_lengths)
            selected = jnp.take_along_axis(logical, jnp.maximum(picked, 0), axis=1)
            return (
                _compact_selected(
                    jnp.where(picked >= 0, selected, -1).astype(jnp.int32)
                ),
                candidates,
            )
        padding = max(0, output_width - scores.shape[1])
        scores = jnp.pad(scores, ((0, 0), (0, padding)), constant_values=-jnp.inf)
        logical = jnp.pad(logical, ((0, 0), (0, padding)), constant_values=width)
        values, indices = jax.lax.top_k(scores, output_width)
        selected = jnp.take_along_axis(logical, indices, axis=1)
        selected_valid = values > -jnp.inf
        if unordered:
            return (
                _compact_selected(
                    jnp.where(selected_valid, selected, -1).astype(jnp.int32)
                ),
                candidates,
            )
        # The dense reference fills an underfull top-k with the lowest logical
        # -inf positions, then sorts all positions before masking them. Preserve
        # those padding locations too: moving padding across a 64-key attention
        # tile changes BF16 accumulation boundaries.
        occupied = jnp.zeros((q.shape[0], output_width + 1), jnp.bool_)
        occupied_ids = jnp.where(
            selected_valid & (selected < output_width), selected, output_width
        )
        occupied = occupied.at[jnp.arange(q.shape[0])[:, None], occupied_ids].set(True)
        fillers = jnp.sort(
            jnp.where(~occupied[:, :output_width], jnp.arange(output_width), width),
            axis=-1,
        )
        filler_rank = jnp.maximum(
            jnp.arange(output_width) - selected_valid.sum(-1)[:, None], 0
        )
        selected = jnp.where(
            selected_valid, selected, jnp.take_along_axis(fillers, filler_rank, axis=1)
        )
        order = jnp.argsort(selected, axis=-1)
        selected = jnp.take_along_axis(selected, order, axis=1)
        selected_valid = jnp.take_along_axis(selected_valid, order, axis=1)
        return jnp.where(selected_valid, selected, -1).astype(jnp.int32), candidates

    # Preserve the reference's batch-wide short-context bypass, and use the
    # shared-key-tile full_context path whenever the active requests' total
    # key reads are at most the per-token candidate key reads in prefill
    # (q.shape[0] > records.shape[0], where all tokens of a request share one
    # key tile in full_context).
    longest = lengths.max() if batch_max is None else batch_max.max()
    if q.shape[0] > records.shape[0]:
        row_lengths = (
            jnp.zeros((records.shape[0],), jnp.int32)
            .at[requests]
            .max(jnp.maximum(lengths, 0))
        )
        cand_keys = jnp.sum(lengths > 0) * (candidates.shape[1] * block_size)
        use_full_context = (longest <= topk) | (row_lengths.sum() <= cand_keys)
    else:
        use_full_context = longest <= topk
    return jax.lax.cond(
        use_full_context,
        full_context,
        candidate_context,
        None,
    )


def write_index_keys(
    pool: jax.Array | tuple[jax.Array, jax.Array],
    write_records: jax.Array,
    keys: jax.Array,
):
    """Publish BF16 index records into native packed halfplane storage."""
    if keys.dtype != jnp.bfloat16:
        raise ValueError("Index keys must be BF16")
    if keys.ndim != 2 or write_records.shape != (keys.shape[0],):
        raise ValueError("One physical record index per key is required")
    return scatter_index_keys(pool, write_records, keys)


def select_indexer_jax(
    q: jax.Array,
    pool: jax.Array,
    records: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    requests: jax.Array,
    batch_max: jax.Array,
    topk: int,
    block_size: int,
    candidate_write: bool,
    candidate_filter: bool,
    unordered: bool,
    page_records: int = 0,
    prefill_mode: str = "streaming",
) -> tuple[jax.Array, jax.Array]:
    """``page_records`` 0 disables page gathers (see `paged_indexer_jax`)."""
    return paged_indexer_jax(
        q,
        pool,
        records,
        weights,
        lengths,
        candidates,
        requests,
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        unordered,
        batch_max,
        page_records or None,
        prefill_mode,
    )


def select_indexer_2bank_jax(
    q: jax.Array,
    pool_lo: jax.Array,
    pool_hi: jax.Array,
    records: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    requests: jax.Array,
    batch_max: jax.Array,
    topk: int,
    block_size: int,
    candidate_write: bool,
    candidate_filter: bool,
    unordered: bool,
    page_records: int = 0,
    prefill_mode: str = "streaming",
) -> tuple[jax.Array, jax.Array]:
    return paged_indexer_jax(
        q,
        cast(Any, (pool_lo, pool_hi)),
        records,
        weights,
        lengths,
        candidates,
        requests,
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        unordered,
        batch_max,
        page_records or None,
        prefill_mode,
    )


def _register(name, function, fake, arrays, statics, donate=()):
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_pallas_op_mesh,
    )

    op = pallas.jax_op(
        name,
        function,
        donate_argnums=donate,
        mesh=get_pallas_op_mesh(),
        input_partition_specs=cast(Any, (P(),) * arrays + (None,) * statics),
    )
    op.register_fake(fake)
    return op


def initialize_indexer():
    from .candidate_cache import initialize_candidate_cache

    initialize_candidate_cache()
    global _indexer_op, _indexer_2bank_op
    if _indexer_op is not None:
        return

    def fake(
        q,
        pool,
        records,
        weights,
        lengths,
        candidates,
        requests,
        batch_max,
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        unordered,
        page_records=0,
        prefill_mode="streaming",
    ):
        return (
            records.new_empty(
                (q.shape[0], min(topk, records.shape[1])), dtype=torch.int32
            ),
            torch.empty_like(candidates),
        )

    _indexer_op = _register(
        "pallas::dsv41v2_exact_indexer", select_indexer_jax, fake, 8, 7, None
    )

    def fake_2bank(
        q,
        pool_lo,
        pool_hi,
        records,
        weights,
        lengths,
        candidates,
        requests,
        batch_max,
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        unordered,
        page_records=0,
        prefill_mode="streaming",
    ):
        return (
            records.new_empty(
                (q.shape[0], min(topk, records.shape[1])), dtype=torch.int32
            ),
            torch.empty_like(candidates),
        )

    _indexer_2bank_op = _register(
        "pallas::dsv41v2_exact_indexer_2bank",
        select_indexer_2bank_jax,
        fake_2bank,
        9,
        7,
        None,
    )


def paged_indexer(
    q,
    cache,
    records,
    weights,
    lengths,
    candidates,
    requests,
    *,
    topk,
    block_size,
    candidate_write,
    candidate_filter,
    unordered=False,
    batch_max=None,
    page_records=None,
    prefill_mode=None,
):
    """``records`` is ``[requests, width]``; ``requests`` maps tokens to rows.

    The model must publish Index-K before selection; this op only reads it.
    ``batch_max`` (default ``lengths.max()``) is the longest length of the
    whole batch when the queries are only part of it. ``page_records``: see
    `paged_index_scores_jax`.
    """
    # Resolve the comparison selector in the torch-to-JAX wrapper and pass it
    # explicitly: compiled JAX functions must not read environment state.
    if prefill_mode is None:
        prefill_mode = "streaming"
    assert _indexer_op is not None
    if batch_max is None:
        batch_max = lengths.max()
    tail_args = (
        records.int(),
        weights,
        lengths.flatten().int(),
        candidates.int(),
        requests.int(),
        batch_max.reshape(1).int(),
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        unordered,
        page_records or 0,
        prefill_mode,
    )
    if _has_hi_bank(cache):
        assert _indexer_2bank_op is not None
        return _indexer_2bank_op(
            q,
            cache.raw,
            cache.raw_hi,
            *tail_args,
        )
    return _indexer_op(
        q,
        cache.raw,
        *tail_args,
    )


_query_op = None
_write_op = None
_read_op = None
_select_op = None
_prepare_op = None
_prepared_select_op = None


def paged_fp8_indexer(
    q: torch.Tensor,
    cache: "PackedFP8IndexCacheRows",
    page_table: torch.Tensor,
    weights: torch.Tensor,
    lengths: torch.Tensor,
    candidates: torch.Tensor,
    requests: torch.Tensor,
    *,
    topk: int,
    block_size: int,
    candidate_write: bool,
    candidate_filter: bool,
    unordered: bool,
    batch_max: torch.Tensor,
    score_width: int,
    stage_cache: dict[Any, Any] | None,
    source_prefix: str,
    candidate_source: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score the published K cache using FP8 Q and weights with Q scales folded in."""
    if (
        candidate_filter
        and stage_cache is not None
        and q.shape[0] > 0
        and block_size in (8, 16, 32, 64, 128)
        and 0 < candidates.shape[1] * block_size < score_width
    ):
        assert candidate_source is not None
        key = (
            "packed_fp8_candidate_keys",
            source_prefix,
            stage_cache.get(("candidate_index_version", source_prefix), 0),
            candidate_source,
            stage_cache.get(("candidate_publication", candidate_source), 0),
            # This cache lives within one batch. Hashing its symbolic row count
            # would force Dynamo to specialize the dynamic token dimension.
            candidates.shape[1],
            block_size,
            score_width,
            cache.layout,
        )
        gathered_candidates = stage_cache.get(key)
        if gathered_candidates is None:
            assert _prepare_op is not None
            gathered_candidates = _prepare_op(
                cache.banks,
                page_table.int(),
                candidates.int(),
                requests.int(),
                cache._layout_args(),
                block_size,
                score_width,
            )
            stage_cache[key] = gathered_candidates
        assert _prepared_select_op is not None
        selected = _prepared_select_op(
            q,
            weights,
            lengths.flatten().int(),
            gathered_candidates,
            topk,
            block_size,
            unordered,
            score_width,
        )
        return selected, candidates
    assert _select_op is not None
    return _select_op(
        q,
        cache.banks,
        page_table.int(),
        weights,
        lengths.flatten().int(),
        candidates.int(),
        requests.int(),
        batch_max.reshape(1).int(),
        cache._layout_args(),
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        unordered,
        score_width,
    )


def initialize_fp8_indexer() -> None:
    global _query_op, _write_op, _read_op, _select_op, _prepare_op, _prepared_select_op
    if _query_op is not None:
        return

    def fake_query(q, positions, cos_sin, weights):
        return torch.empty_like(q, dtype=torch.float8_e4m3fn), torch.empty_like(
            weights, dtype=torch.float32
        )

    def fake_write(banks, records, keys, after, layout):
        return tuple(torch.empty_like(bank) for bank in banks)

    def fake_read(banks, records, layout):
        return banks[0].new_empty((records.numel(), 128), dtype=torch.bfloat16)

    def fake_select(
        q,
        banks,
        page_table,
        weights,
        lengths,
        candidates,
        requests,
        batch_max,
        layout,
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        unordered,
        score_width,
    ):
        return (
            page_table.new_empty((q.shape[0], min(topk, score_width))),
            torch.empty_like(candidates),
        )

    def fake_prepare(banks, page_table, candidates, requests, layout, block, width):
        tokens, count = candidates.shape
        groups = tokens
        count *= 2
        key_pages = (count * block + PAGE_RECORDS - 1) // PAGE_RECORDS
        return (
            banks[0].new_empty((groups * key_pages * PAGE_ROWS, 128)),
            candidates.new_empty(candidates.shape),
            requests.new_empty((groups + 1,)),
            requests.new_empty((tokens,), dtype=torch.bool),
            requests.new_empty((tokens, 2), dtype=torch.int32),
        )

    def fake_prepared(
        q,
        weights,
        lengths,
        gathered_candidates,
        topk,
        candidate_block_size,
        unordered,
        context_width,
    ):
        return q.new_empty((q.shape[0], min(topk, context_width)), dtype=torch.int32)

    _query_op = _register(
        "pallas::dsv41_packed_fp8_quantize_query", _quantize_query, fake_query, 4, 0
    )
    _write_op = _register(
        "pallas::dsv41_packed_fp8_write", _write, fake_write, 4, 1, (0,)
    )
    _read_op = _register("pallas::dsv41_packed_fp8_read", _read, fake_read, 2, 1)
    _select_op = _register(
        "pallas::dsv41_packed_fp8_select", select_fp8_indexer_jax, fake_select, 8, 7
    )
    _prepare_op = _register(
        "pallas::dsv41_packed_fp8_prepare_candidates", _prepare, fake_prepare, 4, 3
    )
    _prepared_select_op = _register(
        "pallas::dsv41_packed_fp8_select_prepared",
        _select_from_prepared_candidates,
        fake_prepared,
        4,
        4,
    )


def _quantize_query(
    q: jax.Array,
    positions: jax.Array,
    cos_sin: jax.Array,
    weights: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    from vllm_torchtpu.kernels.deepseek_v4.rope import rope_quant

    values, scales = rope_quant(q, positions, cos_sin, power_of_two=True)
    return values, weights.astype(jnp.float32) * scales


def _write(
    banks: tuple[jax.Array, ...],
    records: jax.Array,
    keys: jax.Array,
    after: jax.Array,
    layout: tuple[int, ...],
) -> tuple[jax.Array, ...]:
    banks, _ = jax.lax.optimization_barrier((banks, after))
    return write_fp8_records(banks, records, keys, PackedFP8Layout(*layout))


def _read(
    banks: tuple[jax.Array, ...],
    records: jax.Array,
    layout: tuple[int, ...],
) -> jax.Array:
    return read_fp8_records(banks, records, PackedFP8Layout(*layout))


def _select_candidate_tokens(
    scores: jax.Array,
    candidates: jax.Array,
    *,
    k: int,
    block_size: int,
    unordered: bool,
    valid_lengths: jax.Array | None = None,
    scheduling_group_id: int | None = None,
) -> jax.Array:
    """Select compact scores and map winners to logical token IDs."""
    if k == 0:
        return jnp.empty((scores.shape[0], 0), jnp.int32)
    if unordered and streaming_topk.is_supported(
        scores.shape[1],
        k,
        reconstruct_threshold=topk_reconstruction_threshold(k),
        logical_block_count=candidates.shape[1],
    ):
        return streaming_topk.streaming_topk(
            scores,
            valid_lengths=(
                jnp.full((scores.shape[0], 1), scores.shape[1], jnp.int32)
                if valid_lengths is None
                else valid_lengths.reshape(-1, 1)
            ),
            logical_blocks=candidates,
            k=k,
            exclude_neg_inf=True,
            unordered=True,
            reconstruct_threshold=topk_reconstruction_threshold(k),
            scheduling_group_id=scheduling_group_id,
        )
    if unordered:
        values, slots = jax.lax.top_k(scores, k)
        logical = (
            jnp.take_along_axis(candidates, slots // block_size, axis=1) * block_size
            + slots % block_size
        )
        return jnp.where(jnp.isfinite(values), logical, -1)
    slots = jnp.broadcast_to(jnp.arange(scores.shape[1]), scores.shape)
    logical = (
        jnp.take_along_axis(candidates, slots // block_size, axis=1) * block_size
        + slots % block_size
    )
    _, logical, slots = jax.lax.sort(
        (~streaming_topk._ordered_bits(scores), logical, slots), dimension=1, num_keys=2
    )
    logical, slots = logical[:, :k], slots[:, :k]
    values = jnp.take_along_axis(scores, slots, axis=1)
    valid = jnp.isfinite(values) if unordered else values > -jnp.inf
    sentinel = jnp.iinfo(jnp.int32).max
    logical = jnp.sort(jnp.where(valid, logical, sentinel))
    return jnp.where(logical < sentinel, logical, -1)


def _score_and_select_candidates(
    q,
    weights,
    lengths,
    gathered_candidates,
    topk,
    candidate_block_size,
    unordered,
    context_width,
):
    (
        candidate_keys,
        candidate_block_ids,
        query_group_offsets,
        valid_queries,
        block_spans,
    ) = gathered_candidates
    k = min(topk, context_width, candidate_block_ids.shape[1] * candidate_block_size)
    lengths = jnp.where(valid_queries, jnp.clip(lengths, 0, context_width), 0)
    windows = candidate_windows(
        candidate_block_ids, block_spans, lengths, candidate_block_size
    )
    candidate_token_counts = jnp.maximum(windows[:, 2], windows[:, 3]) - windows[:, 0]
    num_queries = q.shape[0]
    candidate_token_capacity = candidate_block_ids.shape[1] * candidate_block_size
    query_chunk_size = indexer_query_rows(num_queries, PACKED_FP8_MQA_CONFIG.query_tile)
    query_chunk_size = (
        query_chunk_size
        if unordered and num_queries % query_chunk_size == 0
        else num_queries
    )

    def score(start, group):
        scorer = (
            fp8_gathered_mqa_logits.__wrapped__
            if query_chunk_size < num_queries
            else fp8_gathered_mqa_logits
        )
        scores = scorer(
            q,
            candidate_keys,
            weights.astype(jnp.float32),
            windows,
            query_start_loc=query_group_offsets,
            output_width=candidate_token_capacity,
            query_start=start,
            query_count=query_chunk_size,
        )
        return (
            scores
            if group is None
            else xla_metadata.set_xla_metadata(scores, _scheduling_group_id=group)
        )

    if unordered:

        def select(tile, scores, group):
            return _select_candidate_tokens(
                scores,
                jax.lax.dynamic_slice_in_dim(
                    candidate_block_ids, tile * query_chunk_size, query_chunk_size
                ),
                k=k,
                block_size=candidate_block_size,
                unordered=True,
                valid_lengths=jax.lax.dynamic_slice_in_dim(
                    candidate_token_counts, tile * query_chunk_size, query_chunk_size
                ),
                scheduling_group_id=group,
            )

        selected = indexer.pipeline_score_topk_unrolled(
            num_queries // query_chunk_size,
            lambda tile: tile * query_chunk_size,
            score,
            select,
            lambda tile, src, dst: jax.lax.dynamic_update_slice_in_dim(
                dst, src, tile * query_chunk_size, axis=0
            ),
            jnp.full((num_queries, k), -1, jnp.int32),
        )
    else:
        scores = score(jnp.int32(0), None)
        selected = _select_candidate_tokens(
            scores,
            candidate_block_ids,
            k=k,
            block_size=candidate_block_size,
            unordered=False,
        )
    return jnp.pad(
        selected,
        ((0, 0), (0, min(topk, context_width) - k)),
        constant_values=-1,
    )


def _select_fp8_indexer_jax(
    q: jax.Array,
    banks: tuple[jax.Array, ...],
    page_table: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    requests: jax.Array,
    batch_max: jax.Array,
    layout_args: tuple[int, ...],
    topk: int,
    block_size: int,
    candidate_write: bool,
    candidate_filter: bool,
    unordered: bool,
    score_width: int,
    *,
    prefill_chunk_rows: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    layout = PackedFP8Layout(*layout_args)
    tokens, width = q.shape[0], score_width
    if tokens == 0:
        return jnp.empty((0, min(topk, width)), jnp.int32), candidates
    if q.shape[1:] != (32, 128) or width < 1:
        raise ValueError(
            "Packed FP8 model scorer requires H32/D128 and nonempty context"
        )
    if candidate_write and candidate_filter:
        raise ValueError("Candidate sources must not filter their own scores")
    if (
        candidate_filter
        and block_size in (8, 16, 32, 64, 128)
        and candidates.shape[1] * block_size < width
    ):

        def sparse():
            if not candidates.shape[1]:
                return jnp.full(
                    (q.shape[0], min(topk, width)), -1, jnp.int32
                ), candidates
            gathered_candidates = prepare_candidates(
                banks, page_table, candidates, requests, layout, block_size, width
            )
            return _score_and_select_candidates(
                q,
                weights,
                lengths,
                gathered_candidates,
                topk,
                block_size,
                unordered,
                width,
            ), candidates

        return jax.lax.cond(
            jnp.minimum(batch_max.max(), width) <= topk,
            lambda: select_fp8_indexer_jax(
                q,
                banks,
                page_table,
                weights,
                lengths,
                candidates,
                requests,
                batch_max,
                layout_args,
                topk,
                block_size,
                False,
                False,
                unordered,
                width,
            ),
            sparse,
        )
    key_tile = paged_mqa_key_tile(
        width, layout.page_size, PACKED_FP8_MQA_CONFIG.key_tile
    )
    query_tile = PACKED_FP8_MQA_CONFIG.query_tile
    capacity = (width + key_tile - 1) // key_tile * key_tile
    pages = page_table[:, : (width + layout.page_size - 1) // layout.page_size]
    pages = jnp.pad(
        pages,
        ((0, 0), (0, capacity // layout.page_size - pages.shape[1])),
        constant_values=-1,
    )

    rows = prefill_chunk_rows or min(
        max(tokens, query_tile),
        max(query_tile, _INDEX_UNTILED_SCORE_ELEMENTS // capacity),
    )
    padded = (tokens + rows - 1) // rows * rows
    safe_requests = jnp.clip(requests, 0, page_table.shape[0] - 1)
    valid_requests = (requests >= 0) & (requests < page_table.shape[0])
    lengths = jnp.where(valid_requests, jnp.clip(lengths, 0, width), 0)
    inputs = (q, weights, lengths, candidates, safe_requests)
    inputs = tuple(
        jnp.pad(x, ((0, padded - tokens),) + ((0, 0),) * (x.ndim - 1)) for x in inputs
    )
    counts = jnp.bincount(inputs[-1], length=page_table.shape[0])
    query_offsets = jnp.concatenate((jnp.zeros((1,), jnp.int32), jnp.cumsum(counts)))
    stream_select = (
        unordered
        and not candidate_filter
        and streaming_topk.is_supported(
            width,
            topk,
            reconstruct_threshold=topk_reconstruction_threshold(topk),
        )
    )
    publish_rows = candidate_write and candidates.shape[1] * block_size < width
    fuse_publication = (
        publish_rows
        and prefill_chunk_rows is None
        and supports_candidate_publication(block_size)
    )
    select_topk = functools.partial(
        streaming_topk.streaming_topk,
        k=topk,
        exclude_neg_inf=True,
        reconstruct_threshold=topk_reconstruction_threshold(topk),
    )

    def block(start, rows, inputs, *, score_only=False, group=None):
        shared_inputs = score_only and prefill_chunk_rows is None
        queries, row_weights, row_lengths, row_candidates, owners = (
            inputs
            if shared_inputs
            else tuple(jax.lax.dynamic_slice_in_dim(x, start, rows) for x in inputs)
        )
        offsets = (
            query_offsets if shared_inputs else jnp.clip(query_offsets - start, 0, rows)
        )
        pool = banks[0] if len(banks) == 1 else banks

        # Inline inside the outer model jit so scheduling metadata reaches
        # the Pallas call instead of being lost at the inner jit boundary.
        scorer = (
            fp8_paged_mqa_logits.__wrapped__ if score_only else fp8_paged_mqa_logits
        )
        scores = scorer(
            queries,
            pool,
            row_weights.astype(jnp.float32),
            row_lengths,
            pages,
            query_start_loc=offsets,
            max_model_len=capacity if stream_select else width,
            q_scale=jnp.ones_like(row_weights, dtype=jnp.float32),
            query_size=query_tile,
            cache_layout=layout,
            query_start=start if shared_inputs else 0,
            query_count=rows,
            reuse_keys=prefill_chunk_rows is not None and capacity == key_tile,
            candidate_block_size=block_size
            if score_only and fuse_publication
            else None,
        )
        if score_only:
            return (
                scores
                if group is None
                else jax.tree.map(
                    lambda value: xla_metadata.set_xla_metadata(
                        value, _scheduling_group_id=group
                    ),
                    scores,
                )
            )
        if stream_select:
            selected = select_topk(scores, valid_lengths=row_lengths)
        else:
            selected, row_candidates = select_indices_jax(
                scores,
                row_lengths,
                row_candidates,
                topk,
                block_size,
                candidate_write,
                candidate_filter,
                unordered,
                batch_max,
            )
        if stream_select and publish_rows:
            row_candidates = _publish_candidate_blocks(
                scores, row_lengths, candidates.shape[1], block_size, unordered
            )
        return selected, row_candidates

    def select_serial():
        order = jnp.argsort(inputs[-1], stable=True)
        sorted_inputs = tuple(x[order] for x in inputs)
        output = jnp.full((padded, min(topk, width)), -1, jnp.int32)
        published = jnp.full((padded, candidates.shape[1]), -1, jnp.int32)
        # Unroll SparseCore calls to avoid XLA loop lowering failures;
        # barriers bound score-buffer lifetimes.
        for index in range(padded // rows):
            start, output, published = jax.lax.optimization_barrier(
                (jnp.int32(index * rows), output, published)
            )
            result = block(start, rows, sorted_inputs)
            output, published = (
                jax.lax.dynamic_update_slice_in_dim(dst, src, start, axis=0)
                for dst, src in zip((output, published), result)
            )
        inverse = jnp.argsort(order)
        published = published[inverse]
        return output[inverse][:tokens], published[:tokens]

    short = jnp.minimum(batch_max.max(), width) <= topk

    chunk_rows = prefill_chunk_rows or indexer_query_rows(tokens, query_tile)
    if not stream_select or chunk_rows is None or chunk_rows > rows or padded != tokens:
        return select_serial()

    def select_pipelined():
        def score(start, group):
            return block(start, chunk_rows, inputs, score_only=True, group=group)

        def select(tile, scores, group):
            block_scores = None
            if fuse_publication:
                scores, block_scores = scores
            row_lengths = jax.lax.dynamic_slice_in_dim(
                inputs[2], tile * chunk_rows, chunk_rows
            )
            selected = select_topk(
                scores,
                valid_lengths=row_lengths,
                scheduling_group_id=group,
            )
            if publish_rows:
                return selected, _publish_candidate_blocks(
                    scores,
                    row_lengths,
                    candidates.shape[1],
                    block_size,
                    unordered,
                    block_scores=block_scores,
                )
            return selected

        def write(tile, selected, output):
            if publish_rows:
                return tuple(
                    jax.lax.dynamic_update_slice_in_dim(dst, src, tile * size, axis=0)
                    for dst, src, size in zip(
                        output, selected, (chunk_rows, chunk_rows)
                    )
                )
            return jax.lax.dynamic_update_slice_in_dim(
                output, selected, tile * chunk_rows, axis=0
            )

        output = jnp.full((tokens, topk), -1, jnp.int32)
        if publish_rows:
            output = (
                output,
                jnp.full(candidates.shape, -1, jnp.int32),
            )
        pipeline = (
            indexer.pipeline_score_topk_bounded
            if prefill_chunk_rows is not None or publish_rows
            else indexer.pipeline_score_topk_unrolled
        )
        selected = pipeline(
            padded // chunk_rows,
            lambda tile: tile * chunk_rows,
            score,
            select,
            write,
            output,
        )
        if publish_rows:
            selected, published = selected
            return selected, jnp.where(valid_requests[:, None], published, -1)
        return selected, candidates

    # Already grouped requests need neither sorting nor output reordering.
    # Decode preserves complete six-query groups. Prefill chunks instead match
    # the serial score-buffer boundaries, including groups cut by that boundary.
    # Both keep the same four-row/two-row head reduction and BF16 rounding.
    grouped = jnp.all(inputs[-1][1:] >= inputs[-1][:-1])
    complete = (
        jnp.bool_(True)
        if prefill_chunk_rows is not None
        else jnp.all(counts % query_tile == 0)
    )
    return jax.lax.cond(grouped & complete & ~short, select_pipelined, select_serial)


def select_fp8_indexer_jax(
    q: jax.Array,
    banks: tuple[jax.Array, ...],
    page_table: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    requests: jax.Array,
    batch_max: jax.Array,
    layout_args: tuple[int, ...],
    topk: int,
    block_size: int,
    candidate_write: bool,
    candidate_filter: bool,
    unordered: bool,
    score_width: int,
) -> tuple[jax.Array, jax.Array]:
    """Select exact FP8 scores with a bounded, overlapped prefill working set."""
    arguments = (
        q,
        banks,
        page_table,
        weights,
        lengths,
        candidates,
        requests,
        batch_max,
        layout_args,
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        unordered,
    )
    layout = PackedFP8Layout(*layout_args)
    width = paged_mqa_key_tile(
        score_width, layout.page_size, PACKED_FP8_MQA_CONFIG.key_tile
    )
    capacity = (score_width + width - 1) // width * width
    chunk = max(
        PACKED_FP8_MQA_CONFIG.query_tile,
        _INDEX_UNTILED_SCORE_ELEMENTS // max(capacity, 1),
    )
    if (
        width >= score_width
        or chunk >= q.shape[0]
        or q.shape[0] % chunk
        or candidate_write
        or candidate_filter
        or not unordered
        or topk > width
        or jax.default_backend() != "tpu"
    ):
        return _select_fp8_indexer_jax(*arguments, score_width)

    # Preserve score-buffer row boundaries for both widths. Only contexts
    # covered by one K tile can narrow the scores and retain their keys.
    live = (requests >= 0) & (requests < page_table.shape[0])
    longest = jnp.where(live, lengths, 0).max()
    grouped = jnp.all(requests[1:] >= requests[:-1])
    return jax.lax.cond(
        grouped & (longest <= width),
        lambda: _select_fp8_indexer_jax(*arguments, width, prefill_chunk_rows=chunk),
        lambda: _select_fp8_indexer_jax(
            *arguments, score_width, prefill_chunk_rows=chunk
        ),
    )


def _prepare(
    banks: tuple[jax.Array, ...],
    page_table: jax.Array,
    candidates: jax.Array,
    requests: jax.Array,
    layout: tuple[int, ...],
    block: int,
    width: int,
) -> tuple[jax.Array, ...]:
    return tuple(
        prepare_candidates(
            banks,
            page_table,
            candidates,
            requests,
            PackedFP8Layout(*layout),
            block,
            width,
        )
    )


def _select_from_prepared_candidates(
    q: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    gathered_candidates: tuple[jax.Array, ...],
    topk: int,
    candidate_block_size: int,
    unordered: bool,
    context_width: int,
) -> jax.Array:
    gathered_candidates = PreparedCandidates(*gathered_candidates)
    lengths = jnp.where(gathered_candidates.valid_queries, lengths, 0)

    def short():
        ids = jnp.broadcast_to(
            jnp.arange(min(topk, context_width), dtype=jnp.int32),
            (q.shape[0], min(topk, context_width)),
        )
        return jnp.where(ids < lengths[:, None], ids, -1)

    return jax.lax.cond(
        jnp.max(lengths) <= topk,
        short,
        lambda: _score_and_select_candidates(
            q,
            weights,
            lengths,
            gathered_candidates,
            topk,
            candidate_block_size,
            unordered,
            context_width,
        ),
    )
