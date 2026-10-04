"""V4.1 index selection over compact FP8 keys and companion scale words."""

import jax
import jax.numpy as jnp
import torch
from vllm.config import get_current_vllm_config
from vllm.models.deepseek_v41 import attention as upstream

from vllm_torchtpu.kernels.deepseek_v41.cache_write import write_index_cache
from vllm_torchtpu.kernels.deepseek_v41.indexer import (
    QUERY_TILE,
    native_paged_scores,
    paged_scores,
    pipeline_score_topk,
)

from .deepseek_v41_backend import AttentionBackend
from .deepseek_v41_cache import (
    INDEX_HEAD_DIM,
    ROW_BYTES,
    ROW_WORDS,
    IndexPageGeometry,
    NativeCacheBinding,
    RecordLayout,
    cache_byte_layout,
    masked_row_indices,
    row_indices,
)

KEYS_PER_WORD_ROW = ROW_BYTES // INDEX_HEAD_DIM


class IndexerCache(NativeCacheBinding, upstream.DeepseekV4IndexerCache):
    def __init__(self, *args, **kwargs):
        if (
            get_current_vllm_config().model_config.hf_config.index_head_dim
            != INDEX_HEAD_DIM
        ):
            raise ValueError("V4.1 native index cache requires 128-dimensional keys")
        kwargs["head_dim"] = INDEX_HEAD_DIM
        kwargs["dtype"] = torch.uint8
        super().__init__(*args, **kwargs)

    def bind_kv_cache(self, cache, *, raw_cache=None, layout=None):
        self._native = None
        if raw_cache is None and isinstance(cache, torch.Tensor):
            super().bind_kv_cache(cache)
            return
        if not isinstance(cache, tuple) or len(cache) != 2:
            raise ValueError("V4.1 index cache requires a (keys, scales) pair")
        keys, scales = cache
        super().bind_kv_cache(keys)
        if raw_cache is not None:
            layout = cache_byte_layout(raw_cache, self.bound_cache, layout)
            self.bind_native(
                raw_cache,
                index_record_layout(raw_cache, scales, layout),
                scales=scales,
            )

    def get_kv_cache_spec(self, vllm_config):
        from vllm_torchtpu.runner.kv_cache_dsv4 import deepseek_v41_cache_spec

        return deepseek_v41_cache_spec(
            upstream.DeepseekV4IndexerCache.get_kv_cache_spec(self, vllm_config),
            vllm_config,
            indexer=True,
        )

    def get_attn_backend(self):
        return AttentionBackend


def index_record_layout(raw, scales, layout):
    """Validate compact keys and the existing companion array at binding."""
    shape, dtype, stride_bytes, offset_bytes = layout
    if dtype != torch.uint8 or shape[2:] != (INDEX_HEAD_DIM,):
        raise ValueError("Index cache requires 128-byte uint8 keys")
    if raw.dtype != torch.int32 or raw.ndim != 2 or raw.shape[1] != ROW_WORDS:
        raise ValueError("Index cache requires native int32 word rows")
    if stride_bytes % ROW_BYTES or offset_bytes != 0:
        raise ValueError("Index pages require complete word rows at pool offset zero")
    if (
        scales is None
        or scales.dtype != torch.int32
        or scales.ndim != 3
        or scales.shape[0] != shape[0]
        or scales.shape[2] != ROW_WORDS
    ):
        raise ValueError("Index scales require the native companion int32 array")
    geometry = IndexPageGeometry(
        stride_bytes, scales.shape[1] * scales.shape[2] * scales.element_size()
    )
    page_size = geometry.keys_per_page
    if shape[1] != page_size or raw.nbytes != shape[0] * stride_bytes:
        raise ValueError("Index key pages must fill the native NoPE array")
    if shape[0] * page_size > torch.iinfo(torch.int32).max:
        raise ValueError("Index record addresses must fit int32")
    return RecordLayout(page_size, page_size, 0, INDEX_HEAD_DIM, dtype)


def quantize_index_rows(values: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One power-of-two scale for all 128 post-RoPE channels of each Q/K."""
    if values.shape[-1] != INDEX_HEAD_DIM:
        raise ValueError("V4.1 FP8 index vectors must have 128 channels")
    values = values.astype(jnp.float32)
    amax = jnp.maximum(jnp.abs(values).max(axis=-1), 1e-4)
    exponent = jnp.ceil(jnp.log2(amax / 448.0)).astype(jnp.int32)
    scale_bytes = (exponent + 127).astype(jnp.uint8)
    scales = jax.lax.bitcast_convert_type(
        scale_bytes.astype(jnp.uint32) << 23, jnp.float32
    )
    quantized = (values / scales[..., None]).astype(jnp.float8_e4m3fn)
    return quantized, scales, scale_bytes


def pack_index_keys(keys: jax.Array) -> tuple[jax.Array, jax.Array]:
    """128 E4M3 bytes and one int32 word holding the unchanged E8M0 byte."""
    quantized, _, scale_bytes = quantize_index_rows(keys)
    return (
        jax.lax.bitcast_convert_type(quantized, jnp.uint8),
        scale_bytes.astype(jnp.int32),
    )


def gather_index_keys(
    pool: jax.Array, scales: jax.Array, records: jax.Array, page_stride: int
):
    """Decode only selected records, keeping key operands in FP8 for the dot."""

    def arbitrary_rows(_):
        safe = jnp.maximum(records, 0)
        word_rows = safe // KEYS_PER_WORD_ROW
        shifts = (safe % KEYS_PER_WORD_ROW * 8).astype(jnp.int32)
        return ((pool[word_rows] >> shifts[..., None]) & 255).astype(jnp.uint8)

    if records.shape[-1] % KEYS_PER_WORD_ROW:
        key_bytes = arbitrary_rows(None)
    else:
        groups = records.reshape(*records.shape[:-1], -1, KEYS_PER_WORD_ROW)
        first = groups[..., :1]
        lanes = jnp.arange(KEYS_PER_WORD_ROW)
        grouped = jnp.all(
            (groups < 0)
            | (
                (first >= 0)
                & (first % KEYS_PER_WORD_ROW == 0)
                & (groups == first + lanes)
            )
        )

        def four_key_slabs(_):
            # Full pages and aligned candidate blocks share these word reads
            # across four keys, rather than fetching each slab four times.
            rows = jnp.maximum(first[..., 0], 0) // KEYS_PER_WORD_ROW
            shifts = lanes * 8
            keys = (pool[rows][..., None, :] >> shifts[:, None]) & 255
            return keys.astype(jnp.uint8).reshape(*records.shape, INDEX_HEAD_DIM)

        key_bytes = jax.lax.cond(grouped, four_key_slabs, arbitrary_rows, None)
    safe = jnp.maximum(records, 0)
    within_page = safe % page_stride
    scale_bytes = scales[
        safe // page_stride, within_page // ROW_WORDS, within_page % ROW_WORDS
    ].astype(jnp.uint8)
    return (
        jax.lax.bitcast_convert_type(key_bytes, jnp.float8_e4m3fn),
        jax.lax.bitcast_convert_type(scale_bytes, jnp.float8_e8m0fnu).astype(
            jnp.float32
        ),
    )


def paged_index_scores_jax(
    q: jax.Array,
    pool: jax.Array,
    scales: jax.Array,
    records: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    page_stride: int,
) -> jax.Array:
    width = records.shape[1]
    tile = 128
    q, q_scales, _ = quantize_index_rows(q)
    weights = weights.astype(jnp.float32) * q_scales
    records = jnp.pad(records, ((0, 0), (0, -width % tile)), constant_values=-1)
    scores = jnp.full(records.shape, -jnp.inf, jnp.float32)

    def step(index, result):
        # Read only the tile consumed by this iteration. Gathering before
        # the loop materializes [tokens, max_context / ratio, dim] even when
        # lengths restrict execution to a single tile (or no tiles).
        addresses = jax.lax.dynamic_slice_in_dim(records, index * tile, tile, axis=1)
        keys, k_scales = gather_index_keys(pool, scales, addresses, page_stride)
        logits = jnp.einsum("thd,tcd->thc", q, keys, preferred_element_type=jnp.float32)
        # The FP8 dot and weighted head reduction accumulate in FP32. Q's
        # scale is folded into weights once; the shared K scale follows the
        # head reduction, as in V4's FP8 scoring path.
        logits = jnp.maximum(logits, 0)
        values = (logits * weights[:, :, None]).sum(1) * k_scales
        return jax.lax.dynamic_update_slice_in_dim(
            result, values.astype(jnp.float32), index * tile, axis=1
        )

    return jax.lax.fori_loop(
        0,
        jnp.minimum(
            (jnp.maximum(lengths.max(), 0) + tile - 1) // tile, records.shape[1] // tile
        ),
        step,
        scores,
    )[:, :width]


_indexer_op = None
_write_index_op = None


def _select_topk(scores, topk, *, sparsecore, row_lengths=None):
    """Select without a context-wide sort; keep XLA's low-index tie/pad ABI."""
    k = min(topk, scores.shape[1])
    if not sparsecore or k % 16 or k > 4096:
        return jax.lax.top_k(scores, k)
    from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import sparsecore_topk

    padded = jnp.pad(
        scores, ((0, 0), (0, -scores.shape[1] % 16)), constant_values=-jnp.inf
    )
    indices, words = sparsecore_topk(
        padded, k, row_lengths=row_lengths, return_scores=True
    )
    values = jax.lax.bitcast_convert_type(words, jnp.float32)
    valid = indices >= 0
    # XLA fills an underfull selection with the lowest unselected -inf columns.
    # Recreate those columns before the caller's logical sort, using only O(K).
    occupied = jnp.zeros((scores.shape[0], k + 1), jnp.bool_)
    occupied_ids = jnp.where(valid & (indices < k), indices, k)
    occupied = occupied.at[jnp.arange(scores.shape[0])[:, None], occupied_ids].set(True)
    fillers = jnp.sort(
        jnp.where(~occupied[:, :k], jnp.arange(k), scores.shape[1]), axis=-1
    )
    rank = jnp.maximum(jnp.arange(k) - valid.sum(-1)[:, None], 0)
    indices = jnp.where(valid, indices, jnp.take_along_axis(fillers, rank, axis=1))
    return values, indices


def select_indices_jax(
    scores: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    topk: int,
    block_size: int,
    candidate_write: bool,
    candidate_filter: bool,
    *,
    sparsecore: bool = False,
    scores_masked: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Source publishes candidates without filtering its own selection."""
    count, width = scores.shape
    if topk < 1 or width < 1 or block_size < 1:
        raise ValueError("Positive top-k, score width and block size required")
    if candidate_write and candidate_filter:
        raise ValueError("Candidate sources must not filter their own scores")
    if candidates.ndim != 2 or candidates.shape[0] != count:
        raise ValueError("Candidate rows must match query rows")
    lengths = jnp.clip(lengths.reshape(count), 0, width)
    logical = jnp.arange(width)[None, :]
    visible = logical < lengths[:, None]
    if not scores_masked or candidate_write or candidate_filter:
        scores = jnp.where(visible, scores, -jnp.inf)
    published = candidates
    # When the candidate buffer covers every block, publication/filtering are
    # identities. Keep the caller's buffer untouched in this static case.
    if (candidate_write or candidate_filter) and candidates.shape[
        1
    ] * block_size < width:
        blocks = (width + block_size - 1) // block_size
        if candidate_write:
            block_scores = (
                jnp.pad(
                    scores,
                    ((0, 0), (0, blocks * block_size - width)),
                    constant_values=-jnp.inf,
                )
                .reshape(count, blocks, block_size)
                .max(-1)
            )
            last = (lengths - 1) // block_size
            block_scores = jnp.where(
                jnp.arange(blocks)[None] == last[:, None], jnp.inf, block_scores
            )
            values, ids = _select_topk(
                block_scores, candidates.shape[1], sparsecore=sparsecore
            )
            published = jnp.where(values > -jnp.inf, ids, -1).astype(jnp.int32)
        else:
            # One extra column absorbs invalid candidate IDs; no 3-D membership
            # tensor and no dependence on the order of candidate IDs.
            valid = (candidates >= 0) & (candidates < blocks)
            ids = jnp.where(valid, candidates, blocks)
            membership = jnp.zeros((count, blocks + 1), jnp.bool_)
            membership = membership.at[jnp.arange(count)[:, None], ids].set(True)
            allowed = jnp.take_along_axis(membership, logical // block_size, axis=1)
            # Match the upstream batch-wide short-context bypass.
            scores = jnp.where(allowed | (lengths.max() <= topk), scores, -jnp.inf)
    values, indices = _select_topk(
        scores, min(topk, width), sparsecore=sparsecore, row_lengths=lengths
    )
    # Preserve V4.1's logical order and therefore attention accumulation order.
    order = jnp.argsort(indices, axis=-1)
    indices = jnp.take_along_axis(indices, order, axis=1)
    values = jnp.take_along_axis(values, order, axis=1)
    return jnp.where(values > -jnp.inf, indices, -1).astype(jnp.int32), published


def _compact_candidate_indices(
    q,
    pool,
    scales,
    compact,
    weights,
    lengths,
    logical,
    candidates,
    topk,
    capacity,
    page_stride,
    *,
    sparsecore,
):
    """Score compact records and retain dense-reference tie and padding order."""
    # This bounds both key reads and QK work by the candidate pool size.
    compact_lengths = jnp.full_like(lengths, compact.shape[1])
    scores = paged_index_scores_jax(
        q, pool, scales, compact, weights, compact_lengths, page_stride
    )
    scores = jnp.where(compact >= 0, scores, -jnp.inf)
    output_width = min(topk, capacity)
    padding = max(0, output_width - scores.shape[1])
    scores = jnp.pad(scores, ((0, 0), (0, padding)), constant_values=-jnp.inf)
    logical = jnp.pad(logical, ((0, 0), (0, padding)), constant_values=capacity)
    values, indices = _select_topk(scores, output_width, sparsecore=sparsecore)
    selected = jnp.take_along_axis(logical, indices, axis=1)
    selected_valid = values > -jnp.inf
    # The dense reference fills an underfull top-k with the lowest logical
    # -inf positions, then sorts all positions before masking them. Preserve
    # those padding locations in the logical buffer shared by reuse layers.
    occupied = jnp.zeros((q.shape[0], output_width + 1), jnp.bool_)
    occupied_ids = jnp.where(
        selected_valid & (selected < output_width), selected, output_width
    )
    occupied = occupied.at[jnp.arange(q.shape[0])[:, None], occupied_ids].set(True)
    fillers = jnp.sort(
        jnp.where(~occupied[:, :output_width], jnp.arange(output_width), capacity),
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


def paged_indexer_jax(
    q: jax.Array,
    pool: jax.Array,
    scales: jax.Array,
    records: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    topk: int,
    block_size: int,
    candidate_write: bool,
    candidate_filter: bool,
    page_stride: int,
    *,
    sparsecore: bool = False,
    short_context=None,
) -> tuple[jax.Array, jax.Array]:
    def full_context(_):
        scores = paged_index_scores_jax(
            q, pool, scales, records, weights, lengths, page_stride
        )
        scores = jnp.where(records >= 0, scores, -jnp.inf)
        return select_indices_jax(
            scores,
            lengths,
            candidates,
            topk,
            block_size,
            candidate_write,
            candidate_filter,
            sparsecore=sparsecore,
        )

    width = records.shape[1]
    if not candidate_filter or candidates.shape[1] * block_size >= width:
        return full_context(None)

    def candidate_context(_):
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
            records.shape[0], -1
        )
        valid = logical < jnp.minimum(lengths[:, None], width)
        valid &= jnp.repeat(unique & (ids < blocks), block_size, axis=-1)
        compact = jnp.take_along_axis(records, jnp.minimum(logical, width - 1), axis=1)
        compact = jnp.where(valid, compact, -1)
        return _compact_candidate_indices(
            q,
            pool,
            scales,
            compact,
            weights,
            lengths,
            logical,
            candidates,
            topk,
            width,
            page_stride,
            sparsecore=sparsecore,
        )

    # Preserve the reference's batch-wide short-context bypass. That branch
    # scans at most topk keys; long contexts never enter the full-history path.
    short_context = lengths.max() <= topk if short_context is None else short_context
    return jax.lax.cond(short_context, full_context, candidate_context, None)


def bounded_indexer_jax(
    q,
    pool,
    scales,
    block_table,
    weights,
    lengths,
    candidates,
    requests,
    capacity,
    topk,
    block_size,
    candidate_write,
    candidate_filter,
    page_size,
    page_stride,
    row_offset,
):
    """Keep only one query tile's scores and top-k scratch alive at a time."""
    count = q.shape[0]
    lengths = jnp.clip(lengths, 0, min(capacity, block_table.shape[1] * page_size))
    tile = min(QUERY_TILE, count)
    padded_count = (count + tile - 1) // tile * tile
    padding = padded_count - count
    lengths_padded = jnp.pad(lengths, (0, padding))
    requests_padded = jnp.pad(requests, (0, padding))
    candidates_padded = jnp.pad(candidates, ((0, padding), (0, 0)), constant_values=-1)
    q_padded = jnp.pad(q, ((0, padding), (0, 0), (0, 0)))
    weights_padded = jnp.pad(weights, ((0, padding), (0, 0)))
    selected = jnp.full((padded_count, min(topk, capacity)), -1, jnp.int32)
    published = candidates_padded
    sparsecore = jax.default_backend() == "tpu"
    short_context = lengths.max() <= topk

    if not candidate_write and not candidate_filter:

        def prepare(index):
            start = index * tile
            return (
                jax.lax.dynamic_slice_in_dim(q_padded, start, tile),
                jax.lax.dynamic_slice_in_dim(weights_padded, start, tile),
                jax.lax.dynamic_slice_in_dim(lengths_padded, start, tile),
                jax.lax.dynamic_slice_in_dim(requests_padded, start, tile),
            )

        def score(prepared):
            query, gates, lens, reqs = prepared
            return native_paged_scores(
                query,
                gates,
                pool,
                scales,
                block_table,
                reqs,
                lens,
                capacity,
                page_size=page_size,
                page_stride=page_stride,
                row_offset=row_offset,
                interpret=not sparsecore,
            )

        def select(index, scores):
            lens = jax.lax.dynamic_slice_in_dim(lengths_padded, index * tile, tile)
            cand = jax.lax.dynamic_slice_in_dim(candidates_padded, index * tile, tile)
            return select_indices_jax(
                scores,
                lens,
                cand,
                topk,
                block_size,
                False,
                False,
                sparsecore=sparsecore,
                scores_masked=sparsecore,
            )[0]

        def write(index, chosen, result):
            return jax.lax.dynamic_update_slice_in_dim(
                result, chosen, index * tile, axis=0
            )

        selected = pipeline_score_topk(
            padded_count // tile, prepare, score, select, write, selected
        )
        return selected[:count], published[:count]

    def step(index, output):
        start = index * tile
        lens = jax.lax.dynamic_slice_in_dim(lengths_padded, start, tile)
        reqs = jax.lax.dynamic_slice_in_dim(requests_padded, start, tile)
        cand = jax.lax.dynamic_slice_in_dim(candidates_padded, start, tile)

        def full_context(_):
            query = jax.lax.dynamic_slice_in_dim(q_padded, start, tile)
            gates = jax.lax.dynamic_slice_in_dim(weights_padded, start, tile)
            quantized, query_scales, _ = quantize_index_rows(query)
            scores = paged_scores(
                quantized,
                gates.astype(jnp.float32) * query_scales,
                pool,
                scales,
                block_table,
                reqs,
                lens,
                capacity,
                page_size=page_size,
                page_stride=page_stride,
                row_offset=row_offset,
                interpret=not sparsecore,
            )
            return select_indices_jax(
                scores,
                lens,
                cand,
                topk,
                block_size,
                candidate_write,
                False,
                sparsecore=sparsecore,
                scores_masked=True,
            )

        if candidate_filter and candidates.shape[1] * block_size < capacity:

            def candidate_context(_):
                # Form only candidate addresses. A dense [tile, capacity]
                # table gather otherwise costs more than scoring the candidates.
                blocks = (capacity + block_size - 1) // block_size
                ids = jnp.sort(
                    jnp.where((cand >= 0) & (cand < blocks), cand, blocks), axis=-1
                )
                unique = jnp.concatenate(
                    (
                        jnp.ones_like(ids[:, :1], dtype=jnp.bool_),
                        ids[:, 1:] != ids[:, :-1],
                    ),
                    axis=-1,
                )
                logical = (
                    ids[..., None] * block_size + jnp.arange(block_size)
                ).reshape(tile, -1)
                valid = logical < lens[:, None]
                valid &= jnp.repeat(unique & (ids < blocks), block_size, axis=-1)
                safe = jnp.minimum(logical, capacity - 1)
                pages = block_table[reqs[:, None], safe // page_size]
                records = row_indices(
                    pages * page_size + safe % page_size,
                    page_size,
                    page_stride,
                    row_offset,
                )
                records = jnp.where(valid, records, -1)
                return _compact_candidate_indices(
                    jax.lax.dynamic_slice_in_dim(q_padded, start, tile),
                    pool,
                    scales,
                    records,
                    jax.lax.dynamic_slice_in_dim(weights_padded, start, tile),
                    lens,
                    logical,
                    cand,
                    topk,
                    capacity,
                    page_stride,
                    sparsecore=sparsecore,
                )

            chosen, blocks = jax.lax.cond(
                short_context, full_context, candidate_context, None
            )
        else:
            chosen, blocks = full_context(None)
        indices, candidate_output = output
        indices = jax.lax.dynamic_update_slice_in_dim(indices, chosen, start, axis=0)
        candidate_output = jax.lax.dynamic_update_slice_in_dim(
            candidate_output, blocks, start, axis=0
        )
        return indices, candidate_output

    selected, published = jax.lax.fori_loop(
        0, padded_count // tile, step, (selected, published)
    )
    return selected[:count], published[:count]


def write_index_keys(
    pool: jax.Array,
    scales: jax.Array,
    write_records: jax.Array,
    keys: jax.Array,
    page_stride: int,
) -> tuple[jax.Array, jax.Array]:
    """Update native four-key slabs without losing neighboring key bytes."""
    if keys.dtype != jnp.bfloat16:
        raise ValueError("Index keys must be BF16")
    if keys.ndim != 2 or write_records.shape != (keys.shape[0],):
        raise ValueError("One physical record index per key is required")
    if not keys.shape[0]:
        return pool, scales
    if jax.default_backend() == "tpu":
        return write_index_cache(pool, scales, write_records, keys, page_stride)
    packed, scale_words = pack_index_keys(keys)
    records = jnp.maximum(write_records, 0)
    rows = records // KEYS_PER_WORD_ROW
    shifts = (records % KEYS_PER_WORD_ROW * 8)[:, None]
    previous = (pool[rows] >> shifts) & 255
    # Distinct records can own different bytes of the SAME word. Add integer
    # byte deltas so scatter combines those updates rather than racing whole
    # word replacements. Modular int32 addition preserves the other bytes.
    deltas = (packed.astype(jnp.int32) - previous) << shifts
    rows = jnp.where(write_records >= 0, rows, pool.shape[0])
    pool = pool.at[rows[:, None], jnp.arange(128)].add(deltas, mode="drop")
    pages = jnp.where(write_records >= 0, records // page_stride, scales.shape[0])
    within_page = records % page_stride
    scales = scales.at[pages, within_page // ROW_WORDS, within_page % ROW_WORDS].set(
        scale_words, mode="drop"
    )
    return pool, scales


def write_paged_index_keys(
    pool: jax.Array,
    scales: jax.Array,
    slots: jax.Array,
    keys: jax.Array,
    page_size: int,
    page_stride: int,
    row_offset: int,
) -> tuple[jax.Array, jax.Array]:
    records = masked_row_indices(slots, page_size, page_stride, row_offset)
    return write_index_keys(pool, scales, records, keys, page_stride)


def update_and_select_jax(
    q: jax.Array,
    pool: jax.Array,
    scales: jax.Array,
    block_table: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    write_slots: jax.Array,
    keys: jax.Array,
    requests: jax.Array,
    capacity: int,
    topk: int,
    block_size: int,
    candidate_write: bool,
    candidate_filter: bool,
    page_size: int,
    page_stride: int,
    row_offset: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # Donate both original arrays; no slice/view of the companion pool escapes.
    write_records = masked_row_indices(write_slots, page_size, page_stride, row_offset)
    updated, updated_scales = write_index_keys(
        pool, scales, write_records, keys, page_stride
    )
    selected, published = bounded_indexer_jax(
        q,
        updated,
        updated_scales,
        block_table,
        weights,
        lengths,
        candidates,
        requests,
        capacity,
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        page_size,
        page_stride,
        row_offset,
    )
    return updated, updated_scales, selected, published


def initialize_indexer():
    global _indexer_op, _write_index_op
    if _indexer_op is not None:
        return
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    _indexer_op = pallas.jax_op(
        "pallas::deepseek_v41_fp8_indexer",
        update_and_select_jax,
        donate_argnums=(1, 2),
        mesh=get_vllm_model_wrapper_context().mesh,
        input_partition_specs=(P(),) * 10 + (None,) * 8,
    )

    def fake(
        q,
        pool,
        scales,
        block_table,
        weights,
        lengths,
        candidates,
        write_slots,
        keys,
        requests,
        capacity,
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        page_size,
        page_stride,
        row_offset,
    ):
        return (
            torch.empty_like(pool),
            torch.empty_like(scales),
            q.new_empty((q.shape[0], min(topk, capacity)), dtype=torch.int32),
            torch.empty_like(candidates),
        )

    _indexer_op.register_fake(fake)
    _write_index_op = pallas.jax_op(
        "pallas::deepseek_v41_write_index_keys",
        write_paged_index_keys,
        donate_argnums=(0, 1),
        mesh=get_vllm_model_wrapper_context().mesh,
        input_partition_specs=(P(),) * 4 + (None,) * 3,
    )
    _write_index_op.register_fake(
        lambda pool, scales, slots, keys, *geometry: (
            torch.empty_like(pool),
            torch.empty_like(scales),
        )
    )


def publish_index_keys(cache, scales, slots, keys, *, layout):
    assert _write_index_op is not None
    updated, updated_scales = _write_index_op(
        cache, scales, slots.int(), keys, *layout.geometry
    )
    cache.copy_(updated)
    scales.copy_(updated_scales)


def paged_indexer(
    q,
    cache,
    scales,
    block_table,
    requests,
    capacity,
    weights,
    lengths,
    candidates,
    write_slots,
    keys,
    *,
    layout,
    topk,
    block_size,
    candidate_write,
    candidate_filter,
):
    assert _indexer_op is not None
    return _indexer_op(
        q,
        cache,
        scales,
        block_table.int(),
        weights,
        lengths.flatten().int(),
        candidates.int(),
        write_slots.int(),
        keys,
        requests.int(),
        capacity,
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        *layout.geometry,
    )
