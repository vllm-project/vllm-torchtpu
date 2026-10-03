"""V4.1 index selection over compact FP8 keys and companion scale words."""

import jax
import jax.numpy as jnp
import torch
from vllm.config import get_current_vllm_config
from vllm.models.deepseek_v41 import attention as upstream

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


def select_indices_jax(
    scores: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    topk: int,
    block_size: int,
    candidate_write: bool,
    candidate_filter: bool,
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
            values, ids = jax.lax.top_k(block_scores, candidates.shape[1])
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
    values, indices = jax.lax.top_k(scores, min(topk, width))
    # Preserve V4.1's logical order and therefore attention accumulation order.
    indices = jnp.sort(indices, axis=-1)
    values = jnp.take_along_axis(scores, indices, axis=1)
    return jnp.where(values > -jnp.inf, indices, -1).astype(jnp.int32), published


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
        # This bounds both key reads and QK work by the candidate pool size.
        compact_lengths = jnp.full_like(lengths, compact.shape[1])
        scores = paged_index_scores_jax(
            q, pool, scales, compact, weights, compact_lengths, page_stride
        )
        scores = jnp.where(compact >= 0, scores, -jnp.inf)
        output_width = min(topk, width)
        padding = max(0, output_width - scores.shape[1])
        scores = jnp.pad(scores, ((0, 0), (0, padding)), constant_values=-jnp.inf)
        logical = jnp.pad(logical, ((0, 0), (0, padding)), constant_values=width)
        values, indices = jax.lax.top_k(scores, output_width)
        selected = jnp.take_along_axis(logical, indices, axis=1)
        selected_valid = values > -jnp.inf
        # The dense reference fills an underfull top-k with the lowest logical
        # -inf positions, then sorts all positions before masking them. Preserve
        # those padding locations in the logical buffer shared by reuse layers.
        occupied = jnp.zeros((records.shape[0], output_width + 1), jnp.bool_)
        occupied_ids = jnp.where(
            selected_valid & (selected < output_width), selected, output_width
        )
        occupied = occupied.at[jnp.arange(records.shape[0])[:, None], occupied_ids].set(
            True
        )
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

    # Preserve the reference's batch-wide short-context bypass. That branch
    # scans at most topk keys; long contexts never enter the full-history path.
    return jax.lax.cond(lengths.max() <= topk, full_context, candidate_context, None)


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
    logical: jax.Array,
    topk: int,
    block_size: int,
    candidate_write: bool,
    candidate_filter: bool,
    page_size: int,
    page_stride: int,
    row_offset: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # Donate both original arrays; no slice/view of the companion pool escapes.
    safe = jnp.clip(logical, 0, block_table.shape[1] * page_size - 1)
    blocks = block_table[requests[:, None], safe // page_size]
    slots = blocks * page_size + safe % page_size
    records = row_indices(slots, page_size, page_stride, row_offset)
    write_records = masked_row_indices(write_slots, page_size, page_stride, row_offset)
    updated, updated_scales = write_index_keys(
        pool, scales, write_records, keys, page_stride
    )
    selected, published = paged_indexer_jax(
        q,
        updated,
        updated_scales,
        records,
        weights,
        lengths,
        candidates,
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        page_stride,
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
        input_partition_specs=(P(),) * 11 + (None,) * 7,
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
        logical,
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
            logical.new_empty(
                (q.shape[0], min(topk, logical.shape[1])), dtype=torch.int32
            ),
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
    logical,
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
        logical.int(),
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        *layout.geometry,
    )
