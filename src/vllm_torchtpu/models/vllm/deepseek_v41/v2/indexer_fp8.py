# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed FP8 indexer: candidate preparation, scoring, selection and op registration."""

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import torch
from jax.experimental import xla_metadata

from vllm_torchtpu.kernels.deepseek_v41 import (
    indexer,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection import (
    block_topk,
    streaming_topk,
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
    fp8_paged_mqa_logits_inline,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.sparse_logits import (
    candidate_windows,
    fp8_gathered_mqa_logits,
    fp8_gathered_mqa_logits_inline,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.tuned import (
    PACKED_FP8_MQA_CONFIG,
    PACKED_FP8_RESIDENT_KEYS,
    PACKED_FP8_WIDTH_STEP,
    dense_key_tile,
    indexer_query_rows,
    paged_mqa_key_tile,
    topk_reconstruction_threshold,
)

from . import indexer_select
from .indexer_select import (
    _INDEX_UNTILED_SCORE_ELEMENTS,
    _publish_candidate_blocks,
    _register,
)

if TYPE_CHECKING:
    from .indexer_cache import PackedFP8IndexCacheRows

# consume(start, rows, state, following) -> (state, following); see
# `select_fp8_indexer_tiles`.
SelectionConsumer = Callable[[int, jax.Array, Any, Any], tuple[Any, Any]]

_query_op = None
_write_op = None
_read_op = None
_select_op = None
_prepare_op = None
_prepared_select_op = None
# Keys per block maximum that block top-k reduces on SparseCore for layers
# without candidate publication: one maximum per half 16-lane vector.
_BLOCK_TOPK_BLOCK = 8


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
    sparse_logits: bool,
    unordered: bool,
    batch_max: torch.Tensor,
    score_width: int,
    stage_cache: dict[Any, Any] | None,
    source_prefix: str,
    candidate_source: int | None,
    query_start_loc: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score the published K cache using FP8 Q and weights with Q scales folded in.

    ``query_start_loc``: [page_table rows + 1] request row boundaries of ``q``
    (the last may undercount trailing padding rows). None derives them from
    ``requests``, which costs a rows x requests comparison.
    """
    if sparse_logits and uses_prepared_candidates(
        q.shape[0], candidates, block_size, score_width, stage_cache
    ):
        assert candidate_source is not None and stage_cache is not None
        gathered_candidates = prepared_candidates(
            cache,
            page_table,
            candidates,
            requests,
            block_size=block_size,
            score_width=score_width,
            stage_cache=stage_cache,
            source_prefix=source_prefix,
            candidate_source=candidate_source,
        )
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
    if query_start_loc is None:
        # Padding requests (-1) trail the batch and count as the last request.
        last = page_table.shape[0] - 1
        owners = torch.where(requests >= 0, requests.clamp(max=last), last)
        bounds = torch.arange(last + 2, device=requests.device, dtype=owners.dtype)
        query_start_loc = (owners[:, None] < bounds).sum(0)
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
        query_start_loc.int(),
        cache._layout_args(),
        topk,
        block_size,
        candidate_write,
        sparse_logits,
        unordered,
        score_width,
    )


def uses_prepared_candidates(
    rows: int,
    candidates: torch.Tensor,
    block_size: int,
    score_width: int,
    stage_cache: dict[Any, Any] | None,
) -> bool:
    """Whether candidate layers score K gathered once per stage."""
    return (
        stage_cache is not None
        and rows > 0
        and block_size in (8, 16, 32, 64, 128)
        and 0 < candidates.shape[1] * block_size < score_width
    )


def prepared_candidates(
    cache: "PackedFP8IndexCacheRows",
    page_table: torch.Tensor,
    candidates: torch.Tensor,
    requests: torch.Tensor,
    *,
    block_size: int,
    score_width: int,
    stage_cache: dict[Any, Any],
    source_prefix: str,
    candidate_source: int,
) -> tuple[torch.Tensor, ...]:
    """Candidate K and metadata, gathered once per Index-K and candidate version."""
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
    gathered = stage_cache.get(key)
    if gathered is None:
        assert _prepare_op is not None
        gathered = _prepare_op(
            cache.banks,
            page_table.int(),
            candidates.int(),
            requests.int(),
            cache._layout_args(),
            block_size,
            score_width,
        )
        stage_cache[key] = gathered
    return gathered


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
        query_start_loc,
        layout,
        topk,
        block_size,
        candidate_write,
        sparse_logits,
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
        "pallas::dsv41_packed_fp8_select", select_fp8_indexer_jax, fake_select, 9, 7
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
            fp8_gathered_mqa_logits_inline
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
            lambda tile, src, dst, following: (
                jax.lax.dynamic_update_slice_in_dim(
                    dst, src, tile * query_chunk_size, axis=0
                ),
                following,
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
    query_start_loc: jax.Array,
    layout_args: tuple[int, ...],
    topk: int,
    block_size: int,
    candidate_write: bool,
    sparse_logits: bool,
    unordered: bool,
    score_width: int,
    *,
    prefill_chunk_rows: int | None = None,
    consume: "SelectionConsumer | None" = None,
    records: "tuple[jax.Array, jax.Array, int] | None" = None,
) -> tuple[jax.Array, ...]:
    """Selected indices and published candidates; see `select_fp8_indexer_tiles`."""

    def finish(selected, published):
        if consume is None:
            return selected, published
        return selected, published, consume(0, selected, None, None)[0]

    layout = PackedFP8Layout(*layout_args)
    tokens, width = q.shape[0], score_width
    if tokens == 0:
        return finish(jnp.empty((0, min(topk, width)), jnp.int32), candidates)
    if q.shape[1:] != (32, 128) or width < 1:
        raise ValueError(
            "Packed FP8 model scorer requires H32/D128 and nonempty context"
        )
    if candidate_write and sparse_logits:
        raise ValueError("Candidate sources must not use sparse logits")
    if (
        sparse_logits
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
                query_start_loc,
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
    key_tile = dense_key_tile(width, layout.page_size, prefill_chunk_rows is not None)
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
    # Callers group each request's queries contiguously in ascending request
    # order; padding requests (-1) trail them. Clamping padding to the last
    # request keeps that order, which the pipelined path relies on.
    safe_requests = jnp.where(
        requests >= 0,
        jnp.minimum(requests, page_table.shape[0] - 1),
        page_table.shape[0] - 1,
    )
    valid_requests = (requests >= 0) & (requests < page_table.shape[0])
    lengths = jnp.where(valid_requests, jnp.clip(lengths, 0, width), 0)
    inputs = (q, weights, lengths, candidates, safe_requests)
    inputs = tuple(
        jnp.pad(x, ((0, padded - tokens),) + ((0, 0),) * (x.ndim - 1)) for x in inputs
    )
    if query_start_loc.shape != (page_table.shape[0] + 1,):
        raise ValueError("query_start_loc needs one entry per page-table row, plus one")
    # Trailing padding rows belong to the last request, as in safe_requests.
    query_offsets = jnp.minimum(query_start_loc.astype(jnp.int32), tokens)
    query_offsets = query_offsets.at[-1].set(tokens)
    if padded > tokens:
        # Tile padding rows own request 0, after its rows once sorted.
        query_offsets = query_offsets.at[1:].add(padded - tokens)
    stream_select = (
        unordered
        and not sparse_logits
        and streaming_topk.is_supported(
            width,
            topk,
            reconstruct_threshold=topk_reconstruction_threshold(topk),
        )
    )
    publish_rows = candidate_write and candidates.shape[1] * block_size < width
    select_topk = functools.partial(
        streaming_topk.streaming_topk,
        k=topk,
        exclude_neg_inf=True,
        reconstruct_threshold=topk_reconstruction_threshold(topk),
        # Attention re-sorts the records; skip ID tie-breaks and output sorting.
        unordered=unordered,
    )
    chunk_rows = prefill_chunk_rows or indexer_query_rows(tokens, query_tile)
    # Top-k scans only the blocks that can hold a winner. Publishing layers
    # rank the same SparseCore block maxima for their candidates. A prefill
    # tile of 256 x 65,536 selects in 113 us on SparseCore, against 180 us
    # for the streaming top-k, so scoring sets the pipeline's pace.
    select_block = block_size if publish_rows else _BLOCK_TOPK_BLOCK
    block_select = (
        stream_select
        and chunk_rows is not None
        and block_topk.is_supported(chunk_rows, capacity, topk, select_block)
    )
    pool = banks[0] if len(banks) == 1 else banks

    def score(start, rows, inputs, *, pipelined, group=None):
        """Scores of `rows` queries from `start`; decode tiles share the inputs."""
        shared = pipelined and prefill_chunk_rows is None
        queries, row_weights, row_lengths = (
            inputs[:3]
            if shared
            else (jax.lax.dynamic_slice_in_dim(x, start, rows) for x in inputs[:3])
        )
        output_width = capacity if stream_select else width
        # Prefill scores one K tile at a time, each kept resident while every
        # query of the chunk reads it. A multi-tile call reloads every tile
        # for each 6-query group instead.
        tiles = (
            output_width // key_tile
            if prefill_chunk_rows is not None and output_width % key_tile == 0
            else 1
        )
        # Scheduling metadata tags the op producing the scores. A single
        # pipelined call is inlined so that op is the Pallas call itself;
        # K tiles meet in a concatenate, so their calls stay behind a jit
        # boundary, whose lowering is also shared.
        scorer = (
            fp8_paged_mqa_logits_inline
            if pipelined and tiles == 1
            else fp8_paged_mqa_logits
        )
        tile_width = output_width // tiles
        tile_pages = tile_width // layout.page_size
        scores = jnp.concatenate(
            [
                scorer(
                    queries,
                    pool,
                    row_weights.astype(jnp.float32),
                    jnp.clip(row_lengths - t * tile_width, 0, tile_width)
                    if tiles > 1
                    else row_lengths,
                    pages[:, t * tile_pages : (t + 1) * tile_pages]
                    if tiles > 1
                    else pages,
                    query_start_loc=(
                        query_offsets
                        if shared
                        else jnp.clip(query_offsets - start, 0, rows)
                    ),
                    max_model_len=tile_width,
                    q_scale=jnp.ones_like(row_weights, dtype=jnp.float32),
                    query_size=query_tile,
                    cache_layout=layout,
                    query_start=start if shared else 0,
                    query_count=rows,
                    reuse_keys=prefill_chunk_rows is not None
                    and tile_width == key_tile,
                )
                for t in range(tiles)
            ],
            axis=1,
        )
        if group is None:
            return scores
        tag = functools.partial(
            xla_metadata.set_xla_metadata, _scheduling_group_id=group
        )
        return jax.tree.map(tag, scores)

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
            scores = score(start, rows, sorted_inputs, pipelined=False)
            row_lengths, row_candidates = (
                jax.lax.dynamic_slice_in_dim(x, start, rows) for x in sorted_inputs[2:4]
            )
            if not stream_select:
                selected, row_candidates = indexer_select.select_indices_jax(
                    scores,
                    row_lengths,
                    row_candidates,
                    topk,
                    block_size,
                    candidate_write,
                    sparse_logits,
                    unordered,
                    batch_max,
                )
            else:
                selected = select_topk(scores, valid_lengths=row_lengths)
                if publish_rows:
                    row_candidates = _publish_candidate_blocks(
                        scores, row_lengths, candidates.shape[1], block_size, unordered
                    )
            output, published = (
                jax.lax.dynamic_update_slice_in_dim(dst, src, start, axis=0)
                for dst, src in zip((output, published), (selected, row_candidates))
            )
        inverse = jnp.argsort(order)
        return output[inverse][:tokens], published[inverse][:tokens]

    if not stream_select or chunk_rows is None or chunk_rows > rows or padded != tokens:
        return finish(*select_serial())

    def select(tile, scores, group):
        row_lengths = jax.lax.dynamic_slice_in_dim(
            inputs[2], tile * chunk_rows, chunk_rows
        )
        if block_select:
            # One SparseCore pass ranks the block maxima for both outputs.
            return block_topk.block_topk(
                scores,
                row_lengths,
                k=topk,
                block_size=select_block,
                publish=candidates.shape[1] if publish_rows else 0,
                records=None
                if tile_records is None
                else (
                    tile_records[0],
                    jax.lax.dynamic_slice_in_dim(
                        tile_records[1], tile * chunk_rows, chunk_rows
                    ),
                    short.astype(jnp.int32).reshape(1),
                    tile_records[2],
                ),
                scheduling_group_id=group,
            )
        selected = select_topk(
            scores, valid_lengths=row_lengths, scheduling_group_id=group
        )
        if publish_rows:
            return selected, _publish_candidate_blocks(
                scores,
                row_lengths,
                candidates.shape[1],
                block_size,
                unordered,
                scheduling_group_id=group,
            )
        return selected

    # Batch-wide short-context bypass: every visible key, as in the reference.
    # A select rather than a cond keeps the scoring path free of branches.
    short = jnp.minimum(batch_max.max(), width) <= topk
    ids = jnp.arange(topk, dtype=jnp.int32)
    visible = jnp.where(ids < lengths[:tokens, None], ids, -1)
    num_tiles = padded // chunk_rows
    bounded = prefill_chunk_rows is not None or publish_rows
    # Unrolled prefill tiles have static row offsets, so each tile's selection
    # feeds the consumer while later tiles are still scored and selected.
    # Decode tiles (unbounded) gain nothing from it: their SparseCore
    # selections and gathers serialize, so per-tile gathers and attention
    # only add calls (72 rows, 3 tiles: 216.2 vs 210.0 us per layer, bitwise).
    unrolled = bounded and num_tiles <= indexer.MAX_UNROLLED_TILES
    tile_consume = consume if unrolled else None
    # The SparseCore selection also resolves the consumer's records: the
    # TensorCore then has no page lookup between a tile's selection and its
    # gather. Selected IDs stay below `capacity`, so only that many pages of
    # the table reach SparseCore VMEM: the server's full table (8 request
    # slots x 4,096 pages) exceeds the room `block_topk` leaves.
    if records is not None:
        table, row_requests, page = records
        records = (table[:, : -(-capacity // page)], row_requests, page)
    tile_records = (
        records
        if tile_consume is not None
        and records is not None
        and block_select
        and not publish_rows
        and block_topk.records_fit(
            chunk_rows, capacity, topk, select_block, records[0].size
        )
        else None
    )

    def write(tile, selected, output, following):
        update = functools.partial(
            jax.lax.dynamic_update_slice_in_dim, start_index=tile * chunk_rows, axis=0
        )
        if tile_consume is None:
            return jax.tree.map(update, output, selected), following
        output, state = output
        start = tile * chunk_rows
        if tile_records is not None:
            selected, physical, counts = selected
            rows = (physical, counts[:, 0])
        else:
            rows = selected[0] if publish_rows else selected
            rows = jnp.where(short, visible[start : start + chunk_rows], rows)
        state, following = tile_consume(start, rows, state, following)
        return (jax.tree.map(update, output, selected), state), following

    # Grouped requests need neither sorting nor output reordering, so no
    # runtime branch to the serial path: a branch costs an overlay load at its
    # entry. Score tiles may split a request; the kernel resumes it mid-group.
    output = jnp.full((tokens, topk), -1, jnp.int32)
    if publish_rows:
        output = (output, jnp.full(candidates.shape, -1, jnp.int32))
    if tile_consume is not None:
        output = (output, None)
    pipeline = (
        functools.partial(
            indexer.pipeline_score_topk_bounded,
            scheduling_groups=tile_consume is None,
        )
        if bounded
        else indexer.pipeline_score_topk_unrolled
    )
    selected = pipeline(
        num_tiles,
        lambda tile: tile * chunk_rows,
        lambda start, group: score(
            start, chunk_rows, inputs, pipelined=True, group=group
        ),
        select,
        write,
        output,
    )
    state = None
    if tile_consume is not None:
        selected, state = selected
    published = candidates
    if publish_rows:
        selected, published = selected
        published = jnp.where(valid_requests[:, None], published, -1)
    selected = jnp.where(short, visible, selected)
    if consume is None:
        return selected, published
    if tile_consume is None:
        return finish(selected, published)
    return selected, published, state


def select_fp8_indexer_tiles(
    q: jax.Array,
    banks: tuple[jax.Array, ...],
    page_table: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    requests: jax.Array,
    batch_max: jax.Array,
    query_start_loc: jax.Array,
    layout_args: tuple[int, ...],
    topk: int,
    block_size: int,
    candidate_write: bool,
    sparse_logits: bool,
    unordered: bool,
    score_width: int,
    *,
    consume: "SelectionConsumer | None" = None,
    records: "tuple[jax.Array, jax.Array, int] | None" = None,
) -> tuple[jax.Array, ...]:
    """Select exact FP8 scores with a bounded, overlapped prefill working set.

    Returns selected [T, topk] logical indices and published candidates. With
    ``consume``, a third result is its final state: every selected row reaches
    ``consume(start, rows, state, following)`` exactly once, in ascending row
    ranges, with ``state`` None at the first call; it returns the new state
    and ``following``. Prefill tiles feed it as their selection completes,
    so its work overlaps the scoring and selection of later tiles.
    ``following`` is the input of the pipeline's next step (None at the
    end): what the consumer gates it on is issued before that step.
    With ``records`` (page table [R, C], row requests [T], page size), tiles
    selected on SparseCore pass ``rows`` as (physical records, valid counts),
    `csa2.cache.prepare_main_records` of the (short-bypassed) selection;
    other row ranges pass the selected indices.
    """
    arguments = (
        q,
        banks,
        page_table,
        weights,
        lengths,
        candidates,
        requests,
        batch_max,
        query_start_loc,
        layout_args,
        topk,
        block_size,
        candidate_write,
        sparse_logits,
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
        or chunk > q.shape[0]
        or q.shape[0] % chunk
        or candidate_write
        or sparse_logits
        or not unordered
        or topk > width
        or jax.default_backend() != "tpu"
    ):
        return _select_fp8_indexer_jax(*arguments, score_width, consume=consume)

    # Score only the batch's live keys: 16,384-key steps while one resident K
    # tile covers them, then powers of two up to the full score width.
    # Scoring all 524,288 columns for a 147K-token context (73,728 keys) took
    # 8.9 ms per 2,048-row chunk, against 1.8 ms for 57,344 keys in one tile.
    step = PACKED_FP8_WIDTH_STEP
    widths = list(range(width, PACKED_FP8_RESIDENT_KEYS + 1, step)) or [width]
    while widths[-1] < score_width:
        widths.append(min(2 * widths[-1], score_width))
    live = (requests >= 0) & (requests < page_table.shape[0])
    longest = jnp.where(live, lengths, 0).max()
    grouped = jnp.all(requests[1:] >= requests[:-1])
    fitting = jnp.searchsorted(jnp.asarray(widths, jnp.int32), longest)
    branch = jnp.where(grouped, jnp.minimum(fitting, len(widths) - 1), len(widths) - 1)
    return jax.lax.switch(
        branch,
        [
            functools.partial(
                _select_fp8_indexer_jax,
                *arguments,
                w,
                prefill_chunk_rows=chunk,
                consume=consume,
                records=records,
            )
            for w in widths
        ],
    )


def select_fp8_indexer_jax(
    q: jax.Array,
    banks: tuple[jax.Array, ...],
    page_table: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    requests: jax.Array,
    batch_max: jax.Array,
    query_start_loc: jax.Array,
    layout_args: tuple[int, ...],
    topk: int,
    block_size: int,
    candidate_write: bool,
    sparse_logits: bool,
    unordered: bool,
    score_width: int,
) -> tuple[jax.Array, jax.Array]:
    """The registered selection op: `select_fp8_indexer_tiles` without a
    consumer, whose signature holds only arrays and plain values."""
    return select_fp8_indexer_tiles(
        q,
        banks,
        page_table,
        weights,
        lengths,
        candidates,
        requests,
        batch_max,
        query_start_loc,
        layout_args,
        topk,
        block_size,
        candidate_write,
        sparse_logits,
        unordered,
        score_width,
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
