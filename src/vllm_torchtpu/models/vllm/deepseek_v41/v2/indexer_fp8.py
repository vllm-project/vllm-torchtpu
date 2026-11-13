# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed FP8 indexer: candidate preparation, scoring, selection and op registration."""

import functools
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import torch
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu

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
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.prefill_topk import (
    prefill_topk,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.sparse_logits import (
    candidate_windows,
    fp8_gathered_mqa_logits,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.tuned import (
    PACKED_FP8_MQA_CONFIG,
    PACKED_FP8_PREFILL_MAX_ROWS,
    PACKED_FP8_PREFILL_MAX_WIDTH,
    PACKED_FP8_PREFILL_ROWS,
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
    if (
        sparse_logits
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
) -> tuple[jax.Array, jax.Array]:
    layout = PackedFP8Layout(*layout_args)
    tokens, width = q.shape[0], score_width
    if tokens == 0:
        return jnp.empty((0, min(topk, width)), jnp.int32), candidates
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
    # Decode top-k scans only the blocks that can hold a winner. Publishing
    # layers rank the same SparseCore block maxima for their candidates.
    select_block = block_size if publish_rows else _BLOCK_TOPK_BLOCK
    block_select = (
        stream_select
        and prefill_chunk_rows is None
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
        # Pipelined tiles inline inside the outer model jit so scheduling
        # metadata reaches the Pallas call instead of the inner jit boundary.
        scorer = fp8_paged_mqa_logits.__wrapped__ if pipelined else fp8_paged_mqa_logits
        scores = scorer(
            queries,
            pool,
            row_weights.astype(jnp.float32),
            row_lengths,
            pages,
            query_start_loc=(
                query_offsets if shared else jnp.clip(query_offsets - start, 0, rows)
            ),
            max_model_len=capacity if stream_select else width,
            q_scale=jnp.ones_like(row_weights, dtype=jnp.float32),
            query_size=query_tile,
            cache_layout=layout,
            query_start=start if shared else 0,
            query_count=rows,
            reuse_keys=prefill_chunk_rows is not None and capacity == key_tile,
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
        return select_serial()

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

    def write(tile, selected, output):
        update = functools.partial(
            jax.lax.dynamic_update_slice_in_dim, start_index=tile * chunk_rows, axis=0
        )
        return jax.tree.map(update, output, selected)

    # Grouped requests need neither sorting nor output reordering, so no
    # runtime branch to the serial path: a branch costs an overlay load at its
    # entry. Score tiles may split a request; the kernel resumes it mid-group.
    output = jnp.full((tokens, topk), -1, jnp.int32)
    if publish_rows:
        output = (output, jnp.full(candidates.shape, -1, jnp.int32))
    pipeline = (
        indexer.pipeline_score_topk_bounded
        if prefill_chunk_rows is not None or publish_rows
        else indexer.pipeline_score_topk_unrolled
    )
    selected = pipeline(
        padded // chunk_rows,
        lambda tile: tile * chunk_rows,
        lambda start, group: score(
            start, chunk_rows, inputs, pipelined=True, group=group
        ),
        select,
        write,
        output,
    )
    published = candidates
    if publish_rows:
        selected, published = selected
        published = jnp.where(valid_requests[:, None], published, -1)
    # Batch-wide short-context bypass: every visible key, as in the reference.
    # A select rather than a cond keeps the scoring path free of branches.
    short = jnp.minimum(batch_max.max(), width) <= topk
    ids = jnp.arange(topk, dtype=jnp.int32)
    visible = jnp.where(ids < lengths[:tokens, None], ids, -1)
    return jnp.where(short, visible, selected), published


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

    def fallback():
        if (
            width >= score_width
            or chunk >= q.shape[0]
            or q.shape[0] % chunk
            or candidate_write
            or sparse_logits
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
            lambda: _select_fp8_indexer_jax(
                *arguments, width, prefill_chunk_rows=chunk
            ),
            lambda: _select_fp8_indexer_jax(
                *arguments, score_width, prefill_chunk_rows=chunk
            ),
        )

    fused_width = capacity if capacity <= PACKED_FP8_PREFILL_MAX_WIDTH else width
    if (
        jax.default_backend() == "tpu"
        and pltpu.get_tpu_info().generation == 7
        and PACKED_FP8_PREFILL_ROWS[0] <= q.shape[0] <= PACKED_FP8_PREFILL_MAX_ROWS
        and q.shape[0] > page_table.shape[0] * PACKED_FP8_MQA_CONFIG.query_tile
        and q.shape[0] % min(q.shape[0], chunk) == 0
        and topk <= score_width
        and q.shape[1:] == (32, 128)
        and q.dtype == jnp.float8_e4m3fn
        and layout.page_size in (128, 256)
        and fused_width <= PACKED_FP8_PREFILL_MAX_WIDTH
        and not candidate_write
        and not sparse_logits
        and unordered
        and streaming_topk.is_supported(
            fused_width, topk, reconstruct_threshold=topk_reconstruction_threshold(topk)
        )
    ):
        layout.validate(tuple(bank.shape[0] for bank in banks))
        if query_start_loc.shape != (page_table.shape[0] + 1,):
            raise ValueError("query_start_loc needs one boundary per request, plus one")
        valid = (requests >= 0) & (requests < page_table.shape[0])
        visible_lengths = jnp.where(valid, jnp.clip(lengths, 0, score_width), 0)
        offsets = jnp.minimum(query_start_loc.astype(jnp.int32), q.shape[0])
        offsets = offsets.at[-1].set(q.shape[0])

        # Keep the old scoring-call boundaries: regrouping six-query tiles
        # across them can change BF16 head-sum rounding.
        shared_rows = indexer_query_rows(q.shape[0], PACKED_FP8_MQA_CONFIG.query_tile)
        score_rows = chunk if chunk < q.shape[0] else shared_rows
        reference_rows = score_rows if score_rows < q.shape[0] else None
        # The old block selector returns stable ordered IDs even when its
        # caller permits arbitrary order. Preserve that stronger contract.
        ordered_blocks = chunk >= q.shape[0] and block_topk.is_supported(
            shared_rows,
            capacity,
            topk,
            _BLOCK_TOPK_BLOCK,
        )

        def fused():
            selected = prefill_topk(
                q,
                banks,
                weights.astype(jnp.float32),
                visible_lengths,
                page_table,
                offsets,
                requests,
                layout=layout,
                width=fused_width,
                k=topk,
                unordered=unordered and not ordered_blocks,
                reference_rows=reference_rows,
            )
            ids = jnp.arange(topk, dtype=jnp.int32)
            visible = jnp.where(ids < visible_lengths[:, None], ids, -1)
            short = jnp.minimum(batch_max.max(), score_width) <= topk
            return jnp.where(short, visible, selected), candidates

        # A padded decode bucket can have the same Q shape as a short prefill.
        # Use the real request boundaries, before extending trailing padding.
        prefill = jnp.max(jnp.diff(query_start_loc)) > PACKED_FP8_MQA_CONFIG.query_tile
        fits = visible_lengths.max() <= fused_width
        if chunk < q.shape[0]:
            # The legacy wide-score route narrows only monotone request IDs.
            # Keep that route for negative padding IDs to preserve its results.
            fits = fits & jnp.all(requests[1:] >= requests[:-1])
        return jax.lax.cond(prefill & fits, fused, fallback)
    return fallback()


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
