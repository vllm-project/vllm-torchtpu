# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare and gather packed FP8 candidate caches for cross-layer reuse."""

from collections.abc import Callable
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from .cache import PAGE_RECORDS, PAGE_ROWS, PackedFP8Layout
from .tuned import (
    PACKED_FP8_DIRECT_METADATA_SHAPES,
    PACKED_FP8_MATRIX_GROUPING_SHAPES,
    PAGED_MQA_FRAGMENT_GATHER_CONFIG,
    PAGED_MQA_GATHER_CONFIG,
)


class PreparedCandidates(NamedTuple):
    """Group-shared K and per-query mappings for candidate scoring."""

    keys: jax.Array  # Contiguous packed FP8 K region per group.
    # Per-query logical IDs: exclusive/shared or shared/exclusive; -1 is padding.
    candidate_block_ids: jax.Array
    offsets: jax.Array  # Group g owns queries [offsets[g], offsets[g + 1]).
    valid_queries: jax.Array  # Valid request-ID mask in input query order.
    block_spans: jax.Array  # Per-query [union start, sorted-segment split], in blocks.


class CandidateMetadata(NamedTuple):
    """Pre-gather metadata; remaining fields match PreparedCandidates."""

    # Per-group cache block IDs; each spans min(block, page_size) tokens.
    # -1 marks invalid entries or padding.
    physical_block_ids: jax.Array
    candidate_block_ids: jax.Array
    offsets: jax.Array
    valid_queries: jax.Array
    block_spans: jax.Array


def prepare_candidates(
    banks: tuple[jax.Array, ...],
    pages: jax.Array,
    candidates: jax.Array,
    requests_ids: jax.Array,
    layout: PackedFP8Layout,
    block: int,
    width: int,
) -> PreparedCandidates:
    """Gather packed FP8 K for candidate unions shared by query groups.

    Args:
        banks: Physical packed FP8 cache banks.
        pages: Dense [requests, max_pages_per_request] physical page table;
            valid page counts vary by request, bounded by sequence lengths.
        candidates: [queries, 2048] logical block IDs; -1 is padding.
        requests_ids: [queries] request IDs; each request's queries are contiguous.
        layout: Packed cache storage layout.
        block: Tokens per candidate block.
        width: Logical token range upper bound.

    Returns:
        PreparedCandidates with gathered K, logical block mappings, query group
        offsets, validity and the start/split of each query window.
    """
    metadata = prepare_candidate_metadata(
        pages, candidates, requests_ids, layout, block, width
    )
    keys = gather_candidate_keys(
        banks,
        metadata.physical_block_ids,
        metadata.offsets,
        block=min(block, layout.page_size),
        layout=layout,
    )
    return PreparedCandidates(keys, *metadata[1:])


def prepare_candidate_metadata(
    pages: jax.Array,
    candidates: jax.Array,
    requests: jax.Array,
    layout: PackedFP8Layout,
    block: int,
    width: int,
) -> CandidateMetadata:
    """Pair queries of even-sized requests; odd-sized requests stay unpaired.

    Args:
        pages: Dense [requests, max_pages_per_request] physical page table;
            valid page counts vary by request, bounded by sequence lengths.
        candidates: [queries, 2048] logical block IDs; -1 is padding.
        requests: [queries] request IDs; each request's queries are contiguous.
        layout: Packed cache storage layout.
        block: Tokens per candidate block.
        width: Logical token range upper bound.

    Returns:
        Per-group physical fragments, per-query block IDs and window spans,
        grouping offsets and validity mask, all in input query order.
    """
    # The caller supplies contiguous queries per request, followed by padding.
    tokens, count = candidates.shape
    owners = jnp.clip(requests, 0, pages.shape[0] - 1)
    valid = (requests >= 0) & (requests < pages.shape[0])

    # Pair adjacent queries in even-sized requests; leave odd-sized ones unpaired.
    rows = jnp.arange(tokens, dtype=jnp.int32)
    boundary = requests[1:] != requests[:-1]
    generation = (
        pltpu.get_tpu_info().generation if jax.default_backend() == "tpu" else 0
    )
    matrix_grouping = (generation, tokens) in PACKED_FP8_MATRIX_GROUPING_SHAPES
    begin_mask = jnp.concatenate((jnp.ones(1, jnp.bool_), boundary))
    end_mask = jnp.concatenate((boundary, jnp.ones(1, jnp.bool_)))
    if matrix_grouping:
        before = rows[None, :] <= rows[:, None]
        after = rows[None, :] >= rows[:, None]
        first = jnp.max(
            jnp.where(begin_mask[None, :] & before, rows[None, :], 0), axis=1
        )
        end = jnp.min(
            jnp.where(end_mask[None, :] & after, rows[None, :] + 1, tokens), axis=1
        )
    else:
        first = lax.associative_scan(jnp.maximum, jnp.where(begin_mask, rows, 0))
        end = lax.associative_scan(
            jnp.minimum, jnp.where(end_mask, rows + 1, tokens), reverse=True
        )
    paired = valid & ((end - first) % 2 == 0)
    starts = ~paired | ((rows - first) % 2 == 0)
    if matrix_grouping:
        before = rows[None, :] <= rows[:, None]
        groups = jnp.sum(starts[None, :] & before, axis=1, dtype=jnp.int32) - 1
        offsets = jnp.sum(
            groups[None, :] < jnp.arange(tokens + 1)[:, None],
            axis=1,
            dtype=jnp.int32,
        )
    else:
        groups = jnp.cumsum(starts.astype(jnp.int32)) - 1
        # Invert sorted group IDs without SparseCore scatter reductions.
        offsets = jnp.searchsorted(groups, jnp.arange(tokens + 1), side="left")
    sizes = jnp.diff(offsets)

    direct = (generation, tokens, count, width, block, layout.page_size)
    if direct in PACKED_FP8_DIRECT_METADATA_SHAPES:
        candidate_block_ids, physical_block_ids, counts = canonical_candidate_blocks(
            candidates,
            pages,
            offsets,
            block=block,
            width=width,
            cache_pages=layout.pages,
            page_size=layout.page_size,
            requests=requests,
        )
    else:
        # Stack each group's two queries; mask missing partners and empty groups.
        # Row q holds that query's candidates, so the kernel can attribute each
        # union block to the queries that requested it.
        representatives = offsets[:-1]
        representatives = jnp.minimum(representatives, tokens - 1)
        first_candidates = candidates[representatives]
        following = candidates[jnp.minimum(representatives + 1, tokens - 1)]
        second_candidates = jnp.where(sizes[:, None] == 2, following, -1)
        live_group = (sizes > 0)[:, None] & valid[representatives, None]
        group_candidates = jnp.stack(
            (
                jnp.where(live_group, first_candidates, -1),
                jnp.where(live_group, second_candidates, -1),
            ),
            axis=1,
        )

        # Compact into A-only/shared/B-only segments, ascending within each.
        # width bounds logical Index-K record positions, not candidate count.
        candidate_block_ids, physical_block_ids, counts = canonical_candidate_blocks(
            group_candidates,
            pages[owners[representatives]],
            offsets,
            block=block,
            width=width,
            cache_pages=layout.pages,
            page_size=layout.page_size,
        )

        member = rows - offsets[groups]
        start = jnp.where(member == 0, 0, counts[groups, 0])
        size = jnp.where(member == 0, counts[groups, 0], counts[groups, 2])
        size += counts[groups, 1]
        slots = start[:, None] + jnp.arange(count)
        candidate_block_ids = jnp.where(
            jnp.arange(count) < size[:, None],
            jnp.take_along_axis(candidate_block_ids[groups], slots, axis=1),
            -1,
        )

    member = rows - offsets[groups]
    spans = jnp.stack(
        (
            jnp.where(member == 0, 0, counts[groups, 0]),
            jnp.where(member == 0, counts[groups, 0], counts[groups, 1]),
        ),
        axis=1,
    )

    # SparseCore metadata only initializes live group rows.
    physical_block_ids = jnp.where(sizes[:, None] > 0, physical_block_ids, -1)
    # Align fragment capacity so each group's gathered K fills packed pages.
    fragment = min(block, layout.page_size)
    per_page = PAGE_RECORDS // fragment
    padded = (physical_block_ids.shape[1] + per_page - 1) // per_page * per_page
    physical_block_ids = jnp.pad(
        physical_block_ids,
        ((0, 0), (0, padded - physical_block_ids.shape[1])),
        constant_values=-1,
    )
    return CandidateMetadata(
        physical_block_ids,
        candidate_block_ids,
        offsets,
        valid,
        spans,
    )


def canonical_candidate_blocks(
    candidates: jax.Array,
    table: jax.Array,
    offsets: jax.Array,
    *,
    block: int,
    width: int,
    cache_pages: int,
    page_size: int = PAGE_RECORDS,
    requests: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Deduplicate and partition each pair into A-only/shared/B-only blocks.

    Each member occupies one contiguous window in the gathered union;
    each of its two segments retains ascending logical IDs.

    Args:
        candidates: [groups, queries, per_query] logical candidate block IDs;
            [queries, per_query] when requests is supplied.
        table: [groups, max_pages_per_request] physical page table rows.
        offsets: Query boundaries identifying live group rows.
        block: Tokens per candidate block.
        width: Logical token range upper bound.
        cache_pages: Physical cache page count.
        page_size: Tokens per cache page.
        requests: Original query request IDs for direct metadata preparation.

    Returns:
        Union IDs (per-query windows for direct preparation), physical
        fragment IDs, and per-group [A-only, shared, B-only] counts.
        Invalid IDs are -1; trailing empty groups remain unset.
    """
    blocks = pl.cdiv(width, block)
    direct = requests is not None
    if direct:
        rows, queries, per_query = offsets.shape[0] - 1, 2, candidates.shape[1]
    else:
        rows, queries, per_query = candidates.shape
    count = queries * per_query
    # Keep half of subcore VMEM free for compiler temporaries and alignment.
    scratch_words = (
        queries * pl.cdiv(blocks, 128)
        + pl.cdiv(table.shape[1], 128)
        + 4 * pl.cdiv(count, 128)
        + 3
    ) * 128 + 128
    if requests is not None:
        scratch_words += (
            pl.cdiv(offsets.size, 128) + pl.cdiv(requests.size, 128) + 1
        ) * 128
    if jax.default_backend() != "tpu" or (
        scratch_words * 4 > plsc.get_sparse_core_info().vmem_capacity_bytes // 2
    ):
        if direct:
            raise ValueError("Direct candidate metadata exceeds its TPU scratch budget")
        return _canonical_candidate_blocks_ref(
            candidates, table, block, width, cache_pages, page_size
        )

    info = plsc.get_sparse_core_info()
    lanes, units = info.num_lanes, info.num_cores * info.num_subcores
    alignment = pltpu.get_tpu_info().num_lanes
    columns = pl.cdiv(count, alignment) * alignment
    histogram_width = pl.cdiv(blocks, alignment) * alignment
    if not direct:
        candidates = jnp.pad(
            candidates.reshape(rows, count),
            ((0, 0), (0, columns - count)),
            constant_values=-1,
        )
    elif per_query % alignment:
        raise ValueError("Direct candidate metadata requires lane-aligned query rows")
    table = jnp.pad(
        table, ((0, 0), (0, -table.shape[1] % alignment)), constant_values=-1
    )
    live = jnp.max(jnp.where(offsets[1:] > offsets[:-1], jnp.arange(1, rows + 1), 0))
    control = jnp.full((lanes,), live, jnp.int32)
    offset_width = pl.cdiv(offsets.size, alignment) * alignment
    if requests is not None:
        control = jnp.concatenate(
            (
                jnp.full((alignment,), live, jnp.int32),
                jnp.pad(
                    offsets,
                    (0, offset_width - offsets.size),
                    constant_values=candidates.shape[0],
                ),
                jnp.pad(requests, (0, -requests.size % alignment), constant_values=-1),
            )
        )
    per_page, parts = max(1, page_size // block), max(1, block // page_size)
    mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

    @pl.kernel(
        out_type=(
            jax.ShapeDtypeStruct((rows, per_query if direct else columns), jnp.int32),
            jax.ShapeDtypeStruct((rows, columns), jnp.int32),
            jax.ShapeDtypeStruct((rows, alignment), jnp.int32),
        ),
        scratch_types=(
            pltpu.VMEM((queries * histogram_width,), jnp.int32),
            pltpu.VMEM((columns + lanes,), jnp.int32),
            pltpu.VMEM((table.shape[1],), jnp.int32),
            pltpu.VMEM((columns + lanes,), jnp.int32),
            pltpu.VMEM((columns + lanes,), jnp.int32),
            pltpu.VMEM((columns + lanes,), jnp.int32),
            pltpu.VMEM((control.size,), jnp.int32),
            pltpu.SemaphoreType.DMA((5,)),
        ),
        mesh=mesh,
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
        name="paged_mqa_candidate_metadata",
    )
    def prepare(
        src,
        pages,
        control_ref,
        candidate_block_ids,
        physical_block_ids,
        owners,
        hist,
        ids,
        page_ids,
        logical_ids,
        physical_block_id_buffer,
        owner_ids,
        active,
        sem,
    ):
        pltpu.async_copy(control_ref, active, sem.at[0]).wait()
        unit = lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))

        @pl.loop(unit, active[pl.ds(0, lanes)][0], step=units)
        def row(row_id):
            row_start, row_queries, row_valid = row_id, queries, jnp.bool_(True)
            if direct:

                def scalar(position):
                    return plsc.load_gather(
                        active, [jnp.full(lanes, position, jnp.int32)]
                    )[0]

                row_start = scalar(alignment + row_id)
                row_queries = scalar(alignment + row_id + 1) - row_start
                request = scalar(alignment + offset_width + row_start)
                row_valid = (request >= 0) & (request < pages.shape[0])
                read_ids = tuple(
                    pltpu.async_copy(
                        src.at[jnp.minimum(row_start + query, src.shape[0] - 1)],
                        ids.at[pl.ds(query * per_query, per_query)],
                        sem.at[0],
                    )
                    for query in range(queries)
                )
                page_row = jnp.clip(request, 0, pages.shape[0] - 1)
            else:
                read_ids = (
                    pltpu.async_copy(src.at[row_id], ids.at[:columns], sem.at[0]),
                )
                page_row = row_id
            read_pages = pltpu.async_copy(pages.at[page_row], page_ids, sem.at[1])

            @plsc.parallel_loop(
                0, queries * histogram_width, lanes, unroll=alignment // lanes
            )
            def clear_hist(i):
                hist[pl.ds(i, lanes)] = jnp.zeros(lanes, jnp.int32)

            @plsc.parallel_loop(0, columns + lanes, lanes)
            def clear_result(i):
                logical_ids[pl.ds(i, lanes)] = jnp.full(lanes, -1, jnp.int32)
                owner_ids[pl.ds(i, lanes)] = jnp.zeros(lanes, jnp.int32)

            for copy in read_ids:
                copy.wait()
            read_pages.wait()

            # One presence histogram per query; a block's owner bit is set when
            # that query lists it. ``scan_count`` dedupes repeats within a query.
            @plsc.parallel_loop(0, columns, lanes, unroll=alignment // lanes)
            def scatter(i):
                keys = ids[pl.ds(i, lanes)]
                query = (i + jnp.arange(lanes)) // per_query
                base = query * histogram_width
                inc, mask = plsc.scan_count(
                    keys,
                    mask=(keys >= 0)
                    & (keys < blocks)
                    & (query < row_queries)
                    & row_valid,
                )
                plsc.addupdate_scatter(
                    hist,
                    [base + jnp.clip(keys, 0, histogram_width - 1)],
                    inc,
                    mask=mask,
                )

            def compact(i, count):
                logical = i + jnp.arange(lanes)
                owner = jnp.zeros(lanes, jnp.int32)
                for query in range(queries):
                    present = hist[pl.ds(query * histogram_width + i, lanes)] != 0
                    owner = owner | (present.astype(jnp.int32) << query)
                page = plsc.load_gather(
                    page_ids,
                    [jnp.minimum(logical // per_page * parts, table.shape[1] - 1)],
                )
                valid = (
                    (owner != 0)
                    & (logical < blocks)
                    & (logical * block < table.shape[1] * page_size)
                    & (page >= 0)
                    & (page < cache_pages)
                )
                for part in range(1, parts):
                    column = logical * parts + part
                    following = plsc.load_gather(
                        page_ids, [jnp.minimum(column, table.shape[1] - 1)]
                    )
                    valid &= (logical * block + part * page_size >= width) | (
                        (column < table.shape[1])
                        & (following >= 0)
                        & (following < cache_pages)
                    )
                plsc.store_compressed(
                    logical_ids.at[pl.ds(count, lanes)], logical, mask=valid
                )
                plsc.store_compressed(
                    owner_ids.at[pl.ds(count, lanes)], owner, mask=valid
                )
                return count + plsc.all_reduce_population_count(valid)[0]

            loop = cast(
                Callable[[Callable[[jax.Array, jax.Array], jax.Array]], jax.Array],
                plsc.parallel_loop(
                    0,
                    histogram_width,
                    lanes,
                    # Fully unrolling one histogram vector miscompiles the
                    # page-ID gather in libtpu 0.0.50.
                    unroll=1 if histogram_width == alignment else alignment // lanes,
                    carry=jnp.int32(0),
                ),
            )
            union_size = loop(compact)
            ends = []
            position = jnp.int32(0)
            for ownership in (1, 3, 2):

                def partition(i, count, ownership=ownership):
                    logical = logical_ids[pl.ds(i, lanes)]
                    mask = (i + jnp.arange(lanes) < union_size) & (
                        owner_ids[pl.ds(i, lanes)] == ownership
                    )
                    plsc.store_compressed(
                        ids.at[pl.ds(count, lanes)], logical, mask=mask
                    )
                    return count + plsc.all_reduce_population_count(mask)[0]

                position = plsc.parallel_loop(
                    0,
                    pl.cdiv(union_size, lanes) * lanes,
                    lanes,
                    carry=position,
                    unroll=alignment // lanes,
                )(partition)
                ends.append(position)

            # Compressed stores may touch the last vector past the live prefix.
            @plsc.parallel_loop(0, columns, lanes)
            def pad(i):
                live = i + jnp.arange(lanes) < union_size
                logical = ids[pl.ds(i, lanes)]
                page = plsc.load_gather(
                    page_ids,
                    [jnp.clip(logical // per_page * parts, 0, table.shape[1] - 1)],
                )
                ids[pl.ds(i, lanes)] = jnp.where(live, logical, -1)
                physical_block_id_buffer[pl.ds(i, lanes)] = jnp.where(
                    live, page * per_page + logical % per_page, -1
                )

            if direct:
                for query in range(queries):

                    @pl.when(query < row_queries)
                    def publish_query(query=query):
                        start = jnp.int32(0) if query == 0 else ends[0]
                        length = ends[1] if query == 0 else ends[2] - ends[0]

                        @plsc.parallel_loop(
                            0, per_query, lanes, unroll=alignment // lanes
                        )
                        def map_query(i):
                            slots = i + jnp.arange(lanes)
                            values = plsc.load_gather(ids, [start + slots])
                            logical_ids[pl.ds(i, lanes)] = jnp.where(
                                slots < length, values, -1
                            )

                        pltpu.async_copy(
                            logical_ids.at[:per_query],
                            candidate_block_ids.at[row_start + query],
                            sem.at[2],
                        ).wait()
            else:
                pltpu.async_copy(
                    ids.at[:columns], candidate_block_ids.at[row_id], sem.at[2]
                ).wait()
            owner_ids[pl.ds(0, lanes)] = jnp.where(
                jnp.arange(lanes) == 0,
                ends[0],
                jnp.where(
                    jnp.arange(lanes) == 1,
                    ends[1] - ends[0],
                    jnp.where(jnp.arange(lanes) == 2, ends[2] - ends[1], 0),
                ),
            )
            copy_physical = pltpu.async_copy(
                physical_block_id_buffer.at[:columns],
                physical_block_ids.at[row_id],
                sem.at[3],
            )
            copy_counts = pltpu.async_copy(
                owner_ids.at[:alignment], owners.at[row_id], sem.at[4]
            )
            copy_physical.wait()
            copy_counts.wait()

    candidate_block_ids, physical_block_ids, owners = prepare(
        candidates, table, control
    )
    candidate_block_ids = candidate_block_ids[:, : per_query if direct else count]
    physical_block_ids, owners = physical_block_ids[:, :count], owners[:, :3]
    if parts > 1:
        physical_block_ids = resolve_candidate_blocks(
            candidate_block_ids,
            table,
            sparse_block_kv=block,
            cache_pages=cache_pages,
            page_size=page_size,
        )
    return candidate_block_ids, physical_block_ids, owners


def _canonical_candidate_blocks_ref(
    candidates, table, block, width, cache_pages, page_size
):
    """Reference partition into three sorted segments on host XLA."""
    rows, queries, per_query = candidates.shape
    blocks = pl.cdiv(width, block)
    flat = candidates.reshape(rows, queries * per_query)
    physical_block_ids = resolve_candidate_blocks(
        flat,
        table,
        sparse_block_kv=block,
        cache_pages=cache_pages,
        page_size=page_size,
    )
    parts = max(1, block // page_size)
    needed = flat[..., None] * block + jnp.arange(parts) * min(block, page_size) < width
    valid = jnp.all(
        (physical_block_ids.reshape(*flat.shape, parts) >= 0) | ~needed, axis=-1
    )
    valid &= (flat >= 0) & (flat < blocks)
    candidate_block_ids = jnp.sort(jnp.where(valid, flat, blocks), axis=-1)
    unique = jnp.diff(candidate_block_ids, axis=-1, prepend=-1) != 0
    candidate_block_ids = jnp.sort(
        jnp.where(unique, candidate_block_ids, blocks), axis=-1
    )
    candidate_block_ids = jnp.where(
        candidate_block_ids < blocks, candidate_block_ids, -1
    )
    physical_block_ids = resolve_candidate_blocks(
        candidate_block_ids,
        table,
        sparse_block_kv=block,
        cache_pages=cache_pages,
        page_size=page_size,
    )
    # Pack per-query ownership: bit q set when query q lists the union block.
    owners = jnp.zeros_like(candidate_block_ids)
    for query in range(queries):
        present = (
            (candidate_block_ids[:, :, None] == candidates[:, query, None, :])
            & (candidates[:, query, None, :] >= 0)
        ).any(axis=-1)
        owners = owners | (present.astype(jnp.int32) << query)
    owners = jnp.where(candidate_block_ids >= 0, owners, 0)
    category = jnp.where(
        owners == 1, 0, jnp.where(owners == 3, 1, jnp.where(owners == 2, 2, 3))
    )
    order = jnp.argsort(category, axis=1, stable=True)
    candidate_block_ids = jnp.take_along_axis(candidate_block_ids, order, axis=1)
    physical_block_ids = jnp.take_along_axis(
        physical_block_ids.reshape(rows, -1, parts), order[..., None], axis=1
    ).reshape(rows, -1)
    counts = jnp.stack([(owners == flag).sum(axis=1) for flag in (1, 3, 2)], axis=1)
    return candidate_block_ids, physical_block_ids, counts


def _resolve_blocks_sc(blocks, table, block, cache_pages, page_size):
    """Resolve sub-page or page-sized blocks using SparseCore.

    Args:
        blocks: [groups, slots] logical block IDs.
        table: [groups, max_pages_per_request] physical page table rows.
        block: Tokens per block; divides page_size.
        cache_pages: Physical cache page count.
        page_size: Tokens per cache page.

    Returns:
        Physical block IDs shaped like blocks; invalid entries are -1.
    """
    info = plsc.get_sparse_core_info()
    lanes, units = info.num_lanes, info.num_cores * info.num_subcores
    rows, count = blocks.shape
    table_width, per_page = table.shape[1], page_size // block
    columns = pl.cdiv(count, 128) * 128
    blocks = jnp.pad(blocks, ((0, 0), (0, columns - count)), constant_values=-1)
    table = jnp.pad(table, ((0, 0), (0, -table_width % 128)), constant_values=-1)
    mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

    @pl.kernel(
        out_type=jax.ShapeDtypeStruct(blocks.shape, jnp.int32),
        scratch_types=(
            pltpu.VMEM((columns,), jnp.int32),
            pltpu.VMEM((table.shape[1],), jnp.int32),
            pltpu.VMEM((columns,), jnp.int32),
            pltpu.SemaphoreType.DMA((3,)),
        ),
        mesh=mesh,
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
        name="paged_mqa_candidate_addresses",
    )
    def resolve(src, pages, out, ids, page_ids, physical_block_ids, sem):
        unit = lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))

        @pl.loop(unit, rows, step=units)
        def row(row_id):
            read_ids = pltpu.async_copy(src.at[row_id], ids, sem.at[0])
            read_pages = pltpu.async_copy(pages.at[row_id], page_ids, sem.at[1])
            read_ids.wait()
            read_pages.wait()

            @plsc.parallel_loop(0, columns, lanes, unroll=8)
            def address(i):
                selected = ids[pl.ds(i, lanes)]
                safe = jnp.clip(selected, 0, table_width * per_page - 1)
                page = plsc.load_gather(page_ids, [safe // per_page])
                valid = (
                    (selected >= 0)
                    & (selected < table_width * per_page)
                    & (page >= 0)
                    & (page < cache_pages)
                )
                physical_block_ids[pl.ds(i, lanes)] = jnp.where(
                    valid, page * per_page + safe % per_page, -1
                )

            pltpu.async_copy(physical_block_ids, out.at[row_id], sem.at[2]).wait()

    return resolve(blocks, table)[:, :count]


@jax.jit(static_argnames=("sparse_block_kv", "cache_pages", "page_size"))
def resolve_candidate_blocks(
    sparse_blocks: jax.Array,
    block_table: jax.Array,
    *,
    sparse_block_kv: int,
    cache_pages: int,
    page_size: int = PAGE_RECORDS,
) -> jax.Array:
    """Resolve request-shared [G,C] blocks, preserving order and duplicates.

    Blocks spanning pages expand into page-sized fragments in candidate order.
    Per-query causal lengths are applied by the scorer, not this shared metadata.

    Args:
        sparse_blocks: [groups, slots] logical candidate block IDs.
        block_table: [groups, max_pages_per_request] physical page table rows.
        sparse_block_kv: Tokens per candidate block.
        cache_pages: Physical cache page count.
        page_size: Tokens per cache page.

    Returns:
        [groups, slots * max(1, sparse_block_kv // page_size)] physical fragment
        IDs, preserving order and duplicates; invalid entries are -1.
    """
    if sparse_block_kv > page_size:
        parts = sparse_block_kv // page_size
        sparse_blocks = jnp.where(
            sparse_blocks[..., None] >= 0,
            sparse_blocks[..., None] * parts + jnp.arange(parts),
            -1,
        ).reshape(sparse_blocks.shape[0], -1)
        sparse_block_kv = page_size
    per_page = page_size // sparse_block_kv
    limit = block_table.shape[1] * per_page
    safe = jnp.clip(sparse_blocks, 0, limit - 1)
    scratch_words = (
        2 * pl.cdiv(sparse_blocks.shape[1], 128) + pl.cdiv(block_table.shape[1], 128)
    ) * 128
    if jax.default_backend() == "tpu" and (
        scratch_words * 4 <= plsc.get_sparse_core_info().vmem_capacity_bytes // 2
    ):
        return _resolve_blocks_sc(
            sparse_blocks, block_table, sparse_block_kv, cache_pages, page_size
        )
    pages = jnp.take_along_axis(block_table, safe // per_page, axis=1)
    valid = (
        (sparse_blocks >= 0)
        & (sparse_blocks < limit)
        & (pages >= 0)
        & (pages < cache_pages)
    )
    return jnp.where(valid, pages * per_page + safe % per_page, -1)


def gather_candidate_keys(
    banks: tuple[jax.Array, ...],
    physical_block_ids: jax.Array,
    offsets: jax.Array,
    *,
    block: int,
    layout: PackedFP8Layout,
) -> jax.Array:
    """Gather compact pages, skipping empty tiles inside SparseCore.

    Args:
        banks: Physical packed FP8 cache banks.
        physical_block_ids: [groups, blocks] cache block IDs; -1 is padding.
        offsets: Query boundaries identifying live group rows.
        block: Tokens per physical block, at most layout.page_size.
        layout: Packed cache storage layout.

    Returns:
        Packed int32 K of shape [packed_rows, PAGE_RECORDS], with equal-sized
        contiguous regions per group; skipped tiles remain uninitialized.
    """
    info = plsc.get_sparse_core_info()
    lanes, units = info.num_lanes, info.num_cores * info.num_subcores
    config = PAGED_MQA_FRAGMENT_GATHER_CONFIG.get((block, layout.page_size, len(banks)))
    if config is None:
        group = max(lanes, min(128, 1024 // block))
        buffers = min(
            2,
            info.vmem_capacity_bytes
            // (group * PAGE_ROWS * (block + len(banks)) * 4 + 4096),
        )
        config = PAGED_MQA_GATHER_CONFIG.get((block, len(banks)), (group, buffers))
    group, buffers = config
    count = physical_block_ids.size
    per_page = PAGE_RECORDS // block
    source_per_page = layout.page_size // block
    span = group * PAGE_ROWS
    per_unit = pl.cdiv(count, units * group) * group
    steps = per_unit // group
    scratch_words = buffers * (span * (block + len(banks)) + lanes) + lanes
    metadata_rows = min(
        pl.cdiv(per_unit, group * lanes) * group * lanes,
        (info.vmem_capacity_bytes // 4 - scratch_words - 4096)
        // (group * lanes)
        * group
        * lanes,
    )
    meta_steps = metadata_rows // group
    ids = jnp.pad(
        physical_block_ids.reshape(-1),
        (0, units * per_unit - count),
        constant_values=-1,
    ).reshape(-1, group)
    live_rows = jnp.max(
        jnp.where(
            offsets[1:] > offsets[:-1], jnp.arange(physical_block_ids.shape[0]) + 1, 0
        )
    )
    control = jnp.full(
        (lanes,),
        pl.cdiv(live_rows * physical_block_ids.shape[1], units * group),
        jnp.int32,
    )
    # Rotate ownership across rows to spread their partially populated tails.
    rotate = physical_block_ids.shape[1] % (units * group) == 0
    steps_per_group = max(1, physical_block_ids.shape[1] // (units * group))
    mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

    @pl.kernel(
        out_type=jax.ShapeDtypeStruct(
            (units * per_unit // per_page * PAGE_ROWS, PAGE_RECORDS), jnp.int32
        ),
        scratch_types=(
            pltpu.VMEM((meta_steps, group), jnp.int32),
            pltpu.VMEM((meta_steps,), jnp.int32),
            pltpu.VMEM((lanes,), jnp.int32),
            pltpu.VMEM((buffers, lanes), jnp.int32),
            pltpu.VMEM((buffers, span, block), jnp.int32),
            pltpu.VMEM((buffers, len(banks), span), jnp.int32),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((buffers, len(banks))),
            pltpu.SemaphoreType.DMA((buffers,)),
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False, needs_layout_passes=False
        ),
        mesh=mesh,
        name="paged_mqa_candidate_gather_direct",
    )
    def kernel(
        pool,
        physical_block_ids_ref,
        control_ref,
        out,
        selected,
        metadata_ids,
        ctrl,
        flags,
        words,
        read_ids,
        meta_sem,
        read_sem,
        write_sem,
    ):
        unit = lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))
        pltpu.async_copy(control_ref, ctrl, meta_sem).wait()
        active = ctrl[...][0]

        def destination(step):
            return step * units + (
                (unit + step // steps_per_group) % units if rotate else unit
            )

        # Direct indexed metadata DMA avoids an external transpose and flags.
        def load_metadata(start):
            for i in range(0, meta_steps, lanes):
                row = start + i + jnp.arange(lanes, dtype=jnp.int32)
                metadata_ids[pl.ds(i, lanes)] = jnp.where(
                    row < steps, destination(row), -1
                )
            pltpu.async_copy(
                physical_block_ids_ref.at[plsc.Indices(metadata_ids, ignored_value=-1)],
                selected,
                meta_sem,
            ).wait()

        load_metadata(jnp.int32(0))

        def live(step):
            return flags[step % buffers, :][0] != 0

        def read(step, slot, wait=False):
            if not wait:
                index = step if metadata_rows >= per_unit else step % meta_steps
                if metadata_rows < per_unit:

                    @pl.when((step > 0) & (index == 0))
                    def next_metadata():
                        load_metadata(step)

                any_valid = jnp.zeros((lanes,), jnp.bool_)

                def addresses(chosen):
                    safe = jnp.maximum(chosen, 0)
                    row = (
                        layout.page_offset
                        + safe // source_per_page * layout.page_stride
                    )
                    column = safe % source_per_page
                    bank_base = 0
                    for bank, ref in enumerate(pool):
                        valid = (
                            (chosen >= 0)
                            & (row >= bank_base)
                            & (row + layout.page_rows <= bank_base + ref.shape[0])
                        )
                        source = (row - bank_base) * per_page + column
                        yield bank, source, valid
                        bank_base += ref.shape[0]

                if per_page == lanes:
                    for i in range(0, group, lanes):
                        chosen = selected.reshape(-1)[pl.ds(index * group + i, lanes)]
                        any_valid = any_valid | (chosen >= 0)
                        for bank, base, valid in addresses(chosen):
                            for field in range(PAGE_ROWS):
                                read_ids[
                                    slot,
                                    bank,
                                    pl.ds(i * PAGE_ROWS + field * lanes, lanes),
                                ] = jnp.where(valid, base + field * source_per_page, -1)
                else:
                    for i in range(0, group, lanes):
                        any_valid |= (
                            selected.reshape(-1)[pl.ds(index * group + i, lanes)] >= 0
                        )

                    @plsc.parallel_loop(0, span, lanes, unroll=8)
                    def fragment_addresses(i):
                        segment = i + jnp.arange(lanes)
                        field = segment // per_page % PAGE_ROWS
                        column = (
                            segment // (PAGE_ROWS * per_page) * per_page
                            + segment % per_page
                        )
                        chosen = plsc.load_gather(
                            selected.reshape(-1), [index * group + column]
                        )
                        for bank, source, valid in addresses(chosen):
                            read_ids[slot, bank, pl.ds(i, lanes)] = jnp.where(
                                valid, source + field * source_per_page, -1
                            )

                flags[slot, :] = (
                    plsc.all_reduce_population_count(any_valid) != 0
                ).astype(jnp.int32)

            @pl.when(live(step))
            def transfer():
                for bank, ref in enumerate(pool):
                    copy = pltpu.make_async_copy(
                        ref.reshape(-1, block).at[
                            plsc.Indices(read_ids.at[slot, bank], ignored_value=-1)
                        ],
                        words.at[slot],
                        read_sem.at[slot, bank],
                    )
                    (copy.wait if wait else copy.start)()

        def write(step, wait=False):
            slot = step % buffers

            @pl.when(live(step))
            def transfer():
                target = out.reshape(-1, block).at[
                    pl.ds(destination(step) * span, span)
                ]
                copy = pltpu.make_async_copy(words.at[slot], target, write_sem.at[slot])
                (copy.wait if wait else copy.start)()

        for step in range(min(buffers - 1, steps)):
            pl.when(step < active)(
                lambda step=step: read(jnp.int32(step), jnp.int32(step))
            )

        @pl.loop(0, active)
        def body(step):
            if buffers == 1:
                read(step, 0)
            read(step, step % buffers, wait=True)
            write(step)
            if buffers == 1:
                write(step, wait=True)

            @pl.when((buffers > 1) & (step + buffers - 1 < active))
            def prefetch():
                future = step + buffers - 1
                pl.when(future >= buffers)(lambda: write(future - buffers, wait=True))
                read(future, future % buffers)

        for last in range(buffers if buffers > 1 else 0):
            pl.when(last < active)(
                lambda last=last: write(active - last - 1, wait=True)
            )

    return kernel(banks, ids, control)[: count // per_page * PAGE_ROWS]
