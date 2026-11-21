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

from vllm_torchtpu import envs

from .cache import PAGE_RECORDS, PAGE_ROWS, PackedFP8Layout
from .tuned import (
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
        candidates: [queries, candidate_count] logical block IDs; -1 is padding.
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
    fragment = min(block, layout.page_size)
    region_rows = (
        metadata.physical_block_ids.shape[1] // (PAGE_RECORDS // fragment) * PAGE_ROWS
    )
    if region_rows != candidate_region_rows(candidates.shape[1], block):
        raise ValueError(
            f"Candidate K regions of {region_rows} rows, expected "
            f"{candidate_region_rows(candidates.shape[1], block)}"
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
    """Pair consecutive queries of each request into groups that share gathered K.

    Queries are paired two by two inside each run of equal request IDs
    (padding rows, request -1, form their own runs); the last query of an
    odd-sized run stays alone. Each query keeps its own candidates and
    window, so the grouping changes which K bytes are gathered together, not
    any score. The number of groups is therefore at most
    `candidate_group_capacity(tokens, requests)`, about half the queries, and
    every per-group array (and the gathered K, one region per group) is sized
    for that many groups instead of one per query.

    Args:
        pages: Dense [requests, max_pages_per_request] physical page table;
            valid page counts vary by request, bounded by sequence lengths.
        candidates: [queries, candidate_count] logical block IDs; -1 is padding.
        requests: [queries] request IDs; each request's queries are contiguous.
        layout: Packed cache storage layout.
        block: Tokens per candidate block.
        width: Logical token range upper bound.

    Returns:
        Per-group physical fragments, per-query block IDs and window spans,
        grouping offsets and validity mask, all in input query order.
    """
    tokens, count = candidates.shape
    owners = jnp.clip(requests, 0, pages.shape[0] - 1)
    valid = (requests >= 0) & (requests < pages.shape[0])

    rows = jnp.arange(tokens, dtype=jnp.int32)
    boundary = requests[1:] != requests[:-1]
    begin_mask = jnp.concatenate((jnp.ones(1, jnp.bool_), boundary))
    end_mask = jnp.concatenate((boundary, jnp.ones(1, jnp.bool_)))
    # cummax/cummin lower to one reduce-window; associative_scan unrolls into
    # log2(tokens) levels of tiny serial slice/pad ops.
    first = lax.cummax(jnp.where(begin_mask, rows, 0))
    end = lax.cummin(jnp.where(end_mask, rows + 1, tokens), reverse=True)
    # Pair rows two by two from the start of their run; an odd run's last row
    # stays alone. Example: runs of 3, 2 and 1 rows give groups
    # [0, 1] [2] [3, 4] [5].
    length = end - first
    paired = (rows - first) < length - length % 2
    starts = ~paired | ((rows - first) % 2 == 0)
    groups = jnp.cumsum(starts.astype(jnp.int32)) - 1
    capacity = candidate_group_capacity(tokens, pages.shape[0])
    if tokens <= 4096:
        # Decode sizes: (capacity + 1) x tokens compares fuse into one cheap op.
        offsets = jnp.searchsorted(
            groups, jnp.arange(capacity + 1), side="left", method="compare_all"
        )
    else:
        # Group g begins at the g-th start row; trailing groups are empty at
        # tokens. Linear in tokens, where compare_all grows quadratically.
        (offsets,) = jnp.nonzero(starts, size=capacity + 1, fill_value=tokens)
    sizes = jnp.diff(offsets)

    direct = False
    if jax.default_backend() == "tpu":
        info = pltpu.get_tpu_info()
        if info.sparse_core is not None and count % info.num_lanes == 0:
            words = _metadata_scratch_words(
                2,
                count,
                pl.cdiv(width, block),
                pages.shape[1],
                offsets=offsets.size,
                requests=requests.size,
            )
            direct = (
                words * jnp.dtype(jnp.int32).itemsize
                <= info.sparse_core.vmem_capacity_bytes // 2
            )
    if direct:
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
        # Preserve per-query ownership when forming each pair's union.
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


def _metadata_scratch_words(
    queries: int,
    per_query: int,
    blocks: int,
    table_columns: int,
    *,
    offsets: int = 0,
    requests: int = 0,
) -> int:
    info = pltpu.get_tpu_info()
    sc = info.sparse_core
    assert sc is not None
    alignment = info.num_lanes

    def aligned(size):
        return pl.cdiv(size, alignment) * alignment

    control = (
        alignment + aligned(offsets) + aligned(requests) if requests else sc.num_lanes
    )
    return (
        queries * aligned(blocks)
        + aligned(table_columns)
        + 4 * (aligned(queries * per_query) + sc.num_lanes)
        + control
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
    info = pltpu.get_tpu_info().sparse_core if jax.default_backend() == "tpu" else None
    fits = (
        info is not None
        and _metadata_scratch_words(
            queries,
            per_query,
            blocks,
            table.shape[1],
            offsets=offsets.size if direct else 0,
            requests=requests.size if requests is not None else 0,
        )
        * jnp.dtype(jnp.int32).itemsize
        <= info.vmem_capacity_bytes // 2
    )
    if not fits:
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
            # Direct preparation writes one window per query, not per group.
            jax.ShapeDtypeStruct(
                (candidates.shape[0], per_query) if direct else (rows, columns),
                jnp.int32,
            ),
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


def _sorted_union_owners(keys, queries, per_query, blocks):
    """Sorted unique union of candidate blocks and its per-query owner bits.

    Gives the same result as sorting, deduplicating and then comparing every
    union block with every candidate of every query, without that
    [rows, union, per_query] compare (36 us per call at 4 rows and 72 us at
    8 rows on tpu7x for 2 queries of 2048 candidates). The owner bit of each
    query travels through both sorts as a payload: after the first sort equal
    blocks are adjacent, the first entry of each run ORs the bits of the next
    ``queries - 1`` entries of the same block, and the second sort compacts
    one entry per run. Assumes each query lists a block at most once (the
    candidates are the indices of a top-k).

    Args:
        keys: [rows, queries * per_query] int32 logical block ids, query-major,
            with ``blocks`` for every invalid entry.
        queries: Queries per row (at most 30, one bit each).
        per_query: Candidates per query.
        blocks: The invalid id, larger than every valid id.

    Returns:
        ([rows, queries * per_query] ascending unique ids followed by
        ``blocks``, [rows, queries * per_query] int32 owner bits, bit q set
        when query q lists the block; zero on ``blocks`` entries).
    """
    rows, count = keys.shape
    bits = jnp.broadcast_to(
        (1 << jnp.arange(queries, dtype=jnp.int32))[:, None], (queries, per_query)
    ).reshape(1, count)
    bits = jnp.broadcast_to(bits, (rows, count))
    keys, bits = lax.sort((keys, bits), dimension=1, num_keys=1)
    start = jnp.diff(keys, axis=-1, prepend=-1) != 0
    # Each query lists a block at most once, so a run of equal blocks has at
    # most `queries` entries, and the run's first entry (the one kept) sees
    # the whole run within the next `queries` positions.
    owners = bits
    for shift in range(1, queries):
        same = jnp.pad(keys[:, shift:] == keys[:, :-shift], ((0, 0), (0, shift)))
        later = jnp.pad(bits[:, shift:], ((0, 0), (0, shift)))
        owners = owners | jnp.where(same, later, 0)
    keys = jnp.where(start, keys, blocks)
    owners = jnp.where(keys < blocks, owners, 0)
    keys, owners = lax.sort((keys, owners), dimension=1, num_keys=1)
    return keys, owners


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
    if envs.TPU_DEEPSEEK_V41_CANDIDATE_SORTED_OWNERS and queries <= 30:
        candidate_block_ids, owners = _sorted_union_owners(
            jnp.where(valid, flat, blocks), queries, per_query, blocks
        )
    else:
        candidate_block_ids, owners = None, None
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
    if owners is None:
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


def _gather_config(block: int, page_size: int, num_banks: int) -> tuple[int, int]:
    """(ids per gather step, pipeline buffers) of `gather_candidate_keys`."""
    info = plsc.get_sparse_core_info()
    config = PAGED_MQA_FRAGMENT_GATHER_CONFIG.get((block, page_size, num_banks))
    if config is None:
        group = max(info.num_lanes, min(128, 1024 // block))
        buffers = min(
            2,
            info.vmem_capacity_bytes
            // (group * PAGE_ROWS * (block + num_banks) * 4 + 4096),
        )
        config = PAGED_MQA_GATHER_CONFIG.get((block, num_banks), (group, buffers))
    return config


def candidate_key_rows(
    groups: int, blocks: int, block: int, page_size: int, num_banks: int
) -> int:
    """Rows of the K array `gather_candidate_keys` returns.

    The gather kernel writes whole steps of ids, so its output is rounded up
    to the SparseCore units times the ids per step, past the
    ``groups * blocks // (PAGE_RECORDS // block) * PAGE_ROWS`` rows of the
    per-group regions. The array is returned untrimmed, because slicing the
    exact rows off copies the whole gathered K (8.25 GiB for 2,048 queries with
    16,384 candidate tokens each); consumers address each region by the region
    size (`candidate_region_rows`), not by the array size.

    Example: 2,048 groups of 4,096 blocks of 8 tokens, 16 blocks per page,
    32 units x 96 ids per step: the 8,388,608 ids are padded to 8,389,632, so
    the array has 524,352 pages (17,303,616 rows) for 524,288 region pages.

    Uses only integer arithmetic, so it also takes symbolic sizes (the
    fake implementation of the prepare op calls it during tracing).

    Args:
        groups: Query groups (rows of the physical block ids).
        blocks: Physical blocks per group (columns of the physical block ids,
            a multiple of the blocks per page).
        block: Tokens per gathered block, at most page_size.
        page_size: Tokens per cache page.
        num_banks: Physical cache banks.

    Returns:
        The number of rows of the returned K array.
    """
    info = plsc.get_sparse_core_info()
    units = info.num_cores * info.num_subcores
    group, _ = _gather_config(block, page_size, num_banks)
    per_page = PAGE_RECORDS // block
    steps = (groups * blocks + units * group - 1) // (units * group)
    return steps * group * units // per_page * PAGE_ROWS


def candidate_group_capacity(
    tokens: int, request_slots: int, minimum: Callable[[int, int], int] = min
) -> int:
    """Static upper bound on the query groups of `prepare_candidate_metadata`.

    Groups pair queries inside runs of equal request IDs, so a run of n rows
    makes ceil(n / 2) groups. Each request's rows are contiguous and padding
    rows trail them, so there are at most request_slots + 1 runs, and even
    if padding sat between every two requests there would be at most
    2 * request_slots + 1. The groups are then at most
    (tokens + 2 * request_slots + 1) / 2, that is at most
    (tokens + 1) // 2 + request_slots, and never more than one per query.

    Example: a 2,048-token prefill step with a page table of 8 request slots
    has at most 1,024 + 8 = 1,032 groups instead of 2,048, which halves
    the gathered candidate K (8.25 GiB to 4.16 GiB for 2,048 candidate blocks
    of 8 tokens per query).

    Uses only integer arithmetic, so it also takes symbolic sizes when the
    caller passes a symbolic minimum (``torch.sym_min`` in the fake
    implementation of the prepare op).

    Args:
        tokens: Query rows of the step.
        request_slots: Rows of the step's page table (request slots).
        minimum: The minimum of two sizes.

    Returns:
        The number of group rows every per-group array is sized for.
    """
    return minimum(tokens, (tokens + 1) // 2 + request_slots)


def candidate_region_rows(candidates_per_query: int, block: int) -> int:
    """Rows of one query group's region in the gathered candidate K.

    A group is a pair of queries, so its region holds up to twice one query's
    candidate blocks, in whole packed pages of PAGE_RECORDS tokens. Example:
    2,048 candidate blocks of 8 tokens per query give
    ceil(2 x 2,048 x 8 / 128) = 256 pages of PAGE_ROWS = 33 rows, 8,448 rows.
    """
    pages = (2 * candidates_per_query * block + PAGE_RECORDS - 1) // PAGE_RECORDS
    return pages * PAGE_ROWS


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
        Packed int32 K of shape [candidate_key_rows(...), PAGE_RECORDS]: equal
        contiguous regions per group, then unused rows; skipped tiles remain
        uninitialized.
    """
    info = plsc.get_sparse_core_info()
    lanes, units = info.num_lanes, info.num_cores * info.num_subcores
    group, buffers = _gather_config(block, layout.page_size, len(banks))
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

    # Untrimmed (see candidate_key_rows): the slice to the regions' rows
    # would copy the whole gathered K.
    return kernel(banks, ids, control)
