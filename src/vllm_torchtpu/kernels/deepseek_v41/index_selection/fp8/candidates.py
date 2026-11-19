# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare and gather packed FP8 candidate caches for cross-layer reuse."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

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
    *,
    ascending: bool = False,
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
        ascending: Whether candidate rows already ascend with -1 suffix
            padding, as candidate publication emits them; others are sorted.

    Returns:
        PreparedCandidates with gathered K, logical block mappings, query group
        offsets, validity and the start/split of each query window.
    """
    metadata = prepare_candidate_metadata(
        pages, candidates, requests_ids, layout, block, width, ascending=ascending
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
    *,
    ascending: bool = False,
) -> CandidateMetadata:
    """Pair queries of even-sized requests; odd-sized requests stay unpaired.

    Args:
        pages: Dense [requests, max_pages_per_request] physical page table;
            valid page counts vary by request, bounded by sequence lengths.
        candidates: [queries, candidate_count] logical block IDs; -1 is padding.
        requests: [queries] request IDs; each request's queries are contiguous.
        layout: Packed cache storage layout.
        block: Tokens per candidate block.
        width: Logical token range upper bound.
        ascending: Whether candidate rows already ascend with -1 suffix
            padding, as candidate publication emits them; others are sorted.

    Returns:
        Per-group physical fragments, per-query block IDs and window spans,
        grouping offsets and validity mask, all in input query order.
    """
    tokens = candidates.shape[0]
    valid = (requests >= 0) & (requests < pages.shape[0])
    if not ascending:
        # The SparseCore kernel takes ascending rows with -1 suffix padding.
        limit = jnp.iinfo(jnp.int32).max
        candidates = jnp.sort(jnp.where(candidates >= 0, candidates, limit), axis=-1)
        candidates = jnp.where(candidates == limit, -1, candidates)

    rows = jnp.arange(tokens, dtype=jnp.int32)
    boundary = requests[1:] != requests[:-1]
    begin_mask = jnp.concatenate((jnp.ones(1, jnp.bool_), boundary))
    end_mask = jnp.concatenate((boundary, jnp.ones(1, jnp.bool_)))
    # cummax/cummin lower to one reduce-window; associative_scan unrolls into
    # log2(tokens) levels of tiny serial slice/pad ops.
    first = lax.cummax(jnp.where(begin_mask, rows, 0))
    end = lax.cummin(jnp.where(end_mask, rows + 1, tokens), reverse=True)
    paired = valid & ((end - first) % 2 == 0)
    starts = ~paired | ((rows - first) % 2 == 0)
    groups = jnp.cumsum(starts.astype(jnp.int32)) - 1
    if tokens <= 4096:
        # Decode sizes: (tokens + 1) x tokens compares fuse into one cheap op.
        offsets = jnp.searchsorted(
            groups, jnp.arange(tokens + 1), side="left", method="compare_all"
        )
    else:
        # Group g begins at the g-th start row; trailing groups are empty at
        # tokens. Linear in tokens, where compare_all grows quadratically.
        (offsets,) = jnp.nonzero(starts, size=tokens + 1, fill_value=tokens)
    sizes = jnp.diff(offsets)

    candidate_block_ids, physical_block_ids, counts = canonical_candidate_blocks(
        candidates,
        pages,
        offsets,
        requests,
        block=block,
        width=width,
        cache_pages=layout.pages,
        page_size=layout.page_size,
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
    requests: jax.Array,
    *,
    block: int,
    width: int,
    cache_pages: int,
    page_size: int = PAGE_RECORDS,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Deduplicate and partition each query pair into A-only/shared/B-only blocks.

    Each query of a pair marks its blocks in its own bitmap of blocks / 32
    words. A's blocks split by B's bits into A-only and shared, and B's by A's
    into B-only, so the work scales with the candidate count, not the width.

    Args:
        candidates: [queries, per_query] logical block IDs. Rows ascend with
            -1 suffix padding, as candidate publication emits them, so repeats
            are adjacent.
        table: [requests, max_pages_per_request] physical page table.
        offsets: Pair g owns queries [offsets[g], offsets[g + 1]).
        requests: [queries] request IDs; queries of invalid requests get no
            blocks.
        block: Tokens per candidate block; page_size // block is a power of two.
        width: Logical token range upper bound.
        cache_pages: Physical cache page count.
        page_size: Tokens per cache page.

    Returns:
        Per-query windows [queries, per_query] (A: A-only then shared; B:
        shared then B-only), per-pair physical union [pairs, 2 * per_query]
        (A-only, shared, B-only) and per-pair [A-only, shared, B-only] counts.
        Segments ascend; invalid IDs and padding are -1. Rows of trailing
        empty pairs remain unset.
    """
    sc = plsc.get_sparse_core_info()
    lanes, units = sc.num_lanes, sc.num_cores * sc.num_subcores
    alignment = pltpu.get_tpu_info().num_lanes
    unroll = alignment // lanes
    # TODO: retry 2 on libtpu > 0.0.50; it gave wrong windows with correct counts.
    classify_unroll = 4
    tokens, per_query = candidates.shape
    pairs = offsets.shape[0] - 1
    blocks = pl.cdiv(width, block)
    per_page = page_size // block
    shift = per_page.bit_length() - 1
    if per_query % alignment or page_size % block or per_page & (per_page - 1):
        raise NotImplementedError(
            f"Candidate metadata needs {alignment}-aligned candidate rows and "
            f"page_size // block a power of two; got per_query={per_query}, "
            f"block={block}, page_size={page_size}"
        )
    shr = lax.shift_right_logical
    table = jnp.pad(
        table, ((0, 0), (0, -table.shape[1] % alignment)), constant_values=-1
    )
    table_columns = table.shape[1]

    def aligned(size):
        return pl.cdiv(size, alignment) * alignment

    offset_words = aligned(offsets.size)
    # Words per scratch buffer; two-slot buffers hold two pairs in flight.
    bitmap_words = aligned(pl.cdiv(blocks, 32))
    words = dict(
        bits_a=bitmap_words,
        bits_b=bitmap_words,
        # Per slot: [pad][A][pad][B]; each pad ends a row's previous-ID compare.
        ids=2 * (2 * lanes + 2 * per_query),
        page_ids=2 * table_columns,
        # Compressed stores address `lanes` words past each live prefix. B's
        # window holds a pair, so rows that break the ascending contract
        # cannot write past it.
        window_a=2 * (per_query + lanes),
        window_b=2 * (2 * per_query + lanes),
        union=2 * (2 * per_query + lanes),
        shared_union=per_query + lanes,
        counts=2 * alignment,
        control=alignment + offset_words + aligned(tokens),
    )
    # Keep half of subcore VMEM free for compiler temporaries and alignment.
    if 4 * sum(words.values()) > sc.vmem_capacity_bytes // 2:
        raise ValueError(
            f"Candidate metadata for {tokens} queries at width {width} exceeds "
            "SparseCore VMEM; lower max_num_batched_tokens"
        )
    id_slot, a_slot, b_slot, union_slot = (
        words[name] // 2 for name in ("ids", "window_a", "window_b", "union")
    )
    a_base, b_base = lanes, 2 * lanes + per_query

    # [live pairs][offsets][requests], each lane-aligned.
    live = jnp.max(jnp.where(offsets[1:] > offsets[:-1], jnp.arange(1, pairs + 1), 0))
    request_base = alignment + offset_words
    control = jnp.concatenate(
        (
            jnp.full((alignment,), live, jnp.int32),
            jnp.pad(offsets, (0, offset_words - offsets.size), constant_values=tokens),
            jnp.pad(requests, (0, -tokens % alignment), constant_values=-1),
        )
    )
    mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")
    scratch = {name: pltpu.VMEM((size,), jnp.int32) for name, size in words.items()}

    @pl.kernel(
        out_type=(
            jax.ShapeDtypeStruct((tokens, per_query), jnp.int32),
            jax.ShapeDtypeStruct((pairs, 2 * per_query), jnp.int32),
            jax.ShapeDtypeStruct((pairs, alignment), jnp.int32),
        ),
        mesh=mesh,
        scratch_types=dict(scratch, sems=pltpu.SemaphoreType.DMA((2, 6))),
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
        name="paged_mqa_candidate_metadata",
    )
    def kernel(
        src,
        pages,
        control_hbm,
        out_ids,
        out_union,
        out_counts,
        *,
        bits_a,
        bits_b,
        ids,
        page_ids,
        window_a,
        window_b,
        union,
        shared_union,
        counts,
        control,
        sems,
    ):
        pltpu.async_copy(control_hbm, control, sems.at[0, 0]).wait()
        unit = lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))
        lane = jnp.arange(lanes, dtype=jnp.int32)
        zeros, ones = jnp.zeros(lanes, jnp.int32), jnp.ones(lanes, jnp.int32)

        def scalar(position):
            return plsc.load_gather(control, [jnp.full(lanes, position, jnp.int32)])[0]

        def count(mask):
            return plsc.all_reduce_population_count(mask)[0]

        def pair_info(step):
            pair = unit + step * units
            start = scalar(alignment + pair)
            queries = scalar(alignment + pair + 1) - start
            return pair, start, queries, scalar(request_base + start)

        def page_row(request):
            return jnp.clip(request, 0, pages.shape[0] - 1)

        def reads(step, wait=False):
            """Copy a pair's IDs, and its page row unless the slot holds it."""
            slot = step % 2
            _, start, _, request = pair_info(step)
            for query, base in enumerate((a_base, b_base)):
                copy = pltpu.make_async_copy(
                    src.at[jnp.minimum(start + query, tokens - 1)],
                    ids.at[pl.ds(slot * id_slot + base, per_query)],
                    sems.at[slot, 0],
                )
                (copy.wait if wait else copy.start)()

            # The slot still holds the page row of the pair two steps back.
            before = pair_info(jnp.maximum(step - 2, 0))[3]
            copy = pltpu.make_async_copy(
                pages.at[page_row(request)],
                page_ids.at[pl.ds(slot * table_columns, table_columns)],
                sems.at[slot, 1],
            )
            pl.when((step < 2) | (page_row(request) != page_row(before)))(
                copy.wait if wait else copy.start
            )

        def writes(step, wait=False):
            """Copy a pair's windows, physical union and counts out."""
            slot = step % 2
            pair, start, queries, _ = pair_info(step)
            first = (
                pltpu.make_async_copy(
                    window_a.at[pl.ds(slot * a_slot, per_query)],
                    out_ids.at[start],
                    sems.at[slot, 2],
                ),
                pltpu.make_async_copy(
                    union.at[pl.ds(slot * union_slot, 2 * per_query)],
                    out_union.at[pair],
                    sems.at[slot, 3],
                ),
                pltpu.make_async_copy(
                    counts.at[pl.ds(slot * alignment, alignment)],
                    out_counts.at[pair],
                    sems.at[slot, 4],
                ),
            )
            for copy in first:
                (copy.wait if wait else copy.start)()
            second = pltpu.make_async_copy(
                window_b.at[pl.ds(slot * b_slot, per_query)],
                out_ids.at[jnp.minimum(start + 1, tokens - 1)],
                sems.at[slot, 5],
            )
            pl.when(queries > 1)(second.wait if wait else second.start)

        def pad(ref, base, end, size):
            """-1 from end to size: the slot holds an earlier pair's IDs."""
            start = pl.multiple_of(end // lanes * lanes, lanes)

            @plsc.parallel_loop(0, size - start, lanes, unroll=4)
            def fill(i):
                at = pl.ds(base + start + i, lanes)
                ref[at] = jnp.where(start + i + lane >= end, -1, ref[at])

        def emit(targets, offset, mask):
            """Compress the mask's lanes of each (ref, base, value) to base + offset."""
            for ref, base, value in targets:
                plsc.store_compressed(
                    ref.at[pl.ds(base + offset, lanes)], value, mask=mask
                )
            return offset + count(mask)

        def clear_bitmaps():
            @plsc.parallel_loop(0, bitmap_words, lanes, unroll=unroll)
            def clear(i):
                bits_a[pl.ds(i, lanes)] = zeros
                bits_b[pl.ds(i, lanes)] = zeros

        clear_bitmaps()
        for slot in range(2):
            for base in (a_base, b_base):
                at = pl.ds(slot * id_slot + base - lanes, lanes)
                ids[at] = jnp.full(lanes, -2, jnp.int32)

        steps = (jnp.maximum(scalar(0) - unit, 0) + units - 1) // units

        @pl.when(steps > 0)
        def prologue():
            reads(jnp.int32(0))

        @pl.loop(0, steps)
        def body(step):
            slot = step % 2
            _, _, queries, request = pair_info(step)
            reads(step, wait=True)

            @pl.when(step + 1 < steps)
            def prefetch():
                reads(step + 1)

            bases = (slot * id_slot + a_base, slot * id_slot + b_base)
            page_base = slot * table_columns
            valid_request = (request >= 0) & (request < pages.shape[0])
            bounds = (
                jnp.where(valid_request, per_query, 0),
                jnp.where(valid_request & (queries > 1), per_query, 0),
            )

            def first(query, i, key):
                """Rows ascend, so a repeat equals the ID before it."""
                return key != ids[pl.ds(bases[query] + i - 1, lanes)]

            def word(key):
                return jnp.minimum(shr(key, 5), bitmap_words - 1)

            # Each query marks its in-range first occurrences in its own bitmap.
            for query, bits in enumerate((bits_a, bits_b)):

                @plsc.parallel_loop(0, bounds[query], lanes, unroll=unroll)
                def mark(i, query=query, bits=bits):
                    key = ids[pl.ds(bases[query] + i, lanes)]
                    marked = (key >= 0) & (key < blocks) & first(query, i, key)
                    # Distinct IDs set distinct bits, so adding them is an OR.
                    plsc.addupdate_scatter(
                        bits, [word(key)], ones << (key & 31), mask=marked
                    )

            def classify(query, i, other):
                """Block ID, physical ID, validity and whether `other` lists it."""
                key = ids[pl.ds(bases[query] + i, lanes)]
                column = shr(key, shift)
                page = plsc.load_gather(
                    page_ids, [page_base + jnp.clip(column, 0, table_columns - 1)]
                )
                valid = (key >= 0) & (key < blocks) & (page >= 0) & (page < cache_pages)
                if blocks > table_columns * per_page:
                    valid &= column < table_columns
                valid &= first(query, i, key)
                listed = plsc.load_gather(other, [word(key)]) >> (key & 31)
                physical = (page << shift) | (key & (per_page - 1))
                return key, physical, valid, (listed & 1) != 0

            a_window, b_window = slot * a_slot, slot * b_slot
            union_base = slot * union_slot

            # The pair two steps back must have drained this slot's outputs.
            @pl.when(step >= 2)
            def drain():
                writes(step - 2, wait=True)

            # A against B's bits: A-only opens A's window, shared opens B's.
            def classify_a(i, carry):
                a_only, shared = carry
                key, physical, valid, in_b = classify(0, i, bits_b)
                a_targets = ((window_a, a_window, key), (union, union_base, physical))
                b_targets = ((window_b, b_window, key), (shared_union, 0, physical))
                only, both = valid & ~in_b, valid & in_b
                return emit(a_targets, a_only, only), emit(b_targets, shared, both)

            a_only, shared = plsc.parallel_loop(
                0,
                bounds[0],
                lanes,
                unroll=classify_unroll,
                carry=(jnp.int32(0), jnp.int32(0)),
            )(classify_a)

            # Shared closes A's window and follows A-only in the union.
            @plsc.parallel_loop(0, pl.cdiv(shared, lanes) * lanes, lanes, unroll=unroll)
            def append_shared(i):
                live = i + lane < shared
                plsc.store_compressed(
                    window_a.at[pl.ds(a_window + a_only + i, lanes)],
                    window_b[pl.ds(b_window + i, lanes)],
                    mask=live,
                )
                plsc.store_compressed(
                    union.at[pl.ds(union_base + a_only + i, lanes)],
                    shared_union[pl.ds(i, lanes)],
                    mask=live,
                )

            # B against A's bits: B-only follows shared in B's window and the union.
            def classify_b(i, b_only):
                key, physical, valid, in_a = classify(1, i, bits_a)
                targets = (
                    (window_b, b_window + shared, key),
                    (union, union_base + a_only + shared, physical),
                )
                return emit(targets, b_only, valid & ~in_a)

            b_only = plsc.parallel_loop(
                0, bounds[1], lanes, unroll=classify_unroll, carry=jnp.int32(0)
            )(classify_b)

            pad(window_a, a_window, a_only + shared, per_query)
            pad(window_b, b_window, shared + b_only, per_query)
            pad(union, union_base, a_only + shared + b_only, 2 * per_query)
            counts[pl.ds(slot * alignment, lanes)] = jnp.where(
                lane == 0,
                a_only,
                jnp.where(lane == 1, shared, jnp.where(lane == 2, b_only, 0)),
            )
            writes(step)

            # Clear this pair's bits while its outputs drain: whole bitmaps
            # when they are no longer than a row, else only the marked words.
            if bitmap_words <= per_query:
                clear_bitmaps()
            else:
                for query, bits in enumerate((bits_a, bits_b)):

                    @plsc.parallel_loop(0, bounds[query], lanes, unroll=unroll)
                    def unmark(i, query=query, bits=bits):
                        # Words hold only this pair's bits, so zeroing the
                        # words of unmarked IDs is harmless.
                        key = ids[pl.ds(bases[query] + i, lanes)]
                        plsc.store_scatter(bits, [word(key)], zeros)

        for back in (2, 1):

            @pl.when(steps >= back)
            def epilogue(back=back):
                writes(steps - back, wait=True)

    candidate_block_ids, physical_block_ids, owners = kernel(candidates, table, control)
    return candidate_block_ids, physical_block_ids, owners[:, :3]


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
