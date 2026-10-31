# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent paged decode scores with double-buffered keys and aligned output DMA.

BF16 queries/keys produce FP32 weighted ReLU scores and optional block-eight maxima.
Decode dispatch uses the per-request shape; keys use canonical halfplane rows.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41.index_key_layout import DIM, SLOTS
from vllm_torchtpu.kernels.deepseek_v41.index_selection import (
    prefill_scores,
    tuned_block_sizes,
)


def _reduce_heads(logits, weights, queries, stripe):
    query_group = math.gcd(queries, stripe)
    head_group = stripe // query_group
    grouped_rows = queries * head_group
    width = logits.shape[-1]
    accumulated = jnp.zeros((grouped_rows, width), jnp.float32)
    # Consume aligned head/query stripes without padding Q=6 to eight queries.
    # Reassociate the FP32 reduction; identical FP32 bits are not required.
    for chunk in range(weights.shape[0]):
        activated = jnp.maximum(
            logits[chunk * grouped_rows : (chunk + 1) * grouped_rows, :], 0
        )
        accumulated = accumulated + activated * weights[chunk, :, None]
    accumulated = accumulated.reshape(queries // query_group, stripe, width)
    outputs = []
    for group in range(queries // query_group):
        head_parts = [
            accumulated[group, h * query_group : (h + 1) * query_group, :]
            for h in range(head_group)
        ]
        # Preserve the existing cyclic reduction's FP32 addition order, while
        # leaving only the dependency tree for returned query rows live.
        for shift in reversed(range(head_group.bit_length() - 1)):
            head_parts = [
                head_parts[h] + head_parts[(h - (1 << shift)) % head_group]
                for h in range(head_group)
            ]
        outputs.append(head_parts[0])
    return jnp.concatenate(outputs, axis=0)


def _prepare_queries(q, weights, query_out, weight_out, head_group, start):
    requests, queries, heads, dim = q.shape

    def prepare_one(local, unused):
        request = jnp.minimum(start + local, requests - 1)
        query = q[request].reshape(1, queries, heads // head_group, head_group, dim)
        query_out[local] = (
            query.transpose(2, 0, 3, 1, 4).reshape(queries * heads, dim).T
        )
        weight = weights[:, request, :].reshape(
            1, queries, heads // head_group, head_group
        )
        weight = weight.transpose(2, 0, 3, 1).reshape(
            heads // head_group, queries * head_group
        )
        weight_out[local] = weight
        return unused

    jax.lax.fori_loop(0, query_out.shape[0], prepare_one, None)


def _prepare_pages(pages, lengths, limits_out, contiguous_out, banks, tile, page):
    requests, page_count = pages.shape
    tiles = math.ceil(page_count * page / tile)
    tile_pages = tile // page
    limits = lengths[...].max(axis=0)
    limits_out[...] = limits
    addresses = (
        jnp.pad(
            pages[...],
            ((0, 0), (0, tiles * tile_pages - page_count)),
            constant_values=-1,
        ).reshape(requests, tiles, tile_pages)
        // SLOTS
    )
    first = addresses[:, :, :1]
    live_pages = (limits[:, None] - jnp.arange(tiles) * tile + page - 1) // page
    page_ids = jnp.arange(tile_pages)
    contiguous = jnp.all(
        (page_ids >= live_pages[:, :, None])
        | (addresses == first + page_ids * (page // SLOTS)),
        axis=-1,
    )
    base = first[:, :, 0]
    read_rows = jnp.clip(live_pages, 0, tile_pages) * (page // SLOTS)
    contiguous &= (base >= 0) & (base + read_rows <= sum(b.shape[0] for b in banks))
    contiguous &= (base >= banks[0].shape[0]) | (base + read_rows <= banks[0].shape[0])
    contiguous_out[...] = contiguous.astype(jnp.int32)


def grouped_decode_scores(
    q: jax.Array,
    weights: jax.Array,
    pool: jax.Array | tuple[jax.Array, ...],
    pages: jax.Array,
    lengths: jax.Array,
    *,
    page_records: int,
    score_config: tuned_block_sizes.PersistentScoreConfig | None = None,
    block_size: int = 8,
    disable_bounds_checks: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Score request-major BF16 queries against paged halfplane keys on TPU7x.

    Args:
        q: BF16 [request, query, head, feature] in split order (even64 then odd64).
        weights: FP32 [request, query, head].
        pool: One or two int32 [packed_record,128] banks; low16 holds the even record
            and high16 the odd, each in split order (even64 then odd64).
            Banks contain complete pages.
        pages: Int32 [request, page] page-aligned physical record starts;
            -1 denotes a hole.
        lengths: Int32 [request, query] visible lengths, bounded by capacity.
        page_records: Records per physical cache page, from the cache layout.
        score_config: Optional benchmark override; None uses the tuned table.
        block_size: Zero for token scores only; eight to also return block maxima.
        disable_bounds_checks: Disable Mosaic's runtime bounds instrumentation.

    Returns:
        FP32 scores [request * query, capacity], with block maxima when requested.
        Positions outside a query's length or in a page hole are negative infinity.
    """
    if (
        q.ndim != 4
        or min(q.shape) < 1
        or q.shape[2:] != (32, DIM)
        or weights.shape != q.shape[:3]
        or pages.ndim != 2
        or pages.shape[0] != q.shape[0]
        or pages.shape[1] < 1
        or lengths.shape != q.shape[:2]
    ):
        raise ValueError("Expected request-major Q [R,Q,32,128] and weights [R,Q,32]")
    pool = pool if isinstance(pool, tuple) else (pool,)
    if not 1 <= len(pool) <= 2:
        raise ValueError("Persistent decode requires one or two packed cache banks")
    if block_size not in (0, 8):
        raise NotImplementedError("Persistent decode supports block_size 0 or 8")
    if jax.default_backend() != "tpu" or pltpu.get_tpu_info().generation != 7:
        raise NotImplementedError("Persistent decode requires TPU v7")
    key = (*q.shape[1:], pages.shape[1] * page_records, len(pool))
    config = (
        tuned_block_sizes.get_persistent_score_config(key)
        if score_config is None
        else score_config
    )
    return _grouped_decode_scores(
        q,
        weights,
        pool,
        pages,
        lengths,
        page_records=page_records,
        key_tile=config.key_tile,
        compute_tile=config.compute_tile,
        head_group=config.head_group,
        block_size=block_size,
        disable_bounds_checks=disable_bounds_checks,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "page_records",
        "key_tile",
        "compute_tile",
        "head_group",
        "block_size",
        "disable_bounds_checks",
    ),
)
def _grouped_decode_scores(
    q,
    weights,
    pool,
    pages,
    lengths,
    *,
    page_records,
    key_tile,
    compute_tile=None,
    head_group=None,
    block_size=8,
    disable_bounds_checks=False,
):
    info = pltpu.get_tpu_info()
    requests, queries, heads, dim = q.shape
    page, buffers = page_records, 2
    if page < 1 or page % (SLOTS * info.num_sublanes):
        raise ValueError("Packed cache pages must align to TPU sublanes")
    page_rows = page // SLOTS
    if compute_tile is None:
        compute_tile = info.mxu_column_size
    if head_group is None:
        head_group = min(
            heads,
            info.num_sublanes
            * (info.num_sublanes // math.gcd(queries, info.num_sublanes)),
        )
    if compute_tile < 1 or compute_tile % info.mxu_column_size:
        raise ValueError("Compute tiles must align to MXU columns")
    if compute_tile % page:
        raise ValueError("Compute tiles must contain whole cache pages")
    if head_group < 1 or heads % head_group or head_group & (head_group - 1):
        raise ValueError("Head groups must be powers of two that divide the heads")
    stripe = queries * head_group
    width = pages.shape[1] * page
    # Publish complete lane vectors, including the last partial block-score vector.
    output_alignment = info.num_lanes * (block_size or 1)
    tile_alignment = math.lcm(compute_tile, output_alignment)
    key_tile = pl.cdiv(min(key_tile, width), tile_alignment) * tile_alignment
    output_width = pl.cdiv(width, output_alignment) * output_alignment
    tiles = math.ceil(width / key_tile)
    request_group = info.num_sublanes // math.gcd(queries, info.num_sublanes)
    groups = math.ceil(requests / request_group)
    output_rows = request_group * queries
    output_chunks = output_rows // info.num_sublanes
    output_buffers = output_chunks
    buffer_rows = output_buffers * info.num_sublanes
    padded_rows = groups * output_rows
    task_count, tile_pages = groups * tiles, key_tile // page
    # Dispatch and the tuning table fix hardware, query shape and DMA geometry.
    if pages.shape[0] != requests or lengths.shape != (requests, queries):
        raise ValueError("Page and length metadata must match the query batch")
    if weights.shape != (requests, queries, heads):
        raise ValueError("Weights must have shape [request, query, head]")
    if q.dtype != jnp.bfloat16 or weights.dtype != jnp.float32:
        raise ValueError("Queries must be BF16 and weights FP32")
    if pages.dtype != jnp.int32 or lengths.dtype != jnp.int32:
        raise ValueError("Page starts and query lengths must be int32")
    if any(
        b.dtype != jnp.int32
        or b.ndim != 2
        or b.shape[1] != dim
        or b.shape[0] < page_rows
        or b.shape[0] % page_rows
        for b in pool
    ):
        raise ValueError("The halfplane cache requires page-aligned int32 banks")
    total_rows = sum(b.shape[0] for b in pool)

    def kernel(
        page_indices_ref,
        lengths_ref,
        q_ref,
        weight_ref,
        pool_ref,
        metadata,
        token_ref,
        block_ref,
        keys,
        tokens,
        blocks,
        input_sem,
        output_sem,
        tasks,
        prepared,
    ):
        queries_buf, weights_buf, limits_v, contiguous_v, limits, contiguous = prepared
        page_values, length_values = metadata
        _prepare_pages(
            page_values, length_values, limits_v, contiguous_v, pool_ref, key_tile, page
        )
        pltpu.sync_copy((limits_v, contiguous_v), (limits, contiguous))

        def original_coordinates(step):
            return step % groups, step // groups

        def group_active(group, tile):
            active = jnp.bool_(False)
            for r in range(min(requests, request_group)):
                request = group * request_group + r
                active |= (request < requests) & (
                    tile * key_tile < limits[jnp.minimum(request, requests - 1)]
                )
            return active

        def add_task(step, count):
            group, tile = original_coordinates(step)
            active = group_active(group, tile)

            @pl.when(active)
            def append():
                tasks[count, 0], tasks[count, 1] = group, tile

            return count + active.astype(jnp.int32)

        num_steps = jax.lax.fori_loop(0, task_count, add_task, jnp.int32(0))

        def coordinates(step):
            group_step, r = step // request_group, step % request_group
            return tasks[group_step, 0] * request_group + r, tasks[group_step, 1]

        def request_active(request, tile):
            return (request < requests) & (
                tile * key_tile < limits[jnp.minimum(request, requests - 1)]
            )

        def live_pages(request, tile):
            return jnp.minimum(
                (limits[request] - tile * key_tile + page - 1) // page, tile_pages
            )

        def fetch(step):
            slot = step % buffers
            request, tile = coordinates(step)

            @pl.when(request_active(request, tile))
            def load_request():
                fetch_request(request, tile, step, slot)

        def fetch_request(request, tile, step, slot):
            first_page = tile * tile_pages
            count = live_pages(request, tile)
            read_rows = count * page_rows

            def copy(address, size, dest):
                if len(pool_ref) == 1:
                    pltpu.make_async_copy(
                        pool_ref[0].at[pl.ds(address, size), :],
                        dest,
                        input_sem.at[slot],
                    ).start()
                else:
                    low, high = pool_ref

                    @pl.when(address < low.shape[0])
                    def low_copy():
                        pltpu.make_async_copy(
                            low.at[pl.ds(address, size), :], dest, input_sem.at[slot]
                        ).start()

                    @pl.when(address >= low.shape[0])
                    def high_copy():
                        pltpu.make_async_copy(
                            high.at[pl.ds(address - low.shape[0], size), :],
                            dest,
                            input_sem.at[slot],
                        ).start()

            @pl.when((read_rows < key_tile // SLOTS) & (step < buffers))
            def clear_partial():
                # Initialize each slot before its first DMA. Later partial loads
                # may retain old keys: masks discard everything past the length.
                clear_rows = min(compute_tile, key_tile) // SLOTS
                last_start = key_tile // SLOTS - clear_rows

                @pl.when(read_rows >= last_start)
                def clear_end():
                    keys[slot, pl.ds(last_start, clear_rows), :] = jnp.zeros(
                        (clear_rows, dim), jnp.int32
                    )

                @pl.when(read_rows < last_start)
                def clear_all():
                    keys[slot] = jnp.zeros((key_tile // SLOTS, dim), jnp.int32)

            @pl.when(contiguous[request, tile] != 0)
            def contiguous_load():
                copy(
                    page_indices_ref[request, first_page] * page_rows,
                    read_rows,
                    keys.at[slot, pl.ds(0, read_rows), :],
                )

            @pl.when(contiguous[request, tile] == 0)
            def scattered_load():
                def one_page(index, unused):
                    page_index = jnp.clip(
                        page_indices_ref[request, first_page + index],
                        0,
                        total_rows // page_rows - 1,
                    )
                    copy(
                        page_index * page_rows,
                        page_rows,
                        keys.at[slot, pl.ds(index * page_rows, page_rows), :],
                    )
                    return unused

                jax.lax.fori_loop(0, count, one_page, None)

        def copies_for(group, tile, chunk, padding=False):
            slot = 0 if padding else chunk % output_buffers
            chunk_row = slot * info.num_sublanes
            first_row = group * output_rows + chunk * info.num_sublanes
            rows = pl.ds(chunk_row, info.num_sublanes)
            count = (
                jnp.minimum(key_tile, output_width - tile * key_tile)
                if output_width % key_tile
                else key_tile
            )
            copies = (
                pltpu.make_async_copy(
                    tokens.at[rows, pl.ds(0, count)],
                    token_ref.at[
                        pl.ds(first_row, info.num_sublanes),
                        pl.ds(tile * key_tile, count),
                    ],
                    output_sem.at[slot, 0],
                ),
            )
            if not block_size:
                return copies
            block_count = (
                jnp.minimum(
                    key_tile // (block_size * info.num_lanes),
                    output_width // (block_size * info.num_lanes)
                    - tile * (key_tile // (block_size * info.num_lanes)),
                )
                * info.num_lanes
                if output_width % key_tile
                else key_tile // block_size
            )
            return copies + (
                pltpu.make_async_copy(
                    blocks.at[rows, pl.ds(0, block_count)],
                    block_ref.at[
                        pl.ds(first_row, info.num_sublanes),
                        pl.ds(tile * (key_tile // block_size), block_count),
                    ],
                    output_sem.at[slot, 1],
                ),
            )

        def output_copies(step, chunk):
            return copies_for(tasks[step, 0], tasks[step, 1], chunk)

        for step in range(buffers):
            pl.when(step < num_steps * request_group)(
                lambda step=step: fetch(jnp.int32(step))
            )

        tokens[: info.num_sublanes] = jnp.full(
            (info.num_sublanes, key_tile), -jnp.inf, jnp.float32
        )
        if block_size:
            blocks[: info.num_sublanes] = jnp.full(
                (info.num_sublanes, key_tile // block_size), -jnp.inf, jnp.float32
            )

        def padding_task(step, unused, wait):
            group, tile = original_coordinates(step)

            @pl.when(~group_active(group, tile))
            def copy():
                for chunk in range(output_chunks):
                    for descriptor in copies_for(group, tile, chunk, padding=True):
                        if wait:
                            descriptor.wait()
                        else:
                            descriptor.start()

            return unused

        # The aligned output buffer must drain before compute reuses it.
        jax.lax.fori_loop(
            0, task_count, functools.partial(padding_task, wait=False), None
        )
        jax.lax.fori_loop(
            0, task_count, functools.partial(padding_task, wait=True), None
        )

        def write_values(output, r, start, values):
            row = r * queries % buffer_rows
            count = min(queries, buffer_rows - row)
            columns = pl.ds(start, values.shape[1])
            output[pl.ds(row, count), columns] = values[:count]
            if count < queries:
                output[pl.ds(0, queries - count), columns] = values[count:]

        def score_request(step, r):
            slot = step % buffers
            request, tile = coordinates(step)
            read_rows = live_pages(request, tile) * page_rows
            pltpu.make_async_copy(
                keys.at[slot, pl.ds(0, read_rows), :],
                keys.at[slot, pl.ds(0, read_rows), :],
                input_sem.at[slot],
            ).wait()

            query = queries_buf[request % request_group]
            weight = weights_buf[request % request_group]
            row_lengths = jnp.stack([lengths_ref[i, request] for i in range(queries)])
            # Keep the matrix, masks and block maxima in one unrolled region.
            # A rolled loop or phase scope changes scheduling substantially.
            fragments = max(1, block_size * info.num_lanes // compute_tile)
            for first in range(0, key_tile // compute_tile, fragments):
                block_parts = []
                for index in range(
                    first, min(first + fragments, key_tile // compute_tile)
                ):
                    start = index * compute_tile
                    fragment = pltpu.bitcast(
                        keys[slot, pl.ds(start // SLOTS, compute_tile // SLOTS), :],
                        jnp.bfloat16,
                    )
                    logits = jax.lax.dot_general(
                        query,
                        fragment,
                        (((0,), (1,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                    values = _reduce_heads(logits, weight, queries, stripe)[:queries]
                    position = tile * key_tile + start + jnp.arange(compute_tile)
                    page_start = (tile * key_tile + start) // page
                    page_valid = jnp.concatenate(
                        [
                            jnp.broadcast_to(
                                (
                                    page_indices_ref[
                                        request,
                                        jnp.minimum(page_start + p, pages.shape[1] - 1),
                                    ]
                                    >= 0
                                ).astype(jnp.int32),
                                (1, page),
                            )
                            for p in range(compute_tile // page)
                        ],
                        axis=1,
                    )
                    valid = (position[None] < row_lengths[:, None]) & (page_valid != 0)
                    values = jnp.where(valid, values, -jnp.inf)
                    write_values(tokens, r, start, values)
                    if block_size:
                        block_parts.append(
                            prefill_scores._block_max(
                                values, info.num_lanes, block_size
                            )
                        )
                if block_size:
                    write_values(
                        blocks,
                        r,
                        first * compute_tile // block_size,
                        jnp.concatenate(block_parts, axis=1),
                    )

        def body(step, unused):
            _prepare_queries(
                q_ref,
                weight_ref,
                queries_buf,
                weights_buf,
                head_group,
                tasks[step, 0] * request_group,
            )
            # Static request positions allow sublane-unaligned stores in VMEM;
            # publish each completed eight-row block while later requests score.
            for r in range(request_group):
                first_new_chunk = (r * queries - 1) // info.num_sublanes + 1
                last_chunk = ((r + 1) * queries - 1) // info.num_sublanes

                for chunk in range(first_new_chunk, last_chunk + 1):
                    if chunk >= output_buffers:
                        for copy in output_copies(step, chunk - output_buffers):
                            copy.wait()
                    else:
                        previous = (
                            chunk
                            + ((output_chunks - 1 - chunk) // output_buffers)
                            * output_buffers
                        )

                        @pl.when(step > 0)
                        def wait_previous(previous=previous):
                            for copy in output_copies(step - 1, previous):
                                copy.wait()

                item = step * request_group + r
                request, tile = coordinates(item)
                active = request_active(request, tile)

                @pl.when(active)
                def score(item=item, r=r):
                    score_request(item, r)

                @pl.when(~active)
                def empty(r=r):
                    write_values(
                        tokens,
                        r,
                        0,
                        jnp.full((queries, key_tile), -jnp.inf, jnp.float32),
                    )
                    if block_size:
                        write_values(
                            blocks,
                            r,
                            0,
                            jnp.full(
                                (queries, key_tile // block_size), -jnp.inf, jnp.float32
                            ),
                        )

                @pl.when(item + buffers < num_steps * request_group)
                def prefetch_next(item=item):
                    fetch(item + buffers)

                for chunk in range(
                    r * queries // info.num_sublanes,
                    (r + 1) * queries // info.num_sublanes,
                ):
                    for copy in output_copies(step, chunk):
                        copy.start()
            return unused

        jax.lax.fori_loop(0, num_steps, body, None)

        @pl.when(num_steps > 0)
        def drain():
            for chunk in range(output_chunks - output_buffers, output_chunks):
                for copy in output_copies(num_steps - 1, chunk):
                    copy.wait()

    hbm, vmem = (
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.VMEM),
    )
    result = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((padded_rows, output_width), jnp.float32),
            jax.ShapeDtypeStruct((padded_rows, output_width // block_size), jnp.float32)
            if block_size
            else None,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(),
            in_specs=(vmem, vmem, (hbm,) * len(pool), (vmem, vmem)),
            out_specs=(hbm, hbm if block_size else None),
            scratch_shapes=(
                pltpu.VMEM((buffers, key_tile // SLOTS, dim), jnp.int32),
                pltpu.VMEM((buffer_rows, key_tile), jnp.float32),
                pltpu.VMEM((buffer_rows, key_tile // block_size), jnp.float32)
                if block_size
                else None,
                pltpu.SemaphoreType.DMA((buffers,)),
                pltpu.SemaphoreType.DMA((output_buffers, 2 if block_size else 1)),
                pltpu.SMEM((task_count, 2), jnp.int32),
                (
                    pltpu.VMEM((request_group, dim, queries * heads), q.dtype),
                    pltpu.VMEM(
                        (request_group, heads // head_group, queries * head_group),
                        weights.dtype,
                    ),
                    pltpu.VMEM((requests,), lengths.dtype),
                    pltpu.VMEM((requests, tiles), jnp.int32),
                    pltpu.SMEM((requests,), lengths.dtype),
                    pltpu.SMEM((requests, tiles), jnp.int32),
                ),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=disable_bounds_checks,
            vmem_limit_bytes=info.vmem_capacity_bytes,
            fuse_transposed_lhs_in_matmul=True,
        ),
        name="decode_grouped_paged_index_scores_persistent",
    )(
        # Prefetch page indices, not record addresses: an SMEM load loses the
        # address's alignment. Multiplying by the static packed-page stride in
        # the kernel makes the DMA's sublane alignment explicit to Mosaic.
        pages // page,
        lengths.transpose(1, 0),
        q,
        weights.transpose(1, 0, 2),
        pool,
        (pages, lengths.transpose(1, 0)),
    )
    tokens, blocks = result
    tokens = tokens[: requests * queries, :width]
    if block_size:
        return tokens, blocks[: requests * queries, : width // block_size]
    return tokens
