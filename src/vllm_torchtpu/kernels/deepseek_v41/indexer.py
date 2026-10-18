# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded query tiles for V4.1's native FP8 index cache.

The query-tile lifetime follows the SGLang-JAX DSA indexer/paged-score pattern:
retain a tile's scores until selection completes, then reuse the score buffer.
Unlike a BF16 cache scorer, this kernel reads four FP8 keys per native int32 row and
reads E8M0 scale words from the separate companion array. Selection reuses
vllm-torchtpu's existing SparseCore top-k, not a full-context XLA sort.
"""

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import config

QUERY_TILE = 32
KEY_TILE = 128
HEAD_DIM = 128
KEYS_PER_WORD = 4


def _score_kernel(
    lengths_ref,
    requests_ref,
    pages_ref,
    request_ref,
    q_ref,
    weights_ref,
    pool_ref,
    scales_ref,
    previous_ref,
    out_ref,
    *,
    page_size,
    page_stride,
    row_offset,
):
    """Score one request's rows; keep other rows in the donated score tile."""
    rows, heads, _ = q_ref.shape
    request = request_ref[0]
    max_length = jnp.int32(0)
    for row in range(rows):
        max_length = jnp.maximum(
            max_length,
            jnp.where(requests_ref[row] == request, lengths_ref[row], 0),
        )
    steps = pl.cdiv(max_length, KEY_TILE)

    def physical_page(step):
        return pages_ref[step * KEY_TILE // page_size]

    def key_map(step):
        page = jnp.clip(physical_page(step), 0, scales_ref.shape[0] - 1)
        within = step * KEY_TILE % page_size + row_offset
        # BlockSpec indices count KEY_TILE/4-word blocks, not word rows.
        return (page * page_stride + within) // KEY_TILE, 0

    def scale_map(step):
        page = jnp.clip(physical_page(step), 0, scales_ref.shape[0] - 1)
        within = step * KEY_TILE % page_size + row_offset
        return page, within // 128, 0

    def score(words_ref, scale_words_ref, old_ref, scores_ref):
        step = pl.program_id(0)
        words = words_ref[...]
        # Byte b of each word holds channel d of key 4*r+b.
        key_bytes = jnp.stack(
            [((words >> (8 * b)) & 255).astype(jnp.uint8) for b in range(4)],
            axis=1,
        ).reshape(KEY_TILE, HEAD_DIM)
        keys = pltpu.bitcast(key_bytes, jnp.float8_e4m3fn)
        scale_bytes = scale_words_ref[...].astype(jnp.uint8)
        key_scales = (
            pltpu.bitcast(scale_bytes, jnp.float8_e8m0fnu)
            .astype(jnp.float32)
            .reshape(1, KEY_TILE)
        )
        similarities = jnp.einsum(
            "nd,kd->nk",
            q_ref[...].reshape(rows * heads, HEAD_DIM),
            keys,
            preferred_element_type=jnp.float32,
        ).reshape(rows, heads, KEY_TILE)
        weights = weights_ref[...].astype(jnp.float32)
        values = jnp.maximum(similarities[:, 0], 0) * weights[:, 0:1]
        for head in range(1, heads):
            values = (
                values
                + jnp.maximum(similarities[:, head], 0) * weights[:, head : head + 1]
            )
        values = values * key_scales
        logical = step * KEY_TILE + jnp.arange(KEY_TILE)
        page_valid = (physical_page(step) >= 0) & (
            physical_page(step) < scales_ref.shape[0]
        )
        for row in range(rows):
            value = jnp.where(
                (logical < lengths_ref[row]) & page_valid, values[row], -jnp.inf
            )
            scores_ref[row] = jnp.where(
                requests_ref[row] == request, value, old_ref[row]
            )

    pltpu.emit_pipeline(
        score,
        grid=(steps,),
        in_specs=(
            pl.BlockSpec(
                (KEY_TILE // KEYS_PER_WORD, HEAD_DIM),
                key_map,
                pipeline_mode=pl.Buffered(2),
            ),
            pl.BlockSpec((None, 1, 128), scale_map, pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec(
                (rows, KEY_TILE),
                lambda step: (0, step),
                pipeline_mode=pl.Buffered(2),
            ),
        ),
        out_specs=pl.BlockSpec(
            (rows, KEY_TILE),
            lambda step: (0, step),
            pipeline_mode=pl.Buffered(2),
        ),
    )(pool_ref, scales_ref, previous_ref, out_ref)


def paged_scores(
    q,
    weights,
    pool,
    scales,
    block_table,
    requests,
    lengths,
    capacity,
    *,
    page_size,
    page_stride,
    row_offset=0,
    interpret=False,
):
    """FP32 scores for at most QUERY_TILE queries, without dense addresses.

    Q is already quantized; its scale is folded into weights by the caller.
    The score buffer is reused across requests. Only a request's page-table
    vector enters SMEM. Inactive query rows keep -inf.
    """
    rows, _, dim = q.shape
    if not 0 < rows <= QUERY_TILE or dim != HEAD_DIM:
        raise ValueError("Indexer scoring requires 1..32 rows and 128 channels")
    if q.dtype != jnp.float8_e4m3fn:
        raise ValueError("Indexer scoring requires FP8 E4M3 queries")
    if capacity < 1 or min(page_size, page_stride) < KEY_TILE:
        raise ValueError("Positive capacity and complete native pages required")
    if page_size % KEY_TILE or page_stride % KEY_TILE or row_offset % KEY_TILE:
        raise ValueError("Native index geometry must be 128-key aligned")
    width = pl.cdiv(capacity, KEY_TILE) * KEY_TILE
    lengths = jnp.clip(lengths, 0, min(capacity, block_table.shape[1] * page_size))
    scores = jnp.full((rows, width), -jnp.inf, jnp.float32)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    def request_scores(row, result):
        request = requests[row]
        # This supports repeated and permuted request IDs, not just packed rows.
        first = ~jnp.any((jnp.arange(rows) < row) & (requests == request))

        def run(previous):
            pages = block_table[request]
            fn = pl.pallas_call(
                functools.partial(
                    _score_kernel,
                    page_size=page_size,
                    page_stride=page_stride,
                    row_offset=row_offset,
                ),
                out_shape=jax.ShapeDtypeStruct(previous.shape, jnp.float32),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=4,
                    grid=(1,),
                    in_specs=(
                        pl.BlockSpec(q.shape, lambda *_: (0, 0, 0)),
                        pl.BlockSpec(weights.shape, lambda *_: (0, 0)),
                        hbm,
                        hbm,
                        hbm,
                    ),
                    out_specs=hbm,
                ),
                input_output_aliases={8: 0},
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("arbitrary",),
                    vmem_limit_bytes=64 * 1024 * 1024,
                ),
                interpret=interpret,
                name="dsv41_paged_fp8_scores",
            )
            return fn(
                lengths,
                requests,
                pages,
                jnp.asarray([request], jnp.int32),
                q,
                weights,
                pool,
                scales,
                previous,
            )

        active = jnp.any((requests == request) & (lengths > 0))
        return lax.cond(first & active, run, lambda x: x, result)

    return lax.fori_loop(0, rows, request_scores, scores)[:, :capacity]


NATIVE_KEY_TILE = 2048


def _native_score_kernel(
    lengths,
    requests,
    pages,
    q_ref,
    weights_ref,
    pool,
    scales,
    _previous,
    output,
    key_buffers,
    scale_buffers,
    score_buffer,
    previous_scores,
    key_sem,
    scale_sem,
    output_sem,
    *,
    page_size,
    page_stride,
    row_offset,
    tile_keys,
    num_requests,
    pages_per_request,
    capacity,
):
    from .cache_write import _quantize

    rows, heads, dim = q_ref.shape
    key_bytes, q_scale_words = _quantize(q_ref[...].reshape(rows * heads, dim))
    query = pltpu.bitcast(key_bytes, jnp.float8_e4m3fn)
    q_scale = pltpu.bitcast(
        (q_scale_words.astype(jnp.uint32) << 23).reshape(rows, heads), jnp.float32
    )
    weighted = weights_ref[...].astype(jnp.float32) * q_scale
    pages_per_tile = tile_keys // page_size

    def request_body(request, _):
        maximum = jnp.int32(0)
        uniform = jnp.bool_(True)
        for row in range(rows):
            maximum = jnp.maximum(
                maximum, jnp.where(requests[row] == request, lengths[row], 0)
            )
            uniform = uniform & ((requests[row] == request) | (lengths[row] == 0))
        steps = pl.cdiv(jnp.minimum(maximum, capacity), tile_keys)

        def fetch(block, bank, wait):
            if wait:
                pltpu.make_async_copy(
                    key_buffers.at[bank], key_buffers.at[bank], key_sem
                ).wait()
                pltpu.make_async_copy(
                    scale_buffers.at[bank], scale_buffers.at[bank], scale_sem
                ).wait()
                return
            for page_in_tile in range(pages_per_tile):
                logical_page = block * pages_per_tile + page_in_tile
                physical = pages[
                    request * pages_per_request
                    + jnp.minimum(logical_page, pages_per_request - 1)
                ]
                safe_page = jnp.clip(physical, 0, scales.shape[0] - 1)
                pltpu.make_async_copy(
                    pool.at[
                        pl.ds(
                            (safe_page * page_stride + row_offset) // 4, page_size // 4
                        ),
                        :,
                    ],
                    key_buffers.at[
                        bank, pl.ds(page_in_tile * page_size // 4, page_size // 4), :
                    ],
                    key_sem,
                ).start()
                pltpu.make_async_copy(
                    scales.at[safe_page, pl.ds(row_offset // 128, page_size // 128), :],
                    scale_buffers.at[bank, page_in_tile, :, :],
                    scale_sem,
                ).start()

        @pl.when(steps > 0)
        def first_fetch():
            fetch(jnp.int32(0), jnp.int32(0), False)

        def block_body(block, _):
            bank = block % 2
            fetch(block, bank, True)

            @pl.when(block + 1 < steps)
            def following():
                fetch(block + 1, 1 - bank, False)

            words = key_buffers[bank]
            bytes_ = jnp.stack(
                [((words >> (8 * lane)) & 255).astype(jnp.uint8) for lane in range(4)],
                axis=1,
            ).reshape(tile_keys, dim)
            keys = pltpu.bitcast(bytes_, jnp.float8_e4m3fn)
            key_scales = pltpu.bitcast(
                scale_buffers[bank].astype(jnp.uint8).reshape(1, tile_keys),
                jnp.float8_e8m0fnu,
            ).astype(jnp.float32)
            similarities = jnp.einsum(
                "nd,kd->nk", query, keys, preferred_element_type=jnp.float32
            ).reshape(rows, heads, tile_keys)
            values = jnp.maximum(similarities[:, 0], 0) * weighted[:, 0:1]
            for head in range(1, heads):
                values = (
                    values
                    + jnp.maximum(similarities[:, head], 0)
                    * weighted[:, head : head + 1]
                )
            logical = block * tile_keys + jnp.arange(tile_keys)
            page_valid = jnp.zeros((tile_keys,), jnp.bool_)
            for page_in_tile in range(pages_per_tile):
                logical_page = block * pages_per_tile + page_in_tile
                physical = pages[
                    request * pages_per_request
                    + jnp.minimum(logical_page, pages_per_request - 1)
                ]
                valid = (
                    (logical_page < pages_per_request)
                    & (physical >= 0)
                    & (physical < scales.shape[0])
                )
                page_valid = page_valid | (
                    valid & (jnp.arange(tile_keys) // page_size == page_in_tile)
                )
            for row in range(rows):
                score_buffer[row] = jnp.where(
                    (logical < lengths[row]) & page_valid,
                    values[row] * key_scales[0],
                    -jnp.inf,
                )

            @pl.when(uniform)
            def full_store():
                pltpu.make_async_copy(
                    score_buffer,
                    output.at[:, pl.ds(block * tile_keys, tile_keys)],
                    output_sem,
                ).start()
                pltpu.make_async_copy(
                    score_buffer,
                    output.at[:, pl.ds(block * tile_keys, tile_keys)],
                    output_sem,
                ).wait()

            @pl.when(~uniform)
            def ragged_store():
                destination = output.at[:, pl.ds(block * tile_keys, tile_keys)]
                # Keep the DMA shape equal to the full score tile. A one-row
                # view can acquire an incompatible HBM layout inside the
                # serving graph. Preserve the other requests as the legacy
                # paged scorer does, then publish the complete tile.
                copy = pltpu.make_async_copy(destination, previous_scores, output_sem)
                copy.start()
                copy.wait()
                for row in range(rows):
                    score_buffer[row] = jnp.where(
                        requests[row] == request,
                        score_buffer[row],
                        previous_scores[row],
                    )
                copy = pltpu.make_async_copy(score_buffer, destination, output_sem)
                copy.start()
                copy.wait()

            return None

        lax.fori_loop(0, steps, block_body, None)
        return None

    lax.fori_loop(0, num_requests, request_body, None)


def native_paged_scores(
    q,
    weights,
    pool,
    scales,
    block_table,
    requests,
    lengths,
    capacity,
    *,
    page_size,
    page_stride,
    row_offset=0,
    interpret=False,
):
    """Fuse BF16 query quantization and paged FP8 scoring with double-buffered DMA.

    The coupled SparseCore selection receives each query's causal length.
    The aliased output preserves masked rows across ragged query groups.
    """
    if (
        q.dtype != jnp.bfloat16
        or q.shape[-1] != HEAD_DIM
        or not 0 < q.shape[0] <= QUERY_TILE
    ):
        raise ValueError("Native scoring requires 1..32 BF16 queries of width 128")
    if (
        page_size > NATIVE_KEY_TILE
        or page_size % 128
        or page_stride % 128
        or row_offset % 128
    ):
        raise ValueError(
            "Native scoring requires 128-aligned pages no larger than 2048"
        )
    tile_keys = max(
        page_size, min(NATIVE_KEY_TILE, pl.cdiv(capacity, page_size) * page_size)
    )
    width = pl.cdiv(capacity, tile_keys) * tile_keys
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    lengths = jnp.clip(lengths, 0, min(capacity, block_table.shape[1] * page_size))
    return pl.pallas_call(
        functools.partial(
            _native_score_kernel,
            page_size=page_size,
            page_stride=page_stride,
            row_offset=row_offset,
            tile_keys=tile_keys,
            num_requests=block_table.shape[0],
            pages_per_request=block_table.shape[1],
            capacity=capacity,
        ),
        out_shape=jax.ShapeDtypeStruct((q.shape[0], width), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(1,),
            in_specs=(
                pl.BlockSpec(q.shape, lambda *_: (0, 0, 0)),
                pl.BlockSpec(weights.shape, lambda *_: (0, 0)),
                hbm,
                hbm,
                hbm,
            ),
            out_specs=hbm,
            scratch_shapes=(
                pltpu.VMEM((2, tile_keys // 4, HEAD_DIM), jnp.int32),
                pltpu.VMEM(
                    (2, tile_keys // page_size, page_size // 128, 128), jnp.int32
                ),
                pltpu.VMEM((q.shape[0], tile_keys), jnp.float32),
                pltpu.VMEM((q.shape[0], tile_keys), jnp.float32),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
            ),
        ),
        input_output_aliases={7: 0},
        interpret=interpret,
        name="dsv41_quantized_paged_scores",
    )(
        lengths,
        requests,
        block_table.reshape(-1),
        q,
        weights,
        pool,
        scales,
        jnp.full((q.shape[0], width), -jnp.inf, jnp.float32),
    )[:, :capacity]


def pipeline_score_topk_bounded(
    num_tiles: int,
    prepare: Callable[[jax.Array], Any],
    score: Callable[[Any, int | None], jax.Array],
    select: Callable[[jax.Array, jax.Array, int | None], Any],
    write: Callable[[jax.Array, Any, Any], Any],
    output: Any,
) -> Any:
    """Bound score lifetime and pair next-tile scoring with current selection.

    Callbacks accept an optional scheduling group after their data arguments.
    Each pair gets a program-unique group so the compiler keeps TensorCore
    scoring between SparseCore selection start and completion.
    """
    if num_tiles < 1:
        raise ValueError("The score/top-k pipeline requires at least one tile")
    if num_tiles > 16:
        return pipeline_score_topk(num_tiles, prepare, score, select, write, output)

    group_base = (
        config.reserve_scheduling_group_ids(num_tiles - 1) if num_tiles > 1 else 0
    )
    # Gate current scores and next preparation on the previous selection
    # so unrolling does not extend every score buffer's lifetime.
    with jax.named_scope("DSV41ScoreTopKBoundedPipeline"):
        current = score(prepare(jnp.int32(0)), None)
        previous = None
        for tile in range(num_tiles - 1):
            prepared = prepare(jnp.int32(tile + 1))
            if previous is not None:
                previous, current, prepared = lax.optimization_barrier(
                    (previous, current, prepared)
                )
                output = write(jnp.int32(tile - 1), previous, output)
            group = group_base + tile
            following = score(prepared, group)
            previous = select(jnp.int32(tile), current, group)
            current = following
        if previous is not None:
            previous, current = lax.optimization_barrier((previous, current))
            output = write(jnp.int32(num_tiles - 2), previous, output)
        last = jnp.int32(num_tiles - 1)
        return write(last, select(last, current, None), output)


def pipeline_score_topk(num_tiles, prepare, score, select, write, output):
    """Order TensorCore scoring and SparseCore selection across two query tiles.

    The barrier gates score-buffer reuse on selection completion, before
    output scatter. Physical execution overlap depends on the compiler.
    """
    first = score(prepare(jnp.int32(0)))

    def advance(tile, state):
        current, result = state
        following = score(prepare(tile + 1))
        return following, write(tile, select(tile, current), result)

    def body(pair, state):
        tile = 2 * pair
        current, result = state
        following = score(prepare(tile + 1))
        selected = select(tile, current)
        prepared, selected = lax.optimization_barrier((prepare(tile + 2), selected))
        replacement = score(prepared)
        following_selected = select(tile + 1, following)
        result = write(tile, selected, result)
        result = write(tile + 1, following_selected, result)
        return replacement, result

    pairs = (num_tiles - 1) // 2
    with jax.named_scope("DSV41ScoreTopKPipeline"):
        state = lax.fori_loop(0, pairs, body, (first, output))
        if 2 * pairs < num_tiles - 1:
            state = advance(jnp.int32(2 * pairs), state)
        last, result = state
        return write(
            jnp.int32(num_tiles - 1), select(jnp.int32(num_tiles - 1), last), result
        )
