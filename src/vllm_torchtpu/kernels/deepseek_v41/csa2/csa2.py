# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CSA2 attention with ragged query tiles and a four-buffer carry-PV pipeline."""

import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _fill_query_tiles(starts, tiles, total_tokens, bq):
    batch = starts.shape[0] - 1

    def request_body(request, first):
        begin = starts[request]
        end = jnp.where(
            request == batch, total_tokens, starts[jnp.minimum(request + 1, batch)]
        )
        count = pl.cdiv(end - begin, bq)

        def tile_body(local, unused):
            offset = begin + local * bq
            tile = (first + local) * 3
            tiles[tile] = request
            tiles[tile + 1] = offset
            tiles[tile + 2] = jnp.minimum(bq, end - offset)
            return None

        lax.fori_loop(0, count, tile_body, None)
        return first + count

    # A final synthetic request covers output padding without an HBM clear pass.
    return lax.fori_loop(0, batch + 1, request_body, jnp.int32(0))


def build_query_tiles(
    starts: jax.Array, total_tokens: int, bq: int
) -> tuple[jax.Array, jax.Array]:
    capacity = min(total_tokens, pl.cdiv(total_tokens, bq) + starts.shape[0] - 1)
    tiles = jax.new_ref(jnp.zeros(capacity * 3, jnp.int32))
    count = _fill_query_tiles(starts, tiles, total_tokens, bq)
    return tiles[...].reshape(capacity, 3), count


def _tile_field(tiles, tile, column):
    if len(tiles.shape) == 1:
        return tiles[tile * 3 + column]
    return tiles[tile, column]


def swa_window(tile, tiles, positions, pages, bounds, *, page_size, non_causal):
    request, start, count = (
        _tile_field(tiles, tile, 0),
        _tile_field(tiles, tile, 1),
        _tile_field(tiles, tile, 2),
    )
    active = request < pages.shape[0]
    request = jnp.minimum(request, pages.shape[0] - 1)
    position = positions[start]
    left = bounds[request, 0] if non_causal else jnp.maximum(0, position - 127)
    right = bounds[request, 1] if non_causal else position + count
    anchor = left // page_size * page_size
    left = jnp.maximum(left, pages[request, 1])
    return position, left, jnp.where(active, right, left), anchor


def _prepare_swa(
    tile,
    tiles_ref,
    positions_ref,
    swa_page_ref,
    bounds_ref,
    swa_kv_ref,
    swa_ref,
    *,
    page_size,
    cache_page_size,
    num_banks,
    low_pages,
    non_causal,
):
    num_pages = 256 // page_size
    request = jnp.minimum(_tile_field(tiles_ref, tile, 0), swa_page_ref.shape[0] - 1)
    active = _tile_field(tiles_ref, tile, 0) < swa_page_ref.shape[0]
    _, left, right, anchor = swa_window(
        tile,
        tiles_ref,
        positions_ref,
        swa_page_ref,
        bounds_ref,
        page_size=page_size,
        non_causal=non_causal,
    )
    bf16_bits = jnp.dtype(jnp.bfloat16).itemsize * 8
    low_mask = (1 << bf16_bits) - 1
    high_mask = ~low_mask
    for block in range(num_pages):
        column = (
            anchor + block * page_size - swa_page_ref[request, 0]
        ) // cache_page_size
        column = jnp.clip(column, 0, swa_page_ref.shape[1] - 3)
        physical_page = swa_page_ref[request, column + 2]
        values = swa_kv_ref[block * num_banks][0]
        if num_banks == 2:
            values = jnp.where(
                physical_page < low_pages, values, swa_kv_ref[block * 2 + 1][0]
            )
        rows = pl.ds(block * page_size, page_size)
        positions = anchor + block * page_size + 2 * jnp.arange(page_size // 2)
        low = active & (positions >= left) & (positions < right)
        high = active & (positions + 1 >= left) & (positions + 1 < right)
        # A BF16 select triggers an invalid predicate layout on the pinned
        # compiler. Mask pairs of rows through their zero-copy INT32 view.
        mask = jnp.where(low, low_mask, 0) | jnp.where(high, high_mask, 0)
        words = pltpu.bitcast(values, jnp.int32) & mask[:, None]
        swa_ref[rows] = pltpu.bitcast(words, jnp.bfloat16)


def _csa2_kernel(
    # prefetch
    tiles_ref: Any,  # SMEM: [tile_capacity, 3], request/start/count
    positions_ref: Any,  # SMEM: [T], int32
    main_lengths_ref: Any,  # SMEM: [T], int32
    swa_page_ref: Any,  # SMEM: [B, 2 + page_columns], int32
    bounds_ref: Any,  # SMEM: [B, 2], int32 or None
    # input
    q_ref: Any,  # VMEM: [bq, H, 512], BF16
    main_cache_ref: Any,  # VMEM: [bq, 512, 128], INT32
    main_scale_ref: Any,  # VMEM: [bq, 1, 512], FP32
    sink_ref: Any,  # VMEM: [H, 1], FP32
    rope_ref: Any,  # VMEM: [bq, 2, 128], FP32
    # output
    output_ref: Any,  # VMEM: [bq, H, 512], BF16
    # scratch
    main_kv_ref: Any,  # VMEM: [bq, 512, 512], BF16
    swa_ref: Any,  # VMEM: [256, 512], BF16
    swa_score_ref: Any,  # VMEM: [bq, H, 256], FP32
    main_p_ref: Any,  # VMEM: [bq, H, 512], BF16
    swa_p_ref: Any,  # VMEM: [bq, H, 256], BF16
    *,
    bq: int,
    block_q: int,
    page_size: int,
    scale: float,
    non_causal: bool,
    accumulators: Any,
    tile_index: Any,
    previous_output: Any = None,
    previous_rope: Any = None,
    previous_tile: Any = None,
    swa_rhs_preloaded: bool = False,
    prepare_next: Any = None,
    preload_swa: Any = None,
    drain_only: bool = False,
) -> None:
    log2_e = math.log2(math.e)
    tile = tile_index
    request, t0 = _tile_field(tiles_ref, tile, 0), _tile_field(tiles_ref, tile, 1)
    position_start, left, right, anchor = swa_window(
        tile,
        tiles_ref,
        positions_ref,
        swa_page_ref,
        bounds_ref,
        page_size=page_size,
        non_causal=non_causal,
    )
    read_start = jnp.minimum(t0, positions_ref.shape[0] - bq)
    row_offset = t0 - read_start
    valid_count = _tile_field(tiles_ref, tile, 2)
    position_start -= row_offset
    t0 = read_start
    active = request < swa_page_ref.shape[0]
    request = jnp.minimum(request, swa_page_ref.shape[0] - 1)
    key_positions = anchor + jnp.arange(256)
    num_heads = q_ref.shape[1]
    mxu_heads = pl.cdiv(num_heads, 16) * 16
    # Separate QK/PV ranges let the next QK overlap the preceding PV.
    # Each range holds block_q results; together they must fit 256 MRB rows.

    def swa_qk_compute(first, size):
        for mxu in range(2):
            ks = pl.ds(mxu * 256, 256)
            if not swa_rhs_preloaded:
                pltpu.matmul_push_rhs(swa_ref[:, ks], 0, mxu, transpose=True)
            for r in range(size):
                lhs = jnp.pad(
                    q_ref[first + r, :, ks], ((0, mxu_heads - num_heads), (0, 0))
                )
                pltpu.matmul_acc_lhs(
                    accumulators[mxu][r],
                    lhs,
                    load_staged_rhs=(1 if swa_rhs_preloaded else 0) if r == 0 else None,
                )

    def swa_qk_pop(first, size):
        for r in range(size):
            halves = [
                pltpu.matmul_pop(accumulators[mxu][r])[:num_heads] for mxu in range(2)
            ]
            if non_causal:
                start, end = left, right
            else:
                position = position_start + first + r
                start = jnp.maximum(left, position - 127)
                end = jnp.minimum(right, position + 1)
            valid = (key_positions >= start) & (key_positions < end)
            swa_score_ref[first + r] = jnp.where(
                valid[None, :], (halves[0] + halves[1]) * (scale * log2_e), -jnp.inf
            )

    def main_unpack(indices):
        # Split each 64-row read into 16-row conversions shared by QK and PV.
        unpack_rows = 64
        for i in indices:
            for row in range(0, 512, unpack_rows):
                rows = pl.ds(row, unpack_rows)
                words = main_cache_ref[i, rows, :]
                for subrow in range(0, unpack_rows, 16):
                    subwords = words[pl.ds(subrow, 16)]
                    for group in range(4):
                        shifted = lax.shift_right_logical(subwords, 8 * group)
                        bits = shifted.astype(jnp.uint8)
                        fp8 = lax.bitcast_convert_type(bits, jnp.float8_e4m3fn)
                        values = fp8.astype(jnp.bfloat16)
                        subrows = pl.ds(row + subrow, 16)
                        main_kv_ref[i, subrows, pl.ds(group * 128, 128)] = values

    def main_qk_compute(indices):
        tiles = [(r, i, block) for r, i in enumerate(indices) for block in range(2)]

        def push(tile_index):
            r, i, block = tiles[tile_index]
            ks = pl.ds(block * 256, 256)
            for mxu in range(2):
                ns = pl.ds(mxu * 256, 256)
                pltpu.matmul_push_rhs(
                    main_kv_ref[i, ns, ks], tile_index % 2, mxu, transpose=True
                )

        # Alternate two RHS staging slots across tile and token boundaries.
        push(0)
        for tile_index, (r, i, block) in enumerate(tiles):
            if tile_index + 1 < len(tiles):
                push(tile_index + 1)
            ks = pl.ds(block * 256, 256)
            for mxu in range(2):
                lhs = jnp.pad(q_ref[i, :, ks], ((0, mxu_heads - num_heads), (0, 0)))
                pltpu.matmul_acc_lhs(
                    accumulators[mxu][r],
                    lhs,
                    load_staged_rhs=tile_index % 2,
                )

    def softmax(indices, qk_values):
        for r, i in enumerate(indices):
            main_logits = qk_values[r]
            # Scale each key column after QK, using FP32 KV scales.
            kv_scales = main_scale_ref[i]
            main_logits = main_logits * (kv_scales * (scale * log2_e))

            sink = sink_ref[...] * log2_e
            main_count = main_lengths_ref[
                jnp.minimum(t0 + i, main_lengths_ref.shape[0] - 1)
            ]
            main_logits = main_logits + jnp.where(
                jnp.arange(512)[None, :] < main_count, 0.0, -jnp.inf
            )
            swa_logits = swa_score_ref[i]
            logits = jnp.concatenate((main_logits, swa_logits), axis=1)
            maximum = jnp.maximum(sink, jnp.max(logits, axis=1, keepdims=True))
            exponentials = jnp.exp2(logits - maximum)
            sink_exponential = jnp.exp2(sink - maximum)
            denominator = sink_exponential + jnp.sum(
                exponentials, axis=1, keepdims=True
            )
            exponentials = exponentials / denominator
            # The denominator uses unscaled weights; fold KV scales into
            # main PV weights and round once to BF16.
            kv_scales = main_scale_ref[i]
            main_p_ref[i] = (exponentials[:, :512] * kv_scales).astype(jnp.bfloat16)
            swa_p_ref[i] = exponentials[:, 512:].astype(jnp.bfloat16)

    def fused_pv(indices):
        tiles = [(r, i, block) for r, i in enumerate(indices) for block in range(2)]

        def push(tile_index):
            r, i, block = tiles[tile_index]
            ks = pl.ds(block * 256, 256)
            for mxu in range(2):
                ns = pl.ds(mxu * 256, 256)
                pltpu.matmul_push_rhs(
                    main_kv_ref[i, ks, ns], tile_index % 2, mxu, transpose=False
                )

        # Alternate two RHS staging slots across tile and token boundaries.
        push(0)
        for tile_index, (r, i, block) in enumerate(tiles):
            if tile_index + 1 < len(tiles):
                push(tile_index + 1)
            ks = pl.ds(block * 256, 256)
            for mxu in range(2):
                lhs = jnp.pad(
                    main_p_ref[i, :, ks], ((0, mxu_heads - num_heads), (0, 0))
                )
                pltpu.matmul_acc_lhs(
                    accumulators[mxu][block_q + r],
                    lhs,
                    load_staged_rhs=tile_index % 2,
                )

        for mxu in range(2):
            pltpu.matmul_push_rhs(swa_ref[:, pl.ds(mxu * 256, 256)], 0, mxu)
            for r, i in enumerate(indices):
                lhs = jnp.pad(swa_p_ref[i], ((0, mxu_heads - num_heads), (0, 0)))
                pltpu.matmul_acc_lhs(
                    accumulators[mxu][block_q + r],
                    lhs,
                    load_staged_rhs=0 if r == 0 else None,
                )
                if r == bq - 1 and preload_swa is not None:
                    preload_swa(mxu)

    def inverse_rope(indices, target=output_ref, rotations=rope_ref, previous=False):
        for r, i in enumerate(indices):
            output = jnp.concatenate(
                [
                    pltpu.matmul_pop(accumulators[mxu][block_q + r])[:num_heads]
                    for mxu in range(2)
                ],
                axis=1,
            )
            tail = output[:, 384:]
            partner = jnp.take_along_axis(
                tail,
                jnp.broadcast_to((jnp.arange(128) ^ 1)[None, :], tail.shape),
                axis=1,
                mode="promise_in_bounds",
            )
            rotated = (
                tail * rotations[i, 0][None, :] + partner * rotations[i, 1][None, :]
            )
            if previous:
                last_tile = previous_tile
                last_start = _tile_field(tiles_ref, last_tile, 1)
                last_offset = last_start - jnp.minimum(
                    last_start, positions_ref.shape[0] - bq
                )
                valid = (
                    (_tile_field(tiles_ref, last_tile, 0) < swa_page_ref.shape[0])
                    & (i >= last_offset)
                    & (i < last_offset + _tile_field(tiles_ref, last_tile, 2))
                )
            else:
                valid = active & (i >= row_offset) & (i < row_offset + valid_count)
            target[i, :, :384] = jnp.where(valid, output[:, :384], 0).astype(
                jnp.bfloat16
            )
            target[i, :, 384:] = jnp.where(valid, rotated, 0).astype(jnp.bfloat16)

    indices = tuple(range(bq))
    if drain_only:
        inverse_rope(indices)
        return
    swa_qk_compute(0, bq)
    if prepare_next is not None:
        prepare_next()
    main_unpack(indices)
    if previous_output is not None:
        inverse_rope(indices, previous_output, previous_rope, previous=True)
    swa_qk_pop(0, bq)
    main_qk_compute(indices)
    for r, i in enumerate(indices):
        logits = jnp.concatenate(
            [pltpu.matmul_pop(accumulators[mxu][r])[:num_heads] for mxu in range(2)],
            axis=1,
        )
        softmax((i,), (logits,))
    fused_pv(indices)
    return


@functools.partial(
    jax.jit,
    static_argnames=("scale", "compress_ratio", "bq"),
)
def csa2(
    q: jax.Array,
    main_kv: jax.Array,
    main_kv_scale: jax.Array,
    swa_kv: jax.Array | tuple[jax.Array, jax.Array],
    starts: jax.Array,
    positions: jax.Array,
    sink: jax.Array,
    rope_cache: jax.Array,
    *,
    swa_metadata: jax.Array,
    main_lengths: jax.Array | None = None,
    swa_bounds: jax.Array | None = None,
    scale: float = 512**-0.5,
    compress_ratio: int = 2,
    bq: int = 8,
) -> jax.Array:
    """Ragged CSA2 attention with FP32 scales and independent SWA visibility.

    Main KV is query-local packed FP8 [T,512,128], scales are FP32 [T,512].
    SWA uses one or two native BF16 page owners. starts [B+1] describes
    requests; repeated boundaries and output padding are supported.
    Explicit swa_bounds selects visibility independently of model layer stage.
    """
    non_causal = swa_bounds is not None
    if q.ndim != 3 or q.shape[2] != 512:
        raise ValueError("CSA2 q must be [T,H,512]")
    tokens, heads, _ = q.shape
    if heads < 8 or heads > 128 or heads % 8:
        raise ValueError("CSA2 requires H divisible by 8, in [8,128]")
    if type(bq) is not int or not 1 <= bq <= 16:
        raise ValueError("CSA2 bq must be a static integer in [1,16]")
    if type(compress_ratio) is not int or compress_ratio < 1:
        raise ValueError("compress_ratio must be a positive static integer")
    if starts.ndim != 1 or starts.shape[0] < 2:
        raise ValueError("starts must be INT32 [B+1], B >= 1")
    batch = starts.shape[0] - 1
    if (
        swa_metadata.ndim != 2
        or swa_metadata.shape[0] != batch
        or swa_metadata.shape[1] < 3
    ):
        raise ValueError("SWA metadata must be INT32 [B,2+C], C >= 1")
    banks = swa_kv if isinstance(swa_kv, tuple) else (swa_kv,)
    if len(banks) not in (1, 2):
        raise ValueError("SWA requires one native owner or two banks")
    for bank in banks:
        if (
            bank.ndim != 3
            or bank.shape[0] < 1
            or bank.shape[-1] != 512
            or bank.dtype != jnp.bfloat16
        ):
            raise ValueError("SWA requires native BF16 [pages,page_size,512]")
        if bank.shape[1:] != banks[0].shape[1:]:
            raise ValueError("SWA banks must have matching page shapes")
    cache_page_size = banks[0].shape[1]
    if cache_page_size not in (8, 16, 32, 64) and (
        cache_page_size < 128 or cache_page_size % 128
    ):
        raise ValueError("SWA page size must be 8,16,32,64 or a multiple of 128")
    if main_lengths is None:
        main_lengths = jnp.minimum((positions + 1) // compress_ratio, 512)
    specs = (
        (q, (tokens, heads, 512), jnp.bfloat16),
        (main_kv, (tokens, 512, 128), jnp.int32),
        (main_kv_scale, (tokens, 512), jnp.float32),
        (starts, (batch + 1,), jnp.int32),
        (positions, (tokens,), jnp.int32),
        (main_lengths, (tokens,), jnp.int32),
        (sink, (heads,), jnp.float32),
        (rope_cache, (tokens, 2, 128), jnp.float32),
        (swa_metadata, swa_metadata.shape, jnp.int32),
    )
    for value, shape, dtype in specs:
        if value.shape != shape or value.dtype != jnp.dtype(dtype):
            raise ValueError(
                f"CSA2 expects {shape} {dtype}, got {value.shape} {value.dtype}"
            )
    if swa_bounds is not None and (
        swa_bounds.shape != (batch, 2) or swa_bounds.dtype != jnp.int32
    ):
        raise ValueError("SWA bounds must be INT32 [B,2]")
    if not tokens:
        return jnp.zeros_like(q)
    bq = min(bq, tokens, 128 // (pl.cdiv(heads, 16) * 4))
    fixed_tiles = batch == 1 and tokens % bq == 0
    tile_capacity = (
        tokens // bq if fixed_tiles else min(tokens, pl.cdiv(tokens, bq) + batch)
    )
    bounds = swa_bounds if non_causal else jnp.zeros((batch, 2), jnp.int32)
    scalars = (
        starts,
        positions,
        main_lengths,
        swa_metadata,
        bounds,
    )
    # Reserve three quarters of VMEM for four Q/KV buffers, decoded KV,
    # probabilities and compiler temporaries. Large batches stream metadata.
    metadata_bytes = main_kv_scale.nbytes + rope_cache.nbytes
    resident_metadata = metadata_bytes <= pltpu.get_tpu_info().vmem_capacity_bytes // 4
    page_size = min(cache_page_size, 64)
    num_pages = 256 // page_size

    kernel_fn = functools.partial(
        _csa2_kernel,
        non_causal=non_causal,
        bq=bq,
        block_q=min(bq, 128 // (pl.cdiv(heads, 16) * 4)),
        page_size=page_size,
        scale=scale,
    )

    def kernel(
        plan_ref,
        positions_ref,
        lengths_ref,
        pages_ref,
        bounds_ref,
        q_hbm,
        kv_hbm,
        scales_hbm,
        swa_hbm,
        sink_ref,
        rope_hbm,
        out_hbm,
        q_buf,
        kv_buf,
        scales_buf,
        rope_buf,
        rope_saved,
        swa_buf,
        outputs,
        decoded,
        swa_work,
        swa_scores,
        main_p,
        swa_p,
        input_sems,
        swa_sems,
        output_sems,
        tile_scratch,
    ):
        tiles_ref = tile_scratch
        if fixed_tiles:
            count = tokens // bq

            def tile_body(tile, unused):
                offset = tile * bq
                tiles_ref[tile * 3] = (offset >= plan_ref[1]).astype(jnp.int32)
                tiles_ref[tile * 3 + 1] = offset
                tiles_ref[tile * 3 + 2] = jnp.clip(plan_ref[1] - offset, 0, bq)
                return None

            lax.fori_loop(0, count, tile_body, None)
        else:
            count = _fill_query_tiles(plan_ref, tiles_ref, tokens, bq)
            if bq == 1:
                count = tokens
        mxu_heads = pl.cdiv(heads, 16) * 16
        accumulators = tuple(
            tuple(
                jax.empty_ref(
                    jax.ShapeDtypeStruct((mxu_heads, 256), jnp.float32),
                    memory_space=pltpu.ACC(mxu),
                )
                for _ in range(2 * bq)
            )
            for mxu in range(2)
        )

        def safe_tile(tile):
            if isinstance(tile, int):
                limit = count if isinstance(count, int) else tile_capacity
                return min(max(tile, 0), limit - 1)
            return tile

        def rows(tile):
            tile = safe_tile(tile)
            start = jnp.minimum(_tile_field(tiles_ref, tile, 1), tokens - bq)
            return pl.ds(start, bq)

        def input_copies(tile, bank):
            sources = ((q_hbm, q_buf), (kv_hbm, kv_buf))
            if not resident_metadata:
                sources += ((scales_hbm, scales_buf), (rope_hbm, rope_buf))
            return tuple(
                pltpu.make_async_copy(
                    source.at[rows(tile)], target.at[bank], input_sems.at[bank, i]
                )
                for i, (source, target) in enumerate(sources)
            )

        def swa_copies(tile, bank):
            tile = safe_tile(tile)
            request, left, right, anchor = 0, 0, 256, 0
            request = jnp.minimum(_tile_field(tiles_ref, tile, 0), batch - 1)
            _, left, right, anchor = swa_window(
                tile,
                tiles_ref,
                positions_ref,
                pages_ref,
                bounds_ref,
                page_size=page_size,
                non_causal=non_causal,
            )
            copies = []
            for block in range(num_pages):
                for owner, source in enumerate(swa_hbm):
                    position = anchor + block * page_size
                    column = (position - pages_ref[request, 0]) // cache_page_size
                    physical = pages_ref[
                        request, jnp.clip(column, 0, pages_ref.shape[1] - 3) + 2
                    ]
                    valid = (
                        (position < right)
                        & (position + page_size > left)
                        & (column >= 0)
                        & (column < pages_ref.shape[1] - 2)
                        & (physical >= 0)
                        & (physical < sum(owner.shape[0] for owner in banks))
                    )
                    if len(banks) == 2:
                        valid &= (
                            (physical < banks[0].shape[0])
                            if owner == 0
                            else (physical >= banks[0].shape[0])
                        )
                        if owner:
                            physical -= banks[0].shape[0]
                    physical = jnp.where(valid, physical, 0)
                    offset = jnp.where(valid, position % cache_page_size, 0)
                    index = block * len(banks) + owner
                    copies.append(
                        pltpu.make_async_copy(
                            source.at[pl.ds(physical, 1), pl.ds(offset, page_size)],
                            swa_buf.at[bank, index],
                            swa_sems.at[bank, index],
                        )
                    )
            return copies

        def start(tile, bank):
            for copy in (*input_copies(tile, bank), *swa_copies(tile, bank)):
                copy.start()

        def output_copy(tile, bank):
            tile = safe_tile(tile)
            if fixed_tiles:
                source_rows, target_rows = pl.ds(0, bq), pl.ds(tile * bq, bq)
            else:
                start = _tile_field(tiles_ref, tile, 1)
                size = _tile_field(tiles_ref, tile, 2)
                offset = start - jnp.minimum(start, tokens - bq)
                source_rows, target_rows = pl.ds(offset, size), pl.ds(start, size)
            return pltpu.make_async_copy(
                outputs.at[bank, source_rows],
                out_hbm.at[target_rows],
                output_sems.at[bank],
            )

        def prepare(tile, bank, work_bank):
            tile = safe_tile(tile)
            if len(banks) == 1:
                values = swa_buf[bank].reshape(256, 512)
                active = _tile_field(tiles_ref, tile, 0) < batch
                _, left, right, anchor = swa_window(
                    tile,
                    tiles_ref,
                    positions_ref,
                    pages_ref,
                    bounds_ref,
                    page_size=page_size,
                    non_causal=non_causal,
                )
                positions = anchor + 2 * jnp.arange(128)
                low = active & (positions >= left) & (positions < right)
                high = active & (positions + 1 >= left) & (positions + 1 < right)
                mask = jnp.where(low, 0xFFFF, 0) | jnp.where(high, ~0xFFFF, 0)
                values = pltpu.bitcast(
                    pltpu.bitcast(values, jnp.int32) & mask[:, None], jnp.bfloat16
                )
                swa_work[work_bank] = values
                return
            _prepare_swa(
                tile,
                tiles_ref,
                positions_ref,
                pages_ref,
                bounds_ref,
                tuple(swa_buf.at[bank, i] for i in range(num_pages * len(banks))),
                swa_work.at[work_bank],
                page_size=page_size,
                cache_page_size=cache_page_size,
                num_banks=len(banks),
                low_pages=banks[0].shape[0],
                non_causal=non_causal,
            )

        def compute(tile, bank, sb, *, previous=True, drain=False, next_tile=True):
            if not resident_metadata and not drain:
                rope_saved[bank] = rope_buf[sb]

            def prepare_next():
                if next_tile is True:
                    prepare(tile + 1, (sb + 1) % 4, 1 - bank)
                else:

                    @pl.when(tile + 1 < count)
                    def _():
                        prepare(tile + 1, (sb + 1) % 4, 1 - bank)

            def preload_next(mxu):
                def push():
                    pltpu.matmul_push_rhs(
                        (swa_work[1 - bank])[:, pl.ds(mxu * 256, 256)],
                        1,
                        mxu,
                        transpose=True,
                    )

                if next_tile is True:
                    push()
                else:

                    @pl.when(tile + 1 < count)
                    def _():
                        push()

            kernel_fn(
                tiles_ref,
                positions_ref,
                lengths_ref,
                pages_ref,
                bounds_ref,
                q_buf.at[sb],
                kv_buf.at[sb],
                scales_hbm.at[rows(tile)] if resident_metadata else scales_buf.at[sb],
                sink_ref,
                rope_hbm.at[rows(tile)] if resident_metadata else rope_saved.at[bank],
                outputs.at[bank],
                decoded,
                swa_work.at[bank],
                swa_scores,
                main_p,
                swa_p,
                accumulators=accumulators,
                tile_index=safe_tile(tile),
                previous_tile=safe_tile(tile - 1),
                previous_output=outputs.at[1 - bank]
                if previous and not drain
                else None,
                previous_rope=(
                    rope_hbm.at[rows(tile - 1)]
                    if resident_metadata
                    else rope_saved.at[1 - bank]
                )
                if previous and not drain
                else None,
                drain_only=drain,
                swa_rhs_preloaded=previous,
                prepare_next=prepare_next if not drain else None,
                preload_swa=preload_next if not drain else None,
            )
            if previous and not drain:
                output_copy(tile - 1, 1 - bank).start()

        def step(tile, bank, sb, *, previous=True, wait_output=True, steady=False):
            for copy in input_copies(tile, sb):
                copy.wait()
            if not previous:
                for copy in swa_copies(tile, sb):
                    copy.wait()
                prepare(tile, sb, bank)

            def wait_next():
                for copy in swa_copies(tile + 1, (sb + 1) % 4):
                    copy.wait()

            if steady:
                wait_next()
            else:

                @pl.when(tile + 1 < count)
                def _():
                    wait_next()

            if wait_output:
                output_copy(tile - 3, 1 - bank).wait()
            compute(
                tile, bank, sb, previous=previous, next_tile=True if steady else None
            )
            if steady:
                start(tile + 4, sb)
            else:

                @pl.when(tile + 4 < count)
                def _():
                    start(tile + 4, sb)

        for i in range(4):

            @pl.when(i < count)
            def _(*, i=i):
                start(i, i)

        for i in range(4):

            @pl.when(i < count)
            def _(*, i=i):
                step(i, i % 2, i, previous=i > 0, wait_output=i >= 3)

        groups = (
            max((count - 8) // 4, 0)
            if isinstance(count, int)
            else jnp.maximum((count - 8) // 4, 0)
        )

        def body(group, unused):
            for j in range(4):
                step(4 + group * 4 + j, j % 2, j, steady=True)
            return None

        lax.fori_loop(0, groups, body, None)

        # A four-tile steady iteration leaves at most seven drain tiles.
        # Keep their buffer slots static while PV remains live across steps.
        for j in range(7):
            tile = 4 + groups * 4 + j

            @pl.when(tile < count)
            def _(*, tile=tile, j=j):
                step(tile, j % 2, j % 4)

        @pl.when(count >= 3)
        def _():
            output_copy(count - 3, (count - 1) % 2).wait()

        compute(count - 1, (count - 1) % 2, (count - 1) % 4, drain=True, next_tile=None)
        output_copy(count - 1, (count - 1) % 2).start()

        @pl.when(count >= 2)
        def _():
            output_copy(count - 2, (count - 2) % 2).wait()

        output_copy(count - 1, (count - 1) % 2).wait()

    metadata_space = pltpu.VMEM if resident_metadata else pltpu.HBM
    spaces = (
        pltpu.HBM,
        pltpu.HBM,
        metadata_space,
        tuple(pltpu.HBM for _ in banks),
        pltpu.VMEM,
        metadata_space,
    )
    inputs = (q, main_kv, main_kv_scale[:, None, :], banks, sink[:, None], rope_cache)
    # BlockSpec alone does not prevent an upstream whole-owner memory-space copy.
    inputs = jax.tree.map(pltpu.with_memory_space_constraint, inputs, spaces)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(),
            in_specs=jax.tree.map(
                lambda space: pl.BlockSpec(memory_space=space), spaces
            ),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(
                pltpu.VMEM((4, bq, heads, 512), jnp.bfloat16),
                pltpu.VMEM((4, bq, 512, 128), jnp.int32),
                None if resident_metadata else pltpu.VMEM((4, bq, 1, 512), jnp.float32),
                None if resident_metadata else pltpu.VMEM((4, bq, 2, 128), jnp.float32),
                None if resident_metadata else pltpu.VMEM((2, bq, 2, 128), jnp.float32),
                pltpu.VMEM(
                    (4, num_pages * len(banks), 1, page_size, 512), jnp.bfloat16
                ),
                pltpu.VMEM((2, bq, heads, 512), jnp.bfloat16),
                pltpu.VMEM((bq, 512, 512), jnp.bfloat16),
                pltpu.VMEM((2, 256, 512), jnp.bfloat16),
                pltpu.VMEM((bq, heads, 256), jnp.float32),
                pltpu.VMEM((bq, heads, 512), jnp.bfloat16),
                pltpu.VMEM((bq, heads, 256), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((4, 2 if resident_metadata else 4)),
                pltpu.SemaphoreType.DMA((4, num_pages * len(banks))),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SMEM((tile_capacity * 3,), jnp.int32),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            disable_semaphore_checks=True,
        ),
        name="csa2",
    )(
        *scalars,
        *inputs,
    )
