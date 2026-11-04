# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified CSA2 attention for decode and both prefill layer groups."""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _decode_scale_rounding(starts, positions, *, num_banks, non_causal):
    """Resolve legacy precision from the whole request layout, before chunking."""
    tokens = positions.shape[0]
    decode_batch, tail = divmod(tokens, 6)
    batch = starts.shape[0] - 1
    if not decode_batch or tail or batch < decode_batch or num_banks != 1 or non_causal:
        return False
    expected = jnp.minimum(jnp.arange(batch + 1) * 6, tokens)
    consecutive = positions[::6, None] + jnp.arange(6)
    return jnp.all(starts == expected) & jnp.all(
        positions.reshape(decode_batch, 6) == consecutive
    )


def build_query_tiles(
    starts: jax.Array, total_tokens: int, bq: int
) -> tuple[jax.Array, jax.Array]:
    # A final synthetic request covers output padding without an HBM clear pass.
    boundaries = jnp.concatenate((starts, jnp.array([total_tokens], jnp.int32)))
    sizes = boundaries[1:] - boundaries[:-1]
    counts = pl.cdiv(sizes, bq)
    ends = jnp.cumsum(counts)
    begins = jnp.concatenate((jnp.zeros(1, jnp.int32), ends[:-1]))
    capacity = min(total_tokens, pl.cdiv(total_tokens, bq) + starts.shape[0] - 1)
    ids = jnp.arange(capacity, dtype=jnp.int32)
    requests = jnp.minimum(
        jnp.searchsorted(ends, ids, side="right"), sizes.shape[0] - 1
    )
    offsets = boundaries[requests] + (ids - begins[requests]) * bq
    valid = ids < ends[-1]
    lengths = jnp.where(valid, jnp.minimum(bq, boundaries[requests + 1] - offsets), 0)
    offsets = jnp.where(valid, offsets, 0)
    return jnp.stack((requests, offsets, lengths), axis=1), ends[-1]


def swa_window(tile, tiles, positions, pages, bounds, *, page_size, non_causal):
    request, start, count = tiles[tile, 0], tiles[tile, 1], tiles[tile, 2]
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
    request = jnp.minimum(tiles_ref[tile, 0], swa_page_ref.shape[0] - 1)
    active = tiles_ref[tile, 0] < swa_page_ref.shape[0]
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
    # Scalar prefetch
    tiles_ref,  # [tile_capacity, 3]: request, start, count
    positions_ref,
    lengths_ref,
    pages_ref,
    bounds_ref,
    count_ref,
    # Inputs
    q_hbm,  # [T, H, 512], BF16
    kv_hbm,
    scales_hbm,
    swa_hbm,
    sink_ref,
    rope_hbm,
    # Output
    out_hbm,  # [tile_capacity * bq, H, 512], BF16
    # Scratch
    q_buf,  # [4, bq, H, 512], BF16
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
    *,
    scale: float,
    non_causal: bool,
) -> None:
    """Persistent attention with one steady tile body and rotating buffers."""
    tokens, heads, _ = q_hbm.shape
    bq = q_buf.shape[1]
    batch = pages_ref.shape[0]
    num_banks = len(swa_hbm)
    cache_page_size = swa_hbm[0].shape[1]
    page_size = swa_buf.shape[-2]
    num_pages = 256 // page_size
    resident_metadata = scales_buf is None
    # These layouts have one tile per fixed query block, including padding.
    count = (
        tokens // bq if bq == 1 or (batch == 1 and tokens % bq == 0) else count_ref[0]
    )
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

    def rows(tile):
        start = jnp.minimum(tiles_ref[tile, 1], tokens - bq)
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
        request = jnp.minimum(tiles_ref[tile, 0], batch - 1)
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
                )
                if num_banks == 2:
                    valid &= (
                        (physical < swa_hbm[0].shape[0])
                        if owner == 0
                        else (physical >= swa_hbm[0].shape[0])
                    )
                    if owner:
                        physical -= swa_hbm[0].shape[0]
                physical = jnp.where(valid, physical, 0)
                offset = jnp.where(valid, position % cache_page_size, 0)
                index = block * num_banks + owner
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
        return pltpu.make_async_copy(
            outputs.at[bank], out_hbm.at[pl.ds(tile * bq, bq)], output_sems.at[bank]
        )

    def prepare(tile, bank, work_bank):
        _prepare_swa(
            tile,
            tiles_ref,
            positions_ref,
            pages_ref,
            bounds_ref,
            tuple(swa_buf.at[bank, i] for i in range(num_pages * num_banks)),
            swa_work.at[work_bank],
            page_size=page_size,
            cache_page_size=cache_page_size,
            num_banks=num_banks,
            low_pages=swa_hbm[0].shape[0],
            non_causal=non_causal,
        )

    def finish(tile, bank):
        """Drain pending PV, apply inverse RoPE, and mask padded rows."""
        target = outputs.at[bank]
        rotations = (
            rope_hbm.at[rows(tile)] if resident_metadata else rope_saved.at[bank]
        )
        start = tiles_ref[tile, 1]
        row_offset = start - jnp.minimum(start, tokens - bq)
        for i in range(bq):
            output = jnp.concatenate(
                [
                    pltpu.matmul_pop(accumulators[mxu][bq + i])[:heads]
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
            valid = (
                (tiles_ref[tile, 0] < pages_ref.shape[0])
                & (i >= row_offset)
                & (i < row_offset + tiles_ref[tile, 2])
            )
            target[i, :, :384] = jnp.where(valid, output[:, :384], 0).astype(
                jnp.bfloat16
            )
            target[i, :, 384:] = jnp.where(valid, rotated, 0).astype(jnp.bfloat16)

    def compute(tile, bank, sb, *, steady=False):
        """Finish the preceding tile while leaving this tile's PV live."""
        first_tile = isinstance(tile, int) and tile == 0
        if not resident_metadata:
            rope_saved[bank] = rope_buf[sb]
        q_ref = q_buf.at[sb]
        main_cache_ref = kv_buf.at[sb]
        main_scale_ref = (
            scales_hbm.at[rows(tile)] if resident_metadata else scales_buf.at[sb]
        )
        main_kv_ref = decoded
        swa_ref = swa_work.at[bank]
        swa_score_ref = swa_scores
        main_p_ref = main_p
        swa_p_ref = swa_p
        swa_page_ref = pages_ref
        main_lengths_ref = lengths_ref
        log2_e = math.log2(math.e)
        t0 = tiles_ref[tile, 1]
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
        position_start -= row_offset
        t0 = read_start
        key_positions = anchor + jnp.arange(256)

        def prepare_next():
            if steady:
                # Select the next resident input slot for the pipelined body.
                next_tile = jnp.minimum(tile + 1, count - 1)
                prepare(next_tile, next_tile & 3, 1 - bank)
            else:

                @pl.when(tile + 1 < count)
                def _():
                    prepare(tile + 1, (sb + 1) & 3, 1 - bank)

        def preload_next(mxu):
            def push():
                pltpu.matmul_push_rhs(
                    swa_work[1 - bank, :, pl.ds(mxu * 256, 256)],
                    1,
                    mxu,
                    transpose=True,
                )

            if steady:
                push()
            else:

                @pl.when(tile + 1 < count)
                def _():
                    push()

        def swa_qk_compute(first, size):
            for mxu in range(2):
                ks = pl.ds(mxu * 256, 256)
                if first_tile:
                    pltpu.matmul_push_rhs(swa_ref[:, ks], 0, mxu, transpose=True)
                for r in range(size):
                    lhs = jnp.pad(
                        q_ref[first + r, :, ks], ((0, mxu_heads - heads), (0, 0))
                    )
                    pltpu.matmul_acc_lhs(
                        accumulators[mxu][r],
                        lhs,
                        load_staged_rhs=(0 if first_tile else 1) if r == 0 else None,
                    )

        def swa_qk_pop(first, size):
            for r in range(size):
                halves = [
                    pltpu.matmul_pop(accumulators[mxu][r])[:heads] for mxu in range(2)
                ]
                if non_causal:
                    start, end = left, right
                else:
                    position = position_start + first + r
                    start = jnp.maximum(left, position - 127)
                    end = jnp.minimum(right, position + 1)
                valid = (key_positions >= start) & (key_positions < end)
                swa_score_ref[first + r] = (halves[0] + halves[1]) * (
                    scale * log2_e
                ) + jnp.where(valid[None, :], 0.0, -jnp.inf)

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
                    lhs = jnp.pad(q_ref[i, :, ks], ((0, mxu_heads - heads), (0, 0)))
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
                        main_p_ref[i, :, ks], ((0, mxu_heads - heads), (0, 0))
                    )
                    pltpu.matmul_acc_lhs(
                        accumulators[mxu][bq + r],
                        lhs,
                        load_staged_rhs=tile_index % 2,
                    )

            for mxu in range(2):
                pltpu.matmul_push_rhs(swa_ref[:, pl.ds(mxu * 256, 256)], 0, mxu)
                for r, i in enumerate(indices):
                    lhs = jnp.pad(swa_p_ref[i], ((0, mxu_heads - heads), (0, 0)))
                    pltpu.matmul_acc_lhs(
                        accumulators[mxu][bq + r],
                        lhs,
                        load_staged_rhs=0 if r == 0 else None,
                    )
                    if r == bq - 1:
                        preload_next(mxu)

        indices = tuple(range(bq))
        swa_qk_compute(0, bq)
        prepare_next()
        main_unpack(indices)
        if not first_tile:
            finish(tile - 1, 1 - bank)
        swa_qk_pop(0, bq)
        main_qk_compute(indices)
        for r, i in enumerate(indices):
            logits = jnp.concatenate(
                [pltpu.matmul_pop(accumulators[mxu][r])[:heads] for mxu in range(2)],
                axis=1,
            )
            softmax((i,), (logits,))
        fused_pv(indices)
        if not first_tile:
            output_copy(tile - 1, 1 - bank).start()

    for i in range(4):

        @pl.when(i < count)
        def _(*, i=i):
            start(i, i)

    for copy in (*input_copies(0, 0), *swa_copies(0, 0)):
        copy.wait()
    prepare(0, 0, 0)

    # compute(0) prepares and stages tile 1 before the next iteration.
    @pl.when(count > 1)
    def _():
        for copy in swa_copies(1, 1):
            copy.wait()

    compute(0, 0, 0)

    @pl.when(count > 4)
    def _():
        start(4, 0)

    def step(tile, *, steady, last=False):
        sb = tile & 3
        bank = tile & 1
        for copy in input_copies(tile, sb):
            copy.wait()

        def wait_next():
            for copy in swa_copies(tile + 1, (sb + 1) & 3):
                copy.wait()

        if steady:
            wait_next()
        else:
            pl.when(tile + 1 < count)(wait_next)

        @pl.when(tile >= 3)
        def _():
            output_copy(tile - 3, 1 - bank).wait()

        compute(tile, bank, sb, steady=not last)

        if steady:
            start(tile + 4, sb)
        else:
            pl.when(tile + 4 < count)(lambda: start(tile + 4, sb))

    # Retain one tile body per loop; the drain loop handles the prefetched tail.
    end = max(count - 4, 1) if isinstance(count, int) else jnp.maximum(count - 4, 1)
    lax.fori_loop(1, end, lambda tile, _: step(tile, steady=True), None, unroll=False)
    lax.fori_loop(
        end,
        jnp.maximum(count - 1, end),
        lambda tile, _: step(tile, steady=False),
        None,
        unroll=False,
    )

    # Every staged RHS must be consumed before the kernel returns.
    @pl.when(count > 1)
    def _():
        step(count - 1, steady=False, last=True)

    @pl.when(count >= 3)
    def _():
        output_copy(count - 3, (count - 1) % 2).wait()

    finish(count - 1, (count - 1) % 2)
    output_copy(count - 1, (count - 1) % 2).start()

    @pl.when(count >= 2)
    def _():
        output_copy(count - 2, (count - 2) % 2).wait()

    output_copy(count - 1, (count - 1) % 2).wait()


@functools.partial(
    jax.jit,
    static_argnames=("scale", "compress_ratio", "bq", "round_scales"),
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
    round_scales: bool | None = None,
) -> jax.Array:
    """Paged ragged attention with one request-local query tile per program.

    Q is BF16 [T,H,512], main KV is packed FP8 INT32 [T,512,128], and
    scales are FP32 [T,512]. SWA is native BF16 [pages,page_size,512],
    optionally split into two banks with global page IDs.
    starts [B+1] partitions valid Qs; repeated starts denote empty requests.
    positions [T] are consecutive within each request. Metadata [B,2+C]
    stores absolute page base, minimum visibility and C physical page IDs.
    Bounds [B,2], when supplied, replace causal windows with [left,right),
    at most 192 tokens wide. Selected main KV must be finite; SWA outside
    visibility is masked before matrix multiplication. Sink is [H] FP32,
    inverse RoPE is [T,2,128] FP32. Output padding is zero.

    The persistent kernel uses four input buffers and carries pending PV
    across query tiles. It preloads the next SWA RHS before finishing this tile.
    Outputs use two buffers. Encoder/decoder denote the first/last 20 layers;
    visibility is independently specified by swa_bounds. round_scales preserves
    ordinary decode BF16-effective scales; False retains FP32 scales. None
    preserves legacy dispatch semantics by checking six-query boundaries.
    bq is an upper bound, limited by the two live accumulator ranges
    (at most eight queries for H=64, four for H=128). Ragged tails may read
    extra queries; only valid output rows are compacted.
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
    if round_scales is None:
        round_scales = _decode_scale_rounding(
            starts, positions, num_banks=len(banks), non_causal=non_causal
        )
    rounded = main_kv_scale.astype(jnp.bfloat16).astype(jnp.float32)
    main_kv_scale = jnp.where(round_scales, rounded, main_kv_scale)
    bq = min(bq, tokens, 128 // (pl.cdiv(heads, 16) * 4))
    direct_output = batch == 1 and tokens % bq == 0
    if direct_output:
        offsets = jnp.arange(tokens // bq, dtype=jnp.int32) * bq
        counts = jnp.clip(starts[-1] - offsets, 0, bq)
        tiles = jnp.stack(
            ((offsets >= starts[-1]).astype(jnp.int32), offsets, counts), 1
        )
        num_tiles = tokens // bq
    else:
        tiles, num_tiles = build_query_tiles(starts, tokens, bq)
        # With one query per tile, empty requests add no tiles and padding
        # contributes exactly its token count. Keep this bound static so
        # single-token chunks do not retain unreachable RHS preloads/DMAs.
        if bq == 1:
            num_tiles = tokens
    bounds = swa_bounds if non_causal else jnp.zeros((batch, 2), jnp.int32)
    scalars = (tiles, positions, main_lengths, swa_metadata, bounds)
    # Reserve three quarters of VMEM for the attention buffers and temporaries.
    metadata_bytes = main_kv_scale.nbytes + rope_cache.nbytes
    resident_metadata = metadata_bytes <= pltpu.get_tpu_info().vmem_capacity_bytes // 4
    metadata_space = pltpu.VMEM if resident_metadata else pltpu.HBM
    page_size = min(cache_page_size, 64)
    num_pages = 256 // page_size

    tiled_output = pl.pallas_call(
        functools.partial(_csa2_kernel, scale=scale, non_causal=non_causal),
        out_shape=jax.ShapeDtypeStruct((tiles.shape[0] * bq, heads, 512), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            grid=(),
            in_specs=(
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=metadata_space),
                tuple(pl.BlockSpec(memory_space=pltpu.HBM) for _ in banks),
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=metadata_space),
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
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            disable_semaphore_checks=True,
        ),
        name="csa2",
    )(
        *scalars,
        jnp.asarray([num_tiles], jnp.int32),
        q,
        main_kv,
        main_kv_scale[:, None, :],
        banks,
        sink[:, None],
        rope_cache,
    )
    if direct_output:
        return tiled_output[:tokens]
    # Fixed output windows avoid overlapping writes at ragged request tails.
    # Compact only attention outputs; the much larger main KV stays in place.
    boundaries = jnp.concatenate((starts, jnp.array([tokens], jnp.int32)))
    counts = pl.cdiv(jnp.diff(boundaries), bq)
    begins = jnp.concatenate((jnp.zeros(1, jnp.int32), jnp.cumsum(counts)[:-1]))
    token_ids = jnp.arange(tokens)
    requests = jnp.minimum(jnp.searchsorted(starts[1:], token_ids, side="right"), batch)
    tile_ids = begins[requests] + (token_ids - boundaries[requests]) // bq
    local_ids = token_ids - jnp.minimum(tiles[tile_ids, 1], tokens - bq)
    return tiled_output[tile_ids * bq + local_ids]
