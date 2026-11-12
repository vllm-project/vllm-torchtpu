# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CSA2 W8A16 attention over row-packed FP8 main KV and native BF16 SWA."""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41.csa2.csa2 import (
    _fill_query_tiles,
    _prepare_swa,
    _tile_field,
    swa_window,
)


def _csa2_kernel(
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
    qk_scores,
    swa_work,
    swa_scores,
    main_p,
    swa_p,
    denominators,
    input_sems,
    swa_sems,
    output_sems,
    tile_scratch,
    *,
    scale: float,
    non_causal: bool,
    uniform_queries: bool,
) -> None:
    (tokens, heads) = (positions_ref.shape[0], q_hbm.shape[1])
    bq = q_buf.shape[1]
    batch = pages_ref.shape[0]
    num_banks = len(swa_hbm)
    cache_page_size = swa_hbm[0].shape[1]
    page_size = swa_buf.shape[-2]
    num_pages = 256 // page_size
    tile_capacity = tile_scratch.shape[0] // 3
    fixed_tiles = uniform_queries or (batch == 1 and tokens % bq == 0)
    resident_metadata = scales_buf is None
    tiles_ref = tile_scratch
    if uniform_queries:
        count = tokens // bq
    elif fixed_tiles:
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

    def field(tile, column):
        if uniform_queries:
            return tile if column == 0 else tile * bq if column == 1 else bq
        return _tile_field(tiles_ref, tile, column)

    def window(tile):
        if not uniform_queries:
            return swa_window(
                tile,
                tiles_ref,
                positions_ref,
                pages_ref,
                bounds_ref,
                page_size=page_size,
                non_causal=non_causal,
            )
        position = positions_ref[tile * bq]
        left = bounds_ref[tile, 0] if non_causal else jnp.maximum(0, position - 127)
        right = bounds_ref[tile, 1] if non_causal else position + bq
        anchor = left // page_size * page_size
        left = jnp.maximum(left, pages_ref[tile, 1])
        return (position, left, right, anchor)

    def safe_tile(tile):
        if isinstance(tile, int):
            limit = count if isinstance(count, int) else tile_capacity
            return min(max(tile, 0), limit - 1)
        return tile

    def rows(tile):
        if uniform_queries:
            return pl.ds(tile * bq, bq)
        tile = safe_tile(tile)
        start = jnp.minimum(field(tile, 1), tokens - bq)
        return pl.ds(start, bq)

    def input_copies(tile, bank):
        sources = ((q_hbm, q_buf), (kv_hbm, kv_buf))
        if not resident_metadata:
            sources += ((scales_hbm, scales_buf), (rope_hbm, rope_buf))
        return tuple(
            pltpu.make_async_copy(
                source.at[rows(tile)],
                target.at[bank],
                input_sems.at[bank, i],
            )
            for (i, (source, target)) in enumerate(sources)
        )

    def swa_copies(tile, bank):
        tile = safe_tile(tile)
        request = tile if uniform_queries else jnp.minimum(field(tile, 0), batch - 1)
        (_, left, right, anchor) = window(tile)
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
                    & (physical < sum(owner.shape[0] for owner in swa_hbm))
                )
                if num_banks == 2:
                    valid &= (
                        physical < swa_hbm[0].shape[0]
                        if owner == 0
                        else physical >= swa_hbm[0].shape[0]
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
        tile = safe_tile(tile)
        if fixed_tiles:
            source_rows = pl.ds(0, bq)
            target_rows = pl.ds(tile * bq, bq)
        else:
            start = field(tile, 1)
            size = field(tile, 2)
            offset = start - jnp.minimum(start, tokens - bq)
            source_rows = pl.ds(offset, size)
            target_rows = pl.ds(start, size)
        return pltpu.make_async_copy(
            outputs.at[bank, source_rows], out_hbm.at[target_rows], output_sems.at[bank]
        )

    def prepare(tile, bank, work_bank):
        tile = safe_tile(tile)
        if num_banks == 1:
            values = swa_buf[bank].reshape(256, 512)
            active = True if uniform_queries else field(tile, 0) < batch
            (_, left, right, anchor) = window(tile)
            positions = anchor + 2 * jnp.arange(128)
            low = active & (positions >= left) & (positions < right)
            high = active & (positions + 1 >= left) & (positions + 1 < right)
            mask = jnp.where(low, 65535, 0) | jnp.where(high, ~65535, 0)
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
        tile = safe_tile(tile)
        target = outputs.at[bank]
        rotations = (
            rope_hbm.at[rows(tile)] if resident_metadata else rope_saved.at[bank]
        )
        start = field(tile, 1)
        row_offset = start - jnp.minimum(start, tokens - bq)
        for i in range(bq):
            output = jnp.concatenate(
                [
                    pltpu.matmul_pop(accumulators[mxu][bq + i])[:heads]
                    for mxu in range(2)
                ],
                axis=1,
            )
            output = output * jnp.tile(denominators[bank, i], (1, 4))
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
                True
                if uniform_queries
                else (field(tile, 0) < batch)
                & (i >= row_offset)
                & (i < row_offset + field(tile, 2))
            )
            target[i, :, :384] = jnp.where(valid, output[:, :384], 0).astype(
                jnp.bfloat16
            )
            target[i, :, 384:] = jnp.where(valid, rotated, 0).astype(jnp.bfloat16)

    def qk(tile, bank, sb, *, first=False, vpu_callback=None):
        q_ref = q_buf.at[sb]
        main_kv_ref = kv_buf.at[sb].bitcast(jnp.float8_e4m3fn)
        swa_ref = swa_work.at[bank]
        for mxu in range(2):
            ks = pl.ds(mxu * 256, 256)
            if first:
                pltpu.matmul_push_rhs(swa_ref[:, ks], 0, mxu, transpose=True)
            for i in range(bq):
                lhs = jnp.pad(q_ref[i, :, ks], ((0, mxu_heads - heads), (0, 0)))
                pltpu.matmul_acc_lhs(
                    accumulators[mxu][i],
                    lhs,
                    load_staged_rhs=(0 if first else 1) if i == 0 else None,
                )
        for i in range(bq):
            for mxu in range(2):
                swa_scores[bank, mxu, i] = pltpu.matmul_pop(accumulators[mxu][i])[
                    :heads
                ]
        tiles = [(i, k) for i in range(bq) for k in range(2)]

        def push(t):
            (i, k) = tiles[t]
            for mxu in range(2):
                pltpu.matmul_push_rhs(
                    main_kv_ref[i, pl.ds(mxu * 256, 256), pl.ds(k * 256, 256)],
                    t % 2,
                    mxu,
                    transpose=True,
                )

        push(0)
        for t, (i, k) in enumerate(tiles):
            if t + 1 < len(tiles):
                push(t + 1)
            lhs = jnp.pad(
                q_ref[i, :, pl.ds(k * 256, 256)], ((0, mxu_heads - heads), (0, 0))
            )
            for mxu in range(2):
                pltpu.matmul_acc_lhs(accumulators[mxu][i], lhs, load_staged_rhs=t % 2)
            if (
                vpu_callback is not None
                and k == 1
                and ((i + 1) % 3 == 0 or i == bq - 1)
            ):
                first = i // 3 * 3
                vpu_callback(first, i + 1 - first)

    def pop_qk(bank):
        for i in range(bq):
            qk_scores[bank, i] = jnp.concatenate(
                [pltpu.matmul_pop(accumulators[mxu][i])[:heads] for mxu in range(2)],
                axis=1,
            )

    # Fold 256-wide key blocks and process three independent queries at a time.
    def softmax(tile, bank, sb, first=0, size=None):
        size = bq if size is None else size
        width = 256
        scales = scales_hbm[rows(tile)] if resident_metadata else scales_buf[sb]
        (position, left, right, anchor) = window(tile)
        start = field(tile, 1)
        read_start = start if uniform_queries else jnp.minimum(start, tokens - bq)
        position -= start - read_start
        query_rows = pl.ds(first, size)
        lane_max = jnp.full((size, heads, width), -jnp.inf, jnp.float32)
        logit_blocks = []
        for block in range(768 // width):
            main = block < 512 // width
            offset = block * width if main else block * width - 512
            cols = pl.ds(offset, width)
            values = []
            for i in range(first, first + size):
                if main:
                    length = lengths_ref[jnp.minimum(read_start + i, tokens - 1)]
                    logits = qk_scores[bank, i, :, cols] * (
                        scales[i, :, offset : offset + width]
                        * (scale * math.log2(math.e))
                    )
                    logits += jnp.where(
                        (offset + jnp.arange(width))[None, :] < length, 0.0, -jnp.inf
                    )
                else:
                    (lo, hi) = (
                        (left, right)
                        if non_causal
                        else (
                            jnp.maximum(left, position + i - 127),
                            jnp.minimum(right, position + i + 1),
                        )
                    )
                    keys = anchor + offset + jnp.arange(width)
                    valid = (keys >= lo) & (keys < hi)
                    logits = jnp.where(
                        valid[None, :],
                        (
                            swa_scores[bank, 0, i, :, cols]
                            + swa_scores[bank, 1, i, :, cols]
                        )
                        * (scale * math.log2(math.e)),
                        -jnp.inf,
                    )
                values.append(logits)
            logits = jnp.stack(values)
            logit_blocks.append(logits)
            lane_max = jnp.maximum(lane_max, logits)
        sink = sink_ref[...][None, :, :] * math.log2(math.e)
        maximum = jnp.maximum(sink, jnp.max(lane_max, axis=2, keepdims=True))
        lane_sum = jnp.zeros((size, heads, width), jnp.float32)
        for block in range(768 // width):
            main = block < 512 // width
            offset = block * width if main else block * width - 512
            cols = pl.ds(offset, width)
            logits = logit_blocks[block]
            e = jnp.exp2(logits - maximum)
            lane_sum = lane_sum + e
            if main:
                main_p[0, query_rows, :, cols] = (
                    e
                    * (
                        scales[first : first + size, :, offset : offset + width]
                        * 2.0 ** (-10)
                    )
                ).astype(jnp.bfloat16)
            else:
                swa_p[0, query_rows, :, cols] = (e * 2.0 ** (-10)).astype(jnp.bfloat16)
        denominator = jnp.exp2(sink - maximum) + jnp.sum(
            lane_sum, axis=2, keepdims=True
        )
        denominators[bank, query_rows] = jnp.broadcast_to(
            1024.0 / denominator, (size, heads, 128)
        )

    def pv(tile, bank, sb):
        main_kv_ref = kv_buf.at[sb].bitcast(jnp.float8_e4m3fn)
        if not resident_metadata:
            rope_saved[bank] = rope_buf[sb]
        tiles = [(i, k) for i in range(bq) for k in range(2)]

        def push(t):
            (i, k) = tiles[t]
            for mxu in range(2):
                pltpu.matmul_push_rhs(
                    main_kv_ref[i, pl.ds(k * 256, 256), pl.ds(mxu * 256, 256)],
                    t % 2,
                    mxu,
                )

        push(0)
        for t, (i, k) in enumerate(tiles):
            if t + 1 < len(tiles):
                push(t + 1)
            lhs = jnp.pad(
                main_p[0, i, :, pl.ds(k * 256, 256)], ((0, mxu_heads - heads), (0, 0))
            )
            for mxu in range(2):
                pltpu.matmul_acc_lhs(
                    accumulators[mxu][bq + i], lhs, load_staged_rhs=t % 2
                )
        for mxu in range(2):
            pltpu.matmul_push_rhs(swa_work[bank, :, pl.ds(mxu * 256, 256)], 0, mxu)
            for i in range(bq):
                lhs = jnp.pad(swa_p[0, i], ((0, mxu_heads - heads), (0, 0)))
                pltpu.matmul_acc_lhs(
                    accumulators[mxu][bq + i],
                    lhs,
                    load_staged_rhs=0 if i == 0 else None,
                )

    def prepare_qk(tile, bank, sb):
        for copy in swa_copies(tile, sb):
            copy.wait()
        prepare(tile, sb, bank)
        for mxu in range(2):
            pltpu.matmul_push_rhs(
                swa_work[bank, :, pl.ds(mxu * 256, 256)], 1, mxu, transpose=True
            )

    for i in range(4):

        @pl.when(i < count)
        def _(*, i=i):
            start(i, i)

    for copy in (*input_copies(0, 0), *swa_copies(0, 0)):
        copy.wait()
    prepare(0, 0, 0)
    qk(0, 0, 0, first=True)
    pop_qk(0)

    # QK(t+1), softmax(t), and retirement of PV(t-1) share one schedule.
    # Issue PV(t) before starting the previous output DMA.
    def step(tile, *, first=False, last=False, steady=False, bank_override=None):
        bank = tile & 1 if bank_override is None else bank_override
        sb = tile & 3

        def next_qk():
            for copy in input_copies(tile + 1, sb + 1 & 3):
                copy.wait()
            prepare_qk(tile + 1, 1 - bank, sb + 1 & 3)

            def qk_vpu(first_query, size):
                softmax(tile, bank, sb, first_query, size)

            qk(tile + 1, 1 - bank, sb + 1 & 3, vpu_callback=qk_vpu)

        if last:
            softmax(tile, bank, sb)
        elif steady:
            next_qk()
        else:
            pl.when(tile + 1 < count)(next_qk)
            pl.when(tile + 1 >= count)(lambda: softmax(tile, bank, sb))
        if not first:

            @pl.when(tile >= 3)
            def _():
                output_copy(tile - 3, 1 - bank).wait()

            finish(tile - 1, 1 - bank)
        pv(tile, bank, sb)
        if not first:
            output_copy(tile - 1, 1 - bank).start()
        if not last:
            if steady:
                pop_qk(1 - bank)
            else:
                pl.when(tile + 1 < count)(lambda: pop_qk(1 - bank))
        if steady:
            start(tile + 4, sb)
        else:
            pl.when(tile + 4 < count)(lambda: start(tile + 4, sb))

    step(0, first=True, last=isinstance(count, int) and count == 1)
    pairs = (
        max((count - 5) // 2, 0)
        if isinstance(count, int)
        else jnp.maximum((count - 5) // 2, 0)
    )

    def pair_body(pair, _):
        for j in range(2):
            step(1 + 2 * pair + j, steady=True, bank_override=1 + j & 1)

    lax.fori_loop(0, pairs, pair_body, None, unroll=False)
    end = 1 + 2 * pairs
    if isinstance(count, int):
        for tile in range(end, max(count - 1, end)):
            step(tile, bank_override=tile & 1)
    else:
        lax.fori_loop(
            end,
            jnp.maximum(count - 1, end),
            lambda tile, _: step(tile),
            None,
            unroll=False,
        )

    @pl.when(count > 1)
    def _():
        step(count - 1, last=True)

    @pl.when(count >= 3)
    def _():
        output_copy(count - 3, count - 1 & 1).wait()

    finish(count - 1, count - 1 & 1)
    output_copy(count - 1, count - 1 & 1).start()

    @pl.when(count >= 2)
    def _():
        output_copy(count - 2, count - 2 & 1).wait()

    output_copy(count - 1, count - 1 & 1).wait()


@functools.partial(
    jax.jit, static_argnames=("scale", "compress_ratio", "bq", "uniform_queries")
)
def csa2_matmulw8a16(
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
    scale: float = 512 ** (-0.5),
    compress_ratio: int = 2,
    bq: int = 8,
    uniform_queries: bool = False,
) -> jax.Array:
    """Ragged CSA2 attention with FP32 scales and independent SWA visibility.

    Main KV is INT32 [T,128,512]: byte r of word [t,k,d] holds the
    FP8 e4m3fn value at [t,4*k+r,d]. Scales are FP32 [T,512], one per
    key row; both QK and PV consume the same native FP8 Ref view.
    SWA must use native BF16 token pages. This entry point does not accept
    the legacy column-packed main KV or the shared INT32 SWA word pool.
    With uniform_queries=True, every request must contain exactly bq queries
    and starts must be arange(B+1)*bq. The default supports ragged requests.
    SWA uses one or two native BF16 page owners. starts [B+1] describes
    requests; repeated boundaries and output padding are supported.
    Explicit swa_bounds selects visibility independently of model layer stage.
    """
    if main_kv is None or main_kv_scale is None:
        raise ValueError("W8A16 CSA2 requires row-packed main KV and its scales")
    non_causal = swa_bounds is not None
    if q.ndim != 3 or q.shape[2] != 512:
        raise ValueError("CSA2 q must be [T,H,512]")
    (tokens, heads) = (positions.shape[0], q.shape[1])
    if q.shape[0] != tokens:
        raise ValueError("W8A16 CSA2 q and positions must have the same token count")
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
    if uniform_queries and len(banks) != 1:
        raise ValueError("Split SWA banks require uniform_queries=False")
    for bank in banks:
        if (
            bank.ndim != 3
            or bank.shape[0] < 1
            or bank.shape[-1] != 512
            or (bank.dtype != jnp.bfloat16)
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
        (q, (q.shape[0], heads, 512), jnp.bfloat16),
        (main_kv, (tokens, 128, 512), jnp.int32),
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
    if uniform_queries and tokens != batch * bq:
        raise ValueError("uniform_queries requires exactly bq queries per request")
    fixed_tiles = uniform_queries or (batch == 1 and tokens % bq == 0)
    tile_capacity = (
        tokens // bq if fixed_tiles else min(tokens, pl.cdiv(tokens, bq) + batch)
    )
    bounds = swa_bounds if non_causal else jnp.zeros((batch, 2), jnp.int32)
    scalars = (starts, positions, main_lengths, swa_metadata, bounds)
    metadata_bytes = main_kv_scale.nbytes + rope_cache.nbytes
    resident_metadata = metadata_bytes <= pltpu.get_tpu_info().vmem_capacity_bytes // 4
    page_size = min(cache_page_size, 64)
    num_pages = 256 // page_size
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
    out_shape = pltpu.HBM(shape=q.shape, dtype=q.dtype)
    out_specs = pl.BlockSpec(memory_space=pltpu.HBM)
    inputs = jax.tree.map(pltpu.with_memory_space_constraint, inputs, spaces)
    return pl.pallas_call(
        functools.partial(
            _csa2_kernel,
            scale=scale,
            non_causal=non_causal,
            uniform_queries=uniform_queries,
        ),
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(),
            in_specs=jax.tree.map(
                lambda space: pl.BlockSpec(memory_space=space), spaces
            ),
            out_specs=out_specs,
            scratch_shapes=(
                pltpu.VMEM((4, bq, heads, 512), jnp.bfloat16),
                pltpu.VMEM((4, bq, 128, 512), jnp.int32),
                None if resident_metadata else pltpu.VMEM((4, bq, 1, 512), jnp.float32),
                None if resident_metadata else pltpu.VMEM((4, bq, 2, 128), jnp.float32),
                None if resident_metadata else pltpu.VMEM((2, bq, 2, 128), jnp.float32),
                pltpu.VMEM(
                    (4, num_pages * len(banks), 1, page_size, 512), jnp.bfloat16
                ),
                pltpu.VMEM((2, bq, heads, 512), jnp.bfloat16),
                pltpu.VMEM((2, bq, heads, 512), jnp.float32),
                pltpu.VMEM((2, 256, 512), jnp.bfloat16),
                pltpu.VMEM((2, 2, bq, heads, 256), jnp.float32),
                pltpu.VMEM((1, bq, heads, 512), jnp.bfloat16),
                pltpu.VMEM((1, bq, heads, 256), jnp.bfloat16),
                pltpu.VMEM((2, bq, heads, 128), jnp.float32),
                pltpu.SemaphoreType.DMA((4, 2 if resident_metadata else 4)),
                pltpu.SemaphoreType.DMA((4, num_pages * len(banks))),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SMEM((tile_capacity * 3,), jnp.int32),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, disable_semaphore_checks=True
        ),
        name="csa2_matmulw8a16",
    )(*scalars, *inputs)
